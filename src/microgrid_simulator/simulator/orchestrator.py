"""Authoritative simulation loop over injectable telemetry and dispatch providers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd

from microgrid_simulator.config import Settings
from microgrid_simulator.contracts import (
    DispatchCommand,
    DispatchResult,
    PlantObservation,
    TelemetrySessionRequest,
)
from microgrid_simulator.controllers import RuleBasedController
from microgrid_simulator.digital_twin.replay import TelemetryWindow
from microgrid_simulator.env import MicrogridEnv
from microgrid_simulator.simulator.providers import DispatchProvider, TelemetryProvider
from microgrid_simulator.ui.rollout import (
    _dispatch_trace,
    _episode_meta,
    _slack_bus_id,
    _step_row,
    _totals,
)

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class SimulatorOrchestrator:
    """Own simulation time, command validation, fallback, and realized results."""

    def __init__(
        self,
        settings: Settings,
        telemetry: TelemetryProvider,
        dispatch: DispatchProvider,
        *,
        command_timeout_ms: int = 1000,
        pace_seconds: float = 0.0,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.telemetry = telemetry
        self.dispatch = dispatch
        self.command_timeout_ms = command_timeout_ms
        self.pace_seconds = max(0.0, pace_seconds)
        self.session_id = session_id or str(uuid4())
        self.fallback = RuleBasedController(settings)
        self.env: MicrogridEnv | None = None
        # Real-time audit counters: deadline overruns (no valid command within
        # the per-tick TTL -> rule fallback) and recoveries (a fresh valid command
        # applied again after a fallback). Reset at the start of each run().
        self.deadline_overruns = 0
        self.recoveries = 0
        self._last_overrun = False

    @staticmethod
    def _window(payload: Any) -> TelemetryWindow:
        return TelemetryWindow(
            demand_mw=np.asarray(
                [sample.load_demand_kw / 1000.0 for sample in payload.samples],
                dtype=np.float64,
            ),
            pv_mw=np.asarray(
                [sample.pv_available_kw / 1000.0 for sample in payload.samples],
                dtype=np.float64,
            ),
            timestamps=pd.DatetimeIndex([sample.observed_at for sample in payload.samples]),
            first_evaluated_timestamp=pd.Timestamp(payload.session.first_evaluated_time),
            source_files=payload.source_files,
            timestamps_are_observed=payload.session.timestamps_are_observed,
        )

    def _observation(
        self, sequence: int, observation: np.ndarray, info: dict[str, Any], source_id: str
    ) -> PlantObservation:
        if self.env is None:
            raise RuntimeError("simulator environment is not initialized")
        state = self.env._last_state  # noqa: SLF001
        forecast_available = bool(info.get("forecast_available", False))
        return PlantObservation(
            session_id=self.session_id,
            sequence_id=sequence,
            source_id=source_id,
            observed_at=datetime.now(timezone.utc),
            simulation_time_hours=max(0.0, state.timestamp),
            timestep_hours=self.settings.topology.timestep_hours,
            observation=observation.tolist(),
            observation_shape=int(observation.size),
            action_shape=self.env.action_dim,
            pv_available_kw=max(0.0, state.pv_available_mw * 1000.0),
            pv_used_kw=max(0.0, state.pv_used_mw * 1000.0),
            load_demand_kw=max(0.0, state.load_demand_mw * 1000.0),
            load_served_kw=max(0.0, state.load_served_mw * 1000.0),
            battery_soc=state.soc[0] if state.soc else self.env.backend.battery.soc,
            battery_soh=state.soh[0] if state.soh else self.env.backend.battery.soh,
            battery_power_kw=state.battery_p_mw * 1000.0,
            diesel_on=state.diesel_on,
            diesel_power_kw=max(0.0, state.diesel_p_mw * 1000.0),
            grid_connected=not state.islanded,
            grid_import_kw=state.grid_import_mw * 1000.0,
            forecast_available=forecast_available,
            forecast_pv_kw=(
                float(info["pv_forecast_mw"]) * 1000.0
                if forecast_available and info.get("pv_forecast_mw") is not None
                else None
            ),
            forecast_demand_kw=(
                float(info["demand_forecast_mw"]) * 1000.0
                if forecast_available and info.get("demand_forecast_mw") is not None
                else None
            ),
            data_quality="good" if state.solver_ok else "invalid",
            constraints={
                "battery_soc_min": self.settings.battery.soc_min,
                "battery_soc_max": self.settings.battery.soc_max,
                "battery_max_charge_kw": self.settings.battery.max_charge_mw * 1000.0,
                "battery_max_discharge_kw": self.settings.battery.max_discharge_mw * 1000.0,
                "diesel_max_kw": self.settings.diesel.max_kw,
            },
        )

    def _fallback_action(self) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("simulator environment is not initialized")
        forecast_available = self.env._forecast_is_available()  # noqa: SLF001
        pv_horizon, demand_horizon = self.env._forecast_vectors()  # noqa: SLF001
        control = self.fallback.act(
            self.env._last_state,  # noqa: SLF001
            pv_forecast_mw=self.env._current_pv_forecast_mw(),  # noqa: SLF001
            demand_forecast_mw=self.env._current_demand_forecast_mw(),  # noqa: SLF001
            pv_forecast_horizon_mw=pv_horizon if forecast_available else None,
            demand_forecast_horizon_mw=demand_horizon if forecast_available else None,
        )
        return self.env.encode_action(control)

    @staticmethod
    def _command_is_valid(command: DispatchCommand, observation: PlantObservation) -> bool:
        return (
            command.session_id == observation.session_id
            and command.telemetry_sequence_id == observation.sequence_id
            and command.expires_at >= datetime.now(timezone.utc)
            and len(command.normalized_action) == observation.action_shape
            and all(np.isfinite(command.normalized_action))
        )

    async def run(
        self,
        *,
        max_steps: int | None = None,
        seed: int = 0,
        on_event: EventCallback | None = None,
    ) -> list[DispatchResult]:
        configured_steps = int(
            round(self.settings.episode.horizon_hours / self.settings.topology.timestep_hours)
        )
        limit = min(max_steps or configured_steps, configured_steps)
        # Alignment validation requires four samples including context. Prefetch
        # three evaluated steps for shorter smoke runs, but execute only `limit`.
        telemetry_steps = min(configured_steps, max(3, limit))
        payload = await self.telemetry.window(
            TelemetrySessionRequest(
                start_at=self.settings.episode.telemetry_start,
                n_steps=telemetry_steps,
            )
        )
        if payload.session.source_id != (
            self.settings.forecast.source_id or self.settings.scenario.name
        ):
            raise ValueError("telemetry source does not match the active simulator scenario")

        env_settings = self.settings.model_copy(
            update={
                "episode": self.settings.episode.model_copy(
                    update={
                        "horizon_hours": limit * self.settings.topology.timestep_hours,
                        "telemetry_start": None,
                    }
                )
            }
        )
        telemetry_window = self._window(payload)
        self.env = await asyncio.to_thread(
            MicrogridEnv,
            settings=env_settings,
            telemetry_window=telemetry_window,
        )
        observation, info = await asyncio.to_thread(self.env.reset, seed=seed)
        results: list[DispatchResult] = []
        rows: list[dict[str, Any]] = []
        slack_id = _slack_bus_id(env_settings)
        self.deadline_overruns = 0
        self.recoveries = 0
        self._last_overrun = False
        try:
            if on_event is not None:
                await on_event(
                    {
                        "type": "ems_start",
                        "schema_version": "1.0",
                        "session_id": self.session_id,
                        "boundary": "simulation-only",
                    }
                )
                await on_event(
                    {
                        "type": "meta",
                        "meta": {
                            **_episode_meta(
                                env_settings,
                                self.env,
                                "ems",
                                seed,
                                info,
                                slack_id,
                            ),
                            "expected_steps": limit,
                            "ems_session_id": self.session_id,
                        },
                    }
                )
            for sequence in range(limit):
                plant_observation = self._observation(
                    sequence, observation, info, payload.session.source_id
                )
                command: DispatchCommand | None = None
                status: Literal["applied", "fallback", "rejected"] = "fallback"
                message = "command timeout or service error; rule fallback applied"
                overrun = True
                try:
                    candidate = await asyncio.wait_for(
                        self.dispatch.dispatch(plant_observation),
                        timeout=self.command_timeout_ms / 1000.0,
                    )
                    if self._command_is_valid(candidate, plant_observation):
                        command = candidate
                        action = np.asarray(candidate.normalized_action, dtype=np.float32)
                        status = "fallback" if candidate.fallback_active else "applied"
                        message = "EMS command applied"
                        # A valid command met the deadline. A shield-active token
                        # is a protected dispatch (not a deadline overrun).
                        overrun = False
                    else:
                        action = self._fallback_action()
                        status = "rejected"
                        overrun = True
                        message = "stale or mismatched command rejected; rule fallback applied"
                except (TimeoutError, ConnectionError, ValueError):
                    action = self._fallback_action()
                    overrun = True

                if overrun:
                    self.deadline_overruns += 1
                    self._last_overrun = True
                elif self._last_overrun:
                    self.recoveries += 1
                    self._last_overrun = False

                dispatch_policy = (
                    command.controller
                    if command is not None and not command.fallback_active
                    else "rule"
                )
                dispatch_trace = _dispatch_trace(self.env, dispatch_policy, action)
                observation, reward, terminated, truncated, info = await asyncio.to_thread(
                    self.env.step, action
                )
                row = _step_row(
                    env_settings,
                    self.env,
                    sequence,
                    reward,
                    info,
                    slack_id,
                    dispatch_trace,
                )
                rows.append(row)
                state = self.env._last_state  # noqa: SLF001
                result = DispatchResult(
                    session_id=self.session_id,
                    telemetry_sequence_id=sequence,
                    command_id=command.command_id if command is not None else None,
                    status=status,
                    status_message=message,
                    realized_battery_power_kw=state.battery_p_mw * 1000.0,
                    realized_diesel_on=state.diesel_on,
                    realized_diesel_power_kw=max(0.0, state.diesel_p_mw * 1000.0),
                    realized_pv_used_kw=max(0.0, state.pv_used_mw * 1000.0),
                    unserved_load_kw=max(0.0, state.unserved_mw * 1000.0),
                    terminated=terminated,
                    truncated=truncated,
                )
                results.append(result)
                await self.dispatch.record_result(result)
                if on_event is not None:
                    await on_event(
                        {
                            "type": "ems_tick",
                            "schema_version": "1.0",
                            "telemetry": plant_observation.model_dump(mode="json"),
                            "command": command.model_dump(mode="json") if command else None,
                            "result": result.model_dump(mode="json"),
                        }
                    )
                    await on_event({"type": "row", "row": row})
                if terminated or truncated:
                    break
                if self.pace_seconds:
                    await asyncio.sleep(self.pace_seconds)
            if on_event is not None:
                await on_event(
                    {
                        "type": "end",
                        "totals": _totals(rows, env_settings.topology.timestep_hours),
                        "meta": {
                            "steps": len(rows),
                            "diesel_starts": self.env.backend.diesel.starts,
                            "diesel_runtime_hours": self.env.backend.diesel.runtime_hours,
                            **{
                                key: self.env._forecast_meta().get(key)  # noqa: SLF001
                                for key in (
                                    "forecast_available",
                                    "forecast_error",
                                    "forecast_source",
                                    "forecast_requested_source",
                                    "forecast_context_time",
                                    "forecast_context_steps",
                                    "forecast_stale",
                                    "forecast_age_steps",
                                    "forecast_cold_start",
                                    "forecast_covariate_mode",
                                )
                            },
                        },
                    }
                )
                await on_event(
                    {
                        "type": "ems_end",
                        "schema_version": "1.0",
                        "session_id": self.session_id,
                        "ticks": len(results),
                    }
                )
                await on_event(
                    {
                        "type": "ems_metrics",
                        "schema_version": "1.0",
                        "session_id": self.session_id,
                        "ticks": len(results),
                        "deadline_overruns": self.deadline_overruns,
                        "recoveries": self.recoveries,
                    }
                )
            return results
        finally:
            self.env.close()
