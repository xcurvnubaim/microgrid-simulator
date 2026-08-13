"""EMS-owned scenario clock and episode lifecycle."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Literal
from uuid import uuid4

import pandas as pd

from microgrid_simulator.config import Settings
from microgrid_simulator.contracts import (
    PlantFrame,
    PlantObservation,
    PlantPhysicalState,
    PlantSessionStartRequest,
    PlantStepRequest,
    TelemetrySessionRequest,
)
from microgrid_simulator.core.types import GridState
from microgrid_simulator.ems.forecast import EMSForecastRuntime
from microgrid_simulator.ems.service import EMSService
from microgrid_simulator.messaging.nats import DashboardTracePublisher
from microgrid_simulator.plant.providers import NatsPlantProvider
from microgrid_simulator.rl.env import build_observation
from microgrid_simulator.runtime import settings_fingerprint
from microgrid_simulator.simulator.providers import NatsTelemetryProvider
from microgrid_simulator.ui.rollout import _totals

RunState = Literal[
    "initializing",
    "running",
    "awaiting_continuation",
    "completed",
    "cancelling",
    "cancelled",
    "failed",
]


class EMSRun:
    def __init__(self, manager: EMSRunManager, run_id: str, seed: int) -> None:
        self.manager = manager
        self.run_id = run_id
        self.seed = seed
        self.state: RunState = "initializing"
        self.episode_index = 0
        self.steps = 0
        self.total_steps = 0
        self.error: str | None = None
        self.start_at = manager.settings.episode.telemetry_start
        self.next_start_at: str | None = None
        self.carried_state: PlantPhysicalState | None = None
        self.rows: list[dict[str, Any]] = []
        self.episode_metrics: list[dict[str, Any]] = []
        self.cancel_requested = False
        self.task: asyncio.Task[None] | None = None
        self.publisher = DashboardTracePublisher(manager.js, run_id, producer="ems")
        self.forecast = EMSForecastRuntime(manager.settings)
        self._plant_start: Any = None

    def status(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "episode_index": self.episode_index,
            "steps": self.steps,
            "expected_steps": self.manager.steps_per_episode,
            "total_steps": self.total_steps,
            "start_at": self.start_at,
            "next_start_at": self.next_start_at,
            "policy": self.manager.ems.policy,
            "policy_version": self.manager.ems.policy_version,
            "error": self.error,
            "episode_metrics": self.episode_metrics,
            "boundary": "simulation-only",
        }

    def _observation(self, frame: PlantFrame) -> PlantObservation:
        state_data = dict(frame.state)
        state_data["violations"] = []
        state = GridState(**state_data)
        forecast = self.forecast.view(self.steps)
        available = forecast.available
        pv = forecast.pv_mw
        demand = forecast.demand_mw
        canonical = build_observation(
            state,
            self.manager.settings.diesel.enabled,
            pv_forecast_mw=pv if available else None,
            demand_forecast_mw=demand if available else None,
            forecast_available=available,
            initial_soc=self.carried_state.battery_soc if self.carried_state else None,
            target_soc=self.carried_state.battery_soc if self.carried_state else None,
            episode_progress=self.steps / max(1, self.manager.steps_per_episode),
            forecast_horizon=self.manager.settings.forecast.forecast_steps,
        )
        return PlantObservation(
            session_id=self.run_id,
            sequence_id=self.steps,
            source_id=frame.source_id,
            observed_at=frame.observed_at,
            simulation_time_hours=frame.simulation_time_hours,
            timestep_hours=self.manager.settings.topology.timestep_hours,
            observation=canonical.tolist(),
            observation_shape=int(canonical.size),
            action_shape=frame.action_shape,
            pv_available_kw=max(0.0, state.pv_available_mw * 1000.0),
            pv_used_kw=max(0.0, state.pv_used_mw * 1000.0),
            load_demand_kw=max(0.0, state.load_demand_mw * 1000.0),
            load_served_kw=max(0.0, state.load_served_mw * 1000.0),
            battery_soc=frame.physical_state.battery_soc,
            battery_soh=frame.physical_state.battery_soh,
            battery_power_kw=state.battery_p_mw * 1000.0,
            diesel_on=state.diesel_on,
            diesel_power_kw=max(0.0, state.diesel_p_mw * 1000.0),
            grid_connected=not state.islanded,
            grid_import_kw=state.grid_import_mw * 1000.0,
            forecast_available=available,
            forecast_pv_kw=(
                forecast.current_pv_mw * 1000.0 if forecast.current_pv_mw is not None else None
            ),
            forecast_demand_kw=(
                forecast.current_demand_mw * 1000.0
                if forecast.current_demand_mw is not None
                else None
            ),
            data_quality="good" if state.solver_ok else "invalid",
            constraints={
                "battery_soc_min": self.manager.settings.battery.soc_min,
                "battery_soc_max": self.manager.settings.battery.soc_max,
                "battery_max_charge_kw": self.manager.settings.battery.max_charge_mw * 1000,
                "battery_max_discharge_kw": self.manager.settings.battery.max_discharge_mw * 1000,
                "diesel_max_kw": self.manager.settings.diesel.max_kw,
            },
        )

    async def initialize_episode(self) -> None:
        payload = await self.manager.telemetry.window(
            TelemetrySessionRequest(start_at=self.start_at, n_steps=self.manager.steps_per_episode)
        )
        expected_source = (
            self.manager.settings.forecast.source_id or self.manager.settings.scenario.name
        )
        if payload.session.source_id != expected_source:
            raise ValueError("telemetry source does not match the configured EMS scenario")
        self._plant_start = await self.manager.plant.start(
            PlantSessionStartRequest(
                session_id=self.run_id,
                episode_index=self.episode_index,
                seed=self.seed,
                config_fingerprint=self.manager.config_fingerprint,
                telemetry=payload,
                carried_state=self.carried_state,
            )
        )
        self.forecast.reset(payload)
        observation = self._observation(self._plant_start.frame)
        self.manager.ems.validate_contract(observation.observation_shape, observation.action_shape)
        first = pd.Timestamp(payload.session.first_evaluated_time)
        self.next_start_at = str(
            first.to_pydatetime()
            + timedelta(hours=float(self.manager.settings.topology.timestep_hours))
        )
        self.steps = 0
        self.rows = []

    async def run_episode(self) -> None:
        try:
            if self.episode_index == 0:
                await self.publisher.publish(
                    {
                        "type": "ems_start",
                        "schema_version": "1.0",
                        "session_id": self.run_id,
                        "boundary": "simulation-only",
                        "policy": self.manager.ems.policy,
                    }
                )
            await self.publisher.publish(
                {
                    "type": "episode_start",
                    "episode_index": self.episode_index,
                    "start_at": self.start_at,
                    "expected_steps": self.manager.steps_per_episode,
                }
            )
            await self.publisher.publish(
                {
                    "type": "meta",
                    "meta": {
                        **self._plant_start.meta,
                        "expected_steps": self.manager.steps_per_episode,
                        "ems_session_id": self.run_id,
                        "episode_index": self.episode_index,
                        "external_policy": self.manager.ems.policy,
                        **self.forecast.view(self.steps).meta,
                    },
                }
            )
            self.state = "running"
            frame = self._plant_start.frame
            while self.steps < self.manager.steps_per_episode:
                if self.cancel_requested:
                    self.state = "cancelled"
                    await self.publisher.publish(
                        {"type": "ems_cancelled", "episode_index": self.episode_index}
                    )
                    await self.manager.plant.close(self.run_id)
                    return
                observation = self._observation(frame)
                command = await asyncio.to_thread(
                    self.manager.ems.build_command, observation.to_telemetry_frame()
                )
                response = await self.manager.plant.step(
                    PlantStepRequest(
                        session_id=self.run_id,
                        episode_index=self.episode_index,
                        sequence_id=self.steps,
                        command=command,
                    )
                )
                self.rows.append(response.row)
                self.steps += 1
                self.total_steps += 1
                frame = response.frame
                self.carried_state = frame.physical_state
                await self.publisher.publish(
                    {
                        "type": "ems_tick",
                        "schema_version": "1.0",
                        "episode_index": self.episode_index,
                        "telemetry": observation.model_dump(mode="json"),
                        "command": command.model_dump(mode="json"),
                        "result": response.result.model_dump(mode="json"),
                    }
                )
                await self.publisher.publish({"type": "row", "row": response.row})
                if response.terminated or response.truncated:
                    break
                if self.manager.pace_seconds:
                    await asyncio.sleep(self.manager.pace_seconds)
            metrics = _totals(self.rows, self.manager.settings.topology.timestep_hours)
            metrics_record = {
                "episode_index": self.episode_index,
                "start_at": self.start_at,
                "steps": self.steps,
                "totals": metrics,
            }
            self.episode_metrics.append(metrics_record)
            await self.publisher.publish({"type": "episode_end", **metrics_record})
            self.state = "awaiting_continuation"
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.state = "failed"
            await self.publisher.publish(
                {"type": "ems_failed", "error": self.error, "episode_index": self.episode_index}
            )
            await self.manager.plant.close(self.run_id)

    async def finish(self) -> None:
        if self.state not in {"awaiting_continuation", "cancelled", "failed"}:
            raise ValueError(f"cannot finish a run in state {self.state}")
        if self.state == "awaiting_continuation":
            self.state = "completed"
            await self.publisher.publish(
                {
                    "type": "end",
                    "totals": self.episode_metrics[-1]["totals"] if self.episode_metrics else {},
                    "meta": {"episodes": len(self.episode_metrics), "steps": self.total_steps},
                }
            )
            await self.publisher.publish(
                {
                    "type": "ems_end",
                    "session_id": self.run_id,
                    "episodes": len(self.episode_metrics),
                    "steps": self.total_steps,
                }
            )
        await self.manager.plant.close(self.run_id)


class EMSRunManager:
    def __init__(
        self,
        settings: Settings,
        ems: EMSService,
        telemetry: NatsTelemetryProvider,
        plant: NatsPlantProvider,
        js: Any,
        *,
        pace_seconds: float = 0.0,
    ) -> None:
        self.settings = settings
        self.ems = ems
        self.telemetry = telemetry
        self.plant = plant
        self.js = js
        self.pace_seconds = max(0.0, pace_seconds)
        self.config_fingerprint = settings_fingerprint(settings)
        self.steps_per_episode = int(
            round(settings.episode.horizon_hours / settings.topology.timestep_hours)
        )
        self.runs: dict[str, EMSRun] = {}
        self.latest_id: str | None = None
        self.lock = asyncio.Lock()

    async def start(self, seed: int = 0) -> EMSRun:
        async with self.lock:
            active = [
                run
                for run in self.runs.values()
                if run.state in {"initializing", "running", "cancelling"}
            ]
            if active:
                raise ValueError("an EMS run is already active")
            run = EMSRun(self, str(uuid4()), seed)
            self.runs[run.run_id] = run
            self.latest_id = run.run_id
            try:
                await run.initialize_episode()
            except Exception:
                self.runs.pop(run.run_id, None)
                self.latest_id = next(reversed(self.runs), None)
                raise
            run.task = asyncio.create_task(run.run_episode())
            return run

    def get(self, run_id: str) -> EMSRun:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise KeyError("unknown EMS run") from exc

    async def continue_run(self, run_id: str) -> EMSRun:
        run = self.get(run_id)
        if run.state != "awaiting_continuation":
            raise ValueError(f"cannot continue a run in state {run.state}")
        run.episode_index += 1
        run.start_at = run.next_start_at
        run.state = "initializing"
        await run.initialize_episode()
        run.task = asyncio.create_task(run.run_episode())
        return run

    async def cancel(self, run_id: str) -> EMSRun:
        run = self.get(run_id)
        if run.state in {"completed", "cancelled", "failed"}:
            return run
        if run.state == "awaiting_continuation":
            run.cancel_requested = True
            run.state = "cancelled"
            await run.publisher.publish(
                {"type": "ems_cancelled", "episode_index": run.episode_index}
            )
            await self.plant.close(run.run_id)
            return run
        run.cancel_requested = True
        run.state = "cancelling"
        return run
