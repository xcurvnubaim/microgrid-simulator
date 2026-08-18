"""Wall-clock-capable simulator adapter for the EMS event contract."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from microgrid_simulator.config import Settings
from microgrid_simulator.controllers import RuleBasedController
from microgrid_simulator.ems.broker import EventBroker
from microgrid_simulator.ems.types import DispatchCommand, DispatchResult, TelemetryFrame
from microgrid_simulator.env import MicrogridEnv

ResultCallback = Callable[[TelemetryFrame, DispatchCommand | None, DispatchResult], Awaitable[None]]


class SimulatorBridge:
    """Publish canonical observations and apply only fresh, matching commands."""

    def __init__(
        self,
        settings: Settings,
        broker: EventBroker,
        *,
        command_timeout_ms: int = 1000,
        pace_seconds: float = 0.0,
        session_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.command_timeout_ms = command_timeout_ms
        self.pace_seconds = max(0.0, pace_seconds)
        self.session_id = session_id or str(uuid4())
        self.env = MicrogridEnv(settings=settings)
        self.fallback = RuleBasedController(settings)

    def _frame(
        self, sequence: int, observation: np.ndarray, info: dict[str, Any]
    ) -> TelemetryFrame:
        state = self.env._last_state  # noqa: SLF001
        forecast_available = bool(info.get("forecast_available", False))
        return TelemetryFrame(
            session_id=self.session_id,
            sequence_id=sequence,
            source_id=self.settings.scenario.name,
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
        )

    def _fallback_action(self) -> np.ndarray:
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
    def _command_is_valid(command: DispatchCommand, frame: TelemetryFrame) -> bool:
        now = datetime.now(timezone.utc)
        return (
            command.session_id == frame.session_id
            and command.telemetry_sequence_id == frame.sequence_id
            and command.expires_at >= now
            and len(command.normalized_action) == frame.action_shape
            and all(np.isfinite(command.normalized_action))
        )

    async def _receive_matching_command(
        self, frame: TelemetryFrame
    ) -> tuple[DispatchCommand | None, DispatchCommand | None]:
        """Wait once per tick deadline, discarding commands for older frames.

        A command that arrives after its tick timed out can otherwise remain at
        the head of the bounded queue. Consuming only that stale command would
        keep the bridge one tick behind indefinitely, even when the EMS has
        recovered. Return the accepted command and the last rejected candidate
        so the caller can distinguish a clean timeout from a protocol rejection.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.command_timeout_ms / 1000.0
        rejected: DispatchCommand | None = None
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                return None, rejected
            try:
                candidate = await asyncio.wait_for(self.broker.receive_command(), timeout=remaining)
            except TimeoutError:
                return None, rejected
            if self._command_is_valid(candidate, frame):
                return candidate, rejected
            rejected = candidate

    async def run(
        self,
        *,
        max_steps: int | None = None,
        seed: int = 0,
        on_result: ResultCallback | None = None,
    ) -> list[DispatchResult]:
        observation, info = self.env.reset(seed=seed)
        results: list[DispatchResult] = []
        limit = min(max_steps or self.env.max_steps, self.env.max_steps)
        try:
            for sequence in range(limit):
                frame = self._frame(sequence, observation, info)
                await self.broker.publish_telemetry(frame)
                command: DispatchCommand | None = None
                rejected_command: DispatchCommand | None = None
                status: Literal["applied", "fallback", "rejected"] = "fallback"
                message = "command timeout; rule fallback applied"
                command, rejected_command = await self._receive_matching_command(frame)
                if command is not None:
                    action = np.asarray(command.normalized_action, dtype=np.float32)
                    status = "fallback" if command.fallback_active else "applied"
                    message = "EMS command applied"
                else:
                    action = self._fallback_action()
                    if rejected_command is not None:
                        status = "rejected"
                        message = (
                            "stale or mismatched command rejected; "
                            "deadline expired; rule fallback applied"
                        )

                observation, _, terminated, truncated, info = await asyncio.to_thread(
                    self.env.step, action
                )
                state = self.env._last_state  # noqa: SLF001
                result = DispatchResult(
                    session_id=self.session_id,
                    telemetry_sequence_id=sequence,
                    command_id=(
                        command.command_id
                        if command is not None
                        else rejected_command.command_id
                        if rejected_command is not None
                        else None
                    ),
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
                await self.broker.publish_result(result)
                if on_result is not None:
                    await on_result(frame, command, result)
                if terminated or truncated:
                    break
                if self.pace_seconds:
                    await asyncio.sleep(self.pace_seconds)
        finally:
            self.env.close()
        return results
