"""Authoritative, transport-neutral plant sessions."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

import numpy as np
import pandas as pd

from microgrid_simulator.config import Settings
from microgrid_simulator.contracts import (
    PlantFrame,
    PlantPhysicalState,
    PlantSessionCloseResponse,
    PlantSessionStartRequest,
    PlantSessionStartResponse,
    PlantStepRequest,
    PlantStepResponse,
)
from microgrid_simulator.digital_twin.replay import TelemetryWindow
from microgrid_simulator.ems.types import DispatchResult
from microgrid_simulator.env import MicrogridEnv
from microgrid_simulator.ui.rollout import (
    _dispatch_trace,
    _episode_meta,
    _slack_bus_id,
    _step_row,
)


class _Session:
    def __init__(self, settings: Settings, request: PlantSessionStartRequest) -> None:
        self.settings = settings
        self.request = request
        self.sequence = 0
        self.responses: dict[int, tuple[str, PlantStepResponse]] = {}
        payload = request.telemetry
        window = TelemetryWindow(
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
        episode_settings = settings.model_copy(
            update={
                "episode": settings.episode.model_copy(
                    update={
                        "horizon_hours": payload.session.n_steps * settings.topology.timestep_hours,
                        "telemetry_start": None,
                    }
                ),
                # Forecast context belongs to the EMS orchestration service.
                # The plant keeps the fixed observation dimensions but never
                # contacts a forecast service or loads a forecast cache.
                "forecast": settings.forecast.model_copy(update={"enabled": False}),
            }
        )
        self.env = MicrogridEnv(settings=episode_settings, telemetry_window=window)
        observation, info = self.env.reset(seed=request.seed)
        if request.carried_state is not None:
            self._restore(request.carried_state)
            observation = self.env._build_obs(self.env._last_state)  # noqa: SLF001
        self.observation = observation
        self.info = info
        self.slack_id = _slack_bus_id(episode_settings)
        self.meta = _episode_meta(
            episode_settings, self.env, "ems", request.seed, info, self.slack_id
        )

    def _restore(self, state: PlantPhysicalState) -> None:
        battery = self.env.backend.battery
        battery.soc = state.battery_soc
        battery.soh = state.battery_soh
        battery.throughput_mwh = state.battery_throughput_mwh
        diesel = self.env.backend.diesel
        diesel.is_on = state.diesel_on
        diesel.p_mw = state.diesel_power_mw
        diesel.p_end_mw = state.diesel_power_mw
        diesel.runtime_hours = state.diesel_runtime_hours
        diesel.starts = state.diesel_starts
        diesel.hours_in_state = state.diesel_hours_in_state
        current = self.env._last_state  # noqa: SLF001
        current.soc = [state.battery_soc]
        current.soh = [state.battery_soh]
        current.diesel_on = state.diesel_on
        current.diesel_p_mw = state.diesel_power_mw
        current.diesel_starts = state.diesel_starts

    def physical_state(self) -> PlantPhysicalState:
        battery = self.env.backend.battery
        diesel = self.env.backend.diesel
        return PlantPhysicalState(
            battery_soc=float(battery.soc),
            battery_soh=float(battery.soh),
            battery_throughput_mwh=float(battery.throughput_mwh),
            diesel_on=bool(diesel.is_on),
            diesel_power_mw=max(0.0, float(diesel.p_mw)),
            diesel_runtime_hours=float(diesel.runtime_hours),
            diesel_starts=int(diesel.starts),
            diesel_hours_in_state=min(float(diesel.hours_in_state), 1.0e12),
        )

    def frame(self) -> PlantFrame:
        state = self.env._last_state  # noqa: SLF001
        sample_index = min(self.sequence, len(self.request.telemetry.samples) - 1)
        sample = self.request.telemetry.samples[sample_index]
        observed_at = pd.Timestamp(sample.observed_at)
        if observed_at.tzinfo is None:
            observed_at = observed_at.tz_localize("UTC")
        return PlantFrame(
            sequence_id=self.sequence,
            source_id=self.request.telemetry.session.source_id,
            observed_at=observed_at.to_pydatetime(),
            simulation_time_hours=max(0.0, float(state.timestamp)),
            observation=np.asarray(self.observation, dtype=float).tolist(),
            action_shape=self.env.action_dim,
            state=asdict(state),
            info=self.info,
            physical_state=self.physical_state(),
        )

    def step(self, request: PlantStepRequest) -> PlantStepResponse:
        cached = self.responses.get(request.sequence_id)
        if cached is not None:
            if cached[0] != request.command.command_id:
                raise ValueError("sequence already applied with a different command")
            return cached[1]
        if request.episode_index != self.request.episode_index:
            raise ValueError("plant episode mismatch")
        if request.sequence_id != self.sequence:
            raise ValueError(f"expected plant sequence {self.sequence}")
        if request.command.session_id != request.session_id:
            raise ValueError("command session mismatch")
        if request.command.telemetry_sequence_id != request.sequence_id:
            raise ValueError("command sequence mismatch")
        action = np.asarray(request.command.normalized_action, dtype=np.float32)
        if action.shape != (self.env.action_dim,) or not np.all(np.isfinite(action)):
            raise ValueError("invalid plant action")
        trace = _dispatch_trace(self.env, request.command.controller, action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        row = _step_row(
            self.env.settings,
            self.env,
            request.sequence_id,
            reward,
            info,
            self.slack_id,
            trace,
        )
        state = self.env._last_state  # noqa: SLF001
        result = DispatchResult(
            session_id=request.session_id,
            telemetry_sequence_id=request.sequence_id,
            command_id=request.command.command_id,
            status="fallback" if request.command.fallback_active else "applied",
            status_message="EMS command applied by headless plant",
            realized_battery_power_kw=state.battery_p_mw * 1000.0,
            realized_diesel_on=state.diesel_on,
            realized_diesel_power_kw=max(0.0, state.diesel_p_mw * 1000.0),
            realized_pv_used_kw=max(0.0, state.pv_used_mw * 1000.0),
            unserved_load_kw=max(0.0, state.unserved_mw * 1000.0),
            terminated=terminated,
            truncated=truncated,
        )
        self.sequence += 1
        self.observation = observation
        self.info = info
        response = PlantStepResponse(
            session_id=request.session_id,
            episode_index=request.episode_index,
            sequence_id=request.sequence_id,
            frame=self.frame(),
            result=result,
            reward=reward,
            row=row,
            terminated=terminated,
            truncated=truncated,
        )
        self.responses[request.sequence_id] = (request.command.command_id, response)
        return response


class PlantService:
    """Own plant physics and idempotent state transitions, never orchestration."""

    def __init__(self, settings: Settings, *, config_fingerprint: str) -> None:
        self.settings = settings
        self.config_fingerprint = config_fingerprint
        self.sessions: dict[str, _Session] = {}

    async def start(self, request: PlantSessionStartRequest) -> PlantSessionStartResponse:
        if request.config_fingerprint != self.config_fingerprint:
            raise ValueError("EMS and plant configuration fingerprints differ")
        old = self.sessions.pop(request.session_id, None)
        if old is not None:
            await asyncio.to_thread(old.env.close)
        session = await asyncio.to_thread(_Session, self.settings, request)
        self.sessions[request.session_id] = session
        return PlantSessionStartResponse(
            session_id=request.session_id,
            episode_index=request.episode_index,
            frame=session.frame(),
            meta=session.meta,
        )

    async def step(self, request: PlantStepRequest) -> PlantStepResponse:
        session = self.sessions.get(request.session_id)
        if session is None:
            raise KeyError("unknown plant session")
        return await asyncio.to_thread(session.step, request)

    async def close(self, session_id: str) -> PlantSessionCloseResponse:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            await asyncio.to_thread(session.env.close)
        return PlantSessionCloseResponse(session_id=session_id, closed=session is not None)
