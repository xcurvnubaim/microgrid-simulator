"""Stateful pull service over strict, timestamp-aligned telemetry replay windows."""

from __future__ import annotations

from dataclasses import dataclass

from microgrid_simulator.config import Settings
from microgrid_simulator.contracts import (
    TelemetrySample,
    TelemetrySession,
    TelemetrySessionRequest,
    TelemetryWindowPayload,
)
from microgrid_simulator.digital_twin.replay import load_fixed_telemetry_window


@dataclass
class _SessionState:
    payload: TelemetryWindowPayload
    cursor: int = 0


class ReplayTelemetryService:
    """Own telemetry files and expose immutable windows plus a deterministic cursor."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sessions: dict[str, _SessionState] = {}

    def create_session(self, request: TelemetrySessionRequest) -> TelemetrySession:
        n_steps = request.n_steps or int(
            round(self.settings.episode.horizon_hours / self.settings.topology.timestep_hours)
        )
        settings = self.settings
        if request.start_at is not None:
            episode = settings.episode.model_copy(update={"telemetry_start": request.start_at})
            settings = settings.model_copy(update={"episode": episode})
        window = load_fixed_telemetry_window(settings, n_steps)
        if window is None:
            raise ValueError("telemetry service requires episode.telemetry_start")
        source_id = settings.forecast.source_id or settings.scenario.name
        session = TelemetrySession(
            source_id=source_id,
            n_steps=n_steps,
            sample_count=n_steps + 1,
            context_time=str(window.context_timestamp),
            first_evaluated_time=str(window.first_evaluated_timestamp),
            last_evaluated_time=str(window.last_evaluated_timestamp),
            timestamps_are_observed=window.timestamps_are_observed,
        )
        samples = [
            TelemetrySample(
                source_id=source_id,
                session_id=session.session_id,
                sequence_id=index,
                observed_at=str(timestamp),
                pv_available_kw=float(window.pv_mw[index] * 1000.0),
                load_demand_kw=float(window.demand_mw[index] * 1000.0),
            )
            for index, timestamp in enumerate(window.timestamps)
        ]
        payload = TelemetryWindowPayload(
            session=session,
            samples=samples,
            source_files=window.source_files,
        )
        self._sessions[session.session_id] = _SessionState(payload)
        return session

    def get_window(self, session_id: str) -> TelemetryWindowPayload:
        return self._state(session_id).payload

    def next_sample(self, session_id: str) -> TelemetrySample | None:
        state = self._state(session_id)
        if state.cursor >= len(state.payload.samples):
            return None
        sample = state.payload.samples[state.cursor]
        state.cursor += 1
        return sample

    def _state(self, session_id: str) -> _SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown telemetry session {session_id}") from exc
