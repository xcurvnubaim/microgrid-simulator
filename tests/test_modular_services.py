from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from microgrid_simulator.config import Settings
from microgrid_simulator.contracts import (
    PlantObservation,
    PlantSessionStartRequest,
    PlantStepRequest,
    TelemetrySample,
    TelemetrySessionRequest,
)
from microgrid_simulator.ems.forecast import EMSForecastRuntime
from microgrid_simulator.ems.orchestrator import EMSRunManager
from microgrid_simulator.ems.service import EMSService
from microgrid_simulator.forecast import ForecastSnapshot
from microgrid_simulator.plant.service import PlantService
from microgrid_simulator.runtime import settings_fingerprint
from microgrid_simulator.simulator.orchestrator import SimulatorOrchestrator
from microgrid_simulator.simulator.providers import LocalDispatchProvider, LocalTelemetryProvider
from microgrid_simulator.telemetry.service import ReplayTelemetryService


def _settings(tmp_path, *, horizon_hours: float = 1.0) -> Settings:
    timestamps = pd.date_range("2026-01-14 23:45", periods=8, freq="15min")
    load_path = tmp_path / "load.csv"
    pv_path = tmp_path / "pv.csv"
    pd.DataFrame({"ts": timestamps, "kw": np.arange(8) + 100.0}).to_csv(load_path, index=False)
    pd.DataFrame({"ts": timestamps, "kw": np.arange(8) * 10.0}).to_csv(pv_path, index=False)
    raw = Settings().model_dump()
    raw["topology"] |= {"timestep_hours": 0.25, "n_ev": 0}
    raw["episode"] |= {
        "horizon_hours": horizon_hours,
        "telemetry_start": "2026-01-15 00:00:00",
    }
    raw["forecast"]["enabled"] = False
    raw["demand"] |= {"enabled": False, "file": None, "random_window": False}
    raw["digital_twin"] = {
        "fill_strategy": "error",
        "max_gap_steps": 0,
        "max_missing_fraction": 0.0,
        "measurements": {
            "load": {
                "file": str(load_path),
                "timestamp_column": "ts",
                "value_column": "kw",
                "unit": "kw",
            },
            "pv": {
                "file": str(pv_path),
                "timestamp_column": "ts",
                "value_column": "kw",
                "unit": "kw",
            },
        },
    }
    return Settings(**raw)


def _observation(settings: Settings) -> PlantObservation:
    return PlantObservation(
        session_id="session",
        sequence_id=0,
        source_id=settings.scenario.name,
        observed_at=datetime.now(timezone.utc),
        simulation_time_hours=0.0,
        timestep_hours=settings.topology.timestep_hours,
        observation=[0.0],
        observation_shape=1,
        action_shape=3,
        pv_available_kw=0.0,
        pv_used_kw=0.0,
        load_demand_kw=100.0,
        load_served_kw=100.0,
        battery_soc=settings.battery.soc_init,
        battery_soh=1.0,
        battery_power_kw=0.0,
        diesel_on=False,
        diesel_power_kw=0.0,
        grid_connected=False,
        grid_import_kw=0.0,
    )


def test_contracts_reject_simulator_fields_in_raw_telemetry() -> None:
    with pytest.raises(ValidationError):
        TelemetrySample(
            source_id="scenario",
            session_id="session",
            sequence_id=0,
            observed_at="2026-01-15 00:00:00",
            pv_available_kw=10.0,
            load_demand_kw=20.0,
            grid_import_kw=5.0,
        )


def test_telemetry_service_exposes_immutable_window_and_pull_cursor(tmp_path) -> None:
    service = ReplayTelemetryService(_settings(tmp_path))
    session = service.create_session(TelemetrySessionRequest(n_steps=4))
    window = service.get_window(session.session_id)

    assert len(window.samples) == 5
    assert [sample.sequence_id for sample in window.samples] == list(range(5))
    assert window.samples[0].load_demand_kw == pytest.approx(100.0)
    assert service.next_sample(session.session_id) == window.samples[0]
    assert service.next_sample(session.session_id) == window.samples[1]


def test_ems_domain_uses_transport_neutral_observation(tmp_path) -> None:
    settings = _settings(tmp_path)
    observation = _observation(settings)
    command = EMSService(settings, policy="rule").build_command(observation.to_telemetry_frame())

    assert command.session_id == "session"
    assert command.telemetry_sequence_id == 0


def test_headless_plant_step_is_idempotent_and_returns_carried_state(tmp_path) -> None:
    async def exercise() -> None:
        settings = _settings(tmp_path)
        telemetry = ReplayTelemetryService(settings)
        telemetry_session = telemetry.create_session(TelemetrySessionRequest(n_steps=4))
        payload = telemetry.get_window(telemetry_session.session_id)
        fingerprint = settings_fingerprint(settings)
        plant = PlantService(settings, config_fingerprint=fingerprint)
        started = await plant.start(
            PlantSessionStartRequest(
                session_id="run-1",
                episode_index=0,
                seed=0,
                config_fingerprint=fingerprint,
                telemetry=payload,
            )
        )
        observation = _observation(settings).model_copy(
            update={
                "session_id": "run-1",
                "observation": started.frame.observation,
                "observation_shape": len(started.frame.observation),
                "action_shape": started.frame.action_shape,
            }
        )
        command = EMSService(settings).build_command(observation.to_telemetry_frame())
        request = PlantStepRequest(
            session_id="run-1", episode_index=0, sequence_id=0, command=command
        )

        first = await plant.step(request)
        repeated = await plant.step(request)

        assert first == repeated
        assert first.frame.sequence_id == 1
        assert first.frame.physical_state.battery_soc >= settings.battery.soc_min
        assert first.result.command_id == command.command_id

    asyncio.run(exercise())


def test_distributed_forecast_context_is_owned_and_refreshed_by_ems(tmp_path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={
            "forecast": _settings(tmp_path).forecast.model_copy(
                update={"enabled": True, "strict_cache": False, "refresh_each_step": True}
            )
        }
    )
    telemetry = ReplayTelemetryService(settings)
    session = telemetry.create_session(TelemetrySessionRequest(n_steps=4))
    payload = telemetry.get_window(session.session_id)
    runtime = EMSForecastRuntime(settings)
    calls: list[int] = []

    class ForecastClient:
        def fetch(self, context, issued_at=None):  # noqa: ANN001
            calls.append(len(context.pv_values_mw))
            return ForecastSnapshot(
                issued_at=str(issued_at),
                horizon_hours=settings.forecast.horizon_hours,
                frequency_hours=1.0,
                model_version="test",
                pv_target=settings.forecast.target,
                demand_target=settings.forecast.demand_target,
                timestamps=(),
                pv_values_mw=(0.1,) * settings.forecast.horizon_hours,
                demand_values_mw=(0.2,) * settings.forecast.horizon_hours,
                source_id=settings.forecast.source_id or settings.scenario.name,
            )

    runtime.client = ForecastClient()  # type: ignore[assignment]
    runtime.reset(payload)

    first = runtime.view(0)
    repeated = runtime.view(0)
    second = runtime.view(1)

    assert calls == [1, 2]
    assert first.available and repeated.available and second.available
    assert first.current_pv_mw == pytest.approx(0.1)


def test_ems_manager_owns_episode_lifecycle_and_shifted_continuation(tmp_path) -> None:
    async def exercise() -> None:
        settings = _settings(tmp_path)
        telemetry = LocalTelemetryProvider(ReplayTelemetryService(settings))
        plant_service = PlantService(settings, config_fingerprint=settings_fingerprint(settings))

        class LocalPlant:
            async def start(self, request):  # noqa: ANN001
                return await plant_service.start(request)

            async def step(self, request):  # noqa: ANN001
                return await plant_service.step(request)

            async def close(self, session_id):  # noqa: ANN001
                return await plant_service.close(session_id)

        class Ack:
            stream = "MGS_DASHBOARD"
            seq = 1

        class FakeJetStream:
            async def publish(self, subject, payload, headers):  # noqa: ANN001
                del subject, payload, headers
                return Ack()

        manager = EMSRunManager(
            settings,
            EMSService(settings),
            telemetry,  # type: ignore[arg-type]
            LocalPlant(),  # type: ignore[arg-type]
            FakeJetStream(),
        )
        first = await manager.start(seed=0)
        assert first.task is not None
        await first.task

        assert first.state == "awaiting_continuation"
        assert first.steps == 4
        original_start = pd.Timestamp(first.start_at)
        shifted_start = pd.Timestamp(first.next_start_at)
        assert (shifted_start - original_start).total_seconds() == 15 * 60
        carried_soc = first.carried_state.battery_soc

        continued = await manager.continue_run(first.run_id)
        assert continued.task is not None
        await continued.task

        assert continued.state == "awaiting_continuation"
        assert continued.episode_index == 1
        assert continued.total_steps == 8
        assert continued.episode_metrics[1]["start_at"] == continued.start_at
        assert carried_soc >= settings.battery.soc_min
        await continued.finish()
        assert continued.state == "completed"

    asyncio.run(exercise())


def test_local_modular_orchestrator_runs_one_step_per_sequence(tmp_path) -> None:
    async def exercise() -> None:
        settings = _settings(tmp_path)
        telemetry = LocalTelemetryProvider(ReplayTelemetryService(settings))
        dispatch = LocalDispatchProvider(EMSService(settings, policy="rule"))
        orchestrator = SimulatorOrchestrator(
            settings,
            telemetry,
            dispatch,
            command_timeout_ms=500,
        )
        events: list[dict[str, object]] = []

        async def capture(event: dict[str, object]) -> None:
            events.append(event)

        results = await orchestrator.run(max_steps=2, seed=0, on_event=capture)

        assert [result.telemetry_sequence_id for result in results] == [0, 1]
        assert all(result.status == "applied" for result in results)
        assert dispatch.results == results
        assert [event["type"] for event in events] == [
            "ems_start",
            "meta",
            "ems_tick",
            "row",
            "ems_tick",
            "row",
            "end",
            "ems_end",
            "ems_metrics",
        ]
        meta = events[1]["meta"]
        assert isinstance(meta, dict)
        assert meta["expected_steps"] == 2
        rows = [event["row"] for event in events if event["type"] == "row"]
        assert all(isinstance(row, dict) for row in rows)
        assert [row["step"] for row in rows] == [1, 2]
        assert all("soc_pct" in row and "per_bus" in row for row in rows)
        end = events[-3]
        assert end["totals"]["load_kwh"] > 0.0
        assert end["meta"]["steps"] == 2
        metrics = events[-1]
        assert metrics["type"] == "ems_metrics"
        assert metrics["ticks"] == 2
        assert metrics["deadline_overruns"] == 0
        assert metrics["recoveries"] == 0

    asyncio.run(exercise())


def test_orchestrator_counts_deadline_overrun_and_recovery(tmp_path) -> None:
    async def exercise() -> None:
        settings = _settings(tmp_path, horizon_hours=1.0)
        telemetry = LocalTelemetryProvider(ReplayTelemetryService(settings))
        dispatch = LocalDispatchProvider(EMSService(settings, policy="rule"))

        class FlakyDispatch(LocalDispatchProvider):
            def __init__(self, service: object) -> None:
                super().__init__(service)  # type: ignore[arg-type]
                self._calls = 0

            async def dispatch(self, observation: PlantObservation):  # noqa: ANN001
                # Timed out on the first call (deadline overrun), then recovers.
                self._calls += 1
                if self._calls == 1:
                    raise TimeoutError("EMS service unavailable")
                return await super().dispatch(observation)

        orchestrator = SimulatorOrchestrator(
            settings,
            telemetry,
            FlakyDispatch(dispatch.service),
            command_timeout_ms=20,
        )
        events: list[dict[str, object]] = []

        async def capture(event: dict[str, object]) -> None:
            events.append(event)

        await orchestrator.run(max_steps=3, seed=0, on_event=capture)

        assert orchestrator.deadline_overruns == 1
        assert orchestrator.recoveries == 1
        metrics = next(event for event in events if event["type"] == "ems_metrics")
        assert metrics["ticks"] == 3
        assert metrics["deadline_overruns"] == 1
        assert metrics["recoveries"] == 1

    asyncio.run(exercise())
