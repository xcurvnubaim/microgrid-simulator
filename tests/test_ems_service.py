from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from microgrid_simulator.config import Settings
from microgrid_simulator.ems import EMSService, InProcessBroker, SafetyShield, SimulatorBridge
from microgrid_simulator.ems.types import TelemetryFrame


def _settings() -> Settings:
    settings = Settings()
    settings.topology.n_ev = 0
    settings.episode.horizon_hours = 1.0
    settings.forecast.enabled = False
    return settings


def _frame(settings: Settings, **overrides: object) -> TelemetryFrame:
    raw = {
        "session_id": "session",
        "sequence_id": 0,
        "source_id": "scenario",
        "observed_at": datetime.now(timezone.utc),
        "simulation_time_hours": 0.0,
        "timestep_hours": settings.topology.timestep_hours,
        "observation": [0.0],
        "observation_shape": 1,
        "action_shape": 3,
        "pv_available_kw": 0.0,
        "pv_used_kw": 0.0,
        "load_demand_kw": 100.0,
        "load_served_kw": 100.0,
        "battery_soc": settings.battery.soc_init,
        "battery_soh": 1.0,
        "battery_power_kw": 0.0,
        "diesel_on": False,
        "diesel_power_kw": 0.0,
        "grid_connected": False,
        "grid_import_kw": 0.0,
    }
    raw.update(overrides)
    return TelemetryFrame(**raw)


def test_safety_shield_blocks_discharge_at_soc_floor() -> None:
    settings = _settings()
    frame = _frame(settings, battery_soc=settings.battery.soc_min)
    from microgrid_simulator.core.types import ControlAction

    result = SafetyShield(settings).apply(ControlAction(battery_p_mw=-1.0), frame)

    assert result.action.battery_p_mw == pytest.approx(0.0)
    assert "battery_soc_headroom" in result.reasons


def test_rule_service_emits_matching_deadline_bound_command() -> None:
    settings = _settings()
    broker = InProcessBroker()
    service = EMSService(settings, broker, policy="rule")
    frame = _frame(settings, sequence_id=4)

    command = service.build_command(frame)

    assert command.session_id == frame.session_id
    assert command.telemetry_sequence_id == 4
    assert len(command.normalized_action) == frame.action_shape
    assert command.expires_at > command.issued_at
    assert command.requested_diesel_on


def test_rule_service_does_not_import_sb3_or_torch() -> None:
    code = """
import sys
from microgrid_simulator.config import Settings
from microgrid_simulator.ems.service import EMSService
EMSService(Settings(), policy='rule')
assert 'stable_baselines3' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bridge_and_service_complete_event_driven_episode() -> None:
    async def exercise() -> None:
        settings = _settings()
        broker = InProcessBroker()
        service = EMSService(settings, broker, policy="rule")
        bridge = SimulatorBridge(settings, broker, command_timeout_ms=500)
        service_task = asyncio.create_task(service.run())
        try:
            results = await bridge.run(max_steps=2, seed=0)
        finally:
            service_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await service_task

        assert len(results) == 2
        assert [result.telemetry_sequence_id for result in results] == [0, 1]
        assert all(result.command_id for result in results)
        assert all(result.status == "applied" for result in results)

    asyncio.run(exercise())


def test_bridge_uses_rule_fallback_when_command_times_out() -> None:
    async def exercise() -> None:
        settings = _settings()
        broker = InProcessBroker()
        bridge = SimulatorBridge(settings, broker, command_timeout_ms=1)
        result = (await bridge.run(max_steps=1, seed=0))[0]
        assert result.status == "fallback"
        assert result.command_id is None

    asyncio.run(exercise())


def test_bridge_discards_late_command_and_recovers_on_current_tick() -> None:
    async def exercise() -> None:
        settings = _settings()
        broker = InProcessBroker()
        service = EMSService(settings, broker, policy="rule")
        bridge = SimulatorBridge(settings, broker, command_timeout_ms=20)

        async def delayed_service() -> None:
            first = True
            while True:
                frame = await broker.receive_telemetry()
                if first:
                    await asyncio.sleep(0.03)
                    first = False
                command = await asyncio.to_thread(service.build_command, frame)
                await broker.publish_command(command)

        service_task = asyncio.create_task(delayed_service())
        try:
            results = await bridge.run(max_steps=4, seed=0)
        finally:
            service_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await service_task

        assert results[0].status == "fallback"
        assert any(result.status == "applied" for result in results[1:])
        first_recovered = next(result for result in results[1:] if result.status == "applied")
        assert first_recovered.command_id is not None
        assert first_recovered.status_message == "EMS command applied"

    asyncio.run(exercise())


def test_bridge_reports_rejected_command_when_no_match_arrives() -> None:
    async def exercise() -> None:
        settings = _settings()
        broker = InProcessBroker()
        bridge = SimulatorBridge(settings, broker, command_timeout_ms=10)
        frame = _frame(settings, session_id=bridge.session_id, sequence_id=99)
        stale = EMSService(settings, broker, policy="rule").build_command(frame)
        await broker.publish_command(stale)

        result = (await bridge.run(max_steps=1, seed=0))[0]

        assert result.status == "rejected"
        assert result.command_id == stale.command_id
        assert "rule fallback applied" in result.status_message

    asyncio.run(exercise())
