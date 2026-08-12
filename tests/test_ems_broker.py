from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from microgrid_simulator.ems.broker import InProcessBroker
from microgrid_simulator.ems.types import TelemetryFrame


def _frame() -> TelemetryFrame:
    return TelemetryFrame(
        session_id="session",
        sequence_id=7,
        source_id="scenario",
        observed_at=datetime.now(timezone.utc),
        simulation_time_hours=0.0,
        timestep_hours=0.25,
        observation=[0.0],
        observation_shape=1,
        action_shape=3,
        pv_available_kw=0.0,
        pv_used_kw=0.0,
        load_demand_kw=0.0,
        load_served_kw=0.0,
        battery_soc=0.5,
        battery_soh=1.0,
        battery_power_kw=0.0,
        diesel_on=False,
        diesel_power_kw=0.0,
        grid_connected=False,
        grid_import_kw=0.0,
    )


def test_inprocess_broker_preserves_typed_message_identity() -> None:
    async def exercise() -> None:
        broker = InProcessBroker()
        frame = _frame()
        await broker.publish_telemetry(frame)
        assert await broker.receive_telemetry() is frame

    asyncio.run(exercise())
