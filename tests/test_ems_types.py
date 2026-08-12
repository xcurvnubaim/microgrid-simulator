from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from microgrid_simulator.ems.types import TelemetryFrame


def _frame(**overrides: object) -> TelemetryFrame:
    raw = {
        "session_id": "session",
        "sequence_id": 0,
        "source_id": "scenario",
        "observed_at": datetime.now(timezone.utc),
        "simulation_time_hours": 0.0,
        "timestep_hours": 0.25,
        "observation": [0.0, 1.0],
        "observation_shape": 2,
        "action_shape": 3,
        "pv_available_kw": 10.0,
        "pv_used_kw": 8.0,
        "load_demand_kw": 20.0,
        "load_served_kw": 20.0,
        "battery_soc": 0.5,
        "battery_soh": 1.0,
        "battery_power_kw": -2.0,
        "diesel_on": True,
        "diesel_power_kw": 12.0,
        "grid_connected": False,
        "grid_import_kw": 0.0,
    }
    raw.update(overrides)
    return TelemetryFrame(**raw)


def test_telemetry_frame_has_explicit_units_and_physical_conversion() -> None:
    frame = _frame()
    state = frame.to_grid_state()

    assert state.battery_p_mw == pytest.approx(-0.002)
    assert state.diesel_p_mw == pytest.approx(0.012)
    assert state.islanded
    assert frame.schema_version == "1.0"


def test_telemetry_rejects_naive_timestamp_and_nonfinite_observation() -> None:
    with pytest.raises(ValidationError):
        _frame(observed_at=datetime.now())
    with pytest.raises(ValidationError):
        _frame(observation=[float("nan"), 1.0])
