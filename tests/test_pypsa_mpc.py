"""PyPSA MPC objective alignment tests."""

from __future__ import annotations

import numpy as np
import pytest

pypsa = pytest.importorskip("pypsa")

from microgrid_simulator.backends.pypsa_backend import build_operational_network  # noqa: E402
from microgrid_simulator.config import Settings  # noqa: E402
from microgrid_simulator.controllers.pypsa_mpc import _resample_hourly_to_quarter  # noqa: E402
from microgrid_simulator.forecast.cache import ForecastCache  # noqa: E402
from microgrid_simulator.forecast.errors import ForecastCacheError  # noqa: E402
from microgrid_simulator.forecast.snapshot import ForecastSnapshot  # noqa: E402


def test_diesel_costs_come_from_shared_reward_configuration() -> None:
    settings = Settings(
        diesel={
            "enabled": True,
            "carbon_kg_per_kwh": 0.7,
            "shut_down_cost": 2.0,
        },
        reward={
            "w_carbon": 4.0,
            "diesel_fuel_cost_per_kwh": 1.2,
            "diesel_start_cost": 3.0,
        },
    )

    network = build_operational_network(
        settings,
        demand_mw=np.array([0.1]),
        pv_available_mw=np.array([0.0]),
        soc_init=0.5,
    )

    diesel = network.generators.loc["diesel"]
    assert diesel.marginal_cost == pytest.approx((1.2 + 4.0 * 0.7) * 1000.0)
    assert diesel.start_up_cost == pytest.approx(3.0)
    assert diesel.shut_down_cost == pytest.approx(2.0)


def test_resample_hourly_forecast_to_quarter_hour() -> None:
    snapshot = ForecastSnapshot(
        issued_at="2026-01-15T00:00:00",
        horizon_hours=2,
        frequency_hours=1.0,
        model_version="cached",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=("2026-01-15T01:00:00", "2026-01-15T02:00:00"),
        pv_values_mw=(0.1, 0.2),
        demand_values_mw=(0.5, 0.6),
        source_id="islanded_72h_2026-01-15",
    )
    cache = ForecastCache({"2026-01-15T00:00:00": snapshot})

    demand, pv = _resample_hourly_to_quarter(cache, "2026-01-15T00:30:00", n_steps=12)

    # Each hourly value held across four 15-min ticks, zero-padded beyond horizon.
    assert pv.tolist() == [0.1] * 4 + [0.2] * 4 + [0.0] * 4
    assert demand.tolist() == [0.5] * 4 + [0.6] * 4 + [0.0] * 4


def test_resample_hourly_forecast_missing_covering_raises() -> None:
    cache = ForecastCache({})
    with pytest.raises(ForecastCacheError):
        _resample_hourly_to_quarter(cache, "2026-01-15T00:30:00", n_steps=4)
