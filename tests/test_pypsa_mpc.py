"""PyPSA MPC objective alignment tests."""

from __future__ import annotations

import numpy as np
import pytest

pypsa = pytest.importorskip("pypsa")

from microgrid_simulator.backends.pypsa_backend import (  # noqa: E402
    build_operational_network,
    optimize_dispatch,
)
from microgrid_simulator.config import Settings  # noqa: E402
from microgrid_simulator.controllers.pypsa_mpc import (  # noqa: E402
    _align_snapshot_to_steps,
    _resample_hourly_to_quarter,
)
from microgrid_simulator.forecast.cache import ForecastCache  # noqa: E402
from microgrid_simulator.forecast.errors import ForecastCacheError, ForecastError  # noqa: E402
from microgrid_simulator.forecast.snapshot import ForecastSnapshot  # noqa: E402


def _snapshot(
    *,
    issued_at: str = "2026-01-15T00:00:00",
    frequency_hours: float = 0.25,
    n_points: int = 96,
    first_offset_steps: int = 1,
    start: str | None = None,
) -> ForecastSnapshot:
    import datetime as _dt

    start_ts = (
        start
        if start is not None
        else (
            _dt.datetime.fromisoformat(issued_at)
            + _dt.timedelta(hours=frequency_hours * first_offset_steps)
        ).isoformat()
    )
    timestamps = [
        (
            _dt.datetime.fromisoformat(start_ts)
            + _dt.timedelta(hours=frequency_hours * i)
        ).isoformat()
        for i in range(n_points)
    ]
    return ForecastSnapshot(
        issued_at=issued_at,
        horizon_hours=int(round(24.0)),
        frequency_hours=frequency_hours,
        model_version="live",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=tuple(timestamps),
        pv_values_mw=tuple(float((0.1 + 0.01 * i) % 0.5) for i in range(n_points)),
        demand_values_mw=tuple(float((0.5 + 0.02 * i) % 1.0) for i in range(n_points)),
        source_id="islanded_72h_2026-01-15",
        context_time=issued_at,
        context_steps=0,
        cold_start=False,
        covariate_mode="live",
        issue_frequency_hours=0.25,
        forecast_steps=n_points,
        target_units="kw",
    )


def test_align_snapshot_15min_consumes_one_point_per_tick() -> None:
    snap = _snapshot()
    demand, pv = _align_snapshot_to_steps(
        snap,
        "2026-01-15T00:00:00",
        8,
        control_interval_hours=0.25,
        expected_source_id="islanded_72h_2026-01-15",
    )
    assert demand.shape == (8,)
    assert np.allclose(demand, snap.demand_values_mw[:8])
    assert np.allclose(pv, snap.pv_values_mw[:8])


def test_align_snapshot_hourly_holds_four_ticks_per_point() -> None:
    snap = _snapshot(frequency_hours=1.0, n_points=24)
    demand, pv = _align_snapshot_to_steps(
        snap,
        "2026-01-15T00:00:00",
        12,
        control_interval_hours=0.25,
        expected_source_id=None,
    )
    assert demand.shape == (12,)
    assert demand[:4].tolist() == [snap.demand_values_mw[0]] * 4
    assert demand[4:8].tolist() == [snap.demand_values_mw[1]] * 4


def test_align_snapshot_rejects_wrong_source() -> None:
    snap = _snapshot()
    with pytest.raises(ForecastError, match="source"):
        _align_snapshot_to_steps(
            snap,
            "2026-01-15T00:00:00",
            8,
            control_interval_hours=0.25,
            expected_source_id="other-scenario",
        )


def test_align_snapshot_rejects_future_issue() -> None:
    snap = _snapshot(issued_at="2026-01-15T01:00:00", start="2026-01-15T01:15:00")
    with pytest.raises(ForecastError, match="future"):
        _align_snapshot_to_steps(
            snap,
            "2026-01-15T00:00:00",
            8,
            control_interval_hours=0.25,
            expected_source_id=None,
        )


def test_align_snapshot_rejects_future_context() -> None:
    snap = _snapshot()
    snap = ForecastSnapshot(
        issued_at=snap.issued_at,
        horizon_hours=snap.horizon_hours,
        frequency_hours=snap.frequency_hours,
        model_version=snap.model_version,
        pv_target=snap.pv_target,
        demand_target=snap.demand_target,
        timestamps=snap.timestamps,
        pv_values_mw=snap.pv_values_mw,
        demand_values_mw=snap.demand_values_mw,
        source_id=snap.source_id,
        context_time="2026-01-15T00:15:00",
        context_steps=snap.context_steps,
        cold_start=snap.cold_start,
        covariate_mode=snap.covariate_mode,
        issue_frequency_hours=snap.issue_frequency_hours,
        forecast_steps=snap.forecast_steps,
        target_units=snap.target_units,
    )
    with pytest.raises(ForecastError, match="context_time"):
        _align_snapshot_to_steps(
            snap,
            "2026-01-15T00:00:00",
            8,
            control_interval_hours=0.25,
            expected_source_id=None,
        )


def test_align_snapshot_rejects_non_integer_frequency_ratio() -> None:
    snap = _snapshot(frequency_hours=0.4)
    with pytest.raises(ForecastError, match="integer multiple"):
        _align_snapshot_to_steps(
            snap,
            "2026-01-15T00:00:00",
            8,
            control_interval_hours=0.25,
            expected_source_id=None,
        )


def test_align_snapshot_rejects_incomplete_horizon() -> None:
    snap = _snapshot(n_points=2)
    with pytest.raises(ForecastError, match="incomplete horizon"):
        _align_snapshot_to_steps(
            snap,
            "2026-01-15T00:00:00",
            8,
            control_interval_hours=0.25,
            expected_source_id=None,
        )


def test_align_snapshot_rejects_missing_timestamps() -> None:
    snap = _snapshot()
    snap = ForecastSnapshot(
        issued_at=snap.issued_at,
        horizon_hours=snap.horizon_hours,
        frequency_hours=snap.frequency_hours,
        model_version=snap.model_version,
        pv_target=snap.pv_target,
        demand_target=snap.demand_target,
        timestamps=(),
        pv_values_mw=snap.pv_values_mw,
        demand_values_mw=snap.demand_values_mw,
        source_id=snap.source_id,
        context_time=snap.context_time,
        context_steps=snap.context_steps,
        cold_start=snap.cold_start,
        covariate_mode=snap.covariate_mode,
        issue_frequency_hours=snap.issue_frequency_hours,
        forecast_steps=snap.forecast_steps,
        target_units=snap.target_units,
    )
    with pytest.raises(ForecastError, match="timestamps"):
        _align_snapshot_to_steps(
            snap,
            "2026-01-15T00:00:00",
            8,
            control_interval_hours=0.25,
            expected_source_id=None,
        )


def test_align_snapshot_rejects_non_leakage_free_first_point() -> None:
    snap = _snapshot(start="2026-01-15T00:00:00")
    with pytest.raises(ForecastError, match="leakage-free"):
        _align_snapshot_to_steps(
            snap,
            "2026-01-15T00:00:00",
            8,
            control_interval_hours=0.25,
            expected_source_id=None,
        )




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
    assert diesel.ramp_limit_up == pytest.approx(1.0)
    assert diesel.ramp_limit_down == pytest.approx(1.0)
    assert diesel.min_up_time == 2
    assert diesel.min_down_time == 1


def _objective_settings(**reward_overrides: float) -> Settings:
    reward = {
        "w_carbon": 4.0,
        "w_health": 0.5,
        "w_waste": 1.0,
        "w_unserved": 20.0,
        "diesel_fuel_cost_per_kwh": 0.4,
    }
    reward.update(reward_overrides)
    settings = Settings(
        topology={"timestep_hours": 1.0},
        backend={"unit_commitment": False},
        diesel={
            "enabled": True,
            "max_kw": 400.0,
            "min_kw": 0.0,
            "min_up_time_min": 0.0,
            "min_down_time_min": 0.0,
        },
        battery={
            "capacity_mwh": 0.5,
            "max_charge_mw": 0.25,
            "max_discharge_mw": 0.25,
            "soc_min": 0.1,
            "soc_max": 0.95,
        },
        reward=reward,
    )
    settings.buses[0].role = "main"
    return settings


def test_e2_battery_health_weight_changes_mpc_cycling() -> None:
    low_health = _objective_settings(w_health=0.1)
    high_health = _objective_settings(w_health=2000.0)
    demand = np.array([0.0, 0.1])
    pv = np.array([0.1, 0.0])

    low_plan = optimize_dispatch(low_health, demand, pv, soc_init=0.1)
    high_plan = optimize_dispatch(high_health, demand, pv, soc_init=0.1)

    assert low_plan["battery_p_mw"].abs().sum() > 0.15
    assert high_plan["battery_p_mw"].abs().sum() == pytest.approx(0.0)
    assert high_plan["diesel_p_mw"].sum() > low_plan["diesel_p_mw"].sum()


def test_e4_pv_waste_weight_changes_mpc_pv_absorption() -> None:
    no_waste_penalty = _objective_settings(
        w_carbon=0.0,
        w_health=0.5,
        w_waste=0.0,
        w_unserved=0.001,
        diesel_fuel_cost_per_kwh=0.0,
    )
    high_waste_penalty = no_waste_penalty.model_copy(deep=True)
    high_waste_penalty.reward.w_waste = 2.0
    demand = np.array([0.0, 0.1])
    pv = np.array([0.1, 0.0])

    no_penalty_plan = optimize_dispatch(no_waste_penalty, demand, pv, soc_init=0.1)
    high_penalty_plan = optimize_dispatch(high_waste_penalty, demand, pv, soc_init=0.1)

    assert no_penalty_plan["pv_used_mw"].sum() == pytest.approx(0.0)
    assert high_penalty_plan["pv_used_mw"].sum() == pytest.approx(0.1)
    assert high_penalty_plan.loc[0, "battery_p_mw"] == pytest.approx(0.1)


def test_e5_unserved_weight_changes_mpc_load_shedding() -> None:
    low_unserved = _objective_settings(w_unserved=1.0)
    high_unserved = _objective_settings(w_unserved=100.0)
    demand = np.array([0.1, 0.1])
    pv = np.array([0.0, 0.0])

    low_plan = optimize_dispatch(
        low_unserved, demand, pv, soc_init=0.1, diesel_on_init=True
    )
    high_plan = optimize_dispatch(
        high_unserved, demand, pv, soc_init=0.1, diesel_on_init=True
    )

    assert low_plan["shed_mw"].sum() == pytest.approx(0.2)
    assert high_plan["shed_mw"].sum() == pytest.approx(0.0)
    assert high_plan["diesel_p_mw"].sum() == pytest.approx(0.2)


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


class _SpyBackend:
    """Minimal stand-in for PyPSAOperationalBackend used by PyPSAMPCController."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.settings = SimpleNamespace(
            backend=SimpleNamespace(
                horizon_hours=2.0,
                rolling_horizon_hours=0.5,
            ),
            topology=SimpleNamespace(timestep_hours=0.25, n_ev=0),
            episode=SimpleNamespace(telemetry_start="2026-01-15 00:00:00"),
            ev=SimpleNamespace(max_charge_mw=0.05),
        )
        self.demand = object()
        self.pv = object()
        self.calls: list[dict] = []
        self._advances = 0

    def horizon_steps(self) -> int:
        return 8

    def rolling_steps(self) -> int:
        return 2

    def advance_window(self) -> None:
        self._advances += 1

    def optimize_horizon(
        self, n_steps, soc_init=None, diesel_on_init=None, demand_mw=None, pv_mw=None
    ):
        self.calls.append(
            {
                "n_steps": n_steps,
                "soc_init": soc_init,
                "diesel_on_init": diesel_on_init,
                "demand_mw": None if demand_mw is None else np.asarray(demand_mw).copy(),
                "pv_mw": None if pv_mw is None else np.asarray(pv_mw).copy(),
            }
        )
        import pandas as pd

        return pd.DataFrame(
            {
                "battery_p_mw": [0.0],
                "diesel_on": [False],
                "diesel_p_mw": [0.1],
            }
        )


def test_mpc_uses_live_snapshot_and_never_perfect_foresight() -> None:
    from microgrid_simulator.controllers.pypsa_mpc import PyPSAMPCController
    from microgrid_simulator.core.types import GridState

    backend = _SpyBackend()
    ctrl = PyPSAMPCController(backend=backend)
    snap = _snapshot()
    ctrl.set_live_forecast_source(
        lambda: ("2026-01-15T00:00:00", snap),
        expected_source_id="islanded_72h_2026-01-15",
    )

    state = GridState(
        timestamp=0.0,
        soc=[0.5],
        diesel_on=False,
        load_demand_mw=0.6,
        pv_available_mw=0.2,
        pv_used_mw=0.2,
    )
    action = ctrl.act(state)

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["soc_init"] == pytest.approx(0.5)
    # Never None: no perfect-foresight fallback in optimize_horizon.
    assert call["demand_mw"] is not None
    assert call["pv_mw"] is not None
    assert call["demand_mw"].shape == (8,)
    assert np.allclose(call["demand_mw"], snap.demand_values_mw[:8])
    assert np.allclose(call["pv_mw"], snap.pv_values_mw[:8])
    assert action is not None
    assert backend._advances == 1


def test_mpc_fails_closed_when_live_snapshot_missing() -> None:
    from microgrid_simulator.controllers.pypsa_mpc import PyPSAMPCController
    from microgrid_simulator.core.types import GridState

    backend = _SpyBackend()
    ctrl = PyPSAMPCController(backend=backend)
    ctrl.set_live_forecast_source(lambda: ("2026-01-15T00:00:00", None))

    state = GridState(
        timestamp=0.0,
        soc=[0.5],
        diesel_on=False,
        load_demand_mw=0.6,
        pv_available_mw=0.2,
        pv_used_mw=0.2,
    )
    with pytest.raises(ForecastError, match="no live forecast snapshot"):
        ctrl.act(state)
    assert backend.calls == []


def test_mpc_fails_closed_without_any_forecast_source() -> None:
    from microgrid_simulator.controllers.pypsa_mpc import PyPSAMPCController
    from microgrid_simulator.core.types import GridState

    backend = _SpyBackend()
    ctrl = PyPSAMPCController(backend=backend)

    state = GridState(
        timestamp=0.0,
        soc=[0.5],
        diesel_on=False,
        load_demand_mw=0.6,
        pv_available_mw=0.2,
        pv_used_mw=0.2,
    )
    with pytest.raises(ForecastError, match="cached forecast or a live forecast snapshot"):
        ctrl.act(state)
    assert backend.calls == []
