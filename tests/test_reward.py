"""Reward: sign convention, term breakdown, infeasible penalty."""

from __future__ import annotations

from microgrid_simulator.config import BatteryCfg, RewardCfg
from microgrid_simulator.grid.battery import BatteryModel
from microgrid_simulator.grid.pandapower_backend import GridState
from microgrid_simulator.model.reward import compute_reward


def _battery() -> BatteryModel:
    return BatteryModel.from_cfg(BatteryCfg())


def test_reward_is_non_positive() -> None:
    state = GridState(
        v_bus=[1.0, 1.0],
        grid_import_mw=0.1,
        pv_available_mw=0.1,
        pv_used_mw=0.1,
        load_demand_mw=0.2,
        load_served_mw=0.2,
        solver_ok=True,
    )
    reward, _ = compute_reward(state, _battery(), RewardCfg(), dt_hours=0.25)
    assert reward <= 0.0


def test_no_import_beats_high_import() -> None:
    cfg, dt = RewardCfg(), 0.25
    good = GridState(
        v_bus=[1.0], grid_import_mw=0.0, pv_available_mw=0.1, pv_used_mw=0.1, solver_ok=True
    )
    bad = GridState(
        v_bus=[1.0], grid_import_mw=0.5, pv_available_mw=0.1, pv_used_mw=0.1, solver_ok=True
    )
    r_good, _ = compute_reward(good, _battery(), cfg, dt)
    r_bad, _ = compute_reward(bad, _battery(), cfg, dt)
    assert r_good > r_bad


def test_curtailed_solar_is_penalised() -> None:
    cfg, dt = RewardCfg(), 0.25
    used = GridState(
        v_bus=[1.0], grid_import_mw=0.0, pv_available_mw=0.1, pv_used_mw=0.1, solver_ok=True
    )
    wasted = GridState(
        v_bus=[1.0], grid_import_mw=0.0, pv_available_mw=0.1, pv_used_mw=0.0, solver_ok=True
    )
    _, b_used = compute_reward(used, _battery(), cfg, dt)
    _, b_wasted = compute_reward(wasted, _battery(), cfg, dt)
    assert b_wasted.waste > b_used.waste


def test_unserved_load_is_penalised() -> None:
    state = GridState(v_bus=[1.0], load_demand_mw=0.3, load_served_mw=0.2, solver_ok=True)
    _, b = compute_reward(state, _battery(), RewardCfg(), dt_hours=0.25)
    assert b.unserved > 0.0


def test_dumped_excess_is_penalised_separately_from_pv_waste() -> None:
    state = GridState(
        v_bus=[1.0],
        pv_available_mw=0.0,
        pv_used_mw=0.0,
        excess_generation_mw=0.045,
        dump_load_mw=0.045,
        solver_ok=True,
    )
    _, breakdown = compute_reward(state, _battery(), RewardCfg(), dt_hours=0.25)

    assert breakdown.waste == 0.0
    assert breakdown.excess == 45.0 * 0.25


def test_infeasible_grid_gets_large_penalty() -> None:
    state = GridState(solver_ok=False)
    reward, b = compute_reward(state, _battery(), RewardCfg(), dt_hours=0.25)
    assert reward <= -1000.0
    assert b.constraint >= 1000.0


def test_higher_diesel_fuel_price_increases_fuel_penalty() -> None:
    state = GridState(v_bus=[1.0], diesel_p_mw=0.1, solver_ok=True)
    low = RewardCfg(w_carbon=4.0, diesel_fuel_cost_per_kwh=0.40)
    high = RewardCfg(w_carbon=4.0, diesel_fuel_cost_per_kwh=1.20)

    _, low_breakdown = compute_reward(state, _battery(), low, dt_hours=0.25)
    _, high_breakdown = compute_reward(state, _battery(), high, dt_hours=0.25)

    assert low_breakdown.fuel == 40.0
    assert high_breakdown.fuel == 120.0
