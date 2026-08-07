"""Real demand trace + diesel generator (the plan's three decisions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandapower")

from microgrid_simulator.config import Settings  # noqa: E402
from microgrid_simulator.env import MicrogridEnv  # noqa: E402
from microgrid_simulator.grid.demand_trace import DemandTrace  # noqa: E402


@pytest.fixture()
def demand_xlsx(tmp_path):
    """Mimic the export: header on row 2, columns incl. statstime + demand (kW)."""
    ts = pd.date_range("2025-12-01", periods=96 * 4, freq="15min")
    df = pd.DataFrame(
        {
            "deviceid": "D1",
            "樓": "A",
            "loadname": "total",
            "statstime": ts,
            "demand": 100.0 + 50.0 * np.sin(np.arange(len(ts)) / 7.0),
        }
    )
    path = tmp_path / "Total Load (net load)_test.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([["meta"]]).to_excel(writer, index=False, header=False)
        df.to_excel(writer, index=False, startrow=1)
    return path


def _settings(**demand_overrides) -> Settings:
    raw = Settings().model_dump()
    raw["demand"] |= demand_overrides
    return Settings(**raw)


# --- demand trace -----------------------------------------------------------
def test_trace_loads_header_row_two(demand_xlsx) -> None:
    cfg = _settings(file=str(demand_xlsx)).demand
    trace = DemandTrace.from_file(cfg, 0.25)
    assert trace is not None
    assert len(trace) == 96 * 4
    assert trace.values_mw.max() < 0.2  # kW -> MW conversion applied


def test_trace_missing_file_falls_back() -> None:
    cfg = _settings(file="/nowhere/nothing.xlsx").demand
    assert DemandTrace.from_file(cfg, 0.25) is None


def test_random_window_varies_and_scales_loads(demand_xlsx) -> None:
    env = MicrogridEnv(settings=_settings(file=str(demand_xlsx)))
    _, i1 = env.reset(seed=1)
    _, i2 = env.reset(seed=2)
    assert i1["demand_window_start_time"] != i2["demand_window_start_time"]

    # Plan decision 3: sum of the 3 scaled loads equals the trace value.
    state = env.backend.reset(demand_window_mw=np.array([0.123] * (env.max_steps + 1)))
    assert state.demand_is_real
    assert sum(state.p_load) == pytest.approx(0.123, rel=1e-6)
    # relative shares preserved
    bases = [p for p, _ in env.backend.load_bases]
    ratios = [pl / b for pl, b in zip(state.p_load, bases, strict=True)]
    assert max(ratios) == pytest.approx(min(ratios), rel=1e-9)
    env.close()


def test_synthetic_fallback_without_file() -> None:
    env = MicrogridEnv(settings=_settings(file=None))
    _, info = env.reset(seed=0)
    assert "demand_window_start_time" not in info
    _, _, _, _, step_info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert step_info["demand_is_real"] is False
    env.close()


# --- diesel -----------------------------------------------------------------
def test_diesel_action_dims_and_clamp() -> None:
    env = MicrogridEnv(settings=Settings())
    # Compact full-EMS layout: battery + EVs + one diesel command + curtail.
    assert env.action_dim == 1 + env.n_ev + 1 + 1
    env.reset(seed=0)

    on = np.zeros(env.action_dim, dtype=np.float32)
    on[1 + env.n_ev], on[-1] = 1.0, -1.0  # diesel command at nameplate, no curtailment
    _, _, _, _, info = env.step(on)
    assert info["diesel_on"] is True
    # Start tick: 0.25 min crank/sync at 0, block-load to 45 kW, ramp to 150 kW
    # in 3.5 min, hold — tick average (97.5*3.5 + 150*11.25)/15 = 135.25 kW.
    assert info["diesel_p_mw"] == pytest.approx(0.13525)

    off = on.copy()
    off[1 + env.n_ev] = -1e-6  # near-zero negative: requests off, but the min-up
    # lockout overrides and the tiny magnitude maps onto the min-stable setpoint,
    # exercising the same "held on at a clamped setpoint" path as the legacy
    # two-dimension layout (whose stranded setpoint stayed at nameplate).
    # 15 min after the start the genset is still on: soft ramp from 150 kW down
    # to the 45 kW minimum stable load (3.5 min), hold — tick average 57.25 kW.
    _, _, _, _, info = env.step(off)
    assert info["diesel_on"] is True
    assert info["diesel_p_mw"] == pytest.approx(0.05725)

    # After 30 min of runtime the off command goes through. The genset is
    # already sitting at minimum stable load, so the breaker opens immediately
    # with no unload tail — zero output for the whole stop tick.
    _, _, _, _, info = env.step(off)
    assert info["diesel_on"] is False
    assert info["diesel_p_mw"] == pytest.approx(0.0)

    # The next tick the genset is fully offline.
    _, _, _, _, info = env.step(off)
    assert info["diesel_on"] is False
    assert info["diesel_p_mw"] == 0.0
    env.close()


def _diesel(**overrides):
    from microgrid_simulator.grid.diesel import DieselModel

    raw = Settings().model_dump()
    raw["diesel"] |= overrides
    return DieselModel.from_cfg(Settings(**raw).diesel)


def test_diesel_min_stable_load_clamp() -> None:
    d = _diesel(min_kw=45.0, max_kw=150.0)
    # A 5 kW setpoint is below the minimum stable load -> clamped up to 45 kW.
    # The start tick averages lower: 0.25 min of crank/sync produce nothing,
    # then 14.75 min at 45 kW -> 44.25 kW average.
    assert d.apply(True, 0.005, 0.25) == pytest.approx(0.04425)
    assert d.p_end_mw == pytest.approx(0.045)
    # Settled ticks sit exactly at min stable load.
    assert d.apply(True, 0.005, 0.25) == pytest.approx(0.045)


def test_diesel_ramp_limits_output_change() -> None:
    d = _diesel(min_kw=45.0, max_kw=150.0, ramp_kw_per_min=30.0)
    dt = 1.0 / 60.0  # 1-minute ticks, where the ramp binds across ticks
    # Start tick: 0.25 min crank/sync at 0, block-load to 45 kW, then ramp at
    # 30 kW/min for 0.75 min -> ends at 67.5 kW, tick average 42.1875 kW.
    assert d.apply(True, 0.150, dt) == pytest.approx(0.0421875)
    assert d.p_end_mw == pytest.approx(0.0675)
    # Steady ramp: averages are the trapezoid midpoints of each 30 kW climb.
    assert d.apply(True, 0.150, dt) == pytest.approx(0.0825)
    assert d.apply(True, 0.150, dt) == pytest.approx(0.1125)
    # Reaches 150 kW after 0.75 min, holds for the rest of the tick.
    assert d.apply(True, 0.150, dt) == pytest.approx(0.1415625)
    assert d.p_end_mw == pytest.approx(0.150)
    assert d.apply(True, 0.150, dt) == pytest.approx(0.150)
    # Ramping down is limited too, and never below min stable load.
    assert d.apply(True, 0.0, dt) == pytest.approx(0.135)
    assert d.p_end_mw == pytest.approx(0.120)


def test_diesel_stop_soft_unloads_before_breaker_opens() -> None:
    d = _diesel(min_kw=45.0, max_kw=150.0, ramp_kw_per_min=30.0, min_up_time_min=0.0)
    d.apply(True, 0.150, 0.25)
    assert d.p_end_mw == pytest.approx(0.150)
    # Off tick: unload 150 -> 45 kW at 30 kW/min (3.5 min), breaker opens,
    # rest of the 15 min tick is silent -> 97.5 * 3.5 / 15 = 22.75 kW average.
    assert d.apply(False, 0.0, 0.25) == pytest.approx(0.02275)
    assert d.is_on is False
    assert d.p_end_mw == 0.0
    assert d.apply(False, 0.0, 0.25) == 0.0


def test_diesel_min_up_and_down_time_lockouts() -> None:
    d = _diesel(min_up_time_min=30.0, min_down_time_min=15.0)
    dt = 0.25  # 15-minute ticks
    d.apply(True, 0.150, dt)
    assert d.starts == 1
    # Off command 15 min in is ignored (min up time = 30 min).
    d.apply(False, 0.0, dt)
    assert d.is_on is True
    # At 30 min the shutdown is allowed...
    d.apply(False, 0.0, dt)
    assert d.is_on is False
    # ...and after 15 min down the restart is allowed again.
    d.apply(True, 0.150, dt)
    assert d.is_on is True
    assert d.starts == 2


def test_diesel_ramp_zero_means_unlimited() -> None:
    d = _diesel(min_kw=0.0, ramp_kw_per_min=0.0, start_delay_min=0.0, min_up_time_min=0.0)
    assert d.apply(True, 0.150, 1.0 / 60.0) == pytest.approx(0.150)


def test_diesel_carbon_term_in_reward() -> None:
    env = MicrogridEnv(settings=Settings())
    env.reset(seed=0)
    base = np.zeros(env.action_dim, dtype=np.float32)
    base[-3], base[-2], base[-1] = -1.0, -1.0, -1.0
    _, _, _, _, info_off = env.step(base)

    env.reset(seed=0)
    hot = base.copy()
    hot[-3], hot[-2] = 1.0, 1.0
    _, _, _, _, info_on = env.step(hot)

    kwh = (env.settings.diesel.max_kw / 1000.0) * env.dt * 1000.0
    expected_extra = env.settings.reward.diesel_carbon_kg_per_kwh * kwh
    # Diesel also offsets grid import, so compare against the analytic term
    # rather than the raw difference: the diesel share must be present.
    assert info_on["diesel_p_mw"] > 0
    assert info_on["reward/carbon"] >= expected_extra * 0.5
    assert expected_extra > 0
    env.close()


def test_diesel_disabled_shrinks_action_space() -> None:
    raw = Settings().model_dump()
    raw["diesel"] |= {"enabled": False}
    env = MicrogridEnv(settings=Settings(**raw))
    assert env.action_dim == 1 + env.n_ev + 1
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert info["diesel_p_mw"] == 0.0
    env.close()
