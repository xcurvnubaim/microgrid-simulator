"""Battery SoC physics: charge/discharge integration, band clamping, degradation."""

from __future__ import annotations

from microgrid_simulator.config import BatteryCfg
from microgrid_simulator.grid.battery import BatteryModel


def _fresh() -> BatteryModel:
    return BatteryModel.from_cfg(BatteryCfg())


def test_charging_raises_soc() -> None:
    b = _fresh()
    start = b.soc
    b.apply(p_mw=0.10, dt_hours=0.25)
    assert b.soc > start


def test_discharging_lowers_soc() -> None:
    b = _fresh()
    start = b.soc
    b.apply(p_mw=-0.10, dt_hours=0.25)
    assert b.soc < start


def test_soc_never_leaves_band() -> None:
    b = _fresh()
    for _ in range(500):  # keep charging hard
        b.apply(p_mw=1.0, dt_hours=1.0)
    assert b.soc <= b.cfg.soc_max + 1e-9
    b.reset()
    for _ in range(500):  # keep discharging hard
        b.apply(p_mw=-1.0, dt_hours=1.0)
    assert b.soc >= b.cfg.soc_min - 1e-9


def test_power_is_clamped_to_limits() -> None:
    b = _fresh()
    applied, _ = b.apply(p_mw=10.0, dt_hours=0.25)  # way over max_charge
    assert applied <= b.cfg.max_charge_mw + 1e-9


def test_degradation_reduces_soh_monotonically() -> None:
    b = _fresh()
    soh0 = b.soh
    for _ in range(50):
        b.apply(p_mw=0.2, dt_hours=0.25)
    assert b.soh < soh0
    assert 0.0 <= b.soh <= 1.0
