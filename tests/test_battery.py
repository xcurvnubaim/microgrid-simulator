"""Battery SoC physics: charge/discharge integration, band clamping, degradation."""

from __future__ import annotations

import pytest

from microgrid_simulator.components.battery import PymgridBatteryModel, create_battery_model
from microgrid_simulator.config import BatteryCfg, Settings
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


def test_pymgrid_model_uses_symmetric_internal_energy_limits() -> None:
    raw = Settings().model_dump()
    raw["topology"]["timestep_hours"] = 1.0
    raw["battery"] |= {
        "model": "pymgrid",
        "capacity_mwh": 1.0,
        "charge_eff": 0.9,
        "discharge_eff": 0.9,
        "limit_basis": "internal_energy_per_step",
        "max_charge_internal_mwh_per_step": 0.2,
        "max_discharge_internal_mwh_per_step": 0.2,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "soc_init": 0.5,
        "degradation_enabled": False,
    }
    settings = Settings(**raw)
    battery = create_battery_model(settings.battery)

    assert isinstance(battery, PymgridBatteryModel)
    assert settings.battery.max_charge_mw == pytest.approx(0.2 / 0.9)
    assert settings.battery.max_discharge_mw == pytest.approx(0.2 * 0.9)

    charged_mw, charge_degradation = battery.apply(10.0, dt_hours=1.0)
    assert charged_mw == pytest.approx(0.2 / 0.9)
    assert battery.current_charge_mwh == pytest.approx(0.7)
    assert battery.soc == pytest.approx(0.7)
    assert charge_degradation == 0.0

    discharged_mw, discharge_degradation = battery.apply(-10.0, dt_hours=1.0)
    assert discharged_mw == pytest.approx(-0.2 * 0.9)
    assert battery.current_charge_mwh == pytest.approx(0.5)
    assert battery.soc == pytest.approx(0.5)
    assert discharge_degradation == 0.0
    assert battery.soh == 1.0


def test_project_battery_remains_default_model() -> None:
    battery = create_battery_model(BatteryCfg())
    assert isinstance(battery, BatteryModel)
