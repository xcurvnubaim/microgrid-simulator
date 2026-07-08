"""Power accounting invariants for islanded operation and UI totals."""

from __future__ import annotations

import pytest

from microgrid_simulator.backends.simple_backend import SimpleBackend
from microgrid_simulator.config import Settings
from microgrid_simulator.core.types import ControlAction
from microgrid_simulator.ui.rollout import _totals


def _islanded_settings() -> Settings:
    raw = Settings().model_dump()
    for bus in raw["buses"]:
        if bus["role"] == "grid":
            bus["role"] = "main"
    raw["topology"] |= {"n_ev": 0, "n_pv": 1, "n_load": 1}
    raw["episode"] |= {"start_hour": 11.75}
    raw["pv_arrays"] = [{"name": "PV", "bus": 2, "p_mw": 1.0}]
    raw["loads"] = [{"name": "Load", "bus": 4, "p_mw": 0.10, "q_mvar": 0.02}]
    raw["diesel"] |= {"enabled": False}
    raw["battery"] |= {
        "capacity_mwh": 10.0,
        "max_charge_mw": 1.0,
        "max_discharge_mw": 1.0,
        "soc_init": 0.10,
        "soc_min": 0.0,
        "soc_max": 0.95,
    }
    return Settings(**raw)


def test_islanded_battery_charge_uses_only_surplus_after_load() -> None:
    backend = SimpleBackend(_islanded_settings())

    state = backend.step(ControlAction(battery_p_mw=1.0, pv_curtail=0.0))

    surplus_after_load = state.pv_used_mw - state.load_demand_mw
    assert state.load_served_mw == pytest.approx(state.load_demand_mw)
    assert state.unserved_mw == pytest.approx(0.0)
    assert state.battery_p_mw == pytest.approx(surplus_after_load)
    assert state.battery_p_mw < 1.0


def test_rollout_totals_do_not_net_export_against_import() -> None:
    rows = [
        {
            "reward": 0.0,
            "grid_import_kw": 100.0,
            "grid_import_positive_kw": 100.0,
            "grid_export_kw": 0.0,
            "diesel_kw": 0.0,
            "pv_used_kw": 0.0,
            "pv_wasted_kw": 0.0,
            "load_kw": 100.0,
            "served_kw": 100.0,
            "unserved_kw": 0.0,
            "blackout": False,
            "soc_pct": 50.0,
            "carbon_kg": 10.0,
        },
        {
            "reward": 0.0,
            "grid_import_kw": -80.0,
            "grid_import_positive_kw": 0.0,
            "grid_export_kw": 80.0,
            "diesel_kw": 0.0,
            "pv_used_kw": 180.0,
            "pv_wasted_kw": 0.0,
            "load_kw": 100.0,
            "served_kw": 100.0,
            "unserved_kw": 0.0,
            "blackout": False,
            "soc_pct": 50.0,
            "carbon_kg": 0.0,
        },
    ]

    totals = _totals(rows, dt=0.25)

    assert totals["grid_import_kwh"] == pytest.approx(25.0)
    assert totals["grid_export_kwh"] == pytest.approx(20.0)
    assert totals["carbon_kg"] == pytest.approx(10.0)
