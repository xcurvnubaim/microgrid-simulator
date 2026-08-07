"""Shared plant contract: parameters, fixture steps, and acceptance criteria.

Both implementations are configured from this one description so the
verification exercises identical physics; only the adapters differ.
"""

from __future__ import annotations

from dataclasses import dataclass

from microgrid_simulator.components.battery import BatteryModel
from microgrid_simulator.config import Settings


@dataclass(frozen=True)
class CommonPlant:
    """Parameters shared by the two single-bus scheduling implementations."""

    timestep_hours: float = 1.0
    battery_capacity_mwh: float = 1.0
    battery_max_charge_mw: float = 0.3
    battery_max_discharge_mw: float = 0.3
    battery_soc_min: float = 0.1
    battery_soc_max: float = 0.9
    battery_soc_init: float = 0.5
    diesel_min_mw: float = 0.1
    diesel_max_mw: float = 0.4


@dataclass(frozen=True)
class FixtureStep:
    """One target interval and its open-loop supervisory action."""

    name: str
    load_mw: float
    pv_available_mw: float
    battery_p_mw: float = 0.0  # project convention: +charge, -discharge
    diesel_on: bool = False
    diesel_setpoint_mw: float = 0.0
    pv_curtail: float = 0.0


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Numerical gates for the deliberately identical common model."""

    power_mw: float = 1e-9
    stored_energy_mwh: float = 1e-9
    soc: float = 1e-9
    aggregate_energy_mwh: float = 1e-9


class _NoDegradationBattery(BatteryModel):
    """Fixture-only battery: degradation is outside pymgrid's common contract."""

    def degradation_step(self, p_mw: float, dt_hours: float) -> float:
        del p_mw, dt_hours
        return 0.0


def common_fixture() -> list[FixtureStep]:
    """Exercise balance, device limits, spill, shortfall, and curtailment."""

    return [
        FixtureStep("pv_equals_load", load_mw=0.20, pv_available_mw=0.20),
        FixtureStep(
            "charge_request_clipped",
            load_mw=0.20,
            pv_available_mw=0.50,
            battery_p_mw=0.40,
        ),
        FixtureStep(
            "discharge_request_clipped",
            load_mw=0.40,
            pv_available_mw=0.10,
            battery_p_mw=-0.40,
        ),
        FixtureStep(
            "discharge_toward_floor",
            load_mw=0.60,
            pv_available_mw=0.0,
            battery_p_mw=-0.30,
        ),
        FixtureStep("discharge_hits_floor", load_mw=0.40, pv_available_mw=0.0, battery_p_mw=-0.30),
        FixtureStep("discharge_at_floor", load_mw=0.20, pv_available_mw=0.0, battery_p_mw=-0.20),
        FixtureStep(
            "diesel_minimum_excess",
            load_mw=0.05,
            pv_available_mw=0.0,
            diesel_on=True,
            diesel_setpoint_mw=0.02,
        ),
        FixtureStep(
            "diesel_pv_charge",
            load_mw=0.20,
            pv_available_mw=0.30,
            battery_p_mw=0.15,
            diesel_on=True,
            diesel_setpoint_mw=0.10,
        ),
        FixtureStep(
            "charge_toward_ceiling_1",
            load_mw=0.10,
            pv_available_mw=0.70,
            battery_p_mw=0.30,
        ),
        FixtureStep(
            "charge_toward_ceiling_2",
            load_mw=0.10,
            pv_available_mw=0.70,
            battery_p_mw=0.30,
        ),
        FixtureStep("charge_hits_ceiling", load_mw=0.10, pv_available_mw=0.70, battery_p_mw=0.30),
        FixtureStep("charge_at_ceiling", load_mw=0.10, pv_available_mw=0.70, battery_p_mw=0.20),
        FixtureStep(
            "explicit_pv_curtailment",
            load_mw=0.20,
            pv_available_mw=0.40,
            pv_curtail=0.50,
        ),
        FixtureStep(
            "diesel_maximum_clamp",
            load_mw=0.40,
            pv_available_mw=0.0,
            diesel_on=True,
            diesel_setpoint_mw=0.60,
        ),
    ]


def _common_settings(plant: CommonPlant, efficiency: float) -> Settings:
    raw = Settings().model_dump()
    raw["scenario"] = {
        "name": "pymgrid-common-model-verification",
        "description": "Lossless single-bus cross-implementation fixture",
    }
    raw["topology"] |= {
        "timestep_hours": plant.timestep_hours,
        "solver": "balance",
        "n_pv": 1,
        "n_storage": 1,
        "n_ev": 0,
        "n_load": 1,
    }
    raw["backend"] |= {"name": "simple", "timestep_hours": plant.timestep_hours}
    raw["episode"] |= {"start_hour": 0.0, "horizon_hours": 24.0}
    raw["buses"] = [
        {"id": 0, "name": "Common bus", "vn_kv": 0.4, "role": "main"},
    ]
    raw["lines"] = []
    raw["pv_arrays"] = [{"name": "PV", "bus": 0, "p_mw": 1.0}]
    raw["loads"] = [{"name": "Load", "bus": 0, "p_mw": 1.0, "q_mvar": 0.0}]
    raw["battery"] |= {
        "bus": 0,
        "capacity_mwh": plant.battery_capacity_mwh,
        "charge_eff": efficiency,
        "discharge_eff": efficiency,
        "max_charge_mw": plant.battery_max_charge_mw,
        "max_discharge_mw": plant.battery_max_discharge_mw,
        "soc_min": plant.battery_soc_min,
        "soc_max": plant.battery_soc_max,
        "soc_init": plant.battery_soc_init,
    }
    raw["diesel"] |= {
        "enabled": True,
        "bus": 0,
        "min_kw": plant.diesel_min_mw * 1000.0,
        "max_kw": plant.diesel_max_mw * 1000.0,
        "ramp_kw_per_min": 0.0,
        "start_delay_min": 0.0,
        "min_up_time_min": 0.0,
        "min_down_time_min": 0.0,
    }
    raw["demand"] |= {"enabled": False, "file": None, "random_window": False}
    return Settings(**raw)
