"""Per-implementation runners and canonical row normalization.

Each runner feeds the shared fixture through one implementation and returns
rows in the canonical accounting schema so :mod:`.compare` can compute deltas
without knowing which side produced them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from microgrid_simulator.backends.simple_backend import SimpleBackend
from microgrid_simulator.core.types import ControlAction
from microgrid_simulator.experiments.pymgrid_verification.plant import (
    CommonPlant,
    FixtureStep,
    _common_settings,
    _NoDegradationBattery,
)


def _pymgrid_battery_action_mwh(fixture: FixtureStep, timestep_hours: float) -> float:
    """Translate only units and sign; leave feasibility to pymgrid itself."""

    # Project convention: +MW charges. pymgrid convention: +MWh discharges.
    return -fixture.battery_p_mw * timestep_hours


def _build_pymgrid(
    plant: CommonPlant,
    steps: list[FixtureStep],
    efficiency: float,
) -> Any:
    try:
        from pymgrid import Microgrid
        from pymgrid.modules import (
            BatteryModule,
            GensetModule,
            LoadModule,
            RenewableModule,
            UnbalancedEnergyModule,
        )
    except ImportError as exc:  # pragma: no cover - exercised by install boundary
        raise RuntimeError("pymgrid verification requires `uv run --extra pymgrid ...`") from exc

    dt = plant.timestep_hours
    load_energy = np.asarray([step.load_mw * dt for step in steps], dtype=float)
    # pymgrid's RenewableModule has no separate curtailment action.  The known
    # open-loop curtailment request is therefore applied to availability here;
    # the canonical evaluator still measures spill against original availability.
    pv_energy = np.asarray(
        [step.pv_available_mw * (1.0 - np.clip(step.pv_curtail, 0.0, 1.0)) * dt for step in steps],
        dtype=float,
    )
    physical_capacity = plant.battery_capacity_mwh
    battery = BatteryModule(
        min_capacity=physical_capacity * plant.battery_soc_min,
        max_capacity=physical_capacity * plant.battery_soc_max,
        # pymgrid limits are internal energy per step; our limits are external
        # terminal power, so efficiency belongs on opposite sides by direction.
        max_charge=plant.battery_max_charge_mw * dt * efficiency,
        max_discharge=plant.battery_max_discharge_mw * dt / efficiency,
        efficiency=efficiency,
        battery_cost_cycle=0.0,
        init_charge=physical_capacity * plant.battery_soc_init,
    )
    genset = GensetModule(
        running_min_production=plant.diesel_min_mw * dt,
        running_max_production=plant.diesel_max_mw * dt,
        genset_cost=0.0,
        co2_per_unit=0.0,
        cost_per_unit_co2=0.0,
        start_up_time=0,
        wind_down_time=0,
        init_start_up=False,
    )
    return Microgrid(
        [
            LoadModule(load_energy, forecaster=None, forecast_horizon=0),
            ("pv", RenewableModule(pv_energy, forecaster=None, forecast_horizon=0)),
            (
                "unbalanced_energy",
                UnbalancedEnergyModule(
                    raise_errors=False,
                    loss_load_cost=1.0,
                    overgeneration_cost=1.0,
                ),
            ),
            genset,
            battery,
        ]
    )


def _info_value(info: dict[str, list[dict[str, float]]], module: str, key: str) -> float:
    values = info.get(module, [])
    if not values:
        return 0.0
    return float(values[0].get(key, 0.0))


def _canonical_row(
    *,
    implementation: str,
    step_index: int,
    fixture: FixtureStep,
    load_mw: float,
    pv_used_mw: float,
    battery_p_mw: float,
    diesel_mw: float,
    unserved_mw: float,
    excess_mw: float,
    stored_energy_mwh: float,
    plant: CommonPlant,
) -> dict[str, Any]:
    served_mw = max(0.0, load_mw - unserved_mw)
    pv_spill_mw = max(0.0, fixture.pv_available_mw - pv_used_mw)
    supply_mw = pv_used_mw + diesel_mw + max(0.0, -battery_p_mw) + unserved_mw
    sink_mw = load_mw + max(0.0, battery_p_mw) + excess_mw
    return {
        "implementation": implementation,
        "step": step_index,
        "fixture": fixture.name,
        "load_mw": load_mw,
        "served_mw": served_mw,
        "unserved_mw": unserved_mw,
        "pv_available_mw": fixture.pv_available_mw,
        "pv_used_mw": pv_used_mw,
        "pv_spill_mw": pv_spill_mw,
        "battery_p_mw": battery_p_mw,
        "diesel_mw": diesel_mw,
        "excess_mw": excess_mw,
        "stored_energy_mwh": stored_energy_mwh,
        "soc": stored_energy_mwh / plant.battery_capacity_mwh,
        "balance_residual_mw": supply_mw - sink_mw,
    }


def _run_simple(
    plant: CommonPlant,
    steps: list[FixtureStep],
    efficiency: float,
) -> list[dict[str, Any]]:
    settings = _common_settings(plant, efficiency)
    backend = SimpleBackend(settings)
    backend.battery = _NoDegradationBattery.from_cfg(settings.battery)
    # The simple backend intentionally advances before applying an action.  A
    # leading context sample aligns its first solved step with fixture row 0.
    demand = np.asarray([steps[0].load_mw, *[step.load_mw for step in steps]], dtype=float)
    pv = np.asarray(
        [steps[0].pv_available_mw, *[step.pv_available_mw for step in steps]], dtype=float
    )
    backend.reset(demand_window_mw=demand, pv_window_mw=pv)

    rows: list[dict[str, Any]] = []
    for index, fixture in enumerate(steps):
        state = backend.step(
            ControlAction(
                battery_p_mw=fixture.battery_p_mw,
                pv_curtail=fixture.pv_curtail,
                diesel_on=fixture.diesel_on,
                diesel_setpoint_mw=fixture.diesel_setpoint_mw,
            )
        )
        excess_mw = max(
            0.0,
            state.pv_used_mw
            + state.diesel_p_mw
            + max(0.0, -state.battery_p_mw)
            - state.load_served_mw
            - max(0.0, state.battery_p_mw),
        )
        rows.append(
            _canonical_row(
                implementation="microgrid-simulator",
                step_index=index,
                fixture=fixture,
                load_mw=state.load_demand_mw,
                pv_used_mw=state.pv_used_mw,
                battery_p_mw=state.battery_p_mw,
                diesel_mw=state.diesel_p_mw,
                unserved_mw=state.unserved_mw,
                excess_mw=excess_mw,
                stored_energy_mwh=state.soc[0] * plant.battery_capacity_mwh,
                plant=plant,
            )
        )
    return rows


def _run_pymgrid(
    plant: CommonPlant,
    steps: list[FixtureStep],
    efficiency: float,
) -> list[dict[str, Any]]:
    microgrid = _build_pymgrid(plant, steps, efficiency)
    battery = microgrid.modules["battery"][0]
    dt = plant.timestep_hours
    rows: list[dict[str, Any]] = []
    for index, fixture in enumerate(steps):
        battery_energy = _pymgrid_battery_action_mwh(fixture, dt)
        _, _, _, info = microgrid.step(
            {
                "genset": [[float(fixture.diesel_on), fixture.diesel_setpoint_mw * dt]],
                "battery": [battery_energy],
            },
            normalized=False,
        )
        charge_mwh = _info_value(info, "battery", "absorbed_energy")
        discharge_mwh = _info_value(info, "battery", "provided_energy")
        pv_used_mwh = _info_value(info, "pv", "provided_energy")
        diesel_mwh = _info_value(info, "genset", "provided_energy")
        unserved_mwh = _info_value(info, "unbalanced_energy", "provided_energy")
        excess_mwh = _info_value(info, "unbalanced_energy", "absorbed_energy")
        rows.append(
            _canonical_row(
                implementation="pymgrid",
                step_index=index,
                fixture=fixture,
                load_mw=fixture.load_mw,
                pv_used_mw=pv_used_mwh / dt,
                battery_p_mw=(charge_mwh - discharge_mwh) / dt,
                diesel_mw=diesel_mwh / dt,
                unserved_mw=unserved_mwh / dt,
                excess_mw=excess_mwh / dt,
                stored_energy_mwh=float(battery.current_charge),
                plant=plant,
            )
        )
    return rows


def _summarize_rows(
    rows: list[dict[str, Any]],
    plant: CommonPlant,
) -> dict[str, float]:
    """Return implementation-local energy and state summaries."""

    dt = plant.timestep_hours
    return {
        "load_mwh": sum(float(row["load_mw"]) for row in rows) * dt,
        "served_mwh": sum(float(row["served_mw"]) for row in rows) * dt,
        "unserved_mwh": sum(float(row["unserved_mw"]) for row in rows) * dt,
        "pv_available_mwh": sum(float(row["pv_available_mw"]) for row in rows) * dt,
        "pv_used_mwh": sum(float(row["pv_used_mw"]) for row in rows) * dt,
        "pv_spill_mwh": sum(float(row["pv_spill_mw"]) for row in rows) * dt,
        "battery_charge_mwh": sum(max(0.0, float(row["battery_p_mw"])) for row in rows) * dt,
        "battery_discharge_mwh": sum(max(0.0, -float(row["battery_p_mw"])) for row in rows) * dt,
        "diesel_mwh": sum(float(row["diesel_mw"]) for row in rows) * dt,
        "excess_mwh": sum(float(row["excess_mw"]) for row in rows) * dt,
        "initial_soc": plant.battery_soc_init,
        "final_soc": float(rows[-1]["soc"]),
        "max_abs_balance_residual_mw": max(abs(float(row["balance_residual_mw"])) for row in rows),
    }
