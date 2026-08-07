"""Field-by-field trajectory comparison against the acceptance criteria.

Deltas are computed as ``candidate - reference`` on the canonical rows; a
field fails when its maximum absolute delta exceeds the matching tolerance.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from microgrid_simulator.experiments.pymgrid_verification.constants import (
    CANDIDATE_IMPLEMENTATION,
    REFERENCE_IMPLEMENTATION,
)
from microgrid_simulator.experiments.pymgrid_verification.harness import _summarize_rows
from microgrid_simulator.experiments.pymgrid_verification.plant import (
    AcceptanceCriteria,
    CommonPlant,
)

COMPARISON_FIELDS = (
    "load_mw",
    "served_mw",
    "unserved_mw",
    "pv_available_mw",
    "pv_used_mw",
    "pv_spill_mw",
    "battery_p_mw",
    "diesel_mw",
    "excess_mw",
    "stored_energy_mwh",
    "soc",
    "balance_residual_mw",
)


def _compare(
    simple_rows: list[dict[str, Any]],
    pymgrid_rows: list[dict[str, Any]],
    plant: CommonPlant,
    criteria: AcceptanceCriteria,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(simple_rows) != len(pymgrid_rows):
        raise AssertionError("implementations produced different trajectory lengths")

    comparison_rows: list[dict[str, Any]] = []
    max_abs_delta = {field: 0.0 for field in COMPARISON_FIELDS}
    for simple, pymgrid in zip(simple_rows, pymgrid_rows, strict=True):
        row: dict[str, Any] = {"step": simple["step"], "fixture": simple["fixture"]}
        for field in COMPARISON_FIELDS:
            simple_value = float(simple[field])
            pymgrid_value = float(pymgrid[field])
            delta = simple_value - pymgrid_value
            row[f"simulator_{field}"] = simple_value
            row[f"pymgrid_{field}"] = pymgrid_value
            row[f"delta_{field}"] = delta
            max_abs_delta[field] = max(max_abs_delta[field], abs(delta))
        comparison_rows.append(row)

    dt = plant.timestep_hours
    energy_fields = ("unserved_mw", "pv_spill_mw", "diesel_mw", "excess_mw")
    aggregate_delta_mwh = {
        field.removesuffix("_mw") + "_mwh": abs(
            sum(float(row[field]) for row in simple_rows) * dt
            - sum(float(row[field]) for row in pymgrid_rows) * dt
        )
        for field in energy_fields
    }
    failed_fields = [
        field
        for field in COMPARISON_FIELDS
        if max_abs_delta[field]
        > (
            criteria.stored_energy_mwh
            if field == "stored_energy_mwh"
            else criteria.soc
            if field == "soc"
            else criteria.power_mw
        )
    ]
    failed_aggregate_fields = [
        field
        for field, delta in aggregate_delta_mwh.items()
        if delta > criteria.aggregate_energy_mwh
    ]
    passed = not failed_fields and not failed_aggregate_fields
    metrics = {
        "passed": passed,
        "disposition": "accepted" if passed else "candidate_requires_review",
        "reference_implementation": REFERENCE_IMPLEMENTATION,
        "candidate_implementation": CANDIDATE_IMPLEMENTATION,
        "delta_convention": "candidate_minus_reference",
        "steps": len(simple_rows),
        "implementation_summaries": {
            "microgrid-simulator": _summarize_rows(simple_rows, plant),
            "pymgrid": _summarize_rows(pymgrid_rows, plant),
        },
        "max_abs_delta": max_abs_delta,
        "aggregate_energy_delta_mwh": aggregate_delta_mwh,
        "failed_fields": failed_fields,
        "failed_aggregate_fields": failed_aggregate_fields,
        "acceptance_criteria": asdict(criteria),
    }
    return pd.DataFrame(comparison_rows), metrics


SUMMARY_LABELS = {
    "load_mwh": "Load demand (MWh)",
    "served_mwh": "Load served (MWh)",
    "unserved_mwh": "Unserved energy (MWh)",
    "pv_available_mwh": "PV available (MWh)",
    "pv_used_mwh": "PV used (MWh)",
    "pv_spill_mwh": "PV spill (MWh)",
    "battery_charge_mwh": "Battery terminal charge (MWh)",
    "battery_discharge_mwh": "Battery terminal discharge (MWh)",
    "diesel_mwh": "Diesel generation (MWh)",
    "excess_mwh": "Excess generation (MWh)",
    "initial_soc": "Initial SOC",
    "final_soc": "Final SOC",
    "max_abs_balance_residual_mw": "Maximum balance residual (MW)",
}
