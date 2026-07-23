"""Quasi-static pandapower checks for a fixed telemetry rule trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microgrid_simulator.config import Settings
from microgrid_simulator.ui.rollout import run_rollout

SCHEDULING_FIELDS = (
    "load_kw",
    "served_kw",
    "unserved_kw",
    "pv_available_kw",
    "pv_used_kw",
    "pv_wasted_kw",
    "battery_charge_kw",
    "battery_discharge_kw",
    "diesel_kw",
    "soc_pct",
)


def _pandapower_settings(config: Path) -> Settings:
    raw = Settings.from_yaml(config).model_dump()
    raw["backend"]["name"] = "pandapower"
    raw["topology"]["solver"] = "ac"
    return Settings(**raw)


def _selected_indices(reference: pd.DataFrame, validation: pd.DataFrame) -> dict[int, list[str]]:
    median_load = float(reference["load_kw"].median())
    candidates = {
        "representative_load": int((reference["load_kw"] - median_load).abs().idxmin()),
        "peak_load": int(reference["load_kw"].idxmax()),
        "peak_pv": int(reference["pv_available_kw"].idxmax()),
        "peak_unserved": int(reference["unserved_kw"].idxmax()),
        "minimum_soc": int(reference["soc_pct"].idxmin()),
        "worst_voltage": int(
            np.maximum(
                (validation["min_voltage_pu"] - 1.0).abs(),
                (validation["max_voltage_pu"] - 1.0).abs(),
            ).idxmax()
        ),
        "peak_line_loading": int(validation["max_line_loading_pct"].idxmax()),
    }
    selected: dict[int, list[str]] = {}
    for label, index in candidates.items():
        selected.setdefault(index, []).append(label)
    return selected


def run_validation(config: Path, reference_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run all 288 AC states and persist a compact selected-state audit."""
    settings = _pandapower_settings(config)
    result = run_rollout(settings, policy="rule", seed=0)
    validation = pd.DataFrame(result["rows"])
    reference = pd.read_csv(reference_path)
    if len(validation) != len(reference):
        raise ValueError(
            f"pandapower/reference length mismatch: {len(validation)} != {len(reference)}"
        )

    deltas = {
        field: float((validation[field] - reference[field]).abs().max())
        for field in SCHEDULING_FIELDS
    }
    selected = _selected_indices(reference, validation)
    selected_rows: list[dict[str, Any]] = []
    for index, labels in sorted(selected.items()):
        row = validation.iloc[index]
        selected_rows.append(
            {
                "labels": ", ".join(labels),
                "step": int(row["step"]),
                "timestamp": str(reference.iloc[index]["timestamp"]),
                "load_kw": float(row["load_kw"]),
                "pv_available_kw": float(row["pv_available_kw"]),
                "diesel_kw": float(row["diesel_kw"]),
                "battery_kw": float(row["battery_kw"]),
                "soc_pct": float(row["soc_pct"]),
                "unserved_kw": float(row["unserved_kw"]),
                "min_voltage_pu": float(row["min_voltage_pu"]),
                "max_voltage_pu": float(row["max_voltage_pu"]),
                "max_line_loading_pct": float(row["max_line_loading_pct"]),
                "violation_types": ", ".join(row["constraint_violation_types"]),
            }
        )

    metrics: dict[str, Any] = {
        "scenario": settings.scenario.name,
        "config": str(config),
        "reference_trajectory": str(reference_path),
        "steps": len(validation),
        "all_power_flows_converged": len(validation) == 288,
        "scheduling_field_max_abs_delta": deltas,
        "max_abs_scheduling_delta": max(deltas.values()),
        "minimum_voltage_pu": float(validation["min_voltage_pu"].min()),
        "maximum_voltage_pu": float(validation["max_voltage_pu"].max()),
        "maximum_line_loading_pct": float(validation["max_line_loading_pct"].max()),
        "voltage_violation_intervals": int(
            validation["constraint_violation_types"].map(lambda x: "voltage" in x).sum()
        ),
        "line_loading_violation_intervals": int(
            validation["constraint_violation_types"].map(lambda x: "line_loading" in x).sum()
        ),
        "selected_states": selected_rows,
        "claim_boundary": (
            "Quasi-static sensitivity check on assumed balanced feeder data; "
            "not physical-campus validation."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_states.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Held-out pandapower scheduling-state check",
        "",
        f"- Full AC solves: **{metrics['steps']}**; all converged: "
        f"**{str(metrics['all_power_flows_converged']).lower()}**",
        f"- Maximum scheduling-field delta from the simple backend: "
        f"**{metrics['max_abs_scheduling_delta']:.3g}**",
        f"- Voltage range: **{metrics['minimum_voltage_pu']:.6f}–"
        f"{metrics['maximum_voltage_pu']:.6f} pu**",
        f"- Maximum line loading: **{metrics['maximum_line_loading_pct']:.3f}%**",
        f"- Voltage / line-loading violation intervals: "
        f"**{metrics['voltage_violation_intervals']} / "
        f"{metrics['line_loading_violation_intervals']}**",
        "",
        "## Selected states",
        "",
        "| Labels | Timestamp | Load kW | PV kW | Diesel kW | Battery kW | SOC % | "
        "Unserved kW | V min | V max | Line % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['labels']} | {row['timestamp']} | {row['load_kw']:.3f} | "
            f"{row['pv_available_kw']:.3f} | {row['diesel_kw']:.3f} | "
            f"{row['battery_kw']:.3f} | {row['soc_pct']:.3f} | "
            f"{row['unserved_kw']:.3f} | {row['min_voltage_pu']:.6f} | "
            f"{row['max_voltage_pu']:.6f} | {row['max_line_loading_pct']:.3f} |"
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "The configured 0.4 kV buses, balanced lines, load split, and reactive power are",
        "assumed. The islanded slack is a numerical grid-forming reference. These results",
        "test implementation consistency and sensitivity only; they do not validate the",
        "physical campus feeder, losses, voltage, phase behavior, or equipment ratings.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = run_validation(args.config, args.reference, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
