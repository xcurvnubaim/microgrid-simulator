"""Cross-implementation verification against a matched pymgrid plant.

This experiment deliberately verifies only the shared, lossless scheduling
contract.  It does not treat pymgrid as physical ground truth and does not
compare either simulator's native reward.  Both implementations receive the
same exogenous series and open-loop actions; their outputs are normalized into
one accounting schema before deltas are computed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from microgrid_simulator.experiments.pymgrid_verification.compare import _compare
from microgrid_simulator.experiments.pymgrid_verification.constants import (
    CANDIDATE_IMPLEMENTATION,
    PYMGRID_UPSTREAM_COMMIT,
    REFERENCE_IMPLEMENTATION,
)
from microgrid_simulator.experiments.pymgrid_verification.harness import (
    _run_pymgrid,
    _run_simple,
)
from microgrid_simulator.experiments.pymgrid_verification.plant import (
    AcceptanceCriteria,
    CommonPlant,
    common_fixture,
)
from microgrid_simulator.experiments.pymgrid_verification.report import _report


def run_verification(
    output_dir: str | Path,
    criteria: AcceptanceCriteria | None = None,
) -> dict[str, dict[str, Any]]:
    """Run lossless and efficiency-aware cases and persist auditable artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plant = CommonPlant()
    steps = common_fixture()
    criteria = criteria or AcceptanceCriteria()
    all_metrics: dict[str, dict[str, Any]] = {}

    for efficiency in (1.0, 0.96):
        label = f"{efficiency:.2f}"
        simple_rows = _run_simple(plant, steps, efficiency)
        pymgrid_rows = _run_pymgrid(plant, steps, efficiency)
        frame, metrics = _compare(simple_rows, pymgrid_rows, plant, criteria)
        pd.DataFrame(simple_rows).to_csv(
            destination / f"microgrid_simulator_efficiency_{label}.csv",
            index=False,
        )
        pd.DataFrame(pymgrid_rows).to_csv(
            destination / f"pymgrid_efficiency_{label}.csv",
            index=False,
        )
        frame.to_csv(destination / f"trajectory_efficiency_{label}.csv", index=False)
        all_metrics[label] = metrics

    payload = {
        "experiment": "pymgrid-common-model-verification",
        "comparison_policy": {
            "scope": "matched lossless islanded scheduling contract",
            "reference_implementation": REFERENCE_IMPLEMENTATION,
            "candidate_implementation": CANDIDATE_IMPLEMENTATION,
            "delta_convention": "candidate_minus_reference",
            "on_mismatch": "fail_candidate_and_review_mapping_then_candidate_implementation",
            "silent_output_substitution": False,
        },
        "software": {
            "microgrid-simulator": {
                "version": version("microgrid-simulator"),
                "backend": "simple",
                "license": "MIT",
            },
            "pymgrid": {
                "distribution": "python-microgrid",
                "version": version("python-microgrid"),
                "commit": PYMGRID_UPSTREAM_COMMIT,
                "license": "LGPL-3.0",
            },
        },
        "plant": asdict(plant),
        "fixtures": [asdict(step) for step in steps],
        "results": all_metrics,
        "overall_passed": all(metrics["passed"] for metrics in all_metrics.values()),
    }
    (destination / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (destination / "report.md").write_text(_report(payload), encoding="utf-8")
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/pymgrid_common_model_verification"),
    )
    args = parser.parse_args()
    results = run_verification(args.output_dir)
    overall_passed = all(metrics["passed"] for metrics in results.values())
    for efficiency, metrics in results.items():
        print(
            f"efficiency={efficiency}: passed={metrics['passed']} "
            f"max_soc_delta={metrics['max_abs_delta']['soc']:.3g}"
        )
    if not overall_passed:
        raise SystemExit(1)
