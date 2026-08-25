#!/usr/bin/env python3
"""Prepare selected SAC forecast-summary jobs without launching them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from microgrid_simulator.config import load_settings

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_NAMES = ("E0", "E1", "E2", "E3", "E4", "E5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=ROOT / "f3_summary_pilot_jobs.json")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--command-prefix", default="uv run")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_NAMES,
        default=["E0", "E1", "E5"],
    )
    args = parser.parse_args()

    jobs: list[dict[str, Any]] = []
    for scenario in args.scenarios:
        config = ROOT / f"configs/f3-{scenario.lower()}.yaml"
        settings = load_settings(config)
        if (
            settings.forecast.forecast_steps != 96
            or settings.forecast.issue_frequency_hours != 0.25
        ):
            raise ValueError(
                f"{scenario}: pilot requires the corrected 96-point F3 contract"
            )
        for seed in range(3):
            artifact = ROOT / "artifacts" / "sac" / "f3-summary" / scenario / f"seed-{seed}"
            jobs.append(
                {
                    "scenario": scenario,
                    "policy": "sac_f3_summary",
                    "forecast_mode": "cached",
                    "forecast_representation": "summary",
                    "seed": seed,
                    "config": str(config.relative_to(ROOT)),
                    "artifact_dir": str(artifact.relative_to(ROOT)),
                    "model": str((artifact / "sac_microgrid.zip").relative_to(ROOT)),
                    "command": (
                        f"{args.command_prefix} microgrid-sim train --algo sac "
                        f"--timesteps {args.timesteps} --config {config.relative_to(ROOT)} "
                        "--forecast-mode cached --forecast-representation summary "
                        f"--artifact-dir {artifact.relative_to(ROOT)} --seed {seed} "
                        f"--run-id f3-summary-{scenario.lower()}-seed{seed}"
                    ),
                }
            )
    args.json.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(jobs)} summary jobs for {args.scenarios}; no training launched.")


if __name__ == "__main__":
    main()
