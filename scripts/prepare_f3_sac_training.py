#!/usr/bin/env python3
"""Prepare matched 15-minute F3 SAC training jobs without launching them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from microgrid_simulator.config import load_settings

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/f3-sac-training.yaml"
F3_CONFIG = ROOT / "configs/forecast-f3-15min-corrected.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=ROOT / "f3_jobs.json")
    parser.add_argument("--commands", action="store_true")
    parser.add_argument("--command-prefix", default="uv run")
    args = parser.parse_args()
    registry: dict[str, Any] = yaml.safe_load(REGISTRY.read_text())
    timesteps = registry["training"].get("total_timesteps", 1000000)
    jobs = []
    for scenario, base_rel in registry["scenarios"].items():
        for policy in registry["policies"]:
            for seed in policy["seeds"]:
                mode = policy["forecast_mode"]
                config = ROOT / base_rel
                settings = load_settings(config)
                if (
                    settings.forecast.forecast_steps != 96
                    or settings.forecast.issue_frequency_hours != 0.25
                ):
                    raise ValueError(
                        f"{scenario}: F3 config must use 96 points and 15-minute issuance"
                    )
                artifact = (
                    ROOT
                    / registry["training"]["artifact_root"]
                    / scenario
                    / mode
                    / f"seed-{seed}"
                )
                jobs.append(
                    {
                        "scenario": scenario,
                        "policy": policy["id"],
                        "forecast_mode": "cached" if mode == "f3" else "none",
                        "seed": seed,
                        "config": str(config.relative_to(ROOT)),
                        "artifact_dir": str(artifact.relative_to(ROOT)),
                        "model": str((artifact / "sac_microgrid.zip").relative_to(ROOT)),
                        "command": (
                            f"{args.command_prefix} microgrid-sim train --algo sac --timesteps {timesteps} "
                            f"--config {config.relative_to(ROOT)} --forecast-mode "
                            f"{'cached' if mode == 'f3' else 'none'} --artifact-dir "
                            f"{artifact.relative_to(ROOT)} --seed {seed} "
                            f"--run-id f3-{scenario.lower()}-{mode}-seed{seed}"
                        ),
                    }
                )
    if len(jobs) not in {36, 42}:
        raise ValueError(f"expected 36 or 42 F3 jobs, got {len(jobs)}")
    args.json.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"Prepared {len(jobs)} F3 jobs; no training launched.")
    if args.commands:
        print("\n".join(job["command"] for job in jobs))


if __name__ == "__main__":
    main()
