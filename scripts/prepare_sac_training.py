#!/usr/bin/env python3
"""Validate and print the 36 SAC training jobs without launching them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from microgrid_simulator.config import load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "configs" / "comparison-matrix.yaml"


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("training", {}).get("do_not_launch") is not True:
        raise ValueError("training registry must remain do_not_launch: true")
    return data


def prepare_jobs(registry: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for scenario_id, config_rel in registry["scenarios"].items():
        config_path = REPO_ROOT / config_rel
        settings = load_settings(config_path)
        if settings.rl.hard_unserved:
            raise ValueError(f"{scenario_id}: hard_unserved must be false for E0-E5")
        for policy in registry["policies"]:
            if not policy["id"].startswith("sac_"):
                continue
            mode = policy["forecast_mode"]
            artifact_mode = "cached" if mode == "cached" else "noforecast"
            for seed in policy["seeds"]:
                artifact_dir = (
                    REPO_ROOT
                    / registry["training"]["artifact_root"]
                    / scenario_id
                    / f"{artifact_mode}-1m"
                    / f"seed-{seed}"
                )
                jobs.append(
                    {
                        "scenario": scenario_id,
                        "config": str(config_path.relative_to(REPO_ROOT)),
                        "policy": policy["id"],
                        "forecast_mode": mode,
                        "seed": seed,
                        "timesteps": registry["training"]["total_timesteps"],
                        "artifact_dir": str(artifact_dir.relative_to(REPO_ROOT)),
                        "model": str((artifact_dir / "sac_microgrid.zip").relative_to(REPO_ROOT)),
                        "command": (
                            "uv run microgrid-sim train --algo sac "
                            f"--timesteps {registry['training']['total_timesteps']} "
                            f"--config {config_rel} --forecast-mode {mode} "
                            f"--artifact-dir {artifact_dir.relative_to(REPO_ROOT)} "
                            f"--seed {seed} --run-id {scenario_id.lower()}-{mode}-seed{seed}"
                        ),
                    }
                )
    if len(jobs) != 36:
        raise ValueError(f"expected 36 SAC jobs, resolved {len(jobs)}")
    return jobs


def missing_artifacts(jobs: list[dict[str, Any]]) -> list[str]:
    """Return missing model/normalizer paths without loading or executing them."""
    missing: list[str] = []
    for job in jobs:
        artifact = REPO_ROOT / job["model"]
        vecnorm = artifact.with_suffix("")
        vecnorm = Path(str(vecnorm) + "_vecnormalize.pkl")
        for path in (artifact, vecnorm):
            if not path.is_file():
                missing.append(str(path.relative_to(REPO_ROOT)))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the prepared job list")
    parser.add_argument("--commands", action="store_true", help="print shell commands")
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="fail if all model and VecNormalize files are not already present",
    )
    args = parser.parse_args()
    jobs = prepare_jobs(load_registry())
    missing = missing_artifacts(jobs)
    if args.require_artifacts and missing:
        raise SystemExit("Missing SAC artifacts:\n" + "\n".join(missing))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(jobs)} SAC training jobs; no training launched.")
    print(f"Artifact contract: {'complete' if not missing else f'{len(missing)} files missing'}")
    if args.commands:
        for job in jobs:
            print(job["command"])


if __name__ == "__main__":
    main()
