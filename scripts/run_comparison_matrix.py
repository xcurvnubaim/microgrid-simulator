#!/usr/bin/env python3
"""Concurrency-enabled runner for the Module 6 comparison matrix.

Spans E0-E5, 9 March telemetry windows, and all policies: Rule-F0, Schedule,
PyPSA-RH-F0, SAC-F0, and SAC-none (evaluation of 3 seeds each). Evaluates in parallel
using a multiprocessing ProcessPoolExecutor to speed up MILP/RL rollouts.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microgrid_simulator.config import load_settings
from microgrid_simulator.experiments.telemetry_replay import run_telemetry_replay
from microgrid_simulator.ui.rollout import run_rollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
E0_YAML = CONFIG_DIR / "islanded-baseline-72h.yaml"

SCENARIOS = {
    "E0": E0_YAML,
    "E1": CONFIG_DIR / "economic-e1-high-diesel-fuel.yaml",
    "E2": CONFIG_DIR / "economic-e2-low-battery-use.yaml",
    "E3": CONFIG_DIR / "economic-e3-high-fuel-low-battery-use.yaml",
    "E4": CONFIG_DIR / "economic-e4-high-pv-waste.yaml",
    "E5": CONFIG_DIR / "economic-e5-high-unserved.yaml",
}

MARCH_WINDOWS = json.loads(
    (REPO_ROOT / "configs" / "march-windows-2026.json").read_text(encoding="utf-8")
)["starts"]


@dataclass
class JobSpec:
    scenario_id: str
    config_path: Path
    window_start: str
    policy: str
    seed: int
    output_dir: Path


def run_single_job(spec: JobSpec) -> dict[str, Any]:
    """Execute one rollout in a separate process."""
    try:
        settings = load_settings(spec.config_path)
        settings.episode.telemetry_start = spec.window_start
        settings.episode.horizon_hours = 72.0

        # RL policy loading mapping once artifacts are generated:
        # artifacts/sac/{scenario_id}/cached-1m/seed-{seed}/
        # For Rule/Schedule/PyPSA-RH, standard replay paths are used.
        # PyPSA-RH uses the project rolling-horizon wrapper.
        if spec.policy in {"sac_f0", "sac_none"}:
            mode = "cached" if spec.policy == "sac_f0" else "noforecast"
            artifact_dir = (
                REPO_ROOT
                / "artifacts"
                / "sac"
                / spec.scenario_id
                / f"{mode}-1m"
                / f"seed-{spec.seed}"
            )
            model_zip = artifact_dir / "sac_microgrid.zip"
            if not model_zip.exists():
                raise FileNotFoundError(f"Missing SAC model: {model_zip}")
            settings.rl.forecast_mode = "cached" if spec.policy == "sac_f0" else "none"
            result = run_rollout(
                settings,
                policy="rl",
                seed=spec.seed,
                rl_artifact=model_zip,
                rl_algo="sac",
            )
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            import pandas as pd

            pd.DataFrame(result["rows"]).to_csv(spec.output_dir / "trajectory.csv", index=False)
            (spec.output_dir / "metrics.json").write_text(
                json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
            )
            metrics = result
            return {
                "status": "success",
                "scenario": spec.scenario_id,
                "window": spec.window_start,
                "policy": spec.policy,
                "seed": spec.seed,
                "unserved_kwh": metrics["totals"]["unserved_kwh"],
                "diesel_kwh": metrics["totals"]["diesel_kwh"],
                "reward": metrics["totals"].get("total_reward", 0.0),
            }

        metrics = run_telemetry_replay(
            settings,
            spec.config_path,
            spec.output_dir,
            policy=spec.policy,
            seed=spec.seed,
        )
        return {
            "status": "success",
            "scenario": spec.scenario_id,
            "window": spec.window_start,
            "policy": spec.policy,
            "seed": spec.seed,
            "unserved_kwh": metrics["totals"]["unserved_kwh"],
            "diesel_kwh": metrics["totals"]["diesel_kwh"],
            "reward": metrics["totals"].get("total_reward", 0.0),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "scenario": spec.scenario_id,
            "window": spec.window_start,
            "policy": spec.policy,
            "seed": spec.seed,
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EMS comparison matrix")
    parser.add_argument("--workers", type=int, default=4, help="Process pool workers")
    parser.add_argument(
        "--policy",
        type=str,
        choices=["rule", "schedule", "pypsa_rh", "sac_f0", "sac_none"],
        help="Filter one policy",
    )
    parser.add_argument(
        "--include-sac",
        action="store_true",
        help="Explicitly enable SAC jobs; omitted by default until artifacts are validated",
    )
    parser.add_argument(
        "--scenario", type=str, choices=list(SCENARIOS.keys()), help="Filter single scenario"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print job specs without executing")
    args = parser.parse_args()

    jobs: list[JobSpec] = []
    output_base = REPO_ROOT / "reports" / "experiments" / "comparison_matrix"

    for scenario_id, config_path in SCENARIOS.items():
        if args.scenario and scenario_id != args.scenario:
            continue
        for window in MARCH_WINDOWS:
            # Rule, Schedule, and PyPSA-RH baselines
            for policy in ["rule", "schedule", "pypsa_rh"]:
                if args.policy and policy != args.policy:
                    continue
                out_dir = (
                    output_base / scenario_id / policy / window.replace(" ", "_").replace(":", "-")
                )
                if (out_dir / "metrics.json").exists() and (out_dir / "trajectory.csv").exists():
                    continue
                jobs.append(JobSpec(scenario_id, config_path, window, policy, 0, out_dir))

            # RL policies: SAC-F0 and SAC-none (three seeds each, retrained per scenario)
            # Reserved in spec; execution requires retraining.
            if not args.policy or args.policy.startswith("sac_"):
                for policy in ["sac_f0", "sac_none"]:
                    if args.policy and policy != args.policy:
                        continue
                    for seed in range(3):
                        out_dir = (
                            output_base
                            / scenario_id
                            / f"{policy}_seed{seed}"
                            / window.replace(" ", "_").replace(":", "-")
                        )
                        if (out_dir / "metrics.json").exists() and (
                            out_dir / "trajectory.csv"
                        ).exists():
                            continue
                        jobs.append(
                            JobSpec(scenario_id, config_path, window, policy, seed, out_dir)
                        )

    LOGGER.info("Resolved %d total evaluation jobs", len(jobs))
    if args.dry_run:
        for job in jobs[:10]:
            print(
                f"Dry-run spec: {job.scenario_id} | {job.policy} | "
                f"{job.window_start} -> {job.output_dir.name}"
            )
        if len(jobs) > 10:
            print(f"... and {len(jobs) - 10} more")
        return

    # Check baseline filters to prevent running SAC before artifacts exist
    baseline_policies = {"rule", "schedule", "pypsa_rh"}
    active_jobs = [
        j
        for j in jobs
        if j.policy in baseline_policies or (args.include_sac and j.policy.startswith("sac_"))
    ]
    LOGGER.info(
        "Launching %d jobs (%s)",
        len(active_jobs),
        "including SAC" if args.include_sac else "baselines only",
    )

    results = []
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_job, job): job for job in active_jobs}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            if res["status"] == "success":
                LOGGER.info(
                    "COMPLETED: %s | %s | %s | Reward %.1f",
                    res["scenario"],
                    res["policy"],
                    res["window"],
                    res["reward"],
                )
            else:
                LOGGER.error(
                    "FAILED: %s | %s | %s | Error: %s",
                    res["scenario"],
                    res["policy"],
                    res["window"],
                    res["error"],
                )
                failed += 1

    summary_file = output_base / "baseline_run_summary.json"
    output_base.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info(
        "Baseline runs complete. Total %d | Failed %d. Summary at %s",
        len(results),
        failed,
        summary_file,
    )


if __name__ == "__main__":
    main()
