#!/usr/bin/env python3
"""Concurrency-enabled runner for the Module 6 F3 comparison matrix.

Evaluates SAC-F3, SAC-none-F3, SAC-F3-summary, and PyPSA-RH-F3 across the configured
economic scenarios and 9 March telemetry windows in parallel.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from microgrid_simulator.config import load_settings
from microgrid_simulator.experiments.telemetry_replay import run_telemetry_replay
from microgrid_simulator.ui.rollout import run_rollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

SCENARIOS = {
    "E0": CONFIG_DIR / "f3-e0.yaml",
    "E1": CONFIG_DIR / "f3-e1.yaml",
    "E2": CONFIG_DIR / "f3-e2.yaml",
    "E3": CONFIG_DIR / "f3-e3.yaml",
    "E4": CONFIG_DIR / "f3-e4.yaml",
    "E5": CONFIG_DIR / "f3-e5.yaml",
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

        # Point test/evaluation cache paths to the test/val split files
        # The F3 configs default to the train paths. For evaluation on March,
        # we override the forecast cache paths to point to the test split.
        settings.forecast.cache_path = "data/f3/test/forecasts.jsonl"
        settings.forecast.manifest_path = "data/f3/test/forecast_manifest.json"
        settings.forecast.val_cache_path = "data/f3/val/forecasts.jsonl"
        settings.forecast.val_manifest_path = "data/f3/val/forecast_manifest.json"

        if spec.policy in {
            "sac_f3",
            "sac_none_f3",
            "sac_f3_summary",
            "sac_f3_hard",
            "sac_none_f3_hard",
        }:
            mode = "f3" if "sac_f3" in spec.policy else "none"
            scenario_folder = "E1_hard" if "_hard" in spec.policy else spec.scenario_id
            if spec.policy == "sac_f3_summary":
                artifact_dir = (
                    REPO_ROOT
                    / "artifacts"
                    / "sac"
                    / "f3-summary"
                    / scenario_folder
                    / f"seed-{spec.seed}"
                )
            else:
                artifact_dir = (
                    REPO_ROOT
                    / "artifacts"
                    / "sac"
                    / "f3-15min"
                    / scenario_folder
                    / mode
                    / f"seed-{spec.seed}"
                )
            model_zip = artifact_dir / "sac_microgrid.zip"
            if not model_zip.exists():
                raise FileNotFoundError(f"Missing SAC model: {model_zip}")

            settings.rl.forecast_mode = "cached" if mode == "f3" else "none"
            settings.rl.forecast_representation = (
                "summary" if spec.policy == "sac_f3_summary" else "raw"
            )
            result = run_rollout(
                settings,
                policy="rl",
                seed=spec.seed,
                rl_artifact=model_zip,
                rl_algo="sac",
            )
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result["rows"]).to_csv(spec.output_dir / "trajectory.csv", index=False)
            (spec.output_dir / "metrics.json").write_text(
                json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
            )
            return {
                "status": "success",
                "scenario": spec.scenario_id,
                "window": spec.window_start,
                "policy": spec.policy,
                "seed": spec.seed,
                "unserved_kwh": result["totals"]["unserved_kwh"],
                "diesel_kwh": result["totals"]["diesel_kwh"],
                "reward": result["totals"].get("total_reward", 0.0),
            }

        # Otherwise, run the PyPSA rolling-horizon replay.
        metrics = run_telemetry_replay(
            settings,
            spec.config_path,
            spec.output_dir,
            policy="pypsa_rh",
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
    parser = argparse.ArgumentParser(description="Run F3 EMS comparison matrix")
    parser.add_argument("--workers", type=int, default=4, help="Process pool workers")
    parser.add_argument(
        "--policy",
        type=str,
        choices=[
            "pypsa_rh_f3",
            "sac_f3",
            "sac_none_f3",
            "sac_f3_summary",
            "sac_f3_hard",
            "sac_none_f3_hard",
        ],
        help="Filter one policy",
    )
    parser.add_argument(
        "--scenario", type=str, choices=list(SCENARIOS.keys()), help="Filter single scenario"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print job specs without executing")
    parser.add_argument(
        "--summary-pilot",
        action="store_true",
        help="Evaluate only SAC-F3-summary on E0, E1, and E5",
    )
    parser.add_argument(
        "--summary-remaining",
        action="store_true",
        help="Evaluate only SAC-F3-summary on E2, E3, and E4",
    )
    args = parser.parse_args()
    if args.summary_pilot and args.summary_remaining:
        parser.error("choose only one summary scenario group")

    jobs: list[JobSpec] = []
    output_base = REPO_ROOT / "reports" / "experiments" / "comparison_matrix_f3"

    scenario_items = SCENARIOS.items()
    if args.summary_pilot:
        scenario_items = ((name, SCENARIOS[name]) for name in ("E0", "E1", "E5"))
    elif args.summary_remaining:
        scenario_items = ((name, SCENARIOS[name]) for name in ("E2", "E3", "E4"))
    for scenario_id, config_path in scenario_items:
        if args.scenario and scenario_id != args.scenario:
            continue
        for window in MARCH_WINDOWS:
            if args.summary_pilot or args.summary_remaining:
                out_dir = (
                    output_base
                    / scenario_id
                    / "sac_f3_summary"
                    / window.replace(" ", "_").replace(":", "-")
                )
                for seed in range(3):
                    seed_dir = out_dir.with_name(f"{out_dir.name}_seed{seed}")
                    if not (
                        (seed_dir / "metrics.json").exists()
                        and (seed_dir / "trajectory.csv").exists()
                    ):
                        jobs.append(
                            JobSpec(
                                scenario_id,
                                config_path,
                                window,
                                "sac_f3_summary",
                                seed,
                                seed_dir,
                            )
                        )
                continue
            # PyPSA-RH baseline using F3
            if not args.policy or args.policy == "pypsa_rh_f3":
                out_dir = (
                    output_base
                    / scenario_id
                    / "pypsa_rh_f3"
                    / window.replace(" ", "_").replace(":", "-")
                )
                if not (
                    (out_dir / "metrics.json").exists() and (out_dir / "trajectory.csv").exists()
                ):
                    jobs.append(
                        JobSpec(scenario_id, config_path, window, "pypsa_rh_f3", 0, out_dir)
                    )

            # RL policies: existing F3 ablations and the isolated summary pilot.
            if not args.policy or args.policy.startswith("sac_"):
                policies = ["sac_f3", "sac_none_f3"]
                if args.policy == "sac_f3_summary" or not args.policy:
                    policies.append("sac_f3_summary")
                for policy in policies:
                    if args.policy and policy != args.policy:
                        continue
                    for seed in range(3):
                        out_dir = (
                            output_base
                            / scenario_id
                            / f"{policy}_seed{seed}"
                            / window.replace(" ", "_").replace(":", "-")
                        )
                        if not (
                            (out_dir / "metrics.json").exists()
                            and (out_dir / "trajectory.csv").exists()
                        ):
                            jobs.append(
                                JobSpec(scenario_id, config_path, window, policy, seed, out_dir)
                            )
                # Diagnostic hard-unserved trained policy under E1
                if scenario_id == "E1":
                    for policy in ["sac_f3_hard", "sac_none_f3_hard"]:
                        if args.policy and policy != args.policy:
                            continue
                        for seed in range(3):
                            out_dir = (
                                output_base
                                / scenario_id
                                / f"{policy}_seed{seed}"
                                / window.replace(" ", "_").replace(":", "-")
                            )
                            if not (
                                (out_dir / "metrics.json").exists()
                                and (out_dir / "trajectory.csv").exists()
                            ):
                                jobs.append(
                                    JobSpec(scenario_id, config_path, window, policy, seed, out_dir)
                                )

    LOGGER.info("Resolved %d total F3 evaluation jobs", len(jobs))
    if args.dry_run:
        for job in jobs[:10]:
            print(
                f"Dry-run spec: {job.scenario_id} | {job.policy} | "
                f"{job.window_start} -> {job.output_dir.name}"
            )
        if len(jobs) > 10:
            print(f"... and {len(jobs) - 10} more")
        return

    results = []
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_job, job): job for job in jobs}
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

    summary_file = output_base / "f3_run_summary.json"
    output_base.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info(
        "F3 evaluation runs complete. Total %d | Failed %d. Summary at %s",
        len(results),
        failed,
        summary_file,
    )


if __name__ == "__main__":
    main()
