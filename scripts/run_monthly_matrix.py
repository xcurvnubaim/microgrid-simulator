#!/usr/bin/env python3
"""CLI runner for the continuous one-month frozen-policy evaluation.

Examples:
    # Resolve and validate the 54-job matrix without running it
    uv run python scripts/run_monthly_matrix.py --dry-run

    # One Rule-F3 monthly smoke job (full 2,976 steps)
    uv run python scripts/run_monthly_matrix.py --scenario E0 --policy rule_f3

    # Short smoke runs (dev only; written under _smoke/ and never aggregated)
    uv run python scripts/run_monthly_matrix.py \
        --scenario E0 --policy sac_f3 --seed 0 --smoke-steps 48

    # Full matrix with parallel workers, then aggregate
    uv run python scripts/run_monthly_matrix.py --all --workers 4

Completed jobs are never overwritten implicitly: a job reruns only when its
trajectory/metrics are missing or fail validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from microgrid_simulator.experiments.monthly_evaluation import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    aggregate_results,
    build_job_registry,
    run_monthly_job,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--all", action="store_true", help="run the full 54-job matrix")
    parser.add_argument(
        "--include-summary",
        action="store_true",
        help="extend the selection with the 18-job sac_f3_summary arm",
    )
    parser.add_argument("--scenario", action="append", help="E0..E5 (repeatable)")
    parser.add_argument(
        "--policy",
        action="append",
        help="rule_f3 | schedule | pypsa_rh_f3 | sac_f3 | sac_none_f3",
    )
    parser.add_argument("--seed", type=int, action="append", help="SAC seed filter (repeatable)")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve jobs and artifacts, run nothing"
    )
    parser.add_argument(
        "--aggregate-only", action="store_true", help="only rebuild combined summaries"
    )
    parser.add_argument(
        "--smoke-steps", type=int, default=None, help="dev: truncate jobs to N steps under _smoke/"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="result root (default: reports/monthly_march_2026)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = args.seed
    jobs = build_job_registry(
        output_root=args.output_root,
        scenarios=args.scenario,
        policies=args.policy,
        seeds=seeds,
        include_summary=args.include_summary,
    )
    if args.include_summary and not args.policy:
        summary_only = build_job_registry(
            output_root=args.output_root,
            scenarios=args.scenario,
            policies=["sac_f3_summary"],
            seeds=seeds,
        )
        known = {job.job_id for job in jobs}
        jobs = jobs + [job for job in summary_only if job.job_id not in known]

    if args.dry_run:
        missing = [
            str(job.artifact_path)
            for job in jobs
            if job.artifact_path is not None and not job.artifact_path.is_file()
        ]
        print(json.dumps({"jobs": len(jobs), "missing_artifacts": missing}, indent=2))
        for job in jobs:
            state = (
                "complete"
                if (args.output_root / job.job_id / "metrics.json").is_file()
                else "pending"
            )
            artifact = str(job.artifact_path) if job.artifact_path else "-"
            print(f"{job.job_id:28s} [{state:8s}] {job.config_path.name} <- {artifact}")
        return 1 if missing else 0

    if args.aggregate_only:
        result = aggregate_results(output_root=args.output_root)
        print(json.dumps(result, indent=2))
        return 0

    selected = jobs if args.all else jobs
    if not args.all and not (args.scenario or args.policy):
        print("Nothing to do: pass --all or a --scenario/--policy filter.", file=sys.stderr)
        return 2

    failures: list[str] = []
    if args.workers <= 1 or len(selected) == 1:
        for job in selected:
            try:
                metrics = run_monthly_job(
                    job, output_root=args.output_root, smoke_steps=args.smoke_steps
                )
                status = metrics["status"]
                print(
                    f"{job.job_id}: steps={status['steps']} "
                    f"completed={status['completed']} "
                    f"unserved_kwh={metrics['summary']['unserved_kwh']:.3f}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{job.job_id}: {exc}")
                print(f"{job.job_id}: FAILED - {exc}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_monthly_job, job, args.output_root, args.smoke_steps): job
                for job in selected
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    metrics = future.result()
                    status = metrics["status"]
                    print(
                        f"{job.job_id}: steps={status['steps']} "
                        f"completed={status['completed']} "
                        f"unserved_kwh={metrics['summary']['unserved_kwh']:.3f}"
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{job.job_id}: {exc}")
                    print(f"{job.job_id}: FAILED - {exc}", file=sys.stderr)

    result = aggregate_results(output_root=args.output_root)
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
