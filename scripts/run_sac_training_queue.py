#!/usr/bin/env python3
"""Run the prepared SAC training queue with a pool of parallel workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_job(job: dict[str, str | int]) -> tuple[dict[str, str | int], int, float]:
    scenario = job["scenario"]
    mode = job["forecast_mode"]
    seed = job["seed"]
    command = job["command"]
    
    log_dir = REPO_ROOT / "logs" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{scenario.lower()}_{mode}_seed{seed}.log"
    
    print(f"[{time.strftime('%H:%M:%S')}] STARTING: {scenario} ({mode}) seed {seed}", flush=True)
    
    start_time = time.time()
    with log_file.open("w", encoding="utf-8") as fh:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(REPO_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        rc = process.wait()
        
    duration = time.time() - start_time
    status = "SUCCESS" if rc == 0 else f"FAILED (rc={rc})"
    print(
        f"[{time.strftime('%H:%M:%S')}] FINISHED: {scenario} ({mode}) seed {seed} "
        f"-> {status} in {duration:.1f}s (log: logs/training/{log_file.name})",
        flush=True
    )
    return job, rc, duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-json", type=Path, default=REPO_ROOT / "jobs.json", help="path to jobs.json")
    parser.add_argument("--workers", type=int, default=4, help="number of parallel training jobs")
    args = parser.parse_args()

    if not args.jobs_json.is_file():
        print(f"jobs.json not found, generating at {args.jobs_json}...", flush=True)
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "prepare_sac_training.py"), "--json", str(args.jobs_json)],
            check=True,
            cwd=str(REPO_ROOT),
        )

    with args.jobs_json.open("r", encoding="utf-8") as fh:
        jobs = json.load(fh)

    print(f"Loaded {len(jobs)} jobs from {args.jobs_json}.", flush=True)
    print(f"Running queue with {args.workers} workers...", flush=True)

    start_all = time.time()
    results = []
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())
    except KeyboardInterrupt:
        print("\nQueue interrupted! Terminating active jobs...", flush=True)
        sys.exit(1)

    failures = [res for res in results if res[1] != 0]
    total_time = time.time() - start_all
    
    print("\n" + "=" * 60, flush=True)
    print(f"Queue complete in {total_time/3600:.2f} hours.", flush=True)
    print(f"Successful: {len(results) - len(failures)}/{len(jobs)}", flush=True)
    if failures:
        print(f"Failed: {len(failures)}/{len(jobs)}", flush=True)
        for job, rc, dur in failures:
            print(f"  - {job['scenario']} ({job['forecast_mode']}) seed {job['seed']} (rc={rc})", flush=True)
        sys.exit(1)
    else:
        print("All jobs completed successfully!", flush=True)


if __name__ == "__main__":
    main()
