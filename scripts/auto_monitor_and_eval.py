#!/usr/bin/env python3
"""Autonomous monitor for SAC forecast-summary pilot training and matrix evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
def get_queue_status(jobs_file: Path, queue_log: Path) -> dict | None:
    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "monitor_queue.py"),
            "--json",
            str(jobs_file),
            "--log",
            str(queue_log),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--queue-log", type=Path, required=True)
    parser.add_argument("--matrix-log", type=Path, required=True)
    parser.add_argument(
        "--matrix-flag",
        choices=["summary-pilot", "summary-remaining"],
        required=True,
    )
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--expected-artifacts", type=int, default=9)
    args = parser.parse_args()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Autonomous monitor started on ma012...",
        flush=True,
    )

    while True:
        status = get_queue_status(args.jobs, args.queue_log)
        if not status:
            print(
                f"[{time.strftime('%H:%M:%S')}] Warn: No queue status. Retrying...",
                flush=True,
            )
            time.sleep(60)
            continue

        counts = status.get("counts", {})
        progress = status.get("total_progress", {})
        gpu = status.get("gpu_status", {})
        running = counts.get("running", 0)
        pending = counts.get("pending", 0)
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)

        steps_done = progress.get("steps_done", 0)
        steps_total = progress.get("steps_total", 9_000_000)
        pct = progress.get("pct", 0.0)

        print("\n" + "=" * 60, flush=True)
        print(
            f"[{time.strftime('%H:%M:%S')}] Total: {steps_done:,}/{steps_total:,} ({pct:.2f}%)",
            flush=True,
        )
        print(
            f"Jobs: {running} running, {completed} completed, {failed} failed, {pending} pending",
            flush=True,
        )
        u = gpu.get("utilization", "N/A")
        t = gpu.get("temperature", "N/A")
        m_u = gpu.get("memory_used", "N/A")
        m_t = gpu.get("memory_total", "N/A")
        print(f"GPU: Util {u} | Temp {t} | Mem {m_u}/{m_t}", flush=True)
        print("-" * 60, flush=True)

        for job in status.get("running_details", []):
            sc = job.get("scenario")
            sd = job.get("seed")
            st = job.get("progress_steps", 0)
            fps = job.get("fps", 0)
            pct_j = job.get("progress_pct", 0.0)
            print(
                f"  - [{sc}] Seed {sd}: {st:,}/1M ({pct_j:.1f}%) @ {fps} FPS",
                flush=True,
            )

        if failed > 0:
            print("\n[ERROR] Training failures detected!", flush=True)
            for err in status.get("failed_details", []):
                print(f"  Failed job: {err}", flush=True)
            sys.exit(1)

        if running == 0 and pending == 0:
            print("\nAll training jobs have finished!", flush=True)
            break

        time.sleep(300)

    # Verify artifacts
    artifact_root = REPO_ROOT / "artifacts" / "sac" / "f3-summary"
    artifacts = [
        path
        for scenario in args.scenarios
        for path in (artifact_root / scenario).glob("**/sac_microgrid.zip")
    ]
    print(
        f"\n[{time.strftime('%H:%M:%S')}] Found {len(artifacts)}/{args.expected_artifacts} "
        "model artifacts",
        flush=True,
    )
    if len(artifacts) != args.expected_artifacts:
        print(
            f"[ERROR] Expected {args.expected_artifacts} artifacts, found {len(artifacts)}. "
            "Aborting.",
            flush=True,
        )
        sys.exit(1)

    print(
        f"\n[{time.strftime('%H:%M:%S')}] Launching F3 Summary Pilot Matrix Eval (81 jobs)...",
        flush=True,
    )
    with args.matrix_log.open("w") as fh:
        p = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_f3_comparison_matrix.py"),
                f"--{args.matrix_flag}",
                "--workers",
                "9",
            ],
            cwd=str(REPO_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
        )

    if p.returncode != 0:
        print(
            f"[ERROR] Matrix evaluation failed (rc={p.returncode})! "
            f"Check {args.matrix_log}",
            flush=True,
        )
        sys.exit(1)

    print(
        f"\n[{time.strftime('%H:%M:%S')}] Evaluation completed! Saved to {args.matrix_log}",
        flush=True,
    )


if __name__ == "__main__":
    main()
