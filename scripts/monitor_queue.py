#!/usr/bin/env python3
"""Monitor the training queue status, detect errors, and tune concurrency."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_LOG = Path("/tmp/f3-sac-training-queue-2.log")
DEFAULT_JOBS_JSON = REPO_ROOT / "f3_jobs.json"
TMUX_SESSION = "f3-sac-training-2"


def get_completed_jobs(jobs: list[dict]) -> set[int]:
    """Return indices of jobs that have finished training (model artifact exists)."""
    completed = set()
    for idx, job in enumerate(jobs):
        model_path = REPO_ROOT / job["model"]
        if model_path.exists() and model_path.stat().st_size > 0:
            completed.add(idx)
    return completed


def check_tmux_status() -> bool:
    """Check if the tmux training session is alive."""
    try:
        res = subprocess.run(
            ["tmux", "has-session", "-t", TMUX_SESSION],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def get_gpu_status() -> dict:
    """Retrieve GPU utilization and memory stats using nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [x.strip() for x in res.stdout.strip().split(",")]
            return {
                "utilization": f"{parts[0]}%",
                "memory_used": f"{parts[1]} MB",
                "memory_total": f"{parts[2]} MB",
                "temperature": f"{parts[3]}°C",
            }
    except Exception:
        pass
    return {"utilization": "N/A", "memory_used": "N/A", "memory_total": "N/A", "temperature": "N/A"}


def scan_log_for_errors(log_path: Path) -> list[str]:
    """Scan a log file for critical error keywords or tracebacks."""
    errors = []
    if not log_path.exists():
        return errors

    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
        # Common failure signatures
        patterns = [
            r"Traceback \(most recent call last\):",
            r"CUDA out of memory",
            r"Segmentation fault",
            r"Error:",
            r"FAILED \(rc=\d+\)",
        ]
        for p in patterns:
            matches = re.findall(p, content, re.IGNORECASE)
            if matches:
                # Get the surrounding lines or exact error line
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if any(re.search(p, line, re.IGNORECASE) for p in patterns):
                        start = max(0, i - 1)
                        end = min(len(lines), i + 5)
                        snippet = "\n".join(lines[start:end])
                        errors.append(f"Match '{p}' around line {i+1}:\n{snippet}")
                        break  # Report one snippet per pattern to avoid log bloat
    except Exception as e:
        errors.append(f"Failed to read log {log_path.name}: {e}")
    return errors


def analyze_active_job(job: dict) -> dict:
    """Parse log file of an active job to compute current steps, speed, and ETA."""
    scenario = job["scenario"]
    mode = job["forecast_mode"]
    seed = job["seed"]
    
    log_file = REPO_ROOT / "logs" / "training" / f"{scenario.lower()}_{mode}_seed{seed}.log"
    info = {
        "status": "UNKNOWN",
        "timesteps": 0,
        "fps": 0,
        "n_updates": 0,
        "errors": [],
    }

    if not log_file.exists():
        info["status"] = "STARTING/LOG_PENDING"
        return info

    info["errors"] = scan_log_for_errors(log_file)
    if info["errors"]:
        info["status"] = "CRASHED"
        return info

    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
        # Extract total timesteps
        ts_matches = re.findall(r"total_timesteps\s*\|\s*(\d+)", text)
        fps_matches = re.findall(r"fps\s*\|\s*(\d+)", text)
        updates_matches = re.findall(r"n_updates\s*\|\s*(\d+)", text)
        
        if ts_matches:
            info["timesteps"] = int(ts_matches[-1])
            info["status"] = "TRAINING"
        if fps_matches:
            info["fps"] = int(fps_matches[-1])
        if updates_matches:
            info["n_updates"] = int(updates_matches[-1])
    except Exception:
        pass

    return info


def monitor_queue(jobs_path: Path, queue_log: Path) -> dict:
    """Analyze the entire queue and return a structured report."""
    if not jobs_path.is_file():
        return {"error": f"Jobs JSON file not found at {jobs_path}"}

    with jobs_path.open("r", encoding="utf-8") as fh:
        jobs = json.load(fh)

    completed_indices = get_completed_jobs(jobs)
    tmux_alive = check_tmux_status()
    gpu = get_gpu_status()

    # Read queue log to find active/running and finished
    active_in_queue = set()
    failed_in_queue = {}
    
    if queue_log.exists():
        try:
            log_text = queue_log.read_text(encoding="utf-8", errors="ignore")
            # Parse starting entries
            # [07:30:17] STARTING: E0 (cached) seed 0
            starts = re.findall(r"STARTING:\s+(\w+)\s+\(([^)]+)\)\s+seed\s+(\d+)", log_text)
            # Parse finishing entries
            # [07:30:17] FINISHED: E0 (cached) seed 0 -> SUCCESS/FAILED (rc=127) in 12.3s
            finishes = re.findall(
                r"FINISHED:\s+(\w+)\s+\(([^)]+)\)\s+seed\s+(\d+)\s+->\s+([^\s]+)",
                log_text
            )

            started_keys = {(s[0], s[1], int(s[2])) for s in starts}
            finished_keys = {}
            for f in finishes:
                finished_keys[(f[0], f[1], int(f[2]))] = f[3]

            for idx, job in enumerate(jobs):
                if idx in completed_indices:
                    continue
                
                key = (job["scenario"], job["forecast_mode"], job["seed"])
                if key in started_keys:
                    if key in finished_keys:
                        status = finished_keys[key]
                        if "FAIL" in status:
                            failed_in_queue[idx] = status
                    else:
                        active_in_queue.add(idx)
        except Exception as e:
            pass

    # Build job details
    job_reports = []
    total_timesteps_done = 0
    total_target_timesteps = len(jobs) * 1_000_000

    for idx, job in enumerate(jobs):
        is_completed = idx in completed_indices
        is_active = idx in active_in_queue
        is_failed = idx in failed_in_queue
        
        status = "PENDING"
        if is_completed:
            status = "COMPLETED"
            progress_steps = 1_000_000
            fps = 0
            n_updates = 250000
            errors = []
        elif is_active:
            # Parse active log
            active_info = analyze_active_job(job)
            status = active_info["status"]
            progress_steps = active_info["timesteps"]
            fps = active_info["fps"]
            n_updates = active_info["n_updates"]
            errors = active_info["errors"]
        elif is_failed:
            status = f"FAILED ({failed_in_queue[idx]})"
            progress_steps = 0
            fps = 0
            n_updates = 0
            # Scan log for failure reason
            log_file = REPO_ROOT / "logs" / "training" / f"{job['scenario'].lower()}_{job['forecast_mode']}_seed{job['seed']}.log"
            errors = scan_log_for_errors(log_file)
        else:
            progress_steps = 0
            fps = 0
            n_updates = 0
            errors = []

        total_timesteps_done += progress_steps
        
        job_reports.append({
            "index": idx,
            "scenario": job["scenario"],
            "forecast_mode": job["forecast_mode"],
            "seed": job["seed"],
            "status": status,
            "progress_steps": progress_steps,
            "progress_pct": round((progress_steps / 1_000_000) * 100, 2),
            "fps": fps,
            "n_updates": n_updates,
            "errors": errors,
            "artifact_dir": job["artifact_dir"],
        })

    # Summary calculations
    running_jobs = [r for r in job_reports if r["status"] in ("TRAINING", "STARTING/LOG_PENDING")]
    completed_jobs = [r for r in job_reports if r["status"] == "COMPLETED"]
    failed_jobs = [r for r in job_reports if "FAIL" in r["status"] or r["status"] == "CRASHED"]
    pending_jobs = [r for r in job_reports if r["status"] == "PENDING"]

    total_progress_pct = round((total_timesteps_done / total_target_timesteps) * 100, 2)

    return {
        "tmux_session_alive": tmux_alive,
        "tmux_session_name": TMUX_SESSION,
        "gpu_status": gpu,
        "counts": {
            "total": len(jobs),
            "completed": len(completed_jobs),
            "running": len(running_jobs),
            "failed": len(failed_jobs),
            "pending": len(pending_jobs),
        },
        "total_progress": {
            "steps_done": total_timesteps_done,
            "steps_total": total_target_timesteps,
            "pct": total_progress_pct,
        },
        "running_details": running_jobs,
        "failed_details": failed_jobs,
    }


def tune_concurrency(jobs_path: Path, queue_log: Path, workers: int) -> None:
    """Tune concurrency: stop the running queue, prune completed jobs, and restart."""
    print("=== Concurrency Tuning & Queue Restart ===")
    
    if not jobs_path.is_file():
        print(f"Error: {jobs_path} not found.")
        sys.exit(1)

    with jobs_path.open("r", encoding="utf-8") as fh:
        jobs = json.load(fh)

    completed_indices = get_completed_jobs(jobs)
    pruned_jobs = [job for idx, job in enumerate(jobs) if idx not in completed_indices]

    if not pruned_jobs:
        print("All jobs are already completed! Nothing to run.")
        return

    print(f"Original jobs: {len(jobs)}")
    print(f"Completed jobs: {len(completed_indices)}")
    print(f"Remaining jobs to run: {len(pruned_jobs)}")

    # Write the pruned jobs to a temporary json file
    pruned_jobs_path = REPO_ROOT / "f3_jobs_pruned.json"
    with pruned_jobs_path.open("w", encoding="utf-8") as fh:
        json.dump(pruned_jobs, fh, indent=2)
    print(f"Saved pruned jobs list to {pruned_jobs_path.relative_to(REPO_ROOT)}")

    # Stop current tmux session if running
    if check_tmux_status():
        print(f"Terminating active tmux session: '{TMUX_SESSION}'...")
        subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], check=False)
        # Give processes a moment to release ports/resources
        import time
        time.sleep(2)

    # Launch new tmux session running the queue script on the pruned jobs file
    # We must run it using the absolute path to python inside the environment or uv run
    uv_bin = Path(os.environ.get("HOME", "/home/ma012")) / ".local/bin/uv"
    if not uv_bin.exists():
        uv_bin = Path("/usr/local/bin/uv")
    if not uv_bin.exists():
        uv_bin = Path("uv")

    queue_script = REPO_ROOT / "scripts" / "run_sac_training_queue.py"
    
    tmux_cmd = (
        f"tmux new-session -d -s {TMUX_SESSION} "
        f"'{uv_bin} run {queue_script} --jobs-json {pruned_jobs_path} --workers {workers} > {queue_log} 2>&1'"
    )
    
    print(f"Executing: {tmux_cmd}")
    res = subprocess.run(tmux_cmd, shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Successfully started new tmux session '{TMUX_SESSION}' with {workers} workers.")
        print(f"Log file redirected to: {queue_log}")
    else:
        print(f"Failed to start tmux session: {res.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JOBS_JSON, help="Path to f3_jobs.json")
    parser.add_argument("--log", type=Path, default=DEFAULT_QUEUE_LOG, help="Path to queue log")
    parser.add_argument("--tune-workers", type=int, help="Stop current queue and restart with this many workers")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    # Resolve paths absolute
    jobs_path = args.json.resolve()
    queue_log = args.log.resolve()

    if args.tune_workers is not None:
        tune_concurrency(jobs_path, queue_log, args.tune_workers)
        return

    report = monitor_queue(jobs_path, queue_log)

    if args.format == "json":
        print(json.dumps(report, indent=2))
        return

    if "error" in report:
        print(f"Error: {report['error']}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"QUEUE MONITOR: tmux session '{report['tmux_session_name']}' is {'ALIVE' if report['tmux_session_alive'] else 'DEAD'}")
    print(f"GPU status: Util {report['gpu_status']['utilization']} | Temp {report['gpu_status']['temperature']} | Mem {report['gpu_status']['memory_used']}/{report['gpu_status']['memory_total']}")
    print("-" * 60)
    counts = report["counts"]
    print(f"Jobs Summary: Total {counts['total']} | Completed {counts['completed']} | Running {counts['running']} | Failed {counts['failed']} | Pending {counts['pending']}")
    prog = report["total_progress"]
    print(f"Overall Timesteps: {prog['steps_done']:,} / {prog['steps_total']:,} steps ({prog['pct']}%)")
    print("=" * 60)

    if report["running_details"]:
        print("\nActive Training Jobs:")
        for r in report["running_details"]:
            print(f"  - [{r['scenario']} ({r['forecast_mode']})] Seed {r['seed']}: {r['progress_steps']:,}/1,000,000 steps ({r['progress_pct']}%) @ {r['fps']} FPS | Updates: {r['n_updates']:,} | Dir: {r['artifact_dir']}")

    if report["failed_details"]:
        print("\n❌ Failed/Crashed Jobs:")
        for r in report["failed_details"]:
            print(f"  - [{r['scenario']} ({r['forecast_mode']})] Seed {r['seed']} Status: {r['status']}")
            for err in r["errors"]:
                print(f"    Error Snippet:\n    {err}")
    
    # Check if dead but jobs pending/failed
    if not report["tmux_session_alive"] and (counts["pending"] > 0 or counts["failed"] > 0):
        print("\n⚠️ WARNING: tmux queue runner session is DEAD, but unfinished/failed jobs remain.")
        print("Use '--tune-workers N' to restart or run queue manager directly.")


if __name__ == "__main__":
    main()
