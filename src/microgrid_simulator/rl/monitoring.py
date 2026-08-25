"""Deterministic monitoring status collector for the agent-gated loop.

Collects structured status from training processes, GPU state, and artifact
directories. A lightweight LLM may narrate the structured status into concise
prose, but the numerical values, state classifications, and failure detections
are deterministic.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProcessStatus:
    """Status of one training or evaluation process."""

    index: int
    scenario: str
    seed: int
    status: str  # PENDING | TRAINING | COMPLETED | FAILED
    progress_steps: int = 0
    progress_pct: float = 0.0
    fps: float = 0.0
    artifact_dir: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class GpuStatus:
    """GPU state snapshot."""

    utilization_pct: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_c: float = 0.0


@dataclass
class MonitoringSnapshot:
    """Complete structured monitoring snapshot."""

    timestamp: str = ""
    queue_progress_pct: float = 0.0
    total_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    pending_jobs: int = 0
    gpu: GpuStatus = field(default_factory=GpuStatus)
    processes: list[ProcessStatus] = field(default_factory=list)
    artifacts_found: int = 0
    expected_artifacts: int = 0
    status: str = ""  # RUNNING | COMPLETED | FAILED | STALLED


def parse_queue_log(log_path: Path, max_lines: int = 200) -> dict[str, Any]:
    """Parse a training queue log and extract per-job progress.

    Returns a dict of ``{job_index: {status, steps, fps, errors}}``.
    """
    jobs: dict[int, dict[str, Any]] = {}
    if not log_path.is_file():
        return jobs

    lines = log_path.read_text().splitlines()
    for line in lines[-max_lines:]:
        # Match lines like:
        # "  0 | E5 | cached | seed 0 | TRAINING | steps=250000 fps=102"
        # "  0 | E5 | cached | seed 0 | COMPLETED"
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0].strip())
        except ValueError:
            continue
        entry = jobs.setdefault(idx, {"status": "PENDING", "steps": 0, "fps": 0, "errors": []})
        entry["status"] = parts[4].strip() if len(parts) > 4 else "PENDING"
        for part in parts:
            if "steps=" in part:
                with contextlib.suppress(ValueError, IndexError):
                    entry["steps"] = int(part.split("=")[1])
            if "fps=" in part:
                with contextlib.suppress(ValueError, IndexError):
                    entry["fps"] = float(part.split("=")[1])
    return jobs


def snapshot_to_text(snapshot: MonitoringSnapshot) -> str:
    """Render a monitoring snapshot as concise structured text.

    This is the input for a lightweight LLM to narrate. The numbers are
    deterministic; the LLM only rephrases them.
    """
    lines: list[str] = []
    lines.append(f"Status: {snapshot.status}")
    lines.append(
        f"Queue: {snapshot.completed_jobs}/{snapshot.total_jobs} completed, "
        f"{snapshot.running_jobs} running, {snapshot.failed_jobs} failed, "
        f"{snapshot.pending_jobs} pending ({snapshot.queue_progress_pct:.1f}%)"
    )
    lines.append(
        f"GPU: {snapshot.gpu.utilization_pct:.0f}% util | "
        f"{snapshot.gpu.memory_used_mb:.0f}/{snapshot.gpu.memory_total_mb:.0f} MB | "
        f"{snapshot.gpu.temperature_c:.0f}°C"
    )
    lines.append(f"Artifacts: {snapshot.artifacts_found}/{snapshot.expected_artifacts}")
    for p in sorted(snapshot.processes, key=lambda x: x.index):
        lines.append(
            f"  [{p.index}] {p.scenario} seed-{p.seed}: {p.status} "
            f"({p.progress_steps:,} steps, {p.progress_pct:.1f}%"
            + (f", {p.fps:.0f} FPS)" if p.fps else ")")
        )
        for err in p.errors:
            lines.append(f"    ERROR: {err}")
    return "\n".join(lines)