"""Experiment registry for the agent-gated SAC retraining loop.

Deterministic schemas and validation. An LLM may *produce* a proposal, but
every proposal passes through :func:`validate_proposal` here before any job
is launched. This module also owns iteration paths (no-overwrite), budget
accounting, and the incumbent record.

Schemas follow "Agent-Gated Autonomous SAC Retraining Plan - Module 6"
sections 6, 8, 12, 13, and 14.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProposalChange(BaseModel):
    """Exactly one conceptual change per iteration."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    old_value: float | int | str | list[int] | None = None
    new_value: float | int | str | list[int] | None = None


class ProposalV1(BaseModel):
    """Versioned agent proposal (plan section 8)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent-retrain-proposal-v1"]
    iteration: int = Field(ge=1)
    experiment_type: Literal["scratch", "fine_tune"]
    parent_policy: str
    change: ProposalChange
    hypothesis: str = Field(min_length=8)
    smoke_steps: int = Field(default=50_000, ge=1)
    full_steps: int = Field(default=500_000, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1)


class LoopLimits(BaseModel):
    """Frozen loop budgets (plan section 13)."""

    max_iterations: int = 8
    max_total_training_steps: int = 12_000_000
    max_wall_time_hours: float = 48.0
    max_consecutive_no_improvement: int = 3
    max_failed_iterations: int = 2


class PromotionGate(BaseModel):
    """Deterministic lexicographic promotion thresholds (plan section 12)."""

    required_seeds: int = 3
    artifact_validation_pass: bool = True
    solver_failure_count_max: int = 0
    indeterminate_outage_count_max: int = 0
    avoidable_unserved_kwh_max: float = 0.0
    catastrophic_episode_count_max: int = 0
    # Reported for sustainability diagnostics; intentionally not used by the
    # current load-service promotion gate.
    terminal_soc_violation_count_max: int | None = None
    mean_unserved_relative_improvement_min: float = 0.10
    worst_episode_unserved_regression_max: float = 0.0
    reward_relative_regression_max: float = 0.05


class BudgetState(BaseModel):
    """Mutable budget consumption tracked across iterations."""

    completed_iterations: int = 0
    total_training_steps: int = 0
    wall_time_hours: float = 0.0
    consecutive_no_improvement: int = 0
    failed_iterations: int = 0


# Plan section 6 allowlist. Parameter paths match Settings traversal.
ALLOWED_CHANGES: dict[str, set[float | int | str | tuple[int, ...]]] = {
    "rl.sac.learning_rate": {0.0001, 0.0003},
    "rl.sac.batch_size": {256, 512},
    "rl.sac.gradient_steps": {1, 4},
    "rl.sac.gamma": {0.99, 0.995},
    "rl.sac.learning_starts": {10_000},
    "rl.sac.net_arch": {(256, 256), (512, 512)},
    "reward.w_unserved": {100.0, 250.0},
}

FIXED_CONTRACT: dict[str, Any] = {
    "scenario": "E5",
    "forecast_representation": "raw",
    "training_split": "train",
    "selection_split": "validation",
    "seeds": [0, 1, 2],
    "hard_unserved": False,
    "oracle_forecast": False,
}


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def validate_proposal(
    proposal: ProposalV1,
    budget: BudgetState,
    limits: LoopLimits,
    completed_changes: set[tuple[str, Any]] | None = None,
) -> list[str]:
    """Return a list of rejection reasons; empty means accepted.

    Deterministic checks only: schema validity is enforced by pydantic at
    construction; this function enforces the allowlist, one-change rule,
    duplicate detection, and remaining budget.
    """
    reasons: list[str] = []

    param = proposal.change.parameter
    if param not in ALLOWED_CHANGES:
        reasons.append(f"PARAMETER_NOT_ALLOWLISTED: {param}")
    else:
        new_val = _normalize(proposal.change.new_value)
        if new_val not in ALLOWED_CHANGES[param]:
            reasons.append(f"VALUE_NOT_ALLOWLISTED: {param}={new_val!r}")

    if proposal.experiment_type == "fine_tune" and not proposal.parent_policy:
        reasons.append("FINE_TUNE_REQUIRES_PARENT_POLICY")

    if proposal.seeds != FIXED_CONTRACT["seeds"]:
        reasons.append(f"SEEDS_MUST_MATCH_CONTRACT: {FIXED_CONTRACT['seeds']}")

    if completed_changes:
        key = (param, _normalize(proposal.change.new_value))
        if key in completed_changes:
            reasons.append(f"DUPLICATE_EXPERIMENT: {param}={proposal.change.new_value!r}")

    steps_needed = proposal.smoke_steps + proposal.full_steps * len(proposal.seeds)
    if budget.total_training_steps + steps_needed > limits.max_total_training_steps:
        reasons.append(
            f"BUDGET_EXCEEDED: needs {steps_needed} steps, "
            f"remaining {limits.max_total_training_steps - budget.total_training_steps}"
        )
    if budget.completed_iterations + 1 > limits.max_iterations:
        reasons.append("MAX_ITERATIONS_REACHED")
    if budget.failed_iterations >= limits.max_failed_iterations:
        reasons.append("FAILURE_BUDGET_EXHAUSTED")
    if budget.consecutive_no_improvement >= limits.max_consecutive_no_improvement:
        reasons.append("CONSECUTIVE_NO_IMPROVEMENT_STOP")

    return reasons


class IncumbentRecord(BaseModel):
    """Content of ``incumbent.json`` (plan section 14)."""

    policy_id: str
    artifact_dir: str
    checkpoint_sha256: str | None = None
    vecnormalize_sha256: str | None = None
    mean_unserved_kwh: float
    worst_episode_unserved_kwh: float
    mean_reward: float
    promoted_at_iteration: int
    history: list[dict[str, Any]] = Field(default_factory=list)


class IterationRegistry:
    """Owns isolated iteration paths with a strict no-overwrite guarantee."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def iteration_dir(self, iteration: int, create: bool = False) -> Path:
        path = self.root / f"iteration-{iteration:03d}"
        if create:
            if path.exists():
                raise FileExistsError(f"iteration directory already exists: {path}")
            path.mkdir(parents=True)
        return path

    def report_dir(self, iteration: int, create: bool = False) -> Path:
        path = self.root.parent / "reports" / self.root.name / f"iteration-{iteration:03d}"
        if create:
            if path.exists():
                raise FileExistsError(f"report directory already exists: {path}")
            path.mkdir(parents=True)
        return path

    def write_proposal(self, iteration: int, proposal: ProposalV1) -> Path:
        path = self.iteration_dir(iteration, create=False) / "proposal.json"
        if path.exists():
            raise FileExistsError(f"proposal already recorded: {path}")
        path.write_text(proposal.model_dump_json(indent=2) + "\n")
        return path

    def load_completed_changes(self) -> set[tuple[str, Any]]:
        """Scan recorded proposals for (parameter, new_value) pairs."""
        done: set[tuple[str, Any]] = set()
        if not self.root.is_dir():
            return done
        for proposal_file in sorted(self.root.glob("iteration-*/proposal.json")):
            with contextlib_suppress():
                data = json.loads(proposal_file.read_text())
                change = data.get("change", {})
                key = (
                    change.get("parameter", ""),
                    _normalize(change.get("new_value")),
                )
                done.add(key)
        return done

    def load_incumbent(self) -> IncumbentRecord | None:
        path = self.root / "incumbent.json"
        if not path.is_file():
            return None
        return IncumbentRecord(**json.loads(path.read_text()))

    def promote_incumbent(self, record: IncumbentRecord) -> Path:
        """Atomically replace ``incumbent.json`` after gate pass."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "incumbent.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(record.model_dump_json(indent=2) + "\n")
        tmp.replace(path)
        return path


def contextlib_suppress() -> Any:
    """Small indirection so registry imports stay tidy in tests."""
    import contextlib

    return contextlib.suppress(json.JSONDecodeError, OSError)


def check_stop_conditions(budget: BudgetState, limits: LoopLimits) -> str | None:
    """Return the triggered stop reason, or None if the loop may continue."""
    if budget.completed_iterations >= limits.max_iterations:
        return "MAX_ITERATIONS_REACHED"
    if budget.total_training_steps >= limits.max_total_training_steps:
        return "TRAINING_STEP_BUDGET_EXHAUSTED"
    if budget.wall_time_hours >= limits.max_wall_time_hours:
        return "WALL_TIME_BUDGET_EXHAUSTED"
    if budget.consecutive_no_improvement >= limits.max_consecutive_no_improvement:
        return "CONSECUTIVE_NO_IMPROVEMENT"
    if budget.failed_iterations >= limits.max_failed_iterations:
        return "FAILURE_BUDGET_EXHAUSTED"
    return None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
