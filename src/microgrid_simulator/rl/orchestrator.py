"""Agent-loop orchestrator: deterministic control plane (WP4).

Owns one iteration of the gated loop: stop-condition check, proposal
acquisition (provider or frozen fallback), validation, smoke gate, full run,
promotion gate, and incumbent/lineage updates. All physical work is delegated
to injectable runner callables — with no runners wired this module cannot
launch a job, which preserves the scope boundary in the plan (section 2).

Gate semantics follow plan sections 10 and 12; the fallback proposal sequence
follows section 6.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from microgrid_simulator.rl.registry import (
    BudgetState,
    IncumbentRecord,
    IterationRegistry,
    LoopLimits,
    PromotionGate,
    ProposalV1,
    check_stop_conditions,
    validate_proposal,
)

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class SmokeRunResult:
    """Facts produced by an injected smoke runner (plan section 10)."""

    config_schema_valid: bool = True
    observation_action_contract_valid: bool = True
    train_validation_leakage_detected: bool = False
    nan_or_inf_count: int = 0
    unexpected_solver_failure_count: int = 0
    model_artifact_present: bool = False
    vecnormalize_artifact_present: bool = False
    evaluation_dry_run_job_count_matches: bool = True
    training_throughput_fps: float = 0.0
    steps_completed: int = 0


@dataclass
class CandidateMetrics:
    """Aggregated candidate facts required by the promotion gate."""

    artifact_validation_pass: bool = False
    solver_failure_count: int = 0
    indeterminate_outage_count: int = 0
    avoidable_unserved_kwh: float = 0.0
    catastrophic_episode_count: int = 0
    terminal_soc_violation_count: int = 0
    mean_unserved_kwh: float = float("inf")
    worst_episode_unserved_kwh: float = float("inf")
    mean_reward: float = float("-inf")
    seeds_evaluated: int = 0
    artifact_dir: str = ""
    checkpoint_sha256: str | None = None
    vecnormalize_sha256: str | None = None


@dataclass
class IterationOutcome:
    """Result of one orchestrator iteration."""

    iteration: int
    # PROMOTED | RETAINED | SMOKE_FAILED | REJECTED_PROPOSAL | STOPPED | EXHAUSTED
    status: str
    detail: str = ""
    gate_reasons: list[str] = field(default_factory=list)
    promoted_policy_id: str | None = None


# ---------------------------------------------------------------------------
# Gates (deterministic)
# ---------------------------------------------------------------------------

MIN_SMOKE_FPS = 20.0


def evaluate_smoke_gate(result: SmokeRunResult) -> tuple[bool, list[str]]:
    """Apply plan section 10 checks to a smoke run."""
    reasons: list[str] = []
    if not result.config_schema_valid:
        reasons.append("config_schema_invalid")
    if not result.observation_action_contract_valid:
        reasons.append("observation_action_contract_invalid")
    if result.train_validation_leakage_detected:
        reasons.append("train_validation_leakage")
    if result.nan_or_inf_count > 0:
        reasons.append(f"nan_or_inf_count={result.nan_or_inf_count}")
    if result.unexpected_solver_failure_count > 0:
        reasons.append(f"solver_failures={result.unexpected_solver_failure_count}")
    if not result.model_artifact_present:
        reasons.append("model_artifact_missing")
    if not result.vecnormalize_artifact_present:
        reasons.append("vecnormalize_artifact_missing")
    if not result.evaluation_dry_run_job_count_matches:
        reasons.append("evaluation_dry_run_job_count_mismatch")
    if result.training_throughput_fps < MIN_SMOKE_FPS:
        reasons.append(
            f"throughput_below_min ({result.training_throughput_fps:.1f} < {MIN_SMOKE_FPS})"
        )
    return (not reasons), reasons


def evaluate_promotion_gate(
    candidate: CandidateMetrics,
    incumbent: IncumbentRecord | None,
    gate: PromotionGate,
) -> tuple[bool, list[str]]:
    """Lexicographic promotion gate (plan section 12).

    Reliability and physical validity first; economic reward last. With no
    incumbent recorded yet, improvement criteria compare against the gate's
    absolute avoidable-unserved requirement only.
    """
    reasons: list[str] = []

    if candidate.seeds_evaluated < gate.required_seeds:
        reasons.append(f"seeds={candidate.seeds_evaluated}<{gate.required_seeds}")
    if gate.artifact_validation_pass and not candidate.artifact_validation_pass:
        reasons.append("artifact_validation_failed")
    if candidate.solver_failure_count > gate.solver_failure_count_max:
        reasons.append(f"solver_failures={candidate.solver_failure_count}")
    if candidate.indeterminate_outage_count > gate.indeterminate_outage_count_max:
        reasons.append(f"indeterminate_outages={candidate.indeterminate_outage_count}")
    if candidate.avoidable_unserved_kwh > gate.avoidable_unserved_kwh_max:
        reasons.append(f"avoidable_unserved={candidate.avoidable_unserved_kwh:.2f}")
    if candidate.catastrophic_episode_count > gate.catastrophic_episode_count_max:
        reasons.append(f"catastrophic_episodes={candidate.catastrophic_episode_count}")
    # Terminal SOC remains a reported sustainability diagnostic, but is not a
    # promotion blocker in this load-service-focused comparison. The inherited
    # episode-end SOC contract was not part of the user's current objective.

    if incumbent is not None and not reasons:
        improvement = (incumbent.mean_unserved_kwh - candidate.mean_unserved_kwh) / (
            incumbent.mean_unserved_kwh
        ) if incumbent.mean_unserved_kwh > 0 else 0.0
        if improvement < gate.mean_unserved_relative_improvement_min:
            reasons.append(
                f"mean_unserved_improvement={improvement:.3f}"
                f"<{gate.mean_unserved_relative_improvement_min}"
            )
        regression = candidate.worst_episode_unserved_kwh - incumbent.worst_episode_unserved_kwh
        if regression > gate.worst_episode_unserved_regression_max:
            reasons.append(f"worst_episode_regression={regression:.2f}")
        if incumbent.mean_reward != 0:
            reward_regression = (incumbent.mean_reward - candidate.mean_reward) / abs(
                incumbent.mean_reward
            )
            if reward_regression > gate.reward_relative_regression_max:
                reasons.append(f"reward_regression={reward_regression:.3f}")

    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Proposal providers
# ---------------------------------------------------------------------------


class ProposalProvider(Protocol):
    def propose(
        self,
        completed_changes: set[tuple[str, Any]],
        budget: BudgetState,
        limits: LoopLimits,
        iteration: int,
    ) -> ProposalV1: ...


FALLBACK_SEQUENCE: list[tuple[str, str, float | int | list[int], str]] = [
    # (experiment_type, parameter, new_value, hypothesis) — plan section 6 order.
    (
        "scratch",
        "rl.sac.gradient_steps",
        4,
        "Four gradient updates match the four transitions collected per vectorized step.",
    ),
    (
        "fine_tune",
        "rl.sac.gamma",
        0.995,
        "A longer discount horizon credits outages to earlier SOC and diesel decisions.",
    ),
    (
        "scratch",
        "rl.sac.batch_size",
        512,
        "Larger batches reduce critic gradient noise under the stronger update rule.",
    ),
    (
        "fine_tune",
        "reward.w_unserved",
        250.0,
        "A stronger soft penalty is a diagnostic for penalty-sensitivity of outages.",
    ),
    (
        "scratch",
        "rl.sac.net_arch",
        [512, 512],
        "More capacity is tested only after smaller-network learning plateaus.",
    ),
]


class DeterministicFallbackProvider:
    """Frozen proposal sequence used when no LLM provider is wired (section 8)."""

    def __init__(self, parent_policy: str = "e5_raw_f3") -> None:
        self.parent_policy = parent_policy

    def propose(
        self,
        completed_changes: set[tuple[str, Any]],
        budget: BudgetState,
        limits: LoopLimits,
        iteration: int,
    ) -> ProposalV1:
        from microgrid_simulator.rl.registry import ProposalChange

        for experiment_type, parameter, new_value, hypothesis in FALLBACK_SEQUENCE:
            key = (parameter, _normalize(new_value))
            if key in completed_changes:
                continue
            return ProposalV1(
                schema_version="agent-retrain-proposal-v1",
                iteration=iteration,
                experiment_type=experiment_type,  # type: ignore[arg-type]
                parent_policy=self.parent_policy if experiment_type == "fine_tune" else "",
                change=ProposalChange(parameter=parameter, new_value=new_value),
                hypothesis=hypothesis,
            )
        raise StopIteration("fallback sequence exhausted")


def _normalize(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


# ---------------------------------------------------------------------------
# Runners (injectable; nothing runs unless wired)
# ---------------------------------------------------------------------------


class Runners(Protocol):
    def smoke(self, proposal: ProposalV1, iteration_dir: Any) -> SmokeRunResult: ...

    def full(
        self, proposal: ProposalV1, iteration_dir: Any
    ) -> tuple[CandidateMetrics, dict[str, Any]]: ...


class NoRunnerWired:
    """Default runner set: fails closed without launching anything."""

    def smoke(self, proposal: ProposalV1, iteration_dir: Any) -> SmokeRunResult:
        raise NotImplementedError(
            "no smoke runner wired; training jobs require the recorded scope decision"
        )

    def full(
        self, proposal: ProposalV1, iteration_dir: Any
    ) -> tuple[CandidateMetrics, dict[str, Any]]:
        raise NotImplementedError(
            "no full-run runner wired; training jobs require the recorded scope decision"
        )


MAX_CONSECUTIVE_INVALID_PROPOSALS = 3


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AgentLoopOrchestrator:
    """One-iteration control plane. Deterministic; runners are injected."""

    def __init__(
        self,
        registry: IterationRegistry,
        limits: LoopLimits | None = None,
        gate: PromotionGate | None = None,
        provider: ProposalProvider | None = None,
        runners: Runners | None = None,
    ) -> None:
        self.registry = registry
        self.limits = limits or LoopLimits()
        self.gate = gate or PromotionGate()
        self.provider: ProposalProvider = provider or DeterministicFallbackProvider()
        self.runners: Runners = runners or NoRunnerWired()
        self.budget = BudgetState()
        # Resume support: advance completed_iterations to skip already recorded iterations.
        completed = self.registry.load_completed_changes()
        self.budget.completed_iterations = len(completed)
        self._consecutive_invalid = 0

    def check_stop(self) -> str | None:
        reason = check_stop_conditions(self.budget, self.limits)
        if reason is None and self._consecutive_invalid >= MAX_CONSECUTIVE_INVALID_PROPOSALS:
            return "REPEATED_OUT_OF_SCHEMA_PROPOSALS"
        return reason

    def run_iteration(self) -> IterationOutcome:
        started = time.time()

        stop = self.check_stop()
        if stop is not None:
            return IterationOutcome(
                iteration=self.budget.completed_iterations + 1,
                status="STOPPED",
                detail=stop,
            )

        iteration = self.budget.completed_iterations + 1
        incumbent = self.registry.load_incumbent()
        completed = self.registry.load_completed_changes()

        try:
            proposal = self.provider.propose(completed, self.budget, self.limits, iteration)
        except StopIteration as exc:
            return IterationOutcome(iteration=iteration, status="EXHAUSTED", detail=str(exc))

        proposal.iteration = iteration
        reasons = validate_proposal(proposal, self.budget, self.limits, completed)
        if reasons:
            self._consecutive_invalid += 1
            return IterationOutcome(
                iteration=iteration,
                status="REJECTED_PROPOSAL",
                detail="; ".join(reasons),
                gate_reasons=reasons,
            )
        self._consecutive_invalid = 0

        it_dir = self.registry.iteration_dir(iteration, create=True)
        self.registry.write_proposal(iteration, proposal)

        try:
            smoke_result = self.runners.smoke(proposal, it_dir)
        except NotImplementedError as exc:
            return IterationOutcome(iteration=iteration, status="NO_RUNNER", detail=str(exc))

        smoke_passed, smoke_reasons = evaluate_smoke_gate(smoke_result)
        self.budget.total_training_steps += smoke_result.steps_completed
        if not smoke_passed:
            self.budget.failed_iterations += 1
            return IterationOutcome(
                iteration=iteration,
                status="SMOKE_FAILED",
                detail="; ".join(smoke_reasons),
                gate_reasons=smoke_reasons,
            )

        try:
            candidate, meta = self.runners.full(proposal, it_dir)
        except NotImplementedError as exc:
            return IterationOutcome(iteration=iteration, status="NO_RUNNER", detail=str(exc))

        self.budget.total_training_steps += proposal.full_steps * len(proposal.seeds)
        self.budget.wall_time_hours += (time.time() - started) / 3600.0

        passed, gate_reasons = evaluate_promotion_gate(candidate, incumbent, self.gate)
        self.budget.completed_iterations += 1

        if passed:
            record = IncumbentRecord(
                policy_id=f"{proposal.experiment_type}-it{iteration:03d}",
                artifact_dir=candidate.artifact_dir,
                checkpoint_sha256=candidate.checkpoint_sha256,
                vecnormalize_sha256=candidate.vecnormalize_sha256,
                mean_unserved_kwh=candidate.mean_unserved_kwh,
                worst_episode_unserved_kwh=candidate.worst_episode_unserved_kwh,
                mean_reward=candidate.mean_reward,
                promoted_at_iteration=iteration,
                history=[meta] if meta else [],
            )
            self.registry.promote_incumbent(record)
            self.budget.consecutive_no_improvement = 0
            return IterationOutcome(
                iteration=iteration,
                status="PROMOTED",
                promoted_policy_id=record.policy_id,
            )

        self.budget.consecutive_no_improvement += 1
        return IterationOutcome(
            iteration=iteration,
            status="RETAINED",
            detail="candidate did not pass promotion gate",
            gate_reasons=gate_reasons,
        )
