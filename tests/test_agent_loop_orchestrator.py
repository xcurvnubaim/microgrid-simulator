"""WP4 orchestrator tests: gates, fallback provider, injectable runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from microgrid_simulator.rl.orchestrator import (
    AgentLoopOrchestrator,
    CandidateMetrics,
    DeterministicFallbackProvider,
    PromotionGate,
    SmokeRunResult,
    evaluate_promotion_gate,
    evaluate_smoke_gate,
)
from microgrid_simulator.rl.registry import (
    BudgetState,
    IncumbentRecord,
    IterationRegistry,
    LoopLimits,
)

# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------


def _good_smoke() -> SmokeRunResult:
    return SmokeRunResult(
        model_artifact_present=True,
        vecnormalize_artifact_present=True,
        training_throughput_fps=95.0,
        steps_completed=50_000,
    )


def test_smoke_gate_pass() -> None:
    passed, reasons = evaluate_smoke_gate(_good_smoke())
    assert passed and reasons == []


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("config_schema_valid", False),
        ("observation_action_contract_valid", False),
        ("train_validation_leakage_detected", True),
        ("model_artifact_present", False),
        ("vecnormalize_artifact_present", False),
        ("evaluation_dry_run_job_count_matches", False),
    ],
)
def test_smoke_gate_boolean_failures(field_name: str, value: bool) -> None:
    result = _good_smoke()
    setattr(result, field_name, value)
    passed, reasons = evaluate_smoke_gate(result)
    assert not passed and reasons


def test_smoke_gate_numeric_failures() -> None:
    result = _good_smoke()
    result.nan_or_inf_count = 2
    result.unexpected_solver_failure_count = 1
    result.training_throughput_fps = 5.0
    passed, reasons = evaluate_smoke_gate(result)
    assert not passed
    joined = "; ".join(reasons)
    assert "nan_or_inf" in joined and "solver_failures" in joined and "throughput" in joined


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def _incumbent() -> IncumbentRecord:
    return IncumbentRecord(
        policy_id="e5_raw_f3",
        artifact_dir="artifacts/sac/f3/E5/seed-0",
        mean_unserved_kwh=15.7,
        worst_episode_unserved_kwh=40.0,
        mean_reward=-42_535.0,
        promoted_at_iteration=0,
    )


def _candidate(**overrides: Any) -> CandidateMetrics:
    data: dict[str, Any] = {
        "artifact_validation_pass": True,
        "mean_unserved_kwh": 10.0,  # ~36% better than incumbent
        "worst_episode_unserved_kwh": 35.0,  # no regression
        "mean_reward": -43_000.0,  # ~1.1% worse, within 5%
        "seeds_evaluated": 3,
    }
    data.update(overrides)
    return CandidateMetrics(**data)


def test_promotion_gate_pass() -> None:
    passed, reasons = evaluate_promotion_gate(_candidate(), _incumbent(), PromotionGate())
    assert passed and reasons == []


def test_promotion_gate_insufficient_improvement() -> None:
    c = _candidate(mean_unserved_kwh=14.5)  # only ~7.6% better
    passed, reasons = evaluate_promotion_gate(c, _incumbent(), PromotionGate())
    assert not passed and any("mean_unserved_improvement" in r for r in reasons)


def test_promotion_gate_worst_episode_regression() -> None:
    c = _candidate(worst_episode_unserved_kwh=45.0)
    passed, reasons = evaluate_promotion_gate(c, _incumbent(), PromotionGate())
    assert not passed and any("worst_episode_regression" in r for r in reasons)


def test_promotion_gate_reward_regression() -> None:
    c = _candidate(mean_reward=-46_000.0)  # ~8% worse
    passed, reasons = evaluate_promotion_gate(c, _incumbent(), PromotionGate())
    assert not passed and any("reward_regression" in r for r in reasons)


def test_promotion_gate_avoidable_unserved_blocks() -> None:
    c = _candidate(avoidable_unserved_kwh=3.2)
    passed, reasons = evaluate_promotion_gate(c, _incumbent(), PromotionGate())
    assert not passed and any("avoidable_unserved" in r for r in reasons)


def test_promotion_gate_indeterminate_blocks() -> None:
    c = _candidate(indeterminate_outage_count=1)
    passed, reasons = evaluate_promotion_gate(c, _incumbent(), PromotionGate())
    assert not passed and any("indeterminate_outages" in r for r in reasons)


def test_promotion_gate_ignores_terminal_soc_diagnostic() -> None:
    candidate = _candidate(terminal_soc_violation_count=27)
    passed, reasons = evaluate_promotion_gate(candidate, _incumbent(), PromotionGate())
    assert passed and reasons == []


def test_promotion_gate_missing_seeds_block() -> None:
    c = _candidate(seeds_evaluated=2)
    passed, reasons = evaluate_promotion_gate(c, _incumbent(), PromotionGate())
    assert not passed and any("seeds=" in r for r in reasons)


def test_promotion_gate_no_incumbent_skips_relative_checks() -> None:
    passed, reasons = evaluate_promotion_gate(_candidate(), None, PromotionGate())
    assert passed and reasons == []


# ---------------------------------------------------------------------------
# Fallback provider
# ---------------------------------------------------------------------------


def test_fallback_provider_order_and_dedup() -> None:
    provider = DeterministicFallbackProvider()
    p1 = provider.propose(set(), BudgetState(), LoopLimits(), iteration=1)
    assert p1.change.parameter == "rl.sac.gradient_steps"
    assert p1.change.new_value == 4
    assert p1.experiment_type == "scratch"

    completed = {("rl.sac.gradient_steps", 4)}
    p2 = provider.propose(completed, BudgetState(), LoopLimits(), iteration=2)
    assert (p2.change.parameter, p2.change.new_value) == ("rl.sac.gamma", 0.995)
    assert p2.experiment_type == "fine_tune"
    assert p2.parent_policy == "e5_raw_f3"


def test_fallback_provider_exhausts() -> None:
    provider = DeterministicFallbackProvider()
    all_done = {
        ("rl.sac.gradient_steps", 4),
        ("rl.sac.gamma", 0.995),
        ("rl.sac.batch_size", 512),
        ("reward.w_unserved", 250.0),
        ("rl.sac.net_arch", (512, 512)),
    }
    with pytest.raises(StopIteration):
        provider.propose(all_done, BudgetState(), LoopLimits(), iteration=9)


# ---------------------------------------------------------------------------
# Orchestrator with stub runners
# ---------------------------------------------------------------------------


class StubRunners:
    def __init__(
        self,
        smoke_result: SmokeRunResult | None = None,
        candidate: CandidateMetrics | None = None,
    ) -> None:
        self.smoke_calls: list[Any] = []
        self.full_calls: list[Any] = []
        self._smoke = smoke_result or _good_smoke()
        self._candidate = candidate or _candidate()

    def smoke(self, proposal: Any, iteration_dir: Any) -> SmokeRunResult:
        self.smoke_calls.append(proposal)
        return self._smoke

    def full(self, proposal: Any, iteration_dir: Any) -> tuple[CandidateMetrics, dict[str, Any]]:
        self.full_calls.append(proposal)
        return self._candidate, {"note": "stub"}


def _registry(tmp_path: Path) -> IterationRegistry:
    return IterationRegistry(tmp_path / "artifacts" / "agent-loop" / "e5")


def _seed_incumbent(registry: IterationRegistry) -> IncumbentRecord:
    """Record the pre-existing E5 raw F3 incumbent, as in real deployment."""
    record = _incumbent()
    registry.promote_incumbent(record)
    return record


def test_orchestrator_no_runner_fails_closed(tmp_path: Path) -> Path:
    orch = AgentLoopOrchestrator(registry=_registry(tmp_path))
    outcome = orch.run_iteration()
    assert outcome.status == "NO_RUNNER"
    assert "scope decision" in outcome.detail
    return tmp_path


def test_orchestrator_promotes_candidate(tmp_path: Path) -> None:
    runners = StubRunners()
    orch = AgentLoopOrchestrator(
        registry=_registry(tmp_path), runners=runners, limits=LoopLimits(max_iterations=1)
    )
    outcome = orch.run_iteration()
    assert outcome.status == "PROMOTED"
    incumbent = orch.registry.load_incumbent()
    assert incumbent is not None
    assert incumbent.mean_unserved_kwh == pytest.approx(10.0)
    assert orch.budget.completed_iterations == 1
    assert len(runners.smoke_calls) == 1 and len(runners.full_calls) == 1
    # Iteration dir + proposal recorded.
    assert (tmp_path / "artifacts/agent-loop/e5/iteration-001/proposal.json").is_file()


def test_orchestrator_retains_on_gate_failure(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    _seed_incumbent(reg)
    weak = _candidate(mean_unserved_kwh=15.0)  # <10% improvement
    runners = StubRunners(candidate=weak)
    orch = AgentLoopOrchestrator(registry=reg, runners=runners)
    outcome = orch.run_iteration()
    assert outcome.status == "RETAINED"
    assert any("mean_unserved_improvement" in r for r in outcome.gate_reasons)
    # Incumbent unchanged.
    assert orch.registry.load_incumbent().mean_unserved_kwh == pytest.approx(15.7)
    assert orch.budget.consecutive_no_improvement == 1


def test_orchestrator_smoke_failure_blocks_full_run(tmp_path: Path) -> None:
    bad_smoke = _good_smoke()
    bad_smoke.model_artifact_present = False
    runners = StubRunners(smoke_result=bad_smoke)
    orch = AgentLoopOrchestrator(registry=_registry(tmp_path), runners=runners)
    outcome = orch.run_iteration()
    assert outcome.status == "SMOKE_FAILED"
    assert any("model_artifact_missing" in r for r in outcome.gate_reasons)
    assert runners.full_calls == []  # full training never launched
    assert orch.budget.failed_iterations == 1


def test_orchestrator_stops_after_failure_budget(tmp_path: Path) -> None:
    bad_smoke = _good_smoke()
    bad_smoke.vecnormalize_artifact_present = False
    runners = StubRunners(smoke_result=bad_smoke)
    orch = AgentLoopOrchestrator(
        registry=_registry(tmp_path),
        runners=runners,
        limits=LoopLimits(max_failed_iterations=1),
    )
    first = orch.run_iteration()
    assert first.status == "SMOKE_FAILED"
    second = orch.run_iteration()
    assert second.status == "STOPPED"
    assert second.detail == "FAILURE_BUDGET_EXHAUSTED"


def test_orchestrator_stops_after_stagnation(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    _seed_incumbent(reg)
    weak = _candidate(mean_unserved_kwh=15.5)
    runners = StubRunners(candidate=weak)
    orch = AgentLoopOrchestrator(
        registry=reg,
        runners=runners,
        limits=LoopLimits(max_consecutive_no_improvement=2),
    )
    assert orch.run_iteration().status == "RETAINED"
    # Second iteration proposes gamma (fallback skips completed gradient_steps).
    second = orch.run_iteration()
    assert second.status == "RETAINED"
    third = orch.run_iteration()
    assert third.status == "STOPPED"
    assert third.detail == "CONSECUTIVE_NO_IMPROVEMENT"


def test_orchestrator_exhausted_provider(tmp_path: Path) -> None:
    class EmptyProvider:
        def propose(self, *a: Any, **k: Any) -> Any:
            raise StopIteration("nothing left")

    orch = AgentLoopOrchestrator(
        registry=_registry(tmp_path), runners=StubRunners(), provider=EmptyProvider()
    )
    outcome = orch.run_iteration()
    assert outcome.status == "EXHAUSTED"
