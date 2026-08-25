"""Agent-gated loop modules: SAC config, reliability, report, inventory, registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microgrid_simulator.config import RLCfg, SACCfg
from microgrid_simulator.rl.registry import (
    ALLOWED_CHANGES,
    BudgetState,
    IncumbentRecord,
    IterationRegistry,
    LoopLimits,
    ProposalChange,
    ProposalV1,
    check_stop_conditions,
    validate_proposal,
)
from microgrid_simulator.rl.reliability import (
    TrajectoryStep,
    classify_avoidability,
    compute_episode_metrics,
    extract_outages,
)

# ---------------------------------------------------------------------------
# WP1: SACCfg configuration
# ---------------------------------------------------------------------------


def test_sac_cfg_defaults_are_none() -> None:
    cfg = SACCfg()
    assert cfg.learning_rate is None
    assert cfg.gamma is None
    assert cfg.net_arch is None


def test_rlcfg_accepts_nested_sac_section() -> None:
    rl = RLCfg(sac={"learning_rate": 0.0001, "gradient_steps": 4})
    assert rl.sac.learning_rate == 0.0001
    assert rl.sac.gradient_steps == 4


def test_sac_cfg_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra"):
        SACCfg(not_a_real_field=1)


def test_build_sac_kwargs_filters_none() -> None:
    from microgrid_simulator.config import Settings
    from microgrid_simulator.rl.train import _build_sac_kwargs

    s = Settings()
    assert _build_sac_kwargs(s) == {}

    s.rl.sac.learning_rate = 0.0001
    s.rl.sac.net_arch = [512, 512]
    kwargs = _build_sac_kwargs(s)
    assert kwargs["learning_rate"] == 0.0001
    assert kwargs["policy_kwargs"]["net_arch"] == [512, 512]
    assert "gamma" not in kwargs


# ---------------------------------------------------------------------------
# WP2: reliability diagnostics
# ---------------------------------------------------------------------------


def _step(**overrides: object) -> TrajectoryStep:
    base: dict = {
        "step": 0,
        "hour": 0.0,
        "load_kw": 100.0,
        "pv_available_kw": 0.0,
        "pv_used_kw": 0.0,
        "battery_kw": 0.0,
        "battery_discharge_kw": 0.0,
        "diesel_kw": 200.0,
        "diesel_on": True,
        "diesel_load_serving_kw": 200.0,
        "unserved_kw": 50.0,
        "soc_pct": 50.0,
        "reward": -10.0,
        "penalty_carbon": 0.0,
        "penalty_autonomy": 0.0,
        "penalty_health": 0.0,
        "penalty_waste": 0.0,
        "penalty_excess": 0.0,
        "penalty_unserved": 0.0,
        "penalty_fuel": 0.0,
        "penalty_constraint": 0.0,
        "constraint_violation_count": 0,
    }
    base.update(overrides)
    return TrajectoryStep(**base)


def test_classify_immediately_avoidable() -> None:
    assert classify_avoidability(_step()) == "immediately_avoidable"


def test_classify_energy_planning_failure() -> None:
    # Diesel committed at max (no ramp headroom) and SOC below the floor:
    # capacity exists, but reserve was spent earlier.
    s = _step(load_kw=100.0, diesel_kw=400.0, soc_pct=8.0, unserved_kw=100.0)
    assert classify_avoidability(s) == "energy_planning_failure"


def test_classify_commitment_limited_diesel_off() -> None:
    # Diesel idle: covering the shortfall requires a start, which is subject
    # to startup/min-down constraints — not instant headroom.
    s = _step(diesel_on=False, diesel_kw=0.0, soc_pct=8.0, unserved_kw=100.0)
    assert classify_avoidability(s) == "commitment_limited"


def test_classify_physically_unavoidable() -> None:
    # Load exceeds everything installable (pv 0 + battery 250 + diesel 400).
    s = _step(load_kw=700.0, diesel_kw=400.0, soc_pct=5.0, unserved_kw=100.0)
    assert classify_avoidability(s) == "physically_unavoidable"


def test_extract_outages_contiguous_events() -> None:
    steps = [
        _step(step=i, unserved_kw=50.0 if 3 <= i <= 6 else 0.0) for i in range(10)
    ]
    events = extract_outages(steps)
    assert len(events) == 1
    assert events[0].duration_steps == 4
    assert events[0].peak_unserved_kw == 50.0
    assert events[0].unserved_kwh == pytest.approx(50.0)


def test_compute_episode_metrics_classification() -> None:
    steps = [
        _step(step=i, unserved_kw=50.0 if i < 4 else 0.0) for i in range(8)
    ]
    metrics = compute_episode_metrics(steps)
    assert metrics.total_unserved_kwh == pytest.approx(50.0)
    assert len(metrics.outage_events) == 1
    assert metrics.outage_events[0].classification == "immediately_avoidable"
    assert metrics.min_soc_pct == 50.0


# ---------------------------------------------------------------------------
# Lightweight: report + inventory + monitoring
# ---------------------------------------------------------------------------


def test_report_builder_tables() -> None:
    from microgrid_simulator.rl.report import (
        build_comparison_table,
        build_metrics_table,
        build_outage_table,
        render_iteration_report,
    )

    table = build_metrics_table({"total_reward": -50000.0})
    assert "-50,000.00" in table

    comparison = build_comparison_table(
        [("a", {"x": 1.0}), ("b", {"x": 2.0})], ["x"]
    )
    assert "| a |" in comparison and "| b |" in comparison

    outage = build_outage_table(
        [{"classification": "immediately_avoidable", "unserved_kwh": 10.0}]
    )
    assert "immediately_avoidable" in outage

    report = render_iteration_report(
        iteration=1,
        proposal={
            "change": {"parameter": "p", "old_value": 1, "new_value": 2},
            "hypothesis": "h",
        },
        smoke_results={"passed": True},
        seed_metrics={0: {"reward": -1.0}},
        comparison_table=comparison,
        outage_table=outage,
        promotion={"passed": True},
    )
    assert "Iteration 001" in report and "PROMOTED" in report


def test_inventory_manifest_text() -> None:
    from microgrid_simulator.rl.inventory import ArtifactManifest, manifest_to_text

    text = manifest_to_text(ArtifactManifest(iteration=1))
    assert "Models: 0" in text and "Evaluations: 0" in text


def test_monitoring_snapshot_text() -> None:
    from microgrid_simulator.rl.monitoring import (
        GpuStatus,
        MonitoringSnapshot,
        ProcessStatus,
        snapshot_to_text,
    )

    snap = MonitoringSnapshot(
        status="RUNNING",
        total_jobs=9,
        completed_jobs=4,
        gpu=GpuStatus(utilization_pct=85.0),
        processes=[
            ProcessStatus(index=0, scenario="E5", seed=0, status="TRAINING",
                          progress_steps=500_000, fps=102.0)
        ],
    )
    text = snapshot_to_text(snap)
    assert "RUNNING" in text and "4/9" in text and "102 FPS" in text


# ---------------------------------------------------------------------------
# WP3: registry validation and paths
# ---------------------------------------------------------------------------


def _proposal(**overrides: object) -> ProposalV1:
    data: dict = {
        "schema_version": "agent-retrain-proposal-v1",
        "iteration": 2,
        "experiment_type": "fine_tune",
        "parent_policy": "e5_raw_f3",
        "change": ProposalChange(parameter="rl.sac.gradient_steps", old_value=1, new_value=4),
        "hypothesis": "Four updates match four transitions per vectorized step.",
    }
    data.update(overrides)
    return ProposalV1(**data)


def test_valid_proposal_accepted() -> None:
    assert validate_proposal(_proposal(), BudgetState(), LoopLimits()) == []


def test_non_allowlisted_parameter_rejected() -> None:
    p = _proposal(change=ProposalChange(parameter="rl.sac.ent_coef", new_value=0.01))
    reasons = validate_proposal(p, BudgetState(), LoopLimits())
    assert any("PARAMETER_NOT_ALLOWLISTED" in r for r in reasons)


def test_non_allowlisted_value_rejected() -> None:
    p = _proposal(change=ProposalChange(parameter="rl.sac.gamma", new_value=0.999))
    reasons = validate_proposal(p, BudgetState(), LoopLimits())
    assert any("VALUE_NOT_ALLOWLISTED" in r for r in reasons)


def test_duplicate_experiment_rejected() -> None:
    done = {("rl.sac.gradient_steps", 4)}
    reasons = validate_proposal(_proposal(), BudgetState(), LoopLimits(), completed_changes=done)
    assert any("DUPLICATE_EXPERIMENT" in r for r in reasons)


def test_budget_exceeded_rejected() -> None:
    big = BudgetState(total_training_steps=11_900_000)
    reasons = validate_proposal(_proposal(), big, LoopLimits())
    assert any("BUDGET_EXCEEDED" in r for r in reasons)


def test_net_arch_list_normalization() -> None:
    p = _proposal(
        experiment_type="scratch",
        change=ProposalChange(
            parameter="rl.sac.net_arch", old_value=[256, 256], new_value=[512, 512]
        ),
    )
    assert validate_proposal(p, BudgetState(), LoopLimits()) == []


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (BudgetState(completed_iterations=8), "MAX_ITERATIONS_REACHED"),
        (BudgetState(consecutive_no_improvement=3), "CONSECUTIVE_NO_IMPROVEMENT"),
        (BudgetState(failed_iterations=2), "FAILURE_BUDGET_EXHAUSTED"),
        (BudgetState(wall_time_hours=48.0), "WALL_TIME_BUDGET_EXHAUSTED"),
        (BudgetState(), None),
    ],
)
def test_stop_conditions(budget: BudgetState, expected: str | None) -> None:
    assert check_stop_conditions(budget, LoopLimits()) == expected


def test_registry_no_overwrite_and_duplicates(tmp_path: Path) -> None:
    reg = IterationRegistry(tmp_path / "artifacts" / "agent-loop" / "e5")
    reg.iteration_dir(1, create=True)
    with pytest.raises(FileExistsError):
        reg.iteration_dir(1, create=True)

    reg.write_proposal(1, _proposal())
    with pytest.raises(FileExistsError):
        reg.write_proposal(1, _proposal())

    assert ("rl.sac.gradient_steps", 4) in reg.load_completed_changes()


def test_incumbent_atomic_promotion(tmp_path: Path) -> None:
    reg = IterationRegistry(tmp_path / "inc")
    rec = IncumbentRecord(
        policy_id="e5_raw_f3",
        artifact_dir="artifacts/sac/f3/E5/seed-0",
        mean_unserved_kwh=15.7,
        worst_episode_unserved_kwh=40.0,
        mean_reward=-42535.0,
        promoted_at_iteration=0,
    )
    reg.promote_incumbent(rec)
    assert reg.load_incumbent().mean_unserved_kwh == 15.7
    reg.promote_incumbent(rec.model_copy(update={"mean_unserved_kwh": 10.0}))
    assert reg.load_incumbent().mean_unserved_kwh == 10.0
    # No leftover temp files after atomic replace.
    assert not list(reg.root.glob("*.tmp"))


def test_allowlist_matches_plan_search_space() -> None:
    assert set(ALLOWED_CHANGES) == {
        "rl.sac.learning_rate",
        "rl.sac.batch_size",
        "rl.sac.gradient_steps",
        "rl.sac.gamma",
        "rl.sac.learning_starts",
        "rl.sac.net_arch",
        "reward.w_unserved",
    }


def test_proposal_json_round_trip(tmp_path: Path) -> None:
    p = _proposal()
    path = tmp_path / "proposal.json"
    path.write_text(p.model_dump_json(indent=2))
    loaded = ProposalV1(**json.loads(path.read_text()))
    assert loaded == p