"""Runner tests with stubbed training/evaluation — no jobs launched."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from microgrid_simulator.config import load_settings
from microgrid_simulator.rl.registry import ProposalChange, ProposalV1
from microgrid_simulator.rl.runners import (
    SACTrainRunners,
    SeedEvaluation,
    apply_proposal_to_settings,
)

CONFIG = Path("configs/islanded-baseline-72h.yaml")


def _proposal(**overrides: Any) -> ProposalV1:
    data: dict[str, Any] = {
        "schema_version": "agent-retrain-proposal-v1",
        "iteration": 1,
        "experiment_type": "scratch",
        "parent_policy": "",
        "change": ProposalChange(parameter="rl.sac.gradient_steps", new_value=4),
        "hypothesis": "More updates per transition improve critic learning.",
    }
    data.update(overrides)
    return ProposalV1(**data)


# ---------------------------------------------------------------------------
# apply_proposal_to_settings
# ---------------------------------------------------------------------------


def test_apply_dotted_path_rl_sac() -> None:
    settings = apply_proposal_to_settings(load_settings(CONFIG), _proposal())
    assert settings.rl.sac.gradient_steps == 4
    # Base object untouched (deep copy).
    assert load_settings(CONFIG).rl.sac.gradient_steps in (None, 1)


def test_apply_reward_path() -> None:
    p = _proposal(
        change=ProposalChange(parameter="reward.w_unserved", old_value=100.0, new_value=250.0)
    )
    settings = apply_proposal_to_settings(load_settings(CONFIG), p)
    assert settings.reward.w_unserved == 250.0


def test_fine_tune_default_learning_rate() -> None:
    base = load_settings(CONFIG)
    assert base.rl.sac.learning_rate is None  # SB3 default would be used
    p = _proposal(experiment_type="fine_tune", parent_policy="e5_raw_f3")
    settings = apply_proposal_to_settings(base, p)
    assert settings.rl.sac.learning_rate == pytest.approx(1e-4)


def test_fine_tune_keeps_proposed_learning_rate() -> None:
    p = _proposal(
        experiment_type="fine_tune",
        parent_policy="e5_raw_f3",
        change=ProposalChange(parameter="rl.sac.learning_rate", new_value=0.0003),
    )
    settings = apply_proposal_to_settings(load_settings(CONFIG), p)
    assert settings.rl.sac.learning_rate == pytest.approx(3e-4)


def test_scratch_does_not_force_learning_rate() -> None:
    settings = apply_proposal_to_settings(load_settings(CONFIG), _proposal())
    assert settings.rl.sac.learning_rate is None


# ---------------------------------------------------------------------------
# SACTrainRunners with stubbed train/eval
# ---------------------------------------------------------------------------


class StubTrainFn:
    """Writes fake artifacts instead of training."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, settings: Any, **kwargs: Any) -> Path:
        self.calls.append({"settings": settings, **kwargs})
        artifact_dir: Path = kwargs["artifact_dir"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        zip_path = artifact_dir / "sac_microgrid.zip"
        zip_path.write_bytes(b"stub-zip")
        # train() writes vecnormalize next to the model; mimic that.
        vn = artifact_dir / "sac_microgrid_vecnormalize.pkl"
        vn.write_bytes(b"stub-vn")
        return zip_path


def _seed_eval(seed: int, mean_unserved: float = 12.0) -> SeedEvaluation:
    return SeedEvaluation(
        seed=seed,
        episode_unserved_kwh=[mean_unserved] * 3,
        episode_rewards=[-40_000.0] * 3,
        worst_episode_unserved_kwh=mean_unserved + 5.0,
        avoidable_kwh_per_episode=0.0,
    )


@pytest.fixture()
def runners(tmp_path: Path) -> tuple[SACTrainRunners, StubTrainFn]:
    train_fn = StubTrainFn()
    runners = SACTrainRunners(
        base_config_path=CONFIG,
        incumbent_root=tmp_path / "incumbent",
        eval_episodes=3,
        train_fn=train_fn,
        eval_fn=lambda settings, zip_path, episodes, eval_seed: _seed_eval(eval_seed % 10),
    )
    return runners, train_fn


def test_smoke_runner_produces_pass_result(tmp_path: Path, runners) -> None:
    r, train_fn = runners
    result = r.smoke(_proposal(), tmp_path / "it001")
    assert train_fn.calls and train_fn.calls[0]["init_artifact"] is None
    assert result.model_artifact_present and result.vecnormalize_artifact_present
    assert result.nan_or_inf_count == 0
    assert result.observation_action_contract_valid
    assert result.training_throughput_fps > 0
    assert result.steps_completed == 50_000


def test_full_runner_aggregates_seeds(tmp_path: Path, runners) -> None:
    r, train_fn = runners
    candidate, meta = r.full(_proposal(seeds=[0, 1, 2]), tmp_path / "it001")
    assert len(train_fn.calls) == 3
    assert {c["seed"] for c in train_fn.calls} == {0, 1, 2}
    assert candidate.seeds_evaluated == 3
    assert candidate.artifact_validation_pass
    assert candidate.mean_unserved_kwh == pytest.approx(12.0)
    assert candidate.worst_episode_unserved_kwh == pytest.approx(17.0)
    assert candidate.mean_reward == pytest.approx(-40_000.0)
    assert meta["catastrophic_threshold_kwh"] == pytest.approx(5.0 * 15.7)


def test_full_runner_fine_tune_loads_incumbent(tmp_path: Path, runners) -> None:
    r, train_fn = runners
    # Create incumbent artifacts for seed 2 only.
    inc = tmp_path / "incumbent" / "seed-2"
    inc.mkdir(parents=True)
    (inc / "sac_microgrid.zip").write_bytes(b"parent")
    (inc / "sac_microgrid_vecnormalize.pkl").write_bytes(b"parent-vn")

    p = _proposal(experiment_type="fine_tune", parent_policy="e5_raw_f3", seeds=[2])
    r.full(p, tmp_path / "it001")
    call = train_fn.calls[0]
    assert call["init_artifact"] == inc / "sac_microgrid.zip"
    assert call["init_vecnorm"] == inc / "sac_microgrid_vecnormalize.pkl"
    assert call["init_replay_buffer"] is None  # fresh buffer, §7


def test_full_runner_fine_tune_missing_parent_fails(tmp_path: Path, runners) -> None:
    r, _ = runners
    p = _proposal(experiment_type="fine_tune", parent_policy="e5_raw_f3", seeds=[1])
    with pytest.raises(FileNotFoundError):
        r.full(p, tmp_path / "it001")


def test_catastrophic_threshold_uses_incumbent_mean(tmp_path: Path) -> None:
    r = SACTrainRunners(
        base_config_path=CONFIG,
        incumbent_root=tmp_path,
        incumbent_mean_unserved_kwh=20.0,
        train_fn=StubTrainFn(),
        eval_fn=lambda *a, **k: _seed_eval(0),
    )
    assert r.catastrophic_threshold() == pytest.approx(100.0)


def test_meta_records_per_seed_and_threshold(tmp_path: Path, runners) -> None:
    r, _ = runners
    _, meta = r.full(_proposal(seeds=[0]), tmp_path / "it001")
    assert meta["per_seed"][0]["seed"] == 0
    out = json.dumps(meta)  # serializable for lineage records
    assert "catastrophic_threshold_kwh" in out