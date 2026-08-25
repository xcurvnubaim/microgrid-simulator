"""Tests for the continuous one-month frozen-policy evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from microgrid_simulator.config import load_settings
from microgrid_simulator.experiments.monthly_evaluation import (
    EXPECTED_STEPS,
    SAC_POLICIES,
    apply_forecast_mode_override,
    artifact_paths_for,
    build_job_registry,
    extract_outage_events,
    load_masked_rl_components,
    masked_rl_action,
    validate_completed_job,
)
from microgrid_simulator.rl.env import get_episode_progress_index


def test_registry_resolves_exactly_54_jobs(tmp_path: Path) -> None:
    jobs = build_job_registry(output_root=tmp_path)
    assert len(jobs) == 54
    deterministic = [j for j in jobs if j.policy not in SAC_POLICIES]
    sac = [j for j in jobs if j.policy in SAC_POLICIES]
    assert len(deterministic) == 18  # 3 policies x 6 scenarios
    assert len(sac) == 36  # 2 policies x 6 scenarios x 3 seeds
    ids = [job.job_id for job in jobs]
    assert len(ids) == len(set(ids))
    # E1_hard artifacts must never enter the matrix
    assert not any("E1_hard" in str(job.artifact_path) for job in jobs)


def test_registry_normalizes_legacy_mpc_identifier(tmp_path: Path) -> None:
    jobs = build_job_registry(output_root=tmp_path, scenarios=["E0"], policies=["mpc_f3"])
    assert len(jobs) == 1
    assert jobs[0].policy == "pypsa_rh_f3"
    assert "mpc" not in jobs[0].job_id


def test_registry_artifacts_exist() -> None:
    for job in build_job_registry():
        if job.artifact_path is None:
            continue
        assert job.artifact_path.is_file(), f"missing {job.artifact_path}"
        assert job.vecnorm_path is not None and job.vecnorm_path.is_file()


def test_progress_index_matches_observation_layout() -> None:
    settings = load_settings("configs/f3-e0-monthly.yaml")
    index = get_episode_progress_index(settings)
    # 4 buses + 3 loads + 2 pv + soc + soh + 3 scalars + diesel + 2 time feats
    # + soc + initial_soc + target_soc = 20; forecast flag follows at 21.
    assert index == 20


def test_masked_action_zeroes_only_progress_coordinate() -> None:
    class _Rms:
        mean = np.zeros(214, dtype=np.float64)
        var = np.ones(214, dtype=np.float64)

    class _Model:
        def predict(self, obs, deterministic=True):  # noqa: ARG002
            return np.full(3, 7.0, dtype=np.float32), None

    class _Env:
        _last_observation = np.arange(214, dtype=np.float32)

    env = _Env()
    action = masked_rl_action(env, _Model(), _Rms(), progress_index=20)  # type: ignore[arg-type]
    assert np.allclose(action, 7.0)
    # The function must not mutate the env's observation buffer.
    assert env._last_observation[20] == 20.0


def test_sac_none_override_sets_forecast_mode_in_memory_only() -> None:
    settings = load_settings("configs/f3-e0-monthly.yaml")
    assert settings.rl.forecast_mode == "cached"
    overridden = apply_forecast_mode_override(settings, "sac_none_f3")
    assert overridden.rl.forecast_mode == "none"
    fresh = load_settings("configs/f3-e0-monthly.yaml")
    assert fresh.rl.forecast_mode == "cached"


def test_outage_events_collapse_contiguous_intervals() -> None:
    rows = [
        {"timestamp": f"2026-03-01 00:{i * 15:02d}:00", "unserved_kw": kw}
        for i, kw in enumerate([0.0, 5.0, 3.0, 0.0, 0.0, 8.0])
    ]
    events = extract_outage_events(rows, dt_hours=0.25)
    assert len(events) == 2
    first, second = events
    assert first["intervals"] == 2
    assert first["duration_hours"] == pytest.approx(0.5)
    assert first["unserved_kwh"] == pytest.approx((5.0 + 3.0) * 0.25)
    assert first["peak_unserved_kw"] == pytest.approx(5.0)
    assert second["intervals"] == 1
    assert [e["event_id"] for e in events] == [1, 2]


def test_validate_completed_job_requires_both_files_and_steps(tmp_path: Path) -> None:
    assert not validate_completed_job(tmp_path)

    metrics = {"status": {"completed": True, "steps": EXPECTED_STEPS}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    import pandas as pd

    pd.DataFrame({"timestamp": [f"t{i}" for i in range(EXPECTED_STEPS)]}).to_csv(
        tmp_path / "trajectory.csv", index=False
    )
    assert validate_completed_job(tmp_path)

    short = json.dumps({"status": {"completed": True, "steps": 96}})
    (tmp_path / "metrics.json").write_text(short, encoding="utf-8")
    assert not validate_completed_job(tmp_path)

    incomplete = json.dumps({"status": {"completed": False, "steps": EXPECTED_STEPS}})
    (tmp_path / "metrics.json").write_text(incomplete, encoding="utf-8")
    assert not validate_completed_job(tmp_path)


def test_load_masked_components_fail_closed_without_stats(tmp_path: Path) -> None:
    from microgrid_simulator.experiments.monthly_evaluation import MonthlyJob

    artifact, vecnorm = artifact_paths_for("E0", "sac_f3", 0)
    job = MonthlyJob(
        scenario="E0",
        policy="sac_f3",
        seed=0,
        config_path=Path("configs/f3-e0-monthly.yaml"),
        artifact_path=artifact,
        vecnorm_path=tmp_path / "missing.pkl",
        output_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="fail closed"):
        load_masked_rl_components(job)


def test_monthly_configs_pin_march_contract() -> None:
    for i in range(6):
        settings = load_settings(f"configs/f3-e{i}-monthly.yaml")
        assert settings.episode.horizon_hours == 744.0
        assert settings.episode.telemetry_start == "2026-03-01 00:00:00"
        assert settings.forecast.cache_path == "data/f3/test/forecasts.jsonl"
        assert settings.forecast.strict_cache is True
        assert settings.topology.timestep_hours == 0.25
        assert (
            int(round(settings.episode.horizon_hours / settings.topology.timestep_hours))
            == EXPECTED_STEPS
        )
