"""15-minute forecast resolution migration contract (plan §Test Plan / Gate 1)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from microgrid_simulator.config import ForecastCfg
from microgrid_simulator.forecast import (
    ForecastCache,
    ForecastCacheError,
    ForecastContext,
    StrictCachedForecastClient,
)
from microgrid_simulator.forecast.snapshot import ForecastSnapshot

EXPECTED_SOURCE = "campus-test"


def _record(
    issued_at: str,
    *,
    horizon: int = 24,
    frequency: float = 0.25,
    issue_frequency: float = 0.25,
    first_offset_steps: int = 1,
) -> dict:
    n = int(round(horizon / frequency))
    start = datetime.fromisoformat(issued_at) + timedelta(hours=frequency * first_offset_steps)
    timestamps = [(start + timedelta(hours=frequency * i)).isoformat() for i in range(n)]
    return {
        "issued_at": issued_at,
        "horizon_hours": horizon,
        "frequency_hours": frequency,
        "issue_frequency_hours": issue_frequency,
        "forecast_steps": n,
        "target_units": "kw",
        "model_version": "test-model",
        "pv_target": "pv_avg",
        "demand_target": "demand",
        "timestamps": timestamps,
        "pv_values_kw": [0.0] * n,
        "demand_values_kw": [100.0] * n,
        "source_id": EXPECTED_SOURCE,
        "context_time": issued_at,
        "context_steps": 512,
        "cold_start": False,
        "covariate_mode": "shortwave",
    }


def _manifest(
    *,
    horizon: int = 24,
    frequency: float = 0.25,
    issue_frequency: float = 0.25,
    total_records: int = 1,
) -> dict:
    return {
        "cache_version": "1.0",
        "source_id": EXPECTED_SOURCE,
        "mode": "http",
        "total_records": total_records,
        "horizon_hours": horizon,
        "frequency_hours": frequency,
        "issue_frequency_hours": issue_frequency,
        "forecast_steps": int(round(horizon / frequency)),
        "target_units": "kw",
        "pv_model": "Chronos-2",
        "demand_model": "Chronos-2",
        "causal_preprocessing": True,
        "splits": {},
    }


def _write(tmp_path, records, manifest) -> tuple[str, str]:
    jsonl_path = tmp_path / "forecasts.jsonl"
    jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest_path = tmp_path / "forecast_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return str(jsonl_path), str(manifest_path)


def test_config_derives_96_points_from_24h_at_quarter_hour() -> None:
    cfg = ForecastCfg(
        enabled=True, horizon_hours=24, target_frequency_hours=0.25, issue_frequency_hours=0.25
    )
    assert cfg.forecast_steps == 96


def test_config_rejects_non_integer_horizon_frequency() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ForecastCfg(horizon_hours=24, target_frequency_hours=0.35)


def test_config_rejects_issue_frequency_exceeding_horizon() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        ForecastCfg(horizon_hours=24, target_frequency_hours=0.25, issue_frequency_hours=25.0)


def test_strict_load_accepts_96_point_15min_cache(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00")]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=1))
    cache = ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)
    snap = cache.get("2026-01-15T00:00:00")
    assert snap is not None
    assert snap.steps == 96
    assert snap.frequency_hours == 0.25
    assert snap.issue_frequency_hours == 0.25
    assert snap.target_units == "kw"
    assert snap.timestamps[0] == "2026-01-15T00:15:00"
    assert snap.timestamps[-1] == "2026-01-16T00:00:00"


def test_strict_load_rejects_manifest_steps_mismatch(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00")]
    manifest = _manifest(total_records=1)
    manifest["forecast_steps"] = 24  # wrong: should be 96
    jsonl_path, manifest_path = _write(tmp_path, records, manifest)
    with pytest.raises(ForecastCacheError, match="forecast_steps"):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_strict_load_rejects_issue_frequency_mismatch(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00", issue_frequency=1.0)]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=1))
    with pytest.raises(ForecastCacheError, match="issue_frequency_hours"):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_strict_load_rejects_target_units_mismatch(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00")]
    records[0]["target_units"] = "mw"
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=1))
    with pytest.raises(ForecastCacheError, match="target_units"):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_first_target_is_strictly_after_issued_at(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00", first_offset_steps=0)]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=1))
    with pytest.raises(ForecastCacheError, match="leakage-free"):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_96_targets_exactly_15_minutes_apart(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00")]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=1))
    cache = ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)
    snap = cache.get("2026-01-15T00:00:00")
    import pandas as pd

    ts = [pd.Timestamp(t) for t in snap.timestamps]
    spacings = [(b - a).total_seconds() for a, b in zip(ts, ts[1:], strict=False)]
    assert all(s == 900 for s in spacings)
    assert len(ts) == 96


def test_each_issue_uses_context_no_later_than_cutoff(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00")]
    records[0]["context_time"] = "2026-01-15T00:15:00"  # after issued_at
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=1))
    with pytest.raises(ForecastCacheError, match="context_time is after issued_at"):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_strict_client_uses_15min_covering_snapshot(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00")]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=1))
    cache = ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)
    client = StrictCachedForecastClient(ForecastCfg(enabled=True), cache)
    context = ForecastContext(
        source_id=EXPECTED_SOURCE,
        frequency_hours=0.25,
        pv_values_mw=(0.0,),
        demand_values_mw=(0.1,),
    )
    snap = client.fetch(context, issued_at="2026-01-15T00:00:00")
    assert snap.issued_at == "2026-01-15T00:00:00"
    # A request one full issue-interval later has no covering snapshot.
    with pytest.raises(ForecastCacheError, match="no cached forecast"):
        client.fetch(context, issued_at="2026-01-15T00:15:00")


def test_pypsa_rh_direct_15min_path_consumes_one_point_per_tick() -> None:
    from microgrid_simulator.controllers.pypsa_rolling_horizon import _align_forecast_to_steps

    snapshot = ForecastSnapshot(
        issued_at="2026-01-15T00:00:00",
        horizon_hours=2,
        frequency_hours=0.25,
        issue_frequency_hours=0.25,
        model_version="cached",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=("2026-01-15T00:15:00", "2026-01-15T00:30:00"),
        pv_values_mw=(0.1, 0.2),
        demand_values_mw=(0.5, 0.6),
        source_id="islanded_72h_2026-01-15",
    )
    cache = ForecastCache({"2026-01-15T00:00:00": snapshot})
    demand, pv = _align_forecast_to_steps(
        cache, "2026-01-15T00:00:00", n_steps=8, control_interval_hours=0.25
    )
    # one point per tick, no repetition; zero-padded beyond horizon.
    assert pv.tolist() == [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert demand.tolist() == [0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_observation_shape_scales_with_forecast_steps() -> None:
    from microgrid_simulator.core.types import GridState
    from microgrid_simulator.rl.env import build_observation

    state = GridState()
    f24 = build_observation(state, False, forecast_horizon=24)
    f96 = build_observation(state, False, forecast_horizon=96)
    base = len(f24) - 24 * 2 - 1
    assert base == len(f96) - 96 * 2 - 1
    assert len(f96) - len(f24) == (96 - 24) * 2


def test_summary_forecast_is_causal_and_keeps_raw_observation_width() -> None:
    from microgrid_simulator.core.types import GridState
    from microgrid_simulator.rl.env import build_observation, summarize_forecast

    state = GridState()
    pv = np.full(96, 0.2, dtype=np.float32)
    demand = np.full(96, 0.5, dtype=np.float32)
    summary = summarize_forecast(pv, demand, 0.25)

    assert summary.shape == (15,)
    assert summary[2] == pytest.approx(0.3)
    assert summary[9] == pytest.approx(1.2)
    assert summary[11] == pytest.approx(7.2)
    assert summary[12] == pytest.approx(0.0)

    raw = build_observation(
        state,
        False,
        pv_forecast_mw=pv,
        demand_forecast_mw=demand,
        forecast_available=True,
        forecast_horizon=96,
        forecast_timestep_hours=0.25,
    )
    compact = build_observation(
        state,
        False,
        pv_forecast_mw=pv,
        demand_forecast_mw=demand,
        forecast_available=True,
        forecast_horizon=96,
        forecast_representation="summary",
        forecast_timestep_hours=0.25,
    )
    assert compact.shape == raw.shape
    assert compact[-192:-177].tolist() == pytest.approx(summary.tolist())
    assert np.all(compact[-177:] == 0.0)


def test_summary_forecast_is_zeroed_when_unavailable() -> None:
    from microgrid_simulator.core.types import GridState
    from microgrid_simulator.rl.env import build_observation

    state = GridState()
    obs = build_observation(
        state,
        False,
        pv_forecast_mw=np.ones(96),
        demand_forecast_mw=np.ones(96),
        forecast_horizon=96,
        forecast_representation="summary",
        forecast_timestep_hours=0.25,
    )
    assert np.all(obs[-192:] == 0.0)
