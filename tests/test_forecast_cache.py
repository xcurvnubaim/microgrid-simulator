"""Strict forecast-cache loader tests (RL Plan step 2 / Week 1 infrastructure).

Covers ``ForecastCache.load``: valid load, source_id mismatch, malformed JSONL,
missing manifest, misaligned timestamps, and cold-start flag handling, plus the
rolling-hourly ``get_covering`` lookup that reuses one hourly forecast across
the four 15-minute decisions in its hour.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from microgrid_simulator.config import ForecastCfg
from microgrid_simulator.forecast import (
    ForecastCache,
    ForecastCacheError,
    ForecastContext,
    ForecastSourceError,
    StrictCachedForecastClient,
)

EXPECTED_SOURCE = "campus-test"


def _record(
    issued_at: str,
    *,
    source_id: str = EXPECTED_SOURCE,
    horizon: int = 2,
    frequency: float = 1.0,
    cold_start: bool = False,
    first_offset_steps: int = 1,
) -> dict:
    n = int(round(horizon / frequency))
    start = datetime.fromisoformat(issued_at) + timedelta(hours=frequency * first_offset_steps)
    timestamps = [(start + timedelta(hours=frequency * i)).isoformat() for i in range(n)]
    return {
        "issued_at": issued_at,
        "horizon_hours": horizon,
        "frequency_hours": frequency,
        "model_version": "test-model",
        "pv_target": "pv_avg",
        "demand_target": "demand",
        "timestamps": timestamps,
        "pv_values_kw": [0.0] * n,
        "demand_values_kw": [100.0] * n,
        "source_id": source_id,
        "context_time": issued_at,
        "context_steps": 24,
        "cold_start": cold_start,
        "covariate_mode": "ecmwf",
    }


def _manifest(
    source_id: str = EXPECTED_SOURCE,
    total_records: int = 2,
    horizon: int = 2,
    frequency: float = 1.0,
) -> dict:
    return {
        "cache_version": "1.0",
        "source_id": source_id,
        "mode": "http",
        "total_records": total_records,
        "horizon_hours": horizon,
        "frequency_hours": frequency,
        "pv_model": "Chronos-2",
        "demand_model": "Chronos-2",
        "splits": {},
    }


def _write(tmp_path, records: list[dict], manifest: dict) -> tuple[str, str]:
    jsonl_path = tmp_path / "forecasts.jsonl"
    jsonl_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest_path = tmp_path / "forecast_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return str(jsonl_path), str(manifest_path)


def test_strict_cached_client_uses_hourly_covering_snapshot(tmp_path) -> None:
    record = _record("2026-01-15T00:00:00")
    cache_path, manifest_path = _write(tmp_path, [record], _manifest(total_records=1))
    cache = ForecastCache.load(
        cache_path,
        manifest_path,
        expected_source_id=EXPECTED_SOURCE,
    )
    client = StrictCachedForecastClient(ForecastCfg(enabled=True), cache)
    context = ForecastContext(
        source_id=EXPECTED_SOURCE,
        frequency_hours=0.25,
        pv_values_mw=(0.0,),
        demand_values_mw=(0.1,),
    )
    snapshot = client.fetch(context, issued_at="2026-01-15T00:45:00")
    assert snapshot.issued_at == "2026-01-15T00:00:00"


def test_strict_cached_client_fails_without_coverage(tmp_path) -> None:
    record = _record("2026-01-15T00:00:00")
    cache_path, manifest_path = _write(tmp_path, [record], _manifest(total_records=1))
    cache = ForecastCache.load(cache_path, manifest_path, expected_source_id=EXPECTED_SOURCE)
    client = StrictCachedForecastClient(ForecastCfg(enabled=True), cache)
    context = ForecastContext(
        source_id=EXPECTED_SOURCE,
        frequency_hours=0.25,
        pv_values_mw=(0.0,),
        demand_values_mw=(0.1,),
    )
    with pytest.raises(ForecastCacheError):
        client.fetch(context, issued_at="2026-01-14T23:45:00")


def test_strict_cached_client_rejects_stale_covering_snapshot(tmp_path) -> None:
    record = _record("2026-01-15T00:00:00")
    cache_path, manifest_path = _write(tmp_path, [record], _manifest(total_records=1))
    cache = ForecastCache.load(cache_path, manifest_path, expected_source_id=EXPECTED_SOURCE)
    client = StrictCachedForecastClient(ForecastCfg(enabled=True), cache)
    context = ForecastContext(
        source_id=EXPECTED_SOURCE,
        frequency_hours=0.25,
        pv_values_mw=(0.0,),
        demand_values_mw=(0.1,),
    )
    with pytest.raises(ForecastCacheError, match="no cached forecast was issued"):
        client.fetch(context, issued_at="2026-01-15T01:00:00")


def test_load_valid_cache(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00"), _record("2026-01-15T01:00:00")]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=2))
    cache = ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)
    assert len(cache) == 2
    snap = cache.get("2026-01-15T00:00:00")
    assert snap is not None
    assert snap.source_id == EXPECTED_SOURCE
    assert snap.demand_values_mw == pytest.approx((0.1, 0.1))  # kW -> MW at the boundary
    assert snap.pv_values_mw == (0.0, 0.0)
    assert cache.cold_start_issued_at() == ()


def test_load_rejects_manifest_source_mismatch(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00")]
    jsonl_path, manifest_path = _write(
        tmp_path, records, _manifest(source_id="other-scenario", total_records=1)
    )
    with pytest.raises(ForecastSourceError):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_load_rejects_record_source_mismatch(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00", source_id="intruder-scenario")]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=1))
    with pytest.raises(ForecastSourceError):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)



def test_load_rejects_malformed_jsonl(tmp_path) -> None:
    jsonl_path = tmp_path / "forecasts.jsonl"
    jsonl_path.write_text(json.dumps(_record("2026-01-15T00:00:00")) + "\n{not valid json\n")
    manifest_path = tmp_path / "forecast_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(total_records=2)))
    with pytest.raises(ForecastCacheError, match="not valid JSON"):
        ForecastCache.load(
            str(jsonl_path), str(manifest_path), expected_source_id=EXPECTED_SOURCE
        )


def test_load_rejects_missing_manifest(tmp_path) -> None:
    jsonl_path, _ = _write(tmp_path, [_record("2026-01-15T00:00:00")], _manifest(total_records=1))
    missing = tmp_path / "absent_manifest.json"
    with pytest.raises(ForecastCacheError, match="manifest not found"):
        ForecastCache.load(jsonl_path, str(missing), expected_source_id=EXPECTED_SOURCE)


def test_load_rejects_missing_jsonl(tmp_path) -> None:
    manifest_path = tmp_path / "forecast_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(total_records=1)))
    with pytest.raises(ForecastCacheError, match="cache not found"):
        ForecastCache.load(
            str(tmp_path / "absent.jsonl"), str(manifest_path), expected_source_id=EXPECTED_SOURCE
        )


def test_load_rejects_misaligned_issued_at(tmp_path) -> None:
    records = [_record("2026-01-15T00:07:00")]  # 7 minutes off the 15-minute action grid
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=1))
    with pytest.raises(ForecastCacheError, match="action grid"):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_load_rejects_misaligned_forecast_timestamps(tmp_path) -> None:
    record = _record("2026-01-15T00:00:00")
    record["timestamps"] = ["2026-01-15T01:10:00", "2026-01-15T02:10:00"]  # 10 min off grid
    jsonl_path, manifest_path = _write(tmp_path, [record], _manifest(total_records=1))
    with pytest.raises(ForecastCacheError, match="action grid"):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_load_rejects_leaky_horizon(tmp_path) -> None:
    # first_offset_steps=0 makes the first forecast point coincide with issued_at.
    record = _record("2026-01-15T00:00:00", first_offset_steps=0)
    jsonl_path, manifest_path = _write(tmp_path, [record], _manifest(total_records=1))
    with pytest.raises(ForecastCacheError, match="leakage-free"):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_load_rejects_manifest_record_count_mismatch(tmp_path) -> None:
    records = [_record("2026-01-15T00:00:00")]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=5))
    with pytest.raises(ForecastCacheError, match="manifest declares"):
        ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)


def test_load_cold_start_flag_handling(tmp_path) -> None:
    records = [
        _record("2026-01-15T00:00:00", cold_start=True),
        _record("2026-01-15T01:00:00", cold_start=False),
    ]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=2))
    cache = ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)
    assert cache.cold_start_issued_at() == ("2026-01-15T00:00:00",)
    cold = cache.get("2026-01-15T00:00:00")
    warm = cache.get("2026-01-15T01:00:00")
    assert cold is not None and cold.cold_start is True
    assert warm is not None and warm.cold_start is False


def test_get_covering_reuses_hourly_forecast(tmp_path) -> None:
    records = [
        _record("2026-01-15T00:00:00"),
        _record("2026-01-15T01:00:00"),
        _record("2026-01-15T02:00:00"),
    ]
    jsonl_path, manifest_path = _write(tmp_path, records, _manifest(total_records=3))
    cache = ForecastCache.load(jsonl_path, manifest_path, expected_source_id=EXPECTED_SOURCE)
    # One hourly forecast covers the four 15-minute decisions in its hour.
    for minute in ("00", "15", "30", "45"):
        snap = cache.get_covering(f"2026-01-15T01:{minute}:00")
        assert snap is not None
        assert snap.issued_at == "2026-01-15T01:00:00"
    # Before the first issue there is no forecast to reuse.
    assert cache.get_covering("2026-01-14T23:45:00") is None
