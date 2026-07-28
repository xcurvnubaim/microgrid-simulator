"""Tests for ForecastCache loader and RandomEpisodeSampler."""

from __future__ import annotations

import pandas as pd
import pytest

from microgrid_simulator.config import Settings
from microgrid_simulator.forecast import ForecastCache
from microgrid_simulator.rl.sampler import RandomEpisodeSampler


def test_forecast_cache_jsonl_roundtrip(tmp_path) -> None:
    cache_file = tmp_path / "forecasts.jsonl"
    jsonl_data = (
        '{"issued_at": "2026-01-15T00:00:00Z", "horizon_hours": 24, "frequency_hours": 1.0, '
        '"pv_values_kw": [0.0, 10.0], "demand_values_kw": [50.0, 100.0], "source_id": "test"}\n'
    )
    cache_file.write_text(jsonl_data)

    cache = ForecastCache.from_jsonl(str(cache_file))
    assert len(cache) == 1
    snap = cache.get("2026-01-15T00:00:00Z")
    assert snap is not None
    assert snap.pv_values_mw == (0.0, 0.01)
    assert snap.demand_values_mw == (0.05, 0.1)

    out_file = tmp_path / "out.jsonl"
    cache.to_jsonl(str(out_file))
    reloaded = ForecastCache.from_jsonl(str(out_file))
    assert len(reloaded) == 1
    snap2 = reloaded.get("2026-01-15T00:00:00Z")
    assert snap2 is not None
    assert snap2.pv_values_mw == (0.0, 0.01)


def test_random_episode_sampler_splits() -> None:
    settings = Settings()
    train_sampler = RandomEpisodeSampler(settings, split="train", seed=42)
    val_sampler = RandomEpisodeSampler(settings, split="val", seed=42)
    test_sampler = RandomEpisodeSampler(settings, split="test", seed=42)

    assert len(train_sampler.valid_start_timestamps) > 0
    assert len(val_sampler.valid_start_timestamps) > 0
    assert len(test_sampler.valid_start_timestamps) > 0

    # Ensure train starts are strictly before Feb 2026
    for start in train_sampler.valid_start_timestamps:
        assert start < pd.Timestamp("2026-02-01")

    # Ensure val starts are in Feb 2026 and do not overlap Feb 4 gap
    feb4_start = pd.Timestamp("2026-02-04 00:00:00")
    feb4_end = pd.Timestamp("2026-02-05 00:00:00")
    for start in val_sampler.valid_start_timestamps:
        end = start + pd.to_timedelta(settings.episode.horizon_hours, unit="h")
        assert start >= pd.Timestamp("2026-02-01") and end <= pd.Timestamp("2026-03-01")
        assert end <= feb4_start or start >= feb4_end

    # Ensure test starts are in Mar 2026
    for start in test_sampler.valid_start_timestamps:
        end = start + pd.to_timedelta(settings.episode.horizon_hours, unit="h")
        assert start >= pd.Timestamp("2026-03-01") and end <= pd.Timestamp("2026-04-01")
