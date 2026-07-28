"""Chronos service contract and Gym observation integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandapower")

from microgrid_simulator.config import ForecastCfg, Settings  # noqa: E402
from microgrid_simulator.digital_twin.replay import TelemetryWindow  # noqa: E402
from microgrid_simulator.forecast import (  # noqa: E402
    ForecastClient,
    ForecastContext,
    ForecastError,
    ForecastSnapshot,
    ForecastSourceError,
)
from microgrid_simulator.rl.env import MicrogridEnv  # noqa: E402


def _context(
    source_id: str = "default", pv: tuple[float, ...] = (0.1,), demand: tuple[float, ...] = (0.2,)
) -> ForecastContext:
    return ForecastContext(
        source_id=source_id,
        frequency_hours=1.0,
        pv_values_mw=pv,
        demand_values_mw=demand,
    )


def _payload(horizon: int = 3, source_id: str = "default") -> dict[str, object]:
    return {
        "issued_at": "2026-07-23T12:00:00",
        "horizon_h": horizon,
        "frequency_h": 1.0,
        "model_version": "autogluon/chronos-2-small",
        "source_id": source_id,
        "context_time": "2026-07-23T12:00:00",
        "context_steps": 1,
        "cold_start": True,
        "covariate_mode": "none",
        "timestamps": [f"2026-07-23T{12 + i:02d}:00:00" for i in range(horizon)],
        "units": {"pv_avg": "kw", "demand": "kw"},
        "forecast": {
            "pv_avg": [-0.1, 100.0, 200.0][:horizon],
            "demand": [50.0, 150.0, 250.0][:horizon],
        },
    }


def test_client_requests_configured_horizon_and_converts_kw(monkeypatch) -> None:
    cfg = ForecastCfg(enabled=True, horizon_hours=3)
    client = ForecastClient(cfg)
    seen: dict[str, object] = {}

    def fake_post(url: str, body: dict[str, object]) -> dict[str, object]:
        seen.update({"url": url, "body": body})
        return _payload()

    monkeypatch.setattr(client, "_post_json", fake_post)

    snapshot = client.fetch(_context())

    assert seen["url"] == "http://127.0.0.1:8000/forecast"
    assert seen["body"] == {
        "horizon_h": 3,
        "context": {
            "source_id": "default",
            "frequency_h": 1.0,
            "pv_kw": [100.0],
            "demand_kw": [200.0],
        },
    }
    assert snapshot.pv_values_mw == pytest.approx((0.0, 0.1, 0.2))
    assert snapshot.demand_values_mw == pytest.approx((0.05, 0.15, 0.25))
    assert snapshot.source_id == "default"


def test_client_sends_step_timestamp_when_available(monkeypatch) -> None:
    client = ForecastClient(ForecastCfg(enabled=True, horizon_hours=3))
    seen: dict[str, object] = {}

    def fake_post(_url: str, body: dict[str, object]) -> dict[str, object]:
        seen.update(body)
        return _payload()

    monkeypatch.setattr(client, "_post_json", fake_post)
    client.fetch(_context(), issued_at="2026-01-15T00:00:00")

    assert seen["horizon_h"] == 3
    assert seen["issued_at"] == "2026-01-15T00:00:00"
    assert seen["context"]["source_id"] == "default"


def test_client_rejects_a_mismatched_horizon(monkeypatch) -> None:
    client = ForecastClient(ForecastCfg(enabled=True, horizon_hours=2))
    monkeypatch.setattr(client, "_post_json", lambda *_: _payload(horizon=3))

    with pytest.raises(ForecastError, match="requested 2 h"):
        client.fetch(_context())


def test_client_rejects_an_incompatible_forecast_source(monkeypatch) -> None:
    client = ForecastClient(ForecastCfg(enabled=True, horizon_hours=3))
    monkeypatch.setattr(
        client,
        "_post_json",
        lambda *_: _payload(source_id="campus-telemetry-2025-2026"),
    )

    with pytest.raises(
        ForecastSourceError,
        match="forecast source is incompatible with pymgrid25-scenario-2",
    ):
        client.fetch(_context(source_id="pymgrid25-scenario-2"))


def test_client_rejects_a_cross_source_scale_leak(monkeypatch) -> None:
    client = ForecastClient(ForecastCfg(enabled=True, horizon_hours=3))
    payload = _payload()
    payload["forecast"] = {"pv_avg": [60.0] * 3, "demand": [60.0] * 3}
    monkeypatch.setattr(client, "_post_json", lambda *_: payload)

    with pytest.raises(ForecastError, match="range is incompatible"):
        client.fetch(_context(pv=(20.0,), demand=(20.0,)))


class _StubForecastClient:
    def __init__(self, snapshot: ForecastSnapshot | None = None, error: str | None = None):
        self.snapshot = snapshot
        self.error = error
        self.calls = 0
        self.issued_at_calls: list[str | None] = []
        self.context_calls: list[ForecastContext] = []

    def fetch(self, context: ForecastContext, issued_at: str | None = None) -> ForecastSnapshot:
        self.calls += 1
        self.issued_at_calls.append(issued_at)
        self.context_calls.append(context)
        if self.error:
            raise ForecastError(self.error)
        assert self.snapshot is not None
        return self.snapshot


def _settings(
    *,
    enabled: bool,
    horizon: int = 3,
    refresh_each_step: bool = True,
    timestep_hours: float = 0.25,
    episode_hours: float = 1.0,
) -> Settings:
    raw = Settings().model_dump()
    raw["forecast"] = {
        "enabled": enabled,
        "horizon_hours": horizon,
        "service_url": "http://forecaster.test",
        "refresh_each_step": refresh_each_step,
    }
    raw["episode"]["horizon_hours"] = episode_hours
    raw["topology"]["timestep_hours"] = timestep_hours
    return Settings(**raw)


def test_enabled_forecast_is_appended_to_observation_and_info() -> None:
    snapshot = ForecastSnapshot(
        issued_at="2026-07-23T12:00:00",
        horizon_hours=3,
        frequency_hours=1.0,
        model_version="chronos-test",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=(),
        pv_values_mw=(0.1, 0.2, 0.3),
        demand_values_mw=(0.7, 0.8, 0.9),
        source_id="default",
    )
    client = _StubForecastClient(snapshot=snapshot)
    env = MicrogridEnv(settings=_settings(enabled=True), forecast_client=client)

    obs, reset_info = env.reset(seed=0)
    base_obs_size = len(obs) - 7  # availability + three PV + three demand values

    assert client.calls == 1
    assert obs[base_obs_size:].tolist() == pytest.approx([1.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert reset_info["forecast_available"] is True
    assert reset_info["forecast_horizon_hours"] == 3

    client.snapshot = ForecastSnapshot(
        issued_at="2026-07-23T12:15:00",
        horizon_hours=3,
        frequency_hours=1.0,
        model_version="chronos-test",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=(),
        pv_values_mw=(0.4, 0.5, 0.6),
        demand_values_mw=(1.0, 1.1, 1.2),
        source_id="default",
    )
    idle = np.zeros(env.action_dim, dtype=np.float32)
    next_obs, _, _, _, step_info = env.step(idle)
    assert client.calls == 2
    assert next_obs[base_obs_size:].tolist() == pytest.approx([1.0, 0.4, 0.5, 0.6, 1.0, 1.1, 1.2])
    assert step_info["forecast_available"] is True
    assert step_info["pv_forecast_mw"] == pytest.approx(0.4)
    assert step_info["demand_forecast_mw"] == pytest.approx(1.0)
    assert step_info["forecast_horizon_mw"] == pytest.approx([0.4, 0.5, 0.6])
    assert step_info["demand_forecast_horizon_mw"] == pytest.approx([1.0, 1.1, 1.2])
    assert step_info["forecast_issued_at"] == "2026-07-23T12:15:00"
    env.close()


def test_source_mismatch_is_explicitly_unavailable_in_environment(monkeypatch) -> None:
    settings = _settings(enabled=True, horizon=2)
    settings.scenario.name = "pymgrid25-scenario-2"
    client = ForecastClient(settings.forecast)
    monkeypatch.setattr(
        client,
        "_post_json",
        lambda *_: _payload(horizon=2, source_id="campus-telemetry-2025-2026"),
    )
    env = MicrogridEnv(settings=settings, forecast_client=client)

    _, info = env.reset(seed=0)

    assert info["forecast_available"] is False
    assert info["forecast_error"] == "forecast source is incompatible with pymgrid25-scenario-2"
    assert info["forecast_source"] == "campus-telemetry-2025-2026"
    env.close()


def test_unavailable_forecast_is_masked_without_stopping_plant() -> None:
    client = _StubForecastClient(error="service offline")
    env = MicrogridEnv(settings=_settings(enabled=True, horizon=2), forecast_client=client)

    obs, reset_info = env.reset(seed=0)

    assert obs[-5:].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert reset_info["forecast_available"] is False
    assert reset_info["forecast_error"] == "service offline"
    env.close()


def test_disabled_forecast_preserves_legacy_observation_shape() -> None:
    disabled = MicrogridEnv(settings=_settings(enabled=False))
    enabled = MicrogridEnv(
        settings=_settings(enabled=True),
        forecast_client=_StubForecastClient(error="service offline"),
    )

    disabled_obs, _ = disabled.reset(seed=0)
    enabled_obs, _ = enabled.reset(seed=0)

    assert len(enabled_obs) == len(disabled_obs) + 7
    disabled.close()
    enabled.close()


def test_per_step_refresh_can_be_disabled() -> None:
    snapshot = ForecastSnapshot(
        issued_at="2026-07-23T12:00:00",
        horizon_hours=2,
        frequency_hours=1.0,
        model_version="chronos-test",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=(),
        pv_values_mw=(0.1, 0.2),
        demand_values_mw=(0.3, 0.4),
        source_id="default",
    )
    client = _StubForecastClient(snapshot=snapshot)
    env = MicrogridEnv(
        settings=_settings(enabled=True, horizon=2, refresh_each_step=False),
        forecast_client=client,
    )
    env.reset(seed=0)

    env.step(np.zeros(env.action_dim, dtype=np.float32))

    assert client.calls == 1
    env.close()


def test_fixed_replay_requests_forecast_at_each_exact_step_timestamp() -> None:
    snapshot = ForecastSnapshot(
        issued_at="2026-01-14T23:45:00",
        horizon_hours=2,
        frequency_hours=1.0,
        model_version="chronos-test",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=(),
        pv_values_mw=(0.1, 0.2),
        demand_values_mw=(0.3, 0.4),
        source_id="default",
    )
    client = _StubForecastClient(snapshot=snapshot)
    env = MicrogridEnv(
        settings=_settings(enabled=True, horizon=2),
        forecast_client=client,
    )
    timestamps = pd.date_range("2026-01-14 23:45:00", periods=5, freq="15min")
    env.telemetry_window = TelemetryWindow(
        demand_mw=np.full(5, 0.1),
        pv_mw=np.full(5, 0.05),
        timestamps=timestamps,
        first_evaluated_timestamp=timestamps[1],
        source_files={"load": "test", "pv": "test"},
    )

    env.reset(seed=0)
    env.step(np.zeros(env.action_dim, dtype=np.float32))

    assert client.issued_at_calls == [
        "2026-01-14T23:45:00",
        "2026-01-15T00:00:00",
    ]
    assert client.context_calls[0].demand_values_mw == pytest.approx((0.1,))
    assert client.context_calls[1].demand_values_mw == pytest.approx((0.1, 0.1))
    env.close()


def test_synthetic_replay_clock_does_not_anchor_historical_forecasts() -> None:
    snapshot = ForecastSnapshot(
        issued_at="2026-03-31T23:00:00",
        horizon_hours=2,
        frequency_hours=1.0,
        model_version="chronos-test",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=(),
        pv_values_mw=(0.1, 0.2),
        demand_values_mw=(0.3, 0.4),
        source_id="default",
    )
    client = _StubForecastClient(snapshot=snapshot)
    env = MicrogridEnv(
        settings=_settings(enabled=True, horizon=2),
        forecast_client=client,
    )
    timestamps = pd.date_range("2000-01-01", periods=5, freq="h")
    env.telemetry_window = TelemetryWindow(
        demand_mw=np.full(5, 0.1),
        pv_mw=np.full(5, 0.05),
        timestamps=timestamps,
        first_evaluated_timestamp=timestamps[1],
        source_files={"load": "synthetic", "pv": "synthetic"},
        timestamps_are_observed=False,
    )

    _, reset_info = env.reset(seed=0)
    env.step(np.zeros(env.action_dim, dtype=np.float32))

    assert client.issued_at_calls == [None, None]
    assert client.context_calls[0].source_id == "default"
    assert reset_info["forecast_uses_observed_replay_timestamp"] is False
    env.close()


def test_repeated_issue_time_advances_the_existing_horizon() -> None:
    snapshot = ForecastSnapshot(
        issued_at="2026-07-23T12:00:00",
        horizon_hours=3,
        frequency_hours=1.0,
        model_version="chronos-test",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=(),
        pv_values_mw=(0.1, 0.2, 0.3),
        demand_values_mw=(0.4, 0.5, 0.6),
        source_id="default",
    )
    client = _StubForecastClient(snapshot=snapshot)
    env = MicrogridEnv(
        settings=_settings(enabled=True, horizon=3, timestep_hours=1.0),
        forecast_client=client,
    )

    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.action_dim, dtype=np.float32))

    assert info["forecast_stale"] is True
    assert info["forecast_age_steps"] == 1
    assert info["pv_forecast_mw"] == pytest.approx(0.2)
    assert info["demand_forecast_mw"] == pytest.approx(0.5)
    env.close()


def test_forecast_context_advances_without_future_replay_values() -> None:
    snapshot = ForecastSnapshot(
        issued_at="2026-07-23T12:00:00",
        horizon_hours=2,
        frequency_hours=1.0,
        model_version="chronos-test",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=(),
        pv_values_mw=(0.1, 0.2),
        demand_values_mw=(0.3, 0.4),
        source_id="default",
    )
    client = _StubForecastClient(snapshot=snapshot)
    env = MicrogridEnv(settings=_settings(enabled=True, horizon=2), forecast_client=client)
    timestamps = pd.date_range("2026-01-01", periods=5, freq="h")
    env.telemetry_window = TelemetryWindow(
        demand_mw=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
        pv_mw=np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
        timestamps=timestamps,
        first_evaluated_timestamp=timestamps[1],
        source_files={"load": "test", "pv": "test"},
    )

    env.reset(seed=0)
    env.step(np.zeros(env.action_dim, dtype=np.float32))

    assert client.context_calls[0].pv_values_mw == pytest.approx((0.10,))
    assert client.context_calls[0].demand_values_mw == pytest.approx((0.01,))
    assert client.context_calls[1].pv_values_mw == pytest.approx((0.10, 0.20))
    assert client.context_calls[1].demand_values_mw == pytest.approx((0.01, 0.02))
    assert 0.30 not in client.context_calls[1].pv_values_mw
    assert 0.03 not in client.context_calls[1].demand_values_mw
    env.close()


@pytest.mark.parametrize("rollout_steps", [24, 168])
def test_stale_rollout_advances_then_reports_horizon_exhaustion(rollout_steps: int) -> None:
    snapshot = ForecastSnapshot(
        issued_at="2026-07-23T12:00:00",
        horizon_hours=2,
        frequency_hours=1.0,
        model_version="chronos-test",
        pv_target="pv_avg",
        demand_target="demand",
        timestamps=(),
        pv_values_mw=(0.1, 0.2),
        demand_values_mw=(0.3, 0.4),
        source_id="default",
    )
    client = _StubForecastClient(snapshot=snapshot)
    env = MicrogridEnv(
        settings=_settings(
            enabled=True,
            horizon=2,
            timestep_hours=1.0,
            episode_hours=float(rollout_steps),
        ),
        forecast_client=client,
    )

    env.reset(seed=0)
    rows = []
    for _ in range(rollout_steps):
        _, _, _, _, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
        rows.append(info)

    available_values = [row["pv_forecast_mw"] for row in rows if row["forecast_available"]]
    assert available_values == pytest.approx([0.2])
    assert all(row["forecast_stale"] or not row["forecast_available"] for row in rows)
    assert rows[-1]["forecast_available"] is False
    assert rows[-1]["forecast_stale"] is True
    assert client.calls == rollout_steps + 1
    env.close()


def test_forecast_cache_loading(tmp_path) -> None:
    jsonl_file = tmp_path / "forecasts.jsonl"
    jsonl_file.write_text(
        '{"issued_at": "2026-01-15T08:00:00Z", "horizon_hours": 24, "frequency_hours": 1.0, '
        '"pv_values_kw": [10.0, 20.0], "demand_values_kw": [100.0, 200.0], "source_id": "test_src"}\n'
    )

    from microgrid_simulator.forecast import CachedForecastClient, ForecastCache

    cache = ForecastCache.from_jsonl(str(jsonl_file))
    assert len(cache) == 1
    snapshot = cache.get("2026-01-15T08:00:00Z")
    assert snapshot is not None
    assert snapshot.pv_values_mw == pytest.approx((0.01, 0.02))
    assert snapshot.demand_values_mw == pytest.approx((0.1, 0.2))

    cfg = ForecastCfg(enabled=True, horizon_hours=24)
    client = CachedForecastClient(cfg, cache)
    ctx = ForecastContext("test_src", 1.0, (0.01,), (0.1,))
    fetched = client.fetch(ctx, issued_at="2026-01-15T08:00:00Z")
    assert fetched.pv_values_mw == pytest.approx((0.01, 0.02))

