"""Dashboard PyPSA-RH (PyPSA planner + pandapower plant) integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from microgrid_simulator.config import Settings
from microgrid_simulator.ui.rollout import (
    LEGACY_POLICY_ALIASES,
    POLICIES,
    _setup_pypsa_rh_controller,
)


def test_pypsa_rh_is_in_dashboard_policies() -> None:
    assert "pypsa_rh" in POLICIES
    assert "mpc" not in POLICIES
    assert LEGACY_POLICY_ALIASES["mpc"] == "pypsa_rh"


def test_pypsa_rh_rejects_missing_cache_wired() -> None:
    settings = Settings.from_yaml("configs/islanded-baseline-72h.yaml")
    settings.forecast.cache_path = ""
    settings.forecast.manifest_path = ""

    with pytest.raises(ValueError, match="strict cached forecasting requires"):
        from microgrid_simulator.rl.env import MicrogridEnv

        MicrogridEnv(settings=settings)


def test_pypsa_rh_rejects_non_causal_manifest_for_replay(tmp_path) -> None:
    import json
    import shutil

    settings = Settings.from_yaml("configs/islanded-baseline-72h.yaml")
    manifest_path = Path(settings.forecast.manifest_path)
    cache_path = Path(settings.forecast.cache_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["causal_preprocessing"] = False
    tmp_cache = tmp_path / "cache.jsonl"
    tmp_manifest = tmp_path / "manifest.json"
    shutil.copyfile(cache_path, tmp_cache)
    tmp_manifest.write_text(json.dumps(manifest))
    overrides = {
        "forecast": settings.forecast.model_copy(
            update={
                "cache_path": str(tmp_cache),
                "manifest_path": str(tmp_manifest),
            }
        )
    }
    settings = settings.model_copy(update=overrides)

    from microgrid_simulator.rl.env import MicrogridEnv

    with pytest.raises((ValueError, Exception), match="causal_preprocessing"):
        MicrogridEnv(settings=settings)


def test_pypsa_rh_backend_constructor_accepted() -> None:
    """Smoke: PyPSA backend + controller create against the active scenario."""
    settings = Settings.from_yaml("configs/islanded-baseline-72h.yaml")
    from microgrid_simulator.backends.pypsa_backend import PyPSAOperationalBackend
    from microgrid_simulator.controllers.pypsa_rolling_horizon import PyPSARollingHorizonController
    from microgrid_simulator.forecast.cache import ForecastCache
    from microgrid_simulator.rl.env import MicrogridEnv

    env = MicrogridEnv(settings=settings)
    env.reset(seed=0)
    try:
        pypsa_backend = PyPSAOperationalBackend(settings)
        win = getattr(env, "telemetry_window", None)
        pypsa_backend.reset(
            demand_window_mw=win.demand_mw if win is not None else None,
            pv_window_mw=win.pv_mw if win is not None else None,
        )
        cache = ForecastCache.load(
            settings.forecast.cache_path,
            settings.forecast.manifest_path,
            expected_source_id=settings.forecast.source_id or settings.scenario.name,
            action_interval_hours=settings.topology.timestep_hours,
        )
        ctrl = PyPSARollingHorizonController(backend=pypsa_backend, forecast_cache=cache)
        assert ctrl is not None
    finally:
        env.close()


def test_pypsa_rh_setup_controller_labels() -> None:
    settings = Settings.from_yaml("configs/islanded-baseline-72h.yaml")
    from microgrid_simulator.rl.env import MicrogridEnv

    env = MicrogridEnv(settings=settings)
    env.reset(seed=0)
    try:
        ctrl, summary = _setup_pypsa_rh_controller(settings, env)
        assert summary["planner"] == "pypsa"
        assert summary["plant"] == "pandapower"
        assert summary["forecast_mode"] == "strict_cache"
        assert summary["replan_interval_hours"] == pytest.approx(
            settings.backend.rolling_horizon_hours
        )
        ctrl.reset()
    finally:
        env.close()


def _make_live_env() -> tuple[Settings, object]:
    """Env settings wired to a live (strict_cache=false) forecast service."""
    base = Settings.from_yaml("configs/islanded-baseline-72h.yaml")
    forecast = base.forecast.model_copy(
        update={
            "enabled": True,
            "strict_cache": False,
            "service_url": "http://localhost:8000",
            "cache_path": None,
            "manifest_path": None,
        }
    )
    settings = base.model_copy(update={"forecast": forecast})
    return settings, None


class _FakeForecastClient:
    def __init__(self, settings):
        self.settings = settings
        self.issued_at = "2026-01-15T00:00:00"
        self._snap = None

    def fetch(self, context, issued_at=None):  # noqa: ANN001
        from datetime import datetime as dt
        from datetime import timedelta

        from microgrid_simulator.forecast.snapshot import ForecastSnapshot

        self.cur_issued = issued_at or self.issued_at
        n = self.settings.forecast.forecast_steps
        frequency = self.settings.forecast.target_frequency_hours
        start = dt.fromisoformat(self.cur_issued) + timedelta(hours=frequency)
        timestamps = tuple((start + timedelta(hours=frequency * i)).isoformat() for i in range(n))
        self._snap = ForecastSnapshot(
            issued_at=self.cur_issued,
            horizon_hours=self.settings.forecast.horizon_hours,
            frequency_hours=frequency,
            model_version="fake-live",
            pv_target="pv_avg",
            demand_target="demand",
            timestamps=timestamps,
            pv_values_mw=tuple(0.1 for _ in range(n)),
            demand_values_mw=tuple(0.5 for _ in range(n)),
            source_id=self.settings.forecast.source_id or self.settings.scenario.name,
            context_time=self.cur_issued,
            context_steps=1,
            cold_start=False,
            covariate_mode="live",
            issue_frequency_hours=self.settings.forecast.issue_frequency_hours,
            forecast_steps=n,
            target_units="kw",
        )
        return self._snap

    @property
    def snapshot(self):
        return self._snap if self._snap is not None else self.fetch(None)


def test_pypsa_rh_live_mode_setup_and_metadata() -> None:
    settings, _ = _make_live_env()
    from microgrid_simulator.rl.env import MicrogridEnv
    from microgrid_simulator.ui.rollout import _setup_pypsa_rh_controller

    env = MicrogridEnv(settings=settings, forecast_client=_FakeForecastClient(settings))
    env.reset(seed=0)
    try:
        ctrl, summary = _setup_pypsa_rh_controller(settings, env)
        assert summary["forecast_mode"] == "live_service"
        assert summary["service_url"] == "http://localhost:8000"
        assert summary["planner"] == "pypsa"
        assert summary["plant"] == "pandapower"
        ctrl.reset()
    finally:
        env.close()


def test_pypsa_rh_live_mode_run_rollout_uses_live_snapshot() -> None:
    settings, _ = _make_live_env()
    # run_rollout constructs its own env; patch the module-level ForecastClient
    # the env refers to at line 241 so the internal env uses the fake too.
    import microgrid_simulator.rl.env as env_mod
    from microgrid_simulator.ui.rollout import run_rollout

    original = env_mod.ForecastClient
    env_mod.ForecastClient = lambda cfg: _FakeForecastClient(settings)
    try:
        result = run_rollout(settings, policy="pypsa_rh", seed=0)
        assert "rows" in result
        assert result["rows"], "PyPSA-RH live rollout produced no rows"
        meta = result["meta"]
        assert meta["pypsa_rh"]["forecast_mode"] == "live_service"
        assert meta["pypsa_rh"]["planner"] == "pypsa"
    finally:
        env_mod.ForecastClient = original


def test_pypsa_rh_requires_service_url_when_live() -> None:
    settings, _ = _make_live_env()
    settings.forecast.service_url = ""
    from microgrid_simulator.rl.env import MicrogridEnv
    from microgrid_simulator.ui.rollout import _setup_pypsa_rh_controller

    env = MicrogridEnv(settings=settings, forecast_client=_FakeForecastClient(settings))
    env.reset(seed=0)
    try:
        with pytest.raises(ValueError, match="service_url"):
            _setup_pypsa_rh_controller(settings, env)
    finally:
        env.close()


def test_pypsa_rh_rejects_forecast_mode_none() -> None:
    from microgrid_simulator.rl.env import MicrogridEnv
    from microgrid_simulator.ui.rollout import _setup_pypsa_rh_controller

    settings = Settings.from_yaml("configs/islanded-baseline-72h.yaml")
    settings.rl.forecast_mode = "none"
    env = MicrogridEnv(settings=settings, forecast_client=_FakeForecastClient(settings))
    env.reset(seed=0)
    try:
        with pytest.raises(ValueError, match="forecast_mode=none"):
            _setup_pypsa_rh_controller(settings, env)
    finally:
        env.close()
