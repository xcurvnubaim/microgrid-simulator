"""Shared runtime-default scenario selection."""

from __future__ import annotations

from pathlib import Path

from microgrid_simulator.config import DEFAULT_CONFIG_ENV, find_config_path, load_settings
from microgrid_simulator.ui.server import _base_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = REPO_ROOT / "configs" / "islanded-baseline-72h.yaml"


def test_runtime_default_is_nominal_islanded_campus(monkeypatch) -> None:
    monkeypatch.delenv(DEFAULT_CONFIG_ENV, raising=False)
    monkeypatch.chdir(REPO_ROOT)

    assert find_config_path() == DEFAULT_SCENARIO
    settings = load_settings()
    assert settings.scenario.name == "islanded_72h_2026-01-15"
    assert settings.external_reference is None


def test_dashboard_uses_shared_default_and_respects_override(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    active = REPO_ROOT / "configs" / "islanded-baseline-72h.yaml"
    monkeypatch.setenv(DEFAULT_CONFIG_ENV, str(active))

    assert _base_settings().scenario.name == "islanded_72h_2026-01-15"


def test_nominal_contract_values_are_explicit_in_yaml() -> None:
    settings = load_settings(DEFAULT_SCENARIO)

    assert settings.topology.timestep_hours == 0.25
    assert settings.episode.horizon_hours == 72.0
    assert settings.forecast.horizon_hours == 24
    assert settings.battery.capacity_mwh == 0.50
    assert settings.battery.max_charge_mw == 0.25
    assert settings.battery.max_discharge_mw == 0.25
    assert settings.diesel.max_kw == 400.0
    assert settings.diesel.min_kw == 80.0
    assert settings.diesel.ramp_kw_per_min == 160.0
    assert settings.reward.w_carbon == 4.0
    assert settings.reward.w_health == 0.5
    assert settings.reward.w_unserved == 20.0


def test_recursive_extends(tmp_path) -> None:
    grandparent = tmp_path / "grandparent.yaml"
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"

    grandparent.write_text("reward:\n  w_carbon: 4.0\n  w_health: 0.5\n")
    parent.write_text(f"extends: {grandparent.name}\nreward:\n  w_health: 1.0\n  w_unserved: 20.0\n")
    child.write_text(f"extends: {parent.name}\nreward:\n  w_unserved: 30.0\n")

    from microgrid_simulator.config import Settings
    settings = Settings.from_yaml(child)
    assert settings.reward.w_carbon == 4.0
    assert settings.reward.w_health == 1.0
    assert settings.reward.w_unserved == 30.0

