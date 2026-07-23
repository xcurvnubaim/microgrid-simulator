"""Shared runtime-default scenario selection."""

from __future__ import annotations

from pathlib import Path

from microgrid_simulator.config import DEFAULT_CONFIG_ENV, find_config_path, load_settings
from microgrid_simulator.ui.server import _base_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = REPO_ROOT / "configs" / "pymgrid25-scenario-2.yaml"


def test_runtime_default_is_native_pymgrid_scenario_2(monkeypatch) -> None:
    monkeypatch.delenv(DEFAULT_CONFIG_ENV, raising=False)
    monkeypatch.chdir(REPO_ROOT)

    assert find_config_path() == DEFAULT_SCENARIO
    settings = load_settings()
    assert settings.scenario.name == "pymgrid25-scenario-2"
    assert settings.external_reference is not None
    assert settings.external_reference.scenario_number == 2


def test_dashboard_uses_shared_default_and_respects_override(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv(DEFAULT_CONFIG_ENV, str(REPO_ROOT / "configs" / "simulator.yaml"))

    assert _base_settings().scenario.name == "default"
