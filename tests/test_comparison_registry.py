"""Contract checks for the deferred Module 6 comparison execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_preparation_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_sac_training", REPO_ROOT / "scripts" / "prepare_sac_training.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_resolves_36_sac_jobs_without_launching() -> None:
    module = _load_preparation_module()
    registry = module.load_registry()
    jobs = module.prepare_jobs(registry)

    assert len(jobs) == 36
    assert {job["scenario"] for job in jobs} == {"E0", "E1", "E2", "E3", "E4", "E5"}
    assert {job["forecast_mode"] for job in jobs} == {"cached", "none"}
    assert {job["seed"] for job in jobs} == {0, 1, 2}
    assert registry["training"]["do_not_launch"] is True


def test_march_manifest_is_frozen_and_has_nine_windows() -> None:
    import json

    manifest = json.loads((REPO_ROOT / "configs" / "march-windows-2026.json").read_text())
    assert manifest["status"] == "frozen-post-hoc"
    assert len(manifest["starts"]) == 9
    assert manifest["window_hours"] == 72
