from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_ems_scenario.sh"
RULE_CONFIG = ROOT / "configs" / "docker-islanded-72h.yaml"
SAC_ARTIFACT = (
    ROOT
    / "artifacts"
    / "sac"
    / "hardunserved-v3-60k"
    / "seed-0"
    / "sac_microgrid.zip"
)


def _run(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=ROOT,
        env={**os.environ, "LC_ALL": "C"},
        capture_output=True,
        text=True,
        input=input_text,
        check=False,
    )


def test_launcher_dry_run_resolves_rule_scenario() -> None:
    result = _run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert f"{RULE_CONFIG} -> /app/configs/docker-islanded-72h.yaml" in result.stdout
    assert "policy:   rule" in result.stdout
    assert "artifact: none" in result.stdout


def test_launcher_dry_run_resolves_sac_and_ppo_artifacts() -> None:
    relative = SAC_ARTIFACT.relative_to(ROOT)
    for policy in ("sac", "ppo"):
        result = _run("--dry-run", "--policy", policy, "--artifact", str(relative))
        assert result.returncode == 0, result.stderr
        assert f"policy:   {policy}" in result.stdout
        assert "/app/artifacts/sac/hardunserved-v3-60k/seed-0/sac_microgrid.zip" in (
            result.stdout
        )


def test_launcher_interactive_menu_discovers_artifacts_from_directory() -> None:
    result = _run("--interactive", "--dry-run", input_text="2\n1\n2\n1\n")

    assert result.returncode == 0, result.stderr
    assert "Available ems scenarios (discovered from configs/)" in result.stdout
    assert "configs/docker-islanded-72h.yaml" in result.stdout
    assert "Available EMS policies (discovered from artifacts/)" in result.stdout
    assert "sac (" in result.stdout
    assert "Available SAC artifacts:" in result.stdout
    assert "policy:   sac" in result.stdout
    assert "/app/artifacts/sac/" in result.stdout


def test_launcher_rejects_invalid_inputs(tmp_path: Path) -> None:
    outside_artifact = tmp_path / "model.zip"
    outside_artifact.write_bytes(b"not a model")
    cases = (
        ("--dry-run", "--config", "configs/missing.yaml"),
        ("--dry-run", "--policy", "dqn"),
        ("--dry-run", "--policy", "sac"),
        ("--dry-run", "--policy", "sac", "--artifact", "artifacts/missing.zip"),
        ("--dry-run", "--policy", "sac", "--artifact", str(outside_artifact)),
        ("--dry-run", "--policy", "rule", "--artifact", str(SAC_ARTIFACT)),
    )

    for args in cases:
        result = _run(*args)
        assert result.returncode == 2, (args, result.stdout, result.stderr)
