"""Deterministic artifact inventory scanner for the agent-gated loop.

Produces a structured JSON manifest from artifact directories. A lightweight
LLM may narrate this manifest into human-readable prose, but may not alter
the manifest, its identities, or its provenance fields.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ArtifactEntry:
    """One discovered artifact file."""

    rel_path: str
    size_bytes: int
    sha256: str


@dataclass
class ModelArtifact:
    """One trained model (checkpoint + optional VecNormalize)."""

    scenario: str
    seed: int
    policy: str
    model_zip: ArtifactEntry | None = None
    vecnormalize_pkl: ArtifactEntry | None = None


@dataclass
class EvaluationArtifact:
    """One evaluation result (trajectory + metrics)."""

    scenario: str
    policy: str
    seed: int | None
    window: str
    trajectory_csv: ArtifactEntry | None = None
    metrics_json: ArtifactEntry | None = None


@dataclass
class ArtifactManifest:
    """Full inventory for one agent-loop iteration or baseline."""

    iteration: int | None
    models: list[ModelArtifact] = field(default_factory=list)
    evaluations: list[EvaluationArtifact] = field(default_factory=list)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path: Path, root: Path) -> ArtifactEntry:
    return ArtifactEntry(
        rel_path=str(path.relative_to(root)),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def scan_models(root: Path, scenario: str = "E5") -> list[ModelArtifact]:
    """Scan a model artifact directory for zip + vecnormalize files.

    Expected layout: ``<root>/<scenario>/seed-{0,1,2}/sac_microgrid.zip``
    """
    models: list[ModelArtifact] = []
    scenario_dir = root / scenario
    if not scenario_dir.is_dir():
        return models
    for seed_dir in sorted(scenario_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.replace("seed-", ""))
        except (ValueError, IndexError):
            continue
        model = ModelArtifact(scenario=scenario, seed=seed, policy="sac_f3")
        zip_file = seed_dir / "sac_microgrid.zip"
        if zip_file.is_file():
            model.model_zip = _entry(zip_file, root)
        vn_file = zip_file.with_name(zip_file.stem + "_vecnormalize.pkl")
        if vn_file.is_file():
            model.vecnormalize_pkl = _entry(vn_file, root)
        models.append(model)
    return models


def scan_evaluations(
    root: Path,
    scenario: str = "E5",
    policy: str | None = None,
) -> list[EvaluationArtifact]:
    """Scan evaluation result directories.

    Expected layout:
        ``<root>/<scenario>/<policy>[_seedN]/<window>/trajectory.csv``
        ``<root>/<scenario>/<policy>[_seedN]/<window>/metrics.json``
    """
    evals: list[EvaluationArtifact] = []
    scenario_dir = root / scenario
    if not scenario_dir.is_dir():
        return evals
    for policy_dir in sorted(scenario_dir.iterdir()):
        if not policy_dir.is_dir():
            continue
        pname = policy_dir.name
        if policy and policy not in pname:
            continue
        seed: int | None = None
        pol = pname
        if "_seed" in pname:
            pol, seed_str = pname.rsplit("_seed", 1)
            try:
                seed = int(seed_str)
            except ValueError:
                seed = None
        for window_dir in sorted(policy_dir.iterdir()):
            if not window_dir.is_dir():
                continue
            ev = EvaluationArtifact(
                scenario=scenario,
                policy=pol,
                seed=seed,
                window=window_dir.name,
            )
            traj = window_dir / "trajectory.csv"
            if traj.is_file():
                ev.trajectory_csv = _entry(traj, root)
            metrics = window_dir / "metrics.json"
            if metrics.is_file():
                ev.metrics_json = _entry(metrics, root)
            evals.append(ev)
    return evals


def to_dict(manifest: ArtifactManifest) -> dict[str, Any]:
    """Convert manifest to JSON-serializable dict."""
    return {
        "iteration": manifest.iteration,
        "models": [asdict(m) for m in manifest.models],
        "evaluations": [asdict(e) for e in manifest.evaluations],
    }


def manifest_to_text(manifest: ArtifactManifest) -> str:
    """Produce a human-readable inventory summary.

    A lightweight LLM may narrate this text, but must not alter the identities,
    counts, hashes, or paths.
    """
    lines: list[str] = []
    models = manifest.models
    evals = manifest.evaluations
    lines.append(f"Models: {len(models)}")
    for m in models:
        lines.append(
            f"  {m.scenario}/seed-{m.seed}/{m.policy}: "
            f"{'model=' + (m.model_zip.rel_path if m.model_zip else 'MISSING')}, "
            f"{'vecnorm=' + (m.vecnormalize_pkl.rel_path if m.vecnormalize_pkl else 'MISSING')}"
        )
    lines.append(f"Evaluations: {len(evals)}")
    for e in sorted(evals, key=lambda x: (x.scenario, x.policy, x.window)):
        lines.append(
            f"  {e.scenario}/{e.policy}_seed{e.seed or 0}/{e.window}: "
            f"{'traj=' + (e.trajectory_csv.rel_path if e.trajectory_csv else 'MISSING')}, "
            f"{'metrics=' + (e.metrics_json.rel_path if e.metrics_json else 'MISSING')}"
        )
    for m in models:
        for entry in (m.model_zip, m.vecnormalize_pkl):
            if entry:
                lines.append(f"  SHA256:{entry.sha256[:16]}...  {entry.rel_path}")
    for e in evals:
        for entry in (e.trajectory_csv, e.metrics_json):
            if entry:
                lines.append(f"  SHA256:{entry.sha256[:16]}...  {entry.rel_path}")
    return "\n".join(lines)