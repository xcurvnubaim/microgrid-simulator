"""Compatibility shim — training/evaluation moved to ``microgrid_simulator.rl``.

Old signatures are preserved; new code should call
``microgrid_simulator.rl.train.train`` / ``microgrid_simulator.rl.evaluate.evaluate``.
"""

from __future__ import annotations

from pathlib import Path

from microgrid_simulator.config import Settings


def train(
    settings: Settings,
    algo: str = "ppo",
    total_timesteps: int = 50_000,
    artifact_dir: Path = Path("artifacts"),
    tensorboard_log: Path | None = None,
    seed: int = 0,
) -> Path:
    from microgrid_simulator.rl.train import train as _train

    return _train(
        settings,
        algo=algo,
        total_timesteps=total_timesteps,
        artifact_dir=artifact_dir,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )


def evaluate(
    settings: Settings,
    artifact: Path,
    algo: str = "ppo",
    episodes: int = 3,
) -> dict[str, float]:
    from microgrid_simulator.rl.evaluate import evaluate as _evaluate

    return _evaluate(settings, artifact=artifact, algo=algo, episodes=episodes)
