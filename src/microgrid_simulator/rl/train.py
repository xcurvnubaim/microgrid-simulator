"""Train an SB3 policy against the microgrid env — reproducible by config.

Every run writes to ``<rl.log_dir>/<algo>_<timestamp>/``: Monitor episode CSVs,
``evaluations.npz`` from EvalCallback, periodic checkpoints, and the resolved
config the run used. The final policy (plus VecNormalize statistics when
enabled) lands in ``rl.artifact_dir``.

SAC is the preferred algorithm for the continuous battery-dispatch action
space (off-policy, sample efficient); PPO stays supported.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from microgrid_simulator.config import Settings
from microgrid_simulator.rl.callbacks import build_callbacks
from microgrid_simulator.rl.wrappers import make_training_env

LOGGER = logging.getLogger(__name__)

ALGOS: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "sac": SAC}


def train(
    settings: Settings,
    algo: str | None = None,
    total_timesteps: int | None = None,
    artifact_dir: Path | None = None,
    tensorboard_log: Path | None = None,
    seed: int | None = None,
    backend_name: str | None = None,
) -> Path:
    """Train an agent and save the policy artifact. Returns the artifact path.

    Explicit arguments override the ``rl:`` section of the config.
    """
    rl = settings.rl
    algo = (algo or rl.algo).lower()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo {algo!r}; choose from {sorted(ALGOS)}")
    total_timesteps = total_timesteps if total_timesteps is not None else rl.total_timesteps
    seed = seed if seed is not None else rl.seed
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else Path(rl.artifact_dir)

    run_dir = Path(rl.log_dir) / f"{algo}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(settings.model_dump(), sort_keys=False))

    n_envs = rl.n_envs or (4 if algo == "ppo" else 1)  # SAC is off-policy, 1 env is fine
    vec_env = make_training_env(
        settings,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=run_dir / "monitor",
        backend_name=backend_name,
    )
    eval_env = make_training_env(
        settings,
        n_envs=1,
        seed=seed + 10_000,
        monitor_dir=run_dir / "eval_monitor",
        training=False,
        backend_name=backend_name,
    )

    model_cls = cast(Any, ALGOS[algo])
    model = model_cls(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )
    LOGGER.info(
        "Training %s for %d timesteps (%d envs) -> %s",
        algo.upper(),
        total_timesteps,
        n_envs,
        run_dir,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=build_callbacks(settings, eval_env, run_dir, algo),
        progress_bar=False,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{algo}_microgrid"
    model.save(str(path))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(path) + "_vecnormalize.pkl")
    LOGGER.info("Saved policy -> %s.zip", path)
    return path.with_suffix(".zip")
