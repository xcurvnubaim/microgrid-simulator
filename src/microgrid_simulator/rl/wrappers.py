"""Standard SB3 env wrapping: Monitor CSVs + vectorisation + VecNormalize."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from microgrid_simulator.config import Settings
from microgrid_simulator.rl.env import MicrogridEnv


def make_env_fn(settings: Settings, backend_name: str | None = None) -> Callable[[], MicrogridEnv]:
    def _init() -> MicrogridEnv:
        return MicrogridEnv(settings=settings, backend_name=backend_name)

    return _init


def make_training_env(
    settings: Settings,
    n_envs: int,
    seed: int,
    monitor_dir: str | Path | None = None,
    normalize: bool | None = None,
    training: bool = True,
    backend_name: str | None = None,
) -> VecEnv:
    """Vectorised env with per-episode Monitor CSVs and optional VecNormalize."""
    vec = make_vec_env(
        make_env_fn(settings, backend_name),
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir) if monitor_dir else None,
    )
    if normalize if normalize is not None else settings.rl.normalize:
        vec = VecNormalize(vec, training=training, norm_obs=True, norm_reward=training)
    return vec


def load_normalization(vec: VecEnv, stats_path: str | Path) -> VecEnv:
    """Restore saved VecNormalize statistics for evaluation (obs only)."""
    vec = VecNormalize.load(str(stats_path), vec)
    vec.training = False
    vec.norm_reward = False
    return vec
