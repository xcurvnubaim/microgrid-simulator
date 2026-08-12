"""Training callbacks: periodic evaluation + checkpointing.

EvalCallback keeps the best model and writes ``evaluations.npz`` under the run
directory; CheckpointCallback snapshots the policy (and VecNormalize stats)
so long runs can be resumed or inspected mid-flight.

Each callback's frequency is divided by the number of parallel environments so
the actual wall-clock interval matches the configured step count regardless of
vectorisation.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

from microgrid_simulator.config import Settings


def build_callbacks(
    settings: Settings,
    eval_env: VecEnv,
    run_dir: Path,
    algo: str,
    stage: str | None = None,
    n_envs: int = 1,
) -> list[BaseCallback]:
    rl = settings.rl
    n_envs = rl.n_envs or n_envs
    prefix = f"{algo}_microgrid"
    if stage:
        prefix = f"{prefix}-{stage}"
    return [
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir),
            eval_freq=max(1, rl.eval_freq // n_envs),
            n_eval_episodes=rl.eval_episodes,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(1, rl.checkpoint_freq // n_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=prefix,
            save_vecnormalize=True,
        ),
    ]
