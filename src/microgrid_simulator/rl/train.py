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
from microgrid_simulator.forecast import ForecastCache, StrictCachedForecastClient
from microgrid_simulator.rl.callbacks import build_callbacks
from microgrid_simulator.rl.wrappers import make_training_env

LOGGER = logging.getLogger(__name__)

ALGOS: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "sac": SAC}


def _build_sac_kwargs(settings: Settings) -> dict[str, Any]:
    """Build SAC keyword arguments from config, filtering out None values.

    This lets the YAML override individual hyperparameters while falling back
    to SB3 defaults for anything left unset.
    """
    cfg = settings.rl.sac
    kwargs: dict[str, Any] = {}
    for field in ("learning_rate", "buffer_size", "learning_starts",
                  "batch_size", "gamma", "tau", "train_freq", "gradient_steps",
                  "n_critics", "use_sde"):
        val = getattr(cfg, field, None)
        if val is not None:
            kwargs[field] = val
    if cfg.ent_coef is not None:
        kwargs["ent_coef"] = cfg.ent_coef
    if cfg.target_entropy is not None:
        kwargs["target_entropy"] = cfg.target_entropy
    if cfg.net_arch is not None:
        kwargs["policy_kwargs"] = {"net_arch": cfg.net_arch}
    return kwargs


def _override_sac_params(model: SAC, settings: Settings) -> SAC:
    """Apply config-level SAC hyperparameters to an already loaded model.

    This is used for fine-tuning: the loaded model keeps its old constructor
    values, so we override the ones the user explicitly set.
    """
    cfg = settings.rl.sac
    if cfg.learning_rate is not None:
        from stable_baselines3.common.utils import constant_fn
        model.lr_schedule = constant_fn(cfg.learning_rate)
    if cfg.gamma is not None:
        model.gamma = cfg.gamma
    if cfg.tau is not None:
        model.tau = cfg.tau
    if cfg.batch_size is not None:
        model.batch_size = cfg.batch_size
    if cfg.gradient_steps is not None:
        model.gradient_steps = cfg.gradient_steps
    if cfg.train_freq is not None:
        from stable_baselines3.common.type_aliases import TrainFreq
        model.train_freq = TrainFreq(cfg.train_freq, "step")
    if cfg.learning_starts is not None:
        model.learning_starts = cfg.learning_starts
    return model


def _resolve_device(device: str) -> str:
    """Resolve the RL device string, honouring PyTorch CUDA availability."""
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train(
    settings: Settings,
    algo: str | None = None,
    total_timesteps: int | None = None,
    artifact_dir: Path | None = None,
    tensorboard_log: Path | None = None,
    seed: int | None = None,
    backend_name: str | None = None,
    init_artifact: Path | None = None,
    init_vecnorm: Path | None = None,
    init_replay_buffer: Path | None = None,
    save_replay_buffer: bool = False,
    reset_num_timesteps: bool = False,
    n_envs: int | None = None,
    stage_name: str | None = None,
    run_id: str | None = None,
    eval_env_nominal: bool = False,
) -> Path:
    """Train an agent and save the policy artifact. Returns the artifact path.

    Explicit arguments override the ``rl:`` section of the config.

    When ``init_artifact`` is provided the run *continues* from an existing
    policy (fine-tuning/resume): the model is loaded, the optional
    ``init_vecnorm`` statistics are restored in training mode, the optional SAC
    replay buffer is reloaded, and timestep counters continue from the parent
    unless ``reset_num_timesteps`` is True. The parent artifact is never
    overwritten — a new artifact is written under ``artifact_dir``.
    """
    rl = settings.rl
    if rl.train_split != "train" or rl.eval_split != "val":
        raise ValueError("training requires train_split='train' and eval_split='val'")
    forecast_mode = getattr(rl, "forecast_mode", "cached")
    if forecast_mode not in {"cached", "none"}:
        raise ValueError("training forecast_mode must be 'cached' or 'none'")
    if rl.forecast_representation not in {"raw", "summary"}:
        raise ValueError("forecast_representation must be 'raw' or 'summary'")
    if forecast_mode == "cached" and (
        not settings.forecast.enabled or not settings.forecast.strict_cache
    ):
        raise ValueError("cached training requires forecast.enabled and forecast.strict_cache")
    algo = (algo or rl.algo).lower()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo {algo!r}; choose from {sorted(ALGOS)}")
    if init_artifact is not None and ALGOS[algo].__name__.lower() != algo:
        pass  # init artifact algorithm check happens on load below
    total_timesteps = total_timesteps if total_timesteps is not None else rl.total_timesteps
    seed = seed if seed is not None else rl.seed
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else Path(rl.artifact_dir)

    forecast_client = None
    eval_forecast_client = None
    if settings.forecast.enabled and settings.forecast.strict_cache and forecast_mode == "cached":
        if not settings.forecast.cache_path or not settings.forecast.manifest_path:
            raise ValueError("strict cached forecasting requires cache_path and manifest_path")
        source_id = settings.forecast.source_id or settings.scenario.name
        cache = ForecastCache.load(
            settings.forecast.cache_path,
            settings.forecast.manifest_path,
            expected_source_id=source_id,
            action_interval_hours=settings.topology.timestep_hours,
        )
        forecast_client = StrictCachedForecastClient(settings.forecast, cache)
        # The eval environment samples the val split, which lives in a distinct
        # cache. Load it separately so the eval callback does not query the train
        # cache for February windows.
        if settings.forecast.val_cache_path and settings.forecast.val_manifest_path:
            eval_cache = ForecastCache.load(
                settings.forecast.val_cache_path,
                settings.forecast.val_manifest_path,
                expected_source_id=source_id,
                action_interval_hours=settings.topology.timestep_hours,
            )
            eval_forecast_client = StrictCachedForecastClient(settings.forecast, eval_cache)
        else:
            eval_forecast_client = forecast_client

    ts = run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(rl.log_dir) / f"{algo}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump = settings.model_dump()
    dump["_lineage"] = {
        "init_artifact": str(init_artifact) if init_artifact else None,
        "init_vecnorm": str(init_vecnorm) if init_vecnorm else None,
        "init_replay_buffer": str(init_replay_buffer) if init_replay_buffer else None,
        "stage_name": stage_name,
        "reset_num_timesteps": bool(reset_num_timesteps),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dump, sort_keys=False))

    n_envs = n_envs if n_envs is not None else (rl.n_envs or (4 if algo == "ppo" else 1))
    vec_env = make_training_env(
        settings,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=run_dir / "monitor",
        backend_name=backend_name,
        split=rl.train_split,
        forecast_client=forecast_client,
    )
    eval_env = make_training_env(
        settings,
        n_envs=1,
        seed=seed + 10_000,
        monitor_dir=run_dir / "eval_monitor",
        training=False,
        backend_name=backend_name,
        split=rl.eval_split,
        forecast_client=eval_forecast_client,
    )

    model_cls = cast(Any, ALGOS[algo])
    device = _resolve_device(rl.device)
    if init_artifact is not None:
        load_cls = SAC if algo == "sac" else PPO
        model = load_cls.load(str(Path(init_artifact)), env=vec_env, device=device)
        if init_vecnorm is not None:
            from microgrid_simulator.rl.wrappers import (
                load_normalization,
                load_training_normalization,
            )

            vec_env = load_training_normalization(vec_env, init_vecnorm)
            eval_env = load_normalization(eval_env, init_vecnorm)
            model = load_cls.load(str(Path(init_artifact)), env=vec_env, device=device)
        if algo == "sac" and init_replay_buffer is not None:
            model.load_replay_buffer(str(init_replay_buffer))
        if algo == "sac":
            model = _override_sac_params(cast(SAC, model), settings)
        LOGGER.info(
            "Continuing from %s (fine-tune/resume), %d steps", init_artifact, total_timesteps
        )
    else:
        sac_kwargs = _build_sac_kwargs(settings) if algo == "sac" else {}
        model = model_cls(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=seed,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            device=device,
            **sac_kwargs,
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
        callback=build_callbacks(settings, eval_env, run_dir, algo, stage=stage_name),
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{stage_name}" if stage_name else ""
    path = artifact_dir / f"{algo}_microgrid{suffix}"
    model.save(str(path))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(path) + "_vecnormalize.pkl")
    if save_replay_buffer and algo == "sac":
        model.save_replay_buffer(str(path) + "_replay_buffer.pkl")

    # Overwrite the final model with the best validation model if available
    best_model_path = run_dir / "best_model" / "best_model.zip"
    if best_model_path.is_file():
        import shutil
        shutil.copy(best_model_path, path.with_suffix(".zip"))
        LOGGER.info("Deployed best validation checkpoint from %s to %s.zip", best_model_path, path)

    LOGGER.info("Saved policy -> %s.zip", path)
    return path.with_suffix(".zip")
