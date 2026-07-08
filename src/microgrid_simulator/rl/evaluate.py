"""Roll out a trained policy and report/export evaluation metrics.

If VecNormalize statistics were saved next to the artifact
(``<artifact>_vecnormalize.pkl``) they are restored so the policy sees the
observation scaling it was trained with.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from microgrid_simulator.config import Settings
from microgrid_simulator.rl.env import MicrogridEnv
from microgrid_simulator.rl.train import ALGOS
from microgrid_simulator.rl.wrappers import load_normalization

LOGGER = logging.getLogger(__name__)


def evaluate(
    settings: Settings,
    artifact: Path,
    algo: str | None = None,
    episodes: int = 3,
    csv_path: Path | None = None,
    backend_name: str | None = None,
) -> dict[str, float]:
    """Roll out a trained policy; returns mean reward + operating averages.

    When ``csv_path`` is given, one row per step (episode, step, reward, grid
    import, SoC, diesel, unserved, ...) is exported for offline analysis.
    """
    algo = (algo or settings.rl.algo).lower()
    model = cast(Any, ALGOS[algo]).load(str(artifact))

    inner = MicrogridEnv(settings=settings, backend_name=backend_name)
    env: Any = DummyVecEnv([lambda: inner])
    stats = Path(str(artifact)).with_suffix("").as_posix() + "_vecnormalize.pkl"
    if Path(stats).exists():
        env = load_normalization(env, stats)
        LOGGER.info("Restored VecNormalize stats from %s", stats)

    rows: list[dict[str, Any]] = []
    ep_rewards: list[float] = []
    imports: list[float] = []
    unserved: list[float] = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        step = 0
        step_imports: list[float] = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            info = infos[0]
            total += float(reward[0])
            done = bool(dones[0])
            step += 1
            step_imports.append(max(0.0, info["grid_import_mw"]))
            state = inner._last_state  # noqa: SLF001 - evaluation diagnostics
            rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "reward": float(reward[0]),
                    "grid_import_mw": info["grid_import_mw"],
                    "soc": info["soc"],
                    "soh": info["soh"],
                    "diesel_p_mw": info["diesel_p_mw"],
                    "pv_wasted_mw": info["pv_wasted_mw"],
                    "unserved_mw": state.unserved_mw,
                    "load_demand_mw": state.load_demand_mw,
                }
            )
        ep_rewards.append(total)
        imports.append(float(np.mean(step_imports)) if step_imports else 0.0)
        unserved.append(float(sum(r["unserved_mw"] for r in rows if r["episode"] == episode)))

    env.close()

    if csv_path is not None and rows:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote per-step evaluation rows -> %s", csv_path)

    return {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "mean_grid_import_mw": float(np.mean(imports)),
        "mean_unserved_mwh_per_episode": float(
            np.mean(unserved) * settings.topology.timestep_hours
        ),
    }
