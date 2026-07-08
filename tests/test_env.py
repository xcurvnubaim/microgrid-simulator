"""Environment: Gymnasium contract, spaces, episode termination, determinism."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandapower")

from microgrid_simulator.config import Settings  # noqa: E402
from microgrid_simulator.env import MicrogridEnv  # noqa: E402


def _env() -> MicrogridEnv:
    return MicrogridEnv(settings=Settings())


def test_reset_returns_obs_and_info() -> None:
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert "timestamp" in info
    env.close()


def test_step_contract() -> None:
    env = _env()
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert reward <= 0.0
    for key in ("grid_import_mw", "soc", "soh"):
        assert key in info
    env.close()


def test_episode_truncates_at_horizon() -> None:
    env = _env()
    env.reset(seed=0)
    idle = np.zeros(env.action_dim, dtype=np.float32)
    truncated = False
    for _ in range(env.max_steps + 5):
        _, _, terminated, truncated, _ = env.step(idle)
        if terminated or truncated:
            break
    assert truncated
    env.close()


def test_action_bounds_are_respected() -> None:
    env = _env()
    env.reset(seed=0)
    # Max charge command should not blow SoC past the band.
    charge = np.zeros(env.action_dim, dtype=np.float32)
    charge[0] = 1.0
    for _ in range(20):
        env.step(charge)
    assert env.backend.battery.soc <= env.settings.battery.soc_max + 1e-6
    env.close()
