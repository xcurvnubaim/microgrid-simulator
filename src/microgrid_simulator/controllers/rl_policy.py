"""RLPolicyController — a trained SB3 policy behind the controller interface.

Lets a learned policy be evaluated with exactly the same rollout/replay code
as the classical baselines. Rebuilds the env's observation from the grid state
and decodes the normalised action back to physical units; if VecNormalize
statistics were saved next to the artifact they are applied to observations.

Requires the ``rl`` extra.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from microgrid_simulator.config import Settings
from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState
from microgrid_simulator.rl.env import build_observation, decode_action


class RLPolicyController(Controller):
    name = "rl"

    def __init__(
        self,
        settings: Settings,
        artifact: str | Path,
        algo: str | None = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__(settings)
        from microgrid_simulator.rl.train import ALGOS

        algo = (algo or settings.rl.algo).lower()
        self.model: Any = ALGOS[algo].load(str(artifact))
        self.deterministic = deterministic
        self.n_ev = settings.topology.n_ev
        self.diesel_enabled = settings.diesel.enabled

        self._obs_rms: Any = None
        stats = Path(str(artifact)).with_suffix("").as_posix() + "_vecnormalize.pkl"
        if Path(stats).exists():
            with open(stats, "rb") as fh:
                self._obs_rms = pickle.load(fh).obs_rms

    def act(self, state: GridState) -> ControlAction:
        obs = build_observation(state, self.diesel_enabled)
        if self._obs_rms is not None:
            obs = (obs - self._obs_rms.mean) / (self._obs_rms.var + 1e-8) ** 0.5
            obs = obs.clip(-10.0, 10.0).astype("float32")
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return decode_action(action, self.settings, self.n_ev, self.diesel_enabled)
