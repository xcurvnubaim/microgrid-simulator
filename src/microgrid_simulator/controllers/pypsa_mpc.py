"""PyPSAMPCController — rolling-horizon optimal dispatch baseline.

Every ``rolling_horizon_hours`` the controller re-solves a PyPSA
unit-commitment/dispatch over the next ``horizon_hours`` (perfect foresight of
demand and PV from the driving backend) and plays the plan back one tick at a
time. This is the classic MPC/expert baseline an RL policy should be compared
against — and, run offline, a per-episode dispatch benchmark.

Requires the ``ops`` extra and a
:class:`~microgrid_simulator.backends.pypsa_backend.PyPSAOperationalBackend`
driving the episode (that backend supplies both the forecast and the physics).
"""

from __future__ import annotations

import pandas as pd

from microgrid_simulator.backends.pypsa_backend import PyPSAOperationalBackend, plan_to_actions
from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState


class PyPSAMPCController(Controller):
    name = "pypsa_mpc"

    def __init__(
        self,
        settings: object = None,  # kept positional-compatible; settings come from the backend
        backend: PyPSAOperationalBackend | None = None,
        horizon_steps: int | None = None,
        replan_every: int | None = None,
    ) -> None:
        if backend is None:
            raise ValueError("PyPSAMPCController needs a PyPSAOperationalBackend instance")
        super().__init__(backend.settings)
        self.backend = backend
        self.horizon_steps = horizon_steps or backend.horizon_steps()
        self.replan_every = replan_every or backend.rolling_steps()
        self._actions: list[ControlAction] = []
        self._plans: list[pd.DataFrame] = []
        self._since_replan = 0

    def reset(self) -> None:
        self._actions = []
        self._plans = []
        self._since_replan = 0

    @property
    def last_plan(self) -> pd.DataFrame | None:
        return self._plans[-1] if self._plans else None

    def act(self, state: GridState) -> ControlAction:
        if not self._actions or self._since_replan >= self.replan_every:
            plan = self.backend.optimize_horizon(self.horizon_steps)
            self._plans.append(plan)
            self._actions = plan_to_actions(plan)
            self._since_replan = 0
        self._since_replan += 1
        action = self._actions.pop(0)
        action.ev_p_mw = self._ev_charge()
        return action
