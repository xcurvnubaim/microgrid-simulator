"""IdleController — the do-nothing lower bound.

No battery action, no PV curtailment, diesel off. EV chargers keep the legacy
half-rate draw so idle episodes stay comparable with historical rollouts.
"""

from __future__ import annotations

from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState


class IdleController(Controller):
    name = "idle"

    def act(self, state: GridState) -> ControlAction:
        return ControlAction(ev_p_mw=self._ev_charge())
