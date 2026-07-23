"""Controller contract — a policy mapping grid state to a physical action.

Controllers see only :class:`GridState` and emit :class:`ControlAction` in
physical units (MW); they never touch a backend or an RL framework directly.
The env's ``encode_action`` bridges them onto the normalised Gymnasium action
space when needed (e.g. for the dashboard rollout).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from microgrid_simulator.config import Settings
from microgrid_simulator.core.types import ControlAction, GridState

# The dashboard's legacy policies drove EVs with a normalised 0.0 action, which
# decodes to half the maximum charge rate. Baseline controllers keep that EV
# behaviour so their results stay comparable with historical rollouts.
LEGACY_EV_FRACTION = 0.5


class Controller(ABC):
    """One EMS policy. Stateless controllers only need ``act``."""

    name: str = "controller"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def reset(self) -> None:
        """Called at episode start; override for stateful controllers."""
        return None

    @abstractmethod
    def act(self, state: GridState) -> ControlAction:
        """Decide the next tick's action from the last solved state."""

    # -- shared helpers ------------------------------------------------------
    def _ev_charge(self) -> list[float]:
        rate = LEGACY_EV_FRACTION * self.settings.ev.max_charge_mw
        return [rate] * self.settings.topology.n_ev

    def _diesel_max_mw(self) -> float:
        return self.settings.diesel.max_kw / 1000.0 if self.settings.diesel.enabled else 0.0
