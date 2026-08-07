"""The backend contract every simulator implementation must satisfy.

The RL environment, controllers, replay, and API talk to *this* interface only;
which engine actually solves the tick (algebraic balance, pandapower AC power
flow, PyPSA, OpenDSS) is a config choice, not an import.

Implementations additionally expose these attributes, which the environment
and dashboard rely on:

* ``settings``       — the :class:`~microgrid_simulator.config.Settings` in force
* ``battery``        — the selected project or pymgrid battery state model
* ``diesel``         — the :class:`DieselModel` (runtime/starts counters)
* ``demand_trace``   — the loaded historical trace, or ``None``
* ``grid_connected`` — False when the topology has no utility intertie
* ``timestamp``      — simulated hours since episode start
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from microgrid_simulator.components.battery import BatteryLike
from microgrid_simulator.components.diesel import DieselModel
from microgrid_simulator.config import Settings
from microgrid_simulator.core.scenario import Scenario
from microgrid_simulator.core.time_series import DemandTrace
from microgrid_simulator.core.types import ControlAction, GridState


def clamp_action(action: ControlAction, settings: Settings) -> ControlAction:
    """Clamp a requested action to the device power limits in ``settings``.

    This is the *stateless* feasibility projection (power ratings, fractions in
    range). Stateful constraints — SoC headroom, ramp limits, min up/down
    lockouts — are enforced by the component models inside ``step()``.
    """
    b = settings.battery
    clamped = action.copy()
    clamped.battery_p_mw = max(-b.max_discharge_mw, min(b.max_charge_mw, action.battery_p_mw))
    ev_max = settings.ev.max_charge_mw
    clamped.ev_p_mw = [max(0.0, min(ev_max, float(p))) for p in action.ev_p_mw]
    clamped.pv_curtail = max(0.0, min(1.0, action.pv_curtail))
    if settings.diesel.enabled:
        clamped.diesel_setpoint_mw = max(
            0.0, min(settings.diesel.max_kw / 1000.0, action.diesel_setpoint_mw)
        )
    else:
        clamped.diesel_on = False
        clamped.diesel_setpoint_mw = 0.0
    return clamped


class MicrogridBackend(ABC):
    """Abstract simulation backend: one microgrid, stepped one tick at a time."""

    settings: Settings
    # Runtime attributes documented in the module docstring above; every
    # concrete backend exposes them at construction time (attribute or
    # read-only property).
    battery: BatteryLike
    diesel: DieselModel
    grid_connected: bool
    timestamp: float

    @property
    def demand_trace(self) -> DemandTrace | None:
        """The loaded historical demand trace, or ``None``."""
        return None

    @property
    def demand_is_real(self) -> bool:
        """Whether the active demand series comes from measured telemetry."""
        return self.demand_trace is not None

    @abstractmethod
    def reset(
        self,
        scenario: Scenario | None = None,
        seed: int | None = None,
        demand_window_mw: np.ndarray | None = None,
        pv_window_mw: np.ndarray | None = None,
    ) -> GridState:
        """Start a fresh episode and return the initial state.

        ``scenario`` may carry new settings (topology rebuild) and/or aligned
        demand/PV windows; ``None`` re-runs the constructed scenario.
        """

    @abstractmethod
    def step(self, action: ControlAction) -> GridState:
        """Advance one tick under ``action`` and return the solved state."""

    def validate_action(self, action: ControlAction) -> ControlAction:
        """Project a requested action onto the statically feasible set."""
        return clamp_action(action, self.settings)

    @abstractmethod
    def get_state(self) -> GridState:
        """Return the most recently solved state without advancing time."""

    @abstractmethod
    def close(self) -> None:
        """Release solver resources; the backend is unusable afterwards."""
