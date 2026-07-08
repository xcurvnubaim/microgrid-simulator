"""Diesel generator — discrete on/off + continuous setpoint (plan decision 1).

Action schema (matches Research Brainstorm §1):
    diesel_on          ∈ {0, 1}
    diesel_setpoint_kw ∈ [0, diesel_max_kw]

The setpoint is continuous — a real genset governor modulates fuel injection to
follow any load — but the realised output is shaped by genset physics:

* **Minimum stable load** (``min_kw``, ~30 % of nameplate): sustained operation
  below it causes wet stacking, so while on the output is clamped up to it.
* **Ramp limit** (``ramp_kw_per_min``): output moves toward the setpoint at a
  bounded rate. Diesels ramp fast (tens of %/min), so this only binds at fine
  timesteps.
* **Minimum up/down time** (``min_up_time_min`` / ``min_down_time_min``): the
  genset controller ignores an off command until the engine has run long
  enough, and an on command until it has cooled down — real controllers lock
  out rapid cycling because starts dominate engine wear.
* **Start transient**: on the start tick the unit synchronises and picks up at
  most min load plus one tick of ramp, not an arbitrary jump to full power.

No real nameplate rating exists yet; ``max_kw`` defaults to ~150 kW (roughly
half the 352.8 kW observed historical peak) until the actual generator spec is
available. Carbon accounting lives in the reward
(``diesel_setpoint_kw * carbon_per_kwh_diesel * dt``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from microgrid_simulator.config import DieselCfg


@dataclass
class DieselModel:
    """Dispatchable generator with an on/off state and a clamped setpoint."""

    cfg: DieselCfg
    is_on: bool = False
    p_mw: float = 0.0
    runtime_hours: float = 0.0
    starts: int = 0
    # Time spent in the current on/off state; starts at inf so a fresh genset
    # is past any cool-down lockout and may start on the first tick.
    hours_in_state: float = math.inf

    @classmethod
    def from_cfg(cls, cfg: DieselCfg) -> DieselModel:
        return cls(cfg=cfg)

    def reset(self) -> None:
        self.is_on = False
        self.p_mw = 0.0
        self.runtime_hours = 0.0
        self.starts = 0
        self.hours_in_state = math.inf

    @property
    def max_mw(self) -> float:
        return self.cfg.max_kw / 1000.0

    @property
    def min_mw(self) -> float:
        return min(self.cfg.min_kw, self.cfg.max_kw) / 1000.0

    @property
    def ramp_mw_per_hour(self) -> float:
        if self.cfg.ramp_kw_per_min <= 0:
            return math.inf  # 0 disables the ramp limit
        return self.cfg.ramp_kw_per_min * 60.0 / 1000.0

    def apply(self, on: bool, setpoint_mw: float, dt_hours: float) -> float:
        """Advance one tick; returns the realised output in MW.

        The realised on/off state and output may differ from the command:
        min up/down lockouts can override the switch, and the min-load /
        ramp clamps can override the setpoint.
        """
        if not self.cfg.enabled:
            on = False
        elif self.is_on and not on and self.hours_in_state < self.cfg.min_up_time_min / 60.0:
            on = True  # min up time: too soon to shut down
        elif not self.is_on and on and self.hours_in_state < self.cfg.min_down_time_min / 60.0:
            on = False  # min down time: still cooling down

        starting = on and not self.is_on
        if starting:
            self.starts += 1
        self.hours_in_state = dt_hours if on != self.is_on else self.hours_in_state + dt_hours
        self.is_on = bool(on)

        if not self.is_on:
            self.p_mw = 0.0
            return 0.0

        ramp_mw = self.ramp_mw_per_hour * dt_hours
        target = max(self.min_mw, min(self.max_mw, float(setpoint_mw)))
        if starting:
            # Synchronise and pick up load: min load plus one tick of ramp.
            self.p_mw = min(target, min(self.max_mw, self.min_mw + ramp_mw))
        else:
            self.p_mw = max(self.p_mw - ramp_mw, min(self.p_mw + ramp_mw, target))
        self.runtime_hours += dt_hours
        return self.p_mw
