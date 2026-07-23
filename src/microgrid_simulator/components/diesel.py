"""Diesel generator — discrete on/off + continuous setpoint (plan decision 1).

Action schema (matches Research Brainstorm §1):
    diesel_on          ∈ {0, 1}
    diesel_setpoint_kw ∈ [0, diesel_max_kw]

The setpoint is continuous — a real genset governor modulates fuel injection to
follow any load — but the realised output is shaped by genset physics:

* **Minimum stable load** (``min_kw``, ~30 % of nameplate): sustained operation
  below it causes wet stacking, so while on the output is clamped up to it.
* **Ramp limit** (``ramp_kw_per_min``): output moves toward the setpoint at a
  bounded rate.
* **Start sequence**: crank + synchronise for ``start_delay_min`` at zero
  output, then the breaker closes onto the minimum stable load (gensets accept
  a block step) and the governor ramps toward the setpoint.
* **Stop sequence**: the governor soft-unloads down to min stable load at the
  ramp limit, then the breaker opens — output never teleports to zero from
  high load, but the drop from min load is a genuine breaker-opening step.
* **Minimum up/down time** (``min_up_time_min`` / ``min_down_time_min``): the
  genset controller ignores an off command until the engine has run long
  enough, and an on command until it has cooled down — real controllers lock
  out rapid cycling because starts dominate engine wear.

Because simulation ticks (15 min by default) are much longer than these
transients, the trajectory is integrated *within* the tick and ``apply``
returns the tick-average power — the energy-correct value for the power
balance and carbon accounting — while ``p_end_mw`` keeps the instantaneous
end-of-tick output so ramping stays continuous across ticks.

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
    p_mw: float = 0.0  # tick-average output (energy-correct for the power balance)
    p_end_mw: float = 0.0  # instantaneous output at tick end (ramp continuity)
    runtime_hours: float = 0.0
    starts: int = 0
    # Time spent in the current on/off state; starts at inf so a fresh genset
    # is past any cool-down lockout and may start on the first tick.
    hours_in_state: float = math.inf

    @classmethod
    def from_cfg(cls, cfg: DieselCfg) -> DieselModel:
        return cls(cfg=cfg, is_on=cfg.initial_on)

    def reset(self) -> None:
        self.is_on = self.cfg.initial_on
        self.p_mw = 0.0
        self.p_end_mw = 0.0
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

    def _ramp_toward(self, p0: float, target: float, duration_h: float) -> tuple[float, float]:
        """Ramp-limited move from ``p0`` toward ``target`` over ``duration_h``.

        Returns ``(energy_mwh, p_end_mw)`` — the integral of the piecewise-
        linear trajectory and the instantaneous output when time runs out.
        """
        if duration_h <= 0.0:
            return 0.0, p0
        rate = self.ramp_mw_per_hour
        if math.isinf(rate):
            return target * duration_h, target
        reach_h = abs(target - p0) / rate
        if reach_h >= duration_h:
            p_end = p0 + math.copysign(rate * duration_h, target - p0)
            return (p0 + p_end) / 2.0 * duration_h, p_end
        ramp_mwh = (p0 + target) / 2.0 * reach_h
        return ramp_mwh + target * (duration_h - reach_h), target

    def apply(self, on: bool, setpoint_mw: float, dt_hours: float) -> float:
        """Advance one tick; returns the tick-average output in MW.

        The realised on/off state and output may differ from the command:
        min up/down lockouts can override the switch, and the start/stop
        sequences, min-load clamp, and ramp limit shape the within-tick
        trajectory that the returned average integrates.
        """
        if not self.cfg.enabled:
            on = False
        elif self.is_on and not on and self.hours_in_state < self.cfg.min_up_time_min / 60.0:
            on = True  # min up time: too soon to shut down
        elif not self.is_on and on and self.hours_in_state < self.cfg.min_down_time_min / 60.0:
            on = False  # min down time: still cooling down

        starting = on and not self.is_on
        stopping = self.is_on and not on
        if starting:
            self.starts += 1
        self.hours_in_state = dt_hours if on != self.is_on else self.hours_in_state + dt_hours
        self.is_on = bool(on)

        if not self.is_on:
            energy_mwh = 0.0
            if stopping and self.p_end_mw > 0.0:
                # Soft unload to min stable load at the ramp limit, then the
                # breaker opens; the tail of the tick produces nothing. If the
                # unload can't finish inside the tick the breaker trips at the
                # tick end regardless.
                rate = self.ramp_mw_per_hour
                unload_h = (
                    0.0
                    if math.isinf(rate)
                    else min(dt_hours, max(0.0, self.p_end_mw - self.min_mw) / rate)
                )
                p_open = self.p_end_mw - (0.0 if math.isinf(rate) else rate * unload_h)
                energy_mwh = (self.p_end_mw + p_open) / 2.0 * unload_h
            self.p_end_mw = 0.0
            self.p_mw = energy_mwh / dt_hours if dt_hours > 0 else 0.0
            return self.p_mw

        target = max(self.min_mw, min(self.max_mw, float(setpoint_mw)))
        if starting:
            # Crank + synchronise at zero output, breaker closes onto min
            # stable load, then ramp toward the setpoint.
            sync_h = min(dt_hours, max(0.0, self.cfg.start_delay_min) / 60.0)
            energy_mwh, p_end = self._ramp_toward(self.min_mw, target, dt_hours - sync_h)
            if dt_hours - sync_h <= 0.0:
                p_end = 0.0  # still synchronising at tick end
        else:
            # p_end below min only if the whole start tick was consumed by the
            # sync delay; the breaker then closes onto min load now.
            p0 = max(self.p_end_mw, self.min_mw)
            energy_mwh, p_end = self._ramp_toward(p0, target, dt_hours)
        self.p_end_mw = p_end
        self.p_mw = energy_mwh / dt_hours if dt_hours > 0 else 0.0
        self.runtime_hours += dt_hours
        return self.p_mw
