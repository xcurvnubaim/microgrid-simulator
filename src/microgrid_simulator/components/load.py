"""Aggregate demand model — synthetic time-of-day curve or a historical window.

Owns the demand trace and the per-episode window position so every backend
computes the exact same demand numbers: when a real window is installed the
configured static loads are scaled proportionally so their sum matches the
historical total; otherwise the legacy sinusoid drives the yaml base MW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sin

import numpy as np

from microgrid_simulator.config import Settings
from microgrid_simulator.core.time_series import DemandTrace


def load_factor(timestamp_hours: float) -> float:
    """Legacy synthetic campus curve: daytime bump on a 0.90 base."""
    hour = timestamp_hours % 24.0
    return 0.90 + 0.18 * max(0.0, sin(pi * (hour - 7.0) / 13.0))


@dataclass
class DemandModel:
    """Static loads plus the (optional) historical total-demand window."""

    load_bases: list[tuple[float, float]]  # (p_mw, q_mvar) per configured load
    demand_trace: DemandTrace | None = None
    window_mw: np.ndarray | None = None
    window_pos: int = 0
    _default_bases: list[tuple[float, float]] = field(default_factory=list, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> DemandModel:
        bases = [(ld.p_mw, ld.q_mvar) for ld in settings.loads[: settings.topology.n_load]]
        trace = DemandTrace.from_file(settings.demand, settings.topology.timestep_hours)
        return cls(load_bases=bases, demand_trace=trace)

    @property
    def demand_is_real(self) -> bool:
        return self.window_mw is not None

    def reset(self, window_mw: np.ndarray | None = None) -> None:
        self.window_mw = window_mw
        self.window_pos = 0

    def advance(self) -> None:
        self.window_pos += 1

    def total_demand_mw(self, timestamp_hours: float) -> float:
        """Target total static demand for this tick (EV charging not included)."""
        base_total = sum(p for p, _ in self.load_bases)
        if self.window_mw is not None and len(self.window_mw) > 0:
            pos = min(self.window_pos, len(self.window_mw) - 1)
            return float(self.window_mw[pos])
        return base_total * load_factor(timestamp_hours)

    def per_load_mw(self, timestamp_hours: float) -> list[tuple[float, float]]:
        """(p_mw, q_mvar) per static load, scaled so the p sum hits the target
        total while relative shares stay untouched."""
        base_total = sum(p for p, _ in self.load_bases)
        lf = self.total_demand_mw(timestamp_hours) / base_total if base_total > 1e-12 else 0.0
        return [(p * lf, q * lf) for p, q in self.load_bases]
