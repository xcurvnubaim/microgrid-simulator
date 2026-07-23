"""PV fleet — availability profile plus curtailment.

Availability is the sinusoidal daylight curve the simulator has always used
(0 at night, peak near solar noon); replace ``availability_factor`` with a
measured irradiance series through the digital-twin replay when calibrating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sin

import numpy as np

from microgrid_simulator.config import Settings


def solar_factor(timestamp_hours: float) -> float:
    """Sinusoidal daylight curve, 0 at night, peak near solar noon."""
    hour = timestamp_hours % 24.0
    return max(0.0, sin(pi * (hour - 6.0) / 12.0))


@dataclass
class PVFleetModel:
    """All PV arrays in the scenario, driven by one shared availability factor."""

    bases_mw: list[float] = field(default_factory=list)
    window_mw: np.ndarray | None = None
    window_pos: int = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> PVFleetModel:
        arrays = settings.pv_arrays[: settings.topology.n_pv]
        return cls(bases_mw=[pv.p_mw for pv in arrays])

    def availability_factor(self, timestamp_hours: float) -> float:
        return solar_factor(timestamp_hours)

    @property
    def pv_is_real(self) -> bool:
        return self.window_mw is not None

    def reset(self, window_mw: np.ndarray | None = None) -> None:
        self.window_mw = window_mw
        self.window_pos = 0

    def advance(self) -> None:
        self.window_pos += 1

    def available_mw(self, timestamp_hours: float) -> float:
        if self.window_mw is not None and len(self.window_mw) > 0:
            pos = min(self.window_pos, len(self.window_mw) - 1)
            return float(self.window_mw[pos])
        return self.availability_factor(timestamp_hours) * sum(self.bases_mw)

    def per_array_mw(self, timestamp_hours: float, curtail: float) -> list[float]:
        """Realised output per array after curtailment (fraction in [0, 1])."""
        base_total = sum(self.bases_mw)
        available = self.available_mw(timestamp_hours)
        used = available * (1.0 - max(0.0, min(1.0, curtail)))
        if base_total <= 1e-12:
            return [0.0] * len(self.bases_mw)
        return [used * base / base_total for base in self.bases_mw]

    def used_mw(self, timestamp_hours: float, curtail: float) -> float:
        return sum(self.per_array_mw(timestamp_hours, curtail))
