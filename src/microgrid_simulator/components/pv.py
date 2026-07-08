"""PV fleet — availability profile plus curtailment.

Availability is the sinusoidal daylight curve the simulator has always used
(0 at night, peak near solar noon); replace ``availability_factor`` with a
measured irradiance series through the digital-twin replay when calibrating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sin

from microgrid_simulator.config import Settings


def solar_factor(timestamp_hours: float) -> float:
    """Sinusoidal daylight curve, 0 at night, peak near solar noon."""
    hour = timestamp_hours % 24.0
    return max(0.0, sin(pi * (hour - 6.0) / 12.0))


@dataclass
class PVFleetModel:
    """All PV arrays in the scenario, driven by one shared availability factor."""

    bases_mw: list[float] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: Settings) -> PVFleetModel:
        arrays = settings.pv_arrays[: settings.topology.n_pv]
        return cls(bases_mw=[pv.p_mw for pv in arrays])

    def availability_factor(self, timestamp_hours: float) -> float:
        return solar_factor(timestamp_hours)

    def available_mw(self, timestamp_hours: float) -> float:
        return self.availability_factor(timestamp_hours) * sum(self.bases_mw)

    def per_array_mw(self, timestamp_hours: float, curtail: float) -> list[float]:
        """Realised output per array after curtailment (fraction in [0, 1])."""
        scale = self.availability_factor(timestamp_hours) * (1.0 - max(0.0, min(1.0, curtail)))
        return [base * scale for base in self.bases_mw]

    def used_mw(self, timestamp_hours: float, curtail: float) -> float:
        return sum(self.per_array_mw(timestamp_hours, curtail))
