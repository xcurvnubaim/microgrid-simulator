"""Scenario — everything a backend needs to (re)build and drive an episode.

A scenario is the typed :class:`~microgrid_simulator.config.Settings` (topology,
devices, limits) plus per-episode data such as the demand window drawn from a
historical trace. Backends receive it in ``reset()`` so the same backend object
can be reused across episodes and scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from microgrid_simulator.config import Settings


@dataclass
class Scenario:
    """One reproducible episode setup."""

    settings: Settings
    # Total-demand trajectory for the episode, one MW value per tick (index 0 is
    # the reset tick). ``None`` -> the synthetic time-of-day curve drives loads.
    demand_window_mw: np.ndarray | None = None
    name: str = "default"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        settings = Settings.from_yaml(path)
        return cls(settings=settings, name=settings.scenario.name)

    @classmethod
    def from_settings(cls, settings: Settings) -> Scenario:
        return cls(settings=settings, name=settings.scenario.name)
