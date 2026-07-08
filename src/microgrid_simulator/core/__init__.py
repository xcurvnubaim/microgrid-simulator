"""Core layer: backend contract, shared types, scenario, and time series.

Depends only on ``config``. Backends, controllers, the RL env, and the digital
twin all build on this layer.
"""

from __future__ import annotations

from microgrid_simulator.core.backend import MicrogridBackend, clamp_action
from microgrid_simulator.core.scenario import Scenario
from microgrid_simulator.core.time_series import DemandTrace
from microgrid_simulator.core.types import (
    BackendResult,
    ConstraintViolation,
    ControlAction,
    DeviceStatus,
    GridState,
    RewardBreakdown,
)

__all__ = [
    "BackendResult",
    "ConstraintViolation",
    "ControlAction",
    "DemandTrace",
    "DeviceStatus",
    "GridState",
    "MicrogridBackend",
    "RewardBreakdown",
    "Scenario",
    "clamp_action",
]
