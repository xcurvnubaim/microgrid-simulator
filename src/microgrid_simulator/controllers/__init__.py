"""Baseline and learned controllers, all speaking GridState -> ControlAction.

``PyPSAMPCController`` and ``RLPolicyController`` are imported lazily because
they need the optional ``ops`` / ``rl`` extras.
"""

from __future__ import annotations

from typing import Any

from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.controllers.deterministic import DeterministicController
from microgrid_simulator.controllers.greedy_peak_shaving import GreedyPeakShavingController
from microgrid_simulator.controllers.idle import IdleController
from microgrid_simulator.controllers.rule_based import RuleBasedController

__all__ = [
    "Controller",
    "DeterministicController",
    "GreedyPeakShavingController",
    "IdleController",
    "PyPSAMPCController",
    "RLPolicyController",
    "RuleBasedController",
]


def __getattr__(name: str) -> Any:
    if name == "PyPSAMPCController":
        from microgrid_simulator.controllers.pypsa_mpc import PyPSAMPCController

        return PyPSAMPCController
    if name == "RLPolicyController":
        from microgrid_simulator.controllers.rl_policy import RLPolicyController

        return RLPolicyController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
