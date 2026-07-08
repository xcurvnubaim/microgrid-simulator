"""Compatibility shims — ``grid/`` moved to ``core/``, ``components/``, ``backends/``.

* ``grid.battery``            -> ``components.battery``
* ``grid.diesel``             -> ``components.diesel``
* ``grid.demand_trace``       -> ``core.time_series``
* ``grid.pandapower_backend`` -> ``backends.pandapower_backend`` (+ ``core.types.GridState``)

``PandapowerBackend`` is resolved lazily so importing this package does not
require pandapower to be installed.
"""

from __future__ import annotations

from typing import Any

from microgrid_simulator.components.battery import BatteryModel
from microgrid_simulator.core.types import GridState

__all__ = ["BatteryModel", "GridState", "PandapowerBackend"]


def __getattr__(name: str) -> Any:
    if name == "PandapowerBackend":
        from microgrid_simulator.backends.pandapower_backend import PandapowerBackend

        return PandapowerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
