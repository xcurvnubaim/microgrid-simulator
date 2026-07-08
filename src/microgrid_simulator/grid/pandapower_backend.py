"""Compatibility shim — moved to ``microgrid_simulator.backends.pandapower_backend``.

``GridState`` now lives in ``microgrid_simulator.core.types`` and is importable
from here without pandapower installed; the backend class itself is resolved
lazily because it does require pandapower.
"""

from __future__ import annotations

from typing import Any

from microgrid_simulator.core.types import GridState

__all__ = ["GridState", "PandapowerBackend"]


def __getattr__(name: str) -> Any:
    if name == "PandapowerBackend":
        from microgrid_simulator.backends.pandapower_backend import PandapowerBackend

        return PandapowerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
