"""Backend registry for the temporarily pandapower-only runtime."""

from __future__ import annotations

from microgrid_simulator.config import RUNTIME_PHYSICS_ENGINE, Settings
from microgrid_simulator.core.backend import MicrogridBackend

BACKEND_NAMES = (RUNTIME_PHYSICS_ENGINE,)


def resolve_backend_name(name: str, settings: Settings) -> str:
    """Return the hardcoded runtime engine, ignoring config and overrides."""
    del name, settings
    return RUNTIME_PHYSICS_ENGINE


def create_backend(settings: Settings, name: str | None = None) -> MicrogridBackend:
    """Instantiate pandapower regardless of config or caller override."""
    resolve_backend_name(name or settings.backend.name, settings)
    settings.backend.name = RUNTIME_PHYSICS_ENGINE
    settings.topology.solver = "ac"
    from microgrid_simulator.backends.pandapower_backend import PandapowerBackend

    return PandapowerBackend(settings)


__all__ = [
    "BACKEND_NAMES",
    "RUNTIME_PHYSICS_ENGINE",
    "create_backend",
    "resolve_backend_name",
]
