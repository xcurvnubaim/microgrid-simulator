"""Backend registry — resolve a config name to a simulator implementation.

Heavy engines (pandapower, PyPSA, OpenDSS) are imported lazily so the default
install only ever loads what the chosen backend actually needs.
"""

from __future__ import annotations

import importlib.util
import logging

from microgrid_simulator.config import Settings
from microgrid_simulator.core.backend import MicrogridBackend

LOGGER = logging.getLogger(__name__)

BACKEND_NAMES = ("simple", "pandapower", "pypsa", "opendss")


def resolve_backend_name(name: str, settings: Settings) -> str:
    """Resolve ``auto`` using the legacy ``topology.solver`` switch.

    ``solver: ac`` historically meant pandapower Newton-Raphson and
    ``solver: balance`` the algebraic engine, so old configs keep their exact
    behaviour; if pandapower is not installed we fall back to the simple
    backend with a warning instead of failing the default install.
    """
    name = (name or "auto").lower()
    if name != "auto":
        return name
    if settings.topology.solver == "balance":
        return "simple"
    if importlib.util.find_spec("pandapower") is not None:
        return "pandapower"
    LOGGER.warning(
        "backend.name=auto wants pandapower (topology.solver=ac) but it is not "
        "installed — falling back to the simple backend. Reinstall project "
        "dependencies with `pip install -e .`."
    )
    return "simple"


def create_backend(settings: Settings, name: str | None = None) -> MicrogridBackend:
    """Instantiate the backend selected by ``name`` (default: config's choice)."""
    resolved = resolve_backend_name(name or settings.backend.name, settings)
    if resolved == "simple":
        from microgrid_simulator.backends.simple_backend import SimpleBackend

        return SimpleBackend(settings)
    if resolved == "pandapower":
        from microgrid_simulator.backends.pandapower_backend import PandapowerBackend

        return PandapowerBackend(settings)
    if resolved == "pypsa":
        from microgrid_simulator.backends.pypsa_backend import PyPSAOperationalBackend

        return PyPSAOperationalBackend(settings)
    if resolved == "opendss":
        from microgrid_simulator.backends.opendss_backend import OpenDSSBackend

        return OpenDSSBackend(settings)
    raise ValueError(f"Unknown backend {resolved!r}; choose from {BACKEND_NAMES}")


__all__ = ["BACKEND_NAMES", "create_backend", "resolve_backend_name"]
