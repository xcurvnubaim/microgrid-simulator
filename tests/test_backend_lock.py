"""Temporary pandapower-only runtime lock."""

from __future__ import annotations

import pytest

pytest.importorskip("pandapower")

from microgrid_simulator.backends import (  # noqa: E402
    BACKEND_NAMES,
    create_backend,
    resolve_backend_name,
)
from microgrid_simulator.backends.pandapower_backend import PandapowerBackend  # noqa: E402
from microgrid_simulator.backends.simple_backend import SimpleBackend  # noqa: E402
from microgrid_simulator.config import Settings  # noqa: E402
from microgrid_simulator.rl.env import MicrogridEnv  # noqa: E402
from microgrid_simulator.ui.rollout import POLICIES  # noqa: E402


def test_settings_normalize_backend_and_solver_to_pandapower_ac() -> None:
    settings = Settings(backend={"name": "simple"}, topology={"solver": "balance"})

    assert settings.backend.name == "pandapower"
    assert settings.topology.solver == "ac"


def test_factory_ignores_backend_override() -> None:
    settings = Settings()
    settings.backend.name = "opendss"
    settings.topology.solver = "balance"

    assert BACKEND_NAMES == ("pandapower",)
    assert resolve_backend_name("opendss", settings) == "pandapower"
    backend = create_backend(settings, name="pypsa")
    try:
        assert settings.backend.name == "pandapower"
        assert settings.topology.solver == "ac"
        assert isinstance(backend, PandapowerBackend)
        assert backend.solver == "ac"
    finally:
        backend.close()


def test_environment_rejects_injected_backend() -> None:
    settings = Settings()
    injected = SimpleBackend(settings)

    with pytest.raises(ValueError, match="custom backend injection is disabled"):
        MicrogridEnv(settings=settings, backend=injected)


def test_dashboard_exposes_pypsa_rh_through_shared_rollout() -> None:
    """PyPSA-RH plans separately but executes through pandapower."""
    assert "pypsa_rh" in POLICIES
    assert "mpc" not in POLICIES
