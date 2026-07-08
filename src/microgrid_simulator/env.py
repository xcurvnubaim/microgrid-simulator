"""Compatibility shim — the environment moved to ``microgrid_simulator.rl.env``.

Kept so ``from microgrid_simulator.env import MicrogridEnv`` (and the
``MicrogridEnv-v0`` gymnasium entry point) keep working.
"""

from __future__ import annotations

from microgrid_simulator.rl.env import MicrogridEnv

__all__ = ["MicrogridEnv"]
