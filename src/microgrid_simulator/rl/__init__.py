"""RL layer: Gymnasium env, SB3 training/evaluation, wrappers, callbacks.

Only ``rl.env`` is imported eagerly — training modules pull in SB3/torch, so
they are imported on demand (``from microgrid_simulator.rl.train import train``).
"""

from __future__ import annotations

from microgrid_simulator.rl.env import MicrogridEnv

__all__ = ["MicrogridEnv"]
