"""Model layer: reward shaping + RL agent glue.

Depends on ``grid/`` and ``config``. Never imports the top-level env upward.
"""

from __future__ import annotations

from microgrid_simulator.model.reward import RewardBreakdown, compute_reward

__all__ = ["RewardBreakdown", "compute_reward"]
