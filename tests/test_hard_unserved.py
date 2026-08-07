"""Hard unserved-load constraint: islanded blackout ends the episode with a penalty.

Covers the RL-side enforcement of the islanded scenario's "load service is
non-negotiable" contract: when ``rl.hard_unserved`` is enabled, any tick whose
unserved demand exceeds the solver-noise tolerance terminates the episode
immediately and applies the one-shot ``hard_unserved_penalty``, on top of the
soft ``w_unserved`` reward term that is still computed for reporting.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandapower")

from microgrid_simulator.config import Settings  # noqa: E402
from microgrid_simulator.rl.env import MicrogridEnv  # noqa: E402


def _islanded_settings(*, hard_unserved: bool, penalty: float = 1000.0) -> Settings:
    """Islanded scenario with insufficient supply, so an idle action sheds load.

    No grid bus, diesel disabled, zero PV: demand exceeds supply, producing a
    deterministic blackout on the first step.
    """
    raw = Settings().model_dump()
    raw["topology"]["n_ev"] = 0
    raw["diesel"]["enabled"] = False
    # Remove the grid bus so the backend is islanded.
    raw["buses"] = [b for b in raw["buses"] if b["role"] != "grid"]
    # Remove lines that referenced the removed grid bus (bus 0).
    raw["lines"] = [
        line
        for line in raw["lines"]
        if line["from_bus"] != 0 and line["to_bus"] != 0
    ]
    # Zero PV availability so nothing covers the demand.
    for pv in raw["pv_arrays"]:
        pv["p_mw"] = 0.0
    # Tiny battery that starts nearly empty and cannot discharge, so demand is
    # shed rather than served from storage.
    raw["battery"]["capacity_mwh"] = 0.001
    raw["battery"]["soc_init"] = 0.10
    raw["battery"]["soc_min"] = 0.10
    raw["battery"]["max_discharge_mw"] = 0.0
    raw["rl"]["hard_unserved"] = hard_unserved
    raw["rl"]["hard_unserved_penalty"] = penalty
    return Settings(**raw)


def test_hard_unserved_terminates_on_blackout_and_applies_penalty() -> None:
    env = MicrogridEnv(settings=_islanded_settings(hard_unserved=True, penalty=1000.0))
    env.reset(seed=0)
    idle = np.zeros(env.action_dim, dtype=np.float32)

    _, reward_soft, terminated, truncated, info = env.step(idle)

    assert info["hard_unserved_enabled"] is True
    assert info["hard_unserved_triggered"] is True
    assert info["unserved_mw"] > 0.0
    assert terminated is True
    assert truncated is False
    # Reward includes both the soft w_unserved term and the one-shot hard penalty;
    # with the default weights it must be clearly dominated by the 1000 penalty.
    assert reward_soft <= -1000.0
    env.close()


def test_soft_unserved_does_not_terminate_or_apply_penalty() -> None:
    env = MicrogridEnv(settings=_islanded_settings(hard_unserved=False))
    env.reset(seed=0)
    idle = np.zeros(env.action_dim, dtype=np.float32)

    _, reward_soft, terminated, truncated, info = env.step(idle)

    assert info["hard_unserved_enabled"] is False
    assert info["hard_unserved_triggered"] is False
    assert info["unserved_mw"] > 0.0
    # The episode is not ended by unserved load when the hard constraint is off;
    # it may still end for other reasons (e.g. solver), but the reward must not
    # include the 1000 one-shot penalty.
    assert reward_soft > -1000.0
    env.close()


def test_hard_unserved_respects_tolerance_threshold() -> None:
    # Set tolerance high enough (e.g. 1.0 MW) so the ~0.33 MW demand shortfall
    # does NOT trigger termination.
    s = _islanded_settings(hard_unserved=True, penalty=1000.0)
    s.rl.hard_unserved_tol_mw = 1.0
    env = MicrogridEnv(settings=s)
    env.reset(seed=0)
    idle = np.zeros(env.action_dim, dtype=np.float32)

    _, reward, terminated, _, info = env.step(idle)

    assert info["hard_unserved_enabled"] is True
    assert info["hard_unserved_triggered"] is False
    assert terminated is False
    assert reward > -1000.0
    env.close()
