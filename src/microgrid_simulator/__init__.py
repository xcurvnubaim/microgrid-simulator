"""microgrid_simulator — digital twin + RL energy-management-system framework.

A modular research framework for a campus microgrid:

* ``core/``         backend contract, shared types, scenario, time series
* ``components/``   battery, diesel, PV, load, grid-intertie device models
* ``backends/``     simple (fast RL), pandapower (AC validation),
                    PyPSA (operational scheduling), OpenDSS (skeleton)
* ``controllers/``  idle / rule-based / greedy / PyPSA-MPC / RL baselines
* ``digital_twin/`` measured-data ingestion, alignment, calibration, replay,
                    validation metrics
* ``rl/``           Gymnasium env + SB3 training/evaluation
* ``api/``          FastAPI server for the React dashboard

Registers the env with gymnasium so ``gym.make("MicrogridEnv-v0")`` works.
"""

from __future__ import annotations

from gymnasium.envs.registration import register

__version__ = "0.2.0"

register(
    id="MicrogridEnv-v0",
    entry_point="microgrid_simulator.rl.env:MicrogridEnv",
)

__all__ = ["__version__"]
