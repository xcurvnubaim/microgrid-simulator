"""Deprecated compatibility imports for the former PyPSA MPC terminology.

Use :mod:`microgrid_simulator.controllers.pypsa_rolling_horizon` and
``PyPSARollingHorizonController`` in new code. PyPSA supplies the optimization
model; the surrounding rolling-horizon orchestration belongs to this project.
"""

from microgrid_simulator.controllers.pypsa_rolling_horizon import (
    PyPSARollingHorizonController,
    SnapshotProvider,
    _align_forecast_to_steps,
    _align_snapshot_to_steps,
    _resample_hourly_to_quarter,
)

# Compatibility only. Do not use this name in new code or research-facing output.
PyPSAMPCController = PyPSARollingHorizonController

__all__ = [
    "PyPSAMPCController",
    "PyPSARollingHorizonController",
    "SnapshotProvider",
    "_align_forecast_to_steps",
    "_align_snapshot_to_steps",
    "_resample_hourly_to_quarter",
]
