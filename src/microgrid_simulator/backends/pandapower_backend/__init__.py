"""PandapowerBackend — AC power-flow validation of the campus microgrid.

Owns the editable network, driving profiles (solar / load time-of-day curves),
the battery SoC integration, and one Newton-Raphson AC solve per tick. Use it
to validate voltages and line loading of a dispatch; RL-specific logic lives in
``rl/`` and never in here.

Topology is data-driven from ``Settings.buses`` and ``Settings.lines`` so the
same scenario can be edited in the dashboard like a light Cisco Packet Tracer
model: create buses, connect them, then place PV, storage, diesel, EV, and load
assets on those buses.

The package splits the backend by responsibility:

* ``network``  editable topology construction (buses, lines, assets)
* ``profiles`` solar/load driving curves and islanded feasibility clipping
* ``snapshot`` solved-state accounting into the shared ``GridState`` contract
* ``backend``  the :class:`PandapowerBackend` lifecycle (reset/step/solve)

Requires pandapower, which ships with the default install (``pip install -e .``).
"""

from microgrid_simulator.backends.pandapower_backend.backend import PandapowerBackend
from microgrid_simulator.backends.pandapower_backend.snapshot import V_MAX_PU, V_MIN_PU

__all__ = ["V_MAX_PU", "V_MIN_PU", "PandapowerBackend"]
