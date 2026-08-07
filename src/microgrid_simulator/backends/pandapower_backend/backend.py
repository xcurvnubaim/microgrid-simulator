"""PandapowerBackend — AC power-flow validation of the campus microgrid.

Owns the editable network, driving profiles (solar / load time-of-day curves),
the battery SoC integration, and one Newton-Raphson AC solve per tick. Use it
to validate voltages and line loading of a dispatch; RL-specific logic lives in
``rl/`` and never in here.

The implementation is split into focused mixins by responsibility:

* ``network``  editable topology construction (buses, lines, assets)
* ``profiles`` solar/load driving curves and islanded feasibility clipping
* ``snapshot`` solved-state accounting into the shared ``GridState`` contract

Requires pandapower, which ships with the default install (``pip install -e .``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandapower as pp

from microgrid_simulator.backends.pandapower_backend.network import _NetworkBuilderMixin
from microgrid_simulator.backends.pandapower_backend.profiles import _ProfilesMixin
from microgrid_simulator.backends.pandapower_backend.snapshot import _SnapshotMixin
from microgrid_simulator.components import DieselModel, create_battery_model
from microgrid_simulator.config import BusCfg, Settings
from microgrid_simulator.core.backend import MicrogridBackend
from microgrid_simulator.core.scenario import Scenario
from microgrid_simulator.core.time_series import DemandTrace
from microgrid_simulator.core.types import ControlAction, GridState

LOGGER = logging.getLogger(__name__)


class PandapowerBackend(
    _NetworkBuilderMixin, _ProfilesMixin, _SnapshotMixin, MicrogridBackend
):
    """Campus microgrid using pandapower AC power flow."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._configure(settings)

    def _configure(self, settings: Settings) -> None:
        self.settings = settings
        self.topo = settings.topology
        self.dt = float(self.topo.timestep_hours)
        self.timestamp = float(settings.episode.start_hour)
        self.solver = str(getattr(self.topo, "solver", "ac")).lower()
        self._warm = False  # previous AC solution available for warm start

        self.battery = create_battery_model(settings.battery)
        self.diesel = DieselModel.from_cfg(settings.diesel)
        self.ev_soc = [settings.ev.soc_init for _ in range(self.topo.n_ev)]

        # Real demand trace. None -> synthetic curve.
        self._demand_trace = DemandTrace.from_file(settings.demand, self.dt)
        self._demand_window: np.ndarray | None = None
        self._pv_window: np.ndarray | None = None
        self._window_pos: int = 0

        self.bus_lookup: dict[int, int] = {}
        self.bus_cfg_by_id: dict[int, BusCfg] = {}
        self.static_load_indices: list[int] = []
        self.ev_load_indices: list[int] = []
        self.pv_indices: list[int] = []
        self.storage_indices: list[int] = []
        self.diesel_index: int | None = None
        self.dump_load_index: int | None = None
        # ``grid_connected`` is False when the topology has no utility intertie
        # (no bus with role grid/slack/utility). Islanded operation must serve
        # demand from PV + diesel + battery alone; any shortfall is a blackout.
        self.grid_connected: bool = True
        self.pv_bases_mw = [pv.p_mw for pv in settings.pv_arrays[: self.topo.n_pv]]
        self.load_bases = [(load.p_mw, load.q_mvar) for load in settings.loads[: self.topo.n_load]]
        self.net: Any = None
        self._profile_static_p_mw: list[float] = []
        self._profile_static_q_mvar: list[float] = []
        self._profile_ev_p_mw: list[float] = []
        self._served_load_fraction: float = 1.0
        self._build_network()
        self._last_state: GridState = self._snapshot(
            battery_p_mw=0.0, pv_curtail=0.0, solver_ok=True
        )

    # -- lifecycle ---------------------------------------------------------
    def reset(
        self,
        scenario: Scenario | None = None,
        seed: int | None = None,
        demand_window_mw: np.ndarray | None = None,
        pv_window_mw: np.ndarray | None = None,
    ) -> GridState:
        """Reset physics; optionally install new settings and/or a demand window.

        ``demand_window_mw`` holds one total-demand value per tick (index 0 is
        the reset tick). When ``None``, the synthetic sinusoid drives the loads.
        """
        if scenario is not None:
            self._configure(scenario.settings)
            if demand_window_mw is None:
                demand_window_mw = scenario.demand_window_mw
            if pv_window_mw is None:
                pv_window_mw = scenario.pv_window_mw
        self.timestamp = float(self.settings.episode.start_hour)
        self._warm = False
        self.battery.reset()
        self.diesel.reset()
        self.ev_soc = [self.settings.ev.soc_init for _ in range(self.topo.n_ev)]
        self._demand_window = demand_window_mw
        self._pv_window = pv_window_mw
        self._window_pos = 0
        self._apply_profiles(battery_p_mw=0.0, ev_p_mw=[0.0] * self.topo.n_ev, curtail_pv=0.0)
        ok = self._run_power_flow()
        self._last_state = self._snapshot(battery_p_mw=0.0, pv_curtail=0.0, solver_ok=ok)
        return self._last_state

    def step(self, action: ControlAction) -> GridState:
        """Advance one tick: dispatch constraints -> profiles -> power flow."""
        self.diesel.apply(action.diesel_on, action.diesel_setpoint_mw, self.dt)

        requested_ev = list(action.ev_p_mw)[: self.topo.n_ev]
        requested_ev += [0.0] * (self.topo.n_ev - len(requested_ev))
        ev_applied = []
        for i, req in enumerate(requested_ev):
            rate = max(0.0, min(self.settings.ev.max_charge_mw, float(req)))
            self.ev_soc[i] = min(
                1.0, self.ev_soc[i] + rate * self.dt / max(self.settings.ev.capacity_mwh, 1e-9)
            )
            ev_applied.append(rate)

        self.timestamp += self.dt
        self._window_pos += 1
        battery_request = self._feasible_battery_request(action, ev_applied)
        applied_bp, delta_soh = self.battery.apply(battery_request, self.dt)
        self._apply_profiles(
            battery_p_mw=applied_bp, ev_p_mw=ev_applied, curtail_pv=action.pv_curtail
        )
        ok = self._run_power_flow()
        snap = self._snapshot(battery_p_mw=applied_bp, pv_curtail=action.pv_curtail, solver_ok=ok)
        snap.delta_soh = delta_soh
        self._last_state = snap
        return snap

    def get_state(self) -> GridState:
        return self._last_state

    def close(self) -> None:
        self.net = None

    @property
    def demand_trace(self) -> DemandTrace | None:
        return self._demand_trace

    def _run_power_flow(self) -> bool:
        if self.solver == "balance":
            return True
        # Warm-starting from the previous tick's solution roughly halves the
        # Newton-Raphson time; fall back to a flat start if it diverges.
        inits = ("results", "flat") if self._warm else ("flat",)
        for init in inits:
            try:
                pp.runpp(self.net, init=init, max_iteration=30, numba=False)
                self._warm = True
                return True
            except Exception as exc:  # pragma: no cover - solver dependent
                LOGGER.warning("Pandapower power flow failed (init=%s): %s", init, exc)
        self._warm = False
        return False
