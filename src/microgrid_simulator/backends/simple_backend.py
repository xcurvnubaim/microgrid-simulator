"""SimpleBackend — fast algebraic energy balance, no power-flow solver.

This is the former ``solver: balance`` mode extracted into a standalone,
pure-python backend (no pandapower import), used for quick RL training and
smoke tests. It preserves:

* battery SoC band and charge/discharge power limits (via ``BatteryModel``),
* diesel min-load / ramp / min up-down lockouts (via ``DieselModel``),
* PV availability and curtailment (via ``PVFleetModel``),
* grid import/export limits (via ``GridIntertieModel``; unlimited by default),
* islanding: without a utility bus, local sources must cover demand and any
  shortfall is reported as unserved load.

It does **not** model voltages or line loading — those fields come back as a
flat 1.0 pu profile and an empty list. Use the pandapower/OpenDSS backends to
validate electrical feasibility of a dispatch.
"""

from __future__ import annotations

import numpy as np

from microgrid_simulator.components import (
    DemandModel,
    DieselModel,
    GridIntertieModel,
    PVFleetModel,
    create_battery_model,
)
from microgrid_simulator.config import Settings
from microgrid_simulator.core.backend import MicrogridBackend
from microgrid_simulator.core.scenario import Scenario
from microgrid_simulator.core.time_series import DemandTrace
from microgrid_simulator.core.types import (
    UNSERVED_TOLERANCE_MW,
    ConstraintViolation,
    ControlAction,
    GridState,
)

_GRID_ROLES = {"grid", "slack", "utility"}


class SimpleBackend(MicrogridBackend):
    """Lossless algebraic balance: import = demand + charging - PV - diesel."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._configure(settings)

    # -- lifecycle -----------------------------------------------------------
    def _configure(self, settings: Settings) -> None:
        self.settings = settings
        self.topo = settings.topology
        self.dt = float(self.topo.timestep_hours)
        self.timestamp = float(settings.episode.start_hour)
        self.battery = create_battery_model(settings.battery)
        self.diesel = DieselModel.from_cfg(settings.diesel)
        self.ev_soc = [settings.ev.soc_init for _ in range(self.topo.n_ev)]
        self.pv = PVFleetModel.from_settings(settings)
        self.demand = DemandModel.from_settings(settings)
        self.intertie = GridIntertieModel.from_settings(settings)
        buses = settings.buses
        self.n_buses = len(buses)
        self.grid_connected = any(b.role.lower() in _GRID_ROLES for b in buses)
        self._last_state = self._snapshot(ControlAction(), applied_battery_mw=0.0)

    def reset(
        self,
        scenario: Scenario | None = None,
        seed: int | None = None,
        demand_window_mw: np.ndarray | None = None,
        pv_window_mw: np.ndarray | None = None,
    ) -> GridState:
        if scenario is not None:
            self._configure(scenario.settings)
            if demand_window_mw is None:
                demand_window_mw = scenario.demand_window_mw
            if pv_window_mw is None:
                pv_window_mw = scenario.pv_window_mw
        self.timestamp = float(self.settings.episode.start_hour)
        self.battery.reset()
        self.diesel.reset()
        self.ev_soc = [self.settings.ev.soc_init for _ in range(self.topo.n_ev)]
        self.demand.reset(demand_window_mw)
        self.pv.reset(pv_window_mw)
        self._last_state = self._snapshot(ControlAction(), applied_battery_mw=0.0)
        return self._last_state

    def step(self, action: ControlAction) -> GridState:
        self.diesel.apply(action.diesel_on, action.diesel_setpoint_mw, self.dt)

        ev_max = self.settings.ev.max_charge_mw
        capacity = max(self.settings.ev.capacity_mwh, 1e-9)
        requested_ev = list(action.ev_p_mw)[: self.topo.n_ev]
        requested_ev += [0.0] * (self.topo.n_ev - len(requested_ev))
        ev_applied: list[float] = []
        for i, req in enumerate(requested_ev):
            rate = max(0.0, min(ev_max, float(req)))
            self.ev_soc[i] = min(1.0, self.ev_soc[i] + rate * self.dt / capacity)
            ev_applied.append(rate)

        self.timestamp += self.dt
        self.demand.advance()
        self.pv.advance()
        battery_request = self._feasible_battery_request(action, ev_applied)
        applied_bp, delta_soh = self.battery.apply(battery_request, self.dt)
        state = self._snapshot(action, applied_battery_mw=applied_bp, ev_p_mw=ev_applied)
        state.delta_soh = delta_soh
        self._last_state = state
        return state

    def get_state(self) -> GridState:
        return self._last_state

    def close(self) -> None:
        pass

    # -- convenience accessors kept for the env / dashboard -------------------
    @property
    def demand_trace(self) -> DemandTrace | None:
        return self.demand.demand_trace

    @property
    def demand_is_real(self) -> bool:
        return self.demand.demand_is_real

    @property
    def load_bases(self) -> list[tuple[float, float]]:
        return self.demand.load_bases

    # -- balance solve ---------------------------------------------------------
    def _snapshot(
        self,
        action: ControlAction,
        applied_battery_mw: float,
        ev_p_mw: list[float] | None = None,
    ) -> GridState:
        ev_p_mw = ev_p_mw if ev_p_mw is not None else [0.0] * self.topo.n_ev
        per_load = self.demand.per_load_mw(self.timestamp)
        static_mw = sum(p for p, _ in per_load)
        ev_mw = sum(ev_p_mw)
        demand_mw = static_mw + ev_mw
        pv_available = self.pv.available_mw(self.timestamp)
        pv_per_array = self.pv.per_array_mw(self.timestamp, action.pv_curtail)
        pv_used = sum(pv_per_array)

        violations: list[ConstraintViolation] = []
        if self.grid_connected:
            grid_import = demand_mw + applied_battery_mw - pv_used - self.diesel.p_mw
            flow, violation = self.intertie.clamp(grid_import)
            unserved = 0.0
            if violation is not None:
                violations.append(violation)
                if violation.kind == "import_limit":
                    # The contract caps what the utility can supply; the excess
                    # demand goes unserved this tick.
                    unserved = grid_import - flow
                else:
                    # Export beyond the contract is spilled PV first.
                    excess = -grid_import - (self.intertie.max_export_mw or 0.0)
                    spill = min(pv_used, excess)
                    pv_used -= spill
                    scale = pv_used / (pv_used + spill) if (pv_used + spill) > 1e-12 else 0.0
                    pv_per_array = [p * scale for p in pv_per_array]
                grid_import = flow
            served = demand_mw - unserved
        else:
            # An island has no export path. After the battery request is
            # realized, curtail PV that cannot serve load or battery charging.
            # Any remaining overgeneration is therefore attributable to an
            # inflexible source such as diesel minimum stable output.
            pv_absorption_limit = max(0.0, demand_mw + applied_battery_mw - self.diesel.p_mw)
            if pv_used > pv_absorption_limit:
                pre_spill = pv_used
                pv_used = pv_absorption_limit
                scale = pv_used / pre_spill if pre_spill > 1e-12 else 0.0
                pv_per_array = [p * scale for p in pv_per_array]
            # Islanded: local sources must cover demand; the shortfall is shed.
            # Served stays in [0, demand]: shedding cannot exceed the demand
            # itself, even if the commanded battery charge outstrips generation.
            grid_import = 0.0
            supply_for_load = pv_used + self.diesel.p_mw - applied_battery_mw
            served = min(demand_mw, max(0.0, supply_for_load))
            unserved = demand_mw - served

        if unserved > UNSERVED_TOLERANCE_MW:
            violations.append(
                ConstraintViolation(
                    kind="unserved",
                    device="loads",
                    magnitude=unserved,
                    message=f"{unserved * 1000.0:.1f} kW of demand unserved",
                )
            )

        state = GridState(
            v_bus=[1.0] * self.n_buses,
            p_load=[p for p, _ in per_load],
            p_gen=pv_per_array,
            soc=[self.battery.soc],
            soh=[self.battery.soh],
            ev_soc=list(self.ev_soc),
            line_loading=[],
            grid_import_mw=grid_import,
            pv_available_mw=pv_available,
            pv_used_mw=pv_used,
            load_demand_mw=demand_mw,
            load_served_mw=served,
            unserved_mw=unserved,
            islanded=not self.grid_connected,
            battery_p_mw=applied_battery_mw,
            diesel_p_mw=self.diesel.p_mw,
            diesel_on=self.diesel.is_on,
            demand_is_real=self.demand_is_real,
            pv_is_real=self.pv.pv_is_real,
            solver_ok=True,
            timestamp=self.timestamp,
            violations=violations,
        )
        self._electrical_snapshot(state)
        return state

    def _feasible_battery_request(
        self, action: ControlAction, ev_p_mw: list[float] | None = None
    ) -> float:
        """Clamp islanded battery power to current load balance.

        Battery charging is flexible demand and can only use local surplus.
        Battery discharging is flexible supply and should not exceed the
        residual campus demand unless the model exposes an explicit sink.
        """
        requested = float(action.battery_p_mw)
        if self.grid_connected:
            return requested

        ev_mw = sum(ev_p_mw or [])
        static_mw = sum(p for p, _ in self.demand.per_load_mw(self.timestamp))
        demand_mw = static_mw + ev_mw
        pv_mw = sum(self.pv.per_array_mw(self.timestamp, action.pv_curtail))
        local_without_battery_mw = pv_mw + self.diesel.p_mw

        if requested >= 0.0:
            surplus_mw = max(0.0, local_without_battery_mw - demand_mw)
            return min(requested, surplus_mw)

        residual_mw = max(0.0, demand_mw - local_without_battery_mw)
        return max(requested, -residual_mw)

    def _electrical_snapshot(self, state: GridState) -> None:
        """Hook for subclasses that add an electrical validation solve
        (e.g. the OpenDSS backend fills ``v_bus`` from a snapshot solve)."""
