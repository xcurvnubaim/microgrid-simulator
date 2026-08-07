"""Grid-state snapshot and electrical accounting for :class:`PandapowerBackend`.

Translates one solved pandapower tick into the shared ``GridState`` contract,
including the islanded served/unserved split, the explicit dump-load path for
non-exportable surplus, and voltage/line-loading constraint violations.
"""

from __future__ import annotations

from typing import Any

from microgrid_simulator.components import BatteryLike, DieselModel
from microgrid_simulator.core.types import ConstraintViolation, GridState

V_MIN_PU = 0.95
V_MAX_PU = 1.05


class _SnapshotMixin:
    """Solved-state accounting split out of the backend lifecycle.

    Also reads ``demand_is_real``, ``pv_is_real``, and
    ``current_pv_available_mw`` from :class:`_ProfilesMixin`; the concrete
    backend inherits from both, so those resolve through normal attribute
    lookup at runtime.
    """

    # Shared state owned by PandapowerBackend._configure.
    timestamp: float
    solver: str
    net: Any
    diesel: DieselModel
    battery: BatteryLike
    grid_connected: bool
    bus_lookup: dict[int, int]
    ev_soc: list[float]
    pv_indices: list[int]
    static_load_indices: list[int]
    ev_load_indices: list[int]
    dump_load_index: int | None
    _profile_static_p_mw: list[float]
    _profile_ev_p_mw: list[float]
    _served_load_fraction: float

    def _electrical_violations(
        self, v_bus: list[float], line_loading: list[float]
    ) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []
        for i, v in enumerate(v_bus):
            if v < V_MIN_PU or v > V_MAX_PU:
                limit = V_MIN_PU if v < V_MIN_PU else V_MAX_PU
                violations.append(
                    ConstraintViolation(
                        kind="voltage",
                        device=f"bus_{i}",
                        magnitude=abs(v - limit),
                        limit=limit,
                        message=f"bus {i} at {v:.4f} pu outside [{V_MIN_PU}, {V_MAX_PU}]",
                    )
                )
        for i, loading in enumerate(line_loading):
            if loading > 100.0:
                violations.append(
                    ConstraintViolation(
                        kind="line_loading",
                        device=f"line_{i}",
                        magnitude=loading - 100.0,
                        limit=100.0,
                        message=f"line {i} at {loading:.1f}% loading",
                    )
                )
        return violations

    def _snapshot(self, battery_p_mw: float, pv_curtail: float, solver_ok: bool) -> GridState:
        net = self.net
        balance = self.solver == "balance"
        if not solver_ok or (not balance and not len(net.res_ext_grid)):
            return GridState(
                solver_ok=False,
                timestamp=self.timestamp,
                soc=[self.battery.soc],
                soh=[self.battery.soh],
                ev_soc=list(self.ev_soc),
                battery_p_mw=battery_p_mw,
                diesel_p_mw=self.diesel.p_mw,
                diesel_on=self.diesel.is_on,
                diesel_starts=self.diesel.starts,
                demand_is_real=self.demand_is_real,
                pv_is_real=self.pv_is_real,
                islanded=not self.grid_connected,
            )

        pv_available = self.current_pv_available_mw()
        pv_used = float(net.sgen.loc[self.pv_indices, "p_mw"].sum()) if self.pv_indices else 0.0
        static_demand = (
            sum(self._profile_static_p_mw)
            if self._profile_static_p_mw
            else float(net.load.loc[self.static_load_indices, "p_mw"].sum())
        )
        ev_demand = (
            sum(self._profile_ev_p_mw)
            if self._profile_ev_p_mw
            else float(net.load.loc[self.ev_load_indices, "p_mw"].sum())
        )
        demand = static_demand + ev_demand

        if self.grid_connected:
            # Utility intertie balances the network; all load is served. The
            # balance engine closes the books algebraically (lossless) instead
            # of reading the slack injection from an AC solution.
            if balance:
                grid_import = demand + battery_p_mw - pv_used - self.diesel.p_mw
            else:
                grid_import = float(net.res_ext_grid.p_mw.sum())
            served = demand
            unserved = 0.0
        else:
            # Islanded: local sources must cover demand. Net power available to
            # load = PV used + diesel output - battery power (battery charging is
            # an extra sink, discharging is a source). Anything demand exceeds
            # that by is shed — an islanded blackout / brownout. Served stays in
            # [0, demand]: shedding cannot exceed the demand itself, even if the
            # commanded battery charge outstrips local generation.
            grid_import = 0.0
            served = demand * self._served_load_fraction
            unserved = demand - served

        dump_load_mw = (
            float(net.load.at[self.dump_load_index, "p_mw"])
            if self.dump_load_index is not None
            else 0.0
        )
        if balance:
            network_loss_mw = 0.0
            reference_balance_mw = 0.0
        else:
            network_loss_mw = 0.0
            for result_name in ("res_line", "res_trafo", "res_trafo3w"):
                result = getattr(net, result_name, None)
                if result is not None and "pl_mw" in result:
                    network_loss_mw += float(result.pl_mw.sum())
            # Connected ext_grid power is already ``grid_import_mw``. For an
            # island this is only the disclosed numerical grid-forming balance.
            reference_balance_mw = (
                0.0 if self.grid_connected else float(net.res_ext_grid.p_mw.sum())
            )
        hidden_absorption_mw = max(0.0, -reference_balance_mw)
        excess_generation_mw = dump_load_mw + hidden_absorption_mw

        v_bus = [1.0] * len(self.bus_lookup) if balance else [float(v) for v in net.res_bus.vm_pu]
        line_loading = (
            []
            if balance
            else [float(x) for x in net.res_line.loading_percent]
            if len(net.res_line)
            else []
        )

        return GridState(
            v_bus=v_bus,
            p_load=[
                float(p)
                for p in (
                    self._profile_static_p_mw
                    if self._profile_static_p_mw
                    else net.load.loc[self.static_load_indices, "p_mw"]
                )
            ],
            p_gen=[float(p) for p in net.sgen.loc[self.pv_indices, "p_mw"]],
            soc=[self.battery.soc],
            soh=[self.battery.soh],
            ev_soc=list(self.ev_soc),
            line_loading=line_loading,
            grid_import_mw=grid_import,
            pv_available_mw=pv_available,
            pv_used_mw=pv_used,
            load_demand_mw=demand,
            load_served_mw=served,
            unserved_mw=unserved,
            islanded=not self.grid_connected,
            battery_p_mw=battery_p_mw,
            diesel_p_mw=self.diesel.p_mw,
            diesel_on=self.diesel.is_on,
            diesel_starts=self.diesel.starts,
            excess_generation_mw=excess_generation_mw,
            dump_load_mw=dump_load_mw,
            network_loss_mw=max(0.0, network_loss_mw),
            reference_balance_mw=reference_balance_mw,
            demand_is_real=self.demand_is_real,
            pv_is_real=self.pv_is_real,
            solver_ok=True,
            timestamp=self.timestamp,
            violations=self._electrical_violations(v_bus, line_loading),
        )
