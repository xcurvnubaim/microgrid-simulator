"""PandapowerBackend — AC power-flow validation of the campus microgrid.

Owns the editable network, driving profiles (solar / load time-of-day curves),
the battery SoC integration, and one Newton-Raphson AC solve per tick. Use it
to validate voltages and line loading of a dispatch; RL-specific logic lives in
``rl/`` and never in here.

Topology is data-driven from ``Settings.buses`` and ``Settings.lines`` so the
same scenario can be edited in the dashboard like a light Cisco Packet Tracer
model: create buses, connect them, then place PV, storage, diesel, EV, and load
assets on those buses.

Requires pandapower, which ships with the default install (``pip install -e .``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandapower as pp

from microgrid_simulator.components import DieselModel, create_battery_model
from microgrid_simulator.config import BusCfg, LineCfg, Settings
from microgrid_simulator.core.backend import MicrogridBackend
from microgrid_simulator.core.scenario import Scenario
from microgrid_simulator.core.time_series import DemandTrace
from microgrid_simulator.core.types import ConstraintViolation, ControlAction, GridState

LOGGER = logging.getLogger(__name__)

V_MIN_PU = 0.95
V_MAX_PU = 1.05


class PandapowerBackend(MicrogridBackend):
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
        self.demand_trace = DemandTrace.from_file(settings.demand, self.dt)
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

    # -- network construction ---------------------------------------------
    def _build_network(self) -> None:
        net = pp.create_empty_network(sn_mva=1.0)
        self.bus_lookup = {}
        self.bus_cfg_by_id = {}

        buses = self.settings.buses
        for bus in buses:
            self.bus_lookup[int(bus.id)] = int(pp.create_bus(net, vn_kv=bus.vn_kv, name=bus.name))
            self.bus_cfg_by_id[int(bus.id)] = bus

        if not self.bus_lookup:
            raise ValueError("topology must contain at least one bus")

        ref_id, self.grid_connected = self._reference_bus_id()
        self.reference_bus_id = ref_id
        # pandapower needs a voltage reference (slack) to solve. When grid-connected
        # this is the utility intertie (effectively infinite). When islanded it sits
        # on a grid-forming local source (diesel/battery) purely as the numerical
        # reference. Its active balance is exposed separately (normally AC
        # losses after the explicit dump load is applied), while reported
        # utility grid_import remains zero.
        ref_name = self.bus_cfg_by_id[ref_id].name
        pp.create_ext_grid(
            net,
            bus=self._bus(ref_id),
            vm_pu=1.0,
            name=(
                f"{ref_name} utility intertie"
                if self.grid_connected
                else f"{ref_name} grid-forming reference (islanded)"
            ),
        )

        for link in self.settings.lines:
            if int(link.from_bus) not in self.bus_lookup or int(link.to_bus) not in self.bus_lookup:
                LOGGER.warning("Skipping link %s because its bus id is missing", link.name)
                continue
            self._connection(net, link)

        self.pv_indices = []
        for i, pv in enumerate(self.settings.pv_arrays[: self.topo.n_pv]):
            self.pv_indices.append(
                pp.create_sgen(
                    net,
                    bus=self._bus(pv.bus, fallback_role="pv"),
                    p_mw=pv.p_mw,
                    q_mvar=0.0,
                    name=pv.name or f"PV array {i}",
                    type="PV",
                )
            )

        self.diesel_index = None
        if self.settings.diesel.enabled:
            self.diesel_index = int(
                pp.create_sgen(
                    net,
                    bus=self._bus(self.settings.diesel.bus, fallback_role="main"),
                    p_mw=0.0,
                    q_mvar=0.0,
                    name="Diesel genset",
                    type="diesel",
                )
            )

        # An island has no export path. Surplus from an inflexible source such
        # as diesel minimum output therefore needs an explicit sink; otherwise
        # pandapower's numerical ext_grid silently absorbs it.
        self.dump_load_index = None
        if not self.grid_connected:
            self.dump_load_index = int(
                pp.create_load(
                    net,
                    bus=self._bus(ref_id),
                    p_mw=0.0,
                    q_mvar=0.0,
                    name="Explicit excess-generation dump load",
                    type="dump_load",
                )
            )

        self.storage_indices = []
        if self.topo.n_storage > 0:
            self.storage_indices = [
                pp.create_storage(
                    net,
                    bus=self._bus(self.settings.battery.bus, fallback_role="battery"),
                    p_mw=0.0,
                    q_mvar=0.0,
                    max_e_mwh=self.settings.battery.capacity_mwh,
                    soc_percent=self.settings.battery.soc_init * 100.0,
                    name="Campus battery",
                    type="battery",
                )
            ]

        self.static_load_indices = []
        for i, load in enumerate(self.settings.loads[: self.topo.n_load]):
            self.static_load_indices.append(
                pp.create_load(
                    net,
                    bus=self._bus(load.bus, fallback_role="load"),
                    p_mw=load.p_mw,
                    q_mvar=load.q_mvar,
                    name=load.name or f"Load {i}",
                )
            )

        self.ev_load_indices = [
            pp.create_load(
                net,
                bus=self._bus(self.settings.ev.bus, fallback_role="ev"),
                p_mw=0.0,
                q_mvar=0.0,
                name=f"EV charger {i}",
            )
            for i in range(self.topo.n_ev)
        ]
        self.net = net

    def _find_slack_bus_id(self) -> int:
        for bus_id, bus in self.bus_cfg_by_id.items():
            if bus.role.lower() in {"grid", "slack", "utility"}:
                return bus_id
        return min(self.bus_lookup)

    def _reference_bus_id(self) -> tuple[int, bool]:
        """Pick the power-flow reference bus and whether it is a real utility tie.

        Returns ``(bus_id, grid_connected)``. A utility bus (role grid/slack/
        utility) always wins and marks the network grid-connected. Otherwise the
        network is islanded and the reference falls on a grid-forming local
        source — diesel first, then battery — so the AC solve has a slack while
        the reported grid import stays zero.
        """
        for bus_id, bus in self.bus_cfg_by_id.items():
            if bus.role.lower() in {"grid", "slack", "utility"}:
                return bus_id, True
        if self.settings.diesel.enabled and int(self.settings.diesel.bus) in self.bus_lookup:
            return int(self.settings.diesel.bus), False
        if int(self.settings.battery.bus) in self.bus_lookup:
            return int(self.settings.battery.bus), False
        return min(self.bus_lookup), False

    def _bus(self, bus_id: int, fallback_role: str | None = None) -> int:
        bid = int(bus_id)
        if bid in self.bus_lookup:
            return self.bus_lookup[bid]
        if fallback_role:
            for candidate_id, bus in self.bus_cfg_by_id.items():
                if bus.role.lower() == fallback_role.lower():
                    return self.bus_lookup[candidate_id]
        return self.bus_lookup[self._find_slack_bus_id()]

    def _connection(self, net: Any, link: LineCfg) -> int:
        from_cfg = self.bus_cfg_by_id[int(link.from_bus)]
        to_cfg = self.bus_cfg_by_id[int(link.to_bus)]
        kind = link.kind.lower()
        voltage_differs = abs(float(from_cfg.vn_kv) - float(to_cfg.vn_kv)) > 1e-9
        if kind == "transformer" or (kind == "auto" and voltage_differs):
            return self._transformer(net, link, from_cfg, to_cfg)
        return self._line(
            net,
            self._bus(link.from_bus),
            self._bus(link.to_bus),
            link.length_km,
            link.name,
            link.max_i_ka,
        )

    def _transformer(self, net: Any, link: LineCfg, from_cfg: BusCfg, to_cfg: BusCfg) -> int:
        if from_cfg.vn_kv >= to_cfg.vn_kv:
            hv_id, lv_id = int(link.from_bus), int(link.to_bus)
            hv_kv, lv_kv = from_cfg.vn_kv, to_cfg.vn_kv
        else:
            hv_id, lv_id = int(link.to_bus), int(link.from_bus)
            hv_kv, lv_kv = to_cfg.vn_kv, from_cfg.vn_kv
        return int(
            pp.create_transformer_from_parameters(
                net,
                hv_bus=self._bus(hv_id),
                lv_bus=self._bus(lv_id),
                sn_mva=1.6,
                vn_hv_kv=hv_kv,
                vn_lv_kv=lv_kv,
                vk_percent=6.0,
                vkr_percent=0.5,
                pfe_kw=1.5,
                i0_percent=0.2,
                shift_degree=0.0,
                name=link.name,
            )
        )

    @staticmethod
    def _line(
        net: Any,
        from_bus: int,
        to_bus: int,
        length_km: float,
        name: str,
        max_i_ka: float = 1.0,
    ) -> int:
        return int(
            pp.create_line_from_parameters(
                net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=length_km,
                r_ohm_per_km=0.05,
                x_ohm_per_km=0.02,
                c_nf_per_km=0.0,
                max_i_ka=max_i_ka,
                name=name,
            )
        )

    # -- driving profiles --------------------------------------------------
    def solar_factor(self) -> float:
        """Sinusoidal daylight curve, 0 at night, peak near solar noon."""
        from microgrid_simulator.components.pv import solar_factor

        return solar_factor(self.timestamp)

    @property
    def pv_is_real(self) -> bool:
        return self._pv_window is not None

    def current_pv_available_mw(self) -> float:
        """Measured PV availability when replaying, otherwise synthetic PV."""
        if self._pv_window is not None and len(self._pv_window) > 0:
            pos = min(self._window_pos, len(self._pv_window) - 1)
            return float(self._pv_window[pos])
        return self.solar_factor() * sum(self.pv_bases_mw)

    def load_factor(self) -> float:
        from microgrid_simulator.components.load import load_factor

        return load_factor(self.timestamp)

    @property
    def demand_is_real(self) -> bool:
        return self._demand_window is not None

    def current_demand_mw(self) -> float:
        """Target total static demand for this tick.

        Real window when installed: existing loads are scaled proportionally so
        their sum equals the historical total; otherwise the synthetic sinusoid
        uses the yaml base MW.
        """
        base_total = sum(p for p, _ in self.load_bases)
        if self._demand_window is not None and len(self._demand_window) > 0:
            pos = min(self._window_pos, len(self._demand_window) - 1)
            return float(self._demand_window[pos])
        return base_total * self.load_factor()

    def _apply_profiles(self, battery_p_mw: float, ev_p_mw: list[float], curtail_pv: float) -> None:
        # Scale all static loads by one factor so their sum hits the target
        # total while relative shares (and buses) stay untouched.
        base_total = sum(p for p, _ in self.load_bases)
        lf = self.current_demand_mw() / base_total if base_total > 1e-12 else 0.0
        static_p_mw = [p * lf for p, _ in self.load_bases]
        static_q_mvar = [q * lf for _, q in self.load_bases]
        ev_p_mw = list(ev_p_mw)

        demand_mw = sum(static_p_mw) + sum(ev_p_mw)
        pv_base_total = sum(self.pv_bases_mw)
        pv_used_mw = self.current_pv_available_mw() * (1.0 - curtail_pv)
        if not self.grid_connected:
            pv_used_mw = min(
                pv_used_mw,
                max(0.0, demand_mw + battery_p_mw - self.diesel.p_mw),
            )
        pv_scale = pv_used_mw / pv_base_total if pv_base_total > 1e-12 else 0.0
        for idx, base in zip(self.pv_indices, self.pv_bases_mw, strict=False):
            self.net.sgen.at[idx, "p_mw"] = base * pv_scale

        if self.grid_connected or demand_mw <= 1e-12:
            served_fraction = 1.0
        else:
            supply_for_load = (
                float(self.net.sgen.loc[self.pv_indices, "p_mw"].sum()) if self.pv_indices else 0.0
            )
            supply_for_load += self.diesel.p_mw - battery_p_mw
            served_fraction = min(1.0, max(0.0, supply_for_load / demand_mw))

        self._profile_static_p_mw = static_p_mw
        self._profile_static_q_mvar = static_q_mvar
        self._profile_ev_p_mw = ev_p_mw
        self._served_load_fraction = served_fraction
        for load_pos, idx in enumerate(self.static_load_indices):
            self.net.load.at[idx, "p_mw"] = static_p_mw[load_pos] * served_fraction
            self.net.load.at[idx, "q_mvar"] = static_q_mvar[load_pos] * served_fraction

        if self.diesel_index is not None:
            self.net.sgen.at[self.diesel_index, "p_mw"] = self.diesel.p_mw

        for idx, rate in zip(self.ev_load_indices, ev_p_mw, strict=False):
            self.net.load.at[idx, "p_mw"] = rate * served_fraction
            self.net.load.at[idx, "q_mvar"] = rate * 0.12 * served_fraction

        # pandapower storage: p_mw > 0 = charging (a load). Our convention matches.
        for idx in self.storage_indices:
            self.net.storage.at[idx, "p_mw"] = battery_p_mw
            self.net.storage.at[idx, "soc_percent"] = self.battery.soc * 100.0

        if self.dump_load_index is not None:
            local_supply_mw = pv_used_mw + self.diesel.p_mw + max(0.0, -battery_p_mw)
            named_sinks_mw = demand_mw * served_fraction + max(0.0, battery_p_mw)
            self.net.load.at[self.dump_load_index, "p_mw"] = max(
                0.0, local_supply_mw - named_sinks_mw
            )
            self.net.load.at[self.dump_load_index, "q_mvar"] = 0.0

    def _feasible_battery_request(
        self, action: ControlAction, ev_p_mw: list[float] | None = None
    ) -> float:
        requested = float(action.battery_p_mw)
        if self.grid_connected:
            return requested

        pv_mw = self.current_pv_available_mw() * (1.0 - action.pv_curtail)
        demand_mw = self.current_demand_mw() + sum(ev_p_mw or [])
        local_without_battery_mw = pv_mw + self.diesel.p_mw

        if requested >= 0.0:
            surplus_mw = max(0.0, local_without_battery_mw - demand_mw)
            return min(requested, surplus_mw)

        residual_mw = max(0.0, demand_mw - local_without_battery_mw)
        return max(requested, -residual_mw)

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

    # -- snapshot ----------------------------------------------------------
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
