"""Pandapower network construction for :class:`PandapowerBackend`.

Topology is data-driven from ``Settings.buses`` and ``Settings.lines`` so the
same scenario can be edited in the dashboard like a light Cisco Packet Tracer
model: create buses, connect them, then place PV, storage, diesel, EV, and load
assets on those buses.
"""

from __future__ import annotations

import logging
from typing import Any

import pandapower as pp

from microgrid_simulator.config import BusCfg, LineCfg, Settings, TopologyCfg

LOGGER = logging.getLogger(__name__)


class _NetworkBuilderMixin:
    """Editable-network construction split out of the backend lifecycle."""

    # Shared state owned by PandapowerBackend._configure.
    settings: Settings
    topo: TopologyCfg
    net: Any
    bus_lookup: dict[int, int]
    bus_cfg_by_id: dict[int, BusCfg]
    grid_connected: bool
    reference_bus_id: int
    pv_indices: list[int]
    diesel_index: int | None
    dump_load_index: int | None
    storage_indices: list[int]
    static_load_indices: list[int]
    ev_load_indices: list[int]

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
