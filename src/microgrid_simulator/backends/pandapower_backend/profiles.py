"""Driving profiles and dispatch feasibility for :class:`PandapowerBackend`.

Owns the solar/load curves (synthetic or replayed windows), per-tick profile
application onto the pandapower elements, and the islanded battery-request
clipping that keeps local sources from overserving the island.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from microgrid_simulator.components import BatteryLike, DieselModel
from microgrid_simulator.config import Settings, TopologyCfg
from microgrid_simulator.core.types import ControlAction


class _ProfilesMixin:
    """Time-series driven profiles split out of the backend lifecycle."""

    # Shared state owned by PandapowerBackend._configure.
    settings: Settings
    topo: TopologyCfg
    timestamp: float
    net: Any
    diesel: DieselModel
    battery: BatteryLike
    grid_connected: bool
    pv_bases_mw: list[float]
    load_bases: list[tuple[float, float]]
    pv_indices: list[int]
    diesel_index: int | None
    dump_load_index: int | None
    storage_indices: list[int]
    static_load_indices: list[int]
    ev_load_indices: list[int]
    _demand_window: np.ndarray | None
    _pv_window: np.ndarray | None
    _window_pos: int
    _profile_static_p_mw: list[float]
    _profile_static_q_mvar: list[float]
    _profile_ev_p_mw: list[float]
    _served_load_fraction: float

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

    def _apply_profiles(
        self, battery_p_mw: float, ev_p_mw: list[float], curtail_pv: float
    ) -> None:
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
