"""DeterministicController — SoC-aware greedy merit-order dispatch.

Fulfils each tick's demand with the cheapest local resource first, as a pure
function of the last solved state: PV serves load directly, surplus PV charges
the battery, the battery shaves everything above the peak threshold
(grid-connected) or covers the residual (islanded), and diesel firms only what
the battery physically cannot deliver this tick. No randomness, no clock
windows, no look-ahead — the same state always maps to the same action.

Unlike ``rule`` (fixed morning-charge / evening-discharge windows) this
controller derives the battery's true deliverable power from its SoC, so
islanded it hands the residual over to diesel exactly when the battery runs
out of headroom instead of blacking out at the SoC floor.
"""

from __future__ import annotations

from microgrid_simulator.config import Settings
from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState


class DeterministicController(Controller):
    name = "deterministic"

    def __init__(
        self,
        settings: Settings,
        threshold_mw: float | None = None,
        recharge_fraction: float = 0.5,
    ) -> None:
        super().__init__(settings)
        # Default to the reward's soft peak cap so "good" matches the reward.
        self.threshold_mw = (
            threshold_mw if threshold_mw is not None else settings.reward.peak_threshold_mw
        )
        # Recharge only when expected import is below this share of the threshold.
        self.recharge_below_mw = recharge_fraction * self.threshold_mw

    def _battery_caps_mw(self, state: GridState) -> tuple[float, float]:
        """(charge, discharge) power the battery can actually take/deliver this
        tick, limited by both its converter rating and its SoC headroom —
        mirrors :meth:`BatteryModel.apply` so requests are realised as asked."""
        b = self.settings.battery
        dt = max(float(self.settings.topology.timestep_hours), 1e-9)
        soc = state.soc[0] if state.soc else b.soc_init
        soh = state.soh[0] if state.soh else 1.0
        capacity_mwh = max(b.capacity_mwh * soh, 1e-9)
        charge_mw = min(
            b.max_charge_mw, max(0.0, b.soc_max - soc) * capacity_mwh / (b.charge_eff * dt)
        )
        discharge_mw = min(
            b.max_discharge_mw, max(0.0, soc - b.soc_min) * capacity_mwh * b.discharge_eff / dt
        )
        return charge_mw, discharge_mw

    def act(self, state: GridState) -> ControlAction:
        s = self.settings
        charge_cap_mw, discharge_cap_mw = self._battery_caps_mw(state)
        residual_mw = max(0.0, state.load_demand_mw - state.pv_used_mw)
        surplus_mw = max(0.0, state.pv_used_mw - state.load_demand_mw)
        battery_p_mw = 0.0
        diesel_gap_mw = 0.0

        if surplus_mw > 0.0:
            battery_p_mw = min(charge_cap_mw, surplus_mw)  # soak up free PV
        elif state.islanded:
            # Battery covers the residual up to what its SoC can deliver;
            # diesel firms the remainder so the lights stay on.
            battery_p_mw = -min(discharge_cap_mw, residual_mw)
            diesel_gap_mw = residual_mw + battery_p_mw
        elif residual_mw > self.threshold_mw:
            # Shave the peak; anything the battery can't shave goes to diesel.
            battery_p_mw = -min(discharge_cap_mw, residual_mw - self.threshold_mw)
            diesel_gap_mw = residual_mw + battery_p_mw - self.threshold_mw
        elif residual_mw < self.recharge_below_mw:
            # Valley filling: recharge without pushing import past the threshold.
            battery_p_mw = min(charge_cap_mw, self.threshold_mw - residual_mw)

        diesel_on = False
        diesel_setpoint_mw = 0.0
        if s.diesel.enabled and diesel_gap_mw > 1e-9:
            min_mw = s.diesel.min_kw / 1000.0
            # Grid-connected, don't start the genset below its minimum stable
            # load — the grid covers small peaks. Islanded, any deficit
            # justifies a start (surplus below min load is dumped).
            if state.islanded or diesel_gap_mw >= min_mw:
                diesel_on = True
                diesel_setpoint_mw = min(self._diesel_max_mw(), max(min_mw, diesel_gap_mw))

        return ControlAction(
            battery_p_mw=battery_p_mw,
            ev_p_mw=self._ev_charge(),
            pv_curtail=0.0,
            diesel_on=diesel_on,
            diesel_setpoint_mw=diesel_setpoint_mw,
        )
