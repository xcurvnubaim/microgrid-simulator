"""RuleBasedController — the interpretable heuristic baseline.

Grid-connected: solar-charge the battery in the morning window, discharge over
the evening peak, and let diesel peak-shave whatever exceeds the soft peak
threshold.

Islanded (no utility tie): there is no grid to backfill, so the controller
instead tries to *keep the lights on* — diesel carries the residual first (it
is the firm dispatchable source), the battery only tops up what diesel can't
and charges on PV surplus. Leaning on the battery first would black out once
its SoC floor is hit with diesel never asked to compensate.

Ported one-to-one from the legacy dashboard ``rule`` policy.
"""

from __future__ import annotations

from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState


class RuleBasedController(Controller):
    name = "rule"

    def act(self, state: GridState) -> ControlAction:
        s = self.settings
        hour = state.timestamp % 24.0
        islanded = state.islanded
        # Plan against available PV, not the previously dispatched/curtailed
        # value. Using pv_used creates a feedback lock: diesel displaces PV,
        # the next observation appears to have no PV, and diesel stays on.
        residual_mw = max(0.0, state.load_demand_mw - state.pv_available_mw)
        battery_p_mw = 0.0

        if islanded:
            surplus_mw = max(0.0, state.pv_available_mw - state.load_demand_mw)
            diesel_covered_mw = min(residual_mw, self._diesel_max_mw())
            battery_gap_mw = max(0.0, residual_mw - diesel_covered_mw)
            if surplus_mw > 0.0:
                battery_p_mw = min(s.battery.max_charge_mw, surplus_mw)
            elif battery_gap_mw > 0.0:
                battery_p_mw = -min(s.battery.max_discharge_mw, battery_gap_mw)
            # Diesel covers the whole residual it is capable of, regardless of
            # battery state, so the lights stay on whenever capacity exists. A
            # small headroom margin absorbs the one-tick control lag at ramps.
            diesel_target_mw = residual_mw * 1.12
        else:
            if 8 <= hour < 15:
                battery_p_mw = 0.6 * s.battery.max_charge_mw
            elif 18 <= hour < 22:
                battery_p_mw = -0.6 * s.battery.max_discharge_mw
            diesel_target_mw = max(0.0, residual_mw - s.reward.peak_threshold_mw)

        diesel_on = False
        diesel_setpoint_mw = 0.0
        if s.diesel.enabled:
            # Grid-connected, don't start the genset for less than its minimum
            # stable load — the grid covers small peaks. Islanded, any residual
            # justifies a start (surplus below min load is dumped, lights stay on).
            floor_mw = 0.0 if islanded else s.diesel.min_kw / 1000.0
            if diesel_target_mw > max(floor_mw, 0.0):
                diesel_on = True
                diesel_setpoint_mw = min(self._diesel_max_mw(), diesel_target_mw)

        return ControlAction(
            battery_p_mw=battery_p_mw,
            ev_p_mw=self._ev_charge(),
            pv_curtail=0.0,
            diesel_on=diesel_on,
            diesel_setpoint_mw=diesel_setpoint_mw,
        )
