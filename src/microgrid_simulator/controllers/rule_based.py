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

When a matched, leakage-free PV/load forecast is available, the forecast replaces
the islanded rule's blind 12% diesel margin with bounded, trend-aware headroom:
the currently observed residual remains the safety floor, while a rising forecast
may increase the target by at most 12%. Missing or invalid forecasts preserve the
legacy rule exactly.
"""

from __future__ import annotations

import math

from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState

MAX_FORECAST_HEADROOM_FRACTION = 0.12


class RuleBasedController(Controller):
    name = "rule"

    @staticmethod
    def _forecast_residual_mw(
        pv_forecast_mw: float | None,
        demand_forecast_mw: float | None,
    ) -> float | None:
        """Return forecast net load only when both inputs are usable."""

        if pv_forecast_mw is None or demand_forecast_mw is None:
            return None
        pv = float(pv_forecast_mw)
        demand = float(demand_forecast_mw)
        if not math.isfinite(pv) or not math.isfinite(demand) or pv < 0.0 or demand < 0.0:
            return None
        return max(0.0, demand - pv)

    @staticmethod
    def _forecast_aware_target_mw(current_target_mw: float, forecast_target_mw: float) -> float:
        """Add only bounded headroom for a forecast rise.

        Generation never falls below the current target, so a forecast cannot
        create an immediate deficit. The 12% cap matches the legacy islanded
        margin and prevents an hourly forecast from causing large dump-load
        production during a 15-minute control interval.
        """

        rise_mw = max(0.0, forecast_target_mw - current_target_mw)
        max_headroom_mw = current_target_mw * MAX_FORECAST_HEADROOM_FRACTION
        return current_target_mw + min(rise_mw, max_headroom_mw)

    def act(
        self,
        state: GridState,
        *,
        pv_forecast_mw: float | None = None,
        demand_forecast_mw: float | None = None,
    ) -> ControlAction:
        s = self.settings
        hour = state.timestamp % 24.0
        islanded = state.islanded
        # Plan against available PV, not the previously dispatched/curtailed
        # value. Using pv_used creates a feedback lock: diesel displaces PV,
        # the next observation appears to have no PV, and diesel stays on.
        residual_mw = max(0.0, state.load_demand_mw - state.pv_available_mw)
        forecast_residual_mw = self._forecast_residual_mw(
            pv_forecast_mw, demand_forecast_mw
        )
        battery_p_mw = 0.0

        if islanded:
            surplus_mw = max(0.0, state.pv_available_mw - state.load_demand_mw)
            diesel_covered_mw = min(residual_mw, self._diesel_max_mw())
            battery_gap_mw = max(0.0, residual_mw - diesel_covered_mw)
            if surplus_mw > 0.0:
                battery_p_mw = min(s.battery.max_charge_mw, surplus_mw)
            elif battery_gap_mw > 0.0:
                battery_p_mw = -min(s.battery.max_discharge_mw, battery_gap_mw)
            # Diesel covers the whole observed residual it is capable of,
            # regardless of battery state. Without a forecast, preserve the
            # legacy 12% margin. With a forecast, use only the portion of that
            # margin supported by a near-term rise in forecast net load.
            if forecast_residual_mw is None:
                diesel_target_mw = residual_mw * (1.0 + MAX_FORECAST_HEADROOM_FRACTION)
            else:
                diesel_target_mw = self._forecast_aware_target_mw(
                    residual_mw, forecast_residual_mw
                )
        else:
            if 8 <= hour < 15:
                battery_p_mw = 0.6 * s.battery.max_charge_mw
            elif 18 <= hour < 22:
                battery_p_mw = -0.6 * s.battery.max_discharge_mw
            diesel_target_mw = max(0.0, residual_mw - s.reward.peak_threshold_mw)
            if forecast_residual_mw is not None:
                forecast_peak_mw = max(
                    0.0, forecast_residual_mw - s.reward.peak_threshold_mw
                )
                diesel_target_mw = self._forecast_aware_target_mw(
                    diesel_target_mw, forecast_peak_mw
                )

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
