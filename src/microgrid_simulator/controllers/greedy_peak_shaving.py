"""GreedyPeakShavingController — keep grid import under a threshold, myopically.

Uses the last solved tick as the forecast for the next one (one-tick lag):
discharge just enough battery to pull the expected import below the threshold,
and recharge only during clearly low-demand periods so energy is available for
the next peak. No look-ahead — that is exactly what the PyPSA-RH baseline
adds on top.
"""

from __future__ import annotations

from microgrid_simulator.config import Settings
from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState


class GreedyPeakShavingController(Controller):
    name = "greedy"

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

    def act(self, state: GridState) -> ControlAction:
        b = self.settings.battery
        expected_import_mw = max(0.0, state.load_demand_mw - state.pv_used_mw)
        surplus_mw = max(0.0, state.pv_used_mw - state.load_demand_mw)
        battery_p_mw = 0.0

        if surplus_mw > 0.0:
            battery_p_mw = min(b.max_charge_mw, surplus_mw)  # soak up free PV
        elif expected_import_mw > self.threshold_mw:
            battery_p_mw = -min(b.max_discharge_mw, expected_import_mw - self.threshold_mw)
        elif expected_import_mw < self.recharge_below_mw:
            # Valley filling: recharge without pushing import past the threshold.
            battery_p_mw = min(b.max_charge_mw, self.threshold_mw - expected_import_mw)

        return ControlAction(battery_p_mw=battery_p_mw, ev_p_mw=self._ev_charge())
