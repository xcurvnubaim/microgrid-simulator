"""ManualScheduleController — fixed time-of-day diesel dispatch.

Every rule-baseline blackout traced back to the same cause (see
[[Simulator Accounting Findings - Module 6]]): the diesel setpoint reacts to
last tick's residual, so a PV dip or load spike between two 15-minute samples
outruns diesel for one tick, and ``RuleBasedController`` never routes that gap
to the battery because it assumes diesel (well under its nameplate cap)
already covers the residual.

This controller sidesteps the lag by not reacting to the residual at all for
diesel: it plays back the clock-driven timetable in ``settings.diesel_schedule``
(per-scenario YAML; editable in the dashboard's Diesel schedule panel). Each
segment maps an hour-of-day range to a level — nameplate ``max``, minimum
stable load ``min``, ``off``, or an explicit kW value; uncovered hours mean
off. The setpoint is deliberately not sized to the instantaneous residual, so
it overgenerates relative to actual need — the surplus shows up as extra
``diesel_kwh``/``carbon_kg`` and ``excess_generation_kw`` (no PV/battery sink
is commanded for it), the reliability-for-waste trade this baseline is meant
to demonstrate against the reactive rule and PyPSA-RH baselines.

The default timetable (full nameplate outside 09:00-14:00, min load inside
it) was hand-derived from the 2026-01-15 window's hourly load/PV profile. A
first cut that turned diesel fully *off* in that window was worse than the
rule baseline — PV can still swing ~47 kW in one 15-minute step inside it,
which outran the same laggy battery-only backstop the rule controller has;
holding ``min_kw`` (assumed 45 kW, close to the window's observed 45.9 kW
worst-case residual) removes most of that gap.

Battery behaviour defaults to the ``RuleBasedController``'s reactive policy
(charge on PV surplus, discharge to cover any residual the scheduled diesel
doesn't reach) so the diesel-only comparison stays isolated. When
``settings.battery_schedule`` has segments the battery follows its own clock
instead: charge windows are binary (full ``max_charge_mw``; the backend clips
by SOC and resolves the PV→diesel→grid source split), discharge windows carry
a continuous kW level, and uncovered hours idle.
"""

from __future__ import annotations

from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState


class ManualScheduleController(Controller):
    name = "schedule"

    def _scheduled_diesel_mw(self, hour: float) -> tuple[bool, float]:
        """Resolve the timetable to (on, setpoint_mw) for this hour-of-day."""
        s = self.settings
        if not s.diesel.enabled:
            return False, 0.0
        level = s.diesel_schedule.level_at(hour)
        if level == "off":
            return False, 0.0
        if level == "max":
            return True, self._diesel_max_mw()
        if level == "min":
            return True, s.diesel.min_kw / 1000.0
        # Explicit kW value, clamped to the genset's stable operating band.
        setpoint_mw = min(max(float(level), s.diesel.min_kw), s.diesel.max_kw) / 1000.0
        return True, setpoint_mw

    def _scheduled_battery_mw(
        self,
        hour: float,
        residual_mw: float,
        surplus_mw: float,
        diesel_setpoint_mw: float,
        islanded: bool,
    ) -> float | None:
        """Resolve the battery timetable; ``None`` = no timetable configured."""
        s = self.settings
        if not s.battery_schedule.segments:
            return None
        segment = s.battery_schedule.segment_at(hour)
        if segment is None or segment.mode == "idle":
            return 0.0
        if segment.mode == "reactive":
            if not islanded:
                return 0.0
            if surplus_mw > 0.0:
                return min(s.battery.max_charge_mw, surplus_mw)
            battery_gap_mw = max(0.0, residual_mw - diesel_setpoint_mw)
            return -min(s.battery.max_discharge_mw, battery_gap_mw)
        if segment.mode == "charge":
            if segment.level == "max":
                return s.battery.max_charge_mw
            return min(max(0.0, float(segment.level)) / 1000.0, s.battery.max_charge_mw)
        if segment.level == "max":
            return -s.battery.max_discharge_mw
        return -min(max(0.0, float(segment.level)) / 1000.0, s.battery.max_discharge_mw)

    def act(self, state: GridState) -> ControlAction:
        s = self.settings
        hour = state.timestamp % 24.0
        islanded = state.islanded
        residual_mw = max(0.0, state.load_demand_mw - state.pv_available_mw)
        surplus_mw = max(0.0, state.pv_available_mw - state.load_demand_mw)

        diesel_on, diesel_setpoint_mw = self._scheduled_diesel_mw(hour)

        battery_p_mw = self._scheduled_battery_mw(
            hour,
            residual_mw,
            surplus_mw,
            diesel_setpoint_mw,
            islanded,
        )
        if battery_p_mw is None:
            # Reactive fallback (islanded only), unchanged from RuleBasedController.
            battery_p_mw = 0.0
            if islanded:
                if surplus_mw > 0.0:
                    battery_p_mw = min(s.battery.max_charge_mw, surplus_mw)
                else:
                    battery_gap_mw = max(0.0, residual_mw - diesel_setpoint_mw)
                    if battery_gap_mw > 0.0:
                        battery_p_mw = -min(s.battery.max_discharge_mw, battery_gap_mw)

        return ControlAction(
            battery_p_mw=battery_p_mw,
            ev_p_mw=self._ev_charge(),
            pv_curtail=0.0,
            diesel_on=diesel_on,
            diesel_setpoint_mw=diesel_setpoint_mw,
        )
