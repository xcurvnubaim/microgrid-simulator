"""Independent command projection before a request reaches the plant."""

from __future__ import annotations

from dataclasses import dataclass

from microgrid_simulator.config import Settings
from microgrid_simulator.core.backend import clamp_action
from microgrid_simulator.core.types import ControlAction
from microgrid_simulator.ems.types import TelemetryFrame


@dataclass(frozen=True)
class ShieldResult:
    action: ControlAction
    reasons: tuple[str, ...]


class SafetyShield:
    """Project commands against ratings, SOC headroom, and diesel ramp bounds."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def apply(self, action: ControlAction, frame: TelemetryFrame) -> ShieldResult:
        projected = clamp_action(action, self.settings)
        reasons: list[str] = []
        if projected != action:
            reasons.append("static_device_limit")

        battery = self.settings.battery
        capacity = max(battery.capacity_mwh * frame.battery_soh, 1e-9)
        dt = frame.timestep_hours
        max_charge = max(
            0.0,
            (battery.soc_max - frame.battery_soc) * capacity / (battery.charge_eff * dt),
        )
        max_discharge = max(
            0.0,
            (frame.battery_soc - battery.soc_min) * capacity * battery.discharge_eff / dt,
        )
        bounded_battery = min(max_charge, max(-max_discharge, projected.battery_p_mw))
        if abs(bounded_battery - projected.battery_p_mw) > 1e-9:
            projected.battery_p_mw = bounded_battery
            reasons.append("battery_soc_headroom")

        if projected.diesel_on:
            lo = min(self.settings.diesel.min_kw, self.settings.diesel.max_kw) / 1000.0
            hi = self.settings.diesel.max_kw / 1000.0
            if self.settings.diesel.ramp_kw_per_min > 0.0 and frame.diesel_on:
                ramp = self.settings.diesel.ramp_kw_per_min * dt * 60.0 / 1000.0
                current = frame.diesel_power_kw / 1000.0
                lo = max(lo, current - ramp)
                hi = min(hi, current + ramp)
            bounded_diesel = min(hi, max(lo, projected.diesel_setpoint_mw))
            if abs(bounded_diesel - projected.diesel_setpoint_mw) > 1e-9:
                projected.diesel_setpoint_mw = bounded_diesel
                reasons.append("diesel_operating_band")

        return ShieldResult(projected, tuple(reasons))
