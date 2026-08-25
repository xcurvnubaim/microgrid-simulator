from __future__ import annotations

from microgrid_simulator.config import Settings
from microgrid_simulator.core.types import ControlAction, GridState
from microgrid_simulator.rl.env import project_rl_action


def _state(**updates: object) -> GridState:
    data = {
        "soc": [0.50],
        "soh": [1.0],
        "load_demand_mw": 0.30,
        "pv_available_mw": 0.0,
        "pv_used_mw": 0.0,
    }
    data.update(updates)
    return GridState(**data)


def test_shield_commits_diesel_for_current_shortage() -> None:
    action = ControlAction(battery_p_mw=0.10, diesel_on=False)
    projected, reasons = project_rl_action(action, _state(), Settings(), 0.50, 0, 288)

    assert projected.battery_p_mw == 0.0
    assert projected.diesel_on
    assert projected.diesel_setpoint_mw > 0.0
    assert "cancel_battery_charge_for_load" in reasons
    assert "load_service_diesel_commitment" in reasons


def test_shield_blocks_terminal_discharge() -> None:
    action = ControlAction(battery_p_mw=-0.20, diesel_on=False)
    projected, reasons = project_rl_action(action, _state(soc=[0.40]), Settings(), 0.50, 250, 288)

    assert projected.battery_p_mw == 0.0
    assert projected.diesel_on
    assert "terminal_soc_discharge_block" in reasons
    assert "load_service_diesel_commitment" in reasons


def test_shield_does_not_change_safe_action_outside_terminal_window() -> None:
    action = ControlAction(battery_p_mw=-0.10, diesel_on=False)
    projected, reasons = project_rl_action(
        action, _state(load_demand_mw=0.05), Settings(), 0.50, 0, 288
    )

    assert projected.battery_p_mw == action.battery_p_mw
    assert projected.diesel_on == action.diesel_on
    assert reasons == ()
