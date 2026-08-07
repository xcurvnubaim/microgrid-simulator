"""DeterministicController: merit-order dispatch, SoC awareness, determinism."""

from __future__ import annotations

import pytest

from microgrid_simulator.config import Settings
from microgrid_simulator.controllers import DeterministicController
from microgrid_simulator.core.types import GridState


def _state(**overrides: object) -> GridState:
    state = GridState(soc=[0.5], soh=[1.0])
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_same_state_same_action() -> None:
    ctrl = DeterministicController(Settings())
    state = _state(load_demand_mw=0.5, pv_used_mw=0.1)
    assert ctrl.act(state) == ctrl.act(state)


def test_pv_surplus_charges_battery() -> None:
    ctrl = DeterministicController(Settings())
    action = ctrl.act(_state(load_demand_mw=0.1, pv_used_mw=0.3))
    assert action.battery_p_mw > 0.0
    assert not action.diesel_on


def test_peak_above_threshold_discharges() -> None:
    settings = Settings()
    ctrl = DeterministicController(settings)
    threshold = settings.reward.peak_threshold_mw
    action = ctrl.act(_state(load_demand_mw=threshold + 0.1, pv_used_mw=0.0))
    assert action.battery_p_mw < 0.0
    # Battery can shave the whole 0.1 MW overshoot: no genset start.
    assert not action.diesel_on


def test_empty_battery_hands_peak_to_diesel() -> None:
    settings = Settings()
    ctrl = DeterministicController(settings)
    threshold = settings.reward.peak_threshold_mw
    overshoot = settings.diesel.min_kw / 1000.0 + 0.01
    state = _state(
        load_demand_mw=threshold + overshoot, pv_used_mw=0.0, soc=[settings.battery.soc_min]
    )
    action = ctrl.act(state)
    assert action.battery_p_mw == 0.0  # nothing left below the SoC floor
    assert action.diesel_on
    assert action.diesel_setpoint_mw >= overshoot - 1e-9


def test_islanded_diesel_covers_battery_shortfall() -> None:
    settings = Settings()
    ctrl = DeterministicController(settings)
    residual = settings.battery.max_discharge_mw + 0.05
    state = _state(load_demand_mw=residual, pv_used_mw=0.0, islanded=True, soc=[0.9])
    action = ctrl.act(state)
    assert action.battery_p_mw < 0.0
    assert action.diesel_on
    assert action.diesel_setpoint_mw + -action.battery_p_mw >= residual - 1e-9


def test_valley_fill_stays_below_threshold() -> None:
    settings = Settings()
    ctrl = DeterministicController(settings)
    action = ctrl.act(_state(load_demand_mw=0.05, pv_used_mw=0.0, soc=[0.2]))
    assert action.battery_p_mw > 0.0
    residual = 0.05
    assert residual + action.battery_p_mw <= settings.reward.peak_threshold_mw + 1e-9


def test_rollout_accepts_deterministic_policy() -> None:
    from microgrid_simulator.ui.rollout import POLICIES, run_rollout

    assert "deterministic" in POLICIES
    result = run_rollout(Settings(), policy="deterministic", seed=0)
    assert result["rows"]
    assert result["meta"]["policy"] == "deterministic"


def test_policies_expose_rl() -> None:
    from microgrid_simulator.ui.rollout import POLICIES

    assert "rl" in POLICIES


def test_rl_policy_requires_artifact() -> None:
    from microgrid_simulator.ui.rollout import run_rollout

    with pytest.raises(ValueError, match="rl"):
        run_rollout(Settings(), policy="rl", seed=0)
