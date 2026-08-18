"""Forecast-aware diesel headroom for the interpretable rule controller."""

from __future__ import annotations

import math

import pytest

from microgrid_simulator.config import Settings
from microgrid_simulator.controllers.rule_based import RuleBasedController
from microgrid_simulator.core.types import GridState
from microgrid_simulator.ui.rollout import policy_action


def _settings() -> Settings:
    raw = Settings().model_dump()
    raw["diesel"] |= {"enabled": True, "min_kw": 45.0, "max_kw": 500.0}
    return Settings(**raw)


def _state(*, islanded: bool = True, demand_mw: float = 0.200, pv_mw: float = 0.100) -> GridState:
    return GridState(
        islanded=islanded,
        load_demand_mw=demand_mw,
        pv_available_mw=pv_mw,
        pv_used_mw=pv_mw,
    )


def _night_state(*, soc: float, demand_mw: float = 0.200, pv_mw: float = 0.0) -> GridState:
    state = _state(islanded=True, demand_mw=demand_mw, pv_mw=pv_mw)
    state.timestamp = 22.0
    state.soc = [soc]
    return state


def test_no_forecast_preserves_legacy_islanded_headroom() -> None:
    action = RuleBasedController(_settings()).act(_state())

    assert action.diesel_on
    assert action.diesel_setpoint_mw == pytest.approx(0.112)


def test_flat_or_falling_forecast_removes_blind_extra_headroom() -> None:
    controller = RuleBasedController(_settings())

    flat = controller.act(_state(), pv_forecast_mw=0.100, demand_forecast_mw=0.200)
    falling = controller.act(_state(), pv_forecast_mw=0.150, demand_forecast_mw=0.200)

    assert flat.diesel_setpoint_mw == pytest.approx(0.100)
    assert falling.diesel_setpoint_mw == pytest.approx(0.100)


def test_rising_forecast_adds_only_supported_bounded_headroom() -> None:
    controller = RuleBasedController(_settings())

    small_rise = controller.act(_state(), pv_forecast_mw=0.100, demand_forecast_mw=0.206)
    large_rise = controller.act(_state(), pv_forecast_mw=0.100, demand_forecast_mw=0.400)

    assert small_rise.diesel_setpoint_mw == pytest.approx(0.106)
    assert large_rise.diesel_setpoint_mw == pytest.approx(0.112)


def test_forecast_headroom_fraction_is_configurable() -> None:
    settings = _settings()
    settings.rule.forecast_headroom_fraction = 0.25
    controller = RuleBasedController(settings)

    no_forecast = controller.act(_state())
    rising_forecast = controller.act(
        _state(), pv_forecast_mw=0.100, demand_forecast_mw=0.400
    )

    assert no_forecast.diesel_setpoint_mw == pytest.approx(0.125)
    assert rising_forecast.diesel_setpoint_mw == pytest.approx(0.125)


def test_night_discharges_only_energy_above_reserve_floor() -> None:
    controller = RuleBasedController(_settings())

    above = controller.act(_night_state(soc=0.50))
    at_floor = controller.act(_night_state(soc=0.30))

    assert above.battery_p_mw == pytest.approx(-0.20)
    assert above.diesel_on is False
    assert at_floor.battery_p_mw == 0.0
    assert at_floor.diesel_setpoint_mw == pytest.approx(0.224)


def test_forecast_net_load_raises_diesel_off_night_reserve() -> None:
    controller = RuleBasedController(_settings())

    action = controller.act(
        _night_state(soc=0.50, demand_mw=0.20),
        pv_forecast_horizon_mw=[0.0, 0.0, 0.3],
        demand_forecast_horizon_mw=[0.2, 0.2, 0.2],
    )

    # Two forecast hours each require 200 kW with diesel off, lifting the reserve
    # above the configured 30% floor and preventing discharge at 50% SOC.
    assert action.battery_p_mw == 0.0


def test_forecast_pv_surplus_ends_diesel_off_reserve_horizon() -> None:
    controller = RuleBasedController(_settings())

    action = controller.act(
        _night_state(soc=0.50, demand_mw=0.20),
        pv_forecast_horizon_mw=[0.0, 0.3, 0.0],
        demand_forecast_horizon_mw=[0.1, 0.2, 0.4],
    )

    # Only 100 kWh before the forecast PV-surplus point is reserved. At 50% SOC,
    # energy above the resulting reserve remains available for current discharge.
    assert action.battery_p_mw < 0.0


def test_daytime_charging_uses_only_measured_pv_surplus() -> None:
    controller = RuleBasedController(_settings())

    no_surplus = controller.act(
        _state(demand_mw=0.20, pv_mw=0.10),
        pv_forecast_mw=0.40,
        demand_forecast_mw=0.20,
    )
    measured_surplus = controller.act(_state(demand_mw=0.10, pv_mw=0.20))

    assert no_surplus.battery_p_mw == 0.0
    assert measured_surplus.battery_p_mw == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("pv_forecast_mw", "demand_forecast_mw"),
    [
        (None, 0.2),
        (0.1, None),
        (math.nan, 0.2),
        (0.1, math.inf),
        (-0.1, 0.2),
    ],
)
def test_incomplete_or_invalid_forecast_preserves_legacy_rule(
    pv_forecast_mw: float | None,
    demand_forecast_mw: float | None,
) -> None:
    action = RuleBasedController(_settings()).act(
        _state(),
        pv_forecast_mw=pv_forecast_mw,
        demand_forecast_mw=demand_forecast_mw,
    )

    assert action.diesel_setpoint_mw == pytest.approx(0.112)


def test_grid_connected_forecast_adjustment_respects_current_peak_floor() -> None:
    settings = _settings()
    settings.reward.peak_threshold_mw = 0.050
    controller = RuleBasedController(settings)
    state = _state(islanded=False, demand_mw=0.200, pv_mw=0.100)

    falling = controller.act(state, pv_forecast_mw=0.100, demand_forecast_mw=0.120)
    rising = controller.act(state, pv_forecast_mw=0.100, demand_forecast_mw=0.300)

    assert falling.diesel_setpoint_mw == pytest.approx(0.050)
    assert rising.diesel_setpoint_mw == pytest.approx(0.056)


def test_rollout_rule_policy_forwards_current_forecast() -> None:
    class ForecastEnv:
        settings = _settings()
        _last_state = _state()

        @staticmethod
        def _current_pv_forecast_mw() -> float:
            return 0.100

        @staticmethod
        def _current_demand_forecast_mw() -> float:
            return 0.206

        @staticmethod
        def encode_action(action):
            return action

    action = policy_action(ForecastEnv(), "rule")

    assert action.diesel_setpoint_mw == pytest.approx(0.106)
