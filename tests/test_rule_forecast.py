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
