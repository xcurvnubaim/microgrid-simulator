"""ManualScheduleController: YAML-driven diesel + battery timetable resolution."""

from __future__ import annotations

import pytest

from microgrid_simulator.config import (
    BatteryScheduleCfg,
    BatteryScheduleSegmentCfg,
    DieselScheduleCfg,
    DieselScheduleSegmentCfg,
    Settings,
)
from microgrid_simulator.controllers import ManualScheduleController
from microgrid_simulator.core.types import GridState


def _state(hour: float, load_mw: float = 0.08, pv_mw: float = 0.0) -> GridState:
    return GridState(
        v_bus=[1.0],
        p_load=[load_mw],
        p_gen=[pv_mw],
        soc=[0.5],
        soh=[1.0],
        ev_soc=[],
        line_loading=[],
        load_demand_mw=load_mw,
        pv_available_mw=pv_mw,
        islanded=True,
        timestamp=hour,
    )


def _settings(segments: list[DieselScheduleSegmentCfg] | None = None) -> Settings:
    settings = Settings()
    settings.diesel.enabled = True
    if segments is not None:
        settings.diesel_schedule = DieselScheduleCfg(segments=segments)
    return settings


def test_default_timetable_matches_documented_baseline() -> None:
    controller = ManualScheduleController(_settings())
    max_mw = controller.settings.diesel.max_kw / 1000.0
    min_mw = controller.settings.diesel.min_kw / 1000.0

    night = controller.act(_state(hour=3.0))
    assert night.diesel_on and night.diesel_setpoint_mw == pytest.approx(max_mw)

    midday = controller.act(_state(hour=11.0))
    assert midday.diesel_on and midday.diesel_setpoint_mw == pytest.approx(min_mw)

    evening = controller.act(_state(hour=18.0))
    assert evening.diesel_on and evening.diesel_setpoint_mw == pytest.approx(max_mw)


def test_uncovered_hours_and_off_level_turn_diesel_off() -> None:
    settings = _settings(
        [
            DieselScheduleSegmentCfg(start_hour=6.0, end_hour=9.0, level="max"),
            DieselScheduleSegmentCfg(start_hour=9.0, end_hour=12.0, level="off"),
        ]
    )
    controller = ManualScheduleController(settings)
    assert controller.act(_state(hour=10.0)).diesel_on is False  # explicit off
    assert controller.act(_state(hour=20.0)).diesel_on is False  # uncovered


def test_explicit_kw_level_is_clamped_to_stable_band() -> None:
    settings = _settings(
        [
            DieselScheduleSegmentCfg(start_hour=0.0, end_hour=12.0, level=80.0),
            DieselScheduleSegmentCfg(start_hour=12.0, end_hour=18.0, level=10.0),
            DieselScheduleSegmentCfg(start_hour=18.0, end_hour=24.0, level=500.0),
        ]
    )
    controller = ManualScheduleController(settings)
    d = settings.diesel
    assert controller.act(_state(hour=6.0)).diesel_setpoint_mw == pytest.approx(0.080)
    # Below min stable load clamps up; above nameplate clamps down.
    assert controller.act(_state(hour=15.0)).diesel_setpoint_mw == pytest.approx(
        d.min_kw / 1000.0
    )
    assert controller.act(_state(hour=21.0)).diesel_setpoint_mw == pytest.approx(
        d.max_kw / 1000.0
    )


def test_overnight_wraparound_segment() -> None:
    settings = _settings([DieselScheduleSegmentCfg(start_hour=22.0, end_hour=6.0, level="max")])
    controller = ManualScheduleController(settings)
    assert controller.act(_state(hour=23.0)).diesel_on is True
    assert controller.act(_state(hour=2.0)).diesel_on is True
    assert controller.act(_state(hour=12.0)).diesel_on is False


def test_battery_covers_gap_beyond_scheduled_diesel() -> None:
    settings = _settings([DieselScheduleSegmentCfg(start_hour=0.0, end_hour=24.0, level="min")])
    controller = ManualScheduleController(settings)
    min_mw = settings.diesel.min_kw / 1000.0
    action = controller.act(_state(hour=3.0, load_mw=min_mw + 0.020))
    assert action.battery_p_mw == pytest.approx(-0.020)


def test_disabled_diesel_ignores_timetable() -> None:
    settings = _settings()
    settings.diesel.enabled = False
    action = ManualScheduleController(settings).act(_state(hour=3.0))
    assert action.diesel_on is False and action.diesel_setpoint_mw == 0.0


# --- battery timetable -------------------------------------------------------
def _battery_settings(segments: list[BatteryScheduleSegmentCfg]) -> Settings:
    settings = _settings()
    settings.battery_schedule = BatteryScheduleCfg(segments=segments)
    return settings


def test_battery_charge_window_is_binary_full_rate() -> None:
    settings = _battery_settings(
        [BatteryScheduleSegmentCfg(start_hour=10.0, end_hour=15.0, mode="charge")]
    )
    controller = ManualScheduleController(settings)
    action = controller.act(_state(hour=12.0))
    assert action.battery_p_mw == pytest.approx(settings.battery.max_charge_mw)


def test_battery_discharge_window_takes_continuous_kw() -> None:
    settings = _battery_settings(
        [
            BatteryScheduleSegmentCfg(start_hour=18.0, end_hour=20.0, mode="discharge", level=80.0),
            BatteryScheduleSegmentCfg(
                start_hour=20.0, end_hour=22.0, mode="discharge", level="max"
            ),
            BatteryScheduleSegmentCfg(
                start_hour=22.0, end_hour=24.0, mode="discharge", level=9000.0
            ),
        ]
    )
    controller = ManualScheduleController(settings)
    assert controller.act(_state(hour=19.0)).battery_p_mw == pytest.approx(-0.080)
    assert controller.act(_state(hour=21.0)).battery_p_mw == pytest.approx(
        -settings.battery.max_discharge_mw
    )
    # Explicit kW above the inverter rating clamps to max_discharge_mw.
    assert controller.act(_state(hour=23.0)).battery_p_mw == pytest.approx(
        -settings.battery.max_discharge_mw
    )


def test_battery_timetable_idles_uncovered_hours_even_with_residual() -> None:
    # With a timetable present the battery no longer reacts to the residual:
    # uncovered/idle hours command 0 regardless of load beyond diesel.
    settings = _battery_settings(
        [BatteryScheduleSegmentCfg(start_hour=10.0, end_hour=12.0, mode="idle")]
    )
    controller = ManualScheduleController(settings)
    assert controller.act(_state(hour=11.0, load_mw=0.5)).battery_p_mw == 0.0  # idle
    assert controller.act(_state(hour=3.0, load_mw=0.5)).battery_p_mw == 0.0  # uncovered


def test_battery_without_timetable_keeps_reactive_fallback() -> None:
    settings = _settings([DieselScheduleSegmentCfg(start_hour=0.0, end_hour=24.0, level="min")])
    assert settings.battery_schedule.segments == []
    controller = ManualScheduleController(settings)
    min_mw = settings.diesel.min_kw / 1000.0
    action = controller.act(_state(hour=3.0, load_mw=min_mw + 0.020))
    assert action.battery_p_mw == pytest.approx(-0.020)
