"""Native pymgrid25 scenario translation and source-data regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("pymgrid")

from microgrid_simulator.backends.simple_backend import SimpleBackend  # noqa: E402
from microgrid_simulator.components.battery import PymgridBatteryModel  # noqa: E402
from microgrid_simulator.config import Settings  # noqa: E402
from microgrid_simulator.core.types import ControlAction  # noqa: E402
from microgrid_simulator.digital_twin.replay import load_fixed_telemetry_window  # noqa: E402
from microgrid_simulator.experiments.pymgrid_scenario_import import (  # noqa: E402
    load_native_pymgrid,
    translate_pymgrid_scenario,
    write_translated_config,
)
from microgrid_simulator.model.reward import compute_reward  # noqa: E402
from microgrid_simulator.ui.rollout import run_rollout  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = Path(
    "/home/xcurv/teep-taiwan/python-microgrid/src/pymgrid/data/scenario/"
    "pymgrid25/microgrid_2/microgrid_2.yaml"
)
TRANSLATED_YAML = REPO_ROOT / "archive/scenarios/pymgrid25-scenario-2.yaml"


def test_checked_in_scenario_matches_fresh_native_translation() -> None:
    expected = translate_pymgrid_scenario(SOURCE_YAML, scenario_number=2)
    actual = yaml.safe_load(TRANSLATED_YAML.read_text(encoding="utf-8"))

    assert actual == expected
    settings = Settings(**actual)
    assert settings.external_reference is not None
    assert settings.external_reference.reference_priority is True
    assert settings.battery.capacity_mwh == pytest.approx(66.116)
    assert settings.battery.model == "pymgrid"
    assert settings.battery.limit_basis == "internal_energy_per_step"
    assert settings.battery.max_charge_internal_mwh_per_step == pytest.approx(16.529)
    assert settings.battery.max_discharge_internal_mwh_per_step == pytest.approx(16.529)
    assert settings.battery.max_charge_mw == pytest.approx(18.365555555555556)
    assert settings.battery.max_discharge_mw == pytest.approx(14.8761)
    assert settings.battery.degradation_enabled is False
    assert [bus.role for bus in settings.buses] == ["main", "pv", "load", "battery"]
    assert len(settings.lines) == 3
    assert settings.pv_arrays[0].bus == 1
    assert settings.loads[0].bus == 2
    assert settings.battery.bus == 3
    assert settings.diesel.bus == 0
    assert settings.diesel.initial_on is True
    assert settings.reward.mode == "pymgrid"


def test_translated_logical_feeders_converge_for_dashboard_ac_rollout() -> None:
    settings = Settings.from_yaml(TRANSLATED_YAML)
    settings.backend.name = "pandapower"
    settings.topology.solver = "ac"

    result = run_rollout(settings, policy="rule", seed=0)

    assert len(result["rows"]) == 24
    assert all(row["solver_ok"] for row in result["rows"])
    assert min(row["min_voltage_pu"] for row in result["rows"]) >= 0.95
    assert max(row["max_line_loading_pct"] for row in result["rows"]) <= 100.0


def test_translated_telemetry_is_native_pymgrid_data() -> None:
    settings = Settings.from_yaml(TRANSLATED_YAML)
    window = load_fixed_telemetry_window(settings, n_steps=24)
    native = load_native_pymgrid(SOURCE_YAML)

    assert window is not None
    native_load_mw = -np.asarray(native.modules["load"][0].time_series[:24]).reshape(-1) / 1000.0
    native_pv_mw = np.asarray(native.modules["pv"][0].time_series[:24]).reshape(-1) / 1000.0
    assert window.demand_mw[0] == pytest.approx(native_load_mw[0])
    assert window.pv_mw[0] == pytest.approx(native_pv_mw[0])
    np.testing.assert_allclose(window.demand_mw[1:], native_load_mw, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(window.pv_mw[1:], native_pv_mw, rtol=0.0, atol=1e-12)
    assert str(window.context_timestamp) == "1999-12-31 23:00:00"
    assert str(window.first_evaluated_timestamp) == "2000-01-01 00:00:00"


def test_generated_yaml_round_trips_through_project_settings(tmp_path: Path) -> None:
    raw = translate_pymgrid_scenario(SOURCE_YAML, scenario_number=2)
    destination = write_translated_config(raw, tmp_path / "scenario.yaml")
    settings = Settings.from_yaml(destination)

    assert settings.scenario.name == "pymgrid25-scenario-2"
    assert settings.topology.timestep_hours == 1.0
    assert settings.external_reference is not None
    assert settings.external_reference.source_yaml == str(SOURCE_YAML)


def test_first_native_step_matches_project_dispatch_and_reward() -> None:
    settings = Settings.from_yaml(TRANSLATED_YAML)
    window = load_fixed_telemetry_window(settings, n_steps=24)
    assert window is not None

    backend = SimpleBackend(settings)
    assert isinstance(backend.battery, PymgridBatteryModel)
    backend.reset(demand_window_mw=window.demand_mw, pv_window_mw=window.pv_mw)
    target_load_mw = float(window.demand_mw[1])
    project_state = backend.step(ControlAction(diesel_on=True, diesel_setpoint_mw=target_load_mw))
    project_reward, _ = compute_reward(
        project_state,
        backend.battery,
        settings.reward,
        settings.topology.timestep_hours,
    )

    native = load_native_pymgrid(SOURCE_YAML)
    _, native_reward, _, native_info = native.step(
        {
            "genset": [[1.0, target_load_mw * 1000.0]],
            "battery": [0.0],
        },
        normalized=False,
    )

    assert project_state.load_demand_mw == pytest.approx(target_load_mw)
    assert project_state.diesel_p_mw == pytest.approx(
        native_info["genset"][0]["provided_energy"] / 1000.0
    )
    assert project_state.unserved_mw == pytest.approx(
        native_info["unbalanced_energy"][0]["provided_energy"] / 1000.0
    )
    assert project_state.soc[0] == pytest.approx(float(native.modules["battery"][0].soc))
    assert project_reward == pytest.approx(native_reward)


def test_native_charge_limit_excess_and_cycle_cost_match_project() -> None:
    settings = Settings.from_yaml(TRANSLATED_YAML)
    window = load_fixed_telemetry_window(settings, n_steps=24)
    assert window is not None

    backend = SimpleBackend(settings)
    backend.reset(demand_window_mw=window.demand_mw, pv_window_mw=window.pv_mw)
    project_state = backend.step(
        ControlAction(
            diesel_on=True,
            diesel_setpoint_mw=settings.diesel.max_kw / 1000.0,
            battery_p_mw=settings.battery.max_charge_mw,
        )
    )
    project_reward, _ = compute_reward(
        project_state,
        backend.battery,
        settings.reward,
        settings.topology.timestep_hours,
    )

    native = load_native_pymgrid(SOURCE_YAML)
    _, native_reward, _, native_info = native.step(
        {
            "genset": [[1.0, settings.diesel.max_kw]],
            "battery": [-settings.battery.max_charge_mw * 1000.0],
        },
        normalized=False,
    )

    assert project_state.battery_p_mw == pytest.approx(
        native_info["battery"][0]["absorbed_energy"] / 1000.0
    )
    assert project_state.soc[0] == pytest.approx(float(native.modules["battery"][0].soc))
    assert project_reward == pytest.approx(native_reward)


def test_dedicated_battery_transition_sequence_matches_native_module() -> None:
    settings = Settings.from_yaml(TRANSLATED_YAML)
    project = PymgridBatteryModel.from_cfg(settings.battery)
    native = load_native_pymgrid(SOURCE_YAML).modules["battery"][0]

    for request_mw in (100.0, 100.0, -100.0, -100.0, 100.0):
        previous_internal_mwh = project.current_charge_mwh
        project_applied_mw, project_degradation = project.apply(request_mw, 1.0)
        _, native_reward, _, native_info = native.step(
            -request_mw * 1000.0,
            normalized=False,
        )
        info_key = "absorbed_energy" if project_applied_mw >= 0.0 else "provided_energy"
        native_applied_mw = native_info[info_key] / 1000.0
        if project_applied_mw < 0.0:
            native_applied_mw *= -1.0

        internal_change_kwh = abs(project.current_charge_mwh - previous_internal_mwh) * 1000.0
        expected_reward = -internal_change_kwh * settings.reward.pymgrid_battery_cost_cycle
        assert project_applied_mw == pytest.approx(native_applied_mw)
        assert project.current_charge_mwh == pytest.approx(float(native.current_charge) / 1000.0)
        assert project.soc == pytest.approx(float(native.soc))
        assert project_degradation == 0.0
        assert native_reward == pytest.approx(expected_reward)
