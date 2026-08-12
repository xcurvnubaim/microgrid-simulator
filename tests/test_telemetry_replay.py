"""Aligned measured load/PV replay and paper-report contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microgrid_simulator.cli import _replay_output_dir
from microgrid_simulator.config import Settings
from microgrid_simulator.controllers.rule_based import RuleBasedController
from microgrid_simulator.core.types import GridState
from microgrid_simulator.digital_twin.alignment import AlignmentError
from microgrid_simulator.digital_twin.replay import load_fixed_telemetry_window
from microgrid_simulator.experiments.telemetry_replay import run_telemetry_replay
from microgrid_simulator.rl.env import MicrogridEnv


def _telemetry_settings(tmp_path, *, horizon_hours: float = 1.0) -> Settings:
    timestamps = pd.date_range("2026-01-14 23:45", periods=8, freq="15min")
    load_path = tmp_path / "load.csv"
    pv_path = tmp_path / "pv.csv"
    pd.DataFrame({"ts": timestamps, "kw": np.arange(8) + 100.0}).to_csv(load_path, index=False)
    pd.DataFrame({"ts": timestamps, "kw": [-1.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}).to_csv(
        pv_path, index=False
    )

    raw = Settings().model_dump()
    raw["topology"] |= {"timestep_hours": 0.25, "solver": "balance", "n_ev": 0}
    raw["backend"] |= {"name": "simple"}
    raw["episode"] |= {
        "horizon_hours": horizon_hours,
        "telemetry_start": "2026-01-15 00:00:00",
    }
    raw["demand"] |= {"enabled": False, "file": None, "random_window": False}
    raw["digital_twin"] = {
        "fill_strategy": "error",
        "max_gap_steps": 0,
        "max_missing_fraction": 0.0,
        "measurements": {
            "load": {
                "file": str(load_path),
                "timestamp_column": "ts",
                "value_column": "kw",
                "unit": "kw",
            },
            "pv": {
                "file": str(pv_path),
                "timestamp_column": "ts",
                "value_column": "kw",
                "unit": "kw",
            },
        },
    }
    return Settings(**raw)


def test_default_replay_output_uses_configured_telemetry_date(tmp_path) -> None:
    settings = _telemetry_settings(tmp_path)
    assert _replay_output_dir(settings, "mpc").as_posix().endswith("islanded_72h_mpc_2026-01-15")


def test_heldout_scenario_changes_only_identity_and_telemetry_window() -> None:
    baseline = Settings.from_yaml(Path("configs/islanded-baseline-72h.yaml")).model_dump()
    heldout = Settings.from_yaml(Path("configs/islanded-heldout-72h.yaml")).model_dump()

    baseline["scenario"] = heldout["scenario"]
    baseline["episode"]["telemetry_start"] = heldout["episode"]["telemetry_start"]

    assert baseline == heldout


def test_high_fuel_scenario_inherits_baseline_and_overrides_dispatch_costs() -> None:
    baseline = Settings.from_yaml(Path("configs/islanded-baseline-72h.yaml"))
    high_fuel = Settings.from_yaml(Path("configs/islanded-high-fuel-battery-pv-72h.yaml"))

    assert high_fuel.scenario.name == "islanded_high_fuel_battery_pv_72h"
    assert high_fuel.reward.diesel_fuel_cost_per_kwh == 1.20
    assert high_fuel.reward.w_health < baseline.reward.w_health
    assert high_fuel.reward.w_waste > baseline.reward.w_waste
    assert high_fuel.episode.terminal_soc_penalty > baseline.episode.terminal_soc_penalty
    assert high_fuel.battery == baseline.battery
    assert high_fuel.digital_twin == baseline.digital_twin


def test_fixed_replay_uses_context_then_exact_evaluated_samples(tmp_path) -> None:
    settings = _telemetry_settings(tmp_path)
    window = load_fixed_telemetry_window(settings, n_steps=4)
    assert window is not None
    assert window.timestamps_are_observed is True
    assert list(window.demand_mw * 1000.0) == pytest.approx([100, 101, 102, 103, 104])
    assert list(window.pv_mw * 1000.0) == pytest.approx([0, 0, 10, 20, 30])

    env = MicrogridEnv(settings=settings)
    _, info = env.reset(seed=0)
    assert info["telemetry_context_time"] == "2026-01-14 23:45:00"
    assert env._last_state.load_demand_mw == pytest.approx(0.100)  # noqa: SLF001
    action = np.zeros(env.action_dim, dtype=np.float32)
    env.step(action)
    assert env._last_state.load_demand_mw == pytest.approx(0.101)  # noqa: SLF001
    assert env._last_state.pv_available_mw == pytest.approx(0.0)  # noqa: SLF001
    assert env._last_state.demand_is_real is True  # noqa: SLF001
    assert env._last_state.pv_is_real is True  # noqa: SLF001
    env.close()


def test_positional_reference_replay_marks_its_clock_as_synthetic(tmp_path) -> None:
    settings = _telemetry_settings(tmp_path)
    raw = settings.model_dump()
    for name in ("load", "pv"):
        path = tmp_path / f"{name}-positional.csv"
        pd.DataFrame({"value": np.arange(8) + 1.0}).to_csv(path, index=False)
        raw["digital_twin"]["measurements"][name] = {
            "file": str(path),
            "timestamp_column": None,
            "value_column": "value",
            "unit": "kw",
            "synthetic_start": "2000-01-01T00:00:00",
            "synthetic_step_hours": 0.25,
            "prepend_first_as_context": True,
        }
    raw["episode"]["telemetry_start"] = "2000-01-01T00:00:00"

    window = load_fixed_telemetry_window(Settings(**raw), n_steps=4)

    assert window is not None
    assert window.timestamps_are_observed is False


def test_fixed_replay_fails_when_pv_measurement_is_not_configured(tmp_path) -> None:
    settings = _telemetry_settings(tmp_path)
    raw = settings.model_dump()
    del raw["digital_twin"]["measurements"]["pv"]
    with pytest.raises(AlignmentError, match="missing.*pv"):
        load_fixed_telemetry_window(Settings(**raw), n_steps=4)


def test_rule_plans_from_available_pv_not_previous_curtailed_pv() -> None:
    settings = Settings()
    state = GridState(
        load_demand_mw=0.10,
        pv_available_mw=0.20,
        pv_used_mw=0.0,
        islanded=True,
    )

    action = RuleBasedController(settings).act(state)

    assert action.battery_p_mw == pytest.approx(0.10)
    assert action.diesel_on is False


def test_paper_report_writes_288_measured_intervals(tmp_path) -> None:
    settings = _telemetry_settings(tmp_path, horizon_hours=72.0)
    # Supply enough exact samples for the 72 h contract.
    timestamps = pd.date_range("2026-01-14 23:45", periods=289, freq="15min")
    for name, value in (("load", 100.0), ("pv", 20.0)):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"ts": timestamps, "kw": value}).to_csv(path, index=False)
        settings.digital_twin.measurements[name].file = str(path)
    for bus in settings.buses:
        if bus.role == "grid":
            bus.role = "main"
    settings.diesel.enabled = False

    output = tmp_path / "report"
    metrics = run_telemetry_replay(
        settings, tmp_path / "config.yaml", output, policy="rule", seed=0
    )

    trajectory = pd.read_csv(output / "trajectory.csv")
    stored = json.loads((output / "metrics.json").read_text())
    assert len(trajectory) == 288
    assert trajectory["dispatch_policy"].eq("rule").all()
    assert trajectory["dispatch_rule"].str.startswith("islanded rule:").all()
    assert {
        "requested_battery_kw",
        "requested_diesel_on",
        "requested_diesel_kw",
        "requested_pv_curtailment_pct",
    }.issubset(trajectory.columns)
    assert metrics["meta"]["demand_is_real"] is True
    assert metrics["meta"]["pv_is_real"] is True
    assert stored["totals"]["load_kwh"] == pytest.approx(7200.0)
    assert (output / "report.md").is_file()
    assert (output / "visualization.png").stat().st_size > 10_000


def test_paper_report_accepts_672_measured_intervals(tmp_path) -> None:
    settings = _telemetry_settings(tmp_path, horizon_hours=168.0)
    timestamps = pd.date_range("2026-01-14 23:45", periods=673, freq="15min")
    for name, value in (("load", 100.0), ("pv", 20.0)):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"ts": timestamps, "kw": value}).to_csv(path, index=False)
        settings.digital_twin.measurements[name].file = str(path)
    for bus in settings.buses:
        if bus.role == "grid":
            bus.role = "main"
    settings.diesel.enabled = False

    output = tmp_path / "report-168h"
    metrics = run_telemetry_replay(
        settings, tmp_path / "config.yaml", output, policy="rule", seed=0
    )

    trajectory = pd.read_csv(output / "trajectory.csv")
    assert len(trajectory) == 672
    assert metrics["meta"]["evaluation_end_exclusive"] == "2026-01-22 00:00:00"
