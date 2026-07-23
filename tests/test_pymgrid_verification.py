"""Cross-implementation verification of the shared pymgrid scheduling contract."""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("pymgrid")

from microgrid_simulator.experiments.pymgrid_verification import (  # noqa: E402
    AcceptanceCriteria,
    CommonPlant,
    FixtureStep,
    _compare,
    _pymgrid_battery_action_mwh,
    _run_pymgrid,
    _run_simple,
    common_fixture,
    run_verification,
)


def test_pymgrid_action_translation_does_not_preclip_request() -> None:
    charge_request = FixtureStep(
        "charge",
        load_mw=0.2,
        pv_available_mw=0.5,
        battery_p_mw=0.4,
    )
    discharge_request = FixtureStep(
        "discharge",
        load_mw=0.4,
        pv_available_mw=0.1,
        battery_p_mw=-0.4,
    )

    assert _pymgrid_battery_action_mwh(charge_request, 1.0) == pytest.approx(-0.4)
    assert _pymgrid_battery_action_mwh(discharge_request, 1.0) == pytest.approx(0.4)


def test_common_model_matches_pymgrid(tmp_path) -> None:
    results = run_verification(tmp_path)

    assert results["1.00"]["passed"] is True
    assert results["0.96"]["passed"] is True
    assert (tmp_path / "microgrid_simulator_efficiency_1.00.csv").is_file()
    assert (tmp_path / "pymgrid_efficiency_1.00.csv").is_file()
    assert (tmp_path / "trajectory_efficiency_1.00.csv").is_file()
    assert (tmp_path / "trajectory_efficiency_0.96.csv").is_file()
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["overall_passed"] is True
    assert payload["comparison_policy"]["reference_implementation"] == "pymgrid"
    assert payload["comparison_policy"]["candidate_implementation"] == "microgrid-simulator"
    assert payload["comparison_policy"]["silent_output_substitution"] is False
    assert payload["results"]["1.00"]["failed_fields"] == []
    assert payload["results"]["1.00"]["failed_aggregate_fields"] == []

    comparison = pd.read_csv(tmp_path / "trajectory_efficiency_1.00.csv").set_index("fixture")
    charge = comparison.loc["charge_request_clipped"]
    discharge = comparison.loc["discharge_request_clipped"]
    assert charge["simulator_battery_p_mw"] == pytest.approx(0.3)
    assert charge["pymgrid_battery_p_mw"] == pytest.approx(0.3)
    assert discharge["simulator_battery_p_mw"] == pytest.approx(-0.3)
    assert discharge["pymgrid_battery_p_mw"] == pytest.approx(-0.3)


def test_mismatch_rejects_project_candidate_in_favor_of_pymgrid() -> None:
    plant = CommonPlant()
    fixtures = common_fixture()
    candidate_rows = _run_simple(plant, fixtures, efficiency=1.0)
    reference_rows = _run_pymgrid(plant, fixtures, efficiency=1.0)
    candidate_rows[0]["served_mw"] += 0.01

    _, metrics = _compare(
        candidate_rows,
        reference_rows,
        plant,
        AcceptanceCriteria(),
    )

    assert metrics["passed"] is False
    assert metrics["disposition"] == "candidate_requires_review"
    assert metrics["reference_implementation"] == "pymgrid"
    assert metrics["candidate_implementation"] == "microgrid-simulator"
    assert "served_mw" in metrics["failed_fields"]
