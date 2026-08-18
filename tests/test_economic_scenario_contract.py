"""E1-E5 must differ from nominal only in declared economic fields."""

from __future__ import annotations

from pathlib import Path

import pytest

from microgrid_simulator.config import load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
E0 = CONFIG_DIR / "islanded-baseline-72h.yaml"

SCENARIOS = {
    "E1": (
        CONFIG_DIR / "economic-e1-high-diesel-fuel.yaml",
        {"reward.diesel_fuel_cost_per_kwh": (0.4, 1.2)},
    ),
    "E2": (
        CONFIG_DIR / "economic-e2-low-battery-use.yaml",
        {"reward.w_health": (0.5, 0.1)},
    ),
    "E3": (
        CONFIG_DIR / "economic-e3-high-fuel-low-battery-use.yaml",
        {
            "reward.diesel_fuel_cost_per_kwh": (0.4, 1.2),
            "reward.w_health": (0.5, 0.1),
        },
    ),
    "E4": (
        CONFIG_DIR / "economic-e4-high-pv-waste.yaml",
        {"reward.w_waste": (1.0, 2.0)},
    ),
    "E5": (
        CONFIG_DIR / "economic-e5-high-unserved.yaml",
        {"reward.w_unserved": (20.0, 100.0)},
    ),
}


def _differences(left: object, right: object, prefix: str = "") -> dict[str, tuple[object, object]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: dict[str, tuple[object, object]] = {}
        for key in sorted(left.keys() | right.keys()):
            path = f"{prefix}.{key}" if prefix else key
            differences.update(_differences(left.get(key), right.get(key), path))
        return differences
    if left != right:
        return {prefix: (left, right)}
    return {}


@pytest.mark.parametrize(("scenario_id", "overlay", "declared"), [
    (scenario_id, overlay, declared)
    for scenario_id, (overlay, declared) in SCENARIOS.items()
])
def test_economic_overlay_changes_only_declared_fields(
    scenario_id: str,
    overlay: Path,
    declared: dict[str, tuple[object, object]],
) -> None:
    nominal = load_settings(E0).model_dump(mode="json")
    scenario = load_settings(overlay).model_dump(mode="json")
    nominal["scenario"] = scenario["scenario"]

    differences = _differences(nominal, scenario)

    assert differences == declared, f"{scenario_id} has undeclared configuration changes"
    assert scenario["rl"]["hard_unserved"] is False
