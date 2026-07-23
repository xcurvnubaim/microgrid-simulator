"""Translate a native pymgrid25 YAML scenario into simulator configuration.

The native pymgrid YAML remains authoritative. The generated project YAML
references its compressed time series in place and records every unit/sign
conversion needed by the project simulator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from microgrid_simulator.config import Settings
from microgrid_simulator.experiments.pymgrid_verification import PYMGRID_UPSTREAM_COMMIT


def _one_module(microgrid: Any, name: str) -> Any:
    modules = list(microgrid.modules[name])
    if len(modules) != 1:
        raise ValueError(f"expected exactly one {name!r} module, found {len(modules)}")
    return modules[0]


def load_native_pymgrid(source_yaml: str | Path) -> Any:
    """Load the authoritative custom-tagged pymgrid YAML."""

    try:
        from pymgrid import Microgrid
    except ImportError as exc:  # pragma: no cover - install boundary
        raise RuntimeError("scenario import requires `uv run --extra pymgrid ...`") from exc

    source = Path(source_yaml).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"pymgrid source YAML not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        return Microgrid.load(stream)


def translate_pymgrid_scenario(
    source_yaml: str | Path,
    *,
    scenario_number: int,
    timestep_hours: float = 1.0,
    source_commit: str = PYMGRID_UPSTREAM_COMMIT,
) -> dict[str, Any]:
    """Return a project config whose plant and series match one pymgrid25 YAML."""

    if timestep_hours <= 0:
        raise ValueError("timestep_hours must be positive")
    source = Path(source_yaml).resolve()
    microgrid = load_native_pymgrid(source)
    module_names = set(microgrid.modules.names())
    required = {"load", "pv", "battery", "genset", "unbalanced_energy"}
    missing = required - module_names
    if missing:
        raise ValueError(f"unsupported pymgrid scenario; missing modules {sorted(missing)}")
    if "grid" in module_names:
        raise ValueError("this importer currently accepts islanded pymgrid scenarios only")

    load = _one_module(microgrid, "load")
    pv = _one_module(microgrid, "pv")
    battery = _one_module(microgrid, "battery")
    genset = _one_module(microgrid, "genset")
    imbalance = _one_module(microgrid, "unbalanced_energy")

    load_energy_kwh = -np.asarray(load.time_series, dtype=float).reshape(-1)
    pv_energy_kwh = np.asarray(pv.time_series, dtype=float).reshape(-1)
    if len(load_energy_kwh) != len(pv_energy_kwh):
        raise ValueError("pymgrid load and PV series have different lengths")
    if len(load_energy_kwh) < 2:
        raise ValueError("pymgrid scenario needs a context row and at least one evaluated row")
    if np.any(load_energy_kwh < 0) or np.any(pv_energy_kwh < 0):
        raise ValueError("unexpected pymgrid load/PV signs after canonical conversion")

    # Pymgrid battery power bounds are internal energy per step. The project
    # bounds are terminal MW, hence the asymmetric efficiency conversion.
    efficiency = float(battery.efficiency)
    max_charge_mw = float(battery.max_charge) / efficiency / 1000.0 / timestep_hours
    max_discharge_mw = float(battery.max_discharge) * efficiency / 1000.0 / timestep_hours
    capacity_mwh = float(battery.max_capacity) / 1000.0

    data_root = source.parent / "data" / "cls_params"
    load_file = data_root / "LoadModule" / "time_series.csv.gz"
    pv_file = data_root / "RenewableModule" / "time_series.csv.gz"
    for series_file in (load_file, pv_file):
        if not series_file.is_file():
            raise FileNotFoundError(f"pymgrid sidecar series not found: {series_file}")

    synthetic_start = "2000-01-01T00:00:00"
    first_evaluated = synthetic_start
    native_parameters = {
        "rows": int(len(load_energy_kwh)),
        "final_step": int(microgrid.final_step),
        "battery_max_capacity_kwh": float(battery.max_capacity),
        "battery_min_capacity_kwh": float(battery.min_capacity),
        "battery_max_charge_internal_kwh_per_step": float(battery.max_charge),
        "battery_max_discharge_internal_kwh_per_step": float(battery.max_discharge),
        "battery_efficiency": efficiency,
        "battery_cost_cycle": float(battery.battery_cost_cycle),
        "genset_min_kwh_per_step": float(genset.running_min_production),
        "genset_max_kwh_per_step": float(genset.running_max_production),
        "genset_cost": float(genset.genset_cost),
        "genset_co2_per_unit": float(genset.co2_per_unit),
        "genset_cost_per_unit_co2": float(genset.cost_per_unit_co2),
        "loss_load_cost": float(imbalance.loss_load_cost),
        "overgeneration_cost": float(imbalance.overgeneration_cost),
    }
    raw: dict[str, Any] = {
        "scenario": {
            "name": f"pymgrid25-scenario-{scenario_number}",
            "description": (
                "Simulator translation of the native pymgrid25 islanded scenario; "
                "pymgrid is authoritative for the declared shared contract."
            ),
        },
        "external_reference": {
            "implementation": "pymgrid",
            "benchmark": "pymgrid25",
            "scenario_number": scenario_number,
            "source_yaml": str(source),
            "source_commit": source_commit,
            "reference_priority": True,
            "native_parameters": native_parameters,
        },
        "topology": {
            "timestep_hours": timestep_hours,
            "solver": "balance",
            "n_pv": 1,
            "n_storage": 1,
            "n_ev": 0,
            "n_load": 1,
        },
        "backend": {"name": "simple", "timestep_hours": timestep_hours},
        # Pymgrid's scheduling contract is electrically single-bus. Keep that
        # balance model, but expose a feeder-style logical topology for the
        # dashboard, matching the layout used by the campus profile. The
        # benchmark reaches 54 MW, so use a 20 kV logical distribution level
        # rather than copying the campus profile's sub-MW 0.4 kV ratings.
        "buses": [
            {
                "id": 0,
                "name": "Pymgrid main distribution",
                "vn_kv": 20.0,
                "role": "main",
                "x": 290,
                "y": 150,
            },
            {
                "id": 1,
                "name": "Pymgrid PV yard",
                "vn_kv": 20.0,
                "role": "pv",
                "x": 520,
                "y": 60,
            },
            {
                "id": 2,
                "name": "Pymgrid load",
                "vn_kv": 20.0,
                "role": "load",
                "x": 760,
                "y": 150,
            },
            {
                "id": 3,
                "name": "Pymgrid battery storage",
                "vn_kv": 20.0,
                "role": "battery",
                "x": 760,
                "y": 240,
            },
        ],
        "lines": [
            {
                "name": "Pymgrid PV feeder",
                "from_bus": 0,
                "to_bus": 1,
                "kind": "line",
                "length_km": 0.18,
                "max_i_ka": 2.0,
            },
            {
                "name": "Pymgrid load feeder",
                "from_bus": 0,
                "to_bus": 2,
                "kind": "line",
                "length_km": 0.28,
                "max_i_ka": 2.0,
            },
            {
                "name": "Pymgrid battery feeder",
                "from_bus": 0,
                "to_bus": 3,
                "kind": "line",
                "length_km": 0.16,
                "max_i_ka": 2.0,
            },
        ],
        "pv_arrays": [
            {
                "name": "pymgrid renewable",
                "bus": 1,
                "p_mw": float(pv_energy_kwh.max()) / 1000.0 / timestep_hours,
            }
        ],
        "loads": [
            {
                "name": "pymgrid load",
                "bus": 2,
                "p_mw": float(load_energy_kwh.max()) / 1000.0 / timestep_hours,
                "q_mvar": 0.0,
            }
        ],
        "battery": {
            "model": "pymgrid",
            "bus": 3,
            "capacity_mwh": capacity_mwh,
            "charge_eff": efficiency,
            "discharge_eff": efficiency,
            "limit_basis": "internal_energy_per_step",
            "max_charge_internal_mwh_per_step": float(battery.max_charge) / 1000.0,
            "max_discharge_internal_mwh_per_step": float(battery.max_discharge) / 1000.0,
            "soc_min": float(battery.min_capacity) / float(battery.max_capacity),
            "soc_max": 1.0,
            "soc_init": float(battery.init_soc),
            "degradation_enabled": False,
        },
        "diesel": {
            "enabled": True,
            "bus": 0,
            "min_kw": float(genset.running_min_production) / timestep_hours,
            "max_kw": float(genset.running_max_production) / timestep_hours,
            "ramp_kw_per_min": 0.0,
            "start_delay_min": float(genset.start_up_time) * timestep_hours * 60.0,
            "min_up_time_min": 0.0,
            "min_down_time_min": 0.0,
            "initial_on": bool(genset.current_status),
            "carbon_kg_per_kwh": float(genset.co2_per_unit),
            "fuel_cost_per_kwh": float(genset.genset_cost),
        },
        "demand": {"enabled": False, "file": None, "random_window": False},
        "digital_twin": {
            "fill_strategy": "error",
            "max_gap_steps": 0,
            "max_missing_fraction": 0.0,
            "measurements": {
                "load": {
                    "file": str(load_file),
                    "timestamp_column": None,
                    "value_column": "0",
                    "unit": "kwh_per_step",
                    "header_row": 0,
                    "value_multiplier": -1.0,
                    "synthetic_start": synthetic_start,
                    "synthetic_step_hours": timestep_hours,
                    "prepend_first_as_context": True,
                },
                "pv": {
                    "file": str(pv_file),
                    "timestamp_column": None,
                    "value_column": "0",
                    "unit": "kwh_per_step",
                    "header_row": 0,
                    "value_multiplier": 1.0,
                    "synthetic_start": synthetic_start,
                    "synthetic_step_hours": timestep_hours,
                    "prepend_first_as_context": True,
                },
            },
        },
        "episode": {
            # The loader duplicates native row zero as preceding controller
            # context, so the first applied action still evaluates native row zero.
            "telemetry_start": first_evaluated,
            "start_hour": 0.0,
            "horizon_hours": min(24.0, len(load_energy_kwh) * timestep_hours),
        },
        "reward": {
            "mode": "pymgrid",
            "pymgrid_battery_cost_cycle": float(battery.battery_cost_cycle),
            "pymgrid_genset_cost": float(genset.genset_cost),
            "pymgrid_co2_per_unit": float(genset.co2_per_unit),
            "pymgrid_cost_per_unit_co2": float(genset.cost_per_unit_co2),
            "pymgrid_loss_load_cost": float(imbalance.loss_load_cost),
            "pymgrid_overgeneration_cost": float(imbalance.overgeneration_cost),
        },
    }
    settings = Settings(**raw)  # fail before writing if translated config is invalid
    if not np.isclose(settings.battery.max_charge_mw, max_charge_mw):
        raise AssertionError("derived terminal charge limit does not match pymgrid")
    if not np.isclose(settings.battery.max_discharge_mw, max_discharge_mw):
        raise AssertionError("derived terminal discharge limit does not match pymgrid")
    return raw


def write_translated_config(raw: dict[str, Any], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated from native pymgrid YAML; do not hand-edit mapped plant values.\n"
        "# Regenerate with microgrid_simulator.experiments.pymgrid_scenario_import.\n"
    )
    destination.write_text(
        header + yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-yaml", type=Path, required=True)
    parser.add_argument("--scenario-number", type=int, required=True)
    parser.add_argument("--timestep-hours", type=float, default=1.0)
    parser.add_argument("--source-commit", default=PYMGRID_UPSTREAM_COMMIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = translate_pymgrid_scenario(
        args.source_yaml,
        scenario_number=args.scenario_number,
        timestep_hours=args.timestep_hours,
        source_commit=args.source_commit,
    )
    destination = write_translated_config(raw, args.output)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
