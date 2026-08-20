#!/usr/bin/env python3
"""Evaluate frozen SAC-F0 checkpoints under counterfactual forecast masking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from microgrid_simulator.config import load_settings
from microgrid_simulator.ui.rollout import run_rollout

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "E1": REPO_ROOT / "configs" / "economic-e1-high-diesel-fuel.yaml",
    "E3": REPO_ROOT / "configs" / "economic-e3-high-fuel-low-battery-use.yaml",
}
WINDOWS = json.loads(
    (REPO_ROOT / "configs" / "march-windows-2026.json").read_text(encoding="utf-8")
)["starts"]


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dt_hours = 0.25
    unserved_rows = [row for row in rows if float(row["unserved_kw"]) > 1e-6]
    diesel_starts = sum(
        bool(row["diesel_on"]) and (index == 0 or not bool(rows[index - 1]["diesel_on"]))
        for index, row in enumerate(rows)
    )
    return {
        "unserved_hours": len(unserved_rows) * dt_hours,
        "diesel_starts": diesel_starts,
        "diesel_requested_on_steps": sum(bool(row["requested_diesel_on"]) for row in rows),
        "diesel_on_steps": sum(bool(row["diesel_on"]) for row in rows),
        "minimum_soc_pct": min(float(row["soc_pct"]) for row in rows),
        "soc_floor_unserved_steps": sum(
            float(row["soc_pct"]) <= 10.1 for row in unserved_rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed SAC-F0 models with recorded or zeroed forecast observations"
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--forecast-input", choices=["recorded", "zero"], required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: reports/experiments/forecast_diagnostics/<name>.json)",
    )
    args = parser.parse_args()

    output = args.output or (
        REPO_ROOT
        / "reports"
        / "experiments"
        / "forecast_diagnostics"
        / f"{args.scenario.lower()}_sac_f0_{args.forecast_input}.json"
    )
    results: list[dict[str, Any]] = []

    for seed in range(3):
        artifact = (
            REPO_ROOT
            / "artifacts"
            / "sac"
            / args.scenario
            / "cached-1m"
            / f"seed-{seed}"
            / "sac_microgrid.zip"
        )
        if not artifact.exists():
            raise FileNotFoundError(f"Missing SAC-F0 artifact: {artifact}")

        for window in WINDOWS:
            settings = load_settings(SCENARIOS[args.scenario])
            settings.episode.telemetry_start = window
            settings.episode.horizon_hours = 72.0
            settings.rl.forecast_mode = (
                "cached" if args.forecast_input == "recorded" else "none"
            )
            rollout = run_rollout(
                settings,
                policy="rl",
                seed=seed,
                rl_artifact=artifact,
                rl_algo="sac",
            )
            results.append(
                {
                    "scenario": args.scenario,
                    "forecast_input": args.forecast_input,
                    "seed": seed,
                    "window": window,
                    "reward": rollout["totals"]["total_reward"],
                    "unserved_kwh": rollout["totals"]["unserved_kwh"],
                    "diesel_kwh": rollout["totals"]["diesel_kwh"],
                    **_summarize_rows(rollout["rows"]),
                }
            )
            print(
                f"{args.scenario} seed={seed} window={window} "
                f"unserved={results[-1]['unserved_kwh']:.2f} kWh"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} diagnostic runs to {output}")


if __name__ == "__main__":
    main()
