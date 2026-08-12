#!/usr/bin/env python3
"""Evaluate high-fuel SAC artifacts on fixed February validation windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_feb_validation_suite import FEB_WINDOWS, rollout_episode

from microgrid_simulator.config import Settings

OUTPUT_DIR = Path("reports/experiments/high_fuel_battery_pv_feb_2026-08-07")
ARTIFACT_DIR = Path("artifacts/sac/high-fuel-battery-pv-1m")


def main() -> None:
    settings = Settings.from_yaml("configs/islanded-high-fuel-battery-pv-72h.yaml")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for seed in (0, 1, 2):
        artifact = ARTIFACT_DIR / f"seed-{seed}" / "sac_microgrid.zip"
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        episodes = []
        for window in FEB_WINDOWS:
            _, totals = rollout_episode(
                settings,
                "rl",
                "cached",
                window,
                artifact_path=artifact,
            )
            episodes.append(totals)

        results.append(
            {
                "condition_id": f"high_fuel_sac_seed_{seed}",
                "served_pct": float(np.mean([e["served_energy_pct"] for e in episodes])),
                "unserved_kwh_per_ep": float(np.mean([e["unserved_kwh"] for e in episodes])),
                "diesel_kwh_per_ep": float(np.mean([e["diesel_kwh"] for e in episodes])),
                "battery_throughput_kwh_per_ep": float(
                    np.mean([e["battery_throughput_kwh"] for e in episodes])
                ),
                "pv_used_kwh_per_ep": float(np.mean([e["pv_used_kwh"] for e in episodes])),
                "pv_wasted_kwh_per_ep": float(
                    np.mean([e["pv_wasted_kwh"] for e in episodes])
                ),
                "final_soc_deviation_pp": float(
                    np.mean([abs(e["final_soc_pct"] - 50.0) for e in episodes])
                ),
                "constraint_violations": int(
                    sum(e["constraint_violation_events"] for e in episodes)
                ),
            }
        )

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps({"results": results}, indent=2))
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    (OUTPUT_DIR / "state.json").write_text(json.dumps({"status": "completed"}, indent=2))


if __name__ == "__main__":
    main()
