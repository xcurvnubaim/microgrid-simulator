#!/usr/bin/env python3
"""Run deterministic February validation for Module 6 controller artifacts.

Evaluates every specified condition on identical, non-overlapping 72-hour
February telemetry windows using pandapower AC power flow:
  - 2026-02-06 00:00:00
  - 2026-02-09 00:00:00
  - 2026-02-12 00:00:00
  - 2026-02-15 00:00:00
  - 2026-02-18 00:00:00
  - 2026-02-21 00:00:00

Exports per-step trajectories and aggregates paired metrics across windows/seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microgrid_simulator.config import Settings
from microgrid_simulator.env import MicrogridEnv
from microgrid_simulator.ui.rollout import (
    _dispatch_trace,
    _load_rl_model,
    _slack_bus_id,
    _step_row,
    _totals,
    policy_action,
)

LOGGER = logging.getLogger("feb_validation")

FEB_WINDOWS = (
    "2026-02-06 00:00:00",
    "2026-02-09 00:00:00",
    "2026-02-12 00:00:00",
    "2026-02-15 00:00:00",
    "2026-02-18 00:00:00",
    "2026-02-21 00:00:00",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic February validation across SAC seeds & baselines."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/feb_validation_2026-08-07"),
        help="Directory to save validation outputs",
    )
    parser.add_argument(
        "--include-hardunserved",
        action="store_true",
        help="Also run the hard-unserved stress variant as a diagnostic",
    )
    return parser.parse_args()


def load_base_settings(hard_unserved: bool = False) -> Settings:
    config_file = (
        "configs/islanded-baseline-72h-hardunserved.yaml"
        if hard_unserved
        else "configs/islanded-baseline-72h.yaml"
    )
    return Settings.from_yaml(config_file)


def rollout_episode(
    settings: Settings,
    policy_name: str,
    forecast_mode: str,
    start_ts: str,
    artifact_path: Path | None = None,
    hard_unserved: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Clone settings so telemetry_start and forecast_mode are set strictly
    cfg_dump = settings.model_dump()
    cfg_dump["episode"]["telemetry_start"] = start_ts
    if "rl" in cfg_dump:
        cfg_dump["rl"]["forecast_mode"] = forecast_mode
        if hard_unserved:
            cfg_dump["rl"]["hard_unserved"] = True
    env_settings = Settings(**cfg_dump)

    env = MicrogridEnv(settings=env_settings, backend_name="pandapower")
    env.reset()
    slack_id = _slack_bus_id(env_settings)

    mpc_ctrl = None
    if policy_name == "mpc":
        from microgrid_simulator.backends.pypsa_backend import PyPSAOperationalBackend
        from microgrid_simulator.controllers.pypsa_mpc import PyPSAMPCController
        from microgrid_simulator.forecast.cache import ForecastCache

        pypsa_backend = PyPSAOperationalBackend(env_settings)
        # Feed the MPC the same real telemetry window the env replays, so the
        # PyPSA forecast is not synthetic (constant/zero) and the comparison is fair.
        win = getattr(env, "telemetry_window", None)
        pypsa_backend.reset(
            demand_window_mw=win.demand_mw if win is not None else None,
            pv_window_mw=win.pv_mw if win is not None else None,
        )
        cache: ForecastCache | None = None
        if forecast_mode == "cached" and env_settings.forecast.strict_cache:
            cache = ForecastCache.load(
                env_settings.forecast.cache_path,
                env_settings.forecast.manifest_path,
                expected_source_id=(
                    env_settings.forecast.source_id or env_settings.scenario.name
                ),
                action_interval_hours=env_settings.topology.timestep_hours,
            )
        mpc_ctrl = PyPSAMPCController(backend=pypsa_backend, forecast_cache=cache)

    rl_model, rl_norm = None, None
    if policy_name == "rl":
        if artifact_path is None or not artifact_path.exists():
            raise FileNotFoundError(f"Missing artifact for RL: {artifact_path}")
        rl_model, rl_norm = _load_rl_model(
            artifact_path, env_settings.rl.algo, env_settings
        )

    rows: list[dict[str, Any]] = []
    total_reward = 0.0
    ep_done = False
    step = 0

    while not ep_done and step < env.max_steps:
        step += 1
        if policy_name == "mpc":
            assert mpc_ctrl is not None
            action = env.encode_action(mpc_ctrl.act(env._last_state))  # noqa: SLF001
        else:
            action = policy_action(
                env, policy_name, rl_model=rl_model, rl_norm=rl_norm
            )

        dispatch = _dispatch_trace(env, policy_name, action)
        _, reward_val, terminated, truncated, step_info = env.step(action)
        ep_done = terminated or truncated
        total_reward += float(reward_val)

        row = _step_row(
            env_settings, env, step - 1, reward_val, step_info, slack_id, dispatch
        )
        rows.append(row)

    env.close()

    dt = env_settings.topology.timestep_hours
    totals = _totals(rows, dt)
    totals["total_reward"] = total_reward
    totals["steps_executed"] = step
    totals["early_termination"] = step < 288

    return rows, totals


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    trajectories_dir = out_dir / "trajectories"
    logs_dir.mkdir(exist_ok=True)
    trajectories_dir.mkdir(exist_ok=True)

    state_file = out_dir / "state.json"
    summary_file = out_dir / "feb_validation_summary.json"

    # Define all condition configurations
    base_settings = load_base_settings(hard_unserved=False)

    conditions: list[dict[str, Any]] = [
        {
            "condition_id": "rule_baseline",
            "policy": "rule",
            "forecast_mode": "cached",
            "artifact": None,
            "hard_unserved": False,
        },
        {
            "condition_id": "mpc_baseline",
            "policy": "mpc",
            "forecast_mode": "cached",
            "artifact": None,
            "hard_unserved": False,
        },
    ]

    # RL conditions
    for seed in (0, 1, 2):
        conditions.append(
            {
                "condition_id": f"cached_sac_seed_{seed}",
                "policy": "rl",
                "forecast_mode": "cached",
                "artifact": Path(
                    f"artifacts/sac/cached-1m-v3/seed-{seed}/sac_microgrid.zip"
                ),
                "hard_unserved": False,
            }
        )
        conditions.append(
            {
                "condition_id": f"noforecast_sac_seed_{seed}",
                "policy": "rl",
                "forecast_mode": "none",
                "artifact": Path(
                    f"artifacts/sac/noforecast-1m/seed-{seed}/sac_microgrid.zip"
                ),
                "hard_unserved": False,
            }
        )
        if args.include_hardunserved:
            conditions.append(
                {
                    "condition_id": f"hardunserved_sac_seed_{seed}",
                    "policy": "rl",
                    "forecast_mode": "cached",
                    "artifact": Path(
                        f"artifacts/sac/hardunserved-500k-v3/seed-{seed}/sac_microgrid.zip"
                    ),
                    "hard_unserved": True,
                }
            )

    LOGGER.info(
        "Starting February validation for %d conditions across %d windows...",
        len(conditions),
        len(FEB_WINDOWS),
    )

    all_results: list[dict[str, Any]] = []

    for cond in conditions:
        cid = cond["condition_id"]
        policy = cond["policy"]
        fmode = cond["forecast_mode"]
        art = cond["artifact"]
        hu = cond["hard_unserved"]

        LOGGER.info("Evaluating condition: %s", cid)
        cond_totals: list[dict[str, Any]] = []

        for window_start in FEB_WINDOWS:
            w_tag = window_start[:10]
            LOGGER.info(" -> window: %s", w_tag)

            try:
                rows, totals = rollout_episode(
                    settings=base_settings,
                    policy_name=policy,
                    forecast_mode=fmode,
                    start_ts=window_start,
                    artifact_path=art,
                    hard_unserved=hu,
                )
            except Exception as exc:
                LOGGER.error(
                    "FAILED condition %s on window %s: %s", cid, w_tag, exc
                )
                with open(state_file, "w") as sf:
                    json.dump(
                        {
                            "status": "failed",
                            "condition": cid,
                            "window": w_tag,
                            "error": str(exc),
                        },
                        sf,
                        indent=2,
                    )
                return 1

            # Save per-step CSV
            csv_path = trajectories_dir / f"{cid}_{w_tag}.csv"
            keys = [
                k
                for k in rows[0].keys()
                if not isinstance(rows[0][k], (list, dict))
            ]
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                for r in rows:
                    row_copy = {k: r[k] for k in keys}
                    writer.writerow(row_copy)

            totals["window_start"] = window_start
            cond_totals.append(totals)

        # Aggregate across windows
        mean_reward = float(np.mean([t["total_reward"] for t in cond_totals]))
        std_reward = float(np.std([t["total_reward"] for t in cond_totals]))
        mean_served_pct = float(
            np.mean([t["served_energy_pct"] for t in cond_totals])
        )
        sum_unserved_kwh = float(
            sum(t["unserved_kwh"] for t in cond_totals)
        )
        mean_unserved_kwh = float(
            np.mean([t["unserved_kwh"] for t in cond_totals])
        )
        mean_diesel_kwh = float(
            np.mean([t["diesel_kwh"] for t in cond_totals])
        )
        mean_pv_used_kwh = float(
            np.mean([t["pv_used_kwh"] for t in cond_totals])
        )
        mean_pv_wasted_kwh = float(
            np.mean([t["pv_wasted_kwh"] for t in cond_totals])
        )
        mean_carbon_kg = float(np.mean([t["carbon_kg"] for t in cond_totals]))
        mean_soc_diff = float(
            np.mean([abs(t["final_soc_pct"] - 50.0) for t in cond_totals])
        )

        all_results.append(
            {
                "condition_id": cid,
                "policy": policy,
                "forecast_mode": fmode,
                "hard_unserved": hu,
                "windows_evaluated": len(cond_totals),
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_served_energy_pct": mean_served_pct,
                "sum_unserved_kwh": sum_unserved_kwh,
                "mean_unserved_kwh_per_episode": mean_unserved_kwh,
                "mean_diesel_kwh": mean_diesel_kwh,
                "mean_pv_used_kwh": mean_pv_used_kwh,
                "mean_pv_wasted_kwh": mean_pv_wasted_kwh,
                "mean_carbon_kg": mean_carbon_kg,
                "mean_abs_soc_deviation_pct": mean_soc_diff,
                "per_window_totals": cond_totals,
            }
        )

    # Save summary json
    with open(summary_file, "w") as sf:
        json.dump({"results": all_results}, sf, indent=2)

    # Save summary CSV table for paper view
    csv_summary_file = out_dir / "feb_validation_metrics.csv"
    table_rows = []
    for res in all_results:
        table_rows.append(
            {
                "condition_id": res["condition_id"],
                "policy": res["policy"],
                "forecast_mode": res["forecast_mode"],
                "mean_reward": f"{res['mean_reward']:.2f}",
                "std_reward": f"{res['std_reward']:.2f}",
                "served_pct": f"{res['mean_served_energy_pct']:.3f}%",
                "unserved_kwh_per_ep": f"{res['mean_unserved_kwh_per_episode']:.3f}",
                "diesel_kwh": f"{res['mean_diesel_kwh']:.2f}",
                "pv_used_kwh": f"{res['mean_pv_used_kwh']:.2f}",
                "pv_wasted_kwh": f"{res['mean_pv_wasted_kwh']:.2f}",
                "carbon_kg": f"{res['mean_carbon_kg']:.2f}",
                "soc_dev_pct": f"{res['mean_abs_soc_deviation_pct']:.2f}%",
            }
        )
    df_summary = pd.DataFrame(table_rows)
    df_summary.to_csv(csv_summary_file, index=False)

    with open(state_file, "w") as sf:
        json.dump({"status": "completed"}, sf, indent=2)

    LOGGER.info(
        "February validation completed successfully. Summary saved to %s",
        summary_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
