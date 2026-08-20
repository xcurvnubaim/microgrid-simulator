#!/usr/bin/env python3
"""Score F3 cached PV/load forecasts against processed telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
DEFAULT_PV = Path(
    "/home/xcurv/teep-taiwan/data/processed/pv_15min_chronos_reconstruction_candidate.csv"
)
DEFAULT_DEMAND = Path(
    "/home/xcurv/teep-taiwan/data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv"
)


def _load_target(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    return pd.Series(
        frame[column].astype(float).to_numpy(),
        index=pd.DatetimeIndex(frame["timestamp"]),
        name=column,
    ).sort_index()


def _metrics(errors: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(errors) & np.isfinite(actual)
    errors = errors[finite]
    actual = actual[finite]
    return {
        "n": int(errors.size),
        "mae_kw": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        "rmse_kw": float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan"),
        "bias_kw": float(np.mean(errors)) if errors.size else float("nan"),
        "actual_mean_kw": float(np.mean(actual)) if actual.size else float("nan"),
    }


def _score_split(
    cache_path: Path,
    pv: pd.Series,
    demand: pd.Series,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    per_horizon: list[dict[str, Any]] = []
    pv_errors: list[list[float]] = []
    demand_errors: list[list[float]] = []
    net_errors: list[list[float]] = []
    pv_actuals: list[list[float]] = []
    demand_actuals: list[list[float]] = []
    net_actuals: list[list[float]] = []
    pv_ramp_errors: list[float] = []
    demand_ramp_errors: list[float] = []
    net_ramp_errors: list[float] = []

    for record in records:
        timestamps = pd.DatetimeIndex(record["timestamps"])
        actual_pv = pv.reindex(timestamps).to_numpy(dtype=float)
        actual_demand = demand.reindex(timestamps).to_numpy(dtype=float)
        forecast_pv = np.asarray(record["pv_values_kw"], dtype=float)
        forecast_demand = np.asarray(record["demand_values_kw"], dtype=float)
        actual_net = actual_demand - actual_pv
        forecast_net = forecast_demand - forecast_pv
        pv_errors.append(forecast_pv - actual_pv)
        demand_errors.append(forecast_demand - actual_demand)
        net_errors.append(forecast_net - actual_net)
        pv_actuals.append(actual_pv)
        demand_actuals.append(actual_demand)
        net_actuals.append(actual_net)
        if len(actual_pv) > 1:
            pv_ramp = np.diff(forecast_pv) - np.diff(actual_pv)
            demand_ramp = np.diff(forecast_demand) - np.diff(actual_demand)
            pv_ramp_errors.extend(pv_ramp[np.isfinite(pv_ramp)])
            demand_ramp_errors.extend(demand_ramp[np.isfinite(demand_ramp)])
            net_ramp = (
                np.diff(forecast_demand - forecast_pv)
                - np.diff(actual_demand - actual_pv)
            )
            net_ramp_errors.extend(net_ramp[np.isfinite(net_ramp)])

    pv_err = np.asarray(pv_errors)
    demand_err = np.asarray(demand_errors)
    net_err = np.asarray(net_errors)
    pv_act = np.asarray(pv_actuals)
    demand_act = np.asarray(demand_actuals)
    net_act = np.asarray(net_actuals)
    for horizon in range(pv_err.shape[1]):
        per_horizon.append(
            {
                "lead_minutes": (horizon + 1) * 15,
                "pv": _metrics(pv_err[:, horizon], pv_act[:, horizon]),
                "demand": _metrics(demand_err[:, horizon], demand_act[:, horizon]),
                "net_load": _metrics(net_err[:, horizon], net_act[:, horizon]),
            }
        )
    return {
        "records": len(records),
        "points": int(pv_err.size),
        "pv": _metrics(pv_err.ravel(), pv_act.ravel()),
        "demand": _metrics(demand_err.ravel(), demand_act.ravel()),
        "net_load": _metrics(net_err.ravel(), net_act.ravel()),
        "pv_ramp_mae_kw": float(
            np.nanmean(np.abs(np.asarray(pv_ramp_errors, dtype=float)))
        ),
        "demand_ramp_mae_kw": float(
            np.nanmean(np.abs(np.asarray(demand_ramp_errors, dtype=float)))
        ),
        "net_load_ramp_mae_kw": float(
            np.nanmean(np.abs(np.asarray(net_ramp_errors, dtype=float)))
        ),
        "by_horizon": per_horizon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data/f3"))
    parser.add_argument("--pv-path", type=Path, default=DEFAULT_PV)
    parser.add_argument("--demand-path", type=Path, default=DEFAULT_DEMAND)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/f3_forecast_scores.json"),
    )
    args = parser.parse_args()

    pv = _load_target(args.pv_path, "pv_kw")
    demand = _load_target(args.demand_path, "demand_kw")
    result = {
        "cache_root": str(args.cache_root),
        "pv_path": str(args.pv_path),
        "demand_path": str(args.demand_path),
        "target_frequency_hours": 0.25,
        "splits": {
            split: _score_split(args.cache_root / split / "forecasts.jsonl", pv, demand)
            for split in SPLITS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for split, metrics in result["splits"].items():
        print(
            f"{split}: PV MAE {metrics['pv']['mae_kw']:.2f} kW; "
            f"demand MAE {metrics['demand']['mae_kw']:.2f} kW; "
            f"net-load MAE {metrics['net_load']['mae_kw']:.2f} kW; "
            f"net-load bias {metrics['net_load']['bias_kw']:.2f} kW"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
