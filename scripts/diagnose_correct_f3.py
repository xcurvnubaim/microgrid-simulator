#!/usr/bin/env python3
"""Diagnose and validate leakage-safe residual corrections for F3."""

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
    return pd.Series(frame[column].to_numpy(float), index=pd.DatetimeIndex(frame.timestamp))


def _rows(cache: Path, pv: pd.Series, demand: pd.Series) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        issued = pd.Timestamp(record["issued_at"])
        timestamps = pd.DatetimeIndex(record["timestamps"])
        actual_pv = pv.reindex(timestamps).to_numpy(float)
        actual_demand = demand.reindex(timestamps).to_numpy(float)
        forecast_pv = np.asarray(record["pv_values_kw"], float)
        forecast_demand = np.asarray(record["demand_values_kw"], float)
        for horizon, target_time in enumerate(timestamps):
            if not np.isfinite(actual_pv[horizon] + actual_demand[horizon]):
                continue
            output.append(
                {
                    "issued_at": issued,
                    "target_time": target_time,
                    "lead": horizon,
                    "hour": target_time.hour,
                    "pv_error": forecast_pv[horizon] - actual_pv[horizon],
                    "demand_error": forecast_demand[horizon] - actual_demand[horizon],
                    "net_error": (forecast_demand[horizon] - forecast_pv[horizon])
                    - (actual_demand[horizon] - actual_pv[horizon]),
                }
            )
    return pd.DataFrame(output)


def _summary(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label, "rows": len(frame)}
    for column in ("pv_error", "demand_error", "net_error"):
        values = frame[column].to_numpy(float)
        result[column] = {
            "mae_kw": float(np.mean(np.abs(values))),
            "bias_kw": float(np.mean(values)),
            "rmse_kw": float(np.sqrt(np.mean(values**2))),
        }
    return result


def _correction(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    # Fit only on train+validation, stratified by target horizon and target-time hour.
    fit = pd.concat([train, validation], ignore_index=True)
    grouped = fit.groupby(["lead", "hour"])[["pv_error", "demand_error", "net_error"]].mean()
    global_bias = fit[["pv_error", "demand_error", "net_error"]].mean()
    return {
        "grouped": {
            f"{lead}:{hour}": {key: float(value) for key, value in row.items()}
            for (lead, hour), row in grouped.iterrows()
        },
        "global": {key: float(value) for key, value in global_bias.items()},
        "fit_rows": len(fit),
    }


def _apply(frame: pd.DataFrame, correction: dict[str, Any], mode: str) -> pd.DataFrame:
    result = frame.copy()
    if mode == "global":
        biases = correction["global"]
        for column in ("pv_error", "demand_error", "net_error"):
            result[column] -= biases[column]
        return result
    for index, row in result.iterrows():
        biases = correction["grouped"].get(
            f"{int(row.lead)}:{int(row.hour)}", correction["global"]
        )
        for column in ("pv_error", "demand_error", "net_error"):
            result.at[index, column] -= biases[column]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data/f3"))
    parser.add_argument("--pv-path", type=Path, default=DEFAULT_PV)
    parser.add_argument("--demand-path", type=Path, default=DEFAULT_DEMAND)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/f3_shift_diagnosis.json"),
    )
    args = parser.parse_args()

    pv = _load_target(args.pv_path, "pv_kw")
    demand = _load_target(args.demand_path, "demand_kw")
    frames = {
        split: _rows(args.cache_root / split / "forecasts.jsonl", pv, demand)
        for split in SPLITS
    }
    correction = _correction(frames["train"], frames["val"])
    result: dict[str, Any] = {
        "raw": {split: _summary(frame, split) for split, frame in frames.items()},
        "correction": correction,
        "corrected": {},
        "by_hour_test": {},
        "by_lead_test": {},
    }
    for mode in ("global", "grouped"):
        result["corrected"][mode] = {
            split: _summary(_apply(frame, correction, mode), f"{split}_{mode}")
            for split, frame in frames.items()
        }
    test_corrected = _apply(frames["test"], correction, "grouped")
    for key, group in test_corrected.groupby("hour"):
        result["by_hour_test"][str(key)] = _summary(group, f"test_hour_{key}")
    for key, group in test_corrected.groupby("lead"):
        result["by_lead_test"][str(key)] = _summary(group, f"test_lead_{key}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    for label, metrics in result["raw"].items():
        print(
            label,
            "raw net MAE",
            round(metrics["net_error"]["mae_kw"], 2),
            "bias",
            round(metrics["net_error"]["bias_kw"], 2),
        )
    for label, metrics in result["corrected"]["grouped"].items():
        print(
            label,
            "corrected net MAE",
            round(metrics["net_error"]["mae_kw"], 2),
            "bias",
            round(metrics["net_error"]["bias_kw"], 2),
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
