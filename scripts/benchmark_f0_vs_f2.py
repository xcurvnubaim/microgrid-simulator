#!/usr/bin/env python3
"""Benchmark F0 (hourly, held 4 ticks) versus F2 (direct 15-minute) forecasts.

Evaluates PV, load, and net-load accuracy on fixed February validation origins
using only training and February data. Reports the pre-declared metrics from the
15-Minute Forecast Resolution Migration Plan (§Forecast Evaluation):

  - point accuracy (MAE, RMSE, bias) for PV, load, net load;
  - horizon accuracy grouped into 0-1h, 1-6h, 6-12h, 12-24h;
  - operational shape (net-load MAE, ramp MAE, max load underprediction,
    max PV overprediction);
  - event timing (peak-load and PV-sunset timing error);
  - data quality (coverage, cold starts, stale/missing counts).

Both caches must be split-specific and leakage-safe (strict manifest). The
comparison is a forecast-quality gate (Gate 2), not an operational rollout.

Usage:
    .venv/bin/python scripts/benchmark_f0_vs_f2.py \
        --f0-cache data/forecasts.jsonl --f0-manifest data/forecast_manifest.json \
        --f2-cache data/f2/val/forecasts.jsonl --f2-manifest data/f2/val/forecast_manifest.json \
        --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microgrid_simulator.forecast import ForecastCache, ForecastCacheError

PROCESSED_PV = (
    "/home/xcurv/teep-taiwan/data/processed/pv_15min_chronos_reconstruction_candidate.csv"
)
PROCESSED_DEMAND = (
    "/home/xcurv/teep-taiwan/data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv"
)

VAL_WINDOWS = (
    "2026-02-06 00:00:00",
    "2026-02-09 00:00:00",
    "2026-02-12 00:00:00",
    "2026-02-15 00:00:00",
    "2026-02-18 00:00:00",
    "2026-02-21 00:00:00",
)

HORIZON_GROUPS = (
    ("0-1h", 1, 4),
    ("1-6h", 4, 24),
    ("6-12h", 24, 48),
    ("12-24h", 48, 96),
)


def _load_truth() -> tuple[pd.Series, pd.Series]:
    pv = pd.read_csv(PROCESSED_PV, parse_dates=["timestamp"])
    dem = pd.read_csv(PROCESSED_DEMAND, parse_dates=["timestamp"])
    pv_series = pd.Series(pv["pv_kw"].to_numpy(), index=pv["timestamp"]).sort_index()
    dem_series = pd.Series(
        dem["demand_kw"].to_numpy(), index=dem["timestamp"]
    ).sort_index()
    return pv_series, dem_series


@dataclass
class Metrics:
    pv_mae: float
    pv_rmse: float
    pv_bias: float
    load_mae: float
    load_rmse: float
    load_bias: float
    net_mae: float
    net_rmse: float
    net_bias: float
    ramp_mae: float
    max_load_underprediction: float
    max_pv_overprediction: float
    peak_load_timing_error_min: float
    pv_sunset_timing_error_min: float
    coverage: int
    cold_starts: int
    n_horizon: dict[str, dict[str, float]]


def _eval_origin(
    cache: ForecastCache,
    issued_at: pd.Timestamp,
    pv_truth: pd.Series,
    dem_truth: pd.Series,
    direct_15min: bool,
) -> Metrics | None:
    """Evaluate one forecast origin against realized right-endpoint targets.

    For F0 (hourly), each hourly point is held across four 15-minute ticks and
    compared to the four realized quarter-hour values. For F2 (direct 15-min),
    each point maps one-to-one to a realized quarter-hour value.
    """
    snapshot = cache.get_covering(issued_at.isoformat())
    if snapshot is None:
        return None
    freq = snapshot.frequency_hours
    n = snapshot.steps
    pv_f = np.asarray(snapshot.pv_values_mw) * 1000.0  # MW -> kW
    dem_f = np.asarray(snapshot.demand_values_mw) * 1000.0

    interval = pd.to_timedelta(freq, unit="h")
    start = issued_at + interval
    target_idx = pd.date_range(start=start, periods=n, freq=interval)

    # Realized right-endpoint targets (mean over the preceding interval).
    pv_r = pv_truth.reindex(target_idx).to_numpy(dtype=float)
    dem_r = dem_truth.reindex(target_idx).to_numpy(dtype=float)

    valid = ~(np.isnan(pv_r) | np.isnan(dem_r))
    if valid.sum() == 0:
        return None

    if direct_15min:
        # one-to-one comparison
        pv_pred = pv_f[: len(target_idx)]
        dem_pred = dem_f[: len(target_idx)]
    else:
        # F0: hold each hourly value across its four 15-minute control ticks.
        ticks = max(1, int(round(freq / 0.25)))
        pv_pred = np.repeat(pv_f, ticks)[: len(target_idx)]
        dem_pred = np.repeat(dem_f, ticks)[: len(target_idx)]

    pv_pred = pv_pred[valid]
    dem_pred = dem_pred[valid]
    pv_r = pv_r[valid]
    dem_r = dem_r[valid]

    net_pred = dem_pred - pv_pred
    net_r = dem_r - pv_r

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def bias(a, b):
        return float(np.mean(b - a))

    # ramp MAE on net load (difference between consecutive points).
    ramp_pred = np.diff(net_pred)
    ramp_r = np.diff(net_r)
    ramp_mae = float(np.mean(np.abs(ramp_pred - ramp_r))) if len(ramp_r) else 0.0

    # operational shape
    max_load_under = float(np.max(dem_r - dem_pred))  # actual - predicted load
    max_pv_over = float(np.max(pv_pred - pv_r))  # predicted - actual PV

    # event timing
    peak_load_timing_error_min = 0.0
    pv_sunset_timing_error_min = 0.0
    if len(dem_r) and len(pv_r):
        # peak-load timing: index of max realized vs max predicted load.
        true_peak_i = int(np.argmax(dem_r))
        pred_peak_i = int(np.argmax(dem_pred))
        peak_load_timing_error_min = abs(true_peak_i - pred_peak_i) * (freq * 60.0)

        # PV sunset: first index where realized PV stays near zero after noon.
        noon_i = max(0, len(pv_r) // 2)
        tail = pv_r[noon_i:]
        zero_tail = np.where(tail < 1.0)[0]
        true_sunset_i = noon_i + int(zero_tail[0]) if len(zero_tail) else len(pv_r) - 1
        tail_p = pv_pred[noon_i:]
        zero_tail_p = np.where(tail_p < 1.0)[0]
        pred_sunset_i = noon_i + int(zero_tail_p[0]) if len(zero_tail_p) else len(pv_pred) - 1
        pv_sunset_timing_error_min = abs(true_sunset_i - pred_sunset_i) * (freq * 60.0)

    horizon_metrics: dict[str, dict[str, float]] = {}
    for label, lo, hi in HORIZON_GROUPS:
        lo = min(lo, len(net_r))
        hi = min(hi, len(net_r))
        if hi <= lo:
            continue
        horizon_metrics[label] = {
            "net_mae": mae(net_pred[lo:hi], net_r[lo:hi]),
            "pv_mae": mae(pv_pred[lo:hi], pv_r[lo:hi]),
            "load_mae": mae(dem_pred[lo:hi], dem_r[lo:hi]),
        }

    return Metrics(
        pv_mae=mae(pv_pred, pv_r),
        pv_rmse=rmse(pv_pred, pv_r),
        pv_bias=bias(pv_pred, pv_r),
        load_mae=mae(dem_pred, dem_r),
        load_rmse=rmse(dem_pred, dem_r),
        load_bias=bias(dem_pred, dem_r),
        net_mae=mae(net_pred, net_r),
        net_rmse=rmse(net_pred, net_r),
        net_bias=bias(net_pred, net_r),
        ramp_mae=ramp_mae,
        max_load_underprediction=max_load_under,
        max_pv_overprediction=max_pv_over,
        peak_load_timing_error_min=peak_load_timing_error_min,
        pv_sunset_timing_error_min=pv_sunset_timing_error_min,
        coverage=int(valid.sum()),
        cold_starts=1 if snapshot.cold_start else 0,
        n_horizon=horizon_metrics,
    )


def _aggregate(metrics: list[Metrics]) -> dict:
    if not metrics:
        return {}
    keys = [
        "pv_mae", "pv_rmse", "pv_bias", "load_mae", "load_rmse", "load_bias",
        "net_mae", "net_rmse", "net_bias", "ramp_mae",
        "max_load_underprediction", "max_pv_overprediction",
        "peak_load_timing_error_min", "pv_sunset_timing_error_min",
    ]
    out: dict = {}
    for k in keys:
        vals = [getattr(m, k) for m in metrics]
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    out["coverage"] = int(sum(m.coverage for m in metrics))
    out["cold_starts"] = int(sum(m.cold_starts for m in metrics))
    out["origins"] = len(metrics)
    horizon: dict[str, dict[str, float]] = {}
    for label, _, _ in HORIZON_GROUPS:
        for k in ("net_mae", "pv_mae", "load_mae"):
            vals = [m.n_horizon[label][k] for m in metrics if label in m.n_horizon]
            horizon.setdefault(label, {})[k] = float(np.mean(vals)) if vals else float("nan")
    out["horizon"] = horizon
    return out


def _load_cache(path: Path, manifest_path: Path, source_id: str) -> ForecastCache:
    return ForecastCache.load(
        str(path),
        str(manifest_path),
        expected_source_id=source_id,
        action_interval_hours=0.25,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-cache", required=True, type=Path)
    parser.add_argument("--f0-manifest", required=True, type=Path)
    parser.add_argument("--f2-cache", required=True, type=Path)
    parser.add_argument("--f2-manifest", required=True, type=Path)
    parser.add_argument("--source-id", default="islanded_72h_2026-01-15")
    parser.add_argument(
        "--windows",
        nargs="*",
        default=list(VAL_WINDOWS),
        help="override the fixed February origins",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON output path")
    args = parser.parse_args()

    f0 = _load_cache(args.f0_cache, args.f0_manifest, args.source_id)
    f2 = _load_cache(args.f2_cache, args.f2_manifest, args.source_id)
    pv, dem = _load_truth()

    f0_metrics: list[Metrics] = []
    f2_metrics: list[Metrics] = []
    for window in args.windows:
        origin = pd.Timestamp(window)
        m0 = _eval_origin(f0, origin, pv, dem, direct_15min=False)
        m2 = _eval_origin(f2, origin, pv, dem, direct_15min=True)
        if m0:
            f0_metrics.append(m0)
        if m2:
            f2_metrics.append(m2)

    result = {
        "source_id": args.source_id,
        "windows": list(args.windows),
        "F0_hourly": _aggregate(f0_metrics),
        "F2_15min": _aggregate(f2_metrics),
    }

    print(json.dumps(result, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ForecastCacheError as exc:
        print(f"cache error: {exc}", file=sys.stderr)
        sys.exit(2)
