"""Align measured series onto the simulator's regular timestep grid.

Missing data is handled by an *explicit* strategy and hard limits: long gaps
or a high missing share raise :class:`AlignmentError` instead of being bridged
silently — a digital twin calibrated on invented data is worse than no twin.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class AlignmentError(ValueError):
    """Raised when the measured data cannot honestly cover the requested grid."""


def _gap_runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Contiguous runs of missing samples as (start, end, length)."""
    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    run_start: pd.Timestamp | None = None
    length = 0
    for ts, missing in mask.items():
        if missing:
            run_start = ts if run_start is None else run_start
            length += 1
        elif run_start is not None:
            runs.append((run_start, ts, length))
            run_start, length = None, 0
    if run_start is not None:
        runs.append((run_start, mask.index[-1], length))
    return runs


def align_series(
    series: dict[str, pd.Series],
    timestep_hours: float,
    fill_strategy: str = "interpolate",
    max_gap_steps: int = 8,
    max_missing_fraction: float = 0.05,
) -> pd.DataFrame:
    """Resample all series to one regular grid over their common time range.

    Strategies: ``interpolate`` (linear, short gaps only), ``ffill`` (hold last
    value, short gaps only), ``error`` (any missing sample is fatal).
    """
    if not series:
        raise AlignmentError("no series to align")
    if fill_strategy not in {"interpolate", "ffill", "error"}:
        raise AlignmentError(f"unknown fill strategy {fill_strategy!r}")

    start = max(s.index[0] for s in series.values())
    end = min(s.index[-1] for s in series.values())
    if start >= end:
        raise AlignmentError(
            "series do not overlap in time: "
            + ", ".join(f"{k}: {v.index[0]}..{v.index[-1]}" for k, v in series.items())
        )

    freq = pd.Timedelta(hours=timestep_hours)
    aligned: dict[str, pd.Series] = {}
    for name, s in series.items():
        regular = s.resample(freq).mean().loc[start:end]
        missing = regular.isna()
        n_missing = int(missing.sum())
        if n_missing:
            frac = n_missing / len(regular)
            gaps = [g for g in _gap_runs(missing) if g[2] > max_gap_steps]
            if fill_strategy == "error":
                raise AlignmentError(f"{name}: {n_missing} missing samples (strategy=error)")
            if frac > max_missing_fraction:
                raise AlignmentError(
                    f"{name}: {frac:.1%} of samples missing exceeds "
                    f"max_missing_fraction={max_missing_fraction:.1%}"
                )
            if gaps:
                detail = "; ".join(f"{s0} .. {s1} ({n} steps)" for s0, s1, n in gaps[:5])
                raise AlignmentError(
                    f"{name}: gap(s) longer than {max_gap_steps} steps — refusing to "
                    f"fill silently: {detail}"
                )
            LOGGER.info("%s: filling %d missing samples via %s", name, n_missing, fill_strategy)
            if fill_strategy == "interpolate":
                regular = regular.interpolate(limit=max_gap_steps, limit_direction="both")
            else:
                regular = regular.ffill(limit=max_gap_steps).bfill(limit=max_gap_steps)
        aligned[name] = regular

    frame = pd.DataFrame(aligned).dropna()
    if len(frame) < 4:
        raise AlignmentError("fewer than 4 aligned samples remain")
    frame.index.name = "timestamp"
    return frame.astype(np.float64)
