"""Strict, timestamp-aligned telemetry windows for deterministic experiments.

Unlike the training-oriented demand fallback, a configured fixed replay fails
closed: missing files, misaligned timestamps, or gaps raise an error instead of
silently substituting a synthetic curve.  Index 0 is the preceding context
sample; indices 1..N are the N evaluated control intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microgrid_simulator.config import Settings
from microgrid_simulator.digital_twin.alignment import AlignmentError, align_series
from microgrid_simulator.digital_twin.data_ingestion import load_measurements

REQUIRED_REPLAY_SERIES = ("load", "pv")


@dataclass(frozen=True)
class TelemetryWindow:
    """One aligned episode window in canonical units (MW)."""

    demand_mw: np.ndarray
    pv_mw: np.ndarray
    timestamps: pd.DatetimeIndex
    first_evaluated_timestamp: pd.Timestamp
    source_files: dict[str, str]

    @property
    def context_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.timestamps[0])

    @property
    def last_evaluated_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.timestamps[-1])


def load_fixed_telemetry_window(settings: Settings, n_steps: int) -> TelemetryWindow | None:
    """Load the configured fixed load/PV replay, or ``None`` when not requested.

    ``episode.telemetry_start`` denotes the first of ``n_steps`` evaluated
    samples. One preceding sample is included as context because the rule
    controller acts on the last observed state.
    """
    start_text = settings.episode.telemetry_start
    if not start_text:
        return None
    if n_steps < 1:
        raise AlignmentError("fixed telemetry replay requires at least one step")

    missing = [
        name for name in REQUIRED_REPLAY_SERIES if name not in settings.digital_twin.measurements
    ]
    if missing:
        raise AlignmentError(
            "fixed telemetry replay requires digital_twin.measurements "
            f"{list(REQUIRED_REPLAY_SERIES)}; missing {missing}"
        )

    dt = pd.to_timedelta(float(settings.topology.timestep_hours), unit="h")
    first = pd.Timestamp(start_text)
    context = first - dt
    last = first + (n_steps - 1) * dt
    expected = pd.date_range(context, last, freq=dt, name="timestamp")

    all_series = load_measurements(settings.digital_twin)
    selected = {name: all_series[name].loc[context:last] for name in REQUIRED_REPLAY_SERIES}
    empty = [name for name, series in selected.items() if series.empty]
    if empty:
        raise AlignmentError(
            f"telemetry has no samples for {empty} in required window {context} .. {last}"
        )

    frame = align_series(
        selected,
        timestep_hours=float(settings.topology.timestep_hours),
        fill_strategy=settings.digital_twin.fill_strategy,
        max_gap_steps=settings.digital_twin.max_gap_steps,
        max_missing_fraction=settings.digital_twin.max_missing_fraction,
    ).reindex(expected)
    if frame.isna().any().any():
        detail = {name: int(frame[name].isna().sum()) for name in frame.columns}
        raise AlignmentError(
            f"fixed replay does not exactly cover {context} .. {last}: missing {detail}"
        )
    if len(frame) != n_steps + 1:
        raise AlignmentError(f"expected {n_steps + 1} context/evaluation samples, got {len(frame)}")

    # Meter standby draw can make night-time PV slightly negative. The plant
    # model represents generation availability, so clip it to zero explicitly.
    frame["load"] = frame["load"].clip(lower=0.0)
    frame["pv"] = frame["pv"].clip(lower=0.0)
    source_files = {
        name: settings.digital_twin.measurements[name].file for name in REQUIRED_REPLAY_SERIES
    }
    return TelemetryWindow(
        demand_mw=frame["load"].to_numpy(dtype=np.float64),
        pv_mw=frame["pv"].to_numpy(dtype=np.float64),
        timestamps=pd.DatetimeIndex(frame.index),
        first_evaluated_timestamp=first,
        source_files=source_files,
    )
