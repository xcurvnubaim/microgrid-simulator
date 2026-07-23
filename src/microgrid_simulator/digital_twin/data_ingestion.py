"""Load measured series (grid meter, PV, SOC, load) from CSV/Excel exports.

Every series comes back as a pandas Series with a sorted DatetimeIndex and
normalised units: power in MW, state-of-charge as a fraction in [0, 1].
Duplicate timestamps (multiple meters per tick) are summed for power series
and averaged for SOC — the campus exports carry one row per device.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from microgrid_simulator.config import DigitalTwinCfg, MeasurementCfg

LOGGER = logging.getLogger(__name__)

# Multiplier to the canonical unit (MW for power, fraction for SOC).
UNIT_SCALES = {
    "w": 1e-6,
    "kw": 1e-3,
    "mw": 1.0,
    "pct": 0.01,
    "percent": 0.01,
    "fraction": 1.0,
}
_MEAN_UNITS = {"pct", "percent", "fraction"}  # levels average; powers sum


def load_series(cfg: MeasurementCfg) -> pd.Series:
    """Read one measurement file and return a unit-normalised, time-indexed series."""
    path = Path(cfg.file)
    if not path.exists():
        raise FileNotFoundError(f"measurement file not found: {path}")

    unit = cfg.unit.lower()
    energy_units = {"kwh_per_step", "mwh_per_step"}
    if unit not in UNIT_SCALES and unit not in energy_units:
        expected = sorted([*UNIT_SCALES, *energy_units])
        raise ValueError(f"unknown unit {cfg.unit!r}; expected one of {expected}")

    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        frame = pd.read_excel(
            path, header=cfg.header_row, sheet_name=cfg.sheet or 0, engine="openpyxl"
        )
    else:
        frame = pd.read_csv(path, header=cfg.header_row)

    required_columns = {cfg.value_column}
    if cfg.timestamp_column is not None:
        required_columns.add(cfg.timestamp_column)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing column(s) {sorted(missing)}; found {list(frame.columns)}"
        )

    if cfg.timestamp_column is None:
        if cfg.synthetic_start is None or cfg.synthetic_step_hours is None:
            raise ValueError(
                f"{path.name}: positional series requires synthetic_start and synthetic_step_hours"
            )
        ts = pd.date_range(
            cfg.synthetic_start,
            periods=len(frame),
            freq=pd.to_timedelta(cfg.synthetic_step_hours, unit="h"),
        )
    else:
        ts = pd.to_datetime(frame[cfg.timestamp_column], errors="coerce")
    values = pd.to_numeric(frame[cfg.value_column], errors="coerce")
    series = pd.Series(values.to_numpy() * cfg.value_multiplier, index=ts).dropna()
    series = series[series.index.notna()]
    if series.empty:
        raise ValueError(f"{path.name}: no usable rows after parsing timestamps/values")

    agg = "mean" if unit in _MEAN_UNITS else "sum"
    series = series.groupby(level=0).agg(agg).sort_index()
    if unit == "kwh_per_step":
        if not cfg.synthetic_step_hours or cfg.synthetic_step_hours <= 0:
            raise ValueError(f"{path.name}: kwh_per_step requires positive synthetic_step_hours")
        scale = 1e-3 / cfg.synthetic_step_hours
    elif unit == "mwh_per_step":
        if not cfg.synthetic_step_hours or cfg.synthetic_step_hours <= 0:
            raise ValueError(f"{path.name}: mwh_per_step requires positive synthetic_step_hours")
        scale = 1.0 / cfg.synthetic_step_hours
    else:
        scale = UNIT_SCALES[unit]
    series = series * scale
    if cfg.prepend_first_as_context:
        if not cfg.synthetic_step_hours or cfg.synthetic_step_hours <= 0:
            raise ValueError(
                f"{path.name}: prepend_first_as_context requires positive synthetic_step_hours"
            )
        context_timestamp = series.index[0] - pd.to_timedelta(cfg.synthetic_step_hours, unit="h")
        context = pd.Series([series.iloc[0]], index=pd.DatetimeIndex([context_timestamp]))
        series = pd.concat([context, series])
    return series


def load_measurements(cfg: DigitalTwinCfg) -> dict[str, pd.Series]:
    """Load every series configured under ``digital_twin.measurements``.

    Keys are the config names (conventionally ``grid_import``, ``pv``, ``soc``,
    ``load``, ``battery_power``).
    """
    out: dict[str, pd.Series] = {}
    for name, spec in cfg.measurements.items():
        out[name] = load_series(spec)
        LOGGER.info(
            "Loaded %s: %d samples, %s .. %s",
            name,
            len(out[name]),
            out[name].index[0],
            out[name].index[-1],
        )
    return out
