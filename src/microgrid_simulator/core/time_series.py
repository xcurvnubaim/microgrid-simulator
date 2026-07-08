"""Historical demand trace — a regular-interval time series feeding episodes.

Loads the campus net-load export (xlsx via openpyxl, or csv), aggregates it
into a single total-demand series at the simulator timestep, and hands out
random episode windows so the agent trains against *actual* demand shapes
(spikes, weekday/weekend variation, noise) instead of a perfect sinusoid.

Expected source shape (Total Load (net load)_*.xlsx):
    header on row 2  ->  pd.read_excel(..., header=1)
    columns          ->  deviceid, 樓, loadname, statstime, demand
    resolution       ->  15 min (matches ``topology.timestep_hours: 0.25``)
    unit             ->  kW

Only demand is wired in for now; PV and SOC real series stay synthetic per the
plan ("demand is the trustworthy one to start with").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microgrid_simulator.config import DemandCfg

LOGGER = logging.getLogger(__name__)


@dataclass
class DemandTrace:
    """A regular-interval total-demand series in MW."""

    values_mw: np.ndarray  # shape (n,), total campus demand per tick
    start: pd.Timestamp | None
    step_hours: float
    source: str

    # -- construction --------------------------------------------------------
    @classmethod
    def from_file(cls, cfg: DemandCfg, timestep_hours: float) -> DemandTrace | None:
        """Load the trace, or return ``None`` (with a warning) when unavailable.

        Returning ``None`` keeps the simulator usable everywhere: the backend
        falls back to the synthetic sinusoid, exactly the pre-plan behaviour.
        """
        if not cfg.enabled or not cfg.file:
            return None
        path = Path(cfg.file)
        if not path.exists():
            LOGGER.warning("Demand file not found: %s — using synthetic load curve", path)
            return None
        try:
            frame = cls._read(path, cfg)
            return cls._from_frame(frame, cfg, timestep_hours, source=str(path))
        except Exception as exc:  # noqa: BLE001 - any parse failure degrades gracefully
            LOGGER.warning("Failed to load demand trace %s: %s — using synthetic", path, exc)
            return None

    @staticmethod
    def _read(path: Path, cfg: DemandCfg) -> pd.DataFrame:
        if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            # Header is on row 2 of the export -> header=cfg.header_row (default 1).
            return pd.read_excel(path, header=cfg.header_row, engine="openpyxl")
        return pd.read_csv(path, header=cfg.header_row)

    @classmethod
    def _from_frame(
        cls, frame: pd.DataFrame, cfg: DemandCfg, timestep_hours: float, source: str
    ) -> DemandTrace:
        ts_col, val_col = cfg.timestamp_column, cfg.value_column
        missing = {ts_col, val_col} - set(frame.columns)
        if missing:
            raise ValueError(f"missing column(s) {sorted(missing)}; found {list(frame.columns)}")

        series = (
            frame[[ts_col, val_col]]
            .assign(**{ts_col: pd.to_datetime(frame[ts_col], errors="coerce")})
            .dropna(subset=[ts_col])
        )
        series[val_col] = pd.to_numeric(series[val_col], errors="coerce")
        series = series.dropna(subset=[val_col])
        if series.empty:
            raise ValueError("no usable rows after parsing timestamps/values")

        # Multiple meters per timestamp (deviceid rows) -> sum to campus total.
        total = series.groupby(ts_col)[val_col].sum().sort_index()

        # Snap onto the simulator's regular grid; short gaps are interpolated.
        freq = pd.Timedelta(hours=timestep_hours)
        regular = total.resample(freq).mean().interpolate(limit=8, limit_direction="both")
        regular = regular.dropna()
        if len(regular) < 4:
            raise ValueError("trace too short after resampling")

        scale = 1e-3 if cfg.unit.lower() == "kw" else 1.0  # -> MW
        values_mw = np.clip(regular.to_numpy(dtype=np.float64) * scale, 0.0, None)
        return cls(
            values_mw=values_mw,
            start=regular.index[0],
            step_hours=timestep_hours,
            source=source,
        )

    # -- episode sampling ----------------------------------------------------
    def __len__(self) -> int:
        return int(self.values_mw.shape[0])

    def sample_window(
        self, rng: np.random.Generator, n_steps: int
    ) -> tuple[np.ndarray, int]:
        """Draw a random contiguous window (plan decision 2: random, not sequential).

        Returns ``(window_mw, start_index)``. If the trace is shorter than the
        requested horizon it is tiled to cover it.
        """
        if len(self) <= n_steps:
            reps = int(np.ceil((n_steps + 1) / max(len(self), 1)))
            return np.tile(self.values_mw, reps)[: n_steps + 1], 0
        start = int(rng.integers(0, len(self) - n_steps))
        return self.values_mw[start : start + n_steps + 1].copy(), start

    def window_start_time(self, start_index: int) -> str | None:
        if self.start is None:
            return None
        return str(self.start + pd.Timedelta(hours=start_index * self.step_hours))

    # -- reporting (used by the dashboard API) --------------------------------
    def stats(self) -> dict[str, float | str | int | None]:
        kw = self.values_mw * 1000.0
        return {
            "source": self.source,
            "rows": len(self),
            "start": str(self.start) if self.start is not None else None,
            "end": self.window_start_time(len(self) - 1),
            "min_kw": float(kw.min()),
            "mean_kw": float(kw.mean()),
            "max_kw": float(kw.max()),
            "step_hours": self.step_hours,
        }
