"""Forecast context owned by the distributed EMS orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microgrid_simulator.config import Settings
from microgrid_simulator.contracts import TelemetryWindowPayload
from microgrid_simulator.forecast import (
    ForecastCache,
    ForecastClient,
    ForecastContext,
    ForecastSnapshot,
    StrictCachedForecastClient,
)


@dataclass(frozen=True)
class ForecastView:
    available: bool
    pv_mw: np.ndarray
    demand_mw: np.ndarray
    current_pv_mw: float | None
    current_demand_mw: float | None
    meta: dict[str, object]


class EMSForecastRuntime:
    """Build leakage-free rolling forecast context from the EMS telemetry window."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        cfg = settings.forecast
        mode = getattr(settings.rl, "forecast_mode", "cached")
        if mode not in {"cached", "none"}:
            raise ValueError("forecast_mode must be 'cached' or 'none'")
        if cfg.enabled and cfg.strict_cache and mode == "cached":
            if not cfg.cache_path or not cfg.manifest_path:
                raise ValueError("strict cached forecasting requires cache_path and manifest_path")
            cache = ForecastCache.load(
                cfg.cache_path,
                cfg.manifest_path,
                expected_source_id=cfg.source_id or settings.scenario.name,
                action_interval_hours=settings.topology.timestep_hours,
            )
            self.client: ForecastClient = StrictCachedForecastClient(cfg, cache)
        else:
            self.client = ForecastClient(cfg)
        self.payload: TelemetryWindowPayload | None = None
        self.snapshot: ForecastSnapshot | None = None
        self.origin_sequence = 0
        self.stale = False
        self.error: str | None = None
        self.last_refresh_sequence = -1

    def reset(self, payload: TelemetryWindowPayload) -> None:
        self.payload = payload
        self.snapshot = None
        self.origin_sequence = 0
        self.stale = False
        self.error = None
        self.last_refresh_sequence = -1

    def _context(self, sequence: int) -> ForecastContext:
        if self.payload is None:
            raise RuntimeError("forecast telemetry window is not initialized")
        samples = self.payload.samples[: sequence + 1]
        return ForecastContext(
            source_id=self.settings.forecast.source_id or self.settings.scenario.name,
            frequency_hours=self.settings.topology.timestep_hours,
            pv_values_mw=tuple(sample.pv_available_kw / 1000.0 for sample in samples),
            demand_values_mw=tuple(sample.load_demand_kw / 1000.0 for sample in samples),
        )

    def _refresh(self, sequence: int) -> None:
        cfg = self.settings.forecast
        mode = getattr(self.settings.rl, "forecast_mode", "cached")
        if not cfg.enabled or mode == "none":
            self.snapshot = None
            return
        if self.payload is None:
            raise RuntimeError("forecast telemetry window is not initialized")
        issued_at = pd.Timestamp(self.payload.samples[sequence].observed_at).isoformat()
        candidate = self.client.fetch(self._context(sequence), issued_at=issued_at)
        if candidate is None:
            self.snapshot = None
            return
        if self.snapshot is not None and candidate.issued_at == self.snapshot.issued_at:
            self.stale = True
            return
        self.snapshot = candidate
        self.origin_sequence = sequence
        self.stale = False
        self.error = None

    def view(self, sequence: int) -> ForecastView:
        cfg = self.settings.forecast
        if (sequence == 0 or cfg.refresh_each_step) and sequence != self.last_refresh_sequence:
            self._refresh(sequence)
            self.last_refresh_sequence = sequence
        horizon = cfg.forecast_steps
        pv = np.zeros(horizon, dtype=np.float32)
        demand = np.zeros(horizon, dtype=np.float32)
        offset = 0
        if self.snapshot is not None:
            elapsed = max(0, sequence - self.origin_sequence)
            offset = int(
                np.floor(
                    elapsed
                    * float(self.settings.topology.timestep_hours)
                    / self.snapshot.frequency_hours
                    + 1e-12
                )
            )
            pv_values = self.snapshot.pv_values_mw[offset : offset + horizon]
            demand_values = self.snapshot.demand_values_mw[offset : offset + horizon]
            pv[: len(pv_values)] = pv_values
            demand[: len(demand_values)] = demand_values
        available = bool(
            self.snapshot is not None
            and offset < len(self.snapshot.pv_values_mw)
            and offset < len(self.snapshot.demand_values_mw)
        )
        meta: dict[str, object] = {
            "forecast_enabled": cfg.enabled,
            "forecast_available": available,
            "forecast_requested_horizon_hours": horizon,
            "forecast_refresh_each_step": cfg.refresh_each_step,
            "forecast_requested_source": cfg.source_id or self.settings.scenario.name,
            "forecast_source": self.snapshot.source_id if self.snapshot else None,
            "forecast_stale": self.stale,
            "forecast_age_steps": max(0, sequence - self.origin_sequence),
            "forecast_error": self.error,
        }
        if self.snapshot is not None:
            meta.update(self.snapshot.as_meta())
        return ForecastView(
            available=available,
            pv_mw=pv,
            demand_mw=demand,
            current_pv_mw=float(pv[0]) if available else None,
            current_demand_mw=float(demand[0]) if available else None,
            meta=meta,
        )
