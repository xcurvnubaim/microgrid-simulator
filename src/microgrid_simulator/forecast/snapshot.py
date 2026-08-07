"""Validated forecast container shared by the HTTP client and the cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ForecastSnapshot:
    """Validated, timestamp-aligned hourly PV and demand forecasts in MW."""

    issued_at: str
    horizon_hours: int
    frequency_hours: float
    model_version: str
    pv_target: str
    demand_target: str
    timestamps: tuple[str, ...]
    pv_values_mw: tuple[float, ...]
    demand_values_mw: tuple[float, ...]
    source_id: str = ""
    context_time: str | None = None
    context_steps: int = 0
    cold_start: bool = False
    covariate_mode: str = "unknown"

    def as_meta(self) -> dict[str, Any]:
        return {
            "forecast_available": True,
            "forecast_issued_at": self.issued_at,
            "forecast_horizon_hours": self.horizon_hours,
            "forecast_frequency_hours": self.frequency_hours,
            "forecast_model_version": self.model_version,
            "forecast_target": self.pv_target,
            "forecast_demand_target": self.demand_target,
            "forecast_timestamps": list(self.timestamps),
            "forecast_values_kw": [value * 1000.0 for value in self.pv_values_mw],
            "forecast_demand_values_kw": [value * 1000.0 for value in self.demand_values_mw],
            "forecast_source": self.source_id,
            "forecast_context_time": self.context_time,
            "forecast_context_steps": self.context_steps,
            "forecast_cold_start": self.cold_start,
            "forecast_covariate_mode": self.covariate_mode,
        }
