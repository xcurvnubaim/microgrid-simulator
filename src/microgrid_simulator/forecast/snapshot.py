"""Validated forecast container shared by the HTTP client and the cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ForecastSnapshot:
    """Validated, timestamp-aligned PV and demand forecasts in MW.

    ``frequency_hours`` is the target spacing (points per snapshot are
    ``forecast_steps``). ``issue_frequency_hours`` is the interval between
    issue times and is carried so consumers can fail closed when a snapshot is
    too old for a decision. ``target_units`` records the API-boundary units the
    values arrived in (always ``kw`` at the boundary).
    """

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
    pv_context_steps: int = 0
    demand_context_steps: int = 0
    cold_start: bool = False
    covariate_mode: str = "unknown"
    issue_frequency_hours: float | None = None
    forecast_steps: int | None = None
    target_units: str = "kw"

    @property
    def steps(self) -> int:
        """Resolved point count, preferring the explicit field when present."""
        if self.forecast_steps is not None:
            return self.forecast_steps
        return int(round(self.horizon_hours / self.frequency_hours))

    def as_meta(self) -> dict[str, Any]:
        return {
            "forecast_available": True,
            "forecast_issued_at": self.issued_at,
            "forecast_horizon_hours": self.horizon_hours,
            "forecast_frequency_hours": self.frequency_hours,
            "forecast_issue_frequency_hours": self.issue_frequency_hours,
            "forecast_steps": self.steps,
            "forecast_first_target": self.timestamps[0] if self.timestamps else None,
            "forecast_last_target": self.timestamps[-1] if self.timestamps else None,
            "forecast_model_version": self.model_version,
            "forecast_target": self.pv_target,
            "forecast_demand_target": self.demand_target,
            "forecast_target_units": self.target_units,
            "forecast_timestamps": list(self.timestamps),
            "forecast_values_kw": [value * 1000.0 for value in self.pv_values_mw],
            "forecast_demand_values_kw": [value * 1000.0 for value in self.demand_values_mw],
            "forecast_source": self.source_id,
            "forecast_context_time": self.context_time,
            "forecast_context_steps": self.context_steps,
            "forecast_pv_context_steps": self.pv_context_steps,
            "forecast_demand_context_steps": self.demand_context_steps,
            "forecast_cold_start": self.cold_start,
            "forecast_covariate_mode": self.covariate_mode,
        }
