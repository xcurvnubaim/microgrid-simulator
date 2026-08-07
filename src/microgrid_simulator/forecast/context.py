"""Request-side forecast context: internal MW history for the API contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from microgrid_simulator.forecast.errors import ForecastError


@dataclass(frozen=True)
class ForecastContext:
    """Internal MW history serialized to the forecast API's kW contract once."""

    source_id: str
    frequency_hours: float
    pv_values_mw: tuple[float, ...]
    demand_values_mw: tuple[float, ...]

    def as_request(self) -> dict[str, Any]:
        if not self.source_id:
            raise ForecastError("forecast context requires a scenario source_id")
        if not math.isfinite(self.frequency_hours) or self.frequency_hours <= 0.0:
            raise ForecastError("forecast context frequency_h must be positive and finite")
        if not self.pv_values_mw or len(self.pv_values_mw) != len(self.demand_values_mw):
            raise ForecastError(
                "forecast context requires matching non-empty PV and demand history"
            )

        def kw(values: tuple[float, ...], name: str) -> list[float]:
            converted: list[float] = []
            for value in values:
                if not math.isfinite(value) or value < 0.0:
                    raise ForecastError(
                        f"forecast context {name} contains a non-finite or negative MW value"
                    )
                # The HTTP boundary is intentionally the only MW -> kW conversion.
                converted.append(value * 1000.0)
            return converted

        return {
            "source_id": self.source_id,
            "frequency_h": self.frequency_hours,
            "pv_kw": kw(self.pv_values_mw, "pv"),
            "demand_kw": kw(self.demand_values_mw, "demand"),
        }
