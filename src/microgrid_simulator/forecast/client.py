"""HTTP client for the Chronos forecast service (+ cache-aware variant)."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib import error, request

import pandas as pd

from microgrid_simulator.config import ForecastCfg
from microgrid_simulator.forecast.cache import ForecastCache
from microgrid_simulator.forecast.context import ForecastContext
from microgrid_simulator.forecast.errors import (
    ForecastCacheError,
    ForecastError,
    ForecastSourceError,
)
from microgrid_simulator.forecast.snapshot import ForecastSnapshot


class ForecastClient:
    """Small standard-library HTTP client for ``POST /forecast``."""

    def __init__(self, cfg: ForecastCfg):
        self.cfg = cfg

    def fetch(self, context: ForecastContext, issued_at: str | None = None) -> ForecastSnapshot:
        body: dict[str, Any] = {
            "horizon_h": self.cfg.horizon_hours,
            "context": context.as_request(),
        }
        if issued_at is not None:
            body["issued_at"] = issued_at
        payload = self._post_json(
            f"{self.cfg.service_url.rstrip('/')}/forecast",
            body,
        )
        return self._parse(payload, context)

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(body).encode("utf-8")
        req = request.Request(
            url,
            data=encoded,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.cfg.timeout_seconds) as response:  # noqa: S310
                raw = response.read()
        except (error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
            raise ForecastError(f"forecast service request failed: {exc}") from exc
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ForecastError("forecast service returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ForecastError("forecast service response must be a JSON object")
        return parsed

    def _parse(self, payload: dict[str, Any], context: ForecastContext) -> ForecastSnapshot:
        try:
            horizon = int(payload["horizon_h"])
            frequency = float(payload.get("frequency_h", 1.0))
            forecasts = payload["forecast"]
            issued_at = str(payload["issued_at"])
            model_version = str(payload["model_version"])
            source_id = str(payload["source_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ForecastError("forecast response is missing series or metadata") from exc

        if source_id != context.source_id:
            raise ForecastSourceError(context.source_id, source_id)

        if horizon != self.cfg.horizon_hours:
            raise ForecastError(
                f"forecast returned {horizon} h; requested {self.cfg.horizon_hours} h"
            )
        if frequency <= 0.0 or not math.isfinite(frequency):
            raise ForecastError("forecast frequency_h must be a positive finite number")
        if not math.isclose(frequency, 1.0):
            raise ForecastError(
                f"forecast frequency_h is {frequency}; the simulator expects hourly values"
            )
        expected_points = int(round(horizon / frequency))
        units = payload.get("units", {})
        if not isinstance(forecasts, dict):
            raise ForecastError("forecast series must be a JSON object")

        def series_mw(target: str) -> tuple[float, ...]:
            values = forecasts.get(target)
            if not isinstance(values, list) or len(values) != expected_points:
                actual = len(values) if isinstance(values, list) else "missing/non-list"
                raise ForecastError(
                    f"forecast target {target!r} has {actual} values; expected {expected_points}"
                )
            unit = units.get(target, "") if isinstance(units, dict) else ""
            unit = str(unit).lower()
            if unit != "kw":
                raise ForecastError(f"forecast target {target!r} must use 'kw' at the API boundary")
            converted: list[float] = []
            for value in values:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise ForecastError(
                        f"forecast target {target!r} contains a non-numeric value"
                    ) from exc
                if not math.isfinite(numeric):
                    raise ForecastError(f"forecast target {target!r} contains a non-finite value")
                # The HTTP boundary is intentionally the only kW -> MW conversion.
                converted.append(max(0.0, numeric / 1000.0))
            return tuple(converted)

        pv_values_mw = series_mw(self.cfg.target)
        demand_values_mw = series_mw(self.cfg.demand_target)
        self._check_source_scale(context, pv_values_mw, demand_values_mw)

        timestamps_raw = payload.get("timestamps", [])
        if timestamps_raw and (
            not isinstance(timestamps_raw, list) or len(timestamps_raw) != expected_points
        ):
            raise ForecastError("forecast timestamps must align one-to-one with values")
        timestamps = tuple(str(value) for value in timestamps_raw)

        return ForecastSnapshot(
            issued_at=issued_at,
            horizon_hours=horizon,
            frequency_hours=frequency,
            model_version=model_version,
            pv_target=self.cfg.target,
            demand_target=self.cfg.demand_target,
            timestamps=timestamps,
            pv_values_mw=pv_values_mw,
            demand_values_mw=demand_values_mw,
            source_id=source_id,
            context_time=(
                str(payload["context_time"]) if payload.get("context_time") is not None else None
            ),
            context_steps=int(payload.get("context_steps", len(context.pv_values_mw))),
            cold_start=bool(payload.get("cold_start", False)),
            covariate_mode=str(payload.get("covariate_mode", "unknown")),
        )

    @staticmethod
    def _check_source_scale(
        context: ForecastContext,
        pv_values_mw: tuple[float, ...],
        demand_values_mw: tuple[float, ...],
    ) -> None:
        """Catch obvious cross-source scale leaks before the dashboard sees them."""

        for name, history, forecast in (
            ("PV", context.pv_values_mw, pv_values_mw),
            ("demand", context.demand_values_mw, demand_values_mw),
        ):
            history_peak = max(history, default=0.0)
            forecast_peak = max(forecast, default=0.0)
            # A 60 kW curve alongside a 20 MW scenario is not a plausible low
            # load period; it is a source/unit mismatch.  Keep zero-PV night
            # forecasts valid by requiring a materially large observed context.
            if history_peak >= 10.0 and forecast_peak < history_peak * 0.01:
                raise ForecastError(
                    f"forecast {name} range is incompatible with {context.source_id}: "
                    f"{forecast_peak:.3f} MW forecast for {history_peak:.3f} MW context"
                )


class CachedForecastClient(ForecastClient):
    """ForecastClient variant that queries an in-memory ForecastCache first."""

    def __init__(self, cfg: ForecastCfg, cache: ForecastCache | None = None):
        super().__init__(cfg)
        self.cache = cache or ForecastCache()

    def fetch(self, context: ForecastContext, issued_at: str | None = None) -> ForecastSnapshot:
        if issued_at and (snapshot := self.cache.get_covering(issued_at)):
            if snapshot.source_id and snapshot.source_id != context.source_id:
                raise ForecastSourceError(context.source_id, snapshot.source_id)
            return snapshot
        return super().fetch(context, issued_at)


class StrictCachedForecastClient(CachedForecastClient):
    """Cache-only forecast client for leakage-safe RL experiments."""

    def fetch(self, context: ForecastContext, issued_at: str | None = None) -> ForecastSnapshot:
        if issued_at is None:
            raise ForecastError("strict cached forecasting requires an issued timestamp")
        snapshot = self.cache.get_covering(issued_at)
        if snapshot is None:
            raise ForecastCacheError(f"no cached forecast covers {issued_at}")
        requested = pd.Timestamp(issued_at).tz_localize(None)
        issued = pd.Timestamp(snapshot.issued_at).tz_localize(None)
        age_hours = (requested - issued).total_seconds() / 3600.0
        if age_hours < 0.0 or age_hours >= snapshot.frequency_hours:
            raise ForecastCacheError(
                f"no cached forecast was issued in the {snapshot.frequency_hours:g}h interval "
                f"covering {issued_at}"
            )
        if snapshot.source_id and snapshot.source_id != context.source_id:
            raise ForecastSourceError(context.source_id, snapshot.source_id)
        return snapshot
