"""Client boundary for the separately deployed Chronos PV/demand forecaster.

The simulator intentionally talks to ``microgrid-forecaster`` over HTTP rather
than importing its heavy foundation-model stack. One request is made at episode
reset. Forecast values are observations for controllers/agents and dashboard
comparison only; the physical backend continues to use its own trajectories.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from microgrid_simulator.config import ForecastCfg


class ForecastError(RuntimeError):
    """The forecast service failed or returned an invalid payload."""


class ForecastSourceError(ForecastError):
    """A response belongs to a different scenario and must not be consumed."""

    def __init__(self, expected_source: str, actual_source: str):
        super().__init__(f"forecast source is incompatible with {expected_source}")
        self.expected_source = expected_source
        self.actual_source = actual_source


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


class ForecastCache:
    """In-memory cache for JSONL forecast snapshots keyed by timestamp or index."""

    def __init__(self, records: dict[str, ForecastSnapshot] | None = None) -> None:
        self._records: dict[str, ForecastSnapshot] = records or {}

    @classmethod
    def from_jsonl(cls, jsonl_path: str) -> ForecastCache:
        """Load ForecastCache from a JSONL file containing forecast snapshot records."""
        records: dict[str, ForecastSnapshot] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                issued_at = data["issued_at"]
                horizon = int(data.get("horizon_hours", data.get("horizon_h", 24)))
                freq = float(data.get("frequency_hours", data.get("frequency_h", 1.0)))
                pv_mw = tuple(val / 1000.0 for val in data.get("pv_values_kw", []))
                demand_mw = tuple(val / 1000.0 for val in data.get("demand_values_kw", []))
                if not pv_mw and "pv_values_mw" in data:
                    pv_mw = tuple(data["pv_values_mw"])
                if not demand_mw and "demand_values_mw" in data:
                    demand_mw = tuple(data["demand_values_mw"])

                snapshot = ForecastSnapshot(
                    issued_at=issued_at,
                    horizon_hours=horizon,
                    frequency_hours=freq,
                    model_version=str(data.get("model_version", "cached")),
                    pv_target=str(data.get("pv_target", "pv_avg")),
                    demand_target=str(data.get("demand_target", "demand")),
                    timestamps=tuple(str(ts) for ts in data.get("timestamps", [])),
                    pv_values_mw=pv_mw,
                    demand_values_mw=demand_mw,
                    source_id=str(data.get("source_id", "")),
                    context_time=data.get("context_time"),
                    context_steps=int(data.get("context_steps", 0)),
                    cold_start=bool(data.get("cold_start", False)),
                    covariate_mode=str(data.get("covariate_mode", "cached")),
                )
                records[issued_at] = snapshot
        return cls(records)

    def get(self, issued_at: str) -> ForecastSnapshot | None:
        return self._records.get(issued_at)

    def to_jsonl(self, jsonl_path: str) -> None:
        """Write all ForecastSnapshot records to a JSONL file."""
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for snapshot in self._records.values():
                record = {
                    "issued_at": snapshot.issued_at,
                    "horizon_hours": snapshot.horizon_hours,
                    "frequency_hours": snapshot.frequency_hours,
                    "model_version": snapshot.model_version,
                    "pv_target": snapshot.pv_target,
                    "demand_target": snapshot.demand_target,
                    "timestamps": list(snapshot.timestamps),
                    "pv_values_kw": [val * 1000.0 for val in snapshot.pv_values_mw],
                    "demand_values_kw": [val * 1000.0 for val in snapshot.demand_values_mw],
                    "source_id": snapshot.source_id,
                    "context_time": snapshot.context_time,
                    "context_steps": snapshot.context_steps,
                    "cold_start": snapshot.cold_start,
                    "covariate_mode": snapshot.covariate_mode,
                }
                f.write(json.dumps(record) + "\n")

    def __len__(self) -> int:
        return len(self._records)


class CachedForecastClient(ForecastClient):
    """ForecastClient variant that queries an in-memory ForecastCache first."""

    def __init__(self, cfg: ForecastCfg, cache: ForecastCache | None = None):
        super().__init__(cfg)
        self.cache = cache or ForecastCache()

    def fetch(self, context: ForecastContext, issued_at: str | None = None) -> ForecastSnapshot:
        if issued_at and (snapshot := self.cache.get(issued_at)):
            if snapshot.source_id and snapshot.source_id != context.source_id:
                raise ForecastSourceError(context.source_id, snapshot.source_id)
            return snapshot
        return super().fetch(context, issued_at)
