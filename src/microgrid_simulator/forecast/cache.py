"""On-disk forecast cache: JSONL records, manifest, and strict loading.

``ForecastCache.from_jsonl`` is the lenient dashboard/replay path, while
``ForecastCache.load`` is the fail-closed RL-training path: it requires the
manifest, scenario identity, 15-minute grid alignment, and a leakage-free
horizon before any snapshot may be consumed.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import pandas as pd

from microgrid_simulator.forecast.errors import ForecastCacheError, ForecastSourceError
from microgrid_simulator.forecast.snapshot import ForecastSnapshot


@dataclass(frozen=True)
class ForecastCacheManifest:
    """Validated ``forecast_manifest.json`` describing a pre-generated cache.

    The manifest pins the scenario ``source_id``, record count, horizon, and
    frequency so a cache cannot be silently paired with the wrong scenario or a
    partially written JSONL file.
    """

    cache_version: str
    source_id: str
    mode: str
    total_records: int
    horizon_hours: int
    frequency_hours: float
    pv_model: str = ""
    demand_model: str = ""
    splits: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, manifest_path: str) -> ForecastCacheManifest:
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as exc:
            raise ForecastCacheError(
                f"forecast cache manifest not found: {manifest_path}"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ForecastCacheError("forecast cache manifest is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ForecastCacheError("forecast cache manifest must be a JSON object")
        try:
            return cls(
                cache_version=str(payload["cache_version"]),
                source_id=str(payload["source_id"]),
                mode=str(payload.get("mode", "unknown")),
                total_records=int(payload["total_records"]),
                horizon_hours=int(payload["horizon_hours"]),
                frequency_hours=float(payload["frequency_hours"]),
                pv_model=str(payload.get("pv_model", "")),
                demand_model=str(payload.get("demand_model", "")),
                splits=payload["splits"] if isinstance(payload.get("splits"), dict) else {},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ForecastCacheError(
                "forecast cache manifest is missing required fields"
            ) from exc


def _strict_series_mw(
    data: dict[str, Any], name: str, expected_points: int, line_number: int
) -> tuple[float, ...]:
    """Convert one cached PV/demand series to MW, failing closed on bad data."""
    kw_key = f"{name}_values_kw"
    mw_key = f"{name}_values_mw"
    if kw_key in data:
        values, scale, used_key = data[kw_key], 1.0 / 1000.0, kw_key
    elif mw_key in data:
        values, scale, used_key = data[mw_key], 1.0, mw_key
    else:
        raise ForecastCacheError(f"forecast cache line {line_number}: missing {kw_key!r}")
    if not isinstance(values, list) or len(values) != expected_points:
        actual = len(values) if isinstance(values, list) else "non-list"
        raise ForecastCacheError(
            f"forecast cache line {line_number}: {used_key!r} has {actual} values; "
            f"expected {expected_points}"
        )
    converted: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ForecastCacheError(
                f"forecast cache line {line_number}: {name} forecast is non-numeric"
            ) from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ForecastCacheError(
                f"forecast cache line {line_number}: {name} forecast is negative or non-finite"
            )
        converted.append(numeric * scale)
    return tuple(converted)


def _parse_grid_timestamp(
    text: str, action_interval_hours: float, line_number: int, field_name: str
) -> pd.Timestamp:
    """Parse a timestamp and require it to lie on the 15-minute action grid."""
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError) as exc:
        raise ForecastCacheError(
            f"forecast cache line {line_number}: {field_name} {text!r} is not a timestamp"
        ) from exc
    interval_seconds = int(round(action_interval_hours * 3600.0))
    if interval_seconds <= 0:
        raise ForecastCacheError(
            f"forecast cache line {line_number}: action_interval_hours is too small"
        )
    seconds_into_day = ts.hour * 3600 + ts.minute * 60 + ts.second
    if ts.microsecond or seconds_into_day % interval_seconds != 0:
        raise ForecastCacheError(
            f"forecast cache line {line_number}: {field_name} {text!r} is not aligned to "
            f"the {action_interval_hours}h action grid"
        )
    return ts


class ForecastCache:
    """In-memory cache for JSONL forecast snapshots keyed by timestamp or index."""

    def __init__(self, records: dict[str, ForecastSnapshot] | None = None) -> None:
        self._records: dict[str, ForecastSnapshot] = records or {}
        self._covering_ts: list[pd.Timestamp] | None = None
        self._covering_keys: list[str] | None = None

    @classmethod
    def from_jsonl(cls, jsonl_path: str) -> ForecastCache:
        """Load ForecastCache from a JSONL file containing forecast snapshot records."""
        records: dict[str, ForecastSnapshot] = {}
        with open(jsonl_path, encoding="utf-8") as f:
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

    @classmethod
    def load(
        cls,
        jsonl_path: str,
        manifest_path: str,
        *,
        expected_source_id: str,
        action_interval_hours: float = 0.25,
    ) -> ForecastCache:
        """Strictly load a pre-generated cache plus its manifest for RL training.

        Unlike :meth:`from_jsonl`, this fails closed: a missing manifest,
        malformed JSONL, a scenario/source mismatch, timestamps off the
        15-minute action grid, or a horizon that is not leakage-free all raise
        instead of being silently defaulted.
        """
        manifest = ForecastCacheManifest.from_json(manifest_path)
        if manifest.source_id != expected_source_id:
            raise ForecastSourceError(expected_source_id, manifest.source_id)
        if action_interval_hours <= 0.0:
            raise ForecastCacheError("action_interval_hours must be positive")

        records: dict[str, ForecastSnapshot] = {}
        try:
            with open(jsonl_path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except FileNotFoundError as exc:
            raise ForecastCacheError(f"forecast cache not found: {jsonl_path}") from exc

        for line_number, raw in enumerate(lines, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ForecastCacheError(
                    f"forecast cache line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(data, dict):
                raise ForecastCacheError(
                    f"forecast cache line {line_number} must be a JSON object"
                )
            snapshot = cls._parse_strict_record(
                data, line_number, manifest, expected_source_id, action_interval_hours
            )
            records[snapshot.issued_at] = snapshot

        if len(records) != manifest.total_records:
            raise ForecastCacheError(
                f"forecast cache holds {len(records)} records; manifest declares "
                f"{manifest.total_records}"
            )
        return cls(records)

    @staticmethod
    def _parse_strict_record(
        data: dict[str, Any],
        line_number: int,
        manifest: ForecastCacheManifest,
        expected_source_id: str,
        action_interval_hours: float,
    ) -> ForecastSnapshot:
        try:
            issued_at = str(data["issued_at"])
            horizon = int(data.get("horizon_hours", manifest.horizon_hours))
            frequency = float(data.get("frequency_hours", manifest.frequency_hours))
            source_id = str(data["source_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ForecastCacheError(
                f"forecast cache line {line_number}: missing issued_at/source_id or "
                "non-numeric horizon/frequency"
            ) from exc

        if not source_id:
            raise ForecastCacheError(f"forecast cache line {line_number}: missing source_id")
        if source_id != expected_source_id:
            raise ForecastSourceError(expected_source_id, source_id)
        if horizon != manifest.horizon_hours or not math.isclose(
            frequency, manifest.frequency_hours
        ):
            raise ForecastCacheError(
                f"forecast cache line {line_number}: horizon/frequency {horizon}h/"
                f"{frequency}h disagree with manifest {manifest.horizon_hours}h/"
                f"{manifest.frequency_hours}h"
            )
        if horizon <= 0 or frequency <= 0.0 or not math.isfinite(frequency):
            raise ForecastCacheError(
                f"forecast cache line {line_number}: horizon/frequency must be positive"
            )

        expected_points = int(round(horizon / frequency))
        pv_mw = _strict_series_mw(data, "pv", expected_points, line_number)
        demand_mw = _strict_series_mw(data, "demand", expected_points, line_number)

        raw_timestamps = data.get("timestamps")
        if not isinstance(raw_timestamps, list) or len(raw_timestamps) != expected_points:
            raise ForecastCacheError(
                f"forecast cache line {line_number}: timestamps must list "
                f"{expected_points} entries"
            )
        timestamps = tuple(str(ts) for ts in raw_timestamps)

        issued = _parse_grid_timestamp(issued_at, action_interval_hours, line_number, "issued_at")
        points = [
            _parse_grid_timestamp(ts, action_interval_hours, line_number, "timestamps")
            for ts in timestamps
        ]
        step_seconds = frequency * 3600.0
        try:
            first_offset = (points[0] - issued).total_seconds()
            spacings = [(later - earlier).total_seconds() for earlier, later in pairwise(points)]
        except TypeError as exc:
            raise ForecastCacheError(
                f"forecast cache line {line_number}: timestamps mix timezone-aware and "
                "naive values"
            ) from exc
        if first_offset <= 0.0:
            raise ForecastCacheError(
                f"forecast cache line {line_number}: horizon is not leakage-free; "
                "the first point is not after issued_at"
            )
        if first_offset > step_seconds:
            raise ForecastCacheError(
                f"forecast cache line {line_number}: horizon starts more than one "
                "frequency step after issued_at"
            )
        if any(not math.isclose(spacing, step_seconds) for spacing in spacings):
            raise ForecastCacheError(
                f"forecast cache line {line_number}: timestamps are not evenly spaced "
                "by frequency_hours"
            )

        context_time = data.get("context_time")
        if context_time is not None:
            context_ts = _parse_grid_timestamp(
                str(context_time), action_interval_hours, line_number, "context_time"
            )
            try:
                leaked = context_ts > issued
            except TypeError as exc:
                raise ForecastCacheError(
                    f"forecast cache line {line_number}: context_time mixes timezone-aware "
                    "and naive values"
                ) from exc
            if leaked:
                raise ForecastCacheError(
                    f"forecast cache line {line_number}: context_time is after issued_at"
                )

        return ForecastSnapshot(
            issued_at=issued_at,
            horizon_hours=horizon,
            frequency_hours=frequency,
            model_version=str(data.get("model_version", "cached")),
            pv_target=str(data.get("pv_target", "pv_avg")),
            demand_target=str(data.get("demand_target", "demand")),
            timestamps=timestamps,
            pv_values_mw=pv_mw,
            demand_values_mw=demand_mw,
            source_id=source_id,
            context_time=context_time,
            context_steps=int(data.get("context_steps", 0)),
            cold_start=bool(data.get("cold_start", False)),
            covariate_mode=str(data.get("covariate_mode", "cached")),
        )

    def cold_start_issued_at(self) -> tuple[str, ...]:
        """Issued-at keys whose snapshot was flagged as a cold-start forecast."""
        return tuple(key for key, snap in self._records.items() if snap.cold_start)

    def get_covering(self, timestamp: str) -> ForecastSnapshot | None:
        """Return the latest snapshot issued at or before ``timestamp``.

        This implements the rolling hourly contract in which one hourly forecast
        is reused for the four 15-minute decisions in its hour: a request stamped
        ``HH:15``, ``HH:30``, or ``HH:45`` resolves to the ``HH:00`` snapshot.
        Returns ``None`` when no forecast has been issued yet.
        """
        if self._covering_ts is None or self._covering_keys is None:
            ordered = sorted(
                (pd.Timestamp(snap.issued_at), key) for key, snap in self._records.items()
            )
            self._covering_ts = [ts for ts, _ in ordered]
            self._covering_keys = [key for _, key in ordered]
        target = pd.Timestamp(timestamp)
        index = bisect.bisect_right(self._covering_ts, target) - 1
        if index < 0:
            return None
        return self._records[self._covering_keys[index]]
