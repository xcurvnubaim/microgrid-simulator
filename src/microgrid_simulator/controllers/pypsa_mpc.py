"""PyPSAMPCController — rolling-horizon optimal dispatch baseline.

Every ``rolling_horizon_hours`` the controller re-solves a PyPSA
unit-commitment/dispatch over the next ``horizon_hours`` (perfect foresight of
demand and PV from the driving backend) and plays the plan back one tick at a
time. This is the classic MPC/expert baseline an RL policy should be compared
against — and, run offline, a per-episode dispatch benchmark.

Requires the ``ops`` extra and a
:class:`~microgrid_simulator.backends.pypsa_backend.PyPSAOperationalBackend`
driving the episode (that backend supplies both the forecast and the physics).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from microgrid_simulator.backends.pypsa_backend import PyPSAOperationalBackend, plan_to_actions
from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState
from microgrid_simulator.forecast.cache import ForecastCache
from microgrid_simulator.forecast.errors import ForecastCacheError, ForecastError
from microgrid_simulator.forecast.snapshot import ForecastSnapshot

SnapshotProvider = Callable[[], ForecastSnapshot | tuple[str | None, ForecastSnapshot | None] | None]


def _align_snapshot_to_steps(
    snapshot: ForecastSnapshot,
    issued_at: str,
    n_steps: int,
    *,
    control_interval_hours: float,
    expected_source_id: str | None,
    pad_incomplete: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Align a validated forecast snapshot onto control-interval decision steps.

    Frequency-aware: a direct 15-minute forecast (``frequency_hours`` equals the
    control interval) is consumed one point per tick, while a coarser forecast
    (e.g. hourly F0) has each point held across ``frequency / control_interval``
    ticks.

    Fails closed instead of silently padding: the snapshot must be leakage-free,
    source-matching, aligned to the decision grid, cover the full requested
    horizon, and have timestamps one-to-one with values. A partial horizon is
    rejected rather than zero-padding missing demand/PV to zero, unless
    ``pad_incomplete`` is set (retained for the legacy hourly F0 baseline).
    """
    if (
        expected_source_id is not None
        and snapshot.source_id
        and snapshot.source_id != expected_source_id
    ):
        raise ForecastError(
            f"forecast source {snapshot.source_id!r} does not match expected "
            f"{expected_source_id!r}"
        )

    if not snapshot.timestamps:
        raise ForecastError("forecast snapshot carries no timestamps; cannot align to the grid")
    if len(snapshot.timestamps) != len(snapshot.pv_values_mw) != len(snapshot.demand_values_mw):
        raise ForecastError(
            "forecast snapshot timestamps must align one-to-one with pv/demand values"
        )

    decision = _parse_ts(issued_at, "decision time")
    issued_snap = _parse_ts(snapshot.issued_at, "snapshot issued_at")
    if issued_snap > decision:
        raise ForecastError(
            "forecast snapshot was issued in the future of the decision time; "
            "refusing for leakage-free MPC replay"
        )
    if snapshot.context_time is not None:
        context = _parse_ts(snapshot.context_time, "snapshot context_time")
        if context > decision:
            raise ForecastError(
                "forecast snapshot context_time is after the decision time; it is not causal"
            )

    start_tss = [_parse_ts(ts, "snapshot target timestamp") for ts in snapshot.timestamps]
    if start_tss[0] <= issued_snap:
        raise ForecastError(
            "forecast horizon is not leakage-free; the first target is not after issued_at"
        )

    frequency = snapshot.frequency_hours
    if frequency < control_interval_hours - 1e-9:
        raise ForecastError(
            f"forecast frequency {frequency:g}h is finer than the "
            f"{control_interval_hours:g}h control interval"
        )
    ratio = frequency / control_interval_hours
    if not math.isclose(ratio, round(ratio)):
        raise ForecastError(
            f"forecast frequency {frequency:g}h is not an integer multiple of the "
            f"{control_interval_hours:g}h control interval"
        )
    ticks_per_point = max(1, int(round(ratio)))

    covered_ticks = len(snapshot.pv_values_mw) * ticks_per_point
    if covered_ticks < n_steps:
        if pad_incomplete:
            covered_ticks = n_steps
        else:
            raise ForecastError(
                f"forecast horizon covers {len(snapshot.pv_values_mw) * ticks_per_point} "
                f"control ticks but MPC needs {n_steps}; refusing an incomplete horizon"
            )

    demand = np.zeros(covered_ticks, dtype=np.float64)
    pv = np.zeros(covered_ticks, dtype=np.float64)
    for index, (pv_v, demand_v) in enumerate(
        zip(snapshot.pv_values_mw, snapshot.demand_values_mw, strict=False)
    ):
        start = index * ticks_per_point
        end = min(start + ticks_per_point, covered_ticks)
        demand[start:end] = demand_v
        pv[start:end] = pv_v

    return demand[:n_steps], pv[:n_steps]


def _parse_ts(value: str, label: str) -> pd.Timestamp:
    """Parse a naive timestamp for grid alignment; fail closed on error."""
    try:
        ts = pd.Timestamp(value).tz_localize(None)
    except (TypeError, ValueError) as exc:
        raise ForecastError(f"forecast snapshot has an invalid {label}: {value!r}") from exc
    if pd.isna(ts):
        raise ForecastError(f"forecast snapshot has an invalid {label}: {value!r}")
    return ts


def _align_forecast_to_steps(
    cache: ForecastCache,
    issued_at: str,
    n_steps: int,
    *,
    control_interval_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Align the cached forecast covering ``issued_at`` onto control-interval steps.

    Legacy cache-backed path retained for offline experiments and tests. New
    snapshot-backed callers use :func:`_align_snapshot_to_steps` directly.
    """
    snapshot = cache.get_covering(issued_at)
    if snapshot is None:
        raise ForecastCacheError(f"no cached forecast covers {issued_at}")
    return _align_snapshot_to_steps(
        snapshot,
        issued_at,
        n_steps,
        control_interval_hours=control_interval_hours,
        expected_source_id=None,
        pad_incomplete=True,  # legacy F0 baseline zero-pads beyond its horizon
    )


def _resample_hourly_to_quarter(
    cache: ForecastCache, issued_at: str, n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy hourly F0 path: hold each hourly value across four 15-minute ticks.

    Retained for the incumbent hourly-forecast baseline (F0). New direct
    15-minute F2 caches are consumed through
    :func:`_align_forecast_to_steps` instead.
    """
    return _align_forecast_to_steps(cache, issued_at, n_steps, control_interval_hours=0.25)


class PyPSAMPCController(Controller):
    name = "pypsa_mpc"

    def __init__(
        self,
        settings: object = None,  # kept positional-compatible; settings come from the backend
        backend: PyPSAOperationalBackend | None = None,
        horizon_steps: int | None = None,
        replan_every: int | None = None,
        forecast_cache: ForecastCache | None = None,
        forecast_issued_at: str | None = None,
    ) -> None:
        if backend is None:
            raise ValueError("PyPSAMPCController needs a PyPSAOperationalBackend instance")
        super().__init__(backend.settings)
        self.backend = backend
        self.horizon_steps = horizon_steps or backend.horizon_steps()
        self.replan_every = replan_every or backend.rolling_steps()
        self._actions: list[ControlAction] = []
        self._plans: list[pd.DataFrame] = []
        self._since_replan = 0
        # Cached-forecast MPC: when a cache is given the controller re-solves
        # against the leakage-free cached forecast for the current issued-at
        # time instead of the backend's perfect-foresight telemetry.
        self._forecast_cache = forecast_cache
        self._forecast_issued_at = forecast_issued_at
        self._control_interval_hours = float(backend.settings.topology.timestep_hours)
        # Live-forecast MPC: when a snapshot callback is supplied the controller
        # uses the current validated snapshot at each replan instead of the
        # backend's perfect-foresight telemetry.
        self._snapshot_provider: SnapshotProvider | None = None
        self._expected_source_id: str | None = None

    def set_live_forecast_source(
        self,
        provider: SnapshotProvider,
        source_id: str | None = None,
        expected_source_id: str | None = None,
    ) -> None:
        """Route live forecasting through a per-replan snapshot provider.

        ``provider`` is a zero-argument callable returning the current validated
        ``ForecastSnapshot`` (or ``None`` when none is available). ``source_id``
        and ``expected_source_id`` are accepted for compatibility; the callable
        may also return a tuple.
        """
        self._snapshot_provider = provider
        self._expected_source_id = expected_source_id or source_id

    def reset(self) -> None:
        self._actions = []
        self._plans = []
        self._since_replan = 0

    @property
    def last_plan(self) -> pd.DataFrame | None:
        return self._plans[-1] if self._plans else None

    def _current_issued_at(self, state: GridState) -> str:
        """Absolute replay time used to look up the covering cached forecast."""
        if self._forecast_issued_at is not None:
            return self._forecast_issued_at
        if not np.isfinite(state.timestamp):
            raise ForecastCacheError("cannot resolve cached forecast time from state")
        base = pd.Timestamp(self.backend.settings.episode.telemetry_start)
        return str((base + pd.to_timedelta(state.timestamp, unit="h")).isoformat())

    def act(self, state: GridState) -> ControlAction:
        if not self._actions or self._since_replan >= self.replan_every:
            soc = state.soc[0] if state.soc else None
            if self._forecast_cache is not None:
                issued_at = self._current_issued_at(state)
                demand_mw, pv_mw = _align_forecast_to_steps(
                    self._forecast_cache,
                    issued_at,
                    self.horizon_steps,
                    control_interval_hours=self._control_interval_hours,
                )
            elif self._snapshot_provider is not None:
                demand_mw, pv_mw = self._resolve_live_snapshot(state)
            else:
                raise ForecastError(
                    "PyPSA MPC needs a cached forecast or a live forecast snapshot; "
                    "refusing the perfect-foresight telemetry fallback"
                )
            plan = self.backend.optimize_horizon(
                self.horizon_steps,
                soc_init=soc,
                diesel_on_init=state.diesel_on,
                demand_mw=demand_mw,
                pv_mw=pv_mw,
            )
            self._plans.append(plan)
            self._actions = plan_to_actions(plan)
            self._since_replan = 0
        self._since_replan += 1
        action = self._actions.pop(0)
        action.ev_p_mw = self._ev_charge()
        # Advance the telemetry window so the next re-plan forecasts from the
        # offset that matches the current tick.
        self.backend.advance_window()
        return action

    def _resolve_live_snapshot(self, state: GridState) -> tuple[np.ndarray, np.ndarray]:
        if self._snapshot_provider is None:
            raise ForecastError(
                "no live forecast snapshot provider is configured for MPC "
            )
        provided = self._snapshot_provider()
        if isinstance(provided, tuple):
            if len(provided) == 2:
                issued_at, snapshot = provided
            else:
                snapshot = None
        else:
            issued_at = None
            snapshot = provided
        if snapshot is None:
            raise ForecastError(
                "no live forecast snapshot is available for MPC at this replan; "
                "failing closed instead of using perfect foresight"
            )
        if not isinstance(snapshot, ForecastSnapshot):
            raise TypeError(
                "live forecast provider must return a ForecastSnapshot or None"
            )
        if issued_at is None:
            issued_at = self._current_issued_at(state)
        return _align_snapshot_to_steps(
            snapshot,
            issued_at,
            self.horizon_steps,
            control_interval_hours=self._control_interval_hours,
            expected_source_id=self._expected_source_id,
        )
