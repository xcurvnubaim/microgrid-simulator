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

import numpy as np
import pandas as pd

from microgrid_simulator.backends.pypsa_backend import PyPSAOperationalBackend, plan_to_actions
from microgrid_simulator.controllers.base import Controller
from microgrid_simulator.core.types import ControlAction, GridState
from microgrid_simulator.forecast.cache import ForecastCache
from microgrid_simulator.forecast.errors import ForecastCacheError


def _resample_hourly_to_quarter(
    cache: ForecastCache, issued_at: str, n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the hourly cached forecast covering ``issued_at`` to 15-minute steps.

    Each hourly value is held constant across its four 15-minute control ticks,
    matching the env's forecast reuse contract. Values are clipped to length
    ``n_steps`` and zero-padded beyond the cached horizon.
    """
    snapshot = cache.get_covering(issued_at)
    if snapshot is None:
        raise ForecastCacheError(f"no cached forecast covers {issued_at}")
    ticks_per_hour = 4  # 60 min / 15 min
    demand = np.zeros(n_steps, dtype=np.float64)
    pv = np.zeros(n_steps, dtype=np.float64)
    for hour_index, (pv_h, demand_h) in enumerate(
        zip(snapshot.pv_values_mw, snapshot.demand_values_mw, strict=False)
    ):
        start = hour_index * ticks_per_hour
        if start >= n_steps:
            break
        end = min(start + ticks_per_hour, n_steps)
        demand[start:end] = demand_h
        pv[start:end] = pv_h
    return demand, pv


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
            demand_mw: np.ndarray | None = None
            pv_mw: np.ndarray | None = None
            if self._forecast_cache is not None:
                issued_at = self._current_issued_at(state)
                demand_mw, pv_mw = _resample_hourly_to_quarter(
                    self._forecast_cache, issued_at, self.horizon_steps
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
