"""MicrogridEnv — the Gymnasium RL environment.

Wraps the temporarily hardcoded pandapower AC runtime behind the standard
Gymnasium contract. Backend arguments remain API-compatible but are normalized
by the backend factory while the lock is active.

Action (compact full-EMS, continuous, normalised to [-1, 1] for SB3/SAC;
    RL Plan §Policy and environment contract):
    a[0]  battery command  -> [-max_discharge, +max_charge] MW
    a[1]  diesel command   -> > 0 commits the genset and maps linearly onto its
                               [min_kw, max_kw] setpoint band; <= 0 requests off.
                               The plant's diesel model still shapes the realised
                               output (start delay, ramp, min up/down lockouts).
    a[2]  PV curtailment   -> [0, 1] fraction. Kept as an explicit dimension so
                               legacy ``ControlAction`` controllers stay
                               encodable; trained policies should hold it at 0
                               and let the plant resolve spill.

Observation (float32 vector): bus voltages, static loads, PV generation,
battery SoC/SoH, EV SoCs, grid import, PV availability, diesel output, plus
cyclical time-of-day features. When forecasting is enabled, a forecast
availability flag followed by the configured number of hourly Chronos PV and
demand values in MW are appended.

Episodes: when a real demand trace is configured (``demand.file``) each
``reset()`` draws a random 24h window from it, so the agent sees varied real
demand shapes instead of one synthetic sinusoid.

Terminal-SOC contract (RL Plan §Acceptance criteria): an episode whose final
SOC lies outside ``settings.episode.terminal_soc_tolerance`` of the recorded
starting SOC terminates with a one-shot terminal penalty on the last step, so
the agent is trained to return the battery to its start-of-episode charge
(the same target the rule/MPC comparisons are judged against). The deviation
and penalty are exposed in ``info``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from microgrid_simulator.backends import create_backend
from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.core.backend import MicrogridBackend
from microgrid_simulator.core.scenario import Scenario
from microgrid_simulator.core.types import ControlAction, GridState
from microgrid_simulator.digital_twin.replay import TelemetryWindow, load_fixed_telemetry_window
from microgrid_simulator.forecast import (
    ForecastCache,
    ForecastClient,
    ForecastContext,
    ForecastError,
    ForecastSnapshot,
    ForecastSourceError,
    StrictCachedForecastClient,
)
from microgrid_simulator.model.reward import compute_reward
from microgrid_simulator.rl.sampler import RandomEpisodeSampler, SplitName
from microgrid_simulator.rl.uncertainty import (
    SampledUncertainty,
    UncertaintySampler,
    apply_plant_mismatch,
)


def decode_action(
    a: np.ndarray, settings: Settings, n_ev: int, diesel_enabled: bool
) -> ControlAction:
    """Map a normalised [-1, 1] compact action vector onto physical units.

    Compact layout: ``[battery_command, diesel_command, pv_curtail]``. Extra
    dimensions between the battery command and the final curtailment slot are
    treated as per-EV charge commands so legacy EV-enabled callers keep
    decoding; the active campus topology has ``n_ev == 0``.
    """
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    bcfg = settings.battery

    # battery: symmetric map [-1,1] -> [-max_discharge, +max_charge]
    raw_b = float(a[0])
    battery_p_mw = raw_b * bcfg.max_charge_mw if raw_b >= 0 else raw_b * bcfg.max_discharge_mw

    # EVs: [-1,1] -> [0, ev_max] (unipolar charge)
    ev_max = settings.ev.max_charge_mw
    ev_p_mw = [((float(a[1 + i]) + 1.0) / 2.0) * ev_max for i in range(n_ev)]

    # diesel: one continuous command. > 0 commits the genset and maps onto the
    # [min_kw, max_kw] setpoint band (dispatch-to-setpoint); <= 0 requests off.
    # The genset model still applies its own start/ramp/min-up-down shaping.
    diesel_on = False
    diesel_set_mw = 0.0
    if diesel_enabled:
        raw_d = float(a[1 + n_ev])
        diesel_on = raw_d > 0.0
        if diesel_on:
            lo_mw = min(settings.diesel.min_kw, settings.diesel.max_kw) / 1000.0
            hi_mw = settings.diesel.max_kw / 1000.0
            diesel_set_mw = lo_mw + raw_d * (hi_mw - lo_mw)

    # curtailment: [-1,1] -> [0, 1]
    curtail = (float(a[-1]) + 1.0) / 2.0
    return ControlAction(
        battery_p_mw=battery_p_mw,
        ev_p_mw=ev_p_mw,
        pv_curtail=curtail,
        diesel_on=diesel_on,
        diesel_setpoint_mw=diesel_set_mw,
    )


def encode_action(
    action: ControlAction, settings: Settings, n_ev: int, diesel_enabled: bool
) -> np.ndarray:
    """Inverse of :func:`decode_action` — lets controllers that emit physical
    :class:`ControlAction` drive the normalised Gymnasium action space."""
    dim = 1 + n_ev + (1 if diesel_enabled else 0) + 1
    a = np.zeros(dim, dtype=np.float32)
    bcfg = settings.battery
    p = action.battery_p_mw
    a[0] = p / max(bcfg.max_charge_mw, 1e-9) if p >= 0 else p / max(bcfg.max_discharge_mw, 1e-9)
    ev_max = max(settings.ev.max_charge_mw, 1e-9)
    for i in range(n_ev):
        p_ev = action.ev_p_mw[i] if i < len(action.ev_p_mw) else 0.0
        a[1 + i] = (p_ev / ev_max) * 2.0 - 1.0
    if diesel_enabled:
        if action.diesel_on:
            lo_mw = min(settings.diesel.min_kw, settings.diesel.max_kw) / 1000.0
            hi_mw = max(settings.diesel.max_kw / 1000.0, lo_mw + 1e-9)
            frac = np.clip((action.diesel_setpoint_mw - lo_mw) / (hi_mw - lo_mw), 0.0, 1.0)
            # Lift the exact band minimum just above zero: the decoder treats
            # a non-positive command as "off", so the smallest admissible
            # setpoint must stay strictly positive to round-trip.
            a[1 + n_ev] = max(frac, 1e-6)
        else:
            a[1 + n_ev] = -1.0
    a[-1] = action.pv_curtail * 2.0 - 1.0
    return np.clip(a, -1.0, 1.0)


def build_observation(
    state: GridState,
    diesel_enabled: bool,
    pv_forecast_mw: np.ndarray | None = None,
    demand_forecast_mw: np.ndarray | None = None,
    forecast_available: bool = False,
    initial_soc: float | None = None,
    target_soc: float | None = None,
    episode_progress: float = 0.0,
    forecast_horizon: int = 24,
) -> np.ndarray:
    """Flatten a :class:`GridState` into the env's float32 observation vector.

    ``forecast_horizon`` is the number of forecast points appended (the
    ``forecast.forecast_steps`` derivation — 96 for a 24 h / 0.25 h target), not
    ``horizon_hours``.
    """
    hour = state.timestamp % 24.0
    time_feats = [np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)]
    vals: list[float] = []
    vals += list(state.v_bus)
    vals += list(state.p_load)
    vals += list(state.p_gen)
    vals += list(state.soc)
    vals += list(state.soh)
    vals += list(state.ev_soc)
    vals += [state.grid_import_mw, state.pv_available_mw, state.load_demand_mw]
    if diesel_enabled:
        vals += [state.diesel_p_mw]
    vals += time_feats
    vals += [state.soc[0] if state.soc else 0.0]
    vals += [initial_soc if initial_soc is not None else (state.soc[0] if state.soc else 0.0)]
    vals += [target_soc if target_soc is not None else (state.soc[0] if state.soc else 0.0)]
    vals += [float(np.clip(episode_progress, 0.0, 1.0))]
    pv_values = np.zeros(forecast_horizon, dtype=np.float32)
    demand_values = np.zeros(forecast_horizon, dtype=np.float32)
    if pv_forecast_mw is not None:
        pv_values[: min(forecast_horizon, len(pv_forecast_mw))] = pv_forecast_mw[:forecast_horizon]
    if demand_forecast_mw is not None:
        demand_values[: min(forecast_horizon, len(demand_forecast_mw))] = (
            demand_forecast_mw[:forecast_horizon]
        )
    vals += [1.0 if forecast_available else 0.0]
    vals += list(pv_values)
    vals += list(demand_values)
    return np.asarray(vals, dtype=np.float32)


class MicrogridEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment over a pluggable microgrid backend."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        settings: Settings | None = None,
        config_path: str | Path | None = None,
        render_mode: str | None = None,
        backend: MicrogridBackend | None = None,
        backend_name: str | None = None,
        forecast_client: ForecastClient | None = None,
        episode_sampler: RandomEpisodeSampler | None = None,
        split: SplitName | None = None,
        telemetry_window: TelemetryWindow | None = None,
    ) -> None:
        super().__init__()
        if settings is None:
            settings = load_settings(config_path)
        self.settings = settings
        self.render_mode = render_mode
        self.dt = float(settings.topology.timestep_hours)
        self.max_steps = int(round(settings.episode.horizon_hours / self.dt))
        self._episode_sampler = episode_sampler or (
            RandomEpisodeSampler(settings, split=split, seed=0) if split else None
        )
        self.telemetry_window: TelemetryWindow | None = telemetry_window
        if telemetry_window is None and not self._episode_sampler:
            self.telemetry_window = load_fixed_telemetry_window(settings, self.max_steps)
        if forecast_client is not None:
            self._forecast_client = forecast_client
        elif (
            settings.forecast.enabled
            and settings.forecast.strict_cache
            and getattr(settings.rl, "forecast_mode", "cached") == "cached"
        ):
            if not settings.forecast.cache_path or not settings.forecast.manifest_path:
                raise ValueError("strict cached forecasting requires cache_path and manifest_path")
            source_id = settings.forecast.source_id or settings.scenario.name
            cache = ForecastCache.load(
                settings.forecast.cache_path,
                settings.forecast.manifest_path,
                expected_source_id=source_id,
                action_interval_hours=settings.topology.timestep_hours,
            )
            self._forecast_client = StrictCachedForecastClient(settings.forecast, cache)
        else:
            self._forecast_client = ForecastClient(settings.forecast)
        self._forecast_snapshot: ForecastSnapshot | None = None
        self._forecast_error: str | None = None
        self._forecast_origin_step = 0
        self._forecast_stale = False
        self._forecast_response_source: str | None = None
        self._forecast_pv_history_mw: list[float] = []
        self._forecast_demand_history_mw: list[float] = []

        if backend is not None:
            raise ValueError(
                "custom backend injection is disabled while runtime physics is "
                "hardcoded to pandapower AC"
            )
        self.backend = create_backend(settings, backend_name)
        self.n_ev = settings.topology.n_ev
        self._steps = 0
        self.diesel_enabled = settings.diesel.enabled
        self._uncertainty_sampler = UncertaintySampler.from_settings(settings, 0)
        self._active_uncertainty = SampledUncertainty()
        self._active_settings = settings
        self._last_state: GridState = self.backend.reset()
        # Terminal-SOC tracking (return-to-start contract); populated by reset().
        self._initial_soc: float = self.backend.battery.soc
        self._terminal_soc_deviation: float = 0.0
        self._terminal_soc_penalty: float = 0.0
        self._hard_unserved_triggered: bool = False
        self._last_observation: np.ndarray = self._build_obs(self._last_state)

        # --- action space: compact full-EMS [battery, ev_0..ev_k, (diesel_cmd), curtail] ---
        self.action_dim = 1 + self.n_ev + (1 if self.diesel_enabled else 0) + 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # --- observation space ---
        obs = self._build_obs(self._last_state)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    # -- gym API -----------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0
        self._reset_forecast_state()
        # Draw a fresh random window of the real demand trace per episode
        # (never walk the file sequentially).
        window = None
        pv_window = None
        info_extra: dict[str, Any] = {}
        trace = self.backend.demand_trace
        episode_window = (
            self._episode_sampler.sample_window()
            if self._episode_sampler
            else self.telemetry_window
        )
        self.telemetry_window = episode_window
        if episode_window is not None:
            window = episode_window.demand_mw
            pv_window = episode_window.pv_mw
            info_extra = {
                "demand_window_start_index": 0,
                "demand_window_start_time": str(episode_window.first_evaluated_timestamp),
                "telemetry_context_time": str(episode_window.context_timestamp),
                "telemetry_end_time": str(episode_window.last_evaluated_timestamp),
                "telemetry_source_files": episode_window.source_files,
                "pv_is_real": True,
            }
        elif trace is not None:
            if self.settings.demand.random_window:
                window, start_idx = trace.sample_window(self.np_random, self.max_steps)
            else:
                window, start_idx = trace.values_mw[: self.max_steps + 1].copy(), 0
            info_extra = {
                "demand_window_start_index": start_idx,
                "demand_window_start_time": trace.window_start_time(start_idx),
            }

        self._last_state = self._reset_plant(
            seed=seed, window=window, pv_window=pv_window
        )
        self._initial_soc = float(self.backend.battery.soc)
        self._terminal_soc_deviation = 0.0
        self._terminal_soc_penalty = 0.0
        self._initialize_forecast_history(self._last_state)
        self._load_forecast()
        obs = self._build_obs(self._last_state)
        self._last_observation = obs
        return obs, {
            "timestamp": self._last_state.timestamp,
            "initial_soc": self._initial_soc,
            "target_soc": self._initial_soc,
            **info_extra,
            **self._forecast_meta(),
            **self._active_uncertainty.as_info(),
        }

    def _reset_plant(
        self,
        seed: int | None,
        window: np.ndarray | None,
        pv_window: np.ndarray | None,
    ) -> GridState:
        """Reset the plant with this episode's uncertainty applied.

        Samples a fresh plant-mismatch draw from a reproducible, seed-derived
        stream, applies it to a thumbnail copy of ``settings``, and reconfigures
        the backend through a :class:`Scenario`. Forecast dropout/residual state
        resets here as well.
        """
        if self._uncertainty_sampler.enabled:
            base = self._uncertainty_sampler.cfg.seed or 0
            stream = np.random.default_rng((base * 10_003 + (seed or 0)) & 0xFFFFFFFF)
            sampler = UncertaintySampler(self.settings, stream)
            self._active_uncertainty = sampler.sample()
            self._active_settings = apply_plant_mismatch(
                self.settings, self._active_uncertainty
            )
        else:
            self._active_uncertainty = SampledUncertainty()
            self._active_settings = self.settings

        scenario = Scenario.from_settings(self._active_settings)
        return self.backend.reset(
            scenario=scenario, seed=seed, demand_window_mw=window, pv_window_mw=pv_window
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        control = self._decode_action(action)
        state = self.backend.step(control)
        self._last_state = state

        reward, breakdown = compute_reward(
            state, self.backend.battery, self.settings.reward, self.dt
        )

        self._steps += 1
        self._append_forecast_history(state)
        terminated = not state.solver_ok
        truncated = self._steps >= self.max_steps

        # Hard unserved-load constraint (islanded scenario): a tick with real
        # unserved load is an infeasible dispatch, so end the episode with a
        # one-shot penalty instead of only charging the per-kWh reward term.
        # Shortfalls at or below ``hard_unserved_tol_mw`` (sub-kW command
        # precision residuals around the SOC floor) are tolerated: continuous
        # battery actions cannot hit demand to 1e-6 MW precision, and ending
        # every episode on a microscopic shortfall prevents training from ever
        # observing a full-length episode. The soft ``w_unserved`` reward term
        # is still computed above and folded into the breakdown.
        self._hard_unserved_triggered = False
        hard_tol = float(getattr(self.settings.rl, "hard_unserved_tol_mw", 0.002))
        if getattr(self.settings.rl, "hard_unserved", False) and state.unserved_mw > hard_tol:
            penalty = float(getattr(self.settings.rl, "hard_unserved_penalty", 1000.0))
            reward -= penalty
            breakdown.constraint += penalty
            breakdown.total -= penalty
            terminated = True
            truncated = False
            self._hard_unserved_triggered = True

        # Terminal-SOC return-to-start contract: ending the episode away from
        # the recorded starting SOC ends the episode with a one-shot penalty,
        # so a policy cannot drain or overfill the battery for a short-horizon
        # gain the way an unconstrained finite-horizon MPC does.
        epcfg = self.settings.episode
        final_soc = float(self.backend.battery.soc)
        self._terminal_soc_deviation = abs(final_soc - self._initial_soc)
        self._terminal_soc_penalty = 0.0
        terminal_soc_met = self._terminal_soc_deviation <= epcfg.terminal_soc_tolerance
        if truncated and not terminal_soc_met:
            self._terminal_soc_penalty = (
                self._terminal_soc_deviation * epcfg.terminal_soc_penalty
            )
            reward -= self._terminal_soc_penalty
            breakdown.constraint += self._terminal_soc_penalty
            breakdown.total -= self._terminal_soc_penalty
            terminated = True
            truncated = False

        if self.settings.forecast.enabled and self.settings.forecast.refresh_each_step:
            self._load_forecast()

        obs = self._build_obs(state)
        self._last_observation = obs
        pv_forecast, demand_forecast = self._forecast_vectors()
        info: dict[str, Any] = {
            "timestamp": state.timestamp,
            "grid_import_mw": state.grid_import_mw,
            "soc": self.backend.battery.soc,
            "soh": self.backend.battery.soh,
            "initial_soc": self._initial_soc,
            "target_soc": self._initial_soc,
            "episode_progress": min(1.0, self._steps / max(1, self.max_steps)),
            "terminal_soc_deviation": self._terminal_soc_deviation,
            "terminal_soc_tolerance": self.settings.episode.terminal_soc_tolerance,
            "terminal_soc_penalty": self._terminal_soc_penalty,
            "terminal_soc_met": terminal_soc_met,
            "hard_unserved_enabled": bool(getattr(self.settings.rl, "hard_unserved", False)),
            "hard_unserved_triggered": self._hard_unserved_triggered,
            "unserved_mw": state.unserved_mw,
            "load_demand_mw": state.load_demand_mw,
            "pv_wasted_mw": max(0.0, state.pv_available_mw - state.pv_used_mw),
            "diesel_p_mw": state.diesel_p_mw,
            "diesel_on": state.diesel_on,
            "diesel_load_serving_mw": state.diesel_load_serving_mw,
            "diesel_overgeneration_mw": state.diesel_overgeneration_mw,
            "excess_generation_mw": state.excess_generation_mw,
            "dump_load_mw": state.dump_load_mw,
            "network_loss_mw": state.network_loss_mw,
            "reference_balance_mw": state.reference_balance_mw,
            "demand_is_real": state.demand_is_real,
            "pv_is_real": state.pv_is_real,
            "pv_forecast_mw": self._current_pv_forecast_mw(),
            "demand_forecast_mw": self._current_demand_forecast_mw(),
            "forecast_available": self._forecast_is_available(),
            "forecast_horizon_mw": (pv_forecast.tolist() if self._forecast_is_available() else []),
            "demand_forecast_horizon_mw": (
                demand_forecast.tolist() if self._forecast_is_available() else []
            ),
            "forecast_issued_at": (
                self._forecast_snapshot.issued_at if self._forecast_snapshot else None
            ),
            "forecast_model_version": (
                self._forecast_snapshot.model_version if self._forecast_snapshot else None
            ),
            **self._forecast_meta(),
            **breakdown.as_info(),
            **self._active_uncertainty.as_info(),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        s = self._last_state
        return (
            f"t={s.timestamp:6.2f}h  import={s.grid_import_mw * 1000:7.2f}kW  "
            f"soc={self.backend.battery.soc:5.2f}  soh={self.backend.battery.soh:6.4f}  "
            f"pv={s.pv_used_mw * 1000:6.2f}/{s.pv_available_mw * 1000:6.2f}kW"
        )

    def close(self) -> None:
        self.backend.close()

    # -- action / observation encoding ------------------------------------
    def _decode_action(self, action: np.ndarray) -> ControlAction:
        return decode_action(action, self.settings, self.n_ev, self.diesel_enabled)

    def encode_action(self, action: ControlAction) -> np.ndarray:
        return encode_action(action, self.settings, self.n_ev, self.diesel_enabled)

    def _build_obs(self, state: GridState) -> np.ndarray:
        pv_forecast: np.ndarray | None
        demand_forecast: np.ndarray | None
        pv_forecast, demand_forecast = self._forecast_vectors()
        if not self.settings.forecast.enabled:
            pv_forecast = None
            demand_forecast = None
        return build_observation(
            state,
            self.diesel_enabled,
            pv_forecast_mw=pv_forecast,
            demand_forecast_mw=demand_forecast,
            forecast_available=self._forecast_is_available(),
            initial_soc=self._initial_soc,
            target_soc=self._initial_soc,
            episode_progress=self._steps / max(1, self.max_steps),
            forecast_horizon=self.settings.forecast.forecast_steps,
        )

    # -- forecast observation ---------------------------------------------
    def _reset_forecast_state(self) -> None:
        self._forecast_snapshot = None
        self._forecast_error = None
        self._forecast_origin_step = 0
        self._forecast_stale = False
        self._forecast_response_source = None
        self._forecast_pv_history_mw = []
        self._forecast_demand_history_mw = []

    def _initialize_forecast_history(self, state: GridState) -> None:
        """Start fallback history with exactly one preceding/current measurement."""
        if self.telemetry_window is None:
            self._forecast_pv_history_mw = [max(0.0, state.pv_available_mw)]
            self._forecast_demand_history_mw = [max(0.0, state.load_demand_mw)]

    def _append_forecast_history(self, state: GridState) -> None:
        """Append only the solved current step; never inspect a later replay point."""
        if self.telemetry_window is None:
            self._forecast_pv_history_mw.append(max(0.0, state.pv_available_mw))
            self._forecast_demand_history_mw.append(max(0.0, state.load_demand_mw))

    def _forecast_context(self) -> ForecastContext:
        if self.telemetry_window is not None:
            # Telemetry index zero is the preceding controller context.  At
            # reset this slices one value; after step N it ends at N.
            end = min(self._steps + 1, len(self.telemetry_window.demand_mw))
            pv_history = tuple(float(value) for value in self.telemetry_window.pv_mw[:end])
            demand_history = tuple(float(value) for value in self.telemetry_window.demand_mw[:end])
        else:
            pv_history = tuple(self._forecast_pv_history_mw)
            demand_history = tuple(self._forecast_demand_history_mw)
        return ForecastContext(
            source_id=self.settings.forecast.source_id or self.settings.scenario.name,
            frequency_hours=self.dt,
            pv_values_mw=pv_history,
            demand_values_mw=demand_history,
        )

    def _forecast_mode(self) -> str:
        """Return the deployable forecast input mode, failing closed."""
        mode = getattr(self.settings.rl, "forecast_mode", "cached")
        if mode not in {"cached", "none"}:
            raise ValueError("forecast_mode must be 'cached' or 'none'")
        return mode

    def _load_forecast(self) -> None:
        if not self.settings.forecast.enabled:
            return
        mode = self._forecast_mode()
        if mode == "none":
            # No-forecast ablation: keep the forecast observation dimensions
            # present but zeroed with availability off, so the observation
            # shape matches cached policies exactly.
            self._forecast_snapshot = None
            return
        if self._unc_forecast_dropped():
            # Forecast-service dropout: keep the retained snapshot (or none) so
            # the horizon advances/goes stale as if the update was suppressed.
            self._forecast_stale = self._forecast_snapshot is not None
            return
        try:
            snapshot = self._forecast_client.fetch(
                self._forecast_context(), issued_at=self._forecast_request_timestamp()
            )
        except ForecastSourceError as exc:
            # This is a correctness guard, not a transient outage: a campus
            # curve must disappear immediately from a pymgrid rollout.
            self._forecast_snapshot = None
            self._forecast_error = str(exc)
            self._forecast_stale = False
            self._forecast_response_source = exc.actual_source
            if self.settings.forecast.strict_cache:
                raise
            return
        except ForecastError as exc:
            # Preserve an unexhausted valid horizon through a transient error,
            # but make the state visible rather than silently refreshing it.
            self._forecast_error = str(exc)
            self._forecast_stale = self._forecast_snapshot is not None
            if self.settings.forecast.strict_cache:
                raise
            return

        self._forecast_response_source = snapshot.source_id
        if (
            self._forecast_snapshot is not None
            and snapshot.issued_at == self._forecast_snapshot.issued_at
        ):
            # Do not reset the origin.  The old horizon keeps advancing until
            # it is exhausted, after which the availability mask turns off.
            self._forecast_stale = True
            self._forecast_error = None
            return

        self._forecast_snapshot = snapshot
        self._forecast_origin_step = self._steps
        self._forecast_stale = False
        self._forecast_error = None

    def _forecast_request_timestamp(self) -> str | None:
        """Absolute replay time for leakage-free historical rolling forecasts."""
        if self.telemetry_window is None or not self.telemetry_window.timestamps_are_observed:
            return None
        index = min(self._steps, len(self.telemetry_window.timestamps) - 1)
        return str(pd.Timestamp(self.telemetry_window.timestamps[index]).isoformat())

    def _forecast_offset(self) -> int:
        snapshot = self._forecast_snapshot
        frequency = snapshot.frequency_hours if snapshot else 1.0
        elapsed_steps = max(0, self._steps - self._forecast_origin_step)
        return int(np.floor((elapsed_steps * self.dt + 1e-12) / frequency))

    def _forecast_is_available(self) -> bool:
        snapshot = self._forecast_snapshot
        return (
            snapshot is not None
            and self._forecast_offset() < len(snapshot.pv_values_mw)
            and self._forecast_offset() < len(snapshot.demand_values_mw)
        )

    def _forecast_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        horizon = self.settings.forecast.forecast_steps
        snapshot = self._forecast_snapshot
        pv_vector = np.zeros(horizon, dtype=np.float32)
        demand_vector = np.zeros(horizon, dtype=np.float32)
        if snapshot is None:
            return pv_vector, demand_vector
        offset = self._forecast_offset()
        pv_remaining = snapshot.pv_values_mw[offset : offset + horizon]
        demand_remaining = snapshot.demand_values_mw[offset : offset + horizon]
        pv_vector[: len(pv_remaining)] = pv_remaining
        demand_vector[: len(demand_remaining)] = demand_remaining
        return self._corrupt_forecast_vectors(pv_vector, demand_vector)

    def _current_pv_forecast_mw(self) -> float | None:
        if not self.settings.forecast.enabled or not self._forecast_is_available():
            return None
        return float(self._forecast_vectors()[0][0])

    def _current_demand_forecast_mw(self) -> float | None:
        if not self.settings.forecast.enabled or not self._forecast_is_available():
            return None
        return float(self._forecast_vectors()[1][0])

    def _unc_forecast_dropped(self) -> bool:
        """True when forecast-service dropout should suppress this refresh."""
        if not self._uncertainty_sampler.enabled:
            return False
        f = self._active_uncertainty
        if f.forecast_dropout_until_step > self._steps:
            self._forecast_stale = self._forecast_snapshot is not None
            return True
        fc = self.settings.uncertainty.forecast
        if fc.dropout_prob <= 0.0:
            return False
        rng = np.random.default_rng(
            (self._uncertainty_sampler.cfg.seed or 0) * 10_003 + self._steps
        )
        if rng.random() < fc.dropout_prob:
            f.forecast_dropout_until_step = self._steps + max(1, fc.dropout_block_steps)
            self._forecast_stale = self._forecast_snapshot is not None
            return True
        return False

    def _corrupt_forecast_vectors(
        self, pv_vector: np.ndarray, demand_vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add horizon-dependent residual noise to forecast vectors when enabled."""
        f = self._active_uncertainty
        pv_std = f.forecast_pv_residual_std
        demand_std = f.forecast_demand_residual_std
        if pv_std <= 0.0 and demand_std <= 0.0:
            return pv_vector, demand_vector
        rng = np.random.default_rng(
            (self._uncertainty_sampler.cfg.seed or 0) * 31_337 + self._steps
        )
        horizon = len(pv_vector)
        slope = self.settings.uncertainty.forecast.horizon_slope
        pv_noise = rng.normal(0.0, 1.0, size=horizon) * np.abs(pv_vector) * pv_std
        demand_noise = rng.normal(0.0, 1.0, size=horizon) * np.abs(demand_vector) * demand_std
        for h in range(horizon):
            pv_noise[h] *= 1.0 + slope * h
            demand_noise[h] *= 1.0 + slope * h
        return pv_vector + pv_noise, demand_vector + demand_noise

    def current_forecast(
        self,
    ) -> tuple[str | None, ForecastSnapshot | None]:
        """Return the live forecast snapshot for the current decision tick.

        Returns ``(request_timestamp, snapshot)`` where ``request_timestamp`` is
        the absolute replay time the current forecast was issued against and the
        snapshot is the most recently fetched (possibly stale) response. A stale
        or unavailable snapshot is still returned so the caller can decide how to
        treat it; the dashboard MPC path fails closed when it cannot provide a
        non-stale, full-horizon forecast.
        """
        if not self.settings.forecast.enabled:
            return None, None
        return self._forecast_request_timestamp(), self._forecast_snapshot

    def _forecast_meta(self) -> dict[str, Any]:
        cfg = self.settings.forecast
        base: dict[str, Any] = {
            "forecast_enabled": cfg.enabled,
            "forecast_available": self._forecast_is_available(),
            "forecast_requested_horizon_hours": cfg.horizon_hours,
            "forecast_refresh_each_step": cfg.refresh_each_step,
            "forecast_service_url": cfg.service_url,
            "forecast_requested_source": (
                self.settings.forecast.source_id or self.settings.scenario.name
            ),
            "forecast_source": self._forecast_response_source,
            "forecast_stale": self._forecast_stale,
            "forecast_age_steps": max(0, self._steps - self._forecast_origin_step),
            "forecast_uses_observed_replay_timestamp": bool(
                self.telemetry_window is not None and self.telemetry_window.timestamps_are_observed
            ),
            "forecast_error": self._forecast_error,
        }
        if self._forecast_snapshot is not None:
            base.update(self._forecast_snapshot.as_meta())
            base["forecast_source"] = self._forecast_snapshot.source_id
        # Snapshot metadata describes a valid response; availability also
        # depends on whether its retained rolling horizon still has an element.
        base["forecast_available"] = self._forecast_is_available()
        return base
