"""MicrogridEnv — the Gymnasium RL environment.

Wraps *any* :class:`~microgrid_simulator.core.backend.MicrogridBackend` behind
the standard Gymnasium contract. The backend is chosen by name from the config
(``backend.name``: simple | pandapower | pypsa | opendss | auto) — this module
never imports a simulation engine directly.

Action (continuous, normalised to [-1, 1] for SB3 compatibility):
    a[0]        battery power      -> [-max_discharge, +max_charge] MW
    a[1..n_ev]  EV charge rates    -> [0, ev_max_charge] MW each
    a[-3]       diesel on/off      -> on when a[-3] > 0 (only if diesel enabled)
    a[-2]       diesel setpoint    -> [0, diesel_max_kw] (only if diesel enabled)
    a[-1]       PV curtailment     -> [0, 1] fraction

Observation (float32 vector): bus voltages, static loads, PV generation,
battery SoC/SoH, EV SoCs, grid import, PV availability, diesel output, plus
cyclical time-of-day features.

Episodes: when a real demand trace is configured (``demand.file``) each
``reset()`` draws a random 24h window from it, so the agent sees varied real
demand shapes instead of one synthetic sinusoid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from microgrid_simulator.backends import create_backend
from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.core.backend import MicrogridBackend
from microgrid_simulator.core.types import ControlAction, GridState
from microgrid_simulator.digital_twin.replay import TelemetryWindow, load_fixed_telemetry_window
from microgrid_simulator.model.reward import compute_reward


def decode_action(
    a: np.ndarray, settings: Settings, n_ev: int, diesel_enabled: bool
) -> ControlAction:
    """Map a normalised [-1, 1] action vector onto physical units."""
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    bcfg = settings.battery

    # battery: symmetric map [-1,1] -> [-max_discharge, +max_charge]
    raw_b = float(a[0])
    battery_p_mw = raw_b * bcfg.max_charge_mw if raw_b >= 0 else raw_b * bcfg.max_discharge_mw

    # EVs: [-1,1] -> [0, ev_max] (unipolar charge)
    ev_max = settings.ev.max_charge_mw
    ev_p_mw = [((float(a[1 + i]) + 1.0) / 2.0) * ev_max for i in range(n_ev)]

    # diesel: discrete on/off (threshold at 0) + continuous setpoint
    diesel_on = False
    diesel_set_mw = 0.0
    if diesel_enabled:
        diesel_on = float(a[-3]) > 0.0
        diesel_set_mw = ((float(a[-2]) + 1.0) / 2.0) * (settings.diesel.max_kw / 1000.0)

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
    dim = 1 + n_ev + (2 if diesel_enabled else 0) + 1
    a = np.zeros(dim, dtype=np.float32)
    bcfg = settings.battery
    p = action.battery_p_mw
    a[0] = p / max(bcfg.max_charge_mw, 1e-9) if p >= 0 else p / max(bcfg.max_discharge_mw, 1e-9)
    ev_max = max(settings.ev.max_charge_mw, 1e-9)
    for i in range(n_ev):
        p_ev = action.ev_p_mw[i] if i < len(action.ev_p_mw) else 0.0
        a[1 + i] = (p_ev / ev_max) * 2.0 - 1.0
    if diesel_enabled:
        a[-3] = 1.0 if action.diesel_on else -1.0
        max_mw = max(settings.diesel.max_kw / 1000.0, 1e-9)
        a[-2] = (action.diesel_setpoint_mw / max_mw) * 2.0 - 1.0
    a[-1] = action.pv_curtail * 2.0 - 1.0
    return np.clip(a, -1.0, 1.0)


def build_observation(state: GridState, diesel_enabled: bool) -> np.ndarray:
    """Flatten a :class:`GridState` into the env's float32 observation vector."""
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
    ) -> None:
        super().__init__()
        if settings is None:
            settings = load_settings(config_path)
        self.settings = settings
        self.render_mode = render_mode
        self.dt = float(settings.topology.timestep_hours)
        self.max_steps = int(round(settings.episode.horizon_hours / self.dt))
        self.telemetry_window: TelemetryWindow | None = load_fixed_telemetry_window(
            settings, self.max_steps
        )

        self.backend = backend if backend is not None else create_backend(settings, backend_name)
        self.n_ev = settings.topology.n_ev
        self._steps = 0
        self._last_state: GridState = self.backend.reset()

        # --- action space: [battery, ev_0..ev_k, (diesel_on, diesel_set), curtail] ---
        self.diesel_enabled = settings.diesel.enabled
        self.action_dim = 1 + self.n_ev + (2 if self.diesel_enabled else 0) + 1
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

        # Draw a fresh random window of the real demand trace per episode
        # (never walk the file sequentially).
        window = None
        pv_window = None
        info_extra: dict[str, Any] = {}
        trace = self.backend.demand_trace
        if self.telemetry_window is not None:
            window = self.telemetry_window.demand_mw
            pv_window = self.telemetry_window.pv_mw
            info_extra = {
                "demand_window_start_index": 0,
                "demand_window_start_time": str(self.telemetry_window.first_evaluated_timestamp),
                "telemetry_context_time": str(self.telemetry_window.context_timestamp),
                "telemetry_end_time": str(self.telemetry_window.last_evaluated_timestamp),
                "telemetry_source_files": self.telemetry_window.source_files,
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

        self._last_state = self.backend.reset(
            seed=seed, demand_window_mw=window, pv_window_mw=pv_window
        )
        obs = self._build_obs(self._last_state)
        return obs, {"timestamp": self._last_state.timestamp, **info_extra}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        control = self._decode_action(action)
        state = self.backend.step(control)
        self._last_state = state

        reward, breakdown = compute_reward(
            state, self.backend.battery, self.settings.reward, self.dt
        )

        self._steps += 1
        terminated = not state.solver_ok
        truncated = self._steps >= self.max_steps

        obs = self._build_obs(state)
        info: dict[str, Any] = {
            "timestamp": state.timestamp,
            "grid_import_mw": state.grid_import_mw,
            "soc": self.backend.battery.soc,
            "soh": self.backend.battery.soh,
            "pv_wasted_mw": max(0.0, state.pv_available_mw - state.pv_used_mw),
            "diesel_p_mw": state.diesel_p_mw,
            "diesel_on": state.diesel_on,
            "demand_is_real": state.demand_is_real,
            "pv_is_real": state.pv_is_real,
            **breakdown.as_info(),
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
        return build_observation(state, self.diesel_enabled)
