"""Shared rollout runner for the dashboard API.

Runs a controller (rule / idle / random / deterministic / schedule / rl / pypsa_rh) through one
episode and returns per-timestep rows the frontend can chart directly, plus episode meta
(demand source, window start, totals). ``stream_rollout`` yields the same rows one tick at
a time so the API can stream them to the UI as they are solved.

The ``rl`` policy loads a trained SB3 artifact and predicts from the env's own observation
builder, so training and dashboard evaluation see identical inputs.

The ``pypsa_rh`` policy uses a persistent ``PyPSARollingHorizonController`` that
re-plans at ``backend.rolling_horizon_hours`` intervals against the configured leakage-free
cached forecast. Each commanded action is executed against the pandapower AC plant, and
realized SOC/diesel state is fed back into every re-plan.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from microgrid_simulator.backends import resolve_backend_name
from microgrid_simulator.config import Settings
from microgrid_simulator.controllers import (
    DeterministicController,
    ManualScheduleController,
    RuleBasedController,
)
from microgrid_simulator.core.types import ControlAction, GridState
from microgrid_simulator.env import MicrogridEnv
from microgrid_simulator.rl.env import decode_action

LOGGER = logging.getLogger(__name__)

POLICIES = ("rule", "idle", "random", "deterministic", "schedule", "rl", "pypsa_rh")
LEGACY_POLICY_ALIASES = {"mpc": "pypsa_rh"}


class _RollingHorizonController:
    """Persistent project rolling-horizon controller that replans across ticks."""

    def __init__(self, controller: Any, rh_summary: dict[str, Any]) -> None:
        self._controller = controller
        self.rh_summary = rh_summary

    def act(self, env: MicrogridEnv) -> np.ndarray:
        return env.encode_action(self._controller.act(env._last_state))  # noqa: SLF001

    def reset(self) -> None:
        self._controller.reset()

    @property
    def last_plan(self) -> Any:
        return self._controller.last_plan


def _setup_pypsa_rh_controller(
    settings: Settings,
    env: MicrogridEnv,
) -> tuple[Any, dict[str, Any]]:
    """Create the project PyPSA rolling-horizon controller with causal forecasts.

    The planning backend receives the same telemetry window as the pandapower
    plant so its physical dimensions are consistent, but the controller override
    (a strict cache or a validated live snapshot) ensures only forecasts drive
    optimization decisions — never the backend's perfect-foresight telemetry.

    ``strict_cache: true`` uses the pre-generated, manifest-validated cache.
    ``strict_cache: false`` requires a live HTTP forecast service (``forecast``
    enabled and ``forecast_mode != none``) whose snapshots are consumed at each
    replan through the environment.

    Returns the controller and a PyPSA-RH summary dict for reporting. Raises with a
    descriptive message when no valid forecast source is configured or available.
    """
    from microgrid_simulator.backends.pypsa_backend import PyPSAOperationalBackend
    from microgrid_simulator.controllers.pypsa_rolling_horizon import (
        PyPSARollingHorizonController,
    )
    from microgrid_simulator.forecast.cache import ForecastCache

    if not settings.forecast.enabled:
        raise ValueError("PyPSA-RH policy requires forecast.enabled in the scenario YAML.")

    forecast_mode = getattr(settings.rl, "forecast_mode", "cached")
    if forecast_mode == "none":
        raise ValueError(
            "PyPSA-RH policy cannot run with forecast_mode=none; it needs a forecast source."
        )

    strict_cache = settings.forecast.strict_cache

    pypsa_backend = PyPSAOperationalBackend(settings)
    win = getattr(env, "telemetry_window", None)
    pypsa_backend.reset(
        demand_window_mw=win.demand_mw if win is not None else None,
        pv_window_mw=win.pv_mw if win is not None else None,
    )

    summary: dict[str, Any] = {
        "forecast_mode": "strict_cache" if strict_cache else "live_service",
        "planner": "pypsa",
        "plant": "pandapower",
        "replan_interval_hours": float(settings.backend.rolling_horizon_hours),
        "horizon_hours": settings.backend.horizon_hours,
    }

    if strict_cache:
        if not settings.forecast.cache_path or not settings.forecast.manifest_path:
            raise ValueError(
                "PyPSA-RH policy with strict_cache=true needs forecast.cache_path and "
                "forecast.manifest_path in the scenario YAML."
            )
        source_id = settings.forecast.source_id or settings.scenario.name
        try:
            cache = ForecastCache.load(
                settings.forecast.cache_path,
                settings.forecast.manifest_path,
                expected_source_id=source_id,
                action_interval_hours=settings.topology.timestep_hours,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"PyPSA-RH forecast cache rejected: {exc}") from exc
        ctrl = PyPSARollingHorizonController(backend=pypsa_backend, forecast_cache=cache)
        summary.update(
            {
                "cache_source_id": source_id,
                "cache_source": source_id,
                "cache_path": settings.forecast.cache_path,
                "cache_manifest_path": settings.forecast.manifest_path,
            }
        )
    else:
        if not settings.forecast.service_url:
            raise ValueError(
                "PyPSA-RH policy with strict_cache=false needs forecast.service_url "
                "pointing at a live forecast service."
            )
        ctrl = PyPSARollingHorizonController(backend=pypsa_backend)
        ctrl.set_live_forecast_source(
            lambda: env.current_forecast(),
            expected_source_id=settings.forecast.source_id or settings.scenario.name,
        )
        summary.update(
            {
                "service_url": settings.forecast.service_url,
                "source_id": settings.forecast.source_id or settings.scenario.name,
            }
        )

    return ctrl, summary


def _rule_action(env: MicrogridEnv) -> np.ndarray:
    controller = RuleBasedController(env.settings)
    horizon_provider = getattr(env, "_forecast_vectors", None)
    availability_provider = getattr(env, "_forecast_is_available", None)
    forecast_available = bool(availability_provider()) if availability_provider else False
    pv_horizon, demand_horizon = horizon_provider() if forecast_available else (None, None)
    return env.encode_action(
        controller.act(
            env._last_state,  # noqa: SLF001
            pv_forecast_mw=env._current_pv_forecast_mw(),  # noqa: SLF001
            demand_forecast_mw=env._current_demand_forecast_mw(),  # noqa: SLF001
            pv_forecast_horizon_mw=pv_horizon if forecast_available else None,
            demand_forecast_horizon_mw=demand_horizon if forecast_available else None,
        )
    )


def _load_rl_model(
    artifact: str | Path | None,
    algo: str | None,
    settings: Settings,
) -> Any:
    if not artifact:
        raise ValueError("policy 'rl' requires a trained model artifact")
    from microgrid_simulator.rl.train import ALGOS

    algo = (algo or settings.rl.algo).lower()
    if algo not in ALGOS:
        raise ValueError(f"Unknown RL algo {algo!r}; choose from {sorted(ALGOS)}")
    model = ALGOS[algo].load(str(artifact))

    norm: Any = None
    stats = Path(str(artifact)).with_suffix("").as_posix() + "_vecnormalize.pkl"
    if Path(stats).exists():
        with open(stats, "rb") as fh:
            norm = pickle.load(fh).obs_rms
    return model, norm


def policy_action(
    env: MicrogridEnv,
    policy: str,
    *,
    rl_model: Any = None,
    rl_norm: Any = None,
    rh_ctrl: _RollingHorizonController | None = None,
) -> np.ndarray:
    if policy == "random":
        return env.action_space.sample()
    if policy == "deterministic":
        controller = DeterministicController(env.settings)
        return env.encode_action(controller.act(env._last_state))  # noqa: SLF001
    if policy == "schedule":
        controller = ManualScheduleController(env.settings)
        return env.encode_action(controller.act(env._last_state))  # noqa: SLF001
    if policy == "idle":
        idle = np.zeros(env.action_dim, dtype=np.float32)
        if env.diesel_enabled:
            idle[1 + env.n_ev] = -1.0
        idle[-1] = -1.0
        return idle
    if policy == "rl":
        if rl_model is None:
            raise ValueError("policy 'rl' requires a trained model artifact")
        obs = env._last_observation  # noqa: SLF001
        if rl_norm is not None:
            obs = (obs - rl_norm.mean) / (rl_norm.var + 1e-8) ** 0.5
            obs = obs.clip(-10.0, 10.0).astype("float32")
        action, _ = rl_model.predict(obs, deterministic=True)
        return action
    policy = LEGACY_POLICY_ALIASES.get(policy, policy)
    if policy == "pypsa_rh":
        if rh_ctrl is None:
            raise ValueError(
                "PyPSA-RH controller was not initialized; call "
                "_setup_pypsa_rh_controller before the rollout"
            )
        return rh_ctrl.act(env)
    return _rule_action(env)


def _dispatch_rule(
    settings: Settings,
    policy: str,
    state: GridState,
    control: ControlAction,
) -> str:
    eps = 1e-9
    if control.battery_p_mw > eps:
        battery = "charge battery"
    elif control.battery_p_mw < -eps:
        battery = "discharge battery"
    else:
        battery = "hold battery"
    diesel = "run diesel" if control.diesel_on else "diesel off"

    if policy == "rule":
        if state.islanded:
            if control.battery_p_mw > eps:
                battery = "charge battery from PV surplus"
            elif control.battery_p_mw < -eps:
                battery = "discharge battery for capacity gap"
            diesel = "diesel follows residual with headroom" if control.diesel_on else "diesel off"
            return f"islanded rule: {battery}; {diesel}"

        hour = state.timestamp % 24.0
        if 8.0 <= hour < 15.0:
            battery = "morning charge window"
        elif 18.0 <= hour < 22.0:
            battery = "evening discharge window"
        else:
            battery = "hold battery outside time windows"
        diesel = "diesel peak shaving" if control.diesel_on else "diesel off below peak threshold"
        return f"grid rule: {battery}; {diesel}"

    if policy == "schedule":
        diesel_level = settings.diesel_schedule.level_at(state.timestamp % 24.0)
        battery_mode = "reactive battery" if not settings.battery_schedule.segments else battery
        return f"schedule rule: diesel {diesel_level}; {battery_mode}"
    if policy == "deterministic":
        return f"deterministic merit order: {battery}; {diesel}"
    if policy == "random":
        return f"random sampled action: {battery}; {diesel}"
    if policy == "idle":
        return "idle rule: hold battery; diesel off; no PV curtailment"
    if policy == "rl":
        return f"trained RL policy: {battery}; {diesel}"
    if policy == "pypsa_rh":
        return f"PyPSA-RH rollout: {battery}; {diesel}"
    return f"{policy} policy: {battery}; {diesel}"


def _dispatch_trace(env: MicrogridEnv, policy: str, action: np.ndarray) -> dict[str, Any]:
    state = env._last_state  # noqa: SLF001
    control = decode_action(action, env.settings, env.n_ev, env.diesel_enabled)
    forecast_pv_mw = env._current_pv_forecast_mw()  # noqa: SLF001
    forecast_demand_mw = env._current_demand_forecast_mw()  # noqa: SLF001
    rule_used_forecast = (
        policy == "rule" and forecast_pv_mw is not None and forecast_demand_mw is not None
    )
    return {
        "dispatch_policy": policy,
        "dispatch_rule": _dispatch_rule(env.settings, policy, state, control),
        "requested_battery_kw": control.battery_p_mw * 1000.0,
        "requested_diesel_on": bool(control.diesel_on),
        "requested_diesel_kw": control.diesel_setpoint_mw * 1000.0,
        "requested_pv_curtailment_pct": control.pv_curtail * 100.0,
        "rule_forecast_considered": rule_used_forecast,
        "rule_forecast_pv_kw": forecast_pv_mw * 1000.0 if rule_used_forecast else None,
        "rule_forecast_demand_kw": forecast_demand_mw * 1000.0 if rule_used_forecast else None,
    }


def _slack_bus_id(settings: Settings) -> int:
    for bus in settings.buses:
        if bus.role.lower() in {"grid", "slack", "utility"}:
            return bus.id
    return min(bus.id for bus in settings.buses)


def _per_bus_state(settings: Settings, s: Any, slack_id: int) -> dict[int, dict[str, Any]]:
    load_buses = [load.bus for load in settings.loads[: settings.topology.n_load]]
    pv_buses = [pv.bus for pv in settings.pv_arrays[: settings.topology.n_pv]]
    static_mw = sum(s.p_load)
    ev_mw = max(0.0, s.load_demand_mw - static_mw)
    diesel_enabled = settings.diesel.enabled
    total_demand_mw = s.load_demand_mw
    unserved_frac = (s.unserved_mw / total_demand_mw) if total_demand_mw > 1e-9 else 0.0

    per_bus: dict[int, dict[str, Any]] = {}
    for i, bus in enumerate(settings.buses):
        demand_mw = sum(p for p, b in zip(s.p_load, load_buses, strict=False) if b == bus.id)
        if bus.id == settings.ev.bus:
            demand_mw += ev_mw
        pv_mw = sum(p for p, b in zip(s.p_gen, pv_buses, strict=False) if b == bus.id)
        on_diesel_bus = diesel_enabled and bus.id == settings.diesel.bus
        unserved_mw = demand_mw * unserved_frac
        per_bus[bus.id] = {
            "demand_kw": demand_mw * 1000.0,
            "served_kw": (demand_mw - unserved_mw) * 1000.0,
            "unserved_kw": unserved_mw * 1000.0,
            "pv_kw": pv_mw * 1000.0,
            "battery_kw": s.battery_p_mw * 1000.0 if bus.id == settings.battery.bus else 0.0,
            "diesel_kw": s.diesel_p_mw * 1000.0 if on_diesel_bus else 0.0,
            "diesel_excess_kw": (s.diesel_overgeneration_mw * 1000.0 if on_diesel_bus else 0.0),
            "diesel_on": bool(s.diesel_on) if on_diesel_bus else False,
            "grid_kw": s.grid_import_mw * 1000.0 if bus.id == slack_id else 0.0,
            "v_pu": round(s.v_bus[i], 5) if i < len(s.v_bus) else None,
        }
    return per_bus


def _step_row(
    settings: Settings,
    env: MicrogridEnv,
    step: int,
    reward: float,
    info: dict[str, Any],
    slack_id: int,
    dispatch: dict[str, Any],
) -> dict[str, Any]:
    s = env._last_state  # noqa: SLF001
    dt = settings.topology.timestep_hours
    grid_import_kw = max(0.0, s.grid_import_mw) * 1000.0
    grid_export_kw = max(0.0, -s.grid_import_mw) * 1000.0
    battery_charge_kw = max(0.0, s.battery_p_mw) * 1000.0
    battery_discharge_kw = max(0.0, -s.battery_p_mw) * 1000.0
    local_supply_kw = (s.pv_used_mw + s.diesel_p_mw) * 1000.0 + battery_discharge_kw
    excess_generation_kw = max(0.0, s.excess_generation_mw * 1000.0)
    dump_load_kw = max(0.0, s.dump_load_mw * 1000.0)
    network_loss_kw = max(0.0, s.network_loss_mw * 1000.0)
    reference_balance_kw = s.reference_balance_mw * 1000.0
    balance_residual_kw = (
        local_supply_kw
        + grid_import_kw
        + max(0.0, reference_balance_kw)
        - s.load_served_mw * 1000.0
        - battery_charge_kw
        - grid_export_kw
        - dump_load_kw
        - network_loss_kw
        - max(0.0, -reference_balance_kw)
    )
    gross_generation_kw = (
        s.pv_used_mw + s.diesel_p_mw + max(0.0, -s.battery_p_mw) + max(0.0, s.grid_import_mw)
    ) * 1000.0
    carbon_kg = (
        grid_import_kw * dt * settings.reward.grid_carbon_kg_per_kwh
        + max(0.0, s.diesel_p_mw) * 1000.0 * dt * settings.reward.diesel_carbon_kg_per_kwh
    )
    return {
        "step": step + 1,
        "hour": round(s.timestamp, 4),
        **dispatch,
        "reward": reward,
        "grid_import_kw": s.grid_import_mw * 1000.0,
        "grid_import_positive_kw": grid_import_kw,
        "grid_export_kw": grid_export_kw,
        "load_kw": s.load_demand_mw * 1000.0,
        "served_kw": s.load_served_mw * 1000.0,
        "load_serving_supply_kw": s.load_served_mw * 1000.0,
        "unserved_kw": s.unserved_mw * 1000.0,
        "blackout": bool(s.blackout),
        "islanded": bool(s.islanded),
        "demand_is_real": bool(s.demand_is_real),
        "pv_is_real": bool(s.pv_is_real),
        "pv_available_kw": s.pv_available_mw * 1000.0,
        "pv_used_kw": s.pv_used_mw * 1000.0,
        "pv_forecast_kw": (
            float(info["pv_forecast_mw"]) * 1000.0
            if info.get("pv_forecast_mw") is not None
            else None
        ),
        "pv_forecast_horizon_kw": [
            float(value) * 1000.0 for value in info.get("forecast_horizon_mw", [])
        ],
        "demand_forecast_kw": (
            float(info["demand_forecast_mw"]) * 1000.0
            if info.get("demand_forecast_mw") is not None
            else None
        ),
        "demand_forecast_horizon_kw": [
            float(value) * 1000.0 for value in info.get("demand_forecast_horizon_mw", [])
        ],
        "forecast_issued_at": info.get("forecast_issued_at"),
        "forecast_model_version": info.get("forecast_model_version"),
        "forecast_available": bool(info.get("forecast_available", False)),
        "forecast_source": info.get("forecast_source"),
        "forecast_requested_source": info.get("forecast_requested_source"),
        "forecast_context_time": info.get("forecast_context_time"),
        "forecast_context_steps": info.get("forecast_context_steps"),
        "forecast_stale": bool(info.get("forecast_stale", False)),
        "forecast_age_steps": info.get("forecast_age_steps"),
        "forecast_cold_start": bool(info.get("forecast_cold_start", False)),
        "forecast_covariate_mode": info.get("forecast_covariate_mode"),
        "forecast_error": info.get("forecast_error"),
        "pv_wasted_kw": max(0.0, s.pv_available_mw - s.pv_used_mw) * 1000.0,
        "battery_kw": s.battery_p_mw * 1000.0,
        "battery_charge_kw": battery_charge_kw,
        "battery_discharge_kw": battery_discharge_kw,
        "battery_charge_from_pv_kw": s.battery_charge_from_pv_mw * 1000.0,
        "battery_charge_from_grid_kw": s.battery_charge_from_grid_mw * 1000.0,
        "battery_charge_from_diesel_kw": s.battery_charge_from_diesel_mw * 1000.0,
        "diesel_kw": s.diesel_p_mw * 1000.0,
        "diesel_load_serving_kw": s.diesel_load_serving_mw * 1000.0,
        "diesel_overgeneration_kw": s.diesel_overgeneration_mw * 1000.0,
        "diesel_on": bool(s.diesel_on),
        "gross_generation_kw": gross_generation_kw,
        "excess_generation_kw": excess_generation_kw,
        "dump_load_kw": dump_load_kw,
        "network_loss_kw": network_loss_kw,
        "reference_balance_kw": reference_balance_kw,
        "power_balance_residual_kw": balance_residual_kw,
        "carbon_kg": carbon_kg,
        "solver_ok": bool(s.solver_ok),
        "soc_pct": (s.soc[0] * 100.0) if s.soc else None,
        "soh_pct": (s.soh[0] * 100.0) if s.soh else None,
        "min_voltage_pu": min(s.v_bus) if s.v_bus else None,
        "max_voltage_pu": max(s.v_bus) if s.v_bus else None,
        "v_bus": [round(v, 5) for v in s.v_bus],
        "p_load_kw": [p * 1000.0 for p in s.p_load],
        "p_pv_kw": [p * 1000.0 for p in s.p_gen],
        "ev_soc_pct": [x * 100.0 for x in s.ev_soc],
        "max_line_loading_pct": max(s.line_loading) if s.line_loading else None,
        "constraint_violation_count": len(s.violations),
        "constraint_violation_types": sorted({v.kind for v in s.violations}),
        "per_bus": _per_bus_state(settings, s, slack_id),
        **{
            k.replace("reward/", "penalty_"): v
            for k, v in info.items()
            if k.startswith("reward/") and k != "reward/total"
        },
    }


_BACKEND_CLASS_NAMES = {
    "SimpleBackend": "simple",
    "PandapowerBackend": "pandapower",
    "PyPSAOperationalBackend": "pypsa",
    "OpenDSSBackend": "opendss",
}


def _resolved_backend_name(env: MicrogridEnv, settings: Settings) -> str:
    return _BACKEND_CLASS_NAMES.get(
        type(env.backend).__name__, resolve_backend_name(settings.backend.name, settings)
    )


def _episode_meta(
    settings: Settings,
    env: MicrogridEnv,
    policy: str,
    seed: int,
    reset_info: dict[str, Any],
    slack_id: int,
) -> dict[str, Any]:
    demand_real = bool(env.backend.demand_is_real)
    pv_real = bool(env._last_state.pv_is_real)  # noqa: SLF001
    meta: dict[str, Any] = {
        "policy": policy,
        "seed": seed,
        "backend": _resolved_backend_name(env, settings),
        "timestep_hours": settings.topology.timestep_hours,
        "demand_source": "historical trace" if demand_real else "synthetic sinusoid",
        "demand_is_real": demand_real,
        "pv_source": "historical trace" if pv_real else "synthetic daylight curve",
        "pv_is_real": pv_real,
        "demand_window_start_time": reset_info.get("demand_window_start_time"),
        "telemetry_context_time": reset_info.get("telemetry_context_time"),
        "telemetry_end_time": reset_info.get("telemetry_end_time"),
        "telemetry_source_files": reset_info.get("telemetry_source_files"),
        "forecast_enabled": bool(reset_info.get("forecast_enabled", False)),
        "forecast_available": bool(reset_info.get("forecast_available", False)),
        "forecast_error": reset_info.get("forecast_error"),
        "forecast_service_url": reset_info.get("forecast_service_url"),
        "forecast_requested_horizon_hours": reset_info.get("forecast_requested_horizon_hours"),
        "forecast_refresh_each_step": bool(reset_info.get("forecast_refresh_each_step", False)),
        "forecast_horizon_hours": reset_info.get("forecast_horizon_hours"),
        "forecast_frequency_hours": reset_info.get("forecast_frequency_hours"),
        "forecast_issued_at": reset_info.get("forecast_issued_at"),
        "forecast_model_version": reset_info.get("forecast_model_version"),
        "forecast_source": reset_info.get("forecast_source"),
        "forecast_requested_source": reset_info.get("forecast_requested_source"),
        "forecast_context_time": reset_info.get("forecast_context_time"),
        "forecast_context_steps": reset_info.get("forecast_context_steps"),
        "forecast_stale": bool(reset_info.get("forecast_stale", False)),
        "forecast_age_steps": reset_info.get("forecast_age_steps"),
        "forecast_cold_start": bool(reset_info.get("forecast_cold_start", False)),
        "forecast_covariate_mode": reset_info.get("forecast_covariate_mode"),
        "forecast_target": reset_info.get("forecast_target"),
        "forecast_demand_target": reset_info.get("forecast_demand_target"),
        "forecast_timestamps": reset_info.get("forecast_timestamps", []),
        "forecast_values_kw": reset_info.get("forecast_values_kw", []),
        "forecast_demand_values_kw": reset_info.get("forecast_demand_values_kw", []),
        "grid_connected": env.backend.grid_connected,
        "islanded": not env.backend.grid_connected,
        "diesel_enabled": env.diesel_enabled,
        "topology": {
            "buses": [bus.model_dump() for bus in settings.buses],
            "lines": [line.model_dump() for line in settings.lines],
            "slack_bus": slack_id,
            "battery_bus": settings.battery.bus,
            "ev_bus": settings.ev.bus,
            "diesel_bus": settings.diesel.bus,
            "pv_buses": [pv.bus for pv in settings.pv_arrays[: settings.topology.n_pv]],
            "load_buses": [load.bus for load in settings.loads[: settings.topology.n_load]],
        },
    }
    if policy == "pypsa_rh":
        meta["planner"] = "pypsa"
        meta["plant"] = "pandapower"
    return meta


def run_rollout(
    settings: Settings,
    policy: str = "rule",
    seed: int = 0,
    rl_artifact: str | Path | None = None,
    rl_algo: str | None = None,
) -> dict[str, Any]:
    policy = LEGACY_POLICY_ALIASES.get(policy, policy)
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    rl_model, rl_norm = (
        _load_rl_model(rl_artifact, rl_algo, settings) if policy == "rl" else (None, None)
    )
    env = MicrogridEnv(settings=settings)
    _, reset_info = env.reset(seed=seed)
    dt = settings.topology.timestep_hours
    slack_id = _slack_bus_id(settings)

    rh_ctrl: _RollingHorizonController | None = None
    rh_summary: dict[str, Any] | None = None
    if policy == "pypsa_rh":
        ctrl, rh_summary = _setup_pypsa_rh_controller(settings, env)
        rh_ctrl = _RollingHorizonController(ctrl, rh_summary)

    rows: list[dict[str, Any]] = []
    for step in range(env.max_steps):
        action = policy_action(env, policy, rl_model=rl_model, rl_norm=rl_norm, rh_ctrl=rh_ctrl)
        dispatch = _dispatch_trace(env, policy, action)
        _, reward, terminated, truncated, info = env.step(action)
        rows.append(_step_row(settings, env, step, reward, info, slack_id, dispatch))
        if terminated or truncated:
            break

    totals = _totals(rows, dt)
    meta: dict[str, Any] = {
        **_episode_meta(settings, env, policy, seed, reset_info, slack_id),
        **{
            key: env._forecast_meta().get(key)  # noqa: SLF001
            for key in (
                "forecast_available",
                "forecast_error",
                "forecast_source",
                "forecast_requested_source",
                "forecast_context_time",
                "forecast_context_steps",
                "forecast_stale",
                "forecast_age_steps",
                "forecast_cold_start",
                "forecast_covariate_mode",
            )
        },
        "steps": len(rows),
        "diesel_starts": env.backend.diesel.starts,
        "diesel_runtime_hours": env.backend.diesel.runtime_hours,
    }
    if rl_artifact is not None:
        meta["rl_artifact"] = str(rl_artifact)
        meta["rl_algo"] = rl_algo or settings.rl.algo
    if rh_summary is not None:
        meta["pypsa_rh"] = rh_summary
    env.close()
    return {"rows": rows, "totals": totals, "meta": meta}


def stream_rollout(
    settings: Settings,
    policy: str = "rule",
    seed: int = 0,
    rl_artifact: str | Path | None = None,
    rl_algo: str | None = None,
) -> Iterator[dict[str, Any]]:
    policy = LEGACY_POLICY_ALIASES.get(policy, policy)
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    rl_model, rl_norm = (
        _load_rl_model(rl_artifact, rl_algo, settings) if policy == "rl" else (None, None)
    )
    env = MicrogridEnv(settings=settings)
    try:
        _, reset_info = env.reset(seed=seed)
        dt = settings.topology.timestep_hours
        slack_id = _slack_bus_id(settings)
        meta = _episode_meta(settings, env, policy, seed, reset_info, slack_id)
        if rl_artifact is not None:
            meta["rl_artifact"] = str(rl_artifact)
            meta["rl_algo"] = rl_algo or settings.rl.algo

        rh_ctrl: _RollingHorizonController | None = None
        rh_summary: dict[str, Any] | None = None
        if policy == "pypsa_rh":
            ctrl, rh_summary = _setup_pypsa_rh_controller(settings, env)
            rh_ctrl = _RollingHorizonController(ctrl, rh_summary)
            meta["pypsa_rh"] = rh_summary

        yield {
            "type": "meta",
            "meta": meta,
        }

        rows: list[dict[str, Any]] = []
        for step in range(env.max_steps):
            action = policy_action(env, policy, rl_model=rl_model, rl_norm=rl_norm, rh_ctrl=rh_ctrl)
            dispatch = _dispatch_trace(env, policy, action)
            _, reward, terminated, truncated, info = env.step(action)
            row = _step_row(settings, env, step, reward, info, slack_id, dispatch)
            rows.append(row)
            yield {"type": "row", "row": row}
            if terminated or truncated:
                break

        end_meta: dict[str, Any] = {
            **{
                key: env._forecast_meta().get(key)  # noqa: SLF001
                for key in (
                    "forecast_available",
                    "forecast_error",
                    "forecast_source",
                    "forecast_requested_source",
                    "forecast_context_time",
                    "forecast_context_steps",
                    "forecast_stale",
                    "forecast_age_steps",
                    "forecast_cold_start",
                    "forecast_covariate_mode",
                )
            },
            "steps": len(rows),
            "diesel_starts": env.backend.diesel.starts,
            "diesel_runtime_hours": env.backend.diesel.runtime_hours,
        }
        if rl_artifact is not None:
            end_meta["rl_artifact"] = str(rl_artifact)
            end_meta["rl_algo"] = rl_algo or settings.rl.algo
        if rh_summary is not None:
            end_meta["pypsa_rh"] = rh_summary
        yield {"type": "end", "totals": _totals(rows, dt), "meta": end_meta}
    finally:
        env.close()


def _totals(rows: list[dict[str, Any]], dt: float) -> dict[str, float]:
    if not rows:
        return {}

    def _sum(key: str) -> float:
        return float(sum(r.get(key) or 0.0 for r in rows))

    soc_values = [
        float(value)
        for row in rows
        if (value := row.get("soc_pct")) is not None and np.isfinite(float(value))
    ]
    soh_values = [
        float(value)
        for row in rows
        if (value := row.get("soh_pct")) is not None and np.isfinite(float(value))
    ]
    grid_kwh = _sum("grid_import_positive_kw") * dt
    grid_export_kwh = _sum("grid_export_kw") * dt
    diesel_kwh = _sum("diesel_kw") * dt
    diesel_load_serving_kwh = _sum("diesel_load_serving_kw") * dt
    diesel_overgeneration_kwh = _sum("diesel_overgeneration_kw") * dt
    pv_available_kwh = _sum("pv_available_kw") * dt
    battery_charge_kwh = _sum("battery_charge_kw") * dt
    battery_charge_from_pv_kwh = _sum("battery_charge_from_pv_kw") * dt
    battery_charge_from_grid_kwh = _sum("battery_charge_from_grid_kw") * dt
    battery_charge_from_diesel_kwh = _sum("battery_charge_from_diesel_kw") * dt
    battery_discharge_kwh = _sum("battery_discharge_kw") * dt
    blackout_steps = sum(1 for r in rows if r.get("blackout"))
    load_kwh = _sum("load_kw") * dt
    served_kwh = _sum("served_kw") * dt
    pv_used_kwh = _sum("pv_used_kw") * dt
    return {
        "total_reward": _sum("reward"),
        "grid_import_kwh": grid_kwh,
        "grid_export_kwh": grid_export_kwh,
        "diesel_kwh": diesel_kwh,
        "diesel_load_serving_kwh": diesel_load_serving_kwh,
        "diesel_overgeneration_kwh": diesel_overgeneration_kwh,
        "diesel_useful_pct": (
            100.0 * (diesel_kwh - diesel_overgeneration_kwh) / diesel_kwh if diesel_kwh else 100.0
        ),
        "pv_available_kwh": pv_available_kwh,
        "pv_used_kwh": pv_used_kwh,
        "pv_wasted_kwh": _sum("pv_wasted_kw") * dt,
        "pv_utilization_pct": 100.0 * pv_used_kwh / pv_available_kwh if pv_available_kwh else 0.0,
        "battery_charge_kwh": battery_charge_kwh,
        "battery_charge_from_pv_kwh": battery_charge_from_pv_kwh,
        "battery_charge_from_grid_kwh": battery_charge_from_grid_kwh,
        "battery_charge_from_diesel_kwh": battery_charge_from_diesel_kwh,
        "battery_discharge_kwh": battery_discharge_kwh,
        "battery_throughput_kwh": battery_charge_kwh + battery_discharge_kwh,
        "load_kwh": load_kwh,
        "served_kwh": served_kwh,
        "unserved_kwh": _sum("unserved_kw") * dt,
        "served_energy_pct": 100.0 * served_kwh / load_kwh if load_kwh else 100.0,
        "blackout_steps": blackout_steps,
        "blackout_hours": blackout_steps * dt,
        "peak_unserved_kw": float(max((r["unserved_kw"] for r in rows), default=0.0)),
        "peak_import_kw": float(
            max(r.get("grid_import_positive_kw", r["grid_import_kw"]) for r in rows)
        ),
        "peak_load_kw": float(max(r["load_kw"] for r in rows)),
        "peak_pv_kw": float(max((r.get("pv_available_kw", 0.0) for r in rows), default=0.0)),
        "excess_generation_kwh": _sum("excess_generation_kw") * dt,
        "dump_load_kwh": _sum("dump_load_kw") * dt,
        "network_loss_kwh": _sum("network_loss_kw") * dt,
        "reference_balance_import_kwh": sum(
            max(0.0, float(r.get("reference_balance_kw", 0.0))) for r in rows
        )
        * dt,
        "reference_balance_absorption_kwh": sum(
            max(0.0, -float(r.get("reference_balance_kw", 0.0))) for r in rows
        )
        * dt,
        "peak_excess_generation_kw": float(
            max((r.get("excess_generation_kw", 0.0) for r in rows), default=0.0)
        ),
        "peak_diesel_overgeneration_kw": float(
            max((r.get("diesel_overgeneration_kw", 0.0) for r in rows), default=0.0)
        ),
        "max_abs_power_balance_residual_kw": float(
            max((abs(r.get("power_balance_residual_kw", 0.0)) for r in rows), default=0.0)
        ),
        "constraint_violation_events": int(_sum("constraint_violation_count")),
        "min_soc_pct": min(soc_values, default=0.0),
        "max_soc_pct": max(soc_values, default=0.0),
        "final_soc_pct": soc_values[-1] if soc_values else 0.0,
        "soh_loss_pct_points": 100.0 - (soh_values[-1] if soh_values else 100.0),
        "carbon_kg": _sum("carbon_kg"),
    }
