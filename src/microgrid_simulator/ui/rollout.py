"""Shared rollout runner for the dashboard API.

Runs a non-learned controller (rule / idle / random / deterministic) through one episode and
returns per-timestep rows the frontend can chart directly, plus episode meta
(demand source, window start, totals). ``stream_rollout`` yields the same rows
one tick at a time so the API can stream them to the UI as they are solved.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from microgrid_simulator.backends import resolve_backend_name
from microgrid_simulator.config import Settings
from microgrid_simulator.controllers import DeterministicController
from microgrid_simulator.env import MicrogridEnv

POLICIES = ("rule", "idle", "random", "deterministic")


def _rule_action(env: MicrogridEnv) -> np.ndarray:
    """Interpretable heuristic.

    Grid-connected: solar-charge / evening-discharge the battery and let diesel
    peak-shave whatever exceeds the soft peak threshold.

    Islanded (no utility tie): there is no grid to backfill, so the controller
    instead tries to *keep the lights on* — discharge the battery and dispatch
    diesel to cover the full residual demand. When PV + battery + diesel still
    can't meet it, the shortfall shows up as an islanded blackout.
    """
    action = np.zeros(env.action_dim, dtype=np.float32)
    s = env._last_state  # noqa: SLF001 - diagnostics-grade access is fine here
    hour = env.backend.timestamp % 24.0
    islanded = not env.backend.grid_connected

    residual_mw = max(0.0, s.load_demand_mw - s.pv_used_mw)

    if islanded:
        # Diesel is the firm dispatchable source, so let it carry the residual
        # first; the battery only tops up what diesel can't and charges on PV
        # surplus. (Leaning on the battery first would black out once its SoC
        # floor is hit and diesel was never asked to compensate.)
        surplus_mw = max(0.0, s.pv_used_mw - s.load_demand_mw)
        diesel_max_mw = env.settings.diesel.max_kw / 1000.0 if env.diesel_enabled else 0.0
        diesel_target_mw = min(residual_mw, diesel_max_mw)
        battery_gap_mw = max(0.0, residual_mw - diesel_target_mw)
        if surplus_mw > 0.0:
            action[0] = min(1.0, surplus_mw / max(env.settings.battery.max_charge_mw, 1e-9))
        elif battery_gap_mw > 0.0:
            action[0] = -min(1.0, battery_gap_mw / max(env.settings.battery.max_discharge_mw, 1e-9))
        # diesel should cover the whole residual it's capable of, regardless of
        # battery state, so the lights stay on whenever capacity exists. A small
        # headroom margin absorbs the one-tick control lag at demand ramps.
        diesel_target_mw = residual_mw * 1.12
    else:
        action[0] = 0.6 if 8 <= hour < 15 else (-0.6 if 18 <= hour < 22 else 0.0)
        diesel_target_mw = max(0.0, residual_mw - env.settings.reward.peak_threshold_mw)

    if env.diesel_enabled:
        # Grid-connected, don't start the genset for less than its minimum
        # stable load — the grid covers small peaks. Islanded, any residual
        # justifies a start (surplus below min load is dumped, lights stay on).
        floor_mw = 0.0 if islanded else env.settings.diesel.min_kw / 1000.0
        if diesel_target_mw > max(floor_mw, 0.0):
            diesel_max_mw = env.settings.diesel.max_kw / 1000.0
            frac = min(1.0, diesel_target_mw / max(diesel_max_mw, 1e-9))
            action[-3] = 1.0  # on
            action[-2] = frac * 2.0 - 1.0  # setpoint -> [-1, 1]
        else:
            action[-3] = -1.0  # off
            action[-2] = -1.0

    action[-1] = -1.0  # no curtailment
    return action


def policy_action(env: MicrogridEnv, policy: str) -> np.ndarray:
    if policy == "random":
        return env.action_space.sample()
    if policy == "deterministic":
        controller = DeterministicController(env.settings)
        return env.encode_action(controller.act(env._last_state))  # noqa: SLF001
    if policy == "idle":
        idle = np.zeros(env.action_dim, dtype=np.float32)
        if env.diesel_enabled:
            idle[-3] = -1.0
            idle[-2] = -1.0
        idle[-1] = -1.0
        return idle
    return _rule_action(env)


def _slack_bus_id(settings: Settings) -> int:
    """Mirror PandapowerBackend._find_slack_bus_id for per-bus reporting."""
    for bus in settings.buses:
        if bus.role.lower() in {"grid", "slack", "utility"}:
            return bus.id
    return min(bus.id for bus in settings.buses)


def _per_bus_state(settings: Settings, s: Any, slack_id: int) -> dict[int, dict[str, Any]]:
    """Split the grid snapshot into one state record per topology bus.

    v_bus follows settings.buses order (bus creation order), p_load follows
    settings.loads[:n_load], p_gen follows pv_arrays[:n_pv]. Unserved demand in
    an islanded blackout is attributed to each bus in proportion to its share of
    total demand.
    """
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
) -> dict[str, Any]:
    s = env._last_state  # noqa: SLF001
    dt = settings.topology.timestep_hours
    grid_import_kw = max(0.0, s.grid_import_mw) * 1000.0
    grid_export_kw = max(0.0, -s.grid_import_mw) * 1000.0
    battery_charge_kw = max(0.0, s.battery_p_mw) * 1000.0
    battery_discharge_kw = max(0.0, -s.battery_p_mw) * 1000.0
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
        "reward": reward,
        "grid_import_kw": s.grid_import_mw * 1000.0,
        "grid_import_positive_kw": grid_import_kw,
        "grid_export_kw": grid_export_kw,
        "load_kw": s.load_demand_mw * 1000.0,
        "served_kw": s.load_served_mw * 1000.0,
        "load_serving_supply_kw": s.load_served_mw * 1000.0,
        "unserved_kw": s.unserved_mw * 1000.0,
        "blackout": bool(s.unserved_mw > 1e-6),
        "islanded": bool(s.islanded),
        "pv_available_kw": s.pv_available_mw * 1000.0,
        "pv_used_kw": s.pv_used_mw * 1000.0,
        "pv_wasted_kw": max(0.0, s.pv_available_mw - s.pv_used_mw) * 1000.0,
        "battery_kw": s.battery_p_mw * 1000.0,
        "battery_charge_kw": battery_charge_kw,
        "battery_discharge_kw": battery_discharge_kw,
        "diesel_kw": s.diesel_p_mw * 1000.0,
        "diesel_on": bool(s.diesel_on),
        "gross_generation_kw": gross_generation_kw,
        "carbon_kg": carbon_kg,
        "soc_pct": (s.soc[0] * 100.0) if s.soc else None,
        "soh_pct": (s.soh[0] * 100.0) if s.soh else None,
        "min_voltage_pu": min(s.v_bus) if s.v_bus else None,
        "max_voltage_pu": max(s.v_bus) if s.v_bus else None,
        "v_bus": [round(v, 5) for v in s.v_bus],
        "p_load_kw": [p * 1000.0 for p in s.p_load],
        "p_pv_kw": [p * 1000.0 for p in s.p_gen],
        "ev_soc_pct": [x * 100.0 for x in s.ev_soc],
        "max_line_loading_pct": max(s.line_loading) if s.line_loading else None,
        "per_bus": _per_bus_state(settings, s, slack_id),
        **{
            k.replace("reward/", "penalty_"): v
            for k, v in info.items()
            if k.startswith("reward/") and k != "reward/total"
        },
    }


def _episode_meta(
    settings: Settings,
    env: MicrogridEnv,
    policy: str,
    seed: int,
    reset_info: dict[str, Any],
    slack_id: int,
) -> dict[str, Any]:
    demand_real = bool(env.backend.demand_is_real)
    return {
        "policy": policy,
        "seed": seed,
        "backend": resolve_backend_name(settings.backend.name, settings),
        "timestep_hours": settings.topology.timestep_hours,
        "demand_source": "historical trace" if demand_real else "synthetic sinusoid",
        "demand_is_real": demand_real,
        "demand_window_start_time": reset_info.get("demand_window_start_time"),
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


def run_rollout(settings: Settings, policy: str = "rule", seed: int = 0) -> dict[str, Any]:
    env = MicrogridEnv(settings=settings)
    _, reset_info = env.reset(seed=seed)
    dt = settings.topology.timestep_hours
    slack_id = _slack_bus_id(settings)

    rows: list[dict[str, Any]] = []
    for step in range(env.max_steps):
        action = policy_action(env, policy)
        _, reward, terminated, truncated, info = env.step(action)
        rows.append(_step_row(settings, env, step, reward, info, slack_id))
        if terminated or truncated:
            break

    totals = _totals(rows, dt)
    meta = {
        **_episode_meta(settings, env, policy, seed, reset_info, slack_id),
        "steps": len(rows),
        "diesel_starts": env.backend.diesel.starts,
        "diesel_runtime_hours": env.backend.diesel.runtime_hours,
    }
    env.close()
    return {"rows": rows, "totals": totals, "meta": meta}


def stream_rollout(
    settings: Settings, policy: str = "rule", seed: int = 0
) -> Iterator[dict[str, Any]]:
    """Yield one episode as a sequence of events for the streaming API.

    Event order: one ``meta`` (topology + demand window, sent before any tick so
    the UI can set up), ``row`` per solved tick, then one ``end`` carrying the
    episode totals and the counters only known at the end. The env is closed
    even when the consumer stops iterating early (client disconnect).
    """
    env = MicrogridEnv(settings=settings)
    try:
        _, reset_info = env.reset(seed=seed)
        dt = settings.topology.timestep_hours
        slack_id = _slack_bus_id(settings)
        yield {
            "type": "meta",
            "meta": _episode_meta(settings, env, policy, seed, reset_info, slack_id),
        }

        rows: list[dict[str, Any]] = []
        for step in range(env.max_steps):
            action = policy_action(env, policy)
            _, reward, terminated, truncated, info = env.step(action)
            row = _step_row(settings, env, step, reward, info, slack_id)
            rows.append(row)
            yield {"type": "row", "row": row}
            if terminated or truncated:
                break

        yield {
            "type": "end",
            "totals": _totals(rows, dt),
            "meta": {
                "steps": len(rows),
                "diesel_starts": env.backend.diesel.starts,
                "diesel_runtime_hours": env.backend.diesel.runtime_hours,
            },
        }
    finally:
        env.close()


def _totals(rows: list[dict[str, Any]], dt: float) -> dict[str, float]:
    if not rows:
        return {}

    def _sum(key: str) -> float:
        return float(sum(r.get(key) or 0.0 for r in rows))

    grid_kwh = _sum("grid_import_positive_kw") * dt
    grid_export_kwh = _sum("grid_export_kw") * dt
    diesel_kwh = _sum("diesel_kw") * dt
    blackout_steps = sum(1 for r in rows if r.get("blackout"))
    return {
        "total_reward": _sum("reward"),
        "grid_import_kwh": grid_kwh,
        "grid_export_kwh": grid_export_kwh,
        "diesel_kwh": diesel_kwh,
        "pv_used_kwh": _sum("pv_used_kw") * dt,
        "pv_wasted_kwh": _sum("pv_wasted_kw") * dt,
        "load_kwh": _sum("load_kw") * dt,
        "served_kwh": _sum("served_kw") * dt,
        "unserved_kwh": _sum("unserved_kw") * dt,
        "blackout_steps": blackout_steps,
        "blackout_hours": blackout_steps * dt,
        "peak_unserved_kw": float(max((r["unserved_kw"] for r in rows), default=0.0)),
        "peak_import_kw": float(max(r.get("grid_import_positive_kw", r["grid_import_kw"]) for r in rows)),
        "peak_load_kw": float(max(r["load_kw"] for r in rows)),
        "final_soc_pct": float(rows[-1]["soc_pct"] or 0.0),
        "carbon_kg": _sum("carbon_kg"),
    }
