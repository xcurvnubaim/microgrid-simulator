"""Continuous one-month frozen-policy evaluation (March 2026 stress test).

Implements [[Continuous One-Month Frozen-Policy Evaluation Plan - Module 6]]:

* one uninterrupted 744-hour islanded replay per job (2,976 x 15-minute
  decisions) with a single physical reset before the first March decision;
* frozen Rule-F3 / Schedule / PyPSA-RH-F3 baselines and frozen SAC-F3 /
  SAC-none-F3 checkpoints (seeds 0-2) across E0-E5 (54 jobs);
* single masked-progress SAC mode: after applying the checkpoint's saved
  VecNormalize statistics, the normalized ``episode_progress`` coordinate is
  set to ``0.0`` (its training mean). Native-month and 72-hour-cycled progress
  are rejected modes and are never evaluated;
* per-job outputs (``trajectory.csv``, ``metrics.json``, ``daily.csv``,
  ``outage_events.csv``), safe resume, and combined summaries.

No training, tuning, or checkpoint selection from March happens here.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.env import MicrogridEnv


def get_episode_progress_index(settings: Settings) -> int:
    """Return the index of ``episode_progress`` in the flat observation.

    Mirrors ``build_observation`` ordering: buses, loads, PV, SOC, SOH, EVs,
    three plant scalars, optional diesel output, two time features, then SOC,
    initial-SOC, and target-SOC context coordinates. Kept here so the monthly
    runner stays self-contained about the masking contract.
    """
    n_buses = len(settings.buses)
    n_load = settings.topology.n_load
    n_pv = settings.topology.n_pv
    n_storage = settings.topology.n_storage
    n_ev = settings.topology.n_ev
    diesel_idx = 1 if settings.diesel.enabled else 0
    return n_buses + n_load + n_pv + n_storage + n_storage + n_ev + 3 + diesel_idx + 2 + 1 + 1 + 1


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "monthly_march_2026"

EVALUATION_START = "2026-03-01 00:00:00"
EVALUATION_END_EXCLUSIVE = "2026-04-01 00:00:00"
EXPECTED_STEPS = 2976
BATTERY_CAPACITY_KWH = 500.0
EQUIVALENT_CYCLE_DENOMINATOR_KWH = 2.0 * BATTERY_CAPACITY_KWH

SCENARIOS: dict[str, Path] = {f"E{i}": CONFIG_DIR / f"f3-e{i}-monthly.yaml" for i in range(6)}

# Deterministic baselines run once each (seed 0); SAC runs three frozen seeds.
DETERMINISTIC_POLICIES: dict[str, str] = {
    "rule_f3": "rule",
    "schedule": "schedule",
    "pypsa_rh_f3": "pypsa_rh",
}
LEGACY_POLICY_ALIASES = {"mpc_f3": "pypsa_rh_f3"}
SAC_POLICIES: dict[str, str] = {
    "sac_f3": "f3",
    "sac_none_f3": "none",
}
# Post-plan extension (2026-08-21): fixed-width causal F3 summary representation,
# trained as the E0-E5 pilot under artifacts/sac/f3-summary. Not part of the
# frozen 54-job matrix; included only when explicitly requested.
SAC_SUMMARY_POLICIES: dict[str, str] = {
    "sac_f3_summary": "summary",
}
SAC_SEEDS = (0, 1, 2)
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "sac" / "f3-15min"
SUMMARY_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "sac" / "f3-summary"

TRAJECTORY_COLUMNS = (
    "timestamp",
    "step",
    "dispatch_policy",
    "dispatch_rule",
    "requested_battery_kw",
    "requested_diesel_on",
    "requested_diesel_kw",
    "requested_pv_curtailment_pct",
    "load_kw",
    "served_kw",
    "unserved_kw",
    "pv_available_kw",
    "pv_used_kw",
    "pv_wasted_kw",
    "battery_charge_kw",
    "battery_charge_from_pv_kw",
    "battery_charge_from_grid_kw",
    "battery_charge_from_diesel_kw",
    "battery_discharge_kw",
    "diesel_kw",
    "diesel_load_serving_kw",
    "diesel_overgeneration_kw",
    "diesel_on",
    "soc_pct",
    "soh_pct",
    "excess_generation_kw",
    "dump_load_kw",
    "network_loss_kw",
    "reference_balance_kw",
    "power_balance_residual_kw",
    "carbon_kg",
    "reward",
    "constraint_violation_count",
)

SUMMARY_METRICS = (
    "total_reward",
    "load_kwh",
    "served_kwh",
    "unserved_kwh",
    "served_energy_pct",
    "blackout_hours",
    "outage_count",
    "max_outage_hours",
    "worst_daily_unserved_kwh",
    "affected_days",
    "initial_soc_pct",
    "final_soc_pct",
    "soc_change_pct_points",
    "battery_throughput_kwh",
    "equivalent_full_cycles",
    "soh_loss_pct_points",
    "diesel_kwh",
    "diesel_starts",
    "diesel_runtime_hours",
    "diesel_overgeneration_kwh",
    "pv_available_kwh",
    "pv_used_kwh",
    "pv_wasted_kwh",
    "excess_generation_kwh",
    "dump_load_kwh",
    "network_loss_kwh",
    "carbon_kg",
    "peak_unserved_kw",
)


@dataclass(frozen=True)
class MonthlyJob:
    """One frozen monthly evaluation job."""

    scenario: str
    policy: str
    seed: int
    config_path: Path
    artifact_path: Path | None
    vecnorm_path: Path | None
    output_dir: Path

    @property
    def job_id(self) -> str:
        return f"{self.scenario}_{self.policy}_seed{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["job_id"] = self.job_id
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in data.items()
        }


def artifact_paths_for(scenario: str, policy: str, seed: int) -> tuple[Path, Path]:
    """Resolve the frozen checkpoint and VecNormalize stats for a SAC job."""
    if policy in SAC_SUMMARY_POLICIES:
        base = SUMMARY_ARTIFACT_ROOT / scenario / f"seed-{seed}"
    else:
        mode = SAC_POLICIES[policy]
        base = ARTIFACT_ROOT / scenario / mode / f"seed-{seed}"
    return base / "sac_microgrid.zip", base / "sac_microgrid_vecnormalize.pkl"


def build_job_registry(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    scenarios: list[str] | None = None,
    policies: list[str] | None = None,
    seeds: list[int] | None = None,
    include_summary: bool = False,
) -> list[MonthlyJob]:
    """Resolve the frozen 54-job matrix (or a filtered/extended subset).

    ``include_summary=True`` appends the 18-job F3-summary extension arm; the
    default registry remains exactly the plan's 54 jobs.
    """
    known_policies: dict[str, str] = {}
    known_policies.update(DETERMINISTIC_POLICIES)
    known_policies.update(SAC_POLICIES)
    known_policies.update(SAC_SUMMARY_POLICIES)
    selected_scenarios = scenarios if scenarios else list(SCENARIOS)
    selected_policies = (
        [LEGACY_POLICY_ALIASES.get(policy, policy) for policy in policies]
        if policies
        else [
            *DETERMINISTIC_POLICIES,
            *(SAC_SUMMARY_POLICIES if include_summary else ()),
            *SAC_POLICIES,
        ]
    )
    unknown = [p for p in selected_policies if p not in known_policies]
    if unknown:
        raise ValueError(f"unknown policies: {unknown}; known: {sorted(known_policies)}")
    jobs: list[MonthlyJob] = []
    for scenario in selected_scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
        config_path = SCENARIOS[scenario]
        for policy in selected_policies:
            if policy in DETERMINISTIC_POLICIES:
                jobs.append(
                    MonthlyJob(
                        scenario=scenario,
                        policy=policy,
                        seed=0,
                        config_path=config_path,
                        artifact_path=None,
                        vecnorm_path=None,
                        output_dir=output_root / f"{scenario}_{policy}_seed0",
                    )
                )
            else:
                for seed in seeds if seeds is not None else SAC_SEEDS:
                    artifact, vecnorm = artifact_paths_for(scenario, policy, seed)
                    jobs.append(
                        MonthlyJob(
                            scenario=scenario,
                            policy=policy,
                            seed=seed,
                            config_path=config_path,
                            artifact_path=artifact,
                            vecnorm_path=vecnorm,
                            output_dir=output_root / f"{scenario}_{policy}_seed{seed}",
                        )
                    )
    return jobs


def apply_forecast_mode_override(settings: Settings, policy: str) -> Settings:
    """Point the env at the right forecast inputs for each SAC arm, in memory.

    SAC-none-F3 receives zero-filled forecast slots; the F3-summary arm keeps
    cached forecasts but consumes the fixed-width causal summary instead of the
    raw 96-point vectors (observation shape stays 214). Scenario YAML files and
    trained artifacts are never rewritten.
    """
    if policy == "sac_none_f3":
        settings.rl.forecast_mode = "none"
    elif policy == "sac_f3_summary":
        settings.rl.forecast_representation = "summary"
    return settings


def load_masked_rl_components(job: MonthlyJob) -> tuple[Any, Any, int]:
    """Load a frozen SAC checkpoint with fail-closed normalization stats.

    Returns ``(model, obs_rms, progress_index)``. Missing or shape-incompatible
    VecNormalize statistics abort the job instead of substituting a raw
    constant.
    """
    from microgrid_simulator.rl.train import ALGOS

    assert job.artifact_path is not None and job.vecnorm_path is not None
    if not job.artifact_path.is_file():
        raise FileNotFoundError(f"frozen checkpoint missing: {job.artifact_path}")
    if not job.vecnorm_path.is_file():
        raise FileNotFoundError(
            f"VecNormalize statistics missing for {job.artifact_path}; "
            "masking requires the saved training statistics (fail closed)"
        )
    model = ALGOS["sac"].load(str(job.artifact_path))
    with open(job.vecnorm_path, "rb") as fh:
        norm = pickle.load(fh)
    obs_rms = norm.obs_rms
    settings = load_settings(job.config_path)
    progress_index = get_episode_progress_index(settings)
    if obs_rms.mean.shape[0] != model.observation_space.shape[0]:
        raise ValueError(
            f"VecNormalize shape {obs_rms.mean.shape} does not match checkpoint "
            f"observation space {model.observation_space.shape}"
        )
    return model, obs_rms, progress_index


def masked_rl_action(
    env: MicrogridEnv,
    model: Any,
    obs_rms: Any,
    progress_index: int,
) -> np.ndarray:
    """Predict one frozen-policy action with masked episode progress.

    Applies the checkpoint's saved VecNormalize statistics to the env's raw
    observation, then sets only the normalized progress coordinate to ``0.0``
    (the training mean) so the controller receives no episode-position
    information.
    """
    obs = env._last_observation  # noqa: SLF001
    obs = (obs - obs_rms.mean) / (obs_rms.var + 1e-8) ** 0.5
    obs = obs.clip(-10.0, 10.0).astype("float32")
    obs = obs.copy()
    obs[progress_index] = 0.0
    action, _ = model.predict(obs, deterministic=True)
    return action


def extract_outage_events(
    rows: list[dict[str, Any]], dt_hours: float, threshold_kw: float = 0.0
) -> list[dict[str, Any]]:
    """Collapse contiguous unserved-load intervals into outage events."""
    events: list[dict[str, Any]] = []
    open_event: dict[str, Any] | None = None
    for row in rows:
        unserved_kw = float(row.get("unserved_kw") or 0.0)
        if unserved_kw > threshold_kw:
            if open_event is None:
                open_event = {
                    "start_timestamp": row["timestamp"],
                    "end_timestamp": row["timestamp"],
                    "intervals": 0,
                    "unserved_kwh": 0.0,
                    "peak_unserved_kw": 0.0,
                }
            open_event["end_timestamp"] = row["timestamp"]
            open_event["intervals"] += 1
            open_event["unserved_kwh"] += unserved_kw * dt_hours
            open_event["peak_unserved_kw"] = max(open_event["peak_unserved_kw"], unserved_kw)
        elif open_event is not None:
            events.append(open_event)
            open_event = None
    if open_event is not None:
        events.append(open_event)
    for i, event in enumerate(events, start=1):
        event["event_id"] = i
        event["duration_hours"] = event["intervals"] * dt_hours
    return events


def _daily_rows(rows: list[dict[str, Any]], dt_hours: float) -> list[dict[str, Any]]:
    """Aggregate the trajectory into calendar-day summaries."""
    from microgrid_simulator.ui.rollout import _totals

    per_day = int(round(24.0 / dt_hours))
    daily: list[dict[str, Any]] = []
    for start in range(0, len(rows), per_day):
        chunk = rows[start : start + per_day]
        totals = _totals(chunk, dt_hours)
        daily.append(
            {
                "day": start // per_day + 1,
                "date": str(chunk[0]["timestamp"])[:10],
                "load_kwh": totals["load_kwh"],
                "served_pct": totals["served_energy_pct"],
                "unserved_kwh": totals["unserved_kwh"],
                "blackout_hours": totals["blackout_hours"],
                "pv_used_kwh": totals["pv_used_kwh"],
                "pv_wasted_kwh": totals["pv_wasted_kwh"],
                "diesel_kwh": totals["diesel_kwh"],
                "battery_throughput_kwh": totals["battery_throughput_kwh"],
                "carbon_kg": totals["carbon_kg"],
                "final_soc_pct": totals["final_soc_pct"],
            }
        )
    return daily


def _sustainability_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    soc_values = [
        float(row["soc_pct"])
        for row in rows
        if row.get("soc_pct") is not None and np.isfinite(float(row["soc_pct"]))
    ]
    soh_values = [
        float(row["soh_pct"])
        for row in rows
        if row.get("soh_pct") is not None and np.isfinite(float(row["soh_pct"]))
    ]
    initial = soc_values[0] if soc_values else float("nan")
    final = soc_values[-1] if soc_values else float("nan")
    near_floor = sum(1 for v in soc_values if abs(v - 10.0) <= 0.1)
    near_ceiling = sum(1 for v in soc_values if abs(v - 95.0) <= 0.1)
    return {
        "initial_soc_pct": initial,
        "final_soc_pct": final,
        "soc_change_pct_points": final - initial,
        "intervals_near_soc_floor": near_floor,
        "intervals_near_soc_ceiling": near_ceiling,
        "soh_loss_pct_points": 100.0 - (soh_values[-1] if soh_values else 100.0),
    }


def validate_completed_job(output_dir: Path, expected_steps: int = EXPECTED_STEPS) -> bool:
    """A job is complete only when its trajectory and metrics both validate."""
    trajectory = output_dir / "trajectory.csv"
    metrics_path = output_dir / "metrics.json"
    if not trajectory.is_file() or not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    status = metrics.get("status", {})
    if not status.get("completed"):
        return False
    if status.get("steps") != expected_steps:
        return False
    try:
        frame = pd.read_csv(trajectory, usecols=["timestamp"])
    except (OSError, ValueError):
        return False
    return len(frame) == expected_steps


def _reward_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row if key.startswith("penalty_")})
    return {key: float(sum(float(row.get(key) or 0.0) for row in rows)) for key in keys}


def run_monthly_job(
    job: MonthlyJob,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    smoke_steps: int | None = None,
) -> dict[str, Any]:
    """Run one frozen controller over the uninterrupted March replay.

    The plant resets once before the first decision; battery SOC, diesel
    commitment, degradation state, and cumulative accounting carry through all
    former 72-hour boundaries. Terminal-SOC accounting applies only after the
    final decision of the month.
    """
    from microgrid_simulator.ui.rollout import (
        _dispatch_trace,
        _episode_meta,
        _RollingHorizonController,
        _setup_pypsa_rh_controller,
        _slack_bus_id,
        _step_row,
        _totals,
        policy_action,
    )

    if smoke_steps is not None:
        output_dir = output_root / "_smoke" / job.job_id
    else:
        output_dir = job.output_dir
        if validate_completed_job(output_dir):
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            metrics["resumed"] = True
            return metrics

    settings = load_settings(job.config_path)
    apply_forecast_mode_override(settings, job.policy)

    rl_model = rl_norm = None
    progress_index: int | None = None
    if job.policy in SAC_POLICIES or job.policy in SAC_SUMMARY_POLICIES:
        rl_model, rl_norm, progress_index = load_masked_rl_components(job)

    rollout_policy = DETERMINISTIC_POLICIES.get(job.policy, "rl")
    env = MicrogridEnv(settings=settings)
    _, reset_info = env.reset(seed=job.seed)
    dt = settings.topology.timestep_hours
    slack_id = _slack_bus_id(settings)

    rh_ctrl: _RollingHorizonController | None = None
    rh_summary: dict[str, Any] | None = None
    if rollout_policy == "pypsa_rh":
        ctrl, rh_summary = _setup_pypsa_rh_controller(settings, env)
        rh_ctrl = _RollingHorizonController(ctrl, rh_summary)

    expected_steps = EXPECTED_STEPS if smoke_steps is None else smoke_steps
    rows: list[dict[str, Any]] = []
    unexpected_termination = False
    solver_failures = 0
    for step in range(expected_steps):
        if rollout_policy == "rl":
            assert rl_model is not None and rl_norm is not None and progress_index is not None
            action = masked_rl_action(env, rl_model, rl_norm, progress_index)
        else:
            action = policy_action(
                env, rollout_policy, rl_model=rl_model, rl_norm=rl_norm, rh_ctrl=rh_ctrl
            )
        dispatch = _dispatch_trace(env, rollout_policy, action)
        _, reward, terminated, truncated, info = env.step(action)
        rows.append(_step_row(settings, env, step, reward, info, slack_id, dispatch))
        if not info.get("solver_ok", True):
            solver_failures += 1
        if terminated or truncated:
            if step + 1 < expected_steps:
                unexpected_termination = True
            break

    first = pd.Timestamp(settings.episode.telemetry_start)
    for i, row in enumerate(rows):
        row["timestamp"] = str(first + i * pd.to_timedelta(dt, unit="h"))

    completed = len(rows) == expected_steps and not unexpected_termination
    totals = _totals(rows, dt)
    sustainability = _sustainability_metrics(rows)
    daily = _daily_rows(rows, dt)
    events = extract_outage_events(rows, dt)
    worst_daily = max((d["unserved_kwh"] for d in daily), default=0.0)
    affected_days = sum(1 for d in daily if d["unserved_kwh"] > 0.0)

    meta = _episode_meta(settings, env, rollout_policy, job.seed, reset_info, slack_id)
    identity: dict[str, Any] = {
        **job.as_dict(),
        "progress_masked": job.policy in SAC_POLICIES or job.policy in SAC_SUMMARY_POLICIES,
        "progress_mode": (
            "masked"
            if (job.policy in SAC_POLICIES or job.policy in SAC_SUMMARY_POLICIES)
            else "not_applicable"
        ),
        "progress_observation_index": progress_index,
        "rollout_policy": rollout_policy,
    }
    provenance: dict[str, Any] = {
        "telemetry_source_files": meta.get("telemetry_source_files"),
        "forecast_cache_path": settings.forecast.cache_path,
        "forecast_manifest_path": settings.forecast.manifest_path,
        "forecast_source_id": settings.forecast.source_id,
        "forecast_mode": getattr(settings.rl, "forecast_mode", "cached"),
        "backend": meta.get("backend"),
        "timestep_hours": dt,
        "expected_steps": expected_steps,
        "evaluation_start_time": str(first),
        "evaluation_end_exclusive": str(first + len(rows) * pd.to_timedelta(dt, unit="h")),
        "pypsa_rh": rh_summary,
    }

    summary: dict[str, Any] = {
        **totals,
        **sustainability,
        "outage_count": len(events),
        "max_outage_hours": max((e["duration_hours"] for e in events), default=0.0),
        "worst_daily_unserved_kwh": worst_daily,
        "affected_days": affected_days,
        "equivalent_full_cycles": totals.get("battery_throughput_kwh", 0.0)
        / EQUIVALENT_CYCLE_DENOMINATOR_KWH,
    }

    metrics: dict[str, Any] = {
        "identity": identity,
        "provenance": provenance,
        "status": {
            "completed": completed,
            "steps": len(rows),
            "unexpected_termination": unexpected_termination,
            "solver_failures": solver_failures,
            "smoke": smoke_steps is not None,
        },
        "summary": summary,
        "reward_penalty_totals": _reward_totals(rows),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = pd.DataFrame(rows).loc[:, list(TRAJECTORY_COLUMNS)]
    trajectory.to_csv(output_dir / "trajectory.csv", index=False)
    pd.DataFrame(daily).to_csv(output_dir / "daily.csv", index=False)
    pd.DataFrame(events).to_csv(output_dir / "outage_events.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    env.close()
    return metrics


def aggregate_results(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    jobs: list[MonthlyJob] | None = None,
) -> dict[str, Any]:
    """Write combined summaries over every validated job under ``output_root``.

    The aggregation registry always includes the F3-summary extension arm so
    completed summary jobs are never silently dropped from the summaries.
    """
    registry = (
        jobs
        if jobs is not None
        else build_job_registry(output_root=output_root, include_summary=True)
    )
    records: list[dict[str, Any]] = []
    for job in registry:
        if not validate_completed_job(job.output_dir):
            continue
        metrics = json.loads((job.output_dir / "metrics.json").read_text(encoding="utf-8"))
        identity = metrics["identity"]
        record: dict[str, Any] = {
            "scenario": identity["scenario"],
            "policy": identity["policy"],
            "seed": identity["seed"],
            "job_id": identity["job_id"],
            "completed": metrics["status"]["completed"],
            "steps": metrics["status"]["steps"],
        }
        record.update({name: metrics["summary"].get(name) for name in SUMMARY_METRICS})
        records.append(record)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(records)
    summary_path = output_root / "monthly_summary.csv"
    if records:
        columns = [
            "scenario",
            "policy",
            "seed",
            "job_id",
            "completed",
            "steps",
            *SUMMARY_METRICS,
        ]
        summary_frame = summary_frame.sort_values(["scenario", "policy", "seed"])
        summary_frame.loc[:, columns].to_csv(summary_path, index=False)

    sac_records = [
        r for r in records if r["policy"] in SAC_POLICIES or r["policy"] in SAC_SUMMARY_POLICIES
    ]
    seed_path = output_root / "sac_seed_summary.csv"
    if sac_records:
        frame = pd.DataFrame(sac_records)
        grouped = frame.groupby(["scenario", "policy"])[list(SUMMARY_METRICS)]
        agg = grouped.agg(["mean", "std", "min", "max"]).reset_index()
        agg.columns = [
            f"{col}_{stat}" if stat else col
            for col, stat in zip(
                agg.columns.get_level_values(0), agg.columns.get_level_values(1), strict=False
            )
        ]
        agg.to_csv(seed_path, index=False)

    return {
        "jobs_total": len(registry),
        "jobs_completed": len(records),
        "monthly_summary": str(summary_path) if records else None,
        "sac_seed_summary": str(seed_path) if sac_records else None,
    }
