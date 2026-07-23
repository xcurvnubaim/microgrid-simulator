"""Generate a reproducible 72-hour telemetry replay and paper-facing report."""

# Markdown table rows are intentionally kept as complete string literals.
# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from microgrid_simulator.config import Settings
from microgrid_simulator.ui.rollout import _totals, run_rollout

TRAJECTORY_COLUMNS = (
    "timestamp",
    "step",
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
    "diesel_on",
    "soc_pct",
    "soh_pct",
    "excess_generation_kw",
    "power_balance_residual_kw",
    "carbon_kg",
    "reward",
    "constraint_violation_count",
)


POLICY_LABELS = {
    "rule": "rule controller",
    "mpc": "PyPSA MPC (rolling-horizon optimal dispatch) controller",
    "schedule": "manual fixed-schedule diesel controller",
}

POLICY_LIMITATIONS = {
    "mpc": [
        "- The MPC controller has perfect foresight of load/PV over its optimization "
        "horizon (no forecast error is modeled); near the end of the 72 h episode the "
        "lookahead clamps to the last measured sample instead of running out of data.",
    ],
    "schedule": [
        "- The manual-schedule controller deliberately overgenerates diesel outside its "
        "low-risk window regardless of actual residual; the surplus has no PV/battery sink "
        "commanded for it and shows up as `excess_generation_kwh` plus extra "
        "`diesel_kwh`/`carbon_kg` rather than being curtailed or stored.",
    ],
}

POLICY_EVIDENCE_ROWS = {
    "rule": (
        "| Rule controller | designed baseline | diesel-first residual coverage, "
        "battery gap/surplus handling |"
    ),
    "mpc": (
        "| MPC controller | designed baseline | PyPSA rolling-horizon MILP, perfect-foresight "
        "forecast from the driving backend, replanned every "
        "`backend.rolling_horizon_hours` |"
    ),
    "schedule": (
        "| Manual-schedule controller | designed baseline | clock-driven, not "
        "residual-reactive: plays back the per-scenario `diesel_schedule` timetable "
        "(hour-of-day segments at max / min / off / explicit kW); battery covers any "
        "remaining gap beyond the scheduled diesel setpoint |"
    ),
}


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{value:,.{decimals}f}"


def _daily_summary(rows: list[dict[str, Any]], dt: float) -> list[dict[str, Any]]:
    per_day = round(24.0 / dt)
    daily: list[dict[str, Any]] = []
    for start in range(0, len(rows), per_day):
        subset = rows[start : start + per_day]
        totals = _totals(subset, dt)
        daily.append(
            {
                "day": start // per_day + 1,
                "date": str(subset[0]["timestamp"])[:10],
                "load_kwh": totals["load_kwh"],
                "served_pct": totals["served_energy_pct"],
                "unserved_kwh": totals["unserved_kwh"],
                "pv_used_kwh": totals["pv_used_kwh"],
                "diesel_kwh": totals["diesel_kwh"],
                "battery_throughput_kwh": totals["battery_throughput_kwh"],
                "carbon_kg": totals["carbon_kg"],
                "final_soc_pct": totals["final_soc_pct"],
            }
        )
    return daily


def _reward_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row if key.startswith("penalty_")})
    return {key: float(sum(float(row.get(key) or 0.0) for row in rows)) for key in keys}


def _violation_types(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("constraint_violation_types") or [])
    return dict(sorted(counts.items()))


def _report_markdown(
    settings: Settings,
    metrics: dict[str, Any],
    config_path: Path,
) -> str:
    total = metrics["totals"]
    meta = metrics["meta"]
    daily = metrics["daily"]
    reward = metrics["reward_penalty_totals"]
    violations = metrics["violation_interval_counts"]
    policy = meta["policy"]
    policy_label = POLICY_LABELS.get(policy, policy)
    evidence_row = POLICY_EVIDENCE_ROWS.get(
        policy, f"| Controller | designed baseline | `{policy}` |"
    )

    headline = [
        ("Measured load energy", total["load_kwh"], "kWh", "measured input"),
        ("Served load energy", total["served_kwh"], "kWh", "model output"),
        ("Load served", total["served_energy_pct"], "%", "headline reliability"),
        ("Unserved energy", total["unserved_kwh"], "kWh", "headline failure metric"),
        (
            "Blackout/brownout duration",
            total["blackout_hours"],
            "h",
            "15-min intervals with unserved load",
        ),
        ("Peak unserved power", total["peak_unserved_kw"], "kW", "maximum interval"),
        (
            "Measured PV available",
            total["pv_available_kwh"],
            "kWh",
            "negative standby clipped to zero",
        ),
        ("PV used", total["pv_used_kwh"], "kWh", "before source attribution"),
        ("PV curtailed/spilled", total["pv_wasted_kwh"], "kWh", "PV-only waste definition"),
        ("Diesel generation", total["diesel_kwh"], "kWh", "assumed generator model"),
        ("Diesel runtime", meta["diesel_runtime_hours"], "h", "realized on-state"),
        ("Diesel starts", meta["diesel_starts"], "count", "realized starts"),
        ("Diesel carbon proxy", total["carbon_kg"], "kg CO2e", "0.70 kg/kWh assumed"),
        ("Battery charge energy", total["battery_charge_kwh"], "kWh", "AC-side terminal energy"),
        (
            "Battery charge from PV",
            total["battery_charge_from_pv_kwh"],
            "kWh",
            "surplus-PV share of charging",
        ),
        (
            "Battery charge from grid",
            total["battery_charge_from_grid_kwh"],
            "kWh",
            "grid-import share of charging",
        ),
        (
            "Battery charge from diesel",
            total["battery_charge_from_diesel_kwh"],
            "kWh",
            "surplus-diesel share of charging",
        ),
        (
            "Battery discharge energy",
            total["battery_discharge_kwh"],
            "kWh",
            "AC-side terminal energy",
        ),
        ("Battery throughput", total["battery_throughput_kwh"], "kWh", "charge + discharge"),
        ("Final SOC", total["final_soc_pct"], "%", "assumed 500 kWh battery"),
        (
            "SoH loss proxy",
            total["soh_loss_pct_points"],
            "percentage points",
            "placeholder degradation law",
        ),
        (
            "Excess-generation proxy",
            total["excess_generation_kwh"],
            "kWh",
            "not an explicit modeled sink",
        ),
        (
            "Maximum balance residual",
            total["max_abs_power_balance_residual_kw"],
            "kW",
            "after named excess proxy",
        ),
    ]

    lines = [
        f"# 72-hour islanded telemetry replay — {policy_label}",
        "",
        "## Experiment identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Scenario | `{settings.scenario.name}` |",
        f"| Controller | `{meta['policy']}` |",
        f"| Backend | `{meta['backend']}` (lossless algebraic scheduling model) |",
        f"| Evaluated interval | {meta['evaluation_start_time']} to {meta['evaluation_end_exclusive']} |",
        f"| Resolution | {settings.topology.timestep_hours * 60:g} min; {meta['steps']} intervals |",
        f"| Configuration | `{config_path}` |",
        f"| Load source | `{meta['telemetry_source_files']['load']}` |",
        f"| PV source | `{meta['telemetry_source_files']['pv']}` |",
        "| Grid | unavailable for all intervals |",
        "",
        "The load and PV series are measured and timestamp-aligned. One preceding 15-minute",
        "sample is controller context; it is not included in the energy totals.",
        "",
    ]
    if policy == "schedule":
        lines.extend(
            [
                "## Diesel timetable (designed scenario assumption)",
                "",
                "| Hours | Diesel level |",
                "|---|---|",
                *(
                    f"| {seg.start_hour:g}:00 - {seg.end_hour:g}:00 | `{seg.level}` |"
                    for seg in settings.diesel_schedule.segments
                ),
                "",
                "Hours not covered by a segment mean diesel off. Levels: `max` = nameplate,",
                "`min` = minimum stable load, numeric = explicit kW setpoint.",
                "",
            ]
        )
    lines += [
        "## Paper evaluation table",
        "",
        "| Metric | Value | Unit | Interpretation |",
        "|---|---:|---|---|",
    ]
    lines.extend(
        f"| {name} | {_fmt(float(value))} | {unit} | {meaning} |"
        for name, value, unit, meaning in headline
    )
    lines.extend(
        [
            "",
            "## Daily breakdown",
            "",
            "| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        "| {day} | {date} | {load} | {served} | {unserved} | {pv} | {diesel} | {batt} | {carbon} | {soc} |".format(
            day=row["day"],
            date=row["date"],
            load=_fmt(row["load_kwh"]),
            served=_fmt(row["served_pct"]),
            unserved=_fmt(row["unserved_kwh"]),
            pv=_fmt(row["pv_used_kwh"]),
            diesel=_fmt(row["diesel_kwh"]),
            batt=_fmt(row["battery_throughput_kwh"]),
            carbon=_fmt(row["carbon_kg"]),
            soc=_fmt(row["final_soc_pct"]),
        )
        for row in daily
    )
    lines.extend(
        [
            "",
            "## Reward and constraint audit",
            "",
            "Reward terms below are raw accumulated penalty magnitudes, before their weights",
            "are combined into the reported total reward.",
            "",
            "| Term | Accumulated magnitude |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{key}` | {_fmt(value, 6)} |" for key, value in reward.items())
    lines.extend(
        [
            "",
            f"Constraint-violation events: **{total['constraint_violation_events']}**; interval counts by type: `{json.dumps(violations, sort_keys=True)}`.",
            "",
            "## Parameter evidence",
            "",
            "| Quantity | Evidence class | Value/use |",
            "|---|---|---|",
            "| Load and PV | measured | aligned campus telemetry for this exact window |",
            "| PV negative night readings | designed preprocessing | clipped to zero generation availability |",
            "| Battery | assumed/project constraint | 500 kWh, ±250 kW, 96%/96%, SOC 10–95% |",
            "| Diesel | assumed | 150 kW, 45 kW minimum, ramp/up/down limits, 0.70 kg CO2e/kWh |",
            "| Load allocation and Q | assumed | 3:2:2 and Q=0.2P; Q unused by the simple backend |",
            evidence_row,
            "",
            "## Limitations",
            "",
            "- This is a historical-telemetry-fed simulation, not measured historical dispatch.",
            "- The simple backend is lossless and reports flat 1 pu voltage; it does not validate feeder voltage, losses, or line loading.",
            "- Battery charge is source-resolved into PV/grid/diesel shares by a fixed PV-then-diesel-then-grid merit order; the split is an accounting attribution over the lossless single-bus balance, not a metered physical routing.",
            "- `excess_generation_kwh` exposes supply above served load and battery charging, but no physical dump load is modeled.",
            "- Battery degradation is a placeholder throughput/SOC-stress proxy, not a calibrated life model.",
            "- Equipment and feeder assumptions remain subject to the provenance blockers; results establish a controller baseline, not physical-campus validation.",
        ]
        + POLICY_LIMITATIONS.get(policy, [])
        + [
            "",
            "## Reproduction",
            "",
            "```bash",
            f"uv run microgrid-sim replay --config {config_path} --output-dir "
            f"{metrics['output_dir']} --policy {policy}",
            "```",
            "",
            "`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains",
            "the machine-readable aggregate, daily, reward, and violation results.",
            "`visualization.png` is the corresponding four-panel paper-style time-series figure.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_telemetry_replay(
    trajectory: pd.DataFrame, output_path: Path, policy: str = "rule"
) -> None:
    """Render the measured inputs, dispatch, SOC, and reliability diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    frame = trajectory.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    time = frame["timestamp"]
    battery_to_bus_kw = frame["battery_discharge_kw"] - frame["battery_charge_kw"]
    policy_label = POLICY_LABELS.get(policy, policy)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"72-hour islanded campus microgrid replay — measured load/PV, {policy_label}",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(time, frame["load_kw"], color="#202124", linewidth=2.2, label="Load demand")
    ax.plot(
        time,
        frame["served_kw"],
        color="#188038",
        linewidth=1.6,
        linestyle="--",
        label="Load served",
    )
    ax.plot(time, frame["pv_used_kw"], color="#f9ab00", linewidth=1.5, label="PV used")
    ax.plot(time, frame["diesel_kw"], color="#c5221f", linewidth=1.5, label="Diesel")
    ax.plot(
        time,
        battery_to_bus_kw,
        color="#1a73e8",
        linewidth=1.3,
        label="Battery to bus (+ discharge)",
    )
    ax.axhline(0.0, color="#5f6368", linewidth=0.8)
    ax.set_ylabel("Power (kW)")
    ax.set_title("Dispatch and load")
    ax.legend(ncol=5, loc="upper right", fontsize=8)

    ax = axes[1]
    ax.fill_between(
        time,
        0.0,
        frame["pv_available_kw"],
        color="#fde293",
        alpha=0.65,
        label="PV available",
    )
    ax.plot(time, frame["pv_used_kw"], color="#e37400", linewidth=1.7, label="PV used")
    ax.plot(
        time,
        frame["pv_wasted_kw"],
        color="#d93025",
        linewidth=1.2,
        label="PV spill",
    )
    ax.set_ylabel("Power (kW)")
    ax.set_title("Measured PV availability and realized use")
    ax.legend(ncol=3, loc="upper right", fontsize=8)

    ax = axes[2]
    ax.fill_between(
        time,
        0.0,
        battery_to_bus_kw,
        where=battery_to_bus_kw >= 0.0,
        color="#1a73e8",
        alpha=0.55,
        label="Discharge",
    )
    ax.fill_between(
        time,
        0.0,
        battery_to_bus_kw,
        where=battery_to_bus_kw < 0.0,
        color="#00acc1",
        alpha=0.55,
        label="Charge",
    )
    ax.axhline(0.0, color="#5f6368", linewidth=0.8)
    ax.set_ylabel("Battery to bus (kW)")
    ax.set_title("Battery action and state of charge")
    soc_ax = ax.twinx()
    soc_ax.plot(time, frame["soc_pct"], color="#7b1fa2", linewidth=2.0, label="SOC")
    soc_ax.set_ylabel("SOC (%)", color="#7b1fa2")
    soc_ax.set_ylim(5.0, 100.0)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = soc_ax.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, ncol=3, loc="upper right", fontsize=8)

    ax = axes[3]
    ax.fill_between(
        time,
        0.0,
        frame["unserved_kw"],
        color="#d93025",
        alpha=0.65,
        label="Unserved load",
    )
    ax.plot(
        time,
        frame["excess_generation_kw"],
        color="#9334e6",
        linewidth=1.5,
        label="Excess-generation proxy",
    )
    ax.set_ylabel("Power (kW)")
    ax.set_title("Reliability and unresolved surplus")
    ax.legend(ncol=2, loc="upper right", fontsize=8)

    for ax in axes:
        for boundary in pd.date_range(time.iloc[0].normalize(), time.iloc[-1], freq="1D"):
            ax.axvline(boundary, color="#9aa0a6", linewidth=0.7, alpha=0.45)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    axes[-1].set_xlabel("Measured telemetry timestamp")
    fig.text(
        0.01,
        0.002,
        "Simple lossless backend; BESS and diesel parameters are assumed. "
        "PV spill is explicit; excess generation is a proxy, not a modeled dump load.",
        fontsize=9,
        color="#5f6368",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


SUPPORTED_POLICIES = ("rule", "mpc", "schedule")


def run_telemetry_replay(
    settings: Settings,
    config_path: Path,
    output_dir: Path,
    policy: str = "rule",
    seed: int = 0,
) -> dict[str, Any]:
    """Run and persist one deterministic paper baseline."""
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"paper telemetry baseline currently supports policy in {SUPPORTED_POLICIES}"
        )
    result = run_rollout(settings, policy=policy, seed=seed)
    rows = result["rows"]
    if len(rows) != 288 or settings.topology.timestep_hours != 0.25:
        raise ValueError(
            f"paper baseline requires 288 x 15-minute rows; got {len(rows)} x "
            f"{settings.topology.timestep_hours * 60:g} minutes"
        )
    if not result["meta"]["demand_is_real"] or not result["meta"]["pv_is_real"]:
        raise ValueError("paper baseline refuses synthetic load or PV")

    first = pd.Timestamp(settings.episode.telemetry_start)
    dt = pd.to_timedelta(settings.topology.timestep_hours, unit="h")
    for i, row in enumerate(rows):
        row["timestamp"] = str(first + i * dt)

    result["meta"]["evaluation_start_time"] = str(first)
    result["meta"]["evaluation_end_exclusive"] = str(first + len(rows) * dt)
    daily = _daily_summary(rows, settings.topology.timestep_hours)
    metrics: dict[str, Any] = {
        "scenario": settings.scenario.model_dump(),
        "meta": result["meta"],
        "totals": result["totals"],
        "daily": daily,
        "reward_penalty_totals": _reward_totals(rows),
        "violation_interval_counts": _violation_types(rows),
        "output_dir": str(output_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = pd.DataFrame(rows).loc[:, list(TRAJECTORY_COLUMNS)]
    trajectory.to_csv(output_dir / "trajectory.csv", index=False)
    plot_telemetry_replay(trajectory, output_dir / "visualization.png", policy=policy)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        _report_markdown(settings, metrics, config_path), encoding="utf-8"
    )
    return metrics
