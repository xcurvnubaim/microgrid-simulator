"""Deterministic reliability diagnostics for the agent-gated SAC loop.

Classes and functions in this module are never assigned to an LLM. They
calculate facts, classify outages, and aggregate reward components from
trajectory CSVs produced by the evaluation runner.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UNSERVED_TOLERANCE_KW = 2.0


@dataclass
class OutageEvent:
    """One contiguous interval where unserved power exceeded tolerance."""

    start_step: int
    end_step: int
    duration_steps: int
    unserved_kwh: float
    peak_unserved_kw: float
    classification: str = "indeterminate"  # filled by the classifier


@dataclass
class TrajectoryStep:
    """One 15-minute decision step extracted from a trajectory CSV."""

    step: int
    hour: float
    load_kw: float
    pv_available_kw: float
    pv_used_kw: float
    battery_kw: float
    battery_discharge_kw: float
    diesel_kw: float
    diesel_on: bool
    diesel_load_serving_kw: float
    unserved_kw: float
    soc_pct: float
    reward: float
    penalty_carbon: float
    penalty_autonomy: float
    penalty_health: float
    penalty_waste: float
    penalty_excess: float
    penalty_unserved: float
    penalty_fuel: float
    penalty_constraint: float
    constraint_violation_count: int


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for one evaluation episode."""

    total_reward: float
    total_unserved_kwh: float
    total_diesel_kwh: float
    total_diesel_starts: int
    total_pv_curtailed_kwh: float
    total_excess_kwh: float
    total_health_penalty: float
    avg_soc_pct: float
    min_soc_pct: float
    terminal_soc_pct: float
    terminal_soc_deviation: float
    outage_events: list[OutageEvent] = field(default_factory=list)


def load_trajectory(path: Path) -> list[TrajectoryStep]:
    """Parse a trajectory CSV into a list of TrajectoryStep."""
    steps: list[TrajectoryStep] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            steps.append(
                TrajectoryStep(
                    step=int(row["step"]),
                    hour=float(row["hour"]),
                    load_kw=_f(row, "load_kw"),
                    pv_available_kw=_f(row, "pv_available_kw"),
                    pv_used_kw=_f(row, "pv_used_kw"),
                    battery_kw=_f(row, "battery_kw"),
                    battery_discharge_kw=_f(row, "battery_discharge_kw"),
                    diesel_kw=_f(row, "diesel_kw"),
                    diesel_on=row.get("diesel_on", "False").strip().lower() == "true",
                    diesel_load_serving_kw=_f(row, "diesel_load_serving_kw"),
                    unserved_kw=_f(row, "unserved_kw"),
                    soc_pct=_f(row, "soc_pct"),
                    reward=_f(row, "reward"),
                    penalty_carbon=_f(row, "penalty_carbon"),
                    penalty_autonomy=_f(row, "penalty_autonomy"),
                    penalty_health=_f(row, "penalty_health"),
                    penalty_waste=_f(row, "penalty_waste"),
                    penalty_excess=_f(row, "penalty_excess"),
                    penalty_unserved=_f(row, "penalty_unserved"),
                    penalty_fuel=_f(row, "penalty_fuel"),
                    penalty_constraint=_f(row, "penalty_constraint"),
                    constraint_violation_count=int(_f(row, "constraint_violation_count")),
                )
            )
    return steps


def _f(row: dict[str, str], key: str) -> float:
    """Safely parse a float from a CSV row."""
    val = row.get(key, "0.0")
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def classify_avoidability(
    step: TrajectoryStep,
    battery_max_discharge_kw: float = 250.0,
    battery_soc_min_pct: float = 10.0,
    diesel_max_kw: float = 400.0,
    diesel_min_kw: float = 80.0,
) -> str:
    """Classify whether an unserved-load interval was avoidable.

    Returns one of:
      - "immediately_avoidable"
      - "energy_planning_failure"
      - "commitment_limited"
      - "physically_unavoidable"
      - "indeterminate"

    Only headroom from resources already realizable this tick counts as
    immediate: battery discharge above the SOC floor, plus ramp headroom of
    an already-committed diesel. Starting an idle genset is subject to
    startup delay / minimum-down constraints, so an off diesel contributes
    zero immediate headroom and instead yields "commitment_limited".
    """
    if step.unserved_kw <= UNSERVED_TOLERANCE_KW:
        return "indeterminate"

    # Immediate headroom at this tick.
    battery_headroom = battery_max_discharge_kw if step.soc_pct > battery_soc_min_pct else 0.0
    diesel_headroom = max(0.0, diesel_max_kw - step.diesel_kw) if step.diesel_on else 0.0

    if battery_headroom + diesel_headroom >= step.unserved_kw:
        return "immediately_avoidable"

    # Episode-level resource check against frozen plant capacity.
    total_capacity = battery_max_discharge_kw + diesel_max_kw
    if step.pv_available_kw + total_capacity < step.load_kw:
        return "physically_unavoidable"

    # Capacity exists somewhere; identify why it was not usable now.
    if not step.diesel_on:
        # A start would have been required: startup delay / min-down / ramp.
        return "commitment_limited"
    if step.soc_pct <= battery_soc_min_pct:
        # Diesel committed but exhausted; reserve was spent earlier.
        return "energy_planning_failure"
    return "commitment_limited"


def extract_outages(
    steps: list[TrajectoryStep],
    dt_hours: float = 0.25,
    tolerance_kw: float = UNSERVED_TOLERANCE_KW,
) -> list[OutageEvent]:
    """Extract contiguous unserved-load events from a trajectory."""
    events: list[OutageEvent] = []
    current: list[TrajectoryStep] = []
    for s in steps:
        if s.unserved_kw > tolerance_kw:
            current.append(s)
        elif current:
            peak = max(e.unserved_kw for e in current)
            total_kwh = sum(e.unserved_kw for e in current) * dt_hours
            events.append(
                OutageEvent(
                    start_step=current[0].step,
                    end_step=current[-1].step,
                    duration_steps=len(current),
                    unserved_kwh=total_kwh,
                    peak_unserved_kw=peak,
                )
            )
            current = []
    if current:
        peak = max(e.unserved_kw for e in current)
        total_kwh = sum(e.unserved_kw for e in current) * dt_hours
        events.append(
            OutageEvent(
                start_step=current[0].step,
                end_step=current[-1].step,
                duration_steps=len(current),
                unserved_kwh=total_kwh,
                peak_unserved_kw=peak,
            )
        )
    return events


def compute_episode_metrics(
    steps: list[TrajectoryStep],
    terminal_soc_target: float = 50.0,
    terminal_soc_tolerance: float = 1.0,
) -> EpisodeMetrics:
    """Aggregate metrics from a full trajectory."""
    total_reward = sum(s.reward for s in steps)
    total_unserved_kwh = sum(s.unserved_kw for s in steps) * 0.25
    total_diesel_kwh = sum(s.diesel_kw for s in steps) * 0.25
    total_pv_curtailed_kwh = sum(
        max(0.0, s.pv_available_kw - s.pv_used_kw) for s in steps
    ) * 0.25
    total_excess_kwh = sum(
        max(0.0, s.diesel_kw - s.diesel_load_serving_kw) for s in steps
    ) * 0.25

    socs = [s.soc_pct for s in steps]
    outages = extract_outages(steps)
    for event in outages:
        relevant = steps[event.start_step : event.end_step + 1]
        classifications = [classify_avoidability(s) for s in relevant]
        event.classification = max(set(classifications), key=classifications.count)

    return EpisodeMetrics(
        total_reward=total_reward,
        total_unserved_kwh=total_unserved_kwh,
        total_diesel_kwh=total_diesel_kwh,
        total_diesel_starts=_count_diesel_starts(steps),
        total_pv_curtailed_kwh=total_pv_curtailed_kwh,
        total_excess_kwh=total_excess_kwh,
        total_health_penalty=sum(s.penalty_health for s in steps),
        avg_soc_pct=sum(socs) / len(socs) if socs else 0.0,
        min_soc_pct=min(socs) if socs else 0.0,
        terminal_soc_pct=socs[-1] if socs else 0.0,
        terminal_soc_deviation=abs(socs[-1] - terminal_soc_target) if socs else 0.0,
        outage_events=outages,
    )


def _count_diesel_starts(steps: list[TrajectoryStep]) -> int:
    """Count diesel engine starts (off -> on transitions)."""
    starts = 0
    was_on = False
    for s in steps:
        if s.diesel_on and not was_on:
            starts += 1
        was_on = s.diesel_on
    return starts


def evaluate_trajectory_file(
    csv_path: Path,
    terminal_soc_target: float = 50.0,
) -> dict[str, Any]:
    """Load a trajectory CSV, compute metrics, return dict."""
    steps = load_trajectory(csv_path)
    metrics = compute_episode_metrics(steps, terminal_soc_target=terminal_soc_target)

    def _sum_classified(label: str) -> float:
        return sum(e.unserved_kwh for e in metrics.outage_events if e.classification == label)

    return {
        "total_reward": metrics.total_reward,
        "total_unserved_kwh": metrics.total_unserved_kwh,
        "total_diesel_kwh": metrics.total_diesel_kwh,
        "total_diesel_starts": metrics.total_diesel_starts,
        "total_pv_curtailed_kwh": metrics.total_pv_curtailed_kwh,
        "total_excess_kwh": metrics.total_excess_kwh,
        "total_health_penalty": metrics.total_health_penalty,
        "avg_soc_pct": metrics.avg_soc_pct,
        "min_soc_pct": metrics.min_soc_pct,
        "terminal_soc_pct": metrics.terminal_soc_pct,
        "terminal_soc_deviation": metrics.terminal_soc_deviation,
        "outage_count": len(metrics.outage_events),
        "total_outage_kwh": sum(e.unserved_kwh for e in metrics.outage_events),
        "avoidable_outage_kwh": _sum_classified("immediately_avoidable"),
        "energy_planning_kwh": _sum_classified("energy_planning_failure"),
        "commitment_limited_kwh": _sum_classified("commitment_limited"),
        "unavoidable_kwh": _sum_classified("physically_unavoidable"),
    }