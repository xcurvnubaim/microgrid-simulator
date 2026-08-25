"""Deterministic iteration report builder for the agent-gated loop.

This module assembles structured data into a Markdown report template.
A lightweight LLM fills narrative sections from verified facts; the
structural template and metric values are never LLM-generated.
"""

from __future__ import annotations

from typing import Any


def build_metrics_table(metrics: dict[str, Any]) -> str:
    """Render a key-value metrics dict as a Markdown table."""
    rows = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            rows.append(f"| {k} | {v:,.2f} |")
        else:
            rows.append(f"| {k} | {v} |")
    return "| Metric | Value |\n|--------|------|\n" + "\n".join(rows) + "\n"


def build_comparison_table(
    candidates: list[tuple[str, dict[str, Any]]],
    metrics: list[str],
) -> str:
    """Build a per-arm comparison table.

    ``candidates`` is a list of ``(label, metrics_dict)`` pairs.
    ``metrics`` is the ordered list of metric keys to include.
    """
    header = "| Arm | " + " | ".join(metrics) + " |\n"
    sep = "|---" + "|---" * len(metrics) + "|\n"
    rows = []
    for label, m in candidates:
        vals = []
        for key in metrics:
            v = m.get(key, 0.0)
            if isinstance(v, float):
                vals.append(f"{v:,.2f}")
            else:
                vals.append(str(v))
        rows.append(f"| {label} | " + " | ".join(vals) + " |")
    return header + sep + "\n".join(rows) + "\n"


def build_outage_table(outages: list[dict[str, Any]]) -> str:
    """Render outage classifications as a table."""
    if not outages:
        return "_No outage events recorded._\n"
    header = "| Classification | Count | Total kWh |\n"
    sep = "|---|---:|---:|\n"
    by_class: dict[str, dict[str, float | int]] = {}
    for o in outages:
        c = o.get("classification", "indeterminate")
        entry = by_class.setdefault(c, {"count": 0, "kwh": 0.0})
        entry["count"] += 1
        entry["kwh"] += o.get("unserved_kwh", 0.0)
    rows = []
    for c in sorted(by_class):
        e = by_class[c]
        rows.append(f"| {c} | {e['count']} | {e['kwh']:,.2f} |")
    return header + sep + "\n".join(rows) + "\n"


def render_iteration_report(
    iteration: int,
    proposal: dict[str, Any] | None,
    smoke_results: dict[str, Any] | None,
    seed_metrics: dict[int, dict[str, Any]] | None,
    comparison_table: str,
    outage_table: str,
    promotion: dict[str, Any] | None,
    narrative_sections: dict[str, str] | None = None,
) -> str:
    """Render a complete iteration report.

    ``narrative_sections`` is a dict keyed by section heading, filled by a
    lightweight LLM from verified facts. All metric tables are deterministic.
    """
    lines: list[str] = []
    lines.append(f"# Agent Loop — Iteration {iteration:03d}\n")

    # Proposal
    lines.append("## Proposal\n")
    if proposal:
        change = proposal.get("change", {})
        param = change.get("parameter", "unknown")
        old = change.get("old_value", "?")
        new = change.get("new_value", "?")
        lines.append(f"- **Change:** `{param}`: {old} → {new}")
        lines.append(f'- **Hypothesis:** {proposal.get("hypothesis", "N/A")}')
        lines.append("")
    else:
        lines.append("_No proposal — initial or baseline iteration._\n")

    # Smoke gate
    lines.append("## Smoke Gate\n")
    if smoke_results:
        passed = smoke_results.get("passed", False)
        lines.append(f"- **Status:** {'PASSED' if passed else 'FAILED'}")
        for k, v in sorted(smoke_results.items()):
            if k != "passed":
                lines.append(f"- `{k}`: {v}")
        lines.append("")
    else:
        lines.append("_No smoke gate — full training only._\n")

    # Seed results
    lines.append("## Training Results\n")
    if seed_metrics:
        for seed, m in sorted(seed_metrics.items()):
            lines.append(f"### Seed {seed}\n")
            lines.append(build_metrics_table(m))
    else:
        lines.append("_No seed data._\n")

    # Comparison
    lines.append("## Comparison\n")
    lines.append(comparison_table)
    lines.append("")

    # Outages
    lines.append("## Outage Diagnostics\n")
    lines.append(outage_table)
    lines.append("")

    # Promotion
    lines.append("## Promotion Gate\n")
    if promotion:
        passed = promotion.get("passed", False)
        lines.append(f"- **Result:** {'PROMOTED' if passed else 'REJECTED'}")
        for k, v in sorted(promotion.items()):
            if k != "passed":
                lines.append(f"- `{k}`: {v}")
        lines.append("")
    else:
        lines.append("_Not evaluated._\n")

    # Narrative (LLM-filled)
    if narrative_sections:
        lines.append("## Analysis\n")
        for heading, text in narrative_sections.items():
            lines.append(f"### {heading}\n")
            lines.append(text)
            lines.append("")

    return "\n".join(lines)


def render_loop_summary(
    iterations: list[dict[str, Any]],
    total_budget: dict[str, Any],
) -> str:
    """Render a summary across all completed iterations."""
    lines: list[str] = []
    lines.append("# Agent Loop Summary\n")
    lines.append(f"- **Iterations completed:** {total_budget.get('completed', 0)}")
    lines.append(f"- **Promoted:** {total_budget.get('promoted', 0)}")
    lines.append(f"- **Budget consumed:** {total_budget.get('steps_consumed', 0):,} steps")
    lines.append(f"- **Wall time:** {total_budget.get('wall_time_hours', 0):.1f} hours")
    lines.append("")

    if iterations:
        lines.append("| Iteration | Change | Result | Unserved Δ |\n")
        lines.append("|---|---:|---:|---:|\n")
        for it in iterations:
            change = it.get("change", "—")
            result = it.get("result", "?")
            delta = it.get("unserved_delta_kwh", 0.0)
            lines.append(f"| {it['iteration']} | {change} | {result} | {delta:+,.2f} |\n")

    return "\n".join(lines)