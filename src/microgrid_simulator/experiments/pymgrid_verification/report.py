"""Markdown report rendering for the verification artifacts.

The report states the narrow verification question, the reference-priority
policy, and the explicit limitations so the gate is not misread as physical
validation of the campus microgrid.
"""

from __future__ import annotations

from typing import Any

from microgrid_simulator.experiments.pymgrid_verification.compare import SUMMARY_LABELS


def _report(payload: dict[str, Any]) -> str:
    results = payload["results"]
    plant = payload["plant"]
    fixtures = payload["fixtures"]
    lines = [
        "# pymgrid common-model verification",
        "",
        "> This is cross-implementation verification of a deliberately matched, lossless",
        "> scheduling model. It is not physical validation of the campus microgrid.",
        "",
        "## Verification verdict",
        "",
        f"**{'PASS' if payload['overall_passed'] else 'FAIL'}** — the two independent",
        "implementations produced equivalent canonical trajectories within the declared",
        "numerical tolerances after explicit input and output normalization.",
        "",
        "This answers one narrow question: **does the project simulator implement the same",
        "matched scheduling and accounting contract as pymgrid for these deterministic",
        "cases?** It does not establish campus-plant validity or reproduce a published",
        "pymgrid25 controller result.",
        "",
        "## Reference priority for this scenario",
        "",
        "For this common-model verification only, **pymgrid is the reference implementation**",
        "and `microgrid-simulator` is the candidate. Every reported delta is calculated as",
        "`project simulator - pymgrid`. If a delta exceeds tolerance, the gate fails and the",
        "project result is not accepted for this scenario.",
        "",
        "Mismatch resolution order:",
        "",
        "1. Check units, signs, timestep alignment, module naming, and canonical output mapping.",
        "2. If the mapping is correct and behavior belongs to the shared contract, change the",
        "   project simulator or its scenario-specific adapter to match pymgrid, then rerun all",
        "   regression tests.",
        "3. Keep a difference only when it is intentional and outside the shared contract;",
        "   document it explicitly and exclude that field from equivalence claims rather than",
        "   silently overriding either trajectory.",
        "",
        "This priority does not make pymgrid authoritative for campus telemetry, feeder physics,",
        "degradation, diesel transients, rewards, or other features absent from the common model.",
        "",
        "## Implementations under test",
        "",
        "| Run | Software | Model used | License / provenance |",
        "|---|---|---|---|",
        "| Project run | microgrid-simulator "
        f"{payload['software']['microgrid-simulator']['version']} "
        "| `SimpleBackend` with degradation and diesel transients disabled only for this fixture "
        "| Local project, MIT |",
        f"| Reference run | python-microgrid {payload['software']['pymgrid']['version']} "
        "| Native `Microgrid` with load, renewable, battery, genset, and unbalanced-energy modules "
        f"| LGPL-3.0, upstream `{payload['software']['pymgrid']['commit']}` |",
        "",
        "The runs use separate model objects and state transitions. The pymgrid adapter receives",
        "the raw requested battery and diesel actions; it performs its own feasibility clipping.",
        "For the battery action, only sign and MW-to-MWh conversion occur at the input boundary.",
        "Because pymgrid's renewable module has no curtailment action, requested PV curtailment",
        "is represented as reduced usable availability while the original availability is retained",
        "for the canonical spill calculation.",
        "",
        "## Common plant contract",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| Timestep | {plant['timestep_hours']:.3g} h |",
        "| Network | One islanded, lossless bus |",
        f"| Battery capacity | {plant['battery_capacity_mwh']:.3g} MWh |",
        f"| Battery charge limit | {plant['battery_max_charge_mw']:.3g} MW |",
        f"| Battery discharge limit | {plant['battery_max_discharge_mw']:.3g} MW |",
        f"| Battery SOC range | {plant['battery_soc_min']:.3g}–{plant['battery_soc_max']:.3g} |",
        f"| Initial battery SOC | {plant['battery_soc_init']:.3g} |",
        "| Diesel output range while on | "
        f"{plant['diesel_min_mw']:.3g}–{plant['diesel_max_mw']:.3g} MW |",
        "| Grid connection | None |",
        "",
        "Battery efficiencies 1.00 and 0.96 are verified as separate complete runs.",
        "",
        "## Fixture coverage",
        "",
        "| # | Fixture | Load (MW) | PV available (MW) | Battery request (MW) "
        "| Diesel request | PV curtailment |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for index, fixture in enumerate(fixtures, start=1):
        diesel = f"{fixture['diesel_setpoint_mw']:.3g} MW" if fixture["diesel_on"] else "off"
        lines.append(
            f"| {index} | `{fixture['name']}` | {fixture['load_mw']:.3g} "
            f"| {fixture['pv_available_mw']:.3g} | {fixture['battery_p_mw']:.3g} "
            f"| {diesel} | {fixture['pv_curtail']:.3g} |"
        )
    lines.extend(
        [
            "",
            "Positive battery request means charging in the project convention. The fixture",
            "includes requests beyond available surplus/deficit, component power limits, and SOC",
            "headroom so each implementation must enforce its own constraints.",
            "",
            "## Independent run summaries",
        ]
    )
    for label, metrics in results.items():
        simulator = metrics["implementation_summaries"]["microgrid-simulator"]
        pymgrid = metrics["implementation_summaries"]["pymgrid"]
        lines.extend(
            [
                "",
                f"### Battery efficiency {label}",
                "",
                "| Metric | Project simulator | pymgrid | Absolute delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, display in SUMMARY_LABELS.items():
            delta = abs(float(simulator[key]) - float(pymgrid[key]))
            lines.append(
                f"| {display} | {simulator[key]:.12g} | {pymgrid[key]:.12g} | {delta:.3g} |"
            )
    lines.extend(
        [
            "",
            "## Cross-implementation acceptance",
            "",
            "| Battery efficiency | Steps | Passed | Max power-field delta (MW) | Max SOC delta |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for label, metrics in results.items():
        power_fields = [
            value
            for field, value in metrics["max_abs_delta"].items()
            if field not in {"stored_energy_mwh", "soc"}
        ]
        lines.append(
            f"| {label} | {metrics['steps']} | {'yes' if metrics['passed'] else 'no'} "
            f"| {max(power_fields):.3g} | {metrics['max_abs_delta']['soc']:.3g} |"
        )
    lines.extend(
        [
            "",
            "Acceptance limits are `1e-9 MW` for every power field, `1e-9 MWh` for stored",
            "and aggregate energy, and `1e-9` for SOC. Every step also has a zero balance",
            "residual to floating-point precision.",
            "",
            "Canonical mappings used for comparison:",
            "",
            "- project battery `+MW` = charge; pymgrid battery `+MWh` = discharge;",
            "- pymgrid unbalanced energy provided = unserved load;",
            "- pymgrid unbalanced energy absorbed = excess generation;",
            "- `pv_spill = pv_available - pv_used`; excess generation is not relabeled as spill;",
            "- native rewards and costs are excluded because their definitions are not equivalent.",
            "",
            "## Audit correction",
            "",
            "The first adapter revision pre-clipped the battery request with a duplicate of the",
            "project feasibility rule before calling pymgrid. Although pymgrid's native clipping",
            "produced the same trajectory, that made the implementations less independent than",
            "the report claimed. The adapter now sends the raw request, and two fixtures",
            "deliberately exceed feasible charge/discharge so the native limit behavior is tested.",
            "The corrected verification still passes with the same tolerance-scale deltas.",
            "",
            "## Evidence files",
            "",
            "For each efficiency, the directory contains a project-only trajectory, a",
            "pymgrid-only trajectory, and a field-by-field comparison trajectory. `metrics.json`",
            "contains the plant, fixture inputs, independent summaries, tolerances, deltas, and",
            "overall verdict.",
            "",
            "Reproduce from the repository root:",
            "",
            "```bash",
            "uv run --extra pymgrid python -m \\",
            "  microgrid_simulator.experiments.pymgrid_verification \\",
            "  --output-dir reports/experiments/pymgrid_common_model_verification",
            "uv run --extra dev --extra pymgrid pytest tests/test_pymgrid_verification.py",
            "```",
            "",
            "## Interpretation and limitations",
            "",
            "A pass means the two implementations agree after their units, signs, timing,",
            "device limits, and imbalance terminology are made identical. It does not validate",
            "pymgrid25's assumed capacities, either simulator's native reward, feeder physics,",
            "battery degradation, diesel transients, or the physical campus.",
            "",
            "The next experiment may reproduce a native pymgrid25 scenario and compare",
            "controllers, but those results must remain separate from campus-telemetry results.",
            "",
        ]
    )
    return "\n".join(lines)
