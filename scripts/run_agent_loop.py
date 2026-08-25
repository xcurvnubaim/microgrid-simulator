#!/usr/bin/env python3
"""Run the agent-gated SAC retraining loop (plan sections 3-14).

Constructs the orchestrator with real runners, seeds the incumbent record
from a frozen validation evaluation of the existing E5 raw-F3 checkpoints if
absent, then executes up to ``--iterations`` loop iterations. Every outcome
is written under ``reports/agent-loop/e5/``. Nothing is launched unless this
script is explicitly invoked; the canonical scope decision must exist first.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from microgrid_simulator.rl.orchestrator import AgentLoopOrchestrator
from microgrid_simulator.rl.registry import (
    IncumbentRecord,
    IterationRegistry,
    LoopLimits,
    PromotionGate,
)
from microgrid_simulator.rl.report import render_iteration_report
from microgrid_simulator.rl.runners import SACTrainRunners, evaluate_seed_on_validation

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "f3-e5.yaml"
DEFAULT_INCUMBENT_ROOT = REPO / "artifacts" / "sac" / "f3-15min" / "E5" / "f3"
DEFAULT_ART_ROOT = REPO / "artifacts" / "agent-loop" / "e5"
DEFAULT_REPORT_ROOT = REPO / "reports" / "agent-loop" / "e5"


def seed_incumbent_if_missing(
    registry: IterationRegistry,
    runners: SACTrainRunners,
    eval_episodes: int,
) -> IncumbentRecord:
    record = registry.load_incumbent()
    if record is not None:
        return record

    print("[loop] no incumbent record; evaluating E5 raw-F3 checkpoints on validation...")
    episode_unserved: list[float] = []
    rewards: list[float] = []
    worst = 0.0
    for seed in (0, 1, 2):
        zip_path, _ = runners._seed_paths(seed)
        if not zip_path.is_file():
            raise FileNotFoundError(f"incumbent checkpoint missing: {zip_path}")
        evaluation = evaluate_seed_on_validation(
            runners.base_settings,
            zip_path,
            episodes=eval_episodes,
            eval_seed=runners.eval_seed + seed,
        )
        episode_unserved.extend(evaluation.episode_unserved_kwh)
        rewards.extend(evaluation.episode_rewards)
        worst = max(worst, evaluation.worst_episode_unserved_kwh)
        seed_mean = sum(evaluation.episode_unserved_kwh) / len(evaluation.episode_unserved_kwh)
        print(
            f"[loop] incumbent seed-{seed}: mean={seed_mean:.2f} kWh "
            f"worst={evaluation.worst_episode_unserved_kwh:.2f} kWh"
        )
    record = IncumbentRecord(
        policy_id="e5_raw_f3",
        artifact_dir=str(runners.incumbent_root / "seed-0"),
        mean_unserved_kwh=sum(episode_unserved) / len(episode_unserved),
        worst_episode_unserved_kwh=worst,
        mean_reward=sum(rewards) / len(rewards),
        promoted_at_iteration=0,
        history=[{"provenance": "frozen validation evaluation of existing checkpoints"}],
    )
    registry.promote_incumbent(record)
    print(
        f"[loop] incumbent seeded: mean={record.mean_unserved_kwh:.2f} kWh "
        f"worst={record.worst_episode_unserved_kwh:.2f} kWh reward={record.mean_reward:.1f}"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--incumbent-root", type=Path, default=DEFAULT_INCUMBENT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ART_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=9)
    args = parser.parse_args()

    registry = IterationRegistry(args.artifact_root)
    runners = SACTrainRunners(
        base_config_path=args.config,
        incumbent_root=args.incumbent_root,
        eval_episodes=args.eval_episodes,
    )
    incumbent = seed_incumbent_if_missing(registry, runners, args.eval_episodes)
    runners.incumbent_mean_unserved_kwh = incumbent.mean_unserved_kwh

    orchestrator = AgentLoopOrchestrator(
        registry=registry,
        limits=LoopLimits(max_iterations=args.iterations),
        gate=PromotionGate(),
        runners=runners,
    )

    summary: list[dict[str, object]] = []
    for _ in range(args.iterations):
        started = time.time()
        outcome = orchestrator.run_iteration()
        elapsed_min = (time.time() - started) / 60.0
        print(
            f"[loop] iteration {outcome.iteration}: {outcome.status} "
            f"({elapsed_min:.1f} min) {outcome.detail}",
            flush=True,
        )
        summary.append(
            {
                "iteration": outcome.iteration,
                "status": outcome.status,
                "detail": outcome.detail,
                "gate_reasons": outcome.gate_reasons,
                "promoted_policy_id": outcome.promoted_policy_id,
                "elapsed_min": elapsed_min,
            }
        )

        report_dir = args.report_root / f"iteration-{outcome.iteration:03d}"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "outcome.json").write_text(
            json.dumps(summary[-1], indent=2) + "\n", encoding="utf-8"
        )
        proposal_path = args.artifact_root / f"iteration-{outcome.iteration:03d}" / "proposal.json"
        proposal = json.loads(proposal_path.read_text()) if proposal_path.is_file() else None
        (report_dir / "report.md").write_text(
            render_iteration_report(
                iteration=outcome.iteration,
                proposal=proposal,
                smoke_results={"status": outcome.status},
                seed_metrics=None,
                comparison_table="",
                outage_table="_See per-seed meta in outcome.json lineage._",
                promotion={
                    "passed": outcome.status == "PROMOTED",
                    "reasons": outcome.gate_reasons,
                },
            ),
            encoding="utf-8",
        )

        if outcome.status in {"STOPPED", "EXHAUSTED", "NO_RUNNER"}:
            break

    args.report_root.mkdir(parents=True, exist_ok=True)
    final_incumbent = registry.load_incumbent()
    (args.report_root / "loop-summary.json").write_text(
        json.dumps(
            {
                "budget": orchestrator.budget.model_dump(),
                "iterations": summary,
                "incumbent": final_incumbent.model_dump() if final_incumbent else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[loop] done; budget={orchestrator.budget.model_dump()}", flush=True)


if __name__ == "__main__":
    main()
