#!/usr/bin/env python3
"""Run one matched gamma=0.995 fine-tuning iteration for an E0-E5 scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_agent_loop import seed_incumbent_if_missing

from microgrid_simulator.rl.orchestrator import AgentLoopOrchestrator
from microgrid_simulator.rl.registry import (
    IterationRegistry,
    LoopLimits,
    ProposalChange,
    ProposalV1,
)
from microgrid_simulator.rl.runners import SACTrainRunners


class FixedFineTuneProvider:
    def __init__(self, policy: str) -> None:
        self.policy = policy

    def propose(self, completed_changes, budget, limits, iteration) -> ProposalV1:
        return ProposalV1(
            schema_version="agent-retrain-proposal-v1",
            iteration=iteration,
            experiment_type="fine_tune",
            parent_policy=self.policy,
            change=ProposalChange(
                parameter="rl.sac.gamma",
                old_value=0.99,
                new_value=0.995,
            ),
            hypothesis="Longer discount horizon improves reserve credit assignment.",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=[f"E{i}" for i in range(6)])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--episodes", type=int, default=9)
    args = parser.parse_args()

    scenario = args.scenario
    repo = args.repo
    config = repo / "configs" / f"f3-{scenario.lower()}.yaml"
    incumbent_root = repo / "artifacts" / "sac" / "f3-15min" / scenario / "f3"
    artifact_root = repo / "artifacts" / "agent-loop-cross" / scenario.lower()
    report_root = repo / "reports" / "agent-loop-cross" / scenario.lower()
    registry = IterationRegistry(artifact_root)
    runners = SACTrainRunners(
        base_config_path=config,
        incumbent_root=incumbent_root,
        eval_episodes=args.episodes,
    )
    incumbent = seed_incumbent_if_missing(registry, runners, args.episodes)
    runners.incumbent_mean_unserved_kwh = incumbent.mean_unserved_kwh
    orchestrator = AgentLoopOrchestrator(
        registry=registry,
        limits=LoopLimits(max_iterations=1),
        provider=FixedFineTuneProvider(f"{scenario.lower()}_raw_f3"),
        runners=runners,
    )
    outcome = orchestrator.run_iteration()
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": scenario,
        "outcome": {
            "iteration": outcome.iteration,
            "status": outcome.status,
            "detail": outcome.detail,
            "gate_reasons": outcome.gate_reasons,
            "elapsed_budget": orchestrator.budget.model_dump(),
        },
        "incumbent": registry.load_incumbent().model_dump() if registry.load_incumbent() else None,
    }
    (report_root / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
