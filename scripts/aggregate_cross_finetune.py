#!/usr/bin/env python3
"""Detailed per-scenario objective metrics for cross-scenario fine-tuned checkpoints."""
from __future__ import annotations

from pathlib import Path

from microgrid_simulator.config import load_settings
from microgrid_simulator.rl.runners import evaluate_seed_on_validation


def main() -> None:
    root = Path("artifacts/agent-loop-cross")
    for scenario in ("e0", "e1", "e2", "e3", "e4"):
        settings = load_settings(Path("configs") / f"f3-{scenario}.yaml")
        results = []
        for seed in range(3):
            artifact = root / scenario / "iteration-001" / f"seed-{seed}" / "sac_microgrid.zip"
            results.append(
                evaluate_seed_on_validation(settings, artifact, episodes=9, eval_seed=1234 + seed)
            )
        n = len(results)
        unserved = [u for r in results for u in r.episode_unserved_kwh]
        rewards = [u for r in results for u in r.episode_rewards]
        served = sum(r.served_kwh for r in results)
        load = sum(r.load_kwh for r in results)
        soc = sum(sum(r.final_soc_pcts) for r in results) / sum(
            len(r.final_soc_pcts) for r in results
        )
        diesel = sum(r.diesel_kwh for r in results) / n
        carbon = sum(r.carbon_kg for r in results) / n
        health = sum(r.health_penalty for r in results) / n
        pv_waste = sum(r.pv_wasted_kwh for r in results) / n
        excess = sum(r.excess_kwh for r in results) / n
        avoidable = sum(r.avoidable_kwh_per_episode for r in results) / n
        solver = sum(r.solver_failure_steps for r in results)
        terminal = sum(r.terminal_soc_violations for r in results)
        rate = 1.2 if scenario in ("e1", "e3") else 0.4
        print(
            f"{scenario.upper()} | {sum(unserved)/len(unserved):.2f} | "
            f"{100.0*served/max(load,1):.2f}% | {soc:.1f}% | "
            f"{diesel:.0f} / ${diesel*rate:,.0f} / {carbon:,.0f} | "
            f"{health:.1f} | {pv_waste:.0f} / {excess:.0f} / {excess:.0f} | "
            f"avoid={avoidable:.2f} solver={solver} terminal={terminal} "
            f"reward={sum(rewards)/len(rewards):.1f}"
        )


if __name__ == "__main__":
    main()
