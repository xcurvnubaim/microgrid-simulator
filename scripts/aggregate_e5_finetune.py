#!/usr/bin/env python3
"""Evaluate the E5 gamma=0.995 fine-tuned arm under the cross-scenario protocol."""
from __future__ import annotations

from pathlib import Path

from microgrid_simulator.config import load_settings
from microgrid_simulator.rl.runners import evaluate_seed_on_validation


def main() -> None:
    settings = load_settings(Path("configs/f3-e5.yaml"))
    results = []
    for seed in range(3):
        artifact = Path(f"artifacts/agent-loop/e5/iteration-002/seed-{seed}/sac_microgrid.zip")
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
    pv = sum(r.pv_wasted_kwh for r in results) / n
    excess = sum(r.excess_kwh for r in results) / n
    avoid = sum(r.avoidable_kwh_per_episode for r in results) / n
    solver = sum(r.solver_failure_steps for r in results)
    terminal = sum(r.terminal_soc_violations for r in results)
    rate = 0.4
    worst = max(unserved)
    print(
        f"E5 | mean_unserved={sum(unserved)/len(unserved):.2f} | "
        f"served={100*served/max(load,1):.2f}% | final_soc={soc:.1f}% | "
        f"diesel={diesel:.0f} fuel=${diesel*rate:,.0f} carbon={carbon:,.0f} | "
        f"health={health:.1f} | pv_waste={pv:.0f} excess={excess:.0f} | "
        f"avoidable={avoid:.2f} solver={solver} terminal={terminal} | "
        f"worst={worst:.2f} reward={sum(rewards)/len(rewards):.1f}"
    )


if __name__ == "__main__":
    main()
