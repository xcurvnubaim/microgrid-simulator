#!/usr/bin/env python3
"""Evaluate the retained E5 raw-F3 SAC seeds with the opt-in RL shield."""

from __future__ import annotations

import argparse
from pathlib import Path

from microgrid_simulator.config import load_settings
from microgrid_simulator.rl.runners import evaluate_seed_on_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/f3-e5.yaml"))
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/sac/f3-15min/E5/f3")
    )
    parser.add_argument("--episodes", type=int, default=9)
    args = parser.parse_args()

    settings = load_settings(args.config)
    settings.rl.action_shield = True
    settings.rl.shield_terminal_hours = 12.0
    unserved: list[float] = []
    rewards: list[float] = []
    worst = 0.0
    solver_failures = 0
    terminal_violations = 0
    avoidable = 0.0
    for seed in range(3):
        artifact = args.root / f"seed-{seed}" / "sac_microgrid.zip"
        result = evaluate_seed_on_validation(settings, artifact, args.episodes, 1234 + seed)
        unserved.extend(result.episode_unserved_kwh)
        rewards.extend(result.episode_rewards)
        worst = max(worst, result.worst_episode_unserved_kwh)
        solver_failures += result.solver_failure_steps
        terminal_violations += result.terminal_soc_violations
        avoidable += result.avoidable_kwh_per_episode
        seed_mean_unserved = sum(result.episode_unserved_kwh) / len(result.episode_unserved_kwh)
        seed_mean_reward = sum(result.episode_rewards) / len(result.episode_rewards)
        print(
            f"seed-{seed}: mean_unserved={seed_mean_unserved:.2f} "
            f"worst={result.worst_episode_unserved_kwh:.2f} "
            f"mean_reward={seed_mean_reward:.1f} "
            f"avoidable={result.avoidable_kwh_per_episode:.2f}"
        )

    print("=== SHIELDED E5 SUMMARY ===")
    print(f"mean_unserved_kwh={sum(unserved) / len(unserved):.2f}")
    print(f"worst_episode_unserved_kwh={worst:.2f}")
    print(f"mean_reward={sum(rewards) / len(rewards):.1f}")
    print(f"avoidable_unserved_kwh_per_episode={avoidable / 3:.2f}")
    print(f"solver_failure_steps={solver_failures}")
    print(f"terminal_soc_violations={terminal_violations}")


if __name__ == "__main__":
    main()
