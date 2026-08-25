# Agent Loop — Iteration 002

## Proposal

- **Change:** `rl.sac.gamma`: None → 0.995
- **Hypothesis:** A longer discount horizon credits outages to earlier SOC and diesel decisions.

## Smoke Gate

- **Status:** FAILED
- `status`: RETAINED

## Training Results

_No seed data._

## Comparison



## Outage Diagnostics

_See per-seed meta in outcome.json lineage._

## Promotion Gate

- **Result:** REJECTED
- `reasons`: ['solver_failures=20', 'avoidable_unserved=2.22', 'terminal_soc_violations=27']
