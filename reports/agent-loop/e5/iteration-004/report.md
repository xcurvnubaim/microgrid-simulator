# Agent Loop — Iteration 004

## Proposal

- **Change:** `reward.w_unserved`: None → 250.0
- **Hypothesis:** A stronger soft penalty is a diagnostic for penalty-sensitivity of outages.

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
- `reasons`: ['solver_failures=39', 'avoidable_unserved=2.46', 'terminal_soc_violations=27']
