# Native pymgrid Scenario 2 — RBC and MPC Verification

Verified 2026-07-23 from
`/home/xcurv/teep-taiwan/notebook/pymgrid.ipynb` and an independent rerun against the
editable checkout at commit `09089ccfaab3e95becfda1b26fbce6f9c6195f6a`.

This is an external one-hour-resolution benchmark using pymgrid's native scenario 2. It
is not a campus-telemetry result and must not be placed in the same result table as the
campus 15-minute rollouts.

## Result

| Controller | Loss load total | Loss-load steps > 1e-8 | Overgeneration total | Overgeneration steps > 1e-8 | PV curtailment total |
|---|---:|---:|---:|---:|---:|
| Native RBC | 109,727,236.813353 | 6,808 | 0 | 0 | 6,885,166.508064 |
| Native MPC | 1.70246e-9 | 0 | 2.40470e-9 | 0 | 6,885,166.508064 |

Across all 8,759 evaluated steps, native MPC therefore has numerical-zero blackout
(`loss_load`) and numerical-zero `overgeneration` at a `1e-8` tolerance. This confirms
the result visible in the notebook.

Zero `overgeneration` is not zero solar waste. The PV module separately records
6,885,166.508064 energy units of `curtailment`; with a one-hour timestep these are
hourly energy quantities in the scenario's native scale.

## Why zero imbalance is feasible

- Load and PV use `OracleForecaster`; MPC sees the current step plus 23 forecast steps,
  giving a 24-hour perfect-foresight horizon.
- The genset has zero startup and wind-down time and begins online.
- Its maximum per-step production is 43,725.6, approximately the scenario's peak load.
- The battery has 66,116 maximum stored-energy units and 16,529 internal-energy units
  per step of symmetric charge/discharge capability.
- The optimization penalizes loss load and has sufficient controllable capacity to
  satisfy the balance.

These conditions explain the result; they do not establish that MPC will have zero
imbalance under campus sizing, measured forecast error, diesel transition constraints,
or a different terminal-SOC policy.

## Reproduction boundary

The native RBC/MPC behavior has been reproduced. The published Halev et al. Table 2
cost comparison, native PPO/RL result, other scenarios, and project-controller
substitution remain separate unfinished tasks. `metrics.json` contains the complete
machine-readable totals from this run.
