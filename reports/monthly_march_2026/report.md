# Continuous One-Month Frozen-Policy Evaluation — March 2026 (E0–E5)

> Frozen policies trained on 72-hour episodes were evaluated post hoc for transfer to an
> uninterrupted 31-day islanded simulation, with the artificial episode-progress input
> masked to its training mean so the policies received no episode-position information.

## Contract

| Item | Value |
|---|---|
| Interval | 2026-03-01 00:00 – 2026-03-31 23:45 (744 h) |
| Decisions | 2,976 consecutive × 15 min, one physical reset |
| Policies | Rule-F3, Schedule, MPC-F3; frozen SAC-F3 / SAC-none-F3 seeds 0–2 |
| SAC progress | masked only (normalized `episode_progress` := 0.0 after VecNormalize; obs index 20 of 214) |
| Forecast | leakage-safe F3 test cache (2,977 records incl. 2026-02-28 23:45 context issue) |
| Jobs | 54/54 completed, no unexpected termination, no solver failures |
| Training/tuning from March | none |

## Deterministic baselines (identical across scenarios: dispatch is not economics-aware)

| Policy | Unserved kWh | Served % | Blackout h | Final SOC % | Diesel kWh | PV spill kWh |
|---|---:|---:|---:|---:|---:|---:|
| schedule | 20.9 | 99.97 | 2.8 | 95.0 | 119,330 | 19,452 |
| rule_f3 | 1,487.1 | 98.04 | 133.2 | 40.6 | 68,086 | 4,837 |
| mpc_f3 | 8,565.4 | 88.73 | 256.5 | 27.8 | 54,320 | 3,386 |

(MPC-F3 varies marginally by scenario through economics in its objective: unserved 8,565–8,654 kWh.)

## SAC seed summary (masked progress)

| Scenario | Policy | Unserved kWh mean | min | max | Served % mean | Final SOC % mean |
|---|---|---:|---:|---:|---:|---:|
| E0 | sac_f3 | 15,159 | 5,843 | 23,876 | 80.05 | 18.4 |
| E0 | sac_none_f3 | 3,147 | 34 | 8,968 | 95.86 | 39.0 |
| E1 | sac_f3 | 11,519 | 10,479 | 12,809 | 84.84 | 39.0 |
| E1 | sac_none_f3 | 20,408 | 3,107 | 36,158 | 73.14 | 11.9 |
| E2 | sac_f3 | 16,098 | 13,834 | 17,848 | 78.81 | 14.3 |
| E2 | sac_none_f3 | 5,834 | 75 | 17,279 | 92.32 | 24.1 |
| E3 | sac_f3 | 22,136 | 12,621 | 35,362 | 70.86 | 21.7 |
| E3 | sac_none_f3 | 31,559 | 22,998 | 44,419 | 58.46 | 10.0 |
| E4 | sac_f3 | 9,155 | 3,981 | 14,761 | 87.95 | 26.1 |
| E4 | sac_none_f3 | 9,964 | 495 | 20,508 | 86.88 | 27.9 |
| E5 | sac_f3 | 6,742 | 943 | 16,941 | 91.13 | 18.7 |
| E5 | sac_none_f3 | 11,390 | 6,804 | 15,587 | 85.01 | 12.1 |

## Interpretation

- **Dispatch sustainability failure dominates.** Most SAC seeds deplete the battery toward the
  10% floor within the month and accumulate multi-MWh unserved energy (up to ~44 MWh, E3
  sac_none_f3). Seed variance is extreme (e.g. E0 sac_none_f3: 34 kWh vs 8,968 kWh), consistent
  with the E1/E3 seed-sensitivity diagnosed in the 72-hour F3 matrix.
- **Schedule is the reliability ceiling** (~21 kWh unserved, 99.97% served) but burns the most
  diesel (119 MWh) and spills the most PV (19.5 GWh → kWh scale: 19,452 kWh); it ends the month
  at the SOC ceiling (95%).
- **Rule-F3** is the best reward-aware baseline on reliability (1,487 kWh, 98.04%) at moderate
  diesel use (68 MWh).
- **MPC-F3 degrades in continuous operation**: rolling-horizon replanning holds ~88.7% served but
  drifts SOC down (final 28–37%) without any month-scale reserve management.
- **Masked-progress transfer is fragile but not uniform**: a minority of seeds sustain service
  near-baselines-level (E0/E2 sac_none_f3 seeds 1–2: 34–149 kWh), showing the frozen policies
  retain useful state-feedback control when their SOC trajectory avoids the floor.

## Claim boundary

Post-hoc transfer/stress evidence only. Not a continuing controller, not trained for monthly
episodes, not a live or physical EMS, and not long-term reliability proof. One autocorrelated
month, three seeds, already-inspected telemetry; descriptive paired differences only. Battery/
diesel parameters are assumed; diesel fuel is unlimited by assumption. Deployment-oriented SAC
requires a separately approved retrain without `episode_progress` under a continuing-task
reward/safety contract.

## Artifacts

- Per job: `<scenario>_<policy>_seed<n>/trajectory.csv` (2,976 rows), `metrics.json`, `daily.csv` (31 rows), `outage_events.csv`
- Combined: `monthly_summary.csv`, `sac_seed_summary.csv`
- Runner: `scripts/run_monthly_matrix.py`; module: `src/microgrid_simulator/experiments/monthly_evaluation.py`

---

# Extension Arm: Fixed-Width F3 Summary (executed on ma012, 2026-08-21)

Post-plan addition: frozen `sac_f3_summary` checkpoints (fixed-width causal forecast
summaries inside the unchanged 214-value observation, masked progress) evaluated over
the same continuous March contract on the remote host (tmux `monthly-summary-eval`,
8 workers). **18/18 jobs completed, zero failures**; merged into this result root.
Default plan matrix remains 54 jobs; the registry resolves 72 with `--include-summary`.

## Raw vs summary scenario means

| Scenario | Raw unserved mean | Summary unserved mean [min-max] | Delta | Summary served % | Summary final SOC % |
|---|---:|---|---:|---:|---:|
| E0 | 15,159 | 732 [122-1,844] | -95% | 99.0 | 33.2 |
| E1 | 11,519 | 16,961 [722-30,016] | +47% | 77.7 | 15.0 |
| E2 | 16,098 | 2,348 [22-5,804] | -85% | 96.9 | 30.5 |
| E3 | 22,136 | 14,965 [503-34,041] | -32% | 80.3 | 28.0 |
| E4 | 9,155 | 8,973 [5,091-15,366] | -2% | 88.2 | 10.0 |
| E5 | 6,742 | 17,856 [10,540-31,974] | +165% | 76.5 | 15.4 |
| **Sum** | **80,809** | **61,835** | **-23%** | | |

## Per-seed outcomes (summary arm)

| Scenario | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| E0 | 122 kWh / 99.8% / SOC 30.8 | 232 kWh / 99.7% / SOC 57.7 | 1,844 kWh / 97.6% / SOC 11.0 |
| E1 | 722 kWh / 99.1% / SOC 24.9 | 20,146 kWh / 73.5% / SOC 10.0 (floor) | 30,016 kWh / 60.5% / SOC 10.0 (floor) |
| E2 | 22 kWh / 100.0% / SOC 63.4 | 5,804 kWh / 92.4% / SOC 13.4 | 1,217 kWh / 98.4% / SOC 14.7 |
| E3 | 503 kWh / 99.3% / SOC 48.1 | 10,352 kWh / 86.4% / SOC 25.8 | 34,041 kWh / 55.2% / SOC 10.0 (floor) |
| E4 | 15,366 kWh / 79.8% / SOC 10.0 (floor) | 6,462 kWh / 91.5% / SOC 10.0 (floor) | 5,091 kWh / 93.3% / SOC 10.0 (floor) |
| E5 | 11,053 kWh / 85.5% / SOC 26.1 | 10,540 kWh / 86.1% / SOC 10.0 (floor) | 31,974 kWh / 57.9% / SOC 10.0 (floor) |

## Reading

- Aggregate unserved energy drops ~23% under the fixed-width encoding; E0 reaches
  99.0% served — the best frozen-SAC reliability observed in this month study.
- The effect inverts in E1 (+47%) and E5 (+165%): forecast-representation value remains
  strongly scenario-dependent, consistent with the F0/F3 episode-scale findings.
- Sustainability failure, seed fragility, and SOC floor-pinning persist regardless of
  encoding (8/18 summary seeds end at the 10% floor; all three E4 seeds): these are
  properties of frozen-policy transfer, not of one observation representation.

Provenance: checkpoints `artifacts/sac/f3-summary/{E0..E5}/seed-{0,1,2}` trained on
`ma012`; evaluation executed remotely and merged locally; no git operations performed
on either host.
