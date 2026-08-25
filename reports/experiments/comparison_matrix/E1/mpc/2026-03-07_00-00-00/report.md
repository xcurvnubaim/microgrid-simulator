# 72-hour islanded telemetry replay — PyPSA MPC (rolling-horizon optimal dispatch) controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e1_high_diesel_fuel` |
| Controller | `mpc` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-07 00:00:00 to 2026-03-10 00:00:00 |
| Resolution | 15 min; 288 intervals |
| Configuration | `/home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml` |
| Load source | `../data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv` |
| PV source | `../data/processed/pv_15min_chronos_reconstruction_candidate.csv` |
| Grid | unavailable for all intervals |

The load and PV series are measured and timestamp-aligned. One preceding 15-minute
sample is controller context; it is not included in the energy totals.

## Paper evaluation table

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Measured load energy | 6,625.300 | kWh | measured input |
| Served load energy | 6,239.739 | kWh | model output |
| Load served | 94.180 | % | headline reliability |
| Unserved energy | 385.561 | kWh | headline failure metric |
| Blackout/brownout duration | 17.000 | h | 15-min intervals with unserved load |
| Peak unserved power | 120.693 | kW | maximum interval |
| Measured PV available | 895.268 | kWh | negative standby clipped to zero |
| PV used | 536.717 | kWh | before source attribution |
| PV curtailed/spilled | 358.551 | kWh | PV-only waste definition |
| Diesel generation | 6,580.546 | kWh | assumed generator model |
| Diesel load-serving energy | 5,282.489 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 661.193 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 89.952 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 62.000 | h | realized on-state |
| Diesel starts | 5.000 | count | realized starts |
| Diesel carbon proxy | 4,606.382 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 636.864 | kWh | AC-side terminal energy |
| Battery charge from PV | 0.000 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 636.864 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 420.533 | kWh | AC-side terminal energy |
| Battery throughput | 1,057.396 | kWh | charge + discharge |
| Final SOC | 84.684 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.043 | percentage points | placeholder degradation law |
| Excess generation | 661.193 | kWh | non-load-serving generation |
| Dump-load energy | 661.193 | kWh | explicit sink for non-exportable surplus |
| Network losses | 65.817 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-07 | 2,788.500 | 96.245 | 104.711 | 211.566 | 2,548.648 | 285.121 | 1,784.053 | 42.733 |
| 2 | 2026-03-08 | 1,503.900 | 99.842 | 2.375 | 194.270 | 1,785.188 | 483.447 | 1,249.632 | 45.891 |
| 3 | 2026-03-09 | 2,332.900 | 88.063 | 278.476 | 130.881 | 2,246.710 | 288.829 | 1,572.697 | 84.684 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 18,425.528262 |
| `penalty_constraint` | 34.684265 |
| `penalty_excess` | 661.192950 |
| `penalty_fuel` | 35,306.619878 |
| `penalty_health` | 2.141450 |
| `penalty_unserved` | 7,711.216965 |
| `penalty_waste` | 358.550759 |

Constraint-violation events: **0**; interval counts by type: `{}`.

## Parameter evidence

| Quantity | Evidence class | Value/use |
|---|---|---|
| Load and PV | measured | aligned campus telemetry for this exact window |
| PV negative night readings | designed preprocessing | clipped to zero generation availability |
| Battery | assumed/project constraint | 500 kWh, ±250 kW, 96%/96%, SOC 10–95% |
| Diesel | assumed | 400 kW (re-anchored to measured 352.8 kW peak), 80 kW minimum, ramp/up/down limits, 0.70 kg CO2e/kWh |
| Load allocation and Q | assumed | 3:2:2 and Q=0.2P; Q unused by the simple backend |
| MPC controller | designed baseline | PyPSA rolling-horizon MILP, perfect-foresight forecast from the driving backend, replanned every `backend.rolling_horizon_hours` |

## Limitations

- This is a historical-telemetry-fed simulation, not measured historical dispatch.
- The simple backend is lossless and reports flat 1 pu voltage; it does not validate feeder voltage, losses, or line loading.
- Battery charge is source-resolved into PV/grid/diesel shares by a fixed PV-then-diesel-then-grid merit order; the split is an accounting attribution over the lossless single-bus balance, not a metered physical routing.
- `excess_generation_kwh` exposes supply above served load and battery charging; non-exportable surplus is routed to the explicit `dump_load_kwh` sink.
- AC network losses and any numerical reference-bus balance are reported separately and are never relabeled as waste.
- Battery degradation is a placeholder throughput/SOC-stress proxy, not a calibrated life model.
- Equipment and feeder assumptions remain subject to the provenance blockers; results establish a controller baseline, not physical-campus validation.
- The MPC controller has perfect foresight of load/PV over its optimization horizon (no forecast error is modeled); near the end of the 72 h episode the lookahead clamps to the last measured sample instead of running out of data.

## Reproduction

```bash
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E1/mpc/2026-03-07_00-00-00 --policy mpc
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
