# 72-hour islanded telemetry replay — PyPSA MPC (rolling-horizon optimal dispatch) controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e1_high_diesel_fuel` |
| Controller | `mpc` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-13 00:00:00 to 2026-03-16 00:00:00 |
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
| Measured load energy | 6,504.900 | kWh | measured input |
| Served load energy | 6,109.215 | kWh | model output |
| Load served | 93.917 | % | headline reliability |
| Unserved energy | 395.685 | kWh | headline failure metric |
| Blackout/brownout duration | 17.000 | h | 15-min intervals with unserved load |
| Peak unserved power | 132.225 | kW | maximum interval |
| Measured PV available | 3,955.120 | kWh | negative standby clipped to zero |
| PV used | 2,939.090 | kWh | before source attribution |
| PV curtailed/spilled | 1,016.030 | kWh | PV-only waste definition |
| Diesel generation | 3,559.896 | kWh | assumed generator model |
| Diesel load-serving energy | 2,717.602 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 317.669 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 91.076 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 39.500 | h | realized on-state |
| Diesel starts | 13.000 | count | realized starts |
| Diesel carbon proxy | 2,491.927 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 825.577 | kWh | AC-side terminal energy |
| Battery charge from PV | 300.952 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 524.625 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 753.475 | kWh | AC-side terminal energy |
| Battery throughput | 1,579.052 | kWh | charge + discharge |
| Final SOC | 51.544 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.063 | percentage points | placeholder degradation law |
| Excess generation | 317.669 | kWh | non-load-serving generation |
| Dump-load energy | 317.669 | kWh | explicit sink for non-exportable surplus |
| Network losses | 83.417 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-13 | 2,270.800 | 96.626 | 76.612 | 927.483 | 1,302.561 | 530.834 | 911.793 | 48.845 |
| 2 | 2026-03-14 | 2,310.200 | 98.005 | 46.086 | 1,059.926 | 1,265.117 | 476.402 | 885.582 | 51.011 |
| 3 | 2026-03-15 | 1,923.900 | 85.811 | 272.987 | 951.682 | 992.217 | 571.816 | 694.552 | 51.544 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 9,967.707849 |
| `penalty_constraint` | 1.543863 |
| `penalty_excess` | 317.669232 |
| `penalty_fuel` | 25,895.499169 |
| `penalty_health` | 3.164388 |
| `penalty_unserved` | 7,913.700119 |
| `penalty_waste` | 1,016.029760 |

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
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E1/mpc/2026-03-13_00-00-00 --policy mpc
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
