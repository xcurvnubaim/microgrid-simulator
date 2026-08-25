# 72-hour islanded telemetry replay — PyPSA MPC (rolling-horizon optimal dispatch) controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e1_high_diesel_fuel` |
| Controller | `mpc` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-25 00:00:00 to 2026-03-28 00:00:00 |
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
| Measured load energy | 7,938.400 | kWh | measured input |
| Served load energy | 7,176.548 | kWh | model output |
| Load served | 90.403 | % | headline reliability |
| Unserved energy | 761.852 | kWh | headline failure metric |
| Blackout/brownout duration | 29.000 | h | 15-min intervals with unserved load |
| Peak unserved power | 118.944 | kW | maximum interval |
| Measured PV available | 1,484.190 | kWh | negative standby clipped to zero |
| PV used | 1,144.465 | kWh | before source attribution |
| PV curtailed/spilled | 339.725 | kWh | PV-only waste definition |
| Diesel generation | 6,277.679 | kWh | assumed generator model |
| Diesel load-serving energy | 5,488.637 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 256.293 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 95.917 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 60.250 | h | realized on-state |
| Diesel starts | 8.000 | count | realized starts |
| Diesel carbon proxy | 4,394.376 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 532.749 | kWh | AC-side terminal energy |
| Battery charge from PV | 0.000 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 532.749 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 543.446 | kWh | AC-side terminal energy |
| Battery throughput | 1,076.196 | kWh | charge + discharge |
| Final SOC | 39.067 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.043 | percentage points | placeholder degradation law |
| Excess generation | 256.293 | kWh | non-load-serving generation |
| Dump-load energy | 256.293 | kWh | explicit sink for non-exportable surplus |
| Network losses | 88.362 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-25 | 2,553.700 | 90.739 | 236.494 | 646.838 | 1,740.192 | 435.577 | 1,218.134 | 37.157 |
| 2 | 2026-03-26 | 2,885.200 | 85.515 | 417.923 | 411.808 | 2,149.111 | 302.716 | 1,504.378 | 50.504 |
| 3 | 2026-03-27 | 2,499.500 | 95.702 | 107.434 | 85.819 | 2,388.376 | 337.903 | 1,671.863 | 39.067 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 17,577.502540 |
| `penalty_constraint` | 10.932781 |
| `penalty_excess` | 256.293393 |
| `penalty_fuel` | 36,940.861497 |
| `penalty_health` | 2.152391 |
| `penalty_unserved` | 15,237.038126 |
| `penalty_waste` | 339.725157 |

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
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E1/mpc/2026-03-25_00-00-00 --policy mpc
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
