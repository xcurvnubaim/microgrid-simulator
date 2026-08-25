# 72-hour islanded telemetry replay — PyPSA MPC (rolling-horizon optimal dispatch) controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e3_high_fuel_low_battery_use` |
| Controller | `mpc` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-13 00:00:00 to 2026-03-16 00:00:00 |
| Resolution | 15 min; 288 intervals |
| Configuration | `/home/ma012/projects/microgrid-simulator/configs/economic-e3-high-fuel-low-battery-use.yaml` |
| Load source | `../data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv` |
| PV source | `../data/processed/pv_15min_chronos_reconstruction_candidate.csv` |
| Grid | unavailable for all intervals |

The load and PV series are measured and timestamp-aligned. One preceding 15-minute
sample is controller context; it is not included in the energy totals.

## Paper evaluation table

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Measured load energy | 6,504.900 | kWh | measured input |
| Served load energy | 6,105.074 | kWh | model output |
| Load served | 93.853 | % | headline reliability |
| Unserved energy | 399.826 | kWh | headline failure metric |
| Blackout/brownout duration | 17.750 | h | 15-min intervals with unserved load |
| Peak unserved power | 132.225 | kW | maximum interval |
| Measured PV available | 3,955.120 | kWh | negative standby clipped to zero |
| PV used | 2,921.638 | kWh | before source attribution |
| PV curtailed/spilled | 1,033.482 | kWh | PV-only waste definition |
| Diesel generation | 3,596.206 | kWh | assumed generator model |
| Diesel load-serving energy | 2,726.281 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 335.654 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 90.666 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 40.000 | h | realized on-state |
| Diesel starts | 11.000 | count | realized starts |
| Diesel carbon proxy | 2,517.344 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 835.224 | kWh | AC-side terminal energy |
| Battery charge from PV | 300.952 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 534.271 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 758.107 | kWh | AC-side terminal energy |
| Battery throughput | 1,593.330 | kWh | charge + discharge |
| Final SOC | 52.427 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.064 | percentage points | placeholder degradation law |
| Excess generation | 335.654 | kWh | non-load-serving generation |
| Dump-load energy | 335.654 | kWh | explicit sink for non-exportable surplus |
| Network losses | 83.370 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-13 | 2,270.800 | 96.626 | 76.612 | 924.521 | 1,339.274 | 509.173 | 937.492 | 53.428 |
| 2 | 2026-03-14 | 2,310.200 | 98.067 | 44.658 | 1,045.436 | 1,338.434 | 490.710 | 936.904 | 66.976 |
| 3 | 2026-03-15 | 1,923.900 | 85.521 | 278.556 | 951.682 | 918.499 | 593.447 | 642.949 | 52.427 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 10,069.377218 |
| `penalty_constraint` | 2.427407 |
| `penalty_excess` | 335.653533 |
| `penalty_fuel` | 24,549.789516 |
| `penalty_health` | 0.637375 |
| `penalty_unserved` | 7,996.523392 |
| `penalty_waste` | 1,033.481785 |

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
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e3-high-fuel-low-battery-use.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E3/mpc/2026-03-13_00-00-00 --policy mpc
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
