# 72-hour islanded telemetry replay — PyPSA MPC (rolling-horizon optimal dispatch) controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e3_high_fuel_low_battery_use` |
| Controller | `mpc` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-19 00:00:00 to 2026-03-22 00:00:00 |
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
| Measured load energy | 7,290.300 | kWh | measured input |
| Served load energy | 6,629.629 | kWh | model output |
| Load served | 90.938 | % | headline reliability |
| Unserved energy | 660.671 | kWh | headline failure metric |
| Blackout/brownout duration | 22.500 | h | 15-min intervals with unserved load |
| Peak unserved power | 136.589 | kW | maximum interval |
| Measured PV available | 1,949.456 | kWh | negative standby clipped to zero |
| PV used | 1,612.487 | kWh | before source attribution |
| PV curtailed/spilled | 336.969 | kWh | PV-only waste definition |
| Diesel generation | 5,347.410 | kWh | assumed generator model |
| Diesel load-serving energy | 4,487.213 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 380.755 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 92.880 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 55.750 | h | realized on-state |
| Diesel starts | 6.000 | count | realized starts |
| Diesel carbon proxy | 3,743.187 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 488.530 | kWh | AC-side terminal energy |
| Battery charge from PV | 9.088 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 479.442 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 539.018 | kWh | AC-side terminal energy |
| Battery throughput | 1,027.547 | kWh | charge + discharge |
| Final SOC | 31.492 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.041 | percentage points | placeholder degradation law |
| Excess generation | 380.755 | kWh | non-load-serving generation |
| Dump-load energy | 380.755 | kWh | explicit sink for non-exportable surplus |
| Network losses | 79.403 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-19 | 2,697.800 | 81.124 | 509.230 | 278.516 | 1,962.568 | 347.000 | 1,373.798 | 47.455 |
| 2 | 2026-03-20 | 2,397.700 | 98.581 | 34.032 | 1,002.947 | 1,575.051 | 290.051 | 1,102.536 | 73.054 |
| 3 | 2026-03-21 | 2,194.800 | 94.651 | 117.409 | 331.023 | 1,809.791 | 390.497 | 1,266.853 | 31.492 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 14,972.747069 |
| `penalty_constraint` | 18.507842 |
| `penalty_excess` | 380.754840 |
| `penalty_fuel` | 30,251.566403 |
| `penalty_health` | 0.411704 |
| `penalty_unserved` | 13,213.410824 |
| `penalty_waste` | 336.968917 |

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
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e3-high-fuel-low-battery-use.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E3/mpc/2026-03-19_00-00-00 --policy mpc
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
