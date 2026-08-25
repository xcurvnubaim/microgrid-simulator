# 72-hour islanded telemetry replay — rule controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e3_high_fuel_low_battery_use` |
| Controller | `rule` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-25 00:00:00 to 2026-03-28 00:00:00 |
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
| Measured load energy | 7,938.400 | kWh | measured input |
| Served load energy | 7,760.333 | kWh | model output |
| Load served | 97.757 | % | headline reliability |
| Unserved energy | 178.067 | kWh | headline failure metric |
| Blackout/brownout duration | 13.250 | h | 15-min intervals with unserved load |
| Peak unserved power | 102.964 | kW | maximum interval |
| Measured PV available | 1,484.190 | kWh | negative standby clipped to zero |
| PV used | 1,118.400 | kWh | before source attribution |
| PV curtailed/spilled | 365.790 | kWh | PV-only waste definition |
| Diesel generation | 7,474.970 | kWh | assumed generator model |
| Diesel load-serving energy | 6,650.234 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 824.736 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 88.967 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 71.000 | h | realized on-state |
| Diesel starts | 2.000 | count | realized starts |
| Diesel carbon proxy | 5,232.479 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 8.302 | kWh | AC-side terminal energy |
| Battery charge from PV | 8.302 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 0.000 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 0.000 | kWh | AC-side terminal energy |
| Battery throughput | 8.302 | kWh | charge + discharge |
| Final SOC | 51.594 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.000 | percentage points | placeholder degradation law |
| Excess generation | 824.736 | kWh | non-load-serving generation |
| Dump-load energy | 824.736 | kWh | explicit sink for non-exportable surplus |
| Network losses | 101.116 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-25 | 2,553.700 | 97.791 | 56.405 | 627.634 | 2,129.789 | 8.302 | 1,490.852 | 51.594 |
| 2 | 2026-03-26 | 2,885.200 | 97.248 | 79.398 | 422.283 | 2,665.357 | 0.000 | 1,865.750 | 51.594 |
| 3 | 2026-03-27 | 2,499.500 | 98.309 | 42.264 | 68.483 | 2,679.824 | 0.000 | 1,875.877 | 51.594 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 20,929.916027 |
| `penalty_constraint` | 1.593986 |
| `penalty_excess` | 824.735716 |
| `penalty_fuel` | 37,983.856046 |
| `penalty_health` | 0.003321 |
| `penalty_unserved` | 3,561.348714 |
| `penalty_waste` | 365.790034 |

Constraint-violation events: **0**; interval counts by type: `{}`.

## Parameter evidence

| Quantity | Evidence class | Value/use |
|---|---|---|
| Load and PV | measured | aligned campus telemetry for this exact window |
| PV negative night readings | designed preprocessing | clipped to zero generation availability |
| Battery | assumed/project constraint | 500 kWh, ±250 kW, 96%/96%, SOC 10–95% |
| Diesel | assumed | 400 kW (re-anchored to measured 352.8 kW peak), 80 kW minimum, ramp/up/down limits, 0.70 kg CO2e/kWh |
| Load allocation and Q | assumed | 3:2:2 and Q=0.2P; Q unused by the simple backend |
| Rule controller | designed baseline | diesel-first residual coverage, battery gap/surplus handling |

## Limitations

- This is a historical-telemetry-fed simulation, not measured historical dispatch.
- The simple backend is lossless and reports flat 1 pu voltage; it does not validate feeder voltage, losses, or line loading.
- Battery charge is source-resolved into PV/grid/diesel shares by a fixed PV-then-diesel-then-grid merit order; the split is an accounting attribution over the lossless single-bus balance, not a metered physical routing.
- `excess_generation_kwh` exposes supply above served load and battery charging; non-exportable surplus is routed to the explicit `dump_load_kwh` sink.
- AC network losses and any numerical reference-bus balance are reported separately and are never relabeled as waste.
- Battery degradation is a placeholder throughput/SOC-stress proxy, not a calibrated life model.
- Equipment and feeder assumptions remain subject to the provenance blockers; results establish a controller baseline, not physical-campus validation.

## Reproduction

```bash
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e3-high-fuel-low-battery-use.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E3/rule/2026-03-25_00-00-00 --policy rule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
