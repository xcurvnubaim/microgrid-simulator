# 72-hour islanded telemetry replay — rule controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e1_high_diesel_fuel` |
| Controller | `rule` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-04 00:00:00 to 2026-03-07 00:00:00 |
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
| Measured load energy | 7,754.900 | kWh | measured input |
| Served load energy | 7,610.419 | kWh | model output |
| Load served | 98.137 | % | headline reliability |
| Unserved energy | 144.481 | kWh | headline failure metric |
| Blackout/brownout duration | 15.000 | h | 15-min intervals with unserved load |
| Peak unserved power | 104.069 | kW | maximum interval |
| Measured PV available | 842.734 | kWh | negative standby clipped to zero |
| PV used | 640.214 | kWh | before source attribution |
| PV curtailed/spilled | 202.519 | kWh | PV-only waste definition |
| Diesel generation | 7,902.062 | kWh | assumed generator model |
| Diesel load-serving energy | 6,970.205 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 931.857 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 88.207 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 72.000 | h | realized on-state |
| Diesel starts | 1.000 | count | realized starts |
| Diesel carbon proxy | 5,531.443 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 0.000 | kWh | AC-side terminal energy |
| Battery charge from PV | 0.000 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 0.000 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 0.000 | kWh | AC-side terminal energy |
| Battery throughput | 0.000 | kWh | charge + discharge |
| Final SOC | 50.000 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.000 | percentage points | placeholder degradation law |
| Excess generation | 931.857 | kWh | non-load-serving generation |
| Dump-load energy | 931.857 | kWh | explicit sink for non-exportable surplus |
| Network losses | 94.802 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-04 | 2,434.800 | 98.198 | 43.870 | 121.613 | 2,588.019 | 0.000 | 1,811.614 | 50.000 |
| 2 | 2026-03-05 | 2,778.400 | 97.814 | 60.749 | 355.602 | 2,652.917 | 0.000 | 1,857.042 | 50.000 |
| 3 | 2026-03-06 | 2,541.700 | 98.432 | 39.862 | 162.999 | 2,661.126 | 0.000 | 1,862.788 | 50.000 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 22,125.773228 |
| `penalty_constraint` | 0.000000 |
| `penalty_excess` | 931.857143 |
| `penalty_fuel` | 39,081.896963 |
| `penalty_health` | 0.000000 |
| `penalty_unserved` | 2,889.619810 |
| `penalty_waste` | 202.519298 |

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
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E1/rule/2026-03-04_00-00-00 --policy rule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
