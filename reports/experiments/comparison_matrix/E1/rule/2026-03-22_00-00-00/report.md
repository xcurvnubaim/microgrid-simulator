# 72-hour islanded telemetry replay — rule controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e1_high_diesel_fuel` |
| Controller | `rule` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-22 00:00:00 to 2026-03-25 00:00:00 |
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
| Measured load energy | 7,517.000 | kWh | measured input |
| Served load energy | 7,309.386 | kWh | model output |
| Load served | 97.238 | % | headline reliability |
| Unserved energy | 207.614 | kWh | headline failure metric |
| Blackout/brownout duration | 10.500 | h | 15-min intervals with unserved load |
| Peak unserved power | 155.600 | kW | maximum interval |
| Measured PV available | 1,772.531 | kWh | negative standby clipped to zero |
| PV used | 1,274.626 | kWh | before source attribution |
| PV curtailed/spilled | 497.906 | kWh | PV-only waste definition |
| Diesel generation | 7,026.222 | kWh | assumed generator model |
| Diesel load-serving energy | 6,064.380 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 956.808 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 86.382 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 68.250 | h | realized on-state |
| Diesel starts | 5.000 | count | realized starts |
| Diesel carbon proxy | 4,918.356 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 34.653 | kWh | AC-side terminal energy |
| Battery charge from PV | 29.619 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 5.034 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 0.000 | kWh | AC-side terminal energy |
| Battery throughput | 34.653 | kWh | charge + discharge |
| Final SOC | 56.653 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.001 | percentage points | placeholder degradation law |
| Excess generation | 956.808 | kWh | non-load-serving generation |
| Dump-load energy | 956.808 | kWh | explicit sink for non-exportable surplus |
| Network losses | 95.152 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-22 | 2,055.200 | 97.216 | 57.216 | 103.734 | 2,322.359 | 23.050 | 1,625.651 | 54.426 |
| 2 | 2026-03-23 | 2,522.400 | 97.198 | 70.676 | 716.034 | 2,028.600 | 6.569 | 1,420.020 | 55.687 |
| 3 | 2026-03-24 | 2,939.400 | 97.288 | 79.722 | 454.857 | 2,675.263 | 5.034 | 1,872.684 | 56.653 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 19,673.422438 |
| `penalty_constraint` | 6.653407 |
| `penalty_excess` | 956.808427 |
| `penalty_fuel` | 37,621.867037 |
| `penalty_health` | 0.069306 |
| `penalty_unserved` | 4,152.270602 |
| `penalty_waste` | 497.905575 |

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
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E1/rule/2026-03-22_00-00-00 --policy rule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
