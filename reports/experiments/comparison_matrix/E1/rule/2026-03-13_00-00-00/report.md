# 72-hour islanded telemetry replay — rule controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e1_high_diesel_fuel` |
| Controller | `rule` |
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
| Served load energy | 6,440.137 | kWh | model output |
| Load served | 99.004 | % | headline reliability |
| Unserved energy | 64.763 | kWh | headline failure metric |
| Blackout/brownout duration | 7.500 | h | 15-min intervals with unserved load |
| Peak unserved power | 46.156 | kW | maximum interval |
| Measured PV available | 3,955.120 | kWh | negative standby clipped to zero |
| PV used | 2,925.531 | kWh | before source attribution |
| PV curtailed/spilled | 1,029.589 | kWh | PV-only waste definition |
| Diesel generation | 4,608.019 | kWh | assumed generator model |
| Diesel load-serving energy | 3,755.579 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 843.090 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 81.704 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 53.750 | h | realized on-state |
| Diesel starts | 7.000 | count | realized starts |
| Diesel carbon proxy | 3,225.613 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 438.094 | kWh | AC-side terminal energy |
| Battery charge from PV | 428.744 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 9.350 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 187.771 | kWh | AC-side terminal energy |
| Battery throughput | 625.865 | kWh | charge + discharge |
| Final SOC | 95.000 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.027 | percentage points | placeholder degradation law |
| Excess generation | 843.090 | kWh | non-load-serving generation |
| Dump-load energy | 843.090 | kWh | explicit sink for non-exportable surplus |
| Network losses | 89.046 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-13 | 2,270.800 | 98.905 | 24.871 | 994.215 | 1,662.220 | 123.072 | 1,163.554 | 73.630 |
| 2 | 2026-03-14 | 2,310.200 | 99.274 | 16.770 | 1,027.403 | 1,675.778 | 111.292 | 1,173.045 | 95.000 |
| 3 | 2026-03-15 | 1,923.900 | 98.798 | 23.122 | 903.913 | 1,270.021 | 391.501 | 889.015 | 95.000 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 12,902.452739 |
| `penalty_constraint` | 45.000000 |
| `penalty_excess` | 843.090125 |
| `penalty_fuel` | 26,358.490410 |
| `penalty_health` | 1.329401 |
| `penalty_unserved` | 1,295.256537 |
| `penalty_waste` | 1,029.589269 |

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
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E1/rule/2026-03-13_00-00-00 --policy rule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
