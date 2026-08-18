# 72-hour islanded telemetry replay — rule controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e4_high_pv_waste` |
| Controller | `rule` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-16 00:00:00 to 2026-03-19 00:00:00 |
| Resolution | 15 min; 288 intervals |
| Configuration | `/home/xcurv/teep-taiwan/microgrid-simulator/configs/economic-e4-high-pv-waste.yaml` |
| Load source | `/home/xcurv/teep-taiwan/data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv` |
| PV source | `/home/xcurv/teep-taiwan/data/processed/pv_15min_chronos_reconstruction_candidate.csv` |
| Grid | unavailable for all intervals |

The load and PV series are measured and timestamp-aligned. One preceding 15-minute
sample is controller context; it is not included in the energy totals.

## Paper evaluation table

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Measured load energy | 7,728.900 | kWh | measured input |
| Served load energy | 7,562.999 | kWh | model output |
| Load served | 97.853 | % | headline reliability |
| Unserved energy | 165.901 | kWh | headline failure metric |
| Blackout/brownout duration | 10.750 | h | 15-min intervals with unserved load |
| Peak unserved power | 97.011 | kW | maximum interval |
| Measured PV available | 3,363.202 | kWh | negative standby clipped to zero |
| PV used | 2,777.025 | kWh | before source attribution |
| PV curtailed/spilled | 586.177 | kWh | PV-only waste definition |
| Diesel generation | 5,461.240 | kWh | assumed generator model |
| Diesel load-serving energy | 4,748.302 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 712.133 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 86.960 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 59.500 | h | realized on-state |
| Diesel starts | 10.000 | count | realized starts |
| Diesel carbon proxy | 3,822.868 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 178.085 | kWh | AC-side terminal energy |
| Battery charge from PV | 177.280 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 0.805 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 214.952 | kWh | AC-side terminal energy |
| Battery throughput | 393.038 | kWh | charge + discharge |
| Final SOC | 39.411 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.016 | percentage points | placeholder degradation law |
| Excess generation | 712.133 | kWh | non-load-serving generation |
| Dump-load energy | 712.133 | kWh | explicit sink for non-exportable surplus |
| Network losses | 110.989 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-16 | 2,400.100 | 98.170 | 43.931 | 1,032.445 | 1,561.480 | 193.825 | 1,093.036 | 48.784 |
| 2 | 2026-03-17 | 2,754.000 | 97.295 | 74.497 | 798.465 | 2,112.501 | 100.120 | 1,478.751 | 45.681 |
| 3 | 2026-03-18 | 2,574.800 | 98.156 | 47.473 | 946.115 | 1,787.259 | 99.093 | 1,251.081 | 39.411 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 15,291.471579 |
| `penalty_constraint` | 10.589354 |
| `penalty_excess` | 712.133003 |
| `penalty_fuel` | 15,501.983759 |
| `penalty_health` | 0.786075 |
| `penalty_unserved` | 3,318.026778 |
| `penalty_waste` | 1,172.353582 |

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
uv run microgrid-sim replay --config /home/xcurv/teep-taiwan/microgrid-simulator/configs/economic-e4-high-pv-waste.yaml --output-dir /home/xcurv/teep-taiwan/microgrid-simulator/reports/experiments/comparison_matrix/E4/rule/2026-03-16_00-00-00 --policy rule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
