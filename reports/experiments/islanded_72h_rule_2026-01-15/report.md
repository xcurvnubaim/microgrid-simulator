# 72-hour islanded telemetry replay — rule controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `islanded_72h_2026-01-15` |
| Controller | `rule` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-01-15 00:00:00 to 2026-01-18 00:00:00 |
| Resolution | 15 min; 288 intervals |
| Configuration | `configs/islanded-baseline-72h.yaml` |
| Load source | `/home/xcurv/teep-taiwan/data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv` |
| PV source | `/home/xcurv/teep-taiwan/data/processed/pv_15min_chronos_reconstruction_candidate.csv` |
| Grid | unavailable for all intervals |

The load and PV series are measured and timestamp-aligned. One preceding 15-minute
sample is controller context; it is not included in the energy totals.

## Paper evaluation table

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Measured load energy | 4,553.500 | kWh | measured input |
| Served load energy | 4,541.232 | kWh | model output |
| Load served | 99.731 | % | headline reliability |
| Unserved energy | 12.268 | kWh | headline failure metric |
| Blackout/brownout duration | 3.500 | h | 15-min intervals with unserved load |
| Peak unserved power | 13.439 | kW | maximum interval |
| Measured PV available | 1,903.514 | kWh | negative standby clipped to zero |
| PV used | 1,461.794 | kWh | before source attribution |
| PV curtailed/spilled | 441.720 | kWh | PV-only waste definition |
| Diesel generation | 3,616.944 | kWh | assumed generator model |
| Diesel load-serving energy | 3,313.802 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 303.142 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 91.619 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 60.000 | h | realized on-state |
| Diesel starts | 3.000 | count | realized starts |
| Diesel carbon proxy | 2,531.861 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 234.364 | kWh | AC-side terminal energy |
| Battery charge from PV | 234.364 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 0.000 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 0.000 | kWh | AC-side terminal energy |
| Battery throughput | 234.364 | kWh | charge + discharge |
| Final SOC | 95.000 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.010 | percentage points | placeholder degradation law |
| Excess generation | 303.142 | kWh | non-load-serving generation |
| Dump-load energy | 303.142 | kWh | explicit sink for non-exportable surplus |
| Network losses | 34.463 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-01-15 | 1,649.700 | 99.768 | 3.826 | 740.646 | 1,176.811 | 163.516 | 823.767 | 81.396 |
| 2 | 2026-01-16 | 1,537.200 | 99.679 | 4.939 | 227.576 | 1,407.602 | 0.000 | 985.322 | 81.396 |
| 3 | 2026-01-17 | 1,366.600 | 99.744 | 3.503 | 493.572 | 1,032.531 | 70.849 | 722.772 | 95.000 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 2,531.860803 |
| `penalty_constraint` | 45.000000 |
| `penalty_excess` | 303.142073 |
| `penalty_health` | 0.990149 |
| `penalty_unserved` | 61.340479 |
| `penalty_waste` | 441.719989 |

Constraint-violation events: **0**; interval counts by type: `{}`.

## Parameter evidence

| Quantity | Evidence class | Value/use |
|---|---|---|
| Load and PV | measured | aligned campus telemetry for this exact window |
| PV negative night readings | designed preprocessing | clipped to zero generation availability |
| Battery | assumed/project constraint | 500 kWh, ±250 kW, 96%/96%, SOC 10–95% |
| Diesel | assumed | 150 kW, 45 kW minimum, ramp/up/down limits, 0.70 kg CO2e/kWh |
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
uv run microgrid-sim replay --config configs/islanded-baseline-72h.yaml --output-dir reports/experiments/islanded_72h_rule_2026-01-15 --policy rule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
