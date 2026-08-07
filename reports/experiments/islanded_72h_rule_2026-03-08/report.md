# 72-hour islanded telemetry replay — rule controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `islanded_heldout_72h_2026-03-08` |
| Controller | `rule` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-08 00:00:00 to 2026-03-11 00:00:00 |
| Resolution | 15 min; 288 intervals |
| Configuration | `configs/islanded-heldout-72h.yaml` |
| Load source | `/home/xcurv/teep-taiwan/data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv` |
| PV source | `/home/xcurv/teep-taiwan/data/processed/pv_15min_chronos_reconstruction_candidate.csv` |
| Grid | unavailable for all intervals |

The load and PV series are measured and timestamp-aligned. One preceding 15-minute
sample is controller context; it is not included in the energy totals.

## Paper evaluation table

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Measured load energy | 6,549.600 | kWh | measured input |
| Served load energy | 6,478.411 | kWh | model output |
| Load served | 98.913 | % | headline reliability |
| Unserved energy | 71.189 | kWh | headline failure metric |
| Blackout/brownout duration | 6.250 | h | 15-min intervals with unserved load |
| Peak unserved power | 124.464 | kW | maximum interval |
| Measured PV available | 738.972 | kWh | negative standby clipped to zero |
| PV used | 438.689 | kWh | before source attribution |
| PV curtailed/spilled | 300.283 | kWh | PV-only waste definition |
| Diesel generation | 6,342.419 | kWh | assumed generator model |
| Diesel load-serving energy | 6,030.685 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 305.529 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 95.183 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 71.000 | h | realized on-state |
| Diesel starts | 3.000 | count | realized starts |
| Diesel carbon proxy | 4,439.694 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 12.269 | kWh | AC-side terminal energy |
| Battery charge from PV | 6.063 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 6.205 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 15.101 | kWh | AC-side terminal energy |
| Battery throughput | 27.369 | kWh | charge + discharge |
| Final SOC | 49.210 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.001 | percentage points | placeholder degradation law |
| Excess generation | 305.529 | kWh | non-load-serving generation |
| Dump-load energy | 305.529 | kWh | explicit sink for non-exportable surplus |
| Network losses | 68.721 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-08 | 1,503.900 | 99.630 | 5.567 | 312.120 | 1,280.288 | 12.269 | 896.202 | 52.356 |
| 2 | 2026-03-09 | 2,332.900 | 98.503 | 34.922 | 65.474 | 2,349.281 | 8.542 | 1,644.497 | 50.576 |
| 3 | 2026-03-10 | 2,712.800 | 98.868 | 30.700 | 61.094 | 2,712.851 | 6.559 | 1,898.995 | 49.210 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 4,439.693514 |
| `penalty_constraint` | 0.000000 |
| `penalty_excess` | 305.528884 |
| `penalty_health` | 0.109477 |
| `penalty_unserved` | 355.945074 |
| `penalty_waste` | 300.283016 |

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
uv run microgrid-sim replay --config configs/islanded-heldout-72h.yaml --output-dir reports/experiments/islanded_72h_rule_2026-03-08 --policy rule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
