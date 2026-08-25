# 72-hour islanded telemetry replay — rule controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e1_high_diesel_fuel` |
| Controller | `rule` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-19 00:00:00 to 2026-03-22 00:00:00 |
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
| Measured load energy | 7,290.300 | kWh | measured input |
| Served load energy | 7,165.762 | kWh | model output |
| Load served | 98.292 | % | headline reliability |
| Unserved energy | 124.538 | kWh | headline failure metric |
| Blackout/brownout duration | 10.250 | h | 15-min intervals with unserved load |
| Peak unserved power | 108.364 | kW | maximum interval |
| Measured PV available | 1,949.456 | kWh | negative standby clipped to zero |
| PV used | 1,549.012 | kWh | before source attribution |
| PV curtailed/spilled | 400.444 | kWh | PV-only waste definition |
| Diesel generation | 6,583.125 | kWh | assumed generator model |
| Diesel load-serving energy | 5,686.388 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 896.737 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 86.378 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 68.250 | h | realized on-state |
| Diesel starts | 2.000 | count | realized starts |
| Diesel carbon proxy | 4,608.188 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 69.638 | kWh | AC-side terminal energy |
| Battery charge from PV | 69.638 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 0.000 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 0.000 | kWh | AC-side terminal energy |
| Battery throughput | 69.638 | kWh | charge + discharge |
| Final SOC | 63.371 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.003 | percentage points | placeholder degradation law |
| Excess generation | 896.737 | kWh | non-load-serving generation |
| Dump-load energy | 896.737 | kWh | explicit sink for non-exportable surplus |
| Network losses | 89.813 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-19 | 2,697.800 | 97.904 | 56.557 | 239.595 | 2,683.028 | 0.000 | 1,878.120 | 50.000 |
| 2 | 2026-03-20 | 2,397.700 | 99.153 | 20.309 | 1,033.905 | 1,682.577 | 69.638 | 1,177.804 | 63.371 |
| 3 | 2026-03-21 | 2,194.800 | 97.828 | 47.671 | 275.512 | 2,217.520 | 0.000 | 1,552.264 | 63.371 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 18,432.750262 |
| `penalty_constraint` | 13.370665 |
| `penalty_excess` | 896.737102 |
| `penalty_fuel` | 33,299.000450 |
| `penalty_health` | 0.139276 |
| `penalty_unserved` | 2,490.757582 |
| `penalty_waste` | 400.443575 |

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
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E1/rule/2026-03-19_00-00-00 --policy rule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
