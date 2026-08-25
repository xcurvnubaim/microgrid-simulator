# 72-hour islanded telemetry replay — manual fixed-schedule diesel controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e1_high_diesel_fuel` |
| Controller | `schedule` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-04 00:00:00 to 2026-03-07 00:00:00 |
| Resolution | 15 min; 288 intervals |
| Configuration | `/home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml` |
| Load source | `../data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv` |
| PV source | `../data/processed/pv_15min_chronos_reconstruction_candidate.csv` |
| Grid | unavailable for all intervals |

The load and PV series are measured and timestamp-aligned. One preceding 15-minute
sample is controller context; it is not included in the energy totals.

## Diesel timetable (designed scenario assumption)

| Hours | Diesel level |
|---|---|
| 0:00 - 7:00 | `80.0` |
| 7:00 - 9:00 | `220.0` |
| 9:00 - 21:00 | `200.0` |
| 21:00 - 24:00 | `150.0` |

Hours not covered by a segment mean diesel off. Levels: `max` = nameplate,
`min` = minimum stable load, numeric = explicit kW setpoint.

## Paper evaluation table

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Measured load energy | 7,754.900 | kWh | measured input |
| Served load energy | 7,754.900 | kWh | model output |
| Load served | 100.000 | % | headline reliability |
| Unserved energy | 0.000 | kWh | headline failure metric |
| Blackout/brownout duration | 0.000 | h | 15-min intervals with unserved load |
| Peak unserved power | 0.000 | kW | maximum interval |
| Measured PV available | 842.734 | kWh | negative standby clipped to zero |
| PV used | 61.244 | kWh | before source attribution |
| PV curtailed/spilled | 781.490 | kWh | PV-only waste definition |
| Diesel generation | 11,547.574 | kWh | assumed generator model |
| Diesel load-serving energy | 7,693.656 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 3,619.554 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 68.655 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 72.000 | h | realized on-state |
| Diesel starts | 1.000 | count | realized starts |
| Diesel carbon proxy | 8,083.302 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 234.365 | kWh | AC-side terminal energy |
| Battery charge from PV | 0.000 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 234.365 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 0.000 | kWh | AC-side terminal energy |
| Battery throughput | 234.365 | kWh | charge + discharge |
| Final SOC | 95.000 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.010 | percentage points | placeholder degradation law |
| Excess generation | 3,619.554 | kWh | non-load-serving generation |
| Dump-load energy | 3,619.554 | kWh | explicit sink for non-exportable surplus |
| Network losses | 98.231 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-04 | 2,434.800 | 100.000 | 0.000 | 61.244 | 3,848.799 | 234.365 | 2,694.159 | 95.000 |
| 2 | 2026-03-05 | 2,778.400 | 100.000 | 0.000 | 0.000 | 3,849.388 | 0.000 | 2,694.571 | 95.000 |
| 3 | 2026-03-06 | 2,541.700 | 100.000 | 0.000 | 0.000 | 3,849.388 | 0.000 | 2,694.571 | 95.000 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 32,333.208422 |
| `penalty_constraint` | 45.000000 |
| `penalty_excess` | 3,619.553604 |
| `penalty_fuel` | 56,580.357294 |
| `penalty_health` | 0.496604 |
| `penalty_unserved` | 0.000000 |
| `penalty_waste` | 781.489872 |

Constraint-violation events: **0**; interval counts by type: `{}`.

## Parameter evidence

| Quantity | Evidence class | Value/use |
|---|---|---|
| Load and PV | measured | aligned campus telemetry for this exact window |
| PV negative night readings | designed preprocessing | clipped to zero generation availability |
| Battery | assumed/project constraint | 500 kWh, ±250 kW, 96%/96%, SOC 10–95% |
| Diesel | assumed | 400 kW (re-anchored to measured 352.8 kW peak), 80 kW minimum, ramp/up/down limits, 0.70 kg CO2e/kWh |
| Load allocation and Q | assumed | 3:2:2 and Q=0.2P; Q unused by the simple backend |
| Manual-schedule controller | designed baseline | clock-driven, not residual-reactive: plays back the per-scenario `diesel_schedule` timetable (hour-of-day segments at max / min / off / explicit kW); battery covers any remaining gap beyond the scheduled diesel setpoint |

## Limitations

- This is a historical-telemetry-fed simulation, not measured historical dispatch.
- The simple backend is lossless and reports flat 1 pu voltage; it does not validate feeder voltage, losses, or line loading.
- Battery charge is source-resolved into PV/grid/diesel shares by a fixed PV-then-diesel-then-grid merit order; the split is an accounting attribution over the lossless single-bus balance, not a metered physical routing.
- `excess_generation_kwh` exposes supply above served load and battery charging; non-exportable surplus is routed to the explicit `dump_load_kwh` sink.
- AC network losses and any numerical reference-bus balance are reported separately and are never relabeled as waste.
- Battery degradation is a placeholder throughput/SOC-stress proxy, not a calibrated life model.
- Equipment and feeder assumptions remain subject to the provenance blockers; results establish a controller baseline, not physical-campus validation.
- The manual-schedule controller deliberately overgenerates diesel outside its low-risk window regardless of actual residual; the surplus has no PV/battery sink commanded for it and is routed to the explicit `dump_load_kwh` sink, with extra `diesel_kwh`/`carbon_kg` rather than being curtailed or stored.

## Reproduction

```bash
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E1/schedule/2026-03-04_00-00-00 --policy schedule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
