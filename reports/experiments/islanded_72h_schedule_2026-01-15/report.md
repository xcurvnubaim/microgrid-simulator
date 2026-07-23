# 72-hour islanded telemetry replay — manual fixed-schedule diesel controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `islanded_72h_2026-01-15` |
| Controller | `schedule` |
| Backend | `simple` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-01-15 00:00:00 to 2026-01-18 00:00:00 |
| Resolution | 15 min; 288 intervals |
| Configuration | `configs/islanded-baseline-72h.yaml` |
| Load source | `/home/xcurv/teep-taiwan/data/Total Load (net load)_2025-12-01_2026-04-30.xlsx` |
| PV source | `/home/xcurv/teep-taiwan/data/PV_2025-12-01_2026-03-31.xlsx` |
| Grid | unavailable for all intervals |

The load and PV series are measured and timestamp-aligned. One preceding 15-minute
sample is controller context; it is not included in the energy totals.

## Diesel timetable (designed scenario assumption)

| Hours | Diesel level |
|---|---|
| 0:00 - 9:00 | `max` |
| 9:00 - 14:00 | `min` |
| 14:00 - 24:00 | `max` |

Hours not covered by a segment mean diesel off. Levels: `max` = nameplate,
`min` = minimum stable load, numeric = explicit kW setpoint.

## Paper evaluation table

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Measured load energy | 4,553.500 | kWh | measured input |
| Served load energy | 4,553.146 | kWh | model output |
| Load served | 99.992 | % | headline reliability |
| Unserved energy | 0.354 | kWh | headline failure metric |
| Blackout/brownout duration | 0.750 | h | 15-min intervals with unserved load |
| Peak unserved power | 0.806 | kW | maximum interval |
| Measured PV available | 1,903.514 | kWh | negative standby clipped to zero |
| PV used | 637.390 | kWh | before source attribution |
| PV curtailed/spilled | 1,266.125 | kWh | PV-only waste definition |
| Diesel generation | 9,221.312 | kWh | assumed generator model |
| Diesel runtime | 72.000 | h | realized on-state |
| Diesel starts | 1.000 | count | realized starts |
| Diesel carbon proxy | 6,454.919 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 234.636 | kWh | AC-side terminal energy |
| Battery charge from PV | 28.206 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 206.430 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 0.250 | kWh | AC-side terminal energy |
| Battery throughput | 234.886 | kWh | charge + discharge |
| Final SOC | 95.000 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.010 | percentage points | placeholder degradation law |
| Excess-generation proxy | 5,071.171 | kWh | not an explicit modeled sink |
| Maximum balance residual | 0.000 | kW | after named excess proxy |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-01-15 | 1,649.700 | 100.000 | 0.000 | 363.566 | 3,071.312 | 194.495 | 2,149.919 | 87.344 |
| 2 | 2026-01-16 | 1,537.200 | 99.977 | 0.354 | 149.633 | 3,075.000 | 0.250 | 2,152.500 | 87.292 |
| 3 | 2026-01-17 | 1,366.600 | 100.000 | 0.000 | 124.191 | 3,075.000 | 40.141 | 2,152.500 | 95.000 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 6,454.918750 |
| `penalty_constraint` | 0.000000 |
| `penalty_health` | 0.992588 |
| `penalty_unserved` | 1.770417 |
| `penalty_waste` | 1,266.124559 |

Constraint-violation events: **3**; interval counts by type: `{"unserved": 3}`.

## Parameter evidence

| Quantity | Evidence class | Value/use |
|---|---|---|
| Load and PV | measured | aligned campus telemetry for this exact window |
| PV negative night readings | designed preprocessing | clipped to zero generation availability |
| Battery | assumed/project constraint | 500 kWh, ±250 kW, 96%/96%, SOC 10–95% |
| Diesel | assumed | 150 kW, 45 kW minimum, ramp/up/down limits, 0.70 kg CO2e/kWh |
| Load allocation and Q | assumed | 3:2:2 and Q=0.2P; Q unused by the simple backend |
| Manual-schedule controller | designed baseline | clock-driven, not residual-reactive: plays back the per-scenario `diesel_schedule` timetable (hour-of-day segments at max / min / off / explicit kW); battery covers any remaining gap beyond the scheduled diesel setpoint |

## Limitations

- This is a historical-telemetry-fed simulation, not measured historical dispatch.
- The simple backend is lossless and reports flat 1 pu voltage; it does not validate feeder voltage, losses, or line loading.
- Battery charge is source-resolved into PV/grid/diesel shares by a fixed PV-then-diesel-then-grid merit order; the split is an accounting attribution over the lossless single-bus balance, not a metered physical routing.
- `excess_generation_kwh` exposes supply above served load and battery charging, but no physical dump load is modeled.
- Battery degradation is a placeholder throughput/SOC-stress proxy, not a calibrated life model.
- Equipment and feeder assumptions remain subject to the provenance blockers; results establish a controller baseline, not physical-campus validation.
- The manual-schedule controller deliberately overgenerates diesel outside its low-risk window regardless of actual residual; the surplus has no PV/battery sink commanded for it and shows up as `excess_generation_kwh` plus extra `diesel_kwh`/`carbon_kg` rather than being curtailed or stored.

## Reproduction

```bash
uv run microgrid-sim replay --config configs/islanded-baseline-72h.yaml --output-dir reports/experiments/islanded_72h_schedule_2026-01-15 --policy schedule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
