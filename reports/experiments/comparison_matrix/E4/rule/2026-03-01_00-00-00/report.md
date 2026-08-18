# 72-hour islanded telemetry replay — rule controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e4_high_pv_waste` |
| Controller | `rule` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-01 00:00:00 to 2026-03-04 00:00:00 |
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
| Measured load energy | 6,602.800 | kWh | measured input |
| Served load energy | 6,461.315 | kWh | model output |
| Load served | 97.857 | % | headline reliability |
| Unserved energy | 141.485 | kWh | headline failure metric |
| Blackout/brownout duration | 12.750 | h | 15-min intervals with unserved load |
| Peak unserved power | 131.674 | kW | maximum interval |
| Measured PV available | 1,423.957 | kWh | negative standby clipped to zero |
| PV used | 1,009.599 | kWh | before source attribution |
| PV curtailed/spilled | 414.357 | kWh | PV-only waste definition |
| Diesel generation | 6,351.166 | kWh | assumed generator model |
| Diesel load-serving energy | 5,397.520 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 953.646 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 84.985 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 61.250 | h | realized on-state |
| Diesel starts | 5.000 | count | realized starts |
| Diesel carbon proxy | 4,445.816 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 260.136 | kWh | AC-side terminal energy |
| Battery charge from PV | 260.136 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 0.000 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 314.331 | kWh | AC-side terminal energy |
| Battery throughput | 574.468 | kWh | charge + discharge |
| Final SOC | 34.453 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.024 | percentage points | placeholder degradation law |
| Excess generation | 953.646 | kWh | non-load-serving generation |
| Dump-load energy | 953.646 | kWh | explicit sink for non-exportable surplus |
| Network losses | 77.331 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-01 | 1,300.900 | 99.672 | 4.264 | 536.717 | 1,459.006 | 283.891 | 1,021.304 | 95.000 |
| 2 | 2026-03-02 | 2,529.200 | 96.670 | 84.213 | 384.303 | 1,955.393 | 290.576 | 1,368.775 | 34.453 |
| 3 | 2026-03-03 | 2,772.700 | 98.088 | 53.009 | 88.579 | 2,936.767 | 0.000 | 2,055.737 | 34.453 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 17,783.265436 |
| `penalty_constraint` | 15.547450 |
| `penalty_excess` | 953.646091 |
| `penalty_fuel` | 14,581.865964 |
| `penalty_health` | 1.194274 |
| `penalty_unserved` | 2,829.700300 |
| `penalty_waste` | 828.714585 |

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
uv run microgrid-sim replay --config /home/xcurv/teep-taiwan/microgrid-simulator/configs/economic-e4-high-pv-waste.yaml --output-dir /home/xcurv/teep-taiwan/microgrid-simulator/reports/experiments/comparison_matrix/E4/rule/2026-03-01_00-00-00 --policy rule
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
