# 72-hour islanded telemetry replay — PyPSA MPC (rolling-horizon optimal dispatch) controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e1_high_diesel_fuel` |
| Controller | `mpc` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-01 00:00:00 to 2026-03-04 00:00:00 |
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
| Measured load energy | 6,602.800 | kWh | measured input |
| Served load energy | 5,449.668 | kWh | model output |
| Load served | 82.536 | % | headline reliability |
| Unserved energy | 1,153.132 | kWh | headline failure metric |
| Blackout/brownout duration | 29.500 | h | 15-min intervals with unserved load |
| Peak unserved power | 146.361 | kW | maximum interval |
| Measured PV available | 1,423.957 | kWh | negative standby clipped to zero |
| PV used | 992.961 | kWh | before source attribution |
| PV curtailed/spilled | 430.996 | kWh | PV-only waste definition |
| Diesel generation | 4,533.396 | kWh | assumed generator model |
| Diesel load-serving energy | 3,808.265 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 130.907 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 97.112 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 47.750 | h | realized on-state |
| Diesel starts | 7.000 | count | realized starts |
| Diesel carbon proxy | 3,173.377 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 617.395 | kWh | AC-side terminal energy |
| Battery charge from PV | 23.172 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 594.224 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 671.614 | kWh | AC-side terminal energy |
| Battery throughput | 1,289.009 | kWh | charge + discharge |
| Final SOC | 28.611 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.052 | percentage points | placeholder degradation law |
| Excess generation | 130.907 | kWh | non-load-serving generation |
| Dump-load energy | 130.907 | kWh | explicit sink for non-exportable surplus |
| Network losses | 54.798 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-01 | 1,300.900 | 98.634 | 17.771 | 347.462 | 1,060.761 | 553.605 | 742.532 | 62.761 |
| 2 | 2026-03-02 | 2,529.200 | 79.057 | 529.686 | 525.838 | 1,410.003 | 434.865 | 987.002 | 39.737 |
| 3 | 2026-03-03 | 2,772.700 | 78.156 | 605.675 | 119.661 | 2,062.632 | 300.539 | 1,443.843 | 28.611 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 12,693.508173 |
| `penalty_constraint` | 21.388667 |
| `penalty_excess` | 130.907139 |
| `penalty_fuel` | 26,560.299726 |
| `penalty_health` | 2.590398 |
| `penalty_unserved` | 23,062.641843 |
| `penalty_waste` | 430.996247 |

Constraint-violation events: **0**; interval counts by type: `{}`.

## Parameter evidence

| Quantity | Evidence class | Value/use |
|---|---|---|
| Load and PV | measured | aligned campus telemetry for this exact window |
| PV negative night readings | designed preprocessing | clipped to zero generation availability |
| Battery | assumed/project constraint | 500 kWh, ±250 kW, 96%/96%, SOC 10–95% |
| Diesel | assumed | 400 kW (re-anchored to measured 352.8 kW peak), 80 kW minimum, ramp/up/down limits, 0.70 kg CO2e/kWh |
| Load allocation and Q | assumed | 3:2:2 and Q=0.2P; Q unused by the simple backend |
| MPC controller | designed baseline | PyPSA rolling-horizon MILP, perfect-foresight forecast from the driving backend, replanned every `backend.rolling_horizon_hours` |

## Limitations

- This is a historical-telemetry-fed simulation, not measured historical dispatch.
- The simple backend is lossless and reports flat 1 pu voltage; it does not validate feeder voltage, losses, or line loading.
- Battery charge is source-resolved into PV/grid/diesel shares by a fixed PV-then-diesel-then-grid merit order; the split is an accounting attribution over the lossless single-bus balance, not a metered physical routing.
- `excess_generation_kwh` exposes supply above served load and battery charging; non-exportable surplus is routed to the explicit `dump_load_kwh` sink.
- AC network losses and any numerical reference-bus balance are reported separately and are never relabeled as waste.
- Battery degradation is a placeholder throughput/SOC-stress proxy, not a calibrated life model.
- Equipment and feeder assumptions remain subject to the provenance blockers; results establish a controller baseline, not physical-campus validation.
- The MPC controller has perfect foresight of load/PV over its optimization horizon (no forecast error is modeled); near the end of the 72 h episode the lookahead clamps to the last measured sample instead of running out of data.

## Reproduction

```bash
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e1-high-diesel-fuel.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E1/mpc/2026-03-01_00-00-00 --policy mpc
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
