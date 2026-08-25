# 72-hour islanded telemetry replay — PyPSA MPC (rolling-horizon optimal dispatch) controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `economic_e3_high_fuel_low_battery_use` |
| Controller | `mpc` |
| Backend | `pandapower` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-03-10 00:00:00 to 2026-03-13 00:00:00 |
| Resolution | 15 min; 288 intervals |
| Configuration | `/home/ma012/projects/microgrid-simulator/configs/economic-e3-high-fuel-low-battery-use.yaml` |
| Load source | `../data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv` |
| PV source | `../data/processed/pv_15min_chronos_reconstruction_candidate.csv` |
| Grid | unavailable for all intervals |

The load and PV series are measured and timestamp-aligned. One preceding 15-minute
sample is controller context; it is not included in the energy totals.

## Paper evaluation table

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Measured load energy | 7,700.100 | kWh | measured input |
| Served load energy | 6,919.467 | kWh | model output |
| Load served | 89.862 | % | headline reliability |
| Unserved energy | 780.633 | kWh | headline failure metric |
| Blackout/brownout duration | 30.000 | h | 15-min intervals with unserved load |
| Peak unserved power | 102.860 | kW | maximum interval |
| Measured PV available | 1,859.584 | kWh | negative standby clipped to zero |
| PV used | 1,474.589 | kWh | before source attribution |
| PV curtailed/spilled | 384.995 | kWh | PV-only waste definition |
| Diesel generation | 5,427.657 | kWh | assumed generator model |
| Diesel load-serving energy | 4,811.775 | kWh | PV-first accounting attribution |
| Diesel overgeneration | 86.363 | kWh | diesel share routed to the dump-load path |
| Diesel useful output | 98.409 | % | generation not classified as dumped diesel excess |
| Diesel runtime | 53.250 | h | realized on-state |
| Diesel starts | 7.000 | count | realized starts |
| Diesel carbon proxy | 3,799.360 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 556.753 | kWh | AC-side terminal energy |
| Battery charge from PV | 27.234 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 529.519 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 660.337 | kWh | AC-side terminal energy |
| Battery throughput | 1,217.091 | kWh | charge + discharge |
| Final SOC | 19.322 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.050 | percentage points | placeholder degradation law |
| Excess generation | 86.363 | kWh | non-load-serving generation |
| Dump-load energy | 86.363 | kWh | explicit sink for non-exportable surplus |
| Network losses | 84.456 | kWh | separate from excess generation |
| Maximum balance residual | 0.000 | kW | after dump load, losses, and reference balance |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-03-10 | 2,712.800 | 90.097 | 268.649 | 66.904 | 2,370.505 | 318.492 | 1,659.354 | 45.646 |
| 2 | 2026-03-11 | 2,362.600 | 98.014 | 46.915 | 779.264 | 1,545.401 | 476.319 | 1,081.781 | 32.387 |
| 3 | 2026-03-12 | 2,624.700 | 82.281 | 465.070 | 628.421 | 1,511.750 | 422.280 | 1,058.225 | 19.322 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 15,197.439785 |
| `penalty_constraint` | 30.677861 |
| `penalty_excess` | 86.362619 |
| `penalty_fuel` | 30,688.753917 |
| `penalty_health` | 0.496641 |
| `penalty_unserved` | 15,612.663243 |
| `penalty_waste` | 384.995105 |

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
uv run microgrid-sim replay --config /home/ma012/projects/microgrid-simulator/configs/economic-e3-high-fuel-low-battery-use.yaml --output-dir /home/ma012/projects/microgrid-simulator/reports/experiments/comparison_matrix/E3/mpc/2026-03-10_00-00-00 --policy mpc
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
