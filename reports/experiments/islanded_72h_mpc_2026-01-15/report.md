# 72-hour islanded telemetry replay — PyPSA MPC (rolling-horizon optimal dispatch) controller

## Experiment identity

| Field | Value |
|---|---|
| Scenario | `islanded_72h_2026-01-15` |
| Controller | `mpc` |
| Backend | `pypsa` (lossless algebraic scheduling model) |
| Evaluated interval | 2026-01-15 00:00:00 to 2026-01-18 00:00:00 |
| Resolution | 15 min; 288 intervals |
| Configuration | `configs/islanded-baseline-72h.yaml` |
| Load source | `/home/xcurv/teep-taiwan/data/Total Load (net load)_2025-12-01_2026-04-30.xlsx` |
| PV source | `/home/xcurv/teep-taiwan/data/PV_2025-12-01_2026-03-31.xlsx` |
| Grid | unavailable for all intervals |

The load and PV series are measured and timestamp-aligned. One preceding 15-minute
sample is controller context; it is not included in the energy totals.

## Paper evaluation table

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Measured load energy | 4,553.500 | kWh | measured input |
| Served load energy | 4,549.716 | kWh | model output |
| Load served | 99.917 | % | headline reliability |
| Unserved energy | 3.784 | kWh | headline failure metric |
| Blackout/brownout duration | 18.250 | h | 15-min intervals with unserved load |
| Peak unserved power | 2.968 | kW | maximum interval |
| Measured PV available | 1,903.514 | kWh | negative standby clipped to zero |
| PV used | 1,903.400 | kWh | before source attribution |
| PV curtailed/spilled | 0.114 | kWh | PV-only waste definition |
| Diesel generation | 2,490.504 | kWh | assumed generator model |
| Diesel runtime | 45.250 | h | realized on-state |
| Diesel starts | 13.000 | count | realized starts |
| Diesel carbon proxy | 1,743.353 | kg CO2e | 0.70 kg/kWh assumed |
| Battery charge energy | 459.084 | kWh | AC-side terminal energy |
| Battery charge from PV | 448.988 | kWh | surplus-PV share of charging |
| Battery charge from grid | 0.000 | kWh | grid-import share of charging |
| Battery charge from diesel | 10.096 | kWh | surplus-diesel share of charging |
| Battery discharge energy | 615.048 | kWh | AC-side terminal energy |
| Battery throughput | 1,074.133 | kWh | charge + discharge |
| Final SOC | 10.000 | % | assumed 500 kWh battery |
| SoH loss proxy | 0.044 | percentage points | placeholder degradation law |
| Excess-generation proxy | 0.152 | kWh | not an explicit modeled sink |
| Maximum balance residual | 0.000 | kW | after named excess proxy |

## Daily breakdown

| Day | Date | Load kWh | Served % | Unserved kWh | PV used kWh | Diesel kWh | Battery throughput kWh | Carbon kg | Final SOC % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026-01-15 | 1,649.700 | 99.851 | 2.452 | 823.549 | 750.626 | 462.114 | 525.438 | 31.589 |
| 2 | 2026-01-16 | 1,537.200 | 99.959 | 0.623 | 346.034 | 1,095.731 | 104.080 | 767.012 | 11.741 |
| 3 | 2026-01-17 | 1,366.600 | 99.948 | 0.709 | 733.818 | 644.146 | 507.938 | 450.902 | 10.000 |

## Reward and constraint audit

Reward terms below are raw accumulated penalty magnitudes, before their weights
are combined into the reported total reward.

| Term | Accumulated magnitude |
|---|---:|
| `penalty_autonomy` | 0.000000 |
| `penalty_carbon` | 1,743.352535 |
| `penalty_constraint` | 0.000000 |
| `penalty_health` | 4.406911 |
| `penalty_unserved` | 18.919315 |
| `penalty_waste` | 0.113943 |

Constraint-violation events: **73**; interval counts by type: `{"unserved": 73}`.

## Parameter evidence

| Quantity | Evidence class | Value/use |
|---|---|---|
| Load and PV | measured | aligned campus telemetry for this exact window |
| PV negative night readings | designed preprocessing | clipped to zero generation availability |
| Battery | assumed/project constraint | 500 kWh, ±250 kW, 96%/96%, SOC 10–95% |
| Diesel | assumed | 150 kW, 45 kW minimum, ramp/up/down limits, 0.70 kg CO2e/kWh |
| Load allocation and Q | assumed | 3:2:2 and Q=0.2P; Q unused by the simple backend |
| MPC controller | designed baseline | PyPSA rolling-horizon MILP, perfect-foresight forecast from the driving backend, replanned every `backend.rolling_horizon_hours` |

## Limitations

- This is a historical-telemetry-fed simulation, not measured historical dispatch.
- The simple backend is lossless and reports flat 1 pu voltage; it does not validate feeder voltage, losses, or line loading.
- Battery charge is source-resolved into PV/grid/diesel shares by a fixed PV-then-diesel-then-grid merit order; the split is an accounting attribution over the lossless single-bus balance, not a metered physical routing.
- `excess_generation_kwh` exposes supply above served load and battery charging, but no physical dump load is modeled.
- Battery degradation is a placeholder throughput/SOC-stress proxy, not a calibrated life model.
- Equipment and feeder assumptions remain subject to the provenance blockers; results establish a controller baseline, not physical-campus validation.
- The MPC controller has perfect foresight of load/PV over its optimization horizon (no forecast error is modeled); near the end of the 72 h episode the lookahead clamps to the last measured sample instead of running out of data.

## Reproduction

```bash
uv run microgrid-sim replay --config configs/islanded-baseline-72h.yaml --output-dir reports/experiments/islanded_72h_mpc_2026-01-15 --policy mpc
```

`trajectory.csv` contains all 288 evaluated intervals and `metrics.json` contains
the machine-readable aggregate, daily, reward, and violation results.
`visualization.png` is the corresponding four-panel paper-style time-series figure.
