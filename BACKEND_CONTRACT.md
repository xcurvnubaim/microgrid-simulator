# Fast reduced-order backend contract

This is the executable scheduling-tier contract shared by the Python environment,
controller baselines, reports, and the 900 s Simulink derivative. It documents the
project-created reduction; it is not an EMT, feeder, or physical-campus validation claim.

## Time, units, and signs

- One scheduling tick is `dt = topology.timestep_hours`; the campus experiments use
  `dt = 0.25 h` (900 s).
- Power is MW inside Python and converted to kW only at report/API boundaries. Energy is
  `power_MW * dt_h` in MWh.
- `ControlAction.battery_p_mw > 0` means charging (bus demand);
  `battery_p_mw < 0` means discharging (bus supply).
- Grid import is positive and export is negative. Diesel and PV generation are
  non-negative. `pv_curtail` is a fraction in `[0, 1]`.
- A returned `GridState` describes the interval just solved. Requested actions may be
  clipped by power, SOC, commitment, ramp, islanding, and intertie constraints.

The public interface is `MicrogridBackend.reset()`, `step()`, `get_state()`, and
`close()` in `src/microgrid_simulator/core/backend.py`. Every backend returns the common
`GridState` schema in `core/types.py`.

## Scheduling equations

For a lossless single scheduling bus, using realized rather than requested power:

```text
P_grid = P_load_served + P_batt - P_pv_used - P_diesel
P_load_served = P_load_demand - P_unserved
P_pv_spill = max(0, P_pv_available - P_pv_used)
```

In an island, `P_grid = 0`. Battery discharge is clipped to residual demand,
`max(0, P_load_demand - P_pv - P_diesel)`. Battery charging is clipped to local
generation surplus. PV with no load, storage, or export sink is curtailed. Remaining
inflexible surplus is reported as `excess_generation_kw`; it is not called waste and is
not evidence of a physical dump load.

Battery SOC uses AC-terminal power:

```text
SOC[k+1] = SOC[k] + eta_charge * P_charge * dt / E_usable
SOC[k+1] = SOC[k] - P_discharge * dt / (eta_discharge * E_usable)
E_usable = nameplate_capacity * SOH
```

Power and SOC are clamped to the configured limits, and realized terminal power is backed
out from the clamped SOC transition. The current SOH loss is a designed throughput/SOC
stress proxy, not a calibrated degradation model.

Diesel output is the tick-average power after on/off lockouts, start delay, minimum stable
load, and ramp limits. This preserves interval energy when a transition occurs inside a
15-minute tick.

## Required outputs and identities

Every paper-facing rollout reports at least:

- demand, served load, and unserved load;
- available/used/spilled PV;
- realized battery charge/discharge and SOC;
- battery charging attributed to PV, diesel, and grid;
- diesel power, state, starts, and runtime;
- grid import/export when connected;
- excess-generation proxy and named-path balance residual;
- each reward penalty and each constraint violation separately.

The fixed charging attribution is PV surplus, then diesel surplus, then grid import. The
three source fields must sum to realized battery charging. Attribution is an accounting
rule over the single-bus balance, not a metered physical routing claim.

## Evidence classes for the campus scenario

| Quantity | Evidence | Current value/source |
|---|---|---|
| Load and PV trajectories | measured | timestamp-aligned campus exports |
| Asset topology/roles | inherited ancestry plus project reduction | preserved MathWorks architecture and provenance matrix |
| Battery dispatch cap | inherited/design mapping | ±250 kW scheduling reference |
| Battery capacity, efficiencies, SOC band | assumed | 500 kWh, 96%/96%, 10–95% |
| Diesel limits and emissions | assumed | 150 kW, 45 kW minimum, configured transition limits, 0.70 kg CO2e/kWh |
| Balance, curtailment, attribution, reward | designed | project equations and tests |
| Feeder voltage/loss/line limits | unknown or assumed | not established by the simple backend |

Do not replace missing equipment or feeder evidence with unlabeled typical values.

## Verification and tolerances

- Unit/regression tests enforce SOC direction/bounds, device limits, islanded
  charge/discharge feasibility, PV spill, source attribution, deterministic replay, and
  reward/accounting outputs.
- The January rule trajectory matched `CampusMicrogridScheduling.slx` within
  `4.71e-13 kW`; the Simulink model matched the direct MATLAB implementation within
  `1.42e-14 kW`.
- Scheduling trajectory comparisons use `1e-6 kW` for reported power and `1e-9` for
  direct in-process numerical parity unless a runner states a tighter tolerance.
- `unserved_mw <= 1e-6` is treated as optimizer noise rather than a blackout.
- pandapower checks are quasi-static feasibility evidence only. They cannot become
  physical-campus validation until the feeder, transformer, phase, and reactive-power
  data are available.

The controlling implementation tests are in `tests/test_battery.py`,
`tests/test_demand_diesel.py`, `tests/test_power_accounting.py`, and
`tests/test_telemetry_replay.py`.
