# pymgrid common-model verification

> This is cross-implementation verification of a deliberately matched, lossless
> scheduling model. It is not physical validation of the campus microgrid.

## Verification verdict

**PASS** — the two independent
implementations produced equivalent canonical trajectories within the declared
numerical tolerances after explicit input and output normalization.

This answers one narrow question: **does the project simulator implement the same
matched scheduling and accounting contract as pymgrid for these deterministic
cases?** It does not establish campus-plant validity or reproduce a published
pymgrid25 controller result.

## Reference priority for this scenario

For this common-model verification only, **pymgrid is the reference implementation**
and `microgrid-simulator` is the candidate. Every reported delta is calculated as
`project simulator - pymgrid`. If a delta exceeds tolerance, the gate fails and the
project result is not accepted for this scenario.

Mismatch resolution order:

1. Check units, signs, timestep alignment, module naming, and canonical output mapping.
2. If the mapping is correct and behavior belongs to the shared contract, change the
   project simulator or its scenario-specific adapter to match pymgrid, then rerun all
   regression tests.
3. Keep a difference only when it is intentional and outside the shared contract;
   document it explicitly and exclude that field from equivalence claims rather than
   silently overriding either trajectory.

This priority does not make pymgrid authoritative for campus telemetry, feeder physics,
degradation, diesel transients, rewards, or other features absent from the common model.

## Implementations under test

| Run | Software | Model used | License / provenance |
|---|---|---|---|
| Project run | microgrid-simulator 0.1.0 | `SimpleBackend` with degradation and diesel transients disabled only for this fixture | Local project, MIT |
| Reference run | python-microgrid 1.4.1 | Native `Microgrid` with load, renewable, battery, genset, and unbalanced-energy modules | LGPL-3.0, upstream `09089ccfaab3e95becfda1b26fbce6f9c6195f6a` |

The runs use separate model objects and state transitions. The pymgrid adapter receives
the raw requested battery and diesel actions; it performs its own feasibility clipping.
For the battery action, only sign and MW-to-MWh conversion occur at the input boundary.
Because pymgrid's renewable module has no curtailment action, requested PV curtailment
is represented as reduced usable availability while the original availability is retained
for the canonical spill calculation.

## Common plant contract

| Parameter | Value |
|---|---:|
| Timestep | 1 h |
| Network | One islanded, lossless bus |
| Battery capacity | 1 MWh |
| Battery charge limit | 0.3 MW |
| Battery discharge limit | 0.3 MW |
| Battery SOC range | 0.1–0.9 |
| Initial battery SOC | 0.5 |
| Diesel output range while on | 0.1–0.4 MW |
| Grid connection | None |

Battery efficiencies 1.00 and 0.96 are verified as separate complete runs.

## Fixture coverage

| # | Fixture | Load (MW) | PV available (MW) | Battery request (MW) | Diesel request | PV curtailment |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `pv_equals_load` | 0.2 | 0.2 | 0 | off | 0 |
| 2 | `charge_request_clipped` | 0.2 | 0.5 | 0.4 | off | 0 |
| 3 | `discharge_request_clipped` | 0.4 | 0.1 | -0.4 | off | 0 |
| 4 | `discharge_toward_floor` | 0.6 | 0 | -0.3 | off | 0 |
| 5 | `discharge_hits_floor` | 0.4 | 0 | -0.3 | off | 0 |
| 6 | `discharge_at_floor` | 0.2 | 0 | -0.2 | off | 0 |
| 7 | `diesel_minimum_excess` | 0.05 | 0 | 0 | 0.02 MW | 0 |
| 8 | `diesel_pv_charge` | 0.2 | 0.3 | 0.15 | 0.1 MW | 0 |
| 9 | `charge_toward_ceiling_1` | 0.1 | 0.7 | 0.3 | off | 0 |
| 10 | `charge_toward_ceiling_2` | 0.1 | 0.7 | 0.3 | off | 0 |
| 11 | `charge_hits_ceiling` | 0.1 | 0.7 | 0.3 | off | 0 |
| 12 | `charge_at_ceiling` | 0.1 | 0.7 | 0.2 | off | 0 |
| 13 | `explicit_pv_curtailment` | 0.2 | 0.4 | 0 | off | 0.5 |
| 14 | `diesel_maximum_clamp` | 0.4 | 0 | 0 | 0.6 MW | 0 |

Positive battery request means charging in the project convention. The fixture
includes requests beyond available surplus/deficit, component power limits, and SOC
headroom so each implementation must enforce its own constraints.

## Independent run summaries

### Battery efficiency 1.00

| Metric | Project simulator | pymgrid | Absolute delta |
|---|---:|---:|---:|
| Load demand (MWh) | 3.25 | 3.25 | 0 |
| Load served (MWh) | 2.45 | 2.45 | 4.44e-16 |
| Unserved energy (MWh) | 0.8 | 0.8 | 1.11e-16 |
| PV available (MWh) | 4.3 | 4.3 | 0 |
| PV used (MWh) | 2.3 | 2.3 | 0 |
| PV spill (MWh) | 2 | 2 | 0 |
| Battery terminal charge (MWh) | 1.1 | 1.1 | 2.22e-16 |
| Battery terminal discharge (MWh) | 0.7 | 0.7 | 1.11e-16 |
| Diesel generation (MWh) | 0.6 | 0.6 | 0 |
| Excess generation (MWh) | 0.05 | 0.05 | 0 |
| Initial SOC | 0.5 | 0.5 | 0 |
| Final SOC | 0.9 | 0.9 | 0 |
| Maximum balance residual (MW) | 0 | 0 | 0 |

### Battery efficiency 0.96

| Metric | Project simulator | pymgrid | Absolute delta |
|---|---:|---:|---:|
| Load demand (MWh) | 3.25 | 3.25 | 0 |
| Load served (MWh) | 2.41048 | 2.41048 | 0 |
| Unserved energy (MWh) | 0.83952 | 0.83952 | 1.11e-16 |
| PV available (MWh) | 4.3 | 4.3 | 0 |
| PV used (MWh) | 2.33333333333 | 2.33333333333 | 0 |
| PV spill (MWh) | 1.96666666667 | 1.96666666667 | 0 |
| Battery terminal charge (MWh) | 1.13333333333 | 1.13333333333 | 2.22e-16 |
| Battery terminal discharge (MWh) | 0.66048 | 0.66048 | 0 |
| Diesel generation (MWh) | 0.6 | 0.6 | 0 |
| Excess generation (MWh) | 0.05 | 0.05 | 0 |
| Initial SOC | 0.5 | 0.5 | 0 |
| Final SOC | 0.9 | 0.9 | 0 |
| Maximum balance residual (MW) | 0 | 0 | 0 |

## Cross-implementation acceptance

| Battery efficiency | Steps | Passed | Max power-field delta (MW) | Max SOC delta |
|---:|---:|---|---:|---:|
| 1.00 | 14 | yes | 5.55e-17 | 0 |
| 0.96 | 14 | yes | 5.55e-17 | 0 |

Acceptance limits are `1e-9 MW` for every power field, `1e-9 MWh` for stored
and aggregate energy, and `1e-9` for SOC. Every step also has a zero balance
residual to floating-point precision.

Canonical mappings used for comparison:

- project battery `+MW` = charge; pymgrid battery `+MWh` = discharge;
- pymgrid unbalanced energy provided = unserved load;
- pymgrid unbalanced energy absorbed = excess generation;
- `pv_spill = pv_available - pv_used`; excess generation is not relabeled as spill;
- native rewards and costs are excluded because their definitions are not equivalent.

## Audit correction

The first adapter revision pre-clipped the battery request with a duplicate of the
project feasibility rule before calling pymgrid. Although pymgrid's native clipping
produced the same trajectory, that made the implementations less independent than
the report claimed. The adapter now sends the raw request, and two fixtures
deliberately exceed feasible charge/discharge so the native limit behavior is tested.
The corrected verification still passes with the same tolerance-scale deltas.

## Evidence files

For each efficiency, the directory contains a project-only trajectory, a
pymgrid-only trajectory, and a field-by-field comparison trajectory. `metrics.json`
contains the plant, fixture inputs, independent summaries, tolerances, deltas, and
overall verdict.

Reproduce from the repository root:

```bash
uv run --extra pymgrid python -m \
  microgrid_simulator.experiments.pymgrid_verification \
  --output-dir reports/experiments/pymgrid_common_model_verification
uv run --extra dev --extra pymgrid pytest tests/test_pymgrid_verification.py
```

## Interpretation and limitations

A pass means the two implementations agree after their units, signs, timing,
device limits, and imbalance terminology are made identical. It does not validate
pymgrid25's assumed capacities, either simulator's native reward, feeder physics,
battery degradation, diesel transients, or the physical campus.

The next experiment may reproduce a native pymgrid25 scenario and compare
controllers, but those results must remain separate from campus-telemetry results.
