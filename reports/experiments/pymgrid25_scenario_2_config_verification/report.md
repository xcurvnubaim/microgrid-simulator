# pymgrid25 scenario 2 configuration verification

## Verdict

**PASS** — `configs/pymgrid25-scenario-2.yaml` is a validated project translation of
the native pymgrid25 scenario 2 plant and data configuration at upstream commit
`09089ccfaab3e95becfda1b26fbce6f9c6195f6a`.

Pymgrid is authoritative for the shared scheduling contract in this scenario. The
campus configuration remains unchanged and is not subordinate to this benchmark.

## Source and generated input

| Role | Path |
|---|---|
| Native source | `/home/xcurv/teep-taiwan/python-microgrid/src/pymgrid/data/scenario/pymgrid25/microgrid_2/microgrid_2.yaml` |
| Simulator input | `configs/pymgrid25-scenario-2.yaml` |
| Importer | `src/microgrid_simulator/experiments/pymgrid_scenario_import.py` |
| Regression test | `tests/test_pymgrid_scenario_import.py` |

The generated input references the native compressed load and renewable sidecars in
place. It does not copy benchmark data into the project repository.

## Verified mapping

| Parameter | Native pymgrid value | Simulator value | Conversion |
|---|---:|---:|---|
| Timestep | 1 h | 1 h | direct benchmark assumption |
| Topology | islanded single bus | islanded single bus | direct |
| Load/PV rows | 8,760 | 8,760 | original `.csv.gz` sidecars |
| Battery maximum capacity | 66,116 kWh | 66.116 MWh | kWh → MWh |
| Battery minimum/initial SOC | 0.20 | 0.20 | direct |
| Battery maximum SOC | 1.00 | 1.00 | direct |
| Battery efficiency | 0.90 | charge/discharge 0.90 | direct |
| Battery internal charge bound | 16.529 MWh/step | 16.529 MWh/step | direct, native limit is authoritative |
| Battery internal discharge bound | 16.529 MWh/step | 16.529 MWh/step | direct, native limit is authoritative |
| Derived charge action bound | — | 18.3655556 MW terminal | internal limit ÷ efficiency ÷ timestep |
| Derived discharge action bound | — | 14.8761 MW terminal | internal limit × efficiency ÷ timestep |
| Genset minimum | 2,429.2 kWh/step | 2.4292 MW | hourly energy → average power |
| Genset maximum | 43,725.6 kWh/step | 43.7256 MW | hourly energy → average power |
| Genset initial status | on | on | direct |
| Battery degradation | absent | disabled | shared-contract alignment |

Native marginal costs are also preserved: battery cycle `0.02`, genset fuel `0.4`,
genset CO2 `2.0` at `0.1` cost per unit, loss load `10.0`, and overgeneration `1.0`.
The scenario selects `reward.mode: pymgrid`, which evaluates the same additive module
costs rather than the campus five-objective reward.

The translated YAML uses `limit_basis: internal_energy_per_step`, so the two symmetric
`16.529 MWh/step` values are stored exactly as pymgrid defines them. Efficiency is not
disabled: terminal action bounds are derived at runtime because equal terminal MW limits
would produce unequal internal-energy changes and would no longer match pymgrid SOC.

## Battery model separation

The scenario explicitly selects `battery.model: pymgrid`. This creates a dedicated
`PymgridBatteryModel` whose state of truth is current stored energy, whose limits are
internal MWh per step, whose efficiency transition follows the native module, and whose
SOH remains fixed. It does not execute the campus model's degradation or capacity-fade
path.

The default `battery.model: project` remains a separate implementation using terminal MW,
SOH, throughput degradation, and faded usable capacity. Configuration validation rejects
internal-energy limits for the project model and rejects terminal-power limits for the
pymgrid model, preventing the two contracts from being mixed accidentally.

## Data and timing checks

- Native load is stored as negative absorbed energy; the input converts it to positive
  demand.
- Native load/PV energy per step is converted to average MW using the one-hour timestep.
- Native row 0 is duplicated as controller context because the project backend advances
  exogenous inputs before applying an action; the first evaluated project interval is
  still native row 0.
- All first 24 evaluated load/PV values match the native arrays within `1e-12 MW`, the
  tolerance required only for decimal CSV round-trip noise.
- First-step exact-load and maximum-diesel/maximum-charge cases match native pymgrid in
  load, diesel production, battery terminal energy, excess generation cost, unserved
  energy, battery SOC, cycle cost, and total reward.
- A direct five-transition charge/discharge sequence hits both symmetric internal limits
  and matches the native `BatteryModule` in terminal energy, stored energy, SOC, zero SOH
  degradation, and cycle reward at every step.
- The full project regression suite passes 65 tests with the pymgrid optional dependency
  enabled.

## Regeneration and validation

```bash
uv run --extra pymgrid python -m \
  microgrid_simulator.experiments.pymgrid_scenario_import \
  --source-yaml /home/xcurv/teep-taiwan/python-microgrid/src/pymgrid/data/scenario/pymgrid25/microgrid_2/microgrid_2.yaml \
  --scenario-number 2 \
  --output configs/pymgrid25-scenario-2.yaml

uv run --extra dev --extra pymgrid pytest tests/test_pymgrid_scenario_import.py
```

The regression test regenerates the mapping in memory and compares it with the checked-in
YAML. A native plant parameter, source path, cost, or conversion drift therefore fails the
test rather than silently changing the benchmark.

## Boundary and next step

This verifies configuration translation, source telemetry, one native dispatch step, and
native reward accounting. It is not yet an 8,760-hour controller reproduction and does
not reproduce the paper's RBC, MPC, or RL totals. The next gate is to execute the same
native pymgrid controller action sequence in both implementations, first over 24–168
hours and then over the complete native horizon.
