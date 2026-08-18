# Experiment Archive

This directory preserves completed or inactive Module 6 runs and scenarios. Nothing here
is part of the normal runtime or current experiment plan.

## Active Scenario

The single active scenario is `configs/islanded-baseline-72h.yaml`:

- 72-hour islanded campus replay at 15-minute control resolution;
- pandapower AC plant with PV, battery, and diesel;
- F0 causal hourly PV/load forecasts held for four controller ticks;
- nominal dispatch economics;
- rule, PyPSA MPC, no-forecast SAC, and F0 forecast-aware SAC controllers.

`configs/docker-islanded-72h.yaml` is only a container path/transport overlay for this same
scenario. It is not a separate research scenario.

## Layout

- `scenarios/`: inactive YAML scenarios and external reproduction configurations.
- `runs/reports/`: completed experiment outputs and the submitted report evidence.
- `runs/scripts/`: completed training, validation, sensitivity, and evaluation runners.
- `runs/data-f2/`: rejected F2 forecast caches.
- `runs/docs/`: implementation notes for completed experiments.
- `artifacts/sac/`: model artifacts from inactive SAC variants.

The retained active SAC artifacts are `artifacts/sac/cached-1m-v3/` and
`artifacts/sac/noforecast-1m/`. Archived material may be inspected for provenance, but it
must not be treated as an active default or revived without an explicit scope decision.
