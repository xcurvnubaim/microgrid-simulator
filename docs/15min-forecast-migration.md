# 15-Minute Forecast Resolution Migration — Implementation Notes

Status: **code + caches + benchmark complete**; SAC-F2 training and paired
operational validation deferred (see the plan's Execution Sequence).

## What changed

### Timing contract

`ForecastCfg` now carries an explicit target/issue frequency in addition to the
horizon. `forecast_steps` is derived, never configured directly:

```yaml
forecast:
  horizon_hours: 24
  target_frequency_hours: 0.25   # 15-minute forecast targets
  issue_frequency_hours: 0.25    # one snapshot per controller decision
```

`forecast_steps = horizon_hours / target_frequency_hours = 96`. Invalid
combinations (non-integer steps, issue frequency above horizon) fail closed at
config load.

### Cache schema

- `ForecastCacheManifest` and each JSONL record now declare
  `issue_frequency_hours`, `forecast_steps`, and `target_units` (always `kw` at
  the boundary) in addition to `horizon_hours`/`frequency_hours`.
- `ForecastSnapshot` exposes `steps`, `issue_frequency_hours`,
  `forecast_steps`, and `target_units`, plus first/last-target metadata via
  `as_meta()`.
- Strict loading validates that the derived point count matches
  `forecast_steps`, that per-record issue frequency and target units agree with
  the manifest, and that the first target is strictly after `issued_at` at
  15-minute spacing. Any future-value substitution fails closed.

### Consumers

- `MicrogridEnv` / `build_observation` / EMS orchestrator now append
  `forecast.forecast_steps` forecast points (96) instead of `horizon_hours`.
- `PyPSAMPCController` uses a frequency-aware alignment (`_align_forecast_to_steps`):
  a direct 15-minute F2 snapshot is consumed one point per tick; a coarser F0
  snapshot is held across `frequency / control_interval` ticks. The old hourly
  F0 path (`_resample_hourly_to_quarter`) is preserved for the incumbent baseline.
- `StrictCachedForecastClient` uses `issue_frequency_hours` (not target
  frequency) for staleness.

### Forecaster (microgrid-forecaster repo)

- `ForecastRequest`/`ForecastPayload` support `target_frequency_h` and
  `issue_frequency_h`.
- `Predictor.predict` produces `steps = horizon / target_frequency` points at the
  requested frequency and sets `frequency_h`/`issue_frequency_h` accordingly.
- `Service.run_once` selects the **shortwave** covariate mode when the target
  frequency is sub-hourly, reading the existing interpolated
  `shortwave_radiation_wm2` trajectory verbatim from the processed PV candidate
  (no weather alignment is regenerated). Context history is consumed at the
  caller frequency (15-minute), not resampled to hourly.

## F2 cache generation

The forecaster changes live in the **separate** `microgrid-forecaster` directory
(empty `.git`, no remote — not part of this repository). They must be copied to
the training host alongside this repo:

- `microgrid-forecaster/src/microgrid_forecaster/` (15-min target frequency,
  `shortwave` covariate mode, `issue_frequency_h`)
- `microgrid-forecaster/configs/forecaster-f2-15min.yaml`
- `microgrid-forecaster/scripts/generate_f2_cache.py`

```bash
cd <training-host>/microgrid-forecaster
uv sync --extra foundation   # pulls chronos-forecasting + torch
uv run python scripts/generate_f2_cache.py --split train
uv run python scripts/generate_f2_cache.py --split val
```

Each run writes a split-specific, causal, provenance-labelled cache under
`microgrid-simulator/data/f2/<split>/` (`forecasts.jsonl` + `forecast_manifest.json`).
The caches are git-ignored (`data/f2/`) and are regenerated on the training host,
not committed. Context is the preceding quarter-hour history up to 512 steps
(~5 days 8 hours); the first target is the next right-endpoint interval; PV is
clamped non-negative to match the HTTP boundary. The future shortwave trajectory
is used under an **assumed-known weather** condition (`covariate_mode:
"shortwave"`); archived ECMWF issue/model-run times are not verified.

## F0/F2 benchmark

```bash
cd /home/xcurv/teep-taiwan/microgrid-simulator
.venv/bin/python scripts/benchmark_f0_vs_f2.py \
  --f0-cache data/forecasts.jsonl \
  --f0-manifest reports/experiments/f2_benchmark/f0_manifest.json \
  --f2-cache data/f2/val/forecasts.jsonl \
  --f2-manifest data/f2/val/forecast_manifest.json
```

On six fixed February origins, F2 improves the controller-relevant net-load and
ramp metrics (net MAE, ramp MAE, 0-1h net MAE) over the hourly F0 baseline. The
JSON report is written to `reports/experiments/f2_benchmark/f0_vs_f2.json`.

## Remaining (deferred to the training host)

- Train SAC-F2 (≥3 seeds, nominal islanded economics) using
  `configs/islanded-f2-15min-72h.yaml` once the train cache is regenerated.
- Run the paired February operational validation (MPC-F0 vs MPC-F2, SAC-F0 vs
  SAC-F2, no-forecast ablation).
- Record the Gate 3/4 acceptance decision and freeze the representation before
  starting the islanded economic scenario plan.

## Training-host smoke test (verified)

The full training path was smoke-tested on this host before commit:

- `train_policy` on `islanded-f2-15min-72h.yaml` (40 steps, CPU) writes
  `sac_microgrid.zip`;
- a train-split env reset yields a 214-dim observation (21 base + 1 availability
  + 96 PV + 96 demand) with `forecast_available=True` at every step;
- `PyPSAMPCController` re-plans against the F2 cache (one point per tick).
