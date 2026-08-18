# microgrid-simulator

The **RL Environment (Region 4)** of the Microgrid Brain digital twin — the
training ground everything else trains against.

A [Gymnasium](https://gymnasium.farama.org/) environment for a **editable-bus campus
microgrid**, backed by **pandapower** Newton-Raphson AC power flow, with
**battery state-of-charge physics** and a **multi-term reward** from the
architecture diagram (carbon · autonomy · battery health · energy waste ·
excess generation · unserved load). Trainable out of the box with
**stable-baselines3** (PPO / SAC).

> Companion to `simple-sim/` (the multi-backend + Isaac Sim bridge) and
> `microgrid-forecaster/` (the Data Aggregator). This package is the clean,
> installable RL env described in *How to Build the Microgrid Brain Simulator*.

## The environment loop

```text
reset()  -> (obs, info)
step(a)  -> (obs, reward, terminated, truncated, info)
```

Each `step()`:
1. integrate battery SoC from the action (efficiency-aware, band-clamped)
2. update EV / load powers, apply PV curtailment + time-of-day scaling
3. run `pandapower.runpp()` on the editable-bus campus
4. compute the multi-term reward
5. advance the timestamp

### Active scenario

The runtime default is `configs/islanded-baseline-72h.yaml`: a deterministic 72-hour
islanded campus replay at 15-minute control resolution, using pandapower AC with PV,
battery, and diesel. It uses the F0 causal hourly forecast cache and nominal dispatch
economics for rule, MPC, and SAC comparison. `configs/docker-islanded-72h.yaml` is a
container path/transport overlay for the same scenario, not a separate research scenario.

Completed and inactive scenarios, runs, scripts, caches, and model variants are preserved
under [`archive/`](archive/README.md) and are not normal runtime choices.

| Bus | Voltage | Role |
|---|---|---|
| Bus 0 | 20 kV | External grid connection (slack) |
| Bus 1 | 0.4 kV | Main distribution (transformer from Bus 0) |
| Bus 2 | 0.4 kV | PV yard |
| Bus 3 | 0.4 kV | EV charging hub |
| Bus 4 | 0.4 kV | Academic building loads |
| Bus 5 | 0.4 kV | Battery storage |

### Action (continuous, normalised `[-1, 1]`)

| Index | Meaning | Maps to |
|---|---|---|
| `a[0]` | battery power | `[-max_discharge, +max_charge]` MW |
| `a[1..n_ev]` | EV charge rates | `[0, ev_max]` MW each |
| `a[-3]` | diesel on/off | on when `a[-3] > 0` (diesel enabled only) |
| `a[-2]` | diesel setpoint | `[0, diesel_max_kw]` (diesel enabled only) |
| `a[-1]` | PV curtailment | `[0, 1]` fraction |

### Chronos PV and demand forecast observation

Run the separately deployable forecaster, then enable `forecast.enabled` in a
scenario YAML or in the dashboard's **PV + demand forecast** panel:

```bash
cd ../microgrid-forecaster
uv sync --extra foundation
uv run forecaster serve --config configs/forecaster.yaml
```

The default forecaster configuration is HTTP-only; no Redis process is required.

At reset and, by default, after every simulator step, the simulator sends
`POST /forecast` with the user-selected `forecast.horizon_hours` (1–168) and a
scenario-aligned history context: `source_id`, sampling frequency, and only
observed PV/demand values in kW. The forecaster must echo that source identity;
a campus response is explicitly unavailable in a pymgrid rollout. Responses
contain timestamp-aligned hourly `pv_avg` and `demand` curves in kW, converted
once to MW for the rolling N-hour observation. Step metadata retains the
shifted horizon, source, issue/context time, cold-start and covariate mode,
plus fresh/stale age status; the dashboard shows these diagnostics alongside
the PV and dispatch overlays.

For fixed campus-telemetry replays, each request includes the exact observed
step timestamp and the context ends at the solved step, so the forecaster
cannot use future history. Positional external-reference data (such as pymgrid)
has an artificial clock and deliberately omits that timestamp; it still sends
its own context and runs without campus weather covariates. A repeated service
issue time is marked stale and consumes the retained horizon until exhaustion,
when availability becomes false rather than repeating element zero.

Forecasts are controller information only: measured or synthetic PV and demand
remain the plant inputs. If the service is unavailable, the flag is `0`, both
forecast vectors are zero-filled, and the reason is exposed in rollout metadata.
Because the horizon fixes the observation shape (`1 + N PV + N demand` appended
values), train and evaluate a policy with the same horizon.

### Reconstructed demand + diesel

The campus profile points `demand.file` at the provenance-labelled weekly-seasonal
candidate (`data/processed/demand_15min_weekly_seasonal_reconstruction_candidate.csv`,
15-min `timestamp` + `demand_kw` in kW). The fixed campus replay scenarios point their
load and PV measurements at the reconstruction candidates. Each `reset()` draws a
**random 24h window** from the Dec–Apr history; the three static loads are scaled proportionally so their
sum tracks the real total (topology untouched). Without a file the original
synthetic sinusoid drives the loads. The diesel genset (default 150 kW — a
placeholder until the real nameplate spec exists) adds
`diesel_kw * carbon_per_kwh_diesel * dt` to the carbon reward term. PV and SOC
are now exposed in the dashboard as a combined PV availability/usage and battery
SoC chart. Non-exportable diesel surplus is routed to an explicit dump-load
account, while PV spill, AC losses, and numerical reference-bus balance remain
separate fields. The dashboard exposes diesel-overgeneration energy, peak power,
useful-output percentage, per-step values, and a dedicated source/sink chart.
The configured diesel-bus chart also overlays `Diesel excess (kW)` as a red dashed
series on its own right-side scale, keeping small surplus visible beside total output.

The active `configs/islanded-baseline-72h.yaml` loads
timestamp-aligned reconstruction candidates, fails on missing samples instead of falling back
to synthetic data, and evaluates 288 intervals from 2026-01-15 through 2026-01-17.
The equations, units, signs, evidence labels, required outputs, and verification tolerances are frozen in
[`BACKEND_CONTRACT.md`](BACKEND_CONTRACT.md).

### Archived scenario evidence

The pymgrid reproduction, F2 migration, hard-unserved stress, high-fuel sensitivity,
held-out one-off configurations, and robust-training stages are archived. Their reports and
configuration files remain available for provenance under `archive/`; they are not active
alternatives to the nominal islanded scenario.

### Reward (always ≤ 0)

```text
reward = -( w_carbon·carbon + w_autonomy·grid_import + w_health·degradation
          + w_waste·curtailed_solar + w_excess·dumped_overgeneration
          + w_unserved·unmet_load + constraints )
```

Default weights and physics live in `configs/islanded-baseline-72h.yaml`.

## Quickstart

```bash
# install (uv creates .venv and resolves from pyproject)
uv sync

# sanity rollout — nominal islanded campus scenario by default
uv run microgrid-sim play --steps 24 --policy rule

# strict 72-hour measured load/PV replay + Markdown/JSON/CSV evaluation
uv run microgrid-sim replay --config configs/islanded-baseline-72h.yaml

# control-room dashboard (FastAPI + React): topology replay, dispatch charts
uv run microgrid-sim dashboard

# broker-decoupled, simulation-only EMS with rule fallback and acknowledgments
uv run microgrid-sim ems-run --config configs/islanded-baseline-72h.yaml --policy rule

# independently runnable boundaries after starting a NATS server with JetStream
uv run microgrid-sim telemetry-serve --config configs/islanded-baseline-72h.yaml \
  --nats-url nats://127.0.0.1:4222
uv run microgrid-sim plant-serve --config configs/islanded-baseline-72h.yaml \
  --nats-url nats://127.0.0.1:4222
uv run microgrid-sim ems-control-serve --config configs/islanded-baseline-72h.yaml \
  --policy rule --nats-url nats://127.0.0.1:4222

# train PPO on the default scenario, save artifacts/ppo_microgrid.zip
uv run microgrid-sim train --algo ppo --timesteps 50000

# evaluate the trained policy
uv run microgrid-sim eval artifacts/ppo_microgrid.zip --algo ppo --episodes 3
```

Use it as a plain Gymnasium env too:

```python
import gymnasium as gym
import microgrid_simulator  # registers "MicrogridEnv-v0"

env = gym.make("MicrogridEnv-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Docker Compose EMS Setup

The Compose setup runs the same **simulation-only** EMS boundary. It does not connect to
SCADA, field telemetry, or physical equipment.

Prerequisites:

- Docker Engine with Compose;
- campus telemetry under `../data/processed/` relative to this repository, or set
  `TEEP_DATA_DIR` to the host data directory;

```bash
# inspect or customize telemetry and ports
cp .env.example .env

# validate without building or starting containers
./scripts/run_ems_scenario.sh --dry-run

# run interactively: choose Normal/EMS, scenario, policy, then artifact
./scripts/run_ems_scenario.sh

# bypass the menu and run the rule controller directly
./scripts/run_ems_scenario.sh --policy rule

# run one learned controller in the EMS container
./scripts/run_ems_scenario.sh \
  --config configs/docker-islanded-72h.yaml \
  --policy sac \
  --artifact artifacts/sac/noforecast-1m/seed-1/sac_microgrid.zip
```

When launched in a terminal without explicit selections, the script first chooses a
Compose profile. **Normal** starts the integrated simulator whose dashboard owns Run/Stop
and editable settings. **EMS** discovers container-compatible YAML below `configs/`, then
discovers SAC/PPO `.zip` files below `artifacts/<policy>/`. Explicit arguments remain
available for automation, and `--interactive` forces the menus.

In EMS mode the CLI starts services in the background, starts the run through the EMS
control API, and prints live episode progress. One episode is the configured 72-hour
window (288 15-minute ticks). After it completes, the CLI asks whether to continue. A
continuation uses the same run identity, carries battery/diesel physical state, and moves
the next 72-hour window forward by one timestep. Ctrl-C requests cancellation at a tick
boundary before teardown. Cleanup uses `docker compose down --remove-orphans` without
`-v`, preserving the `nats-data` JetStream volume and artifacts.

Telemetry, plant, and the passive EMS dashboard use the lightweight non-root `runtime`
image. The EMS service and integrated Normal dashboard use the `cpu-rl` target, which
installs CPU-only PyTorch and Stable-Baselines3. The Normal dashboard needs it only when
the user selects RL; imports and model loading remain lazy. Distributed EMS validates the
observation/action shape before a run begins. Training remains host-side. Configs and
artifacts are mounted read-only, and Compose runs exactly one EMS controller at a time.

In the distributed profile the EMS owns the scenario clock, telemetry/forecast context,
canonical controller observation, dispatch, and episode lifecycle. The headless plant owns
pandapower physics, feasibility, and authoritative state transitions/accounting. Core NATS
request/reply carries telemetry windows and idempotent plant transitions; JetStream persists
ordered dashboard events in `nats-data`. The dashboard is passive: it discovers the latest
run through its same-origin API, replays/follows the JetStream trace, shows the external
policy, and never starts simulation or performs RL inference. NATS monitoring is available
at `http://localhost:8222` by default.

## Layout

```
src/microgrid_simulator/
  config.py                  typed settings (yaml + MGS_ env overrides)
  core/                      backend contract, shared types, scenario, time series
  components/                battery, diesel, PV, load, grid-intertie device models
  backends/                  create_backend registry (runtime locked to pandapower AC)
    simple_backend.py        fast lossless balance plant for RL
    pandapower_backend/      AC power-flow plant: network / profiles / snapshot / backend
    pypsa_backend.py         operational scheduling research utility
    opendss_backend.py       OpenDSS skeleton
  forecast/                  Chronos HTTP client + strict leakage-free JSONL cache
  messaging/                 NATS subjects, JSON envelopes, JetStream trace publisher
  contracts/                 telemetry, plant-observation, command, and result schemas
  telemetry/                 strict replay sessions and HTTP service
  simulator/                 transport-neutral orchestrator, providers, and HTTP service
  controllers/               idle / rule / deterministic / manual / MPC / RL policies
  digital_twin/              measured-data ingestion, alignment, replay
  model/reward.py            the multi-term reward / punish
  rl/                        Gymnasium env, SB3 train/eval, episode sampler
  experiments/               reproducible runners (telemetry replay, pymgrid verification)
  ems/                       inference, shield, broker compatibility, and HTTP service
  grid/                      compatibility shims for the pre-backends layout
  ui/server.py               FastAPI dashboard API (+ demand upload)
  ui/rollout.py              rule / idle / random rollout runner
  ui/web/                    React control room (prebuilt in web/dist)
  cli.py                     research commands plus telemetry/EMS/simulator service entry points
configs/islanded-baseline-72h.yaml  active nominal campus scenario
configs/docker-islanded-72h.yaml    container overlay for the active scenario
archive/                          inactive scenarios and completed run evidence
tests/                       battery, reward, env, accounting (mirrors src/)
```

Layering (one-way imports): `rl/` + `controllers/` → `backends/` → `components/` → `core/` → `config`.

## Dev

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src
```

## Where it sits in the build order

Phase 3 of *How to Build the Microgrid Brain Simulator* — the RL agent training
loop. It builds on the pandapower physics (Phase 1) and the five-term reward +
SoC physics (Phase 2). The Chronos service is now wired into the observation
through its horizon-aware HTTP endpoint. Next up is the Digital Twin state
store + EnKF (Phase 4).

## Dashboard

`uv run microgrid-sim dashboard` serves the control room at `http://127.0.0.1:8501`:
edit the active scenario graph in the main topology designer, then tune battery, diesel,
EVs, PV/loads, and reward weights in the left rail. Drop the demand xlsx straight
onto **Demand source** and run an episode. The topology diagram replays the rollout tick-by-tick with a
scrubber; below it: the PV + SoC chart, the dispatch stack (PV / battery /
diesel / grid vs the demand line), battery SoC, voltage band, reward
decomposition, and the raw per-timestep table with CSV export.

Use **Service map** in the top navigation for a human-readable view of the running
simulator, telemetry, EMS, NATS, and browser communication paths. The page includes a
bounded message inspector with expandable request, reply, and dashboard-event payloads.
In Compose, it reads NATS monitoring from `NATS_MONITOR_URL` (default `http://nats:8222`)
and observes application messages through `NATS_URL` (default `nats://nats:4222`),
refreshing status and recent messages every five seconds.

Frontend dev loop (optional — a built copy ships in `ui/web/dist`):

```bash
cd src/microgrid_simulator/ui/web
npm install
npm run dev        # Vite dev server proxying /api to :8501
npm run build      # refresh dist/ served by microgrid-sim dashboard
```
