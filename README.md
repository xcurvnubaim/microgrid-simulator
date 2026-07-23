# microgrid-simulator

The **RL Environment (Region 4)** of the Microgrid Brain digital twin — the
training ground everything else trains against.

A [Gymnasium](https://gymnasium.farama.org/) environment for a **editable-bus campus
microgrid**, backed by **pandapower** Newton-Raphson AC power flow, with
**battery state-of-charge physics** and the **five-term reward** from the
architecture diagram (carbon · autonomy · battery health · energy waste ·
unserved load). Trainable out of the box with **stable-baselines3** (PPO / SAC).

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
4. compute the five-term reward
5. advance the timestamp

### Campus topology (editable)

The editable campus topology is defined from `buses` and `lines` in
`configs/simulator.yaml`, similar to a light Cisco Packet Tracer model: create bus nodes,
connect them with lines/transformers, then place PV, battery, diesel, and loads on any bus.
That alternate campus profile separates the PV yard and battery storage; the runtime
default is the single-bus native pymgrid25 scenario 2 translation described below.

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

### Real demand + diesel

Point `demand.file` at the campus net-load export
(`Total Load (net load)_*.xlsx`, header on row 2, 15-min `statstime` +
`demand` in kW) and each `reset()` draws a **random 24h window** from the
Dec–Apr history; the three static loads are scaled proportionally so their
sum tracks the real total (topology untouched). Without a file the original
synthetic sinusoid drives the loads. The diesel genset (default 150 kW — a
placeholder until the real nameplate spec exists) adds
`diesel_kw * carbon_per_kwh_diesel * dt` to the carbon reward term. PV and SOC
are now exposed in the dashboard as a combined PV availability/usage and battery
SoC chart.

For the deterministic paper baseline, `configs/islanded-baseline-72h.yaml` loads
timestamp-aligned measured load and PV, fails on missing samples instead of falling back
to synthetic data, and evaluates 288 intervals from 2026-01-15 through 2026-01-17.
`configs/islanded-heldout-72h.yaml` uses the same plant assumptions on the disjoint
2026-03-08 through 2026-03-10 window for rule/MPC evaluation. The equations, units,
signs, evidence labels, required outputs, and verification tolerances are frozen in
[`BACKEND_CONTRACT.md`](BACKEND_CONTRACT.md).

### Native pymgrid25 scenario input

`configs/pymgrid25-scenario-2.yaml` is the simulator's default scenario and a generated
translation of the authoritative native pymgrid25 scenario 2 YAML. It keeps the benchmark
separate from the campus config, references the original 8,760-row compressed load/PV series, converts
pymgrid's negative-load and internal-battery-energy conventions explicitly, disables
project-only battery degradation, preserves the initial genset state, and selects the
native pymgrid additive cost function. Pymgrid is the reference for this scenario's
declared shared scheduling contract.

The YAML selects a dedicated `battery.model: pymgrid` implementation and stores pymgrid's
symmetric battery limits directly as `16.529 MWh/step` charge and discharge with
`limit_basis: internal_energy_per_step`. The separate campus profile in
`configs/simulator.yaml` uses the `project` model with terminal-power limits and optional
SOH degradation. The pymgrid
model retains native `0.90` efficiency and derives terminal action bounds at runtime;
disabling efficiency or forcing symmetric terminal MW would change its native SOC
transition.

Regenerate the config after changing or updating the pinned pymgrid checkout:

```bash
uv run --extra pymgrid python -m \
  microgrid_simulator.experiments.pymgrid_scenario_import \
  --source-yaml /home/xcurv/teep-taiwan/python-microgrid/src/pymgrid/data/scenario/pymgrid25/microgrid_2/microgrid_2.yaml \
  --scenario-number 2 \
  --output configs/pymgrid25-scenario-2.yaml
```

A simulator smoke rollout can use `uv run microgrid-sim play`; pass
`--config configs/simulator.yaml` to select the campus profile instead.
That runs a project controller on native inputs; it is not yet a reproduction of the
paper's native RBC/MPC/RL results. See
`reports/experiments/pymgrid25_scenario_2_config_verification/report.md` for the mapping
and validation boundary.

### Reward (always ≤ 0)

```text
reward = -( w_carbon·carbon + w_autonomy·grid_import + w_health·degradation
          + w_waste·curtailed_solar + w_unserved·unmet_load + constraints )
```

Default weights and physics live in `configs/pymgrid25-scenario-2.yaml`; alternate
scenario YAMLs remain selectable with `--config` or `MGS_CONFIG`.

## Quickstart

```bash
# install (uv creates .venv and resolves from pyproject)
uv sync

# sanity rollout — native pymgrid25 scenario 2 by default
uv run microgrid-sim play --steps 24 --policy rule

# strict 72-hour measured load/PV replay + Markdown/JSON/CSV evaluation
uv run microgrid-sim replay --config configs/islanded-baseline-72h.yaml

# disjoint held-out rule and MPC evaluations
uv run microgrid-sim replay --config configs/islanded-heldout-72h.yaml --policy rule
uv run microgrid-sim replay --config configs/islanded-heldout-72h.yaml --policy mpc

# control-room dashboard (FastAPI + React): topology replay, dispatch charts
uv run microgrid-sim dashboard

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

## Layout

```
src/microgrid_simulator/
  config.py                  typed settings (yaml + MGS_ env overrides)
  grid/
    battery.py               SoC physics + empirical degradation (PINN hook)
    demand_trace.py          real net-load xlsx/csv -> random episode windows
    diesel.py                diesel genset (on/off + clamped setpoint)
    pandapower_backend.py    editable-bus campus network + power flow + profiles
  model/
    reward.py                the five-term reward / punish
    agent.py                 SB3 PPO/SAC train + evaluate
  env.py                     MicrogridEnv (Gymnasium reset/step/spaces)
  ui/server.py               FastAPI dashboard API (+ demand upload)
  ui/rollout.py              rule / idle / random rollout runner
  ui/web/                    React control room (prebuilt in web/dist)
  cli.py                     train | eval | play | dashboard
configs/pymgrid25-scenario-2.yaml  default native benchmark translation
configs/simulator.yaml       alternate campus topology and project-model profile
tests/                       battery, reward, env (mirrors src/)
```

Layering (one-way imports): `env` → `model/` → `grid/` → `config`.

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
SoC physics (Phase 2). Next up: the Digital Twin state store + EnKF (Phase 4)
and wiring the forecaster's `forecaster:load` Redis feed into the observation.
```

## Dashboard

`uv run microgrid-sim dashboard` serves the control room at `http://127.0.0.1:8501`:
edit the active scenario graph in the main topology designer, then tune battery, diesel,
EVs, PV/loads, and reward weights in the left rail. Drop the demand xlsx straight
onto **Demand source** and run an episode. The topology diagram replays the rollout tick-by-tick with a
scrubber; below it: the PV + SoC chart, the dispatch stack (PV / battery /
diesel / grid vs the demand line), battery SoC, voltage band, reward
decomposition, and the raw per-timestep table with CSV export.

Frontend dev loop (optional — a built copy ships in `ui/web/dist`):

```bash
cd src/microgrid_simulator/ui/web
npm install
npm run dev        # Vite dev server proxying /api to :8501
npm run build      # refresh dist/ served by microgrid-sim dashboard
```
