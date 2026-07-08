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

The topology is defined from `buses` and `lines` in `configs/simulator.yaml`, similar to a light Cisco Packet Tracer model: create bus nodes, connect them with lines/transformers, then place PV, battery, diesel, EV chargers, and loads on any bus. The default scenario now separates the PV yard and battery storage so the battery can charge from PV, grid import, diesel, or any other connected source.

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

### Reward (always ≤ 0)

```text
reward = -( w_carbon·carbon + w_autonomy·grid_import + w_health·degradation
          + w_waste·curtailed_solar + w_unserved·unmet_load + constraints )
```

Weights and physics live in `configs/simulator.yaml` (typed, env-overridable
with the `MGS_` prefix).

## Quickstart

```bash
# install (uv creates .venv and resolves from pyproject)
uv sync

# sanity rollout — no learning, rule-of-thumb controller
uv run microgrid-sim play --steps 96 --policy rule

# control-room dashboard (FastAPI + React): topology replay, dispatch charts
uv run microgrid-sim dashboard

# train PPO, save artifacts/ppo_microgrid.zip
uv run microgrid-sim train --algo ppo --timesteps 50000 --config configs/simulator.yaml

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
configs/simulator.yaml       topology, battery, episode, reward weights
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
edit the campus graph in the main topology designer, then tune battery, diesel,
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
