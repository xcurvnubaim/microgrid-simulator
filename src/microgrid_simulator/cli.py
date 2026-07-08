"""Command-line entrypoint: `microgrid-sim train | eval | play`.

    microgrid-sim train --algo ppo --timesteps 50000 --config configs/simulator.yaml
    microgrid-sim eval  --algo ppo --artifact artifacts/ppo_microgrid.zip
    microgrid-sim play  --steps 96          # rule-of-thumb rollout, no policy
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import typer

from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.env import MicrogridEnv

app = typer.Typer(add_completion=False, help="Campus microgrid RL simulator (Region 4).")


def _load_settings(config: Path | None) -> Settings:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return load_settings(config)


@app.command()
def train(
    algo: str = typer.Option("ppo", help="ppo | sac"),
    timesteps: int = typer.Option(50_000, help="total training timesteps"),
    config: Path | None = typer.Option(None, help="path to simulator.yaml"),
    artifact_dir: Path = typer.Option(Path("artifacts"), help="where to save the policy"),
    tensorboard: Path | None = typer.Option(None, help="TensorBoard log dir"),
    seed: int = typer.Option(0),
) -> None:
    """Train an SB3 agent against the microgrid environment."""
    from microgrid_simulator.model.agent import train as _train

    settings = _load_settings(config)
    path = _train(
        settings,
        algo=algo,
        total_timesteps=timesteps,
        artifact_dir=artifact_dir,
        tensorboard_log=tensorboard,
        seed=seed,
    )
    typer.echo(f"saved: {path}")


@app.command()
def eval(
    artifact: Path = typer.Argument(..., help="trained policy .zip"),
    algo: str = typer.Option("ppo", help="ppo | sac"),
    episodes: int = typer.Option(3),
    config: Path | None = typer.Option(None),
) -> None:
    """Evaluate a trained policy and print reward + grid-import stats."""
    from microgrid_simulator.model.agent import evaluate

    settings = _load_settings(config)
    metrics = evaluate(settings, artifact=artifact, algo=algo, episodes=episodes)
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def play(
    steps: int = typer.Option(96, help="number of ticks (96 = 24h @ 15min)"),
    config: Path | None = typer.Option(None),
    policy: str = typer.Option("rule", help="rule | random | idle | deterministic"),
) -> None:
    """Roll out a non-learned controller for a sanity check (no training)."""
    from microgrid_simulator.controllers import DeterministicController

    settings = _load_settings(config)
    env = MicrogridEnv(settings=settings, render_mode="ansi")
    obs, _ = env.reset()
    total = 0.0
    controller = DeterministicController(settings)

    for _ in range(steps):
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "idle":
            action = np.zeros(env.action_dim, dtype=np.float32)
        elif policy == "deterministic":
            action = env.encode_action(controller.act(env._last_state))  # noqa: SLF001
        else:  # rule: charge battery when sun is up, discharge in evening peak
            hour = env.backend.timestamp % 24.0
            batt = 0.6 if 8 <= hour < 15 else (-0.6 if 18 <= hour < 22 else 0.0)
            action = np.zeros(env.action_dim, dtype=np.float32)
            action[0] = batt
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        typer.echo(env.render())
        if terminated or truncated:
            break

    typer.echo(f"\ntotal reward over {steps} steps: {total:.2f}")
    env.close()


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="local dashboard port"),
    host: str = typer.Option("127.0.0.1", help="bind address"),
) -> None:
    """Launch the interactive dashboard (FastAPI + React control room)."""
    import uvicorn

    typer.echo(f"Microgrid control room -> http://{host}:{port}")
    uvicorn.run("microgrid_simulator.ui.server:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    app()
