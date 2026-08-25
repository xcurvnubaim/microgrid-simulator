"""Command-line entrypoint: `microgrid-sim train | eval | play`.

microgrid-sim train --algo ppo --timesteps 50000
microgrid-sim eval  --algo ppo --artifact artifacts/ppo_microgrid.zip
microgrid-sim play  --steps 24          # one default hourly scenario day
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.env import MicrogridEnv

app = typer.Typer(add_completion=False, help="Campus microgrid RL simulator (Region 4).")


def _load_settings(config: Path | None) -> Settings:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return load_settings(config)


def _replay_output_dir(settings: Settings, policy: str) -> Path:
    """Return a collision-free default directory for a fixed telemetry replay."""
    start = settings.episode.telemetry_start
    if not start:
        raise ValueError("fixed telemetry replay requires episode.telemetry_start")
    date = start.split()[0].split("T")[0]
    return Path("reports/experiments") / f"islanded_72h_{policy}_{date}"


@app.command()
def generate_forecast_cache(
    output: Path = typer.Option(Path("data/forecasts.jsonl"), help="output JSONL file path"),
    config: Path | None = typer.Option(None, help="scenario YAML configuration"),
    manifest: Path | None = typer.Option(
        Path("data/forecast_manifest.json"), help="output manifest file path"
    ),
    mode: str = typer.Option("http", help="http (Chronos-2 for PV and demand)"),
) -> None:
    """Pre-generate a leakage-free 24-hour forecast JSONL cache across telemetry ranges."""
    from microgrid_simulator.digital_twin.alignment import align_series
    from microgrid_simulator.digital_twin.data_ingestion import load_measurements
    from microgrid_simulator.forecast import ForecastCache, ForecastClient, ForecastContext

    settings = _load_settings(config)
    if mode != "http":
        raise typer.BadParameter(
            "only http generation is leakage-safe; hybrid and persistence are disabled",
            param_hint="--mode",
        )
    measurements = load_measurements(settings.digital_twin)
    if "load" not in measurements or "pv" not in measurements:
        typer.echo(
            "Error: digital_twin config must specify load and pv measurement sources.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Forecast context must be reconstructed only from observed values. Reject
    # missing samples rather than interpolating them from later measurements.
    aligned = align_series(
        measurements,
        timestep_hours=settings.topology.timestep_hours,
        fill_strategy="error",
        max_gap_steps=0,
        max_missing_fraction=0.0,
    )
    load_series = aligned["load"].dropna()
    pv_series = aligned["pv"].dropna()

    common_idx = load_series.index.intersection(pv_series.index)
    load_series = load_series.reindex(common_idx)
    pv_series = pv_series.reindex(common_idx)

    timestamps = list(common_idx)
    pv_mw = pv_series.to_numpy()
    demand_mw = load_series.to_numpy()
    n_samples = len(timestamps)
    source_id = settings.forecast.source_id or settings.scenario.name

    typer.echo(
        f"Generating leakage-free ({mode}) forecast cache for {n_samples} aligned timestamps..."
    )

    records = {}
    http_client = ForecastClient(settings.forecast)

    # Step through hourly indices, requiring at least 24h (96 steps) of past context
    for i in range(96, n_samples, 4):
        issued_at_ts = pd.Timestamp(timestamps[i])
        issued_at = issued_at_ts.isoformat()

        # Send past 500 steps (~20 days) context to Chronos HTTP forecaster
        ctx_start_idx = max(0, i - 500)
        past_pv = tuple(max(0.0, float(val)) for val in pv_mw[ctx_start_idx : i + 1 : 4])
        past_demand = tuple(max(0.0, float(val)) for val in demand_mw[ctx_start_idx : i + 1 : 4])
        ctx = ForecastContext(
            source_id=source_id,
            frequency_hours=1.0,
            pv_values_mw=past_pv,
            demand_values_mw=past_demand,
        )

        chronos_snap = http_client.fetch(ctx, issued_at=issued_at)

        records[issued_at] = chronos_snap

    cache = ForecastCache(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache.to_jsonl(str(output))
    typer.echo(f"Saved {len(cache)} forecast snapshot records to {output}")

    if manifest:
        manifest_data = {
            "cache_version": "1.0",
            "source_id": source_id,
            "mode": mode,
            "total_records": len(cache),
            "horizon_hours": settings.forecast.horizon_hours,
            "frequency_hours": settings.forecast.target_frequency_hours,
            "issue_frequency_hours": settings.forecast.issue_frequency_hours,
            "forecast_steps": settings.forecast.forecast_steps,
            "target_units": "kw",
            "pv_model": "Chronos-2 Foundation Model",
            "demand_model": "Chronos-2 Foundation Model",
            "causal_preprocessing": True,
            "splits": {
                "train": {"start": "2025-12-02T00:00:00", "end": "2026-01-29T23:45:00"},
                "val": {
                    "start": "2026-02-06T00:00:00",
                    "end": "2026-02-24T23:45:00",
                    "exclude_gaps": ["2026-02-04"],
                },
                "test": {"start": "2026-03-01T00:00:00", "end": "2026-03-31T23:45:00"},
            },
        }
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(manifest_data, indent=2))
        typer.echo(f"Saved manifest to {manifest}")


@app.command()
def train(
    algo: str = typer.Option("sac", help="sac | ppo"),
    timesteps: int = typer.Option(50_000, help="total training timesteps"),
    config: Path | None = typer.Option(
        None, help="scenario YAML (default: nominal islanded campus)"
    ),
    artifact_dir: Path = typer.Option(Path("artifacts"), help="where to save the policy"),
    tensorboard: Path | None = typer.Option(None, help="TensorBoard log dir"),
    seed: int = typer.Option(0),
    forecast_mode: str | None = typer.Option(
        None, help="forecast input override: cached | none (default: config value)"
    ),
    forecast_representation: str | None = typer.Option(
        None, help="forecast representation override: raw | summary"
    ),
    init_artifact: Path | None = typer.Option(
        None, help="existing policy .zip to continue from (fine-tuning)"
    ),
    init_vecnorm: Path | None = typer.Option(
        None, help="parent VecNormalize stats .pkl to restore in training mode"
    ),
    init_replay_buffer: Path | None = typer.Option(
        None, help="parent SAC replay buffer .pkl to reload"
    ),
    save_replay_buffer: bool = typer.Option(
        False, help="save the SAC replay buffer beside the artifact"
    ),
    reset_num_timesteps: bool = typer.Option(
        False, help="restart timestep counters instead of continuing them"
    ),
    stage: str | None = typer.Option(
        None, help="optional stage label used in artifact/checkpoint names"
    ),
    n_envs: int | None = typer.Option(
        None, help="number of parallel environments (default: 4 for ppo, 1 for sac)"
    ),
    run_id: str | None = typer.Option(None, help="explicit run directory id"),
) -> None:
    """Train an SB3 agent against the microgrid environment."""
    from microgrid_simulator.model.agent import train as _train

    settings = _load_settings(config)
    if forecast_mode is not None and forecast_mode not in {"cached", "none"}:
        raise typer.BadParameter("choose cached or none", param_hint="--forecast-mode")
    if forecast_mode is not None:
        settings.rl.forecast_mode = forecast_mode  # type: ignore[assignment]
    if forecast_representation is not None:
        if forecast_representation not in {"raw", "summary"}:
            raise typer.BadParameter(
                "choose raw or summary", param_hint="--forecast-representation"
            )
        settings.rl.forecast_representation = forecast_representation  # type: ignore[assignment]
    path = _train(
        settings,
        algo=algo,
        total_timesteps=timesteps,
        artifact_dir=artifact_dir,
        tensorboard_log=tensorboard,
        seed=seed,
        init_artifact=init_artifact,
        init_vecnorm=init_vecnorm,
        init_replay_buffer=init_replay_buffer,
        save_replay_buffer=save_replay_buffer,
        reset_num_timesteps=reset_num_timesteps,
        stage_name=stage,
        n_envs=n_envs,
        run_id=run_id,
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
    steps: int = typer.Option(24, help="number of ticks (24 = one default hourly scenario day)"),
    config: Path | None = typer.Option(None),
    policy: str = typer.Option("rule", help="rule | random | idle | deterministic"),
) -> None:
    """Roll out a non-learned controller for a sanity check (no training)."""
    from microgrid_simulator.controllers import DeterministicController, RuleBasedController

    settings = _load_settings(config)
    env = MicrogridEnv(settings=settings, render_mode="ansi")
    obs, _ = env.reset()
    total = 0.0
    deterministic_controller = DeterministicController(settings)
    rule_controller = RuleBasedController(settings)

    for _ in range(steps):
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "idle":
            action = np.zeros(env.action_dim, dtype=np.float32)
        elif policy == "deterministic":
            action = env.encode_action(
                deterministic_controller.act(env._last_state)  # noqa: SLF001
            )
        else:
            forecast_available = env._forecast_is_available()  # noqa: SLF001
            pv_horizon, demand_horizon = env._forecast_vectors()  # noqa: SLF001
            action = env.encode_action(
                rule_controller.act(
                    env._last_state,  # noqa: SLF001
                    pv_forecast_mw=env._current_pv_forecast_mw(),  # noqa: SLF001
                    demand_forecast_mw=env._current_demand_forecast_mw(),  # noqa: SLF001
                    pv_forecast_horizon_mw=pv_horizon if forecast_available else None,
                    demand_forecast_horizon_mw=demand_horizon if forecast_available else None,
                )
            )
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        typer.echo(env.render())
        if terminated or truncated:
            break

    typer.echo(f"\ntotal reward over {steps} steps: {total:.2f}")
    env.close()


@app.command()
def replay(
    config: Path = typer.Option(
        Path("configs/islanded-baseline-72h.yaml"), help="fixed telemetry scenario YAML"
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="paper report, metrics JSON, and interval CSV directory "
        "(default: reports/experiments/islanded_72h_<policy>_<telemetry-date>)",
    ),
    policy: str = typer.Option(
        "rule", help="paper baseline controller (rule | pypsa_rh | schedule)"
    ),
    seed: int = typer.Option(0, help="recorded reproducibility seed"),
) -> None:
    """Run a strict 72-hour measured load/PV baseline and export its report."""
    from microgrid_simulator.experiments.telemetry_replay import run_telemetry_replay

    policy = "pypsa_rh" if policy == "mpc" else policy
    settings = _load_settings(config)
    if output_dir is None:
        output_dir = _replay_output_dir(settings, policy)
    metrics = run_telemetry_replay(settings, config, output_dir, policy=policy, seed=seed)
    typer.echo(json.dumps(metrics["totals"], indent=2))
    typer.echo(f"report: {output_dir / 'report.md'}")


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="local dashboard port"),
    host: str = typer.Option("0.0.0.0", help="bind address"),
) -> None:
    """Launch the interactive dashboard (FastAPI + React control room)."""
    import uvicorn

    typer.echo(f"Microgrid control room -> http://{host}:{port}")
    uvicorn.run("microgrid_simulator.ui.server:app", host=host, port=port, log_level="info")


@app.command("ems-run")
def ems_run(
    config: Path | None = typer.Option(None, help="scenario YAML configuration"),
    policy: str = typer.Option("rule", help="rule | sac | ppo"),
    artifact: Path | None = typer.Option(None, help="trained policy .zip for SAC/PPO"),
    steps: int = typer.Option(96, min=1, help="maximum simulated control ticks"),
    timeout_ms: int = typer.Option(1000, min=1, help="command deadline in milliseconds"),
    pace_seconds: float = typer.Option(0.0, min=0.0, help="wall-clock delay between ticks"),
    seed: int = typer.Option(0, help="episode seed"),
) -> None:
    """Run the broker-decoupled EMS against the simulator (no hardware actuation)."""
    from microgrid_simulator.ems import EMSService, InProcessBroker, SimulatorBridge

    if policy not in {"rule", "sac", "ppo"}:
        raise typer.BadParameter("choose rule, sac, or ppo", param_hint="--policy")
    settings = _load_settings(config)

    async def _run() -> None:
        broker = InProcessBroker()
        service = EMSService(
            settings,
            broker,
            policy=policy,  # type: ignore[arg-type]
            artifact=artifact,
            command_ttl_ms=timeout_ms,
        )
        bridge = SimulatorBridge(
            settings,
            broker,
            command_timeout_ms=timeout_ms,
            pace_seconds=pace_seconds,
        )
        service_task = asyncio.create_task(service.run())
        try:
            results = await bridge.run(max_steps=steps, seed=seed)
        finally:
            service_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await service_task
        for result in results:
            typer.echo(result.model_dump_json())
        fallback_count = sum(result.status == "fallback" for result in results)
        typer.echo(
            f"EMS simulation complete: {len(results)} ticks, "
            f"{fallback_count} fallback commands, no physical actuation"
        )

    asyncio.run(_run())


@app.command("telemetry-serve")
def telemetry_serve(
    config: Path | None = typer.Option(None, help="scenario YAML configuration"),
    nats_url: str = typer.Option("nats://127.0.0.1:4222", help="NATS server URL"),
) -> None:
    """Serve strict historical telemetry windows over NATS request/reply."""
    from microgrid_simulator.telemetry.nats import run_worker
    from microgrid_simulator.telemetry.service import ReplayTelemetryService

    asyncio.run(run_worker(ReplayTelemetryService(_load_settings(config)), nats_url))


@app.command("ems-serve")
def ems_serve(
    config: Path | None = typer.Option(None, help="scenario YAML configuration"),
    policy: str = typer.Option("rule", envvar="MGS_EMS_POLICY", help="rule | sac | ppo"),
    artifact: Path | None = typer.Option(
        None, envvar="MGS_EMS_ARTIFACT", help="trained policy .zip for SAC/PPO"
    ),
    timeout_ms: int = typer.Option(1000, min=1, help="command TTL in milliseconds"),
    nats_url: str = typer.Option("nats://127.0.0.1:4222", help="NATS server URL"),
) -> None:
    """Serve rule or learned EMS dispatch inference over NATS request/reply."""

    from microgrid_simulator.ems.nats import run_worker
    from microgrid_simulator.ems.service import EMSService

    if policy not in {"rule", "sac", "ppo"}:
        raise typer.BadParameter("choose rule, sac, or ppo", param_hint="--policy")
    asyncio.run(
        run_worker(
            EMSService(
                _load_settings(config),
                policy=policy,  # type: ignore[arg-type]
                artifact=artifact,
                command_ttl_ms=timeout_ms,
            ),
            nats_url,
        )
    )


@app.command("plant-serve")
def plant_serve(
    config: Path | None = typer.Option(None, help="scenario YAML configuration"),
    nats_url: str = typer.Option("nats://127.0.0.1:4222", help="NATS server URL"),
) -> None:
    """Serve authoritative headless plant transitions over NATS."""
    from microgrid_simulator.plant.nats import run_worker
    from microgrid_simulator.plant.service import PlantService
    from microgrid_simulator.runtime import settings_fingerprint

    settings = _load_settings(config)
    asyncio.run(
        run_worker(
            PlantService(settings, config_fingerprint=settings_fingerprint(settings)),
            nats_url,
        )
    )


@app.command("ems-control-serve")
def ems_control_serve(
    config: Path | None = typer.Option(None, help="scenario YAML configuration"),
    policy: str = typer.Option("rule", envvar="MGS_EMS_POLICY", help="rule | sac | ppo"),
    artifact: Path | None = typer.Option(
        None, envvar="MGS_EMS_ARTIFACT", help="trained policy .zip for SAC/PPO"
    ),
    nats_url: str = typer.Option("nats://127.0.0.1:4222", help="NATS server URL"),
    pace_seconds: float = typer.Option(0.0, min=0.0, help="optional wall-clock delay per tick"),
    host: str = typer.Option("127.0.0.1", help="bind address"),
    port: int = typer.Option(8004, help="HTTP control port"),
) -> None:
    """Serve the EMS-owned scenario lifecycle and dispatch control plane."""
    import uvicorn

    from microgrid_simulator.ems.http import create_app

    if policy not in {"rule", "sac", "ppo"}:
        raise typer.BadParameter("choose rule, sac, or ppo", param_hint="--policy")
    if policy != "rule" and artifact is None:
        raise typer.BadParameter(f"{policy} requires --artifact", param_hint="--artifact")
    uvicorn.run(
        create_app(
            _load_settings(config),
            policy=policy,
            artifact=str(artifact) if artifact is not None else None,
            nats_url=nats_url,
            pace_seconds=pace_seconds,
        ),
        host=host,
        port=port,
        log_level="info",
    )


@app.command("simulator-serve")
def simulator_serve(
    config: Path | None = typer.Option(None, help="scenario YAML configuration"),
    telemetry_url: str | None = typer.Option(None, help="telemetry service base URL"),
    ems_url: str | None = typer.Option(None, help="EMS service base URL"),
    nats_url: str | None = typer.Option(None, help="NATS URL for services and JetStream"),
    timeout_ms: int = typer.Option(1000, min=1, help="EMS command deadline in milliseconds"),
    host: str = typer.Option("127.0.0.1", help="bind address"),
    port: int = typer.Option(8003, help="HTTP port"),
) -> None:
    """Serve authoritative simulation sessions and event streams over HTTP."""
    import uvicorn

    from microgrid_simulator.simulator.http import create_app

    uvicorn.run(
        create_app(
            _load_settings(config),
            telemetry_url=telemetry_url,
            ems_url=ems_url,
            nats_url=nats_url,
            command_timeout_ms=timeout_ms,
        ),
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    app()
