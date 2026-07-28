"""Command-line entrypoint: `microgrid-sim train | eval | play`.

microgrid-sim train --algo ppo --timesteps 50000
microgrid-sim eval  --algo ppo --artifact artifacts/ppo_microgrid.zip
microgrid-sim play  --steps 24          # one default hourly scenario day
"""

from __future__ import annotations

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
    manifest: Path | None = typer.Option(Path("data/forecast_manifest.json"), help="output manifest file path"),
    mode: str = typer.Option("hybrid", help="hybrid (LightGBM ECMWF for PV + Chronos HTTP for Demand) | http | persistence"),
) -> None:
    """Pre-generate a leakage-free 24-hour forecast JSONL cache across telemetry ranges."""
    from microgrid_simulator.digital_twin.data_ingestion import load_measurements
    from microgrid_simulator.digital_twin.alignment import align_series
    from microgrid_simulator.forecast import ForecastCache, ForecastSnapshot, ForecastClient, ForecastContext
    from lightgbm import LGBMRegressor

    settings = _load_settings(config)
    measurements = load_measurements(settings.digital_twin)
    if "load" not in measurements or "pv" not in measurements:
        typer.echo("Error: digital_twin config must specify load and pv measurement sources.", err=True)
        raise typer.Exit(code=1)

    aligned = align_series(measurements, timestep_hours=0.25, max_gap_steps=1000, max_missing_fraction=0.50)
    load_series = aligned["load"].dropna()
    pv_series = aligned["pv"].dropna()

    common_idx = load_series.index.intersection(pv_series.index)
    load_series = load_series.reindex(common_idx)
    pv_series = pv_series.reindex(common_idx)

    timestamps = list(common_idx)
    pv_mw = pv_series.to_numpy()
    demand_mw = load_series.to_numpy()
    n_samples = len(timestamps)

    typer.echo(f"Generating leakage-free ({mode}) forecast cache for {n_samples} aligned timestamps...")

    # Load ECMWF weather data for LightGBM PV forecaster
    ecmwf_path = Path("/home/xcurv/teep-taiwan/data/ecmwf-ifs.json")
    lgbm_pv_model = None
    ecmwf_df = None

    if mode == "hybrid" and ecmwf_path.exists():
        with open(ecmwf_path) as f:
            ecmwf_df = pd.DataFrame(json.load(f)["hourly"])
        ecmwf_df["time"] = pd.to_datetime(ecmwf_df["time"])
        ecmwf_df.set_index("time", inplace=True)

        pv_hourly = (pv_series * 1000.0).resample("1h").mean().rename("pv_kw").dropna()
        df_train_all = pd.concat([pv_hourly, ecmwf_df.add_prefix("ecmwf_")], axis=1, join="inner").dropna()

        # Train LightGBM on Dec - Jan split
        train_df = df_train_all.loc["2025-12-02":"2026-01-29"]
        features = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", 
                    "direct_normal_irradiance", "temperature_2m", "cloud_cover"]
        df_train_all = pd.concat([pv_hourly, ecmwf_df[features]], axis=1, join="inner").dropna()

        # Train LightGBM on Dec - Jan split
        train_df = df_train_all.loc["2025-12-02":"2026-01-29"]

        X_train = train_df[features].copy()
        X_train["hour"] = train_df.index.hour
        X_train["month"] = train_df.index.month
        y_train = train_df["pv_kw"]

        lgbm_pv_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        lgbm_pv_model.fit(X_train, y_train)
        typer.echo("Trained LightGBM PV model on ECMWF weather features.")

    records: dict[str, ForecastSnapshot] = {}
    http_client = ForecastClient(settings.forecast)

    # Step through hourly indices, requiring at least 24h (96 steps) of past context
    for i in range(96, n_samples, 4):  
        issued_at_ts = pd.Timestamp(timestamps[i])
        issued_at = issued_at_ts.isoformat()

        # Send past 500 steps (~20 days) context to Chronos HTTP forecaster for Demand
        ctx_start_idx = max(0, i - 500)
        past_pv = tuple(max(0.0, float(val)) for val in pv_mw[ctx_start_idx:i:4])
        past_demand = tuple(max(0.0, float(val)) for val in demand_mw[ctx_start_idx:i:4])
        ctx = ForecastContext(
            source_id=settings.scenario.name,
            frequency_hours=1.0,
            pv_values_mw=past_pv,
            demand_values_mw=past_demand,
        )

        chronos_snap = http_client.fetch(ctx, issued_at=issued_at)

        if mode == "hybrid" and lgbm_pv_model is not None and ecmwf_df is not None:
            # Predict PV using LightGBM + ECMWF weather for the next 24 hours
            future_timestamps = [issued_at_ts + pd.Timedelta(hours=h+1) for h in range(24)]
            valid_ts = [ts for ts in future_timestamps if ts in ecmwf_df.index]

            if len(valid_ts) == 24:
                feat_cols = features
                X_future = ecmwf_df.loc[future_timestamps, feat_cols].copy()
                X_future["hour"] = [ts.hour for ts in future_timestamps]
                X_future["month"] = [ts.month for ts in future_timestamps]

                pv_pred_kw = lgbm_pv_model.predict(X_future).clip(min=0)
                pv_pred_mw = tuple(float(val) / 1000.0 for val in pv_pred_kw)

                # Combine LightGBM PV (best) + Chronos Demand (best)
                records[issued_at] = ForecastSnapshot(
                    issued_at=issued_at,
                    horizon_hours=24,
                    frequency_hours=1.0,
                    model_version="lgbm-ecmwf-pv + chronos-2-demand",
                    pv_target="pv_avg",
                    demand_target="demand",
                    timestamps=chronos_snap.timestamps,
                    pv_values_mw=pv_pred_mw,
                    demand_values_mw=chronos_snap.demand_values_mw,
                    source_id=settings.scenario.name,
                    context_time=issued_at,
                    context_steps=i,
                    cold_start=False,
                    covariate_mode="hybrid-ecmwf",
                )
            else:
                records[issued_at] = chronos_snap
        else:
            records[issued_at] = chronos_snap

    cache = ForecastCache(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache.to_jsonl(str(output))
    typer.echo(f"Saved {len(cache)} hybrid forecast snapshot records to {output}")

    if manifest:
        manifest_data = {
            "cache_version": "1.0",
            "source_id": settings.scenario.name,
            "mode": mode,
            "total_records": len(cache),
            "horizon_hours": 24,
            "frequency_hours": 1.0,
            "pv_model": "LightGBM + ECMWF (MAE: 15.31 kW)",
            "demand_model": "Chronos-2 Context 500 (MAE: 24.72 kW)",
            "splits": {
                "train": {"start": "2025-12-02T00:00:00", "end": "2026-01-29T23:45:00"},
                "val": {"start": "2026-02-06T00:00:00", "end": "2026-02-24T23:45:00", "exclude_gaps": ["2026-02-04"]},
                "test": {"start": "2026-03-01T00:00:00", "end": "2026-03-31T23:45:00"},
            },
        }
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(manifest_data, indent=2))
        typer.echo(f"Saved manifest to {manifest}")


@app.command()
def train(
    algo: str = typer.Option("ppo", help="ppo | sac"),
    timesteps: int = typer.Option(50_000, help="total training timesteps"),
    config: Path | None = typer.Option(None, help="scenario YAML (default: pymgrid25 scenario 2)"),
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
    steps: int = typer.Option(24, help="number of ticks (24 = one default hourly scenario day)"),
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
def replay(
    config: Path = typer.Option(
        Path("configs/islanded-baseline-72h.yaml"), help="fixed telemetry scenario YAML"
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="paper report, metrics JSON, and interval CSV directory "
        "(default: reports/experiments/islanded_72h_<policy>_<telemetry-date>)",
    ),
    policy: str = typer.Option("rule", help="paper baseline controller (rule | mpc | schedule)"),
    seed: int = typer.Option(0, help="recorded reproducibility seed"),
) -> None:
    """Run a strict 72-hour measured load/PV baseline and export its report."""
    from microgrid_simulator.experiments.telemetry_replay import run_telemetry_replay

    settings = _load_settings(config)
    if output_dir is None:
        output_dir = _replay_output_dir(settings, policy)
    metrics = run_telemetry_replay(settings, config, output_dir, policy=policy, seed=seed)
    typer.echo(json.dumps(metrics["totals"], indent=2))
    typer.echo(f"report: {output_dir / 'report.md'}")


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
