"""FastAPI dashboard server — replaces the old Streamlit UI.

Serves a JSON API for the React frontend bundled in ``ui/web/dist``:

    GET  /api/defaults        resolved default settings + demand-trace status
    POST /api/simulate        {settings, policy, seed} -> per-timestep rollout
    POST /api/simulate/stream same, but NDJSON events streamed tick by tick
    POST /api/demand/upload   multipart xlsx/csv -> stored trace + stats
    GET  /api/demand/template example csv in the exact shape the upload expects

Run with ``microgrid-sim dashboard`` (uvicorn under the hood).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from math import pi, sin
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.grid.demand_trace import DemandTrace
from microgrid_simulator.ui.rollout import POLICIES, run_rollout, stream_rollout

LOGGER = logging.getLogger(__name__)

WEB_DIST = Path(__file__).parent / "web" / "dist"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "microgrid-sim-demand"
RL_PRESETS = (
    {
        "label": "v3 seed-0 (60k steps)",
        "artifact": "artifacts/sac/hardunserved-v3-60k/seed-0/sac_microgrid.zip",
        "algo": "sac",
    },
    {
        "label": "v3 seed-1 (60k steps)",
        "artifact": "artifacts/sac/hardunserved-v3-60k/seed-1/sac_microgrid.zip",
        "algo": "sac",
    },
    {
        "label": "v3 seed-2 (60k steps)",
        "artifact": "artifacts/sac/hardunserved-v3-60k/seed-2/sac_microgrid.zip",
        "algo": "sac",
    },
    {
        "label": "verified smoke (60k steps)",
        "artifact": "artifacts/sac/hardunserved-v3-smoke/sac_microgrid.zip",
        "algo": "sac",
    },
)


class SimulateRequest(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    policy: str = "rule"
    seed: int = 0
    rl_artifact: str | None = None
    rl_algo: str | None = None


class StreamRequest(SimulateRequest):
    # Minimum wall time per tick so the live charts animate instead of the
    # whole episode arriving in one burst. 0 streams at solver speed.
    pace_ms: int = Field(default=25, ge=0, le=500)


def _base_settings(policy: str | None = None) -> Settings:
    """Use the shared default scenario, including explicit MGS_CONFIG overrides.

    The ``rl`` policy plays a policy trained on the campus controller-study
    scenario (forecast-enabled, islanded pandapower); the plain default
    scenario has forecasting disabled and produces a different observation
    shape, so it is swapped in for RL playback.

    The RL base honors ``$MGS_CONFIG`` first (so a user can point the dashboard
    at the exact scenario a policy was trained on, e.g. the hard-unserved
    islanded replay), then falls back to the hard-unserved campus controller
    scenario, and finally to the shared default. The old hardcoded plain
    ``islanded-baseline-72h.yaml`` had no ``hard_unserved`` constraint and no
    digital-twin telemetry replay, so policies trained under the hard-unserved
    contract saw the synthetic sinusoid instead — the source of dashboard
    blackouts.
    """
    if policy == "rl":
        from microgrid_simulator.config import Settings as S

        env_cfg = os.environ.get("MGS_CONFIG")
        if env_cfg and Path(env_cfg).is_file():
            return S.from_yaml(Path(env_cfg))
        campus = Path(__file__).resolve().parents[3] / "configs" / "islanded-baseline-72h-hardunserved.yaml"
        if campus.is_file():
            return S.from_yaml(campus)
        campus = Path(__file__).resolve().parents[3] / "configs" / "islanded-baseline-72h.yaml"
        if campus.is_file():
            return S.from_yaml(campus)
    return load_settings()


def _merge_settings(overrides: dict[str, Any], policy: str | None = None) -> Settings:
    raw = _base_settings(policy).model_dump()
    # The artifact supplies the controller, not the plant. Dashboard settings
    # therefore remain authoritative for RL just as they are for rule and MPC
    # playback. The policy-specific base only gives the UI a compatible initial
    # scenario; every explicit dashboard edit is applied below.
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return Settings(**raw)


def _playback_settings(overrides: dict[str, Any], policy: str) -> Settings:
    settings = _merge_settings(overrides, policy)
    if policy == "rl" and settings.rl.hard_unserved:
        # Hard unserved-load termination is a training constraint. Dashboard
        # inference should finish the requested horizon so the user can inspect
        # the complete blackout and unserved-energy trajectory.
        settings = settings.model_copy(
            update={"rl": settings.rl.model_copy(update={"hard_unserved": False})}
        )
    return settings


def _demand_template_csv(settings: Settings) -> str:
    """Example demand export in the exact shape the upload parser expects:
    ``header_row`` note rows first, then the header, then one row per tick."""
    cfg = settings.demand
    step_hours = settings.topology.timestep_hours
    unit = cfg.unit.lower()

    note = (
        f"# Demand trace template — keep this note row: the header must stay on "
        f"row {cfg.header_row + 1}. Replace the sample rows with your own "
        f"timestamps and total demand in {unit.upper()}."
    )
    lines = [f"{note},"] * cfg.header_row
    lines.append(f"{cfg.timestamp_column},{cfg.value_column}")

    start = datetime(2025, 1, 1)
    n_steps = int(round(48.0 / step_hours))  # two example days
    for i in range(n_steps):
        t = start + timedelta(hours=i * step_hours)
        hour = (i * step_hours) % 24.0
        # Plausible campus shape: ~180 kW base with a daytime bump and a bit of ripple.
        kw = 180.0 + 160.0 * max(0.0, sin(pi * (hour - 7.0) / 13.0)) + 8.0 * sin(hour * 2.1)
        value = kw if unit == "kw" else kw / 1000.0
        lines.append(f"{t:%Y-%m-%d %H:%M:%S},{value:.1f}")
    return "\n".join(lines) + "\n"


def _demand_status(settings: Settings) -> dict[str, Any]:
    trace = DemandTrace.from_file(settings.demand, settings.topology.timestep_hours)
    if trace is None:
        return {"available": False, "stats": None}
    return {"available": True, "stats": trace.stats()}


def _defaults_payload() -> dict[str, Any]:
    settings = _base_settings()
    return {
        "settings": settings.model_dump(),
        "policy_settings": {"rl": _base_settings("rl").model_dump()},
        "policies": list(POLICIES),
        "rl_presets": list(RL_PRESETS),
        "demand": _demand_status(settings),
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="Microgrid Simulator", docs_url="/api/docs", openapi_url="/api/openapi.json"
    )

    @app.get("/api/defaults")
    def defaults() -> dict[str, Any]:
        return _defaults_payload()

    @app.post("/api/simulate")
    def simulate(req: SimulateRequest) -> dict[str, Any]:
        if req.policy not in POLICIES:
            raise HTTPException(422, f"policy must be one of {POLICIES}")
        if req.policy == "rl" and not req.rl_artifact:
            raise HTTPException(422, "policy 'rl' requires rl_artifact")
        try:
            settings = _playback_settings(req.settings, req.policy)
        except Exception as exc:  # noqa: BLE001 - surface pydantic detail to the UI
            raise HTTPException(422, f"invalid settings: {exc}") from exc

        result = run_rollout(
            settings,
            policy=req.policy,
            seed=req.seed,
            rl_artifact=req.rl_artifact,
            rl_algo=req.rl_algo,
        )
        result["demand"] = _demand_status(settings)
        return result

    @app.post("/api/simulate/stream")
    def simulate_stream(req: StreamRequest) -> StreamingResponse:
        if req.policy not in POLICIES:
            raise HTTPException(422, f"policy must be one of {POLICIES}")
        if req.policy == "rl" and not req.rl_artifact:
            raise HTTPException(422, "policy 'rl' requires rl_artifact")
        try:
            settings = _playback_settings(req.settings, req.policy)
        except Exception as exc:  # noqa: BLE001 - surface pydantic detail to the UI
            raise HTTPException(422, f"invalid settings: {exc}") from exc

        pace = req.pace_ms / 1000.0

        def ndjson() -> Iterator[str]:
            last = time.monotonic()
            for event in stream_rollout(
                settings,
                policy=req.policy,
                seed=req.seed,
                rl_artifact=req.rl_artifact,
                rl_algo=req.rl_algo,
            ):
                if pace and event["type"] == "row":
                    now = time.monotonic()
                    wait = pace - (now - last)
                    if wait > 0:
                        time.sleep(wait)
                    last = time.monotonic()
                yield json.dumps(event) + "\n"

        return StreamingResponse(
            ndjson(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/demand/template")
    def demand_template() -> Response:
        return Response(
            content=_demand_template_csv(_base_settings()),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="demand_template.csv"'},
        )

    @app.post("/api/demand/upload")
    async def upload_demand(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        suffix = Path(file.filename or "demand.xlsx").suffix.lower()
        if suffix not in {".xlsx", ".xlsm", ".xls", ".csv"}:
            raise HTTPException(422, "expected an .xlsx or .csv demand export")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / f"demand{suffix}"
        dest.write_bytes(await file.read())

        settings = _base_settings()
        cfg = settings.demand.model_copy(update={"file": str(dest), "enabled": True})
        trace = DemandTrace.from_file(cfg, settings.topology.timestep_hours)
        if trace is None:
            raise HTTPException(
                422,
                "could not parse the file — expected columns "
                f"'{cfg.timestamp_column}' and '{cfg.value_column}' with the "
                f"header on row {cfg.header_row + 1}",
            )
        return {"file": str(dest), "stats": trace.stats()}

    # --- static frontend (built React app) ---------------------------------
    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = WEB_DIST / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")
    else:  # pragma: no cover - dev convenience

        @app.get("/", include_in_schema=False)
        def missing() -> dict[str, str]:
            return {
                "detail": "frontend not built — run `npm install && npm run build` "
                "in src/microgrid_simulator/ui/web, or use the /api endpoints"
            }

    return app


app = create_app()
