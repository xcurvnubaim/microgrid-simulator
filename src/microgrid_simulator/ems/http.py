"""HTTP control plane for EMS-owned simulation runs."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from microgrid_simulator.config import Settings
from microgrid_simulator.ems.orchestrator import EMSRunManager
from microgrid_simulator.ems.service import EMSService
from microgrid_simulator.messaging.nats import connect_nats, ensure_dashboard_stream
from microgrid_simulator.plant.providers import NatsPlantProvider
from microgrid_simulator.simulator.providers import NatsTelemetryProvider


class StartRunRequest(BaseModel):
    seed: int = 0


def create_app(
    settings: Settings,
    *,
    policy: str,
    artifact: str | None,
    nats_url: str,
    pace_seconds: float = 0.0,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nc = await connect_nats(nats_url, name="microgrid-ems")
        js = await ensure_dashboard_stream(nc)
        app.state.nc = nc
        app.state.manager = EMSRunManager(
            settings,
            EMSService(settings, policy=policy, artifact=artifact),  # type: ignore[arg-type]
            NatsTelemetryProvider(nc),
            NatsPlantProvider(nc),
            js,
            pace_seconds=pace_seconds,
        )
        try:
            yield
        finally:
            for run in app.state.manager.runs.values():
                if run.task and not run.task.done():
                    run.cancel_requested = True
                    await run.task
            await nc.drain()

    app = FastAPI(title="Microgrid EMS control", lifespan=lifespan)

    def manager() -> EMSRunManager:
        return app.state.manager

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        return {"ready": True, "policy": manager().ems.policy}

    @app.get("/runs/latest")
    async def latest() -> dict[str, Any]:
        run_id = manager().latest_id
        return {"run": manager().get(run_id).status() if run_id else None}

    @app.get("/runs/{run_id}")
    async def status(run_id: str) -> dict[str, Any]:
        try:
            return manager().get(run_id).status()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/runs", status_code=201)
    async def start(request: StartRunRequest) -> dict[str, Any]:
        try:
            return (await manager().start(request.seed)).status()
        except (ValueError, ConnectionError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/runs/{run_id}/continue")
    async def continue_run(run_id: str) -> dict[str, Any]:
        try:
            return (await manager().continue_run(run_id)).status()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, ConnectionError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/runs/{run_id}/finish")
    async def finish(run_id: str) -> dict[str, Any]:
        try:
            run = manager().get(run_id)
            await run.finish()
            return run.status()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/runs/{run_id}/cancel")
    async def cancel(run_id: str) -> dict[str, Any]:
        try:
            return (await manager().cancel(run_id)).status()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app
