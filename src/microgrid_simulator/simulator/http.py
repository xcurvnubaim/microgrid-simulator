"""FastAPI adapter and event stream for modular simulator sessions."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from nats.errors import TimeoutError as NatsTimeoutError
from pydantic import BaseModel, ConfigDict, Field

from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.ems.service import EMSService
from microgrid_simulator.messaging.nats import (
    DASHBOARD_STREAM,
    DashboardTracePublisher,
    connect_nats,
    decode_json,
    ensure_dashboard_stream,
    trace_subject,
)
from microgrid_simulator.simulator.orchestrator import SimulatorOrchestrator
from microgrid_simulator.simulator.providers import (
    HttpDispatchProvider,
    HttpTelemetryProvider,
    LocalDispatchProvider,
    LocalTelemetryProvider,
    NatsDispatchProvider,
    NatsTelemetryProvider,
)
from microgrid_simulator.telemetry.service import ReplayTelemetryService


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int | None = Field(default=None, ge=1)
    seed: int = 0
    pace_ms: int = Field(default=0, ge=0, le=60_000)


@dataclass
class SimulationSession:
    orchestrator: SimulatorOrchestrator
    request: SimulationRequest
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)
    task: asyncio.Task[list[Any]] | None = None
    error: str | None = None
    trace_publisher: DashboardTracePublisher | None = None

    async def publish(self, event: dict[str, Any]) -> None:
        if self.trace_publisher is not None:
            event = await self.trace_publisher.publish(event)
        self.events.append(event)
        for queue in tuple(self.subscribers):
            await queue.put(event)

    async def run(self) -> None:
        try:
            await self.orchestrator.run(
                max_steps=self.request.steps,
                seed=self.request.seed,
                on_event=self.publish,
            )
        except Exception as exc:  # noqa: BLE001 - session exposes structured failure
            self.error = str(exc)
            await self.publish({"type": "error", "message": str(exc)})


def create_app(
    settings: Settings | None = None,
    *,
    telemetry_url: str | None = None,
    ems_url: str | None = None,
    nats_url: str | None = None,
    local_policy: Literal["rule", "sac", "ppo"] = "rule",
    local_artifact: str | Path | None = None,
    command_timeout_ms: int = 1000,
) -> FastAPI:
    active_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.nats = None
        app.state.jetstream = None
        if nats_url:
            app.state.nats = await connect_nats(nats_url, name="microgrid-simulator")
            app.state.jetstream = await ensure_dashboard_stream(app.state.nats)
        yield
        if app.state.nats is not None:
            await app.state.nats.drain()

    app = FastAPI(title="Microgrid Simulator Service", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.sessions = {}

    def orchestrator(pace_seconds: float) -> SimulatorOrchestrator:
        if app.state.nats is not None:
            telemetry = NatsTelemetryProvider(app.state.nats)
            dispatch = NatsDispatchProvider(
                app.state.nats, timeout_seconds=command_timeout_ms / 1000.0
            )
        else:
            telemetry = (
                HttpTelemetryProvider(telemetry_url)
                if telemetry_url
                else LocalTelemetryProvider(ReplayTelemetryService(active_settings))
            )
            dispatch = (
                HttpDispatchProvider(ems_url, timeout_seconds=command_timeout_ms / 1000.0)
                if ems_url
                else LocalDispatchProvider(
                    EMSService(
                        active_settings,
                        policy=local_policy,
                        artifact=local_artifact,
                        command_ttl_ms=command_timeout_ms,
                    )
                )
            )
        return SimulatorOrchestrator(
            active_settings,
            telemetry,
            dispatch,
            command_timeout_ms=command_timeout_ms,
            pace_seconds=pace_seconds,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "schema_version": "1.0"}

    @app.get("/ready")
    def ready() -> dict[str, object]:
        nats_ready = app.state.nats is None or app.state.nats.is_connected
        return {
            "ready": nats_ready,
            "telemetry_mode": (
                "nats" if nats_url else "http" if telemetry_url else "in-process"
            ),
            "ems_mode": "nats" if nats_url else "http" if ems_url else "in-process",
            "trace_store": "jetstream" if app.state.jetstream is not None else "memory",
        }

    @app.post("/simulations", status_code=status.HTTP_201_CREATED)
    async def start_simulation(request: SimulationRequest) -> dict[str, str]:
        runner = orchestrator(request.pace_ms / 1000.0)
        publisher = (
            DashboardTracePublisher(app.state.jetstream, runner.session_id)
            if app.state.jetstream is not None
            else None
        )
        session = SimulationSession(runner, request, trace_publisher=publisher)
        app.state.sessions[runner.session_id] = session
        session.task = asyncio.create_task(session.run())
        return {"session_id": runner.session_id, "status": "running"}

    @app.get("/simulations/{session_id}")
    def simulation_status(session_id: str) -> dict[str, object]:
        session = app.state.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, f"unknown simulation session {session_id}")
        running = session.task is not None and not session.task.done()
        return {
            "session_id": session_id,
            "status": "failed" if session.error else "running" if running else "complete",
            "event_count": len(session.events),
            "error": session.error,
        }

    @app.websocket("/simulations/{session_id}/events")
    async def simulation_events(websocket: WebSocket, session_id: str) -> None:
        session = app.state.sessions.get(session_id)
        if session is None and app.state.jetstream is None:
            await websocket.close(code=4404, reason="unknown simulation session")
            return
        await websocket.accept()
        if app.state.jetstream is not None:
            subscription = await app.state.jetstream.pull_subscribe(
                trace_subject(session_id),
                stream=DASHBOARD_STREAM,
            )
            try:
                while True:
                    messages = await subscription.fetch(1, timeout=30)
                    message = messages[0]
                    event = decode_json(message.data)
                    metadata = message.metadata
                    trace = event.setdefault("_trace", {})
                    trace.update(
                        {
                            "stream": metadata.stream,
                            "stream_sequence": metadata.sequence.stream,
                            "stored_at": metadata.timestamp.astimezone(timezone.utc).isoformat(),
                            "redelivered": metadata.num_delivered > 1,
                        }
                    )
                    await websocket.send_json(event)
                    await message.ack()
                    if event["type"] in {"ems_end", "error"}:
                        return
            except (NatsTimeoutError, WebSocketDisconnect):
                return
            finally:
                await subscription.unsubscribe()
            return

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        session.subscribers.append(queue)
        try:
            for event in session.events:
                await websocket.send_json(event)
            if session.task is not None and session.task.done():
                return
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                if event["type"] in {"ems_end", "error"}:
                    return
        except WebSocketDisconnect:
            return
        finally:
            session.subscribers.remove(queue)

    return app
