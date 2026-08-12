"""FastAPI adapter for the replay telemetry module."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response, status

from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.contracts import (
    TelemetrySample,
    TelemetrySessionRequest,
    TelemetryWindowPayload,
)
from microgrid_simulator.telemetry.service import ReplayTelemetryService


def create_app(settings: Settings | None = None) -> FastAPI:
    service = ReplayTelemetryService(settings or load_settings())
    app = FastAPI(title="Microgrid Telemetry Service")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "schema_version": "1.0"}

    @app.get("/ready")
    def ready() -> dict[str, bool]:
        return {"ready": True}

    @app.post("/sessions", status_code=status.HTTP_201_CREATED)
    def create_session(request: TelemetrySessionRequest) -> dict[str, object]:
        try:
            return service.create_session(request).model_dump(mode="json")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/sessions/{session_id}/window", response_model=TelemetryWindowPayload)
    def window(session_id: str) -> TelemetryWindowPayload:
        try:
            return service.get_window(session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/sessions/{session_id}/next", response_model=TelemetrySample | None)
    def next_sample(session_id: str, response: Response) -> TelemetrySample | None:
        try:
            sample = service.next_sample(session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if sample is None:
            response.status_code = status.HTTP_204_NO_CONTENT
        return sample

    return app
