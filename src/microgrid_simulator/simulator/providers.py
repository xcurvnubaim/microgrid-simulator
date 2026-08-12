"""Local, NATS, and legacy HTTP provider adapters for the simulator."""

from __future__ import annotations

import asyncio
import json
from typing import Protocol
from urllib import error, request

from nats.errors import NoRespondersError
from nats.errors import TimeoutError as NatsTimeoutError

from microgrid_simulator.contracts import (
    DispatchCommand,
    DispatchResult,
    PlantObservation,
    TelemetrySessionRequest,
    TelemetryWindowPayload,
)
from microgrid_simulator.ems.service import EMSService
from microgrid_simulator.messaging.nats import (
    EMS_DISPATCH_SUBJECT,
    EMS_RESULT_SUBJECT,
    TELEMETRY_WINDOW_SUBJECT,
    decode_json,
    encode_json,
)
from microgrid_simulator.telemetry.service import ReplayTelemetryService


class TelemetryProvider(Protocol):
    async def window(self, request: TelemetrySessionRequest) -> TelemetryWindowPayload: ...


class DispatchProvider(Protocol):
    async def dispatch(self, observation: PlantObservation) -> DispatchCommand: ...

    async def record_result(self, result: DispatchResult) -> None: ...


class LocalTelemetryProvider:
    def __init__(self, service: ReplayTelemetryService) -> None:
        self.service = service

    async def window(self, request: TelemetrySessionRequest) -> TelemetryWindowPayload:
        session = await asyncio.to_thread(self.service.create_session, request)
        return self.service.get_window(session.session_id)


class LocalDispatchProvider:
    def __init__(self, service: EMSService) -> None:
        self.service = service
        self.results: list[DispatchResult] = []

    async def dispatch(self, observation: PlantObservation) -> DispatchCommand:
        return await asyncio.to_thread(
            self.service.build_command, observation.to_telemetry_frame()
        )

    async def record_result(self, result: DispatchResult) -> None:
        self.results.append(result)


def _nats_response(data: bytes) -> object:
    payload = decode_json(data)
    if not isinstance(payload, dict) or not payload.get("ok"):
        message = (
            payload.get("error", "invalid NATS service response")
            if isinstance(payload, dict)
            else "invalid NATS service response"
        )
        raise ConnectionError(str(message))
    return payload.get("data")


class NatsTelemetryProvider:
    def __init__(self, nc: object, timeout_seconds: float = 30.0) -> None:
        self.nc = nc
        self.timeout_seconds = timeout_seconds

    async def window(self, request_model: TelemetrySessionRequest) -> TelemetryWindowPayload:
        try:
            message = await self.nc.request(  # type: ignore[attr-defined]
                TELEMETRY_WINDOW_SUBJECT,
                encode_json(request_model),
                timeout=self.timeout_seconds,
            )
        except (NatsTimeoutError, NoRespondersError) as exc:
            raise ConnectionError(f"telemetry NATS request failed: {exc}") from exc
        return TelemetryWindowPayload.model_validate(_nats_response(message.data))


class NatsDispatchProvider:
    def __init__(self, nc: object, timeout_seconds: float = 1.0) -> None:
        self.nc = nc
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, observation: PlantObservation) -> DispatchCommand:
        try:
            message = await self.nc.request(  # type: ignore[attr-defined]
                EMS_DISPATCH_SUBJECT,
                encode_json(observation),
                timeout=self.timeout_seconds,
            )
        except (NatsTimeoutError, NoRespondersError) as exc:
            raise ConnectionError(f"EMS NATS request failed: {exc}") from exc
        return DispatchCommand.model_validate(_nats_response(message.data))

    async def record_result(self, result: DispatchResult) -> None:
        try:
            await self.nc.request(  # type: ignore[attr-defined]
                EMS_RESULT_SUBJECT,
                encode_json(result),
                timeout=self.timeout_seconds,
            )
        except (NatsTimeoutError, NoRespondersError):
            return


class _HttpClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def send(self, method: str, path: str, body: dict[str, object] | None = None) -> object:
        encoded = json.dumps(body).encode("utf-8") if body is not None else None
        req = request.Request(
            f"{self.base_url}{path}",
            data=encoded,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:  # noqa: S310
                raw = response.read()
        except (error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
            raise ConnectionError(f"service request failed: {method} {path}: {exc}") from exc
        return json.loads(raw) if raw else None


class HttpTelemetryProvider:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.client = _HttpClient(base_url, timeout_seconds)

    async def window(self, request_model: TelemetrySessionRequest) -> TelemetryWindowPayload:
        session = await asyncio.to_thread(
            self.client.send,
            "POST",
            "/sessions",
            request_model.model_dump(mode="json"),
        )
        session_id = str(session["session_id"])  # type: ignore[index]
        payload = await asyncio.to_thread(
            self.client.send, "GET", f"/sessions/{session_id}/window"
        )
        return TelemetryWindowPayload.model_validate(payload)


class HttpDispatchProvider:
    def __init__(self, base_url: str, timeout_seconds: float = 1.0) -> None:
        self.client = _HttpClient(base_url, timeout_seconds)

    async def dispatch(self, observation: PlantObservation) -> DispatchCommand:
        payload = await asyncio.to_thread(
            self.client.send,
            "POST",
            "/dispatch",
            observation.model_dump(mode="json"),
        )
        return DispatchCommand.model_validate(payload)

    async def record_result(self, result: DispatchResult) -> None:
        try:
            await asyncio.to_thread(
                self.client.send,
                "POST",
                "/dispatch-results",
                result.model_dump(mode="json"),
            )
        except ConnectionError:
            return
