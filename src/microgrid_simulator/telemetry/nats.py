"""NATS request/reply worker for historical telemetry windows."""

from __future__ import annotations

import asyncio
from typing import Any

from microgrid_simulator.contracts import TelemetrySessionRequest
from microgrid_simulator.messaging.nats import (
    TELEMETRY_WINDOW_SUBJECT,
    connect_nats,
    decode_json,
    encode_json,
)
from microgrid_simulator.telemetry.service import ReplayTelemetryService


async def run_worker(service: ReplayTelemetryService, nats_url: str) -> None:
    nc = await connect_nats(nats_url, name="microgrid-telemetry")

    async def handle(message: Any) -> None:
        try:
            request = TelemetrySessionRequest.model_validate(decode_json(message.data))
            session = await asyncio.to_thread(service.create_session, request)
            payload = service.get_window(session.session_id)
            await message.respond(encode_json({"ok": True, "data": payload}))
        except Exception as exc:  # noqa: BLE001 - return a typed service error to requester
            await message.respond(encode_json({"ok": False, "error": str(exc)}))

    await nc.subscribe(TELEMETRY_WINDOW_SUBJECT, queue="telemetry-workers", cb=handle)
    await nc.flush()
    try:
        await asyncio.Event().wait()
    finally:
        await nc.drain()
