"""NATS request/reply worker for EMS inference and result audit."""

from __future__ import annotations

import asyncio
from typing import Any

from microgrid_simulator.contracts import DispatchResult, PlantObservation
from microgrid_simulator.ems.service import EMSService
from microgrid_simulator.messaging.nats import (
    EMS_DISPATCH_SUBJECT,
    EMS_RESULT_SUBJECT,
    connect_nats,
    decode_json,
    encode_json,
)


async def run_worker(service: EMSService, nats_url: str) -> None:
    nc = await connect_nats(nats_url, name="microgrid-ems")
    results: list[DispatchResult] = []

    async def dispatch(message: Any) -> None:
        try:
            observation = PlantObservation.model_validate(decode_json(message.data))
            command = await asyncio.to_thread(
                service.build_command, observation.to_telemetry_frame()
            )
            await message.respond(encode_json({"ok": True, "data": command}))
        except Exception as exc:  # noqa: BLE001 - return a typed service error to requester
            await message.respond(encode_json({"ok": False, "error": str(exc)}))

    async def record_result(message: Any) -> None:
        try:
            results.append(DispatchResult.model_validate(decode_json(message.data)))
            await message.respond(encode_json({"ok": True, "data": {"accepted": True}}))
        except Exception as exc:  # noqa: BLE001
            await message.respond(encode_json({"ok": False, "error": str(exc)}))

    await nc.subscribe(EMS_DISPATCH_SUBJECT, queue="ems-workers", cb=dispatch)
    await nc.subscribe(EMS_RESULT_SUBJECT, queue="ems-workers", cb=record_result)
    await nc.flush()
    try:
        await asyncio.Event().wait()
    finally:
        await nc.drain()
