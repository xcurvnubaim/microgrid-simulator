"""NATS request/reply adapter for the headless plant."""

from __future__ import annotations

import asyncio
from typing import Any

from microgrid_simulator.contracts import (
    PlantSessionCloseRequest,
    PlantSessionStartRequest,
    PlantStepRequest,
)
from microgrid_simulator.messaging.nats import (
    PLANT_CLOSE_SUBJECT,
    PLANT_START_SUBJECT,
    PLANT_STEP_SUBJECT,
    connect_nats,
    decode_json,
    encode_json,
)
from microgrid_simulator.plant.service import PlantService


async def run_worker(service: PlantService, nats_url: str) -> None:
    nc = await connect_nats(nats_url, name="microgrid-plant")

    async def reply(message: Any, model: type[Any], handler: Any) -> None:
        try:
            request = model.model_validate(decode_json(message.data))
            response = await handler(request)
            await message.respond(encode_json({"ok": True, "data": response}))
        except Exception as exc:  # noqa: BLE001
            await message.respond(encode_json({"ok": False, "error": str(exc)}))

    async def start(message: Any) -> None:
        await reply(message, PlantSessionStartRequest, service.start)

    async def step(message: Any) -> None:
        await reply(message, PlantStepRequest, service.step)

    await nc.subscribe(PLANT_START_SUBJECT, queue="plant-workers", cb=start)
    await nc.subscribe(PLANT_STEP_SUBJECT, queue="plant-workers", cb=step)

    async def close(request: PlantSessionCloseRequest) -> Any:
        return await service.close(request.session_id)

    async def close_message(message: Any) -> None:
        await reply(message, PlantSessionCloseRequest, close)

    await nc.subscribe(PLANT_CLOSE_SUBJECT, queue="plant-workers", cb=close_message)
    await nc.flush()
    try:
        await asyncio.Event().wait()
    finally:
        await nc.drain()
