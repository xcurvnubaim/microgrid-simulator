"""Client adapter for the headless plant NATS service."""

from __future__ import annotations

from nats.errors import NoRespondersError
from nats.errors import TimeoutError as NatsTimeoutError

from microgrid_simulator.contracts import (
    PlantSessionCloseRequest,
    PlantSessionCloseResponse,
    PlantSessionStartRequest,
    PlantSessionStartResponse,
    PlantStepRequest,
    PlantStepResponse,
)
from microgrid_simulator.messaging.nats import (
    PLANT_CLOSE_SUBJECT,
    PLANT_START_SUBJECT,
    PLANT_STEP_SUBJECT,
    decode_json,
    encode_json,
)


def _response(data: bytes) -> object:
    payload = decode_json(data)
    if not isinstance(payload, dict) or not payload.get("ok"):
        error = (
            payload.get("error", "invalid plant response")
            if isinstance(payload, dict)
            else payload
        )
        raise ConnectionError(str(error))
    return payload.get("data")


class NatsPlantProvider:
    def __init__(self, nc: object, timeout_seconds: float = 30.0) -> None:
        self.nc = nc
        self.timeout_seconds = timeout_seconds

    async def _request(self, subject: str, payload: object) -> object:
        try:
            message = await self.nc.request(  # type: ignore[attr-defined]
                subject, encode_json(payload), timeout=self.timeout_seconds
            )
        except (NatsTimeoutError, NoRespondersError) as exc:
            raise ConnectionError(f"plant NATS request failed: {exc}") from exc
        return _response(message.data)

    async def start(self, request: PlantSessionStartRequest) -> PlantSessionStartResponse:
        return PlantSessionStartResponse.model_validate(
            await self._request(PLANT_START_SUBJECT, request)
        )

    async def step(self, request: PlantStepRequest) -> PlantStepResponse:
        return PlantStepResponse.model_validate(await self._request(PLANT_STEP_SUBJECT, request))

    async def close(self, session_id: str) -> PlantSessionCloseResponse:
        return PlantSessionCloseResponse.model_validate(
            await self._request(
                PLANT_CLOSE_SUBJECT, PlantSessionCloseRequest(session_id=session_id)
            )
        )
