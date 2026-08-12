"""Shared NATS subjects, JSON helpers, and JetStream dashboard tracing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from nats.aio.client import Client as NATS
from nats.js.api import RetentionPolicy, StorageType, StreamConfig

TELEMETRY_WINDOW_SUBJECT = "mgs.v1.telemetry.window"
EMS_DISPATCH_SUBJECT = "mgs.v1.ems.dispatch"
EMS_RESULT_SUBJECT = "mgs.v1.ems.result"
PLANT_START_SUBJECT = "mgs.v1.plant.start"
PLANT_STEP_SUBJECT = "mgs.v1.plant.step"
PLANT_CLOSE_SUBJECT = "mgs.v1.plant.close"
DASHBOARD_SUBJECT_PREFIX = "mgs.v1.dashboard"
DASHBOARD_STREAM = "MGS_DASHBOARD"


def encode_json(payload: Any) -> bytes:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")

    def model_default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return json.dumps(payload, separators=(",", ":"), default=model_default).encode("utf-8")


def decode_json(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def trace_subject(session_id: str, event_type: str | None = None) -> str:
    session_token = session_id.replace(".", "-").replace(" ", "-")
    base = f"{DASHBOARD_SUBJECT_PREFIX}.{session_token}"
    return f"{base}.{event_type}" if event_type else f"{base}.>"


async def connect_nats(url: str, *, name: str) -> NATS:
    nc = NATS()
    await nc.connect(
        servers=[url],
        name=name,
        connect_timeout=2,
        reconnect_time_wait=1,
        max_reconnect_attempts=-1,
    )
    return nc


async def ensure_dashboard_stream(nc: NATS) -> Any:
    js = nc.jetstream()
    config = StreamConfig(
        name=DASHBOARD_STREAM,
        subjects=[f"{DASHBOARD_SUBJECT_PREFIX}.>"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.FILE,
        max_age=7 * 24 * 60 * 60,
        duplicate_window=120,
    )
    try:
        await js.stream_info(DASHBOARD_STREAM)
    except Exception:  # noqa: BLE001 - nats-py exception differs by server version
        await js.add_stream(config=config)
    return js


class DashboardTracePublisher:
    """Persist ordered dashboard events with correlation and causation metadata."""

    def __init__(self, js: Any, session_id: str, *, producer: str = "simulator") -> None:
        self.js = js
        self.session_id = session_id
        self.producer = producer
        self.trace_id = str(uuid4())
        self._event_sequence = 0
        self._last_event_id: str | None = None

    async def publish(self, event: dict[str, Any]) -> dict[str, Any]:
        self._event_sequence += 1
        event_type = str(event["type"])
        event_id = f"{self.session_id}:{self._event_sequence}:{event_type}"
        subject = trace_subject(self.session_id, event_type)
        traced = {
            **event,
            "_trace": {
                "trace_id": self.trace_id,
                "event_id": event_id,
                "session_id": self.session_id,
                "event_sequence": self._event_sequence,
                "event_type": event_type,
                "subject": subject,
                "producer": self.producer,
                "produced_at": datetime.now(timezone.utc).isoformat(),
                "causation_event_id": self._last_event_id,
            },
        }
        ack = await self.js.publish(
            subject,
            encode_json(traced),
            headers={"Nats-Msg-Id": event_id, "X-Trace-Id": self.trace_id},
        )
        traced["_trace"]["stream"] = ack.stream
        traced["_trace"]["stream_sequence"] = ack.seq
        self._last_event_id = event_id
        return traced
