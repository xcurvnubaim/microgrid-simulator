from __future__ import annotations

import asyncio

import pytest

from microgrid_simulator.contracts import TelemetrySample, TelemetrySessionRequest
from microgrid_simulator.messaging.nats import DashboardTracePublisher, decode_json, encode_json
from microgrid_simulator.simulator.providers import NatsTelemetryProvider


class _Message:
    def __init__(self, data: bytes) -> None:
        self.data = data


class _RequestClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[tuple[str, bytes, float]] = []

    async def request(self, subject: str, data: bytes, timeout: float) -> _Message:
        self.requests.append((subject, data, timeout))
        return _Message(encode_json(self.response))


class _Ack:
    stream = "MGS_DASHBOARD"

    def __init__(self, sequence: int) -> None:
        self.seq = sequence


class _JetStream:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bytes, dict[str, str]]] = []

    async def publish(self, subject: str, data: bytes, headers: dict[str, str]) -> _Ack:
        self.messages.append((subject, data, headers))
        return _Ack(len(self.messages))


def test_dashboard_trace_events_are_ordered_and_causally_linked() -> None:
    async def exercise() -> None:
        js = _JetStream()
        publisher = DashboardTracePublisher(js, "session")

        first = await publisher.publish({"type": "meta", "meta": {}})
        second = await publisher.publish({"type": "row", "row": {"step": 1}})

        assert first["_trace"]["event_sequence"] == 1
        assert first["_trace"]["stream_sequence"] == 1
        assert second["_trace"]["event_sequence"] == 2
        assert second["_trace"]["subject"] == "mgs.v1.dashboard.session.row"
        assert second["_trace"]["producer"] == "simulator"
        assert second["_trace"]["causation_event_id"] == first["_trace"]["event_id"]
        assert js.messages[0][2]["X-Trace-Id"] == first["_trace"]["trace_id"]
        stored = decode_json(js.messages[1][1])
        assert stored["_trace"]["event_id"] == second["_trace"]["event_id"]

    asyncio.run(exercise())


def test_nats_provider_surfaces_service_errors() -> None:
    async def exercise() -> None:
        provider = NatsTelemetryProvider(_RequestClient({"ok": False, "error": "bad window"}))
        with pytest.raises(ConnectionError, match="bad window"):
            await provider.window(TelemetrySessionRequest(n_steps=4))

    asyncio.run(exercise())


def test_json_envelope_serializes_nested_contract_models() -> None:
    sample = TelemetrySample(
        source_id="scenario",
        session_id="session",
        sequence_id=0,
        observed_at="2026-01-15T00:00:00",
        pv_available_kw=10.0,
        load_demand_kw=20.0,
    )

    payload = decode_json(encode_json({"ok": True, "data": sample}))

    assert payload["data"]["session_id"] == "session"
    assert payload["data"]["pv_available_kw"] == 10.0
