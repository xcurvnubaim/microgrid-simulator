"""Broker abstraction and dependency-free in-process implementation."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from microgrid_simulator.ems.types import DispatchCommand, DispatchResult, TelemetryFrame


class EventBroker(ABC):
    """Typed event boundary; MQTT or durable brokers can implement this API."""

    @abstractmethod
    async def publish_telemetry(self, frame: TelemetryFrame) -> None: ...

    @abstractmethod
    async def receive_telemetry(self) -> TelemetryFrame: ...

    @abstractmethod
    async def publish_command(self, command: DispatchCommand) -> None: ...

    @abstractmethod
    async def receive_command(self) -> DispatchCommand: ...

    @abstractmethod
    async def publish_result(self, result: DispatchResult) -> None: ...

    @abstractmethod
    async def receive_result(self) -> DispatchResult: ...


class InProcessBroker(EventBroker):
    """Bounded asyncio queues for local simulation and deterministic tests."""

    def __init__(self, maxsize: int = 1) -> None:
        self._telemetry: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=maxsize)
        self._commands: asyncio.Queue[DispatchCommand] = asyncio.Queue(maxsize=maxsize)
        # Results are audit events and must not block plant progress when no
        # dashboard/audit consumer is attached.
        self._results: asyncio.Queue[DispatchResult] = asyncio.Queue()

    async def publish_telemetry(self, frame: TelemetryFrame) -> None:
        await self._telemetry.put(frame)

    async def receive_telemetry(self) -> TelemetryFrame:
        return await self._telemetry.get()

    async def publish_command(self, command: DispatchCommand) -> None:
        await self._commands.put(command)

    async def receive_command(self) -> DispatchCommand:
        return await self._commands.get()

    async def publish_result(self, result: DispatchResult) -> None:
        await self._results.put(result)

    async def receive_result(self) -> DispatchResult:
        return await self._results.get()
