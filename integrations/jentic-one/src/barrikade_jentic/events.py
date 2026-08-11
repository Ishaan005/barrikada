"""Bounded, metadata-only local Jentic event delivery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SecurityEvent:
    type: str
    severity: str
    summary: str
    data: dict[str, Any]
    execution_id: str | None = None
    trace_id: str | None = None


class EventQueue:
    def __init__(
        self,
        max_size: int,
        sink: Callable[[SecurityEvent], Awaitable[None]],
    ) -> None:
        self._queue: asyncio.Queue[SecurityEvent] = asyncio.Queue(maxsize=max_size)
        self._sink = sink
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="barrikade-jentic-events")

    def emit(self, event: SecurityEvent) -> bool:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            return False
        return True

    async def close(self) -> None:
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=1.0)
        except TimeoutError:
            # Core owns the authoritative audit record. A stalled local event
            # sink must not delay Jentic shutdown or an execution transaction.
            pass
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._sink(event)
            except Exception:
                # Barrikade Core is authoritative; local event persistence is best effort.
                pass
            finally:
                self._queue.task_done()


def jentic_event_sink(ctx):
    async def sink(event: SecurityEvent) -> None:
        from jentic_one.shared.events import emit_event, valid_trace_id_or_none  # noqa: PLC0415
        from jentic_one.shared.models.events import EventSeverity  # noqa: PLC0415

        severity = EventSeverity(event.severity)
        async with ctx.admin_db.session() as session:
            async with session.begin():
                await emit_event(
                    session,
                    type=event.type,
                    severity=severity,
                    summary=event.summary,
                    created_by="barrikade",
                    data=event.data,
                    execution_id=event.execution_id,
                    trace_id=valid_trace_id_or_none(event.trace_id),
                )

    return sink
