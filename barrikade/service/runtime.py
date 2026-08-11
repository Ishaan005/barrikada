"""Bounded, deadline-aware execution of blocking detector inference."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar


T = TypeVar("T")


class InferenceQueueFullError(RuntimeError):
    pass


class InferenceDeadlineError(RuntimeError):
    pass


class InferenceUnavailableError(RuntimeError):
    pass


class BoundedInferenceRuntime:
    def __init__(self, workers: int = 1, queue_size: int = 32) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="barrikade")
        self._admission = threading.BoundedSemaphore(workers + queue_size)
        self._closed = False
        self._state_lock = threading.Lock()

    async def run(self, deadline_ms: int, operation: Callable[[], T]) -> T:
        with self._state_lock:
            if self._closed:
                raise InferenceUnavailableError("inference runtime is shutting down")
            admitted = self._admission.acquire(blocking=False)
        if not admitted:
            raise InferenceQueueFullError("inference admission queue is full")

        loop = asyncio.get_running_loop()

        def execute() -> T:
            try:
                return operation()
            finally:
                self._admission.release()

        future = loop.run_in_executor(self._executor, execute)
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=deadline_ms / 1000)
        except TimeoutError as exc:
            raise InferenceDeadlineError("assessment deadline exceeded") from exc

    async def close(self, grace_seconds: float) -> None:
        with self._state_lock:
            self._closed = True
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, lambda: self._executor.shutdown(wait=True)),
            timeout=max(0.001, grace_seconds),
        )
