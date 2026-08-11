import asyncio
import threading

from barrikade.service.runtime import (
    BoundedInferenceRuntime,
    InferenceDeadlineError,
    InferenceQueueFullError,
)


def test_runtime_rejects_when_running_capacity_is_full():
    async def scenario():
        runtime = BoundedInferenceRuntime(workers=1, queue_size=0)
        release = threading.Event()
        started = threading.Event()

        def blocking():
            started.set()
            release.wait(timeout=2)
            return "done"

        first = asyncio.create_task(runtime.run(1000, blocking))
        await asyncio.to_thread(started.wait, 1)
        try:
            await runtime.run(1000, lambda: "never")
        except InferenceQueueFullError:
            pass
        else:
            raise AssertionError("expected a full admission queue")
        release.set()
        assert await first == "done"
        await runtime.close(1)

    asyncio.run(scenario())


def test_deadline_returns_without_unbounding_background_work():
    async def scenario():
        runtime = BoundedInferenceRuntime(workers=1, queue_size=0)
        release = threading.Event()
        try:
            await runtime.run(1, lambda: release.wait(timeout=1))
        except InferenceDeadlineError:
            pass
        else:
            raise AssertionError("expected the inference deadline to expire")
        try:
            await runtime.run(100, lambda: None)
        except InferenceQueueFullError:
            pass
        else:
            raise AssertionError("timed-out work must retain its admission slot")
        release.set()
        await runtime.close(1)

    asyncio.run(scenario())
