"""Process-local plugin runtime created only by the enabled application bootstrap."""

from __future__ import annotations

from contextlib import asynccontextmanager

from barrikade_jentic.config import BarrikadeConfig
from barrikade_jentic.events import EventQueue, jentic_event_sink
from barrikade_jentic.generated.client import BarrikadeClient


class PluginRuntime:
    def __init__(
        self, config: BarrikadeConfig, ctx, *, client: BarrikadeClient | None = None
    ) -> None:
        self.config = config
        if client is None:
            token = config.token_file.read_text(encoding="utf-8").strip()
            if not token:
                raise RuntimeError("Barrikade service token file is empty")
            client = BarrikadeClient(config.endpoint, token, config.request_timeout_seconds)
        self.client = client
        self.events = EventQueue(config.event_queue_size, jentic_event_sink(ctx))

    def start(self) -> None:
        self.events.start()

    async def close(self) -> None:
        await self.events.close()
        await self.client.close()


_RUNTIME: PluginRuntime | None = None


def set_runtime(runtime: PluginRuntime | None) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def get_runtime() -> PluginRuntime | None:
    return _RUNTIME


def install_runtime_lifecycle(app, ctx) -> None:
    del ctx
    runtime = get_runtime()
    if runtime is None:
        return
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        closed = False

        async def close_runtime() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                await runtime.close()
            finally:
                set_runtime(None)

        runtime.start()
        try:
            async with original_lifespan(application):
                try:
                    yield
                finally:
                    # Drain events while Jentic's admin database is still live.
                    await close_runtime()
        finally:
            # Also release the client when the stock lifespan fails during startup.
            await close_runtime()

    app.router.lifespan_context = lifespan
