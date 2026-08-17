import asyncio

from barrikade_jentic.events import EventQueue, SecurityEvent


def test_event_queue_overflow_drops_without_blocking():
    delivered = []

    async def sink(event):
        delivered.append(event)

    async def scenario():
        queue = EventQueue(1, sink)
        event = SecurityEvent("barrikade.content_flagged", "warning", "flagged", {})
        assert queue.emit(event) is True
        assert queue.emit(event) is False
        queue.start()
        await queue.close()

    asyncio.run(scenario())
    assert len(delivered) == 1
