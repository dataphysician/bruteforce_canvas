from __future__ import annotations

import asyncio

import pytest

from bruteforce_canvas.transport import EventBus
from bruteforce_canvas.ui import UIStreamEvent


@pytest.mark.asyncio
async def test_publish_subscribe_receives_event() -> None:
    bus = EventBus()
    event = UIStreamEvent(
        event_id="evt_001",
        timestamp="2026-01-01T00:00:00Z",
        event_type="run_started",
        run_id="run_001",
        lifecycle_state="running",
        message="run started",
    )

    async def collect() -> list[UIStreamEvent]:
        return [item async for item in bus.subscribe()]

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    bus.publish(event)
    bus.close()
    received = await task

    assert len(received) == 1
    assert received[0].event_id == "evt_001"


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_same_event() -> None:
    bus = EventBus()
    event = UIStreamEvent(
        event_id="evt_002",
        timestamp="2026-01-01T00:00:00Z",
        event_type="run_started",
        run_id="run_002",
        lifecycle_state="running",
        message="tick executed",
    )

    async def collect(subscriber_id: int) -> list[UIStreamEvent]:
        return [item async for item in bus.subscribe()]

    a = asyncio.create_task(collect(1))
    b = asyncio.create_task(collect(2))
    await asyncio.sleep(0)
    bus.publish(event)
    bus.close()
    first = await a
    second = await b

    assert [item.event_id for item in first] == ["evt_002"]
    assert [item.event_id for item in second] == ["evt_002"]


@pytest.mark.asyncio
async def test_close_ends_subscriptions_cleanly() -> None:
    bus = EventBus()

    async def drain() -> list[UIStreamEvent]:
        return [item async for item in bus.subscribe()]

    task = asyncio.create_task(drain())
    bus.close()
    received = await task

    assert received == []


@pytest.mark.asyncio
async def test_publish_after_close_is_ignored() -> None:
    bus = EventBus()
    event = UIStreamEvent(
        event_id="evt_003",
        timestamp="2026-01-01T00:00:00Z",
        event_type="run_started",
        run_id="run_003",
        lifecycle_state="paused",
        message="state changed",
    )

    async def collect() -> list[UIStreamEvent]:
        return [item async for item in bus.subscribe()]

    bus.close()
    bus.publish(event)
    task = asyncio.create_task(collect())
    bus.close()
    received = await task

    assert received == []
