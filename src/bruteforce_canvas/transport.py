from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import TypeVar

from bruteforce_canvas.ui import UIStreamEvent

T = TypeVar("T")


class EventBus:
    def __init__(self) -> None:
        self._closed = False
        self._subscribers: list[asyncio.Queue[UIStreamEvent | None]] = []
        self._history: OrderedDict[str, UIStreamEvent] = OrderedDict()

    def publish(self, event: UIStreamEvent) -> None:
        if self._closed:
            return
        self._history[event.event_id] = event.model_copy()
        for queue in self._subscribers:
            queue.put_nowait(event.model_copy())

    async def subscribe(self) -> AsyncIterator[UIStreamEvent]:
        queue: asyncio.Queue[UIStreamEvent | None] = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            for event in self._history.values():
                queue.put_nowait(event.model_copy())
            if self._closed and queue.empty():
                return
            while True:
                if self._closed and queue.empty():
                    break
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            self._subscribers.remove(queue)

    def close(self) -> None:
        self._closed = True
        for queue in list(self._subscribers):
            queue.put_nowait(None)
