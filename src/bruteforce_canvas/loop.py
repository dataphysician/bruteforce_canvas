from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import TYPE_CHECKING

from bruteforce_canvas.orchestration import (
    RunConfig,
    RunCounters,
    RunRuntimeState,
    stall_guard_decision,
    watermark_decision,
)
from bruteforce_canvas.shared import StrictModel

if TYPE_CHECKING:
    from bruteforce_canvas.run_service import RunService


class LoopAction(StrEnum):
    STOP = "stop"
    PAUSE = "pause"
    PROPOSE_COORDINATES = "propose_coordinates"
    GENERATE_PENDING_COORDINATE = "generate_pending_coordinate"
    EVALUATE_PENDING = "evaluate_pending"
    WAIT = "wait"
    GATE_BLOCKED = "gate_blocked"


class LoopDecision(StrictModel):
    action: LoopAction
    reason: str
    next_state: RunRuntimeState


def _runtime_state(value: RunRuntimeState | str) -> RunRuntimeState:
    return value if isinstance(value, RunRuntimeState) else RunRuntimeState(value)


def next_loop_action(
    config: RunConfig,
    counters: RunCounters,
    current_state: RunRuntimeState,
    *,
    stop_requested: bool,
    has_pending_coordinates: bool,
    has_pending_candidates: bool,
) -> LoopDecision:
    if stop_requested:
        return LoopDecision(action=LoopAction.STOP, reason="stop_requested", next_state=RunRuntimeState.STOPPED)

    stall = stall_guard_decision(config, counters)
    if stall.stop:
        return LoopDecision(action=LoopAction.STOP, reason=stall.reason, next_state=RunRuntimeState.STOPPED)

    watermark = watermark_decision(config, counters, current_state)
    watermark_state = _runtime_state(watermark.next_state) if watermark.next_state is not None else current_state
    if watermark_state == RunRuntimeState.PAUSED_HIGH_WATERMARK and current_state == RunRuntimeState.RUNNING:
        return LoopDecision(action=LoopAction.PAUSE, reason=watermark.reason, next_state=watermark_state)

    state = watermark_state
    if state == RunRuntimeState.PAUSED:
        return LoopDecision(action=LoopAction.WAIT, reason="paused", next_state=state)
    if state == RunRuntimeState.PAUSED_HIGH_WATERMARK:
        return LoopDecision(action=LoopAction.WAIT, reason="paused_high_watermark", next_state=state)

    if has_pending_candidates:
        return LoopDecision(action=LoopAction.EVALUATE_PENDING, reason="pending_candidates", next_state=RunRuntimeState.RUNNING)
    if has_pending_coordinates:
        return LoopDecision(
            action=LoopAction.GENERATE_PENDING_COORDINATE,
            reason="pending_coordinates",
            next_state=RunRuntimeState.RUNNING,
        )
    return LoopDecision(
        action=LoopAction.PROPOSE_COORDINATES,
        reason="coordinate_budget_available",
        next_state=RunRuntimeState.RUNNING,
    )


class AsyncRunDriver:
    """Asynchronous loop driver that ticks a :class:`RunService` on an interval.

    The driver is bound to the app lifecycle: it holds no persistence, owns a
    single :class:`asyncio.Task`, and exits cleanly on ``stop()``, on a
    ``STOPPED`` decision from the service, on an unhandled exception, or on
    task cancellation.
    """

    def __init__(self, *, service: RunService, tick_interval: float = 1.0) -> None:
        self._service = service
        self._tick_interval = tick_interval
        self._paused = False
        self._stop_requested = False
        self._task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    @property
    def stopped(self) -> bool:
        return self._stop_requested or self._task is None or self._task.done()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_requested = False
        self._paused = False
        self._running = True
        self._task = asyncio.create_task(self._run(), name="AsyncRunDriver")

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._stop_requested = True

    async def join(self) -> None:
        task = self._task
        if task is None:
            return
        if task.done():
            return
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        try:
            while not self._stop_requested:
                await asyncio.sleep(self._tick_interval)
                if self._stop_requested:
                    break
                if self._paused:
                    continue
                decision = self._service.tick()
                if decision.next_state == RunRuntimeState.STOPPED:
                    break
        except asyncio.CancelledError:
            self._stop_requested = True
        except Exception:
            self._stop_requested = True
        finally:
            self._running = False
