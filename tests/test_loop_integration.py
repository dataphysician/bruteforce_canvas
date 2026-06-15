from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from bruteforce_canvas.loop import AsyncRunDriver, LoopAction, LoopDecision
from bruteforce_canvas.orchestration import RunRuntimeState


def _running_decision() -> LoopDecision:
    return LoopDecision(
        action=LoopAction.PROPOSE_COORDINATES,
        reason="coordinate_budget_available",
        next_state=RunRuntimeState.RUNNING,
    )


def _stopped_decision() -> LoopDecision:
    return LoopDecision(
        action=LoopAction.STOP,
        reason="stop_requested",
        next_state=RunRuntimeState.STOPPED,
    )


def _driver_with(*, tick: MagicMock, tick_interval: float = 0.01) -> AsyncRunDriver:
    service = MagicMock()
    service.tick.side_effect = tick
    return AsyncRunDriver(service=service, tick_interval=tick_interval)


@pytest.mark.asyncio
async def test_start_begins_ticking_and_updates_counters() -> None:
    tick = MagicMock(return_value=_running_decision())
    driver = _driver_with(tick=tick)

    await driver.start()
    try:
        await asyncio.sleep(0.15)
    finally:
        driver.stop()
        await driver.join()

    assert tick.call_count >= 2
    assert driver.running is False
    assert driver.stopped is True


@pytest.mark.asyncio
async def test_pause_and_resume_pauses_and_resumes_ticking() -> None:
    tick = MagicMock(return_value=_running_decision())
    driver = _driver_with(tick=tick)

    await driver.start()
    try:
        await asyncio.sleep(0.05)
        baseline = tick.call_count

        driver.pause()
        assert driver.paused is True
        await asyncio.sleep(0.15)
        assert tick.call_count == baseline

        driver.resume()
        assert driver.paused is False
        await asyncio.sleep(0.15)
        assert tick.call_count > baseline
    finally:
        driver.stop()
        await driver.join()


@pytest.mark.asyncio
async def test_stop_ends_the_loop_cleanly() -> None:
    tick = MagicMock(return_value=_running_decision())
    driver = _driver_with(tick=tick)

    await driver.start()
    assert driver.running is True
    assert driver.stopped is False

    await asyncio.sleep(0.05)
    driver.stop()
    await driver.join()

    assert driver.running is False
    assert driver.stopped is True
    assert tick.call_count >= 1


@pytest.mark.asyncio
async def test_loop_exits_when_service_decides_stopped() -> None:
    service = MagicMock()
    service.tick.side_effect = [_running_decision(), _stopped_decision()]
    driver = AsyncRunDriver(service=service, tick_interval=0.01)

    await driver.start()
    await driver.join()

    assert service.tick.call_count == 2
    assert driver.running is False
    assert driver.stopped is True


@pytest.mark.asyncio
async def test_cancelling_the_loop_task_shuts_down_cleanly() -> None:
    tick = MagicMock(return_value=_running_decision())
    driver = _driver_with(tick=tick)

    await driver.start()
    task = driver._task
    assert task is not None
    assert not task.done()

    task.cancel()
    await driver.join()

    assert task.cancelled() or task.done()
    assert driver.running is False
    assert driver.stopped is True
