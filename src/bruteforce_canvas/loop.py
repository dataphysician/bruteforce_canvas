from __future__ import annotations

from enum import StrEnum

from bruteforce_canvas.orchestration import (
    RunConfig,
    RunCounters,
    RunRuntimeState,
    stall_guard_decision,
    watermark_decision,
)
from bruteforce_canvas.shared import StrictModel


class LoopAction(StrEnum):
    STOP = "stop"
    PAUSE = "pause"
    PROPOSE_COORDINATES = "propose_coordinates"
    GENERATE_PENDING_COORDINATE = "generate_pending_coordinate"
    EVALUATE_PENDING = "evaluate_pending"
    WAIT = "wait"


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
