from bruteforce_canvas.loop import LoopAction, next_loop_action
from bruteforce_canvas.orchestration import RunCounters, RunConfig, RunRuntimeState


def test_loop_controller_stops_on_stall_before_generating_more():
    action = next_loop_action(
        RunConfig(run_id="run_001", raw_user_prompt="a bowl"),
        RunCounters(generated_count=100, promoted_curated_count=0, elapsed_seconds=1801),
        RunRuntimeState.RUNNING,
        stop_requested=False,
        has_pending_coordinates=False,
        has_pending_candidates=False,
    )

    assert action.action == LoopAction.STOP
    assert action.reason == "stall_guard"


def test_loop_controller_pauses_on_high_watermark_before_new_coordinates():
    action = next_loop_action(
        RunConfig(
            run_id="run_001",
            raw_user_prompt="a bowl",
            promoted_high_watermark=10,
            promoted_low_watermark=3,
        ),
        RunCounters(promoted_curated_count=10, elapsed_seconds=100),
        RunRuntimeState.RUNNING,
        stop_requested=False,
        has_pending_coordinates=False,
        has_pending_candidates=False,
    )

    assert action.action == LoopAction.PAUSE
    assert action.reason == "high_watermark_reached"


def test_loop_controller_resumes_from_low_watermark_pause():
    action = next_loop_action(
        RunConfig(
            run_id="run_001",
            raw_user_prompt="a bowl",
            promoted_high_watermark=10,
            promoted_low_watermark=3,
        ),
        RunCounters(promoted_curated_count=2, elapsed_seconds=100),
        RunRuntimeState.PAUSED_HIGH_WATERMARK,
        stop_requested=False,
        has_pending_coordinates=False,
        has_pending_candidates=False,
    )

    assert action.action == LoopAction.PROPOSE_COORDINATES
    assert action.next_state == RunRuntimeState.RUNNING


def test_loop_controller_waits_while_manually_paused():
    action = next_loop_action(
        RunConfig(run_id="run_001", raw_user_prompt="a bowl"),
        RunCounters(generated_count=10, promoted_curated_count=5, elapsed_seconds=100),
        RunRuntimeState.PAUSED,
        stop_requested=False,
        has_pending_coordinates=True,
        has_pending_candidates=True,
    )

    assert action.action == LoopAction.WAIT
    assert action.reason == "paused"
    assert action.next_state == RunRuntimeState.PAUSED


def test_loop_controller_prioritizes_pending_candidates_over_new_routing():
    action = next_loop_action(
        RunConfig(run_id="run_001", raw_user_prompt="a bowl"),
        RunCounters(generated_count=10, promoted_curated_count=10, elapsed_seconds=100),
        RunRuntimeState.RUNNING,
        stop_requested=False,
        has_pending_coordinates=True,
        has_pending_candidates=True,
    )

    assert action.action == LoopAction.EVALUATE_PENDING


def test_loop_controller_generates_existing_coordinates_before_proposing_more():
    action = next_loop_action(
        RunConfig(run_id="run_001", raw_user_prompt="a bowl"),
        RunCounters(generated_count=10, promoted_curated_count=10, elapsed_seconds=100),
        RunRuntimeState.RUNNING,
        stop_requested=False,
        has_pending_coordinates=True,
        has_pending_candidates=False,
    )

    assert action.action == LoopAction.GENERATE_PENDING_COORDINATE
