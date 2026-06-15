from bruteforce_canvas.orchestration import (
    RunCounters,
    RunConfig,
    RunRuntimeState,
    stall_guard_decision,
    watermark_decision,
)


def test_stall_guard_stops_unproductive_run_after_configured_window():
    config = RunConfig(run_id="run_001", raw_user_prompt="a bowl on a table")
    counters = RunCounters(generated_count=100, promoted_curated_count=9, elapsed_seconds=1801)

    decision = stall_guard_decision(config, counters)

    assert decision.stop is True
    assert decision.reason == "stall_guard"
    assert decision.details["minimum_required_promoted"] == 10


def test_stall_guard_does_not_stop_before_window_or_when_minimum_met():
    config = RunConfig(run_id="run_001", raw_user_prompt="a bowl on a table")
    early = stall_guard_decision(config, RunCounters(generated_count=100, promoted_curated_count=0, elapsed_seconds=120))
    productive = stall_guard_decision(config, RunCounters(generated_count=100, promoted_curated_count=10, elapsed_seconds=1801))

    assert early.stop is False
    assert productive.stop is False


def test_high_and_low_watermarks_pause_and_resume_when_configured():
    config = RunConfig(
        run_id="run_001",
        raw_user_prompt="a bowl on a table",
        promoted_high_watermark=500,
        promoted_low_watermark=200,
    )

    pause = watermark_decision(
        config,
        RunCounters(promoted_curated_count=500),
        RunRuntimeState.RUNNING,
    )
    resume = watermark_decision(
        config,
        RunCounters(promoted_curated_count=199),
        RunRuntimeState.PAUSED_HIGH_WATERMARK,
    )

    assert pause.next_state == RunRuntimeState.PAUSED_HIGH_WATERMARK
    assert pause.reason == "high_watermark_reached"
    assert resume.next_state == RunRuntimeState.RUNNING
    assert resume.reason == "low_watermark_reached"
