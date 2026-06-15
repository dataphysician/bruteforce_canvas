from bruteforce_canvas.orchestration import (
    RunCounters,
    RunConfig,
    build_stall_diagnostic,
)


def test_stall_diagnostic_persists_counts_failures_penalties_and_restart_hints():
    config = RunConfig(
        run_id="run_001",
        raw_user_prompt="a difficult prompt",
        iqa_cutoff=0.7,
        alignment_cutoff=0.4,
    )
    counters = RunCounters(
        generated_count=100,
        iqa_evaluated_count=100,
        vlm_evaluated_count=30,
        promoted_curated_count=2,
        elapsed_seconds=1801,
    )

    diagnostic = build_stall_diagnostic(
        config,
        counters,
        failure_types=["alignment_below_cutoff", "alignment_below_cutoff", "quality_below_cutoff"],
        penalized_enum_arms={"cinematography.lighting_mood=BLUE_HOUR": 0.12},
        penalized_combos={"lighting=BLUE_HOUR|shot=WIDE_SHOT": -0.6},
    )

    assert diagnostic.run_id == "run_001"
    assert diagnostic.elapsed_seconds == 1801
    assert diagnostic.generated_count == 100
    assert diagnostic.iqa_pass_count == 30
    assert diagnostic.vlm_pass_count == 2
    assert diagnostic.dominant_failure_types[0] == "alignment_below_cutoff"
    assert diagnostic.most_penalized_enum_arms == ["cinematography.lighting_mood=BLUE_HOUR"]
    assert diagnostic.most_penalized_combinations == ["lighting=BLUE_HOUR|shot=WIDE_SHOT"]
    assert "consider_lowering_alignment_cutoff" in diagnostic.restart_hints
    assert diagnostic.threshold_changes_applied is False


def test_stall_diagnostic_suggests_prompt_rehash_when_alignment_dominates():
    diagnostic = build_stall_diagnostic(
        RunConfig(run_id="run_001", raw_user_prompt="unclear thing moving"),
        RunCounters(generated_count=50, iqa_evaluated_count=50, vlm_evaluated_count=40, promoted_curated_count=0, elapsed_seconds=1801),
        failure_types=["missing_locked_element", "missing_locked_relation", "missing_locked_element"],
        penalized_enum_arms={},
        penalized_combos={},
    )

    assert "clarify_or_rehash_prompt" in diagnostic.restart_hints
    assert "narrow_lhs_enum_space" in diagnostic.restart_hints
