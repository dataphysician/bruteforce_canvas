from bruteforce_canvas.actions import ActionName, decide_image_actions, decide_coordinate_actions
from bruteforce_canvas.evaluation import (
    AlignmentEvaluation,
    CoordinateEvaluationAggregate,
    DispositionSignal,
    ImageEvaluationResult,
    LearningUpdateSignal,
    QualityEvaluation,
)


def image(signal: str, *, full_pass: bool = False) -> ImageEvaluationResult:
    return ImageEvaluationResult(
        candidate_id="cand_7",
        image_path="/tmp/cand_7.png",
        seed=7,
        coordinate_id="coord_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        file_valid=signal != "hard_purge_invalid_artifact",
        quality=QualityEvaluation(score=0.9),
        alignment=AlignmentEvaluation(score=0.9),
        pass_flags={"quality": full_pass, "alignment": full_pass, "full": full_pass},
        failure_types=[],
        localized_blame=[],
        disposition_signal=DispositionSignal(class_name=signal, confidence="high", reasons=["test"]),
        confidence="high",
    )


def aggregate(outcome: str, promoted: int) -> CoordinateEvaluationAggregate:
    return CoordinateEvaluationAggregate(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seeds=[7, 42, 156, 8888, 42069],
        generated_count=5,
        evaluated_count=5,
        promoted_count=promoted,
        quality_pass_count=promoted,
        alignment_pass_count=promoted,
        full_pass_count=promoted,
        mean_quality=0.5,
        mean_alignment=0.5,
        best_quality=0.9,
        best_alignment=0.9,
        pass_rate=promoted / 5,
        outcome=outcome,
        aggregate_failure_types=[],
        aggregate_blame=[],
        update_signal=LearningUpdateSignal(
            thompson_alpha_delta=float(promoted),
            thompson_beta_delta=float(5 - promoted),
            gp_affinity_delta=promoted / 5 - 0.5,
        ),
    )


def test_image_action_planner_promotes_only_pass_threshold_signals_with_full_pass():
    actions = decide_image_actions(image("passes_thresholds", full_pass=True))
    false_positive = decide_image_actions(image("passes_thresholds", full_pass=False))

    assert [action.name for action in actions] == [ActionName.PROMOTE_CURATE]
    assert [action.name for action in false_positive] == [ActionName.PERSIST_FOR_LEARNING]


def test_image_action_planner_maps_retry_and_purge_without_semantic_penalty():
    retry = decide_image_actions(image("infrastructure_retry_no_semantic_penalty"))
    purge = decide_image_actions(image("hard_purge_invalid_artifact"))

    assert retry[0].name == ActionName.INFRASTRUCTURE_RETRY
    assert retry[0].semantic_penalty is False
    assert purge[0].name == ActionName.HARD_PURGE_INVALID_ARTIFACT
    assert purge[0].semantic_penalty is False


def test_coordinate_action_planner_retires_fragile_and_failed_coordinates():
    fragile = decide_coordinate_actions(aggregate("fragile", 1))
    failed = decide_coordinate_actions(aggregate("failed", 0))

    assert [action.name for action in fragile] == [ActionName.RETIRE_COORDINATE]
    assert [action.name for action in failed] == [ActionName.RETIRE_COORDINATE]


def test_coordinate_action_planner_quarantines_repeated_zero_pass_failure():
    actions = decide_coordinate_actions(aggregate("failed", 0), quarantine=True)

    assert [action.name for action in actions] == [ActionName.RETIRE_COORDINATE, ActionName.QUARANTINE_COORDINATE]
