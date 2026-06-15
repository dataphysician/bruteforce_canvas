from bruteforce_canvas.evaluation import (
    AlignmentEvaluation,
    TargetObservation,
    apply_target_observations,
)
from bruteforce_canvas.prompt import EvaluationTarget, EvaluationTargetManifest


def manifest() -> EvaluationTargetManifest:
    return EvaluationTargetManifest(
        manifest_id="eval_manifest_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        coordinate_id="coord_001",
        rendered_prompt="Generate red ceramic bowl on wooden table, medium shot",
        targets=[
            EvaluationTarget(
                target_id="object_01",
                target_kind="element",
                label="bowl",
                priority="locked_required",
                lhs_policy="fixed",
                evaluation_policy="must_match",
                evidence="ceramic bowl",
            ),
            EvaluationTarget(
                target_id="object_01.material",
                target_kind="object_attribute",
                value_raw="ceramic",
                enum_value="CERAMIC",
                priority="locked_required",
                lhs_policy="fixed",
                evaluation_policy="must_match",
            ),
            EvaluationTarget(
                target_id="cinematography.shot_size",
                target_kind="cinematography",
                enum_value="MEDIUM_SHOT",
                priority="sampled",
                lhs_policy="sampleable",
                evaluation_policy="should_match",
            ),
        ],
        negative_targets=[
            EvaluationTarget(
                target_id="constraint.no_extra_people",
                target_kind="constraint",
                value_raw="no extra people",
                priority="negative_guard",
                lhs_policy="fixed",
                evaluation_policy="must_not_appear",
            )
        ],
    )


def test_locked_required_miss_fails_alignment_and_localizes_locked_blame():
    result = apply_target_observations(
        AlignmentEvaluation(score=0.8),
        manifest(),
        [
            TargetObservation(target_id="object_01", present=True, confidence="high"),
            TargetObservation(target_id="object_01.material", present=False, confidence="high"),
            TargetObservation(target_id="cinematography.shot_size", present=True, confidence="high"),
            TargetObservation(target_id="constraint.no_extra_people", present=False, confidence="high"),
        ],
    )

    assert result.pass_flags["alignment"] is False
    assert "wrong_material" in result.failure_types
    assert result.localized_blame[0].target_id == "object_01.material"
    assert result.localized_blame[0].source == "locked"
    assert result.disposition_signal.class_name == "fail_persist_for_learning"


def test_sampled_should_match_miss_adds_sampled_blame_without_hard_failure():
    result = apply_target_observations(
        AlignmentEvaluation(score=0.8),
        manifest(),
        [
            TargetObservation(target_id="object_01", present=True, confidence="high"),
            TargetObservation(target_id="object_01.material", present=True, confidence="high"),
            TargetObservation(target_id="cinematography.shot_size", present=False, confidence="high"),
            TargetObservation(target_id="constraint.no_extra_people", present=False, confidence="high"),
        ],
    )

    assert result.pass_flags["alignment"] is True
    assert result.failure_types == ["wrong_shot_size"]
    assert result.localized_blame[0].source == "sampled"
    assert result.disposition_signal.class_name == "passes_thresholds"


def test_negative_guard_presence_is_hard_failure():
    result = apply_target_observations(
        AlignmentEvaluation(score=0.9),
        manifest(),
        [
            TargetObservation(target_id="object_01", present=True, confidence="high"),
            TargetObservation(target_id="object_01.material", present=True, confidence="high"),
            TargetObservation(target_id="cinematography.shot_size", present=True, confidence="high"),
            TargetObservation(target_id="constraint.no_extra_people", present=True, confidence="high"),
        ],
    )

    assert result.pass_flags["alignment"] is False
    assert result.failure_types == ["negative_constraint_violation"]
    assert result.localized_blame[0].blame_type == "constraint_violation"
