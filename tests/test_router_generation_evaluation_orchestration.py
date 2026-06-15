from pathlib import Path

from bruteforce_canvas.evaluation import (
    AlignmentEvaluation,
    EvaluationBatchRequest,
    EvaluationImageInput,
    EvaluationPlan,
    ImageEvaluationResult,
    QualityEvaluation,
    aggregate_seed_sweep,
    evaluate_with_static_scores,
)
from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE, GenerationSettings, seed_sweep_requests
from bruteforce_canvas.orchestration import RunConfig, apply_feedback
from bruteforce_canvas.router import (
    AxisDomain,
    FieldState,
    LHSRouter,
    RouterInput,
    ThompsonArmState,
)
from bruteforce_canvas.shared import CandidateLifecycle, FeedbackAction


def test_router_keeps_locked_fields_fixed_and_samples_only_eligible_axes():
    router = LHSRouter(seed=11)
    batch = router.propose(
        RouterInput(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            fixed_arms={
                "object.material.object_01": AxisDomain(
                    value="CERAMIC",
                    state=FieldState.EXPLICIT_LOCKED,
                    source="llm_canonicalizer",
                )
            },
            sampleable_axes={
                "cinematography.shot_size": [
                    ThompsonArmState(axis="cinematography.shot_size", value="MEDIUM_SHOT", alpha=3, beta=1),
                    ThompsonArmState(axis="cinematography.shot_size", value="WIDE_SHOT", alpha=1, beta=3),
                ],
                "cinematography.lighting_mood": [
                    ThompsonArmState(axis="cinematography.lighting_mood", value="SOFT_WINDOW_LIGHT", alpha=2, beta=1),
                    ThompsonArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=1, beta=2),
                ],
            },
            count=4,
        )
    )

    assert len(batch.coordinates) == 4
    assert {coord.lifecycle for coord in batch.coordinates} == {CandidateLifecycle.PROPOSED}
    for coord in batch.coordinates:
        assert coord.enum_coordinate["object.material.object_01"].value == "CERAMIC"
        assert coord.enum_coordinate["object.material.object_01"].state == FieldState.EXPLICIT_LOCKED
        assert "compatibility_trace" in coord.model_dump()
        assert coord.coordinate_id.startswith("coord_")


def test_generation_seed_sweep_uses_required_fixed_seed_bundle(tmp_path: Path):
    requests = seed_sweep_requests(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        rendered_prompt="Generate a ceramic bowl on wooden table",
        generation_settings=GenerationSettings(),
        output_dir=tmp_path,
        generator_model_id="stub-generator",
        generator_backend="stub",
    )

    assert [request.seed for request in requests] == DEFAULT_SEED_BUNDLE
    assert all(request.rendered_prompt.startswith("Generate ") for request in requests)
    assert all(request.image_path.endswith(".png") for request in requests)


def test_evaluator_promotes_only_quality_and_alignment_passes(tmp_path: Path):
    images = []
    for seed in DEFAULT_SEED_BUNDLE:
        path = tmp_path / f"seed_{seed}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        images.append(
            EvaluationImageInput(
                candidate_id=f"cand_{seed}",
                image_path=str(path),
                seed=seed,
                coordinate_id="coord_001",
                run_id="run_001",
                prompt_document_id="doc_001",
                target_manifest_id="eval_manifest_001",
                generation_settings={},
            )
        )

    request = EvaluationBatchRequest(
        batch_id="batch_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        batch_kind="seed_sweep",
        coordinate_id="coord_001",
        rendered_prompt="Generate a ceramic bowl on wooden table",
        target_manifest={},
        images=images,
        evaluator_plan=EvaluationPlan(quality_cutoff=0.55, alignment_cutoff=0.25),
    )

    result = evaluate_with_static_scores(
        request,
        quality_scores=[0.9, 0.8, 0.7, 0.2, 0.1],
        alignment_scores=[0.9, 0.8, 0.7, 0.9, 0.9],
    )

    assert result.aggregate.promoted_count == 3
    assert result.aggregate.outcome == "strong"
    assert all(image.disposition_signal.class_name != "passes_thresholds" for image in result.images[3:])


def test_seed_sweep_aggregate_is_not_judged_by_one_lucky_seed():
    results = [
        ImageEvaluationResult(
            candidate_id=f"cand_{index}",
            image_path=f"/tmp/{index}.png",
            seed=seed,
            coordinate_id="coord_001",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            file_valid=True,
            quality=QualityEvaluation(score=0.9),
            alignment=AlignmentEvaluation(score=0.9),
            pass_flags={"quality": index == 0, "alignment": index == 0, "full": index == 0},
            failure_types=[] if index == 0 else ["quality_below_cutoff"],
            localized_blame=[],
            disposition_signal={"class_name": "passes_thresholds" if index == 0 else "fail_persist_for_learning", "confidence": "high", "reasons": []},
            confidence="high",
        )
        for index, seed in enumerate(DEFAULT_SEED_BUNDLE)
    ]

    aggregate = aggregate_seed_sweep(results)
    assert aggregate.promoted_count == 1
    assert aggregate.outcome == "fragile"


def test_feedback_application_is_idempotent():
    state = {}
    first = apply_feedback(state, candidate_id="cand_001", action=FeedbackAction.REJECT)
    second = apply_feedback(first.state, candidate_id="cand_001", action=FeedbackAction.REJECT)

    assert first.applied is True
    assert second.applied is False
    assert first.learning_delta == second.learning_delta


def test_run_config_defaults_match_orchestration_contract():
    config = RunConfig(run_id="run_001", raw_user_prompt="a bowl on a table")
    assert config.seed_bundle == DEFAULT_SEED_BUNDLE
    assert config.iqa_cutoff == 0.55
    assert config.stall_window_seconds == 1800
    assert config.stall_min_promoted == 10
