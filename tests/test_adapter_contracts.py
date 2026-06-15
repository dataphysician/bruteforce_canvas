from pathlib import Path

from bruteforce_canvas.evaluation import (
    EvaluationBatchRequest,
    EvaluationImageInput,
    EvaluationPlan,
    StaticImpactAdapter,
    StaticIQAAdapter,
    StaticVLMAdapter,
    staged_evaluate,
)
from bruteforce_canvas.generation import (
    GenerationRequest,
    GenerationSettings,
    StubGeneratorAdapter,
    validate_image_file,
)


def generation_request(tmp_path: Path, seed: int = 7) -> GenerationRequest:
    return GenerationRequest(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seed=seed,
        rendered_prompt="Generate a ceramic bowl on wooden table",
        generation_settings=GenerationSettings(),
        image_path=str(tmp_path / f"seed_{seed}.png"),
        generator_model_id="stub-generator",
        generator_backend="stub",
    )


def evaluation_request(tmp_path: Path, *, metacognitive_impact: bool = False) -> EvaluationBatchRequest:
    images = []
    for seed in [7, 42, 156, 8888, 42069]:
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
    return EvaluationBatchRequest(
        batch_id="batch_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        batch_kind="seed_sweep",
        coordinate_id="coord_001",
        rendered_prompt="Generate a ceramic bowl on wooden table",
        target_manifest={},
        images=images,
        evaluator_plan=EvaluationPlan(
            quality_cutoff=0.55,
            alignment_cutoff=0.25,
            impact_cutoff=0.9,
            metacognitive_impact=metacognitive_impact,
        ),
    )


def test_stub_generator_persists_png_and_validity_metadata(tmp_path: Path):
    adapter = StubGeneratorAdapter()
    result = adapter.generate(generation_request(tmp_path))

    assert result.infrastructure_blocked is False
    assert result.candidate.file_valid is True
    assert validate_image_file(Path(result.candidate.image_path)).valid is True


def test_stub_generator_infrastructure_failure_has_retry_signal_no_semantic_penalty(tmp_path: Path):
    adapter = StubGeneratorAdapter(blocked_seeds={7})
    result = adapter.generate(generation_request(tmp_path))

    assert result.infrastructure_blocked is True
    assert result.candidate.file_valid is False
    assert result.disposition_signal.class_name == "infrastructure_retry_no_semantic_penalty"


def test_invalid_image_file_detection_is_structural(tmp_path: Path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not a png")

    result = validate_image_file(path)

    assert result.valid is False
    assert result.failure_type == "invalid_image_file"


def test_staged_evaluator_runs_vlm_only_for_iqa_survivors_and_impact_is_informational(tmp_path: Path):
    request = evaluation_request(tmp_path, metacognitive_impact=True)
    result = staged_evaluate(
        request,
        iqa=StaticIQAAdapter(scores=[0.9, 0.2, 0.8, 0.1, 0.7]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.9, 0.1]),
        impact=StaticImpactAdapter(scores=[1.0, 1.0], enabled=True),
    )

    assert result.iqa_survivor_candidate_ids == ["cand_7", "cand_156", "cand_42069"]
    assert result.aggregate.promoted_count == 2
    alignment_failure = next(image for image in result.images if image.candidate_id == "cand_42069")
    assert alignment_failure.impact is None
    assert alignment_failure.pass_flags["full"] is False
    assert alignment_failure.disposition_signal.class_name == "fail_persist_for_learning"


def test_staged_evaluator_records_execution_mode_and_elapsed_time_per_stage(tmp_path: Path):
    request = evaluation_request(tmp_path, metacognitive_impact=True)
    result = staged_evaluate(
        request,
        iqa=StaticIQAAdapter(scores=[0.9, 0.2, 0.8, 0.1, 0.7]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.9, 0.1]),
        impact=StaticImpactAdapter(scores=[1.0, 1.0], enabled=True),
    )

    promoted = next(image for image in result.images if image.candidate_id == "cand_7")
    quality_failure = next(image for image in result.images if image.candidate_id == "cand_42")
    alignment_failure = next(image for image in result.images if image.candidate_id == "cand_42069")

    assert promoted.evaluator_request_id == "batch_001"
    assert promoted.evaluator_plan == request.evaluator_plan
    assert promoted.evaluator_telemetry["iqa"]["execution_mode"] == "batch"
    assert promoted.evaluator_telemetry["vlm"]["execution_mode"] == "bounded_batch"
    assert promoted.evaluator_telemetry["impact"]["execution_mode"] == "bounded_batch"
    assert promoted.evaluator_telemetry["iqa"]["elapsed_ms"] >= 0
    assert promoted.evaluator_versions == {
        "iqa": {"model_id": "static-quality-evaluator", "model_version": "1"},
        "vlm": {"model_id": "static-alignment-evaluator", "model_version": "1"},
        "impact": {"model_id": "static-impact-evaluator", "model_version": "1"},
    }
    assert quality_failure.evaluator_versions == {
        "iqa": {"model_id": "static-quality-evaluator", "model_version": "1"}
    }
    assert alignment_failure.evaluator_versions == {
        "iqa": {"model_id": "static-quality-evaluator", "model_version": "1"},
        "vlm": {"model_id": "static-alignment-evaluator", "model_version": "1"},
    }
    assert quality_failure.evaluator_telemetry == {"iqa": promoted.evaluator_telemetry["iqa"]}
    assert "impact" not in alignment_failure.evaluator_telemetry


def test_staged_evaluator_skips_impact_when_plan_disables_metacognitive_stage(tmp_path: Path):
    request = evaluation_request(tmp_path)
    result = staged_evaluate(
        request,
        iqa=StaticIQAAdapter(scores=[0.9, 0.9, 0.9, 0.9, 0.9]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.9, 0.9, 0.9, 0.9]),
        impact=StaticImpactAdapter(scores=[1.0, 1.0, 1.0, 1.0, 1.0], enabled=True),
    )

    assert [image.impact for image in result.images] == [None, None, None, None, None]


def test_staged_evaluator_records_impact_cutoff_without_demoting_base_promotion(tmp_path: Path):
    request = evaluation_request(tmp_path, metacognitive_impact=True)
    result = staged_evaluate(
        request,
        iqa=StaticIQAAdapter(scores=[0.9, 0.9, 0.9, 0.9, 0.9]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.9, 0.9, 0.9, 0.9]),
        impact=StaticImpactAdapter(scores=[1.0, 0.2, 0.91, 0.89, 0.0], enabled=True),
    )

    below_cutoff = next(image for image in result.images if image.candidate_id == "cand_42")
    assert below_cutoff.pass_flags["quality"] is True
    assert below_cutoff.pass_flags["alignment"] is True
    assert below_cutoff.pass_flags["full"] is True
    assert below_cutoff.pass_flags["impact"] is False
    assert "impact_below_cutoff" in below_cutoff.failure_types
    assert below_cutoff.disposition_signal.class_name == "passes_thresholds"
    assert result.aggregate.promoted_count == 5


class FailingVLMAdapter(StaticVLMAdapter):
    def __init__(self) -> None:
        super().__init__(scores=[])

    def score(self, images):
        raise RuntimeError("vlm offline")


class FailingImpactAdapter(StaticImpactAdapter):
    def __init__(self) -> None:
        super().__init__(scores=[], enabled=True)

    def score(self, images):
        raise TimeoutError("impact timeout")


def test_staged_evaluator_returns_iqa_partial_results_when_vlm_fails(tmp_path: Path):
    request = evaluation_request(tmp_path)
    result = staged_evaluate(
        request,
        iqa=StaticIQAAdapter(scores=[0.9, 0.2, 0.8, 0.1, 0.7]),
        vlm=FailingVLMAdapter(),
    )

    assert result.iqa_survivor_candidate_ids == ["cand_7", "cand_156", "cand_42069"]
    assert result.aggregate.outcome == "blocked"
    survivor = next(image for image in result.images if image.candidate_id == "cand_7")
    quality_failure = next(image for image in result.images if image.candidate_id == "cand_42")
    assert survivor.pass_flags == {"quality": True, "alignment": False, "full": False}
    assert survivor.quality.score == 0.9
    assert survivor.alignment.score == 0.0
    assert survivor.failure_types == ["evaluator_unavailable"]
    assert survivor.disposition_signal.class_name == "infrastructure_retry_no_semantic_penalty"
    assert survivor.evaluator_telemetry["vlm"]["error_type"] == "RuntimeError"
    assert quality_failure.failure_types == ["quality_below_cutoff"]
    assert quality_failure.evaluator_versions == {"iqa": {"model_id": "static-quality-evaluator", "model_version": "1"}}


def test_staged_evaluator_preserves_base_promotion_when_optional_impact_fails(tmp_path: Path):
    request = evaluation_request(tmp_path, metacognitive_impact=True)
    result = staged_evaluate(
        request,
        iqa=StaticIQAAdapter(scores=[0.9, 0.9, 0.9, 0.9, 0.9]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.9, 0.9, 0.9, 0.9]),
        impact=FailingImpactAdapter(),
    )

    promoted = next(image for image in result.images if image.candidate_id == "cand_7")
    assert result.aggregate.promoted_count == 5
    assert promoted.pass_flags["full"] is True
    assert promoted.pass_flags["impact"] is False
    assert promoted.failure_types == ["impact_unavailable"]
    assert promoted.disposition_signal.class_name == "passes_thresholds"
    assert promoted.evaluator_telemetry["impact"]["error_type"] == "TimeoutError"
