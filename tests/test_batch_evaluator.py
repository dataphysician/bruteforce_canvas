from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from bruteforce_canvas.evaluation import (
    AlignmentEvaluation,
    BatchEvaluator,
    EvaluationImageInput,
    EvaluationPlan,
    ImpactEvaluation,
    QualityEvaluation,
)


def _images(tmp_path: Path, count: int = 3) -> list[EvaluationImageInput]:
    images: list[EvaluationImageInput] = []
    for index in range(count):
        seed = index + 1
        path = tmp_path / f"batch_{seed}.png"
        path.write_bytes(b"fake-png")
        images.append(
            EvaluationImageInput(
                candidate_id=f"cand_batch_{seed}",
                image_path=str(path),
                seed=seed,
                coordinate_id="coord_001",
                run_id="run_001",
                prompt_document_id="doc_001",
                target_manifest_id="eval_manifest_001",
                generation_settings={},
            )
        )
    return images


def _plan(
    execution_preference: Literal["serialized", "parallel", "tensor_batch", "auto"],
    *,
    metacognitive_impact: bool = False,
) -> EvaluationPlan:
    return EvaluationPlan(
        quality_cutoff=0.5,
        alignment_cutoff=0.5,
        impact_cutoff=0.5,
        metacognitive_impact=metacognitive_impact,
        execution_preference=execution_preference,
    )


class SingleIQA:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.calls: list[str] = []

    def score_one(self, image: EvaluationImageInput, **_kwargs: Any) -> QualityEvaluation:
        self.calls.append(str(image.candidate_id))
        return QualityEvaluation(score=self.score, model_id="single-iqa")


class SingleVLM:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.calls: list[str] = []

    def evaluate_one(self, image: EvaluationImageInput, **_kwargs: Any) -> AlignmentEvaluation:
        self.calls.append(str(image.candidate_id))
        return AlignmentEvaluation(score=self.score, model_id="single-vlm")


class SingleImpact:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.calls: list[str] = []

    def score_one(self, image: EvaluationImageInput, **_kwargs: Any) -> ImpactEvaluation:
        self.calls.append(str(image.candidate_id))
        return ImpactEvaluation(score=self.score, model_id="single-impact")


class BatchIQA:
    def score(self, images: list[EvaluationImageInput]) -> list[QualityEvaluation]:
        return [QualityEvaluation(score=0.9, model_id="batch-iqa") for _image in images]


class BatchVLM:
    def score(self, images: list[EvaluationImageInput]) -> list[AlignmentEvaluation]:
        return [AlignmentEvaluation(score=0.9, model_id="batch-vlm") for _image in images]


class ParallelVLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def evaluate(self, image_paths: list[str], *, prompt: str, manifest: Any) -> list[AlignmentEvaluation]:
        del prompt, manifest
        self.calls.append(tuple(image_paths))
        return [AlignmentEvaluation(score=0.9, model_id="parallel-vlm") for _path in image_paths]


class TensorIQA:
    def __init__(self) -> None:
        self.tensor_calls: list[list[str]] = []

    def score_tensor_batch(
        self,
        images: list[EvaluationImageInput],
        *,
        prompt: str,
        manifest: Any,
    ) -> list[QualityEvaluation]:
        del prompt, manifest
        self.tensor_calls.append([str(image.candidate_id) for image in images])
        return [QualityEvaluation(score=0.9, model_id="tensor-iqa") for _image in images]


def test_batch_evaluator_serialized_mode_uses_single_image_stage_calls(tmp_path: Path) -> None:
    images = _images(tmp_path)
    iqa = SingleIQA()
    vlm = SingleVLM()
    impact = SingleImpact()

    results = BatchEvaluator(iqa, vlm, impact).evaluate(
        images,
        "prompt",
        {},
        _plan("serialized", metacognitive_impact=True),
    )

    assert [str(image.candidate_id) for image in images] == iqa.calls == vlm.calls == impact.calls
    assert all(result.pass_flags["full"] is True for result in results)
    assert results[0].evaluator_telemetry["iqa"]["execution_mode"] == "serialized"
    assert results[0].evaluator_telemetry["vlm"]["execution_mode"] == "serialized"
    assert results[0].evaluator_telemetry["impact"]["execution_mode"] == "serialized"


def test_batch_evaluator_parallel_mode_runs_vlm_as_parallel_single_image_calls(tmp_path: Path) -> None:
    images = _images(tmp_path, count=4)
    vlm = ParallelVLM()

    results = BatchEvaluator(BatchIQA(), vlm).evaluate(
        images,
        "prompt",
        {},
        _plan("parallel"),
    )

    assert len(results) == 4
    assert sorted(len(call) for call in vlm.calls) == [1, 1, 1, 1]
    assert results[0].evaluator_telemetry["vlm"]["execution_mode"] == "parallel"


def test_batch_evaluator_tensor_batch_mode_uses_iqa_tensor_batch_hook(tmp_path: Path) -> None:
    images = _images(tmp_path)
    iqa = TensorIQA()

    results = BatchEvaluator(iqa, BatchVLM()).evaluate(
        images,
        "prompt",
        {},
        _plan("tensor_batch"),
    )

    assert iqa.tensor_calls == [[str(image.candidate_id) for image in images]]
    assert len(results) == 3
    assert results[0].quality.model_id == "tensor-iqa"
    assert results[0].evaluator_telemetry["iqa"]["execution_mode"] == "tensor_batch"


def test_batch_evaluator_metrics_are_present_on_results(tmp_path: Path) -> None:
    results = BatchEvaluator(SingleIQA(), SingleVLM()).evaluate(
        _images(tmp_path, count=2),
        "prompt",
        {},
        _plan("serialized"),
    )

    metrics = results[0].evaluator_telemetry["batch"]
    assert metrics["eval_batch_size"] == 2
    assert metrics["eval_batch_duration_seconds"] >= 0.0
