from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import Field, model_validator

from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE
from bruteforce_canvas.prompt import EvaluationTargetManifest
from bruteforce_canvas.shared import CandidateId, CoordinateId, DocId, RunId, StrictModel, TargetManifestId


FailureType = Literal[
    "invalid_image_file",
    "image_decode_failed",
    "evaluator_unavailable",
    "evaluator_timeout",
    "evaluator_malformed_output",
    "gpu_memory_failure",
    "quality_below_cutoff",
    "blur_or_low_detail",
    "severe_artifact",
    "bad_anatomy",
    "bad_hands",
    "bad_face",
    "deformed_product",
    "bad_text_rendering",
    "overexposed_or_underexposed",
    "alignment_below_cutoff",
    "missing_locked_element",
    "missing_locked_relation",
    "missing_action_actor",
    "missing_action_target",
    "wrong_spatial_relation",
    "wrong_color",
    "wrong_material",
    "wrong_lighting",
    "wrong_camera_angle",
    "wrong_shot_size",
    "wrong_style",
    "invented_major_object",
    "negative_constraint_violation",
    "seed_fragility",
    "single_seed_luck",
    "coordinate_consistent_failure",
    "impact_below_cutoff",
    "impact_unavailable",
    "impact_unlicensed",
    "impact_from_artifact_or_violation",
]


class EvaluationImageInput(StrictModel):
    candidate_id: CandidateId | None
    image_path: str
    seed: int
    coordinate_id: CoordinateId
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    generation_settings: dict[str, Any]


class EvaluationPlan(StrictModel):
    quality: bool = True
    alignment: bool = True
    metacognitive_impact: bool = False
    quality_cutoff: float
    alignment_cutoff: float
    human_quality_cutoff: float | None = None
    impact_cutoff: float | None = None
    execution_preference: Literal["serialized", "parallel", "tensor_batch", "auto"] = "auto"


class EvaluationBatchRequest(StrictModel):
    batch_id: str
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    batch_kind: Literal["seed_sweep", "mixed_image_batch"]
    coordinate_id: CoordinateId | None
    rendered_prompt: str
    target_manifest: dict[str, Any]
    images: list[EvaluationImageInput]
    evaluator_plan: EvaluationPlan

    @model_validator(mode="after")
    def validate_seed_sweep_shape(self) -> "EvaluationBatchRequest":
        if self.batch_kind == "seed_sweep":
            seeds = [image.seed for image in self.images]
            if seeds != DEFAULT_SEED_BUNDLE:
                raise ValueError("seed_sweep requests must use the fixed five-seed bundle")
        return self


class QualityEvaluation(StrictModel):
    score: float
    model_id: str = "static-quality-evaluator"
    model_version: str = "1"
    confidence: Literal["high", "medium", "low"] = "high"


class AlignmentEvaluation(StrictModel):
    score: float
    model_id: str = "static-alignment-evaluator"
    model_version: str = "1"
    confidence: Literal["high", "medium", "low"] = "high"


class ImpactEvaluation(StrictModel):
    score: float
    model_id: str = "static-impact-evaluator"
    model_version: str = "1"
    confidence: Literal["high", "medium", "low"] = "high"
    informational_only: bool = True


class BlameHint(StrictModel):
    target_id: str | None = None
    field_path: str | None = None
    enum_value: str | None = None
    source: Literal["locked", "sampled", "inferred", "proposed", "unknown"] = "unknown"
    blame_type: Literal[
        "semantic_mismatch",
        "visibility_loss",
        "technical_quality",
        "constraint_violation",
        "seed_instability",
        "infrastructure",
    ]
    confidence: Literal["high", "medium", "low"]
    reason: str


class DispositionSignal(StrictModel):
    class_name: Literal[
        "passes_thresholds",
        "fail_persist_for_learning",
        "demote_candidate",
        "coordinate_quarantine_candidate",
        "temporary_enum_suppression_candidate",
        "hard_purge_invalid_artifact",
        "infrastructure_retry_no_semantic_penalty",
    ]
    confidence: Literal["high", "medium", "low"]
    reasons: list[str] = Field(default_factory=list)


class ImageEvaluationResult(StrictModel):
    candidate_id: CandidateId | None
    image_path: str
    seed: int
    coordinate_id: CoordinateId
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    file_valid: bool
    quality: QualityEvaluation
    alignment: AlignmentEvaluation
    impact: dict[str, Any] | None = None
    evaluator_request_id: str | None = None
    evaluator_plan: EvaluationPlan | None = None
    evaluator_telemetry: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evaluator_versions: dict[str, dict[str, str]] = Field(default_factory=dict)
    pass_flags: dict[str, bool]
    failure_types: list[FailureType]
    localized_blame: list[BlameHint]
    disposition_signal: DispositionSignal
    confidence: Literal["high", "medium", "low"]


class TargetObservation(StrictModel):
    target_id: str
    present: bool
    confidence: Literal["high", "medium", "low"]
    reason: str | None = None


class TargetPolicyResult(StrictModel):
    alignment: AlignmentEvaluation
    pass_flags: dict[str, bool]
    failure_types: list[FailureType]
    localized_blame: list[BlameHint]
    disposition_signal: DispositionSignal


class LearningUpdateSignal(StrictModel):
    thompson_alpha_delta: float
    thompson_beta_delta: float
    gp_affinity_delta: float
    source: Literal["automated_evaluation", "user_feedback", "combined"] = "automated_evaluation"


class CoordinateEvaluationAggregate(StrictModel):
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    coordinate_id: CoordinateId
    seeds: list[int]
    generated_count: int
    evaluated_count: int
    promoted_count: int
    quality_pass_count: int
    alignment_pass_count: int
    full_pass_count: int
    mean_quality: float
    mean_alignment: float
    best_quality: float
    best_alignment: float
    pass_rate: float
    outcome: Literal["strong", "viable", "fragile", "failed", "blocked"]
    aggregate_failure_types: list[FailureType]
    aggregate_blame: list[BlameHint]
    update_signal: LearningUpdateSignal


class EvaluationBatchResult(StrictModel):
    images: list[ImageEvaluationResult]
    aggregate: CoordinateEvaluationAggregate


class StagedEvaluationResult(EvaluationBatchResult):
    iqa_survivor_candidate_ids: list[str]


class StaticIQAAdapter:
    def __init__(self, *, scores: list[float], model_id: str = "static-quality-evaluator", model_version: str = "1") -> None:
        self.scores = scores
        self.model_id = model_id
        self.model_version = model_version

    def score(self, images: list[EvaluationImageInput]) -> list[QualityEvaluation]:
        if len(images) != len(self.scores):
            raise ValueError("IQA score count must match image count")
        return [QualityEvaluation(score=score, model_id=self.model_id, model_version=self.model_version) for score in self.scores]


class StaticVLMAdapter:
    def __init__(self, *, scores: list[float], model_id: str = "static-alignment-evaluator", model_version: str = "1") -> None:
        self.scores = scores
        self.model_id = model_id
        self.model_version = model_version

    def score(self, images: list[EvaluationImageInput]) -> list[AlignmentEvaluation]:
        if len(images) != len(self.scores):
            raise ValueError("VLM score count must match survivor image count")
        return [AlignmentEvaluation(score=score, model_id=self.model_id, model_version=self.model_version) for score in self.scores]


class StaticImpactAdapter:
    def __init__(
        self,
        *,
        scores: list[float],
        enabled: bool = False,
        model_id: str = "static-impact-evaluator",
        model_version: str = "1",
    ) -> None:
        self.scores = scores
        self.enabled = enabled
        self.model_id = model_id
        self.model_version = model_version

    def score(self, images: list[EvaluationImageInput]) -> list[dict[str, Any] | None]:
        if not self.enabled:
            return [None for _ in images]
        if len(images) != len(self.scores):
            raise ValueError("impact score count must match survivor image count")
        return [
            {
                "score": score,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "informational_only": True,
            }
            for score in self.scores
        ]


class BatchEvaluator:
    """Execute evaluator stages using serialized, parallel, or tensor-batch dispatch.

    The class accepts both the lightweight test adapters from this module
    (``score(...)`` methods) and the real Phase G adapters from
    :mod:`bruteforce_canvas.real_adapters` (``evaluate(...)`` methods). Heavy
    ML dependencies remain optional because tensor-batch support is discovered
    by adapter capability rather than imported here.
    """

    def __init__(
        self,
        iqa: Any | None = None,
        vlm: Any | None = None,
        impact: Any | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        self.iqa = iqa
        self.vlm = vlm
        self.impact = impact
        self.request_id = request_id
        self.last_metrics: dict[str, float | int] = {}
        self.last_survivor_candidate_ids: list[str] = []

    def evaluate(
        self,
        images: list[EvaluationImageInput],
        prompt: str,
        manifest: Any,
        plan: EvaluationPlan,
    ) -> list[ImageEvaluationResult]:
        started = time.perf_counter()
        self.last_survivor_candidate_ids = []
        batch_metrics: dict[str, float | int] = {
            "eval_batch_duration_seconds": 0.0,
            "eval_batch_size": len(images),
        }

        try:
            results = self._evaluate(images, prompt, manifest, plan, batch_metrics)
        finally:
            batch_metrics["eval_batch_duration_seconds"] = time.perf_counter() - started
            self.last_metrics = batch_metrics

        return [self._attach_batch_metrics(result, batch_metrics) for result in results]

    def _evaluate(
        self,
        images: list[EvaluationImageInput],
        prompt: str,
        manifest: Any,
        plan: EvaluationPlan,
        batch_metrics: dict[str, float | int],
    ) -> list[ImageEvaluationResult]:
        if not images:
            return []
        iqa_mode = self._stage_mode(plan.execution_preference, default="tensor_batch")
        iqa_telemetry = {"execution_mode": iqa_mode, "elapsed_ms": 0}
        try:
            quality_results = self._score_iqa(images, prompt, manifest, iqa_mode)
        except Exception as error:
            iqa_telemetry = _failed_stage_telemetry(iqa_mode, error)
            return [self._unavailable_result(image, plan, {"iqa": iqa_telemetry}) for image in images]

        if len(quality_results) != len(images):
            error = ValueError("IQA result count must match image count")
            iqa_telemetry = _failed_stage_telemetry(iqa_mode, error)
            return [self._unavailable_result(image, plan, {"iqa": iqa_telemetry}) for image in images]

        survivor_pairs: list[tuple[int, EvaluationImageInput, QualityEvaluation]] = [
            (index, image, quality)
            for index, (image, quality) in enumerate(zip(images, quality_results, strict=True))
            if quality.score >= plan.quality_cutoff
        ]
        survivor_images = [image for _index, image, _quality in survivor_pairs]
        self.last_survivor_candidate_ids = [str(image.candidate_id) for image in survivor_images if image.candidate_id is not None]

        vlm_mode = "serialized" if plan.execution_preference == "serialized" else "parallel"
        vlm_telemetry = {"execution_mode": vlm_mode, "elapsed_ms": 0}
        try:
            alignment_results = self._score_vlm(survivor_images, prompt, manifest, vlm_mode)
        except Exception as error:
            vlm_telemetry = _failed_stage_telemetry(vlm_mode, error)
            return self._partial_vlm_failure_results(images, quality_results, plan, iqa_telemetry, vlm_telemetry)

        if len(alignment_results) != len(survivor_images):
            error = ValueError("VLM result count must match survivor image count")
            vlm_telemetry = _failed_stage_telemetry(vlm_mode, error)
            return self._partial_vlm_failure_results(images, quality_results, plan, iqa_telemetry, vlm_telemetry)

        alignment_by_index = {
            index: alignment
            for (index, _image, _quality), alignment in zip(survivor_pairs, alignment_results, strict=True)
        }
        impact_pairs = [
            (index, image)
            for index, image, _quality in survivor_pairs
            if alignment_by_index[index].score >= plan.alignment_cutoff
        ]
        impact_images = [image for _index, image in impact_pairs]
        impact_mode = "serialized"
        impact_telemetry = {"execution_mode": impact_mode, "elapsed_ms": 0}
        impact_enabled = plan.metacognitive_impact and self.impact is not None
        impact_failed = False
        if impact_enabled:
            try:
                impact_payloads = self._score_impact(impact_images, prompt, manifest, impact_mode)
            except Exception as error:
                impact_failed = True
                impact_telemetry = _failed_stage_telemetry(impact_mode, error)
                impact_payloads = [None for _image in impact_images]
            if len(impact_payloads) != len(impact_images):
                impact_payloads = [None for _image in impact_images]
        else:
            impact_payloads = [None for _image in impact_images]
        impact_by_index = {
            index: payload
            for (index, _image), payload in zip(impact_pairs, impact_payloads, strict=True)
        }

        results: list[ImageEvaluationResult] = []
        for index, (image, quality) in enumerate(zip(images, quality_results, strict=True)):
            if index not in alignment_by_index:
                results.append(self._quality_failure_result(image, quality, plan, iqa_telemetry))
                continue
            alignment = alignment_by_index[index]
            impact_payload = impact_by_index.get(index)
            results.append(
                self._full_result(
                    image,
                    quality,
                    alignment,
                    impact_payload,
                    plan,
                    iqa_telemetry,
                    vlm_telemetry,
                    impact_telemetry,
                    impact_enabled,
                    impact_failed,
                )
            )
        return results

    @staticmethod
    def _stage_mode(
        preference: Literal["serialized", "parallel", "tensor_batch", "auto"],
        *,
        default: Literal["serialized", "parallel", "tensor_batch"],
    ) -> Literal["serialized", "parallel", "tensor_batch"]:
        if preference == "auto":
            return default
        return preference

    def _score_iqa(
        self,
        images: list[EvaluationImageInput],
        prompt: str,
        manifest: Any,
        mode: str,
    ) -> list[QualityEvaluation]:
        if self.iqa is None:
            raise RuntimeError("iqa evaluator unavailable")
        if mode == "serialized":
            if self._supports_path_single_call(self.iqa):
                return [self._score_iqa_single(image, prompt, manifest) for image in images]
            return self._score_iqa_batch(images, prompt, manifest)
        if mode == "parallel":
            if self._has_single_method(self.iqa):
                with ThreadPoolExecutor(max_workers=max(1, min(32, len(images)))) as executor:
                    return list(executor.map(lambda image: self._score_iqa_single(image, prompt, manifest), images))
            return self._score_iqa_batch(images, prompt, manifest)
        return self._score_iqa_tensor_batch(images, prompt, manifest)

    def _score_vlm(
        self,
        images: list[EvaluationImageInput],
        prompt: str,
        manifest: Any,
        mode: str,
    ) -> list[AlignmentEvaluation]:
        if self.vlm is None:
            raise RuntimeError("vlm evaluator unavailable")
        if not images:
            return []
        if mode == "serialized":
            if self._supports_vlm_single_call(self.vlm):
                return [self._score_vlm_single(image, prompt, manifest) for image in images]
            return self._score_vlm_batch(images, prompt, manifest)
        if mode == "parallel":
            if self._supports_vlm_single_call(self.vlm):
                with ThreadPoolExecutor(max_workers=max(1, min(32, len(images)))) as executor:
                    return list(executor.map(lambda image: self._score_vlm_single(image, prompt, manifest), images))
            return self._score_vlm_batch(images, prompt, manifest)
        return self._score_vlm_batch(images, prompt, manifest)

    def _score_impact(
        self,
        images: list[EvaluationImageInput],
        prompt: str,
        manifest: Any,
        mode: str,
    ) -> list[dict[str, Any] | None]:
        if self.impact is None:
            return [None for _image in images]
        if not images:
            return []
        if mode == "serialized":
            if self._supports_path_single_call(self.impact):
                return [self._impact_to_payload(self._score_impact_single(image, prompt, manifest)) for image in images]
            return [self._impact_to_payload(result) for result in self._score_impact_batch(images, prompt, manifest)]
        if mode == "parallel" and self._has_single_method(self.impact):
            with ThreadPoolExecutor(max_workers=max(1, min(32, len(images)))) as executor:
                return [
                    self._impact_to_payload(result)
                    for result in executor.map(lambda image: self._score_impact_single(image, prompt, manifest), images)
                ]
        return [self._impact_to_payload(result) for result in self._score_impact_batch(images, prompt, manifest)]

    def _score_iqa_tensor_batch(
        self,
        images: list[EvaluationImageInput],
        prompt: str,
        manifest: Any,
    ) -> list[QualityEvaluation]:
        assert self.iqa is not None
        for method_name in ("score_tensor_batch", "evaluate_tensor_batch"):
            method = getattr(self.iqa, method_name, None)
            if method is not None:
                try:
                    return list(method(images, prompt=prompt, manifest=manifest))
                except TypeError:
                    try:
                        return list(method(images))
                    except TypeError:
                        return list(method([image.image_path for image in images]))
        return self._score_iqa_batch(images, prompt, manifest)

    def _score_iqa_batch(
        self,
        images: list[EvaluationImageInput],
        prompt: str,
        manifest: Any,
    ) -> list[QualityEvaluation]:
        assert self.iqa is not None
        score = getattr(self.iqa, "score", None)
        if score is not None:
            return list(score(images))
        evaluate = getattr(self.iqa, "evaluate", None)
        if evaluate is not None:
            return list(evaluate([image.image_path for image in images]))
        raise RuntimeError("iqa evaluator does not expose score/evaluate")

    def _score_iqa_single(
        self,
        image: EvaluationImageInput,
        prompt: str,
        manifest: Any,
    ) -> QualityEvaluation:
        assert self.iqa is not None
        for method_name in ("score_one", "evaluate_one", "score_single", "evaluate_single"):
            method = getattr(self.iqa, method_name, None)
            if method is not None:
                try:
                    return method(image, prompt=prompt, manifest=manifest)
                except TypeError:
                    try:
                        return method(image)
                    except TypeError:
                        return method(image.image_path)
        return self._score_iqa_batch([image], prompt, manifest)[0]

    def _score_vlm_batch(
        self,
        images: list[EvaluationImageInput],
        prompt: str,
        manifest: Any,
    ) -> list[AlignmentEvaluation]:
        assert self.vlm is not None
        score = getattr(self.vlm, "score", None)
        if score is not None:
            return list(score(images))
        evaluate = getattr(self.vlm, "evaluate", None)
        if evaluate is not None:
            return list(evaluate([image.image_path for image in images], prompt=prompt, manifest=self._adapter_manifest(manifest)))
        raise RuntimeError("vlm evaluator does not expose score/evaluate")

    def _score_vlm_single(
        self,
        image: EvaluationImageInput,
        prompt: str,
        manifest: Any,
    ) -> AlignmentEvaluation:
        assert self.vlm is not None
        for method_name in ("score_one", "evaluate_one", "score_single", "evaluate_single"):
            method = getattr(self.vlm, method_name, None)
            if method is not None:
                try:
                    return method(image, prompt=prompt, manifest=manifest)
                except TypeError:
                    try:
                        return method(image)
                    except TypeError:
                        return method(image.image_path, prompt=prompt, manifest=manifest)
        evaluate = getattr(self.vlm, "evaluate", None)
        if evaluate is not None:
            return list(evaluate([image.image_path], prompt=prompt, manifest=self._adapter_manifest(manifest)))[0]
        return self._score_vlm_batch([image], prompt, manifest)[0]

    def _score_impact_batch(
        self,
        images: list[EvaluationImageInput],
        prompt: str,
        manifest: Any,
    ) -> list[ImpactEvaluation | dict[str, Any] | None]:
        assert self.impact is not None
        score = getattr(self.impact, "score", None)
        if score is not None:
            return list(score(images))
        evaluate = getattr(self.impact, "evaluate", None)
        if evaluate is not None:
            return list(evaluate([image.image_path for image in images]))
        raise RuntimeError("impact evaluator does not expose score/evaluate")

    def _score_impact_single(
        self,
        image: EvaluationImageInput,
        prompt: str,
        manifest: Any,
    ) -> ImpactEvaluation | dict[str, Any] | None:
        assert self.impact is not None
        for method_name in ("score_one", "evaluate_one", "score_single", "evaluate_single"):
            method = getattr(self.impact, method_name, None)
            if method is not None:
                try:
                    return method(image, prompt=prompt, manifest=manifest)
                except TypeError:
                    try:
                        return method(image)
                    except TypeError:
                        return method(image.image_path)
        return self._score_impact_batch([image], prompt, manifest)[0]

    @staticmethod
    def _has_single_method(adapter: Any) -> bool:
        return any(
            getattr(adapter, method_name, None) is not None
            for method_name in ("score_one", "evaluate_one", "score_single", "evaluate_single")
        )

    @classmethod
    def _supports_path_single_call(cls, adapter: Any) -> bool:
        return cls._has_single_method(adapter) or getattr(adapter, "evaluate", None) is not None

    @classmethod
    def _supports_vlm_single_call(cls, adapter: Any) -> bool:
        return cls._supports_path_single_call(adapter)

    @staticmethod
    def _impact_to_payload(value: ImpactEvaluation | dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        return value.model_dump()

    @staticmethod
    def _adapter_manifest(manifest: Any) -> Any:
        if isinstance(manifest, EvaluationTargetManifest):
            return manifest
        if isinstance(manifest, dict):
            try:
                return EvaluationTargetManifest.model_validate(manifest)
            except Exception:
                return manifest
        return manifest

    def _unavailable_result(
        self,
        image: EvaluationImageInput,
        plan: EvaluationPlan,
        telemetry: dict[str, dict[str, Any]],
    ) -> ImageEvaluationResult:
        return ImageEvaluationResult(
            candidate_id=image.candidate_id,
            image_path=image.image_path,
            seed=image.seed,
            coordinate_id=image.coordinate_id,
            run_id=image.run_id,
            prompt_document_id=image.prompt_document_id,
            target_manifest_id=image.target_manifest_id,
            file_valid=True,
            quality=QualityEvaluation(score=0.0, confidence="low"),
            alignment=AlignmentEvaluation(score=0.0, confidence="low"),
            impact=None,
            evaluator_request_id=self.request_id,
            evaluator_plan=plan,
            evaluator_telemetry=telemetry,
            evaluator_versions={},
            pass_flags={"quality": False, "alignment": False, "full": False},
            failure_types=["evaluator_unavailable"],
            localized_blame=[],
            disposition_signal=DispositionSignal(
                class_name="infrastructure_retry_no_semantic_penalty",
                confidence="high",
                reasons=["evaluator unavailable"],
            ),
            confidence="low",
        )

    def _partial_vlm_failure_results(
        self,
        images: list[EvaluationImageInput],
        quality_results: list[QualityEvaluation],
        plan: EvaluationPlan,
        iqa_telemetry: dict[str, Any],
        vlm_telemetry: dict[str, Any],
    ) -> list[ImageEvaluationResult]:
        results: list[ImageEvaluationResult] = []
        for image, quality in zip(images, quality_results, strict=True):
            quality_pass = quality.score >= plan.quality_cutoff
            if quality_pass:
                failure_types: list[FailureType] = ["evaluator_unavailable"]
                disposition = DispositionSignal(
                    class_name="infrastructure_retry_no_semantic_penalty",
                    confidence="high",
                    reasons=["vlm evaluator unavailable"],
                )
                telemetry = {"iqa": iqa_telemetry, "vlm": vlm_telemetry}
            else:
                failure_types = ["quality_below_cutoff"]
                disposition = DispositionSignal(
                    class_name="fail_persist_for_learning",
                    confidence="high",
                    reasons=[str(failure) for failure in failure_types],
                )
                telemetry = {"iqa": iqa_telemetry}
            results.append(
                ImageEvaluationResult(
                    candidate_id=image.candidate_id,
                    image_path=image.image_path,
                    seed=image.seed,
                    coordinate_id=image.coordinate_id,
                    run_id=image.run_id,
                    prompt_document_id=image.prompt_document_id,
                    target_manifest_id=image.target_manifest_id,
                    file_valid=True,
                    quality=quality,
                    alignment=AlignmentEvaluation(score=0.0, confidence="low"),
                    impact=None,
                    evaluator_request_id=self.request_id,
                    evaluator_plan=plan,
                    evaluator_telemetry=telemetry,
                    evaluator_versions={"iqa": _model_version(quality)},
                    pass_flags={"quality": quality_pass, "alignment": False, "full": False},
                    failure_types=failure_types,
                    localized_blame=[],
                    disposition_signal=disposition,
                    confidence="low" if quality_pass else "high",
                )
            )
        return results

    def _quality_failure_result(
        self,
        image: EvaluationImageInput,
        quality: QualityEvaluation,
        plan: EvaluationPlan,
        iqa_telemetry: dict[str, Any],
    ) -> ImageEvaluationResult:
        return ImageEvaluationResult(
            candidate_id=image.candidate_id,
            image_path=image.image_path,
            seed=image.seed,
            coordinate_id=image.coordinate_id,
            run_id=image.run_id,
            prompt_document_id=image.prompt_document_id,
            target_manifest_id=image.target_manifest_id,
            file_valid=True,
            quality=quality,
            alignment=AlignmentEvaluation(score=0.0, confidence="low"),
            impact=None,
            evaluator_request_id=self.request_id,
            evaluator_plan=plan,
            evaluator_telemetry={"iqa": iqa_telemetry},
            evaluator_versions={"iqa": _model_version(quality)},
            pass_flags={"quality": False, "alignment": False, "full": False},
            failure_types=["quality_below_cutoff"],
            localized_blame=[],
            disposition_signal=DispositionSignal(
                class_name="fail_persist_for_learning",
                confidence="high",
                reasons=["quality_below_cutoff"],
            ),
            confidence="high",
        )

    def _full_result(
        self,
        image: EvaluationImageInput,
        quality: QualityEvaluation,
        alignment: AlignmentEvaluation,
        impact_payload: dict[str, Any] | None,
        plan: EvaluationPlan,
        iqa_telemetry: dict[str, Any],
        vlm_telemetry: dict[str, Any],
        impact_telemetry: dict[str, Any],
        impact_enabled: bool,
        impact_failed: bool,
    ) -> ImageEvaluationResult:
        quality_pass = quality.score >= plan.quality_cutoff
        alignment_pass = alignment.score >= plan.alignment_cutoff
        full_pass = quality_pass and alignment_pass
        failure_types: list[FailureType] = [] if full_pass else ["alignment_below_cutoff"]
        pass_flags = {"quality": quality_pass, "alignment": alignment_pass, "full": full_pass}
        if impact_enabled and full_pass and plan.impact_cutoff is not None:
            impact_score = impact_payload.get("score") if impact_payload is not None else None
            impact_pass = isinstance(impact_score, (int, float)) and impact_score >= plan.impact_cutoff
            pass_flags["impact"] = impact_pass
            if not impact_pass:
                failure_types.append("impact_below_cutoff" if impact_score is not None else "impact_unavailable")

        evaluator_telemetry = {"iqa": iqa_telemetry, "vlm": vlm_telemetry}
        evaluator_versions = {"iqa": _model_version(quality), "vlm": _model_version(alignment)}
        if impact_failed and full_pass:
            evaluator_telemetry["impact"] = impact_telemetry
        if impact_payload is not None:
            evaluator_telemetry["impact"] = impact_telemetry
            evaluator_versions["impact"] = _model_version(impact_payload)

        disposition = (
            DispositionSignal(
                class_name="passes_thresholds",
                confidence="high",
                reasons=["quality and alignment passed"],
            )
            if full_pass
            else DispositionSignal(
                class_name="fail_persist_for_learning",
                confidence="high",
                reasons=[str(failure) for failure in failure_types],
            )
        )
        return ImageEvaluationResult(
            candidate_id=image.candidate_id,
            image_path=image.image_path,
            seed=image.seed,
            coordinate_id=image.coordinate_id,
            run_id=image.run_id,
            prompt_document_id=image.prompt_document_id,
            target_manifest_id=image.target_manifest_id,
            file_valid=True,
            quality=quality,
            alignment=alignment,
            impact=impact_payload,
            evaluator_request_id=self.request_id,
            evaluator_plan=plan,
            evaluator_telemetry=evaluator_telemetry,
            evaluator_versions=evaluator_versions,
            pass_flags=pass_flags,
            failure_types=failure_types,
            localized_blame=[],
            disposition_signal=disposition,
            confidence="high",
        )

    @staticmethod
    def _attach_batch_metrics(
        result: ImageEvaluationResult,
        batch_metrics: dict[str, float | int],
    ) -> ImageEvaluationResult:
        telemetry = {**result.evaluator_telemetry, "batch": dict(batch_metrics)}
        return result.model_copy(update={"evaluator_telemetry": telemetry})


def _failure_for_target(target_id: str, target_kind: str) -> FailureType:
    if target_kind == "relation":
        return "missing_locked_relation"
    if target_kind == "element":
        return "missing_locked_element"
    if target_id.endswith(".material"):
        return "wrong_material"
    if target_id.endswith(".color"):
        return "wrong_color"
    if target_id == "cinematography.shot_size":
        return "wrong_shot_size"
    if target_id == "cinematography.camera_angle":
        return "wrong_camera_angle"
    if target_id == "cinematography.lighting_mood":
        return "wrong_lighting"
    return "alignment_below_cutoff"


def _model_version(model: QualityEvaluation | AlignmentEvaluation | ImpactEvaluation | dict[str, Any]) -> dict[str, str]:
    if isinstance(model, dict):
        return {
            "model_id": str(model.get("model_id", "unknown")),
            "model_version": str(model.get("model_version", "unknown")),
        }
    return {"model_id": model.model_id, "model_version": model.model_version}


def _failed_stage_telemetry(execution_mode: str, error: Exception) -> dict[str, Any]:
    return {
        "execution_mode": execution_mode,
        "elapsed_ms": 0,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
    }


def apply_target_observations(
    alignment: AlignmentEvaluation,
    manifest: EvaluationTargetManifest,
    observations: list[TargetObservation],
) -> TargetPolicyResult:
    observation_by_id = {observation.target_id: observation for observation in observations}
    failure_types: list[FailureType] = []
    blame: list[BlameHint] = []
    hard_failure = False

    for target in manifest.targets:
        observation = observation_by_id.get(target.target_id)
        if observation is None or observation.confidence == "low":
            continue
        missing = not observation.present
        if not missing:
            continue
        failure = _failure_for_target(target.target_id, target.target_kind)
        failure_types.append(failure)
        source = "sampled" if target.priority == "sampled" else "locked"
        blame.append(
            BlameHint(
                target_id=target.target_id,
                field_path=target.target_id,
                enum_value=target.enum_value,
                source=source,
                blame_type="semantic_mismatch",
                confidence=observation.confidence,
                reason=observation.reason or f"{target.target_id} was not observed",
            )
        )
        if target.evaluation_policy == "must_match" and observation.confidence == "high":
            hard_failure = True

    for target in manifest.negative_targets:
        observation = observation_by_id.get(target.target_id)
        if observation is None or observation.confidence == "low" or not observation.present:
            continue
        failure_types.append("negative_constraint_violation")
        blame.append(
            BlameHint(
                target_id=target.target_id,
                field_path=target.target_id,
                enum_value=target.enum_value,
                source="locked",
                blame_type="constraint_violation",
                confidence=observation.confidence,
                reason=observation.reason or f"{target.target_id} appeared despite negative guard",
            )
        )
        if target.evaluation_policy == "must_not_appear" and observation.confidence == "high":
            hard_failure = True

    alignment_pass = alignment.score >= 0.0 and not hard_failure
    disposition = (
        DispositionSignal(class_name="passes_thresholds", confidence="high", reasons=["target policy passed"])
        if alignment_pass
        else DispositionSignal(
            class_name="fail_persist_for_learning",
            confidence="high",
            reasons=[str(failure) for failure in failure_types],
        )
    )
    return TargetPolicyResult(
        alignment=alignment,
        pass_flags={"alignment": alignment_pass},
        failure_types=failure_types,
        localized_blame=blame,
        disposition_signal=disposition,
    )


def _outcome(promoted_count: int, failure_types: list[FailureType]) -> Literal["strong", "viable", "fragile", "failed", "blocked"]:
    if any(item in {"evaluator_unavailable", "evaluator_timeout", "gpu_memory_failure"} for item in failure_types):
        return "blocked"
    if promoted_count >= 3:
        return "strong"
    if promoted_count == 1:
        return "fragile"
    if promoted_count > 1:
        return "viable"
    return "failed"


def aggregate_seed_sweep(results: list[ImageEvaluationResult]) -> CoordinateEvaluationAggregate:
    if len(results) != len(DEFAULT_SEED_BUNDLE):
        raise ValueError("coordinate aggregate requires the fixed five-seed result set")
    first = results[0]
    quality_scores = [result.quality.score for result in results]
    alignment_scores = [result.alignment.score for result in results]
    full_pass_count = sum(1 for result in results if result.pass_flags.get("full", False))
    failure_types: list[FailureType] = []
    for result in results:
        failure_types.extend(result.failure_types)
    return CoordinateEvaluationAggregate(
        run_id=first.run_id,
        prompt_document_id=first.prompt_document_id,
        target_manifest_id=first.target_manifest_id,
        coordinate_id=first.coordinate_id,
        seeds=[result.seed for result in results],
        generated_count=len(results),
        evaluated_count=len(results),
        promoted_count=full_pass_count,
        quality_pass_count=sum(1 for result in results if result.pass_flags.get("quality", False)),
        alignment_pass_count=sum(1 for result in results if result.pass_flags.get("alignment", False)),
        full_pass_count=full_pass_count,
        mean_quality=sum(quality_scores) / len(quality_scores),
        mean_alignment=sum(alignment_scores) / len(alignment_scores),
        best_quality=max(quality_scores),
        best_alignment=max(alignment_scores),
        pass_rate=full_pass_count / len(results),
        outcome=_outcome(full_pass_count, failure_types),
        aggregate_failure_types=sorted(set(failure_types)),
        aggregate_blame=[blame for result in results for blame in result.localized_blame],
        update_signal=LearningUpdateSignal(
            thompson_alpha_delta=float(full_pass_count),
            thompson_beta_delta=float(len(results) - full_pass_count),
            gp_affinity_delta=(full_pass_count / len(results)) - 0.5,
        ),
    )


def evaluate_with_static_scores(
    request: EvaluationBatchRequest,
    *,
    quality_scores: list[float],
    alignment_scores: list[float],
) -> EvaluationBatchResult:
    if len(quality_scores) != len(request.images) or len(alignment_scores) != len(request.images):
        raise ValueError("score arrays must match image count")
    results: list[ImageEvaluationResult] = []
    for image, quality_score, alignment_score in zip(request.images, quality_scores, alignment_scores, strict=True):
        quality_pass = quality_score >= request.evaluator_plan.quality_cutoff
        alignment_pass = alignment_score >= request.evaluator_plan.alignment_cutoff
        full_pass = quality_pass and alignment_pass
        failure_types: list[FailureType] = []
        if not quality_pass:
            failure_types.append("quality_below_cutoff")
        if not alignment_pass:
            failure_types.append("alignment_below_cutoff")
        disposition = (
            DispositionSignal(class_name="passes_thresholds", confidence="high", reasons=["quality and alignment passed"])
            if full_pass
            else DispositionSignal(
                class_name="fail_persist_for_learning",
                confidence="high",
                reasons=[str(failure) for failure in failure_types],
            )
        )
        results.append(
            ImageEvaluationResult(
                candidate_id=image.candidate_id,
                image_path=image.image_path,
                seed=image.seed,
                coordinate_id=image.coordinate_id,
                run_id=image.run_id,
                prompt_document_id=image.prompt_document_id,
                target_manifest_id=image.target_manifest_id,
                file_valid=True,
                quality=QualityEvaluation(score=quality_score),
                alignment=AlignmentEvaluation(score=alignment_score),
                evaluator_versions={
                    "iqa": _model_version(QualityEvaluation(score=quality_score)),
                    "vlm": _model_version(AlignmentEvaluation(score=alignment_score)),
                },
                pass_flags={"quality": quality_pass, "alignment": alignment_pass, "full": full_pass},
                failure_types=failure_types,
                localized_blame=[],
                disposition_signal=disposition,
                confidence="high",
            )
        )
    return EvaluationBatchResult(images=results, aggregate=aggregate_seed_sweep(results))


def evaluate_images(
    request: EvaluationBatchRequest,
    *,
    iqa: StaticIQAAdapter,
    vlm: StaticVLMAdapter,
    impact: StaticImpactAdapter | None = None,
) -> tuple[list[ImageEvaluationResult], list[str]]:
    if request.evaluator_plan.execution_preference != "auto":
        evaluator = BatchEvaluator(iqa=iqa, vlm=vlm, impact=impact, request_id=request.batch_id)
        results = evaluator.evaluate(
            request.images,
            request.rendered_prompt,
            request.target_manifest,
            request.evaluator_plan,
        )
        return results, evaluator.last_survivor_candidate_ids

    iqa_telemetry = {"execution_mode": "batch", "elapsed_ms": 0}
    try:
        quality_results = iqa.score(request.images)
    except Exception as error:
        iqa_telemetry = _failed_stage_telemetry("batch", error)
        results = [
            ImageEvaluationResult(
                candidate_id=image.candidate_id,
                image_path=image.image_path,
                seed=image.seed,
                coordinate_id=image.coordinate_id,
                run_id=image.run_id,
                prompt_document_id=image.prompt_document_id,
                target_manifest_id=image.target_manifest_id,
                file_valid=True,
                quality=QualityEvaluation(score=0.0, confidence="low"),
                alignment=AlignmentEvaluation(score=0.0, confidence="low"),
                impact=None,
                evaluator_request_id=request.batch_id,
                evaluator_plan=request.evaluator_plan,
                evaluator_telemetry={"iqa": iqa_telemetry},
                evaluator_versions={},
                pass_flags={"quality": False, "alignment": False, "full": False},
                failure_types=["evaluator_unavailable"],
                localized_blame=[],
                disposition_signal=DispositionSignal(
                    class_name="infrastructure_retry_no_semantic_penalty",
                    confidence="high",
                    reasons=["iqa evaluator unavailable"],
                ),
                confidence="low",
            )
            for image in request.images
        ]
        return results, []
    survivor_pairs: list[tuple[EvaluationImageInput, QualityEvaluation]] = [
        (image, quality)
        for image, quality in zip(request.images, quality_results, strict=True)
        if quality.score >= request.evaluator_plan.quality_cutoff
    ]
    survivor_images = [image for image, _quality in survivor_pairs]
    vlm_telemetry = {"execution_mode": "bounded_batch", "elapsed_ms": 0}
    try:
        alignment_results = vlm.score(survivor_images)
    except Exception as error:
        vlm_telemetry = _failed_stage_telemetry("bounded_batch", error)
        results = []
        for image, quality in zip(request.images, quality_results, strict=True):
            quality_pass = quality.score >= request.evaluator_plan.quality_cutoff
            if quality_pass:
                failure_types: list[FailureType] = ["evaluator_unavailable"]
                disposition = DispositionSignal(
                    class_name="infrastructure_retry_no_semantic_penalty",
                    confidence="high",
                    reasons=["vlm evaluator unavailable"],
                )
                evaluator_telemetry = {"iqa": iqa_telemetry, "vlm": vlm_telemetry}
            else:
                failure_types = ["quality_below_cutoff"]
                disposition = DispositionSignal(
                    class_name="fail_persist_for_learning",
                    confidence="high",
                    reasons=[str(failure) for failure in failure_types],
                )
                evaluator_telemetry = {"iqa": iqa_telemetry}
            results.append(
                ImageEvaluationResult(
                    candidate_id=image.candidate_id,
                    image_path=image.image_path,
                    seed=image.seed,
                    coordinate_id=image.coordinate_id,
                    run_id=image.run_id,
                    prompt_document_id=image.prompt_document_id,
                    target_manifest_id=image.target_manifest_id,
                    file_valid=True,
                    quality=quality,
                    alignment=AlignmentEvaluation(score=0.0, confidence="low"),
                    impact=None,
                    evaluator_request_id=request.batch_id,
                    evaluator_plan=request.evaluator_plan,
                    evaluator_telemetry=evaluator_telemetry,
                    evaluator_versions={"iqa": _model_version(quality)},
                    pass_flags={"quality": quality_pass, "alignment": False, "full": False},
                    failure_types=failure_types,
                    localized_blame=[],
                    disposition_signal=disposition,
                    confidence="low" if quality_pass else "high",
                )
            )
        return results, [str(image.candidate_id) for image in survivor_images if image.candidate_id is not None]
    impact_enabled = request.evaluator_plan.metacognitive_impact and impact is not None
    alignment_by_id = {
        image.candidate_id: alignment
        for image, alignment in zip(survivor_images, alignment_results, strict=True)
    }
    impact_images = [
        image
        for image in survivor_images
        if alignment_by_id[image.candidate_id].score >= request.evaluator_plan.alignment_cutoff
    ]
    impact_telemetry = {"execution_mode": "bounded_batch", "elapsed_ms": 0}
    impact_failed = False
    if impact_enabled:
        assert impact is not None
        try:
            impact_payloads = impact.score(impact_images)
        except Exception as error:
            impact_failed = True
            impact_telemetry = _failed_stage_telemetry("bounded_batch", error)
            impact_payloads = [None for _ in impact_images]
    else:
        impact_payloads = [None for _ in impact_images]
    impact_by_id = {
        image.candidate_id: payload
        for image, payload in zip(impact_images, impact_payloads, strict=True)
    }
    survivor_by_id = {
        image.candidate_id: (quality, alignment, impact_by_id.get(image.candidate_id))
        for (image, quality), alignment in zip(
            survivor_pairs,
            alignment_results,
            strict=True,
        )
    }

    results: list[ImageEvaluationResult] = []
    for image, quality in zip(request.images, quality_results, strict=True):
        if image.candidate_id in survivor_by_id:
            quality_eval, alignment_eval, impact_payload = survivor_by_id[image.candidate_id]
            alignment_pass = alignment_eval.score >= request.evaluator_plan.alignment_cutoff
            quality_pass = quality_eval.score >= request.evaluator_plan.quality_cutoff
            full_pass = quality_pass and alignment_pass
            failure_types: list[FailureType] = [] if full_pass else ["alignment_below_cutoff"]
            pass_flags = {"quality": quality_pass, "alignment": alignment_pass, "full": full_pass}
            if impact_enabled and full_pass and request.evaluator_plan.impact_cutoff is not None:
                impact_score = impact_payload.get("score") if impact_payload is not None else None
                impact_pass = isinstance(impact_score, (int, float)) and impact_score >= request.evaluator_plan.impact_cutoff
                pass_flags["impact"] = impact_pass
                if not impact_pass:
                    failure_types.append("impact_below_cutoff" if impact_score is not None else "impact_unavailable")
            evaluator_telemetry = {"iqa": iqa_telemetry, "vlm": vlm_telemetry}
            evaluator_versions = {"iqa": _model_version(quality_eval), "vlm": _model_version(alignment_eval)}
            if impact_failed and full_pass:
                evaluator_telemetry["impact"] = impact_telemetry
            if impact_payload is not None:
                evaluator_telemetry["impact"] = impact_telemetry
                evaluator_versions["impact"] = _model_version(impact_payload)
            disposition = (
                DispositionSignal(
                    class_name="passes_thresholds",
                    confidence="high",
                    reasons=["quality and alignment passed"],
                )
                if full_pass
                else DispositionSignal(
                    class_name="fail_persist_for_learning",
                    confidence="high",
                    reasons=[str(failure) for failure in failure_types],
                )
            )
            results.append(
                ImageEvaluationResult(
                    candidate_id=image.candidate_id,
                    image_path=image.image_path,
                    seed=image.seed,
                    coordinate_id=image.coordinate_id,
                    run_id=image.run_id,
                    prompt_document_id=image.prompt_document_id,
                    target_manifest_id=image.target_manifest_id,
                    file_valid=True,
                    quality=quality_eval,
                    alignment=alignment_eval,
                    impact=impact_payload,
                    evaluator_request_id=request.batch_id,
                    evaluator_plan=request.evaluator_plan,
                    evaluator_telemetry=evaluator_telemetry,
                    evaluator_versions=evaluator_versions,
                    pass_flags=pass_flags,
                    failure_types=failure_types,
                    localized_blame=[],
                    disposition_signal=disposition,
                    confidence="high",
                )
            )
        else:
            disposition = DispositionSignal(
                class_name="fail_persist_for_learning",
                confidence="high",
                reasons=["quality_below_cutoff"],
            )
            results.append(
                ImageEvaluationResult(
                    candidate_id=image.candidate_id,
                    image_path=image.image_path,
                    seed=image.seed,
                    coordinate_id=image.coordinate_id,
                    run_id=image.run_id,
                    prompt_document_id=image.prompt_document_id,
                    target_manifest_id=image.target_manifest_id,
                    file_valid=True,
                    quality=quality,
                    alignment=AlignmentEvaluation(score=0.0, confidence="low"),
                    impact=None,
                    evaluator_request_id=request.batch_id,
                    evaluator_plan=request.evaluator_plan,
                    evaluator_telemetry={"iqa": iqa_telemetry},
                    evaluator_versions={"iqa": _model_version(quality)},
                    pass_flags={"quality": False, "alignment": False, "full": False},
                    failure_types=["quality_below_cutoff"],
                    localized_blame=[],
                    disposition_signal=disposition,
                    confidence="high",
                )
            )

    return results, [str(image.candidate_id) for image in survivor_images if image.candidate_id is not None]


def staged_evaluate(
    request: EvaluationBatchRequest,
    *,
    iqa: StaticIQAAdapter,
    vlm: StaticVLMAdapter,
    impact: StaticImpactAdapter | None = None,
) -> StagedEvaluationResult:
    results, survivor_ids = evaluate_images(request, iqa=iqa, vlm=vlm, impact=impact)
    return StagedEvaluationResult(
        images=results,
        aggregate=aggregate_seed_sweep(results),
        iqa_survivor_candidate_ids=survivor_ids,
    )
