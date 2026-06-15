from __future__ import annotations

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
    execution_preference: Literal["auto", "serialized", "parallel"] = "auto"


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


def _model_version(model: QualityEvaluation | AlignmentEvaluation | dict[str, Any]) -> dict[str, str]:
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
        else DispositionSignal(class_name="fail_persist_for_learning", confidence="high", reasons=failure_types)
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
            else DispositionSignal(class_name="fail_persist_for_learning", confidence="high", reasons=failure_types)
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
                    reasons=failure_types,
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
                impact_pass = impact_score is not None and impact_score >= request.evaluator_plan.impact_cutoff
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
                    reasons=failure_types,
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
