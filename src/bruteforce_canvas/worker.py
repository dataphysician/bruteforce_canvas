from __future__ import annotations

from typing import Any

from pydantic import Field

from bruteforce_canvas.actions import decide_coordinate_actions, decide_image_actions
from bruteforce_canvas.evaluation import (
    AlignmentEvaluation,
    EvaluationBatchRequest,
    EvaluationImageInput,
    EvaluationPlan,
    ImageEvaluationResult,
    LearningUpdateSignal,
    QualityEvaluation,
    StaticIQAAdapter,
    StaticImpactAdapter,
    StaticVLMAdapter,
    StagedEvaluationResult,
    aggregate_seed_sweep,
    evaluate_images,
    staged_evaluate,
)
from bruteforce_canvas.generation import GenerationRequest, GenerationResult
from bruteforce_canvas.learning import (
    ComboAffinityState,
    LearningEvent,
    LearningState,
    apply_coordinate_learning,
    coordinate_quarantine_decision,
    enum_suppression_decision,
)
from bruteforce_canvas.persistence import PERSISTENCE_VERSION, JsonlEventStore, PersistenceRecord
from bruteforce_canvas.router import CompatibilityTrace
from bruteforce_canvas.seed_surfing import SeedSurfPolicy, enqueue_seed_surf_bundle
from bruteforce_canvas.shared import CoordinateId, RunId, StrictModel


class SeedSweepWorkItem(StrictModel):
    run_id: RunId
    raw_user_prompt: str
    prompt_document_version: str = "1"
    coordinate_id: CoordinateId
    parent_coordinate_id: CoordinateId | None = None
    rendered_prompt: str
    target_manifest: dict[str, Any]
    generation_requests: list[GenerationRequest]
    evaluation_plan: EvaluationPlan
    sampled_arms: dict[str, str] = Field(default_factory=dict)
    locked_arms: dict[str, str] = Field(default_factory=dict)
    lhs_row: dict[str, int] = Field(default_factory=dict)
    lock_configuration: dict[str, Any] = Field(default_factory=dict)
    default_lock_configuration: dict[str, Any] = Field(default_factory=dict)
    effective_lock_configuration: dict[str, Any] = Field(default_factory=dict)
    verifier_result: dict[str, Any] = Field(default_factory=dict)
    compatibility_trace: CompatibilityTrace = Field(default_factory=CompatibilityTrace)
    bayesian_score_before_generation: float = 1.0
    combo_signature: str


class PersistentSeedSweepWorker:
    def __init__(
        self,
        *,
        store: JsonlEventStore,
        generator: object,
        iqa: StaticIQAAdapter,
        vlm: StaticVLMAdapter,
        impact: StaticImpactAdapter | None = None,
        seed_surf_policy: SeedSurfPolicy | None = None,
    ) -> None:
        self.store = store
        self.generator = generator
        self.iqa = iqa
        self.vlm = vlm
        self.impact = impact
        self.seed_surf_policy = seed_surf_policy

    def run_seed_sweep(self, item: SeedSweepWorkItem) -> StagedEvaluationResult:
        first = item.generation_requests[0]
        self.store.append(
            PersistenceRecord(
                record_id=f"run_config:{item.run_id}",
                record_type="run_config",
                run_id=item.run_id,
                idempotency_key=f"run_config:{item.run_id}",
                payload=self._run_config_payload(item, first),
            )
        )
        self.store.append(
            PersistenceRecord(
                record_id=f"coordinate:{item.coordinate_id}",
                record_type="coordinate_record",
                run_id=item.run_id,
                prompt_document_id=first.prompt_document_id,
                target_manifest_id=first.target_manifest_id,
                coordinate_id=item.coordinate_id,
                idempotency_key=f"coordinate:{item.coordinate_id}",
                payload={
                    "rendered_prompt": item.rendered_prompt,
                    "fixed_arms": item.locked_arms,
                    "sampled_arms": item.sampled_arms,
                    "lhs_row": item.lhs_row,
                    "compatibility_trace": item.compatibility_trace.model_dump(),
                    "bayesian_score": item.bayesian_score_before_generation,
                    "combo_signature": item.combo_signature,
                    "parent_coordinate_id": item.parent_coordinate_id,
                    "lifecycle_state": "proposed",
                },
            )
        )

        images: list[EvaluationImageInput] = []
        generation_results: list[GenerationResult] = []
        for request in item.generation_requests:
            generation_result = self.generator.generate(request)
            generation_results.append(generation_result)
            candidate = generation_result.candidate
            self.store.append(
                PersistenceRecord(
                    record_id=f"candidate:{candidate.candidate_id}",
                    record_type="candidate_record",
                    run_id=candidate.run_id,
                    prompt_document_id=candidate.prompt_document_id,
                    target_manifest_id=candidate.target_manifest_id,
                    coordinate_id=candidate.coordinate_id,
                    candidate_id=candidate.candidate_id,
                    seed=candidate.seed,
                    idempotency_key=f"candidate:{candidate.candidate_id}",
                    payload={
                        **candidate.model_dump(),
                        "raw_user_prompt": item.raw_user_prompt,
                        "prompt_document_version": item.prompt_document_version,
                        "promotion_thresholds": self._promotion_thresholds(item.evaluation_plan),
                        "coordinate_enum_json": self._coordinate_enum_json(item),
                        "compatibility_trace": item.compatibility_trace.model_dump(),
                        "bayesian_score_before_generation": item.bayesian_score_before_generation,
                    },
                )
            )
            if candidate.file_valid and not generation_result.infrastructure_blocked:
                images.append(
                    EvaluationImageInput(
                        candidate_id=candidate.candidate_id,
                        image_path=candidate.image_path,
                        seed=candidate.seed,
                        coordinate_id=candidate.coordinate_id,
                        run_id=candidate.run_id,
                        prompt_document_id=candidate.prompt_document_id,
                        target_manifest_id=candidate.target_manifest_id,
                        generation_settings=candidate.generation_settings,
                    )
                )

        request = EvaluationBatchRequest(
            batch_id=f"batch:{item.coordinate_id}",
            run_id=item.run_id,
            prompt_document_id=first.prompt_document_id,
            target_manifest_id=first.target_manifest_id,
            batch_kind="seed_sweep" if len(images) == len(item.generation_requests) else "mixed_image_batch",
            coordinate_id=item.coordinate_id,
            rendered_prompt=item.rendered_prompt,
            target_manifest=item.target_manifest,
            images=images,
            evaluator_plan=item.evaluation_plan,
        )
        if len(images) == len(item.generation_requests):
            valid_evaluation = staged_evaluate(request, iqa=self.iqa, vlm=self.vlm, impact=self.impact)
            valid_results = valid_evaluation.images
            survivor_ids = valid_evaluation.iqa_survivor_candidate_ids
        else:
            valid_results, survivor_ids = self._evaluate_valid_images(request)
        evaluated_by_candidate = {result.candidate_id: result for result in valid_results}
        ordered_results: list[ImageEvaluationResult] = []
        for generation_result in generation_results:
            candidate = generation_result.candidate
            existing = evaluated_by_candidate.get(candidate.candidate_id)
            if existing is not None:
                ordered_results.append(existing)
                continue
            failure_type = "gpu_memory_failure" if generation_result.infrastructure_blocked else "invalid_image_file"
            ordered_results.append(
                ImageEvaluationResult(
                    candidate_id=candidate.candidate_id,
                    image_path=candidate.image_path,
                    seed=candidate.seed,
                    coordinate_id=candidate.coordinate_id,
                    run_id=candidate.run_id,
                    prompt_document_id=candidate.prompt_document_id,
                    target_manifest_id=candidate.target_manifest_id,
                    file_valid=candidate.file_valid,
                    quality=QualityEvaluation(score=0.0, confidence="low"),
                    alignment=AlignmentEvaluation(score=0.0, confidence="low"),
                    evaluator_request_id=request.batch_id,
                    evaluator_plan=item.evaluation_plan,
                    evaluator_telemetry={},
                    pass_flags={"quality": False, "alignment": False, "full": False},
                    failure_types=[failure_type],
                    localized_blame=[],
                    disposition_signal=generation_result.disposition_signal,
                    confidence="low",
                )
            )
        evaluation = StagedEvaluationResult(
            images=ordered_results,
            aggregate=self._aggregate_without_infrastructure_penalty(ordered_results),
            iqa_survivor_candidate_ids=survivor_ids,
        )

        for image_result in evaluation.images:
            self.store.append(
                PersistenceRecord(
                    record_id=f"image_evaluation:{image_result.candidate_id or image_result.seed}",
                    record_type="image_evaluation",
                    run_id=image_result.run_id,
                    prompt_document_id=image_result.prompt_document_id,
                    target_manifest_id=image_result.target_manifest_id,
                    coordinate_id=image_result.coordinate_id,
                    candidate_id=image_result.candidate_id,
                    seed=image_result.seed,
                    idempotency_key=f"image_evaluation:{image_result.candidate_id or image_result.seed}",
                    payload=image_result.model_dump(),
                )
            )
            for action in decide_image_actions(image_result):
                action_name = str(action.name)
                action_key = f"system_action:{action_name}:{image_result.candidate_id or image_result.seed}"
                self.store.append(
                    PersistenceRecord(
                        record_id=action_key,
                        record_type="system_action",
                        run_id=image_result.run_id,
                        prompt_document_id=image_result.prompt_document_id,
                        target_manifest_id=image_result.target_manifest_id,
                        coordinate_id=image_result.coordinate_id,
                        candidate_id=image_result.candidate_id,
                        seed=image_result.seed,
                        idempotency_key=action_key,
                        payload={
                            "action_name": action_name,
                            "reasons": action.reasons,
                            "semantic_penalty": action.semantic_penalty,
                            "persistence_version": PERSISTENCE_VERSION,
                        },
                    )
                )

        self.store.append(
            PersistenceRecord(
                record_id=f"evaluation_aggregate:{item.coordinate_id}",
                record_type="evaluation_aggregate",
                run_id=item.run_id,
                prompt_document_id=first.prompt_document_id,
                target_manifest_id=first.target_manifest_id,
                coordinate_id=item.coordinate_id,
                idempotency_key=f"evaluation_aggregate:{item.coordinate_id}",
                payload=evaluation.aggregate.model_dump(),
            )
        )
        self._persist_seed_surf_bundle(item, evaluation.aggregate, first)
        for action in decide_coordinate_actions(evaluation.aggregate):
            action_name = str(action.name)
            action_key = f"system_action:{action_name}:{item.coordinate_id}"
            self.store.append(
                PersistenceRecord(
                    record_id=action_key,
                    record_type="system_action",
                    run_id=item.run_id,
                    prompt_document_id=first.prompt_document_id,
                    target_manifest_id=first.target_manifest_id,
                    coordinate_id=item.coordinate_id,
                    idempotency_key=action_key,
                    payload={
                        "action_name": action_name,
                        "reasons": action.reasons,
                        "semantic_penalty": action.semantic_penalty,
                        "persistence_version": PERSISTENCE_VERSION,
                    },
                )
            )
        learning_state = apply_coordinate_learning(
            LearningState(),
            LearningEvent(
                event_id=f"eval:{item.coordinate_id}",
                coordinate_id=item.coordinate_id,
                sampled_arms=item.sampled_arms,
                locked_arms=item.locked_arms,
                combo_signature=item.combo_signature,
                aggregate=evaluation.aggregate,
            ),
        )
        self.store.append(
            PersistenceRecord(
                record_id=f"learning_delta:{item.coordinate_id}",
                record_type="learning_delta",
                run_id=item.run_id,
                prompt_document_id=first.prompt_document_id,
                target_manifest_id=first.target_manifest_id,
                coordinate_id=item.coordinate_id,
                idempotency_key=f"eval:{item.coordinate_id}",
                payload={
                    **learning_state.model_dump(),
                    **self._learning_summary(
                        evaluation.aggregate,
                        learning_state,
                        item.combo_signature,
                        item.locked_arms,
                    ),
                    "learning_signal_source": "automated_evaluation",
                    "persistence_version": PERSISTENCE_VERSION,
                },
            )
        )
        return evaluation

    def _aggregate_without_infrastructure_penalty(self, results: list[ImageEvaluationResult]):
        aggregate = aggregate_seed_sweep(results)
        no_semantic_penalty = {
            "invalid_image_file",
            "image_decode_failed",
            "evaluator_unavailable",
            "evaluator_timeout",
            "gpu_memory_failure",
        }
        non_semantic_failures = sum(
            1 for result in results if any(failure in no_semantic_penalty for failure in result.failure_types)
        )
        beta_delta = max(0.0, float(len(results) - aggregate.promoted_count - non_semantic_failures))
        denominator = max(1.0, float(aggregate.promoted_count) + beta_delta)
        return aggregate.model_copy(
            update={
                "update_signal": LearningUpdateSignal(
                    thompson_alpha_delta=float(aggregate.promoted_count),
                    thompson_beta_delta=beta_delta,
                    gp_affinity_delta=(float(aggregate.promoted_count) / denominator) - 0.5,
                )
            }
        )

    def _promotion_thresholds(self, plan: EvaluationPlan) -> dict[str, float | None]:
        return {
            "quality_cutoff": plan.quality_cutoff,
            "alignment_cutoff": plan.alignment_cutoff,
            "human_quality_cutoff": plan.human_quality_cutoff,
            "impact_cutoff": plan.impact_cutoff,
        }

    def _run_config_payload(self, item: SeedSweepWorkItem, first: GenerationRequest) -> dict[str, object]:
        model_versions: dict[str, dict[str, str]] = {
            "generator": {
                "model_id": first.generator_model_id,
                "backend": first.generator_backend,
                "model_version": str(getattr(self.generator, "model_version", "1")),
            },
            "iqa": {
                "model_id": str(getattr(self.iqa, "model_id", "unknown")),
                "model_version": str(getattr(self.iqa, "model_version", "unknown")),
            },
            "vlm": {
                "model_id": str(getattr(self.vlm, "model_id", "unknown")),
                "model_version": str(getattr(self.vlm, "model_version", "unknown")),
            },
        }
        if self.impact is not None:
            model_versions["impact"] = {
                "model_id": str(getattr(self.impact, "model_id", "unknown")),
                "model_version": str(getattr(self.impact, "model_version", "unknown")),
            }
        return {
            "raw_user_prompt": item.raw_user_prompt,
            "prompt_document_id": first.prompt_document_id,
            "prompt_document_version": item.prompt_document_version,
            "target_manifest_id": first.target_manifest_id,
            "thresholds": self._promotion_thresholds(item.evaluation_plan),
            "seed_bundle": [request.seed for request in item.generation_requests],
            "model_versions": model_versions,
            "lock_configuration": item.lock_configuration,
            "default_lock_configuration": item.default_lock_configuration or item.lock_configuration,
            "effective_lock_configuration": item.effective_lock_configuration or item.lock_configuration,
            "verifier_result": item.verifier_result,
        }

    def _coordinate_enum_json(self, item: SeedSweepWorkItem) -> dict[str, object]:
        return {
            "locked_arms": item.locked_arms,
            "sampled_arms": item.sampled_arms,
            "combo_signature": item.combo_signature,
        }

    def _evaluate_valid_images(self, request: EvaluationBatchRequest) -> tuple[list[ImageEvaluationResult], list[str]]:
        return evaluate_images(request, iqa=self.iqa, vlm=self.vlm, impact=self.impact)

    def _persist_seed_surf_bundle(self, item: SeedSweepWorkItem, aggregate, first: GenerationRequest) -> None:
        if self.seed_surf_policy is None:
            return
        bundle = enqueue_seed_surf_bundle(aggregate, self.seed_surf_policy)
        if bundle is None:
            return
        key = f"seed_surf_bundle:{item.coordinate_id}"
        self.store.append(
            PersistenceRecord(
                record_id=key,
                record_type="seed_surf_bundle",
                run_id=item.run_id,
                prompt_document_id=first.prompt_document_id,
                target_manifest_id=first.target_manifest_id,
                coordinate_id=item.coordinate_id,
                idempotency_key=key,
                payload={
                    **bundle.model_dump(),
                    "rendered_prompt": item.rendered_prompt,
                    "fixed_arms": item.locked_arms,
                    "sampled_arms": item.sampled_arms,
                    "compatibility_trace": item.compatibility_trace.model_dump(),
                    "bayesian_score": item.bayesian_score_before_generation,
                    "combo_signature": item.combo_signature,
                },
            )
        )

    def _learning_summary(
        self,
        aggregate,
        learning_state: LearningState,
        combo_signature: str,
        locked_arms: dict[str, str],
    ) -> dict[str, object]:
        lifecycle_by_outcome = {
            "strong": "strong",
            "viable": "viable",
            "fragile": "retired",
            "failed": "retired",
            "blocked": "blocked",
        }
        suppression_records = []
        for arm in learning_state.enum_arms.values():
            if arm.locked_reliability_observations > 0:
                continue
            decision = enum_suppression_decision(
                arm,
                repeated_failure_types=aggregate.aggregate_failure_types,
                user_authored_locked=False,
            )
            suppression_records.append(
                {
                    "field_path": arm.axis,
                    "enum_value": arm.value,
                    "context_key": arm.context_key,
                    "reason": decision.reason,
                    "suppressed": decision.suppress,
                    "suppression_state": decision.state,
                    "suppressed_until": "cooldown:500_generated_candidates" if decision.suppress else None,
                    "min_exploration_probability": 0.01,
                }
            )
        combo = learning_state.combo_affinities.get(combo_signature, ComboAffinityState(combo_signature=combo_signature))
        quarantine = coordinate_quarantine_decision(aggregate, combo)
        quarantine_records = [
            {
                "coordinate_id": aggregate.coordinate_id,
                "combo_signature": combo_signature,
                "reason": quarantine.reason,
                "quarantined": quarantine.quarantine,
            }
        ]
        locked_reliability_records = []
        for axis, value in locked_arms.items():
            key = f"{axis}={value}"
            arm = learning_state.enum_arms[key]
            locked_reliability_records.append(
                {
                    "field_path": axis,
                    "enum_value": value,
                    "observations_delta": 1,
                    "total_observations": arm.locked_reliability_observations,
                    "outcome": aggregate.outcome,
                    "failure_types": aggregate.aggregate_failure_types,
                }
            )
        return {
            "promotion_curation_state": {
                "promoted_count": aggregate.promoted_count,
                "curated_count": aggregate.promoted_count,
                "pass_rate": aggregate.pass_rate,
            },
            "thompson_delta": {
                "alpha": aggregate.update_signal.thompson_alpha_delta,
                "beta": aggregate.update_signal.thompson_beta_delta,
            },
            "gp_combo_affinity_delta": aggregate.update_signal.gp_affinity_delta,
            "coordinate_lifecycle_update": {
                "from": "evaluated",
                "to": lifecycle_by_outcome[aggregate.outcome],
                "reason": aggregate.outcome,
            },
            "suppression_counters": {
                "checked": len(suppression_records),
                "suppressed": sum(1 for record in suppression_records if record["suppressed"]),
            },
            "quarantine_counters": {
                "checked": 1,
                "quarantined": int(quarantine.quarantine),
            },
            "suppression_records": suppression_records,
            "quarantine_records": quarantine_records,
            "locked_reliability_records": locked_reliability_records,
        }
