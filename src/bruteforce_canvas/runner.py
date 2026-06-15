from __future__ import annotations

from pathlib import Path

from pydantic import Field

from bruteforce_canvas.evaluation import (
    CoordinateEvaluationAggregate,
    EvaluationBatchRequest,
    EvaluationImageInput,
    EvaluationPlan,
    evaluate_with_static_scores,
)
from bruteforce_canvas.generation import GenerationSettings, seed_sweep_requests
from bruteforce_canvas.learning import LearningEvent, LearningState, apply_coordinate_learning
from bruteforce_canvas.locking import apply_lock_overrides, build_default_lock_config
from bruteforce_canvas.orchestration import CandidateState, RunConfig, apply_evaluation_disposition
from bruteforce_canvas.persistence import PersistenceRecord
from bruteforce_canvas.prompt import PromptDocument, render_prompt, target_manifest_from_prompt
from bruteforce_canvas.router import AxisDomain, FieldState, LHSRouter, RouterInput, ThompsonArmState
from bruteforce_canvas.shared import StrictModel


PNG_STUB = b"\x89PNG\r\n\x1a\n"


class RunOnceResult(StrictModel):
    generated_seeds: list[int] = Field(default_factory=list)
    generated_paths: list[Path] = Field(default_factory=list)
    curated_count: int = 0
    aggregate: CoordinateEvaluationAggregate | None = None
    learning_state: LearningState = Field(default_factory=LearningState)
    persisted_records: list[PersistenceRecord] = Field(default_factory=list)


class InMemoryRunEngine:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def run_once(
        self,
        *,
        run_id: str,
        document: PromptDocument,
        quality_scores: list[float],
        alignment_scores: list[float],
    ) -> RunOnceResult:
        config = RunConfig(run_id=run_id, raw_user_prompt=document.raw_user_prompt)
        persisted: list[PersistenceRecord] = [
            PersistenceRecord(
                record_id="rec_001",
                record_type="run_config",
                run_id=run_id,
                payload=config.model_dump(),
            )
        ]

        if not document.verification.approved or any(issue.blocking for issue in document.verification.issues):
            persisted.append(
                PersistenceRecord(
                    record_id="rec_002",
                    record_type="prompt_blocked",
                    run_id=run_id,
                    prompt_document_id=document.prompt_document_id,
                    payload={"issues": [issue.model_dump() for issue in document.verification.issues]},
                )
            )
            return RunOnceResult(persisted_records=persisted)

        rendered = render_prompt(document)
        manifest = target_manifest_from_prompt(run_id, rendered, document)
        default_lock_config = build_default_lock_config(document)
        effective_lock_config = apply_lock_overrides(default_lock_config, [])
        persisted.extend(
            [
                PersistenceRecord(
                    record_id="rec_002",
                    record_type="prompt_document",
                    run_id=run_id,
                    prompt_document_id=document.prompt_document_id,
                    payload=document.model_dump(),
                ),
                PersistenceRecord(
                    record_id="rec_003",
                    record_type="target_manifest",
                    run_id=run_id,
                    prompt_document_id=document.prompt_document_id,
                    target_manifest_id=manifest.manifest_id,
                    payload=manifest.model_dump(),
                ),
                PersistenceRecord(
                    record_id="rec_004",
                    record_type="default_lock_config",
                    run_id=run_id,
                    prompt_document_id=document.prompt_document_id,
                    payload=default_lock_config.model_dump(),
                ),
                PersistenceRecord(
                    record_id="rec_005",
                    record_type="effective_lock_config",
                    run_id=run_id,
                    prompt_document_id=document.prompt_document_id,
                    payload=effective_lock_config.model_dump(),
                ),
            ]
        )

        fixed_arms = {
            entry.field_path: AxisDomain(value=entry.enum_value or entry.raw_value or "LOCKED_RAW", state=FieldState.EXPLICIT_LOCKED, source=entry.lock_source)
            for entry in effective_lock_config.entries
            if entry.lhs_policy == "fixed"
        }
        router = LHSRouter(seed=7)
        coordinate = router.propose(
            RouterInput(
                run_id=run_id,
                prompt_document_id=document.prompt_document_id,
                target_manifest_id=manifest.manifest_id,
                fixed_arms=fixed_arms,
                sampleable_axes={
                    "cinematography.shot_size": [
                        ThompsonArmState(axis="cinematography.shot_size", value="MEDIUM_SHOT", alpha=2, beta=1)
                    ]
                },
                count=1,
            )
        ).coordinates[0]

        generated = seed_sweep_requests(
            run_id=run_id,
            prompt_document_id=document.prompt_document_id,
            target_manifest_id=manifest.manifest_id,
            coordinate_id=coordinate.coordinate_id,
            rendered_prompt=rendered.rendered_prompt,
            generation_settings=GenerationSettings(),
            output_dir=self.output_dir,
            generator_model_id="in-memory-stub-generator",
            generator_backend="stub",
        )
        image_inputs: list[EvaluationImageInput] = []
        generated_paths: list[Path] = []
        for index, request in enumerate(generated, start=1):
            path = Path(request.image_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(PNG_STUB)
            generated_paths.append(path)
            candidate_id = f"cand_{request.seed}"
            persisted.append(
                PersistenceRecord(
                    record_id=f"rec_candidate_{index:03d}",
                    record_type="candidate_record",
                    run_id=run_id,
                    prompt_document_id=document.prompt_document_id,
                    target_manifest_id=manifest.manifest_id,
                    coordinate_id=coordinate.coordinate_id,
                    candidate_id=candidate_id,
                    seed=request.seed,
                    payload=request.model_dump(),
                )
            )
            image_inputs.append(
                EvaluationImageInput(
                    candidate_id=candidate_id,
                    image_path=str(path),
                    seed=request.seed,
                    coordinate_id=coordinate.coordinate_id,
                    run_id=run_id,
                    prompt_document_id=document.prompt_document_id,
                    target_manifest_id=manifest.manifest_id,
                    generation_settings=request.generation_settings.model_dump(),
                )
            )

        evaluation = evaluate_with_static_scores(
            EvaluationBatchRequest(
                batch_id="batch_001",
                run_id=run_id,
                prompt_document_id=document.prompt_document_id,
                target_manifest_id=manifest.manifest_id,
                batch_kind="seed_sweep",
                coordinate_id=coordinate.coordinate_id,
                rendered_prompt=rendered.rendered_prompt,
                target_manifest=manifest.model_dump(),
                images=image_inputs,
                evaluator_plan=EvaluationPlan(
                    quality_cutoff=config.iqa_cutoff,
                    alignment_cutoff=config.alignment_cutoff,
                    human_quality_cutoff=config.human_iqa_cutoff,
                ),
            ),
            quality_scores=quality_scores,
            alignment_scores=alignment_scores,
        )

        curated_count = 0
        for result in evaluation.images:
            candidate_state = CandidateState(
                candidate_id=result.candidate_id or "cand_unknown",
                run_id=run_id,
                prompt_document_id=document.prompt_document_id,
                target_manifest_id=manifest.manifest_id,
                coordinate_id=coordinate.coordinate_id,
                seed=result.seed,
            )
            applied = apply_evaluation_disposition(candidate_state, result)
            if applied.curated:
                curated_count += 1

        learning_state = apply_coordinate_learning(
            LearningState(),
            LearningEvent(
                event_id=f"eval:{coordinate.coordinate_id}",
                coordinate_id=coordinate.coordinate_id,
                sampled_arms=coordinate.sampled_arms,
                locked_arms=coordinate.fixed_arms,
                combo_signature=coordinate.combo_signature,
                aggregate=evaluation.aggregate,
            ),
        )
        persisted.extend(
            [
                PersistenceRecord(
                    record_id="rec_evaluation_aggregate",
                    record_type="evaluation_aggregate",
                    run_id=run_id,
                    prompt_document_id=document.prompt_document_id,
                    target_manifest_id=manifest.manifest_id,
                    coordinate_id=coordinate.coordinate_id,
                    payload=evaluation.aggregate.model_dump(),
                ),
                PersistenceRecord(
                    record_id="rec_learning_delta",
                    record_type="learning_delta",
                    run_id=run_id,
                    prompt_document_id=document.prompt_document_id,
                    target_manifest_id=manifest.manifest_id,
                    coordinate_id=coordinate.coordinate_id,
                    idempotency_key=f"eval:{coordinate.coordinate_id}",
                    payload=learning_state.model_dump(),
                ),
            ]
        )

        return RunOnceResult(
            generated_seeds=[request.seed for request in generated],
            generated_paths=generated_paths,
            curated_count=curated_count,
            aggregate=evaluation.aggregate,
            learning_state=learning_state,
            persisted_records=persisted,
        )
