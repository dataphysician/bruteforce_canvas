import pytest

from bruteforce_canvas.evaluation import (
    AlignmentEvaluation,
    DispositionSignal,
    ImageEvaluationResult,
    QualityEvaluation,
)
from bruteforce_canvas.gates import GateError, StageGate
from bruteforce_canvas.generation import CandidateRecord
from bruteforce_canvas.prompt import (
    EvidenceCategory,
    EvidenceSpan,
    PromptDocumentSpec,
    RenderedPrompt,
    SceneGraphDraft,
    VerificationIssue,
    VerificationReport,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Importance
from bruteforce_canvas.prompt_models import Element
from bruteforce_canvas.router import CandidateCoordinateBatch, CompatibilityTrace


def blocked_document() -> PromptDocumentSpec:
    return PromptDocumentSpec(
        raw_user_prompt="person throwing something",
        graph=SceneGraphDraft(
            seed_prompt="person throwing something",
            elements=[
                Element(
                    id="person_01",
                    label="person",
                    entity_type=EntityType.PERSON,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="person", category=EvidenceCategory.EXPLICIT),
                )
            ]
        ),
        verification=VerificationReport(
            approved=False,
            issues=[
                VerificationIssue(
                    issue_type="unresolved_action_target",
                    repair_scope="prompt_improvement",
                    blocking=True,
                    message="Specify target.",
                )
            ],
        ),
    )


def candidate(seed: int, *, file_valid: bool = True) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=f"cand_{seed}",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seed=seed,
        rendered_prompt="Generate a bowl",
        generator_model_id="stub",
        generator_backend="stub",
        generation_settings={},
        image_path=f"/tmp/{seed}.png",
        file_valid=file_valid,
        timestamp="1970-01-01T00:00:00Z",
        generation_elapsed_ms=0,
    )


def image_result(seed: int) -> ImageEvaluationResult:
    return ImageEvaluationResult(
        candidate_id=f"cand_{seed}",
        image_path=f"/tmp/{seed}.png",
        seed=seed,
        coordinate_id="coord_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        file_valid=True,
        quality=QualityEvaluation(score=0.9),
        alignment=AlignmentEvaluation(score=0.9),
        pass_flags={"quality": True, "alignment": True, "full": True},
        failure_types=[],
        localized_blame=[],
        disposition_signal=DispositionSignal(class_name="passes_thresholds", confidence="high", reasons=[]),
        confidence="high",
    )


def test_prompt_gate_blocks_unapproved_or_blocking_documents():
    with pytest.raises(GateError, match="PromptDocument verification did not pass"):
        StageGate.prompt(blocked_document())


def test_router_gate_blocks_batches_with_hard_rejects_or_no_coordinates():
    with pytest.raises(GateError, match="hard compatibility reject"):
        StageGate.router(
            CandidateCoordinateBatch(
                coordinates=[],
                rejected_traces=[CompatibilityTrace(hard_rejects=["rejected combo x"])],
            )
        )


def test_rendering_gate_requires_generate_prefix_and_nonempty_prompt():
    with pytest.raises(GateError, match="must begin with Generate"):
        StageGate.rendering(
            RenderedPrompt.model_construct(
                prompt_document_id="doc_001",
                rendered_prompt="Make a bowl",
                rendering_trace=[],
            )
        )


def test_generation_gate_accepts_valid_or_infrastructure_blocked_only():
    StageGate.generation([candidate(seed) for seed in [7, 42, 156, 8888, 42069]])
    with pytest.raises(GateError, match="not valid and not infrastructure-blocked"):
        StageGate.generation([candidate(seed, file_valid=(seed != 42)) for seed in [7, 42, 156, 8888, 42069]])
    StageGate.generation(
        [candidate(seed, file_valid=(seed != 42)) for seed in [7, 42, 156, 8888, 42069]],
        infrastructure_blocked_candidate_ids={"cand_42"},
    )


def test_evaluation_gate_accepts_three_or_more_seeds():
    StageGate.evaluation([image_result(seed) for seed in [7, 42, 156, 8888, 42069]])
    StageGate.evaluation([image_result(seed) for seed in [7, 42, 156]])
    StageGate.evaluation([image_result(seed) for seed in [7, 42, 156, 8888, 42069, 100, 200]])


def test_evaluation_gate_rejects_below_minimum_seed_count():
    with pytest.raises(GateError, match="at least 3 seeds"):
        StageGate.evaluation([image_result(seed) for seed in [7, 42]])


def test_generation_gate_accepts_three_or_more_seeds():
    StageGate.generation([candidate(seed) for seed in [7, 42, 156]])
    StageGate.generation([candidate(seed) for seed in [7, 42, 156, 8888, 42069]])
    StageGate.generation([candidate(seed) for seed in [7, 42, 156, 8888, 42069, 100, 200]])


def test_generation_gate_rejects_below_minimum_seed_count():
    with pytest.raises(GateError, match="at least 3 seeds"):
        StageGate.generation([candidate(seed) for seed in [7, 42]])
