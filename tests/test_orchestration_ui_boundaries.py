import pytest

from bruteforce_canvas.evaluation import (
    AlignmentEvaluation,
    DispositionSignal,
    ImageEvaluationResult,
    QualityEvaluation,
)
from bruteforce_canvas.orchestration import (
    CandidateState,
    apply_evaluation_disposition,
)
from bruteforce_canvas.prompt import (
    EvidenceCategory,
    EvidenceSpan,
    ObjectLane,
    PromptDocumentSpec,
    SceneGraphDraft,
    VerificationIssue,
    VerificationReport,
    render_prompt_spec,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Importance, LightingMood
from bruteforce_canvas.prompt_models import CinematographyLane, Element, ObjectDescriptor
from bruteforce_canvas.shared import FeedbackAction
from bruteforce_canvas.ui import (
    CandidateCard,
    PreRunModalState,
    RunWorkspaceReadModel,
    catalogue_default_items,
    pre_run_modal_from_prompt,
    submit_feedback_event,
)


def test_blocking_verification_prevents_rendering_and_begin_generation():
    document = PromptDocumentSpec(
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
                    message="Specify what the person is throwing or request no visible thrown object.",
                )
            ],
        ),
    )

    modal = pre_run_modal_from_prompt(document)
    assert modal.state == PreRunModalState.BLOCKED
    assert modal.can_begin_generation is False
    assert modal.prompt_improvement_feedback
    with pytest.raises(ValueError, match="unapproved"):
        render_prompt_spec(document)


def test_pre_run_modal_exposes_backend_lock_configuration_without_graph_edit_controls():
    document = PromptDocumentSpec(
        raw_user_prompt="a ceramic bowl in blue hour",
        graph=SceneGraphDraft(
            seed_prompt="ceramic bowl in blue hour",
            elements=[
                Element(
                    id="object_01",
                    label="bowl",
                    entity_type=EntityType.PRODUCT,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="bowl", category=EvidenceCategory.EXPLICIT),
                )
            ]
        ),
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="object_01", material="ceramic")]),
        cinematography_lane=CinematographyLane(lighting_mood=LightingMood.BLUE_HOUR_TWILIGHT),
        verification=VerificationReport(approved=True, issues=[]),
    )

    modal = pre_run_modal_from_prompt(document)
    by_field = {entry["field_path"]: entry for entry in modal.lock_entries}

    assert by_field["object.material.object_01"]["lock_state"] == "locked"
    assert by_field["object.material.object_01"]["lhs_policy"] == "fixed"
    assert by_field["cinematography.shot_size"]["lock_state"] == "unlocked"
    assert by_field["cinematography.shot_size"]["lhs_policy"] == "sampleable_if_missing"
    assert all(not field.startswith("graph.") for field in modal.editable_fields)


def test_evaluator_signal_does_not_mutate_candidate_until_orchestrator_applies_it():
    candidate = CandidateState(
        candidate_id="cand_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seed=7,
        promoted=False,
        curated=False,
    )
    result = ImageEvaluationResult(
        candidate_id="cand_001",
        image_path="/tmp/cand_001.png",
        seed=7,
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
        disposition_signal=DispositionSignal(
            class_name="passes_thresholds",
            confidence="high",
            reasons=["quality and alignment passed"],
        ),
        confidence="high",
    )

    assert candidate.promoted is False
    applied = apply_evaluation_disposition(candidate, result)
    assert applied.promoted is True
    assert applied.curated is True


def test_catalogue_default_excludes_failed_rejected_and_shredded_items_but_keeps_accepted():
    cards = [
        CandidateCard(candidate_id="cand_001", promoted=True, curated=True, feedback_action=None),
        CandidateCard(candidate_id="cand_002", promoted=True, curated=True, feedback_action=FeedbackAction.ACCEPT),
        CandidateCard(candidate_id="cand_003", promoted=True, curated=True, feedback_action=FeedbackAction.REJECT),
        CandidateCard(candidate_id="cand_004", promoted=True, curated=True, feedback_action=FeedbackAction.SHRED),
        CandidateCard(candidate_id="cand_005", promoted=False, curated=False, feedback_action=None),
    ]

    default_items = catalogue_default_items(cards)
    assert [card.candidate_id for card in default_items] == ["cand_001", "cand_002"]


def test_ui_feedback_event_uses_exact_backend_semantics():
    event = submit_feedback_event(run_id="run_001", candidate_id="cand_001", action=FeedbackAction.SHRED)
    assert event.event_type == "feedback_submitted"
    assert event.payload["action"] == "shred"
    assert event.payload["candidate_id"] == "cand_001"


def test_workspace_read_model_has_required_status_surfaces():
    workspace = RunWorkspaceReadModel(
        run_id="run_001",
        raw_user_prompt="a bowl on a table",
        run_state="waiting_for_pre_run_confirmation",
        generated_count=0,
        iqa_evaluated_count=0,
        vlm_evaluated_count=0,
        promoted_curated_count=0,
        accepted_count=0,
        rejected_count=0,
        shredded_count=0,
        stall_guard_state="inactive",
        notification="Waiting for pre-run confirmation.",
    )

    assert workspace.notification == "Waiting for pre-run confirmation."
    assert workspace.progress_heartbeat["run_state"] == "waiting_for_pre_run_confirmation"
    assert workspace.progress_heartbeat["elapsed_seconds"] == 0
