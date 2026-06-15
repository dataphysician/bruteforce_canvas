import pytest

from bruteforce_canvas.ui import (
    CandidateCard,
    DetailReport,
    GraphEditRejected,
    PreRunEditableField,
    PreRunModalState,
    RunControl,
    begin_generation_event,
    cancel_pre_run_event,
    run_control_event,
    validate_pre_run_edit,
)
from bruteforce_canvas.shared import FeedbackAction


def test_start_control_requires_prompt_and_opens_pre_run_not_generation():
    with pytest.raises(ValueError, match="non-empty prompt"):
        run_control_event(control=RunControl.START, run_id="run_001", prompt="  ")

    event = run_control_event(control=RunControl.START, run_id="run_001", prompt="a bowl on a table")

    assert event.event_type == "run_start_intent"
    assert event.event_id
    assert event.timestamp.endswith("Z")
    assert event.payload["opens_pre_run_modal"] is True
    assert event.payload["begin_generation"] is False


def test_pause_and_stop_emit_intents_without_discarding_history():
    pause = run_control_event(control=RunControl.PAUSE, run_id="run_001", prompt="a bowl")
    stop = run_control_event(control=RunControl.STOP, run_id="run_001", prompt="a bowl")

    assert pause.event_type == "run_pause_intent"
    assert pause.payload["discard_history"] is False
    assert stop.event_type == "run_stop_intent"
    assert stop.payload["erase_run_history"] is False


def test_begin_generation_event_requires_modal_ready_state():
    with pytest.raises(ValueError, match="pre-run modal is not ready"):
        begin_generation_event(run_id="run_001", modal_state=PreRunModalState.BLOCKED)

    event = begin_generation_event(run_id="run_001", modal_state=PreRunModalState.READY_TO_BEGIN)
    assert event.event_type == "pre_run_begin"


def test_cancel_pre_run_event_closes_modal_without_erasing_history():
    event = cancel_pre_run_event(run_id="run_001")

    assert event.event_type == "pre_run_cancel"
    assert event.payload["begin_generation"] is False
    assert event.payload["erase_run_history"] is False


def test_pre_run_edit_rejects_graph_breaking_fields_but_allows_fluid_fields():
    with pytest.raises(GraphEditRejected):
        validate_pre_run_edit(PreRunEditableField(field_path="graph.elements.object_01.label", value="vase"))

    allowed = validate_pre_run_edit(
        PreRunEditableField(field_path="cinematography.shot_size", value="medium shot")
    )
    assert allowed.field_path == "cinematography.shot_size"


def test_detail_report_contains_required_provenance_and_optional_tags_do_not_gate():
    card = CandidateCard(
        candidate_id="cand_001",
        promoted=True,
        curated=True,
        feedback_action=None,
        optional_tags=["minimal", "warm"],
        seed=7,
    )
    report = DetailReport.from_candidate_card(
        card,
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        rendered_prompt="Generate a bowl on a table",
        generator_model_id="stub-generator",
        generator_backend="stub",
        generation_settings={"steps": 4},
        quality_score=0.8,
        alignment_score=0.7,
        promotion_gate_reasons=["quality and alignment passed"],
    )

    assert report.candidate_id == "cand_001"
    assert report.seed == 7
    assert report.optional_tags == ["minimal", "warm"]
    assert report.optional_tags_gate_curation is False
    assert report.feedback_state is None


def test_detail_report_feedback_state_uses_exact_actions():
    card = CandidateCard(
        candidate_id="cand_001",
        promoted=True,
        curated=True,
        feedback_action=FeedbackAction.ACCEPT,
    )
    report = DetailReport.from_candidate_card(
        card,
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        rendered_prompt="Generate a bowl on a table",
        generator_model_id="stub-generator",
        generator_backend="stub",
        generation_settings={},
        quality_score=0.8,
        alignment_score=0.7,
        promotion_gate_reasons=[],
    )

    assert report.feedback_state == "accept"


def test_detail_report_feedback_state_marks_pending_action_in_flight():
    card = CandidateCard(
        candidate_id="cand_001",
        promoted=True,
        curated=True,
        feedback_action=FeedbackAction.REJECT,
        feedback_pending=True,
    )
    report = DetailReport.from_candidate_card(
        card,
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        rendered_prompt="Generate a bowl on a table",
        generator_model_id="stub-generator",
        generator_backend="stub",
        generation_settings={},
        quality_score=0.8,
        alignment_score=0.7,
        promotion_gate_reasons=[],
    )

    assert report.feedback_state == "reject"
    assert report.feedback_pending is True
