import pytest

from bruteforce_canvas.static_ui import render_workspace_html
from bruteforce_canvas.ui import (
    CandidateCard,
    DetailReport,
    DiagnosticsReadModel,
    PreRunModalReadModel,
    PreRunModalState,
    RunWorkspaceReadModel,
)
from bruteforce_canvas.shared import FeedbackAction


def workspace() -> RunWorkspaceReadModel:
    return RunWorkspaceReadModel(
        run_id="run_001",
        raw_user_prompt="a bowl on a table",
        run_state="running",
        generated_count=5,
        iqa_evaluated_count=5,
        vlm_evaluated_count=3,
        promoted_curated_count=2,
        accepted_count=1,
        rejected_count=0,
        shredded_count=0,
        stall_guard_state="healthy",
        notification="Generating seed 42 for coordinate coord_001.",
    )


def detail() -> DetailReport:
    return DetailReport(
        candidate_id="cand_7",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        rendered_prompt="Generate a bowl on a table",
        seed=7,
        generator_model_id="stub-generator",
        generator_backend="stub",
        generation_settings={"steps": 4},
        coordinate_enum_json={
            "locked_arms": {"object.material.object_01": "CERAMIC"},
            "sampled_arms": {"cinematography.shot_size": "MEDIUM_SHOT"},
            "combo_signature": "shot=MEDIUM_SHOT|material=CERAMIC",
        },
        compatibility_trace={
            "prior_score": 0.88,
            "downranks": [{"reason": "cool lighting weakens ceramic warmth"}],
        },
        bayesian_score_before_generation=0.73,
        quality_score=0.8,
        alignment_score=0.7,
        promotion_thresholds={"quality_cutoff": 0.55, "alignment_cutoff": 0.25},
        promotion_gate_reasons=["quality and alignment passed"],
        optional_tags=["warm"],
        image_path="/tmp/cand_7.png",
    )


def test_static_workspace_html_contains_required_regions_and_controls():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'data-region="run-controls"' in html
    assert 'data-action="start"' in html
    assert 'data-action="pause"' in html
    assert 'data-action="stop"' in html
    assert 'data-region="prompt"' in html
    assert "<textarea" in html
    assert 'data-region="catalogue"' in html
    assert 'data-region="image-detail"' in html
    assert 'data-region="progress-heartbeat"' in html
    assert 'data-region="notification"' in html


def test_static_workspace_html_renders_catalogue_thumbnail_and_selected_image():
    html = render_workspace_html(
        workspace(),
        catalogue=[
            CandidateCard(
                candidate_id="cand_7",
                promoted=True,
                curated=True,
                seed=7,
                thumbnail_path="/tmp/cand_7.png",
            )
        ],
        selected=detail(),
    )

    assert 'data-region="catalogue-thumbnail"' in html
    assert 'src="/tmp/cand_7.png"' in html
    assert 'data-region="selected-image"' in html
    assert 'alt="Selected image cand_7"' in html
    assert 'data-region="promotion-thresholds"' in html
    assert "quality_cutoff: 0.55" in html
    assert 'data-region="coordinate-enum-json"' in html
    assert "object.material.object_01" in html
    assert 'data-region="advanced-provenance-trace"' in html
    assert "cool lighting weakens ceramic warmth" in html
    assert "Bayesian score: 0.730" in html


def test_static_workspace_html_shows_feedback_controls_but_no_backend_overrides():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'data-feedback="accept"' in html
    assert 'data-feedback="reject"' in html
    assert 'data-feedback="shred"' in html
    assert "override evaluator" not in html.lower()
    assert "compatibility policy" not in html.lower()
    assert "graph editor" not in html.lower()


def test_static_workspace_html_replaces_feedback_controls_after_feedback_is_accepted():
    selected = detail().model_copy(update={"feedback_state": "accept"})

    html = render_workspace_html(
        workspace(),
        catalogue=[
            CandidateCard(
                candidate_id="cand_7",
                promoted=True,
                curated=True,
                seed=7,
                feedback_action=FeedbackAction.ACCEPT,
            )
        ],
        selected=selected,
    )

    assert 'data-region="feedback-state"' in html
    assert "Feedback: accept" in html
    assert 'data-feedback="accept"' not in html
    assert 'data-feedback="reject"' not in html
    assert 'data-feedback="shred"' not in html


def test_static_workspace_html_disables_feedback_controls_while_request_is_in_flight():
    selected = detail().model_copy(update={"feedback_state": "reject", "feedback_pending": True})

    html = render_workspace_html(
        workspace(),
        catalogue=[
            CandidateCard(
                candidate_id="cand_7",
                promoted=True,
                curated=True,
                seed=7,
                feedback_action=FeedbackAction.REJECT,
                feedback_pending=True,
            )
        ],
        selected=selected,
    )

    assert 'data-region="feedback-controls"' in html
    assert 'data-feedback-pending="true"' in html
    assert 'data-feedback="accept" disabled' in html
    assert 'data-feedback="reject" disabled' in html
    assert 'data-feedback="shred" disabled' in html
    assert "Feedback pending: reject" in html


def test_static_workspace_html_can_render_pre_run_modal_with_lock_controls():
    modal = PreRunModalReadModel(
        prompt_document_id="doc_001",
        state=PreRunModalState.REVIEW,
        can_begin_generation=True,
        parsed_elements=["object_01: bowl"],
        parsed_relations=[],
        editable_fields=["lighting_raw"],
        lock_entries=[
            {
                "field_path": "object.material.object_01",
                "lock_state": "locked",
                "lhs_policy": "fixed",
                "user_adjustable": True,
            },
            {
                "field_path": "cinematography.shot_size",
                "lock_state": "unlocked",
                "lhs_policy": "sampleable_if_missing",
                "user_adjustable": True,
            },
            {
                "field_path": "constraint.no_extra_people",
                "lock_state": "locked",
                "lhs_policy": "fixed",
                "user_adjustable": False,
            },
        ],
    )

    html = render_workspace_html(
        workspace(),
        catalogue=[],
        selected=None,
        pre_run_modal=modal,
    )

    assert 'data-region="pre-run-modal"' in html
    assert 'data-region="parsed-prompt-report"' in html
    assert 'data-lock-field="object.material.object_01"' in html
    assert 'data-lock-field="constraint.no_extra_people"' in html
    assert 'data-user-adjustable="false"' in html
    assert 'data-action="toggle-advanced-pre-run"' in html
    assert 'data-action="begin-generation"' in html
    assert 'data-action="cancel-pre-run"' in html
    assert "graph editor" not in html.lower()
    assert "add element" not in html.lower()


def test_static_workspace_html_blocked_pre_run_modal_shows_feedback_without_begin_action():
    modal = PreRunModalReadModel(
        prompt_document_id="doc_001",
        state=PreRunModalState.BLOCKED,
        can_begin_generation=False,
        parsed_elements=["person_01: person"],
        parsed_relations=[],
        prompt_improvement_feedback=["Specify what the person is throwing."],
    )

    html = render_workspace_html(
        workspace().model_copy(update={"raw_user_prompt": "person throwing something"}),
        catalogue=[],
        selected=None,
        pre_run_modal=modal,
    )

    assert 'data-modal-state="blocked"' in html
    assert "person throwing something" in html
    assert "Specify what the person is throwing." in html
    assert 'data-action="begin-generation"' not in html
    assert 'data-action="cancel-pre-run"' in html


def test_static_workspace_html_can_render_diagnostics_in_separate_region():
    diagnostics = DiagnosticsReadModel(
        record_counts={"image_evaluation": 5, "system_action": 5},
        system_action_count=5,
        infrastructure_retry_count=1,
        infrastructure_retries=[
            {
                "candidate_id": "cand_42069",
                "coordinate_id": "coord_001",
                "semantic_penalty": False,
                "reasons": ["stubbed infrastructure block"],
            }
        ],
    )

    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
        diagnostics=diagnostics,
    )

    assert 'data-region="developer-diagnostics"' in html
    assert "Infrastructure retries: 1" in html
    assert "cand_42069" in html
    assert "system_action: 5" in html
    assert "<pre" not in html


def test_static_workspace_html_escapes_user_supplied_text():
    unsafe = workspace().model_copy(update={"raw_user_prompt": "<script>alert(1)</script>"})
    html = render_workspace_html(unsafe, catalogue=[], selected=None)

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html


def test_static_workspace_html_has_advanced_toggle_button():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'id="advanced-toggle"' in html
    assert 'data-action="toggle-advanced"' in html
    assert 'aria-pressed="false"' in html
    assert 'aria-label="Toggle advanced view"' in html


def test_static_workspace_html_advanced_view_hidden_and_subregions_exist_when_populated():
    populated = workspace().model_copy(
        update={
            "diagnostic_hold_enums": ["arm_x"],
            "suppressed_enums": ["arm_y"],
            "proposed_enums": ["arm_z"],
            "raw_ood_signals": ["signal_1"],
        }
    )
    html = render_workspace_html(
        populated,
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'id="advanced-view"' in html
    assert 'hidden' in html
    assert 'data-region="diagnostic-hold-enums"' in html
    assert 'data-region="suppressed-enums"' in html
    assert 'data-region="proposed-enums"' in html
    assert 'data-region="raw-ood-signals"' in html


def test_static_workspace_html_advanced_view_hidden_when_empty():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'id="advanced-view"' in html
    assert 'hidden' in html


@pytest.mark.parametrize(
    "error_state",
    [
        "no_prompt",
        "parse_blocked",
        "no_curated_images",
        "all_seeds_failed",
        "generator_unavailable",
        "evaluator_unavailable",
        "stalled",
    ],
)
def test_static_workspace_html_renders_error_state(error_state: str):
    model = workspace().model_copy(update={"error_state": error_state})
    html = render_workspace_html(
        model,
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'data-region="error-state"' in html
    assert f'data-error-state="{error_state}"' in html


def test_static_workspace_html_contains_advanced_toggle_vanilla_js():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert "<script>" in html
    assert "advanced-toggle" in html
    assert "advanced-view" in html
    assert "aria-pressed" in html
    assert "hidden" in html
    assert "toggle-advanced" in html
