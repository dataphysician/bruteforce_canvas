from __future__ import annotations

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


def test_every_button_has_aria_label():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
        pre_run_modal=PreRunModalReadModel(
            prompt_document_id="doc_001",
            state=PreRunModalState.REVIEW,
            can_begin_generation=True,
            parsed_elements=[],
            parsed_relations=[],
            editable_fields=[],
            lock_entries=[],
        ),
    )

    assert 'aria-label="Start generation"' in html
    assert 'aria-label="Pause generation"' in html
    assert 'aria-label="Stop generation"' in html
    assert 'aria-label="Toggle advanced view"' in html
    assert 'aria-label="Toggle advanced pre-run view"' in html
    assert 'aria-label="Accept selected image"' in html
    assert 'aria-label="Reject selected image"' in html
    assert 'aria-label="Shred selected image"' in html
    assert 'aria-label="Begin generation"' in html
    assert 'aria-label="Cancel pre-run"' in html


def test_skip_to_main_link_targets_main_content():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert '<a href="#main-content" class="skip-link">Skip to main content</a>' in html
    assert 'id="main-content"' in html


def test_progress_heartbeat_has_aria_live_polite():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'data-region="progress-heartbeat"' in html
    assert 'aria-live="polite"' in html
    assert 'aria-label="Progress heartbeat"' in html


def test_notification_region_has_aria_live_assertive():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'data-region="notification"' in html
    assert 'aria-live="assertive"' in html
    assert 'aria-label="Run notification"' in html


def test_keyboard_handler_script_includes_required_keys():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert "<script>" in html
    assert "'ArrowDown'" in html
    assert "'ArrowUp'" in html
    assert "'Enter'" in html
    assert "'Delete'" in html
    assert "'Escape'" in html
    assert "candidate-card" in html
