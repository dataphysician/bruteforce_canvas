import pytest

from bruteforce_canvas.static_ui import render_workspace_html
from bruteforce_canvas.ui import (
    CandidateCard,
    DetailReport,
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


def test_viewport_meta_tag_is_present():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'name="viewport"' in html
    assert "width=device-width" in html
    assert "initial-scale=1" in html


def test_media_query_css_for_max_width_768px_is_present():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert "@media (max-width: 768px)" in html
    assert "grid-template-columns: 1fr" in html


def test_zoom_pan_javascript_handles_wheel_events_and_enforces_scale_bounds():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert "viewport.addEventListener('wheel'" in html
    assert "var MIN_SCALE = 0.5" in html
    assert "var MAX_SCALE = 3.0" in html
    assert "Math.max(MIN_SCALE, Math.min(MAX_SCALE, value))" in html


def test_zoom_buttons_are_present_with_aria_labels():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'aria-label="Zoom in"' in html
    assert 'aria-label="Zoom out"' in html
    assert 'aria-label="Reset zoom"' in html
    assert 'data-action="zoom-in"' in html
    assert 'data-action="zoom-out"' in html
    assert 'data-action="zoom-reset"' in html


def test_catalogue_container_has_catalogue_viewport_class():
    html = render_workspace_html(
        workspace(),
        catalogue=[CandidateCard(candidate_id="cand_7", promoted=True, curated=True, seed=7)],
        selected=detail(),
    )

    assert 'class="catalogue-viewport"' in html
    assert 'class="candidate-catalogue"' in html
