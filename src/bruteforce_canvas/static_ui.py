from __future__ import annotations

import json
from html import escape

from bruteforce_canvas.ui import (
    CandidateCard,
    DetailReport,
    DiagnosticsReadModel,
    PreRunModalReadModel,
    RunWorkspaceReadModel,
)


def _button(label: str, action: str) -> str:
    return f'<button type="button" data-action="{escape(action)}">{escape(label)}</button>'


def _catalogue(cards: list[CandidateCard]) -> str:
    items = []
    for card in cards:
        seed = "" if card.seed is None else f" seed {card.seed}"
        feedback = "" if card.feedback_action is None else str(card.feedback_action)
        thumbnail = (
            '<img data-region="catalogue-thumbnail" src="'
            + escape(card.thumbnail_path)
            + '" alt="Thumbnail '
            + escape(card.candidate_id)
            + '">'
            if card.thumbnail_path
            else ""
        )
        items.append(
            '<article class="candidate-card" data-candidate-id="'
            + escape(card.candidate_id)
            + '" data-promoted="'
            + str(card.promoted).lower()
            + '" data-curated="'
            + str(card.curated).lower()
            + '" data-feedback-state="'
            + escape(feedback)
            + '">'
            + thumbnail
            + f"<strong>{escape(card.candidate_id)}</strong>{escape(seed)}"
            + "</article>"
        )
    return '<section data-region="catalogue"><h2>Catalogue</h2>' + "".join(items) + "</section>"


def _detail(selected: DetailReport | None) -> str:
    if selected is None:
        return '<section data-region="image-detail"><h2>Selected Image</h2></section>'
    tags = ", ".join(selected.optional_tags)
    image = (
        '<img data-region="selected-image" src="'
        + escape(selected.image_path)
        + '" alt="Selected image '
        + escape(selected.candidate_id)
        + '">'
        if selected.image_path
        else ""
    )
    thresholds = "".join(
        f"<li>{escape(str(key))}: {escape(str(value))}</li>"
        for key, value in selected.promotion_thresholds.items()
    )
    threshold_region = f'<ul data-region="promotion-thresholds">{thresholds}</ul>' if thresholds else ""
    coordinate_json = (
        '<div data-region="coordinate-enum-json">'
        + escape(json.dumps(selected.coordinate_enum_json, sort_keys=True))
        + "</div>"
        if selected.coordinate_enum_json
        else ""
    )
    provenance_trace = (
        '<details data-region="advanced-provenance-trace"><summary>Advanced provenance trace</summary>'
        + (
            f"<p>Bayesian score: {selected.bayesian_score_before_generation:.3f}</p>"
            if selected.bayesian_score_before_generation is not None
            else ""
        )
        + (
            '<div data-region="compatibility-trace-json">'
            + escape(json.dumps(selected.compatibility_trace, sort_keys=True))
            + "</div>"
            if selected.compatibility_trace
            else ""
        )
        + "</details>"
        if selected.compatibility_trace or selected.bayesian_score_before_generation is not None
        else ""
    )
    if selected.feedback_pending:
        pending = "" if selected.feedback_state is None else selected.feedback_state
        feedback = (
            '<div data-region="feedback-controls" data-feedback-pending="true">'
            f'<p>Feedback pending: {escape(pending)}</p>'
            '<button type="button" data-feedback="accept" disabled>Accept</button>'
            '<button type="button" data-feedback="reject" disabled>Reject</button>'
            '<button type="button" data-feedback="shred" disabled>Shred</button>'
            "</div>"
        )
    elif selected.feedback_state is not None:
        feedback = f'<div data-region="feedback-state">Feedback: {escape(selected.feedback_state)}</div>'
    else:
        feedback = (
            '<div data-region="feedback-controls">'
            '<button type="button" data-feedback="accept">Accept</button>'
            '<button type="button" data-feedback="reject">Reject</button>'
            '<button type="button" data-feedback="shred">Shred</button>'
            "</div>"
        )
    return (
        '<section data-region="image-detail">'
        "<h2>Selected Image</h2>"
        + image
        + f"<p>Candidate: {escape(selected.candidate_id)}</p>"
        + f"<p>Run: {escape(selected.run_id)}</p>"
        + f"<p>Prompt document: {escape(selected.prompt_document_id)}</p>"
        + f"<p>Target manifest: {escape(selected.target_manifest_id)}</p>"
        + f"<p>Coordinate: {escape(selected.coordinate_id)}</p>"
        + coordinate_json
        + provenance_trace
        + f"<p>Rendered prompt: {escape(selected.rendered_prompt)}</p>"
        + f"<p>Seed: {'' if selected.seed is None else selected.seed}</p>"
        + f"<p>Generator: {escape(selected.generator_model_id)} / {escape(selected.generator_backend)}</p>"
        + f"<p>Quality: {selected.quality_score:.3f}</p>"
        + f"<p>Alignment: {selected.alignment_score:.3f}</p>"
        + threshold_region
        + f"<p>Tags: {escape(tags)}</p>"
        + feedback
        + "</section>"
    )


def _progress(workspace: RunWorkspaceReadModel) -> str:
    heartbeat = workspace.progress_heartbeat
    items = "".join(
        f"<li>{escape(str(key))}: {escape(str(value))}</li>" for key, value in heartbeat.items()
    )
    return f'<footer data-region="progress-heartbeat"><ul>{items}</ul></footer>'


def _diagnostics(diagnostics: DiagnosticsReadModel | None) -> str:
    if diagnostics is None:
        return ""
    counts = "".join(
        f"<li>{escape(record_type)}: {count}</li>"
        for record_type, count in sorted(diagnostics.record_counts.items())
    )
    retries = "".join(
        "<li>"
        + escape(str(retry.get("candidate_id", "")))
        + " "
        + escape(str(retry.get("coordinate_id", "")))
        + " semantic_penalty="
        + escape(str(retry.get("semantic_penalty", "")))
        + "</li>"
        for retry in diagnostics.infrastructure_retries
    )
    return (
        '<aside data-region="developer-diagnostics">'
        "<h2>Diagnostics</h2>"
        f"<p>System actions: {diagnostics.system_action_count}</p>"
        f"<p>Infrastructure retries: {diagnostics.infrastructure_retry_count}</p>"
        f"<ul>{counts}</ul>"
        f"<ul>{retries}</ul>"
        "</aside>"
    )


def _pre_run_modal(modal: PreRunModalReadModel | None) -> str:
    if modal is None:
        return ""
    elements = "".join(f"<li>{escape(element)}</li>" for element in modal.parsed_elements)
    relations = "".join(f"<li>{escape(relation)}</li>" for relation in modal.parsed_relations)
    editable_fields = "".join(f"<li>{escape(field)}</li>" for field in modal.editable_fields)
    feedback = "".join(f"<li>{escape(item)}</li>" for item in modal.prompt_improvement_feedback)
    lock_entries = "".join(
        '<li data-lock-field="'
        + escape(str(entry.get("field_path", "")))
        + '" data-lock-state="'
        + escape(str(entry.get("lock_state", "")))
        + '" data-lhs-policy="'
        + escape(str(entry.get("lhs_policy", "")))
        + '" data-user-adjustable="'
        + str(bool(entry.get("user_adjustable", False))).lower()
        + '">'
        + escape(str(entry.get("field_path", "")))
        + ": "
        + escape(str(entry.get("lock_state", "")))
        + "</li>"
        for entry in modal.lock_entries
    )
    begin = _button("Begin Generation", "begin-generation") if modal.can_begin_generation else ""
    return (
        '<section data-region="pre-run-modal" data-modal-state="'
        + escape(str(modal.state))
        + '">'
        "<h2>Pre-Run Review</h2>"
        '<section data-region="parsed-prompt-report">'
        f"<p>Prompt document: {escape(modal.prompt_document_id)}</p>"
        f"<ul>{elements}</ul>"
        f"<ul>{relations}</ul>"
        "</section>"
        f'<ul data-region="prompt-improvement-feedback">{feedback}</ul>'
        f'<ul data-region="editable-fluid-fields">{editable_fields}</ul>'
        f'<ul data-region="lock-controls">{lock_entries}</ul>'
        + _button("Advanced", "toggle-advanced-pre-run")
        + begin
        + _button("Cancel", "cancel-pre-run")
        + "</section>"
    )


def render_workspace_html(
    workspace: RunWorkspaceReadModel,
    *,
    catalogue: list[CandidateCard],
    selected: DetailReport | None,
    diagnostics: DiagnosticsReadModel | None = None,
    pre_run_modal: PreRunModalReadModel | None = None,
) -> str:
    controls = (
        '<nav data-region="run-controls">'
        + _button("Start", "start")
        + _button("Pause", "pause")
        + _button("Stop", "stop")
        + "</nav>"
    )
    prompt = (
        '<section data-region="prompt">'
        '<label for="prompt-text">Prompt</label>'
        f'<textarea id="prompt-text">{escape(workspace.raw_user_prompt)}</textarea>'
        "</section>"
    )
    notification = (
        f'<div data-region="notification">{escape(workspace.notification)}</div>'
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Bruteforce Canvas</title></head>"
        "<body>"
        + controls
        + prompt
        + _pre_run_modal(pre_run_modal)
        + _catalogue(catalogue)
        + _detail(selected)
        + _progress(workspace)
        + notification
        + _diagnostics(diagnostics)
        + "</body></html>"
    )
