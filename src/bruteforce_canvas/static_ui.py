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


def _button(
    label: str,
    action: str,
    extra: str = "",
    *,
    aria_label: str = "",
    disabled: bool = False,
    data_feedback: str = "",
) -> str:
    parts = [f' type="button" data-action="{escape(action)}"']
    if data_feedback:
        parts.append(f' data-feedback="{escape(data_feedback)}"')
    if extra:
        parts.append(" " + extra)
    if disabled:
        parts.append(" disabled")
    if aria_label:
        parts.append(f' aria-label="{escape(aria_label)}"')
    attrs = "".join(parts)
    return f"<button{attrs}>{escape(label)}</button>"


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
            + '"'
            + ' role="button"'
            + ' tabindex="0"'
            + ' aria-label="Candidate '
            + escape(card.candidate_id)
            + '">'
            + thumbnail
            + f"<strong>{escape(card.candidate_id)}</strong>{escape(seed)}"
            + "</article>"
        )
    catalogue = (
        '<section class="candidate-catalogue" data-region="catalogue">'
        + "<h2>Catalogue</h2>"
        + "".join(items)
        + "</section>"
    )
    zoom_controls = (
        '<div class="zoom-controls">'
        + _button("+", "zoom-in", aria_label="Zoom in")
        + _button("−", "zoom-out", aria_label="Zoom out")
        + _button("Reset", "zoom-reset", aria_label="Reset zoom")
        + "</div>"
    )
    return (
        '<div class="catalogue-viewport">'
        + zoom_controls
        + catalogue
        + "</div>"
    )


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
            + _button("Accept", "feedback-accept", data_feedback="accept", disabled=True, aria_label="Accept selected image")
            + _button("Reject", "feedback-reject", data_feedback="reject", disabled=True, aria_label="Reject selected image")
            + _button("Shred", "feedback-shred", data_feedback="shred", disabled=True, aria_label="Shred selected image")
            + "</div>"
        )
    elif selected.feedback_state is not None:
        feedback = f'<div data-region="feedback-state">Feedback: {escape(selected.feedback_state)}</div>'
    else:
        feedback = (
            '<div data-region="feedback-controls">'
            + _button("Accept", "feedback-accept", data_feedback="accept", aria_label="Accept selected image")
            + _button("Reject", "feedback-reject", data_feedback="reject", aria_label="Reject selected image")
            + _button("Shred", "feedback-shred", data_feedback="shred", aria_label="Shred selected image")
            + "</div>"
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
    return (
        '<footer data-region="progress-heartbeat"'
        + ' aria-live="polite"'
        + ' aria-label="Progress heartbeat">'
        + f"<ul>{items}</ul>"
        + "</footer>"
    )


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


_ERROR_STATE_MESSAGES: dict[str, str] = {
    "no_prompt": "No prompt was provided.",
    "parse_blocked": "Prompt parsing is blocked.",
    "no_curated_images": "No curated images are available.",
    "all_seeds_failed": "All seeds failed to generate.",
    "generator_unavailable": "The image generator is unavailable.",
    "evaluator_unavailable": "The evaluator is unavailable.",
    "stalled": "Generation has stalled.",
}


def _error_state(workspace: RunWorkspaceReadModel) -> str:
    if workspace.error_state is None:
        return ""
    message = _ERROR_STATE_MESSAGES.get(
        workspace.error_state,
        f"Unknown error state: {workspace.error_state}",
    )
    return (
        '<div data-region="error-state"'
        + f' data-error-state="{escape(workspace.error_state)}">'
        + f"<p>{escape(message)}</p>"
        + "</div>"
    )


def _advanced_view(workspace: RunWorkspaceReadModel) -> str:
    if (
        not workspace.diagnostic_hold_enums
        and not workspace.suppressed_enums
        and not workspace.proposed_enums
        and not workspace.raw_ood_signals
    ):
        return '<div id="advanced-view" hidden><section data-region="diagnostic-hold-enums"><h3>Diagnostic hold enums</h3><ul></ul></section><section data-region="suppressed-enums"><h3>Suppressed enums</h3><ul></ul></section><section data-region="proposed-enums"><h3>Proposed enums</h3><ul></ul></section><section data-region="raw-ood-signals"><h3>Raw OOD signals</h3><ul></ul></section></div>'
    diagnostic_hold = "".join(
        f"<li>{escape(item)}</li>" for item in workspace.diagnostic_hold_enums
    )
    suppressed = "".join(f"<li>{escape(item)}</li>" for item in workspace.suppressed_enums)
    proposed = "".join(f"<li>{escape(item)}</li>" for item in workspace.proposed_enums)
    ood = "".join(f"<li>{escape(item)}</li>" for item in workspace.raw_ood_signals)
    return (
        '<div id="advanced-view" hidden>'
        + f'<section data-region="diagnostic-hold-enums"><h3>Diagnostic hold enums</h3><ul>{diagnostic_hold}</ul></section>'
        + f'<section data-region="suppressed-enums"><h3>Suppressed enums</h3><ul>{suppressed}</ul></section>'
        + f'<section data-region="proposed-enums"><h3>Proposed enums</h3><ul>{proposed}</ul></section>'
        + f'<section data-region="raw-ood-signals"><h3>Raw OOD signals</h3><ul>{ood}</ul></section>'
        + "</div>"
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
    begin = _button("Begin Generation", "begin-generation", aria_label="Begin generation") if modal.can_begin_generation else ""
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
        + _button("Advanced", "toggle-advanced-pre-run", aria_label="Toggle advanced pre-run view")
        + begin
        + _button("Cancel", "cancel-pre-run", aria_label="Cancel pre-run")
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
    advanced_toggle = (
        _button(
            "Advanced",
            "toggle-advanced",
            'id="advanced-toggle" aria-pressed="false"',
            aria_label="Toggle advanced view",
        )
        + '\n        '
    )
    controls = (
        '<nav data-region="run-controls">'
        + advanced_toggle
        + _button("Start", "start", aria_label="Start generation")
        + _button("Pause", "pause", aria_label="Pause generation")
        + _button("Stop", "stop", aria_label="Stop generation")
        + "</nav>"
    )
    prompt = (
        '<section data-region="prompt">'
        '<label for="prompt-text">Prompt</label>'
        f'<textarea id="prompt-text" aria-label="Prompt">{escape(workspace.raw_user_prompt)}</textarea>'
        "</section>"
    )
    notification = (
        '<div data-region="notification"'
        + ' aria-live="assertive"'
        + ' aria-label="Run notification">'
        + f"{escape(workspace.notification)}"
        + "</div>"
    )
    script = """
    <script>
    (function() {
        var toggleButton = document.getElementById('advanced-toggle');
        if (toggleButton) {
            toggleButton.addEventListener('click', function(event) {
                var advancedView = document.getElementById('advanced-view');
                if (!advancedView) return;
                var pressed = toggleButton.getAttribute('aria-pressed') === 'true';
                toggleButton.setAttribute('aria-pressed', String(!pressed));
                advancedView.hidden = pressed;
            });
            toggleButton.addEventListener('keydown', function(event) {
                if (event.key === ' ' || event.key === 'Enter') {
                    event.preventDefault();
                    toggleButton.click();
                }
            });
        }
        var catalogue = document.querySelector('[data-region="catalogue"]');
        if (catalogue) {
            var cards = Array.from(catalogue.querySelectorAll('.candidate-card'));
            catalogue.addEventListener('keydown', function(event) {
                var card = event.target.closest('.candidate-card');
                if (!card) return;
                var index = cards.indexOf(card);
                if (index < 0) return;
                if (event.key === 'ArrowDown') {
                    event.preventDefault();
                    var next = cards[index + 1];
                    if (next) next.focus();
                } else if (event.key === 'ArrowUp') {
                    event.preventDefault();
                    var prev = cards[index - 1];
                    if (prev) prev.focus();
                } else if (event.key === 'Enter') {
                    event.preventDefault();
                    card.dispatchEvent(new MouseEvent('click', {bubbles: true}));
                } else if (event.key === 'Delete') {
                    event.preventDefault();
                    card.dispatchEvent(new MouseEvent('click', {bubbles: true}));
                }
            });
        }
        var modal = document.querySelector('[data-region="pre-run-modal"]');
        if (modal) {
            modal.addEventListener('keydown', function(event) {
                if (event.key === 'Escape') {
                    var cancelButton = modal.querySelector('button[data-action="cancel-pre-run"]');
                    if (cancelButton) {
                        event.preventDefault();
                        cancelButton.click();
                    }
                }
            });
        }
        var viewport = document.querySelector('.catalogue-viewport');
        if (viewport) {
            var scale = 1;
            var panning = false;
            var startX = 0;
            var startY = 0;
            var scrollLeft = 0;
            var scrollTop = 0;
            var MIN_SCALE = 0.5;
            var MAX_SCALE = 3.0;
            var applyTransform = function() {
                viewport.style.transform = 'scale(' + scale + ')';
            };
            var clamp = function(value) {
                return Math.max(MIN_SCALE, Math.min(MAX_SCALE, value));
            };
            viewport.addEventListener('wheel', function(event) {
                if (!event.ctrlKey && !event.metaKey) {
                    return;
                }
                event.preventDefault();
                var delta = event.deltaY > 0 ? -0.1 : 0.1;
                scale = clamp(scale + delta);
                applyTransform();
            });
            viewport.addEventListener('mousedown', function(event) {
                if (scale <= 1.0) {
                    return;
                }
                event.preventDefault();
                panning = true;
                startX = event.pageX - viewport.offsetLeft;
                startY = event.pageY - viewport.offsetTop;
                scrollLeft = viewport.scrollLeft;
                scrollTop = viewport.scrollTop;
            });
            window.addEventListener('mousemove', function(event) {
                if (!panning) {
                    return;
                }
                event.preventDefault();
                var x = event.pageX - viewport.offsetLeft;
                var y = event.pageY - viewport.offsetTop;
                viewport.scrollLeft = scrollLeft - (x - startX);
                viewport.scrollTop = scrollTop - (y - startY);
            });
            window.addEventListener('mouseup', function() {
                panning = false;
            });
            viewport.addEventListener('dblclick', function(event) {
                event.preventDefault();
                scale = 1;
                applyTransform();
            });
            var zoomInButton = document.querySelector('button[data-action="zoom-in"]');
            if (zoomInButton) {
                zoomInButton.addEventListener('click', function() {
                    scale = clamp(scale + 0.2);
                    applyTransform();
                });
            }
            var zoomOutButton = document.querySelector('button[data-action="zoom-out"]');
            if (zoomOutButton) {
                zoomOutButton.addEventListener('click', function() {
                    scale = clamp(scale - 0.2);
                    applyTransform();
                });
            }
            var zoomResetButton = document.querySelector('button[data-action="zoom-reset"]');
            if (zoomResetButton) {
                zoomResetButton.addEventListener('click', function() {
                    scale = 1;
                    applyTransform();
                });
            }
        }
    })();
    </script>
    """
    style = """
    <style>
        .catalogue-viewport {
            overflow: auto;
            transform-origin: 0 0;
        }
        .candidate-catalogue {
            transform-origin: 0 0;
            transition: transform 0.1s ease;
        }
        .zoom-controls {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        @media (max-width: 768px) {
            [data-region="run-controls"] {
                flex-direction: column;
                align-items: stretch;
            }
            [data-region="run-controls"] button {
                width: 100%;
            }
            .candidate-catalogue {
                display: grid;
                grid-template-columns: 1fr;
            }
            body {
                font-size: 14px;
            }
            [data-region="pre-run-modal"] {
                margin: 0.5rem;
                width: calc(100% - 1rem);
            }
        }
        @media (max-width: 479px) {
            main > section,
            main > .catalogue-viewport {
                padding: 0.25rem;
            }
            [data-region="progress-heartbeat"] > ul > li:not(:first-child) {
                display: none;
            }
            [data-region="notification"] {
                font-size: 0.8rem;
            }
            .zoom-controls {
                gap: 0.25rem;
            }
            [data-region="run-controls"] button,
            [data-region="pre-run-modal"] button {
                width: 100%;
            }
        }
    </style>
    """
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Bruteforce Canvas</title>"
        + style
        + "</head>"
        "<body>"
        + '<a href="#main-content" class="skip-link">Skip to main content</a>'
        + "<main id=\"main-content\">"
        + controls
        + prompt
        + _pre_run_modal(pre_run_modal)
        + _catalogue(catalogue)
        + _detail(selected)
        + _progress(workspace)
        + _error_state(workspace)
        + notification
        + _diagnostics(diagnostics)
        + _advanced_view(workspace)
        + "</main>"
        + script
        + "</body></html>"
    )
