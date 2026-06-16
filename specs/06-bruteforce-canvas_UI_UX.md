**Bruteforce Canvas UI/UX Specification**
**Companion to `bruteforce-canvas_DAG_prompt.md`, `bruteforce-canvas_LHS_enum_router.md`, `bruteforce-canvas_fast_image_generation.md`, `bruteforce-canvas_Evaluator_pipeline.md`, and `bruteforce-canvas_Orchestration.md`**

Status: implementation specification

Audience: UI/UX, frontend, local-app, and product implementers

Primary goal: define a framework-agnostic user interface contract for starting a run, confirming prompt parsing, monitoring generation, curating pre-evaluated images, and inspecting individual image reports without exposing low-level backend machinery as ordinary user controls.

This UI spec consumes backend contracts from the orchestration, prompt, router, generation, and evaluator specs. It does not redefine prompt parsing, enum routing, generator behavior, evaluator scoring, promotion policy, suppression policy, quarantine policy, or learning updates.

**1. Product Position**
The UI is a hybrid run workspace.

It combines:
- prompt entry.
- run controls.
- pre-generation prompt refinement.
- pre-curated image catalogue.
- single-image carousel/detail view.
- lightweight run progress.
- unobtrusive status and diagnostic messaging.

It is not:
- a graph editor.
- an enum registry editor.
- a manual evaluator override surface.
- a backend policy override surface.
- a full developer console by default.

The ordinary user should feel they are guiding a high-throughput image exploration run and selecting the best outputs, not operating the internal parser, sampler, generator, evaluator, or learner directly.

**2. Framework Boundary**
This specification is framework-agnostic.

The same interaction contract should be implementable in:
- a Gradio-style proof of concept.
- a local desktop or notebook app.
- a production web interface.
- a production client backed by streamed orchestration events.

The spec defines screens, read models, allowed actions, state transitions, and visibility rules. It does not choose a frontend framework, component library, CSS system, routing model, backend transport, websocket protocol, or persistence layer.

**3. Primary Workspace**
The primary screen contains one holistic run workspace.

Required regions:
- top run controls.
- prompt text area.
- pre-curated image catalogue.
- selected image carousel/detail view.
- persistent progress heartbeat.
- one-line status notification.

Top run controls:
- `Start`
- `Pause`
- `Stop`

Prompt area:
- appears directly below the run controls.
- uses a text area or equivalent multiline prompt widget.
- contains the prompt that initiated or will initiate the run.
- remains visible during the run so the catalogue has clear prompt provenance.

The main catalogue:
- displays images that survived automated post-generation evaluation and entered the pre-curated pool.
- excludes images that were rejected or shredded by user feedback.
- may also show accepted images if the active filter includes accepted images.
- does not show failed, infrastructure-blocked, or non-promoted images in the default catalogue.

Selected image detail:
- opens when the user selects an image from the catalogue.
- presents one image at a time in carousel-style navigation.
- shows the selected image side-by-side with its report.

**4. Run Controls**
`Start`:
- requires a non-empty prompt.
- launches the pre-run refinement modal before mass generation begins.
- does not directly start mass generation until the user selects `Begin Generation` inside the modal.

`Pause`:
- pauses generation scheduling when supported by the orchestration run state.
- does not discard already generated candidates.
- does not cancel in-flight persistence or evaluation that must complete for consistency.

`Stop`:
- requests run shutdown.
- preserves persisted candidates, evaluator payloads, feedback, and learning evidence.
- does not erase run history.

The UI must not silently mutate thresholds, lock policy, prompt text, or feedback meaning when these controls are used.

**5. Pre-Run Refinement Modal**
The pre-run modal is the final prompt refinement checkpoint before mass generation.

Open condition:
- user selects `Start` while a prompt is present.
- backend parsing/canonicalization/verification begins or has completed.

Close conditions:
- user selects `Begin Generation`.
- user selects `Cancel`.

Default path:
- review then begin.
- most users should be able to inspect parser output briefly and proceed.
- advanced editing is available but secondary.

The modal contains:
- parsed prompt report.
- editable fluid prompt-enhancement fields.
- lock/unlock switches beside eligible enum fields.
- advanced mode switch.
- `Begin Generation` action.
- `Cancel` action.

The modal must communicate parser/canonicalizer quality by showing prefilled fields. Prefilled fields should make it easy to see what the system understood without requiring the user to inspect raw JSON.

**6. Pre-Run Editing Boundary**
The modal is not a graph editor.

Read-only major structure:
- elements.
- relations.
- action actors.
- action targets.
- required graph facts.
- target manifest identity.

Editable fields:
- fluid presentation fields.
- prompt-enhancement descriptors.
- user-adjustable raw descriptors that do not change element identity, relation endpoints, action targets, or required graph facts.
- sampleable or weakly specified fields exposed by the backend as editable.

Enum controls:
- eligible enum fields may show lock/unlock switches.
- locked fields remain fixed for the run.
- unlocked sampleable fields may be varied by the LHS router.
- suppressed or diagnostic-hold states may be visible as status but must not become ordinary policy override controls.

Disallowed edits:
- add element.
- remove element.
- rename an element into a different object identity.
- add relation.
- remove relation.
- change relation endpoints.
- add action target.
- change action target.
- create enum registry values.
- override compatibility policy.
- override evaluator policy.

If the parser missed major scene structure, the user should cancel the modal, revise the original prompt, and start parsing again.

Validation warnings for graph-breaking edits are not required in the normal UI because graph-breaking edit controls are not exposed.

**7. Pre-Run Modal Views**
The modal supports two main views.

Report view:
- default view.
- shows what was parsed and canonicalized.
- groups major elements, relations, actions, cinematography, constraints, and editable prompt-enhancement fields.
- marks read-only major structure clearly.
- shows enough confidence/status information to help the user decide whether to trust the parse.

Advanced edit view:
- intended for power users.
- exposes editable fluid fields and lock/unlock switches.
- may show diagnostic-hold fields, suppressed matched fields, raw-only fields, and proposed enum metadata as status.
- must preserve the no-graph-editor boundary.
- the view is rendered by `static_ui._advanced_view`, with the advanced-mode toggle implemented as a real `<button id="advanced-toggle" type="button" aria-pressed="false">` element so screen readers report the pressed state.
- the toggle responds to a `keydown` handler on Space and Enter to flip the pressed state; this is the canonical accessibility binding for the advanced switch.

The view switch should not change backend state by itself. Backend state changes only when the user applies edits or begins generation.

**8. Prompt-Improvement States**
If the prompt parser/verifier returns blocking issues, the modal should present prompt-improvement feedback instead of allowing mass generation.

Blocking examples:
- no resolved visible element.
- required relation target missing.
- unresolved action object that cannot be rendered safely.
- negative constraint conflicts with required graph content.
- ordinary lighting phrase promoted into graph structure without a visible light-emitting object.

Blocked prompt UI:
- show the issue in plain language.
- preserve the original prompt.
- offer precise rewrite guidance when provided by the prompt pipeline.
- allow `Cancel`.
- do not show `Begin Generation` while blocking issues remain.

Non-blocking ambiguity:
- may show in report view as advisory status.
- may allow `Begin Generation` if the backend provides a safe non-inventive downgrade.

**9. Catalogue**
The catalogue shows evaluation-survived images.

Default inclusion:
- `promoted=true`
- `curated=true`
- not rejected by user feedback.
- not shredded by user feedback.

Default exclusion:
- non-promoted candidates.
- evaluator failures.
- infrastructure-blocked candidates.
- retired or quarantined coordinates with no promoted candidate.
- rejected images.
- shredded images.

Catalogue items should show:
- thumbnail.
- lightweight status badge when useful.
- seed or compact provenance when space allows.
- accepted state if the image has been accepted.

Catalogue sorting should optimize for curation flow:
- accepted images may be grouped or pinned when the active filter includes them.
- pre-curated unreviewed images should remain easy to browse.
- optional tags do not gate catalogue inclusion.

**10. Carousel and Image Detail**
Selecting a catalogue image opens a single-image carousel/detail view.

The view contains:
- selected image.
- rendered prompt.
- seed number.
- optional image tags or descriptors.
- evaluation summary.
- feedback controls.

Image report fields:
- `candidate_id`
- `run_id`
- `prompt_document_id`
- `target_manifest_id`
- `coordinate_id`
- rendered prompt.
- seed.
- generator model ID.
- generator backend.
- generation settings.
- quality score.
- alignment score.
- promotion gate reasons.
- optional tags/descriptors.
- feedback state.

The detail panel should present human-readable labels first. Raw JSON and full traces belong in diagnostics or advanced detail expansion.

**11. Feedback Controls**
Feedback controls appear below the selected image.

Controls:
- thumbs up: `accept`
- thumbs down: `reject`
- trash icon: `shred`

`accept` semantics:
- final accepted image.
- enters the positive fine-tuning dataset.
- keeps promoted/curated state.
- applies positive learning evidence to candidate, coordinate, sampled enum arms, and enum combination.

`reject` semantics:
- image did not meet expected quality or alignment.
- leaves the visible curated catalogue and carousel flow.
- may enter the negative fine-tuning dataset.
- applies normal negative prior evidence.
- persists the image and evaluator payload for learning and calibration.

`shred` semantics:
- image has a visible severe defect.
- leaves the visible curated catalogue and carousel flow.
- applies stronger negative prior evidence.
- is not used in the IQA negative fine-tuning dataset.
- may be persisted for archival or cold-storage purposes according to backend storage policy.

The UI must not describe shred as ordinary delete if the backend persists the image for archival/cold-storage evidence.

**12. Feedback Aftermath**
After feedback is submitted:

Accepted image:
- remains available as accepted output.
- may stay visible in the catalogue under accepted or all filters.
- feedback control state should show that it has been accepted.

Rejected image:
- is removed from the default curated catalogue and carousel flow.
- remains persisted for backend learning.
- may be visible only through audit or diagnostics if such views exist.

Shredded image:
- is removed from the default curated catalogue and carousel flow.
- remains excluded from IQA negative fine-tuning.
- may be visible only through archival/audit diagnostics if such views exist.

Feedback idempotency:
- repeated clicks on the same feedback action must not duplicate backend learning updates.
- after feedback is accepted by the backend, the UI should disable or replace the action controls for that candidate.
- if a feedback request is in flight, the UI should prevent contradictory second actions until the result is known.

**13. Progress and Status**
The UI has two lightweight status surfaces.

Persistent bottom progress:
- always visible while a run exists.
- acts as the run heartbeat.
- shows run health and progress.

Recommended progress fields:
- run state.
- generated count.
- IQA evaluated count.
- VLM evaluated count.
- promoted/curated count.
- accepted count.
- rejected count.
- shredded count.
- stall guard state.
- elapsed time.
- `error_state` (string from the documented state set in §18).
- `diagnostic_hold_enums`, `suppressed_enums`, `proposed_enums`, `raw_ood_signals` (additive lists from the read model).
- `vram_telemetry` (list of `VRAMTelemetry` entries from `src/bruteforce_canvas.telemetry`, surfaced through the `progress_heartbeat` computed field).

One-line notification:
- unobtrusive.
- shows current pipeline stage, recent event, error, or action-needed alert.
- may rotate through short status messages.
- should escalate visually only when user action is needed or a run is blocked.

Example notification messages:
- `Parsing prompt...`
- `Waiting for pre-run confirmation.`
- `Generating seed 42 for coordinate coord_001842.`
- `Evaluating alignment for current seed bundle.`
- `Image promoted to curated catalogue.`
- `Prompt parse blocked: relation target missing.`
- `Generator unavailable.`
- `Run stopped by stall guard.`

**14. Optional Image Tags and Descriptors**
Optional taggers, quality descriptors, TRIBE-style metadata, or impact descriptors are informational in the UI contract.

They may appear in the selected image report below prompt and seed.

They must not:
- gate catalogue inclusion.
- override automated promotion.
- alter accept/reject/shred semantics.
- decide positive or negative dataset membership.
- trigger suppression, quarantine, demotion, or promotion by themselves.

Optional tags may be useful for user understanding, search, filtering, or reporting if those features are implemented. They remain secondary to evaluator gates and user feedback actions.

**15. Diagnostics Boundary**
Ordinary UI:
- prompt.
- run controls.
- pre-run report and editable fluid fields.
- catalogue.
- image detail.
- feedback.
- progress heartbeat.
- one-line status notification.

Developer diagnostics:
- raw PromptDocument.
- full enum coordinate JSON.
- compatibility trace.
- evaluator payload.
- target manifest JSON.
- Thompson/GP update details.
- quarantine/suppression counters.
- generator telemetry.
- VRAM telemetry.
- infrastructure retry details.

Diagnostics should be accessible without becoming the ordinary user path. Low-level internals should not clutter the default workflow.

**16. Read Models**
The UI should consume read models from the backend rather than constructing state by reparsing logs.

Run summary:
```json
{
  "run_id": "run_00091",
  "raw_user_prompt": "a woman in a black coat walking toward a glass door",
  "state": "generating",
  "generated_count": 120,
  "promoted_curated_count": 18,
  "accepted_count": 3,
  "rejected_count": 4,
  "shredded_count": 1,
  "stall_guard_state": "healthy",
  "heartbeat_at": "2026-06-13T12:00:00Z"
}
```

Pre-run parse summary:
```json
{
  "run_id": "run_00091",
  "prompt_document_id": "doc_017",
  "target_manifest_id": "eval_manifest_001842",
  "verifier_approved": true,
  "blocking_issues": [],
  "read_only_graph_summary": [],
  "editable_fields": [],
  "lockable_enum_fields": []
}
```

Catalogue item:
```json
{
  "candidate_id": 123,
  "run_id": "run_00091",
  "coordinate_id": "coord_001842",
  "seed": 42,
  "thumbnail_path": "runtime/candidates/123.thumb.png",
  "curation_state": "curated",
  "feedback_state": null,
  "quality_score": 0.82,
  "alignment_score": 0.76
}
```

Selected image report:
```json
{
  "candidate_id": 123,
  "run_id": "run_00091",
  "prompt_document_id": "doc_017",
  "target_manifest_id": "eval_manifest_001842",
  "coordinate_id": "coord_001842",
  "seed": 42,
  "image_path": "runtime/candidates/123.png",
  "rendered_prompt": "Generate woman wearing black coat walking toward glass door...",
  "generator_model_id": "<active generator builder id>",
  "quality_score": 0.82,
  "alignment_score": 0.76,
  "promotion_reasons": [],
  "optional_descriptors": [],
  "feedback_state": null
}
```

Feedback event:
```json
{
  "candidate_id": 123,
  "run_id": "run_00091",
  "coordinate_id": "coord_001842",
  "feedback_action": "reject",
  "signal_source": "swipe_feedback",
  "effective_status": "demoted_false_positive"
}
```

**17. Event Stream**
The default live transport is a CLI-hosted server-sent event stream backed by the event bus. A production UI may wrap the same event concepts in polling, websockets, a local callback loop, or framework-specific refresh, but the event names and payload concepts remain stable.

Required event concepts:
- run started.
- pre-run parse ready.
- pre-run parse blocked.
- generation queued.
- generation started.
- image generated.
- IQA evaluation completed.
- VLM evaluation completed.
- image promoted/curated.
- feedback accepted.
- image removed from visible catalogue.
- run paused.
- run resumed.
- run stopped.
- run stalled.
- infrastructure warning.
- infrastructure error.

Events should carry:
- event ID.
- timestamp.
- run ID.
- coordinate ID when applicable.
- candidate ID when applicable.
- lifecycle state.
- short human-readable message.
- optional payload reference.

**18. Empty and Error States**

The static UI renders the following states. Each maps to a string in `RunWorkspaceReadModel.error_state` and is surfaced by `static_ui._error_state` as an inline error block above the catalogue.

Documented states:

- `no_prompt`: prompt area is empty. `Start` is disabled. Prompt area remains ready for input.
- `parse_blocked`: pre-run modal shows the blocking issue and any rewrite guidance from the prompt pipeline. `Begin Generation` is disabled.
- `no_curated_yet`: pre-curated catalogue is empty for a healthy run. Progress heartbeat and one-line current-stage notification are shown; the run is not in failure.
- `all_seeds_failed`: the active coordinate's seed bundle produced no passing images. The catalogue stays empty for that coordinate and the run continues unless the stall guard trips.
- `generator_unavailable`: the active generator is missing or refused. Actionable status is shown; prompt and run configuration are preserved; user input is not cleared.
- `evaluator_unavailable`: IQA or VLM evaluation could not run. Actionable status is shown; generated artifacts are preserved; no unevaluated image is marked curated.
- `run_stalled`: stall guard tripped. Stall message is shown; prompt, parsed document, generated count, promoted count, and restart hints from orchestration are preserved.

The default catalogue also distinguishes accepted and unaccepted pre-curated images, and removes rejected and shredded images from the visible flow; those transitions are not `error_state` values but ordinary feedback outcomes covered in §11 and §12.

**19. Accessibility and Responsiveness**

The static UI enforces the following accessibility and responsive contracts. The contracts are implemented in `static_ui.py` and asserted by `tests/test_a11y.py` and `tests/test_responsive_ui.py`.

ARIA contracts:
- descriptive `aria-label` on every `<button>` (Start, Pause, Stop, Accept, Reject, Shred, Toggle advanced view, Begin Generation, Cancel).
- candidate cards rendered as `<article class="candidate-card" role="button" tabindex="0" aria-label="Candidate cand_7">` so keyboard users can tab into the catalogue.
- live regions: `<footer data-region="progress-heartbeat" aria-live="polite" aria-label="Progress heartbeat">` and `<div data-region="notification" aria-live="assertive" aria-label="Run notification">`.
- a `<a href="#main-content" class="skip-link">Skip to main content</a>` element is the first child of `<body>`, and the main workspace is wrapped in `<main id="main-content">`.

Keyboard contracts:
- advanced toggle responds to Space and Enter via a `keydown` listener that flips `aria-pressed`.
- catalogue listens for ArrowUp and ArrowDown to move focus between `.candidate-card` elements; Enter and Delete dispatch a click on the focused card, which the existing JS maps to accept or shred via the card's `data-feedback` attribute.
- modal listens for Escape to click its cancel button.

Responsive contracts:
- `<meta name="viewport" content="width=device-width, initial-scale=1">` is present in the rendered head.
- an inline `<style>` block contains two media queries: `@media (max-width: 768px)` stacks the run controls vertically, switches the candidate grid to a single column, and reduces base font size; `@media (max-width: 479px)` further reduces section padding, hides non-essential list items in the progress heartbeat, and shrinks notification text.
- the carousel/detail layout may stack vertically on narrow screens and appear side-by-side on wider screens; the interaction contract remains the same.

Zoom and pan contracts:
- the catalogue is wrapped in `<div class="catalogue-viewport">` with a `.zoom-controls` toolbar.
- the zoom controls expose `+`, `-`, and Reset buttons, all rendered through `_button` with descriptive `aria-label`s.
- wheel events with the ctrl or meta key trigger zoom, with `clamp(value, 0.5, 3.0)` enforcing the scale bounds.
- click-and-drag panning is enabled when the scale is greater than 1.0; double-click resets the scale to 1.0.
- non-color-only status indication is preserved across all breakpoints.

**20. Acceptance Criteria**
The UI spec is implemented when:
- a user can enter a prompt and start the pre-run refinement modal.
- the modal defaults to report view and supports `Begin Generation` and `Cancel`.
- major graph structure is read-only in the modal.
- editable fields are limited to allowed fluid prompt-enhancement fields.
- eligible enum fields expose lock/unlock controls.
- the catalogue shows only pre-curated evaluation-survived images by default.
- selecting an image opens a single-image detail/carousel view.
- prompt and seed are visible in the selected image report.
- optional descriptors can appear in the report without gating curation.
- thumbs up maps to `accept`.
- thumbs down maps to `reject`.
- trash icon maps to `shred`.
- rejected and shredded images leave the default catalogue/carousel flow.
- shred is represented as severe defect feedback, not ordinary delete.
- bottom progress heartbeat remains visible during a run.
- one-line notification reports current stage, errors, or action-needed alerts.
- low-level diagnostics are not part of the default ordinary user path.
- the spec can be implemented in both proof-of-concept and production UI frameworks.

**21. Implementation Contract**

The UI exposes these contract surfaces:

- `src/bruteforce_canvas/static_ui.py` (`render_workspace_html`, `_advanced_view`, `_error_state`, `_catalogue`, `_detail`, `_progress`, `_button(aria_label, disabled, data_feedback)`, the inline `<style>` block with the 768px and 479px media queries, the inline `<script>` block with the keyboard and zoom/pan handlers, the catalogue-viewport wrapper, and the skip link + `<main>` wrap).
- `src/bruteforce_canvas/ui.py` (`RunWorkspaceReadModel` with additive fields `error_state`, `diagnostic_hold_enums`, `suppressed_enums`, `proposed_enums`, `raw_ood_signals`, `elapsed_seconds`, `vram_telemetry`; computed field `progress_heartbeat` returning the serialised heartbeat dict).
- `src/bruteforce_canvas/transport.py` (`EventBus` for CLI SSE; the UI consumes event bus updates when running against a streaming backend).
- `src/bruteforce_canvas/cli.py` (subcommands `render-workspace` and `stream`; the static UI is served by `render-workspace` and the live UI consumes `stream`).
- `src/bruteforce_canvas/telemetry.py` (`VRAMTelemetry` and `collect_vram_telemetry`, used by the read model and the heartbeat).

**22. Summary Requirement**
The UI presents a single hybrid run workspace. The ordinary path is prompt entry, pre-run review, mass generation, pre-curated catalogue browsing, single-image inspection, and accept/reject/shred feedback. The pre-run modal supports trust-but-adjust prompt refinement without becoming a graph editor. The catalogue contains only evaluation-survived images by default. Feedback has explicit downstream meaning for learning and dataset routing. Optional tags are informational only. The implementation remains framework-agnostic so a POC UI and a production UI can share the same interaction contract.
