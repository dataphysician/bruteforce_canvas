# Design

## Source of truth
- Status: Draft
- Last refreshed: 2026-06-15
- Primary product surfaces: Gradio simulation workspace, static replay workspace, future production run workspace.
- Evidence reviewed: `specs/06-bruteforce-canvas_UI_UX.md`, `src/bruteforce_canvas/ui.py`, `src/bruteforce_canvas/static_ui.py`, `src/bruteforce_canvas/app_controller.py`, `tests/test_ui_contract.py`, `tests/test_static_ui_renderer.py`, `tests/test_a11y.py`, user sketch at `/home/damaso/python/bruteforce_canvas-UI.jpg`.

## Brand
- Personality: direct, technical, inspectable, high-throughput, local-first.
- Trust signals: visible prompt provenance, lock state, thresholds, evaluator scores, seed IDs, promotion reasons, feedback state.
- Avoid: marketing pages, decorative hero layouts, graph-editor controls, evaluator override controls, policy jargon as primary UI.

## Product goals
- Goals: let a user enter a prompt, review structured parsing, configure lock and threshold settings, launch 5-seed sweeps, inspect evaluator outcomes, curate survivors, and submit feedback.
- Non-goals: manual graph editing, enum registry editing, raw evaluator policy overrides, full developer console by default.
- Success signals: the user can tell what was parsed, what is locked, which seeds failed, which images survived, and why a selected image is in the curated catalog.

## Personas and jobs
- Primary personas: local image-generation operator, prompt-system developer, evaluator/pipeline debugger.
- User jobs: start a run from natural language, confirm prompt structure, control variation space, monitor seed sweeps, preserve useful images, reject false positives.
- Key contexts of use: local workstation, long-running generation sessions, backend contracts still evolving.

## Information architecture
- Primary navigation: one workspace screen with progressive reveal.
- Core routes/screens: prompt entry, pre-run review, active 5-seed preview, curated catalog, selected-image carousel/detail.
- Content hierarchy: prompt first, parse review second, active run state third, curated results and selected detail below.

## Design principles
- Principle 1: expose decisions, not internals. Show lock states and scores, but keep graph-breaking controls out of ordinary flow.
- Principle 2: make status visible at the point of action. Failed seeds should be visually muted and marked before the next batch refresh.
- Tradeoffs: use an inline Gradio pre-run panel instead of a true modal until a stable built-in modal contract is available.

## Visual language
- Color: neutral workspace base with black text, teal/green pass signals, red fail signals, amber warnings.
- Typography: compact application typography sized for scanning, not hero-scale marketing type.
- Spacing/layout rhythm: dense but breathable 8-16 px rhythm, cards only for individual candidates and detail surfaces.
- Shape/radius/elevation: subtle borders, 8 px radius or less, minimal elevation.
- Motion: generator yields a pending preview, then an evaluated preview; avoid ornamental animation.
- Imagery/iconography: generated raster seed cards, simple tool buttons, mic placeholder reserved for ASR integration.

## Components
- Existing components to reuse: `PreRunModalReadModel`, `RunWorkspaceReadModel`, `CandidateCard`, `DetailReport`, lock configuration, feedback events.
- New/changed components: Gradio Blocks app, lock-state table, IQA/alignment sliders, seed preview gallery, curated gallery, selected-image detail controls.
- Variants and states: pending seed, failed seed, fragile survivor, viable survivor, strong survivor, accepted, rejected, shredded, blocked prompt.
- Token/component ownership: Gradio simulation CSS lives with the simulation module; static HTML renderer keeps its existing contract.

## Accessibility
- Target standard: practical WCAG 2.1 AA intent for local tooling.
- Keyboard/focus behavior: Gradio-native focus for inputs/buttons; galleries keep native preview/select behavior.
- Contrast/readability: red fail states and dark muted overlays must remain readable.
- Screen-reader semantics: component labels should be concise and action-oriented.
- Reduced motion and sensory considerations: no flashing or continuous motion.

## Responsive behavior
- Supported breakpoints/devices: desktop-first local app with usable tablet/mobile wrapping.
- Layout adaptations: prompt row wraps, 5-seed gallery collapses columns as Gradio allows, detail stack becomes vertical.
- Touch/hover differences: feedback actions remain explicit buttons.

## Interaction states
- Loading: pending 5-seed preview appears immediately before evaluator state lands.
- Empty: curated catalog starts empty and remains visible as a region.
- Error: empty prompt and blocked parse prevent generation.
- Success: promoted candidates appear in the curated catalog and selected detail.
- Disabled: mic button is disabled until ASR is added; TRIBE metacognitive score is shown as disabled in detail.
- Offline/slow network, if applicable: app runs local simulation without network.

## Content voice
- Tone: terse operational labels.
- Terminology: use "IQA", "alignment", "seed", "locked", "unlocked", "curated", "fragile", "viable", "strong".
- Microcopy rules: status messages should report current system state, not explain how to use the UI.

## Implementation constraints
- Framework/styling system: latest installed Gradio Blocks, custom CSS scoped by element IDs/classes, PIL-generated raster placeholders.
- Design-token constraints: avoid adding a separate design-system layer.
- Performance constraints: simulation image generation must be lightweight and write only to `/tmp`.
- Compatibility constraints: Gradio remains an optional dependency; CLI imports it lazily.
- Test/screenshot expectations: unit tests cover state transitions and app construction; browser screenshots can be added after backend stabilizes.

## Open questions
- [ ] Should failed 5-seed previews persist in a visible run history, or remain only in the active sweep before refresh?
- [ ] Should lock overrides be persisted as backend `LockOverride` records in this Gradio prototype, or only reflected in generated prompt metadata for now?
- [ ] Should ASR use a local microphone component or external transcript input once the ASR model is selected?
