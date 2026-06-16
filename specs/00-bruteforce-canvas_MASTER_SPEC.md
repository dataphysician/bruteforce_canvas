**Bruteforce Canvas Master Implementation Specification**
**Greenfield build guide for prompt deconstruction, LHS routing, fast generation, evaluation, orchestration, and UI/UX**

Status: authoritative high-level implementation specification

Audience: product owners, architects, backend engineers, ML pipeline engineers, UI/UX implementers, evaluator implementers, and QA owners

Primary goal: define the complete build strategy for Bruteforce Canvas in a way that removes contract ambiguity between subsystem teams. This document explains what must be built, why the system is shaped this way, the order in which the detailed markdowns must be consumed, and how implementation teams should avoid misunderstandings while building the project greenfield.

This document is the top-level entry point. It does not replace the subsystem specifications. It tells the coding team how to use them without conflict.

**1. Source Documents**
The project is specified by one master document plus six subsystem documents.

Read in this order:

1. `bruteforce-canvas_MASTER_SPEC.md`
2. `bruteforce-canvas_DAG_prompt.md`
3. `bruteforce-canvas_LHS_enum_router.md`
4. `bruteforce-canvas_fast_image_generation.md`
5. `bruteforce-canvas_Evaluator_pipeline.md`
6. `bruteforce-canvas_Orchestration.md`
7. `bruteforce-canvas_UI_UX.md`

Subsystem authority:
- Prompt deconstruction, graph ownership, raw-first extraction, canonicalization, verification, repair, and deterministic rendering are defined in `bruteforce-canvas_DAG_prompt.md`.
- LHS field eligibility, enum routing, compatibility priors, Thompson Sampling, GP-style combination memory, and router learning inputs are defined in `bruteforce-canvas_LHS_enum_router.md`.
- Model-agnostic resident fast image generation, GENERATOR_REGISTRY-registered builders, seed-sweep bundles, generator provenance, and generation worker contracts are defined in `bruteforce-canvas_fast_image_generation.md`.
- Batch evaluation, target manifests, IQA scoring, VLM alignment, optional tags/impact descriptors, failure types, blame hints, and disposition signals are defined in `bruteforce-canvas_Evaluator_pipeline.md`.
- Runtime lifecycle, stage handoff gates, shared vocabulary, run configuration, persistence, promotion, suppression, quarantine, retry, learning updates, feedback handling, and system actions are defined in `bruteforce-canvas_Orchestration.md`.
- The framework-agnostic user interface, pre-run refinement modal, catalogue, carousel, feedback controls, read models, events, and UI boundaries are defined in `bruteforce-canvas_UI_UX.md`.

If two documents appear to mention the same behavior, use the authority list above. A downstream document consumes upstream contracts; it does not redefine them.

**2. System Intent**
Bruteforce Canvas turns a user prompt into a continuous, evaluation-driven image exploration loop.

The system should:
- parse the user's prompt into a semantically owned scene document.
- preserve raw user intent.
- canonicalize known fields into locked enum affordances without overwriting the raw phrase.
- verify graph linkage and lane ownership before any generation.
- vary only safe, sampleable presentation fields.
- generate multiple seed images per prompt coordinate to reduce seed bias.
- evaluate every generated image before curation.
- promote only images that pass required post-generation gates.
- let the user accept, reject, or shred only images that reached the pre-curated pool.
- convert evaluator outputs and user feedback into learning updates.
- keep weak enum arms, weak combinations, and weak coordinates from dominating future generation.

The system should not:
- use LHS to repair missing scene facts.
- allow the renderer to invent graph participants.
- let the image generator decide semantic correctness.
- allow evaluator results to execute system actions directly.
- expose manual backend policy overrides in the ordinary UI.
- let optional tags or impact descriptors gate curation.

**3. Architectural Shape**
The architecture is intentionally staged.

The core sequence is:

```text
raw user prompt
-> PromptDocument extraction
-> field-scoped enum canonicalization
-> LLM verification
-> deterministic rendering contract
-> compatibility-constrained LHS routing
-> rendered prompt coordinates
-> default five-seed fast image generation (configurable; minimum bundle size is 3)
-> staged image evaluation
-> promotion/curation
-> user feedback
-> learning and lifecycle actions
-> UI read models and events
```

Each stage owns one decision class.

Prompt pipeline:
- owns what exists.
- owns graph relations, lane ownership, evidence, unresolved slots, prompt-improvement feedback, and renderable prompt construction.

Router:
- owns which missing or weak presentation fields may vary.
- owns compatibility pruning, candidate coordinates, and Bayesian ranking.

Fast generator:
- owns image synthesis from a rendered prompt, seed, dimensions, and generation settings.
- does not parse, judge, or repair prompts.

Evaluator:
- owns image scoring, pass/fail evidence, failure taxonomy, and blame hints.
- does not execute lifecycle actions.

Orchestrator:
- owns lifecycle, persistence, learning updates, promotion, demotion, quarantine, suppression, retry, stall, and feedback application.

UI:
- owns presentation, user confirmation, curation actions, read models, and ordinary workflow ergonomics.
- does not expose low-level backend override controls as the default user path.

**4. Why The System Is Designed This Way**
The design protects semantic intent while enabling high-throughput exploration.

Graph-first parsing exists because object descriptors, clothing, props, actions, locations, lighting, and camera language are easy to attach to the wrong subject when prompt parsing is flat. The graph gives every later lane a stable ownership root.

Raw strings first exists because canonical enum values are internal affordances. The user's wording remains the render source unless a renderer-safe phrase is explicitly selected. This prevents enum canonicalization from becoming destructive prompt rewriting.

Canonicalizers are field-scoped because enum matching is narrow after the primary prompt-LLM has already made scene-level ownership decisions. The default canonicalizer is embedding-first and may call a prompt-LLM fallback only for ambiguous or configured fields. This keeps the primary parse expressive while preserving bounded, parallel normalization.

LLM verification exists because deterministic code can confirm structural references, but it cannot reliably judge whether a phrase like `burgundy leather handbag` should be an element label or object-lane descriptors without prompt context.

LHS exists only after a clean parse because it is an enrichment system, not a missing-scene repair system. It should vary camera, lighting, focus, palette, composition, and other presentation fields without changing what the user asked for.

Compatibility priors exist because some enum combinations are structurally contradictory or model-hostile before rendering. LHS and Thompson Sampling need guardrails so the generator does not waste cycles on known weak combinations.

The seed-sweep bundle exists because a single seed can make a strong prompt coordinate look bad or a weak coordinate look good. Every randomized coordinate runs a seed-sweep bundle. The default bundle is `[7, 42, 156, 8888, 42069]`. The minimum bundle size is 3; a run may override the bundle via `RunConfig.seed_bundle` (see `generation.py` and `gates.py`).

Evaluation exists because writing a valid PNG is not success. A candidate must pass technical quality, prompt alignment, required element/relation/action preservation, constraints, and relevant guardrails before it enters the pre-curated pool.

The user feedback model exists because the user's curation bar is stricter than automated promotion. Accept, reject, and shred are simple UI actions with precise downstream learning meaning.

The UI is a hybrid workspace because the same run needs prompt entry, pre-run confirmation, progress visibility, catalogue browsing, single-image inspection, and feedback without forcing the user through separate operational tools.

**4.1 Model Adapter Conventions**
Local model adapters are the long-term integration surface for system-owned generation and evaluation models. They load local weights, expose explicit lifecycle methods such as `prewarm()`, and own their role-specific result schema.

The OpenAI-compatible server adapter is restricted to the prompt-LLM layer. It may serve:
- primary PromptDocument extraction.
- slice-scoped repair.
- PromptDocument verification.
- canonicalization fallback when embedding canonicalization is unavailable, ambiguous, or explicitly configured to defer.

The prompt-LLM server adapter is configured as one shared prompt service by default. The same `BC_LLM_PROVIDER=openai-compatible-server`, `BC_LLM_BASE_URL`, `BC_LLM_MODEL`, and `BC_LLM_API_KEY` settings serve extraction, repair, verification, and canonicalization fallback. Role-specific prompt-LLM overrides are not part of the base contract unless a subsystem spec defines them.

The prompt-LLM server adapter must not be used as a generic cloud switch for image generation, IQA, VLM alignment, or metacognitive impact. Those roles require their own local adapters and stable role contracts.

The base real-inference evaluator set is:
- JoyQuality SigLIP2 SO400M for IQA, with baseline quality cutoff `0.55`.
- MiniCPM-V 4.6 for image-prompt alignment.
- TRIBE v2 lite-qv as an optional metacognitive impact tagger, disabled by default and not part of the 16 GB VRAM base tier.

On a 16 GB NVIDIA Ampere class GPU, the base resident target is the active local generator plus JoyQuality and MiniCPM-V warmed before user interaction. TRIBE remains disabled unless capacity, licensing, and run policy explicitly enable it.

**5. Required Build Order**
Build the system in the order below. Do not build downstream stages before upstream contracts are implemented enough to emit the required handoff artifacts.

**5.1 Prompt Pipeline**
Reference: `bruteforce-canvas_DAG_prompt.md`

Build:
- Pydantic or equivalent PromptDocument models.
- primary prompt-LLM extraction call.
- graph elements and raw relations.
- object/action/cinematography/constraint lanes.
- evidence spans.
- unresolved and blocked slots.
- embedding-first field-scoped enum canonicalization with prompt-LLM fallback.
- canonical enum metadata using shared statuses.
- prompt-LLM verifier.
- slice-scoped repair loop.
- deterministic renderer that begins prompts with `Generate`.

Exit gate:
- verified PromptDocument exists.
- blocking issues are absent.
- graph facts are stable.
- raw strings are preserved.
- target manifest can be derived.
- rendered prompt preserves required graph facts.

Do not proceed if:
- relation endpoints are missing.
- required elements are unresolved.
- action targets are invented.
- lighting treatment is incorrectly promoted into a graph element.
- negative constraints remove required scene content.

**5.2 LHS Enum Router**
Reference: `bruteforce-canvas_LHS_enum_router.md`

Build:
- field state assignment.
- sampleable axis discovery.
- fixed-arm and sampled-arm separation.
- compatibility prior.
- LHS row generation.
- Thompson arm scoring.
- GP-style combo scoring.
- compatibility trace.
- CandidateCoordinate output.

Exit gate:
- each coordinate has `coordinate_id`, `run_id`, `prompt_document_id`, `target_manifest_id`.
- hard compatibility rejects are absent.
- fixed arms are not mutated.
- sampled arms are eligible.
- compatibility trace is persisted.
- lifecycle state is `proposed`.

Do not proceed if:
- LHS is being used to add missing elements, props, relations, or action targets.
- an explicit locked user fact is replaced by a sampled enum.
- compatibility trace is missing.
- coordinate identity is unstable.

**5.3 Fast Image Generation**
Reference: `bruteforce-canvas_fast_image_generation.md`

Build:
- resident fast generation adapter.
- the registered local fast image generator builder (`GENERATOR_REGISTRY['bonsai']` by default; `stub` is the deterministic fixture builder).
- generation request model.
- seed sweep request model.
- artifact persistence.
- generator provenance.
- file validity check.
- seed-sweep aggregate output.

Required sweep:

```json
[7, 42, 156, 8888, 42069]
```

Exit gate:
- every generated seed image is persisted or infrastructure-blocked.
- candidate records include run ID, prompt document ID, target manifest ID, coordinate ID, seed, rendered prompt, generator model ID, backend, settings, image path, file validity, timestamp.
- seed candidates are grouped by coordinate ID.

Do not proceed if:
- generator output is treated as promoted without evaluation.
- a failed generation applies semantic penalties.
- thresholds are defined inside the generator adapter.
- any specific model name is used as the generic stage name.

**5.4 Evaluator Pipeline**
Reference: `bruteforce-canvas_Evaluator_pipeline.md`

Build:
- evaluator target manifest.
- JoyQuality IQA evaluator adapter.
- MiniCPM-V alignment/guardrail evaluator adapter.
- optional TRIBE v2 descriptor/tag/impact adapter, disabled unless policy and hardware allow it.
- batch request and result schemas.
- per-image result.
- coordinate aggregate.
- failure taxonomy.
- blame hints.
- disposition signals.

Exit gate:
- every evaluated image has quality score, alignment score, pass flags, failure types, localized blame, disposition signal, evaluator versions, and confidence.
- coordinate aggregate groups the bundle's seed results (default 5; minimum 3).
- evaluator payloads are persisted.

Do not proceed if:
- optional tags are used as promotion gates.
- evaluator executes demotion, purge, quarantine, suppression, or retry directly.
- pass/fail is collapsed into a score with no failure evidence.
- seed sweep aggregate is missing.

**5.5 Orchestration**
Reference: `bruteforce-canvas_Orchestration.md`

Build:
- run configuration.
- pre-run lock configuration.
- shared lifecycle state machine.
- stage handoff enforcement.
- persistence contract.
- promotion/curation application.
- feedback application.
- Thompson and GP learning updates.
- coordinate retirement.
- enum suppression policy.
- coordinate quarantine policy.
- infrastructure retry policy.
- stall guard.
- run progress and telemetry.
- read models/events for UI consumption.

Exit gate:
- all system actions are idempotent.
- learning updates are persisted exactly once per eligible event.
- thresholds are run-local.
- stage outputs cannot skip required gates.
- feedback is traceable to automated evaluation, user action, or both.

Do not proceed if:
- evaluator results execute actions directly.
- feedback duplicates learning deltas on repeated clicks.
- infrastructure failures apply semantic penalties.
- run state cannot be reconstructed from persisted records.

**5.6 UI/UX**
Reference: `bruteforce-canvas_UI_UX.md`

Build:
- framework-agnostic run workspace.
- prompt text area.
- start/pause/stop controls.
- pre-run refinement modal.
- parser/canonicalizer report view.
- advanced edit view for fluid fields only.
- lock/unlock enum controls.
- pre-curated image catalogue.
- single-image carousel/detail view.
- accept/reject/shred controls.
- persistent progress heartbeat.
- one-line notification/status area.
- optional descriptor display.
- read models and event consumption.

Exit gate:
- user can start from a prompt.
- pre-run modal blocks generation when parsing is blocked.
- major graph structure is read-only.
- catalogue shows only eligible pre-curated images by default.
- feedback semantics map exactly to backend `accept`, `reject`, and `shred`.
- optional tags are informational only.

Do not proceed if:
- UI exposes graph editing as the ordinary path.
- UI exposes backend policy override controls.
- shred is presented as ordinary delete.
- optional tags affect curation gating.
- framework choices are baked into the product contract.

**6. Shared Vocabulary**
Every team must use the shared vocabulary from `bruteforce-canvas_Orchestration.md`.

Canonical enum statuses:
- `matched_active`
- `matched_suppressed`
- `matched_diagnostic_hold`
- `unmatched_raw_only`
- `proposed_new_enum`
- `rejected_invalid`

Field routing states:
- `explicit_raw`
- `explicit_locked`
- `explicit_locked_suppressed`
- `entailed_locked`
- `entailed_locked_suppressed`
- `missing_sampleable`
- `weak_sampleable`
- `suppressed_sampleable`
- `blocked`
- `conflict`

Candidate lifecycle states:
- `proposed`
- `rendered`
- `generating`
- `generated`
- `evaluating_iqa`
- `evaluating_vlm`
- `evaluating_impact`
- `evaluated`
- `promoted`
- `curated`
- `strong`
- `viable`
- `fragile`
- `failed`
- `demoted`
- `retired`
- `quarantined`
- `blocked`

Feedback actions:
- `accept`
- `reject`
- `shred`

No implementation should introduce alternate names for these states.

**7. Handoff Artifacts**
Each stage must emit the artifact required by the next stage.

Prompt pipeline emits:
- verified PromptDocument.
- canonical enum metadata.
- target manifest.
- rendered prompt compiler.
- verifier report.

Router emits:
- CandidateCoordinate batch.
- fixed arms.
- sampled arms.
- LHS rows.
- compatibility trace.
- Bayesian score.
- combo signature.

Renderer emits:
- rendered prompt coordinate.
- prompt beginning with `Generate`.
- target manifest reference.
- rendering trace.

Generator emits:
- candidate images.
- candidate records.
- file validity.
- generator provenance.
- seed sweep aggregate.

Evaluator emits:
- per-image evaluator results.
- coordinate aggregate.
- failure types.
- blame hints.
- disposition signals.

Orchestrator emits:
- run state.
- persistence records.
- lifecycle actions.
- learning deltas.
- UI read models.
- event stream concepts.

UI emits:
- run start intent.
- pre-run begin/cancel decisions.
- allowed field edits.
- enum lock/unlock choices.
- feedback actions.

**8. Persistence Strategy**
Persistence is not an optional logging detail. It is the system's replay, audit, learning, and debugging backbone.

Persist:
- raw user prompt.
- PromptDocument version.
- target manifest.
- canonicalizer outputs.
- verifier outputs.
- lock configuration.
- run configuration.
- coordinates.
- rendered prompts.
- generation settings.
- image paths.
- seed identity.
- generator model and backend.
- evaluator payloads.
- pass/fail flags.
- promotion/curation state.
- feedback actions.
- learning deltas.
- compatibility traces.
- suppression/quarantine counters.
- infrastructure failures.
- lifecycle transitions.

The implementation should make every promoted image traceable to:

```text
raw prompt
-> PromptDocument
-> target manifest
-> coordinate
-> rendered prompt
-> seed
-> generated artifact
-> evaluator payload
-> promotion decision
-> user feedback state
-> learning update
```

**9. Ambiguity Prevention Strategy**
The coding team should use these rules to avoid uncertainty and misunderstandings.

Rule 1: one owner per decision.
- Prompt facts belong to the prompt pipeline.
- Variation belongs to the router.
- Pixel synthesis belongs to the generator.
- Image judgment belongs to evaluators.
- Actions and learning belong to orchestration.
- Presentation and user gestures belong to UI.

Rule 2: do not infer across stage boundaries.
- A downstream stage consumes upstream artifacts.
- A downstream stage must not recreate upstream decisions by reparsing text or guessing missing fields.

Rule 3: raw user language remains available.
- Enum canonicalization adds metadata.
- It does not destroy the raw phrase.

Rule 4: blocked means blocked.
- If a prompt or coordinate cannot pass its gate, do not let a downstream stage "try anyway" unless the owning spec explicitly defines a safe downgrade.

Rule 5: every generated candidate must be reproducible.
- Store prompt, coordinate, seed, generator settings, model ID, backend, and artifact path.

Rule 6: user feedback is simple but semantically strong.
- `accept`, `reject`, and `shred` are the only ordinary curation actions.
- They must not be overloaded with unrelated meanings.

Rule 7: optional models stay optional.
- Optional tags, descriptors, or impact outputs may inform the report.
- They do not gate promotion or override feedback.

Rule 8: framework choices do not rewrite product contracts.
- A Gradio POC and a production UI must preserve the same interaction semantics.

Rule 9: no hidden final LLM normalizer.
- Prompt naturalness is handled by deterministic rendering and upstream field ownership.
- LHS fanout must not require one more LLM rewrite per candidate.

Rule 10: all handoff objects get stable IDs.
- `run_id`
- `prompt_document_id`
- `target_manifest_id`
- `coordinate_id`
- `candidate_id`

**10. Greenfield Development Phases**
Use these phases to build without crossing contracts.

Phase 1: schema and IDs
- implement shared IDs and persistence shell.
- define run, prompt document, target manifest, coordinate, candidate, evaluator result, feedback, and learning update records.
- no image generation required.

Phase 2: prompt pipeline
- implement PromptDocument creation, canonicalization, verification, and rendering.
- test static scenes, zero-action scenes, multi-object interactions, target-preserving actions, lighting ownership, and unresolved slots.

Phase 3: router
- implement field states, compatibility prior, LHS rows, Thompson scoring, and CandidateCoordinate output.
- test that LHS cannot add missing graph facts.

Phase 4: generator adapter
- implement active generator interface.
- implement the GENERATOR_REGISTRY and the registered builder(s).
- implement the seed-sweep bundle contract (default 5 seeds; minimum 3).
- persist candidate records even before full evaluator integration.

Phase 5: evaluator
- implement IQA, alignment, failure taxonomy, blame hints, and aggregate results.
- keep optional descriptor/impact adapters informational.

Phase 6: orchestrator
- implement run lifecycle, handoff gates, promotion, feedback, learning, suppression/quarantine, retry, stall, and telemetry.
- ensure idempotent actions.

Phase 7: UI
- implement hybrid run workspace.
- implement pre-run modal.
- implement catalogue and carousel.
- implement feedback actions.
- implement progress heartbeat and one-line status.

Phase 8: end-to-end validation
- run a prompt through parse, route, render, generate, evaluate, promote, display, feedback, and learning update.
- verify every handoff artifact is persisted and traceable.

**11. Acceptance Criteria**
The greenfield implementation is aligned with this master spec when:
- each subsystem consumes the documents in the required order.
- each stage emits its required handoff artifact.
- no downstream stage redoes upstream semantic work.
- raw prompt text is preserved through rendering and provenance.
- enums are metadata affordances, not destructive rewrites.
- LHS samples only eligible presentation fields.
- every randomized prompt coordinate runs a seed-sweep bundle. The default bundle is `[7, 42, 156, 8888, 42069]` and the minimum bundle size is 3.
- promotion requires evaluator pass gates.
- UI feedback is limited to `accept`, `reject`, and `shred`.
- reject and shred remove images from the default curated catalogue.
- shred is excluded from IQA negative fine-tuning.
- optional descriptors remain informational.
- all lifecycle actions are applied by orchestration.
- every generated image is reproducible from persisted provenance.
- the UI can be implemented as a POC or production build without changing interaction semantics.

**12. Detailed Reference Map**
Use the detailed specs for implementation depth.

For graph extraction and rendering:
- `bruteforce-canvas_DAG_prompt.md`
- especially sections on graph-first design, raw strings first, evidence-gated inference, Pydantic models, validators, repair contracts, LLM call contracts, rendering helpers, and walkthroughs.

For randomized field enrichment:
- `bruteforce-canvas_LHS_enum_router.md`
- especially field states, clean parse gate, compatibility prior, Bayesian scoring, locked enum penalty policy, OOD enum handling, implementation interfaces, and router algorithm.

For fast generation:
- `bruteforce-canvas_fast_image_generation.md`
- especially runtime loading contract, generation settings, seed-sweep bundle, coordinate outcomes, generation provenance, evaluator coupling, GPU residency, and interfaces.

For evaluation:
- `bruteforce-canvas_Evaluator_pipeline.md`
- especially target manifest, evaluation aspects, model families, cutoff semantics, request/result shapes, failure types, blame hints, disposition signals, and seed sweep aggregation.

For runtime:
- `bruteforce-canvas_Orchestration.md`
- especially build order, handoff gates, shared vocabulary, run configuration, lifecycle, learning, suppression, stall guard, persistence, system actions, feedback integration, and pseudocode.

For UI:
- `bruteforce-canvas_UI_UX.md`
- especially product position, run controls, pre-run refinement modal, editing boundary, catalogue, carousel, feedback semantics, progress/status, optional descriptors, read models, event stream, and acceptance criteria.

The contract surfaces referenced by these specs are:
- `src/bruteforce_canvas/prompt_render.py` for Spec 01 §8 helper surface.
- `src/bruteforce_canvas/canonicalizers.py` and `src/bruteforce_canvas/llm_clients.py` for the embedding canonicalizer and prompt-LLM server adapter boundary.
- `src/bruteforce_canvas/generator_registry.py` for Spec 03 GENERATOR_REGISTRY pattern.
- `src/bruteforce_canvas/real_adapters.py` for Specs 03 and 04 (JoyQuality, MiniCPM-V 4.6, TRIBE v2 lite-qv adapters).
- `src/bruteforce_canvas/evaluation.py` for Spec 04 outcome thresholds.
- `src/bruteforce_canvas/orchestration.py`, `gates.py`, `run_service.py`, `loop.py`, `balancer.py`, `transport.py` for Spec 05.
- `src/bruteforce_canvas/static_ui.py` and `src/bruteforce_canvas/ui.py` for Spec 06.
- `src/bruteforce_canvas/cli.py` for the `render-workspace` and `stream` subcommands that expose the UI.
- `src/bruteforce_canvas/spec_compliance.check_all` for the per-spec compliance check referenced throughout.

**13. Summary Requirement**
Bruteforce Canvas must be built as an ordered, contract-gated pipeline. The system starts with semantic prompt ownership, not image generation. It then enriches only safe fields, runs seed-sweep bundles, evaluates before curation, applies simple user feedback, and learns from all persisted evidence. Every team should implement only the decisions owned by its stage, rely on stable handoff artifacts, and use the detailed markdowns as the source of truth for subsystem depth.
