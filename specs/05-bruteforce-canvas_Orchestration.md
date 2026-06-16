**Generation, Evaluation, and Learning Orchestration Specification**
**Companion to `bruteforce-canvas_DAG_prompt.md`, `bruteforce-canvas_LHS_enum_router.md`, `bruteforce-canvas_fast_image_generation.md`, and `bruteforce-canvas_Evaluator_pipeline.md`**

Status: implementation specification

Audience: backend orchestration, resident worker, evaluator worker, persistence, and learning-loop implementers

Primary goal: define the backend engine that turns one user prompt into a continuous generate/evaluate/learn loop. The orchestrator coordinates prompt deconstruction, enum locking/unlocking, Bayesian LHS prompt variation, fast image generation, batch evaluation, persistence, automated curation, swipe/thumb feedback on pre-curated images, and system actions such as promotion, demotion, coordinate retirement, enum suppression, infrastructure retry, and stall shutdown.

**1. Scope**
The orchestrator owns the backend state machine.

It coordinates:
- raw user prompt intake.
- PromptDocument extraction, enum canonicalization, and verification.
- pre-run enum lock configuration.
- Bayesian Balancer state loading.
- compatibility-constrained LHS coordinate generation.
- seed-sweep generation bundles per prompt coordinate (default 5 seeds; minimum 3).
- resident fast image generation.
- evaluator batch routing.
- post-evaluation persistence.
- Thompson Sampling and Gaussian Process updates.
- promoted/curated state changes.
- coordinate retirement or retry.
- enum suppression or recovery.
- stall guard shutdown.

It does not own:
- primary prompt-LLM parsing details.
- enum canonicalizer internals.
- final prompt rendering templates.
- evaluator model scoring internals.
- UI presentation details for confirmation or swipe/thumb controls.

The orchestrator consumes the contracts from the prompt, LHS, fast generator, and evaluator specs and applies system actions.

**1.1 Build Order and Contract Authority**
Implementation should consume the markdowns in this order:

1. `bruteforce-canvas_DAG_prompt.md`
   - Build the PromptDocument extraction, field-scoped enum canonicalization, LLM verification, repair loop, and deterministic renderer.
   - Output: verified PromptDocument, rendered prompt compiler, target manifest, and canonical enum metadata.
2. `bruteforce-canvas_LHS_enum_router.md`
   - Build sampleable field selection, compatibility prior, LHS coordinate generation, Thompson arm scoring, and GP-style combo scoring.
   - Input: verified PromptDocument and canonical enum metadata.
   - Output: prompt coordinates ready for rendering/generation.
3. `bruteforce-canvas_fast_image_generation.md`
    - Build the resident local fast image generation adapter, the GENERATOR_REGISTRY, registered builder(s), candidate artifact persistence, and seed-sweep grouping. The bonsai builder is the default production builder; `stub` is the deterministic fixture builder.
   - Input: rendered coordinate, seed, generation settings, and provenance.
   - Output: generated candidate artifacts with reproducible metadata.
4. `bruteforce-canvas_Evaluator_pipeline.md`
   - Build low-latency IQA, VLM alignment, optional impact scoring, failure taxonomy, blame hints, and disposition signals.
   - Input: generated candidates plus target manifest and coordinate metadata.
   - Output: per-image evaluator payloads and coordinate aggregate payloads.
5. `bruteforce-canvas_Orchestration.md`
   - Build the runtime state machine that wires the previous contracts together, applies system actions, persists learning state, and handles feedback.

Contract precedence:
- shared lifecycle, threshold, feedback, promotion, suppression, quarantine, retry, and persistence policy lives in this orchestration spec.
- prompt parsing, lane ownership, and deterministic rendering live in the DAG prompt spec.
- candidate enum selection, compatibility scoring, and Bayesian ranking live in the LHS router spec.
- fast image generation and seed-bundle execution live in the generator spec. The bonsai builder in `GENERATOR_REGISTRY` is the default implementation example; the registry itself lives in `src/bruteforce_canvas/generator_registry.py`.
- evaluator model outputs and failure/blame schemas live in the evaluator spec.

When two docs mention the same concept, the more downstream owner consumes the upstream contract but does not redefine it. For example, the generator spec may list active thresholds in examples, but threshold defaults and run-local overrides are owned by this orchestration spec.

**1.1.1 Stage Handoff Gates**
Each stage must emit the listed artifact before the next stage can start. These gates are procedural contracts, not optional documentation examples.

Prompt schema gate:
- artifact: verified PromptDocument.
- required fields: raw user prompt, seed prompt, graph elements, raw relations, object/action/cinematography/constraint lanes, evidence spans, canonical enum metadata, verifier approval or structured verifier issues.
- next stage may start only when verifier approval is true and blocking issues are absent.

Router gate:
- artifact: CandidateCoordinate batch.
- required fields: `coordinate_id`, `run_id`, `prompt_document_id`, `target_manifest_id`, fixed arms, sampled arms, LHS row, compatibility trace, Bayesian score, combo signature, lifecycle state `proposed`.
- next stage may start only when each coordinate has no hard compatibility reject and has a stable `coordinate_id`.

Rendering gate:
- artifact: rendered prompt coordinate.
- required fields: `coordinate_id`, `run_id`, `prompt_document_id`, `target_manifest_id`, rendered prompt beginning with `Generate`, fixed arms, sampled arms, compatibility trace, Bayesian score.
- next stage may start only when required graph facts survive rendering and the rendered prompt is non-empty.

Fast generation gate:
- artifact: generated seed candidates.
- required fields: `run_id`, `prompt_document_id`, `target_manifest_id`, `coordinate_id`, seed, rendered prompt, generator model ID, generator backend, generation settings, image path, file validity state, generation timestamp.
- next stage may start only after each generated artifact is persisted or marked as infrastructure-blocked.

Evaluation gate:
- artifact: per-image evaluator payloads plus coordinate aggregate.
- required fields: candidate ID, run ID, prompt document ID, target manifest ID, coordinate ID, seed, quality score, alignment score, pass flags, failure types, localized blame hints, disposition signal, evaluator versions, coordinate aggregate outcome.
- next stage may start only after evaluator payloads are persisted and grouped by the seed-sweep bundle per coordinate.

Learning/action gate:
- artifact: applied orchestrator actions and learning updates.
- required fields: promotion/curation state, Thompson alpha/beta deltas, GP combo-affinity delta, coordinate lifecycle update, suppression/quarantine counters, feedback source when present, persistence version.
- next iteration may start only after learning updates are idempotently persisted.

**1.2 Shared Vocabulary**
These terms are shared across all implementation stages.

Canonical enum status:
- `matched_active`: a supplied enum matched and is active for its field.
- `matched_suppressed`: a supplied enum matched but is suppressed for the active model/context.
- `matched_diagnostic_hold`: a supplied enum matched but cannot proceed to mass generation because confidence, policy, or suppression context is ambiguous.
- `unmatched_raw_only`: no supplied enum matched; preserve the raw phrase.
- `proposed_new_enum`: no supplied enum matched, but the phrase is reusable enough to enter the enum candidate queue.
- `rejected_invalid`: the canonicalizer returned an invalid field/value combination and the raw phrase remains authoritative.

Field routing state:
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

Candidate lifecycle:
- `proposed`: coordinate has been built but not rendered/generated.
- `rendered`: prompt string was compiled from a coordinate.
- `generating`: resident fast image generation is in progress.
- `generated`: image artifact was produced and persisted.
- `evaluating_iqa`: IQA evaluation is in progress.
- `evaluating_vlm`: VLM alignment or guardrail evaluation is in progress.
- `evaluating_impact`: optional impact evaluation is in progress.
- `evaluated`: required evaluator payloads were persisted.
- `promoted`: automated evaluation passed the active run thresholds.
- `curated`: image is in the pre-curated pool and can receive swipe/thumb feedback.
- `strong`: coordinate seed bundle passed strongly and may seed surf.
- `viable`: coordinate seed bundle produced at least one promoted image.
- `fragile`: coordinate seed bundle produced weak or seed-sensitive pass behavior.
- `failed`: coordinate seed bundle produced no promoted images for semantic or quality reasons.
- `demoted`: image no longer qualifies as curated/promoted after evaluator or feedback update.
- `retired`: exact coordinate is no longer allocated seed budget.
- `quarantined`: coordinate or combo is temporarily excluded by policy.
- `blocked`: infrastructure or unresolved semantic state prevents judgment.

Feedback actions:
- `accept`: pre-curated image meets the user's bar.
- `reject`: pre-curated image was an automated false positive.
- `shred`: pre-curated image was a severe automated false positive.

No implementation stage should introduce alternate names for these shared states.

**2. End-to-End Flow**
The backend flow is:

```text
User prompt
-> PromptDocument extraction
-> enum canonicalization
-> LLM verification
-> priority/lhs/evaluation policy assignment
-> pre-run lock configuration
-> Bayesian Balancer state load
-> compatibility-constrained LHS coordinate proposal
-> seed-sweep bundle per coordinate (default 5 seeds; minimum 3)
-> IQA batch evaluation
-> survivor VLM alignment and guardrail evaluation
-> optional metacognitive impact scoring on capable hardware
-> post-evaluation action application
-> Thompson/GP learning update
-> continue until stop, stall guard, or configured watermark condition
```

Generated attempts are unbounded in long-running mode. The run does not stop because a fixed number of images were generated. It stops or pauses only because of explicit stop, infrastructure failure policy, stall guard, or an optional configured promoted/curated watermark.

**3. Stage Ownership**
Prompt pipeline:
- decides what exists.
- attaches raw relations, lanes, evidence, priorities, and policies.
- decides default locked state for canonicalized fields.
- does not generate images.

Bayesian Balancer:
- samples eligible unlocked enum arms.
- applies Thompson Sampling over enum arms.
- applies Gaussian Process style combination memory over enum combinations.
- consumes evaluator/orchestrator feedback.
- does not parse prompts or evaluate image pixels.

LHS router:
- gives coverage over sampleable domains.
- respects compatibility priors.
- emits prompt coordinates.
- does not decide post-generation success.

Fast image generation backend:
- generates images from rendered prompt coordinates and seeds using a configured fast image model.
- the generator is model-agnostic; builders are registered through `GENERATOR_REGISTRY` and the live builder is selected by the run configuration.
- does not judge quality or alignment.

Evaluator:
- scores images and emits pass/fail, failure types, blame hints, and disposition signals.
- does not execute purge, demotion, retry, or sampler updates.

Orchestrator:
- persists everything.
- applies evaluator disposition signals.
- updates Bayesian state.
- controls loop lifecycle.

**4. Run Configuration**
Each orchestration run begins with an explicit run configuration.

```json
{
  "run_id": "run_00091",
  "raw_user_prompt": "a woman in a black coat walking toward a glass door, camera slowly pushes in, blurry background, low view",
  "mode": "continuous",
  "iqa_cutoff": 0.55,
  "alignment_cutoff": 0.25,
  "human_iqa_cutoff": 0.70,
  "seed_bundle": [7, 42, 156, 8888, 42069],  # default; minimum bundle size is 3 (MIN_SEED_BUNDLE_SIZE)
  "stall_window_seconds": 1800,
  "stall_min_promoted": 10,
  "promoted_high_watermark": null,
  "promoted_low_watermark": null,
  "metacognitive_impact_enabled": false,
  "metacognitive_min_vram_gib": 24
}
```

Defaults:
- IQA cutoff: `0.55`.
- alignment cutoff: run-configured default unless overridden.
- human/animal deformity guardrail: enabled when detected.
- seed bundle: `[7, 42, 156, 8888, 42069]` (default; runs may override via `RunConfig.seed_bundle`; minimum bundle size is `MIN_SEED_BUNDLE_SIZE = 3`).
- stall window: `30` minutes.
- stall minimum promoted/curated images: `10`.
- metacognitive impact evaluation: disabled unless hardware and policy allow it.

Thresholds are user- or system-parameterized before the run starts. The orchestrator treats thresholds as run-local settings and persists them.

Prompt-LLM service defaults:
- `BC_LLM_PROVIDER=openai-compatible-server` selects the prompt-LLM server adapter when an external OpenAI-compatible endpoint is used.
- `BC_LLM_BASE_URL`, `BC_LLM_MODEL`, and `BC_LLM_API_KEY` configure the shared prompt service.
- the same prompt service is used for extraction, repair, verification, and canonicalization fallback.
- evaluator and generator services are not routed through this prompt-LLM adapter.

**5. Pre-Run Lock Configuration**
Before mass generation starts, the system must expose a backend configuration object for pre-run enum lock confirmation.

Default lock policy:
- canonicalized user-stated enums are locked.
- verifier-required graph facts are locked.
- negative guards are locked.
- inferred canonical matches default to locked only when verifier confidence is clear; proposed new enums remain raw-preserving metadata and do not become LHS-sampleable until admitted into the enum registry.
- missing presentation fields default to sampleable.
- weak presentation fields default to sampleable if verifier marks them flexible.
- suppressed canonical matches remain recognized, but do not become active sampled replacements.

Lock configuration shape:

```json
{
  "field_path": "cinematography.lighting_mood",
  "raw_value": "golden hour",
  "enum_value": "GOLDEN_HOUR",
  "canonical_status": "matched_active",
  "priority": "locked_required",
  "lhs_policy": "fixed",
  "render_policy": "use_raw_or_safe_raw_preserving_phrase",
  "evaluation_policy": "must_match",
  "learning_policy": "track_locked_reliability",
  "lock_source": "llm_canonicalizer",
  "user_adjustable": true
}
```

The orchestrator must persist both the default lock state and the effective run lock state. This allows later audit of whether an enum was locked because the parser canonicalized it or because a pre-run configuration changed it.

Canonical status values:

`matched_active`: canonical enum matched and is available for its configured role.

`matched_suppressed`: canonical enum matched but is suppressed for the active model/context. If user-authored or locked, keep the raw phrase fixed and render from raw or safe raw-preserving phrasing. If sampled or missing, exclude from ordinary LHS selection.

`matched_diagnostic_hold`: canonical enum matched but is held before mass generation because confidence, policy, or suppression state is ambiguous.

`unmatched_raw_only`: no enum matched; preserve the raw value.

`proposed_new_enum`: raw phrase appears reusable but is not in the registry.

`rejected_invalid`: canonicalizer output was invalid for the field or outside the supplied enum context; preserve the raw phrase and do not attach enum behavior.

The backend owns the lock configuration contract. The UI owns how that configuration is presented.

**6. Bayesian Balancer**
The Bayesian Balancer is the learning controller around LHS.

It maintains:
- per-enum-arm Thompson Sampling state.
- enum-combination Gaussian Process style affinity state.
- temporary suppression state for repeatedly failing enum arms.
- coordinate history.
- prompt-cluster or model-family context keys when available.

Per-arm state:

```json
{
  "field_path": "cinematography.camera_angle",
  "enum_value": "LOW_ANGLE",
  "alpha": 8.0,
  "beta": 5.0,
  "suppressed_until": null,
  "context_key": "bonsai_ternary_4b"
}
```

Combination state:

```json
{
  "combo_signature": "shot_size=EXTREME_CLOSE_UP|action_complexity=MULTI_TARGET|focus=SHALLOW_DOF",
  "gp_mean": -0.42,
  "gp_uncertainty": 0.31,
  "last_failure_type": "missing_action_target"
}
```

Locked enums:
- are fixed during coordinate generation.
- still receive reliability evidence.
- must not be replaced by LHS for the current run.

Unlocked/sampleable enums:
- can be selected by LHS.
- receive Thompson alpha/beta updates.
- can be temporarily suppressed after repeated failure.

Suppression and quarantine are not direct products of a raw Thompson posterior cutoff. Thompson and GP state provide statistical evidence, but orchestration actions require a policy signal that combines:
- evaluator failure types.
- seed-sweep aggregate outcomes.
- per-arm Thompson posterior and observation count.
- GP combo affinity and uncertainty.
- recurrence across seeds, coordinates, or prompt clusters.
- confidence that the failure is semantic, quality, or guardrail-related rather than infrastructure.
- lock state and user-authored provenance.

A low posterior says an enum has not performed well. It does not by itself explain whether the enum is out-of-distribution, unlucky, harmed by a particular combo, hurt by renderer wording, or affected by evaluator noise.

**7. LHS Coordinate Generation**
The orchestrator requests coordinates from the LHS router in batches.

Each coordinate contains:
- rendered prompt.
- fixed arms.
- sampled arms.
- compatibility trace.
- Bayesian score.
- coordinate ID.
- seed bundle assignment.

The orchestrator must not send a coordinate to the active fast generator unless:
- PromptDocument verification passed.
- lock configuration is resolved.
- compatibility hard rejects are absent.
- sampleable fields have non-empty eligible domains.
- coordinate is not retired or suppressed.
- render output is non-empty.

**8. Seed-Sweep Bundle Contract**
Every LHS coordinate runs a seed-sweep bundle. The default bundle is:

```json
[7, 42, 156, 8888, 42069]
```

Runs may override the bundle via `RunConfig.seed_bundle`; the minimum bundle size is `MIN_SEED_BUNDLE_SIZE = 3` (see `generation.py` and `gates.py`).

Purpose:
- reduce false negatives from a bad random seed.
- reduce false positives from a lucky seed.
- estimate coordinate robustness.
- decide whether seed surfing is worthwhile.

The orchestrator persists one candidate row per seed and one aggregate row per coordinate.

Coordinate outcomes:

`strong`: multiple seeds pass IQA and VLM evaluation. Coordinate is eligible for seed surfing. The absolute threshold is `promoted_count >= 3` (see `evaluation._outcome`).

`viable`: at least one seed passes, but robustness is not high. The absolute threshold is `promoted_count > 1`, which after the strong guard resolves to `promoted_count == 2`.

`fragile`: exactly one seed passes or failures suggest seed sensitivity. The absolute threshold is `promoted_count == 1`.

`failed`: no seeds pass for semantic or quality reasons. The absolute threshold is `promoted_count == 0`.

`blocked`: infrastructure failure prevents judgment. Any infrastructure-only failure type (`invalid_image_file`, `evaluator_unavailable`, `evaluator_timeout`, `evaluator_malformed_output`, `gpu_memory_failure`) is treated as `blocked` and does not produce a semantic penalty.

**9. Seed Surfing**
Seed surfing is an optional continuation for strong coordinates.

Trigger:
- at least a configured number of seeds in the preliminary bundle pass all required evaluations.
- default: `3` of `5`.

Action:
- enqueue an additional 5-random-seed bundle for the same coordinate.
- keep the same rendered prompt and enum coordinate.
- record the surf bundle as a child of the original coordinate.

Stop surfing when:
- pass rate drops below configured continuation threshold.
- promoted/curated high watermark is reached if configured.
- coordinate is automatically retired.
- run stops or stall guard triggers.

Seed surfing rewards robust prompt variations without allowing a single lucky seed to dominate.

**10. Fast Image Generation Stage**
The orchestrator sends each seed request to the resident fast image generation backend. The generator is model-agnostic; the active builder comes from `GENERATOR_REGISTRY` (see `src/bruteforce_canvas/generator_registry.py`).

Generation input:
- rendered prompt.
- seed.
- steps.
- height.
- width.
- coordinate ID.
- run ID.

Generation output:
- image path.
- generation settings.
- generation status.
- infrastructure errors if any.

The orchestrator must persist valid generated image artifacts before evaluation. If generation fails for infrastructure reasons, the failure must not penalize enum arms or combinations.

**11. Evaluation Routing**
Evaluation is staged for latency.

Stage 1: IQA batch evaluation.
- run first on generated images.
- use GPU batch inference where possible.
- default scorer class: IQA encoder such as JoyQuality.
- cutoff default: `0.55`, parameterized per run.
- images below cutoff do not continue to VLM alignment evaluation.
- failures persist for learning and calibration unless the file is invalid.

Stage 2: VLM alignment and guardrail evaluation.
- receives only IQA survivors.
- may run sequentially or in small batches depending on model and VRAM.
- reference evaluator: MiniCPM-V 4.6 through the local alignment adapter. Production real-inference runs use local weights and fail closed when the required inference device or weights are unavailable.
- checks prompt alignment and guardrails.
- guardrail examples: no human/animal deformity, no severe artifacts, no missing required target, no negative constraint violation.
- emits failure types and blame hints.

**11.1 Outcome Thresholds**

The outcome labels emitted by `evaluation._outcome` are bound to absolute counts of promoted images per coordinate:

- `>= 3` promoted: `strong`.
- `> 1` promoted (resolves to `== 2` after the strong guard): `viable`.
- `== 1` promoted: `fragile`.
- `0` promoted: `failed`.
- any infrastructure-only failure type: `blocked`.

These thresholds are the single source of truth for the coordinate outcome. The aggregate labels listed in §8 are the natural-language rendering of the same thresholds.

Stage 3: optional metacognitive impact scoring.
- enabled only on configured hardware, default `24GB+` VRAM.
- disabled on the 16 GB base tier unless capacity and policy explicitly opt in.
- receives only images that survive IQA and VLM evaluation.
- produces neural/cognitive impact scores and heatmap-style metadata when supported.
- does not change eligibility for images that failed IQA or VLM gates.

This staged routing prevents expensive VLM or impact evaluation from running on images that already failed low-latency quality screening.

**12. Post-Evaluation Promotion**
An image is promoted/curated when:
- generation succeeded.
- image artifact is valid.
- IQA score meets run cutoff.
- VLM alignment score meets run cutoff.
- required guardrails pass.
- negative constraints are not violated.

Promoted/curated images:
- count toward run progress.
- are available for future fine-tuning datasets.
- are the positive pool for future JoyQuality fine-tuning or calibration because they survived both IQA and VLM evaluation.
- produce positive learning signals for all selected coordinate enums unless configured otherwise.

Non-promoted valid images:
- remain persisted as failure evidence.
- are eligible for learning and calibration.
- do not count toward promoted/curated progress.

Invalid artifacts:
- may be hard-purge eligible.
- should leave infrastructure telemetry.
- should not produce semantic penalties.

**13. Learning Update Policy**
The orchestrator applies learning updates after a coordinate's seed bundle aggregate is available.

Passing images:
- increase Thompson alpha for sampled enum arms.
- increase reliability evidence for locked arms.
- add positive GP affinity for the coordinate combination.

VLM guardrail or alignment failures:
- increase Thompson beta for sampled enum arms in the failed coordinate.
- add negative GP affinity for the enum combination.
- add reliability evidence for locked arms without making them LHS-replaceable.

IQA failures:
- can penalize sampled visual presentation arms and combinations when failure is likely prompt-coordinate related.
- should be weaker than VLM semantic failure unless failure is repeated across multiple seeds.

Infrastructure failures:
- do not affect Thompson or GP semantic state.
- route to retry or operational telemetry.

Because low-latency VLM evaluation may not identify the exact problematic enum, the default negative update can be distributed across the sampled coordinate arms and the full combination. Good generations that pass guardrails and alignment reward all selected arms. Bad generations that fail guardrails or alignment penalize the sampled arms and the combination together.

This is intentionally conservative. More localized blame can be used when evaluator evidence is strong, but the orchestrator must not require expensive per-enum VLM interrogation to operate.

Learning updates and lifecycle actions are separate. A failed coordinate can update Thompson/GP state without immediately suppressing an enum or quarantining a combo. Suppression and quarantine require the signal policy in sections 14 and 15 to pass.

**14. Coordinate Lifecycle**
Coordinate states:

`proposed`: generated by LHS, not yet rendered.

`rendered`: has a prompt string and seed bundle.

`generating`: resident fast image generation in progress.

`evaluating_iqa`: IQA batch evaluation in progress.

`evaluating_vlm`: VLM evaluation in progress.

`evaluating_impact`: optional impact evaluation in progress.

`strong`: seed bundle passed strongly and may seed surf.

`viable`: at least one seed passed.

`fragile`: weak pass behavior; do not seed surf by default.

`retired`: coordinate is no longer selected for this run.

`quarantined`: coordinate or combo is held out because failures are repeated or severe.

`blocked`: infrastructure failure prevented judgment.

Lifecycle actions:
- strong coordinates can enqueue random seed surf bundles.
- viable coordinates can remain available but are not automatically surfed unless configured.
- fragile coordinates receive weak or mixed learning updates.
- failed coordinates are retired.
- quarantined coordinates are excluded until cooldown or policy release.

Coordinate retirement is local and cheap: the exact coordinate stops receiving more seed budget. Quarantine is stronger: the coordinate or combination is held out because repeated, high-confidence failures indicate that similar selections are unlikely to be productive.

Quarantine trigger policy:
- seed-sweep pass rate is below the configured threshold.
- failures are consistent across seeds or seed-surf bundles.
- failure types are semantic, guardrail, or severe quality failures.
- GP combo affinity is below the configured quarantine floor with enough observations.
- evaluator confidence is high enough.
- infrastructure-only failure is false.

Default coordinate quarantine thresholds:
- the seed-sweep bundle has `0` passes, or pass rate below `0.20` across at least `10` evaluated seeds including seed surfing.
- same failure family appears in at least `60%` of evaluated images.
- GP combo affinity below `-0.35` after minimum observations.

Coordinate retirement trigger policy:
- exact coordinate has no remaining allocated seed bundles.
- preliminary bundle is `failed` or `fragile`.
- coordinate is not strong enough for seed surfing.
- coordinate does not require quarantine because evidence is weak, sparse, or localized only to that exact variant.

**15. Enum Suppression**
Enum suppression is temporary and context-aware.

Suppress an enum arm when:
- it repeatedly appears in failed coordinates.
- failures are post-evaluation semantic or guardrail failures.
- failures persist across multiple seeds or multiple coordinates.
- infrastructure is not the cause.
- enough observations exist to distinguish recurrent failure from seed noise.
- posterior pass probability is below the configured suppression floor.
- recurrence and evaluator confidence meet policy thresholds.
- the enum is not user-authored and locked for the current prompt.

Suppression state:

```json
{
  "field_path": "lighting_mood",
  "enum_value": "BLUE_HOUR",
  "context_key": "bonsai_ternary_4b",
  "reason": "alignment_below_cutoff across repeated coordinates",
  "suppressed_until": "cooldown:500_generated_candidates",
  "min_exploration_probability": 0.01
}
```

The orchestrator should keep an exploration floor for non-safety enum arms so recovery remains possible after model, renderer, or threshold changes.

Suppression trigger policy:
- evaluator failures identify a stable failure family, such as `wrong_lighting`, `missing_locked_element`, `bad_anatomy`, `negative_constraint_violation`, or `alignment_below_cutoff`.
- seed-sweep aggregation confirms the issue across multiple seeds or coordinates.
- Thompson posterior mean is below the configured pass-rate floor with sufficient sample count.
- GP evidence does not indicate the issue is only a narrow combo failure.
- suppression would not erase explicit user intent.

Default enum suppression thresholds:
- at least `10` semantic observations for the enum arm in the active model/context.
- posterior mean pass rate below `0.20`.
- same failure family in at least `60%` of failures.
- evaluator confidence medium or high.
- cooldown of `500` generated candidates.
- exploration floor of `1%` for non-safety enum arms.

Hard suppression is reserved for safety, policy, or high-confidence negative-constraint violations. Otherwise suppression is temporary and recoverable.

Locked enum handling:
- if the user explicitly authored the enum, do not suppress it as a replacement for that prompt.
- mark `matched_suppressed` or `locked_reliability_warning`.
- preserve the raw phrase.
- emit a renderer phrase diagnostic, such as raw-preserving wording that may reduce model confusion.
- keep reliability evidence separate from LHS replaceability.

**16. Stall Guard**
The orchestrator must stop runs that are not producing enough promoted/curated images.

Default:
- stall window: `30` minutes.
- minimum promoted/curated images: `10`.

Trigger:
- if fewer than `10` images have passed required post-evaluation cutoffs after `30` minutes, stop the run.

Reason:
- prompt may be out of the active generator model's training distribution.
- user concept may be poorly trained or damaged by quantization.
- IQA cutoff may be too high.
- alignment or guardrail cutoff may be too high.
- prompt may need rewording.
- enum search space may be poor.

The stall result should persist:
- elapsed time.
- generated count.
- IQA pass count.
- VLM pass count.
- promoted/curated count.
- dominant failure types.
- most penalized enum arms/combinations.
- suggested machine-readable restart hints.

Restart hints should include whether the next run should consider:
- lowering the IQA cutoff from the current run value.
- lowering alignment or guardrail cutoffs.
- rehashing or clarifying the user prompt.
- locking fewer presentation enums.
- narrowing the LHS enum space.
- disabling optional impact scoring.

The orchestrator should not silently lower thresholds during the same run. Threshold changes belong to a new run configuration or user/system restart flow.

**17. Threshold and Watermark Policy**
Thresholds are run-local and persisted.

Default thresholds:
- IQA aesthetic cutoff: `0.55`.
- prompt-alignment cutoff: deployment configured value.
- human/animal guardrail: enabled when detected.
- optional impact cutoff: disabled unless configured.

Watermarks:
- generated image count is not a success condition.
- promoted/curated image count is the progress count.
- high watermark may pause a run if configured.
- low watermark may resume a paused run if configured.
- continuous mode may leave high watermark unset and run until stop or stall.

This lets the engine support both bulk dataset creation and bounded resident-buffer workflows.

**18. Persistence Contract**
Persist all artifacts needed for audit and learning.

Run:
- raw prompt.
- run configuration.
- thresholds.
- model versions.
- lock configuration.
- PromptDocument ID.
- verifier result.

Coordinate:
- coordinate ID.
- rendered prompt.
- fixed arms.
- sampled arms.
- compatibility trace.
- Bayesian score.
- parent coordinate ID for seed surfing.
- lifecycle state.

Seed candidate:
- candidate ID.
- coordinate ID.
- seed.
- image path.
- generation status.
- generation settings.
- IQA result.
- VLM result.
- optional impact result.
- promoted/curated flags.
- failure types.
- disposition action applied.

Learning:
- Thompson alpha/beta deltas.
- GP combo deltas.
- suppression records.
- reliability updates for locked arms.
- swipe/thumb feedback deltas when available.

Telemetry:
- generated count.
- IQA evaluated count.
- IQA survivor count.
- VLM evaluated count.
- VLM survivor count.
- promoted/curated count.
- throughput per stage.
- stall guard state.
- VRAM and error telemetry.

**19. System Actions**
The orchestrator applies these actions from evaluator and aggregate results.

`persist_for_learning`:
- keep image and result payload.
- use failure evidence for calibration and sampler updates.

`promote_curate`:
- mark candidate as promoted/curated.
- count toward run progress.
- add positive learning evidence.

`demote_candidate`:
- remove promoted/curated status.
- retain candidate for negative evidence unless artifact is invalid.

`retire_coordinate`:
- stop generating more seeds for a weak or failed coordinate.
- keep coordinate history for GP updates.

`quarantine_coordinate`:
- hold coordinate out for a longer cooldown after repeated severe failures.

`suppress_enum_arm`:
- temporarily lower or block selection probability for a sampled enum arm in context.

`hard_purge_invalid_artifact`:
- remove or mark invalid/corrupt image artifact according to storage policy.
- preserve failure telemetry.
- do not apply semantic penalties.

`infrastructure_retry`:
- retry generation/evaluation when failure is operational.
- do not penalize prompt, enum, or coordinate.

**20. Pre-Curated Feedback Integration**
Automated evaluation is the default learning signal. Human feedback is limited to swipe/thumb curation on images that already reached the pre-curated pool through automated evaluation. The backend supports only `accept`, `reject`, and `shred` signals.

Allowed feedback actions:

`accept`:
- confirms that a pre-curated image meets the user's quality bar.
- keeps promoted/curated state.
- adds positive evidence to the candidate, coordinate, sampled enum arms, and enum combination.
- strengthens future seed surfing for the coordinate when aggregate performance is already strong.

`reject`:
- marks the pre-curated image as a false positive.
- removes curated state and may remove promoted state depending on storage policy.
- persists the image and evaluator payload for learning and calibration.
- adds negative evidence to the coordinate and enum combination.
- adds weak or medium negative evidence to sampled enum arms.
- contributes to quarantine or suppression only through repeated patterns.

`shred`:
- marks the pre-curated image as a severe false positive.
- removes curated and promoted state.
- persists metadata and, when storage policy allows, keeps the artifact for evaluator calibration.
- adds stronger negative evidence to the coordinate and enum combination.
- contributes to coordinate quarantine or enum suppression only through repeated patterns.

Feedback only tightens the acceptance bar for images that automated evaluation already considered passable.

The orchestrator automatically maps feedback into downstream effects:
- candidate status changes.
- coordinate reward or penalty.
- Thompson alpha/beta deltas for sampled arms.
- GP combo affinity deltas.
- evaluator false-positive calibration records.
- suppression/quarantine counters when repeated feedback patterns occur.

Feedback source must be persisted separately from automated evaluator evidence:

```json
{
  "candidate_id": 123,
  "coordinate_id": "coord_001842",
  "feedback_action": "reject",
  "feedback_scope": "pre_curated_candidate",
  "signal_source": "swipe_feedback",
  "automated_status": "promoted_curated",
  "effective_status": "demoted_false_positive"
}
```

The orchestrator must record whether a learning update came from automated evaluation, swipe/thumb feedback, or both.

**21. Hardware Tiers**
Base tier:
- fast image generation.
- IQA batch evaluation.
- VLM guardrail/alignment evaluation.
- no optional metacognitive impact scoring by default.

24GB+ tier:
- may enable optional metacognitive impact evaluation.
- may batch larger evaluator workloads.
- may keep more evaluator models resident.

The orchestrator should schedule by hardware tier and avoid loading optional evaluators that would reduce generation/evaluation throughput below acceptable limits.

**22. Failure Handling**
Infrastructure failure:
- retry if transient.
- mark blocked if persistent.
- no semantic penalty.

IQA failure:
- persist if valid.
- do not send to VLM.
- apply weak or configured quality-related learning update.

VLM guardrail failure:
- persist.
- demote or prevent promotion.
- apply negative enum/combination update.

VLM prompt-alignment failure:
- persist.
- demote or prevent promotion.
- apply negative enum/combination update.

Impact failure:
- do not promote for impact ranking.
- do not demote quality/alignment-passing image unless product policy requires it.

Invalid image artifact:
- hard-purge or mark invalid.
- do not apply semantic penalty.

**23. Orchestration Pseudocode**
```text
start_run(raw_prompt, run_config):
  document = prompt_pipeline.extract_canonicalize_verify(raw_prompt)
  lock_config = build_default_lock_config(document)
  effective_config = apply_system_or_user_lock_config(lock_config, run_config)
  balancer = load_bayesian_state(context=document.context_key)

  while not stop_requested:
    if stall_guard_tripped(run):
      stop_run("stall_guard")
      break

    if promoted_high_watermark_reached(run_config):
      pause_until_low_watermark_or_stop()
      continue

    coordinates = lhs_router.propose_batch(document, effective_config, balancer)

    for coordinate in coordinates:
      if coordinate_retired_or_suppressed(coordinate):
        continue

      seed_results = []
      for seed in run_config.seed_bundle:
        candidate = image_generator.generate(coordinate.rendered_prompt, seed)
        persist_candidate_generation(candidate)
        seed_results.append(candidate)

      iqa_results = evaluator.run_iqa_batch(seed_results)
      iqa_survivors = persist_and_filter_iqa(iqa_results, cutoff=run_config.iqa_cutoff)

      vlm_results = evaluator.run_vlm_alignment_and_guardrails(iqa_survivors)
      promoted = apply_vlm_results(vlm_results, run_config)

      if run_config.metacognitive_impact_enabled:
        impact_results = evaluator.run_impact_batch(promoted)
        persist_impact_results(impact_results)

      aggregate = aggregate_coordinate_results(coordinate, seed_results)
      actions = decide_actions(aggregate)
      apply_actions(actions)
      balancer.update(aggregate, actions)

      if aggregate.outcome == "strong":
        enqueue_seed_surf_bundle(coordinate)
```

**24. Open Product Parameters**
These are parameters, not blockers:
- exact IQA cutoff per deployment.
- exact alignment cutoff per VLM.
- human/animal guardrail strictness.
- seed-surfing continuation threshold.
- enum suppression cooldown length.
- high/low promoted watermark values.
- metacognitive-impact cutoff and licensing policy.
- whether IQA failures apply weak negative enum updates or only coordinate-level quality telemetry.

The orchestrator must support these as configuration values.

**25. Implementation Contract**

The orchestration, gating, and learning state expose these contract surfaces:

- `src/bruteforce_canvas/orchestration.py` (`RunConfig`, `RunCounters`, `RuntimeGateState`, `RunRuntimeState`, `stall_guard_decision`, `watermark_decision`).
- `src/bruteforce_canvas/gates.py` (`StageGate.prompt`, `.router`, `.rendering`, `.generation`, `.evaluation` enforce handoff gates with absolute seed-bundle and outcome-threshold checks).
- `src/bruteforce_canvas/run_service.py` (`RunService` with `_run_gate_chain` and `tick`; coordinates the orchestrator's interaction with the generator, evaluator, and persistence layers).
- `src/bruteforce_canvas/loop.py` (`AsyncRunDriver`, `next_loop_action`, `LoopAction`, `LoopDecision`).
- `src/bruteforce_canvas/balancer.py` (`BayesianBalancer` wrapper around Thompson and GP state with prompt-cluster or model-family context keys).
- `src/bruteforce_canvas/transport.py` (`EventBus` plus in-process `asyncio.Queue` pubsub for CLI SSE).
- `src/bruteforce_canvas/cli.py` (subcommands `render-workspace` and `stream`; the CLI is the canonical user-facing entry point and is documented in Spec 06).
- `src/bruteforce_canvas/spec_compliance.py` (`check_all`, `check_spec_01..06`, with `phases="A-L"` when the implementation is complete and the spec body matches it).

**26. Summary Requirement**
The orchestration engine continuously transforms one verified user prompt into many LHS-routed prompt coordinates, generates a seed-sweep bundle per coordinate with the active fast image generator, evaluates images through staged IQA and VLM gates, optionally scores metacognitive impact on capable hardware, persists all candidates and evaluation evidence, promotes only images that pass required cutoffs, accepts swipe/thumb feedback only on pre-curated images, and updates Thompson Sampling plus Gaussian Process state so weak enum arms and weak combinations become less likely. It stops unproductive runs through a parameterized stall guard.
