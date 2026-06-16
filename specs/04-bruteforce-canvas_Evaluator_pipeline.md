**Low-Latency Batch Image Evaluator Specification**
**Companion to `bruteforce-canvas_DAG_prompt.md`, `bruteforce-canvas_LHS_enum_router.md`, `bruteforce-canvas_fast_image_generation.md`, and `bruteforce-canvas_Orchestration.md`**

Status: implementation specification

Audience: evaluator worker, resident GPU worker, prompt pipeline, LHS router, and orchestration implementers

Primary goal: define the evaluator-facing contract for fast image scoring after fast image generation. The evaluator determines whether generated images meet quality, prompt-alignment, and optional metacognitive-impact thresholds, emits failure types and blame hints for learning, and returns structured disposition signals. The evaluator does not execute purge, demotion, quarantine, enum suppression, or retry actions; those actions belong to the orchestrator.

Build-order position: implement this spec after `bruteforce-canvas_fast_image_generation.md`. It consumes generated candidates, target manifests, rendered prompts, coordinate metadata, run-local thresholds, and seed-sweep groupings. Shared lifecycle, promotion, feedback, suppression, quarantine, retry, and persistence terms are defined in `bruteforce-canvas_Orchestration.md` and must not be renamed here.

**1. Scope**
The evaluator sits after image generation and before downstream learning/orchestration decisions.

Input:
- generated image paths or tensors.
- rendered prompt.
- LHS coordinate metadata.
- PromptDocument-derived evaluation target manifest.
- seed sweep grouping.
- evaluator model configuration.

Output:
- per-image evaluation scores.
- per-coordinate seed-sweep aggregate scores.
- pass/fail threshold decisions.
- failure types.
- localized blame hints.
- confidence and uncertainty metadata.
- disposition signals for the orchestrator.

Non-role:
- does not parse the raw user prompt.
- does not rewrite prompt strings.
- does not choose LHS enum values.
- does not mutate locked fields.
- does not apply Thompson Sampling or GP updates directly.
- does not demote, purge, quarantine, or retry by itself.

The evaluator is a measurement and labeling subsystem. The orchestrator consumes its output and decides what to do.

**2. Why Evaluation Exists**
Evaluation is required because generation success is not equivalent to writing a valid PNG. A generated image can be technically coherent while violating the prompt, aligned but low-quality, visually striking but unsafe for the current prompt constraints, or caused by a lucky seed rather than a robust prompt coordinate.

The evaluator produces evidence for these downstream decisions:
- fail but persist for learning.
- mark a candidate as passing post-evaluation thresholds.
- demote from promoted/curated status.
- quarantine a prompt coordinate.
- suppress an enum arm temporarily.
- hard purge an invalid or corrupt image artifact.
- retry infrastructure failures with no semantic penalty.

Those are downstream disposition signals, not evaluator-side actions. The evaluator returns the structured facts and suggested disposition class; the orchestrator performs state changes.

**3. Required Fields vs Priority Status**
A separate `required_*` namespace is not mandatory if every evaluator target carries explicit priority, provenance, and mutability fields.

Preferred representation:

```json
{
  "target_id": "object_01.material",
  "target_kind": "object_attribute",
  "value_raw": "ceramic",
  "enum_value": "CERAMIC",
  "priority": "locked_required",
  "provenance": "explicit_user_prompt",
  "lhs_policy": "fixed",
  "evaluation_policy": "must_match",
  "blame_policy": "eligible_for_reliability_update"
}
```

This avoids duplicating the same information as both `required_elements` and ordinary fields. The evaluator can derive required views from priority.

Recommended priority values:

`locked_required`: user-stated or verifier-required field. Must be preserved. LHS cannot replace it.

`locked_context`: user-stated context that should normally be preserved, but may not be a hard failure if visually absent.

`important`: strongly relevant to scoring but not a hard fail by itself.

`sampled`: LHS-selected detail. Can be blamed, downranked, or replaced in future coordinates.

`optional`: useful if present, not a failure if absent.

`negative_guard`: forbidden content or style constraint. Violations are hard failures when confidence is high.

`diagnostic`: logged for analysis, not used for pass/fail decisions.

Recommended LHS policy values:

`fixed`: do not randomize.

`sampleable`: eligible for LHS variation.

`sampleable_if_missing`: eligible only when absent from the user prompt.

`blocked`: cannot be varied because it would invent, repair, or contradict semantic facts.

Recommended evaluation policy values:

`must_match`: high-confidence mismatch fails the image.

`should_match`: mismatch lowers score or emits warning.

`nice_to_have`: absence is logged only.

`must_not_appear`: high-confidence presence fails the image.

`measure_only`: no direct pass/fail effect.

The parser and verifier should assign these priorities before LHS. The evaluator should consume them, not infer them from prose.

**4. Evaluator Target Manifest**
The evaluator target manifest is derived from the verified PromptDocument, canonicalization metadata, LHS coordinate, and renderer trace.

Recommended shape:

```json
{
  "manifest_id": "eval_manifest_001842",
  "run_id": "run_00091",
  "prompt_document_id": "doc_017",
  "coordinate_id": "coord_001842",
  "rendered_prompt": "Generate red ceramic bowl on wooden table, medium shot from eye level, 50mm lens, shallow depth of field, warm light",
  "targets": [
    {
      "target_id": "object_01",
      "target_kind": "element",
      "label": "bowl",
      "priority": "locked_required",
      "lhs_policy": "fixed",
      "evaluation_policy": "must_match",
      "evidence": "red ceramic bowl"
    },
    {
      "target_id": "rel_01",
      "target_kind": "relation",
      "source_id": "object_01",
      "relation_raw": "on",
      "enum_value": "ON_TOP_OF",
      "target_id_ref": "object_02",
      "priority": "locked_required",
      "lhs_policy": "fixed",
      "evaluation_policy": "must_match"
    },
    {
      "target_id": "cinematography.shot_size",
      "target_kind": "cinematography",
      "enum_value": "MEDIUM_SHOT",
      "priority": "sampled",
      "lhs_policy": "sampleable",
      "evaluation_policy": "should_match"
    }
  ],
  "negative_targets": [
    {
      "target_id": "constraint.no_extra_people",
      "target_kind": "constraint",
      "value_raw": "no extra people",
      "priority": "negative_guard",
      "evaluation_policy": "must_not_appear"
    }
  ]
}
```

This manifest is the bridge between prompt assembly, LHS randomization, generation, and evaluation. It lets the evaluator score prompt adherence without re-parsing the user's original prompt.

**5. Primary Evaluation Aspects**
The evaluator measures three aspects before model-specific routing is selected.

**5.1 Aesthetic and Technical Image Quality**
Question: is the image visually usable?

Measures:
- general aesthetic quality.
- sharpness and resolution sufficiency.
- visible artifacts.
- composition quality.
- exposure, contrast, and color handling.
- anatomy/hand/face issues when humans are visible.
- product surface integrity when products are visible.
- corrupt or invalid file detection.

Expected pass behavior:
- image is technically coherent.
- important subjects are not malformed.
- artifacts do not dominate the intended visual.
- quality score meets the active cutoff.

Expected fail behavior:
- severe blur, corruption, broken anatomy, heavy artifacts, incoherent forms, or unusable lighting.
- score falls below the active quality cutoff.

**5.2 Prompt Instruction-to-Image Alignment**
Question: does the image match the rendered prompt and evaluator target manifest?

Measures:
- required element presence.
- required relation presence.
- required action or pose target preservation.
- object attributes such as color, material, finish, and condition.
- cinematography fields such as shot size, camera angle, lens feel, lighting, focus, and style.
- negative constraint violations.
- semantic drift or invented major objects.

Expected pass behavior:
- `locked_required` targets are visually present or adequately represented.
- sampled fields are reasonably expressed when they affect visible output.
- negative guards are not violated.
- alignment score meets the active cutoff.

Expected fail behavior:
- required element missing.
- required relation/action target missing.
- explicit color/material/lighting contradicted.
- sampled field causes prompt drift.
- negative guard violated.
- alignment score falls below the active cutoff.

**5.3 Optional Metacognitive Impact**
Question: is the image likely to have strong human perceptual, cognitive, or attention impact beyond basic correctness?

This is optional and must not be a base promotion requirement unless the product explicitly enables it.

The TRIBE v2 family is the reference model class for this aspect. It is described as a foundation model over vision, audition, and language for in-silico neuroscience, predicting human brain activity for naturalistic and experimental stimuli.

Measures should be treated as aggregate impact proxies, not truth labels:
- predicted neural-response strength.
- region- or feature-level response concentration.
- cross-modal salience when text/audio/video context exists.
- novelty or memorability proxy.
- attention-capture proxy.
- high-level affective or cognitive engagement proxy.

Expected pass behavior:
- impact score meets the product's optional cutoff.
- image remains prompt-aligned and technically acceptable.
- impact does not come from violating constraints or producing artifacts.

Expected fail behavior:
- low impact score for ranker-enabled workflows.
- high impact caused by defects, shock, constraint violations, or prompt drift.
- model unavailable or not licensed for the deployment context.

TRIBE-style scoring is optional and must remain downstream of quality and alignment. A high metacognitive-impact score cannot make an image eligible when it fails required quality or alignment gates.

**6. Evaluator Model Families**
Evaluator responsibilities should be routed by measurement type.

Evaluator models are local role adapters in the base product contract. The prompt-LLM OpenAI-compatible server adapter is not an evaluator adapter and must not be used to route IQA, VLM alignment, or metacognitive-impact scoring.

**6.1 IQA Encoder Models**
Best fit:
- aesthetic and technical quality.
- artifact detection.
- quick post-generation filtering.
- batch scoring many images with low latency.

Expected output:
- scalar quality score.
- optional artifact score.
- optional human/product quality flags.
- confidence.

Reference adapter:
- [JoyQuality SigLIP2 SO400M](https://huggingface.co/fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn) is the local IQA model for quality scoring. The baseline quality cutoff is `0.55`. Production use requires calibration approval and legal clearance for the model license posture.

**6.2 Small VLMs**
Best fit:
- prompt instruction-to-image alignment.
- object/relation/action verification.
- negative-constraint checks.
- structured JSON failure reports.
- human detection and visible artifact commentary.

Expected output:
- alignment score.
- per-target status.
- failure types.
- localized blame hints.
- confidence.
- critique text for logging, not direct decision-making.

Reference adapter:
- [MiniCPM-V 4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) is the local VLM for image-prompt alignment. It is the prompt-alignment evaluator for real inference runs and maps visual/question-answer output into the stable alignment score, failure types, blame hints, and confidence schema.

**6.3 TRIBE v2 Aggregate Impact Models**
Best fit:
- optional metacognitive-impact ranking.
- neuroscience-inspired salience and engagement estimation.
- research or non-commercial ranking experiments.

Expected output:
- scalar impact score.
- optional per-region or latent feature summaries.
- confidence or availability status.

Reference adapter:
- The TRIBE v2 family is consumed through the lite-qv variant, [Jessylg27/tribev2-lite-qv](https://huggingface.co/Jessylg27/tribev2-lite-qv). It is optional, informational-only by default, disabled on the 16 GB base tier, and must be treated as non-commercial unless legal approval says otherwise.

**7. Cutoff Semantics**
Every evaluator score must define what meeting or missing the cutoff means.

Quality cutoff:
- pass means the image is visually usable enough for the current workflow.
- fail means the image should not be promoted on technical/aesthetic grounds.
- failure may still persist for learning if the file is valid.

Alignment cutoff:
- pass means the image sufficiently matches the rendered prompt and priority-weighted target manifest.
- fail means the image does not satisfy prompt intent strongly enough.
- localized failures should identify target IDs where possible.

Metacognitive-impact cutoff:
- pass means the image is strong enough for optional impact ranking.
- fail means the image may remain valid but should not receive impact-ranking boost.
- unavailable or unlicensed means no impact decision should be made.

Corruption cutoff:
- pass means the file is readable, valid, and contains an image.
- fail means the artifact is invalid and may be hard-purge eligible.

Evaluator confidence:
- high confidence may drive hard gate signals.
- medium confidence may drive warnings and weak learning updates.
- low confidence should avoid strong semantic penalties.

**8. Batch Units**
The evaluator supports two separate batch concepts.

Semantic batch unit:
- a seed-sweep bundle (default 5 seeds; minimum 3) for one LHS coordinate.
- all five images share coordinate ID, rendered prompt, target manifest, fixed arms, sampled arms, and compatibility trace.
- aggregate result updates coordinate viability using the absolute thresholds in `evaluation._outcome`.

Performance batch unit:
- a tensor/model batch optimized for the evaluator model.
- may contain images from one coordinate or many coordinates.
- must preserve per-image identity and coordinate grouping.

The evaluator contract must not confuse these. A model may process 16 images in one tensor batch, but the learning system still needs results grouped by the seed-sweep bundle per coordinate.

**9. Serialized vs Parallel Evaluation**
Some evaluators are individual but serialized as one unit batch. Others can run in parallel as one unit batch.

Serialized single-unit batch:
- each image is evaluated one at a time by a model whose API or memory profile makes true batching impractical.
- the enclosing request still represents one seed sweep or one logical image batch.
- output must preserve per-image order and IDs.

Parallel single-unit batch:
- multiple images are passed to the same model call or batched tensor path.
- output returns one result per image.
- failures in one sample must not corrupt other sample results.

Mixed evaluator pipeline:
- IQA encoder runs image tensor batches.
- VLM alignment may run smaller parallel batches or serialized calls.
- optional TRIBE-style impact scoring may run as a separate ranker batch.

Recommended rule:
- keep the semantic request shape stable.
- let evaluator adapters choose serialized or parallel execution internally.
- report execution mode in the result metadata.

**10. Evaluation Request Shape**
Recommended top-level request:

```python
class EvaluationBatchRequest(BaseModel):
    batch_id: str
    run_id: str
    prompt_document_id: str
    target_manifest_id: str
    batch_kind: Literal["seed_sweep", "mixed_image_batch"]
    coordinate_id: str | None
    rendered_prompt: str
    target_manifest: EvaluationTargetManifest
    images: list[EvaluationImageInput]
    evaluator_plan: EvaluationPlan
```

Image input:

```python
class EvaluationImageInput(BaseModel):
    candidate_id: int | None
    image_path: str
    seed: int
    coordinate_id: str
    run_id: str
    prompt_document_id: str
    target_manifest_id: str
    generation_settings: dict
```

Evaluator plan:

```python
class EvaluationPlan(BaseModel):
    quality: bool = True
    alignment: bool = True
    metacognitive_impact: bool = False
    quality_cutoff: float
    alignment_cutoff: float
    human_quality_cutoff: float | None = None
    impact_cutoff: float | None = None
    execution_preference: Literal["auto", "serialized", "parallel"] = "auto"
```

**11. Evaluation Result Shape**
Per-image result:

```python
class ImageEvaluationResult(BaseModel):
    candidate_id: int | None
    image_path: str
    seed: int
    coordinate_id: str
    run_id: str
    prompt_document_id: str
    target_manifest_id: str
    file_valid: bool
    quality: QualityEvaluation
    alignment: AlignmentEvaluation
    impact: ImpactEvaluation | None = None
    pass_flags: EvaluationPassFlags
    failure_types: list[FailureType]
    localized_blame: list[BlameHint]
    disposition_signal: DispositionSignal
    confidence: Literal["high", "medium", "low"]
```

Coordinate aggregate result:

```python
class CoordinateEvaluationAggregate(BaseModel):
    run_id: str
    prompt_document_id: str
    target_manifest_id: str
    coordinate_id: str
    seeds: list[int]  # configurable per run; default DEFAULT_SEED_BUNDLE, minimum MIN_SEED_BUNDLE_SIZE
    generated_count: int
    evaluated_count: int
    promoted_count: int
    quality_pass_count: int
    alignment_pass_count: int
    full_pass_count: int
    mean_quality: float
    mean_alignment: float
    best_quality: float
    best_alignment: float
    pass_rate: float
    outcome: Literal["strong", "viable", "fragile", "failed", "blocked"]
    aggregate_failure_types: list[FailureType]
    aggregate_blame: list[BlameHint]
    update_signal: LearningUpdateSignal
```

**12. Failure Types**
Failure types should be stable strings so the LHS router and orchestrator can consume them.

File and infrastructure:
- `invalid_image_file`
- `image_decode_failed`
- `evaluator_unavailable`
- `evaluator_timeout`
- `evaluator_malformed_output`
- `gpu_memory_failure`

Technical quality:
- `quality_below_cutoff`
- `blur_or_low_detail`
- `severe_artifact`
- `bad_anatomy`
- `bad_hands`
- `bad_face`
- `deformed_product`
- `bad_text_rendering`
- `overexposed_or_underexposed`

Prompt alignment:
- `alignment_below_cutoff`
- `missing_locked_element`
- `missing_locked_relation`
- `missing_action_actor`
- `missing_action_target`
- `wrong_spatial_relation`
- `wrong_color`
- `wrong_material`
- `wrong_lighting`
- `wrong_camera_angle`
- `wrong_shot_size`
- `wrong_style`
- `invented_major_object`
- `negative_constraint_violation`

Seed sweep:
- `seed_fragility`
- `single_seed_luck`
- `coordinate_consistent_failure`

Impact:
- `impact_below_cutoff`
- `impact_unavailable`
- `impact_unlicensed`
- `impact_from_artifact_or_violation`

**13. Blame Hints**
Blame hints are not updates. They are structured evidence that a downstream learner can consume.

```python
class BlameHint(BaseModel):
    target_id: str | None
    field_path: str | None
    enum_value: str | None
    source: Literal["locked", "sampled", "inferred", "proposed", "unknown"]
    blame_type: Literal[
        "semantic_mismatch",
        "visibility_loss",
        "technical_quality",
        "constraint_violation",
        "seed_instability",
        "infrastructure"
    ]
    confidence: Literal["high", "medium", "low"]
    reason: str
```

Use cases:
- sampled `EXTREME_CLOSE_UP` hides a required relation target.
- locked `GOLDEN_HOUR` repeatedly renders as night for this model.
- proposed relation enum causes target mismatch.
- infrastructure timeout prevents semantic judgment.

Locked fields can receive reliability blame hints, but the evaluator must label them as locked so the downstream system does not treat them as LHS-replaceable.

**14. Disposition Signals**
The evaluator may emit disposition signals, but it must not execute them.

```python
class DispositionSignal(BaseModel):
    class_name: Literal[
        "passes_thresholds",
        "fail_persist_for_learning",
        "demote_candidate",
        "coordinate_quarantine_candidate",
        "temporary_enum_suppression_candidate",
        "hard_purge_invalid_artifact",
        "infrastructure_retry_no_semantic_penalty"
    ]
    confidence: Literal["high", "medium", "low"]
    reasons: list[str]
```

Signal meanings:

`passes_thresholds`: image meets active cutoff scores and may be promoted by the orchestrator.

`fail_persist_for_learning`: image failed semantically or aesthetically but is valid evidence for learning.

`demote_candidate`: image likely should not remain in promoted/curated state.

`coordinate_quarantine_candidate`: repeated seed-sweep failures suggest the coordinate should be held out.

`temporary_enum_suppression_candidate`: failures localize to a sampled enum arm strongly enough to recommend suppression.

`hard_purge_invalid_artifact`: file is corrupt or invalid; no learning value except infrastructure logging.

`infrastructure_retry_no_semantic_penalty`: generation/evaluation failed for infrastructure reasons; do not penalize prompt, enum, or coordinate.

The orchestrator markdown owns the actual action rules, persistence mutations, retry scheduling, purge mechanics, and Thompson/GP update application.

**15. Seed Sweep Aggregation**

Every LHS coordinate runs a seed-sweep bundle. The default bundle is:

```json
[7, 42, 156, 8888, 42069]
```

Runs may override the bundle via `RunConfig.seed_bundle`; the minimum bundle size is `MIN_SEED_BUNDLE_SIZE = 3`. The validator in `EvaluationBatchRequest.validate_seed_sweep_shape` enforces the minimum; the gate chain in `gates.py` enforces it again at the prompt and rendering stages.

**15.1 Outcome Thresholds**

The aggregate labels are bound to absolute thresholds in `evaluation._outcome`:

- `strong`: `promoted_count >= 3`.
- `viable`: `promoted_count > 1` (which, after the strong guard, resolves to `== 2`).
- `fragile`: `promoted_count == 1`.
- `failed`: `promoted_count == 0`.
- `blocked`: any infrastructure-only failure type (`invalid_image_file`, `evaluator_unavailable`, `evaluator_timeout`, `evaluator_malformed_output`, `gpu_memory_failure`).

Aggregate labels:

`strong`: at least three of five images pass required quality and alignment thresholds, with no repeated severe artifact.

`viable`: at least one image passes and failures are not consistently tied to a required target.

`fragile`: exactly one image passes or pass behavior appears seed-lucky.

`failed`: no images pass due to semantic or quality failures.

`blocked`: evaluator or infrastructure failure prevents judgment.

The evaluator must report per-image results and coordinate aggregate results. The learner should update from the aggregate, not from a single unlucky seed.

**16. Cutoff Examples**
Quality pass:
- `quality_score >= quality_cutoff`
- expected assumption: image is technically usable for the current workflow.

Quality fail:
- `quality_score < quality_cutoff`
- expected assumption: image should not be promoted on quality grounds.

Alignment pass:
- `alignment_score >= alignment_cutoff`
- expected assumption: image matches the rendered prompt and target manifest sufficiently.

Alignment fail:
- `alignment_score < alignment_cutoff`
- expected assumption: prompt-coordinate or generation failed to preserve required intent.

Human quality pass:
- human is detected and quality score meets stricter human cutoff.
- expected assumption: visible human artifacts are below acceptance threshold.

Human quality fail:
- human is detected and quality score misses stricter cutoff.
- expected assumption: do not promote.

Impact pass:
- optional impact score meets cutoff.
- expected assumption: image may receive downstream rank boost if quality and alignment also pass.

Impact fail:
- optional impact score misses cutoff.
- expected assumption: image can still be valid, but receives no impact boost.

**17. Evaluator Learning Boundaries**
The evaluator does not modify the Bayesian state. It emits update signals.

Recommended update signal:

```python
class LearningUpdateSignal(BaseModel):
    coordinate_id: str
    reward_hint: float
    seed_pass_rate: float
    arm_blame: list[BlameHint]
    combo_blame: list[BlameHint]
    locked_reliability_blame: list[BlameHint]
    infrastructure_only: bool = False
```

Interpretation:
- sampled arms may receive Thompson alpha/beta updates downstream.
- combos may receive GP affinity updates downstream.
- locked arms may receive reliability updates downstream, but not LHS mutation.
- infrastructure-only failures must not create semantic penalties.

**18. Latency Requirements**
The evaluator should support low-latency bulk operation.

Requirements:
- avoid reparsing raw prompts.
- consume the target manifest directly.
- batch IQA image tensors when possible.
- run VLM alignment in bounded small batches or serialized mode depending on VRAM.
- return partial results when one evaluator fails and another succeeds.
- preserve per-image IDs and coordinate grouping.
- record execution mode and elapsed time per evaluator.

The goal is not to make every evaluator call fully parallel. The goal is to make the contract batch-stable so adapters can choose the fastest safe execution mode.

**19. Persistence Requirements**
Persist:
- evaluator request ID.
- evaluator plan.
- model IDs and revisions.
- execution mode.
- image IDs.
- coordinate ID.
- seed.
- per-image scores.
- per-image pass flags.
- per-image failure types.
- localized blame hints.
- disposition signals.
- coordinate aggregate.
- evaluator confidence.
- malformed output or timeout details.

Do not persist only final pass/fail. The self-improver needs failure explanations and provenance.

**20. External Model References**
Examples used by this specification:
- JoyQuality SigLIP2 SO400M: `https://huggingface.co/fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn`
- MiniCPM-V 4.6: `https://huggingface.co/openbmb/MiniCPM-V-4.6`
- TRIBE v2 lite-qv: `https://huggingface.co/Jessylg27/tribev2-lite-qv`

Model choices may change without changing this evaluator contract. Adapters must map concrete model outputs into the stable evaluator result schema.

**21. Implementation Contract**

The evaluator pipeline and its adapters expose these contract surfaces:

- `src/bruteforce_canvas/evaluation.py` (`QualityEvaluation`, `AlignmentEvaluation`, `ImpactEvaluation`, `DispositionSignal`, `FailureType`, `BlameHint`, `_outcome`, `aggregate_seed_sweep`, `validate_seed_sweep_shape`, `BatchEvaluator`).
- `src/bruteforce_canvas/real_adapters.py` (`JoyQualityAdapter`, `MiniCPMVAdapter`, `TRIBEv2Adapter`; production real-inference runs load local weights, while deterministic fixture modes remain available for contract tests).
- `src/bruteforce_canvas/generation.py` (`MIN_SEED_BUNDLE_SIZE`, `DEFAULT_SEED_BUNDLE`).
- `src/bruteforce_canvas/static_ui.py` (read model `RunWorkspaceReadModel` is composed in `ui.py` and rendered here; failure-reason lookups for the report view).

**22. Summary Requirement**
The evaluator measures generated images across quality, prompt alignment, and optional metacognitive impact. It consumes priority-tagged PromptDocument targets, supports both serialized and parallel batch execution, aggregates the seed-sweep bundle per LHS coordinate using the absolute thresholds in `evaluation._outcome` (`>= 3 strong, > 1 viable, == 1 fragile, == 0 failed, infrastructure blocked`), emits stable failure types and blame hints, and returns disposition signals for the orchestrator. It does not execute orchestration actions or directly update Thompson Sampling or Gaussian Process state.
