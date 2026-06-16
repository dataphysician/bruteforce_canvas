**LHS Enum Router Specification**
**Companion to `bruteforce-canvas_DAG_prompt.md`**
**Fast generation companion: `bruteforce-canvas_fast_image_generation.md`**
**Evaluation companion: `bruteforce-canvas_Evaluator_pipeline.md`**
**Orchestration companion: `bruteforce-canvas_Orchestration.md`**

Status: implementation specification

Audience: prompt pipeline, resident worker, and evaluation-loop implementers

Primary goal: enrich underspecified but verified prompt documents with safe enum variation using Latin Hypercube Sampling, Thompson Sampling, and Gaussian Process style combination memory. The router should improve coverage without polluting the scene graph or inventing required prompt facts.

Build-order position: implement this spec after `bruteforce-canvas_DAG_prompt.md`. It consumes verified PromptDocuments and canonical enum metadata from the prompt spec, then emits prompt coordinates for the deterministic renderer and generation stages. Shared lifecycle, feedback, threshold, suppression, quarantine, retry, and persistence terms are defined in `bruteforce-canvas_Orchestration.md` and must not be renamed here.

**1. Core Position**
The deconstruction-reconstruction pipeline is capable of running LHS over missing or weakly specified fields, but only after the prompt has passed extraction, canonicalization, and verification.

The safe division of labor is:
1. Primary LLM extraction decides what exists, how elements relate, what actions occur, and which lane owns each raw phrase.
2. Field-scoped canonicalizers attach optional locked enum matches to raw values.
3. LLM verification confirms the document is faithful, connected, and renderable.
4. The compatibility prior constrains each sampleable field's eligible enum domain against locked context and required prompt facts.
5. LHS proposes coverage over only eligible sampleable fields and eligible enum domains.
6. A Bayesian enum router ranks, filters, and learns from candidate enum selections.
7. The deterministic renderer compiles each candidate without a final LLM normalizer.
8. Evaluators and pre-curated swipe/thumb feedback update enum-arm and enum-combination beliefs.

LHS is not a missing-scene repair system. It is a variation and enrichment system for fields that can be safely varied without changing the user's semantic request.

**2. What LHS May Randomize**
LHS may sample fields that are absent, weakly specified, or explicitly marked as flexible. These are usually presentation fields.

Good LHS axes:
- shot size
- camera angle
- lens family
- camera motion
- focus behavior
- lighting mood
- color treatment
- atmosphere
- composition style
- render style
- texture emphasis
- quality constraints
- non-semantic framing controls

Conditionally sampleable axes:
- material, finish, color, or wardrobe detail only when the element exists and the field is not user-specified.
- environment detail only when the location is broad and the verifier says enrichment will not conflict.
- mood or genre only when it does not contradict explicit camera, lighting, or subject language.

Do not sample by default:
- new primary subjects
- new props required for an action
- new relation targets
- graph relations
- action participants
- action labels
- user-specified object identity
- safety or negative constraints
- exact brand, identity, medical, legal, or other high-risk attributes

If a prompt says `a red ceramic bowl on a wooden table`, LHS may try `medium shot`, `soft window light`, `50mm lens`, or `muted palette`. It must not turn the bowl into a vase, add a spoon as an action prop, move the bowl under the table, or replace ceramic with metal.

**3. Field States**
Each field entering the router should carry an explicit state. This keeps randomization separate from extraction.

`explicit_raw`: The user stated the field, but no locked enum matched. Preserve raw value. Do not sample over it.

`explicit_locked`: The user stated the field and canonicalization matched a locked enum. Keep fixed during LHS, but still include it in candidate provenance and learning updates.

`explicit_locked_suppressed`: The user stated the field and canonicalization matched an enum that is currently suppressed for the active model/context. Preserve the raw value and keep the field fixed. Do not sample a replacement. Render from the raw phrase or a renderer-safe raw-preserving phrase, and record reliability evidence against the suppressed enum without making it LHS-replaceable.

`entailed_locked`: The verifier accepted the field as strongly entailed and canonicalized. Keep fixed unless the verifier marks it flexible.

`entailed_locked_suppressed`: The verifier accepted the field as entailed and canonicalized, but the matched enum is currently suppressed. Keep the raw/entailed value fixed, route to diagnostic hold when confidence is low, and do not allow LHS to replace it silently.

`missing_sampleable`: The user did not specify the field, and the verifier permits enrichment. LHS may choose a value.

`weak_sampleable`: The user gave broad language such as `nice light`, `cinematic`, or `dramatic angle`. LHS may choose from compatible enum values while preserving the raw phrase as context.

`suppressed_sampleable`: The field is otherwise sampleable, but a candidate enum arm is currently suppressed. Exclude that enum arm from ordinary LHS selection or apply only the configured exploration floor.

`blocked`: The field cannot be sampled because it would repair, invent, or contradict the graph.

`conflict`: The field requires extraction, canonicalization, or verifier repair before LHS can run.

Suppression changes selection policy, not semantic recognition. The canonicalizer should not pretend a suppressed enum failed to match. It should return a suppressed match state so the raw phrase, provenance, and reliability learning remain intact.

**4. Clean Parse Gate**
LHS should run only when the verified document satisfies all of these conditions:
- every required element has an ID and evidence.
- every relation source and target resolves to an element.
- every required action has grounded participants or an approved bounded inference patch.
- ordinary lighting treatment lives in cinematography, not graph relations.
- visible light-emitting objects are represented as elements only when they are part of the scene.
- locked enum matches are field-appropriate.
- unresolved slots are non-blocking or explicitly excluded from rendering.
- no hard conflicts exist across graph, action, object, cinematography, and constraint lanes.

If the clean parse gate fails, do not use LHS to cover the gap. Send the failed slice back to extraction, canonicalization, verifier repair, or prompt-improvement feedback.

**5. Why Clean Prompts Benefit Most**
Cleanly parsed prompts give the router a stable base. The graph supplies fixed semantic anchors, and LHS can explore presentation fields without changing what the user asked for.

Underspecified but clean prompt:

`a ceramic bowl on a wooden table`

Stable facts:
- element: bowl
- element: table
- relation: bowl on table
- material: ceramic belongs to bowl
- material: wood belongs to table

Sampleable fields:
- shot size
- camera angle
- lens
- lighting mood
- color treatment
- focus behavior
- composition style

This is a good LHS case because the missing fields are mostly photographic choices. The router can enrich without inventing new scene facts.

Underspecified and not clean:

`make it intense with that thing moving`

The graph lacks grounded elements, action actor, action target, and evidence. LHS must not guess the missing thing. The pipeline should request a clearer prompt or run a bounded extraction repair if prior context exists.

**6. Locked Enums Are Fixed, Not Immune**
Locked enum matches from field-scoped canonicalization are fixed during candidate generation, but they still participate in learning.

Example:
- user phrase: `golden hour`
- canonical enum: `lighting_mood=GOLDEN_HOUR`
- state: `explicit_locked`

The router must not replace `GOLDEN_HOUR` with `BLUE_HOUR` during ordinary LHS. However, if generated images repeatedly fail alignment because the model renders `golden hour` as night, oversaturation, or incorrect sun direction, the enum arm should receive a penalty in that context.

This matters because failures can come from:
- an enum arm that is correct linguistically but poor for the generator.
- an enum arm that is correct alone but bad with a specific combo.
- a locked enum that causes recurrent alignment failures for a model family.
- a canonicalizer that selected an enum too aggressively.

The router should not erase explicit user intent, but it should learn that certain locked enum arms or combinations require caution, prompt wording changes, or diagnostic hold.

**7. Candidate Coordinate Model**
Each candidate coordinate should store a full coordinate, not just the sampled fields.

```json
{
  "candidate_id": "cand_00042",
  "coordinate_id": "coord_00042",
  "run_id": "run_00091",
  "prompt_document_id": "doc_017",
  "target_manifest_id": "eval_manifest_00042",
  "enum_coordinate": {
    "relation_type.rel_01": {
      "value": "ON_TOP_OF",
      "state": "explicit_locked",
      "source": "llm_canonicalizer"
    },
    "object.material.bowl_01": {
      "value": "CERAMIC",
      "state": "explicit_locked",
      "source": "llm_canonicalizer"
    },
    "cinematography.shot_size": {
      "value": "MEDIUM_SHOT",
      "state": "missing_sampleable",
      "source": "lhs_router"
    },
    "cinematography.lighting_mood": {
      "value": "SOFT_WINDOW_LIGHT",
      "state": "missing_sampleable",
      "source": "lhs_router"
    }
  }
}
```

The full coordinate lets evaluation update both sampled and locked arms. Provenance prevents the next render from mutating explicit user facts.

**8. Latin Hypercube Sampling**
LHS gives coverage across eligible axes. It does not decide whether a value is semantically safe by itself. Compatibility filtering and Bayesian scoring must wrap LHS so weak pairings do not become rendered candidates merely because they occupy a useful stratum.

For each batch:
1. Build the candidate axis list from fields in `missing_sampleable` or `weak_sampleable`.
2. Keep `explicit_locked` and `entailed_locked` fields fixed as context axes.
3. Apply suppression policy and the compatibility prior against fixed context to remove impossible or currently suppressed enum arms from each sampleable axis.
4. Generate LHS rows over the remaining sampleable axes.
5. Map each row stratum to enum arms using compatibility-constrained, router-weighted ordering.
6. Attach fixed locked arms to the full candidate coordinate.
7. Apply full-coordinate compatibility checks and Bayesian scoring.
8. Render the top candidates that pass thresholds.

LHS prevents collapse onto a narrow set of values, while Thompson Sampling and GP-style combo memory bias the rows toward values that have historically passed.

**9. Thompson Sampling Per Enum Arm**
Each enum arm should maintain a Beta posterior:

```json
{
  "axis": "cinematography.lighting_mood",
  "value": "SOFT_WINDOW_LIGHT",
  "alpha": 12.0,
  "beta": 3.0,
  "context_key": "optional model/prompt-cluster key"
}
```

Interpretation:
- `alpha` tracks successful outcomes.
- `beta` tracks failed outcomes.
- A Thompson draw samples a plausible pass probability from `Beta(alpha, beta)`.
- The router uses the draw to rank enum arms inside LHS strata.

Pass/fail should not be a single naive image score. It should be derived from evaluator gates:
- prompt alignment pass
- required element preservation
- required relation preservation
- required action preservation
- technical quality pass
- artifact and anatomy checks when relevant
- constraint compliance
- pre-curated swipe/thumb feedback

The arm posterior estimates whether this enum tends to survive the actual generation and evaluation loop, not whether it is semantically valid in the abstract.

**10. GP-Style Combination Memory**
Some enum values are fine alone but fail together. The router needs combination memory in addition to per-arm learning.

Examples:
- `wide shot` plus `tiny product detail` may reduce product alignment.
- `blue hour` plus `warm kitchen practicals` may be fine in some scenes and incoherent in others.
- `dutch angle` plus `flat lay` may conflict unless the renderer has special handling.
- `macro lens` plus `large city street establishing view` may be a poor combo.

The combination model supports these representations:
- pairwise compatibility table from the frontier plan.
- stored combo affinity keyed by selected enum coordinate.
- kernel over one-hot enum coordinates.
- Gaussian Process over candidate embeddings.
- prompt-cluster-conditioned GP for similar scene families.

Required behavior:
- reward combinations that pass alignment and quality gates.
- penalize combinations that fail, especially when evaluator explanations localize the failure.
- retain uncertainty so new combinations can still be explored.
- separate arm-level blame from combo-level blame where possible.

The persistence contract must store both pair-level and full-coordinate outcomes so a GP implementation can replace or augment the combo-affinity store without changing candidate provenance.

**11. Compatibility Prior**
The compatibility prior is mandatory. It is the curated, typed layer that prevents weak cross-field pairings from becoming expensive rendered candidates. LHS gives coverage, Thompson Sampling gives per-arm optimism, and GP-style memory learns from outcomes. None of those replaces compatibility because some combinations are already known to be structurally contradictory, renderer-hostile, or model-hostile before any image is generated.

The compatibility prior operates at two levels:
- domain pruning before LHS value selection, so impossible arms are removed from each axis before rows are mapped.
- full-coordinate scoring after LHS assembly, so multi-field clashes are rejected or downranked before rendering.

The compatibility prior should be a typed matrix over field paths and enum values:

```json
{
  "left": {"field": "cinematography.shot_size", "value": "EXTREME_CLOSE_UP"},
  "right": {"field": "scene_density", "value": "MULTI_SUBJECT_INTERACTION"},
  "severity": "soft_downrank",
  "weight": -0.35,
  "reason": "Extreme close-up often drops secondary actors or targets in multi-subject action prompts.",
  "allowed_when": ["primary_subject_only", "detail_insert_requested"],
  "blocked_when": ["all_interaction_participants_required"]
}
```

Compatibility is not a flat block list. It has four levels:

`hard_reject`: impossible, contradictory, or violates explicit user intent. Candidate is removed before rendering.

`strong_downrank`: usually weak or model-hostile. Candidate may render only when exploration budget is high or context explicitly supports it.

`soft_downrank`: plausible but risky. Candidate remains eligible with a lower score.

`boost`: known useful pairing. Candidate gets a small prior lift, but still must pass evaluation.

**11.1 Compatibility Inputs**
The prior should evaluate the full candidate coordinate plus prompt context:
- fixed locked enum arms.
- sampled enum arms.
- raw user prompt.
- verified graph facts.
- action count and required participants.
- object sizes and importance.
- scene density.
- explicit constraints.
- field states such as `explicit_locked`, `missing_sampleable`, and `weak_sampleable`.
- model family and renderer version when available.

This prevents overblocking creative combinations. For example, `dutch angle` with `product photography` is often weak, but it may be valid if the prompt asks for a chaotic editorial campaign image.

**11.2 Hard Reject Rules**
Hard rejects should stay small and defensible. They are for contradictions, not taste.

Explicit-user conflict:
- sampled `BLUE_HOUR` when the user locked `GOLDEN_HOUR`.
- sampled `HIGH_KEY_LIGHTING` when the user locked `LOW_KEY_LIGHTING`.
- sampled `WIDE_SHOT` when the user asked for `macro detail of the ring engraving`.
- sampled `BLACK_AND_WHITE` when the user explicitly requested `bright red dress` as a required visual fact.

Camera geometry contradiction:
- `FLAT_LAY` plus `LOW_ANGLE`.
- `FLAT_LAY` plus `DUTCH_ANGLE` unless the renderer has a specific tilted-overhead template.
- `TOP_DOWN` plus `WORM_EYE_VIEW`.
- `ORTHOGRAPHIC_PRODUCT` plus `HANDHELD_TRACKING`.

Motion contradiction:
- `LOCKED_OFF_CAMERA` plus `HANDHELD_TRACKING`.
- `FREEZE_FRAME_STILLNESS` plus `MOTION_BLUR_EMPHASIS` unless the action lane requests implied motion blur.
- `LONG_EXPOSURE_TRAILS` plus `CRISP_SPORTS_ACTION` when the action must be sharply readable.

Graph preservation risk:
- `EXTREME_CLOSE_UP` plus required three-or-more participant interaction.
- `MACRO_LENS` plus required wide environment relation.
- `SHALLOW_DEPTH_OF_FIELD` plus required readable background text or small secondary object.
- `CROP_TIGHT_FACE` plus required held object or relation target outside the face.

Scene domain contradiction:
- `UNDERWATER_ATMOSPHERE` plus dry desert realism unless surrealism is explicit.
- `SNOWSTORM` plus indoor clean product-table realism unless the setting supports it.
- `STERILE_STUDIO_BACKGROUND` plus required visible busy city street.

Constraint contradiction:
- any sampled positive field that violates a negative constraint.
- any sampled style that conflicts with safety or compliance constraints.
- any sampled object, material, or palette that overwrites an explicit required field.

Hard rejects should be easy to explain in a trace. If the reason sounds subjective, it probably belongs in a downrank tier instead.

**11.3 Strong Downrank Rules**
Strong downranks are pairings that frequently produce poor alignment but are not impossible.

Subject/action readability:
- `EXTREME_CLOSE_UP` with `MULTI_PERSON_INTERACTION`.
- `WIDE_ESTABLISHING_SHOT` with `SMALL_PRODUCT_DETAIL`.
- `MOTION_BLUR` with `PRECISE_HAND_OBJECT_INTERACTION`.
- `SILHOUETTE_LIGHTING` with required facial expression.
- `HEAVY_FOG` with required object color or material recognition.

Composition and subject count:
- `CENTERED_SINGLE_SUBJECT` with required two-object interaction.
- `MINIMAL_NEGATIVE_SPACE` with dense marketplace, crowd, or cluttered workshop.
- `SYMMETRICAL_COMPOSITION` with chaotic action unless stylized order is requested.
- `TIGHT_CROP` with required background relation.

Lens and spatial relationship:
- `TELEPHOTO_COMPRESSION` with required clear foreground-background separation.
- `ULTRA_WIDE_LENS` with product hero accuracy.
- `FISHEYE` with precise architecture lines unless distortion is requested.
- `MACRO_LENS` with large-scale environment visibility.

Lighting and material:
- `LOW_KEY_LIGHTING` with required black object detail.
- `HIGH_KEY_LIGHTING` with required moody shadow atmosphere.
- `COLORED_GEL_LIGHTING` with required accurate product color.
- `BACKLIGHT_SILHOUETTE` with required garment texture.

Style and realism:
- `PAINTERLY_STYLE` with required photoreal product catalog.
- `ANIME_STYLE` with explicit documentary photo language.
- `VHS_LO_FI` with luxury product surface inspection.
- `HIGH_GRAIN_FILM` with small text readability.

**11.4 Soft Downrank Rules**
Soft downranks are weak priors that should still allow exploration:
- `DUTCH_ANGLE` with calm product still life.
- `HANDHELD_CAMERA` with polished catalog layout.
- `BLUE_HOUR` with indoor warm practicals.
- `SHALLOW_DEPTH_OF_FIELD` with medium-density scene.
- `WIDE_SHOT` with a single small object when the object is still primary.
- `MUTED_PALETTE` with vivid wardrobe when color is important but not the only required feature.
- `FILMIC_GRAIN` with clean commercial beauty image.
- `HIGH_CONTRAST` with dark clothing in dark environment.

Soft downranks should rarely remove a candidate. They mainly stop weak pairings from dominating early batches.

**11.5 Boost Rules**
The compatibility prior can also provide small positive priors:
- `MACRO_LENS` plus jewelry, food texture, fabric weave, insects, product detail.
- `WIDE_SHOT` plus landscape, architecture, crowd, vehicle environment.
- `SHALLOW_DEPTH_OF_FIELD` plus portrait, single product hero, foreground subject.
- `HANDHELD_CAMERA` plus documentary, street scene, chase, urgency.
- `DOLLY_IN` plus subject approaching, reveal, emotional emphasis.
- `DOLLY_OUT` plus isolation, reveal of environment, aftermath.
- `GOLDEN_HOUR` plus outdoor portrait, landscape, travel, warm editorial.
- `LOW_KEY_LIGHTING` plus noir, suspense, dramatic still life.

Boosts must remain modest. They should not overpower alignment failures or repeated negative feedback.

**11.6 Context Gates**
Every compatibility rule should be allowed to depend on context. This is how the system avoids a brittle universal blacklist.

Useful gates:
- `primary_subject_count`: one, two, three-plus.
- `required_relation_count`: none, one, many.
- `action_complexity`: still, single-action, multi-target, multi-actor.
- `object_scale`: tiny, handheld, human-scale, architectural, landscape.
- `scene_density`: sparse, medium, dense.
- `required_background`: true or false.
- `requires_text_readability`: true or false.
- `requires_color_accuracy`: true or false.
- `style_flexibility`: strict, moderate, open.
- `surreal_or_abstract_allowed`: true or false.
- `user_locked_field_present`: true or false.

Example:

```json
{
  "pair": ["cinematography.shot_size=EXTREME_CLOSE_UP", "action.complexity=MULTI_TARGET"],
  "default": {"severity": "strong_downrank", "weight": -0.55},
  "blocked_when": ["all_targets_required"],
  "allowed_when": ["detail_insert_requested"],
  "reason": "A tight crop may hide required targets unless the user requested an insert shot."
}
```

**11.7 Compatibility Matrix Shape**
The frontier plan should store compatibility by field pair, not by prose notes alone.

```json
{
  "field_pair": ["cinematography.shot_size", "cinematography.camera_angle"],
  "left_values": ["EXTREME_CLOSE_UP", "MEDIUM_SHOT", "WIDE_SHOT"],
  "right_values": ["TOP_DOWN", "EYE_LEVEL", "LOW_ANGLE"],
  "scores": {
    "EXTREME_CLOSE_UP|TOP_DOWN": 0.55,
    "EXTREME_CLOSE_UP|EYE_LEVEL": 0.85,
    "EXTREME_CLOSE_UP|LOW_ANGLE": 0.60,
    "WIDE_SHOT|LOW_ANGLE": 0.75
  },
  "rejects": [
    {
      "left": "TOP_DOWN",
      "right": "LOW_ANGLE",
      "reason": "These are mutually exclusive camera angle families."
    }
  ]
}
```

Recommended score range:
- `0.00`: hard reject.
- `0.10` to `0.35`: strong downrank.
- `0.35` to `0.70`: soft downrank or neutral-risk.
- `0.70` to `0.90`: good default.
- `0.90` to `1.00`: strong fit.

For multi-field candidates, aggregate pair scores with a conservative function:

```text
compatibility_prior(candidate) =
  min_pair_score * 0.50
  + mean_pair_score * 0.30
  + required_field_score * 0.20
```

Using the minimum pair score prevents one severe clash from being washed out by many harmless pairs.

**11.8 When to Apply Compatibility**
Apply compatibility twice:

Before LHS value mapping:
- remove enum arms that are impossible given locked fields.
- downrank enum arms that are weak with fixed context.
- keep LHS strata but map them over eligible values only.

After full candidate assembly:
- check pairwise and higher-order compatibility.
- reject hard conflicts.
- score soft conflicts.
- decide whether the candidate deserves rendering.

This two-pass design avoids wasting LHS rows on values that could never pass with the locked prompt context.

**11.9 Interaction with Thompson and GP**
The compatibility prior is not the same thing as learned preference.

Compatibility prior:
- curated or generated offline.
- applies before rendering.
- catches known contradictions and weak pairings.
- protects GPU and evaluator budget.

Thompson Sampling:
- learns per enum arm from observed outcomes.
- balances exploration and exploitation.
- can discover that a generally valid enum performs poorly for a model.

GP-style combo memory:
- learns context-specific combination behavior.
- discovers failures not known to the initial compatibility matrix.
- can correct optimistic assumptions after evidence accumulates.

The final router should combine them:

```text
candidate_allowed =
  no_hard_rejects
  and compatibility_prior >= compatibility_floor
  and no_locked_field_violation

candidate_score =
  lhs_coverage_bonus
  + thompson_arm_score
  + gp_combo_score
  + compatibility_prior_weight * compatibility_prior
  - recent_failure_penalty
```

Compatibility should never permanently silence non-safety creative combinations unless they violate explicit user intent or structural requirements. Use strong downranks plus exploration floors for subjective or style-dependent pairings.

**11.10 Compatibility Trace**
Every candidate should carry compatibility trace data:

```json
{
  "compatibility": {
    "prior_score": 0.62,
    "min_pair_score": 0.35,
    "mean_pair_score": 0.71,
    "hard_rejects": [],
    "downranks": [
      {
        "fields": ["cinematography.shot_size", "graph.required_relation_count"],
        "values": ["EXTREME_CLOSE_UP", "MANY"],
        "severity": "strong_downrank",
        "weight": -0.55,
        "reason": "Tight crop may hide required relation targets."
      }
    ],
    "boosts": [
      {
        "fields": ["cinematography.lens", "object.scale"],
        "values": ["MACRO_LENS", "TINY_PRODUCT"],
        "weight": 0.20,
        "reason": "Macro lens fits tiny product detail."
      }
    ]
  }
}
```

The trace is required for debugging why LHS did or did not produce certain variations. It also gives evaluator feedback a place to attach localized penalties.

**11.11 Required Pair Families**
The compatibility prior must cover these pair families at minimum. Without these matrices, LHS can produce weak or incoherent candidates even when every individual enum arm is reasonable.

Shot and subject/action complexity:
- `shot_size` x `primary_subject_count`
- `shot_size` x `required_relation_count`
- `shot_size` x `action_complexity`
- `shot_size` x `object_scale`

Lens and scene scale:
- `lens_family` x `object_scale`
- `lens_family` x `required_environment_visibility`
- `lens_family` x `architecture_line_accuracy`
- `lens_family` x `product_accuracy_requirement`

Camera angle and composition:
- `camera_angle` x `composition_style`
- `camera_angle` x `spatial_relation_requirement`
- `camera_angle` x `flat_lay_or_tabletop_scene`
- `camera_angle` x `human_pose_readability`

Camera motion and action:
- `camera_motion` x `action_complexity`
- `camera_motion` x `still_life_or_product_scene`
- `camera_motion` x `motion_blur_policy`
- `camera_motion` x `required_object_readability`

Focus and required details:
- `focus_behavior` x `required_background_visibility`
- `focus_behavior` x `requires_text_readability`
- `focus_behavior` x `small_secondary_object_required`
- `focus_behavior` x `multi_subject_interaction`

Lighting and material/color:
- `lighting_mood` x `requires_color_accuracy`
- `lighting_mood` x `material_reflectivity`
- `lighting_mood` x `dark_subject_or_dark_garment`
- `lighting_mood` x `facial_expression_required`

Atmosphere and visibility:
- `atmosphere` x `required_object_detail`
- `atmosphere` x `required_relation_visibility`
- `atmosphere` x `scene_depth_requirement`
- `atmosphere` x `weather_or_environment_locked`

Style and fidelity:
- `render_style` x `photorealism_requirement`
- `render_style` x `product_catalog_requirement`
- `render_style` x `documentary_or_news_requirement`
- `render_style` x `text_or_logo_readability`

Palette and explicit colors:
- `color_treatment` x `explicit_color_fields`
- `color_treatment` x `brand_color_requirement`
- `color_treatment` x `skin_tone_or_material_accuracy`
- `color_treatment` x `lighting_mood`

Constraint and safety:
- every positive sampled field x explicit negative constraints.
- every style field x safety/compliance constraints.
- every sampled object/detail field x graph mutation policy.

Each required pair family should define hard rejects, strong downranks, soft downranks, boosts, and context gates where applicable. Missing pair matrices should be treated as unknown risk, not as perfect compatibility.

**12. Bayesian Router Scoring**
The router should score a candidate before rendering:

```text
score(candidate) =
  lhs_coverage_bonus
  + sum(thompson_draw(enum_arm) * arm_weight)
  + combo_affinity(candidate)
  + compatibility_prior(candidate)
  - recent_failure_penalties
```

Hard conflicts reject a candidate before score computation. Soft risks lower its priority.

Hard reject examples:
- graph mutation required.
- relation target missing.
- explicit user enum replaced by sampled enum.
- sampled material conflicts with explicit material.
- sampled lighting contradicts explicit lighting.
- safety or negative constraint violated.

Soft downrank examples:
- low historical pass rate for an enum arm.
- low combo affinity.
- weak compatibility score.
- repeated prompt-cluster failures.
- evaluator warning from recent similar candidates.

**13. Reward and Penalty Updates**
Every rendered candidate should update learning state after evaluation.

Positive update:
- increment `alpha` for selected enum arms.
- add positive combo affinity.
- store evaluator reasons and prompt cluster.

Negative update:
- increment `beta` for implicated enum arms.
- add negative combo affinity.
- increase penalty for pairs or full coordinate if combo failure is likely.
- quarantine or diagnostic-hold arms that repeatedly fail in the same context.

Penalty should be localized when possible.

If the evaluator says `bowl missing`, penalize fields related to bowl rendering, composition, crop, focus, and any combo that may have hidden the bowl. Do not equally blame unrelated locked fields such as `wooden table`.

If the evaluator says `lighting contradicts prompt`, penalize the lighting enum and lighting-related combinations.

If the evaluator cannot localize the failure, diffuse a smaller penalty across all selected arms and the full combo.

**14. Locked Enum Penalty Policy**
Locked arms should learn differently from sampled arms.

For a sampled arm:
- a failure can reduce future selection probability directly.

For an explicit locked arm:
- do not mutate or replace it in future candidates for the same prompt.
- update its reliability statistics.
- mark context-sensitive failures.
- route repeated failures to prompt wording, renderer template, canonicalizer diagnostics, or model-family policy.

Example:

`golden hour portrait of a runner`

If `GOLDEN_HOUR` is explicit and images repeatedly render as night, the router may not switch to `SOFT_WINDOW_LIGHT`. It can:
- lower trust in `GOLDEN_HOUR` for that model family.
- prefer renderer wording such as `warm low sun at golden hour`.
- flag the canonicalizer or renderer with a diagnostic event.
- preserve the user's requested lighting in all candidates.

If `GOLDEN_HOUR` is currently suppressed when the user explicitly asks for it, canonicalization should return `matched_suppressed`, not `unmatched`. The run should preserve the raw phrase and fixed intent while recording locked reliability evidence. Suppression excludes the enum from ordinary sampled use; it does not erase the meaning of user-authored text.

**15. Out-of-Distribution Enum Handling**
OOD means an enum value or combination repeatedly fails evaluation in contexts where it should have worked, or succeeds only by drifting from the prompt.

Signals:
- alignment fails despite high technical quality.
- required object, relation, or action disappears.
- generated image introduces strong unintended objects.
- model treats a term as a different concept.
- failure clusters around an enum arm across unrelated prompts.
- failure clusters around a pair or combination across similar prompts.

Router responses:
- lower posterior confidence.
- add combo-level penalty.
- require higher exploration threshold.
- quarantine the enum for affected model family or prompt cluster.
- send enum to diagnostic hold if it was proposed rather than locked.
- keep minimum exploration for non-safety arms so recoveries can be detected.

OOD punishment should be empirical. Do not ban an enum because it sounds unusual; penalize it because it fails alignment or quality checks.

**16. Evaluation Inputs**
The router should consume structured evaluator outputs, not just a single score.

Recommended evaluator payload:

```json
{
  "candidate_id": "cand_00042",
  "prompt_alignment": 0.82,
  "technical_quality": 0.74,
  "target_status": {
    "bowl_01": {"priority": "locked_required", "status": "present"},
    "table_01": {"priority": "locked_required", "status": "present"},
    "rel_01": {"priority": "locked_required", "status": "present"}
  },
  "constraint_compliance": "pass",
  "failure_reasons": [],
  "localized_blame": [],
  "user_feedback": null
}
```

Swipe/thumb feedback applies only to pre-curated images that automated evaluation already considered passable:
- `accept`: positive curation reward.
- `reject`: false-positive correction with negative reward.
- `shred`: severe false-positive correction with stronger negative reward.

Automated evaluators should still record why a candidate passed or failed so the router can localize updates.

**17. No Final LLM Normalization**
The final compilation stage should not call another LLM for every LHS candidate. That would multiply compute cost by candidate count and can blur provenance.

Required behavior:
- fix primary extraction if scene facts are missing.
- fix canonicalizer context if enums are wrong.
- fix verifier criteria if unsafe candidates pass.
- fix deterministic renderer templates if prompt prose is awkward.
- fix enum router policies if poor values are sampled.

The renderer should be cheap, deterministic, and traceable. It should compile verified facts plus selected enum affordances into natural enough prompt strings beginning with `Generate`.

**18. Example Walkthrough**
Input:

`a red ceramic bowl on a wooden table, warm light`

Primary extraction:

```json
{
  "elements": [
    {"id": "object_01", "label": "bowl"},
    {"id": "object_02", "label": "table"}
  ],
  "relations": [
    {
      "id": "rel_01",
      "source_id": "object_01",
      "relation_raw": "on",
      "target_id": "object_02"
    }
  ],
  "object_lane": {
    "object_01": {"color": "red", "material": "ceramic"},
    "object_02": {"material": "wooden"}
  },
  "cinematography": {
    "lighting_mood": "warm light"
  }
}
```

Canonicalization:

```json
{
  "relation.rel_01": {"enum": "ON_TOP_OF", "state": "explicit_locked"},
  "object_01.color": {"enum": "RED", "state": "explicit_locked"},
  "object_01.material": {"enum": "CERAMIC", "state": "explicit_locked"},
  "object_02.material": {"enum": "WOOD", "state": "explicit_locked"},
  "cinematography.lighting_mood": {"enum": "WARM_LIGHT", "state": "explicit_locked"}
}
```

Verifier:

```json
{
  "approved": true,
  "sampleable_fields": [
    "cinematography.shot_size",
    "cinematography.camera_angle",
    "cinematography.lens",
    "cinematography.focus_behavior",
    "cinematography.composition"
  ],
  "blocked_fields": [
    "graph.elements",
    "graph.relations",
    "object_01.material",
    "object_02.material"
  ]
}
```

LHS candidate coordinate:

```json
{
  "fixed": {
    "relation.rel_01": "ON_TOP_OF",
    "object_01.color": "RED",
    "object_01.material": "CERAMIC",
    "object_02.material": "WOOD",
    "cinematography.lighting_mood": "WARM_LIGHT"
  },
  "sampled": {
    "cinematography.shot_size": "MEDIUM_SHOT",
    "cinematography.camera_angle": "EYE_LEVEL",
    "cinematography.lens": "50MM",
    "cinematography.focus_behavior": "SHALLOW_DEPTH_OF_FIELD",
    "cinematography.composition": "CENTERED_COMPOSITION"
  }
}
```

Rendered candidate:

`Generate red ceramic bowl on wooden table, medium shot from eye level, 50mm lens, shallow depth of field, warm light, centered composition`

Evaluation:

```json
{
  "prompt_alignment": 0.91,
  "technical_quality": 0.79,
  "target_status": {
    "object_01": {"priority": "locked_required", "status": "present"},
    "object_02": {"priority": "locked_required", "status": "present"},
    "rel_01": {"priority": "locked_required", "status": "present"}
  },
  "failure_reasons": [],
  "localized_blame": []
}
```

Update:
- increment alpha for sampled arms.
- increment alpha for locked arms as context success.
- add positive combo affinity.

Failure variant:

If another candidate sampled `EXTREME_MACRO` and the table disappeared, the update should:
- penalize `EXTREME_MACRO`.
- penalize the combo if paired with shallow depth of field or tight crop.
- leave `CERAMIC`, `RED`, `WOOD`, and `ON_TOP_OF` mostly untouched unless evaluator evidence implicates them.

**19. Implementation Interfaces**
Recommended Pydantic-facing shapes:

```python
class RoutedEnumArm(BaseModel):
    field_path: str
    enum_value: str
    state: Literal[
        "explicit_raw",
        "explicit_locked",
        "explicit_locked_suppressed",
        "entailed_locked",
        "entailed_locked_suppressed",
        "missing_sampleable",
        "weak_sampleable",
        "suppressed_sampleable",
        "blocked",
        "conflict",
    ]
    source: Literal["primary_extraction", "llm_canonicalizer", "llm_verifier", "lhs_router"]
    raw_value: str | None = None
    confidence: Literal["clear", "probable", "unclear"] | None = None

class LhsAxis(BaseModel):
    field_path: str
    enum_values: list[str]
    sample_policy: Literal["fixed", "sample", "blocked"]
    compatibility_tags: list[str] = []

class CompatibilityRule(BaseModel):
    left_field: str
    left_value: str
    right_field: str
    right_value: str
    severity: Literal["hard_reject", "strong_downrank", "soft_downrank", "boost"]
    weight: float
    reason: str
    allowed_when: list[str] = []
    blocked_when: list[str] = []

class CompatibilityTrace(BaseModel):
    prior_score: float
    min_pair_score: float
    mean_pair_score: float
    hard_rejects: list[CompatibilityRule]
    downranks: list[CompatibilityRule]
    boosts: list[CompatibilityRule]
    missing_pair_families: list[str] = []

class CandidateCoordinate(BaseModel):
    coordinate_id: str
    run_id: str
    prompt_document_id: str
    target_manifest_id: str
    fixed_arms: list[RoutedEnumArm]
    sampled_arms: list[RoutedEnumArm]
    lhs_row: list[float]
    compatibility_prior: float
    compatibility_trace: CompatibilityTrace
    bayesian_score: float
    combo_signature: str
    lifecycle_state: Literal["proposed"] = "proposed"

class RouterLearningFeedback(BaseModel):
    candidate_id: str
    coordinate_id: str
    prompt_alignment: float
    technical_quality: float
    target_status: dict[str, dict[str, str]]
    constraint_compliance: Literal["pass", "warn", "fail"]
    failure_reasons: list[str]
    localized_blame: list[str]
    user_feedback: Literal["accept", "reject", "shred"] | None = None
```

**20. Router Algorithm**
```text
route_lhs_candidates(prompt_document, enum_registry, frontier_plan, preference_state):
  assert prompt_document.verifier.approved

  fixed_arms = collect explicit_locked, explicit_locked_suppressed, entailed_locked, and entailed_locked_suppressed enum arms
  sampleable_axes = collect missing_sampleable and weak_sampleable fields
  blocked_fields = collect blocked and conflict fields

  reject if blocked_fields contain required graph or action gaps

  for axis in sampleable_axes:
    axis.domain = compatibility_prune_domain(
      axis=axis,
      fixed_arms=fixed_arms,
      prompt_document=prompt_document,
      frontier_plan=frontier_plan
    )
    reject axis if domain is empty and field is required
    drop axis if domain is empty and field is optional

  lhs_rows = latin_hypercube(dimensions=len(sampleable_axes), count=batch_size)
  candidates = []

  for row in lhs_rows:
    sampled_arms = []

    for axis, stratum in zip(sampleable_axes, row):
      ranked_values = rank_by_thompson_draws(axis.domain excluding ordinary suppressed arms, preference_state)
      sampled_arms.append(select_value_from_stratum(ranked_values, stratum))

    coordinate = fixed_arms + sampled_arms

    compatibility_trace = evaluate_compatibility(
      coordinate=coordinate,
      prompt_document=prompt_document,
      frontier_plan=frontier_plan
    )

    if compatibility_trace.hard_rejects:
      continue

    combo_affinity = preference_state.combo_affinity(coordinate)
    bayesian_score = score(coordinate, compatibility_trace.prior_score, combo_affinity)

    if bayesian_score >= render_threshold:
      coordinate.compatibility_trace = compatibility_trace
      candidates.append(coordinate)

  assign stable coordinate_id to each returned candidate
  return top candidates by bayesian_score with LHS coverage preserved
```

**21. Persistence**
Persist enough data to replay and learn:
- raw user prompt
- PromptDocument version
- primary extraction model version
- canonicalizer model version
- verifier model version
- enum registry version
- frontier plan version
- fixed arms and sampled arms
- LHS row
- rendered prompt
- generation settings
- image model version
- evaluator versions
- evaluator payload
- pre-curated swipe/thumb feedback
- enum-arm alpha/beta updates
- combo affinity updates
- compatibility traces
- compatibility-rule versions
- quarantine or diagnostic-hold flags

Candidate provenance must distinguish fixed locked arms from sampled arms. Evaluation updates may learn from both, but future LHS selection may only mutate fields marked sampleable.

**22. Summary Requirement**
The granular prompt deconstruction-reconstruction system performs LHS over missing fields only after the clean parse gate and compatibility prior approve the candidate space. That is one of the main reasons to deconstruct prompts into graph, object, action, cinematography, and constraint lanes.

The important constraint is that LHS should enrich only the fields that are safe to vary. Missing camera, lighting, composition, focus, palette, and style fields are ideal. Missing graph participants, relation targets, and action objects are not.

The compatibility prior sits before and after LHS value selection. It removes hard clashes, downranks weak pairings, boosts known-good pairings, and emits traceable reasons. The Bayesian router then sits between compatibility-filtered LHS and rendering. It uses Thompson Sampling to avoid enum arms that often fail, and GP-style combination memory to avoid groups of enums that look good independently but fail together. Out-of-distribution enum values are penalized when evaluation proves they hurt alignment or quality. Even locked enum values receive reliability updates, because a term can be linguistically correct and still unreliable for a specific generator, renderer wording, or scene cluster.

Cleanly parsed prompts benefit most. When the graph is faithful and the lanes are owned correctly, LHS can safely fill in the missing photographic decisions and let the evaluation loop learn which enum choices produce images that remain aligned.

**22.1 Implementation Contract**

The LHS enum router and its supporting learning memory expose these contract surfaces:

- `src/bruteforce_canvas/learning.py` (Thompson Sampling, OOD enum detection via `detect_ood_enum`, EnumSuppressionPolicy, `apply_coordinate_learning`).
- `src/bruteforce_canvas/router.py` (compatibility prior, sampleable field selection, candidate coordinate output, Bayesian scoring).
- `src/bruteforce_canvas/gp.py` (GPyTorch combo memory using `ExactGP`, `GaussianLikelihood`, `RBFKernel`, `fast_pred_var`; deterministic fallback when gpytorch is unavailable).
- `src/bruteforce_canvas/combo_gp.py` (pairwise compatibility table store keyed by combo signature).
- `src/bruteforce_canvas/balancer.py` (BayesianBalancer wrapper around Thompson and GP state with prompt-cluster or model-family context keys).
