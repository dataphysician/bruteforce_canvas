**Fast Image Generation Specification**
**Local generator adapter and registry contract**
**Companion to `bruteforce-canvas_DAG_prompt.md`, `bruteforce-canvas_LHS_enum_router.md`, `bruteforce-canvas_Evaluator_pipeline.md`, and `bruteforce-canvas_Orchestration.md`**

Status: implementation specification

Audience: resident GPU worker, generation backend, evaluation-loop, and UI implementers

Primary goal: define how a fast local image generation model is used as the resident image generator inside the brute-force prompt exploration loop, including continuous GPU generation, run-configured post-evaluation threshold triggers, and the required seed-sweep bundle for every LHS-routed prompt coordinate. The generator is model-agnostic; the registered `bonsai` builder in `GENERATOR_REGISTRY` is the default local production builder, and the registry itself is the generic stage name.

Build-order position: implement this spec after `bruteforce-canvas_DAG_prompt.md` and `bruteforce-canvas_LHS_enum_router.md`. It consumes rendered prompt coordinates, seed bundles, generation settings, and provenance. Shared lifecycle, threshold, promotion, feedback, retry, and persistence terms are defined in `bruteforce-canvas_Orchestration.md` and must not be renamed here.

**1. Model Role**
The fast image generation model is the image synthesis engine. It receives finalized prompt strings from the prompt renderer and produces candidate images for evaluation.

Default local generator role:

`<active generator builder id>`  # the bonsai builder in `GENERATOR_REGISTRY`; the registry abstracts over the concrete model card

Runtime role:
- generate 512x512 image candidates from deterministic rendered prompts.
- run resident on CUDA so the worker can keep generating without model reload between candidates.
- provide high-throughput candidate production for LHS/Bayesian prompt exploration.
- act only on the final rendered prompt string, seed, dimensions, and generation settings.

Non-role:
- does not parse user prompts.
- does not canonicalize enums.
- does not verify graph relations.
- does not decide whether an image is good.
- does not choose LHS enum values.
- does not decide post-evaluation promotion.

Those responsibilities belong to primary prompt-LLM extraction, embedding-first enum canonicalization with prompt-LLM fallback, prompt-LLM verification, the LHS enum router, the compatibility prior, JoyQuality, MiniCPM-V 4.6 alignment, and the pre-curated swipe/thumb feedback loop.

**2. Intended Use**
The fast generator should be used for local image candidate generation after the prompt pipeline has already produced a verified renderable prompt.

Correct input:
- one deterministic rendered prompt beginning with `Generate`.
- one explicit seed.
- fixed dimensions.
- fixed generation step count.
- generation metadata containing prompt coordinate provenance.

Incorrect input:
- raw user prompt that has not passed the graph/lane pipeline.
- raw PromptDocument JSON.
- enum dumps without rendered prose.
- prompt text rewritten by a final LLM normalizer.
- unverified LHS combinations that failed compatibility.

The active generator is optimized in this system as a brute-force candidate producer, not as the semantic authority. The surrounding pipeline supplies semantic structure and evaluation.

**3. Runtime Loading Contract**
The resident generator adapter owns model loading, warmup, generation calls, and model-specific runtime settings. The generator is model-agnostic; the active builder is selected from `GENERATOR_REGISTRY` (see `src/bruteforce_canvas/generator_registry.py`).

Required registry handles:
- `bonsai`: the default local production builder; wraps a model-specific inference path. The concrete model id is read from the builder configuration.
- `stub`: the deterministic fixture builder; returns deterministic placeholder images for contract tests and replay verification.

Required builder behavior:
- locate any text encoder, transformer, VAE, and tokenizer under the configured model root (or the stub's pre-baked artefacts).
- expose a `prewarm()` method that the worker calls once before the first generation call.
- accept a single `generate(rendered_prompt, seed, settings) -> GenerationResult` call, returning the image bytes or path, the seed used, the backend string, and the generation settings.
- report a stable backend string in the candidate record (e.g. `bonsai` or `stub`); the string names the registry builder, not a specific model card.

The resident worker must warm the active builder once and keep it resident while generating many candidates. Repeated load/unload is not part of the production loop.

**4. Generation Settings**
Default generation settings for the registered bonsai builder:

```json
{
  "steps": 4,
  "height": 512,
  "width": 512,
      "backend": "bonsai"  # names the registry builder; the model card is a configuration detail
}
```

The backend writes each generated PNG to the candidate image path and returns generation metadata alongside evaluation scores.

Candidate metadata must include:
- generator model ID.
- generator backend name.
- rendered prompt.
- seed.
- LHS enum coordinate.
- compatibility trace.
- Bayesian score.
- JoyQuality model ID and score.
- prompt-alignment model ID and score.
- promotion gate result.
- post-evaluation promoted/curated state.

**5. End-to-End Position in the Pipeline**
The intended production path is:

```text
User prompt
-> primary prompt-LLM PromptDocument extraction
-> embedding-first field-scoped enum canonicalization with prompt-LLM fallback
-> prompt-LLM verification
-> compatibility-constrained LHS enum routing
-> deterministic prompt rendering
-> fast generator seed-sweep bundle (default 5 seeds; minimum 3)
-> JoyQuality + MiniCPM-V 4.6 evaluation
-> post-evaluation promotion gate
-> Thompson/GP feedback update
-> continuous generation until orchestration stop, stall, or watermark trigger
```

The fast generator starts only after the prompt coordinate is ready to render. If extraction or compatibility fails, do not call the generator to discover whether the prompt might work.

**6. Continuous GPU Generation Loop**
The resident worker is designed to generate continuously on the GPU until a trigger point is reached.

Primary pause trigger:
- pause when the current run reaches the high watermark for images that survive post-generation evaluation scoring.

Example configured high watermark:

`500` post-evaluation promoted/curated images for the current run.

Configured resume trigger:
- resume when the current run's curated buffer drops below the configured low watermark.

Example configured low watermark:

`200` post-evaluation promoted/curated images for the current run.

An image counts toward the watermark only after it passes the required post-generation evaluation thresholds. Generated images that fail the promotion gate are persisted for evidence and learning, but they do not count toward the high watermark.

Promotion gate inputs:
- JoyQuality technical quality score.
- MiniCPM-V 4.6 prompt-alignment score.
- stricter quality threshold for human-visible images.
- prompt/constraint preservation signals.

Active thresholds come from the orchestration run configuration. The generation adapter must persist the threshold values used for the run and consume them without redefining promotion policy.

**7. Stall Guard**
The worker must follow the orchestration run's stall guard policy so GPU time is not spent indefinitely on a prompt/enum space that is producing almost no validated images.

Example configured stall guard:
- after `30` minutes of raw generation time, stop the run if fewer than `10` curated images were produced for the current run.

Purpose:
- protect GPU time.
- detect incompatible prompts, bad enum spaces, broken renderer output, or evaluator thresholds that are too strict.
- block invalid combinations and emit a diagnostic event.

Stall guard failure should produce a traceable status message containing:
- curated count.
- minimum required curated count.
- elapsed generation time.
- generated candidate count.
- run ID.
- prompt coordinate or enum-space summary.

**8. LHS Prompt Coordinate**
An LHS-routed prompt coordinate is the rendered prompt plus its selected enum and compatibility metadata.

Example coordinate:

```json
{
  "run_id": "run_00091",
  "prompt_document_id": "doc_017",
  "target_manifest_id": "eval_manifest_001842",
  "coordinate_id": "coord_001842",
  "rendered_prompt": "Generate red ceramic bowl on wooden table, medium shot from eye level, 50mm lens, shallow depth of field, warm light, centered composition",
  "fixed_arms": {
    "relation.rel_01": "ON_TOP_OF",
    "object_01.color": "RED",
    "object_01.material": "CERAMIC",
    "object_02.material": "WOOD",
    "cinematography.lighting_mood": "WARM_LIGHT"
  },
  "sampled_arms": {
    "cinematography.shot_size": "MEDIUM_SHOT",
    "cinematography.camera_angle": "EYE_LEVEL",
    "cinematography.lens": "50MM",
    "cinematography.focus_behavior": "SHALLOW_DEPTH_OF_FIELD",
    "cinematography.composition": "CENTERED_COMPOSITION"
  },
  "compatibility_prior": 0.82,
  "bayesian_score": 0.76
}
```

This coordinate is the unit that gets the seed-sweep bundle.

**9. Required Seed-Sweep Bundle**
Every randomized LHS prompt coordinate must run a seed-sweep bundle before the system makes a durable judgment about that coordinate.

Default seed bundle:

```json
[7, 42, 156, 8888, 42069]
```

The default bundle is a list of five seeds. The minimum bundle size is 3 (`MIN_SEED_BUNDLE_SIZE` in `src/bruteforce_canvas/generation.py`); runs may override the bundle via `RunConfig.seed_bundle`. The gate chain in `src/bruteforce_canvas/gates.py` enforces the minimum at the prompt and rendering stages.

Purpose:
- avoid seed selection bias.
- distinguish bad prompt coordinates from unlucky single-seed renders.
- identify prompt coordinates that are robust across seed variation.
- avoid over-rewarding a weak coordinate that happened to produce one lucky image.
- provide a fairer update signal to Thompson arm posteriors and GP-style combo memory.

The seed sweep belongs to the rendered prompt coordinate, not to the raw user prompt. If LHS produces 100 distinct prompt coordinates, each coordinate runs its own seed-sweep bundle.

**10. Seed Sweep Evaluation**
For each seed in the active seed-sweep bundle:
1. Generate one image with the active fast generator using the same rendered prompt and settings.
2. Evaluate with JoyQuality.
3. Evaluate prompt alignment with MiniCPM-V 4.6.
4. Run the promotion gate.
5. Persist candidate metadata and image path.
6. Mark passing images as promoted and curated.

The coordinate-level result is an aggregate over the five images:

```json
{
  "run_id": "run_00091",
  "prompt_document_id": "doc_017",
  "target_manifest_id": "eval_manifest_001842",
  "coordinate_id": "coord_001842",
  "seeds": [7, 42, 156, 8888, 42069],  # default; runs may override via RunConfig.seed_bundle; minimum bundle size is 3
  "generated": 5,
  "promoted": 3,
  "best_joyquality": 0.78,
  "mean_joyquality": 0.63,
  "best_alignment": 0.84,
  "mean_alignment": 0.71,
  "pass_rate": 0.60,
  "coordinate_status": "viable"
}
```

The coordinate is not judged by a single seed unless the run is explicitly configured for a diagnostic fast path. The production path uses the full bundle.

**11. Coordinate Outcome Policy**
The seed sweep can produce these coordinate outcomes:

`strong`: at least three of the bundle's seeds pass, with no severe repeated artifact issue. The absolute threshold in `evaluation._outcome` is `promoted_count >= 3`.

`viable`: at least one seed passes and aggregate scores are not contradictory. The absolute threshold is `promoted_count > 1` (which after the strong guard resolves to `promoted_count == 2`).

`fragile`: one seed passes but most seeds fail for similar reasons. The absolute threshold is `promoted_count == 1`.

`failed`: no seeds pass. The absolute threshold is `promoted_count == 0`.

`blocked`: generation or evaluation failed for infrastructure reasons. Any infrastructure-only failure type (`invalid_image_file`, `evaluator_unavailable`, `evaluator_timeout`, `evaluator_malformed_output`, `gpu_memory_failure`) routes to `blocked` and never produces a semantic penalty.

Update behavior:
- `strong`: positive Thompson update for sampled enum arms and positive combo affinity.
- `viable`: small positive update, weighted by pass rate.
- `fragile`: mixed update; do not overreward because the coordinate is seed-sensitive.
- `failed`: negative update, localized to implicated arms and combinations.
- `blocked`: no semantic penalty; route to infrastructure retry.

Locked enum arms are included in the coordinate and receive reliability updates, but they are not mutated by LHS for the current prompt.

**12. Post-Evaluation Promoted Images**
The high watermark counts images that meet required cutoff scores after image evaluation, not merely generated images.

An image is post-evaluation promoted when:
- fast image generation succeeds.
- the image file is valid.
- JoyQuality passes the active threshold.
- prompt alignment passes the active threshold.
- human-image quality threshold passes when a human is detected.
- hard constraints are not violated.
- the candidate is marked `promoted=true` and `curated=true`.

The run-level high watermark counts only promoted/curated images from the current run. Images from older runs must not pause a fresh run before that fresh run produces its own threshold-passing images.

**13. Nonstop Generation Contract**
The real-CUDA worker should run in this cycle:

```text
while not stopped:
  if run_curated_count >= high_watermark:
    pause run
    wait until curated buffer drops below low_watermark
    continue

  if stall guard fails:
    stop run with stall_guard status
    break

  prompt_coordinates = route LHS batch

  for coordinate in prompt_coordinates:
    for seed in run_config.seed_bundle:
      image = image_generator.generate(rendered_prompt, seed)
      evaluation = evaluate(image, rendered_prompt)
      persist candidate
      if evaluation.promoted:
        mark promoted and curated
      if run_curated_count >= high_watermark:
        pause after current safe persistence point
        break

    update Bayesian state from coordinate-level seed sweep aggregate
```

The worker should persist after each generated image. A crash or stop request should lose at most the active in-flight generation, not the entire coordinate sweep.

**14. Prompt and Seed Provenance**
Every generated image must carry enough metadata to reproduce and audit it:
- raw user prompt.
- PromptDocument ID/version.
- rendered prompt.
- coordinate ID.
- fixed enum arms.
- sampled enum arms.
- compatibility prior and trace.
- Bayesian score before generation.
- seed.
- generator model ID.
- generator backend.
- steps, height, width.
- generation timestamp.
- evaluator model IDs and versions.
- evaluator payloads.
- promotion gate result.
- promoted/curated state.

Seed sweep metadata must link all bundle candidates back to the same coordinate ID.

**15. Avoiding Seed Selection Bias**
A single seed can make a valid prompt look bad or a weak prompt look good. The seed-sweep bundle reduces that variance.

Bad policy:
- render one seed for each prompt coordinate.
- promote or penalize the entire coordinate based on that one output.

Correct policy:
- render the bundle's seeds (`7`, `42`, `156`, `8888`, and `42069` by default) for each prompt coordinate.
- evaluate all outputs.
- promote any individual image that passes.
- update the coordinate based on aggregate pass behavior using the absolute thresholds in `evaluation._outcome`.
- preserve per-seed evidence for debugging.

This gives the router better learning signals:
- if every seed in the bundle fails, the coordinate is likely bad.
- if one seed passes and the rest fail, the coordinate is fragile.
- if most seeds pass, the coordinate is robust.
- if failures localize to a relation, object, or lighting field, penalize the implicated enum arms or combo.

**16. Evaluator Coupling**
Fast generator outputs are consumed by evaluators immediately. The generator should not be treated as successful merely because a PNG was written.

Evaluator roles:
- JoyQuality scores technical visual quality.
- MiniCPM-V 4.6 evaluates prompt alignment and returns a JSON critique payload. (Human presence and artifact issues are folded into the JoyQuality human-quality threshold and the failure taxonomy, not into the VLM output.)
- `BAAI/bge-small-en-v1.5` is the embedding model used by the enum canonicalizer for prompt-side canonicalization.
- pre-curated swipe/thumb feedback refines posterior state with `accept`, `reject`, and `shred`.

Promotion is a gate over evaluator outputs. Bayesian learning is an update over both evaluator outputs and pre-curated swipe/thumb feedback.

**17. Failure Classes**
Generation failures:
- CUDA unavailable.
- generator model root missing.
- `GpuPipeline` construction fails.
- image bytes fail to write.
- generated PNG is invalid.

Evaluation failures:
- JoyQuality unavailable.
- MiniCPM-V 4.6 unavailable.
- evaluator returns malformed output.
- alignment cannot be computed.

Semantic failures:
- required element missing.
- required relation missing.
- action target missing.
- explicit color/material/lighting contradicted.
- human artifacts visible.
- prompt alignment below threshold.

Infrastructure failures should not penalize enum arms. Semantic failures should update Thompson/GP state according to localized blame.

**18. GPU Residency**
The GPU worker should keep the active generator resident across a run. The surrounding evaluator models may also stay resident when the hardware tier supports it.

Base 16GB capacity target:
- active generator resident.
- JoyQuality resident and warmed before user interaction when real models are enabled.
- MiniCPM-V 4.6 alignment resident and warmed before user interaction when real models are enabled.
- TRIBE v2 disabled unless a higher hardware tier and policy explicitly enable it.
- 512x512 generation for the default bonsai builder.
- evaluator batches sized to avoid VRAM pressure.

Default capacity envelope (the bonsai builder with MiniCPM-V 4.6 alignment and JoyQuality):
- generator load and prewarm peak is roughly 4 GiB to 5 GiB depending on model card.
- single 512x512 generation peak is roughly 4 GiB to 5 GiB depending on model card.
- combined base-tier operation (active generator, JoyQuality, MiniCPM-V 4.6) should remain within about 12 GiB to 14 GiB, leaving headroom for the worker process and event bus.

These numbers are capacity guidelines for the default implementation example. Runtime telemetry records actual hardware behavior for the active generator and evaluator set, surfaced through `RunWorkspaceReadModel.vram_telemetry`.

**19. UI Contract**
The UI should expose the loop simply:
- prompt input.
- start generation.
- stop generation.
- selected image detail.
- `Accept`, `Reject`, and `Shred` feedback for pre-curated images.

The UI should not expose low-level generator internals as ordinary controls. Advanced settings belong in diagnostics or developer configuration.

For each selected image, the UI should show:
- rendered prompt.
- seed.
- coordinate enum JSON.
- generation settings.
- JoyQuality score.
- alignment score.
- promotion thresholds.
- promotion gate reasons.
- feedback controls.

**20. Implementation Interfaces**
Recommended generation request:

```python
class FastGenerationRequest(BaseModel):
    run_id: str
    prompt_document_id: str
    target_manifest_id: str
    coordinate_id: str
    rendered_prompt: str
    seed: int
    steps: int = 4
    height: int = 512
    width: int = 512
    fixed_arms: dict[str, str]
    sampled_arms: dict[str, str]
    compatibility_trace: dict
    bayesian_score: float
```

Recommended seed sweep request:

```python
class SeedSweepRequest(BaseModel):
    run_id: str
    prompt_document_id: str
    target_manifest_id: str
    coordinate_id: str
    rendered_prompt: str
    seeds: list[int] = [7, 42, 156, 8888, 42069]  # default; runs may override; minimum is MIN_SEED_BUNDLE_SIZE
    generation: dict
    fixed_arms: dict[str, str]
    sampled_arms: dict[str, str]
    compatibility_trace: dict
    bayesian_score: float
```

Recommended seed sweep result:

```python
class SeedSweepResult(BaseModel):
    run_id: str
    prompt_document_id: str
    target_manifest_id: str
    coordinate_id: str
    generated_candidate_ids: list[int]
    seeds: list[int]
    promoted_count: int
    pass_rate: float
    mean_joyquality: float
    mean_alignment: float
    best_joyquality: float
    best_alignment: float
    outcome: Literal["strong", "viable", "fragile", "failed", "blocked"]
    localized_blame: list[str]
```

**21. Persistence Requirements**
The database should persist:
- candidate row per generated seed image.
- coordinate ID shared by all five seed images.
- seed.
- rendered prompt.
- enum JSON.
- generation settings.
- evaluator payload.
- promotion gate payload.
- curated/promoted flags.
- seed sweep aggregate payload.
- Bayesian update payload.
- feedback actions.

This allows the system to answer:
- which LHS prompt coordinate produced this image.
- which seed produced this image.
- whether other seeds for the same coordinate passed or failed.
- whether the coordinate was robust or seed-sensitive.
- why a candidate passed or failed post-evaluation promotion.

**22. Operational Summary**
The resident fast image generator is used after prompt deconstruction, enum routing, compatibility filtering, and deterministic rendering. The generator is model-agnostic; the active local builder comes from `GENERATOR_REGISTRY` (the bonsai builder is the default production builder, and `stub` is the deterministic fixture builder). The worker keeps the active generator warm and continuously generates candidate images until the orchestrator stops the run, the stall guard trips, or the current run reaches its configured high watermark for images that meet required post-evaluation cutoff scores. If a high/low watermark pair is configured, generation resumes when that run's promoted/curated buffer drops below the configured low watermark.

Every LHS-routed rendered prompt coordinate must run a seed-sweep bundle. The default bundle is `[7, 42, 156, 8888, 42069]`; runs may override via `RunConfig.seed_bundle`; the minimum bundle size is `MIN_SEED_BUNDLE_SIZE = 3`. The system evaluates each seed image, marks threshold-passing images as promoted/curated, and updates Bayesian enum beliefs from the aggregate sweep result using the absolute thresholds in `evaluation._outcome`. This prevents single-seed luck from distorting enum selection, GP combo memory, and prompt-coordinate viability.

**22.1 Implementation Contract**

The fast image generation pipeline exposes these contract surfaces:

- `src/bruteforce_canvas/generator_registry.py` (`GENERATOR_REGISTRY` mapping builder names to callables, with `BUILDER_INCLUDES = {"stub", "bonsai"}`).
- `src/bruteforce_canvas/generation.py` (`DEFAULT_SEED_BUNDLE = [7, 42, 156, 8888, 42069]`, `MIN_SEED_BUNDLE_SIZE = 3`, seed sweep request and result models).
- `src/bruteforce_canvas/real_adapters.py` (`JoyQualityAdapter`, `MiniCPMVAdapter`, `TRIBEv2Adapter`; production real-inference runs load local weights, and TRIBE remains optional outside the 16 GB base tier).
- `src/bruteforce_canvas/evaluation.py` (`BatchEvaluator`, `validate_seed_sweep_shape`, `aggregate_seed_sweep`, `_outcome` with the absolute thresholds `>= 3 strong, > 1 viable, == 1 fragile, == 0 failed, infrastructure blocked`).
- `src/bruteforce_canvas/learning.py` (Thompson Sampling alpha/beta updates keyed by enum arm and context).
- `src/bruteforce_canvas/gp.py` (GPyTorch combo memory using `ExactGP`, `GaussianLikelihood`, `RBFKernel`, `fast_pred_var`; deterministic fallback when gpytorch is unavailable).
- `src/bruteforce_canvas/persistence.py` (candidate row per seed image and aggregate row per coordinate, with `PERSISTENCE_VERSION` schema marker).
