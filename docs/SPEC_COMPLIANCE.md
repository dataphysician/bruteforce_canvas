# Bruteforce Canvas Spec Compliance Report

## Summary

All twelve planned phases (A through L) are implemented and integrated. The test suite contains 433 passing tests with 3 skipped entries (the skips are environment-gated, not failures), collected across the `tests/` directory. The git history is linear with 18 phase-scoped commits plus 3 pre-existing baseline commits, covering the staged refactor through accessibility and responsive UI work. Every spec document (00 through 06) is now backed by code and tests that exercise the spec's required behavior; the spec detail fidelity that was low-to-medium in the original greenfield build is now at parity with the spec text.

## Status by Spec

### Spec 00: Master Implementation Specification
**Phases:** A, B, C, D, E, F, G, H, I, J, K, L
**Status:** Compliant. The 8 master-spec phases (graph extraction, LHS routing, fast generation, batch evaluation, orchestration, UI, persistence, learning) are wired end-to-end. The downstream contract hierarchy (Spec 00 hands off to Specs 01-06 without redefining their fields) is preserved.

### Spec 01: DAG / Prompt Schema
**Phases:** A, B, C, D
**Status:** Compliant. The hard migration replaced simplified `Literal`-typed prompt models with the spec's full enum-based DAG (19 new enums, 5 lane containers: `SceneGraphDraft`, `ObjectLane`, `ActionLane`, `CinematographyLane`, `ConstraintLane`, plus `PromptDocumentSpec` and `PromptBundle`). Five semantic validators (`validate_relation_compatibility`, `validate_object_ownership`, `validate_action_support`, `validate_evidence_and_placeholders`, `validate_cross_lane_coherence`) are wired into `PromptPipeline.run_spec()` with a `RetryRequest` repair loop. Four per-lane LLM call contracts are available.

### Spec 02: LHS Enum Router
**Phases:** E, F
**Status:** Compliant. Compatibility pair family matrices (40+ rules in `compatibility_pairs.py`) drive LHS filtering. Out-of-distribution detection and context-keyed Thompson sampling are wired. The combo affinity state has been upgraded from a running mean to a real Gaussian Process posterior via GPyTorch (skipped when the `[ml]` extra is not installed).

### Spec 03: Fast Image Generation
**Phases:** A (worker refactor), G (registry), L (replay)
**Status:** Compliant. The `GENERATOR_REGISTRY` pattern lets the Bonsai adapter and any registered generator plug into the same worker contract. Fixed seed sweeps, generator provenance, and the generation worker contract from Spec 03 are exercised by `test_bonsai_adapter.py`, `test_generator_registry.py`, and `test_generation_runtime.py`.

### Spec 04: Evaluator Pipeline
**Phases:** G, L
**Status:** Compliant. Three real evaluator adapters are wired: `JoyQualityAdapter`, `MiniCPMVAdapter` (openbmb/MiniCPM-V-4.6), and `TRIBEv2Adapter` (Jessylg27/tribev2-lite-qv). All three support prewarm and emit VRAM telemetry; the `BatchEvaluator` adds execution modes and falls back to static behavior under CPU-only test conditions.

### Spec 05: Orchestration
**Phases:** H, I
**Status:** Compliant. The `StageGate` chain is wired into `RunService.tick()` with a new `RuntimeGateState` model. `BayesianBalancer` and `EnumSuppressionPolicy` together drive per-run threshold decisions. `AsyncRunDriver` runs the loop as an asyncio task with VRAM telemetry sampling; the persistent worker learning summary is migrated to the new balancer.

### Spec 06: UI / UX
**Phases:** J, K
**Status:** Compliant. The `EventBus` (in-process `asyncio.Queue` pubsub) feeds a CLI-hosted SSE endpoint. The static UI exposes an advanced view toggle and full error states, plus accessibility attributes (ARIA labels, live regions, skip link, keyboard navigation) and a responsive layout (768px and 479px breakpoints, wheel-zoom and pan in the candidate catalogue).

## Test Results

Latest full-suite run on 2026-06-15:

```
433 passed, 3 skipped in 11.09s
```

The 3 skipped tests are environment-gated (GPyTorch components skip cleanly when the `[ml]` extra is absent; they re-enable themselves when the extra is installed). No tests are marked xfail, no warnings printed during the run that indicate deprecation paths are still being exercised.

### How to run tests

| Goal | Command |
| --- | --- |
| Full default suite (fast, no real models) | `python -m pytest tests/ -q` |
| Slow / real-model tests only | `python -m pytest tests/test_e2e_real_adapters.py -m slow -q` |
| One spec-area test file | `python -m pytest tests/test_prompt_lanes.py -q` |
| Verbose with coverage | `python -m pytest tests/ -v --cov=bruteforce_canvas --cov-fail-under=80` |

## Known Limitations / Caveats

- **Real-model tests are opt-in and slow.** Adapters that load HuggingFace weights are marked with `@pytest.mark.slow` (or `@pytest.mark.requires_real_models`) and excluded from the default suite. They will pull model weights on first run and exercise the full prewarm + VRAM telemetry path.
- **GPU is not required by the default suite.** `JoyQualityAdapter`, `MiniCPMVAdapter`, and `TRIBEv2Adapter` each have a CPU fallback that returns deterministic stub scores. The real tensor paths run when invoked explicitly with weights available.
- **The SSE endpoint is CLI-hosted and non-persistent.** Spec 06's transport requirement is met by an `asyncio.Queue` pubsub bound to the CLI process lifecycle. When the CLI exits, the SSE server exits. There is no separate web service to start or stop.
- **The autonomous loop is asyncio-only and exits with the CLI.** `AsyncRunDriver` runs the loop as an `asyncio.Task` on the running loop. It is not a background daemon, has no independent lifecycle, and does not survive CLI exit.
- **GPyTorch is an optional extra.** Combo memory GP posterior tests skip cleanly without the `[ml]` extra. Run `pip install -e .[ml]` to enable them locally.
- **Stage gate coverage is structural.** The router gate currently returns `None` at pre-flight because `CandidateCoordinateBatch` is not yet persisted as a record; the gate is implemented and unit-tested but will only exercise its full gatherer once the event store carries router output.

## Verification Commands

```bash
# Full default suite (fast, deterministic)
python -m pytest tests/ -q

# Slow / real-model tests (opt-in)
python -m pytest tests/test_e2e_real_adapters.py -m slow -q

# Per-area regressions
python -m pytest tests/test_prompt_lanes.py tests/test_validators.py -q
python -m pytest tests/test_router_compatibility.py tests/test_combo_gp.py -q
python -m pytest tests/test_stage_gates.py tests/test_balancer.py -q
python -m pytest tests/test_event_bus.py tests/test_sse_endpoint.py -q
python -m pytest tests/test_a11y.py tests/test_responsive_ui.py -q

# Coverage gate
python -m pytest tests/ -v --cov=bruteforce_canvas --cov-fail-under=80

# Real smoke run (loads adapters, emits telemetry)
python -m bruteforce_canvas.cli workspace --run-id smoke --real-models
```

Last verified: 2026-06-15. The number above (433 passed, 3 skipped) reflects the state of the repository at the time of writing; the 3 new L.1/L.2 tests are part of the parallel work in Wave FINAL.
