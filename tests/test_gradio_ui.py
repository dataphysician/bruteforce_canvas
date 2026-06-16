import sys
from types import SimpleNamespace

import pytest

gr = pytest.importorskip("gradio")

from bruteforce_canvas.gradio_ui import (
    CSS,
    GradioSimulationState,
    _RUNTIME_SESSIONS,
    _runtime_fixed_arms,
    WORKFLOW_EXPLANATION_MARKDOWN,
    WORKFLOW_MERMAID_MARKDOWN,
    _maybe_zero_gpu,
    _zero_gpu_callbacks_enabled,
    build_demo,
    build_prompt_document_for_demo,
    generate_seed_sweep,
    generate_seed_sweep_runtime,
    initial_state,
    start_pre_run,
    start_pre_run_runtime,
    submit_feedback,
    transcribe_microphone_to_prompt,
    transcribe_microphone_to_prompt_steps,
)
from bruteforce_canvas.evaluation import StaticIQAAdapter, StaticVLMAdapter
from bruteforce_canvas.persistence import reconstruct_run_state
from bruteforce_canvas.prompt import VerificationIssue, VerificationReport, render_prompt_spec
from bruteforce_canvas.prompt_pipeline import PromptPipelineSpecResult


class IncrementingClock:
    def __init__(self, start: int = 0, step: int = 1) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> int:
        value = self.value
        self.value += self.step
        return value


class SequenceClock:
    def __init__(self, values: list[int]) -> None:
        self.values = list(values)
        self.last = values[-1]

    def __call__(self) -> int:
        if self.values:
            self.last = self.values.pop(0)
        return self.last


class FakePromptPipeline:
    def run_spec(self, raw_prompt: str):
        document = build_prompt_document_for_demo(raw_prompt)
        rendered = render_prompt_spec(document)
        return PromptPipelineSpecResult(
            approved=True,
            document=document,
            verifier_report=document.verification,
            rendered_prompt=rendered,
        )


class FakeBlockedPromptPipeline:
    def run_spec(self, raw_prompt: str):
        issue = VerificationIssue(
            issue_type="unresolved_action_target",
            repair_scope="prompt_improvement",
            blocking=True,
            message="Specify what the person is throwing.",
        )
        report = VerificationReport(approved=False, issues=[issue])
        document = build_prompt_document_for_demo(raw_prompt).model_copy(update={"verification": report})
        return PromptPipelineSpecResult(
            approved=False,
            document=document,
            verifier_report=report,
            rendered_prompt=None,
            prompt_improvement_feedback=[issue.message],
        )


def prepare_runtime(monkeypatch, tmp_path, *, iqa_scores: list[float] | None = None, vlm_scores: list[float] | None = None):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config: FakePromptPipeline())
    if iqa_scores is not None:
        monkeypatch.setattr(
            "bruteforce_canvas.gradio_ui._build_runtime_iqa",
            lambda _config: StaticIQAAdapter(scores=iqa_scores, model_id="test-runtime-quality"),
        )
    if vlm_scores is not None:
        monkeypatch.setattr(
            "bruteforce_canvas.gradio_ui._build_runtime_vlm",
            lambda _config: StaticVLMAdapter(scores=vlm_scores, model_id="test-runtime-alignment"),
        )
    _RUNTIME_SESSIONS.clear()
    return start_pre_run_runtime("A magical glowing rose encased in glass", initial_state())


def test_demo_prompt_parse_exposes_elements_relations_and_locks():
    document = build_prompt_document_for_demo("A magical glowing rose encased in glass")

    assert document.graph.elements[0].label == "rose"
    assert document.graph.relations[0].relation_raw == "inside"

    state, panel, report, lock_rows, generate_button, status = start_pre_run(
        "A magical glowing rose encased in glass",
        initial_state(),
    )

    assert isinstance(state, GradioSimulationState)
    assert panel["visible"] is True
    assert "object_01: rose" in report
    assert "object_01 inside object_02" in report
    assert any(row[1] == "cinematography.shot_size" and row[0] is False for row in lock_rows)
    assert generate_button["interactive"] is True
    assert "Pre-run parse ready" in status


def test_runtime_blocked_prompt_includes_reason_and_retry_hint(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config: FakeBlockedPromptPipeline())
    _RUNTIME_SESSIONS.clear()

    state, panel, report, _lock_rows, generate_button, status = start_pre_run_runtime(
        "A person throwing something",
        initial_state(),
    )

    assert panel["visible"] is True
    assert generate_button["interactive"] is False
    assert state.notification == (
        "Prompt parse blocked: Specify what the person is throwing. "
        "Try adding that detail, then submit the prompt again."
    )
    assert "Prompt parse blocked: Specify what the person is throwing." in status
    assert "Compiled prompt blocked." in report
    assert "Try adding that detail, then submit the prompt again." in report


def test_runtime_internal_evidence_blocker_is_translated_for_users(monkeypatch, tmp_path):
    class EvidenceBlockedPromptPipeline:
        def run_spec(self, raw_prompt: str):
            issue = VerificationIssue(
                issue_type="graph",
                repair_scope="evidence_or_placeholder",
                blocking=True,
                message=(
                    "Graph elements lack required 'evidence.text' and 'evidence.reason' fields. "
                    "All elements (e_12, e_13, e_14) have unresolved evidence with missing text and reason."
                ),
            )
            report = VerificationReport(approved=False, issues=[issue])
            document = build_prompt_document_for_demo(raw_prompt).model_copy(update={"verification": report})
            return PromptPipelineSpecResult(
                approved=False,
                document=document,
                verifier_report=report,
                rendered_prompt=None,
                prompt_improvement_feedback=[issue.message],
            )

    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config: EvidenceBlockedPromptPipeline())
    _RUNTIME_SESSIONS.clear()

    state, _panel, report, _lock_rows, generate_button, status = start_pre_run_runtime(
        "A surreal studio scene with several objects",
        initial_state(),
    )

    assert generate_button["interactive"] is False
    assert "evidence.text" not in state.notification
    assert "e_12" not in state.notification
    assert "The parser could not trace some parsed objects or relationships back to exact words" in state.notification
    assert 'Try this prompt: "Create a clear image of a surreal studio scene' in state.notification
    assert "red glass sphere" in state.notification
    assert "black pedestal supports it" in state.notification
    assert "Try rephrasing with the main objects, action, and relationship named explicitly" in status
    assert "evidence.reason" not in report
    assert "Compiled prompt blocked." in report
    assert "Try this prompt:" in report


def test_demo_blocks_app_constructs_without_launching():
    demo = build_demo()

    assert isinstance(demo, gr.Blocks)


def test_runtime_blocks_app_constructs_without_launching():
    demo = build_demo(mode="runtime")

    assert isinstance(demo, gr.Blocks)


def test_zero_gpu_metadata_is_declared_for_space():
    readme = open("README.md", encoding="utf-8").read()
    requirements = open("requirements.txt", encoding="utf-8").read()

    assert "suggested_hardware: zero-a10g" in readme
    assert "spaces>=" in requirements


def test_zero_gpu_wrapper_is_optional_without_spaces(monkeypatch):
    def function():
        return "ok"

    monkeypatch.delenv("BC_ZEROGPU_CALLBACKS", raising=False)
    monkeypatch.setitem(sys.modules, "spaces", None)

    assert _zero_gpu_callbacks_enabled() is True
    assert _maybe_zero_gpu(function, duration=5) is function


def test_zero_gpu_wrapper_uses_spaces_gpu_when_available(monkeypatch):
    calls = []

    def fake_gpu(*, duration):
        calls.append(duration)

        def decorator(function):
            def wrapped(*args, **kwargs):
                return function(*args, **kwargs)

            wrapped._zero_gpu_wrapped = True
            return wrapped

        return decorator

    def function():
        return "ok"

    monkeypatch.delenv("BC_ZEROGPU_CALLBACKS", raising=False)
    monkeypatch.setitem(sys.modules, "spaces", SimpleNamespace(GPU=fake_gpu))

    wrapped = _maybe_zero_gpu(function, duration=17)

    assert calls == [17]
    assert wrapped() == "ok"
    assert wrapped._zero_gpu_wrapped is True


def test_zero_gpu_wrapper_can_be_disabled(monkeypatch):
    def function():
        return "ok"

    monkeypatch.setenv("BC_ZEROGPU_CALLBACKS", "false")

    assert _zero_gpu_callbacks_enabled() is False
    assert _maybe_zero_gpu(function, duration=5) is function


def test_runtime_lock_rows_keep_only_locked_values():
    fixed = _runtime_fixed_arms(
        [
            [True, "object.color.object_01", "red", "", "fixed", "matched"],
            [False, "cinematography.shot_size", "", "close_up", "fixed", "matched"],
            [True, "object.material.object_01", "wood", "ceramic", "fixed", "matched"],
        ]
    )

    assert fixed["object.color.object_01"].value == "red"
    assert fixed["object.material.object_01"].value == "ceramic"
    assert "cinematography.shot_size" not in fixed


def test_runtime_seed_sweep_uses_backend_service_with_static_adapters(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_LOOP_LIMIT_SECONDS", 4)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.time.monotonic", IncrementingClock())

    state, _panel, _report, lock_rows, generate_button, _status = prepare_runtime(monkeypatch, tmp_path)
    outputs = list(generate_seed_sweep_runtime(state, lock_rows, 0.10, 0.10))
    final_state = outputs[-1][0]
    session = _RUNTIME_SESSIONS[final_state.run_id]
    records = session.service.store.replay()
    candidate_ids = [
        record.candidate_id
        for record in records
        if record.record_type == "candidate_record" and record.candidate_id is not None
    ]

    assert generate_button["interactive"] is True
    assert session.config.run.mode == "continuous"
    assert session.config.run.stall_window_seconds == 600
    assert session.config.run.stall_min_promoted == 10
    assert final_state.generated_count == 10
    assert final_state.iqa_evaluated_count == 10
    assert final_state.vlm_evaluated_count == 10
    assert len(final_state.current_batch) == 5
    assert {candidate.coordinate_id for candidate in final_state.current_batch} == {"coord_002"}
    assert len(final_state.curated) == 10
    assert {candidate.coordinate_id for candidate in final_state.curated} == {"coord_001", "coord_002"}
    assert len(candidate_ids) == len(set(candidate_ids)) == 10
    assert {"cand_coord_001_7", "cand_coord_002_7"}.issubset(set(candidate_ids))
    assert reconstruct_run_state(records).elapsed_seconds >= 4
    assert final_state.notification == "Stopped at 15-minute time limit."


def test_runtime_loop_stops_after_15_minute_limit_with_visible_transition(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_LOOP_LIMIT_SECONDS", 0)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.time.monotonic", IncrementingClock())
    state, _panel, _report, lock_rows, _generate_button, _status = prepare_runtime(monkeypatch, tmp_path)

    final_state = list(generate_seed_sweep_runtime(state, lock_rows, 0.10, 0.10))[-1][0]
    records = _RUNTIME_SESSIONS[final_state.run_id].service.store.replay()
    transitions = [record for record in records if record.record_type == "loop_transition"]

    assert final_state.generated_count == 0
    assert final_state.notification == "Stopped at 15-minute time limit."
    assert transitions[-1].payload["reason"] == "gradio_runtime_time_limit"


def test_runtime_loop_stops_by_stall_guard_after_10_minutes(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_LOOP_LIMIT_SECONDS", 900)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.time.monotonic", SequenceClock([0, 1, 601]))
    state, _panel, _report, lock_rows, _generate_button, _status = prepare_runtime(monkeypatch, tmp_path)

    final_state = list(generate_seed_sweep_runtime(state, lock_rows, 0.99, 0.99))[-1][0]
    records = _RUNTIME_SESSIONS[final_state.run_id].service.store.replay()

    assert final_state.generated_count == 5
    assert len(final_state.curated) == 0
    assert final_state.notification == "Stopped by stall guard: fewer than 10 curated images after 10 minutes."
    assert any(record.record_type == "stall_diagnostic" for record in records)
    assert reconstruct_run_state(records).elapsed_seconds == 601


def test_runtime_loop_does_not_stall_stop_before_10_minutes(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_LOOP_LIMIT_SECONDS", 599)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.time.monotonic", SequenceClock([0, 1, 599]))
    state, _panel, _report, lock_rows, _generate_button, _status = prepare_runtime(monkeypatch, tmp_path)

    final_state = list(generate_seed_sweep_runtime(state, lock_rows, 0.99, 0.99))[-1][0]
    records = _RUNTIME_SESSIONS[final_state.run_id].service.store.replay()

    assert final_state.notification == "Stopped at 15-minute time limit."
    assert not any(record.record_type == "stall_diagnostic" for record in records)


def test_runtime_curated_fragile_viable_and_strong_candidates_survive_across_batches(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_LOOP_LIMIT_SECONDS", 4)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.time.monotonic", IncrementingClock())
    state, _panel, _report, lock_rows, _generate_button, _status = prepare_runtime(
        monkeypatch,
        tmp_path,
        iqa_scores=[0.62, 0.72, 0.90, 0.82, 0.84],
        vlm_scores=[0.62, 0.72, 0.90, 0.82, 0.84],
    )

    final_state = list(generate_seed_sweep_runtime(state, lock_rows, 0.55, 0.55))[-1][0]
    catalog = {candidate.candidate_id: candidate for candidate in final_state.curated}

    assert len(final_state.current_batch) == 5
    assert {candidate.coordinate_id for candidate in final_state.current_batch} == {"coord_002"}
    assert {candidate.coordinate_id for candidate in final_state.curated} == {"coord_001", "coord_002"}
    assert {candidate.outcome for candidate in final_state.curated} >= {"fragile", "viable", "strong"}
    assert catalog["cand_coord_001_7"].outcome == "fragile"
    assert catalog["cand_coord_001_42"].outcome == "viable"
    assert catalog["cand_coord_001_156"].outcome == "strong"
    assert all(candidate.display_path for candidate in final_state.curated)


def test_microphone_is_positioned_before_prompt_textbox_in_source():
    source = open("src/bruteforce_canvas/gradio_ui.py", encoding="utf-8").read()

    assert source.index("microphone = gr.Microphone") < source.index("prompt = gr.Textbox")
    assert "min_width=180" in source
    assert "aspect-ratio: 4 / 3" in CSS
    assert "aspect-ratio: 3 / 4" not in CSS
    assert "aspect-ratio: 2 / 3" not in CSS
    assert "--bc-recorder-width: clamp(11rem, 14vw, 14rem)" in CSS
    assert "overflow: hidden !important" in CSS
    assert "animation: none !important" in CSS
    assert "display: none !important" in CSS
    assert '[data-testid="recording-waveform"]' in CSS


def test_workflow_accordion_uses_mermaid_markdown_and_requested_terms():
    source = open("src/bruteforce_canvas/gradio_ui.py", encoding="utf-8").read()

    assert source.index('elem_id="workflow-accordion"') < source.index('elem_classes=["bc-prompt-row"]')
    assert 'gr.Accordion("Workflow diagram", open=True' in source
    assert "```mermaid" in WORKFLOW_MERMAID_MARKDOWN
    assert "flowchart TD" in WORKFLOW_MERMAID_MARKDOWN
    assert "primaryTextColor" in WORKFLOW_MERMAID_MARKDOWN
    assert '"htmlLabels": false' in WORKFLOW_MERMAID_MARKDOWN
    assert "classDef step" in WORKFLOW_MERMAID_MARKDOWN
    assert "Mellum2 Thinking 12B" in WORKFLOW_MERMAID_MARKDOWN
    assert "Cohere Transcribe" in WORKFLOW_MERMAID_MARKDOWN
    assert "BGE enum canonicalizer" in WORKFLOW_MERMAID_MARKDOWN
    assert "Bonsai Ternary 4B" in WORKFLOW_MERMAID_MARKDOWN
    assert "MiniCPM-V-4.6" in WORKFLOW_MERMAID_MARKDOWN
    assert "TRIBE v2 lite-qv" in WORKFLOW_MERMAID_MARKDOWN
    assert "Update priors" in WORKFLOW_MERMAID_MARKDOWN
    assert "bc-workflow-row" in CSS
    assert "#workflow-accordion" in CSS
    assert "#workflow-explanation *" in CSS
    assert "#workflow-mermaid svg text" in CSS
    assert "#workflow-mermaid .nodeLabel p" in CSS
    for term in [
        "Decomposition",
        "Repair/Verify",
        "Canonicalization",
        "CohereLabs/cohere-transcribe-03-2026",
        "BAAI/bge-small-en-v1.5",
        "prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
        "fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn",
        "openbmb/MiniCPM-V-4.6",
        "Jessylg27/tribev2-lite-qv",
        "LHS",
        "Thompson Sampling/GP",
        "IQA",
        "VLM alignment",
        "TRIBE v2",
        "Prior updates",
        "enum-arm alpha/beta",
        "enum-combination GP affinity",
    ]:
        assert term in WORKFLOW_EXPLANATION_MARKDOWN


def test_status_chips_use_explicit_high_contrast_colors():
    assert ".bc-chip" in CSS
    assert "background: #ffffff" in CSS
    assert "color: #111827" in CSS
    assert "border: 1px solid #5f6f68" in CSS
    assert "overflow-wrap: anywhere" in CSS


def test_microphone_transcription_updates_prompt_without_loading_real_model(monkeypatch):
    class FakeTranscriber:
        def transcribe(self, audio):
            assert audio == "audio-value"
            return "a narrated glowing rose"

    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())

    prompt, state, status = transcribe_microphone_to_prompt("audio-value", "", initial_state())

    assert prompt == "a narrated glowing rose"
    assert state.raw_prompt == "a narrated glowing rose"
    assert "ASR transcript inserted" in status


def test_microphone_transcription_streams_visible_processing_message(monkeypatch):
    class FakeTranscriber:
        def transcribe(self, audio):
            assert audio == "audio-value"
            return "a narrated glowing rose"

    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())

    steps = list(transcribe_microphone_to_prompt_steps("audio-value", "", initial_state()))

    assert steps[0][0]["value"] == "Transcribing audio..."
    assert steps[0][0]["interactive"] is False
    assert "Transcribing microphone audio" in steps[0][2]
    assert steps[-1][0]["value"] == "a narrated glowing rose"
    assert steps[-1][0]["interactive"] is True


def test_generate_seed_sweep_yields_pending_then_evaluated_catalog():
    state, _panel, _report, lock_rows, _generate_button, _status = start_pre_run(
        "A magical glowing rose encased in glass",
        initial_state(),
    )

    outputs = list(generate_seed_sweep(state, lock_rows, 0.55, 0.25))

    pending_state = outputs[0][0]
    evaluated_state = outputs[-1][0]
    pending_gallery = outputs[0][3]
    evaluated_gallery = outputs[-1][3]
    catalog_gallery = outputs[-1][4]

    assert pending_state.generated_count == 5
    assert len(pending_gallery) == 5
    assert all("pending" in caption for _path, caption in pending_gallery)
    assert len(evaluated_gallery) == 5
    assert evaluated_state.iqa_evaluated_count == 5
    assert len(catalog_gallery) == len([candidate for candidate in evaluated_state.current_batch if candidate.promoted])
    assert any("failed" in caption for _path, caption in evaluated_gallery)


def test_reject_feedback_removes_selected_candidate_from_visible_catalog():
    state, _panel, _report, lock_rows, _generate_button, _status = start_pre_run(
        "A magical glowing rose encased in glass",
        initial_state(),
    )
    evaluated_state = list(generate_seed_sweep(state, lock_rows, 0.55, 0.25))[-1][0]
    if not evaluated_state.curated:
        pytest.skip("deterministic score mix produced no curated candidates")

    selected = evaluated_state.curated[0].candidate_id
    evaluated_state = evaluated_state.model_copy(update={"selected_candidate_id": selected})
    next_state, catalog, detail_panel, _image, _report, status = submit_feedback(evaluated_state, "reject")

    assert next_state.rejected_count == 1
    assert all(selected not in caption for _path, caption in catalog)
    assert "Feedback recorded: reject" in status
    assert detail_panel["visible"] == bool(catalog)
