from types import SimpleNamespace

import time

import pytest

gr = pytest.importorskip("gradio")

from bruteforce_canvas.gradio_ui import (
    CSS,
    GradioSimulationState,
    SimCandidate,
    _RUNTIME_SESSIONS,
    _catalog_gallery_update,
    _catalog_page_controls,
    _catalog_slot_paths,
    _build_runtime_prompt_pipeline,
    _lock_table_from_review,
    _preview_gallery,
    _prewarm_runtime_asr_if_enabled,
    _release_inactive_runtime_sessions,
    _review_markdown,
    _runtime_config_for_prompt,
    _runtime_fixed_arms,
    _runtime_sampleable_axes,
    WORKFLOW_DIAGRAM_HTML,
    WORKFLOW_EXPLANATION_MARKDOWN,
    WORKFLOW_MERMAID_MARKDOWN,
    _asr_runtime_status_snapshot,
    build_demo,
    build_prompt_document_for_demo,
    generate_seed_sweep,
    generate_seed_sweep_runtime,
    initial_state,
    launch,
    move_catalog_page,
    cancel_pre_run_runtime,
    start_pre_run,
    start_pre_run_runtime,
    submit_feedback,
    _set_asr_runtime_status,
    transcribe_microphone_to_prompt,
    transcribe_microphone_to_prompt_steps,
)
from bruteforce_canvas.evaluation import StaticIQAAdapter, StaticVLMAdapter
from bruteforce_canvas.loop import LoopAction, LoopDecision
from bruteforce_canvas.orchestration import RunRuntimeState
from bruteforce_canvas.persistence import PersistenceRecord, reconstruct_run_state
from bruteforce_canvas.prompt import (
    CanonicalEnum,
    EvidenceCategory,
    EvidenceSpan,
    VerificationIssue,
    VerificationReport,
    render_prompt_spec,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Importance, RelationType
from bruteforce_canvas.prompt_models import Element, ObjectDescriptor, ObjectLane, PromptDocumentSpec, RelationDescriptor, SceneGraphDraft
from bruteforce_canvas.prompt_pipeline import PromptPipelineSpecResult
from bruteforce_canvas.shared import CanonicalStatus
from bruteforce_canvas.ui import pre_run_modal_from_prompt


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


class FakeRaisingPromptPipeline:
    def run_spec(self, raw_prompt: str):
        raise ValueError("graph Field required; seed_prompt Extra inputs are not permitted")


def prepare_runtime(monkeypatch, tmp_path, *, iqa_scores: list[float] | None = None, vlm_scores: list[float] | None = None):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config, **_kwargs: FakePromptPipeline())
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
    assert "bc-triplet-row" in report
    assert "rose" in report
    assert "inside" in report
    assert "glass case" in report
    assert any(row[1] == "cinematography.shot_size" and row[0] is False for row in lock_rows)
    assert generate_button["interactive"] is True
    assert "Pre-run parse ready" in status


def test_runtime_blocked_prompt_includes_reason_and_retry_hint(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config, **_kwargs: FakeBlockedPromptPipeline())
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


def test_runtime_submit_parser_exception_returns_blocked_review_panel(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config, **_kwargs: FakeRaisingPromptPipeline())
    _RUNTIME_SESSIONS.clear()

    state, panel, report, lock_rows, generate_button, status = start_pre_run_runtime(
        "A green ball on a bench.",
        initial_state(),
    )

    assert panel["visible"] is True
    assert generate_button["interactive"] is False
    assert state.review is not None
    assert state.review.can_begin_generation is False
    assert "Prompt parser returned an invalid document shape" in state.notification
    assert "Compiled prompt blocked." in report
    assert "Prompt parser returned an invalid document shape" in report
    assert lock_rows
    assert "Prompt parse blocked" in status


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
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config, **_kwargs: EvidenceBlockedPromptPipeline())
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


def test_runtime_launch_prewarm_runs_before_demo_is_built(monkeypatch):
    calls = []

    class FakeDemo:
        def queue(self, **kwargs):
            calls.append(("queue", kwargs))
            return self

        def launch(self, **kwargs):
            calls.append(("launch", kwargs))

    def fake_prewarm(mode):
        calls.append(("prewarm", mode))

    def fake_build_demo(*, mode=None):
        calls.append(("build_demo", mode))
        return FakeDemo()

    monkeypatch.setattr("bruteforce_canvas.gradio_ui._prewarm_runtime_startup_if_enabled", fake_prewarm)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_demo", fake_build_demo)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui._build_theme", lambda: None)

    launch(mode="runtime", server_name="127.0.0.1", server_port=7860)

    assert calls[0] == ("prewarm", "runtime")
    assert calls[1] == ("build_demo", "runtime")
    assert calls[2][0] == "queue"
    assert calls[3][0] == "launch"
    assert str(calls[3][1]["allowed_paths"][0]).endswith("runtime/gradio_runs")


def test_runtime_asr_prewarm_uses_singleton_transcriber(monkeypatch):
    calls = []

    class FakeConfig:
        model_id = "fake-cohere-asr"

    class FakeTranscriber:
        config = FakeConfig()

        def prewarm(self, *, run_dummy_inference: bool = True):
            calls.append(run_dummy_inference)

    monkeypatch.delenv("BC_ASR_PREWARM", raising=False)
    monkeypatch.setenv("BC_ASR_PREWARM_INFERENCE", "false")
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())

    _prewarm_runtime_asr_if_enabled("runtime")
    _prewarm_runtime_asr_if_enabled("simulation")

    assert calls == [False]


def test_runtime_asr_prewarm_failure_marks_unavailable_and_unloads(monkeypatch):
    calls = []

    class FakeConfig:
        model_id = "fake-cohere-asr"

    class FakeTranscriber:
        config = FakeConfig()

        def prewarm(self, *, run_dummy_inference: bool = True):
            calls.append(("prewarm", run_dummy_inference))
            raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)")

        def unload(self):
            calls.append(("unload", None))

    monkeypatch.delenv("BC_ASR_PREWARM", raising=False)
    monkeypatch.delenv("BC_ASR_PREWARM_REQUIRED", raising=False)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())

    try:
        _prewarm_runtime_asr_if_enabled("runtime")

        status = _asr_runtime_status_snapshot()
        assert status.state == "failed"
        assert "CUBLAS_STATUS_ALLOC_FAILED" in str(status.error)
        assert calls == [("prewarm", True), ("unload", None)]
    finally:
        _set_asr_runtime_status("idle", "test reset")


def test_microphone_transcription_waits_for_asr_preload(monkeypatch):
    class FakeTranscriber:
        def transcribe(self, audio):
            raise AssertionError("transcription should not run while ASR is loading")

    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())
    _set_asr_runtime_status("loading", "Cohere Transcribe ASR is warming up.")

    try:
        prompt, state, status = transcribe_microphone_to_prompt("audio-value", "typed prompt", initial_state())

        assert prompt == "typed prompt"
        assert "ASR is still warming up" in state.notification
        assert "try again when ASR is ready" in status
    finally:
        _set_asr_runtime_status("idle", "test reset")


def test_microphone_transcription_keeps_asr_loaded_after_success_by_default(monkeypatch):
    calls = []

    class FakeTranscriber:
        def transcribe(self, audio):
            calls.append(("transcribe", audio))
            return "a narrated glowing rose"

        def unload(self):
            calls.append(("unload", None))

    fake = FakeTranscriber()
    monkeypatch.delenv("BC_ASR_RELEASE_AFTER_TRANSCRIBE", raising=False)
    monkeypatch.delenv("BC_ASR_KEEP_LOADED_AFTER_TRANSCRIBE", raising=False)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: fake)
    _set_asr_runtime_status("idle", "test reset")

    try:
        prompt, state, status = transcribe_microphone_to_prompt("audio-value", "", initial_state())

        assert prompt == "a narrated glowing rose"
        assert state.raw_prompt == "a narrated glowing rose"
        assert "ASR transcript inserted" in status
        assert "released" not in status
        assert _asr_runtime_status_snapshot().state == "ready"
        assert calls == [("transcribe", "audio-value")]
    finally:
        _set_asr_runtime_status("idle", "test reset")


def test_microphone_transcription_unloads_asr_when_low_vram_policy_is_enabled(monkeypatch):
    calls = []

    class FakeTranscriber:
        def transcribe(self, audio):
            calls.append(("transcribe", audio))
            return "a narrated glowing rose"

        def unload(self):
            calls.append(("unload", None))

    fake = FakeTranscriber()
    monkeypatch.setenv("BC_ASR_RELEASE_AFTER_TRANSCRIBE", "true")
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: fake)
    _set_asr_runtime_status("idle", "test reset")

    try:
        prompt, state, status = transcribe_microphone_to_prompt("audio-value", "", initial_state())

        assert prompt == "a narrated glowing rose"
        assert state.raw_prompt == "a narrated glowing rose"
        assert "ASR released by low-VRAM policy" in status
        assert _asr_runtime_status_snapshot().state == "idle"
        assert calls == [("transcribe", "audio-value"), ("unload", None)]
    finally:
        _set_asr_runtime_status("idle", "test reset")


def test_microphone_transcription_does_not_reload_after_asr_release(monkeypatch):
    class FakeTranscriber:
        def transcribe(self, audio):
            raise AssertionError("released ASR should not be lazily reloaded")

    monkeypatch.delenv("BC_ASR_PREWARM", raising=False)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())
    _set_asr_runtime_status("idle", "Cohere Transcribe ASR released after runtime pre-run service build.")

    try:
        prompt, state, status = transcribe_microphone_to_prompt("audio-value", "typed prompt", initial_state())

        assert prompt == "typed prompt"
        assert "released to free GPU memory" in state.notification
        assert "restart the runtime UI before recording again" in status
    finally:
        _set_asr_runtime_status("idle", "test reset")


def test_empty_microphone_transcript_keeps_asr_ready_for_retry(monkeypatch):
    calls = []

    class FakeTranscriber:
        def transcribe(self, audio):
            calls.append(("transcribe", audio))
            return ""

        def unload(self):
            calls.append(("unload", None))

    fake = FakeTranscriber()
    monkeypatch.delenv("BC_ASR_KEEP_LOADED_AFTER_TRANSCRIBE", raising=False)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: fake)
    _set_asr_runtime_status("ready", "Cohere Transcribe ASR ready.")

    try:
        prompt, state, status = transcribe_microphone_to_prompt("audio-value", "typed prompt", initial_state())

        assert prompt == "typed prompt"
        assert "ASR returned an empty transcript" in status
        assert _asr_runtime_status_snapshot().state == "ready"
        assert calls == [("transcribe", "audio-value")]
    finally:
        _set_asr_runtime_status("idle", "test reset")


def test_readme_omits_space_front_matter():
    readme = open("README.md", encoding="utf-8").read()

    assert not readme.startswith("---\n")


def test_runtime_lock_rows_keep_only_locked_values():
    fixed = _runtime_fixed_arms(
        [
            [True, "object.color.object_01", "red", "", "", "", "", "fixed", "matched"],
            [False, "cinematography.shot_size", "", "close_up", "", "", "", "fixed", "matched"],
            [True, "object.material.object_01", "wood", "[selected] ceramic", "", "", "", "fixed", "matched"],
        ]
    )

    assert fixed["object.color.object_01"].value == "red"
    assert fixed["object.material.object_01"].value == "ceramic"
    assert "cinematography.shot_size" not in fixed


def test_lock_table_surfaces_selected_enums_and_preloaded_lhs_priors():
    document = PromptDocumentSpec(
        raw_user_prompt="green ball on top of a bench",
        graph=SceneGraphDraft(
            seed_prompt="green ball on top of a bench",
            elements=[
                Element(
                    id="object_01",
                    label="ball",
                    entity_type=EntityType.PRODUCT,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="ball", category=EvidenceCategory.EXPLICIT),
                ),
                Element(
                    id="object_02",
                    label="bench",
                    entity_type=EntityType.FURNITURE,
                    role=ElementRole.SUPPORTING,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="bench", category=EvidenceCategory.EXPLICIT),
                ),
            ],
            relations=[
                RelationDescriptor(
                    id="rel_01",
                    source_id="object_01",
                    target_id="object_02",
                    relation_raw="on top of",
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="on top of", category=EvidenceCategory.EXPLICIT),
                )
            ],
        ),
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="object_01", color="green")]),
        canonical_metadata={
            "relation.rel_01": CanonicalEnum(
                raw_value="on top of",
                enum_value=RelationType.ON_TOP_OF.name,
                status=CanonicalStatus.MATCHED_ACTIVE,
                confidence="high",
                reason="relation is explicit in the prompt",
            )
        },
        verification=VerificationReport(approved=True, issues=[]),
    )

    rows = _lock_table_from_review(pre_run_modal_from_prompt(document))
    by_field = {row[1]: row for row in rows}

    assert by_field["relation.rel_01"][3] == "[selected] ON_TOP_OF"
    assert "ON_TOP_OF" in by_field["relation.rel_01"][4]
    assert "MEDIUM_SHOT" in by_field["cinematography.shot_size"][4]
    assert "MEDIUM_SHOT: alpha" in by_field["cinematography.shot_size"][5]

    fixed = _runtime_fixed_arms(rows)
    sampleable_axes = _runtime_sampleable_axes(rows, fixed)

    assert fixed["relation.rel_01"].value == "ON_TOP_OF"
    assert sampleable_axes["cinematography.shot_size"][0].value == "MEDIUM_SHOT"
    assert sampleable_axes["cinematography.shot_size"][0].alpha > sampleable_axes["cinematography.shot_size"][0].beta


def test_runtime_fast_parse_disables_llm_fallback_and_repair_loops(monkeypatch, tmp_path):
    captured = {}

    def fake_build_prompt_pipeline(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.delenv("BC_RUNTIME_FAST_PARSE", raising=False)
    monkeypatch.delenv("BC_RUNTIME_LLM_CANONICALIZER_FALLBACK", raising=False)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", fake_build_prompt_pipeline)

    config = _runtime_config_for_prompt("Create a clear image of a green ball on top of a bench.")
    _build_runtime_prompt_pipeline(config)

    assert config.canonicalizer.llm_fallback is False
    assert captured["kwargs"] == {
        "extraction_validation_retries": 0,
        "max_repairs": 0,
        "max_semantic_repairs": 0,
        "run_semantic_validation": False,
        "run_verifier": False,
    }


def test_runtime_submit_keeps_asr_loaded_by_default(monkeypatch, tmp_path):
    calls = []

    class FakeTranscriber:
        def unload(self):
            calls.append("unload")

    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.delenv("BC_ASR_RELEASE_BEFORE_RUNTIME_SERVICE", raising=False)
    monkeypatch.delenv("BC_ASR_RELEASE_BEFORE_GENERATION", raising=False)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config, **_kwargs: FakePromptPipeline())
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())
    _RUNTIME_SESSIONS.clear()
    _set_asr_runtime_status("ready", "Cohere Transcribe ASR ready.")

    try:
        state, _panel, _report, _lock_rows, generate_button, _status = start_pre_run_runtime(
            "A magical glowing rose encased in glass",
            initial_state(),
        )

        assert state.review is not None
        assert state.review.can_begin_generation is True
        assert generate_button["interactive"] is True
        assert _asr_runtime_status_snapshot().state == "ready"
        assert calls == []
    finally:
        _RUNTIME_SESSIONS.clear()
        _set_asr_runtime_status("idle", "test reset")


def test_runtime_submit_can_release_asr_when_low_vram_policy_is_enabled(monkeypatch, tmp_path):
    calls = []

    class FakeTranscriber:
        def unload(self):
            calls.append("unload")

    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.setenv("BC_ASR_RELEASE_BEFORE_RUNTIME_SERVICE", "true")
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config, **_kwargs: FakePromptPipeline())
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())
    _RUNTIME_SESSIONS.clear()
    _set_asr_runtime_status("ready", "Cohere Transcribe ASR ready.")

    try:
        state, _panel, _report, _lock_rows, generate_button, _status = start_pre_run_runtime(
            "A magical glowing rose encased in glass",
            initial_state(),
        )

        assert state.review is not None
        assert state.review.can_begin_generation is True
        assert generate_button["interactive"] is True
        assert _asr_runtime_status_snapshot().state == "idle"
        assert calls == ["unload"]
    finally:
        _RUNTIME_SESSIONS.clear()
        _set_asr_runtime_status("idle", "test reset")


def test_runtime_releases_stale_sessions_before_new_service_build(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config, **_kwargs: FakePromptPipeline())
    _RUNTIME_SESSIONS.clear()

    old_worker = SimpleNamespace(generator=object(), iqa=object(), vlm=object(), impact=object())

    class OldService:
        def __init__(self) -> None:
            self.state = RunRuntimeState.RUNNING
            self.worker = old_worker
            self.requested_stop = False
            self.stop_reasons: list[str] = []

        def request_stop(self) -> None:
            self.requested_stop = True

        def stop_with_reason(self, reason: str, *, details: dict[str, object] | None = None) -> None:
            self.state = RunRuntimeState.STOPPED
            self.stop_reasons.append(reason)

    old_service = OldService()
    _RUNTIME_SESSIONS["old_run"] = SimpleNamespace(
        config=SimpleNamespace(run=SimpleNamespace(run_id="old_run")),
        service=old_service,
    )

    try:
        state, _panel, _report, _lock_rows, generate_button, _status = start_pre_run_runtime(
            "A magical glowing rose encased in glass",
            initial_state(),
        )

        assert generate_button["interactive"] is True
        assert old_service.requested_stop is True
        assert old_service.stop_reasons == ["gradio_runtime_replaced_by_new_prompt"]
        assert old_worker.generator is None
        assert old_worker.iqa is None
        assert old_worker.vlm is None
        assert old_worker.impact is None
        assert "old_run" not in _RUNTIME_SESSIONS
        assert state.run_id in _RUNTIME_SESSIONS
    finally:
        _RUNTIME_SESSIONS.clear()


def test_release_inactive_runtime_sessions_keeps_requested_run() -> None:
    _RUNTIME_SESSIONS.clear()

    class Service:
        def __init__(self) -> None:
            self.state = RunRuntimeState.RUNNING
            self.worker = SimpleNamespace(generator=object(), iqa=object(), vlm=object(), impact=object())
            self.stopped = False

        def request_stop(self) -> None:
            return None

        def stop_with_reason(self, _reason: str, *, details: dict[str, object] | None = None) -> None:
            self.state = RunRuntimeState.STOPPED
            self.stopped = True

    old_service = Service()
    keep_service = Service()
    _RUNTIME_SESSIONS["old"] = SimpleNamespace(config=SimpleNamespace(run=SimpleNamespace(run_id="old")), service=old_service)
    _RUNTIME_SESSIONS["keep"] = SimpleNamespace(
        config=SimpleNamespace(run=SimpleNamespace(run_id="keep")),
        service=keep_service,
    )

    try:
        _release_inactive_runtime_sessions(keep_run_id="keep", reason="test_release")

        assert list(_RUNTIME_SESSIONS) == ["keep"]
        assert old_service.stopped is True
        assert keep_service.stopped is False
        assert old_service.worker.iqa is None
    finally:
        _RUNTIME_SESSIONS.clear()


def test_runtime_seed_sweep_uses_backend_service_with_static_adapters(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_LOOP_LIMIT_SECONDS", 6)
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
    prompts_by_coordinate = {
        record.coordinate_id: record.payload["rendered_prompt"]
        for record in records
        if record.record_type == "coordinate_record" and record.coordinate_id is not None
    }
    batch_prompts_by_coordinate = {
        record.coordinate_id: record.payload["rendered_prompt"]
        for record in records
        if record.record_type == "runtime_batch_prompt" and record.coordinate_id is not None
    }

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
    assert prompts_by_coordinate["coord_001"].startswith("Generate ")
    assert prompts_by_coordinate["coord_001"] != session.rendered_prompt.rendered_prompt
    assert prompts_by_coordinate["coord_001"] != prompts_by_coordinate["coord_002"]
    assert "camera angle" in prompts_by_coordinate["coord_001"]
    assert batch_prompts_by_coordinate["coord_001"] == prompts_by_coordinate["coord_001"]
    assert batch_prompts_by_coordinate["coord_001"] != batch_prompts_by_coordinate["coord_002"]
    assert reconstruct_run_state(records).elapsed_seconds >= 4
    assert final_state.notification == "Stopped at 15-minute time limit."


def test_runtime_seed_sweep_recovers_stale_browser_state_from_prompt_input(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_LOOP_LIMIT_SECONDS", 0)
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.build_prompt_pipeline", lambda _config, **_kwargs: FakePromptPipeline())
    _RUNTIME_SESSIONS.clear()

    outputs = list(
        generate_seed_sweep_runtime(
            initial_state(),
            [],
            0.10,
            0.10,
            "A magical glowing rose encased in glass",
        )
    )
    final_state = outputs[-1][0]

    assert final_state.review is not None
    assert final_state.review.can_begin_generation is True
    assert final_state.raw_prompt == "A magical glowing rose encased in glass"
    assert final_state.run_id in _RUNTIME_SESSIONS
    assert final_state.notification == "Stopped at 15-minute time limit."


def test_runtime_seed_sweep_streams_generated_seed_gallery_before_batch_completion(monkeypatch, tmp_path):
    monkeypatch.setenv("BC_GRADIO_STREAM_POLL_SECONDS", "0.005")
    state, _panel, _report, lock_rows, _generate_button, _status = prepare_runtime(monkeypatch, tmp_path)
    session = _RUNTIME_SESSIONS[state.run_id]

    def append_candidate(seed: int) -> None:
        coordinate_id = "coord_001"
        candidate_id = f"cand_{coordinate_id}_{seed}"
        image_path = tmp_path / f"{candidate_id}.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        session.service.store.append(
            PersistenceRecord(
                record_id=f"candidate:{candidate_id}",
                record_type="candidate_record",
                run_id=session.config.run.run_id,
                prompt_document_id=session.document.prompt_document_id,
                target_manifest_id=session.target_manifest.manifest_id,
                coordinate_id=coordinate_id,
                candidate_id=candidate_id,
                seed=seed,
                payload={
                    "candidate_id": candidate_id,
                    "run_id": session.config.run.run_id,
                    "prompt_document_id": session.document.prompt_document_id,
                    "target_manifest_id": session.target_manifest.manifest_id,
                    "coordinate_id": coordinate_id,
                    "seed": seed,
                    "rendered_prompt": session.rendered_prompt.rendered_prompt,
                    "generator_model_id": "stream-test",
                    "generator_backend": "stream-test",
                    "generation_settings": {},
                    "image_path": str(image_path),
                    "file_valid": True,
                    "timestamp": "1970-01-01T00:00:00Z",
                    "generation_elapsed_ms": 1,
                },
            )
        )

    def fake_tick() -> LoopDecision:
        time.sleep(0.02)
        append_candidate(7)
        time.sleep(0.02)
        append_candidate(42)
        time.sleep(0.02)
        return LoopDecision(
            action=LoopAction.STOP,
            reason="requested_stop",
            next_state=RunRuntimeState.STOPPED,
        )

    monkeypatch.setattr(session.service, "tick", fake_tick)

    outputs = list(generate_seed_sweep_runtime(state, lock_rows, 0.10, 0.10))
    streamed_states = [output[0] for output in outputs]

    assert any(len(streamed.current_batch) >= 2 for streamed in streamed_states)
    assert any("Batch 1 LHS prompt ready: 0/5 seeds displayed" in streamed.notification for streamed in streamed_states)
    assert all(len(output[4:9]) == 5 for output in outputs)
    prompt_outputs = [output for output in outputs if "Batch 1 LHS prompt ready" in output[0].notification]
    assert prompt_outputs
    assert "LHS candidate prompt" in prompt_outputs[0][2]
    assert all(str(path).endswith("_pending.png") for path in prompt_outputs[0][4:9])
    assert any(output[4] and str(output[4]).endswith("cand_coord_001_7.png") for output in outputs)
    assert any(output[5] and str(output[5]).endswith("cand_coord_001_42.png") for output in outputs)
    assert any(output[6] and str(output[6]).endswith("seed_156_pending.png") for output in outputs)


def test_runtime_seed_sweep_flushes_fifth_seed_after_backend_completion(monkeypatch, tmp_path):
    monkeypatch.setenv("BC_GRADIO_FINAL_SEED_REFRESH_SECONDS", "0")
    state, _panel, _report, lock_rows, _generate_button, _status = prepare_runtime(monkeypatch, tmp_path)
    session = _RUNTIME_SESSIONS[state.run_id]

    def append_candidate(seed: int) -> None:
        coordinate_id = "coord_001"
        candidate_id = f"cand_{coordinate_id}_{seed}"
        image_path = tmp_path / f"{candidate_id}.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        session.service.store.append(
            PersistenceRecord(
                record_id=f"candidate:{candidate_id}",
                record_type="candidate_record",
                run_id=session.config.run.run_id,
                prompt_document_id=session.document.prompt_document_id,
                target_manifest_id=session.target_manifest.manifest_id,
                coordinate_id=coordinate_id,
                candidate_id=candidate_id,
                seed=seed,
                payload={
                    "candidate_id": candidate_id,
                    "run_id": session.config.run.run_id,
                    "prompt_document_id": session.document.prompt_document_id,
                    "target_manifest_id": session.target_manifest.manifest_id,
                    "coordinate_id": coordinate_id,
                    "seed": seed,
                    "rendered_prompt": session.rendered_prompt.rendered_prompt,
                    "generator_model_id": "stream-test",
                    "generator_backend": "stream-test",
                    "generation_settings": {},
                    "image_path": str(image_path),
                    "file_valid": True,
                    "timestamp": "1970-01-01T00:00:00Z",
                    "generation_elapsed_ms": 1,
                },
            )
        )

    def fake_tick() -> LoopDecision:
        for seed in [7, 42, 156, 8888, 42069]:
            append_candidate(seed)
        return LoopDecision(
            action=LoopAction.STOP,
            reason="requested_stop",
            next_state=RunRuntimeState.STOPPED,
        )

    monkeypatch.setattr(session.service, "tick", fake_tick)

    outputs = list(generate_seed_sweep_runtime(state, lock_rows, 0.10, 0.10))

    assert any(output[8] and str(output[8]).endswith("cand_coord_001_42069.png") for output in outputs)
    assert any("Batch 1 rendering: 5/5 seeds displayed" in output[0].notification for output in outputs)


def test_runtime_seed_sweep_streams_catalog_when_candidate_is_promoted(monkeypatch, tmp_path):
    monkeypatch.setenv("BC_GRADIO_STREAM_POLL_SECONDS", "0.005")
    state, _panel, _report, lock_rows, _generate_button, _status = prepare_runtime(monkeypatch, tmp_path)
    session = _RUNTIME_SESSIONS[state.run_id]
    coordinate_id = "coord_001"

    def append_candidate(seed: int) -> None:
        candidate_id = f"cand_{coordinate_id}_{seed}"
        image_path = tmp_path / f"{candidate_id}.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        session.service.store.append(
            PersistenceRecord(
                record_id=f"candidate:{candidate_id}",
                record_type="candidate_record",
                run_id=session.config.run.run_id,
                prompt_document_id=session.document.prompt_document_id,
                target_manifest_id=session.target_manifest.manifest_id,
                coordinate_id=coordinate_id,
                candidate_id=candidate_id,
                seed=seed,
                payload={
                    "candidate_id": candidate_id,
                    "run_id": session.config.run.run_id,
                    "prompt_document_id": session.document.prompt_document_id,
                    "target_manifest_id": session.target_manifest.manifest_id,
                    "coordinate_id": coordinate_id,
                    "seed": seed,
                    "rendered_prompt": session.rendered_prompt.rendered_prompt,
                    "generator_model_id": "stream-test",
                    "generator_backend": "stream-test",
                    "generation_settings": {},
                    "image_path": str(image_path),
                    "file_valid": True,
                    "timestamp": "1970-01-01T00:00:00Z",
                    "generation_elapsed_ms": 1,
                },
            )
        )

    def append_promoted_evaluation(seed: int) -> None:
        candidate_id = f"cand_{coordinate_id}_{seed}"
        session.service.store.append(
            PersistenceRecord(
                record_id=f"evaluation:{candidate_id}",
                record_type="image_evaluation",
                run_id=session.config.run.run_id,
                prompt_document_id=session.document.prompt_document_id,
                target_manifest_id=session.target_manifest.manifest_id,
                coordinate_id=coordinate_id,
                candidate_id=candidate_id,
                seed=seed,
                payload={
                    "quality": {"score": 0.9},
                    "alignment": {"score": 0.9},
                    "pass_flags": {"quality": True, "alignment": True, "full": True},
                    "disposition_signal": {
                        "class_name": "passes_thresholds",
                        "reasons": ["quality and alignment passed"],
                    },
                    "failure_types": [],
                    "evaluator_plan": {"quality_cutoff": 0.1, "alignment_cutoff": 0.1},
                },
            )
        )

    def fake_tick() -> LoopDecision:
        for seed in [7, 42, 156, 8888, 42069]:
            append_candidate(seed)
        time.sleep(0.03)
        append_promoted_evaluation(7)
        time.sleep(0.03)
        return LoopDecision(
            action=LoopAction.STOP,
            reason="requested_stop",
            next_state=RunRuntimeState.STOPPED,
        )

    monkeypatch.setattr(session.service, "tick", fake_tick)

    outputs = list(generate_seed_sweep_runtime(state, lock_rows, 0.10, 0.10))
    catalog_updates = [
        output[9]
        for output in outputs
        if isinstance(output[9], dict) and output[9].get("__type__") == "update"
    ]
    assert any(output[0].curated for output in outputs)
    assert any(output[0].curated and output[9].get("value") == [] for output in outputs if isinstance(output[9], dict))
    assert any(str(output[10]).endswith("cand_coord_001_7.png") for output in outputs)
    assert any("Curated catalog" in output[20] for output in outputs)
    assert any(
        any(str(path).endswith("cand_coord_001_7.png") for path, _caption in update.get("value", []))
        for update in catalog_updates
    )
    assert all(len(output) == 25 for output in outputs)


def test_runtime_cancel_requests_stop_without_discarding_active_session(monkeypatch, tmp_path):
    state, _panel, _report, _lock_rows, _generate_button, _status = prepare_runtime(monkeypatch, tmp_path)
    session = _RUNTIME_SESSIONS[state.run_id]
    active_state = state.model_copy(
        update={
            "batch_index": 1,
            "current_batch_expected_seeds": [7, 42, 156, 8888, 42069],
            "notification": "Batch 1 rendering.",
        }
    )

    next_state, panel_update, generate_update, status = cancel_pre_run_runtime(active_state)

    assert next_state.run_id in _RUNTIME_SESSIONS
    assert _RUNTIME_SESSIONS[next_state.run_id] is session
    assert session.service.stop_requested is True
    assert next_state.notification == "Stop requested. Finishing the current 5-seed batch before stopping."
    assert panel_update == {"__type__": "update"}
    assert generate_update["interactive"] is False
    assert "Stop requested" in status


def test_runtime_stop_request_stops_after_current_five_seed_batch(monkeypatch, tmp_path):
    state, _panel, _report, lock_rows, _generate_button, _status = prepare_runtime(monkeypatch, tmp_path)
    session = _RUNTIME_SESSIONS[state.run_id]
    original_tick = session.service.tick
    calls = 0

    def tick_then_request_stop():
        nonlocal calls
        calls += 1
        decision = original_tick()
        session.service.request_stop()
        return decision

    monkeypatch.setattr(session.service, "tick", tick_then_request_stop)

    outputs = list(generate_seed_sweep_runtime(state, lock_rows, 0.10, 0.10))
    final_state = outputs[-1][0]
    records = session.service.store.replay()
    transitions = [record for record in records if record.record_type == "loop_transition"]

    assert calls == 1
    assert final_state.generated_count == 5
    assert {candidate.coordinate_id for candidate in final_state.current_batch} == {"coord_001"}
    assert {candidate.coordinate_id for candidate in final_state.curated} == {"coord_001"}
    assert final_state.notification == "Stopped by backend/requested stop."
    assert not any("Batch 2 LHS prompt ready" in output[0].notification for output in outputs)
    assert transitions[-1].payload["reason"] == "gradio_runtime_cancel_requested"


def test_runtime_preview_gallery_shows_five_slots_before_any_seed_is_generated(monkeypatch, tmp_path):
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_RUN_ROOT", tmp_path)

    gallery = _preview_gallery(
        [],
        expected_seeds=[7, 42, 156, 8888, 42069],
        run_id="run_001",
        batch_index=1,
    )

    assert len(gallery) == 5
    assert all(path.endswith("_pending.png") for path, _caption in gallery)
    assert all("waiting for image" in caption for _path, caption in gallery)


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
    monkeypatch.setattr("bruteforce_canvas.gradio_ui.RUNTIME_LOOP_LIMIT_SECONDS", 6)
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
    assert "aspect-ratio: 3 / 4" in CSS
    assert "aspect-ratio: 4 / 3" not in CSS
    assert "aspect-ratio: 2 / 3" not in CSS
    assert "--bc-recorder-width: clamp(11rem, 14vw, 14rem)" in CSS
    assert "overflow: hidden !important" in CSS
    assert "animation: none !important" in CSS
    assert "display: none !important" in CSS
    assert '[data-testid="recording-waveform"]' in CSS


def test_workflow_accordion_uses_html_diagram_and_keeps_mermaid_reference():
    source = open("src/bruteforce_canvas/gradio_ui.py", encoding="utf-8").read()
    workflow_slice = source[
        source.index('elem_id="workflow-accordion"') : source.index('elem_classes=["bc-prompt-row"]')
    ]

    assert source.index("bc-title") < source.index('elem_id="workflow-accordion"')
    assert source.index('elem_id="workflow-accordion"') < source.index('elem_classes=["bc-prompt-row"]')
    assert source.index('elem_id="pre-run-panel"') < source.index('elem_id="active-panel"')
    assert source.index('elem_id="seed-slot-row"') < source.index('status = gr.HTML')
    assert source.index('status = gr.HTML') < source.index('elem_id="catalog-gallery"')
    assert 'gr.Accordion("Workflow diagram", open=True' in source
    assert "gr.HTML(" in workflow_slice
    assert "WORKFLOW_DIAGRAM_HTML" in workflow_slice
    assert "WORKFLOW_MERMAID_MARKDOWN" not in workflow_slice
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
    assert "bc-flow" in WORKFLOW_DIAGRAM_HTML
    assert "bc-flow-node" in WORKFLOW_DIAGRAM_HTML
    assert "bc-flow-gate" in WORKFLOW_DIAGRAM_HTML
    assert "bc-flow-edge" in WORKFLOW_DIAGRAM_HTML
    assert "bc-workflow-row" in CSS
    assert "#workflow-accordion" in CSS
    assert "#workflow-explanation *" in CSS
    assert "#workflow-diagram" in CSS
    assert ".bc-flow" in CSS
    assert ".bc-flow-node" in CSS
    assert ".bc-flow-gate" in CSS
    assert ".bc-flow-edge" in CSS
    for term in [
        "CohereLabs/cohere-transcribe-03-2026",
        "Mellum2 Thinking 12B",
        "BAAI/bge-small-en-v1.5",
        "prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
        "fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn",
        "openbmb/MiniCPM-V-4.6",
        "Jessylg27/tribev2-lite-qv",
    ]:
        assert term in WORKFLOW_DIAGRAM_HTML
    for term in [
        "Decomposition",
        "Repair/Verify",
        "Canonicalization",
        "CohereLabs/cohere-transcribe-03-2026",
        "Mellum2 Thinking 12B",
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


def test_pre_run_review_uses_compact_object_relation_object_triplets():
    source = open("src/bruteforce_canvas/gradio_ui.py", encoding="utf-8").read()

    assert "bc-triplet-row" in source
    assert "bc-triplet-object" in source
    assert "bc-triplet-relation" in source
    assert "<h3>Scene graph</h3>" in source
    assert "<h3>Objects</h3>" not in source
    assert "<h3>Relations</h3>" not in source
    assert ".bc-triplet-row {" in CSS
    assert "grid-template-columns: minmax(170px, 1fr) minmax(140px, 0.72fr) minmax(170px, 1fr);" in CSS


def test_pre_run_review_displays_color_as_descriptor_not_object_label():
    document = PromptDocumentSpec(
        raw_user_prompt="green ball on top of bench",
        graph=SceneGraphDraft(
            seed_prompt="green ball on top of bench",
            elements=[
                Element(
                    id="object_01",
                    label="green ball",
                    entity_type=EntityType.PRODUCT,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="green ball", category=EvidenceCategory.EXPLICIT),
                ),
                Element(
                    id="object_02",
                    label="bench",
                    entity_type=EntityType.FURNITURE,
                    role=ElementRole.SUPPORTING,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="bench", category=EvidenceCategory.EXPLICIT),
                ),
            ],
            relations=[
                RelationDescriptor(
                    id="rel_01",
                    source_id="object_01",
                    target_id="object_02",
                    relation_raw="on top of",
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="on top of", category=EvidenceCategory.EXPLICIT),
                )
            ],
        ),
        object_lane=ObjectLane(objects=[]),
        verification=VerificationReport(approved=True, issues=[]),
    )
    review = pre_run_modal_from_prompt(document)

    report = _review_markdown("green ball on top of bench", document, review, "Generate a green ball on top of a bench.")

    assert "<strong>ball</strong>" in report
    assert "color: green" in report
    assert "<strong>green ball</strong>" not in report


def test_lower_catalog_and_detail_components_use_high_contrast_surfaces():
    source = open("src/bruteforce_canvas/gradio_ui.py", encoding="utf-8").read()

    assert 'elem_id="catalog-gallery"' in source
    assert 'gr.Group(visible=False, elem_id="catalog-slot-panel")' in source
    assert 'elem_id="lock-table-panel"' in source
    assert "start_with_catalog_hidden" in source
    assert "cancel_with_catalog_hidden" in source
    assert 'elem_id="catalog-pagination"' in source
    assert 'elem_id="catalog-prev-page"' in source
    assert 'elem_id="catalog-page-status"' in source
    assert 'elem_id="catalog-next-page"' in source
    assert 'elem_id="catalog-slot-row"' in source
    assert 'elem_classes=["bc-catalog-slot"]' in source
    assert 'elem_id="detail-image"' in source
    assert 'elem_id="detail-report"' in source
    assert "#catalog-pagination {" in CSS
    assert "#catalog-page-status" in CSS
    assert "#catalog-slot-row {" in CSS
    assert "#catalog-slot-row .bc-catalog-slot img" in CSS
    assert "#catalog-gallery .thumbnail-item" in CSS
    assert "min-height: 184px !important;" in CSS
    assert "object-fit: cover !important;" in CSS
    assert "#detail-panel {" in CSS
    assert "#detail-image .label-wrap" in CSS
    assert "#detail-report .prose strong" in CSS
    assert "#detail-report .prose code" in CSS
    assert "color: var(--bc-ink) !important;" in CSS
    assert "background: #17211f !important;" in CSS


def test_generate_callback_does_not_refresh_selected_detail_pane():
    source = open("src/bruteforce_canvas/gradio_ui.py", encoding="utf-8").read()
    generate_slice = source[source.index("generate.click(") : source.index("for slot_index, catalog_slot")]
    cancel_slice = source[source.index("cancel.click(") : source.index("generate.click(")]

    assert "detail_panel" not in generate_slice
    assert "detail_image" not in generate_slice
    assert "detail_report" not in generate_slice
    assert "catalog_prev_btn" in generate_slice
    assert "catalog_next_btn" in generate_slice
    assert "catalog_page_status" in generate_slice
    assert "lock_table_panel" in generate_slice
    assert "queue=False" in cancel_slice


def test_catalog_slots_page_through_more_than_eight_images():
    curated = [
        SimCandidate(
            candidate_id=f"cand_{index}",
            seed=index,
            rendered_prompt="prompt",
            image_path=f"/tmp/cand_{index}.png",
            preview_path=f"/tmp/cand_{index}.png",
            display_path=f"/tmp/cand_{index}.png",
            quality_score=0.9,
            alignment_score=0.9,
            promoted=True,
            outcome="viable",
        )
        for index in range(10)
    ]
    state = initial_state().model_copy(update={"curated": curated})

    prev_update, next_update, label = _catalog_page_controls(state)
    assert _catalog_slot_paths(state)[0] == "/tmp/cand_0.png"
    assert _catalog_slot_paths(state)[7] == "/tmp/cand_7.png"
    assert prev_update["interactive"] is False
    assert next_update["interactive"] is True
    assert label == "Curated catalog: 1-8 of 10"

    moved_state, *updates = move_catalog_page(state, 1)
    page_slots = updates[:8]
    page_prev_update, page_next_update, page_label = updates[8:11]

    assert moved_state.catalog_page_index == 1
    assert page_slots[:2] == ["/tmp/cand_8.png", "/tmp/cand_9.png"]
    assert page_slots[2:] == [None, None, None, None, None, None]
    assert page_prev_update["interactive"] is True
    assert page_next_update["interactive"] is False
    assert page_label == "Curated catalog: 9-10 of 10"


def test_microphone_transcription_updates_prompt_without_loading_real_model(monkeypatch):
    class FakeTranscriber:
        def transcribe(self, audio):
            assert audio == "audio-value"
            return "a narrated glowing rose"

    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())
    _set_asr_runtime_status("idle", "test reset")

    try:
        prompt, state, status = transcribe_microphone_to_prompt("audio-value", "", initial_state())

        assert prompt == "a narrated glowing rose"
        assert state.raw_prompt == "a narrated glowing rose"
        assert "ASR transcript inserted" in status
    finally:
        _set_asr_runtime_status("idle", "test reset")


def test_microphone_transcription_streams_visible_processing_message(monkeypatch):
    class FakeTranscriber:
        def transcribe(self, audio):
            assert audio == "audio-value"
            return "a narrated glowing rose"

    monkeypatch.setattr("bruteforce_canvas.gradio_ui.default_transcriber", lambda: FakeTranscriber())
    _set_asr_runtime_status("idle", "test reset")

    try:
        steps = list(transcribe_microphone_to_prompt_steps("audio-value", "", initial_state()))

        assert steps[0][0]["value"] == "Transcribing audio..."
        assert steps[0][0]["interactive"] is False
        assert "Transcribing microphone audio" in steps[0][2]
        assert steps[-1][0]["value"] == "a narrated glowing rose"
        assert steps[-1][0]["interactive"] is True
    finally:
        _set_asr_runtime_status("idle", "test reset")


def test_generate_seed_sweep_yields_pending_then_evaluated_catalog():
    state, _panel, _report, lock_rows, _generate_button, _status = start_pre_run(
        "A magical glowing rose encased in glass",
        initial_state(),
    )

    outputs = list(generate_seed_sweep(state, lock_rows, 0.55, 0.25))

    pending_state = outputs[0][0]
    evaluated_state = outputs[-1][0]
    pending_slots = outputs[0][4:9]
    evaluated_slots = outputs[-1][4:9]
    catalog_gallery = outputs[-1][9]

    assert pending_state.generated_count == 5
    assert len(pending_slots) == 5
    assert all(path for path in pending_slots)
    assert len(evaluated_slots) == 5
    assert all(path for path in evaluated_slots)
    assert evaluated_state.iqa_evaluated_count == 5
    assert len(catalog_gallery) == len([candidate for candidate in evaluated_state.current_batch if candidate.promoted])
    assert any(str(path).endswith("_failed.png") for path in evaluated_slots)


def test_catalog_gallery_update_explicitly_repaints_promoted_items():
    state, _panel, _report, lock_rows, _generate_button, _status = start_pre_run(
        "A magical glowing rose encased in glass",
        initial_state(),
    )
    evaluated_state = list(generate_seed_sweep(state, lock_rows, 0.0, 0.0))[-1][0]

    update = _catalog_gallery_update(evaluated_state)

    assert update["__type__"] == "update"
    assert len(update["value"]) == len(evaluated_state.curated)
    assert all(path for path, _caption in update["value"])
    assert all("seed" in caption for _path, caption in update["value"])


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
    assert detail_panel["visible"] is False
    assert next_state.selected_candidate_id is None
