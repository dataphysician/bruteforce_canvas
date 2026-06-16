"""Tests for the Phase G real evaluator adapters.

The test suite focuses on :class:`JoyQualityAdapter` (Spec 04 §6.1) and
guarantees the four behaviors that downstream evaluators depend on:

1. The module is importable without the ``[ml]`` extras.
2. The static mode is deterministic across calls and processes.
3. The real mode reports a sensible ``is_available()`` answer and
   never crashes on missing dependencies.
4. Static scores are always within ``[0, 1]`` (with a small epsilon
   away from the literal boundaries to avoid cutoff edge cases).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

from bruteforce_canvas import real_adapters
from bruteforce_canvas.evaluation import AlignmentEvaluation, ImpactEvaluation, QualityEvaluation
from bruteforce_canvas.prompt import EvaluationTarget, EvaluationTargetManifest
from bruteforce_canvas.real_adapters import (
    JOYQUALITY_PRIMARY_MODEL_ID,
    MINICPM_V_MODEL_ID,
    TRIBE_V2_MODEL_ID,
    JoyQualityImageProcessor,
    JoyQualityAdapter,
    MiniCPMVAdapter,
    OpenAICompatibleVLMAlignmentAdapter,
    TRIBEv2Adapter,
)


def _fake_image_paths(tmp_path: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = tmp_path / f"image_{index:02d}.png"
        path.write_bytes(b"fake-png")
        paths.append(path)
    return paths


def _real_png_path(tmp_path: Path) -> Path:
    path = tmp_path / "real.png"
    Image.new("RGB", (32, 32), (180, 30, 30)).save(path)
    return path


# ---------------------------------------------------------------------------
# 1. Importability — must NOT require [ml] extras
# ---------------------------------------------------------------------------
def test_joyquality_module_is_importable() -> None:
    """``real_adapters`` imports cleanly even if torch/transformers are absent.

    The test passes when the module is importable, irrespective of
    whether the heavy ML stack happens to be installed. We assert on
    the public symbol rather than a side effect so the test stays
    cheap and order-independent.
    """

    assert real_adapters is not None
    assert hasattr(real_adapters, "JoyQualityAdapter")
    assert hasattr(real_adapters, "JOYQUALITY_PRIMARY_MODEL_ID")
    # Public symbols must remain importable from the top-level module.
    assert "JoyQualityAdapter" in real_adapters.__all__


def test_joyquality_module_does_not_import_torch_at_load_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing ``real_adapters`` must not import ``torch``/``transformers``.

    We block the heavy modules in :data:`sys.modules` *before* a
    forced re-import; if the module accidentally imports either one
    at top level, the block will trigger an ``ImportError`` and the
    test fails.
    """

    block = {"torch", "transformers"}
    saved = {name: sys.modules.get(name) for name in block}
    try:
        for name in block:
            monkeypatch.setitem(sys.modules, name, None)  # type: ignore[arg-type]
        # Force a fresh import. ImportError on the blocked modules
        # is expected and caught by the loader.
        for module_name in list(sys.modules):
            if module_name == "bruteforce_canvas.real_adapters" or module_name.startswith(
                "bruteforce_canvas.real_adapters."
            ):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        # The import below must succeed without consulting torch/transformers.
        from bruteforce_canvas import real_adapters as fresh  # noqa: WPS433
        assert fresh.JoyQualityAdapter is not None
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


# ---------------------------------------------------------------------------
# 2. Static mode — deterministic + in [0, 1]
# ---------------------------------------------------------------------------
def test_joyquality_static_mode_is_deterministic(tmp_path: Path) -> None:
    """Static mode returns identical scores for the same path across calls."""

    adapter = JoyQualityAdapter(mode="static")
    paths = _fake_image_paths(tmp_path, 4)

    first = adapter.evaluate(paths)
    second = adapter.evaluate(paths)
    third = adapter.evaluate(list(reversed(paths)))

    assert len(first) == len(paths)
    # Order-independent determinism: same set of paths ⇒ same multiset of scores.
    assert sorted(item.score for item in first) == sorted(item.score for item in second)
    assert sorted(item.score for item in first) == sorted(item.score for item in third)
    # Score objects must be of the documented type and shape.
    for evaluation in first:
        assert isinstance(evaluation, QualityEvaluation)
        assert evaluation.model_id == "static-joyquality"
        assert evaluation.confidence == "high"


def test_joyquality_static_mode_scores_in_unit_interval(tmp_path: Path) -> None:
    """Every static score is strictly within ``(0, 1)``."""

    adapter = JoyQualityAdapter(mode="static")
    paths = _fake_image_paths(tmp_path, 16)

    results = adapter.evaluate(paths)

    assert len(results) == len(paths)
    for evaluation in results:
        assert 0.0 < evaluation.score < 1.0, f"score {evaluation.score} not in (0, 1)"


def test_joyquality_static_prewarm_is_noop() -> None:
    """``prewarm`` in static mode must not touch the heavy ML stack."""

    adapter = JoyQualityAdapter(mode="static")
    # Must not raise even though torch/transformers may be absent.
    adapter.prewarm()
    adapter.prewarm()  # idempotent
    # No state should have been cached.
    assert adapter._model is None
    assert adapter._processor is None


def test_joyquality_real_prewarm_moves_model_to_resolved_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.to_calls: list[str] = []
            self.eval_calls = 0

        def to(self, device: str) -> "FakeModel":
            self.to_calls.append(device)
            return self

        def eval(self) -> None:
            self.eval_calls += 1

    model = FakeModel()
    adapter = JoyQualityAdapter(mode="real", device="cuda")

    monkeypatch.setattr(adapter, "_load_model", lambda: (model, object(), JOYQUALITY_PRIMARY_MODEL_ID, "test"))
    monkeypatch.setattr(adapter, "_resolve_device", lambda: "cuda")

    adapter.prewarm()

    assert model.to_calls == ["cuda"]
    assert model.eval_calls == 1
    assert adapter._resolved_device == "cuda"


def test_joyquality_real_prewarm_records_actual_parameter_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.device = "cpu"

    class FakeModel:
        def __init__(self) -> None:
            self.param = FakeTensor()

        def to(self, device: str) -> "FakeModel":
            self.param.device = "cuda:0" if device == "cuda" else device
            return self

        def eval(self) -> None:
            return None

        def parameters(self) -> object:
            return iter([self.param])

    model = FakeModel()
    adapter = JoyQualityAdapter(mode="real", device="cuda")

    monkeypatch.setattr(adapter, "_load_model", lambda: (model, object(), JOYQUALITY_PRIMARY_MODEL_ID, "test"))
    monkeypatch.setattr(adapter, "_resolve_device", lambda: "cuda")

    adapter.prewarm()

    assert adapter._resolved_device == "cuda:0"


def test_joyquality_real_prewarm_does_not_hide_failed_device_move(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def to(self, _device: str) -> "FakeModel":
            raise RuntimeError("simulated placement failure")

        def eval(self) -> None:
            return None

    adapter = JoyQualityAdapter(mode="real", device="cuda")

    monkeypatch.setattr(adapter, "_load_model", lambda: (FakeModel(), object(), JOYQUALITY_PRIMARY_MODEL_ID, "test"))
    monkeypatch.setattr(adapter, "_resolve_device", lambda: "cuda")

    with pytest.raises(RuntimeError, match="JoyQuality model could not be moved to 'cuda'"):
        adapter.prewarm()


def test_joyquality_real_prewarm_reports_cuda_oom_during_device_move(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def to(self, _device: str) -> "FakeModel":
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB.")

        def eval(self) -> None:
            return None

    adapter = JoyQualityAdapter(mode="real", device="cuda")

    monkeypatch.setattr(adapter, "_load_model", lambda: (FakeModel(), object(), JOYQUALITY_PRIMARY_MODEL_ID, "test"))
    monkeypatch.setattr(adapter, "_resolve_device", lambda: "cuda")

    with pytest.raises(RuntimeError, match="CUDA out of memory while loading IQA"):
        adapter.prewarm()


def test_joyquality_iqa_inputs_move_to_actual_model_device() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.to_calls: list[str] = []

        def to(self, device: str) -> "FakeTensor":
            self.to_calls.append(device)
            return self

    pixel_values = FakeTensor()

    moved = JoyQualityAdapter._move_tensor_mapping_to_device({"pixel_values": pixel_values, "metadata": "kept"}, "cuda:0")

    assert moved["pixel_values"] is pixel_values
    assert pixel_values.to_calls == ["cuda:0"]
    assert moved["metadata"] == "kept"


def test_joyquality_real_mode_scores_five_images_in_single_forward(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    paths: list[Path] = []
    for index in range(5):
        path = tmp_path / f"real_{index}.png"
        Image.new("RGB", (32, 32), (40 * index, 80, 120)).save(path)
        paths.append(path)

    class FakeProcessor:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(self, *, images: object, return_tensors: str) -> dict[str, object]:
            assert return_tensors == "pt"
            assert isinstance(images, list)
            self.batch_sizes.append(len(images))
            return {"pixel_values": torch.zeros((len(images), 3, 16, 16))}

    class FakeModel:
        def __init__(self) -> None:
            self.forward_calls = 0
            self.seen_shape: tuple[int, ...] | None = None

        def __call__(self, **inputs: object) -> object:
            self.forward_calls += 1
            pixel_values = inputs["pixel_values"]
            self.seen_shape = tuple(pixel_values.shape)
            return type("FakeOutput", (), {"logits": torch.tensor([[1.0], [0.0], [-1.0], [2.0], [-2.0]])})()

    processor = FakeProcessor()
    model = FakeModel()
    adapter = JoyQualityAdapter(mode="real", device="cpu")
    adapter._model = model
    adapter._processor = processor
    adapter._resolved_model_id = JOYQUALITY_PRIMARY_MODEL_ID
    adapter._resolved_model_version = "test"
    adapter._resolved_device = "cpu"

    results = adapter.evaluate(paths)

    assert processor.batch_sizes == [5]
    assert model.forward_calls == 1
    assert model.seen_shape == (5, 3, 16, 16)
    assert [round(result.score, 3) for result in results] == [0.731, 0.5, 0.269, 0.881, 0.119]


def test_joyquality_batched_logits_require_expected_batch_size() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="JoyQuality logit batch size mismatch"):
        JoyQualityAdapter._logits_to_scores(torch.ones((1, 1)), expected_count=5)


# ---------------------------------------------------------------------------
# 3. Real mode — availability guard
# ---------------------------------------------------------------------------
def test_joyquality_real_mode_availability_guard() -> None:
    """``is_available`` returns a bool and reflects dep presence sensibly.

    The test is skipped automatically when the heavy dependencies are
    missing so that CI on a minimal image still runs.
    """

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    adapter = JoyQualityAdapter(mode="real", device="auto")
    available = adapter.is_available()
    assert isinstance(available, bool)
    # If the heavy stack is installed, the adapter must be available.
    assert available is True


def test_joyquality_real_mode_rejects_invalid_arguments() -> None:
    """The constructor validates ``mode`` and ``device`` literals."""

    with pytest.raises(ValueError):
        JoyQualityAdapter(mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        JoyQualityAdapter(device="tpu")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. Public model-id constants — protects the spec contract
# ---------------------------------------------------------------------------
def test_joyquality_model_id_constants_match_spec() -> None:
    """The primary model id matches the documented reference."""

    assert JOYQUALITY_PRIMARY_MODEL_ID == "fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn"


# ---------------------------------------------------------------------------
# 5. Real IQA provenance — no static/model/processor fallback in real mode
# ---------------------------------------------------------------------------
def test_joyquality_real_mode_raises_when_primary_model_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing real IQA load must not degrade to static/fallback scores."""

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated network failure")

    from transformers import AutoModelForImageClassification

    monkeypatch.setattr(AutoModelForImageClassification, "from_pretrained", _explode)

    adapter = JoyQualityAdapter(mode="real", device="cpu")
    paths = _fake_image_paths(tmp_path, 2)

    with pytest.raises(RuntimeError, match="simulated network failure"):
        adapter.evaluate(paths)


def test_joyquality_real_mode_uses_only_primary_model_and_internal_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVisionConfig:
        image_size = 384

    class FakeConfig:
        vision_config = FakeVisionConfig()

    class FakeModel:
        config = FakeConfig()

        def to(self, _device: str) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

    class FakeModelLoader:
        calls: list[str] = []

        @classmethod
        def from_pretrained(cls, model_id: str) -> FakeModel:
            cls.calls.append(model_id)
            return FakeModel()

    monkeypatch.setattr("transformers.AutoModelForImageClassification", FakeModelLoader)

    adapter = JoyQualityAdapter(mode="real", device="cpu")
    adapter.prewarm()

    assert FakeModelLoader.calls == [JOYQUALITY_PRIMARY_MODEL_ID]
    assert isinstance(adapter._processor, JoyQualityImageProcessor)
    assert adapter._processor.size == 384
    assert adapter._resolved_model_id == JOYQUALITY_PRIMARY_MODEL_ID


# ---------------------------------------------------------------------------
# MiniCPMVAdapter (Phase G, real VLM alignment evaluator)
# ---------------------------------------------------------------------------
def _sample_manifest(rendered_prompt: str = "A ceramic bowl resting on a wooden table") -> EvaluationTargetManifest:
    """Build a minimal :class:`EvaluationTargetManifest` for unit tests."""

    return EvaluationTargetManifest(
        manifest_id="eval_manifest_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        coordinate_id="coord_001",
        rendered_prompt=rendered_prompt,
        targets=[
            EvaluationTarget(
                target_id="subject.object",
                target_kind="element",
                label="ceramic bowl",
                priority="locked_required",
                lhs_policy="fixed",
                evaluation_policy="must_match",
            ),
            EvaluationTarget(
                target_id="subject.relation",
                target_kind="relation",
                priority="locked_required",
                lhs_policy="fixed",
                evaluation_policy="must_match",
            ),
        ],
    )


def test_minicpmv_module_is_importable() -> None:
    """``MiniCPMVAdapter`` is importable and exported from ``real_adapters``."""

    assert hasattr(real_adapters, "MiniCPMVAdapter")
    assert "MiniCPMVAdapter" in real_adapters.__all__
    assert MINICPM_V_MODEL_ID == "openbmb/MiniCPM-V-4.6"
    # The class can be imported by name from the module.
    from bruteforce_canvas.real_adapters import MiniCPMVAdapter as Imported  # noqa: WPS433

    assert Imported is MiniCPMVAdapter


def test_minicpmv_static_mode_returns_alignment_evaluation(tmp_path: Path) -> None:
    """Static mode returns one :class:`AlignmentEvaluation` per input image.

    The returned objects must match the schema from
    :mod:`bruteforce_canvas.evaluation` and identify the real model id
    so downstream telemetry is consistent across modes.
    """

    adapter = MiniCPMVAdapter(mode="static")
    paths = _fake_image_paths(tmp_path, 3)
    manifest = _sample_manifest()

    results = adapter.evaluate(paths, prompt="ceramic bowl on table", manifest=manifest)

    assert len(results) == 3
    for evaluation, path in zip(results, paths, strict=True):
        assert isinstance(evaluation, AlignmentEvaluation)
        assert evaluation.model_id == MINICPM_V_MODEL_ID
        assert evaluation.confidence == "high"
        # The static score is a deterministic function of the path; the
        # same path on the same manifest/prompt must yield the same score.
        again = adapter.evaluate([path], prompt="ceramic bowl on table", manifest=manifest)
        assert again[0].score == evaluation.score


def test_minicpmv_static_mode_scores_in_unit_interval(tmp_path: Path) -> None:
    """Every static score is strictly within ``(0, 1)`` (with a small epsilon)."""

    adapter = MiniCPMVAdapter(mode="static")
    paths = _fake_image_paths(tmp_path, 24)
    manifest = _sample_manifest(rendered_prompt="A wooden chair beside a window")

    results = adapter.evaluate(
        paths,
        prompt="wooden chair by a window, soft morning light",
        manifest=manifest,
    )

    assert len(results) == len(paths)
    for evaluation in results:
        assert 0.0 < evaluation.score < 1.0, f"score {evaluation.score} not in (0, 1)"


def test_minicpmv_real_mode_availability_guard() -> None:
    """Real-mode ``is_available`` returns a bool; auto-skips without deps.

    When the heavy ML stack is installed the adapter must report
    available; when it is not, the test is skipped so it does not
    gate the CI pipeline.
    """

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    adapter = MiniCPMVAdapter(mode="real", device="auto")
    available = adapter.is_available()
    assert isinstance(available, bool)
    assert available is True


def test_minicpmv_static_prewarm_is_noop(tmp_path: Path) -> None:
    """Static prewarm is a no-op and does not touch the heavy ML stack."""

    adapter = MiniCPMVAdapter(mode="static")
    adapter.prewarm()
    adapter.prewarm()  # idempotent
    assert adapter._model is None
    assert adapter._tokenizer is None
    # Evaluate still works after a static prewarm.
    results = adapter.evaluate(
        _fake_image_paths(tmp_path, 1),
        prompt="anything",
        manifest=_sample_manifest(),
    )
    assert len(results) == 1
    assert isinstance(results[0], AlignmentEvaluation)


def test_minicpmv_real_mode_uses_image_text_to_text_generate(tmp_path: Path) -> None:
    class FakeInputs(dict):
        input_ids = [[1, 2]]

        def to(self, device: str) -> "FakeInputs":
            self["device"] = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.messages = None
            self.decoded = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return FakeInputs(input_ids=self.__class__.input_ids if hasattr(self.__class__, "input_ids") else [[1, 2]])

        def batch_decode(self, generated_ids, **kwargs):
            self.decoded = generated_ids
            return ['{"score": 0.73, "reason": "visible cup"}']

    class FakeModel:
        device = "cuda"

        def __init__(self) -> None:
            self.generate_kwargs = None

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return [[1, 2, 3, 4]]

    processor = FakeProcessor()
    model = FakeModel()
    adapter = MiniCPMVAdapter(mode="real", device="cuda")
    adapter._model = model
    adapter._processor = processor
    adapter._tokenizer = processor
    adapter._resolved_device = "cuda"

    result = adapter.evaluate([_real_png_path(tmp_path)], "red cup", _sample_manifest())[0]

    assert result.score == 0.73
    assert processor.messages[0]["content"][0]["type"] == "image"
    assert processor.messages[0]["content"][1]["type"] == "text"
    assert model.generate_kwargs["downsample_mode"] == "16x"
    assert model.generate_kwargs["max_new_tokens"] == 128


def test_minicpmv_load_model_falls_back_to_auto_model_for_remote_code_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    calls: dict[str, object] = {}

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls["processor_model_id"] = model_id
            calls["processor_kwargs"] = kwargs
            return object()

    class BrokenImageTextToTextFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            calls["image_text_to_text_model_id"] = model_id
            calls["image_text_to_text_kwargs"] = kwargs
            raise ValueError("Transformers does not recognize model type `minicpmv4_6`")

    class FakeModel:
        def __init__(self) -> None:
            self.device = None
            self.eval_called = False

        def to(self, device: str) -> "FakeModel":
            self.device = device
            return self

        def eval(self) -> None:
            self.eval_called = True

    class FallbackAutoModelFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            calls["fallback_model_id"] = model_id
            calls["fallback_kwargs"] = kwargs
            return FakeModel()

    class FakeTransformers:
        AutoProcessor = FakeProcessorFactory
        AutoModelForImageTextToText = BrokenImageTextToTextFactory
        AutoModel = FallbackAutoModelFactory

    monkeypatch.setattr(real_adapters, "_import_transformers_module", lambda: FakeTransformers)

    adapter = MiniCPMVAdapter(mode="real", device="cpu")
    model, processor, resolved = adapter._load_model()

    assert calls["image_text_to_text_model_id"] == MINICPM_V_MODEL_ID
    assert calls["fallback_model_id"] == MINICPM_V_MODEL_ID
    assert calls["processor_model_id"] == MINICPM_V_MODEL_ID
    assert calls["fallback_kwargs"]["trust_remote_code"] is True
    assert model.device == "cpu"
    assert model.eval_called is True
    assert processor is not None
    assert resolved == "cpu"


def test_minicpmv_real_mode_raises_on_malformed_model_response(tmp_path: Path) -> None:
    class FakeInputs(dict):
        input_ids = [[1, 2]]

        def to(self, _device: str) -> "FakeInputs":
            return self

    class FakeProcessor:
        def apply_chat_template(self, *_args, **_kwargs):
            return FakeInputs(input_ids=[[1, 2]])

        def batch_decode(self, *_args, **_kwargs):
            return ["no numeric score"]

    class FakeModel:
        device = "cpu"

        def generate(self, **_kwargs):
            return [[1, 2, 3]]

    adapter = MiniCPMVAdapter(mode="real", device="cpu")
    adapter._model = FakeModel()
    adapter._processor = FakeProcessor()
    adapter._tokenizer = adapter._processor
    adapter._resolved_device = "cpu"

    with pytest.raises(RuntimeError, match="parseable alignment score"):
        adapter.evaluate([_real_png_path(tmp_path)], "red cup", _sample_manifest())


def test_minicpmv_real_mode_rejects_invalid_arguments() -> None:
    """The constructor validates ``mode`` and ``device`` literals."""

    with pytest.raises(ValueError):
        MiniCPMVAdapter(mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MiniCPMVAdapter(device="tpu")  # type: ignore[arg-type]


def test_minicpmv_evaluate_validates_inputs(tmp_path: Path) -> None:
    """``evaluate`` rejects malformed inputs with a clear ``TypeError``."""

    adapter = MiniCPMVAdapter(mode="static")
    paths = _fake_image_paths(tmp_path, 1)
    manifest = _sample_manifest()

    with pytest.raises(TypeError):
        adapter.evaluate("not a list", prompt="x", manifest=manifest)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        adapter.evaluate(paths, prompt=123, manifest=manifest)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        adapter.evaluate(paths, prompt="x", manifest={"not": "a manifest"})  # type: ignore[arg-type]


def test_minicpmv_evaluate_empty_images_returns_empty_list() -> None:
    """An empty image list yields an empty result list without errors."""

    adapter = MiniCPMVAdapter(mode="static")
    results = adapter.evaluate([], prompt="x", manifest=_sample_manifest())
    assert results == []


def test_openai_compatible_vlm_alignment_adapter_posts_multimodal_structured_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(url: str, payload: dict, *, api_key: str | None, timeout_seconds: float) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        captured["api_key"] = api_key
        captured["timeout_seconds"] = timeout_seconds
        return {"choices": [{"message": {"content": '{"score": 0.81, "reason": "aligned"}'}}]}

    monkeypatch.setattr("bruteforce_canvas.real_adapters._post_json", fake_post_json)
    image_path = _real_png_path(tmp_path)
    adapter = OpenAICompatibleVLMAlignmentAdapter(
        base_url="https://vlm.example.test/v1",
        model="remote-minicpm",
        api_key="secret",
        timeout_seconds=11,
    )

    result = adapter.evaluate([image_path], "red cup", _sample_manifest())[0]

    assert result.score == 0.81
    assert captured["url"] == "https://vlm.example.test/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert captured["timeout_seconds"] == 11
    payload = captured["payload"]
    assert payload["model"] == "remote-minicpm"
    assert payload["response_format"]["type"] == "json_schema"
    content = payload["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# TRIBEv2Adapter (Phase G, real impact evaluator)
# ---------------------------------------------------------------------------
def test_TRIBEv2_adapter_is_importable() -> None:
    """``TRIBEv2Adapter`` is importable and exported from ``real_adapters``."""

    assert hasattr(real_adapters, "TRIBEv2Adapter")
    assert "TRIBEv2Adapter" in real_adapters.__all__
    assert TRIBE_V2_MODEL_ID == "Jessylg27/tribev2-lite-qv"
    from bruteforce_canvas.real_adapters import TRIBEv2Adapter as Imported  # noqa: WPS433

    assert Imported is TRIBEv2Adapter


def test_TRIBEv2_adapter_is_disabled_by_default() -> None:
    """Default construction leaves the adapter disabled and no model loaded.

    ``evaluate`` must return an empty list and ``prewarm`` must be a
    no-op when ``enabled=False`` so that disabled impact scoring never
    triggers any model load or download.
    """

    adapter = TRIBEv2Adapter()
    assert adapter.enabled is False
    assert adapter.mode == "static"
    assert adapter.device == "auto"

    images = [Path("/tmp/bruteforce_canvas_disabled_a.png"), Path("/tmp/bruteforce_canvas_disabled_b.png")]
    assert adapter.evaluate(images) == []

    adapter.prewarm()
    adapter.prewarm()  # idempotent
    assert adapter._prewarmed is False
    assert adapter._pipeline is None


def test_TRIBEv2_static_mode_returns_deterministic_per_path_evaluations(tmp_path: Path) -> None:
    """Static mode returns one deterministic :class:`ImpactEvaluation` per path.

    The score is keyed off the path string (SHA-256-derived), so the
    same path must always produce the same evaluation, different paths
    must produce different scores, and the result type must be
    :class:`ImpactEvaluation` with ``informational_only=True``.
    """

    adapter = TRIBEv2Adapter(enabled=True, mode="static")
    paths = [
        tmp_path / "tribev2_alpha.png",
        tmp_path / "tribev2_beta.png",
        tmp_path / "tribev2_alpha.png",
    ]

    first = adapter.evaluate(paths)
    again = adapter.evaluate(paths)

    assert len(first) == 3
    assert all(isinstance(item, ImpactEvaluation) for item in first)
    assert all(item.informational_only is True for item in first)
    assert all(item.confidence == "high" for item in first)
    assert all(0.0 < item.score < 1.0 for item in first)
    assert first[0].model_id == "static-tribev2-evaluator"

    assert first == again
    assert first[0] == first[2]
    assert first[0].score != first[1].score


def test_TRIBEv2_static_mode_handles_string_paths_and_empty_input() -> None:
    """Static mode accepts ``str`` paths and returns ``[]`` for an empty list."""

    adapter = TRIBEv2Adapter(enabled=True, mode="static")

    assert adapter.evaluate([]) == []
    string_results = adapter.evaluate(["/tmp/bruteforce_canvas_string.png"])
    assert len(string_results) == 1
    assert isinstance(string_results[0], ImpactEvaluation)
    assert 0.0 < string_results[0].score < 1.0


def test_TRIBEv2_real_mode_availability_guard() -> None:
    """Real mode is available iff the heavy ML stack is importable.

    The test is auto-skipped when ``torch``/``transformers`` are not
    installed so the suite remains green on minimal images. When the
    deps are present, ``is_available`` must return ``True`` and
    ``prewarm`` must be a safe no-op when the model weights cannot be
    reached (e.g. on offline CI machines).
    """

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    adapter = TRIBEv2Adapter(enabled=True, mode="real", device="auto")
    assert adapter.is_available() is True

    adapter.prewarm()
    adapter.prewarm()  # idempotent — must not raise on a second call
    # ``_prewarmed`` flips to True only when the pipeline actually
    # cached; the guard is correct either way because ``prewarm``
    # swallows infrastructure failures rather than raising.
    assert isinstance(adapter._prewarmed, bool)
    if adapter._prewarmed:
        assert adapter._pipeline is not None


def test_TRIBEv2_real_mode_disabled_short_circuits_evaluate() -> None:
    """Even with ``mode='real'`` set, ``enabled=False`` returns ``[]``."""

    adapter = TRIBEv2Adapter(enabled=False, mode="real")
    assert adapter.evaluate([Path("/tmp/bruteforce_canvas_x.png")]) == []
    adapter.prewarm()
    assert adapter._prewarmed is False
    assert adapter._pipeline is None


def test_TRIBEv2_constructor_rejects_invalid_arguments() -> None:
    """The constructor validates ``mode`` and ``device`` literals."""

    with pytest.raises(ValueError):
        TRIBEv2Adapter(mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TRIBEv2Adapter(device="tpu")  # type: ignore[arg-type]
