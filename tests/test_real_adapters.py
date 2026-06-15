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

from bruteforce_canvas import real_adapters
from bruteforce_canvas.evaluation import AlignmentEvaluation, ImpactEvaluation, QualityEvaluation
from bruteforce_canvas.prompt import EvaluationTarget, EvaluationTargetManifest
from bruteforce_canvas.real_adapters import (
    JOYQUALITY_PRIMARY_MODEL_ID,
    JOYQUALITY_PROCESSOR_MODEL_ID,
    MINICPM_V_MODEL_ID,
    TRIBE_V2_MODEL_ID,
    JoyQualityAdapter,
    MiniCPMVAdapter,
    TRIBEv2Adapter,
    _load_joyquality_processor,
)


def _fake_image_paths(tmp_path: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = tmp_path / f"image_{index:02d}.png"
        path.write_bytes(b"fake-png")
        paths.append(path)
    return paths


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


def test_joyquality_processor_loader_falls_back_to_siglip2_base_processor() -> None:
    class Loader:
        def __init__(self, name: str, available_model_id: str | None = None) -> None:
            self.name = name
            self.available_model_id = available_model_id
            self.calls: list[str] = []

        def from_pretrained(self, model_id: str) -> str:
            self.calls.append(model_id)
            if model_id != self.available_model_id:
                raise OSError(f"{model_id} missing processor")
            return f"{self.name}:{model_id}"

    auto_processor = Loader("processor")
    auto_image_processor = Loader("image_processor", JOYQUALITY_PROCESSOR_MODEL_ID)

    processor = _load_joyquality_processor(auto_processor, auto_image_processor)

    assert processor == f"image_processor:{JOYQUALITY_PROCESSOR_MODEL_ID}"
    assert auto_processor.calls == [JOYQUALITY_PRIMARY_MODEL_ID, JOYQUALITY_PROCESSOR_MODEL_ID]
    assert auto_image_processor.calls == [JOYQUALITY_PRIMARY_MODEL_ID, JOYQUALITY_PROCESSOR_MODEL_ID]


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
    assert JOYQUALITY_PROCESSOR_MODEL_ID == "google/siglip2-so400m-patch16-512"


# ---------------------------------------------------------------------------
# 5. Defensive fallback — real mode that fails still returns a result
# ---------------------------------------------------------------------------
def test_joyquality_real_mode_falls_back_to_static_when_model_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing model load must degrade to deterministic scores, not raise.

    We simulate a broken ``from_pretrained`` call so the real-mode
    branch falls into the catch-all and produces a static result list.
    """

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated network failure")

    # Patch the auto-loader used by the adapter; both primary and
    # fallback attempts will raise, exercising the inner ``except``.
    from transformers import AutoModelForImageClassification, AutoProcessor

    monkeypatch.setattr(AutoModelForImageClassification, "from_pretrained", _explode)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", _explode)

    adapter = JoyQualityAdapter(mode="real", device="cpu")
    paths = _fake_image_paths(tmp_path, 2)
    results = adapter.evaluate(paths)

    assert len(results) == 2
    for evaluation in results:
        assert isinstance(evaluation, QualityEvaluation)
        # Static-fallback path uses the deterministic model id.
        assert evaluation.model_id == "static-joyquality"
        assert 0.0 < evaluation.score < 1.0


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
