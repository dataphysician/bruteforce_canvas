import sys
import json
from pathlib import Path

import pytest

from bruteforce_canvas import generation as generation_module
from bruteforce_canvas.generation import (
    BonsaiHttpAdapter,
    BonsaiHttpConfig,
    BonsaiTernaryAdapter,
    BonsaiTernaryConfig,
    BonsaiRuntimeError,
    GenerationRequest,
    GenerationSettings,
    validate_image_file,
)


class FakePipeline:
    def __init__(self) -> None:
        self.prewarm_count = 0
        self.calls = []

    def prewarm(self) -> None:
        self.prewarm_count += 1

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("FakeBonsaiResult", (), {"images": [FakeImage()]})()


class FakeImage:
    def save(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\x89PNG\r\n\x1a\n")


class FakeFactory:
    def __init__(self) -> None:
        self.pipeline = FakePipeline()
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.pipeline


class FakeHttpResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.data


def request(tmp_path: Path) -> GenerationRequest:
    return GenerationRequest(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seed=7,
        rendered_prompt="Generate a ceramic bowl on wooden table",
        generation_settings=GenerationSettings(),
        image_path=str(tmp_path / "seed_7.png"),
        generator_model_id="prism-ml/bonsai-image-ternary-4B-gemlite-2bit",
        generator_backend="bonsai-ternary-gemlite",
    )


def _clear_backend_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "backend_gpu" or module_name.startswith("backend_gpu."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def _write_backend(root: Path, server_source: str) -> None:
    backend_pkg = root / "backend_gpu"
    backend_pkg.mkdir(parents=True)
    (backend_pkg / "__init__.py").write_text("", encoding="utf-8")
    (backend_pkg / "server.py").write_text(server_source, encoding="utf-8")


def test_bonsai_adapter_constructs_pipeline_with_model_id(tmp_path: Path, monkeypatch):
    factory = FakeFactory()
    config = BonsaiTernaryConfig(
        model_root=tmp_path / "model",
        triton_cache_dir=tmp_path / ".triton",
        smoke_output_dir=tmp_path / "smoke",
    )
    adapter = BonsaiTernaryAdapter(config=config, pipeline_factory=factory)

    adapter.prewarm()

    assert factory.kwargs == {"model_id": config.model_id}
    assert factory.pipeline.prewarm_count == 1
    assert len(factory.pipeline.calls) == 1
    assert factory.pipeline.calls[0]["prompt"] == config.warmup_prompt
    assert factory.pipeline.calls[0]["num_inference_steps"] == 4
    assert monkeypatch.context is not None


def test_bonsai_adapter_generates_candidate_and_reuses_resident_pipeline(tmp_path: Path):
    factory = FakeFactory()
    adapter = BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(
            model_root=tmp_path / "model",
            triton_cache_dir=tmp_path / ".triton",
            smoke_output_dir=tmp_path / "smoke",
        ),
        pipeline_factory=factory,
    )

    first = adapter.generate(request(tmp_path))
    second = adapter.generate(request(tmp_path))

    assert first.candidate.file_valid is True
    assert second.candidate.file_valid is True
    assert factory.pipeline.prewarm_count == 1
    assert len(factory.pipeline.calls) == 3
    assert factory.pipeline.calls[0]["prompt"] == adapter.config.warmup_prompt
    assert factory.pipeline.calls[1]["prompt"].startswith("Generate ")
    assert factory.pipeline.calls[1]["num_inference_steps"] == 4
    assert factory.pipeline.calls[1]["guidance_scale"] == 1.0


def test_bonsai_http_adapter_generates_candidate_from_backend_png(tmp_path: Path, monkeypatch):
    calls = []

    def fake_urlopen(http_request, timeout):
        calls.append((http_request, timeout))
        if http_request.full_url.endswith("/healthz"):
            return FakeHttpResponse(b'{"status":"ok"}')
        return FakeHttpResponse(b"\x89PNG\r\n\x1a\nhttp-png")

    monkeypatch.setattr(generation_module.urllib.request, "urlopen", fake_urlopen)
    adapter = BonsaiHttpAdapter(
        config=BonsaiHttpConfig(
            base_url="http://127.0.0.1:7950",
            token="test-token",
            kernel_warmup=False,
            timeout_seconds=12,
        )
    )

    result = adapter.generate(request(tmp_path))

    assert result.candidate.file_valid is True
    assert Path(result.candidate.image_path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(calls) == 2
    health_request, health_timeout = calls[0]
    generate_request, generate_timeout = calls[1]
    assert health_request.full_url == "http://127.0.0.1:7950/healthz"
    assert generate_request.full_url == "http://127.0.0.1:7950/generate"
    assert health_timeout == 12
    assert generate_timeout == 12
    assert generate_request.get_header("Authorization") == "Bearer test-token"
    payload = json.loads(generate_request.data.decode("utf-8"))
    assert payload == {
        "prompt": "Generate a ceramic bowl on wooden table",
        "seed": 7,
        "steps": 4,
        "guidance": 1.0,
        "backend": "bonsai-ternary-gemlite",
        "height": 512,
        "width": 512,
    }


def test_bonsai_kernel_warmup_can_be_disabled(tmp_path: Path):
    factory = FakeFactory()
    adapter = BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(
            model_root=tmp_path / "model",
            triton_cache_dir=tmp_path / ".triton",
            smoke_output_dir=tmp_path / "smoke",
            kernel_warmup=False,
        ),
        pipeline_factory=factory,
    )

    adapter.prewarm()

    assert factory.pipeline.prewarm_count == 1
    assert factory.pipeline.calls == []


def test_bonsai_kernel_warmup_uses_configured_prompt_and_dimensions(tmp_path: Path):
    factory = FakeFactory()
    adapter = BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(
            model_root=tmp_path / "model",
            triton_cache_dir=tmp_path / ".triton",
            smoke_output_dir=tmp_path / "smoke",
            warmup_prompt="a calibration object",
            warmup_steps=2,
            warmup_height=256,
            warmup_width=384,
        ),
        pipeline_factory=factory,
    )

    adapter.prewarm()

    assert factory.pipeline.calls == [
        {
            "prompt": "Generate a calibration object",
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 256,
            "width": 384,
        }
    ]


def test_bonsai_backend_import_succeeds_when_backend_gpu_importable_normally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_backend_modules(monkeypatch)
    _write_backend(
        tmp_path,
        "def build_pipeline(**kwargs):\n"
        "    return {'built': kwargs}\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    adapter = BonsaiTernaryAdapter(config=BonsaiTernaryConfig(smoke_output_dir=tmp_path / "smoke"))

    factory = adapter._default_pipeline_factory()

    assert factory(model_id="model") == {"built": {"model_id": "model"}}


def test_bonsai_backend_import_succeeds_from_configured_external_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_backend_modules(monkeypatch)
    backend_root = tmp_path / "Bonsai-Image-Demo"
    _write_backend(
        backend_root,
        "def build_pipeline(**kwargs):\n"
        "    return {'built': kwargs}\n",
    )
    adapter = BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(
            backend_root=backend_root,
            smoke_output_dir=tmp_path / "smoke",
        )
    )

    factory = adapter._default_pipeline_factory()

    assert sys.path[0] == str(backend_root)
    assert factory(model_id="model") == {"built": {"model_id": "model"}}


def test_bonsai_backend_import_fails_clearly_when_root_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_backend_modules(monkeypatch)
    real_import_module = generation_module.importlib.import_module

    def missing_backend(name: str):
        if name == "backend_gpu.server":
            raise ModuleNotFoundError("No module named 'backend_gpu'", name="backend_gpu")
        return real_import_module(name)

    monkeypatch.setattr(generation_module.importlib, "import_module", missing_backend)
    adapter = BonsaiTernaryAdapter(config=BonsaiTernaryConfig(smoke_output_dir=tmp_path / "smoke"))

    with pytest.raises(BonsaiRuntimeError, match="BC_BONSAI_BACKEND_ROOT is not set"):
        adapter._default_pipeline_factory()


def test_bonsai_backend_import_fails_clearly_when_root_is_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_backend_modules(monkeypatch)
    real_import_module = generation_module.importlib.import_module

    def missing_backend(name: str):
        if name == "backend_gpu.server":
            raise ModuleNotFoundError("No module named 'backend_gpu'", name="backend_gpu")
        return real_import_module(name)

    monkeypatch.setattr(generation_module.importlib, "import_module", missing_backend)
    adapter = BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(
            backend_root=tmp_path,
            smoke_output_dir=tmp_path / "smoke",
        )
    )

    with pytest.raises(BonsaiRuntimeError, match="does not contain backend_gpu/server.py"):
        adapter._default_pipeline_factory()


def test_bonsai_backend_import_requires_build_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_backend_modules(monkeypatch)
    _write_backend(tmp_path, "NOT_BUILD_PIPELINE = object()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    adapter = BonsaiTernaryAdapter(config=BonsaiTernaryConfig(smoke_output_dir=tmp_path / "smoke"))

    with pytest.raises(BonsaiRuntimeError, match="callable build_pipeline"):
        adapter._default_pipeline_factory()


def test_bonsai_is_available_checks_pipeline_and_output_access(tmp_path: Path) -> None:
    adapter = BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(
            model_root=tmp_path / "model",
            triton_cache_dir=tmp_path / ".triton",
            smoke_output_dir=tmp_path / "smoke",
        ),
        pipeline_factory=FakeFactory(),
    )

    assert adapter.is_available(output_dir=tmp_path / "writable") is True

    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("file", encoding="utf-8")
    assert adapter.is_available(output_dir=not_a_dir) is False


def test_bonsai_is_available_checks_required_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    adapter = BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(
            device="cuda",
            smoke_output_dir=tmp_path / "smoke",
        ),
        pipeline_factory=FakeFactory(),
    )

    assert adapter.is_available(output_dir=tmp_path / "writable") is False


def test_bonsai_smoke_test_generates_valid_png(tmp_path: Path) -> None:
    adapter = BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(
            model_root=tmp_path / "model",
            triton_cache_dir=tmp_path / ".triton",
            smoke_output_dir=tmp_path / "smoke",
        ),
        pipeline_factory=FakeFactory(),
    )

    result = adapter.smoke_test(output_dir=tmp_path / "smoke", steps=4, height=512, width=512)

    assert result.file_valid is True
    assert result.elapsed_ms >= 0
    assert validate_image_file(result.output_path).valid is True
