from pathlib import Path

from bruteforce_canvas.generation import (
    BonsaiTernaryAdapter,
    BonsaiTernaryConfig,
    GenerationRequest,
    GenerationSettings,
)


class FakePipeline:
    def __init__(self) -> None:
        self.prewarm_count = 0
        self.calls = []

    def prewarm(self) -> None:
        self.prewarm_count += 1

    def generate(self, **kwargs) -> None:
        self.calls.append(kwargs)
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"\x89PNG\r\n\x1a\n")


class FakeFactory:
    def __init__(self) -> None:
        self.pipeline = FakePipeline()
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.pipeline


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


def test_bonsai_adapter_constructs_pipeline_with_required_paths_and_backend(tmp_path: Path, monkeypatch):
    factory = FakeFactory()
    config = BonsaiTernaryConfig(model_root=tmp_path / "model", triton_cache_dir=tmp_path / ".triton")
    adapter = BonsaiTernaryAdapter(config=config, pipeline_factory=factory)

    adapter.prewarm()

    assert factory.kwargs["backend"] == "bonsai-ternary-gemlite"
    assert factory.kwargs["binary_transformer_path"] == str(config.transformer_path)
    assert factory.kwargs["ternary_transformer_path"] == str(config.transformer_path)
    assert factory.pipeline.prewarm_count == 1
    assert monkeypatch.context is not None


def test_bonsai_adapter_generates_candidate_and_reuses_resident_pipeline(tmp_path: Path):
    factory = FakeFactory()
    adapter = BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(model_root=tmp_path / "model", triton_cache_dir=tmp_path / ".triton"),
        pipeline_factory=factory,
    )

    first = adapter.generate(request(tmp_path))
    second = adapter.generate(request(tmp_path))

    assert first.candidate.file_valid is True
    assert second.candidate.file_valid is True
    assert factory.pipeline.prewarm_count == 1
    assert len(factory.pipeline.calls) == 2
    assert factory.pipeline.calls[0]["prompt"].startswith("Generate ")
    assert factory.pipeline.calls[0]["seed"] == 7
    assert factory.pipeline.calls[0]["steps"] == 4
