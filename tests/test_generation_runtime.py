from pathlib import Path
from datetime import UTC, datetime

import pytest

from bruteforce_canvas.generation import (
    BonsaiTernaryConfig,
    GenerationRequest,
    GenerationSettings,
    ResidentGeneratorWorker,
    StubGeneratorAdapter,
)


def request(tmp_path: Path, seed: int = 7) -> GenerationRequest:
    return GenerationRequest(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seed=seed,
        rendered_prompt="Generate a ceramic bowl on wooden table",
        generation_settings=GenerationSettings(),
        image_path=str(tmp_path / f"{seed}.png"),
        generator_model_id="stub-generator",
        generator_backend="stub",
    )


def test_bonsai_config_uses_model_specific_backend_but_generic_stage_name(tmp_path: Path):
    config = BonsaiTernaryConfig(model_root=tmp_path / "bonsai-image-4B-ternary-gemlite")

    assert config.stage_name == "fast_image_generation"
    assert config.backend == "bonsai-ternary-gemlite"
    assert config.model_id == "prism-ml/bonsai-image-ternary-4B-gemlite-2bit"
    assert config.binary_transformer_path == config.ternary_transformer_path
    assert str(config.triton_cache_dir).endswith("runtime/.triton_cache")


def test_resident_worker_prewarms_once_and_reuses_adapter(tmp_path: Path):
    adapter = StubGeneratorAdapter()
    worker = ResidentGeneratorWorker(adapter)

    first = worker.generate(request(tmp_path, 7))
    second = worker.generate(request(tmp_path, 42))

    assert adapter.prewarm_count == 1
    assert adapter.generate_count == 2
    assert first.candidate.seed == 7
    assert second.candidate.seed == 42


def test_stub_generation_records_real_utc_timestamp_and_elapsed_time(tmp_path: Path):
    result = StubGeneratorAdapter().generate(request(tmp_path, 7))

    assert result.candidate.timestamp.endswith("Z")
    assert result.candidate.timestamp != "1970-01-01T00:00:00Z"
    parsed = datetime.fromisoformat(result.candidate.timestamp.replace("Z", "+00:00"))
    assert parsed.tzinfo == UTC
    assert result.candidate.generation_elapsed_ms >= 0


def test_resident_worker_rejects_unrendered_prompt_before_adapter_call(tmp_path: Path):
    adapter = StubGeneratorAdapter()
    worker = ResidentGeneratorWorker(adapter)
    bad = GenerationRequest.model_construct(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seed=7,
        rendered_prompt="a raw prompt",
        generation_settings=GenerationSettings(),
        image_path=str(tmp_path / "bad.png"),
        generator_model_id="stub-generator",
        generator_backend="stub",
    )

    with pytest.raises(ValueError, match="rendered prompt beginning with 'Generate'"):
        worker.generate(bad)
    assert adapter.generate_count == 0
