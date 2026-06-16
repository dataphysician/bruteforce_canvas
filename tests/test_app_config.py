from pathlib import Path

from bruteforce_canvas.app_config import (
    AppConfig,
    CanonicalizerProvider,
    GeneratorKind,
    LLMProvider,
    VLMProvider,
    load_app_config,
)
from bruteforce_canvas.canonicalizers import BGE_SMALL_EN_MODEL_ID, DEFAULT_EMBEDDING_MATCH_THRESHOLD
from bruteforce_canvas.llm_clients import (
    OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL,
    OPENAI_COMPATIBLE_SERVER_DEFAULT_MODEL,
    OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS,
)


def test_load_app_config_defaults_are_safe_for_local_tdd(monkeypatch):
    monkeypatch.delenv("BC_GENERATOR", raising=False)
    monkeypatch.delenv("BC_EVENT_STORE", raising=False)
    monkeypatch.delenv("BC_OPENAI_MODEL", raising=False)

    config = load_app_config({})

    assert config.generator.kind == GeneratorKind.STUB
    assert config.event_store_path == Path("runtime/events.jsonl")
    assert config.openai_model == "gpt-4.1-mini"
    assert config.llm.provider == LLMProvider.OPENAI_COMPATIBLE_SERVER
    assert config.llm.base_url == OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL
    assert config.llm.model == OPENAI_COMPATIBLE_SERVER_DEFAULT_MODEL
    assert config.llm.timeout_seconds == OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS
    assert config.vlm.provider == VLMProvider.LOCAL_MINICPM
    assert config.vlm.model == "openbmb/MiniCPM-V-4.6"
    assert config.vlm.timeout_seconds == OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS
    assert config.canonicalizer.provider == CanonicalizerProvider.EMBEDDING
    assert config.canonicalizer.embedding_model == BGE_SMALL_EN_MODEL_ID
    assert config.canonicalizer.match_threshold == DEFAULT_EMBEDDING_MATCH_THRESHOLD
    assert config.canonicalizer.llm_fallback is True
    assert config.run.iqa_cutoff == 0.55
    assert config.hardware.vram_gib == 0


def test_load_app_config_reads_environment_overrides():
    config = load_app_config(
        {
            "BC_GENERATOR": "bonsai",
            "BC_EVENT_STORE": "/tmp/bruteforce/events.jsonl",
            "BC_OPENAI_MODEL": "gpt-4.1",
            "BC_IQA_CUTOFF": "0.72",
            "BC_ALIGNMENT_CUTOFF": "0.31",
            "BC_VRAM_GIB": "24",
            "BC_DEVICE": "cuda",
            "BC_BONSAI_BACKEND_ROOT": "/external/Bonsai-Image-Demo",
            "BC_BONSAI_MODEL_ROOT": "/models/bonsai",
            "BC_BONSAI_KERNEL_WARMUP": "false",
            "BC_BONSAI_WARMUP_PROMPT": "Generate a matte calibration cube",
            "BC_BONSAI_WARMUP_STEPS": "2",
            "BC_BONSAI_WARMUP_HEIGHT": "256",
            "BC_BONSAI_WARMUP_WIDTH": "384",
            "BC_BONSAI_HTTP_URL": "http://127.0.0.1:7950",
            "BC_BONSAI_HTTP_TOKEN": "test-token",
            "BC_LLM_PROVIDER": "openai-compatible-server",
            "BC_LLM_BASE_URL": "https://llm.example.test/v1",
            "BC_LLM_MODEL": "json-reasoner",
            "BC_LLM_API_KEY": "secret",
            "BC_LLM_TIMEOUT_SECONDS": "17",
            "BC_LLM_MAX_COMPLETION_TOKENS": "8192",
            "BC_LLM_TEMPERATURE": "0.2",
            "BC_LLM_STRUCTURED_DECODING": "false",
            "BC_VLM_PROVIDER": "openai-compatible-server",
            "BC_VLM_BASE_URL": "https://vlm.example.test/v1",
            "BC_VLM_MODEL": "remote-minicpm",
            "BC_VLM_API_KEY": "vlm-secret",
            "BC_VLM_TIMEOUT_SECONDS": "31",
            "BC_VLM_MAX_COMPLETION_TOKENS": "333",
            "BC_VLM_TEMPERATURE": "0.1",
            "BC_VLM_STRUCTURED_DECODING": "false",
            "BC_CANONICALIZER_PROVIDER": "embedding",
            "BC_CANONICALIZER_EMBEDDING_MODEL": "BAAI/bge-small-en",
            "BC_CANONICALIZER_MATCH_THRESHOLD": "0.77",
            "BC_CANONICALIZER_LLM_FALLBACK": "false",
            "BC_PROMOTED_HIGH_WATERMARK": "500",
            "BC_PROMOTED_LOW_WATERMARK": "120",
        }
    )

    assert config.generator.kind == GeneratorKind.BONSAI
    assert config.event_store_path == Path("/tmp/bruteforce/events.jsonl")
    assert config.openai_model == "gpt-4.1"
    assert config.run.iqa_cutoff == 0.72
    assert config.run.alignment_cutoff == 0.31
    assert config.hardware.vram_gib == 24
    assert config.device.device == "cuda"
    assert config.generator.bonsai_backend_root == Path("/external/Bonsai-Image-Demo")
    assert config.generator.bonsai_model_root == Path("/models/bonsai")
    assert config.generator.bonsai_kernel_warmup is False
    assert config.generator.bonsai_warmup_prompt == "Generate a matte calibration cube"
    assert config.generator.bonsai_warmup_steps == 2
    assert config.generator.bonsai_warmup_height == 256
    assert config.generator.bonsai_warmup_width == 384
    assert config.generator.bonsai_http_url == "http://127.0.0.1:7950"
    assert config.generator.bonsai_http_token == "test-token"
    assert config.llm.provider == LLMProvider.OPENAI_COMPATIBLE_SERVER
    assert config.llm.base_url == "https://llm.example.test/v1"
    assert config.llm.model == "json-reasoner"
    assert config.llm.api_key == "secret"
    assert config.llm.timeout_seconds == 17
    assert config.llm.max_completion_tokens == 8192
    assert config.llm.temperature == 0.2
    assert config.llm.structured_decoding is False
    assert config.vlm.provider == VLMProvider.OPENAI_COMPATIBLE_SERVER
    assert config.vlm.base_url == "https://vlm.example.test/v1"
    assert config.vlm.model == "remote-minicpm"
    assert config.vlm.api_key == "vlm-secret"
    assert config.vlm.timeout_seconds == 31
    assert config.vlm.max_completion_tokens == 333
    assert config.vlm.temperature == 0.1
    assert config.vlm.structured_decoding is False
    assert config.canonicalizer.provider == CanonicalizerProvider.EMBEDDING
    assert config.canonicalizer.embedding_model == "BAAI/bge-small-en"
    assert config.canonicalizer.match_threshold == 0.77
    assert config.canonicalizer.llm_fallback is False
    assert config.run.promoted_high_watermark == 500
    assert config.run.promoted_low_watermark == 120


def test_app_config_rejects_invalid_generator_kind():
    try:
        AppConfig.model_validate({"generator": {"kind": "diffusers"}})
    except ValueError as error:
        assert "generator" in str(error)
    else:
        raise AssertionError("expected validation error")


def test_app_config_rejects_bonsai_without_cuda_device():
    try:
        AppConfig.model_validate({"generator": {"kind": "bonsai"}, "device": {"device": "auto"}})
    except ValueError as error:
        assert "BC_GENERATOR=bonsai requires BC_DEVICE=cuda" in str(error)
    else:
        raise AssertionError("expected validation error")


def test_app_config_allows_bonsai_http_without_local_cuda():
    config = AppConfig.model_validate(
        {
            "generator": {
                "kind": "bonsai-http",
                "bonsai_http_url": "http://127.0.0.1:7950",
            }
        }
    )

    assert config.generator.kind == GeneratorKind.BONSAI_HTTP
    assert config.device.device == "auto"
