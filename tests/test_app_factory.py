from pathlib import Path

from bruteforce_canvas.app_config import AppConfig, GeneratorConfig, GeneratorKind
from bruteforce_canvas.app_factory import (
    build_canonicalizer_adapter,
    build_evaluation_plan,
    build_event_store,
    build_generator_adapter,
    build_json_llm_client,
    build_prompt_pipeline,
    build_run_service,
    build_stage_plan,
    build_vlm_adapter,
)
from bruteforce_canvas.canonicalizers import EmbeddingCanonicalizerAdapter, FallbackCanonicalizerAdapter
from bruteforce_canvas.evaluation import StaticIQAAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import (
    BonsaiHttpAdapter,
    BonsaiTernaryAdapter,
    GenerationSettings,
    StubGeneratorAdapter,
    seed_sweep_requests,
)
from bruteforce_canvas.llm_adapters import LLMCanonicalizerAdapter
from bruteforce_canvas.llm_clients import OpenAICompatibleServerJsonLLMClient
from bruteforce_canvas.prompt_pipeline import PromptPipeline
from bruteforce_canvas.real_adapters import MiniCPMVAdapter, OpenAICompatibleVLMAlignmentAdapter
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.worker import SeedSweepWorkItem


def test_factory_builds_stub_generator_by_default():
    adapter = build_generator_adapter(AppConfig())

    assert isinstance(adapter, StubGeneratorAdapter)


def test_factory_builds_bonsai_generator_with_configured_paths(tmp_path: Path):
    config = AppConfig(
        generator=GeneratorConfig(
            kind=GeneratorKind.BONSAI.value,
            bonsai_backend_root=tmp_path / "Bonsai-Image-Demo",
            bonsai_model_root=tmp_path / "bonsai",
            bonsai_triton_cache_dir=tmp_path / ".triton",
            bonsai_kernel_warmup=False,
            bonsai_warmup_prompt="Generate a matte calibration cube",
            bonsai_warmup_steps=2,
            bonsai_warmup_height=256,
            bonsai_warmup_width=384,
        ),
        device={"device": "cuda"},
    )

    adapter = build_generator_adapter(config)

    assert isinstance(adapter, BonsaiTernaryAdapter)
    assert adapter.config.backend_root == tmp_path / "Bonsai-Image-Demo"
    assert adapter.config.model_root == tmp_path / "bonsai"
    assert adapter.config.triton_cache_dir == tmp_path / ".triton"
    assert adapter.config.device == "cuda"
    assert adapter.config.kernel_warmup is False
    assert adapter.config.warmup_prompt == "Generate a matte calibration cube"
    assert adapter.config.warmup_steps == 2
    assert adapter.config.warmup_height == 256
    assert adapter.config.warmup_width == 384


def test_factory_builds_bonsai_http_generator_with_configured_endpoint():
    config = AppConfig(
        generator=GeneratorConfig(
            kind=GeneratorKind.BONSAI_HTTP.value,
            bonsai_http_url="http://127.0.0.1:7950",
            bonsai_http_token="test-token",
            bonsai_kernel_warmup=False,
        )
    )

    adapter = build_generator_adapter(config)

    assert isinstance(adapter, BonsaiHttpAdapter)
    assert adapter.config.base_url == "http://127.0.0.1:7950"
    assert adapter.config.token == "test-token"
    assert adapter.config.kernel_warmup is False


def test_factory_builds_event_store_at_configured_path(tmp_path: Path):
    path = tmp_path / "events.jsonl"

    store = build_event_store(AppConfig(event_store_path=path))

    assert store.path == path


def test_factory_builds_openai_compatible_server_json_client_from_config():
    config = AppConfig(
        llm={
            "provider": "openai-compatible-server",
            "base_url": "https://llm.example.test/v1",
            "model": "json-reasoner",
            "api_key": "secret",
            "timeout_seconds": 30,
            "max_completion_tokens": 2048,
            "temperature": 0.1,
            "structured_decoding": False,
        }
    )

    client = build_json_llm_client(config)

    assert isinstance(client, OpenAICompatibleServerJsonLLMClient)
    assert client.base_url == "https://llm.example.test/v1"
    assert client.model == "json-reasoner"
    assert client.api_key == "secret"
    assert client.timeout_seconds == 30
    assert client.max_completion_tokens == 2048
    assert client.temperature == 0.1
    assert client.structured_decoding is False


def test_factory_builds_local_minicpm_vlm_by_default():
    adapter = build_vlm_adapter(AppConfig(device={"device": "cuda"}))

    assert isinstance(adapter, MiniCPMVAdapter)
    assert adapter.mode == "real"
    assert adapter.device == "cuda"


def test_factory_builds_openai_compatible_vlm_endpoint_adapter_from_config():
    config = AppConfig(
        vlm={
            "provider": "openai-compatible-server",
            "base_url": "https://vlm.example.test/v1",
            "model": "remote-minicpm",
            "api_key": "secret",
            "timeout_seconds": 30,
            "max_completion_tokens": 333,
            "temperature": 0.1,
            "structured_decoding": False,
        }
    )

    adapter = build_vlm_adapter(config)

    assert isinstance(adapter, OpenAICompatibleVLMAlignmentAdapter)
    assert adapter.base_url == "https://vlm.example.test/v1"
    assert adapter.model == "remote-minicpm"
    assert adapter.api_key == "secret"
    assert adapter.timeout_seconds == 30
    assert adapter.max_completion_tokens == 333
    assert adapter.temperature == 0.1
    assert adapter.structured_decoding is False


def test_factory_builds_prompt_pipeline_with_configured_llm_client():
    config = AppConfig(llm={"base_url": "https://llm.example.test/v1"})

    pipeline = build_prompt_pipeline(config)

    assert isinstance(pipeline, PromptPipeline)
    assert pipeline.extractor.client.base_url == "https://llm.example.test/v1"
    assert isinstance(pipeline.canonicalizer, FallbackCanonicalizerAdapter)
    assert isinstance(pipeline.canonicalizer.primary, EmbeddingCanonicalizerAdapter)
    assert isinstance(pipeline.canonicalizer.fallback, LLMCanonicalizerAdapter)
    assert "relation." in pipeline.canonicalizer.primary.enum_contexts
    assert "relation." in pipeline.canonicalizer.fallback.enum_contexts
    assert pipeline.verifier.client is pipeline.extractor.client
    assert pipeline.repairer.client is pipeline.extractor.client


def test_factory_defaults_canonicalizer_to_embedding_with_llm_fallback():
    canonicalizer = build_canonicalizer_adapter(AppConfig())

    assert isinstance(canonicalizer, FallbackCanonicalizerAdapter)
    assert isinstance(canonicalizer.primary, EmbeddingCanonicalizerAdapter)
    assert isinstance(canonicalizer.fallback, LLMCanonicalizerAdapter)
    assert "relation." in canonicalizer.primary.enum_contexts
    assert "relation." in canonicalizer.fallback.enum_contexts


def test_factory_can_disable_canonicalizer_llm_fallback():
    canonicalizer = build_canonicalizer_adapter(AppConfig(canonicalizer={"llm_fallback": False}))

    assert isinstance(canonicalizer, EmbeddingCanonicalizerAdapter)


def test_factory_can_use_llm_canonicalizer_when_explicitly_configured():
    canonicalizer = build_canonicalizer_adapter(AppConfig(canonicalizer={"provider": "llm"}))

    assert isinstance(canonicalizer, LLMCanonicalizerAdapter)
    assert "relation." in canonicalizer.enum_contexts


def test_factory_stage_plan_uses_configured_run_and_hardware():
    config = AppConfig(
        run={
            "run_id": "run_001",
            "raw_user_prompt": "configured prompt",
            "metacognitive_impact_enabled": True,
            "metacognitive_min_vram_gib": 24,
        },
        hardware={"vram_gib": 24, "cuda_available": True},
    )

    plan = build_stage_plan(config)

    assert plan.impact is True
    assert plan.max_vlm_batch_size == 8


def test_factory_builds_run_local_evaluation_plan_from_config():
    config = AppConfig(
        run={
            "run_id": "run_001",
            "raw_user_prompt": "configured prompt",
            "iqa_cutoff": 0.71,
            "alignment_cutoff": 0.42,
            "human_iqa_cutoff": 0.88,
            "metacognitive_impact_enabled": True,
            "metacognitive_min_vram_gib": 24,
        },
        hardware={"vram_gib": 24, "cuda_available": True},
    )

    plan = build_evaluation_plan(config)

    assert plan.quality_cutoff == 0.71
    assert plan.alignment_cutoff == 0.42
    assert plan.human_quality_cutoff == 0.88
    assert plan.metacognitive_impact is True


def test_factory_builds_configured_run_service_that_processes_work(tmp_path: Path):
    config = AppConfig(
        event_store_path=tmp_path / "events.jsonl",
        run={"run_id": "run_001", "raw_user_prompt": "configured prompt"},
    )
    service = build_run_service(
        config,
        iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2, 0.1]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
    )
    requests = seed_sweep_requests(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        rendered_prompt="Generate a ceramic bowl on wooden table",
        generation_settings=GenerationSettings(),
        output_dir=tmp_path,
        generator_model_id="stub-generator",
        generator_backend="stub",
    )

    service.enqueue(
        SeedSweepWorkItem(
            run_id="run_001",
            raw_user_prompt="configured prompt",
            coordinate_id="coord_001",
            rendered_prompt="Generate a ceramic bowl on wooden table",
            target_manifest={},
            generation_requests=requests,
            evaluation_plan=build_evaluation_plan(config),
            sampled_arms={"cinematography.shot_size": "MEDIUM_SHOT"},
            combo_signature="shot=MEDIUM_SHOT",
        )
    )

    decision = service.tick()

    assert isinstance(service, RunService)
    assert decision.reason == "pending_coordinates"
    assert config.event_store_path.exists()
    assert service.store.path == config.event_store_path
