from pathlib import Path

from bruteforce_canvas.app_config import AppConfig, GeneratorConfig, GeneratorKind
from bruteforce_canvas.app_factory import (
    build_evaluation_plan,
    build_event_store,
    build_generator_adapter,
    build_run_service,
    build_stage_plan,
)
from bruteforce_canvas.evaluation import StaticIQAAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import BonsaiTernaryAdapter, GenerationSettings, StubGeneratorAdapter, seed_sweep_requests
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.worker import SeedSweepWorkItem


def test_factory_builds_stub_generator_by_default():
    adapter = build_generator_adapter(AppConfig())

    assert isinstance(adapter, StubGeneratorAdapter)


def test_factory_builds_bonsai_generator_with_configured_paths(tmp_path: Path):
    config = AppConfig(
        generator=GeneratorConfig(
            kind=GeneratorKind.BONSAI.value,
            bonsai_model_root=tmp_path / "bonsai",
            bonsai_triton_cache_dir=tmp_path / ".triton",
        )
    )

    adapter = build_generator_adapter(config)

    assert isinstance(adapter, BonsaiTernaryAdapter)
    assert adapter.config.model_root == tmp_path / "bonsai"
    assert adapter.config.triton_cache_dir == tmp_path / ".triton"


def test_factory_builds_event_store_at_configured_path(tmp_path: Path):
    path = tmp_path / "events.jsonl"

    store = build_event_store(AppConfig(event_store_path=path))

    assert store.path == path


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
