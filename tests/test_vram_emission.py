from __future__ import annotations

from pathlib import Path

from bruteforce_canvas.evaluation import EvaluationPlan, StaticIQAAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import GenerationSettings, StubGeneratorAdapter, seed_sweep_requests
from bruteforce_canvas.orchestration import RunConfig
from bruteforce_canvas.persistence import JsonlEventStore
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.telemetry import VRAMTelemetry
from bruteforce_canvas.ui import RunWorkspaceReadModel
from bruteforce_canvas.worker import PersistentSeedSweepWorker, SeedSweepWorkItem


def _work_item(tmp_path: Path) -> SeedSweepWorkItem:
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
    return SeedSweepWorkItem(
        run_id="run_001",
        raw_user_prompt="a ceramic bowl on wooden table",
        coordinate_id="coord_001",
        rendered_prompt="Generate a ceramic bowl on wooden table",
        target_manifest={},
        generation_requests=requests,
        evaluation_plan=EvaluationPlan(quality_cutoff=0.55, alignment_cutoff=0.25),
        sampled_arms={"cinematography.shot_size": "MEDIUM_SHOT"},
        locked_arms={"object.material.object_01": "CERAMIC"},
        combo_signature="shot=MEDIUM_SHOT|material=CERAMIC",
    )


def _service(tmp_path: Path, *, config: RunConfig | None = None) -> RunService:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=StubGeneratorAdapter(),
        iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2, 0.1]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
    )
    return RunService(
        config=config or RunConfig(run_id="run_001", raw_user_prompt="a ceramic bowl on wooden table"),
        store=store,
        worker=worker,
    )


def test_sampling_occurs_at_configured_interval(tmp_path: Path, monkeypatch):
    config = RunConfig(run_id="run_001", raw_user_prompt="a ceramic bowl on wooden table", vram_sample_interval_ticks=2)
    run_service = _service(tmp_path, config=config)
    run_service.enqueue(_work_item(tmp_path))

    sample = VRAMTelemetry(total_gib=24.0, used_gib=10.0, free_gib=14.0, timestamp=1.0)
    monkeypatch.setattr("bruteforce_canvas.run_service.collect_vram_telemetry", lambda: sample)

    run_service.tick()
    run_service.tick()

    assert len(run_service._vram_telemetry) == 1


def test_telemetry_list_is_capped_at_100_entries(tmp_path: Path, monkeypatch):
    config = RunConfig(run_id="run_001", raw_user_prompt="a ceramic bowl on wooden table", vram_sample_interval_ticks=1)
    run_service = _service(tmp_path, config=config)
    run_service.enqueue(_work_item(tmp_path))

    sample = VRAMTelemetry(total_gib=24.0, used_gib=10.0, free_gib=14.0, timestamp=1.0)
    monkeypatch.setattr("bruteforce_canvas.run_service.collect_vram_telemetry", lambda: sample)
    run_service._vram_telemetry = [sample] * 105

    run_service.tick()

    assert len(run_service._vram_telemetry) == 100
    assert run_service._vram_telemetry[0] == sample
    assert run_service._vram_telemetry[-1] == sample


def test_snapshot_and_read_model_include_telemetry(tmp_path: Path, monkeypatch):
    config = RunConfig(run_id="run_001", raw_user_prompt="a ceramic bowl on wooden table", vram_sample_interval_ticks=1)
    run_service = _service(tmp_path, config=config)
    run_service.enqueue(_work_item(tmp_path))

    sample = VRAMTelemetry(total_gib=24.0, used_gib=10.0, free_gib=14.0, timestamp=1.0)
    monkeypatch.setattr("bruteforce_canvas.run_service.collect_vram_telemetry", lambda: sample)
    run_service._vram_telemetry = [sample]

    snap = run_service.snapshot()

    assert snap.vram_telemetry == [sample]
    assert snap.vram_telemetry[0].total_gib == 24.0

    read_model = RunWorkspaceReadModel(
        run_id="run_001",
        raw_user_prompt="a ceramic bowl on wooden table",
        run_state="running",
        generated_count=1,
        iqa_evaluated_count=1,
        vlm_evaluated_count=1,
        promoted_curated_count=1,
        accepted_count=1,
        rejected_count=1,
        shredded_count=1,
        stall_guard_state="inactive",
        notification="",
        elapsed_seconds=0,
        vram_telemetry=[sample],
    )
    assert read_model.progress_heartbeat["vram_telemetry"] == [sample.model_dump()]
