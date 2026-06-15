from pathlib import Path

from bruteforce_canvas.app_controller import RunAppController
from bruteforce_canvas.evaluation import EvaluationPlan, StaticIQAAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import GenerationSettings, StubGeneratorAdapter, seed_sweep_requests
from bruteforce_canvas.orchestration import RunConfig
from bruteforce_canvas.persistence import JsonlEventStore
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.shared import FeedbackAction
from bruteforce_canvas.ui import RunControl, cancel_pre_run_event, run_control_event, submit_feedback_event
from bruteforce_canvas.worker import PersistentSeedSweepWorker, SeedSweepWorkItem


def service(tmp_path: Path) -> RunService:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    return RunService(
        config=RunConfig(run_id="run_001", raw_user_prompt="a ceramic bowl on wooden table"),
        store=store,
        worker=PersistentSeedSweepWorker(
            store=store,
            generator=StubGeneratorAdapter(),
            iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2, 0.1]),
            vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
        ),
    )


def service_with_config(tmp_path: Path, config: RunConfig) -> RunService:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    return RunService(
        config=config,
        store=store,
        worker=PersistentSeedSweepWorker(
            store=store,
            generator=StubGeneratorAdapter(),
            iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2, 0.1]),
            vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
        ),
    )


def service_with_blocked_seed(tmp_path: Path) -> RunService:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    return RunService(
        config=RunConfig(run_id="run_001", raw_user_prompt="a ceramic bowl on wooden table"),
        store=store,
        worker=PersistentSeedSweepWorker(
            store=store,
            generator=StubGeneratorAdapter(blocked_seeds={42069}),
            iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2]),
            vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
        ),
    )


def item(tmp_path: Path) -> SeedSweepWorkItem:
    return SeedSweepWorkItem(
        run_id="run_001",
        raw_user_prompt="a ceramic bowl on wooden table",
        coordinate_id="coord_001",
        rendered_prompt="Generate a ceramic bowl on wooden table",
        target_manifest={},
        generation_requests=seed_sweep_requests(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            rendered_prompt="Generate a ceramic bowl on wooden table",
            generation_settings=GenerationSettings(),
            output_dir=tmp_path,
            generator_model_id="stub-generator",
            generator_backend="stub",
        ),
        evaluation_plan=EvaluationPlan(quality_cutoff=0.55, alignment_cutoff=0.25),
        sampled_arms={"cinematography.shot_size": "MEDIUM_SHOT"},
        combo_signature="shot=MEDIUM_SHOT",
    )


def test_app_controller_builds_workspace_read_bundle_from_service_store(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(item(tmp_path))
    run_service.tick()
    controller = RunAppController(run_service)

    bundle = controller.workspace()

    assert bundle.workspace.run_id == "run_001"
    assert bundle.workspace.generated_count == 5
    assert bundle.workspace.promoted_curated_count == 3
    assert [card.candidate_id for card in bundle.catalogue] == ["cand_7", "cand_42", "cand_156"]
    assert bundle.selected is not None
    assert bundle.selected.candidate_id == "cand_7"
    assert bundle.workspace.notification == "Generating pending coordinate."


def test_app_controller_builds_initial_workspace_before_records_exist(tmp_path: Path):
    controller = RunAppController(service(tmp_path))

    bundle = controller.workspace()

    assert bundle.workspace.run_id == "run_001"
    assert bundle.workspace.raw_user_prompt == "a ceramic bowl on wooden table"
    assert bundle.workspace.generated_count == 0
    assert bundle.catalogue == []
    assert bundle.selected is None
    assert bundle.workspace.notification == "Waiting for pre-run confirmation."


def test_app_controller_dispatches_ui_events_and_refreshes_workspace(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(item(tmp_path))
    run_service.tick()
    controller = RunAppController(run_service)

    result = controller.handle_event(
        submit_feedback_event(run_id="run_001", candidate_id="cand_7", action=FeedbackAction.ACCEPT)
    )
    bundle = controller.workspace()

    assert result.applied is True
    assert bundle.workspace.accepted_count == 1
    assert bundle.selected is not None
    assert bundle.selected.feedback_state == "accept"


def test_app_controller_reports_paused_status_after_pause_event(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(item(tmp_path))
    controller = RunAppController(run_service)

    controller.handle_event(run_control_event(control=RunControl.PAUSE, run_id="run_001", prompt="ignored"))
    run_service.tick()
    bundle = controller.workspace()

    assert bundle.workspace.run_state == "paused"
    assert bundle.workspace.notification == "Run paused."


def test_app_controller_reports_pre_run_cancel_as_action_needed(tmp_path: Path):
    run_service = service(tmp_path)
    controller = RunAppController(run_service)

    controller.handle_event(cancel_pre_run_event(run_id="run_001"))
    run_service.tick()
    bundle = controller.workspace()

    assert bundle.workspace.run_state == "paused"
    assert bundle.workspace.notification == "Pre-run canceled. Revise the prompt or start again."


def test_app_controller_reports_stopped_status_after_stop_event(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(item(tmp_path))
    controller = RunAppController(run_service)

    controller.handle_event(run_control_event(control=RunControl.STOP, run_id="run_001", prompt="ignored"))
    run_service.tick()
    bundle = controller.workspace()

    assert bundle.workspace.run_state == "stopped"
    assert bundle.workspace.notification == "Run stopped."


def test_app_controller_reports_high_watermark_pause(tmp_path: Path):
    run_service = service_with_config(
        tmp_path,
        RunConfig(
            run_id="run_001",
            raw_user_prompt="a ceramic bowl on wooden table",
            promoted_high_watermark=0,
            promoted_low_watermark=0,
        ),
    )
    run_service.enqueue(item(tmp_path))
    controller = RunAppController(run_service)

    run_service.tick()
    bundle = controller.workspace()

    assert bundle.workspace.run_state == "paused_high_watermark"
    assert bundle.workspace.notification == "Paused at promoted image watermark."


def test_app_controller_diagnostics_expose_system_actions_and_infrastructure_retries(tmp_path: Path):
    run_service = service_with_blocked_seed(tmp_path)
    run_service.enqueue(item(tmp_path))
    run_service.tick()
    controller = RunAppController(run_service)

    workspace = controller.workspace()
    diagnostics = controller.diagnostics()

    assert [card.candidate_id for card in workspace.catalogue] == ["cand_7", "cand_42", "cand_156"]
    assert diagnostics.system_action_count == 5
    assert diagnostics.infrastructure_retry_count == 1
    assert diagnostics.infrastructure_retries[0]["candidate_id"] == "cand_42069"
    assert diagnostics.infrastructure_retries[0]["semantic_penalty"] is False
    assert diagnostics.record_counts["image_evaluation"] == 5
    assert diagnostics.record_counts["system_action"] == 5
