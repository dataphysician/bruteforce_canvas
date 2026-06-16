from pathlib import Path

from bruteforce_canvas.evaluation import EvaluationPlan, StaticIQAAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import GenerationSettings, StubGeneratorAdapter, seed_sweep_requests
from bruteforce_canvas.loop import LoopAction
from bruteforce_canvas.orchestration import FeedbackPolicyError, RunConfig, RunRuntimeState
from bruteforce_canvas.persistence import JsonlEventStore, PersistenceRecord, reconstruct_run_state
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.shared import FeedbackAction
from bruteforce_canvas.ui import (
    PreRunModalState,
    begin_generation_event,
    cancel_pre_run_event,
    run_control_event,
    submit_feedback_event,
    RunControl,
)
from bruteforce_canvas.worker import PersistentSeedSweepWorker, SeedSweepWorkItem


def work_item(
    tmp_path: Path,
    *,
    coordinate_id: str = "coord_001",
    candidate_id_prefix: str | None = None,
) -> SeedSweepWorkItem:
    requests = seed_sweep_requests(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id=coordinate_id,
        rendered_prompt="Generate a ceramic bowl on wooden table",
        generation_settings=GenerationSettings(),
        output_dir=tmp_path / coordinate_id,
        generator_model_id="stub-generator",
        generator_backend="stub",
        candidate_id_prefix=candidate_id_prefix,
    )
    return SeedSweepWorkItem(
        run_id="run_001",
        raw_user_prompt="a ceramic bowl on wooden table",
        coordinate_id=coordinate_id,
        rendered_prompt="Generate a ceramic bowl on wooden table",
        target_manifest={},
        generation_requests=requests,
        evaluation_plan=EvaluationPlan(quality_cutoff=0.55, alignment_cutoff=0.25),
        sampled_arms={"cinematography.shot_size": "MEDIUM_SHOT"},
        locked_arms={"object.material.object_01": "CERAMIC"},
        combo_signature="shot=MEDIUM_SHOT|material=CERAMIC",
    )


def service(tmp_path: Path, *, config: RunConfig | None = None) -> RunService:
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


def test_run_service_processes_pending_seed_sweep_and_persists_loop_state(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))

    decision = run_service.tick()

    assert decision.reason == "pending_coordinates"
    assert run_service.pending_count == 0
    records = run_service.store.replay()
    assert any(record.record_type == "loop_transition" for record in records)
    assert reconstruct_run_state(records).promoted_curated_count == 3


def test_run_service_stop_request_prevents_pending_work(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))
    run_service.request_stop()

    decision = run_service.tick()

    assert decision.next_state == RunRuntimeState.STOPPED
    assert run_service.pending_count == 1
    assert [record.record_type for record in run_service.store.replay()] == ["loop_transition"]


def test_run_service_pause_event_prevents_pending_work_until_pre_run_begin(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))

    run_service.handle_ui_event(
        run_control_event(control=RunControl.PAUSE, run_id="run_001", prompt="a ceramic bowl on wooden table")
    )
    paused = run_service.tick()

    assert paused.next_state == RunRuntimeState.PAUSED
    assert paused.reason == "paused"
    assert run_service.pending_count == 1

    run_service.handle_ui_event(begin_generation_event(run_id="run_001", modal_state=PreRunModalState.READY_TO_BEGIN))
    resumed = run_service.tick()

    assert resumed.reason == "pending_coordinates"
    assert run_service.pending_count == 0


def test_run_service_pre_run_cancel_keeps_history_and_prevents_pending_work(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))

    event = cancel_pre_run_event(run_id="run_001")
    run_service.handle_ui_event(event)
    decision = run_service.tick()

    assert decision.next_state == RunRuntimeState.PAUSED
    assert decision.reason == "paused"
    assert run_service.pending_count == 1
    assert [record.record_type for record in run_service.store.replay()] == ["ui_event", "loop_transition"]
    ui_record = run_service.store.replay()[0]
    assert ui_record.record_id == f"ui_event:{event.event_id}"
    assert ui_record.idempotency_key == event.event_id
    assert ui_record.payload["event_id"] == event.event_id
    assert ui_record.payload["timestamp"] == event.timestamp


def test_run_service_high_watermark_pauses_without_consuming_pending_work(tmp_path: Path):
    config = RunConfig(
        run_id="run_001",
        raw_user_prompt="a ceramic bowl on wooden table",
        promoted_high_watermark=0,
        promoted_low_watermark=0,
    )
    run_service = service(tmp_path, config=config)
    run_service.enqueue(work_item(tmp_path))

    decision = run_service.tick()

    assert decision.next_state == RunRuntimeState.PAUSED_HIGH_WATERMARK
    assert run_service.pending_count == 1
    assert [record.payload["action"] for record in run_service.store.replay()] == ["pause"]


def test_run_service_persists_stall_diagnostic_when_guard_stops_run(tmp_path: Path):
    config = RunConfig(
        run_id="run_001",
        raw_user_prompt="a difficult ceramic bowl",
        stall_window_seconds=1,
        stall_min_promoted=1,
    )
    run_service = service(tmp_path, config=config)
    for record in [
        PersistenceRecord(
            record_id="run_config:run_001",
            record_type="run_config",
            run_id="run_001",
            payload={"raw_user_prompt": "a difficult ceramic bowl"},
        ),
        PersistenceRecord(
            record_id="candidate:cand_7",
            record_type="candidate_record",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={"file_valid": True},
        ),
        PersistenceRecord(
            record_id="image_evaluation:cand_7",
            record_type="image_evaluation",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={"failure_types": ["alignment_below_cutoff"]},
        ),
        PersistenceRecord(
            record_id="evaluation_aggregate:coord_001",
            record_type="evaluation_aggregate",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            payload={
                "evaluated_count": 1,
                "quality_pass_count": 1,
                "promoted_count": 0,
                "elapsed_seconds": 2,
            },
        ),
        PersistenceRecord(
            record_id="learning_delta:coord_001",
            record_type="learning_delta",
            run_id="run_001",
            coordinate_id="coord_001",
            payload={
                "enum_arms": {
                    "cinematography.shot_size=WIDE_SHOT": {"alpha": 1.0, "beta": 4.0},
                },
                "combo_affinities": {
                    "shot=WIDE_SHOT|material=CERAMIC": {"gp_mean": -0.7},
                },
            },
        ),
    ]:
        run_service.store.append(record)

    decision = run_service.tick()

    assert decision.next_state == RunRuntimeState.STOPPED
    diagnostic = next(record for record in run_service.store.replay() if record.record_type == "stall_diagnostic")
    assert diagnostic.record_id == "stall_diagnostic:run_001"
    assert diagnostic.idempotency_key == "stall_diagnostic:run_001"
    assert diagnostic.payload["elapsed_seconds"] == 2
    assert diagnostic.payload["generated_count"] == 1
    assert diagnostic.payload["dominant_failure_types"] == ["alignment_below_cutoff"]
    assert diagnostic.payload["most_penalized_enum_arms"] == ["cinematography.shot_size=WIDE_SHOT"]
    assert diagnostic.payload["most_penalized_combinations"] == ["shot=WIDE_SHOT|material=CERAMIC"]
    assert "consider_lowering_alignment_cutoff" in diagnostic.payload["restart_hints"]


def test_run_service_persists_feedback_for_promoted_curated_candidate(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))
    run_service.tick()

    result = run_service.submit_feedback(candidate_id="cand_7", action=FeedbackAction.ACCEPT)

    records = run_service.store.replay()
    feedback = [record for record in records if record.record_type == "feedback"]
    feedback_learning = [record for record in records if record.record_type == "feedback_learning_delta"]
    assert result.effective_status == "accepted"
    assert feedback[0].candidate_id == "cand_7"
    assert feedback[0].payload["feedback_action"] == "accept"
    assert feedback[0].payload["feedback_scope"] == "pre_curated_candidate"
    assert feedback[0].payload["signal_source"] == "swipe_feedback"
    assert feedback[0].payload["automated_status"] == "promoted_curated"
    assert feedback[0].payload["persistence_version"] == "1"
    assert feedback_learning[0].payload["learning_delta"]["alpha"] == 1.0
    assert feedback_learning[0].payload["learning_signal_source"] == "swipe_feedback"
    assert feedback_learning[0].payload["persistence_version"] == "1"
    assert reconstruct_run_state(records).accepted_count == 1


def test_run_service_feedback_ui_event_dispatches_to_backend_feedback(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))
    run_service.tick()

    result = run_service.handle_ui_event(
        submit_feedback_event(run_id="run_001", candidate_id="cand_7", action=FeedbackAction.ACCEPT)
    )

    assert result.applied is True
    assert [record.record_type for record in run_service.store.replay()].count("feedback") == 1


def test_run_service_feedback_is_idempotent_by_candidate_and_action(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))
    run_service.tick()

    first = run_service.submit_feedback(candidate_id="cand_7", action=FeedbackAction.REJECT)
    second = run_service.submit_feedback(candidate_id="cand_7", action=FeedbackAction.REJECT)

    records = run_service.store.replay()
    assert first.applied is True
    assert second.applied is False
    assert [record.record_type for record in records].count("feedback") == 1
    assert reconstruct_run_state(records).rejected_count == 1


def test_run_service_rejects_contradictory_feedback_after_candidate_feedback_exists(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))
    run_service.tick()

    accepted = run_service.submit_feedback(candidate_id="cand_7", action=FeedbackAction.ACCEPT)
    rejected = run_service.submit_feedback(candidate_id="cand_7", action=FeedbackAction.REJECT)

    records = run_service.store.replay()
    assert accepted.applied is True
    assert rejected.applied is False
    assert [record.record_type for record in records].count("feedback") == 1
    assert [record.record_type for record in records].count("feedback_learning_delta") == 1
    assert reconstruct_run_state(records).accepted_count == 1
    assert reconstruct_run_state(records).rejected_count == 0


def test_run_service_rejects_feedback_for_non_curated_candidate(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))
    run_service.tick()

    try:
        run_service.submit_feedback(candidate_id="cand_8888", action=FeedbackAction.ACCEPT)
    except FeedbackPolicyError as error:
        assert str(error) == "feedback is only accepted for promoted and curated candidates"
    else:
        raise AssertionError("expected FeedbackPolicyError")


def test_run_service_invokes_gate_chain_during_tick_with_pending_work(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.enqueue(work_item(tmp_path))

    run_service.tick()

    assert "rendering" in run_service._gate_state.gates_passed
    assert run_service._gate_state.gates_failed == []


def test_run_service_gates_only_pending_coordinate_records_across_continuous_batches(tmp_path: Path):
    run_service = service(tmp_path)

    for index in range(1, 4):
        coordinate_id = f"coord_{index:03d}"
        run_service.enqueue(
            work_item(
                tmp_path,
                coordinate_id=coordinate_id,
                candidate_id_prefix=f"cand_{coordinate_id}",
            )
        )
        decision = run_service.tick()

        assert decision.reason == "pending_coordinates"
        assert decision.action == LoopAction.GENERATE_PENDING_COORDINATE

    records = run_service.store.replay()
    candidate_coordinates = {
        record.coordinate_id
        for record in records
        if record.record_type == "candidate_record"
    }

    assert candidate_coordinates == {"coord_001", "coord_002", "coord_003"}
    assert not any(record.record_type == "gate_blocked" for record in records)


def test_run_service_persists_gate_blocked_record_when_generation_gate_fails(tmp_path: Path):
    run_service = service(tmp_path)
    run_service.store.append(
        PersistenceRecord(
            record_id="candidate:cand_7",
            record_type="candidate_record",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={
                "candidate_id": "cand_7",
                "run_id": "run_001",
                "prompt_document_id": "doc_001",
                "target_manifest_id": "eval_manifest_001",
                "coordinate_id": "coord_001",
                "seed": 7,
                "rendered_prompt": "Generate a ceramic bowl on wooden table",
                "generator_model_id": "stub",
                "generator_backend": "stub",
                "generation_settings": {},
                "image_path": "/tmp/missing.png",
                "file_valid": False,
                "timestamp": "1970-01-01T00:00:00Z",
                "generation_elapsed_ms": 0,
            },
        )
    )
    run_service.enqueue(work_item(tmp_path))

    decision = run_service.tick()

    assert decision.action == LoopAction.GATE_BLOCKED
    assert decision.reason.startswith("generation_gate_blocked:")
    assert decision.next_state == RunRuntimeState.RUNNING
    assert run_service._gate_state.gates_failed == ["generation"]
    gate_blocked_records = [
        record for record in run_service.store.replay() if record.record_type == "gate_blocked"
    ]
    assert len(gate_blocked_records) == 1
    payload = gate_blocked_records[0].payload
    assert payload["gate"] == "generation"
    assert payload["reason"].startswith("generation_gate_blocked:")
    assert payload["gates_failed"] == ["generation"]
    assert run_service.pending_count == 1
