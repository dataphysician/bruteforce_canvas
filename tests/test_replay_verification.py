"""Replay verification tests for persisted runs (Phase L.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bruteforce_canvas.evaluation import EvaluationPlan, StaticIQAAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import GenerationSettings, StubGeneratorAdapter, seed_sweep_requests
from bruteforce_canvas.orchestration import RunConfig
from bruteforce_canvas.persistence import (
    PERSISTENCE_VERSION,
    JsonlEventStore,
    PersistenceRecord,
    ReconstructedRunState,
    reconstruct_run_state,
)
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.shared import FeedbackAction
from bruteforce_canvas.worker import PersistentSeedSweepWorker, SeedSweepWorkItem


class ReplayInconsistencyError(AssertionError):
    """Raised when replay detects that the persisted run is incomplete or drifted."""


EXPECTED_EVENT_TYPES: tuple[str, ...] = (
    "run_config",
    "candidate_record",
    "image_evaluation",
    "evaluation_aggregate",
    "learning_delta",
)


def verify_run_replay(
    records: list[PersistenceRecord],
    *,
    required_event_types: tuple[str, ...] = EXPECTED_EVENT_TYPES,
    expected_persistence_version: str = PERSISTENCE_VERSION,
) -> ReconstructedRunState:
    """Replay persisted records and validate run integrity.

    Contract: records must be non-empty; every ``required_event_types`` entry
    must appear at least once; every record that carries a
    ``persistence_version`` payload marker must match
    ``expected_persistence_version`` (mismatch = schema drift). Returns the
    reconstructed state; raises :class:`ReplayInconsistencyError` otherwise.
    """
    if not records:
        raise ReplayInconsistencyError("cannot replay run from empty event store")

    observed_types = {record.record_type for record in records}
    missing = [event_type for event_type in required_event_types if event_type not in observed_types]
    if missing:
        raise ReplayInconsistencyError(
            f"missing required event types for replay: {sorted(missing)}"
        )

    for record in records:
        version = record.payload.get("persistence_version")
        if version is None:
            continue
        if version != expected_persistence_version:
            raise ReplayInconsistencyError(
                f"record {record.record_id!r} ({record.record_type}) carries "
                f"persistence_version={version!r}, expected {expected_persistence_version!r}"
            )

    return reconstruct_run_state(records)


def _work_item(tmp_path: Path, *, coordinate_id: str = "coord_001") -> SeedSweepWorkItem:
    requests = seed_sweep_requests(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id=coordinate_id,
        rendered_prompt="Generate a ceramic bowl on wooden table",
        generation_settings=GenerationSettings(),
        output_dir=tmp_path,
        generator_model_id="stub-generator",
        generator_backend="stub",
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


def _service(tmp_path: Path) -> RunService:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=StubGeneratorAdapter(),
        iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2, 0.1]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
    )
    return RunService(
        config=RunConfig(run_id="run_001", raw_user_prompt="a ceramic bowl on wooden table"),
        store=store,
        worker=worker,
    )


def test_replay_reconstructs_run_state(tmp_path: Path) -> None:
    run_service = _service(tmp_path)
    run_service.enqueue(_work_item(tmp_path))
    run_service.tick()
    run_service.enqueue(_work_item(tmp_path, coordinate_id="coord_002"))
    run_service.tick()

    live_snapshot = run_service.snapshot()
    live_candidate_ids = [
        record.candidate_id
        for record in run_service.store.replay()
        if record.record_type == "candidate_record" and record.candidate_id
    ]
    assert live_candidate_ids, "expected live run to have produced at least one candidate"

    reconstructed = verify_run_replay(run_service.store.replay())

    assert reconstructed.run_id == run_service.config.run_id
    assert reconstructed.candidate_ids == live_candidate_ids
    assert reconstructed.coordinate_ids == ["coord_001", "coord_002"]
    assert reconstructed.generated_count == live_snapshot.counters.generated_count
    assert reconstructed.iqa_evaluated_count == live_snapshot.counters.iqa_evaluated_count
    assert reconstructed.vlm_evaluated_count == live_snapshot.counters.vlm_evaluated_count
    assert reconstructed.promoted_curated_count == live_snapshot.counters.promoted_curated_count

    again = verify_run_replay(run_service.store.replay())
    assert again.model_dump() == reconstructed.model_dump()


def test_replay_detects_missing_events(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl")

    store.append(
        PersistenceRecord(
            record_id="run_config:run_001",
            record_type="run_config",
            run_id="run_001",
            idempotency_key="run_config:run_001",
            payload={
                "raw_user_prompt": "a ceramic bowl on wooden table",
                "persistence_version": PERSISTENCE_VERSION,
            },
        )
    )
    store.append(
        PersistenceRecord(
            record_id="candidate:cand_7",
            record_type="candidate_record",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={"file_valid": True, "persistence_version": PERSISTENCE_VERSION},
        )
    )

    with pytest.raises(ReplayInconsistencyError) as exc_info:
        verify_run_replay(store.replay())

    message = str(exc_info.value)
    assert "missing required event types" in message
    for missing in ("image_evaluation", "evaluation_aggregate", "learning_delta"):
        assert missing in message, f"expected {missing!r} in inconsistency message"

    empty_store = JsonlEventStore(tmp_path / "empty.jsonl")
    with pytest.raises(ReplayInconsistencyError, match="empty event store"):
        verify_run_replay(empty_store.replay())


def test_replay_schema_version_is_recorded(tmp_path: Path) -> None:
    run_service = _service(tmp_path)
    run_service.enqueue(_work_item(tmp_path))
    run_service.tick()
    run_service.submit_feedback(candidate_id="cand_7", action=FeedbackAction.ACCEPT)

    records = run_service.store.replay()
    versioned_records = [record for record in records if "persistence_version" in record.payload]
    assert versioned_records, "expected post-Phase B events to carry persistence_version"

    versions = {record.payload["persistence_version"] for record in versioned_records}
    assert versions == {PERSISTENCE_VERSION}, (
        f"expected all versioned records to share {PERSISTENCE_VERSION!r}, got {versions!r}"
    )

    verify_run_replay(records)

    tampered: list[PersistenceRecord] = []
    drifted = False
    for record in records:
        if "persistence_version" in record.payload and not drifted:
            tampered.append(
                record.model_copy(
                    update={"payload": {**record.payload, "persistence_version": "999"}}
                )
            )
            drifted = True
        else:
            tampered.append(record)
    assert drifted, "expected to drift at least one versioned record"

    with pytest.raises(ReplayInconsistencyError) as exc_info:
        verify_run_replay(tampered, expected_persistence_version=PERSISTENCE_VERSION)
    assert "persistence_version" in str(exc_info.value)
    assert "999" in str(exc_info.value)
