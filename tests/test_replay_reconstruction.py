from bruteforce_canvas.persistence import PersistenceRecord, reconstruct_run_state


def test_reconstruct_run_state_from_persisted_records():
    records = [
        PersistenceRecord(
            record_id="rec_001",
            record_type="run_config",
            run_id="run_001",
            payload={"raw_user_prompt": "a bowl on a table"},
        ),
        PersistenceRecord(
            record_id="rec_002",
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
            record_id="rec_003",
            record_type="candidate_record",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_42",
            seed=42,
            payload={"file_valid": True},
        ),
        PersistenceRecord(
            record_id="rec_004",
            record_type="evaluation_aggregate",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            payload={
                "promoted_count": 1,
                "evaluated_count": 2,
                "quality_pass_count": 2,
                "alignment_pass_count": 1,
                "elapsed_seconds": 42,
            },
        ),
        PersistenceRecord(
            record_id="rec_005",
            record_type="feedback",
            run_id="run_001",
            candidate_id="cand_7",
            idempotency_key="cand_7:accept",
            payload={"feedback_action": "accept"},
        ),
        PersistenceRecord(
            record_id="rec_006",
            record_type="learning_delta",
            run_id="run_001",
            coordinate_id="coord_001",
            idempotency_key="eval:coord_001",
            payload={"source": "automated_evaluation"},
        ),
        PersistenceRecord(
            record_id="rec_007",
            record_type="feedback_learning_delta",
            run_id="run_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            idempotency_key="feedback_learning_delta:cand_7:accept",
            payload={"source": "swipe_feedback"},
        ),
    ]

    state = reconstruct_run_state(records)

    assert state.run_id == "run_001"
    assert state.raw_user_prompt == "a bowl on a table"
    assert state.generated_count == 2
    assert state.iqa_evaluated_count == 2
    assert state.vlm_evaluated_count == 2
    assert state.promoted_curated_count == 1
    assert state.elapsed_seconds == 42
    assert state.accepted_count == 1
    assert state.learning_update_count == 2
    assert state.coordinate_ids == ["coord_001"]
    assert state.candidate_ids == ["cand_7", "cand_42"]


def test_reconstruct_run_state_deduplicates_idempotent_feedback_and_learning():
    records = [
        PersistenceRecord(
            record_id="rec_001",
            record_type="run_config",
            run_id="run_001",
            payload={"raw_user_prompt": "a bowl on a table"},
        ),
        PersistenceRecord(
            record_id="rec_002",
            record_type="feedback",
            run_id="run_001",
            candidate_id="cand_7",
            idempotency_key="cand_7:reject",
            payload={"feedback_action": "reject"},
        ),
        PersistenceRecord(
            record_id="rec_003",
            record_type="feedback",
            run_id="run_001",
            candidate_id="cand_7",
            idempotency_key="cand_7:reject",
            payload={"feedback_action": "reject"},
        ),
        PersistenceRecord(
            record_id="rec_004",
            record_type="learning_delta",
            run_id="run_001",
            idempotency_key="eval:coord_001",
            payload={},
        ),
        PersistenceRecord(
            record_id="rec_005",
            record_type="learning_delta",
            run_id="run_001",
            idempotency_key="eval:coord_001",
            payload={},
        ),
        PersistenceRecord(
            record_id="rec_006",
            record_type="feedback_learning_delta",
            run_id="run_001",
            candidate_id="cand_7",
            idempotency_key="feedback_learning_delta:cand_7:reject",
            payload={},
        ),
        PersistenceRecord(
            record_id="rec_007",
            record_type="feedback_learning_delta",
            run_id="run_001",
            candidate_id="cand_7",
            idempotency_key="feedback_learning_delta:cand_7:reject",
            payload={},
        ),
    ]

    state = reconstruct_run_state(records)

    assert state.rejected_count == 1
    assert state.learning_update_count == 2
