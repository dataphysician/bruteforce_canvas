from bruteforce_canvas.persistence import JsonlEventStore, PersistenceRecord


def test_jsonl_event_store_replays_records_in_order(tmp_path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    store.append(
        PersistenceRecord(
            record_id="rec_001",
            record_type="run_config",
            run_id="run_001",
            payload={"raw_user_prompt": "a bowl on a table"},
        )
    )
    store.append(
        PersistenceRecord(
            record_id="rec_002",
            record_type="prompt_document",
            run_id="run_001",
            payload={"prompt_document_id": "doc_001"},
        )
    )

    assert [record.record_id for record in store.replay()] == ["rec_001", "rec_002"]


def test_jsonl_event_store_idempotent_append_does_not_duplicate(tmp_path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    record = PersistenceRecord(
        record_id="rec_001",
        record_type="learning_delta",
        run_id="run_001",
        idempotency_key="cand_001:reject",
        payload={"beta": 1.0},
    )

    assert store.append(record).written is True
    assert store.append(record).written is False
    assert len(store.replay()) == 1


def test_persistence_record_keeps_traceability_chain_ids():
    record = PersistenceRecord(
        record_id="rec_001",
        record_type="candidate_record",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        candidate_id="cand_001",
        seed=7,
        payload={"image_path": "/tmp/cand_001.png"},
    )

    assert record.traceability_key == "run_001/doc_001/eval_manifest_001/coord_001/cand_001/7"
