from pathlib import Path

from bruteforce_canvas.cli import (
    candidate_cards_from_records,
    detail_report_from_records,
    event_stream_from_records,
    main,
    render_workspace_from_store,
)
from bruteforce_canvas.persistence import JsonlEventStore, PersistenceRecord
from bruteforce_canvas.shared import FeedbackAction
from bruteforce_canvas.ui import catalogue_default_items


def test_render_workspace_from_store_writes_static_html(tmp_path: Path):
    store_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "workspace.html"
    store = JsonlEventStore(store_path)
    store.append(
        PersistenceRecord(
            record_id="rec_001",
            record_type="run_config",
            run_id="run_001",
            payload={"raw_user_prompt": "a bowl on a table", "elapsed_seconds": 17},
        )
    )
    store.append(
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
        )
    )

    result = render_workspace_from_store(store_path, output_path)

    assert result == output_path
    html = output_path.read_text(encoding="utf-8")
    assert "a bowl on a table" in html
    assert 'data-region="run-controls"' in html
    assert 'data-region="progress-heartbeat"' in html
    assert "elapsed_seconds: 17" in html


def test_render_workspace_catalogue_uses_persisted_evaluations_and_feedback(tmp_path: Path):
    store_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "workspace.html"
    store = JsonlEventStore(store_path)
    store.append(
        PersistenceRecord(
            record_id="rec_001",
            record_type="run_config",
            run_id="run_001",
            payload={"raw_user_prompt": "a bowl on a table"},
        )
    )
    for index, candidate_id, action in [
        (1, "cand_7", "accept"),
        (2, "cand_42", "reject"),
    ]:
        store.append(
            PersistenceRecord(
                record_id=f"eval_{index}",
                record_type="image_evaluation",
                run_id="run_001",
                prompt_document_id="doc_001",
                target_manifest_id="eval_manifest_001",
                coordinate_id="coord_001",
                candidate_id=candidate_id,
                seed=7 if candidate_id == "cand_7" else 42,
                payload={
                    "pass_flags": {"quality": True, "alignment": True, "full": True},
                    "disposition_signal": {"class_name": "passes_thresholds"},
                },
            )
        )
        store.append(
            PersistenceRecord(
                record_id=f"feedback_{index}",
                record_type="feedback",
                run_id="run_001",
                prompt_document_id="doc_001",
                target_manifest_id="eval_manifest_001",
                coordinate_id="coord_001",
                candidate_id=candidate_id,
                seed=7 if candidate_id == "cand_7" else 42,
                payload={"feedback_action": action},
            )
        )

    render_workspace_from_store(store_path, output_path)

    html = output_path.read_text(encoding="utf-8")
    assert 'data-candidate-id="cand_7"' in html
    assert 'data-feedback-state="accept"' in html
    assert 'data-candidate-id="cand_42"' not in html


def test_candidate_cards_from_records_preserves_feedback_state():
    cards = candidate_cards_from_records(
        [
            PersistenceRecord(
                record_id="eval_1",
                record_type="image_evaluation",
                run_id="run_001",
                prompt_document_id="doc_001",
                target_manifest_id="eval_manifest_001",
                coordinate_id="coord_001",
                candidate_id="cand_7",
                seed=7,
                payload={
                    "pass_flags": {"quality": True, "alignment": True, "full": True},
                    "disposition_signal": {"class_name": "passes_thresholds"},
                },
            ),
            PersistenceRecord(
                record_id="feedback_1",
                record_type="feedback",
                run_id="run_001",
                prompt_document_id="doc_001",
                target_manifest_id="eval_manifest_001",
                coordinate_id="coord_001",
                candidate_id="cand_7",
                seed=7,
                payload={"feedback_action": "accept"},
            ),
        ]
    )

    assert cards[0].candidate_id == "cand_7"
    assert cards[0].promoted is True
    assert cards[0].curated is True
    assert cards[0].feedback_action == FeedbackAction.ACCEPT


def test_candidate_cards_preserve_optional_tags_without_gating_catalogue_inclusion():
    cards = candidate_cards_from_records(
        [
            PersistenceRecord(
                record_id="eval_1",
                record_type="image_evaluation",
                run_id="run_001",
                prompt_document_id="doc_001",
                target_manifest_id="eval_manifest_001",
                coordinate_id="coord_001",
                candidate_id="cand_7",
                seed=7,
                payload={
                    "pass_flags": {"quality": True, "alignment": True, "full": True},
                    "disposition_signal": {"class_name": "passes_thresholds"},
                    "optional_tags": ["warm"],
                },
            ),
            PersistenceRecord(
                record_id="eval_2",
                record_type="image_evaluation",
                run_id="run_001",
                prompt_document_id="doc_001",
                target_manifest_id="eval_manifest_001",
                coordinate_id="coord_001",
                candidate_id="cand_42",
                seed=42,
                payload={
                    "pass_flags": {"quality": False, "alignment": False, "full": False},
                    "disposition_signal": {"class_name": "fail_persist_for_learning"},
                    "optional_tags": ["interesting"],
                },
            ),
        ]
    )

    assert cards[0].optional_tags == ["warm"]
    assert cards[1].optional_tags == ["interesting"]
    assert [card.candidate_id for card in catalogue_default_items(cards)] == ["cand_7"]


def test_event_stream_from_records_projects_required_ui_lifecycle_concepts():
    records = [
        PersistenceRecord(
            record_id="run_1",
            record_type="run_config",
            run_id="run_001",
            payload={"raw_user_prompt": "a bowl on a table", "timestamp": "2026-01-01T00:00:00Z"},
        ),
        PersistenceRecord(
            record_id="prompt_1",
            record_type="prompt_document",
            run_id="run_001",
            prompt_document_id="doc_001",
            payload={"timestamp": "2026-01-01T00:00:01Z"},
        ),
        PersistenceRecord(
            record_id="transition_1",
            record_type="loop_transition",
            run_id="run_001",
            payload={
                "action": "generate_pending_coordinate",
                "next_state": "running",
                "timestamp": "2026-01-01T00:00:02Z",
            },
        ),
        PersistenceRecord(
            record_id="candidate_1",
            record_type="candidate_record",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={"image_path": "/tmp/cand_7.png", "timestamp": "2026-01-01T00:00:03Z"},
        ),
        PersistenceRecord(
            record_id="eval_1",
            record_type="image_evaluation",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={
                "pass_flags": {"quality": True, "alignment": True, "full": True},
                "disposition_signal": {"class_name": "passes_thresholds"},
                "timestamp": "2026-01-01T00:00:04Z",
            },
        ),
        PersistenceRecord(
            record_id="aggregate_1",
            record_type="evaluation_aggregate",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            payload={"evaluated_count": 1, "quality_pass_count": 1, "timestamp": "2026-01-01T00:00:05Z"},
        ),
        PersistenceRecord(
            record_id="feedback_1",
            record_type="feedback",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={"feedback_action": "reject", "timestamp": "2026-01-01T00:00:06Z"},
        ),
        PersistenceRecord(
            record_id="action_1",
            record_type="system_action",
            run_id="run_001",
            coordinate_id="coord_002",
            candidate_id="cand_42",
            seed=42,
            payload={
                "action_name": "infrastructure_retry",
                "semantic_penalty": False,
                "timestamp": "2026-01-01T00:00:07Z",
            },
        ),
        PersistenceRecord(
            record_id="ui_pause",
            record_type="ui_event",
            run_id="run_001",
            payload={
                "event_id": "evt_pause",
                "event_type": "run_pause_intent",
                "timestamp": "2026-01-01T00:00:08Z",
            },
        ),
        PersistenceRecord(
            record_id="ui_resume",
            record_type="ui_event",
            run_id="run_001",
            payload={
                "event_id": "evt_resume",
                "event_type": "pre_run_begin",
                "timestamp": "2026-01-01T00:00:09Z",
            },
        ),
        PersistenceRecord(
            record_id="transition_stall",
            record_type="loop_transition",
            run_id="run_001",
            payload={
                "action": "stop",
                "reason": "stall_guard_triggered",
                "next_state": "stopped",
                "timestamp": "2026-01-01T00:00:10Z",
            },
        ),
    ]

    events = event_stream_from_records(records)
    event_types = [event.event_type for event in events]

    assert event_types == [
        "run_started",
        "pre_run_parse_ready",
        "generation_queued",
        "generation_started",
        "image_generated",
        "image_promoted_curated",
        "iqa_evaluation_completed",
        "vlm_evaluation_completed",
        "feedback_accepted",
        "image_removed_from_visible_catalogue",
        "infrastructure_warning",
        "run_paused",
        "run_resumed",
        "run_stalled",
    ]
    generated = next(event for event in events if event.event_type == "image_generated")
    assert generated.event_id == "candidate_1:image_generated"
    assert generated.timestamp == "2026-01-01T00:00:03Z"
    assert generated.run_id == "run_001"
    assert generated.coordinate_id == "coord_001"
    assert generated.candidate_id == "cand_7"
    assert generated.lifecycle_state == "generated"
    assert generated.message == "Image generated."
    assert generated.payload_reference == "candidate_1"
    feedback = next(event for event in events if event.event_type == "feedback_accepted")
    assert feedback.candidate_id == "cand_7"
    assert feedback.lifecycle_state == "feedback_recorded"
    assert feedback.payload_reference == "feedback_1"
    assert next(event for event in events if event.event_type == "run_paused").event_id == "evt_pause"


def test_event_stream_from_records_projects_blocked_prompt_and_infrastructure_error():
    records = [
        PersistenceRecord(
            record_id="prompt_blocked_1",
            record_type="prompt_blocked",
            run_id="run_001",
            prompt_document_id="doc_001",
            payload={},
        ),
        PersistenceRecord(
            record_id="action_1",
            record_type="system_action",
            run_id="run_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            payload={"action_name": "infrastructure_retry", "semantic_penalty": True},
        ),
    ]

    events = event_stream_from_records(records)

    assert [event.event_type for event in events] == ["pre_run_parse_blocked", "infrastructure_error"]
    assert events[0].timestamp == "1970-01-01T00:00:00Z"


def test_detail_report_from_records_uses_generation_evaluation_and_feedback_payloads():
    records = [
        PersistenceRecord(
            record_id="candidate_1",
            record_type="candidate_record",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={
                "image_path": "/tmp/cand_7.png",
                "raw_user_prompt": "a red ceramic bowl on a wooden table",
                "prompt_document_version": "1",
                "promotion_thresholds": {
                    "quality_cutoff": 0.55,
                    "alignment_cutoff": 0.25,
                    "human_quality_cutoff": None,
                    "impact_cutoff": None,
                },
                "coordinate_enum_json": {
                    "locked_arms": {"object.material.object_01": "CERAMIC"},
                    "sampled_arms": {"cinematography.shot_size": "MEDIUM_SHOT"},
                    "combo_signature": "shot=MEDIUM_SHOT|material=CERAMIC",
                },
                "compatibility_trace": {
                    "prior_score": 0.88,
                    "downranks": [{"reason": "cool lighting weakens ceramic warmth"}],
                },
                "bayesian_score_before_generation": 0.73,
                "rendered_prompt": "Generate a red ceramic bowl on a wooden table",
                "generator_model_id": "stub-generator",
                "generator_backend": "stub",
                "generation_settings": {"steps": 4, "width": 512},
            },
        ),
        PersistenceRecord(
            record_id="eval_1",
            record_type="image_evaluation",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={
                "quality": {"score": 0.91},
                "alignment": {"score": 0.82},
                "disposition_signal": {
                    "class_name": "passes_thresholds",
                    "reasons": ["quality and alignment passed"],
                },
                "pass_flags": {"quality": True, "alignment": True, "full": True},
                "optional_tags": ["warm", "minimal"],
            },
        ),
        PersistenceRecord(
            record_id="feedback_1",
            record_type="feedback",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={"feedback_action": "accept"},
        ),
    ]

    detail = detail_report_from_records(records, "cand_7")

    assert detail is not None
    assert detail.candidate_id == "cand_7"
    assert detail.rendered_prompt.startswith("Generate ")
    assert detail.generator_backend == "stub"
    assert detail.image_path == "/tmp/cand_7.png"
    assert detail.raw_user_prompt == "a red ceramic bowl on a wooden table"
    assert detail.prompt_document_version == "1"
    assert detail.promotion_thresholds["quality_cutoff"] == 0.55
    assert detail.promotion_thresholds["alignment_cutoff"] == 0.25
    assert detail.coordinate_enum_json["sampled_arms"] == {"cinematography.shot_size": "MEDIUM_SHOT"}
    assert detail.compatibility_trace["prior_score"] == 0.88
    assert detail.bayesian_score_before_generation == 0.73
    assert detail.optional_tags == ["warm", "minimal"]
    assert detail.optional_tags_gate_curation is False
    assert detail.quality_score == 0.91
    assert detail.alignment_score == 0.82
    assert detail.promotion_gate_reasons == ["quality and alignment passed"]
    assert detail.feedback_state == "accept"


def test_render_workspace_selects_first_catalogue_candidate_detail(tmp_path: Path):
    store_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "workspace.html"
    store = JsonlEventStore(store_path)
    for record in [
        PersistenceRecord(
            record_id="run_1",
            record_type="run_config",
            run_id="run_001",
            payload={"raw_user_prompt": "a bowl on a table"},
        ),
        PersistenceRecord(
            record_id="candidate_1",
            record_type="candidate_record",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={
                "image_path": "/tmp/cand_7.png",
                "promotion_thresholds": {
                    "quality_cutoff": 0.55,
                    "alignment_cutoff": 0.25,
                    "human_quality_cutoff": None,
                    "impact_cutoff": None,
                },
                "coordinate_enum_json": {
                    "locked_arms": {"object.material.object_01": "CERAMIC"},
                    "sampled_arms": {"cinematography.shot_size": "MEDIUM_SHOT"},
                    "combo_signature": "shot=MEDIUM_SHOT|material=CERAMIC",
                },
                "rendered_prompt": "Generate a red ceramic bowl on a wooden table",
                "generator_model_id": "stub-generator",
                "generator_backend": "stub",
                "generation_settings": {"steps": 4},
            },
        ),
        PersistenceRecord(
            record_id="eval_1",
            record_type="image_evaluation",
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            candidate_id="cand_7",
            seed=7,
            payload={
                "quality": {"score": 0.91},
                "alignment": {"score": 0.82},
                "disposition_signal": {
                    "class_name": "passes_thresholds",
                    "reasons": ["quality and alignment passed"],
                },
                "pass_flags": {"quality": True, "alignment": True, "full": True},
            },
        ),
    ]:
        store.append(record)

    render_workspace_from_store(store_path, output_path)

    html = output_path.read_text(encoding="utf-8")
    assert "Selected Image" in html
    assert "Generate a red ceramic bowl on a wooden table" in html
    assert "Quality: 0.910" in html
    assert 'src="/tmp/cand_7.png"' in html
    assert "quality_cutoff: 0.55" in html
    assert 'data-region="coordinate-enum-json"' in html
    assert "cinematography.shot_size" in html


def test_render_workspace_from_empty_store_raises_clear_error(tmp_path: Path):
    try:
        render_workspace_from_store(tmp_path / "missing.jsonl", tmp_path / "workspace.html")
    except ValueError as error:
        assert str(error) == "no persisted records available"
    else:
        raise AssertionError("expected ValueError")


def test_cli_main_renders_workspace_with_arguments(tmp_path: Path):
    store_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "workspace.html"
    store = JsonlEventStore(store_path)
    store.append(
        PersistenceRecord(
            record_id="rec_001",
            record_type="run_config",
            run_id="run_001",
            payload={"raw_user_prompt": "a bowl on a table"},
        )
    )

    exit_code = main(["render-workspace", "--store", str(store_path), "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()


def test_cli_main_can_render_workspace_with_diagnostics(tmp_path: Path):
    store_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "workspace.html"
    store = JsonlEventStore(store_path)
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
            record_id="action_001",
            record_type="system_action",
            run_id="run_001",
            coordinate_id="coord_001",
            candidate_id="cand_42069",
            seed=42069,
            payload={
                "action_name": "infrastructure_retry",
                "semantic_penalty": False,
                "reasons": ["stubbed infrastructure block"],
            },
        )
    )

    exit_code = main(
        ["render-workspace", "--store", str(store_path), "--output", str(output_path), "--diagnostics"]
    )

    html = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert 'data-region="developer-diagnostics"' in html
    assert "Infrastructure retries: 1" in html


def test_cli_main_returns_nonzero_for_unknown_command(tmp_path: Path):
    assert main(["unknown-command"]) == 2
