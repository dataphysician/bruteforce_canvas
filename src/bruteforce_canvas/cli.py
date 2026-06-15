from __future__ import annotations

import argparse
from pathlib import Path

from bruteforce_canvas.persistence import JsonlEventStore, PersistenceRecord, reconstruct_run_state
from bruteforce_canvas.static_ui import render_workspace_html
from bruteforce_canvas.ui import (
    CandidateCard,
    DetailReport,
    DiagnosticsReadModel,
    RunWorkspaceReadModel,
    UIStreamEvent,
    catalogue_default_items,
)
from bruteforce_canvas.shared import FeedbackAction

DEFAULT_EVENT_TIMESTAMP = "1970-01-01T00:00:00Z"


def candidate_cards_from_records(records: list[PersistenceRecord]) -> list[CandidateCard]:
    cards: dict[str, CandidateCard] = {}
    for record in records:
        if record.candidate_id is None:
            continue
        current = cards.get(
            record.candidate_id,
            CandidateCard(candidate_id=record.candidate_id, promoted=False, curated=False, seed=record.seed),
        )
        if record.record_type == "candidate_record":
            cards[record.candidate_id] = current.model_copy(
                update={"seed": record.seed, "thumbnail_path": record.payload.get("image_path")}
            )
        elif record.record_type == "image_evaluation":
            full_pass = bool(record.payload.get("pass_flags", {}).get("full", False))
            disposition = record.payload.get("disposition_signal", {}).get("class_name")
            curated = full_pass and disposition == "passes_thresholds"
            cards[record.candidate_id] = current.model_copy(
                update={
                    "promoted": full_pass,
                    "curated": curated,
                    "seed": record.seed,
                    "optional_tags": list(record.payload.get("optional_tags", [])),
                }
            )
        elif record.record_type == "feedback":
            action = FeedbackAction(record.payload["feedback_action"])
            cards[record.candidate_id] = current.model_copy(
                update={
                    "feedback_action": action,
                    "accepted": action == FeedbackAction.ACCEPT,
                    "seed": record.seed,
                }
            )
    return list(cards.values())


def _stream_timestamp(record: PersistenceRecord) -> str:
    value = record.payload.get("timestamp")
    if isinstance(value, str) and value:
        return value
    value = record.payload.get("generation_timestamp")
    if isinstance(value, str) and value:
        return value
    return DEFAULT_EVENT_TIMESTAMP


def _stream_event(
    record: PersistenceRecord,
    *,
    event_type: str,
    lifecycle_state: str,
    message: str,
    suffix: str | None = None,
) -> UIStreamEvent:
    event_id = str(record.payload.get("event_id") or f"{record.record_id}:{suffix or event_type}")
    return UIStreamEvent(
        event_id=event_id,
        timestamp=_stream_timestamp(record),
        event_type=event_type,
        run_id=record.run_id,
        coordinate_id=record.coordinate_id,
        candidate_id=record.candidate_id,
        lifecycle_state=lifecycle_state,
        message=message,
        payload_reference=record.record_id,
    )


def event_stream_from_records(records: list[PersistenceRecord]) -> list[UIStreamEvent]:
    events: list[UIStreamEvent] = []
    started_coordinates: set[str] = set()
    for record in records:
        if record.record_type == "run_config":
            events.append(
                _stream_event(
                    record,
                    event_type="run_started",
                    lifecycle_state="running",
                    message="Run started.",
                )
            )
        elif record.record_type == "prompt_document":
            events.append(
                _stream_event(
                    record,
                    event_type="pre_run_parse_ready",
                    lifecycle_state="pre_run_ready",
                    message="Pre-run parse ready.",
                )
            )
        elif record.record_type == "prompt_blocked":
            events.append(
                _stream_event(
                    record,
                    event_type="pre_run_parse_blocked",
                    lifecycle_state="pre_run_blocked",
                    message="Pre-run parse blocked.",
                )
            )
        elif record.record_type == "loop_transition":
            action = record.payload.get("action")
            next_state = str(record.payload.get("next_state", ""))
            reason = str(record.payload.get("reason", ""))
            if action == "generate_pending_coordinate":
                events.append(
                    _stream_event(
                        record,
                        event_type="generation_queued",
                        lifecycle_state=next_state or "running",
                        message="Generation queued.",
                    )
                )
            elif action == "pause":
                events.append(
                    _stream_event(
                        record,
                        event_type="run_paused",
                        lifecycle_state=next_state or "paused",
                        message="Run paused.",
                    )
                )
            elif action == "stop" and reason in {"stall_guard", "stall_guard_triggered"}:
                events.append(
                    _stream_event(
                        record,
                        event_type="run_stalled",
                        lifecycle_state=next_state or "stopped",
                        message="Run stalled.",
                    )
                )
            elif action == "stop":
                events.append(
                    _stream_event(
                        record,
                        event_type="run_stopped",
                        lifecycle_state=next_state or "stopped",
                        message="Run stopped.",
                    )
                )
        elif record.record_type == "ui_event":
            ui_type = record.payload.get("event_type")
            if ui_type == "run_pause_intent":
                events.append(
                    _stream_event(
                        record,
                        event_type="run_paused",
                        lifecycle_state="paused",
                        message="Run paused.",
                    )
                )
            elif ui_type == "pre_run_begin":
                events.append(
                    _stream_event(
                        record,
                        event_type="run_resumed",
                        lifecycle_state="running",
                        message="Run resumed.",
                    )
                )
            elif ui_type == "run_stop_intent":
                events.append(
                    _stream_event(
                        record,
                        event_type="run_stopped",
                        lifecycle_state="stopped",
                        message="Run stopped.",
                    )
                )
        elif record.record_type == "candidate_record":
            if record.coordinate_id and record.coordinate_id not in started_coordinates:
                started_coordinates.add(record.coordinate_id)
                events.append(
                    _stream_event(
                        record,
                        event_type="generation_started",
                        lifecycle_state="generating",
                        message="Generation started.",
                        suffix="generation_started",
                    )
                )
            events.append(
                _stream_event(
                    record,
                    event_type="image_generated",
                    lifecycle_state="generated",
                    message="Image generated.",
                )
            )
        elif record.record_type == "evaluation_aggregate":
            events.append(
                _stream_event(
                    record,
                    event_type="iqa_evaluation_completed",
                    lifecycle_state="evaluating_iqa",
                    message="IQA evaluation completed.",
                    suffix="iqa_evaluation_completed",
                )
            )
            events.append(
                _stream_event(
                    record,
                    event_type="vlm_evaluation_completed",
                    lifecycle_state="evaluating_vlm",
                    message="VLM evaluation completed.",
                    suffix="vlm_evaluation_completed",
                )
            )
        elif record.record_type == "image_evaluation":
            pass_flags = record.payload.get("pass_flags", {})
            disposition = record.payload.get("disposition_signal", {})
            if pass_flags.get("full") is True and disposition.get("class_name") == "passes_thresholds":
                events.append(
                    _stream_event(
                        record,
                        event_type="image_promoted_curated",
                        lifecycle_state="curated",
                        message="Image promoted to curated catalogue.",
                    )
                )
        elif record.record_type == "feedback":
            action = record.payload.get("feedback_action")
            if action == "accept":
                events.append(
                    _stream_event(
                        record,
                        event_type="feedback_accepted",
                        lifecycle_state="curated",
                        message="Feedback accepted.",
                    )
                )
            elif action in {"reject", "shred"}:
                events.append(
                    _stream_event(
                        record,
                        event_type="image_removed_from_visible_catalogue",
                        lifecycle_state="removed",
                        message="Image removed from visible catalogue.",
                    )
                )
        elif record.record_type == "system_action" and record.payload.get("action_name") == "infrastructure_retry":
            severity = "infrastructure_error" if record.payload.get("semantic_penalty") is True else "infrastructure_warning"
            events.append(
                _stream_event(
                    record,
                    event_type=severity,
                    lifecycle_state="blocked",
                    message="Infrastructure retry recorded.",
                )
            )
    return events


def detail_report_from_records(records: list[PersistenceRecord], candidate_id: str) -> DetailReport | None:
    candidate = next(
        (
            record
            for record in reversed(records)
            if record.record_type == "candidate_record" and record.candidate_id == candidate_id
        ),
        None,
    )
    evaluation = next(
        (
            record
            for record in reversed(records)
            if record.record_type == "image_evaluation" and record.candidate_id == candidate_id
        ),
        None,
    )
    if candidate is None or evaluation is None:
        return None

    card = next(
        (card for card in candidate_cards_from_records(records) if card.candidate_id == candidate_id),
        CandidateCard(candidate_id=candidate_id, promoted=False, curated=False, seed=candidate.seed),
    )
    disposition = evaluation.payload.get("disposition_signal", {})
    return DetailReport.from_candidate_card(
        card,
        run_id=evaluation.run_id,
        raw_user_prompt=str(candidate.payload.get("raw_user_prompt", "")),
        prompt_document_id=evaluation.prompt_document_id or "",
        prompt_document_version=str(candidate.payload.get("prompt_document_version", "1")),
        target_manifest_id=evaluation.target_manifest_id or "",
        coordinate_id=evaluation.coordinate_id or "",
        rendered_prompt=str(candidate.payload.get("rendered_prompt", "")),
        generator_model_id=str(candidate.payload.get("generator_model_id", "")),
        generator_backend=str(candidate.payload.get("generator_backend", "")),
        generation_settings=dict(candidate.payload.get("generation_settings", {})),
        coordinate_enum_json=dict(candidate.payload.get("coordinate_enum_json", {})),
        compatibility_trace=dict(candidate.payload.get("compatibility_trace", {})),
        bayesian_score_before_generation=candidate.payload.get("bayesian_score_before_generation"),
        quality_score=float(evaluation.payload.get("quality", {}).get("score", 0.0)),
        alignment_score=float(evaluation.payload.get("alignment", {}).get("score", 0.0)),
        promotion_thresholds=dict(candidate.payload.get("promotion_thresholds", {})),
        promotion_gate_reasons=list(disposition.get("reasons", [])),
        image_path=candidate.payload.get("image_path"),
    )


def diagnostics_from_records(records: list[PersistenceRecord]) -> DiagnosticsReadModel:
    record_counts: dict[str, int] = {}
    for record in records:
        record_counts[record.record_type] = record_counts.get(record.record_type, 0) + 1
    system_actions = [record for record in records if record.record_type == "system_action"]
    actions = [
        {
            "action_name": record.payload.get("action_name"),
            "candidate_id": record.candidate_id,
            "coordinate_id": record.coordinate_id,
            "semantic_penalty": record.payload.get("semantic_penalty", True),
            "reasons": record.payload.get("reasons", []),
        }
        for record in system_actions[-10:]
    ]
    retries = [action for action in actions if action["action_name"] == "infrastructure_retry"]
    return DiagnosticsReadModel(
        record_counts=record_counts,
        system_action_count=len(system_actions),
        infrastructure_retry_count=sum(
            1 for record in system_actions if record.payload.get("action_name") == "infrastructure_retry"
        ),
        infrastructure_retries=retries,
        recent_system_actions=actions,
    )


def render_workspace_from_store(store_path: Path, output_path: Path, *, include_diagnostics: bool = False) -> Path:
    store = JsonlEventStore(store_path)
    records = store.replay()
    if not records:
        raise ValueError("no persisted records available")
    state = reconstruct_run_state(records)
    workspace = RunWorkspaceReadModel(
        run_id=state.run_id,
        raw_user_prompt=state.raw_user_prompt or "",
        run_state="replayed",
        generated_count=state.generated_count,
        iqa_evaluated_count=state.iqa_evaluated_count,
        vlm_evaluated_count=state.vlm_evaluated_count,
        promoted_curated_count=state.promoted_curated_count,
        accepted_count=state.accepted_count,
        rejected_count=state.rejected_count,
        shredded_count=state.shredded_count,
        stall_guard_state="replayed",
        notification="Loaded persisted run state.",
        elapsed_seconds=state.elapsed_seconds,
    )
    catalogue = catalogue_default_items(candidate_cards_from_records(records))
    selected = detail_report_from_records(records, catalogue[0].candidate_id) if catalogue else None
    html = render_workspace_html(
        workspace,
        catalogue=catalogue,
        selected=selected,
        diagnostics=diagnostics_from_records(records) if include_diagnostics else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bruteforce-canvas")
    subparsers = parser.add_subparsers(dest="command")
    render = subparsers.add_parser("render-workspace")
    render.add_argument("--store", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--diagnostics", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    if args.command == "render-workspace":
        render_workspace_from_store(Path(args.store), Path(args.output), include_diagnostics=args.diagnostics)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
