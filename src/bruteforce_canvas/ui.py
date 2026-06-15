from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, computed_field

from bruteforce_canvas.locking import build_default_lock_config
from bruteforce_canvas.prompt import PromptDocument
from bruteforce_canvas.shared import CandidateId, FeedbackAction, RunId, StrictModel


class RunControl(StrEnum):
    START = "start"
    PAUSE = "pause"
    STOP = "stop"


class PreRunModalState(StrEnum):
    REVIEW = "review"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    READY_TO_BEGIN = "ready_to_begin"


class PreRunModalReadModel(StrictModel):
    prompt_document_id: str
    state: PreRunModalState
    can_begin_generation: bool
    parsed_elements: list[str]
    parsed_relations: list[str]
    prompt_improvement_feedback: list[str] = Field(default_factory=list)
    editable_fields: list[str] = Field(default_factory=list)
    lock_entries: list[dict[str, Any]] = Field(default_factory=list)


class CandidateCard(StrictModel):
    candidate_id: CandidateId
    promoted: bool
    curated: bool
    feedback_action: FeedbackAction | None = None
    feedback_pending: bool = False
    accepted: bool = False
    thumbnail_path: str | None = None
    seed: int | None = None
    optional_tags: list[str] = Field(default_factory=list)


class PreRunEditableField(StrictModel):
    field_path: str
    value: str


class GraphEditRejected(ValueError):
    pass


class UIEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    event_type: Literal[
        "run_start_intent",
        "run_pause_intent",
        "run_stop_intent",
        "pre_run_begin",
        "pre_run_cancel",
        "feedback_submitted",
    ]
    run_id: RunId
    payload: dict[str, Any]


class UIStreamEvent(StrictModel):
    event_id: str
    timestamp: str
    event_type: Literal[
        "run_started",
        "pre_run_parse_ready",
        "pre_run_parse_blocked",
        "generation_queued",
        "generation_started",
        "image_generated",
        "iqa_evaluation_completed",
        "vlm_evaluation_completed",
        "image_promoted_curated",
        "feedback_accepted",
        "image_removed_from_visible_catalogue",
        "run_paused",
        "run_resumed",
        "run_stopped",
        "run_stalled",
        "infrastructure_warning",
        "infrastructure_error",
    ]
    run_id: RunId
    coordinate_id: str | None = None
    candidate_id: str | None = None
    lifecycle_state: str
    message: str
    payload_reference: str | None = None


class DetailReport(StrictModel):
    candidate_id: CandidateId
    run_id: RunId
    raw_user_prompt: str = ""
    prompt_document_id: str
    prompt_document_version: str = "1"
    target_manifest_id: str
    coordinate_id: str
    image_path: str | None = None
    rendered_prompt: str
    seed: int | None
    generator_model_id: str
    generator_backend: str
    generation_settings: dict[str, Any]
    coordinate_enum_json: dict[str, Any] = Field(default_factory=dict)
    compatibility_trace: dict[str, Any] = Field(default_factory=dict)
    bayesian_score_before_generation: float | None = None
    quality_score: float
    alignment_score: float
    promotion_thresholds: dict[str, Any] = Field(default_factory=dict)
    promotion_gate_reasons: list[str]
    optional_tags: list[str] = Field(default_factory=list)
    optional_tags_gate_curation: bool = False
    feedback_state: str | None = None
    feedback_pending: bool = False

    @classmethod
    def from_candidate_card(
        cls,
        card: CandidateCard,
        *,
        run_id: str,
        raw_user_prompt: str = "",
        prompt_document_id: str,
        prompt_document_version: str = "1",
        target_manifest_id: str,
        coordinate_id: str,
        rendered_prompt: str,
        generator_model_id: str,
        generator_backend: str,
        generation_settings: dict[str, Any],
        coordinate_enum_json: dict[str, Any] | None = None,
        compatibility_trace: dict[str, Any] | None = None,
        bayesian_score_before_generation: float | None = None,
        quality_score: float,
        alignment_score: float,
        promotion_thresholds: dict[str, Any] | None = None,
        promotion_gate_reasons: list[str],
        image_path: str | None = None,
    ) -> "DetailReport":
        return cls(
            candidate_id=card.candidate_id,
            run_id=run_id,
            raw_user_prompt=raw_user_prompt,
            prompt_document_id=prompt_document_id,
            prompt_document_version=prompt_document_version,
            target_manifest_id=target_manifest_id,
            coordinate_id=coordinate_id,
            image_path=image_path or card.thumbnail_path,
            rendered_prompt=rendered_prompt,
            seed=card.seed,
            generator_model_id=generator_model_id,
            generator_backend=generator_backend,
            generation_settings=generation_settings,
            coordinate_enum_json=coordinate_enum_json or {},
            compatibility_trace=compatibility_trace or {},
            bayesian_score_before_generation=bayesian_score_before_generation,
            quality_score=quality_score,
            alignment_score=alignment_score,
            promotion_thresholds=promotion_thresholds or {},
            promotion_gate_reasons=promotion_gate_reasons,
            optional_tags=card.optional_tags,
            feedback_state=str(card.feedback_action) if card.feedback_action else None,
            feedback_pending=card.feedback_pending,
        )


class RunWorkspaceReadModel(StrictModel):
    run_id: RunId
    raw_user_prompt: str
    run_state: str
    generated_count: int
    iqa_evaluated_count: int
    vlm_evaluated_count: int
    promoted_curated_count: int
    accepted_count: int
    rejected_count: int
    shredded_count: int
    stall_guard_state: str
    notification: str
    elapsed_seconds: int = 0

    @computed_field
    @property
    def progress_heartbeat(self) -> dict[str, int | str]:
        return {
            "run_state": self.run_state,
            "generated_count": self.generated_count,
            "iqa_evaluated_count": self.iqa_evaluated_count,
            "vlm_evaluated_count": self.vlm_evaluated_count,
            "promoted_curated_count": self.promoted_curated_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "shredded_count": self.shredded_count,
            "stall_guard_state": self.stall_guard_state,
            "elapsed_seconds": self.elapsed_seconds,
        }


class DiagnosticsReadModel(StrictModel):
    record_counts: dict[str, int] = Field(default_factory=dict)
    system_action_count: int = 0
    infrastructure_retry_count: int = 0
    infrastructure_retries: list[dict[str, object]] = Field(default_factory=list)
    recent_system_actions: list[dict[str, object]] = Field(default_factory=list)


def pre_run_modal_from_prompt(document: PromptDocument) -> PreRunModalReadModel:
    blocking_issues = [issue for issue in document.verification.issues if issue.blocking]
    state = PreRunModalState.BLOCKED if blocking_issues or not document.verification.approved else PreRunModalState.REVIEW
    lock_config = build_default_lock_config(document)
    return PreRunModalReadModel(
        prompt_document_id=document.prompt_document_id,
        state=state,
        can_begin_generation=state != PreRunModalState.BLOCKED,
        parsed_elements=[f"{element.element_id}: {element.label}" for element in document.graph.elements],
        parsed_relations=[
            f"{relation.source_id} {relation.relation_raw} {relation.target_id}" for relation in document.graph.relations
        ],
        prompt_improvement_feedback=[issue.message for issue in blocking_issues],
        editable_fields=[
            field
            for field, value in document.cinematography.model_dump().items()
            if value is not None and field.endswith("_raw")
        ],
        lock_entries=[
            {
                "field_path": entry.field_path,
                "raw_value": entry.raw_value,
                "enum_value": entry.enum_value,
                "canonical_status": entry.canonical_status,
                "lhs_policy": entry.lhs_policy,
                "lock_state": str(entry.lock_state),
                "user_adjustable": entry.user_adjustable,
            }
            for entry in lock_config.entries
        ],
    )


def catalogue_default_items(cards: list[CandidateCard]) -> list[CandidateCard]:
    excluded_feedback = {FeedbackAction.REJECT, FeedbackAction.SHRED}
    return [
        card
        for card in cards
        if card.promoted and card.curated and card.feedback_action not in excluded_feedback
    ]


def run_control_event(*, control: RunControl, run_id: str, prompt: str) -> UIEvent:
    if control == RunControl.START:
        if not prompt.strip():
            raise ValueError("Start requires a non-empty prompt")
        return UIEvent(
            event_type="run_start_intent",
            run_id=run_id,
            payload={"prompt": prompt, "opens_pre_run_modal": True, "begin_generation": False},
        )
    if control == RunControl.PAUSE:
        return UIEvent(
            event_type="run_pause_intent",
            run_id=run_id,
            payload={"discard_history": False, "cancel_in_flight_persistence": False},
        )
    return UIEvent(
        event_type="run_stop_intent",
        run_id=run_id,
        payload={"erase_run_history": False},
    )


def begin_generation_event(*, run_id: str, modal_state: PreRunModalState) -> UIEvent:
    if modal_state != PreRunModalState.READY_TO_BEGIN:
        raise ValueError("pre-run modal is not ready to begin generation")
    return UIEvent(event_type="pre_run_begin", run_id=run_id, payload={"begin_generation": True})


def cancel_pre_run_event(*, run_id: str) -> UIEvent:
    return UIEvent(
        event_type="pre_run_cancel",
        run_id=run_id,
        payload={"begin_generation": False, "erase_run_history": False},
    )


def validate_pre_run_edit(field: PreRunEditableField) -> PreRunEditableField:
    disallowed_prefixes = (
        "graph.",
        "elements.",
        "relations.",
        "actions.",
        "target_manifest.",
        "constraints.negative_guard_policy",
        "compatibility_policy.",
        "evaluator_policy.",
    )
    if field.field_path.startswith(disallowed_prefixes):
        raise GraphEditRejected("pre-run modal does not allow graph-breaking edits")
    return field


def submit_feedback_event(*, run_id: str, candidate_id: str, action: FeedbackAction) -> UIEvent:
    return UIEvent(
        event_type="feedback_submitted",
        run_id=run_id,
        payload={"candidate_id": candidate_id, "action": action.value},
    )
