from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from bruteforce_canvas.evaluation import CoordinateEvaluationAggregate, ImageEvaluationResult
from bruteforce_canvas.shared import StrictModel


class ActionName(StrEnum):
    PERSIST_FOR_LEARNING = "persist_for_learning"
    PROMOTE_CURATE = "promote_curate"
    DEMOTE_CANDIDATE = "demote_candidate"
    RETIRE_COORDINATE = "retire_coordinate"
    QUARANTINE_COORDINATE = "quarantine_coordinate"
    SUPPRESS_ENUM_ARM = "suppress_enum_arm"
    HARD_PURGE_INVALID_ARTIFACT = "hard_purge_invalid_artifact"
    INFRASTRUCTURE_RETRY = "infrastructure_retry"


class SystemAction(StrictModel):
    name: ActionName
    run_id: str
    candidate_id: str | None = None
    coordinate_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    semantic_penalty: bool = True


def decide_image_actions(result: ImageEvaluationResult) -> list[SystemAction]:
    signal = result.disposition_signal.class_name
    base = {
        "run_id": result.run_id,
        "candidate_id": result.candidate_id,
        "coordinate_id": result.coordinate_id,
        "reasons": result.disposition_signal.reasons,
    }
    if signal == "passes_thresholds" and result.pass_flags.get("full", False):
        return [SystemAction(name=ActionName.PROMOTE_CURATE, **base)]
    if signal == "demote_candidate":
        return [SystemAction(name=ActionName.DEMOTE_CANDIDATE, **base)]
    if signal == "hard_purge_invalid_artifact":
        return [SystemAction(name=ActionName.HARD_PURGE_INVALID_ARTIFACT, semantic_penalty=False, **base)]
    if signal == "infrastructure_retry_no_semantic_penalty":
        return [SystemAction(name=ActionName.INFRASTRUCTURE_RETRY, semantic_penalty=False, **base)]
    return [SystemAction(name=ActionName.PERSIST_FOR_LEARNING, **base)]


def decide_coordinate_actions(
    aggregate: CoordinateEvaluationAggregate,
    *,
    quarantine: bool = False,
) -> list[SystemAction]:
    base = {
        "run_id": aggregate.run_id,
        "coordinate_id": aggregate.coordinate_id,
        "reasons": aggregate.aggregate_failure_types,
    }
    if quarantine:
        return [SystemAction(name=ActionName.QUARANTINE_COORDINATE, **base)]
    if aggregate.outcome in {"failed", "fragile"}:
        return [SystemAction(name=ActionName.RETIRE_COORDINATE, **base)]
    return []
