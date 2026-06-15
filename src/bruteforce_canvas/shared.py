from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints


DocId = Annotated[str, StringConstraints(pattern=r"^doc_[0-9]{3,}$")]
RunId = Annotated[str, StringConstraints(pattern=r"^run_[0-9]{3,}$")]
TargetManifestId = Annotated[str, StringConstraints(pattern=r"^eval_manifest_[0-9]{3,}$")]
CoordinateId = Annotated[str, StringConstraints(pattern=r"^coord_[0-9]{3,}$")]
CandidateId = Annotated[str, StringConstraints(pattern=r"^cand_[A-Za-z0-9_]+$")]
ElementId = Annotated[str, StringConstraints(pattern=r"^[a-z]+_[0-9]{2}$")]
RelationId = Annotated[str, StringConstraints(pattern=r"^rel_[0-9]{2}$")]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=180)]

Confidence = Literal["high", "medium", "low"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, use_enum_values=True)


class CanonicalStatus(StrEnum):
    MATCHED_ACTIVE = "matched_active"
    MATCHED_SUPPRESSED = "matched_suppressed"
    MATCHED_DIAGNOSTIC_HOLD = "matched_diagnostic_hold"
    UNMATCHED_RAW_ONLY = "unmatched_raw_only"
    PROPOSED_NEW_ENUM = "proposed_new_enum"
    REJECTED_INVALID = "rejected_invalid"


class CandidateLifecycle(StrEnum):
    PROPOSED = "proposed"
    RENDERED = "rendered"
    GENERATING = "generating"
    GENERATED = "generated"
    EVALUATING_IQA = "evaluating_iqa"
    EVALUATING_VLM = "evaluating_vlm"
    EVALUATING_IMPACT = "evaluating_impact"
    EVALUATED = "evaluated"
    PROMOTED = "promoted"
    CURATED = "curated"
    STRONG = "strong"
    VIABLE = "viable"
    FRAGILE = "fragile"
    FAILED = "failed"
    DEMOTED = "demoted"
    RETIRED = "retired"
    QUARANTINED = "quarantined"
    BLOCKED = "blocked"


class FeedbackAction(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    SHRED = "shred"


def stable_id(prefix: str, number: int, width: int = 3) -> str:
    return f"{prefix}_{number:0{width}d}"
