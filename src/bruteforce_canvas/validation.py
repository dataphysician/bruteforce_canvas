from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from bruteforce_canvas.shared import StrictModel

if TYPE_CHECKING:
    from bruteforce_canvas.prompt_models import SceneGraphDraft
else:
    SceneGraphDraft = Any


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(StrictModel):
    stage: Literal[
        "graph",
        "object_lane",
        "action_lane",
        "cinematography_lane",
        "constraint_lane",
        "document",
        "render",
    ]
    severity: ValidationSeverity
    path: str
    message: str
    retry_scope: str


class RetryRequest(StrictModel):
    failed_stage: str
    frozen_graph: SceneGraphDraft | None = None
    invalid_payload: dict
    issues: list[ValidationIssue]
    instruction: str


__all__ = [
    "RetryRequest",
    "ValidationIssue",
    "ValidationSeverity",
]
