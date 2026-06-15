from __future__ import annotations

import pytest
from pydantic import ValidationError
from typing import Any, Literal, cast

from bruteforce_canvas.prompt import (
    RetryRequest as PromptRetryRequest,
    ValidationIssue as PromptValidationIssue,
    ValidationSeverity as PromptValidationSeverity,
)
from bruteforce_canvas.shared import StrictModel
from bruteforce_canvas.validation import RetryRequest, ValidationIssue, ValidationSeverity

ValidationStage = Literal[
    "graph",
    "object_lane",
    "action_lane",
    "cinematography_lane",
    "constraint_lane",
    "document",
    "render",
]


def test_validation_severity_matches_spec_values() -> None:
    assert {member.value for member in ValidationSeverity} == {"error", "warning"}
    assert len(ValidationSeverity) == 2
    assert not hasattr(ValidationSeverity, "INFO")
    assert not hasattr(ValidationSeverity, "BLOCKING")


def test_validation_issue_requires_spec_fields() -> None:
    issue = ValidationIssue(
        stage="graph",
        severity=ValidationSeverity.ERROR,
        path="relations[0].target_id",
        message="relation target_id does not resolve to an element",
        retry_scope="relation",
    )

    assert issue.stage == "graph"
    assert issue.severity == "error"
    assert issue.path == "relations[0].target_id"
    assert issue.retry_scope == "relation"


def test_validation_issue_serializes_with_spec_field_names_only() -> None:
    issue = ValidationIssue(
        stage="object_lane",
        severity=ValidationSeverity.WARNING,
        path="objects[0].material",
        message="material should attach to textile element",
        retry_scope="object_descriptor",
    )

    assert issue.model_dump() == {
        "stage": "object_lane",
        "severity": "warning",
        "path": "objects[0].material",
        "message": "material should attach to textile element",
        "retry_scope": "object_descriptor",
    }


def test_validation_issue_allows_every_spec_stage() -> None:
    stages: tuple[ValidationStage, ...] = (
        "graph",
        "object_lane",
        "action_lane",
        "cinematography_lane",
        "constraint_lane",
        "document",
        "render",
    )

    for stage in stages:
        issue = ValidationIssue(
            stage=stage,
            severity=ValidationSeverity.ERROR,
            path="",
            message="test issue",
            retry_scope=stage,
        )
        assert issue.stage == stage


def test_validation_issue_rejects_non_spec_stage_and_old_shape() -> None:
    with pytest.raises(ValidationError):
        ValidationIssue(
            stage=cast(Any, "pipeline"),
            severity=ValidationSeverity.ERROR,
            path="",
            message="bad stage",
            retry_scope="pipeline",
        )

    with pytest.raises(ValidationError):
        ValidationIssue.model_validate(
            {
                "severity": ValidationSeverity.ERROR,
                "code": "old_code",
                "message": "old shape is forbidden",
                "field_path": "graph.relations[0]",
                "repair_scope": "graph",
            }
        )


def test_retry_request_requires_spec_fields() -> None:
    issue = ValidationIssue(
        stage="graph",
        severity=ValidationSeverity.ERROR,
        path="relations[0]",
        message="invalid relation",
        retry_scope="relation",
    )
    request = RetryRequest(
        failed_stage="graph",
        frozen_graph=None,
        invalid_payload={"relation_id": "rel_01"},
        issues=[issue],
        instruction="Return only a corrected relation JSON object.",
    )

    assert request.failed_stage == "graph"
    assert request.frozen_graph is None
    assert request.invalid_payload == {"relation_id": "rel_01"}
    assert request.issues == [issue]
    assert request.instruction.startswith("Return only")


def test_retry_request_serializes_cleanly() -> None:
    request = RetryRequest(
        failed_stage="constraint_lane",
        invalid_payload={"negative_phrases": ["handbag"]},
        issues=[
            ValidationIssue(
                stage="constraint_lane",
                severity=ValidationSeverity.ERROR,
                path="negative_phrases[0]",
                message="negative phrase conflicts with required handbag",
                retry_scope="constraint_lane",
            )
        ],
        instruction="Remove only the conflicting negative phrase.",
    )

    assert request.model_dump() == {
        "failed_stage": "constraint_lane",
        "frozen_graph": None,
        "invalid_payload": {"negative_phrases": ["handbag"]},
        "issues": [
            {
                "stage": "constraint_lane",
                "severity": "error",
                "path": "negative_phrases[0]",
                "message": "negative phrase conflicts with required handbag",
                "retry_scope": "constraint_lane",
            }
        ],
        "instruction": "Remove only the conflicting negative phrase.",
    }


def test_retry_request_rejects_old_retry_fields_and_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        RetryRequest.model_validate({"issues": []})

    with pytest.raises(ValidationError):
        RetryRequest.model_validate(
            {
                "max_retries": 3,
                "strategy": "one_issue_at_a_time",
                "issues": [],
            }
        )


def test_strict_model_rejects_coercion_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ValidationIssue(
            stage="graph",
            severity=ValidationSeverity.ERROR,
            path="",
            message=cast(Any, 123),
            retry_scope="graph",
        )

    with pytest.raises(ValidationError):
        RetryRequest(
            failed_stage="graph",
            invalid_payload=cast(Any, []),
            issues=[],
            instruction="repair graph",
        )


def test_validation_models_inherit_strict_model_config() -> None:
    assert issubclass(ValidationIssue, StrictModel)
    assert issubclass(RetryRequest, StrictModel)

    for model_cls in (ValidationIssue, RetryRequest):
        config = model_cls.model_config
        assert config.get("frozen") is True
        assert config.get("extra") == "forbid"
        assert config.get("strict") is True
        assert config.get("use_enum_values") is True


def test_validation_models_are_reexported_from_prompt() -> None:
    assert PromptValidationSeverity is ValidationSeverity
    assert PromptValidationIssue is ValidationIssue
    assert PromptRetryRequest is RetryRequest
