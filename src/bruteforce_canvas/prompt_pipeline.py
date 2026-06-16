from __future__ import annotations

from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import Field

from bruteforce_canvas.prompt import (
    CanonicalEnum,
    RenderedPrompt,
    RelationEnumMatch,
    VerificationIssue,
    VerificationReport,
    render_prompt_spec,
)
from bruteforce_canvas.prompt_enums import (
    EnumMatchConfidence,
    RelationType,
)
from bruteforce_canvas.prompt_models import (
    EnumMatch,
    PromptDocumentSpec,
)
from bruteforce_canvas.shared import CanonicalStatus, StrictModel
from bruteforce_canvas.validation import RetryRequest, ValidationIssue
from bruteforce_canvas.validators import (
    validate_action_support,
    validate_cross_lane_coherence,
    validate_evidence_and_placeholders,
    validate_object_ownership,
    validate_relation_compatibility,
    validators_to_validation_issues,
)


_EnumT = TypeVar("_EnumT", bound=StrEnum)


class ExtractionAdapter(Protocol):
    def extract(self, raw_prompt: str) -> PromptDocumentSpec:
        """Return a draft spec prompt document from one raw user prompt."""
        ...


class CanonicalizerAdapter(Protocol):
    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        """Canonicalize one field-scoped raw value without rewriting the raw phrase."""
        ...


class VerificationAdapter(Protocol):
    def verify(self, document: PromptDocumentSpec) -> VerificationReport:
        """Verify graph linkage, lane ownership, enum fit, and renderability."""
        ...


class RepairAdapter(Protocol):
    def repair(
        self,
        document: PromptDocumentSpec,
        issue: RetryRequest | VerificationIssue,
    ) -> PromptDocumentSpec:
        """Repair only the slice identified by the retry request or verifier issue."""
        ...


class PromptPipelineSpecResult(StrictModel):
    approved: bool
    document: PromptDocumentSpec
    verifier_report: VerificationReport
    rendered_prompt: RenderedPrompt | None = None
    prompt_improvement_feedback: list[str] = Field(default_factory=list)


class _SemanticValidationResult(StrictModel):
    document: PromptDocumentSpec
    issues: list[ValidationIssue] = Field(default_factory=list)


_SEMANTIC_RETRY_LIMITS: dict[str, int] = {
    "graph_skeleton": 2,
    "object_lane": 2,
    "object_descriptor": 2,
    "action_lane": 2,
    "action_descriptor": 2,
    "relation": 2,
    "constraint_lane": 2,
    "document": 2,
    "cross_lane_coherence": 2,
    "evidence_or_placeholder": 2,
}


def _retry_limit(retry_scope: str) -> int:
    return _SEMANTIC_RETRY_LIMITS.get(retry_scope, 2)


def _retry_instruction(retry_scope: str) -> str:
    scope = retry_scope.replace("_", " ")
    return f"Repair the {scope} validation issue while preserving unrelated graph and lane content."


def _validation_report_from_issues(issues: list[ValidationIssue]) -> VerificationReport:
    return VerificationReport(
        approved=False,
        issues=[
            VerificationIssue(
                issue_type=issue.stage,
                repair_scope=issue.retry_scope,
                blocking=True,
                message=issue.message,
            )
            for issue in issues
        ],
    )


def _retry_request(document: PromptDocumentSpec, issues: list[ValidationIssue]) -> RetryRequest:
    first_issue = issues[0]
    return RetryRequest(
        failed_stage=first_issue.stage,
        frozen_graph=document.graph,
        invalid_payload={"issues": [issue.model_dump() for issue in issues]},
        issues=issues,
        instruction=_retry_instruction(first_issue.retry_scope),
    )


def _coerce_str_enum(enum_type: type[_EnumT], raw_value: str | None) -> _EnumT | None:
    if raw_value is None:
        return None
    normalized = raw_value.strip().lower().replace(" ", "_").replace("-", "_")
    if not normalized:
        return None
    for member in enum_type:
        if normalized in {member.value, member.name.lower()}:
            return member
    return None


def _canonical_status(status: str) -> CanonicalStatus:
    coerced = _coerce_str_enum(CanonicalStatus, status)
    return coerced if coerced is not None else CanonicalStatus.UNMATCHED_RAW_ONLY


def _canonical_confidence(confidence: str) -> EnumMatchConfidence:
    if confidence == "high":
        return EnumMatchConfidence.CLEAR
    return EnumMatchConfidence.UNCLEAR


def _relation_match(canonical: CanonicalEnum | None, relation_raw: str) -> RelationEnumMatch | None:
    if canonical is None:
        return None
    enum_value = _coerce_str_enum(RelationType, canonical.enum_value)
    status = _canonical_status(canonical.status)
    if status == CanonicalStatus.MATCHED_ACTIVE and enum_value is None:
        status = CanonicalStatus.UNMATCHED_RAW_ONLY
    return RelationEnumMatch(
        raw=canonical.raw_value or relation_raw,
        enum_value=enum_value,
        status=status,
        confidence="clear" if canonical.confidence == "high" else "weak",
        reason=canonical.reason,
    )


def _movement_match(canonical: CanonicalEnum | None) -> EnumMatch | None:
    if canonical is None:
        return None
    return EnumMatch(
        raw=canonical.raw_value,
        enum_value=canonical.enum_value.lower() if canonical.enum_value else None,
        confidence=_canonical_confidence(canonical.confidence),
        reason=canonical.reason,
    )


class PromptPipeline:
    def __init__(
        self,
        extractor: ExtractionAdapter,
        canonicalizer: CanonicalizerAdapter,
        verifier: VerificationAdapter,
        repairer: RepairAdapter,
        *,
        max_repairs: int = 2,
        max_semantic_repairs: int | None = None,
        run_semantic_validation: bool = True,
        run_verifier: bool = True,
    ) -> None:
        self.extractor = extractor
        self.canonicalizer = canonicalizer
        self.verifier = verifier
        self.repairer = repairer
        self.max_repairs = max_repairs
        self.max_semantic_repairs = max_semantic_repairs
        self.run_semantic_validation = run_semantic_validation
        self.run_verifier = run_verifier

    def run_spec(self, raw_prompt: str) -> PromptPipelineSpecResult:
        extracted = self.extractor.extract(raw_prompt)
        return self._run_spec_document(extracted)

    def run(self, raw_prompt: str) -> PromptPipelineSpecResult:
        return self.run_spec(raw_prompt)

    def _run_spec_document(self, document: PromptDocumentSpec) -> PromptPipelineSpecResult:
        document = self._canonicalize_spec_document(document)
        if self.run_semantic_validation:
            semantic_result = self._repair_semantic_validation_issues(document)
            document = semantic_result.document
            if semantic_result.issues:
                report = _validation_report_from_issues(semantic_result.issues)
                document = document.model_copy(update={"verification": report})
                return PromptPipelineSpecResult(
                    approved=False,
                    document=document,
                    verifier_report=report,
                    prompt_improvement_feedback=[issue.message for issue in semantic_result.issues],
                )

        if not self.run_verifier:
            report = VerificationReport(approved=True, issues=[])
            document = document.model_copy(update={"verification": report})
            try:
                rendered = render_prompt_spec(document)
            except Exception as error:
                blocked = VerificationReport(
                    approved=False,
                    issues=[
                        VerificationIssue(
                            issue_type="render_failed",
                            repair_scope="prompt_improvement",
                            blocking=True,
                            message=f"Compiled prompt could not render: {str(error)[:120]}",
                        )
                    ],
                )
                document = document.model_copy(update={"verification": blocked})
                return PromptPipelineSpecResult(
                    approved=False,
                    document=document,
                    verifier_report=blocked,
                    prompt_improvement_feedback=[issue.message for issue in blocked.issues],
                )
            return PromptPipelineSpecResult(
                approved=True,
                document=document,
                verifier_report=report,
                rendered_prompt=rendered,
            )

        report = self.verifier.verify(document)
        repairs_used = 0
        while not report.approved and repairs_used < self.max_repairs:
            blocking = [issue for issue in report.issues if issue.blocking]
            if not blocking:
                break
            for issue in blocking:
                repaired = self.repairer.repair(document, issue)
                document = repaired
            document = self._canonicalize_spec_document(document)
            repairs_used += 1
            report = self.verifier.verify(document)

        document = document.model_copy(update={"verification": report})
        if report.approved:
            return PromptPipelineSpecResult(
                approved=True,
                document=document,
                verifier_report=report,
                rendered_prompt=render_prompt_spec(document),
            )

        return PromptPipelineSpecResult(
            approved=False,
            document=document,
            verifier_report=report,
            prompt_improvement_feedback=[issue.message for issue in report.issues if issue.blocking],
        )

    def _repair_semantic_validation_issues(self, document: PromptDocumentSpec) -> _SemanticValidationResult:
        issues = self._semantic_validation_issues(document)
        attempts_by_scope: dict[str, int] = {}

        while issues:
            retry_scope = issues[0].retry_scope
            attempts_used = attempts_by_scope.get(retry_scope, 0)
            retry_limit = _retry_limit(retry_scope) if self.max_semantic_repairs is None else self.max_semantic_repairs
            if attempts_used >= retry_limit:
                return _SemanticValidationResult(document=document, issues=issues)

            retry_request = _retry_request(document, issues)
            repaired = self.repairer.repair(document, retry_request)

            attempts_by_scope[retry_scope] = attempts_used + 1
            document = self._canonicalize_spec_document(repaired)
            issues = self._semantic_validation_issues(document)

        return _SemanticValidationResult(document=document, issues=[])

    def _semantic_validation_issues(self, document: PromptDocumentSpec) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        issues.extend(
            validators_to_validation_issues(
                "validate_relation_compatibility",
                validate_relation_compatibility(document.graph),
            )
        )
        issues.extend(
            validators_to_validation_issues(
                "validate_object_ownership",
                validate_object_ownership(document),
            )
        )
        issues.extend(
            validators_to_validation_issues(
                "validate_action_support",
                validate_action_support(document),
            )
        )
        issues.extend(
            validators_to_validation_issues(
                "validate_evidence_and_placeholders",
                validate_evidence_and_placeholders(document),
            )
        )
        issues.extend(
            validators_to_validation_issues(
                "validate_cross_lane_coherence",
                validate_cross_lane_coherence(document),
            )
        )
        return issues

    def _canonicalize_spec_document(self, document: PromptDocumentSpec) -> PromptDocumentSpec:
        canonical_metadata: dict[str, CanonicalEnum] = dict(document.canonical_metadata)

        relations = []
        for relation in document.graph.relations:
            canonical = self.canonicalizer.canonicalize(
                field_path=f"relation.{relation.id}",
                raw_value=relation.relation_raw,
            )
            canonical_metadata[f"relation.{relation.id}"] = canonical
            relations.append(
                relation.model_copy(
                    update={"relation_match": _relation_match(canonical, relation.relation_raw)}
                )
            )

        for descriptor in document.object_lane.objects:
            for field_name in ("description", "material", "color", "finish", "condition", "pattern"):
                value = getattr(descriptor, field_name)
                if value is None:
                    continue
                field_path = f"object.{field_name}.{descriptor.target_id}"
                canonical_metadata[field_path] = self.canonicalizer.canonicalize(
                    field_path=field_path,
                    raw_value=str(value),
                )

        actions = []
        for action in document.action_lane.actions:
            field_path = f"action.{action.actor_id}"
            canonical = self.canonicalizer.canonicalize(field_path=field_path, raw_value=action.movement_raw)
            canonical_metadata[field_path] = canonical
            actions.append(action.model_copy(update={"movement_match": _movement_match(canonical)}))

        graph = document.graph.model_copy(update={"relations": relations})
        action_lane = document.action_lane.model_copy(update={"actions": actions})
        return document.model_copy(
            update={
                "graph": graph,
                "action_lane": action_lane,
                "canonical_metadata": canonical_metadata,
            }
        )
