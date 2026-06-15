from __future__ import annotations

from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import Field

from bruteforce_canvas.prompt import (
    CanonicalEnum,
    EvidenceCategory,
    EvidenceSpan,
    PromptDocument,
    RenderedPrompt,
    RelationEnumMatch,
    VerificationIssue,
    VerificationReport,
    render_prompt,
    render_prompt_spec,
)
from bruteforce_canvas.prompt_enums import (
    ActionSupportStatus,
    CameraAngle,
    ColorTreatment,
    Condition,
    ElementRole,
    EntityType,
    EnumMatchConfidence,
    Finish,
    Framing,
    Guardrail,
    Importance,
    LightingMood,
    OpticCharacter,
    Pattern,
    RelationType,
    ShotSize,
)
from bruteforce_canvas.prompt_models import (
    ActionDescriptor as SpecActionDescriptor,
    ActionLane,
    CinematographyLane as SpecCinematographyLane,
    ConstraintLane,
    Element as SpecElement,
    EnumMatch,
    ObjectDescriptor as SpecObjectDescriptor,
    ObjectLane,
    PromptDocumentSpec,
    RelationDescriptor,
    SceneGraphDraft,
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
    def extract(self, raw_prompt: str) -> PromptDocument | PromptDocumentSpec:
        """Return a draft prompt document from one raw user prompt."""
        ...


class CanonicalizerAdapter(Protocol):
    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        """Canonicalize one field-scoped raw value without rewriting the raw phrase."""
        ...


class VerificationAdapter(Protocol):
    def verify(self, document: PromptDocument | PromptDocumentSpec) -> VerificationReport:
        """Verify graph linkage, lane ownership, enum fit, and renderability."""
        ...


class RepairAdapter(Protocol):
    def repair(
        self,
        document: PromptDocument | PromptDocumentSpec,
        issue: RetryRequest | VerificationIssue,
    ) -> PromptDocument | PromptDocumentSpec:
        """Repair only the slice identified by the retry request or verifier issue."""
        ...


class PromptPipelineResult(StrictModel):
    approved: bool
    document: PromptDocument
    verifier_report: VerificationReport
    rendered_prompt: RenderedPrompt | None = None
    prompt_improvement_feedback: list[str] = Field(default_factory=list)


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


def _evidence_span(category: str, text: str) -> EvidenceSpan:
    evidence_category = _coerce_str_enum(EvidenceCategory, category)
    return EvidenceSpan(
        text=text,
        category=evidence_category if evidence_category is not None else EvidenceCategory.EXPLICIT,
    )


def _legacy_entity_type(raw_value: str) -> EntityType:
    mapped = {
        "abstract": EntityType.ABSTRACT_VISUAL,
        "object": EntityType.PRODUCT,
    }.get(raw_value)
    if mapped is not None:
        return mapped
    coerced = _coerce_str_enum(EntityType, raw_value)
    return coerced if coerced is not None else EntityType.UNKNOWN_SLOT


def _legacy_role(raw_value: str) -> ElementRole:
    if raw_value == "primary":
        return ElementRole.PRIMARY_SUBJECT
    coerced = _coerce_str_enum(ElementRole, raw_value)
    return coerced if coerced is not None else ElementRole.CONTEXT


def _legacy_importance(raw_value: str) -> Importance:
    if raw_value in {"primary", "foreground", "supporting"}:
        return Importance.REQUIRED
    if raw_value in {"background", "context"}:
        return Importance.AMBIENT
    return Importance.UNRESOLVED


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


def _short_residual(parts: list[str]) -> str | None:
    text = "; ".join(part for part in parts if part)
    return text[:180] if text else None


def _spec_objects(document: PromptDocument) -> list[SpecObjectDescriptor]:
    colors: dict[str, str] = {}
    materials: dict[str, str] = {}
    finishes: dict[str, Finish] = {}
    conditions: dict[str, Condition] = {}
    patterns: dict[str, Pattern] = {}
    residuals_by_element: dict[str, list[str]] = {}

    for descriptor in document.objects:
        residuals = residuals_by_element.setdefault(descriptor.element_id, [])
        if descriptor.field_name == "color":
            colors[descriptor.element_id] = descriptor.raw_value
            continue
        if descriptor.field_name == "material":
            materials[descriptor.element_id] = descriptor.raw_value
            continue
        if descriptor.field_name == "finish":
            finish = _coerce_str_enum(Finish, descriptor.raw_value)
            if finish is not None:
                finishes[descriptor.element_id] = finish
                continue
        if descriptor.field_name == "condition":
            condition = _coerce_str_enum(Condition, descriptor.raw_value)
            if condition is not None:
                conditions[descriptor.element_id] = condition
                continue
        if descriptor.field_name == "pattern":
            pattern = _coerce_str_enum(Pattern, descriptor.raw_value)
            if pattern is not None:
                patterns[descriptor.element_id] = pattern
                continue
        residuals.append(f"{descriptor.field_name}: {descriptor.raw_value}")

    element_ids = (
        set(colors)
        | set(materials)
        | set(finishes)
        | set(conditions)
        | set(patterns)
        | set(residuals_by_element)
    )
    return [
        SpecObjectDescriptor(
            target_id=element_id,
            description=_short_residual(residuals_by_element.get(element_id, [])),
            material=materials.get(element_id),
            color=colors.get(element_id),
            finish=finishes.get(element_id),
            condition=conditions.get(element_id),
            pattern=patterns.get(element_id),
        )
        for element_id in sorted(element_ids)
    ]


def _spec_cinematography(document: PromptDocument) -> SpecCinematographyLane:
    lane = document.cinematography
    shot_size = _coerce_str_enum(ShotSize, lane.shot_size_raw)
    camera_angle = _coerce_str_enum(CameraAngle, lane.camera_angle_raw)
    optic_character = _coerce_str_enum(OpticCharacter, lane.lens_raw)
    lighting_mood = _coerce_str_enum(LightingMood, lane.lighting_raw)
    color_treatment = _coerce_str_enum(ColorTreatment, lane.color_treatment_raw)
    framing = _coerce_str_enum(Framing, lane.composition_raw)
    residuals = []
    for label, raw_value, coerced in (
        ("shot size", lane.shot_size_raw, shot_size),
        ("camera angle", lane.camera_angle_raw, camera_angle),
        ("lens", lane.lens_raw, optic_character),
        ("lighting", lane.lighting_raw, lighting_mood),
        ("color treatment", lane.color_treatment_raw, color_treatment),
        ("composition", lane.composition_raw, framing),
    ):
        if raw_value and coerced is None:
            residuals.append(f"{label}: {raw_value}")
    if lane.style_raw:
        residuals.append(f"style: {lane.style_raw}")

    return SpecCinematographyLane(
        shot_size=shot_size,
        camera_angle=camera_angle,
        optic_character=optic_character,
        focus_behavior=lane.focus_raw,
        lighting_mood=lighting_mood,
        color_treatment=color_treatment,
        framing=framing,
        setting_description=_short_residual(residuals),
    )


def _spec_constraints(document: PromptDocument) -> ConstraintLane:
    guardrails: list[Guardrail] = []
    negative_phrases: list[str] = []
    for constraint in document.constraints:
        if not constraint.negative:
            continue
        guardrail = _coerce_str_enum(Guardrail, constraint.value_raw)
        if guardrail is None:
            negative_phrases.append(constraint.value_raw)
        else:
            guardrails.append(guardrail)
    return ConstraintLane(guardrails=guardrails, negative_phrases=negative_phrases)


def legacy_prompt_document_to_spec(document: PromptDocument) -> PromptDocumentSpec:
    graph = SceneGraphDraft(
        seed_prompt=document.seed_prompt,
        elements=[
            SpecElement(
                id=element.element_id,
                entity_type=_legacy_entity_type(element.entity_type),
                label=element.label,
                role=_legacy_role(element.importance),
                importance=_legacy_importance(element.importance),
                evidence=_evidence_span(element.evidence.category, element.evidence.text),
            )
            for element in document.graph.elements
        ],
        relations=[
            RelationDescriptor(
                id=relation.relation_id,
                source_id=relation.source_id,
                target_id=relation.target_id,
                relation_raw=relation.relation_raw,
                relation_match=_relation_match(relation.canonical, relation.relation_raw),
                importance=Importance.REQUIRED,
                evidence=_evidence_span(relation.evidence.category, relation.evidence.text),
            )
            for relation in document.graph.relations
        ],
    )

    return PromptDocumentSpec(
        graph=graph,
        object_lane=ObjectLane(objects=_spec_objects(document)),
        action_lane=ActionLane(
            actions=[
                SpecActionDescriptor(
                    actor_id=action.actor_id,
                    movement_raw=action.action_raw,
                    movement_match=_movement_match(action.canonical),
                    target_id=action.target_id,
                    support_status=_coerce_str_enum(ActionSupportStatus, action.support_state)
                    or ActionSupportStatus.INDETERMINATE,
                )
                for action in document.actions
            ]
        ),
        cinematography_lane=_spec_cinematography(document),
        constraint_lane=_spec_constraints(document),
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
    ) -> None:
        self.extractor = extractor
        self.canonicalizer = canonicalizer
        self.verifier = verifier
        self.repairer = repairer
        self.max_repairs = max_repairs

    def run(self, raw_prompt: str) -> PromptPipelineResult:
        extracted = self.extractor.extract(raw_prompt)
        if isinstance(extracted, PromptDocumentSpec):
            raise TypeError("PromptPipeline.run() requires a legacy PromptDocument; use run_spec() for PromptDocumentSpec")
        return self._run_legacy_document(extracted)

    def run_spec(self, raw_prompt: str) -> PromptPipelineSpecResult:
        extracted = self.extractor.extract(raw_prompt)
        if isinstance(extracted, PromptDocumentSpec):
            return self._run_spec_document(extracted)

        result = self._run_legacy_document(extracted)
        document = legacy_prompt_document_to_spec(result.document)
        return PromptPipelineSpecResult(
            approved=result.approved,
            document=document,
            verifier_report=result.verifier_report,
            rendered_prompt=render_prompt_spec(document) if result.rendered_prompt is not None else None,
            prompt_improvement_feedback=result.prompt_improvement_feedback,
        )

    def _run_legacy_document(self, document: PromptDocument) -> PromptPipelineResult:
        document = self._canonicalize_document(document)
        report = self.verifier.verify(document)
        repairs_used = 0
        while not report.approved and repairs_used < self.max_repairs:
            blocking = [issue for issue in report.issues if issue.blocking]
            if not blocking:
                break
            for issue in blocking:
                repaired = self.repairer.repair(document, issue)
                if isinstance(repaired, PromptDocumentSpec):
                    raise TypeError("legacy pipeline repairer returned PromptDocumentSpec")
                document = repaired
            document = self._canonicalize_document(document)
            repairs_used += 1
            report = self.verifier.verify(document)

        document = document.model_copy(update={"verification": report})
        if report.approved:
            return PromptPipelineResult(
                approved=True,
                document=document,
                verifier_report=report,
                rendered_prompt=render_prompt(document),
            )

        return PromptPipelineResult(
            approved=False,
            document=document,
            verifier_report=report,
            prompt_improvement_feedback=[issue.message for issue in report.issues if issue.blocking],
        )

    def _run_spec_document(self, document: PromptDocumentSpec) -> PromptPipelineSpecResult:
        document = self._canonicalize_spec_document(document)
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

        report = self.verifier.verify(document)
        repairs_used = 0
        while not report.approved and repairs_used < self.max_repairs:
            blocking = [issue for issue in report.issues if issue.blocking]
            if not blocking:
                break
            for issue in blocking:
                repaired = self.repairer.repair(document, issue)
                if not isinstance(repaired, PromptDocumentSpec):
                    raise TypeError("spec pipeline repairer returned legacy PromptDocument")
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
            if attempts_used >= _retry_limit(retry_scope):
                return _SemanticValidationResult(document=document, issues=issues)

            retry_request = _retry_request(document, issues)
            repaired = self.repairer.repair(document, retry_request)
            if not isinstance(repaired, PromptDocumentSpec):
                raise TypeError("spec pipeline repairer returned legacy PromptDocument")

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

    def _canonicalize_document(self, document: PromptDocument) -> PromptDocument:
        canonical_metadata: dict[str, CanonicalEnum] = dict(document.canonical_metadata)

        relations = []
        for relation in document.graph.relations:
            canonical = self.canonicalizer.canonicalize(
                field_path=f"relation.{relation.relation_id}",
                raw_value=relation.relation_raw,
            )
            canonical_metadata[f"relation.{relation.relation_id}"] = canonical
            relations.append(relation.model_copy(update={"canonical": canonical}))

        objects = []
        for descriptor in document.objects:
            field_path = f"object.{descriptor.field_name}.{descriptor.element_id}"
            canonical = self.canonicalizer.canonicalize(field_path=field_path, raw_value=descriptor.raw_value)
            canonical_metadata[field_path] = canonical
            objects.append(descriptor.model_copy(update={"canonical": canonical}))

        actions = []
        for action in document.actions:
            field_path = f"action.{action.actor_id}"
            canonical = self.canonicalizer.canonicalize(field_path=field_path, raw_value=action.action_raw)
            canonical_metadata[field_path] = canonical
            actions.append(action.model_copy(update={"canonical": canonical}))

        graph = document.graph.model_copy(update={"relations": relations})
        return document.model_copy(
            update={
                "graph": graph,
                "objects": objects,
                "actions": actions,
                "canonical_metadata": canonical_metadata,
            }
        )

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
