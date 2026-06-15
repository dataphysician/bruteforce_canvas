from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from bruteforce_canvas.shared import (
    CanonicalStatus,
    Confidence,
    DocId,
    ElementId,
    RelationId,
    RunId,
    ShortText,
    StrictModel,
    TargetManifestId,
)


class EvidenceCategory(StrEnum):
    EXPLICIT = "explicit"
    ENTAILED = "entailed"
    UNRESOLVED = "unresolved"
    BLOCKED = "blocked"


class Evidence(StrictModel):
    text: ShortText
    category: str


class CanonicalEnum(StrictModel):
    raw_value: ShortText
    enum_value: str | None = None
    status: str
    confidence: Confidence
    reason: ShortText


class Element(StrictModel):
    element_id: ElementId
    label: ShortText
    entity_type: Literal["person", "object", "location", "environment", "light_source", "abstract"]
    importance: Literal["primary", "foreground", "supporting", "background", "context"]
    evidence: Evidence


class Relation(StrictModel):
    relation_id: RelationId
    source_id: ElementId
    target_id: ElementId
    relation_raw: ShortText
    canonical: CanonicalEnum | None = None
    evidence: Evidence


class Graph(StrictModel):
    elements: list[Element] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "Graph":
        element_ids = [element.element_id for element in self.elements]
        if len(set(element_ids)) != len(element_ids):
            raise ValueError("element_id values must be unique")
        relation_ids = [relation.relation_id for relation in self.relations]
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("relation_id values must be unique")
        element_id_set = set(element_ids)
        for relation in self.relations:
            if relation.source_id not in element_id_set:
                raise ValueError(f"source_id {relation.source_id} does not resolve to an element")
            if relation.target_id not in element_id_set:
                raise ValueError(f"target_id {relation.target_id} does not resolve to an element")
        return self


class ObjectDescriptor(StrictModel):
    element_id: ElementId
    field_name: Literal["color", "material", "finish", "condition", "texture", "style"]
    raw_value: ShortText
    canonical: CanonicalEnum | None = None


class ActionDescriptor(StrictModel):
    actor_id: ElementId
    action_raw: ShortText
    target_id: ElementId | None = None
    support_state: Literal["supported", "inferred", "unresolved", "indeterminate"] = "supported"
    canonical: CanonicalEnum | None = None


class CinematographyLane(StrictModel):
    shot_size_raw: str | None = None
    camera_angle_raw: str | None = None
    lens_raw: str | None = None
    focus_raw: str | None = None
    lighting_raw: str | None = None
    color_treatment_raw: str | None = None
    composition_raw: str | None = None
    style_raw: str | None = None


class Constraint(StrictModel):
    constraint_id: str
    value_raw: ShortText
    negative: bool = True
    evidence: Evidence | None = None


class VerificationIssue(StrictModel):
    issue_type: ShortText
    repair_scope: ShortText
    blocking: bool
    message: ShortText


class VerificationReport(StrictModel):
    approved: bool
    issues: list[VerificationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def approved_without_blocking_issues(self) -> "VerificationReport":
        if self.approved and any(issue.blocking for issue in self.issues):
            raise ValueError("approved verification cannot contain blocking issues")
        return self


class PromptDocument(StrictModel):
    prompt_document_id: DocId
    prompt_document_version: str = "1"
    raw_user_prompt: str = Field(min_length=1, max_length=2000)
    seed_prompt: str = Field(min_length=1, max_length=500)
    graph: Graph
    objects: list[ObjectDescriptor] = Field(default_factory=list)
    actions: list[ActionDescriptor] = Field(default_factory=list)
    cinematography: CinematographyLane = Field(default_factory=CinematographyLane)
    constraints: list[Constraint] = Field(default_factory=list)
    canonical_metadata: dict[str, CanonicalEnum] = Field(default_factory=dict)
    verification: VerificationReport

    @model_validator(mode="after")
    def validate_lane_references(self) -> "PromptDocument":
        element_ids = {element.element_id for element in self.graph.elements}
        for descriptor in self.objects:
            if descriptor.element_id not in element_ids:
                raise ValueError(f"object descriptor references missing element_id {descriptor.element_id}")
        for action in self.actions:
            if action.actor_id not in element_ids:
                raise ValueError(f"action references missing actor_id {action.actor_id}")
            if action.target_id is not None and action.target_id not in element_ids:
                raise ValueError(f"action references missing target_id {action.target_id}")
        return self


class RenderedPrompt(StrictModel):
    run_id: RunId | None = None
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId | None = None
    coordinate_id: str | None = None
    rendered_prompt: str
    rendering_trace: list[str]

    @model_validator(mode="after")
    def begins_with_generate(self) -> "RenderedPrompt":
        if not self.rendered_prompt.startswith("Generate "):
            raise ValueError("rendered_prompt must begin with 'Generate '")
        return self


class EvaluationTarget(StrictModel):
    target_id: str
    target_kind: Literal["element", "relation", "object_attribute", "cinematography", "constraint"]
    label: str | None = None
    value_raw: str | None = None
    enum_value: str | None = None
    source_id: str | None = None
    target_id_ref: str | None = None
    relation_raw: str | None = None
    priority: Literal["locked_required", "locked_context", "important", "sampled", "optional", "negative_guard", "diagnostic"]
    lhs_policy: Literal["fixed", "sampleable", "sampleable_if_missing", "blocked"]
    evaluation_policy: Literal["must_match", "should_match", "nice_to_have", "must_not_appear", "measure_only"]
    evidence: str | None = None


class EvaluationTargetManifest(StrictModel):
    manifest_id: TargetManifestId
    run_id: RunId
    prompt_document_id: DocId
    coordinate_id: str | None = None
    rendered_prompt: str
    targets: list[EvaluationTarget]
    negative_targets: list[EvaluationTarget] = Field(default_factory=list)


def _descriptor_map(document: PromptDocument) -> dict[str, list[ObjectDescriptor]]:
    result: dict[str, list[ObjectDescriptor]] = {}
    for descriptor in document.objects:
        result.setdefault(descriptor.element_id, []).append(descriptor)
    return result


def render_prompt(document: PromptDocument) -> RenderedPrompt:
    if not document.verification.approved:
        raise ValueError("cannot render an unapproved PromptDocument")

    descriptors = _descriptor_map(document)
    element_by_id = {element.element_id: element for element in document.graph.elements}
    relation_targets = {relation.target_id for relation in document.graph.relations}
    subject_phrases: list[str] = []
    traces: list[str] = []

    for element in document.graph.elements:
        if element.element_id in relation_targets and element.importance != "primary":
            continue
        parts = [descriptor.raw_value for descriptor in descriptors.get(element.element_id, [])]
        parts.append(element.label)
        phrase = " ".join(parts)
        for relation in [item for item in document.graph.relations if item.source_id == element.element_id]:
            target = element_by_id[relation.target_id]
            target_parts = [descriptor.raw_value for descriptor in descriptors.get(target.element_id, [])]
            target_parts.append(target.label)
            phrase = f"{phrase} {relation.relation_raw} {' '.join(target_parts)}"
            traces.append(f"relation:{relation.relation_id}:{relation.relation_raw}")
        subject_phrases.append(phrase)

    for action in document.actions:
        if action.support_state in {"supported", "inferred"}:
            subject_phrases.append(action.action_raw)
            traces.append(f"action:{action.actor_id}:{action.support_state}")

    cinematography_parts = [
        document.cinematography.shot_size_raw,
        document.cinematography.camera_angle_raw,
        document.cinematography.lens_raw,
        document.cinematography.focus_raw,
        document.cinematography.lighting_raw,
        document.cinematography.color_treatment_raw,
        document.cinematography.composition_raw,
        document.cinematography.style_raw,
    ]
    subject_phrases.extend(part for part in cinematography_parts if part)

    positive_constraints = [constraint.value_raw for constraint in document.constraints if not constraint.negative]
    negative_constraints = [constraint.value_raw for constraint in document.constraints if constraint.negative]
    subject_phrases.extend(positive_constraints)

    rendered = "Generate " + ", ".join(part for part in subject_phrases if part)
    if negative_constraints:
        rendered += ". Negative prompt: " + ", ".join(negative_constraints)
    return RenderedPrompt(
        prompt_document_id=document.prompt_document_id,
        rendered_prompt=rendered,
        rendering_trace=traces,
    )


def target_manifest_from_prompt(run_id: str, rendered: RenderedPrompt, document: PromptDocument) -> EvaluationTargetManifest:
    targets: list[EvaluationTarget] = []
    negative_targets: list[EvaluationTarget] = []
    for element in document.graph.elements:
        targets.append(
            EvaluationTarget(
                target_id=element.element_id,
                target_kind="element",
                label=element.label,
                priority="locked_required" if element.importance in {"primary", "foreground", "supporting"} else "locked_context",
                lhs_policy="fixed",
                evaluation_policy="must_match" if element.importance in {"primary", "foreground", "supporting"} else "should_match",
                evidence=element.evidence.text,
            )
        )
    for relation in document.graph.relations:
        targets.append(
            EvaluationTarget(
                target_id=relation.relation_id,
                target_kind="relation",
                source_id=relation.source_id,
                target_id_ref=relation.target_id,
                relation_raw=relation.relation_raw,
                enum_value=relation.canonical.enum_value if relation.canonical else None,
                priority="locked_required",
                lhs_policy="fixed",
                evaluation_policy="must_match",
                evidence=relation.evidence.text,
            )
        )
    for descriptor in document.objects:
        targets.append(
            EvaluationTarget(
                target_id=f"{descriptor.element_id}.{descriptor.field_name}",
                target_kind="object_attribute",
                value_raw=descriptor.raw_value,
                enum_value=descriptor.canonical.enum_value if descriptor.canonical else None,
                priority="locked_required",
                lhs_policy="fixed",
                evaluation_policy="must_match",
            )
        )
    for constraint in document.constraints:
        target = EvaluationTarget(
            target_id=constraint.constraint_id,
            target_kind="constraint",
            value_raw=constraint.value_raw,
            priority="negative_guard" if constraint.negative else "important",
            lhs_policy="fixed",
            evaluation_policy="must_not_appear" if constraint.negative else "should_match",
            evidence=constraint.evidence.text if constraint.evidence else None,
        )
        if constraint.negative:
            negative_targets.append(target)
        else:
            targets.append(target)

    return EvaluationTargetManifest(
        manifest_id="eval_manifest_001",
        run_id=run_id,
        prompt_document_id=document.prompt_document_id,
        coordinate_id=rendered.coordinate_id,
        rendered_prompt=rendered.rendered_prompt,
        targets=targets,
        negative_targets=negative_targets,
    )
