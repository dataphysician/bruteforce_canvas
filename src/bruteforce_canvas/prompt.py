from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from bruteforce_canvas.prompt_enums import EntityType, RelationType
from bruteforce_canvas.shared import (
    CanonicalStatus,
    Confidence,
    DocId,
    ElementId,
    RelationId,
    ResidualDescription,
    RunId,
    SeedPrompt,
    ShortText,
    StrictModel,
    TargetManifestId,
)
from bruteforce_canvas.validation import RetryRequest, ValidationIssue, ValidationSeverity


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
    """# DEPRECATED: use new spec-compliant models from B3."""

    element_id: ElementId
    label: ShortText
    entity_type: Literal["person", "object", "location", "environment", "light_source", "abstract"]
    importance: Literal["primary", "foreground", "supporting", "background", "context"]
    evidence: Evidence


class Relation(StrictModel):
    """# DEPRECATED: use new spec-compliant models from B3."""

    relation_id: RelationId
    source_id: ElementId
    target_id: ElementId
    relation_raw: ShortText
    canonical: CanonicalEnum | None = None
    evidence: Evidence


class Graph(StrictModel):
    """# DEPRECATED: use new spec-compliant models from B3."""

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
    """# DEPRECATED: use new spec-compliant models from B3."""

    element_id: ElementId
    field_name: Literal["color", "material", "finish", "condition", "texture", "style"]
    raw_value: ShortText
    canonical: CanonicalEnum | None = None


class ActionDescriptor(StrictModel):
    """# DEPRECATED: use new spec-compliant models from B3."""

    actor_id: ElementId
    action_raw: ShortText
    target_id: ElementId | None = None
    support_state: Literal["supported", "inferred", "unresolved", "indeterminate"] = "supported"
    canonical: CanonicalEnum | None = None


class CinematographyLane(StrictModel):
    """# DEPRECATED: use new spec-compliant models from B3."""

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
    """# DEPRECATED: use new spec-compliant models from B3."""

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


class EvidenceSpan(StrictModel):
    text: ShortText
    category: EvidenceCategory = EvidenceCategory.EXPLICIT
    reason: ResidualDescription | None = None


class RelationEnumMatch(StrictModel):
    raw: ShortText = Field(
        description="Raw linkage phrase from the prompt, such as resting on, leaning against, or draped over.",
    )
    enum_value: RelationType | None = Field(
        default=None,
        description=(
            "Optional locked relation enum. Null means preserve raw linkage and do not force a bad enum."
        ),
    )
    status: CanonicalStatus
    confidence: ShortText = Field(description="clear, weak, unmatched, or rejected")
    reason: ResidualDescription | None = None


class ProposedRelationEnum(StrictModel):
    code: ShortText = Field(
        description="UPPER_SNAKE_CASE candidate for the enum registry candidate queue.",
    )
    label: ShortText
    aliases: list[ShortText] = Field(default_factory=list, max_length=12)
    description: ResidualDescription
    source_type_examples: list[EntityType] = Field(default_factory=list, max_length=8)
    target_type_examples: list[EntityType] = Field(default_factory=list, max_length=8)


class InferredGraphSupport(StrictModel):
    reason: ShortText
    implied_element_label: ShortText | None = None
    implied_relation_type: RelationType | None = None
    source_action: ShortText
    evidence: EvidenceSpan


class PromptImprovementHint(StrictModel):
    issue: ShortText
    suggested_rewrites: list[ShortText] = Field(default_factory=list, max_length=4)
    safe_downgrade: ShortText | None = None


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


from bruteforce_canvas.prompt_models import (  # noqa: E402
    ActionLane,
    ConstraintLane,
    ObjectLane,
    PromptBundle,
    PromptDocumentSpec,
    SceneGraphDraft,
    CinematographyLane as SpecCinematographyLane,
)


def _clean_spec_text(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _display_spec_value(value: object | None) -> str:
    return _clean_spec_text(value).replace("_", " ")


def _append_unique(parts: list[str], value: object | None, *, display: bool = False) -> None:
    text = _display_spec_value(value) if display else _clean_spec_text(value)
    if text and text not in parts:
        parts.append(text)


def _spec_object_parts(descriptor: object) -> list[str]:
    parts: list[str] = []
    enum_fields = {"condition", "finish", "pattern"}
    for field_name in ("condition", "color", "pattern", "material", "finish", "description"):
        value = getattr(descriptor, field_name, None)
        if value:
            parts.append(_display_spec_value(value) if field_name in enum_fields else _clean_spec_text(value))
    return parts


def object_phrase(element: object) -> str:
    """Return a graph element's raw label for rendering."""
    return _clean_spec_text(getattr(element, "label", None))


def _spec_object_descriptor_map(document: PromptDocumentSpec) -> dict[str, list[object]]:
    object_descriptors: dict[str, list[object]] = {}
    for descriptor in document.object_lane.objects:
        object_descriptors.setdefault(descriptor.target_id, []).append(descriptor)
    return object_descriptors


def _spec_element_phrase(element: object, descriptors: list[object]) -> str:
    parts: list[str] = []
    for descriptor in descriptors:
        parts.extend(_spec_object_parts(descriptor))
    _append_unique(parts, object_phrase(element))
    return " ".join(parts)


def _spec_scene_phrases(document: PromptDocumentSpec) -> list[str]:
    object_descriptors = _spec_object_descriptor_map(document)

    element_by_id = {element.id: element for element in document.graph.elements}
    relation_targets = {relation.target_id for relation in document.graph.relations}
    phrases: list[str] = []

    _append_unique(phrases, document.graph.seed_prompt)

    for element in document.graph.elements:
        if element.id in relation_targets and element.role != "primary_subject":
            continue
        phrase = _spec_element_phrase(element, object_descriptors.get(element.id, []))
        for relation in [item for item in document.graph.relations if item.source_id == element.id]:
            target = element_by_id.get(relation.target_id)
            target_phrase = relation.target_id
            if target is not None:
                target_phrase = _spec_element_phrase(target, object_descriptors.get(target.id, []))
            relation_raw = _clean_spec_text(relation.relation_raw)
            if relation_raw and target_phrase:
                phrase = f"{phrase} {relation_raw} {target_phrase}".strip()
        _append_unique(phrases, phrase)

    for action in document.action_lane.actions:
        if action.support_status in {"supported", "inferred"}:
            _append_unique(phrases, action.movement_raw)

    return phrases


def scene_description(document: PromptDocumentSpec) -> str:
    """Render the seed prompt, graph elements, relations, and actions."""
    return ", ".join(_spec_scene_phrases(document))


def composition_phrase(document: PromptDocumentSpec) -> str:
    """Render camera placement, focus, framing, and setting details."""
    lane = document.cinematography_lane
    parts: list[str] = []
    for field_name in (
        "shot_size",
        "camera_angle",
        "camera_motion",
        "focus_behavior",
        "framing",
        "setting_description",
    ):
        value = getattr(lane, field_name, None)
        if field_name in {"camera_motion", "focus_behavior", "setting_description"}:
            _append_unique(parts, value)
        else:
            _append_unique(parts, value, display=True)
    return ", ".join(parts)


def style_prompt(document: PromptDocumentSpec) -> str:
    """Render visual style, optics, lighting, and color treatment."""
    lane = document.cinematography_lane
    parts: list[str] = []
    for field_name in ("optic_character", "lighting_mood", "color_treatment"):
        _append_unique(parts, getattr(lane, field_name, None), display=True)
    return ", ".join(parts)


def negative_prompt(document: PromptDocumentSpec) -> str:
    """Render guardrails and raw negative phrases as a comma list."""
    parts: list[str] = []
    for guardrail in document.constraint_lane.guardrails:
        _append_unique(parts, guardrail, display=True)
    for phrase in document.constraint_lane.negative_phrases:
        _append_unique(parts, phrase)
    return ", ".join(parts)


def compile_prompt(document: PromptDocumentSpec) -> str:
    """Compile a spec document into the final positive/negative prompt string."""
    positive_parts = [
        scene_description(document),
        composition_phrase(document),
        style_prompt(document),
    ]
    positive = ", ".join(part for part in positive_parts if part)
    rendered = "Generate " + positive
    negative = negative_prompt(document)
    if negative:
        rendered += ". Negative prompt: " + negative
    return rendered


def _spec_rendering_trace(document: PromptDocumentSpec) -> list[str]:
    traces: list[str] = []
    for relation in document.graph.relations:
        traces.append(f"relation:{relation.id}:{relation.relation_raw}")
    for action in document.action_lane.actions:
        if action.support_status in {"supported", "inferred"}:
            traces.append(f"action:{action.actor_id}:{action.support_status}")
    return traces


def render_prompt_spec(document: PromptDocumentSpec) -> RenderedPrompt:
    """Render an approved spec prompt document with relation/action trace."""
    if not document.verification.approved:
        raise ValueError("cannot render an unapproved PromptDocumentSpec")

    return RenderedPrompt(
        run_id=None,
        prompt_document_id=document.prompt_document_id,
        rendered_prompt=compile_prompt(document),
        rendering_trace=_spec_rendering_trace(document),
    )


EvaluationPriority = Literal[
    "locked_required",
    "locked_context",
    "important",
    "sampled",
    "optional",
    "negative_guard",
    "diagnostic",
]
EvaluationPolicy = Literal[
    "must_match",
    "should_match",
    "nice_to_have",
    "must_not_appear",
    "measure_only",
]


def _spec_priority(importance: str, is_required_role: bool) -> EvaluationPriority:
    if importance == "required" or is_required_role:
        return "locked_required"
    if importance == "ambient":
        return "locked_context"
    if importance == "unresolved":
        return "diagnostic"
    return "optional"


def _spec_evaluation_policy(importance: str) -> EvaluationPolicy:
    if importance == "required":
        return "must_match"
    if importance == "unresolved":
        return "measure_only"
    if importance == "optional":
        return "nice_to_have"
    return "should_match"


def target_manifest_from_prompt_spec(document: PromptDocumentSpec) -> EvaluationTargetManifest:
    rendered = render_prompt_spec(document)
    targets: list[EvaluationTarget] = []
    negative_targets: list[EvaluationTarget] = []

    for element in document.graph.elements:
        importance = str(element.importance)
        required_role = str(element.role) in {"primary_subject", "foreground", "supporting"}
        priority = _spec_priority(importance, required_role)
        targets.append(
            EvaluationTarget(
                target_id=element.id,
                target_kind="element",
                label=element.label,
                priority=priority,
                lhs_policy="sampleable_if_missing" if importance == "unresolved" else "fixed",
                evaluation_policy=_spec_evaluation_policy(importance),
                evidence=element.evidence.text if element.evidence else None,
            )
        )

    for relation in document.graph.relations:
        enum_value = relation.relation_match.enum_value if relation.relation_match else None
        importance = str(relation.importance)
        targets.append(
            EvaluationTarget(
                target_id=relation.id,
                target_kind="relation",
                source_id=relation.source_id,
                target_id_ref=relation.target_id,
                relation_raw=relation.relation_raw,
                enum_value=enum_value,
                priority=_spec_priority(importance, True),
                lhs_policy="fixed",
                evaluation_policy=_spec_evaluation_policy(importance),
                evidence=relation.evidence.text if relation.evidence else None,
            )
        )

    for descriptor in document.object_lane.objects:
        for field_name in ("description", "material", "color", "finish", "condition", "pattern"):
            value = getattr(descriptor, field_name)
            if value is None:
                continue
            is_enum_field = field_name in {"finish", "condition", "pattern"}
            targets.append(
                EvaluationTarget(
                    target_id=f"{descriptor.target_id}.{field_name}",
                    target_kind="object_attribute",
                    value_raw=_display_spec_value(str(value)),
                    enum_value=str(value) if is_enum_field else None,
                    priority="locked_required",
                    lhs_policy="fixed",
                    evaluation_policy="must_match",
                )
            )

    for field_name in (
        "shot_size",
        "camera_angle",
        "optic_character",
        "camera_motion",
        "focus_behavior",
        "lighting_mood",
        "color_treatment",
        "framing",
        "setting_description",
    ):
        value = getattr(document.cinematography_lane, field_name)
        if value is None:
            continue
        is_enum_field = field_name in {
            "shot_size",
            "camera_angle",
            "optic_character",
            "lighting_mood",
            "color_treatment",
            "framing",
        }
        targets.append(
            EvaluationTarget(
                target_id=f"cinematography.{field_name}",
                target_kind="cinematography",
                value_raw=_display_spec_value(str(value)),
                enum_value=str(value) if is_enum_field else None,
                priority="important",
                lhs_policy="fixed",
                evaluation_policy="should_match",
            )
        )

    for index, guardrail in enumerate(document.constraint_lane.guardrails, start=1):
        negative_targets.append(
            EvaluationTarget(
                target_id=f"constraint.guardrail.{index}",
                target_kind="constraint",
                value_raw=_display_spec_value(str(guardrail)),
                enum_value=str(guardrail),
                priority="negative_guard",
                lhs_policy="fixed",
                evaluation_policy="must_not_appear",
            )
        )
    for index, phrase in enumerate(document.constraint_lane.negative_phrases, start=1):
        negative_targets.append(
            EvaluationTarget(
                target_id=f"constraint.negative_phrase.{index}",
                target_kind="constraint",
                value_raw=str(phrase),
                priority="negative_guard",
                lhs_policy="fixed",
                evaluation_policy="must_not_appear",
            )
        )

    return EvaluationTargetManifest(
        manifest_id="eval_manifest_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        coordinate_id=rendered.coordinate_id,
        rendered_prompt=rendered.rendered_prompt,
        targets=targets,
        negative_targets=negative_targets,
    )
