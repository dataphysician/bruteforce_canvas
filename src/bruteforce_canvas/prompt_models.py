"""Spec-compliant lane and container models for the prompt domain (Phase B3).

This module holds the seven new spec-compliant models defined in
``specs/01-bruteforce-canvas_DAG_prompt.md`` sections 4.1-4.5, plus the
small element/relation/descriptor models they depend on:

* ``Element`` (spec §356-365) — identity-shaped element with role + importance
* ``RelationDescriptor`` (spec §368-391) — typed relation with optional enum match
* ``SceneGraphDraft`` (spec §394-427) — graph container with structural validators
* ``ObjectDescriptor`` + ``ObjectLane`` (spec §458-475) — appearance container
* ``EnumMatch`` (spec §504-508) — tiny-LLM enum match
* ``ActionDescriptor`` + ``ActionLane`` (spec §532-552) — motion container
* ``CinematographyLane`` (spec §731-749) — typed cinematography container
* ``ConstraintLane`` (spec §764-766) — guardrails + negative phrases container
* ``PromptDocumentSpec`` (spec §769-790) — merged document with reference validators
* ``PromptBundle`` (spec §793-798) — final positive/negative + render trace bundle

The seven container names that must be importable from
``bruteforce_canvas.prompt`` (per the B3 task contract) are re-exported at
the bottom of ``prompt.py``:

    SceneGraphDraft, ObjectLane, ActionLane, CinematographyLane,
    ConstraintLane, PromptDocumentSpec, PromptBundle

The new ``Element`` / ``CinematographyLane`` / ``ObjectDescriptor`` /
``ActionDescriptor`` classes intentionally shadow the legacy versions in
``prompt.py`` once re-exported. The legacy classes remain in ``prompt.py``
for backward compatibility until B5 completes the migration; they are no
longer reachable via the bare module name.

Structural-only validators are implemented here. Semantic validators
(Phase C) are deliberately out of scope.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from bruteforce_canvas.prompt import (
    CanonicalEnum,
    EvidenceSpan,
    InferredGraphSupport,
    PromptImprovementHint,
    ProposedRelationEnum,
    RelationEnumMatch,
    VerificationReport,
)
from bruteforce_canvas.prompt_enums import (
    ActionSupportStatus,
    CameraAngle,
    ColorTreatment,
    Condition,
    ElementRole,
    EnumMatchConfidence,
    EntityType,
    Finish,
    Framing,
    Guardrail,
    Importance,
    LightingMood,
    MotionIntensity,
    OpticCharacter,
    Pattern,
    RelationType,
    ShotSize,
)
from bruteforce_canvas.shared import (
    CanonicalStatus,
    DocId,
    ElementId,
    RelationId,
    ResidualDescription,
    SeedPrompt,
    ShortText,
    StrictModel,
)


class Element(StrictModel):
    id: ElementId
    entity_type: EntityType
    label: ShortText
    role: ElementRole
    importance: Importance = Importance.REQUIRED
    evidence: EvidenceSpan | None = None


class RelationDescriptor(StrictModel):
    id: RelationId
    source_id: ElementId
    target_id: ElementId
    relation_raw: ShortText
    relation_match: RelationEnumMatch | None = None
    proposed_enum: ProposedRelationEnum | None = None
    importance: Importance = Importance.REQUIRED
    evidence: EvidenceSpan | None = None
    description: ResidualDescription | None = None


class SceneGraphDraft(StrictModel):
    seed_prompt: SeedPrompt
    elements: list[Element] = Field(min_length=1, max_length=24)
    relations: list[RelationDescriptor] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def validate_graph(self) -> "SceneGraphDraft":
        element_ids = [element.id for element in self.elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("element IDs must be unique")

        relation_ids = [relation.id for relation in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation IDs must be unique")

        known = set(element_ids)
        for relation in self.relations:
            if relation.source_id not in known:
                raise ValueError(
                    f"relation {relation.id} has unknown source_id {relation.source_id}"
                )
            if relation.target_id not in known:
                raise ValueError(
                    f"relation {relation.id} has unknown target_id {relation.target_id}"
                )
            if relation.source_id == relation.target_id:
                raise ValueError(f"relation {relation.id} cannot point to itself")
            if not relation.relation_raw.strip():
                raise ValueError(f"relation {relation.id} must preserve relation_raw")
            if (
                relation.relation_match is not None
                and relation.relation_match.enum_value is None
                and relation.relation_match.status == CanonicalStatus.MATCHED_ACTIVE
            ):
                raise ValueError(
                    f"relation {relation.id} cannot be matched with null enum_value"
                )
            if (
                relation.proposed_enum is not None
                and relation.relation_match is not None
                and relation.relation_match.status
                not in {CanonicalStatus.PROPOSED_NEW_ENUM, CanonicalStatus.UNMATCHED_RAW_ONLY}
            ):
                raise ValueError(
                    f"relation {relation.id} proposed_enum is only valid for proposed "
                    f"or unmatched relation matches"
                )

        if not any(element.importance != Importance.UNRESOLVED for element in self.elements):
            raise ValueError("at least one resolved visible element is required")

        return self


class ObjectDescriptor(StrictModel):
    target_id: ElementId
    description: ResidualDescription | None = None
    material: ShortText | None = None
    color: ShortText | None = None
    finish: Finish | None = None
    condition: Condition | None = None
    pattern: Pattern | None = None


class ObjectLane(StrictModel):
    objects: list[ObjectDescriptor] = Field(default_factory=list, max_length=32)


class EnumMatch(StrictModel):
    raw: ShortText
    enum_value: ShortText | None = None
    confidence: EnumMatchConfidence
    reason: ResidualDescription


class ActionDescriptor(StrictModel):
    actor_id: ElementId
    movement_raw: ShortText
    movement_match: EnumMatch | None = None
    target_id: ElementId | None = None
    intensity: MotionIntensity = MotionIntensity.SUBTLE
    support_status: ActionSupportStatus = ActionSupportStatus.SUPPORTED
    required_relation_types: list[RelationType] = Field(default_factory=list, max_length=8)
    inferred_support: InferredGraphSupport | None = None
    prompt_improvement: PromptImprovementHint | None = None
    description: ResidualDescription | None = None


class ActionLane(StrictModel):
    actions: list[ActionDescriptor] = Field(default_factory=list, max_length=16)


class CinematographyLane(StrictModel):
    shot_size: ShotSize | None = None
    camera_angle: CameraAngle | None = None
    optic_character: OpticCharacter | None = None
    camera_motion: ShortText | None = None
    focus_behavior: ShortText | None = None
    lighting_mood: LightingMood | None = None
    color_treatment: ColorTreatment | None = None
    framing: Framing | None = None
    setting_description: ResidualDescription | None = None


class ConstraintLane(StrictModel):
    guardrails: list[Guardrail] = Field(default_factory=list, max_length=16)
    negative_phrases: list[ShortText] = Field(default_factory=list, max_length=24)


class PromptDocumentSpec(StrictModel):
    prompt_document_id: DocId = "doc_001"
    prompt_document_version: str = "1"
    raw_user_prompt: ShortText | None = None
    graph: SceneGraphDraft
    object_lane: ObjectLane = Field(default_factory=ObjectLane)
    action_lane: ActionLane = Field(default_factory=ActionLane)
    cinematography_lane: CinematographyLane = Field(default_factory=CinematographyLane)
    constraint_lane: ConstraintLane = Field(default_factory=ConstraintLane)
    canonical_metadata: dict[str, CanonicalEnum] = Field(default_factory=dict)
    verification: VerificationReport = Field(default_factory=lambda: VerificationReport(approved=True, issues=[]))

    @model_validator(mode="after")
    def validate_references(self) -> "PromptDocumentSpec":
        known = {element.id for element in self.graph.elements}

        for obj in self.object_lane.objects:
            if obj.target_id not in known:
                raise ValueError(f"object descriptor has unknown target_id {obj.target_id}")

        for action in self.action_lane.actions:
            if action.actor_id not in known:
                raise ValueError(f"action has unknown actor_id {action.actor_id}")
            if action.target_id is not None and action.target_id not in known:
                raise ValueError(f"action has unknown target_id {action.target_id}")

        return self


class PromptBundle(StrictModel):
    positive_prompt: str
    negative_prompt: str
    alignment_checklist: list[str]
    render_trace: list[str]
    prompt_improvement_hints: list[PromptImprovementHint] = Field(default_factory=list)


__all__ = [
    "ActionDescriptor",
    "ActionLane",
    "CinematographyLane",
    "ConstraintLane",
    "Element",
    "EnumMatch",
    "ObjectDescriptor",
    "ObjectLane",
    "PromptBundle",
    "PromptDocumentSpec",
    "RelationDescriptor",
    "SceneGraphDraft",
]
