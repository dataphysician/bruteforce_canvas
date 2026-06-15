from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from bruteforce_canvas.prompt_enums import (
    ActionSupportStatus,
    ElementRole,
    EntityType,
    Guardrail,
    Importance,
    OpticCharacter,
    RelationType,
    ShotSize,
)
from bruteforce_canvas.shared import CanonicalStatus
from bruteforce_canvas.validation import ValidationIssue, ValidationSeverity

if TYPE_CHECKING:
    from bruteforce_canvas.prompt_models import PromptDocumentSpec, SceneGraphDraft

ValidationStage = Literal[
    "graph",
    "object_lane",
    "action_lane",
    "cinematography_lane",
    "constraint_lane",
    "document",
    "render",
]

RELATION_COMPATIBILITY: dict[RelationType, set[tuple[EntityType, EntityType]]] = {
    RelationType.WEARING: {
        (EntityType.ANIMAL, EntityType.ACCESSORY),
        (EntityType.PERSON, EntityType.TEXTILE),
        (EntityType.PERSON, EntityType.ACCESSORY),
    },
    RelationType.HOLDING: {
        (EntityType.PERSON, EntityType.ACCESSORY),
        (EntityType.PERSON, EntityType.PRODUCT),
        (EntityType.PERSON, EntityType.TOOL),
    },
    RelationType.CARRYING: {
        (EntityType.PERSON, EntityType.ACCESSORY),
        (EntityType.PERSON, EntityType.PRODUCT),
        (EntityType.PERSON, EntityType.TOOL),
    },
    RelationType.ATTACHED_TO: {
        (EntityType.ACCESSORY, EntityType.TEXTILE),
        (EntityType.PRODUCT, EntityType.VEHICLE),
    },
    RelationType.PART_OF: {
        (EntityType.PRODUCT, EntityType.PRODUCT),
        (EntityType.ARCHITECTURE, EntityType.LOCATION),
    },
}

_VALIDATOR_RETRY_MAP: dict[str, tuple[ValidationStage, str]] = {
    "validate_relation_compatibility": ("graph", "relation"),
    "relation_compatibility": ("graph", "relation"),
    "validate_object_ownership": ("object_lane", "object_descriptor"),
    "object_ownership": ("object_lane", "object_descriptor"),
    "validate_action_support": ("action_lane", "action_descriptor"),
    "action_support": ("action_lane", "action_descriptor"),
    "validate_evidence_and_placeholders": ("graph", "evidence_or_placeholder"),
    "evidence_and_placeholders": ("graph", "evidence_or_placeholder"),
    "validate_cross_lane_coherence": ("document", "cross_lane_coherence"),
    "cross_lane_coherence": ("document", "cross_lane_coherence"),
}


def _relation_type(relation: object) -> RelationType | None:
    relation_match = getattr(relation, "relation_match", None)
    if relation_match is None:
        return None
    enum_value = getattr(relation_match, "enum_value", None)
    if enum_value is None:
        return None
    try:
        return RelationType(enum_value)
    except ValueError:
        return None


def _relation_status(relation: object) -> str | None:
    relation_match = getattr(relation, "relation_match", None)
    if relation_match is None:
        return None
    status = getattr(relation_match, "status", None)
    return str(status) if status is not None else None


def _entity_type(element: object) -> EntityType | None:
    try:
        return EntityType(getattr(element, "entity_type"))
    except ValueError:
        return None


def _importance(element: object) -> Importance | None:
    try:
        return Importance(getattr(element, "importance"))
    except ValueError:
        return None


def _role(element: object) -> ElementRole | None:
    try:
        return ElementRole(getattr(element, "role"))
    except ValueError:
        return None


def _evidence_category(item: object) -> str | None:
    evidence = getattr(item, "evidence", None)
    if evidence is None:
        return None
    category = getattr(evidence, "category", None)
    return str(category) if category is not None else None


def _evidence_has_text_and_reason(item: object) -> bool:
    evidence = getattr(item, "evidence", None)
    if evidence is None:
        return False
    return bool(getattr(evidence, "text", None)) and bool(getattr(evidence, "reason", None))


def _known_elements(document: PromptDocumentSpec) -> dict[str, object]:
    return {element.id: element for element in document.graph.elements}


def _matched_relation_types(document: PromptDocumentSpec, actor_id: str, target_id: str | None) -> set[RelationType]:
    relation_types: set[RelationType] = set()
    for relation in document.graph.relations:
        relation_type = _relation_type(relation)
        if relation_type is None:
            continue
        if _relation_status(relation) != str(CanonicalStatus.MATCHED_ACTIVE):
            continue
        if relation.source_id != actor_id:
            continue
        if target_id is not None and relation.target_id != target_id:
            continue
        relation_types.add(relation_type)
    return relation_types


def _cinematography_has_surface(document: PromptDocumentSpec) -> bool:
    lane = document.cinematography_lane
    return any(
        getattr(lane, field_name) is not None
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
        )
    )


def validate_relation_compatibility(graph: SceneGraphDraft) -> list[str]:
    elements = {element.id: element for element in graph.elements}
    issues: list[str] = []

    for relation in graph.relations:
        if not str(getattr(relation, "relation_raw", "")).strip():
            issues.append(f"{relation.id} missing raw linkage text")
            continue

        relation_type = _relation_type(relation)
        if relation_type is None:
            continue

        allowed = RELATION_COMPATIBILITY.get(relation_type)
        if allowed is None:
            continue

        source = elements.get(relation.source_id)
        target = elements.get(relation.target_id)
        if source is None or target is None:
            issues.append(f"{relation.id} endpoint cannot be checked for compatibility")
            continue

        source_type = _entity_type(source)
        target_type = _entity_type(target)
        if source_type is None or target_type is None:
            issues.append(f"{relation.id} endpoint has unknown entity type")
            continue

        if (source_type, target_type) not in allowed:
            issues.append(
                f"{relation.id} invalid: {source_type.value} "
                f"{relation_type.value} {target_type.value}"
            )

    return issues


def validate_object_ownership(document: PromptDocumentSpec) -> list[str]:
    element_by_id = _known_elements(document)
    worn_targets = {
        relation.target_id
        for relation in document.graph.relations
        if _relation_type(relation) == RelationType.WEARING
    }
    person_sources = {
        relation.source_id
        for relation in document.graph.relations
        if _relation_type(relation) == RelationType.WEARING
    }

    issues: list[str] = []
    for obj in document.object_lane.objects:
        element = element_by_id.get(obj.target_id)
        if element is None:
            issues.append(f"{obj.target_id} object descriptor target is not in graph")
            continue

        entity_type = _entity_type(element)
        importance = _importance(element)
        if entity_type == EntityType.UNKNOWN_SLOT or importance == Importance.UNRESOLVED:
            issues.append(
                f"{obj.target_id} is unresolved; object lane must not assign concrete appearance"
            )
        if entity_type == EntityType.PERSON and obj.material and obj.target_id in person_sources:
            issues.append(
                f"{obj.target_id} has material={obj.material!r}; "
                "clothing material should attach to the worn garment element"
            )
        if obj.target_id in worn_targets and entity_type not in {EntityType.TEXTILE, EntityType.ACCESSORY}:
            issues.append(f"{obj.target_id} is worn but is not a textile/accessory")

    return issues


def validate_action_support(document: PromptDocumentSpec) -> list[str]:
    known = {element.id for element in document.graph.elements}
    issues: list[str] = []

    for action in document.action_lane.actions:
        if action.actor_id not in known:
            issues.append(f"action actor {action.actor_id} is not in graph")
            continue

        if action.target_id is not None and action.target_id not in known:
            issues.append(f"action target {action.target_id} is not in graph")

        relation_types = _matched_relation_types(document, action.actor_id, action.target_id)
        missing = [
            RelationType(relation_type)
            for relation_type in action.required_relation_types
            if RelationType(relation_type) not in relation_types
        ]

        if action.support_status == str(ActionSupportStatus.SUPPORTED) and missing:
            issues.append(
                f"supported action {action.movement_raw} lacks required relation preconditions: "
                f"{[relation_type.value for relation_type in missing]}"
            )

        if action.support_status == str(ActionSupportStatus.INFERRED) and action.inferred_support is None:
            issues.append("inferred action must include an inferred graph support request")

        if action.support_status == str(ActionSupportStatus.UNRESOLVED):
            if action.prompt_improvement is None:
                issues.append("unresolved action should include prompt-improvement feedback or a safe downgrade")
            if action.target_id is not None:
                target = next((element for element in document.graph.elements if element.id == action.target_id), None)
                if target is not None and _entity_type(target) != EntityType.UNKNOWN_SLOT:
                    issues.append("unresolved action target must be an unknown_slot or omitted")

        if action.support_status == str(ActionSupportStatus.INDETERMINATE) and action.target_id is not None:
            issues.append("indeterminate action should not claim a concrete target")

    return issues


def validate_evidence_and_placeholders(document: PromptDocumentSpec) -> list[str]:
    issues: list[str] = []
    unresolved_ids: set[str] = set()

    for element in document.graph.elements:
        category = _evidence_category(element)
        if category in {"entailed", "unresolved", "blocked"} and not _evidence_has_text_and_reason(element):
            issues.append(f"{element.id} non-explicit evidence requires text and reason")

        entity_type = _entity_type(element)
        importance = _importance(element)
        if importance == Importance.UNRESOLVED or entity_type == EntityType.UNKNOWN_SLOT:
            unresolved_ids.add(element.id)
            if entity_type != EntityType.UNKNOWN_SLOT:
                issues.append(f"{element.id} unresolved element must use entity_type=unknown_slot")
            if importance != Importance.UNRESOLVED:
                issues.append(f"{element.id} unresolved element must use importance=unresolved")
            if _role(element) == ElementRole.PRIMARY_SUBJECT:
                issues.append(f"{element.id} unresolved element cannot be the primary subject")
            if category != "unresolved":
                issues.append(f"{element.id} unresolved element requires unresolved evidence")

        if category == "blocked":
            issues.append(
                f"{element.id} is blocked and requires prompt-improvement feedback before lane expansion"
            )

    for relation in document.graph.relations:
        if relation.source_id in unresolved_ids or relation.target_id in unresolved_ids:
            if _evidence_category(relation) != "unresolved":
                issues.append(f"{relation.id} touching unresolved slot requires unresolved evidence")

    for obj in document.object_lane.objects:
        if obj.target_id in unresolved_ids:
            issues.append(f"{obj.target_id} unresolved slot cannot receive object descriptors")

    return issues


def validate_cross_lane_coherence(document: PromptDocumentSpec) -> list[str]:
    issues: list[str] = []

    if str(Guardrail.NO_EXTRA_PEOPLE) in set(document.constraint_lane.guardrails):
        required_people = [
            element.id
            for element in document.graph.elements
            if _entity_type(element) == EntityType.PERSON and _importance(element) == Importance.REQUIRED
        ]
        if required_people:
            issues.append(
                "no_extra_people conflicts with required person elements: "
                f"{', '.join(required_people)}"
            )

    required_labels = {
        str(element.label).lower()
        for element in document.graph.elements
        if _importance(element) == Importance.REQUIRED
    }
    for phrase in document.constraint_lane.negative_phrases:
        phrase_key = str(phrase).lower()
        conflicts = sorted(label for label in required_labels if label and label in phrase_key)
        if conflicts:
            issues.append(
                "negative phrase conflicts with required element label: "
                f"{', '.join(conflicts)}"
            )

    lane = document.cinematography_lane
    if lane.optic_character == str(OpticCharacter.MACRO) and lane.shot_size in {
        str(ShotSize.WIDE_SHOT),
        str(ShotSize.ESTABLISHING_SHOT),
    }:
        issues.append("warning: macro optic conflicts with wide or establishing shot scale")

    has_renderable_lane = any(
        (
            bool(document.object_lane.objects),
            bool(document.action_lane.actions),
            _cinematography_has_surface(document),
            bool(document.constraint_lane.guardrails),
            bool(document.constraint_lane.negative_phrases),
        )
    )
    if not has_renderable_lane:
        issues.append("document has only graph content; at least one renderable lane must be present")

    return issues


def validators_to_validation_issues(validator_name: str, issues: list[str]) -> list[ValidationIssue]:
    stage, retry_scope = _VALIDATOR_RETRY_MAP.get(validator_name, ("document", validator_name))
    return [
        ValidationIssue(
            stage=stage,
            severity=ValidationSeverity.ERROR,
            path="",
            message=issue,
            retry_scope=retry_scope,
        )
        for issue in issues
    ]


__all__ = [
    "RELATION_COMPATIBILITY",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_action_support",
    "validate_cross_lane_coherence",
    "validate_evidence_and_placeholders",
    "validate_object_ownership",
    "validate_relation_compatibility",
    "validators_to_validation_issues",
]
