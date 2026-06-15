from __future__ import annotations

from bruteforce_canvas.prompt import EvidenceCategory, EvidenceSpan, PromptImprovementHint, RelationEnumMatch
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
from bruteforce_canvas.prompt_models import (
    ActionDescriptor,
    ActionLane,
    CinematographyLane,
    ConstraintLane,
    Element,
    ObjectDescriptor,
    ObjectLane,
    PromptDocumentSpec,
    RelationDescriptor,
    SceneGraphDraft,
)
from bruteforce_canvas.shared import CanonicalStatus
from bruteforce_canvas.validators import (
    RELATION_COMPATIBILITY,
    ValidationSeverity,
    validate_action_support,
    validate_cross_lane_coherence,
    validate_evidence_and_placeholders,
    validate_object_ownership,
    validate_relation_compatibility,
    validators_to_validation_issues,
)


def _element(
    element_id: str,
    entity_type: EntityType,
    label: str,
    role: ElementRole = ElementRole.SUPPORTING,
    importance: Importance = Importance.REQUIRED,
    evidence: EvidenceSpan | None = None,
) -> Element:
    return Element(
        id=element_id,
        entity_type=entity_type,
        label=label,
        role=role,
        importance=importance,
        evidence=evidence,
    )


def _relation(
    source_id: str,
    target_id: str,
    relation_type: RelationType | None,
    relation_id: str = "rel_01",
    evidence: EvidenceSpan | None = None,
) -> RelationDescriptor:
    relation_match = None
    relation_raw = "near"
    if relation_type is not None:
        relation_raw = str(relation_type)
        relation_match = RelationEnumMatch(
            raw=str(relation_type),
            enum_value=relation_type,
            status=CanonicalStatus.MATCHED_ACTIVE,
            confidence="clear",
        )
    return RelationDescriptor(
        id=relation_id,
        source_id=source_id,
        target_id=target_id,
        relation_raw=relation_raw,
        relation_match=relation_match,
        evidence=evidence,
    )


def _document(
    elements: list[Element],
    relations: list[RelationDescriptor] | None = None,
    object_lane: ObjectLane | None = None,
    action_lane: ActionLane | None = None,
    cinematography_lane: CinematographyLane | None = None,
    constraint_lane: ConstraintLane | None = None,
) -> PromptDocumentSpec:
    return PromptDocumentSpec(
        graph=SceneGraphDraft(seed_prompt="test prompt", elements=elements, relations=relations or []),
        object_lane=object_lane or ObjectLane(),
        action_lane=action_lane or ActionLane(),
        cinematography_lane=cinematography_lane or CinematographyLane(),
        constraint_lane=constraint_lane or ConstraintLane(),
    )


def _document_construct(
    elements: list[Element],
    relations: list[RelationDescriptor] | None = None,
    object_lane: ObjectLane | None = None,
    action_lane: ActionLane | None = None,
    cinematography_lane: CinematographyLane | None = None,
    constraint_lane: ConstraintLane | None = None,
) -> PromptDocumentSpec:
    graph = SceneGraphDraft.model_construct(seed_prompt="test prompt", elements=elements, relations=relations or [])
    return PromptDocumentSpec.model_construct(
        graph=graph,
        object_lane=object_lane or ObjectLane(),
        action_lane=action_lane or ActionLane(),
        cinematography_lane=cinematography_lane or CinematographyLane(),
        constraint_lane=constraint_lane or ConstraintLane(),
        canonical_metadata={},
        verification=None,
    )


def _contains(issues: list[str], text: str) -> bool:
    return any(text in issue for issue in issues)


def test_relation_compatibility_table_matches_spec_pairs() -> None:
    expected = {
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

    assert RELATION_COMPATIBILITY == expected
    assert sum(len(pairs) for pairs in RELATION_COMPATIBILITY.values()) == 13


def test_relation_compatibility_accepts_valid_wearing_pair() -> None:
    graph = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("textile_01", EntityType.TEXTILE, "coat"),
        ],
        [_relation("person_01", "textile_01", RelationType.WEARING)],
    ).graph

    assert validate_relation_compatibility(graph) == []


def test_relation_compatibility_rejects_invalid_wearing_target() -> None:
    graph = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("location_01", EntityType.LOCATION, "street"),
        ],
        [_relation("person_01", "location_01", RelationType.WEARING)],
    ).graph

    assert _contains(validate_relation_compatibility(graph), "person wearing location")


def test_relation_compatibility_ignores_raw_only_relation() -> None:
    graph = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("location_01", EntityType.LOCATION, "street"),
        ],
        [_relation("person_01", "location_01", None)],
    ).graph

    assert validate_relation_compatibility(graph) == []


def test_relation_compatibility_ignores_unlisted_relation_type() -> None:
    graph = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("tool_01", EntityType.TOOL, "hammer"),
        ],
        [_relation("person_01", "tool_01", RelationType.USING)],
    ).graph

    assert validate_relation_compatibility(graph) == []


def test_relation_compatibility_flags_missing_raw_text() -> None:
    relation = RelationDescriptor.model_construct(
        id="rel_01",
        source_id="person_01",
        target_id="textile_01",
        relation_raw="",
        relation_match=None,
        proposed_enum=None,
        importance="required",
        evidence=None,
        description=None,
    )
    graph = SceneGraphDraft.model_construct(
        seed_prompt="x",
        elements=[
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("textile_01", EntityType.TEXTILE, "coat"),
        ],
        relations=[relation],
    )

    assert validate_relation_compatibility(graph) == ["rel_01 missing raw linkage text"]


def test_object_ownership_accepts_material_on_worn_textile() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("textile_01", EntityType.TEXTILE, "coat"),
        ],
        [_relation("person_01", "textile_01", RelationType.WEARING)],
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="textile_01", material="wool")]),
    )

    assert validate_object_ownership(document) == []


def test_object_ownership_rejects_clothing_material_on_person() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("textile_01", EntityType.TEXTILE, "coat"),
        ],
        [_relation("person_01", "textile_01", RelationType.WEARING)],
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="person_01", material="wool")]),
    )

    assert _contains(validate_object_ownership(document), "clothing material should attach")


def test_object_ownership_accepts_handbag_material_on_accessory() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("accessory_01", EntityType.ACCESSORY, "handbag"),
        ],
        [_relation("person_01", "accessory_01", RelationType.HOLDING)],
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="accessory_01", material="leather")]),
    )

    assert validate_object_ownership(document) == []


def test_object_ownership_rejects_descriptor_on_unresolved_slot() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element(
                "object_01",
                EntityType.UNKNOWN_SLOT,
                "something",
                importance=Importance.UNRESOLVED,
                evidence=EvidenceSpan(text="something", category=EvidenceCategory.UNRESOLVED, reason="open slot"),
            ),
        ],
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="object_01", color="red")]),
    )

    assert _contains(validate_object_ownership(document), "must not assign concrete appearance")


def test_object_ownership_rejects_non_textile_worn_target() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("product_01", EntityType.PRODUCT, "box"),
        ],
        [_relation("person_01", "product_01", RelationType.WEARING)],
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="product_01", color="red")]),
    )

    assert _contains(validate_object_ownership(document), "is worn but is not a textile/accessory")


def test_object_ownership_flags_missing_descriptor_target_from_constructed_document() -> None:
    document = _document_construct(
        [_element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT)],
        object_lane=ObjectLane.model_construct(
            objects=[ObjectDescriptor.model_construct(target_id="object_99", color="red")]
        ),
    )

    assert validate_object_ownership(document) == ["object_99 object descriptor target is not in graph"]


def test_action_support_accepts_supported_action_with_required_relation() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("tool_01", EntityType.TOOL, "sword"),
        ],
        [_relation("person_01", "tool_01", RelationType.HOLDING)],
        action_lane=ActionLane(
            actions=[
                ActionDescriptor(
                    actor_id="person_01",
                    target_id="tool_01",
                    movement_raw="swordfighting",
                    required_relation_types=[RelationType.HOLDING],
                )
            ]
        ),
    )

    assert validate_action_support(document) == []


def test_action_support_flags_missing_actor() -> None:
    document = _document_construct(
        [_element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT)],
        action_lane=ActionLane.model_construct(
            actions=[ActionDescriptor.model_construct(actor_id="person_99", movement_raw="running")]
        ),
    )

    assert validate_action_support(document) == ["action actor person_99 is not in graph"]


def test_action_support_flags_missing_target() -> None:
    document = _document_construct(
        [_element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT)],
        action_lane=ActionLane.model_construct(
            actions=[
                ActionDescriptor.model_construct(
                    actor_id="person_01",
                    target_id="object_99",
                    movement_raw="reaching toward object",
                    required_relation_types=[],
                    support_status="supported",
                    inferred_support=None,
                    prompt_improvement=None,
                )
            ]
        ),
    )

    assert validate_action_support(document) == ["action target object_99 is not in graph"]


def test_action_support_flags_supported_action_missing_relation_precondition() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("tool_01", EntityType.TOOL, "sword"),
        ],
        action_lane=ActionLane(
            actions=[
                ActionDescriptor(
                    actor_id="person_01",
                    target_id="tool_01",
                    movement_raw="swordfighting",
                    required_relation_types=[RelationType.HOLDING],
                )
            ]
        ),
    )

    assert _contains(validate_action_support(document), "lacks required relation preconditions")


def test_action_support_flags_inferred_action_without_support_request() -> None:
    document = _document(
        [_element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT)],
        action_lane=ActionLane(
            actions=[
                ActionDescriptor(
                    actor_id="person_01",
                    movement_raw="swordfighting",
                    support_status=ActionSupportStatus.INFERRED,
                )
            ]
        ),
    )

    assert validate_action_support(document) == ["inferred action must include an inferred graph support request"]


def test_action_support_flags_unresolved_action_without_feedback_and_concrete_target() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("product_01", EntityType.PRODUCT, "ball"),
        ],
        action_lane=ActionLane(
            actions=[
                ActionDescriptor(
                    actor_id="person_01",
                    target_id="product_01",
                    movement_raw="throwing something",
                    support_status=ActionSupportStatus.UNRESOLVED,
                )
            ]
        ),
    )

    issues = validate_action_support(document)
    assert _contains(issues, "prompt-improvement feedback")
    assert _contains(issues, "unknown_slot or omitted")


def test_action_support_flags_indeterminate_action_with_target() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element("product_01", EntityType.PRODUCT, "ball"),
        ],
        action_lane=ActionLane(
            actions=[
                ActionDescriptor(
                    actor_id="person_01",
                    target_id="product_01",
                    movement_raw="doing something impossible",
                    support_status=ActionSupportStatus.INDETERMINATE,
                    prompt_improvement=PromptImprovementHint(issue="unclear"),
                )
            ]
        ),
    )

    assert validate_action_support(document) == ["indeterminate action should not claim a concrete target"]


def test_evidence_accepts_entailed_evidence_with_text_and_reason() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element(
                "tool_01",
                EntityType.TOOL,
                "sword",
                evidence=EvidenceSpan(text="swordfighting", category=EvidenceCategory.ENTAILED, reason="requires weapon"),
            ),
        ]
    )

    assert validate_evidence_and_placeholders(document) == []


def test_evidence_rejects_entailed_evidence_without_reason() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element(
                "tool_01",
                EntityType.TOOL,
                "sword",
                evidence=EvidenceSpan(text="swordfighting", category=EvidenceCategory.ENTAILED),
            ),
        ]
    )

    assert validate_evidence_and_placeholders(document) == [
        "tool_01 non-explicit evidence requires text and reason"
    ]


def test_evidence_rejects_unresolved_element_with_wrong_type_role_and_evidence() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman"),
            _element(
                "object_01",
                EntityType.PRODUCT,
                "something",
                ElementRole.PRIMARY_SUBJECT,
                Importance.UNRESOLVED,
                EvidenceSpan(text="something", category=EvidenceCategory.EXPLICIT),
            ),
        ]
    )

    issues = validate_evidence_and_placeholders(document)
    assert _contains(issues, "must use entity_type=unknown_slot")
    assert _contains(issues, "cannot be the primary subject")
    assert _contains(issues, "requires unresolved evidence")


def test_evidence_rejects_blocked_graph_fact() -> None:
    document = _document(
        [
            _element(
                "person_01",
                EntityType.PERSON,
                "person",
                ElementRole.PRIMARY_SUBJECT,
                evidence=EvidenceSpan(text="unsafe ambiguity", category=EvidenceCategory.BLOCKED, reason="blocked"),
            )
        ]
    )

    assert _contains(validate_evidence_and_placeholders(document), "requires prompt-improvement feedback")


def test_evidence_rejects_relation_touching_unresolved_slot_without_unresolved_evidence() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element(
                "object_01",
                EntityType.UNKNOWN_SLOT,
                "something",
                importance=Importance.UNRESOLVED,
                evidence=EvidenceSpan(text="something", category=EvidenceCategory.UNRESOLVED, reason="open slot"),
            ),
        ],
        [_relation("person_01", "object_01", None)],
    )

    assert validate_evidence_and_placeholders(document) == [
        "rel_01 touching unresolved slot requires unresolved evidence"
    ]


def test_evidence_rejects_object_descriptor_on_unresolved_slot() -> None:
    document = _document(
        [
            _element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT),
            _element(
                "object_01",
                EntityType.UNKNOWN_SLOT,
                "something",
                importance=Importance.UNRESOLVED,
                evidence=EvidenceSpan(text="something", category=EvidenceCategory.UNRESOLVED, reason="open slot"),
            ),
        ],
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="object_01", color="red")]),
    )

    assert validate_evidence_and_placeholders(document) == [
        "object_01 unresolved slot cannot receive object descriptors"
    ]


def test_cross_lane_flags_no_extra_people_with_required_person() -> None:
    document = _document(
        [_element("person_01", EntityType.PERSON, "woman", ElementRole.PRIMARY_SUBJECT)],
        constraint_lane=ConstraintLane(guardrails=[Guardrail.NO_EXTRA_PEOPLE]),
    )

    assert _contains(validate_cross_lane_coherence(document), "no_extra_people conflicts")


def test_cross_lane_accepts_no_extra_people_without_required_person() -> None:
    document = _document(
        [_element("person_01", EntityType.PERSON, "woman", importance=Importance.OPTIONAL)],
        constraint_lane=ConstraintLane(guardrails=[Guardrail.NO_EXTRA_PEOPLE]),
    )

    assert validate_cross_lane_coherence(document) == []


def test_cross_lane_flags_negative_phrase_negating_required_label() -> None:
    document = _document(
        [_element("accessory_01", EntityType.ACCESSORY, "handbag", ElementRole.PRIMARY_SUBJECT)],
        constraint_lane=ConstraintLane(negative_phrases=["no handbag"]),
    )

    assert _contains(validate_cross_lane_coherence(document), "handbag")


def test_cross_lane_flags_macro_wide_scale_warning() -> None:
    document = _document(
        [_element("product_01", EntityType.PRODUCT, "watch", ElementRole.PRIMARY_SUBJECT)],
        cinematography_lane=CinematographyLane(
            optic_character=OpticCharacter.MACRO,
            shot_size=ShotSize.WIDE_SHOT,
        ),
    )

    assert validate_cross_lane_coherence(document) == [
        "warning: macro optic conflicts with wide or establishing shot scale"
    ]


def test_cross_lane_flags_graph_only_document_without_renderable_lane() -> None:
    document = _document([_element("product_01", EntityType.PRODUCT, "watch", ElementRole.PRIMARY_SUBJECT)])

    assert validate_cross_lane_coherence(document) == [
        "document has only graph content; at least one renderable lane must be present"
    ]


def test_cross_lane_accepts_object_lane_as_renderable_surface() -> None:
    document = _document(
        [_element("product_01", EntityType.PRODUCT, "watch", ElementRole.PRIMARY_SUBJECT)],
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="product_01", color="silver")]),
    )

    assert validate_cross_lane_coherence(document) == []


def test_validators_to_validation_issues_maps_validator_name() -> None:
    issues = validators_to_validation_issues("validate_action_support", ["action actor person_99 is not in graph"])

    assert len(issues) == 1
    assert issues[0].stage == "action_lane"
    assert issues[0].severity == ValidationSeverity.ERROR
    assert issues[0].path == ""
    assert issues[0].message == "action actor person_99 is not in graph"
    assert issues[0].retry_scope == "action_descriptor"


def test_validators_to_validation_issues_falls_back_to_document_scope() -> None:
    issues = validators_to_validation_issues("custom_validator", ["custom failure"])

    assert issues[0].stage == "document"
    assert issues[0].retry_scope == "custom_validator"
