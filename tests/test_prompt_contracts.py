import pytest
from pydantic import ValidationError

from bruteforce_canvas.prompt import (
    EvidenceCategory,
    EvidenceSpan,
    ObjectLane,
    PromptDocumentSpec,
    RelationEnumMatch,
    SceneGraphDraft,
    render_prompt_spec,
    target_manifest_from_prompt_spec,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Importance, LightingMood, RelationType
from bruteforce_canvas.prompt_models import CinematographyLane, Element, ObjectDescriptor, RelationDescriptor
from bruteforce_canvas.shared import CanonicalStatus


def test_prompt_document_preserves_raw_relation_and_renders_generate_prompt():
    document = PromptDocumentSpec(
        raw_user_prompt="a red ceramic bowl on a wooden table",
        graph=SceneGraphDraft(
            seed_prompt="red ceramic bowl on wooden table",
            elements=[
                Element(
                    id="object_01",
                    label="bowl",
                    entity_type=EntityType.PRODUCT,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="red ceramic bowl", category=EvidenceCategory.EXPLICIT),
                ),
                Element(
                    id="object_02",
                    label="table",
                    entity_type=EntityType.FURNITURE,
                    role=ElementRole.SUPPORTING,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="wooden table", category=EvidenceCategory.EXPLICIT),
                ),
            ],
            relations=[
                RelationDescriptor(
                    id="rel_01",
                    source_id="object_01",
                    target_id="object_02",
                    relation_raw="on",
                    relation_match=RelationEnumMatch(
                        raw="on",
                        enum_value=RelationType.ON_TOP_OF,
                        status=CanonicalStatus.MATCHED_ACTIVE,
                        confidence="clear",
                        reason="clear support relation",
                    ),
                    evidence=EvidenceSpan(text="bowl on a wooden table", category=EvidenceCategory.EXPLICIT),
                )
            ],
        ),
        object_lane=ObjectLane(
            objects=[
                ObjectDescriptor(target_id="object_01", color="red", material="ceramic"),
                ObjectDescriptor(target_id="object_02", material="wooden"),
            ]
        ),
        cinematography_lane=CinematographyLane(lighting_mood=LightingMood.SOFT_NATURAL),
    )

    assert document.prompt_document_version == "1"
    rendered = render_prompt_spec(document)
    assert rendered.rendered_prompt.startswith("Generate ")
    assert "red ceramic bowl" in rendered.rendered_prompt
    assert "on wooden table" in rendered.rendered_prompt
    assert document.graph.relations[0].relation_raw == "on"


def test_prompt_document_rejects_missing_relation_endpoint():
    with pytest.raises(ValidationError, match="target_id"):
        PromptDocumentSpec(
            raw_user_prompt="a bowl on a table",
            graph=SceneGraphDraft(
                seed_prompt="bowl on table",
                elements=[
                    Element(
                        id="object_01",
                        label="bowl",
                        entity_type=EntityType.PRODUCT,
                        role=ElementRole.PRIMARY_SUBJECT,
                        importance=Importance.REQUIRED,
                        evidence=EvidenceSpan(text="bowl", category=EvidenceCategory.EXPLICIT),
                    )
                ],
                relations=[
                    RelationDescriptor(
                        id="rel_01",
                        source_id="object_01",
                        target_id="object_02",
                        relation_raw="on",
                        evidence=EvidenceSpan(text="bowl on a table", category=EvidenceCategory.EXPLICIT),
                    )
                ],
            ),
        )


def test_target_manifest_marks_locked_graph_facts_required_and_fixed():
    document = PromptDocumentSpec(
        raw_user_prompt="a ceramic bowl on a wooden table",
        graph=SceneGraphDraft(
            seed_prompt="ceramic bowl on wooden table",
            elements=[
                Element(
                    id="object_01",
                    label="bowl",
                    entity_type=EntityType.PRODUCT,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="ceramic bowl", category=EvidenceCategory.EXPLICIT),
                )
            ],
            relations=[],
        ),
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="object_01", material="ceramic")]),
    )

    rendered = render_prompt_spec(document)
    manifest = target_manifest_from_prompt_spec(document)

    element_target = next(target for target in manifest.targets if target.target_id == "object_01")
    material_target = next(target for target in manifest.targets if target.target_id == "object_01.material")
    assert element_target.priority == "locked_required"
    assert element_target.lhs_policy == "fixed"
    assert material_target.evaluation_policy == "must_match"
