import pytest
from pydantic import ValidationError

from bruteforce_canvas.prompt import (
    CanonicalEnum,
    CinematographyLane,
    Element,
    Evidence,
    EvidenceCategory,
    Graph,
    ObjectDescriptor,
    PromptDocument,
    Relation,
    VerificationReport,
    render_prompt,
    target_manifest_from_prompt,
)
from bruteforce_canvas.shared import CanonicalStatus


def test_prompt_document_preserves_raw_relation_and_renders_generate_prompt():
    document = PromptDocument(
        prompt_document_id="doc_001",
        raw_user_prompt="a red ceramic bowl on a wooden table",
        seed_prompt="red ceramic bowl on wooden table",
        graph=Graph(
            elements=[
                Element(
                    element_id="object_01",
                    label="bowl",
                    entity_type="object",
                    importance="primary",
                    evidence=Evidence(text="red ceramic bowl", category=EvidenceCategory.EXPLICIT),
                ),
                Element(
                    element_id="object_02",
                    label="table",
                    entity_type="object",
                    importance="supporting",
                    evidence=Evidence(text="wooden table", category=EvidenceCategory.EXPLICIT),
                ),
            ],
            relations=[
                Relation(
                    relation_id="rel_01",
                    source_id="object_01",
                    target_id="object_02",
                    relation_raw="on",
                    canonical=CanonicalEnum(
                        raw_value="on",
                        enum_value="ON_TOP_OF",
                        status=CanonicalStatus.MATCHED_ACTIVE,
                        confidence="high",
                        reason="clear support relation",
                    ),
                    evidence=Evidence(text="bowl on a wooden table", category=EvidenceCategory.EXPLICIT),
                )
            ],
        ),
        objects=[
            ObjectDescriptor(element_id="object_01", field_name="color", raw_value="red"),
            ObjectDescriptor(element_id="object_01", field_name="material", raw_value="ceramic"),
            ObjectDescriptor(element_id="object_02", field_name="material", raw_value="wooden"),
        ],
        cinematography=CinematographyLane(lighting_raw="soft window light"),
        verification=VerificationReport(approved=True, issues=[]),
    )

    assert document.prompt_document_version == "1"
    rendered = render_prompt(document)
    assert rendered.rendered_prompt.startswith("Generate ")
    assert "red ceramic bowl" in rendered.rendered_prompt
    assert "on wooden table" in rendered.rendered_prompt
    assert document.graph.relations[0].relation_raw == "on"


def test_prompt_document_rejects_missing_relation_endpoint():
    with pytest.raises(ValidationError, match="target_id"):
        PromptDocument(
            prompt_document_id="doc_001",
            raw_user_prompt="a bowl on a table",
            seed_prompt="bowl on table",
            graph=Graph(
                elements=[
                    Element(
                        element_id="object_01",
                        label="bowl",
                        entity_type="object",
                        importance="primary",
                        evidence=Evidence(text="bowl", category=EvidenceCategory.EXPLICIT),
                    )
                ],
                relations=[
                    Relation(
                        relation_id="rel_01",
                        source_id="object_01",
                        target_id="object_02",
                        relation_raw="on",
                        evidence=Evidence(text="bowl on a table", category=EvidenceCategory.EXPLICIT),
                    )
                ],
            ),
            verification=VerificationReport(approved=True, issues=[]),
        )


def test_target_manifest_marks_locked_graph_facts_required_and_fixed():
    document = PromptDocument(
        prompt_document_id="doc_001",
        raw_user_prompt="a ceramic bowl on a wooden table",
        seed_prompt="ceramic bowl on wooden table",
        graph=Graph(
            elements=[
                Element(
                    element_id="object_01",
                    label="bowl",
                    entity_type="object",
                    importance="primary",
                    evidence=Evidence(text="ceramic bowl", category=EvidenceCategory.EXPLICIT),
                )
            ],
            relations=[],
        ),
        objects=[ObjectDescriptor(element_id="object_01", field_name="material", raw_value="ceramic")],
        verification=VerificationReport(approved=True, issues=[]),
    )

    rendered = render_prompt(document)
    manifest = target_manifest_from_prompt("run_001", rendered, document)

    element_target = next(target for target in manifest.targets if target.target_id == "object_01")
    material_target = next(target for target in manifest.targets if target.target_id == "object_01.material")
    assert element_target.priority == "locked_required"
    assert element_target.lhs_policy == "fixed"
    assert material_target.evaluation_policy == "must_match"
