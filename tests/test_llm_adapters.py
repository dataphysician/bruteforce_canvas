from bruteforce_canvas.llm_adapters import (
    FieldEnumContext,
    LLMCanonicalizerAdapter,
    LLMPromptExtractionAdapter,
    LLMRepairAdapter,
    LLMVerificationAdapter,
)
from bruteforce_canvas.prompt import (
    EvidenceCategory,
    EvidenceSpan,
    ObjectLane,
    PromptDocumentSpec,
    SceneGraphDraft,
    VerificationIssue,
    VerificationReport,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Importance
from bruteforce_canvas.prompt_models import Element, ObjectDescriptor
from bruteforce_canvas.shared import CanonicalStatus


class FakeJsonClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def generate_json(self, *, system: str, user: dict, schema_name: str) -> dict:
        self.calls.append({"system": system, "user": user, "schema_name": schema_name})
        return self.responses.pop(0)


class RaisingJsonClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_json(self, *, system: str, user: dict, schema_name: str) -> dict:
        self.calls.append({"system": system, "user": user, "schema_name": schema_name})
        raise ValueError("LLM response did not contain a JSON object")


def prompt_document_payload() -> dict:
    return {
        "prompt_document_id": "doc_001",
        "raw_user_prompt": "a ceramic bowl on a wooden table",
        "graph": {
            "seed_prompt": "ceramic bowl on wooden table",
            "elements": [
                {
                    "id": "object_01",
                    "label": "bowl",
                    "entity_type": "product",
                    "role": "primary_subject",
                    "importance": "required",
                    "evidence": {"text": "ceramic bowl", "category": "explicit"},
                }
            ],
            "relations": [],
        },
        "object_lane": {"objects": [{"target_id": "object_01", "material": "ceramic"}]},
        "action_lane": {"actions": []},
        "cinematography_lane": {},
        "constraint_lane": {},
        "canonical_metadata": {},
        "verification": {"approved": False, "issues": []},
    }


def document() -> PromptDocumentSpec:
    return PromptDocumentSpec(
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
            ]
        ),
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="object_01", material="ceramic")]),
        verification=VerificationReport(approved=False, issues=[]),
    )


def test_llm_extraction_adapter_validates_prompt_document_and_preserves_raw_prompt():
    client = FakeJsonClient([prompt_document_payload()])
    adapter = LLMPromptExtractionAdapter(client)

    result = adapter.extract("a ceramic bowl on a wooden table")

    assert result.raw_user_prompt == "a ceramic bowl on a wooden table"
    assert client.calls[0]["schema_name"] == "PromptDocumentSpec"
    assert client.calls[0]["user"]["raw_prompt"] == "a ceramic bowl on a wooden table"
    assert "no rule-based fallback" in client.calls[0]["system"]


def test_llm_extraction_adapter_restores_raw_prompt_when_payload_omits_it():
    payload = prompt_document_payload()
    payload.pop("raw_user_prompt")
    client = FakeJsonClient([payload])

    result = LLMPromptExtractionAdapter(client).extract("a ceramic bowl on a wooden table")

    assert result.raw_user_prompt == "a ceramic bowl on a wooden table"


def test_llm_extraction_adapter_retries_invalid_prompt_document_payload():
    valid_payload = prompt_document_payload()
    client = FakeJsonClient(
        [
            {
                "id": "object_01",
                "label": "bowl",
                "entity_type": "product",
                "role": "primary_subject",
            },
            valid_payload,
        ]
    )

    result = LLMPromptExtractionAdapter(client).extract("a ceramic bowl on a wooden table")

    assert result.raw_user_prompt == "a ceramic bowl on a wooden table"
    assert len(client.calls) == 2
    assert client.calls[1]["schema_name"] == "PromptDocumentSpec"
    assert client.calls[1]["user"]["raw_prompt"] == "a ceramic bowl on a wooden table"
    assert client.calls[1]["user"]["previous_invalid_payload"]["id"] == "object_01"
    assert "PromptDocumentSpec" in client.calls[1]["system"]


def test_llm_extraction_adapter_normalizes_llm_id_padding_and_clears_canonical_fields():
    payload = prompt_document_payload()
    payload["graph"]["elements"][0]["id"] = "obj_001"
    payload["graph"]["relations"] = [
        {
            "id": "rel_001",
            "source_id": "obj_001",
            "target_id": "obj_002",
            "relation_raw": "on",
            "relation_match": {
                "raw": "on",
                "enum_value": "on",
                "confidence": "clear",
                "reason": "LLM prefilled a canonical field",
            },
        }
    ]
    payload["graph"]["elements"].append(
        {
            "id": "obj_002",
            "label": "table",
            "entity_type": "surface",
            "role": "supporting",
            "importance": "required",
            "evidence": {"text": "wooden table", "category": "explicit"},
        }
    )
    payload["object_lane"]["objects"] = [
        {"target_id": "obj_001", "material": "ceramic"},
        {"target_id": "obj_002", "material": "wooden"},
    ]
    payload["action_lane"]["actions"] = [
        {
            "actor_id": "obj_001",
            "movement_raw": "resting",
            "movement_match": {
                "raw": "resting",
                "enum_value": "stationary",
                "confidence": "clear",
                "reason": "LLM prefilled a canonical field",
            },
        }
    ]
    payload["canonical_metadata"] = {
        "relation.rel_001": {
            "raw_value": "on",
            "enum_value": "ON",
            "status": "matched_active",
            "confidence": "high",
            "reason": "LLM prefilled metadata",
        }
    }
    client = FakeJsonClient([payload])

    result = LLMPromptExtractionAdapter(client).extract("a ceramic bowl on a wooden table")

    assert [element.id for element in result.graph.elements] == ["obj_01", "obj_02"]
    assert result.graph.relations[0].id == "rel_01"
    assert result.graph.relations[0].source_id == "obj_01"
    assert result.graph.relations[0].target_id == "obj_02"
    assert result.graph.relations[0].relation_match is None
    assert result.object_lane.objects[0].target_id == "obj_01"
    assert result.action_lane.actions[0].actor_id == "obj_01"
    assert result.action_lane.actions[0].movement_match is None
    assert result.canonical_metadata == {}


def test_llm_canonicalizer_sends_only_field_scoped_context():
    client = FakeJsonClient(
        [
            {
                "raw_value": "ceramic",
                "enum_value": "CERAMIC",
                "status": "matched_active",
                "confidence": "high",
                "reason": "field enum match",
            }
        ]
    )
    adapter = LLMCanonicalizerAdapter(
        client,
        enum_contexts={
            "object.material.object_01": FieldEnumContext(
                field_name="material",
                semantic_role="object material",
                enum_values={"CERAMIC": "fired clay material"},
            )
        },
    )

    result = adapter.canonicalize(field_path="object.material.object_01", raw_value="ceramic")

    assert result.status == CanonicalStatus.MATCHED_ACTIVE
    assert client.calls[0]["schema_name"] == "CanonicalEnum"
    assert client.calls[0]["user"]["raw_value"] == "ceramic"
    assert client.calls[0]["user"]["extracted_or_repaired_value"] == "ceramic"
    assert client.calls[0]["user"]["stored_enum_context"] == {"CERAMIC": "fired clay material"}
    assert "raw_user_prompt" not in client.calls[0]["user"]


def test_llm_canonicalizer_resolves_field_family_context_by_prefix():
    client = FakeJsonClient(
        [
            {
                "raw_value": "on the table",
                "enum_value": "ON_TOP_OF",
                "status": "matched_active",
                "confidence": "high",
                "reason": "field enum match",
            }
        ]
    )
    adapter = LLMCanonicalizerAdapter(
        client,
        enum_contexts={
            "relation.": FieldEnumContext(
                field_name="relation",
                semantic_role="scene graph relation",
                enum_values={"ON_TOP_OF": "on top of", "INSIDE": "inside"},
            )
        },
    )

    result = adapter.canonicalize(field_path="relation.rel_01", raw_value="on the table")

    assert result.enum_value == "ON_TOP_OF"
    assert client.calls[0]["user"]["field_name"] == "relation"
    assert client.calls[0]["user"]["stored_enum_context"] == {"ON_TOP_OF": "on top of", "INSIDE": "inside"}


def test_llm_canonicalizer_rejects_unregistered_enum_output():
    client = FakeJsonClient(
        [
            {
                "raw_value": "on the table",
                "enum_value": "ABOVE_SURFACE",
                "status": "matched_active",
                "confidence": "high",
                "reason": "non-registry enum",
            }
        ]
    )
    adapter = LLMCanonicalizerAdapter(
        client,
        enum_contexts={
            "relation.": FieldEnumContext(
                field_name="relation",
                semantic_role="scene graph relation",
                enum_values={"ON_TOP_OF": "on top of", "INSIDE": "inside"},
            )
        },
    )

    result = adapter.canonicalize(field_path="relation.rel_01", raw_value="on the table")

    assert result.enum_value is None
    assert result.status == CanonicalStatus.UNMATCHED_RAW_ONLY
    assert "non-registered enum" in result.reason


def test_llm_canonicalizer_preserves_raw_value_when_llm_returns_malformed_json():
    client = RaisingJsonClient()
    adapter = LLMCanonicalizerAdapter(
        client,
        enum_contexts={
            "action.movement": FieldEnumContext(
                field_name="movement",
                semantic_role="action movement",
                enum_values={"STATIONARY": "not moving"},
            )
        },
    )

    result = adapter.canonicalize(field_path="action.movement.object_01", raw_value="floating")

    assert result.raw_value == "floating"
    assert result.enum_value is None
    assert result.status == CanonicalStatus.UNMATCHED_RAW_ONLY
    assert result.confidence == "low"
    assert "preserved raw value" in result.reason
    assert client.calls[0]["schema_name"] == "CanonicalEnum"


def test_llm_verifier_adapter_validates_structured_report():
    client = FakeJsonClient([{"approved": True, "issues": []}])
    result = LLMVerificationAdapter(client).verify(document())

    assert result.approved is True
    assert client.calls[0]["schema_name"] == "VerificationReport"
    assert client.calls[0]["user"]["prompt_document_id"] == "doc_001"


def test_llm_repair_adapter_sends_slice_scope_and_returns_repaired_document():
    issue = VerificationIssue(
        issue_type="descriptor_wrong_owner",
        repair_scope="object_descriptor",
        blocking=True,
        message="material assigned to wrong object",
    )
    payload = prompt_document_payload()
    payload["object_lane"] = {"objects": [{"target_id": "object_01", "material": "ceramic"}]}
    client = FakeJsonClient([payload])

    result = LLMRepairAdapter(client).repair(document(), issue)

    assert result.prompt_document_id == "doc_001"
    assert client.calls[0]["schema_name"] == "PromptDocumentSpecRepair"
    assert client.calls[0]["user"]["repair_scope"] == "object_descriptor"


def test_llm_repair_adapter_merges_partial_graph_slice_into_original_document():
    issue = VerificationIssue(
        issue_type="graph_skeleton",
        repair_scope="graph_skeleton",
        blocking=True,
        message="repair graph only",
    )
    client = FakeJsonClient(
        [
            {
                "seed_prompt": "repaired ceramic bowl on wooden table",
                "elements": [
                    {
                        "id": "object_001",
                        "label": "bowl",
                        "entity_type": "product",
                        "role": "primary_subject",
                        "importance": "required",
                        "evidence": {"text": "ceramic bowl", "category": "explicit"},
                    }
                ],
                "relations": [],
            }
        ]
    )

    result = LLMRepairAdapter(client).repair(document(), issue)

    assert result.graph.seed_prompt == "repaired ceramic bowl on wooden table"
    assert result.graph.elements[0].id == "object_01"
    assert result.object_lane.objects[0].target_id == "object_01"
