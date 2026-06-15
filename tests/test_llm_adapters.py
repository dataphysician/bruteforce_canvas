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
    assert client.calls[0]["user"]["enum_context"] == {"CERAMIC": "fired clay material"}
    assert "raw_user_prompt" not in client.calls[0]["user"]


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
