from __future__ import annotations

from typing import Protocol

from pydantic import Field

from bruteforce_canvas.prompt import CanonicalEnum, VerificationIssue, VerificationReport
from bruteforce_canvas.prompt_models import PromptDocumentSpec
from bruteforce_canvas.shared import StrictModel
from bruteforce_canvas.validation import RetryRequest


class JsonLLMClient(Protocol):
    def generate_json(self, *, system: str, user: dict, schema_name: str) -> dict:
        """Return JSON-compatible data for the requested schema."""
        ...


class FieldEnumContext(StrictModel):
    field_name: str
    semantic_role: str
    enum_values: dict[str, str] = Field(default_factory=dict)


class LLMPromptExtractionAdapter:
    def __init__(self, client: JsonLLMClient) -> None:
        self.client = client

    def extract(self, raw_prompt: str) -> PromptDocumentSpec:
        payload = self.client.generate_json(
            system=(
                "Extract a graph-first PromptDocumentSpec from the raw prompt with no rule-based fallback. "
                "Preserve raw user wording, evidence spans, lane ownership, and unresolved slots."
            ),
            user={"raw_prompt": raw_prompt},
            schema_name="PromptDocumentSpec",
        )
        return PromptDocumentSpec.model_validate(payload, strict=False)


class LLMCanonicalizerAdapter:
    def __init__(self, client: JsonLLMClient, *, enum_contexts: dict[str, FieldEnumContext] | None = None) -> None:
        self.client = client
        self.enum_contexts = enum_contexts or {}

    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        context = self.enum_contexts.get(
            field_path,
            FieldEnumContext(field_name=field_path, semantic_role=field_path, enum_values={}),
        )
        payload = self.client.generate_json(
            system=(
                "Canonicalize one field-scoped raw value. Preserve the raw value, use only this field's enum "
                "context, and do not infer scene facts or invent graph participants."
            ),
            user={
                "field_path": field_path,
                "field_name": context.field_name,
                "semantic_role": context.semantic_role,
                "raw_value": raw_value,
                "enum_context": context.enum_values,
            },
            schema_name="CanonicalEnum",
        )
        return CanonicalEnum.model_validate(payload)


class LLMVerificationAdapter:
    def __init__(self, client: JsonLLMClient) -> None:
        self.client = client

    def verify(self, document: PromptDocumentSpec) -> VerificationReport:
        payload = self.client.generate_json(
            system=(
                "Verify graph linkage, lane ownership, enum fit, unresolved slots, prompt faithfulness, "
                "and renderability. Return structured issues without silently rewriting the document."
            ),
            user={
                "prompt_document_id": document.prompt_document_id,
                "document": document.model_dump(),
            },
            schema_name="VerificationReport",
        )
        return VerificationReport.model_validate(payload)


class LLMRepairAdapter:
    def __init__(self, client: JsonLLMClient) -> None:
        self.client = client

    def repair(self, document: PromptDocumentSpec, issue: RetryRequest | VerificationIssue) -> PromptDocumentSpec:
        if isinstance(issue, RetryRequest):
            repair_scope = issue.issues[0].retry_scope if issue.issues else issue.failed_stage
            issue_payload = issue.model_dump()
            instruction = issue.instruction
        else:
            repair_scope = issue.repair_scope
            issue_payload = issue.model_dump()
            instruction = "Repair the verifier issue while preserving unrelated lanes."

        payload = self.client.generate_json(
            system=(
                "Repair only the slice named by repair_scope. Preserve stable IDs, raw user language, "
                "and unrelated lanes."
            ),
            user={
                "prompt_document_id": document.prompt_document_id,
                "repair_scope": repair_scope,
                "issue": issue_payload,
                "instruction": instruction,
                "document": document.model_dump(),
            },
            schema_name="PromptDocumentSpecRepair",
        )
        return PromptDocumentSpec.model_validate(payload, strict=False)
