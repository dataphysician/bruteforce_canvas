from __future__ import annotations

from typing import Protocol

from pydantic import Field

from bruteforce_canvas.prompt import (
    ActionLane,
    CanonicalEnum,
    ConstraintLane,
    ObjectLane,
    VerificationIssue,
    VerificationReport,
)
from bruteforce_canvas.prompt_models import (
    CinematographyLane,
    PromptDocumentSpec,
    SceneGraphDraft,
)
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


class LLMLaneCallProtocol(Protocol):
    """Structural protocol for spec §7 per-lane LLM call contracts.

    Each adapter expands one lane from a frozen ``SceneGraphDraft`` and
    optionally the original ``raw_prompt`` (for cinematography and
    constraint lanes). Adapters return a validated Pydantic lane model
    that the pipeline can stitch into the merged ``PromptDocumentSpec``.
    """

    system_instruction: str

    def expand(
        self,
        graph: SceneGraphDraft,
        **kwargs: object,
    ) -> ObjectLane | ActionLane | CinematographyLane | ConstraintLane:
        ...


class LLMObjectLaneAdapter:
    """Spec §7.2 Object Lane Call.

    Expands appearance descriptors for existing graph elements. The
    adapter must attach appearance to the element it describes, must not
    create relations, must not invent element IDs, and must not describe
    actions or camera.
    """

    system_instruction: str = (
        "Expand appearance descriptors for existing graph elements. "
        "Attach appearance to the element it describes. "
        "Do not create relations. "
        "Do not invent element IDs. "
        "Do not describe actions or camera. "
        "Return JSON matching ObjectLane."
    )

    def __init__(self, client: JsonLLMClient) -> None:
        self.client = client

    def expand(self, graph: SceneGraphDraft) -> ObjectLane:
        payload = self.client.generate_json(
            system=self.system_instruction,
            user={
                "graph": graph.model_dump(),
            },
            schema_name="ObjectLane",
        )
        return ObjectLane.model_validate(payload, strict=False)


class LLMActionLaneAdapter:
    """Spec §7.3 Action Lane Call.

    Expands temporal, fluid, dynamic, or implied-motion behavior. Stable
    spatial, ownership, and contact relations stay in the graph; this
    adapter only fills the action lane and preserves the raw movement
    phrase as the rendering source.
    """

    system_instruction: str = (
        "Expand temporal, fluid, dynamic, or implied-motion behavior for "
        "the validated graph. Preserve the raw movement phrase; do not "
        "rewrite the user's wording. Do not invent elements, do not "
        "create relations, and do not absorb stable spatial, contact, or "
        "ownership state that belongs in the graph. Return JSON matching "
        "ActionLane."
    )

    def __init__(self, client: JsonLLMClient) -> None:
        self.client = client

    def expand(self, graph: SceneGraphDraft) -> ActionLane:
        payload = self.client.generate_json(
            system=self.system_instruction,
            user={
                "graph": graph.model_dump(),
            },
            schema_name="ActionLane",
        )
        return ActionLane.model_validate(payload, strict=False)


class LLMCinematographyLaneAdapter:
    """Spec §7.4 Cinematography Lane Call.

    Extracts cinematography, lighting, framing, lens, color treatment,
    and setting atmosphere. Uses the raw prompt and ``seed_prompt`` as
    evidence for shot language and the validated graph only to preserve
    scene context; must not add, remove, rename, or reassign elements
    and must not leak object materials, clothing ownership, actions,
    relations, or negative constraints into the lane.
    """

    system_instruction: str = (
        "Extract only cinematography, lighting, framing, lens, color "
        "treatment, and setting atmosphere. "
        "Use the raw prompt and graph seed_prompt as evidence for shot "
        "language. Use the validated graph only to preserve scene "
        "context; do not add, remove, rename, or reassign elements. "
        "Do not output object materials, clothing ownership, actions, "
        "relations, or negative constraints. "
        "Return JSON matching CinematographyLane."
    )

    def __init__(self, client: JsonLLMClient) -> None:
        self.client = client

    def expand(self, graph: SceneGraphDraft, raw_prompt: str) -> CinematographyLane:
        payload = self.client.generate_json(
            system=self.system_instruction,
            user={
                "raw_user_prompt": raw_prompt,
                "validated_scene_graph": graph.model_dump(),
            },
            schema_name="CinematographyLane",
        )
        return CinematographyLane.model_validate(payload, strict=False)


class LLMConstraintLaneAdapter:
    """Spec §7.5 Constraint Lane Call.

    Extracts guardrails and negative phrases. Must not negate required
    graph content (no banning of required people, props, or visual
    properties) and must not absorb positive visual content into the
    constraint surface.
    """

    system_instruction: str = (
        "Extract exclusions and guardrails only. "
        "Do not negate required graph elements or required visual "
        "properties; narrow the negative phrase when the prompt still "
        "needs the element. "
        "Do not absorb positive visual content into the constraint "
        "surface. Return JSON matching ConstraintLane."
    )

    def __init__(self, client: JsonLLMClient) -> None:
        self.client = client

    def expand(self, graph: SceneGraphDraft, raw_prompt: str) -> ConstraintLane:
        payload = self.client.generate_json(
            system=self.system_instruction,
            user={
                "raw_user_prompt": raw_prompt,
                "validated_scene_graph": graph.model_dump(),
            },
            schema_name="ConstraintLane",
        )
        return ConstraintLane.model_validate(payload, strict=False)
