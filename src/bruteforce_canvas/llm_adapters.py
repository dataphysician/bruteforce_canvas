from __future__ import annotations

import re
from typing import Protocol

from pydantic import Field, ValidationError

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
from bruteforce_canvas.shared import CanonicalStatus, StrictModel
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
    def __init__(self, client: JsonLLMClient, *, max_validation_retries: int = 1) -> None:
        self.client = client
        self.max_validation_retries = max_validation_retries

    def extract(self, raw_prompt: str) -> PromptDocumentSpec:
        system = (
            "Extract a graph-first PromptDocumentSpec from the raw prompt with no rule-based fallback. "
            "Preserve raw user wording, evidence spans, lane ownership, and unresolved slots."
        )
        user = {"raw_prompt": raw_prompt}
        last_error: ValidationError | None = None
        for _attempt in range(self.max_validation_retries + 1):
            payload = self.client.generate_json(
                system=system,
                user=user,
                schema_name="PromptDocumentSpec",
            )
            prepared = _prepare_extraction_payload(payload, raw_prompt=raw_prompt)
            try:
                return PromptDocumentSpec.model_validate(prepared, strict=False)
            except ValidationError as error:
                last_error = error
                system = (
                    "Retry PromptDocumentSpec extraction. The previous JSON failed local validation. "
                    "Return one complete top-level PromptDocumentSpec object with graph, lanes, and verification. "
                    "Do not return a single element, lane, markdown, commentary, or hidden reasoning."
                )
                user = {
                    "raw_prompt": raw_prompt,
                    "previous_invalid_payload": payload,
                    "validation_error": str(error),
                }
        if last_error is not None:
            raise last_error
        raise RuntimeError("PromptDocumentSpec extraction did not produce a response")


class LLMCanonicalizerAdapter:
    def __init__(self, client: JsonLLMClient, *, enum_contexts: dict[str, FieldEnumContext] | None = None) -> None:
        self.client = client
        self.enum_contexts = enum_contexts or {}

    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        context = self._context_for_field_path(field_path)
        try:
            payload = self.client.generate_json(
                system=(
                    "Canonicalize one extracted or repaired field enum candidate. Preserve the raw value, use only "
                    "the field-scoped stored enum context, and do not infer scene facts or invent graph participants. "
                    "If no stored enum key fits, return enum_value as null with unmatched_raw_only status."
                ),
                user={
                    "field_path": field_path,
                    "field_name": context.field_name,
                    "semantic_role": context.semantic_role,
                    "extracted_or_repaired_value": raw_value,
                    "raw_value": raw_value,
                    "stored_enum_context": context.enum_values,
                },
                schema_name="CanonicalEnum",
            )
            canonical = CanonicalEnum.model_validate(payload)
        except Exception:
            canonical = CanonicalEnum(
                raw_value=raw_value or "unknown",
                enum_value=None,
                status=CanonicalStatus.UNMATCHED_RAW_ONLY,
                confidence="low",
                reason="LLM canonicalization failed; preserved raw value as unmatched.",
            )
        return _enforce_stored_enum(canonical, context)

    def _context_for_field_path(self, field_path: str) -> FieldEnumContext:
        if field_path in self.enum_contexts:
            return self.enum_contexts[field_path]
        for prefix, context in self.enum_contexts.items():
            if field_path.startswith(prefix):
                return context
        return FieldEnumContext(field_name=field_path, semantic_role=field_path, enum_values={})


def _enforce_stored_enum(canonical: CanonicalEnum, context: FieldEnumContext) -> CanonicalEnum:
    if canonical.enum_value is None or not context.enum_values:
        return canonical
    if canonical.enum_value in context.enum_values:
        return canonical
    return CanonicalEnum(
        raw_value=canonical.raw_value,
        enum_value=None,
        status=CanonicalStatus.UNMATCHED_RAW_ONLY,
        confidence="low",
        reason=f"LLM returned non-registered enum {canonical.enum_value}",
    )


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
        merged_payload = _prepare_repair_document_payload(payload, document)
        return PromptDocumentSpec.model_validate(_prepare_prompt_document_payload(merged_payload), strict=False)


_ID_FIELD_NAMES = {"id", "source_id", "target_id", "actor_id"}
_OVER_PADDED_ID = re.compile(r"^(?P<prefix>[a-z]+)_0+(?P<number>[0-9]{2})$")


def _prepare_prompt_document_payload(payload: dict) -> dict:
    """Normalize LLM-produced document JSON before strict validation.

    Mellum2 often emits human-natural three-digit graph IDs such as
    ``obj_001``. The prompt schema intentionally uses compact two-digit
    IDs such as ``obj_01``, and the canonicalizer owns enum match fields.
    This cleanup keeps those mechanical concerns local to the LLM
    adapter without changing raw prompt content or lane descriptors.
    """

    if not isinstance(payload, dict):
        return payload
    prepared = _normalize_id_fields(payload)
    prepared.pop("canonical_metadata", None)
    graph = prepared.get("graph")
    if isinstance(graph, dict):
        for relation in graph.get("relations", []):
            if isinstance(relation, dict):
                relation.pop("relation_match", None)
    action_lane = prepared.get("action_lane")
    if isinstance(action_lane, dict):
        for action in action_lane.get("actions", []):
            if isinstance(action, dict):
                action.pop("movement_match", None)
    return prepared


def _prepare_extraction_payload(payload: dict, *, raw_prompt: str) -> dict:
    prepared = _prepare_prompt_document_payload(payload)
    if isinstance(prepared, dict) and not prepared.get("raw_user_prompt"):
        prepared = {**prepared, "raw_user_prompt": raw_prompt}
    return prepared


def _prepare_repair_document_payload(payload: dict, original: PromptDocumentSpec) -> dict:
    if not isinstance(payload, dict) or "graph" in payload:
        return payload

    merged = original.model_dump(mode="json")
    keys = set(payload)
    if {"seed_prompt", "elements", "relations"}.issubset(keys):
        merged["graph"] = payload
    elif "objects" in payload:
        merged["object_lane"] = payload
    elif "actions" in payload:
        merged["action_lane"] = payload
    elif keys & {
        "shot_size",
        "camera_angle",
        "optic_character",
        "camera_motion",
        "focus_behavior",
        "lighting_mood",
        "color_treatment",
        "framing",
        "setting_description",
    }:
        merged["cinematography_lane"] = payload
    elif keys & {"guardrails", "negative_phrases"}:
        merged["constraint_lane"] = payload
    else:
        return payload
    return merged


def _normalize_id_fields(value: object) -> object:
    if isinstance(value, list):
        return [_normalize_id_fields(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_id(raw) if key in _ID_FIELD_NAMES and isinstance(raw, str) else _normalize_id_fields(raw)
            for key, raw in value.items()
        }
    return value


def _normalize_id(value: str) -> str:
    match = _OVER_PADDED_ID.match(value)
    if match is None:
        return value
    return f"{match.group('prefix')}_{match.group('number')}"


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
