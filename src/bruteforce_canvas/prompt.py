from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from bruteforce_canvas.prompt_enums import EntityType, RelationType
from bruteforce_canvas.shared import (
    CanonicalStatus,
    Confidence,
    DocId,
    ResidualDescription,
    RunId,
    ShortText,
    StrictModel,
    TargetManifestId,
)
from bruteforce_canvas.validation import RetryRequest, ValidationIssue, ValidationSeverity


class EvidenceCategory(StrEnum):
    EXPLICIT = "explicit"
    ENTAILED = "entailed"
    UNRESOLVED = "unresolved"
    BLOCKED = "blocked"


class CanonicalEnum(StrictModel):
    raw_value: ShortText
    enum_value: str | None = None
    status: str
    confidence: Confidence
    reason: ShortText


class VerificationIssue(StrictModel):
    issue_type: ShortText
    repair_scope: ShortText
    blocking: bool
    message: ShortText


class VerificationReport(StrictModel):
    approved: bool
    issues: list[VerificationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def approved_without_blocking_issues(self) -> "VerificationReport":
        if self.approved and any(issue.blocking for issue in self.issues):
            raise ValueError("approved verification cannot contain blocking issues")
        return self


class RenderedPrompt(StrictModel):
    run_id: RunId | None = None
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId | None = None
    coordinate_id: str | None = None
    rendered_prompt: str
    rendering_trace: list[str]

    @model_validator(mode="after")
    def begins_with_generate(self) -> "RenderedPrompt":
        if not self.rendered_prompt.startswith("Generate "):
            raise ValueError("rendered_prompt must begin with 'Generate '")
        return self


class EvaluationTarget(StrictModel):
    target_id: str
    target_kind: Literal["element", "relation", "object_attribute", "cinematography", "constraint"]
    label: str | None = None
    value_raw: str | None = None
    enum_value: str | None = None
    source_id: str | None = None
    target_id_ref: str | None = None
    relation_raw: str | None = None
    priority: Literal["locked_required", "locked_context", "important", "sampled", "optional", "negative_guard", "diagnostic"]
    lhs_policy: Literal["fixed", "sampleable", "sampleable_if_missing", "blocked"]
    evaluation_policy: Literal["must_match", "should_match", "nice_to_have", "must_not_appear", "measure_only"]
    evidence: str | None = None


class EvaluationTargetManifest(StrictModel):
    manifest_id: TargetManifestId
    run_id: RunId
    prompt_document_id: DocId
    coordinate_id: str | None = None
    rendered_prompt: str
    targets: list[EvaluationTarget]
    negative_targets: list[EvaluationTarget] = Field(default_factory=list)


class EvidenceSpan(StrictModel):
    text: ShortText
    category: EvidenceCategory = EvidenceCategory.EXPLICIT
    reason: ResidualDescription | None = None


class RelationEnumMatch(StrictModel):
    raw: ShortText = Field(
        description="Raw linkage phrase from the prompt, such as resting on, leaning against, or draped over.",
    )
    enum_value: RelationType | None = Field(
        default=None,
        description=(
            "Optional locked relation enum. Null means preserve raw linkage and do not force a bad enum."
        ),
    )
    status: CanonicalStatus
    confidence: ShortText = Field(description="clear, weak, unmatched, or rejected")
    reason: ResidualDescription | None = None


class ProposedRelationEnum(StrictModel):
    code: ShortText = Field(
        description="UPPER_SNAKE_CASE candidate for the enum registry candidate queue.",
    )
    label: ShortText
    aliases: list[ShortText] = Field(default_factory=list, max_length=12)
    description: ResidualDescription
    source_type_examples: list[EntityType] = Field(default_factory=list, max_length=8)
    target_type_examples: list[EntityType] = Field(default_factory=list, max_length=8)


class InferredGraphSupport(StrictModel):
    reason: ShortText
    implied_element_label: ShortText | None = None
    implied_relation_type: RelationType | None = None
    source_action: ShortText
    evidence: EvidenceSpan


class PromptImprovementHint(StrictModel):
    issue: ShortText
    suggested_rewrites: list[ShortText] = Field(default_factory=list, max_length=4)
    safe_downgrade: ShortText | None = None


if TYPE_CHECKING:
    from bruteforce_canvas.prompt_models import (
        ActionLane,
        CinematographyLane,
        ConstraintLane,
        ObjectLane,
        PromptBundle,
        PromptDocumentSpec,
        SceneGraphDraft,
    )


_PROMPT_MODEL_EXPORTS = {
    "ActionLane",
    "CinematographyLane",
    "ConstraintLane",
    "ObjectLane",
    "PromptBundle",
    "PromptDocumentSpec",
    "SceneGraphDraft",
}


def __getattr__(name: str) -> object:
    if name in _PROMPT_MODEL_EXPORTS:
        from bruteforce_canvas import prompt_models

        value = getattr(prompt_models, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _clean_spec_text(value: object | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


_PROMPT_COMMAND_PREFIX = re.compile(
    r"^(?:generate|create|make|draw|show)\s+(?:(?:a|an)\s+)?(?:(?:clear|detailed)\s+)?"
    r"(?:image|picture|photo|rendering)\s+of\s+",
    re.IGNORECASE,
)
_RELATION_NEEDS_ARTICLE = re.compile(
    r"\b(on top of|inside|under|over|next to|in front of|behind|on|in)\s+"
    r"(?!a\b|an\b|the\b)([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})(?=,|$)",
    re.IGNORECASE,
)
_TRAILING_FRAGMENT_PUNCTUATION = " \t\n\r,.;:"
_SEMANTIC_DROP_TOKENS = {"a", "an", "the", "of"}
_LEADING_ARTICLE_EXCLUSIONS = {
    "a",
    "an",
    "the",
    "no",
    "two",
    "three",
    "four",
    "five",
    "several",
    "multiple",
    "many",
    "group",
}


def _clean_prompt_fragment(value: object | None) -> str:
    text = _clean_spec_text(value)
    if not text:
        return ""
    text = re.sub(r"^generate\s+", "", text, flags=re.IGNORECASE)
    text = _PROMPT_COMMAND_PREFIX.sub("", text)
    return text.strip(_TRAILING_FRAGMENT_PUNCTUATION)


def _semantic_tokens(value: object | None) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", _clean_prompt_fragment(value).lower())
        if token not in _SEMANTIC_DROP_TOKENS
    ]
    deduped: list[str] = []
    for token in tokens:
        if deduped and deduped[-1] == token:
            continue
        deduped.append(token)
    return deduped


def _semantic_key(value: object | None) -> str:
    return " ".join(_semantic_tokens(value))


def _is_semantically_redundant(candidate: str, existing: list[str]) -> bool:
    candidate_key = _semantic_key(candidate)
    if not candidate_key:
        return True
    candidate_tokens = set(candidate_key.split())
    for phrase in existing:
        existing_key = _semantic_key(phrase)
        if not existing_key:
            continue
        if candidate_key == existing_key or candidate_key in existing_key:
            return True
        existing_tokens = set(existing_key.split())
        if candidate_tokens.issubset(existing_tokens):
            return True
        if existing_key in candidate_key and len(candidate_tokens) <= len(existing_tokens) + 1:
            return True
    return False


def _append_scene_phrase(parts: list[str], value: object | None) -> None:
    text = _clean_prompt_fragment(value)
    if not text or _is_semantically_redundant(text, parts):
        return
    parts.append(text)


def _lower_initial_article(value: str) -> str:
    return re.sub(r"^(A|An|The)\b", lambda match: match.group(1).lower(), value, count=1)


def _lower_initial_word(value: str) -> str:
    match = re.match(r"([A-Z][a-z]+)(\b.*)", value)
    if not match:
        return value
    return match.group(1).lower() + match.group(2)


def _add_missing_relation_articles(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        relation = match.group(1)
        target = match.group(2)
        return f"{relation} a {target}"

    return _RELATION_NEEDS_ARTICLE.sub(replace, value)


def _add_missing_leading_article(value: str) -> str:
    first = value.split(maxsplit=1)[0].strip(",.").lower() if value.split() else ""
    if not first or first in _LEADING_ARTICLE_EXCLUSIONS:
        return value
    return f"a {value}"


def _normalize_positive_prompt_text(value: str) -> str:
    normalized = _lower_initial_article(value)
    normalized = _lower_initial_word(normalized)
    normalized = _add_missing_relation_articles(normalized)
    return _add_missing_leading_article(normalized)


def _display_spec_value(value: object | None) -> str:
    return _clean_spec_text(value).replace("_", " ")


def _append_unique(parts: list[str], value: object | None, *, display: bool = False) -> None:
    text = _display_spec_value(value) if display else _clean_spec_text(value)
    if text and text not in parts:
        parts.append(text)


def _spec_object_parts(descriptor: object) -> list[str]:
    parts: list[str] = []
    enum_fields = {"condition", "finish", "pattern"}
    for field_name in ("condition", "color", "pattern", "material", "finish", "description"):
        value = getattr(descriptor, field_name, None)
        if value:
            parts.append(_display_spec_value(value) if field_name in enum_fields else _clean_spec_text(value))
    return parts


def object_phrase(element: object) -> str:
    """Return a graph element's raw label for rendering."""
    return _clean_spec_text(getattr(element, "label", None))


def _spec_object_descriptor_map(document: PromptDocumentSpec) -> dict[str, list[object]]:
    object_descriptors: dict[str, list[object]] = {}
    for descriptor in document.object_lane.objects:
        object_descriptors.setdefault(descriptor.target_id, []).append(descriptor)
    return object_descriptors


def _spec_element_phrase(element: object, descriptors: list[object]) -> str:
    parts: list[str] = []
    label = object_phrase(element)
    label_tokens = set(_semantic_tokens(label))
    for descriptor in descriptors:
        for part in _spec_object_parts(descriptor):
            part_tokens = set(_semantic_tokens(part))
            if part_tokens and part_tokens.issubset(label_tokens):
                continue
            _append_unique(parts, part)
    _append_unique(parts, label)
    return " ".join(parts)


def _spec_scene_phrases(document: PromptDocumentSpec) -> list[str]:
    object_descriptors = _spec_object_descriptor_map(document)

    element_by_id = {element.id: element for element in document.graph.elements}
    relation_targets = {relation.target_id for relation in document.graph.relations}
    phrases: list[str] = []

    _append_scene_phrase(phrases, document.graph.seed_prompt)

    for element in document.graph.elements:
        if element.id in relation_targets and element.role != "primary_subject":
            continue
        phrase = _spec_element_phrase(element, object_descriptors.get(element.id, []))
        for relation in [item for item in document.graph.relations if item.source_id == element.id]:
            target = element_by_id.get(relation.target_id)
            target_phrase = relation.target_id
            if target is not None:
                target_phrase = _spec_element_phrase(target, object_descriptors.get(target.id, []))
            relation_raw = _clean_spec_text(relation.relation_raw)
            if relation_raw and target_phrase:
                phrase = f"{phrase} {relation_raw} {target_phrase}".strip()
        _append_scene_phrase(phrases, phrase)

    for action in document.action_lane.actions:
        if action.support_status in {"supported", "inferred"}:
            _append_scene_phrase(phrases, action.movement_raw)

    return phrases


def scene_description(document: PromptDocumentSpec) -> str:
    """Render the seed prompt, graph elements, relations, and actions."""
    return ", ".join(_spec_scene_phrases(document))


def composition_phrase(document: PromptDocumentSpec) -> str:
    """Render camera placement, focus, framing, and setting details."""
    lane = document.cinematography_lane
    parts: list[str] = []
    for field_name in (
        "shot_size",
        "camera_angle",
        "camera_motion",
        "focus_behavior",
        "framing",
        "setting_description",
    ):
        value = getattr(lane, field_name, None)
        if field_name in {"camera_motion", "focus_behavior", "setting_description"}:
            _append_unique(parts, value)
        else:
            _append_unique(parts, value, display=True)
    return ", ".join(parts)


def style_prompt(document: PromptDocumentSpec) -> str:
    """Render visual style, optics, lighting, and color treatment."""
    lane = document.cinematography_lane
    parts: list[str] = []
    for field_name in ("optic_character", "lighting_mood", "color_treatment"):
        _append_unique(parts, getattr(lane, field_name, None), display=True)
    return ", ".join(parts)


def negative_prompt(document: PromptDocumentSpec) -> str:
    """Render guardrails and raw negative phrases as a comma list."""
    parts: list[str] = []
    for guardrail in document.constraint_lane.guardrails:
        _append_unique(parts, guardrail, display=True)
    for phrase in document.constraint_lane.negative_phrases:
        _append_unique(parts, phrase)
    return ", ".join(parts)


def compile_prompt(document: PromptDocumentSpec) -> str:
    """Compile a spec document into the final positive/negative prompt string."""
    positive_parts = [
        scene_description(document),
        composition_phrase(document),
        style_prompt(document),
    ]
    positive = ", ".join(_clean_prompt_fragment(part) for part in positive_parts if _clean_prompt_fragment(part))
    positive = _normalize_positive_prompt_text(positive)
    rendered = "Generate " + positive
    negative = negative_prompt(document)
    if negative:
        rendered = rendered.rstrip(".") + ". Negative prompt: " + negative
    elif not rendered.endswith("."):
        rendered += "."
    return rendered


def _spec_rendering_trace(document: PromptDocumentSpec) -> list[str]:
    traces: list[str] = []
    for relation in document.graph.relations:
        traces.append(f"relation:{relation.id}:{relation.relation_raw}")
    for action in document.action_lane.actions:
        if action.support_status in {"supported", "inferred"}:
            traces.append(f"action:{action.actor_id}:{action.support_status}")
    return traces


def render_prompt_spec(document: PromptDocumentSpec) -> RenderedPrompt:
    """Render an approved spec prompt document with relation/action trace."""
    if not document.verification.approved:
        raise ValueError("cannot render an unapproved PromptDocumentSpec")

    return RenderedPrompt(
        run_id=None,
        prompt_document_id=document.prompt_document_id,
        rendered_prompt=compile_prompt(document),
        rendering_trace=_spec_rendering_trace(document),
    )


EvaluationPriority = Literal[
    "locked_required",
    "locked_context",
    "important",
    "sampled",
    "optional",
    "negative_guard",
    "diagnostic",
]
EvaluationPolicy = Literal[
    "must_match",
    "should_match",
    "nice_to_have",
    "must_not_appear",
    "measure_only",
]


def _spec_priority(importance: str, is_required_role: bool) -> EvaluationPriority:
    if importance == "required" or is_required_role:
        return "locked_required"
    if importance == "ambient":
        return "locked_context"
    if importance == "unresolved":
        return "diagnostic"
    return "optional"


def _spec_evaluation_policy(importance: str) -> EvaluationPolicy:
    if importance == "required":
        return "must_match"
    if importance == "unresolved":
        return "measure_only"
    if importance == "optional":
        return "nice_to_have"
    return "should_match"


def target_manifest_from_prompt_spec(document: PromptDocumentSpec) -> EvaluationTargetManifest:
    rendered = render_prompt_spec(document)
    targets: list[EvaluationTarget] = []
    negative_targets: list[EvaluationTarget] = []

    for element in document.graph.elements:
        importance = str(element.importance)
        required_role = str(element.role) in {"primary_subject", "foreground", "supporting"}
        priority = _spec_priority(importance, required_role)
        targets.append(
            EvaluationTarget(
                target_id=element.id,
                target_kind="element",
                label=element.label,
                priority=priority,
                lhs_policy="sampleable_if_missing" if importance == "unresolved" else "fixed",
                evaluation_policy=_spec_evaluation_policy(importance),
                evidence=element.evidence.text if element.evidence else None,
            )
        )

    for relation in document.graph.relations:
        enum_value = relation.relation_match.enum_value if relation.relation_match else None
        importance = str(relation.importance)
        targets.append(
            EvaluationTarget(
                target_id=relation.id,
                target_kind="relation",
                source_id=relation.source_id,
                target_id_ref=relation.target_id,
                relation_raw=relation.relation_raw,
                enum_value=enum_value,
                priority=_spec_priority(importance, True),
                lhs_policy="fixed",
                evaluation_policy=_spec_evaluation_policy(importance),
                evidence=relation.evidence.text if relation.evidence else None,
            )
        )

    for descriptor in document.object_lane.objects:
        for field_name in ("description", "material", "color", "finish", "condition", "pattern"):
            value = getattr(descriptor, field_name)
            if value is None:
                continue
            is_enum_field = field_name in {"finish", "condition", "pattern"}
            targets.append(
                EvaluationTarget(
                    target_id=f"{descriptor.target_id}.{field_name}",
                    target_kind="object_attribute",
                    value_raw=_display_spec_value(str(value)),
                    enum_value=str(value) if is_enum_field else None,
                    priority="locked_required",
                    lhs_policy="fixed",
                    evaluation_policy="must_match",
                )
            )

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
    ):
        value = getattr(document.cinematography_lane, field_name)
        if value is None:
            continue
        is_enum_field = field_name in {
            "shot_size",
            "camera_angle",
            "optic_character",
            "lighting_mood",
            "color_treatment",
            "framing",
        }
        targets.append(
            EvaluationTarget(
                target_id=f"cinematography.{field_name}",
                target_kind="cinematography",
                value_raw=_display_spec_value(str(value)),
                enum_value=str(value) if is_enum_field else None,
                priority="important",
                lhs_policy="fixed",
                evaluation_policy="should_match",
            )
        )

    for index, guardrail in enumerate(document.constraint_lane.guardrails, start=1):
        negative_targets.append(
            EvaluationTarget(
                target_id=f"constraint.guardrail.{index}",
                target_kind="constraint",
                value_raw=_display_spec_value(str(guardrail)),
                enum_value=str(guardrail),
                priority="negative_guard",
                lhs_policy="fixed",
                evaluation_policy="must_not_appear",
            )
        )
    for index, phrase in enumerate(document.constraint_lane.negative_phrases, start=1):
        negative_targets.append(
            EvaluationTarget(
                target_id=f"constraint.negative_phrase.{index}",
                target_kind="constraint",
                value_raw=str(phrase),
                priority="negative_guard",
                lhs_policy="fixed",
                evaluation_policy="must_not_appear",
            )
        )

    return EvaluationTargetManifest(
        manifest_id="eval_manifest_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        coordinate_id=rendered.coordinate_id,
        rendered_prompt=rendered.rendered_prompt,
        targets=targets,
        negative_targets=negative_targets,
    )
