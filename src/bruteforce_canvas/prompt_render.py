"""Spec-compliant rendering helpers for the prompt domain (Phase B6).

This module is the discrete implementation of spec
``specs/01-bruteforce-canvas_DAG_prompt.md`` section 8 (rendering order).
It exposes pure, deterministic string assembly helpers that the orchestrator
(:func:`compile_prompt`) composes to produce a :class:`PromptBundle`.

Helpers are deliberately decoupled from any LLM call. The orchestrator
chains deterministic assembly, not generation.

The seven exported helpers expected by the spec compliance checker are:

* :func:`object_phrase`           - element + descriptor -> ``"burgundy polished leather handbag"``
* :func:`relation_type`           - relation -> enum value (or raw linkage)
* :func:`relation_label`          - relation -> lowercase human label
* :func:`relation_phrase`         - source, relation, target -> ``"source relation target"``
* :func:`action_phrase`           - action -> ``"actor movement_raw target"`` (target preserved)
* :func:`compile_positive_prompt` - bundle -> positive prompt string
* :func:`render_cinematography`   - lane -> cinematography phrase
* :func:`compile_negative_prompt` - bundle -> negative prompt string
* :func:`compile_prompt`          - bundle -> :class:`PromptBundle`

All helpers accept either a pydantic model or a plain dict with matching
keys. They are pure functions: no I/O, no LLM, no globals.
"""

from __future__ import annotations

from typing import Any, Mapping


_OBJECT_DESCRIPTOR_FIELDS: tuple[str, ...] = (
    "description",
    "color",
    "finish",
    "material",
)


def _clean_text(value: Any) -> str:
    """Return ``value`` as a stripped string. ``None`` becomes empty."""
    if value is None:
        return ""
    return str(value).strip()


def _descriptor_value(descriptor: Any, field_name: str) -> str:
    """Read ``field_name`` from a pydantic model or dict, stripping text."""
    if descriptor is None:
        return ""
    if isinstance(descriptor, Mapping):
        raw = descriptor.get(field_name)
    else:
        raw = getattr(descriptor, field_name, None)
    return _clean_text(raw)


def _element_label(element: Any) -> str:
    if element is None:
        return ""
    if isinstance(element, str):
        return _clean_text(element)
    if isinstance(element, Mapping):
        return _clean_text(element.get("label"))
    return _clean_text(getattr(element, "label", None))


def _relation_id(relation: Any) -> str:
    """Read the relation id from a pydantic model or dict."""
    if relation is None:
        return ""
    if isinstance(relation, Mapping):
        return _clean_text(relation.get("id"))
    return _clean_text(getattr(relation, "id", None))


def _relation_raw(relation: Any) -> str:
    """Read the raw linkage phrase from a relation."""
    if relation is None:
        return ""
    if isinstance(relation, Mapping):
        return _clean_text(relation.get("relation_raw"))
    return _clean_text(getattr(relation, "relation_raw", None))


def _relation_match_enum(relation: Any) -> str:
    """Read the matched enum value from ``relation.relation_match`` if present."""
    if relation is None:
        return ""
    if isinstance(relation, Mapping):
        match = relation.get("relation_match")
    else:
        match = getattr(relation, "relation_match", None)
    if match is None:
        return ""
    if isinstance(match, Mapping):
        return _clean_text(match.get("enum_value"))
    return _clean_text(getattr(match, "enum_value", None))


def object_phrase(element: Any, descriptor: Any) -> str:
    """Return ``"<description> <color> <finish> <material> <label>"`` form.

    Descriptor fields are read in the spec-locked order
    ``description, color, finish, material``. ``None`` and empty strings are
    skipped so the phrase is dense. The element label is appended last.

    Accepts pydantic models or plain dicts. Empty inputs return an empty
    string so callers can filter trivially.
    """
    parts: list[str] = []
    for field_name in _OBJECT_DESCRIPTOR_FIELDS:
        text = _descriptor_value(descriptor, field_name)
        if text:
            parts.append(text)
    label = _element_label(element)
    if label:
        parts.append(label)
    return " ".join(parts)


def relation_type(relation: Any) -> str:
    """Return the relation's matched enum value when present, else raw linkage.

    Per spec section 8.2 the locked enum match takes precedence; we fall
    back to the raw linkage phrase when the small-LLM enum matcher has not
    produced a value (status: ``UNMATCHED_RAW_ONLY`` or absent).
    """
    enum_value = _relation_match_enum(relation)
    if enum_value:
        return enum_value
    return _relation_raw(relation)


def relation_label(relation: Any) -> str:
    """Return a human-friendly relation label.

    Uses the matched enum value when present (lower-cased with underscores
    converted to spaces). Otherwise returns the raw linkage phrase
    lower-cased. The lowercase transform is intentional so the phrase
    reads naturally in the middle of a sentence.
    """
    enum_value = _relation_match_enum(relation)
    if enum_value:
        return enum_value.replace("_", " ").lower()
    return _relation_raw(relation).lower()


def relation_phrase(source: Any, relation: Any, target: Any) -> str:
    """Return ``"<source_label> <relation_label> <target_label>"`` phrase.

    Empty components are skipped so partial inputs do not produce leading
    or trailing whitespace. ``source`` and ``target`` may be elements
    (pydantic or dict) or pre-extracted label strings.
    """
    source_text = source if isinstance(source, str) else _element_label(source)
    target_text = target if isinstance(target, str) else _element_label(target)
    label = relation_label(relation)
    return " ".join(part for part in (source_text, label, target_text) if part)


def _action_actor_label(action: Any) -> str:
    if action is None:
        return ""
    if isinstance(action, Mapping):
        actor_id = _clean_text(action.get("actor_id"))
    else:
        actor_id = _clean_text(getattr(action, "actor_id", None))
    return actor_id


def _action_movement_raw(action: Any) -> str:
    if action is None:
        return ""
    if isinstance(action, Mapping):
        return _clean_text(action.get("movement_raw"))
    return _clean_text(getattr(action, "movement_raw", None))


def _action_target_id(action: Any) -> str:
    if action is None:
        return ""
    if isinstance(action, Mapping):
        target_id = action.get("target_id")
    else:
        target_id = getattr(action, "target_id", None)
    return _clean_text(target_id)


def action_phrase(action: Any) -> str:
    """Return ``"actor movement_raw target"`` phrase, preserving the target.

    Per spec 8.2, target-bearing actions must keep the target phrase
    visible. The phrase is assembled in spec order and the target is
    preserved as a separate token so the action reads as
    ``"person walking toward glass door"`` rather than collapsing to
    ``"walking motion"``.
    """
    actor = _action_actor_label(action)
    movement = _action_movement_raw(action)
    target = _action_target_id(action)
    return " ".join(part for part in (actor, movement, target) if part)


_CINEMATOGRAPHY_FIELDS: tuple[tuple[str, str], ...] = (
    ("shot_size", "shot"),
    ("camera_angle", ""),
    ("optic_character", ""),
    ("camera_motion", ""),
    ("focus_behavior", ""),
    ("lighting_mood", "lighting"),
    ("color_treatment", ""),
    ("framing", ""),
    ("setting_description", ""),
)


def _lane_value(lane: Any, field_name: str) -> str:
    if lane is None:
        return ""
    if isinstance(lane, Mapping):
        return _clean_text(lane.get(field_name))
    return _clean_text(getattr(lane, field_name, None))


def _display_value(value: str) -> str:
    """Convert enum-ish values to a displayable form.

    Replaces underscores with spaces and lowercases. The transform is
    uniform so ``EYE_LEVEL`` and ``eye_level`` both render as
    ``"eye level"``.
    """
    return value.replace("_", " ").lower()


def _with_optional_suffix(value: str, suffix: str) -> str:
    """Append ``suffix`` to ``value`` unless it is already present."""
    if not suffix:
        return value
    if suffix in value:
        return value
    return f"{value} {suffix}"


def render_cinematography(lane: Any) -> str:
    """Render a cinematography lane as a comma-separated phrase.

    Fields are read in spec order. Enum values are displayed with
    underscores converted to spaces; for shot size and lighting mood a
    context suffix (``"shot"`` / ``"lighting"``) is added only when the
    value does not already contain that word, which prevents the
    ``"lighting lighting"`` duplication the spec calls out.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for field_name, suffix in _CINEMATOGRAPHY_FIELDS:
        raw = _lane_value(lane, field_name)
        if not raw:
            continue
        text = _display_value(raw)
        text = _with_optional_suffix(text, suffix)
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return ", ".join(parts)


def _bundle_subject(bundle: Any) -> str:
    if bundle is None:
        return ""
    if isinstance(bundle, Mapping):
        graph = bundle.get("graph")
    else:
        graph = getattr(bundle, "graph", None)
    if graph is None:
        return ""
    if isinstance(graph, Mapping):
        return _clean_text(graph.get("seed_prompt"))
    return _clean_text(getattr(graph, "seed_prompt", None))


def _bundle_objects(bundle: Any) -> list[str]:
    if bundle is None:
        return []
    if isinstance(bundle, Mapping):
        lane = bundle.get("object_lane")
    else:
        lane = getattr(bundle, "object_lane", None)
    if lane is None:
        return []
    if isinstance(lane, Mapping):
        objects = lane.get("objects") or []
    else:
        objects = getattr(lane, "objects", None) or []
    element_label_by_id = _element_label_map(bundle)
    phrases: list[str] = []
    for descriptor in objects:
        if isinstance(descriptor, Mapping):
            target_id = _clean_text(descriptor.get("target_id"))
        else:
            target_id = _clean_text(getattr(descriptor, "target_id", None))
        label = element_label_by_id.get(target_id, "")
        phrase = object_phrase(element=label or None, descriptor=descriptor)
        if phrase:
            phrases.append(phrase)
    return phrases


def _element_label_map(bundle: Any) -> dict[str, str]:
    if bundle is None:
        return {}
    if isinstance(bundle, Mapping):
        graph = bundle.get("graph")
    else:
        graph = getattr(bundle, "graph", None)
    if graph is None:
        return {}
    if isinstance(graph, Mapping):
        elements = graph.get("elements") or []
    else:
        elements = getattr(graph, "elements", None) or []
    result: dict[str, str] = {}
    for element in elements:
        if isinstance(element, Mapping):
            eid = _clean_text(element.get("id"))
            label = _clean_text(element.get("label"))
        else:
            eid = _clean_text(getattr(element, "id", None))
            label = _clean_text(getattr(element, "label", None))
        if eid:
            result[eid] = label
    return result


def _bundle_spatial(bundle: Any) -> str:
    if bundle is None:
        return ""
    if isinstance(bundle, Mapping):
        graph = bundle.get("graph")
    else:
        graph = getattr(bundle, "graph", None)
    if graph is None:
        return ""
    if isinstance(graph, Mapping):
        elements = graph.get("elements") or []
        relations = graph.get("relations") or []
    else:
        elements = getattr(graph, "elements", None) or []
        relations = getattr(graph, "relations", None) or []
    element_by_id = {}
    for element in elements:
        if isinstance(element, Mapping):
            eid = _clean_text(element.get("id"))
            label = _clean_text(element.get("label"))
        else:
            eid = _clean_text(getattr(element, "id", None))
            label = _clean_text(getattr(element, "label", None))
        if eid:
            element_by_id[eid] = label

    phrases: list[str] = []
    for relation in relations:
        if isinstance(relation, Mapping):
            source_id = _clean_text(relation.get("source_id"))
            target_id = _clean_text(relation.get("target_id"))
        else:
            source_id = _clean_text(getattr(relation, "source_id", None))
            target_id = _clean_text(getattr(relation, "target_id", None))
        source_label = element_by_id.get(source_id, source_id)
        target_label = element_by_id.get(target_id, target_id)
        phrase = relation_phrase(source_label, relation, target_label)
        if phrase:
            phrases.append(phrase)
    return ", ".join(phrases)


def _bundle_actions(bundle: Any) -> str:
    if bundle is None:
        return ""
    if isinstance(bundle, Mapping):
        lane = bundle.get("action_lane")
    else:
        lane = getattr(bundle, "action_lane", None)
    if lane is None:
        return ""
    if isinstance(lane, Mapping):
        actions = lane.get("actions") or []
    else:
        actions = getattr(lane, "actions", None) or []
    phrases: list[str] = []
    for action in actions:
        if isinstance(action, Mapping):
            support = action.get("support_status", "supported")
        else:
            support = getattr(action, "support_status", "supported")
        if str(support) not in {"supported", "inferred"}:
            continue
        phrase = action_phrase(action)
        if phrase:
            phrases.append(phrase)
    return ", ".join(phrases)


def _bundle_setting(bundle: Any) -> str:
    if bundle is None:
        return ""
    if isinstance(bundle, Mapping):
        lane = bundle.get("cinematography_lane")
    else:
        lane = getattr(bundle, "cinematography_lane", None)
    if lane is None:
        return ""
    if isinstance(lane, Mapping):
        return _clean_text(lane.get("setting_description"))
    return _clean_text(getattr(lane, "setting_description", None))


def _bundle_cinematography(bundle: Any) -> str:
    if bundle is None:
        return ""
    if isinstance(bundle, Mapping):
        lane = bundle.get("cinematography_lane")
    else:
        lane = getattr(bundle, "cinematography_lane", None)
    return render_cinematography(lane)


def _bundle_style(bundle: Any) -> str:
    if bundle is None:
        return ""
    if isinstance(bundle, Mapping):
        lane = bundle.get("cinematography_lane")
    else:
        lane = getattr(bundle, "cinematography_lane", None)
    if lane is None:
        return ""
    parts: list[str] = []
    for field_name in ("optic_character", "color_treatment"):
        if isinstance(lane, Mapping):
            value = _clean_text(lane.get(field_name))
        else:
            value = _clean_text(getattr(lane, field_name, None))
        if value:
            parts.append(_display_value(value))
    return ", ".join(parts)


def compile_positive_prompt(bundle: Any) -> str:
    """Compile a positive prompt from a ``PromptDocumentSpec``-like bundle.

    Order is the spec-locked rendering order: subject, objects, spatial,
    action, setting, cinematography, style. Empty segments are skipped
    so the output is dense.
    """
    segments = [
        _bundle_subject(bundle),
        ", ".join(_bundle_objects(bundle)),
        _bundle_spatial(bundle),
        _bundle_actions(bundle),
        _bundle_setting(bundle),
        _bundle_cinematography(bundle),
        _bundle_style(bundle),
    ]
    return ", ".join(segment for segment in segments if segment)


def _bundle_negative(bundle: Any) -> tuple[list[str], list[str]]:
    if bundle is None:
        return [], []
    if isinstance(bundle, Mapping):
        lane = bundle.get("constraint_lane")
    else:
        lane = getattr(bundle, "constraint_lane", None)
    if lane is None:
        return [], []
    if isinstance(lane, Mapping):
        guardrails = lane.get("guardrails") or []
        phrases = lane.get("negative_phrases") or []
    else:
        guardrails = getattr(lane, "guardrails", None) or []
        phrases = getattr(lane, "negative_phrases", None) or []
    guardrail_text: list[str] = []
    for guardrail in guardrails:
        text = _display_value(_clean_text(guardrail)) if not isinstance(guardrail, str) else _display_value(guardrail)
        if text:
            guardrail_text.append(text)
    phrase_text = [_clean_text(phrase) for phrase in phrases]
    phrase_text = [text for text in phrase_text if text]
    return guardrail_text, phrase_text


def compile_negative_prompt(bundle: Any) -> str:
    """Compile a negative prompt from a bundle's guardrails and phrases."""
    guardrails, phrases = _bundle_negative(bundle)
    parts: list[str] = []
    for item in guardrails:
        if item and item not in parts:
            parts.append(item)
    for item in phrases:
        if item and item not in parts:
            parts.append(item)
    return ", ".join(parts)


def _trace(bundle: Any) -> list[str]:
    traces: list[str] = []
    if bundle is None:
        return traces
    if isinstance(bundle, Mapping):
        graph = bundle.get("graph")
    else:
        graph = getattr(bundle, "graph", None)
    if graph is None:
        return traces
    if isinstance(graph, Mapping):
        relations = graph.get("relations") or []
        actions_lane = bundle.get("action_lane") if isinstance(bundle, Mapping) else None
    else:
        relations = getattr(graph, "relations", None) or []
        actions_lane = getattr(bundle, "action_lane", None) if not isinstance(bundle, Mapping) else None
    for relation in relations:
        rid = _relation_id(relation)
        raw = _relation_raw(relation)
        if rid:
            traces.append(f"relation:{rid}:{raw}")
    if actions_lane is not None:
        if isinstance(actions_lane, Mapping):
            actions = actions_lane.get("actions") or []
        else:
            actions = getattr(actions_lane, "actions", None) or []
        for action in actions:
            if isinstance(action, Mapping):
                actor = _clean_text(action.get("actor_id"))
                support = action.get("support_status", "supported")
            else:
                actor = _clean_text(getattr(action, "actor_id", None))
                support = getattr(action, "support_status", "supported")
            if str(support) in {"supported", "inferred"} and actor:
                traces.append(f"action:{actor}:{support}")
    return traces


def compile_prompt(bundle: Any) -> Any:
    """Compile a positive/negative prompt pair and trace into a :class:`PromptBundle`.

    The orchestrator composes :func:`compile_positive_prompt`,
    :func:`compile_negative_prompt`, and :func:`_trace` into a
    :class:`PromptBundle` so callers receive a single immutable artifact
    that carries the inputs needed by downstream evaluators.

    The :class:`PromptBundle` import is deferred to break the
    ``prompt`` <-> ``prompt_models`` cycle that pre-exists Phase B6; see
    ``bruteforce_canvas.prompt`` line 391 for the matching pattern.
    """
    from bruteforce_canvas.prompt_models import PromptBundle

    positive = compile_positive_prompt(bundle)
    negative = compile_negative_prompt(bundle)
    return PromptBundle(
        positive_prompt=positive,
        negative_prompt=negative,
        alignment_checklist=[],
        render_trace=_trace(bundle),
    )


__all__ = [
    "action_phrase",
    "compile_negative_prompt",
    "compile_positive_prompt",
    "compile_prompt",
    "object_phrase",
    "relation_label",
    "relation_phrase",
    "relation_type",
    "render_cinematography",
]
