from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from bruteforce_canvas.prompt_models import PromptDocumentSpec
from bruteforce_canvas.shared import CanonicalStatus, StrictModel


class LockState(StrEnum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    DIAGNOSTIC_HOLD = "diagnostic_hold"


class LockConfigEntry(StrictModel):
    field_path: str
    raw_value: str | None = None
    enum_value: str | None = None
    canonical_status: str
    priority: Literal["locked_required", "locked_context", "important", "sampled", "optional", "negative_guard", "diagnostic"]
    lhs_policy: Literal["fixed", "sampleable", "sampleable_if_missing", "blocked"]
    render_policy: Literal["use_raw_or_safe_raw_preserving_phrase", "use_sampled_enum_phrase", "measure_only"]
    evaluation_policy: Literal["must_match", "should_match", "nice_to_have", "must_not_appear", "measure_only"]
    learning_policy: Literal["track_locked_reliability", "track_sampled_arm", "none"]
    lock_state: LockState
    lock_source: str
    user_adjustable: bool = True


class LockConfiguration(StrictModel):
    prompt_document_id: str
    entries: list[LockConfigEntry]

    @property
    def by_field_path(self) -> dict[str, LockConfigEntry]:
        return {entry.field_path: entry for entry in self.entries}


class LockOverride(StrictModel):
    field_path: str
    lock_state: LockState
    override_source: str


SPEC_PRESENTATION_AXES = {
    "cinematography.shot_size": "shot_size",
    "cinematography.camera_angle": "camera_angle",
    "cinematography.lens": "optic_character",
    "cinematography.focus": "focus_behavior",
    "cinematography.lighting_mood": "lighting_mood",
    "cinematography.color_treatment": "color_treatment",
    "cinematography.composition": "framing",
    "cinematography.style": "setting_description",
}


def _document_id(document: PromptDocumentSpec) -> str:
    return document.prompt_document_id


def _display_value(value: object) -> str:
    return str(value).replace("_", " ")


def _spec_object_entries(document: PromptDocumentSpec) -> list[LockConfigEntry]:
    entries: list[LockConfigEntry] = []
    enum_fields = {"finish", "condition", "pattern"}
    for descriptor in document.object_lane.objects:
        for field_name in ("description", "material", "color", "finish", "condition", "pattern"):
            value = getattr(descriptor, field_name)
            if value is None:
                continue
            raw_value = _display_value(value)
            entries.append(
                LockConfigEntry(
                    field_path=f"object.{field_name}.{descriptor.target_id}",
                    raw_value=raw_value,
                    enum_value=str(value) if field_name in enum_fields else None,
                    canonical_status=CanonicalStatus.UNMATCHED_RAW_ONLY,
                    priority="locked_required",
                    lhs_policy="fixed",
                    render_policy="use_raw_or_safe_raw_preserving_phrase",
                    evaluation_policy="must_match",
                    learning_policy="track_locked_reliability",
                    lock_state=LockState.LOCKED,
                    lock_source="prompt_document_spec",
                )
            )
    return entries


def _spec_relation_entries(document: PromptDocumentSpec) -> list[LockConfigEntry]:
    entries: list[LockConfigEntry] = []
    for relation in document.graph.relations:
        canonical = document.canonical_metadata.get(f"relation.{relation.id}")
        relation_match = relation.relation_match
        enum_value = None
        canonical_status = CanonicalStatus.UNMATCHED_RAW_ONLY
        lock_source = "prompt_document_spec"
        if canonical is not None:
            enum_value = canonical.enum_value
            canonical_status = canonical.status
            lock_source = "canonicalizer"
        elif relation_match is not None:
            enum_value = str(relation_match.enum_value) if relation_match.enum_value is not None else None
            canonical_status = relation_match.status
            lock_source = "relation_match"
        entries.append(
            LockConfigEntry(
                field_path=f"relation.{relation.id}",
                raw_value=relation.relation_raw,
                enum_value=enum_value,
                canonical_status=canonical_status,
                priority="locked_required",
                lhs_policy="fixed",
                render_policy="use_raw_or_safe_raw_preserving_phrase",
                evaluation_policy="must_match",
                learning_policy="track_locked_reliability",
                lock_state=LockState.LOCKED,
                lock_source=lock_source,
            )
        )
    return entries


def _spec_constraint_entries(document: PromptDocumentSpec) -> list[LockConfigEntry]:
    entries: list[LockConfigEntry] = []
    for guardrail in document.constraint_lane.guardrails:
        value = str(guardrail)
        entries.append(
            LockConfigEntry(
                field_path=f"constraint.{value}",
                raw_value=_display_value(value),
                enum_value=value,
                canonical_status=CanonicalStatus.UNMATCHED_RAW_ONLY,
                priority="negative_guard",
                lhs_policy="fixed",
                render_policy="use_raw_or_safe_raw_preserving_phrase",
                evaluation_policy="must_not_appear",
                learning_policy="track_locked_reliability",
                lock_state=LockState.LOCKED,
                lock_source="prompt_document_spec",
                user_adjustable=False,
            )
        )
    for index, phrase in enumerate(document.constraint_lane.negative_phrases, start=1):
        entries.append(
            LockConfigEntry(
                field_path=f"constraint.negative_phrase.{index}",
                raw_value=str(phrase),
                canonical_status=CanonicalStatus.UNMATCHED_RAW_ONLY,
                priority="negative_guard",
                lhs_policy="fixed",
                render_policy="use_raw_or_safe_raw_preserving_phrase",
                evaluation_policy="must_not_appear",
                learning_policy="track_locked_reliability",
                lock_state=LockState.LOCKED,
                lock_source="prompt_document_spec",
                user_adjustable=False,
            )
        )
    return entries


def _spec_presentation_entries(document: PromptDocumentSpec) -> list[LockConfigEntry]:
    entries: list[LockConfigEntry] = []
    for field_path, attribute in SPEC_PRESENTATION_AXES.items():
        canonical = document.canonical_metadata.get(field_path)
        raw_value = getattr(document.cinematography_lane, attribute)
        if canonical is not None:
            lock_state = (
                LockState.DIAGNOSTIC_HOLD
                if canonical.status == CanonicalStatus.MATCHED_DIAGNOSTIC_HOLD
                else LockState.LOCKED
            )
            entries.append(
                LockConfigEntry(
                    field_path=field_path,
                    raw_value=canonical.raw_value,
                    enum_value=canonical.enum_value,
                    canonical_status=canonical.status,
                    priority="locked_required",
                    lhs_policy="fixed",
                    render_policy="use_raw_or_safe_raw_preserving_phrase",
                    evaluation_policy="must_match",
                    learning_policy="track_locked_reliability",
                    lock_state=lock_state,
                    lock_source="llm_canonicalizer",
                )
            )
        elif raw_value:
            entries.append(
                LockConfigEntry(
                    field_path=field_path,
                    raw_value=_display_value(raw_value),
                    canonical_status=CanonicalStatus.UNMATCHED_RAW_ONLY,
                    priority="locked_context",
                    lhs_policy="fixed",
                    render_policy="use_raw_or_safe_raw_preserving_phrase",
                    evaluation_policy="should_match",
                    learning_policy="track_locked_reliability",
                    lock_state=LockState.LOCKED,
                    lock_source="prompt_document_spec",
                )
            )
        else:
            entries.append(
                LockConfigEntry(
                    field_path=field_path,
                    canonical_status=CanonicalStatus.UNMATCHED_RAW_ONLY,
                    priority="sampled",
                    lhs_policy="sampleable_if_missing",
                    render_policy="use_sampled_enum_phrase",
                    evaluation_policy="should_match",
                    learning_policy="track_sampled_arm",
                    lock_state=LockState.UNLOCKED,
                    lock_source="default_missing_presentation_axis",
                )
            )
    return entries


def build_default_lock_config(document: PromptDocumentSpec) -> LockConfiguration:
    return LockConfiguration(
        prompt_document_id=_document_id(document),
        entries=[
            *_spec_object_entries(document),
            *_spec_relation_entries(document),
            *_spec_constraint_entries(document),
            *_spec_presentation_entries(document),
        ],
    )


def apply_lock_overrides(default: LockConfiguration, overrides: list[LockOverride]) -> LockConfiguration:
    override_by_path = {override.field_path: override for override in overrides}
    effective_entries: list[LockConfigEntry] = []
    for entry in default.entries:
        override = override_by_path.get(entry.field_path)
        if override is None:
            effective_entries.append(entry)
            continue
        effective_entries.append(
            entry.model_copy(update={"lock_state": override.lock_state, "lock_source": override.override_source})
        )
    return LockConfiguration(prompt_document_id=default.prompt_document_id, entries=effective_entries)
