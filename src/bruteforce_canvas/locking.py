from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from bruteforce_canvas.prompt import PromptDocument
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


PRESENTATION_AXES = {
    "cinematography.shot_size": "shot_size_raw",
    "cinematography.camera_angle": "camera_angle_raw",
    "cinematography.lens": "lens_raw",
    "cinematography.focus": "focus_raw",
    "cinematography.lighting_mood": "lighting_raw",
    "cinematography.color_treatment": "color_treatment_raw",
    "cinematography.composition": "composition_raw",
    "cinematography.style": "style_raw",
}

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


def _entry_for_canonical_object(field_path: str, raw_value: str, enum_value: str | None, status: CanonicalStatus) -> LockConfigEntry:
    return LockConfigEntry(
        field_path=field_path,
        raw_value=raw_value,
        enum_value=enum_value,
        canonical_status=status,
        priority="locked_required",
        lhs_policy="fixed",
        render_policy="use_raw_or_safe_raw_preserving_phrase",
        evaluation_policy="must_match",
        learning_policy="track_locked_reliability",
        lock_state=LockState.LOCKED,
        lock_source="llm_canonicalizer",
    )


def _document_id(document: PromptDocument | PromptDocumentSpec) -> str:
    return getattr(document, "prompt_document_id", "doc_001")


def _display_value(value: object) -> str:
    return str(value).replace("_", " ")


def _canonical_status_value(status: str | CanonicalStatus) -> CanonicalStatus:
    try:
        return CanonicalStatus(status)
    except ValueError:
        return CanonicalStatus.UNMATCHED_RAW_ONLY


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


def build_default_lock_config(document: PromptDocument | PromptDocumentSpec) -> LockConfiguration:
    if isinstance(document, PromptDocumentSpec):
        return LockConfiguration(
            prompt_document_id=_document_id(document),
            entries=[
                *_spec_object_entries(document),
                *_spec_constraint_entries(document),
                *_spec_presentation_entries(document),
            ],
        )

    entries: list[LockConfigEntry] = []

    for descriptor in document.objects:
        field_path = f"object.{descriptor.field_name}.{descriptor.element_id}"
        if descriptor.canonical is not None:
            entries.append(
                _entry_for_canonical_object(
                    field_path,
                    descriptor.raw_value,
                    descriptor.canonical.enum_value,
                    _canonical_status_value(descriptor.canonical.status),
                )
            )
        else:
            entries.append(
                LockConfigEntry(
                    field_path=field_path,
                    raw_value=descriptor.raw_value,
                    canonical_status=CanonicalStatus.UNMATCHED_RAW_ONLY,
                    priority="locked_required",
                    lhs_policy="fixed",
                    render_policy="use_raw_or_safe_raw_preserving_phrase",
                    evaluation_policy="must_match",
                    learning_policy="track_locked_reliability",
                    lock_state=LockState.LOCKED,
                    lock_source="prompt_document",
                )
            )

    for constraint in document.constraints:
        entries.append(
            LockConfigEntry(
                field_path=constraint.constraint_id,
                raw_value=constraint.value_raw,
                canonical_status=CanonicalStatus.UNMATCHED_RAW_ONLY,
                priority="negative_guard" if constraint.negative else "important",
                lhs_policy="fixed",
                render_policy="use_raw_or_safe_raw_preserving_phrase",
                evaluation_policy="must_not_appear" if constraint.negative else "should_match",
                learning_policy="track_locked_reliability",
                lock_state=LockState.LOCKED,
                lock_source="prompt_document",
                user_adjustable=False,
            )
        )

    for field_path, attribute in PRESENTATION_AXES.items():
        canonical = document.canonical_metadata.get(field_path)
        raw_value = getattr(document.cinematography, attribute)
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
                    raw_value=raw_value,
                    canonical_status=CanonicalStatus.UNMATCHED_RAW_ONLY,
                    priority="locked_context",
                    lhs_policy="fixed",
                    render_policy="use_raw_or_safe_raw_preserving_phrase",
                    evaluation_policy="should_match",
                    learning_policy="track_locked_reliability",
                    lock_state=LockState.LOCKED,
                    lock_source="prompt_document",
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

    return LockConfiguration(prompt_document_id=_document_id(document), entries=entries)


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
