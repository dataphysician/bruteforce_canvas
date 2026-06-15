"""Spec section 8 rendering helper tests (Phase B6).

Each helper is exercised with both a pydantic model and a plain dict to
guarantee the dict-acceptance contract the spec requires. Where the spec
pins an exact phrase (object_phrase assembly, action_phrase target
preservation, cinematography "no duplicate lighting" rule) the assertion
uses an equality check.
"""

from __future__ import annotations

from typing import Any

import pytest

import bruteforce_canvas.prompt
from bruteforce_canvas.prompt import (
    CinematographyLane as PromptCinematographyLane,
)
from bruteforce_canvas.prompt_enums import (
    ActionSupportStatus,
    CameraAngle,
    ColorTreatment,
    ElementRole,
    EntityType,
    Importance,
    LightingMood,
    MotionIntensity,
    RelationType,
    ShotSize,
)
from bruteforce_canvas.shared import CanonicalStatus
from bruteforce_canvas.prompt_models import (
    ActionDescriptor,
    CinematographyLane,
    Element,
    ObjectDescriptor,
)
from bruteforce_canvas.prompt_render import (
    action_phrase,
    compile_negative_prompt,
    compile_positive_prompt,
    compile_prompt,
    object_phrase,
    relation_label,
    relation_phrase,
    relation_type,
    render_cinematography,
)


def _element(label: str, element_id: str = "object_01") -> dict[str, Any]:
    return {
        "id": element_id,
        "entity_type": EntityType.PRODUCT,
        "label": label,
        "role": ElementRole.PRIMARY_SUBJECT,
        "importance": Importance.REQUIRED,
    }


def _relation(
    *,
    relation_id: str,
    source_id: str,
    target_id: str,
    relation_raw: str,
    enum_value: RelationType | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": relation_id,
        "source_id": source_id,
        "target_id": target_id,
        "relation_raw": relation_raw,
        "importance": Importance.REQUIRED,
    }
    if enum_value is not None:
        payload["relation_match"] = {
            "raw": relation_raw,
            "enum_value": enum_value,
            "status": CanonicalStatus.MATCHED_ACTIVE,
            "confidence": "clear",
            "reason": "test match",
        }
    return payload


def test_object_phrase_assembles_descriptor_fields_in_spec_order() -> None:
    descriptor = {
        "description": "structured",
        "color": "burgundy",
        "finish": "polished",
        "material": "leather",
    }
    element = _element("handbag")
    assert object_phrase(element, descriptor) == "structured burgundy polished leather handbag"


def test_object_phrase_skips_none_and_empty_fields() -> None:
    descriptor = {
        "description": "weathered",
        "color": None,
        "finish": "",
        "material": "canvas",
    }
    element = _element("tote", element_id="object_02")
    assert object_phrase(element, descriptor) == "weathered canvas tote"


def test_object_phrase_accepts_pydantic_descriptor() -> None:
    descriptor = ObjectDescriptor(
        target_id="object_03",
        description="quilted",
        color="navy",
        material="suede",
    )
    element = Element(
        id="object_03",
        entity_type=EntityType.ACCESSORY,
        label="crossbody bag",
        role=ElementRole.PRIMARY_SUBJECT,
        importance=Importance.REQUIRED,
    )
    assert object_phrase(element, descriptor) == "quilted navy suede crossbody bag"


def test_object_phrase_returns_label_only_when_descriptor_is_empty() -> None:
    assert object_phrase(_element("lamp"), {}) == "lamp"


def test_relation_type_prefers_enum_value_over_raw() -> None:
    relation = _relation(
        relation_id="rel_001",
        source_id="elem_a",
        target_id="elem_b",
        relation_raw="resting on",
        enum_value=RelationType.ON_TOP_OF,
    )
    assert relation_type(relation) == "on_top_of"


def test_relation_type_falls_back_to_raw_when_enum_missing() -> None:
    relation = _relation(
        relation_id="rel_002",
        source_id="elem_a",
        target_id="elem_b",
        relation_raw="tucked behind",
    )
    assert relation_type(relation) == "tucked behind"


def test_relation_label_is_lowercase_human_readable() -> None:
    relation = _relation(
        relation_id="rel_003",
        source_id="elem_a",
        target_id="elem_b",
        relation_raw="RESTING ON",
        enum_value=RelationType.ON_TOP_OF,
    )
    assert relation_label(relation) == "on top of"


def test_relation_phrase_assembles_source_relation_target() -> None:
    relation = _relation(
        relation_id="rel_004",
        source_id="elem_a",
        target_id="elem_b",
        relation_raw="resting on",
        enum_value=RelationType.ON_TOP_OF,
    )
    assert relation_phrase("handbag", relation, "shelf") == "handbag on top of shelf"


def test_action_phrase_preserves_target_phrase() -> None:
    action = {
        "actor_id": "person_001",
        "movement_raw": "walking toward",
        "target_id": "glass_door",
    }
    assert action_phrase(action) == "person_001 walking toward glass_door"


def test_action_phrase_accepts_pydantic_action() -> None:
    action = ActionDescriptor(
        actor_id="person_02",
        movement_raw="standing beside",
        target_id="window_02",
        support_status=ActionSupportStatus.SUPPORTED,
    )
    assert action_phrase(action) == "person_02 standing beside window_02"


def test_action_phrase_omits_target_when_none() -> None:
    action = {
        "actor_id": "person_003",
        "movement_raw": "pausing",
        "target_id": None,
    }
    assert action_phrase(action) == "person_003 pausing"


def test_render_cinematography_uses_spec_field_order() -> None:
    lane = CinematographyLane(
        shot_size=ShotSize.MEDIUM_SHOT,
        camera_angle=CameraAngle.EYE_LEVEL,
        focus_behavior="shallow DOF",
        lighting_mood=LightingMood.PRACTICAL_LIGHTING,
        color_treatment=ColorTreatment.FILMIC_CONTRAST,
    )
    rendered = render_cinematography(lane)
    parts = rendered.split(", ")
    assert parts[0] == "medium shot"
    assert "eye level" in parts
    assert "shallow dof" in parts
    assert "practical lighting" in parts
    assert "filmic contrast" in parts
    assert "lighting lighting" not in rendered


def test_render_cinematography_avoids_duplicate_lighting_word() -> None:
    lane = CinematographyLane(lighting_mood=LightingMood.PRACTICAL_LIGHTING)
    rendered = render_cinematography(lane)
    assert rendered == "practical lighting"
    assert "lighting lighting" not in rendered


def test_render_cinematography_accepts_dict_input() -> None:
    lane = {
        "shot_size": "close_up",
        "camera_angle": "low_angle",
        "lighting_mood": "low_key",
    }
    rendered = render_cinematography(lane)
    assert "close up" in rendered
    assert "low angle" in rendered
    assert "low key" in rendered


def test_render_cinematography_dedupes_repeated_fragments() -> None:
    lane = {
        "shot_size": "medium_shot",
        "lighting_mood": "practical_lighting",
        "framing": "centered",
    }
    rendered = render_cinematography(lane)
    occurrences = rendered.count("medium shot")
    assert occurrences == 1
    occurrences = rendered.count("practical lighting")
    assert occurrences == 1


def test_compile_positive_prompt_follows_spec_section_order() -> None:
    bundle = _build_minimal_bundle(
        seed_prompt="Studio still life",
        subject_label="handbag",
        descriptor={
            "description": "structured",
            "color": "burgundy",
            "material": "leather",
        },
        relation_raw="resting on",
        relation_enum=RelationType.ON_TOP_OF,
        movement_raw="leaning against",
        target_id="elem_target",
        target_label="shelf",
        cinematography=CinematographyLane(
            shot_size=ShotSize.MEDIUM_SHOT,
            camera_angle=CameraAngle.EYE_LEVEL,
            lighting_mood=LightingMood.PRACTICAL_LIGHTING,
        ),
    )
    rendered = compile_positive_prompt(bundle)
    subject_index = rendered.index("Studio still life")
    objects_index = rendered.index("structured burgundy leather handbag")
    spatial_index = rendered.index("handbag on top of shelf")
    action_index = rendered.index("person_001 leaning against elem_target")
    cinematography_index = rendered.index("medium shot")
    assert subject_index < objects_index < spatial_index < action_index < cinematography_index


def test_compile_positive_prompt_skips_empty_segments() -> None:
    bundle = _build_minimal_bundle(
        seed_prompt="Quiet interior",
        subject_label="chair",
        descriptor={"material": "walnut"},
    )
    rendered = compile_positive_prompt(bundle)
    assert "Quiet interior" in rendered
    assert "chair" in rendered
    assert "walnut" in rendered
    assert ", ," not in rendered
    assert rendered.startswith("Quiet interior")


def test_compile_negative_prompt_renders_guardrails_and_phrases() -> None:
    bundle = _build_minimal_bundle(
        seed_prompt="",
        subject_label="",
        descriptor={},
    )
    bundle["constraint_lane"] = {
        "guardrails": ["no_extra_people", "no_text"],
        "negative_phrases": ["blurry output", "low contrast"],
    }
    rendered = compile_negative_prompt(bundle)
    assert "no extra people" in rendered
    assert "no text" in rendered
    assert "blurry output" in rendered
    assert "low contrast" in rendered


def test_compile_prompt_returns_prompt_bundle_with_trace() -> None:
    bundle = _build_minimal_bundle(
        seed_prompt="Generate",
        subject_label="vase",
        descriptor={"color": "ceramic"},
        relation_raw="sitting on",
        relation_enum=RelationType.ON_TOP_OF,
    )
    result = compile_prompt(bundle)
    assert result.positive_prompt
    assert isinstance(result.negative_prompt, str)
    assert any(entry.startswith("relation:rel_a_b") for entry in result.render_trace)


def _build_minimal_bundle(
    *,
    seed_prompt: str,
    subject_label: str,
    descriptor: dict[str, Any],
    relation_raw: str | None = None,
    relation_enum: RelationType | None = None,
    movement_raw: str | None = None,
    target_id: str | None = None,
    target_label: str | None = None,
    cinematography: CinematographyLane | dict[str, Any] | None = None,
) -> dict[str, Any]:
    elements = [_element(subject_label or "subject", element_id="elem_a")]
    if target_label is not None or relation_raw is not None:
        elements.append(_element(target_label or "target", element_id="elem_target"))
    relations: list[dict[str, Any]] = []
    if relation_raw is not None:
        relations.append(
            _relation(
                relation_id="rel_a_b",
                source_id="elem_a",
                target_id="elem_target",
                relation_raw=relation_raw,
                enum_value=relation_enum,
            )
        )
    actions: list[dict[str, Any]] = []
    if movement_raw is not None:
        action: dict[str, Any] = {
            "actor_id": "person_001",
            "movement_raw": movement_raw,
            "support_status": "supported",
        }
        if target_id is not None:
            action["target_id"] = target_id
        actions.append(action)
    object_lane: dict[str, Any] = {"objects": []}
    if descriptor:
        object_descriptor = {"target_id": "elem_a", **descriptor}
        object_lane["objects"] = [object_descriptor]
    bundle: dict[str, Any] = {
        "graph": {
            "seed_prompt": seed_prompt,
            "elements": elements,
            "relations": relations,
        },
        "object_lane": object_lane,
        "action_lane": {"actions": actions},
        "cinematography_lane": (
            cinematography.model_dump()
            if isinstance(cinematography, CinematographyLane)
            else (cinematography or {})
        ),
        "constraint_lane": {"guardrails": [], "negative_phrases": []},
    }
    return bundle
