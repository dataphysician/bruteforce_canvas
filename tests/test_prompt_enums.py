"""Tests verifying that ``prompt_enums`` matches the spec exactly.

Each enum is locked by an exact member count and the full set of values.
If the spec changes, the test is expected to fail loudly so the diff is
intentional and reviewable.
"""

from __future__ import annotations

from enum import StrEnum

import pytest

from bruteforce_canvas import prompt_enums
from bruteforce_canvas.prompt_enums import (
    ActionSupportStatus,
    CameraAngle,
    ColorTreatment,
    Condition,
    ElementRole,
    EnumMatchConfidence,
    EntityType,
    Finish,
    Framing,
    Guardrail,
    Importance,
    LightingMood,
    MotionIntensity,
    MovementType,
    OpticCharacter,
    Pattern,
    RelationType,
    ShotSize,
)


def test_module_exports_contain_all_eighteen_enums() -> None:
    expected = {
        "EntityType",
        "ElementRole",
        "RelationType",
        "Importance",
        "Finish",
        "Condition",
        "Pattern",
        "MovementType",
        "MotionIntensity",
        "EnumMatchConfidence",
        "ActionSupportStatus",
        "ShotSize",
        "CameraAngle",
        "OpticCharacter",
        "LightingMood",
        "ColorTreatment",
        "Framing",
        "Guardrail",
    }
    assert set(prompt_enums.__all__) == expected
    assert len(prompt_enums.__all__) == 18


def test_all_enums_subclass_strenum() -> None:
    enum_names = prompt_enums.__all__
    for name in enum_names:
        enum_cls = getattr(prompt_enums, name)
        assert issubclass(enum_cls, StrEnum), f"{name} must subclass StrEnum"


def test_entity_type_member_set() -> None:
    assert {member.value for member in EntityType} == {
        "person",
        "animal",
        "textile",
        "accessory",
        "tool",
        "product",
        "vehicle",
        "architecture",
        "location",
        "environment",
        "light_source",
        "abstract_visual",
        "unknown_slot",
        "surface",
        "container",
        "furniture",
    }
    assert len(EntityType) == 16


def test_element_role_member_set() -> None:
    assert {member.value for member in ElementRole} == {
        "primary_subject",
        "foreground",
        "supporting",
        "background",
        "context",
    }
    assert len(ElementRole) == 5


def test_relation_type_member_set() -> None:
    assert {member.value for member in RelationType} == {
        "wearing",
        "holding",
        "carrying",
        "using",
        "attached_to",
        "part_of",
        "inside",
        "on_top_of",
        "under",
        "over",
        "covering",
        "next_to",
        "in_front_of",
        "behind",
        "looking_at",
        "illuminating",
        "parked_on",
    }
    assert len(RelationType) == 17


def test_importance_member_set() -> None:
    assert {member.value for member in Importance} == {
        "required",
        "optional",
        "ambient",
        "unresolved",
    }
    assert len(Importance) == 4


def test_finish_member_set() -> None:
    assert {member.value for member in Finish} == {
        "matte",
        "satin",
        "glossy",
        "polished",
        "weathered",
        "natural",
    }
    assert len(Finish) == 6


def test_condition_member_set() -> None:
    assert {member.value for member in Condition} == {
        "pristine",
        "new",
        "used",
        "worn",
        "aged",
        "distressed",
    }
    assert len(Condition) == 6


def test_pattern_member_set() -> None:
    assert {member.value for member in Pattern} == {
        "solid",
        "geometric",
        "striped",
        "floral",
        "textured",
        "none",
    }
    assert len(Pattern) == 6


def test_movement_type_member_set() -> None:
    assert {member.value for member in MovementType} == {
        "posture_stance",
        "locomotion",
        "manual_interaction",
        "object_manipulation",
        "gestural",
        "gaze_attention",
        "facial_expression",
        "performance",
        "combat",
        "body_orientation",
    }
    assert len(MovementType) == 10


def test_motion_intensity_member_set() -> None:
    assert {member.value for member in MotionIntensity} == {
        "still",
        "subtle",
        "moderate",
        "energetic",
    }
    assert len(MotionIntensity) == 4


def test_enum_match_confidence_member_set() -> None:
    assert {member.value for member in EnumMatchConfidence} == {
        "clear",
        "unclear",
    }
    assert len(EnumMatchConfidence) == 2


def test_action_support_status_member_set() -> None:
    assert {member.value for member in ActionSupportStatus} == {
        "supported",
        "inferred",
        "unresolved",
        "indeterminate",
    }
    assert len(ActionSupportStatus) == 4


def test_shot_size_member_set() -> None:
    assert {member.value for member in ShotSize} == {
        "extreme_close_up",
        "big_close_up",
        "close_up",
        "medium_close_up",
        "medium_close_shot",
        "medium_shot",
        "medium_long_shot",
        "full_body",
        "wide_shot",
        "establishing_shot",
        "two_shot",
        "three_shot",
        "group_shot",
        "over_the_shoulder_shot",
        "point_of_view_shot",
        "insert_shot",
    }
    assert len(ShotSize) == 16


def test_camera_angle_member_set() -> None:
    assert {member.value for member in CameraAngle} == {
        "eye_level",
        "low_angle",
        "high_angle",
        "dutch_angle",
        "over_the_shoulder",
        "birds_eye",
        "worms_eye",
        "canted",
        "profile",
        "three_quarter",
        "overhead_flat_lay",
    }
    assert len(CameraAngle) == 11


def test_optic_character_member_set() -> None:
    assert {member.value for member in OpticCharacter} == {
        "natural_35mm",
        "portrait_50mm",
        "telephoto_compression",
        "wide_angle",
        "macro",
        "shallow_focus",
        "deep_focus",
        "anamorphic_cinematic",
        "vintage_soft",
        "tilt_shift_selective",
        "fisheye_distorted",
        "dream_glow",
    }
    assert len(OpticCharacter) == 12


def test_lighting_mood_member_set() -> None:
    assert {member.value for member in LightingMood} == {
        "soft_natural",
        "golden_hour",
        "low_key",
        "high_key",
        "neon_noir",
        "studio_softbox",
        "practical_lighting",
        "blue_hour_twilight",
        "tungsten_interior",
        "overcast_soft",
        "high_key_bright",
        "chiaroscuro_extreme",
        "neon_night",
        "rim_silhouette",
        "candlelight_intimate",
    }
    assert len(LightingMood) == 15


def test_color_treatment_member_set() -> None:
    assert {member.value for member in ColorTreatment} == {
        "natural_color",
        "filmic_contrast",
        "muted_palette",
        "rich_saturation",
        "monochrome",
        "cinematic_teal_orange",
        "cross_processed",
        "bleach_bypass",
        "pastel_soft",
        "neon_saturated",
        "earthy_organic",
        "monochromatic_sepia",
        "nocturnal_blue",
    }
    assert len(ColorTreatment) == 13


def test_framing_member_set() -> None:
    assert {member.value for member in Framing} == {
        "centered",
        "rule_of_thirds",
        "symmetrical",
        "negative_space",
        "layered_depth",
        "leading_lines",
        "frame_within_frame",
        "diagonal",
        "asymmetrical_balance",
        "s_curve",
        "golden_ratio",
        "off_center",
    }
    assert len(Framing) == 12


def test_guardrail_member_set() -> None:
    assert {member.value for member in Guardrail} == {
        "no_extra_people",
        "no_text",
        "no_logos",
        "no_distorted_hands",
        "no_extra_limbs",
        "no_blur",
        "no_overexposure",
        "no_underexposure",
    }
    assert len(Guardrail) == 8


@pytest.mark.parametrize(
    "enum_cls",
    [
        EntityType,
        ElementRole,
        RelationType,
        Importance,
        Finish,
        Condition,
        Pattern,
        MovementType,
        MotionIntensity,
        EnumMatchConfidence,
        ActionSupportStatus,
        ShotSize,
        CameraAngle,
        OpticCharacter,
        LightingMood,
        ColorTreatment,
        Framing,
        Guardrail,
    ],
)
def test_enum_value_is_its_str(enum_cls: type[StrEnum]) -> None:
    for member in enum_cls:
        assert member == member.value
        assert isinstance(member, str)
