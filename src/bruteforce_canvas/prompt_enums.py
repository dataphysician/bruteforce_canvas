"""Spec-locked enumeration values for the prompt domain.

This module consolidates the 18 lane and graph enumerations defined in
``specs/01-bruteforce-canvas_DAG_prompt.md`` (sections covering graph,
object, action, cinematography, and constraint lanes).
All enums use :class:`enum.StrEnum` (Python 3.11+) and are pure value sets.

Pre-existing enums intentionally NOT defined here:

* ``EvidenceCategory`` — lives in :mod:`bruteforce_canvas.prompt`
* ``CanonicalStatus`` and ``CandidateLifecycle`` — live in :mod:`bruteforce_canvas.shared`
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "ActionSupportStatus",
    "CameraAngle",
    "ColorTreatment",
    "Condition",
    "ElementRole",
    "EnumMatchConfidence",
    "EntityType",
    "Finish",
    "Framing",
    "Guardrail",
    "Importance",
    "LightingMood",
    "MotionIntensity",
    "MovementType",
    "OpticCharacter",
    "Pattern",
    "RelationType",
    "ShotSize",
]


class EntityType(StrEnum):
    PERSON = "person"
    ANIMAL = "animal"
    TEXTILE = "textile"
    ACCESSORY = "accessory"
    TOOL = "tool"
    PRODUCT = "product"
    VEHICLE = "vehicle"
    ARCHITECTURE = "architecture"
    LOCATION = "location"
    ENVIRONMENT = "environment"
    LIGHT_SOURCE = "light_source"
    ABSTRACT_VISUAL = "abstract_visual"
    UNKNOWN_SLOT = "unknown_slot"
    SURFACE = "surface"
    CONTAINER = "container"
    FURNITURE = "furniture"


class ElementRole(StrEnum):
    PRIMARY_SUBJECT = "primary_subject"
    FOREGROUND = "foreground"
    SUPPORTING = "supporting"
    BACKGROUND = "background"
    CONTEXT = "context"


class RelationType(StrEnum):
    WEARING = "wearing"
    HOLDING = "holding"
    CARRYING = "carrying"
    USING = "using"
    ATTACHED_TO = "attached_to"
    PART_OF = "part_of"
    INSIDE = "inside"
    ON_TOP_OF = "on_top_of"
    UNDER = "under"
    OVER = "over"
    COVERING = "covering"
    NEXT_TO = "next_to"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    LOOKING_AT = "looking_at"
    ILLUMINATING = "illuminating"
    PARKED_ON = "parked_on"


class Importance(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    AMBIENT = "ambient"
    UNRESOLVED = "unresolved"


class Finish(StrEnum):
    MATTE = "matte"
    SATIN = "satin"
    GLOSSY = "glossy"
    POLISHED = "polished"
    WEATHERED = "weathered"
    NATURAL = "natural"


class Condition(StrEnum):
    PRISTINE = "pristine"
    NEW = "new"
    USED = "used"
    WORN = "worn"
    AGED = "aged"
    DISTRESSED = "distressed"


class Pattern(StrEnum):
    SOLID = "solid"
    GEOMETRIC = "geometric"
    STRIPED = "striped"
    FLORAL = "floral"
    TEXTURED = "textured"
    NONE = "none"


class MovementType(StrEnum):
    POSTURE_STANCE = "posture_stance"
    LOCOMOTION = "locomotion"
    MANUAL_INTERACTION = "manual_interaction"
    OBJECT_MANIPULATION = "object_manipulation"
    GESTURAL = "gestural"
    GAZE_ATTENTION = "gaze_attention"
    FACIAL_EXPRESSION = "facial_expression"
    PERFORMANCE = "performance"
    COMBAT = "combat"
    BODY_ORIENTATION = "body_orientation"


class MotionIntensity(StrEnum):
    STILL = "still"
    SUBTLE = "subtle"
    MODERATE = "moderate"
    ENERGETIC = "energetic"


class EnumMatchConfidence(StrEnum):
    CLEAR = "clear"
    UNCLEAR = "unclear"


class ActionSupportStatus(StrEnum):
    SUPPORTED = "supported"
    INFERRED = "inferred"
    UNRESOLVED = "unresolved"
    INDETERMINATE = "indeterminate"


class ShotSize(StrEnum):
    EXTREME_CLOSE_UP = "extreme_close_up"
    BIG_CLOSE_UP = "big_close_up"
    CLOSE_UP = "close_up"
    MEDIUM_CLOSE_UP = "medium_close_up"
    MEDIUM_CLOSE_SHOT = "medium_close_shot"
    MEDIUM_SHOT = "medium_shot"
    MEDIUM_LONG_SHOT = "medium_long_shot"
    FULL_BODY = "full_body"
    WIDE_SHOT = "wide_shot"
    ESTABLISHING_SHOT = "establishing_shot"
    TWO_SHOT = "two_shot"
    THREE_SHOT = "three_shot"
    GROUP_SHOT = "group_shot"
    OVER_THE_SHOULDER_SHOT = "over_the_shoulder_shot"
    POINT_OF_VIEW_SHOT = "point_of_view_shot"
    INSERT_SHOT = "insert_shot"


class CameraAngle(StrEnum):
    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    DUTCH_ANGLE = "dutch_angle"
    OVER_THE_SHOULDER = "over_the_shoulder"
    BIRDS_EYE = "birds_eye"
    WORMS_EYE = "worms_eye"
    CANTED = "canted"
    PROFILE = "profile"
    THREE_QUARTER = "three_quarter"
    OVERHEAD_FLAT_LAY = "overhead_flat_lay"


class OpticCharacter(StrEnum):
    NATURAL_35MM = "natural_35mm"
    PORTRAIT_50MM = "portrait_50mm"
    TELEPHOTO_COMPRESSION = "telephoto_compression"
    WIDE_ANGLE = "wide_angle"
    MACRO = "macro"
    SHALLOW_FOCUS = "shallow_focus"
    DEEP_FOCUS = "deep_focus"
    ANAMORPHIC_CINEMATIC = "anamorphic_cinematic"
    VINTAGE_SOFT = "vintage_soft"
    TILT_SHIFT_SELECTIVE = "tilt_shift_selective"
    FISHEYE_DISTORTED = "fisheye_distorted"
    DREAM_GLOW = "dream_glow"


class LightingMood(StrEnum):
    SOFT_NATURAL = "soft_natural"
    GOLDEN_HOUR = "golden_hour"
    LOW_KEY = "low_key"
    HIGH_KEY = "high_key"
    NEON_NOIR = "neon_noir"
    STUDIO_SOFTBOX = "studio_softbox"
    PRACTICAL_LIGHTING = "practical_lighting"
    BLUE_HOUR_TWILIGHT = "blue_hour_twilight"
    TUNGSTEN_INTERIOR = "tungsten_interior"
    OVERCAST_SOFT = "overcast_soft"
    HIGH_KEY_BRIGHT = "high_key_bright"
    CHIAROSCURO_EXTREME = "chiaroscuro_extreme"
    NEON_NIGHT = "neon_night"
    RIM_SILHOUETTE = "rim_silhouette"
    CANDLELIGHT_INTIMATE = "candlelight_intimate"


class ColorTreatment(StrEnum):
    NATURAL_COLOR = "natural_color"
    FILMIC_CONTRAST = "filmic_contrast"
    MUTED_PALETTE = "muted_palette"
    RICH_SATURATION = "rich_saturation"
    MONOCHROME = "monochrome"
    CINEMATIC_TEAL_ORANGE = "cinematic_teal_orange"
    CROSS_PROCESSED = "cross_processed"
    BLEACH_BYPASS = "bleach_bypass"
    PASTEL_SOFT = "pastel_soft"
    NEON_SATURATED = "neon_saturated"
    EARTHY_ORGANIC = "earthy_organic"
    MONOCHROMATIC_SEPIA = "monochromatic_sepia"
    NOCTURNAL_BLUE = "nocturnal_blue"


class Framing(StrEnum):
    CENTERED = "centered"
    RULE_OF_THIRDS = "rule_of_thirds"
    SYMMETRICAL = "symmetrical"
    NEGATIVE_SPACE = "negative_space"
    LAYERED_DEPTH = "layered_depth"
    LEADING_LINES = "leading_lines"
    FRAME_WITHIN_FRAME = "frame_within_frame"
    DIAGONAL = "diagonal"
    ASYMMETRICAL_BALANCE = "asymmetrical_balance"
    S_CURVE = "s_curve"
    GOLDEN_RATIO = "golden_ratio"
    OFF_CENTER = "off_center"


class Guardrail(StrEnum):
    NO_EXTRA_PEOPLE = "no_extra_people"
    NO_TEXT = "no_text"
    NO_LOGOS = "no_logos"
    NO_DISTORTED_HANDS = "no_distorted_hands"
    NO_EXTRA_LIMBS = "no_extra_limbs"
    NO_BLUR = "no_blur"
    NO_OVEREXPOSURE = "no_overexposure"
    NO_UNDEREXPOSURE = "no_underexposure"
