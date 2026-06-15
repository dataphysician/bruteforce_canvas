"""Static compatibility pair-family rules for the LHS enum router.

All rules are module-level constants consumed by :class:`CompatibilityPrior`
(see ``router.py``).  Each rule is a 4-tuple:

    (axis1_value, axis2_value, severity, reason)

Axis values are stored as plain strings matching the enum value strings in
``bruteforce_canvas.prompt_enums`` so that E2 can map them without importing
those enums at rule-load time.

Rule families are drawn from spec §11.11.
"""

from __future__ import annotations

from enum import StrEnum

from bruteforce_canvas.router import CompatibilitySeverity

__all__ = ["CompatibilitySeverity", "PAIR_FAMILY_RULES"]

# ---------------------------------------------------------------------------
# Shot size × subject / action complexity
# ---------------------------------------------------------------------------

PAIR_FAMILY_RULES: list[tuple[str, str, CompatibilitySeverity, str]] = [
    # shot_size × primary_subject_count
    (
        "wide_shot",
        "single",
        CompatibilitySeverity.HARD_REJECT,
        "wide_shot is too sparse for a single-subject frame; use medium_shot or closer",
    ),
    (
        "establishing_shot",
        "single",
        CompatibilitySeverity.HARD_REJECT,
        "establishing_shot requires multiple subjects or strong environment context",
    ),
    (
        "macro",
        "group",
        CompatibilitySeverity.HARD_REJECT,
        "macro has an impossibly shallow depth of field for group shots",
    ),
    (
        "macro",
        "wide_shot",
        CompatibilitySeverity.HARD_REJECT,
        "macro captures only a tiny patch; wide_shot framing is physically impossible",
    ),
    (
        "close_up",
        "multiple",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "close_up cannot frame multiple distinct subjects with individual clarity",
    ),
    # shot_size × required_relation_count
    (
        "extreme_close_up",
        "multiple",
        CompatibilitySeverity.HARD_REJECT,
        "extreme_close_up has no room to show relations between subjects",
    ),
    # shot_size × action_complexity
    (
        "wide_shot",
        "complex",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "wide_shot spreads complex action across too much area to read clearly",
    ),
    # shot_size × object_scale
    (
        "establishing_shot",
        "tiny",
        CompatibilitySeverity.HARD_REJECT,
        "establishing_shot makes tiny objects invisible in the scene",
    ),
    (
        "macro",
        "large",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "macro captures only a tiny area; a large object cannot be fully shown",
    ),
    # -----------------------------------------------------------------------
    # Lens family × scene scale / accuracy
    # -----------------------------------------------------------------------
    # lens_family × object_scale
    (
        "wide_angle",
        "tiny",
        CompatibilitySeverity.HARD_REJECT,
        "wide_angle lens perspective distorts and swamps tiny objects in the frame",
    ),
    (
        "telephoto_compression",
        "large",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "telephoto_compression flattens large scenes, losing spatial context",
    ),
    # lens_family × required_environment_visibility
    (
        "macro",
        "broad",
        CompatibilitySeverity.HARD_REJECT,
        "macro captures only a tight patch; broad environment visibility is impossible",
    ),
    (
        "fisheye_distorted",
        "broad",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "fisheye_distorted severely restricts the usable visible area for environment context",
    ),
    # lens_family × architecture_line_accuracy
    (
        "telephoto_compression",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "telephoto_compression flattens perspective, destroying architectural line accuracy",
    ),
    (
        "fisheye_distorted",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "fisheye_distorted barrel distortion is incompatible with accurate straight lines",
    ),
    # lens_family × product_accuracy_requirement
    (
        "fisheye_distorted",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "fisheye_distorted warps product proportions beyond acceptable accuracy",
    ),
    (
        "tilt_shift_selective",
        "high",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "tilt_shift_selective selective focus plane does not serve overall product accuracy",
    ),
    # -----------------------------------------------------------------------
    # Camera angle × composition / readability
    # -----------------------------------------------------------------------
    # camera_angle × composition_style
    (
        "dutch_angle",
        "symmetrical",
        CompatibilitySeverity.HARD_REJECT,
        "dutch_angle is inherently asymmetric; it cannot produce a symmetrical composition",
    ),
    (
        "birds_eye",
        "overhead_flat_lay",
        CompatibilitySeverity.BOOST,
        "birds_eye and overhead_flat_lay are the same perspective; pairing is natural",
    ),
    (
        "low_angle",
        "negative_space",
        CompatibilitySeverity.BOOST,
        "low_angle makes sky/background dominant, creating strong negative space above subject",
    ),
    (
        "overhead_flat_lay",
        "centered",
        CompatibilitySeverity.SOFT_DOWNRANK,
        "overhead_flat_lay typically reads better with rule_of_thirds than strict centering",
    ),
    # camera_angle × spatial_relation_requirement
    (
        "worms_eye",
        "foreground_background",
        CompatibilitySeverity.HARD_REJECT,
        "worms_eye extreme upward perspective destroys depth cues needed for foreground/background relations",
    ),
    (
        "birds_eye",
        "foreground_background",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "birds_eye compresses depth; foreground/background relation is ambiguous",
    ),
    # camera_angle × flat_lay_or_tabletop_scene
    (
        "dutch_angle",
        "flat_lay",
        CompatibilitySeverity.HARD_REJECT,
        "dutch_angle cannot produce a usable flat lay or tabletop composition",
    ),
    (
        "worms_eye",
        "flat_lay",
        CompatibilitySeverity.HARD_REJECT,
        "worms_eye is the opposite of the flat lay overhead orientation",
    ),
    # camera_angle × human_pose_readability
    (
        "profile",
        "high",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "profile view limits visible body to one side; full-body pose readability is reduced",
    ),
    (
        "worms_eye",
        "high",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "worms_eye extreme upward angle distorts human proportions severely",
    ),
    (
        "overhead_flat_lay",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "overhead_flat_lay points straight down; human pose is not readable",
    ),
    # -----------------------------------------------------------------------
    # Camera motion × action / readability
    # -----------------------------------------------------------------------
    # camera_motion × action_complexity
    (
        "static",
        "complex",
        CompatibilitySeverity.HARD_REJECT,
        "static camera cannot track or frame complex multi-part action adequately",
    ),
    (
        "dolly",
        "still_life",
        CompatibilitySeverity.HARD_REJECT,
        "dolly motion requires a movable subject; still life is stationary by definition",
    ),
    # camera_motion × still_life_or_product_scene
    (
        "tracking",
        "still_life",
        CompatibilitySeverity.HARD_REJECT,
        "tracking motion presupposes a moving subject; incompatible with still life",
    ),
    # camera_motion × motion_blur_policy
    (
        "tracking",
        "no_blur",
        CompatibilitySeverity.HARD_REJECT,
        "tracking camera motion introduces motion blur; conflicts with no_blur policy",
    ),
    (
        "dolly",
        "no_blur",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "dolly movement at speed introduces blur; requires slow speeds to satisfy no_blur",
    ),
    # camera_motion × required_object_readability
    (
        "crane",
        "high",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "crane motion creates dramatic parallax shifts that can obscure fine object detail",
    ),
    # -----------------------------------------------------------------------
    # Focus behavior × required visibility / interaction
    # -----------------------------------------------------------------------
    # focus_behavior × required_background_visibility
    (
        "shallow_focus",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "shallow_focus intentionally blurs the background; cannot satisfy high background visibility",
    ),
    (
        "selective_focus",
        "high",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "selective_focus puts only one plane in focus; background visibility is partial at best",
    ),
    # focus_behavior × requires_text_readability
    (
        "shallow_focus",
        "yes",
        CompatibilitySeverity.HARD_REJECT,
        "shallow_focus risks blurring text in the background; cannot guarantee text readability",
    ),
    (
        "selective_focus",
        "yes",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "selective_focus may keep only the subject sharp; text in secondary areas may blur",
    ),
    # focus_behavior × small_secondary_object_required
    (
        "shallow_focus",
        "yes",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "small secondary objects frequently fall outside the shallow focal plane",
    ),
    # focus_behavior × multi_subject_interaction
    (
        "shallow_focus",
        "yes",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "multi-subject interaction spans multiple depths; shallow focus isolates only one",
    ),
    # -----------------------------------------------------------------------
    # Lighting mood × material / color / expression requirements
    # -----------------------------------------------------------------------
    # lighting_mood × requires_color_accuracy
    (
        "low_key",
        "high",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "low_key extreme contrast crushes shadow detail, degrading color accuracy",
    ),
    (
        "rim_silhouette",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "rim_silhouette places the subject in shadow, making color information unavailable",
    ),
    (
        "neon_noir",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "neon_noir applies an artificial teal-orange grade that does not represent true scene color",
    ),
    (
        "studio_softbox",
        "high",
        CompatibilitySeverity.BOOST,
        "studio_softbox provides even, controllable illumination ideal for color-accurate capture",
    ),
    # lighting_mood × material_reflectivity
    (
        "chiaroscuro_extreme",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "chiaroscuro_extreme deep shadows obscure specular highlights; reflectivity is lost",
    ),
    (
        "candlelight_intimate",
        "high",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "candlelight_intimate warm flickering light masks subtle material reflectivity",
    ),
    (
        "studio_softbox",
        "high",
        CompatibilitySeverity.BOOST,
        "studio_softbox soft quality accurately renders specular highlights on all surfaces",
    ),
    # lighting_mood × dark_subject_or_dark_garment
    (
        "chiaroscuro_extreme",
        "yes",
        CompatibilitySeverity.HARD_REJECT,
        "chiaroscuro_extreme shadows crush dark garments into near-blackness; they disappear",
    ),
    (
        "high_key_bright",
        "yes",
        CompatibilitySeverity.BOOST,
        "high_key_bright floods the scene with light, keeping dark garments clearly visible",
    ),
    # lighting_mood × facial_expression_required
    (
        "rim_silhouette",
        "yes",
        CompatibilitySeverity.HARD_REJECT,
        "rim_silhouette hides the face in shadow; expressions are not readable",
    ),
    (
        "studio_softbox",
        "yes",
        CompatibilitySeverity.BOOST,
        "studio_softbox soft frontal light renders facial expressions clearly and faithfully",
    ),
    # -----------------------------------------------------------------------
    # Color treatment × lighting mood / explicit colors / skin tone
    # -----------------------------------------------------------------------
    # color_treatment × explicit_color_fields
    (
        "monochrome",
        "yes",
        CompatibilitySeverity.HARD_REJECT,
        "monochrome removes all chromatic information; explicit color fields are violated",
    ),
    (
        "monochromatic_sepia",
        "yes",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "monochromatic_sepia applies a uniform warm brown cast that overrides explicit colors",
    ),
    # color_treatment × brand_color_requirement
    (
        "monochrome",
        "yes",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "monochrome strips brand palette colors; brand recognition is degraded",
    ),
    (
        "monochromatic_sepia",
        "yes",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "monochromatic_sepia distorts brand colors with a non-neutral warm cast",
    ),
    # color_treatment × skin_tone_or_material_accuracy
    (
        "monochrome",
        "yes",
        CompatibilitySeverity.HARD_REJECT,
        "monochrome eliminates hue information entirely; skin tone accuracy is undefined",
    ),
    (
        "monochromatic_sepia",
        "yes",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "monochromatic_sepia shifts all skin tones to warm brown, distorting natural accuracy",
    ),
    # color_treatment × lighting_mood
    (
        "pastel_soft",
        "high_key_bright",
        CompatibilitySeverity.BOOST,
        "pastel_soft and high_key_bright share a light, desaturated aesthetic; pairing is natural",
    ),
    (
        "neon_saturated",
        "neon_noir",
        CompatibilitySeverity.BOOST,
        "neon_saturated and neon_noir share a vivid neon-lit aesthetic; pairing is natural",
    ),
    (
        "natural_color",
        "soft_natural",
        CompatibilitySeverity.BOOST,
        "natural_color faithfully renders soft_natural daylight without color distortion",
    ),
    # -----------------------------------------------------------------------
    # Atmosphere × visibility / depth requirements
    # -----------------------------------------------------------------------
    # atmosphere × required_object_detail
    (
        "mist",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "mist obscures fine object detail through volumetric scattering",
    ),
    (
        "fog",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "fog severely limits visibility range; fine object detail is not reachable",
    ),
    (
        "haze",
        "high",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "haze softly scatters light and reduces contrast; fine detail is degraded",
    ),
    # atmosphere × required_relation_visibility
    (
        "mist",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "mist occludes edges and boundaries needed to perceive relations between objects",
    ),
    (
        "shallow_depth_of_field",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "shallow_depth_of_field blurs non-focal subjects; relations in the background are invisible",
    ),
    # atmosphere × scene_depth_requirement
    (
        "shallow_depth_of_field",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "shallow_depth_of_field intentionally flattens depth; cannot satisfy a depth requirement",
    ),
    (
        "fog",
        "high",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "fog reduces perceived depth through atmospheric perspective flattening",
    ),
    # -----------------------------------------------------------------------
    # Render style × fidelity / readability requirements
    # -----------------------------------------------------------------------
    # render_style × photorealism_requirement
    (
        "tilt_shift",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "tilt_shift produces a miniature effect; incompatible with photorealism",
    ),
    (
        "low_poly",
        "high",
        CompatibilitySeverity.HARD_REJECT,
        "low_poly is intentionally geometric and stylized; incompatible with photorealism",
    ),
    (
        "photorealistic",
        "high",
        CompatibilitySeverity.BOOST,
        "photorealistic is the canonical style for photorealism requirement",
    ),
    # render_style × product_catalog_requirement
    (
        "photorealistic",
        "yes",
        CompatibilitySeverity.BOOST,
        "photorealistic accurately represents product appearance for catalog use",
    ),
    (
        "low_poly",
        "yes",
        CompatibilitySeverity.HARD_REJECT,
        "low_poly abstraction hides product detail; unsuitable for a product catalog",
    ),
    (
        "sketch",
        "yes",
        CompatibilitySeverity.HARD_REJECT,
        "sketch is interpretive and imprecise; does not serve product catalog accuracy",
    ),
    # render_style × documentary_or_news_requirement
    (
        "cel_shaded",
        "yes",
        CompatibilitySeverity.HARD_REJECT,
        "cel_shaded cartoon style violates documentary fidelity requirements",
    ),
    (
        "neon_glow",
        "yes",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "neon_glow is a stylized artistic effect; it undermines documentary credibility",
    ),
    # render_style × text_or_logo_readability
    (
        "watercolor",
        "yes",
        CompatibilitySeverity.HARD_REJECT,
        "watercolor bleeding obscures crisp edges; text and logos are not reliably readable",
    ),
    (
        "sketch",
        "yes",
        CompatibilitySeverity.STRONG_DOWNRANK,
        "sketch hand-drawn quality does not guarantee legible text or logo reproduction",
    ),
    # -----------------------------------------------------------------------
    # Constraint and safety families
    # -----------------------------------------------------------------------
    # every sampled field x explicit negative constraints
    (
        "__any_sampled__",
        "__negative_constraint__",
        CompatibilitySeverity.HARD_REJECT,
        "sampled value conflicts with an explicit negative constraint; negative constraints are absolute",
    ),
    # every style field x safety/compliance constraints
    (
        "__any_style__",
        "__safety_constraint__",
        CompatibilitySeverity.HARD_REJECT,
        "style selection violates an active safety or compliance constraint",
    ),
    # every sampled object/detail field x graph mutation policy
    (
        "__any_sampled__",
        "__graph_mutation__",
        CompatibilitySeverity.HARD_REJECT,
        "sampled object or detail field would require mutating the locked scene graph",
    ),
    # explicit locked field x conflicting sampled value
    (
        "__explicit_locked__",
        "__conflicting_sampled__",
        CompatibilitySeverity.HARD_REJECT,
        "sampled value conflicts with an explicit locked field; user specification takes priority",
    ),
]
