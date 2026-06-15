"""Phase B3 spec-compliance tests for the seven lane/container models.

Source of truth: ``specs/01-bruteforce-canvas_DAG_prompt.md`` sections
4.1-4.5. The 14 tests below cover one happy path and one structural
violation per container model:

    SceneGraphDraft, ObjectLane, ActionLane, CinematographyLane,
    ConstraintLane, PromptDocumentSpec, PromptBundle

The container names are imported from ``bruteforce_canvas.prompt`` per
the B3 task contract; the re-export must resolve to the classes in
``bruteforce_canvas.prompt_models``. Auxiliary models (``Element``,
``RelationDescriptor``, ``ObjectDescriptor``, ``EnumMatch``,
``ActionDescriptor``) are imported from the same location.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bruteforce_canvas.prompt import (
    ActionLane,
    ConstraintLane,
    ObjectLane,
    PromptBundle,
    PromptDocumentSpec,
    SceneGraphDraft,
)
from bruteforce_canvas.prompt_models import (
    ActionDescriptor,
    CinematographyLane,
    Element,
    EnumMatch,
    ObjectDescriptor,
    RelationDescriptor,
)
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
    OpticCharacter,
    Pattern,
    RelationType,
    ShotSize,
)
from bruteforce_canvas.shared import (
    CanonicalStatus,
    SeedPrompt,
    ShortText,
    StrictModel,
)


def _make_element(
    element_id: str = "person_01",
    label: str = "woman",
    entity_type: EntityType = EntityType.PERSON,
    role: ElementRole = ElementRole.PRIMARY_SUBJECT,
    importance: Importance = Importance.REQUIRED,
) -> Element:
    return Element(
        id=element_id,
        entity_type=entity_type,
        label=label,
        role=role,
        importance=importance,
    )


def test_scene_graph_draft_happy_path_minimal() -> None:
    """Spec §394-427: SceneGraphDraft accepts one element and a seed prompt."""
    graph = SceneGraphDraft(
        seed_prompt="a woman stands in a rain-slicked alley",
        elements=[_make_element()],
    )

    assert isinstance(graph, SceneGraphDraft)
    assert graph.seed_prompt == "a woman stands in a rain-slicked alley"
    assert len(graph.elements) == 1
    assert graph.relations == []
    assert graph.elements[0].importance == "required"


def test_scene_graph_draft_rejects_duplicate_element_ids() -> None:
    """Spec §402-403: duplicate element IDs must raise."""
    with pytest.raises(ValidationError) as exc_info:
        SceneGraphDraft(
            seed_prompt="x",
            elements=[
                _make_element(element_id="person_01", label="woman"),
                _make_element(element_id="person_01", label="man", role=ElementRole.FOREGROUND),
            ],
        )
    assert "unique" in str(exc_info.value).lower()


def test_object_lane_happy_path_with_typed_fields() -> None:
    """Spec §458-475: ObjectLane accepts ObjectDescriptor with all typed fields."""
    lane = ObjectLane(
        objects=[
            ObjectDescriptor(
                target_id="coat_01",
                material="wool",
                color="burgundy",
                finish=Finish.MATTE,
                condition=Condition.WORN,
                pattern=Pattern.SOLID,
            )
        ]
    )

    assert len(lane.objects) == 1
    obj = lane.objects[0]
    assert obj.target_id == "coat_01"
    assert obj.material == "wool"
    assert obj.color == "burgundy"
    assert obj.finish == "matte"
    assert obj.condition == "worn"
    assert obj.pattern == "solid"


def test_object_lane_enforces_max_length_32() -> None:
    """Spec §475: ``objects: list[ObjectDescriptor] = Field(max_length=32)``."""
    too_many = [
        ObjectDescriptor(target_id=f"obj_{i:02d}") for i in range(33)
    ]
    with pytest.raises(ValidationError):
        ObjectLane(objects=too_many)


def test_action_lane_happy_path_with_enum_match() -> None:
    """Spec §504-552: ActionLane accepts a single ActionDescriptor with EnumMatch."""
    lane = ActionLane(
        actions=[
            ActionDescriptor(
                actor_id="person_01",
                movement_raw="swordfighting",
                movement_match=EnumMatch(
                    raw="swordfighting",
                    enum_value="combat",
                    confidence=EnumMatchConfidence.CLEAR,
                    reason="distinct combat motion vocabulary",
                ),
                intensity=MotionIntensity.ENERGETIC,
                support_status=ActionSupportStatus.SUPPORTED,
                required_relation_types=[RelationType.HOLDING],
            )
        ]
    )

    assert len(lane.actions) == 1
    action = lane.actions[0]
    assert action.actor_id == "person_01"
    assert action.movement_raw == "swordfighting"
    assert action.movement_match is not None
    assert action.movement_match.enum_value == "combat"
    assert action.intensity == "energetic"
    assert action.support_status == "supported"
    assert action.required_relation_types == ["holding"]


def test_action_lane_enforces_max_length_16() -> None:
    """Spec §552: ``actions: list[ActionDescriptor] = Field(max_length=16)``."""
    too_many = [
        ActionDescriptor(actor_id="person_01", movement_raw=f"motion {i}")
        for i in range(17)
    ]
    with pytest.raises(ValidationError):
        ActionLane(actions=too_many)


def test_cinematography_lane_happy_path_typed_enums() -> None:
    """Spec §731-749: CinematographyLane uses typed enums + residual fields."""
    lane = CinematographyLane(
        shot_size=ShotSize.MEDIUM_SHOT,
        camera_angle=CameraAngle.EYE_LEVEL,
        optic_character=OpticCharacter.NATURAL_35MM,
        camera_motion="locked-off",
        focus_behavior="shallow depth of field",
        lighting_mood=LightingMood.GOLDEN_HOUR,
        color_treatment=ColorTreatment.CINEMATIC_TEAL_ORANGE,
        framing=Framing.RULE_OF_THIRDS,
        setting_description="rain-slicked alley at dusk",
    )

    assert lane.shot_size == "medium_shot"
    assert lane.camera_angle == "eye_level"
    assert lane.optic_character == "natural_35mm"
    assert lane.camera_motion == "locked-off"
    assert lane.focus_behavior == "shallow depth of field"
    assert lane.lighting_mood == "golden_hour"
    assert lane.color_treatment == "cinematic_teal_orange"
    assert lane.framing == "rule_of_thirds"
    assert lane.setting_description == "rain-slicked alley at dusk"


def test_cinematography_lane_rejects_unknown_enum_value() -> None:
    """Spec §731-733: shot_size is a typed ShotSize enum, not free text."""
    with pytest.raises(ValidationError):
        CinematographyLane(shot_size="super_extreme_close_up")


def test_constraint_lane_happy_path_guardrails_and_negatives() -> None:
    """Spec §764-766: ConstraintLane holds guardrails and negative phrases."""
    lane = ConstraintLane(
        guardrails=[Guardrail.NO_TEXT, Guardrail.NO_LOGOS, Guardrail.NO_BLUR],
        negative_phrases=["cartoon", "sketch", "drawing"],
    )

    assert lane.guardrails == ["no_text", "no_logos", "no_blur"]
    assert lane.negative_phrases == ["cartoon", "sketch", "drawing"]


def test_constraint_lane_enforces_guardrails_max_length_16() -> None:
    """Spec §765: ``guardrails: list[Guardrail] = Field(max_length=16)``."""
    with pytest.raises(ValidationError):
        ConstraintLane(
            guardrails=[Guardrail.NO_TEXT] * 17,
            negative_phrases=[],
        )


def test_prompt_document_spec_happy_path_resolves_all_references() -> None:
    """Spec §769-790: PromptDocumentSpec accepts a graph and resolved object/action lanes."""
    graph = SceneGraphDraft(
        seed_prompt="a woman walks",
        elements=[_make_element(element_id="person_01", label="woman")],
    )
    spec = PromptDocumentSpec(
        graph=graph,
        object_lane=ObjectLane(
            objects=[ObjectDescriptor(target_id="person_01", color="burgundy")]
        ),
        action_lane=ActionLane(
            actions=[ActionDescriptor(actor_id="person_01", movement_raw="walks")]
        ),
        cinematography_lane=CinematographyLane(shot_size=ShotSize.MEDIUM_SHOT),
        constraint_lane=ConstraintLane(guardrails=[Guardrail.NO_TEXT]),
    )

    assert spec.graph is graph
    assert spec.object_lane.objects[0].target_id == "person_01"
    assert spec.action_lane.actions[0].actor_id == "person_01"
    assert spec.cinematography_lane.shot_size == "medium_shot"
    assert spec.constraint_lane.guardrails == ["no_text"]


def test_prompt_document_spec_rejects_object_target_not_in_graph() -> None:
    """Spec §780-782: object target_id must resolve to a graph element id."""
    graph = SceneGraphDraft(
        seed_prompt="a woman stands",
        elements=[_make_element(element_id="person_01")],
    )
    with pytest.raises(ValidationError) as exc_info:
        PromptDocumentSpec(
            graph=graph,
            object_lane=ObjectLane(
                objects=[ObjectDescriptor(target_id="unknown_99")]
            ),
        )
    assert "unknown target_id" in str(exc_info.value)


def test_prompt_bundle_happy_path() -> None:
    """Spec §793-798: PromptBundle carries positive/negative + checklist + trace."""
    bundle = PromptBundle(
        positive_prompt="Generate a woman in a burgundy coat in a rain-slicked alley",
        negative_prompt="cartoon, sketch, blurry",
        alignment_checklist=["woman", "coat", "alley"],
        render_trace=["relation:rel_01:holding"],
    )

    assert bundle.positive_prompt.startswith("Generate ")
    assert bundle.negative_prompt == "cartoon, sketch, blurry"
    assert bundle.alignment_checklist == ["woman", "coat", "alley"]
    assert bundle.render_trace == ["relation:rel_01:holding"]
    assert bundle.prompt_improvement_hints == []


def test_prompt_bundle_rejects_missing_positive_prompt() -> None:
    """Spec §794: ``positive_prompt: str`` is required (no default)."""
    with pytest.raises(ValidationError):
        PromptBundle(
            negative_prompt="cartoon",
            alignment_checklist=[],
            render_trace=[],
        )
