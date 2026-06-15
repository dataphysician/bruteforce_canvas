"""Spec §7 per-lane LLM adapter contract tests.

Each adapter gets two tests:
1. Happy path — fake client returns a valid lane dict, adapter returns a
   validated Pydantic lane model and forwards the expected schema name
   and user payload shape to the client.
2. Validation rejection — fake client returns malformed JSON (wrong
   shape or wrong types), adapter must raise a Pydantic ``ValidationError``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bruteforce_canvas.llm_adapters import (
    LLMActionLaneAdapter,
    LLMCinematographyLaneAdapter,
    LLMConstraintLaneAdapter,
    LLMObjectLaneAdapter,
)
from bruteforce_canvas.prompt import (
    ActionLane,
    ConstraintLane,
    ObjectLane,
)
from bruteforce_canvas.prompt_enums import (
    ActionSupportStatus,
    ElementRole,
    EntityType,
    Finish,
    Guardrail,
    Importance,
    MotionIntensity,
    ShotSize,
)
from bruteforce_canvas.prompt_models import (
    CinematographyLane,
    Element,
    SceneGraphDraft,
)


class FakeJsonClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate_json(self, *, system: str, user: dict, schema_name: str) -> dict:
        self.calls.append({"system": system, "user": user, "schema_name": schema_name})
        return self.responses.pop(0)


def _graph() -> SceneGraphDraft:
    return SceneGraphDraft(
        seed_prompt="a woman in a charcoal wool coat on a city street",
        elements=[
            Element(
                id="person_01",
                label="woman",
                entity_type=EntityType.PERSON,
                role=ElementRole.PRIMARY_SUBJECT,
                importance=Importance.REQUIRED,
            ),
            Element(
                id="garment_01",
                label="coat",
                entity_type=EntityType.TEXTILE,
                role=ElementRole.FOREGROUND,
                importance=Importance.REQUIRED,
            ),
            Element(
                id="location_01",
                label="city street",
                entity_type=EntityType.LOCATION,
                role=ElementRole.BACKGROUND,
                importance=Importance.REQUIRED,
            ),
        ],
    )


def test_object_lane_adapter_happy_path_returns_validated_lane() -> None:
    payload: dict = {
        "objects": [
            {
                "target_id": "garment_01",
                "material": "wool",
                "color": "charcoal",
                "finish": "matte",
                "pattern": "solid",
            },
            {
                "target_id": "location_01",
                "description": "rain-slick pavement",
                "finish": "glossy",
            },
        ]
    }
    client = FakeJsonClient([payload])
    adapter = LLMObjectLaneAdapter(client)

    result = adapter.expand(_graph())

    assert isinstance(result, ObjectLane)
    assert len(result.objects) == 2
    assert result.objects[0].target_id == "garment_01"
    assert result.objects[0].material == "wool"
    assert result.objects[1].finish == Finish.GLOSSY
    call = client.calls[0]
    assert call["schema_name"] == "ObjectLane"
    assert call["system"].startswith(
        "Expand appearance descriptors for existing graph elements."
    )
    assert "graph" in call["user"]
    assert call["user"]["graph"]["seed_prompt"].startswith("a woman in a charcoal wool coat")


def test_object_lane_adapter_rejects_malformed_payload() -> None:
    bad_payload: dict = {
        "objects": [
            {
                "target_id": "garment_01",
                "finish": "not_a_real_finish_value",
            }
        ]
    }
    client = FakeJsonClient([bad_payload])
    adapter = LLMObjectLaneAdapter(client)

    with pytest.raises(ValidationError):
        adapter.expand(_graph())


# ---------------------------------------------------------------------------
# LLMActionLaneAdapter
# ---------------------------------------------------------------------------


def test_action_lane_adapter_happy_path_returns_validated_lane() -> None:
    payload: dict = {
        "actions": [
            {
                "actor_id": "person_01",
                "movement_raw": "posing",
                "intensity": MotionIntensity.SUBTLE.value,
                "support_status": ActionSupportStatus.SUPPORTED.value,
                "description": "composed stillness suitable for a film still",
            }
        ]
    }
    client = FakeJsonClient([payload])
    adapter = LLMActionLaneAdapter(client)

    result = adapter.expand(_graph())

    assert isinstance(result, ActionLane)
    assert len(result.actions) == 1
    assert result.actions[0].movement_raw == "posing"
    assert result.actions[0].actor_id == "person_01"
    call = client.calls[0]
    assert call["schema_name"] == "ActionLane"
    assert "Expand temporal" in call["system"]
    assert call["user"]["graph"]["seed_prompt"].startswith("a woman in a charcoal wool coat")


def test_action_lane_adapter_rejects_malformed_payload() -> None:
    bad_payload: dict = {
        "actions": [
            {
                "actor_id": "person_01",
                "movement_raw": "posing",
                "support_status": "not_a_real_status",
            }
        ]
    }
    client = FakeJsonClient([bad_payload])
    adapter = LLMActionLaneAdapter(client)

    with pytest.raises(ValidationError):
        adapter.expand(_graph())


# ---------------------------------------------------------------------------
# LLMCinematographyLaneAdapter
# ---------------------------------------------------------------------------


def test_cinematography_lane_adapter_happy_path_returns_validated_lane() -> None:
    payload: dict = {
        "shot_size": "medium_shot",
        "camera_angle": "eye_level",
        "optic_character": "portrait_50mm",
        "lighting_mood": "practical_lighting",
        "color_treatment": "filmic_contrast",
        "framing": "layered_depth",
        "setting_description": "rain-slick city street with soft practical lights",
    }
    client = FakeJsonClient([payload])
    adapter = LLMCinematographyLaneAdapter(client)
    graph = _graph()
    raw_prompt = "a woman in a charcoal wool coat on a city street"

    result = adapter.expand(graph, raw_prompt)

    assert isinstance(result, CinematographyLane)
    assert result.shot_size == ShotSize.MEDIUM_SHOT
    assert result.lighting_mood == "practical_lighting"
    call = client.calls[0]
    assert call["schema_name"] == "CinematographyLane"
    assert call["user"]["raw_user_prompt"] == raw_prompt
    assert call["user"]["validated_scene_graph"] == graph.model_dump()
    assert "Extract only cinematography" in call["system"]


def test_cinematography_lane_adapter_rejects_invalid_enum() -> None:
    bad_payload: dict = {
        "shot_size": "super_telephoto",
        "lighting_mood": "practical_lighting",
    }
    client = FakeJsonClient([bad_payload])
    adapter = LLMCinematographyLaneAdapter(client)

    with pytest.raises(ValidationError):
        adapter.expand(_graph(), "a woman in a charcoal wool coat on a city street")


# ---------------------------------------------------------------------------
# LLMConstraintLaneAdapter
# ---------------------------------------------------------------------------


def test_constraint_lane_adapter_happy_path_returns_validated_lane() -> None:
    payload: dict = {
        "guardrails": [Guardrail.NO_TEXT.value, Guardrail.NO_LOGOS.value],
        "negative_phrases": ["duplicate subject"],
    }
    client = FakeJsonClient([payload])
    adapter = LLMConstraintLaneAdapter(client)
    graph = _graph()
    raw_prompt = "a woman in a charcoal wool coat on a city street, no text"

    result = adapter.expand(graph, raw_prompt)

    assert isinstance(result, ConstraintLane)
    assert Guardrail.NO_TEXT in result.guardrails
    assert Guardrail.NO_LOGOS in result.guardrails
    assert result.negative_phrases == ["duplicate subject"]
    call = client.calls[0]
    assert call["schema_name"] == "ConstraintLane"
    assert call["user"]["raw_user_prompt"] == raw_prompt
    assert call["user"]["validated_scene_graph"] == graph.model_dump()
    assert "Extract exclusions" in call["system"]


def test_constraint_lane_adapter_rejects_unknown_guardrail() -> None:
    bad_payload: dict = {
        "guardrails": ["no_such_guardrail"],
        "negative_phrases": [],
    }
    client = FakeJsonClient([bad_payload])
    adapter = LLMConstraintLaneAdapter(client)

    with pytest.raises(ValidationError):
        adapter.expand(_graph(), "a woman in a charcoal wool coat on a city street")


# ---------------------------------------------------------------------------
# Protocol-shape spot check
# ---------------------------------------------------------------------------


def test_lane_adapters_expose_system_instruction_class_constant() -> None:
    """Spec §7: each adapter exposes a ``system_instruction`` class constant.

    Plan QA: LLMObjectLaneAdapter.system_instruction[:80] must start with
    'Expand appearance descriptors for existing graph elements.'.
    """
    assert LLMObjectLaneAdapter.system_instruction[:80].startswith(
        "Expand appearance descriptors for existing graph elements."
    )
    assert isinstance(LLMObjectLaneAdapter.system_instruction, str)
    assert isinstance(LLMActionLaneAdapter.system_instruction, str)
    assert isinstance(LLMCinematographyLaneAdapter.system_instruction, str)
    assert isinstance(LLMConstraintLaneAdapter.system_instruction, str)
    # Sanity: cinematography instruction must reference cinematography output
    assert "CinematographyLane" in LLMCinematographyLaneAdapter.system_instruction
    assert "ActionLane" in LLMActionLaneAdapter.system_instruction
    assert "ObjectLane" in LLMObjectLaneAdapter.system_instruction
    assert "ConstraintLane" in LLMConstraintLaneAdapter.system_instruction
