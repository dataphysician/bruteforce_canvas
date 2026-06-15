from __future__ import annotations

import math

from bruteforce_canvas.evaluation import (
    CoordinateEvaluationAggregate,
    LearningUpdateSignal,
)
from bruteforce_canvas.learning import (
    DEFAULT_CONTEXT_KEY,
    EnumArmState,
    LearningEvent,
    LearningState,
    apply_coordinate_learning,
    detect_ood_enum,
    enum_suppression_decision,
)


def _aggregate(*, outcome: str, promoted_count: int, pass_rate: float) -> CoordinateEvaluationAggregate:
    return CoordinateEvaluationAggregate(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seeds=[7, 42, 156, 8888, 42069],
        generated_count=5,
        evaluated_count=5,
        promoted_count=promoted_count,
        quality_pass_count=promoted_count,
        alignment_pass_count=promoted_count,
        full_pass_count=promoted_count,
        mean_quality=0.2 if promoted_count == 0 else 0.7,
        mean_alignment=0.2 if promoted_count == 0 else 0.7,
        best_quality=0.3 if promoted_count == 0 else 0.9,
        best_alignment=0.3 if promoted_count == 0 else 0.9,
        pass_rate=pass_rate,
        outcome=outcome,
        aggregate_failure_types=["alignment_below_cutoff"] if promoted_count == 0 else [],
        aggregate_blame=[],
        update_signal=LearningUpdateSignal(
            thompson_alpha_delta=float(promoted_count),
            thompson_beta_delta=float(5 - promoted_count),
            gp_affinity_delta=pass_rate - 0.5,
        ),
    )


def test_detect_ood_enum_flags_high_variance_low_observation_arms():
    fresh_arm = EnumArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=1.0, beta=1.0)
    near_uniform_arm = EnumArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=1.0, beta=2.0)
    sparse_arm = EnumArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=1.5, beta=1.5)

    assert detect_ood_enum(fresh_arm, "BLUE_HOUR") is True
    assert near_uniform_arm.posterior_variance > 0.05
    assert detect_ood_enum(near_uniform_arm, "BLUE_HOUR") is True
    assert sparse_arm.observations < 4
    assert detect_ood_enum(sparse_arm, "BLUE_HOUR") is True


def test_detect_ood_enum_does_not_flag_well_observed_arms():
    confident_arm = EnumArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=30.0, beta=3.0)
    balanced_well_observed = EnumArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=12.0, beta=12.0)

    assert detect_ood_enum(confident_arm, "BLUE_HOUR") is False
    assert detect_ood_enum(balanced_well_observed, "BLUE_HOUR") is False


def test_context_arms_nested_dict_supports_independent_context_families():
    state = LearningState()
    state.context_arms["model_family=sdxl"] = {}
    state.context_arms["model_family=flux"] = {}

    state.context_arms["model_family=sdxl"]["cinematography.lighting_mood=BLUE_HOUR"] = EnumArmState(
        axis="cinematography.lighting_mood",
        value="BLUE_HOUR",
        context_key="model_family=sdxl",
        alpha=2.0,
        beta=4.0,
    )
    state.context_arms["model_family=flux"]["cinematography.lighting_mood=BLUE_HOUR"] = EnumArmState(
        axis="cinematography.lighting_mood",
        value="BLUE_HOUR",
        context_key="model_family=flux",
        alpha=10.0,
        beta=1.0,
    )

    assert (
        state.context_arms["model_family=sdxl"]["cinematography.lighting_mood=BLUE_HOUR"].posterior_mean == 1 / 3
    )
    assert (
        state.context_arms["model_family=flux"]["cinematography.lighting_mood=BLUE_HOUR"].posterior_mean
        == 10 / 11
    )


def test_apply_coordinate_learning_writes_to_context_keyed_arms():
    state = LearningState()
    event = LearningEvent(
        event_id="eval:coord_001",
        coordinate_id="coord_001",
        sampled_arms={"cinematography.lighting_mood": "BLUE_HOUR"},
        locked_arms={"object.material.object_01": "CERAMIC"},
        combo_signature="lighting=BLUE_HOUR|material=CERAMIC",
        aggregate=_aggregate(outcome="failed", promoted_count=0, pass_rate=0.0),
    )

    updated = apply_coordinate_learning(state, event, context_key="model_family=sdxl")

    assert "model_family=sdxl" in updated.context_arms
    sdxl_arm = updated.context_arms["model_family=sdxl"]["cinematography.lighting_mood=BLUE_HOUR"]
    sdxl_locked = updated.context_arms["model_family=sdxl"]["object.material.object_01=CERAMIC"]
    assert sdxl_arm.context_key == "model_family=sdxl"
    assert sdxl_arm.alpha == 1.0
    assert sdxl_arm.beta == 6.0
    assert sdxl_locked.locked_reliability_observations == 1
    assert "cinematography.lighting_mood=BLUE_HOUR" in updated.enum_arms


def test_apply_coordinate_learning_falls_back_to_default_context_when_no_key_given():
    state = LearningState()
    event = LearningEvent(
        event_id="eval:coord_001",
        coordinate_id="coord_001",
        sampled_arms={"cinematography.lighting_mood": "BLUE_HOUR"},
        locked_arms={},
        combo_signature="lighting=BLUE_HOUR",
        aggregate=_aggregate(outcome="viable", promoted_count=2, pass_rate=0.4),
    )

    updated = apply_coordinate_learning(state, event)

    assert DEFAULT_CONTEXT_KEY in updated.context_arms
    assert "cinematography.lighting_mood=BLUE_HOUR" in updated.context_arms[DEFAULT_CONTEXT_KEY]


def test_enum_suppression_decision_suppresses_ood_arms_below_stricter_floor():
    ood_arm = EnumArmState(
        axis="cinematography.lighting_mood",
        value="BLUE_HOUR",
        alpha=1.0,
        beta=2.0,
    )
    assert ood_arm.observations == 1
    stable_failures = ["alignment_below_cutoff"] * 10

    ood_decision = enum_suppression_decision(
        ood_arm,
        repeated_failure_types=stable_failures,
        user_authored_locked=False,
        min_observations=1,
    )
    assert ood_decision.suppress is True
    assert ood_decision.reason == "ood_evidence_below_floor"
    assert ood_decision.state == "matched_diagnostic_hold"


def test_enum_suppression_decision_lets_ood_arm_survive_with_strong_posterior():
    ood_arm = EnumArmState(
        axis="cinematography.lighting_mood",
        value="BLUE_HOUR",
        alpha=2.5,
        beta=0.5,
    )
    assert ood_arm.posterior_mean > 0.6
    assert ood_arm.alpha + ood_arm.beta < 4
    stable_failures = ["alignment_below_cutoff"] * 10

    decision = enum_suppression_decision(
        ood_arm,
        repeated_failure_types=stable_failures,
        user_authored_locked=False,
        min_observations=1,
    )

    assert decision.suppress is False
    assert decision.reason == "posterior_above_ood_floor"


def test_learning_state_migrates_legacy_enum_arms_into_default_context():
    legacy = LearningState(
        enum_arms={
            "cinematography.lighting_mood=BLUE_HOUR": EnumArmState(
                axis="cinematography.lighting_mood",
                value="BLUE_HOUR",
                alpha=2.0,
                beta=3.0,
            )
        }
    )

    assert DEFAULT_CONTEXT_KEY in legacy.context_arms
    migrated = legacy.context_arms[DEFAULT_CONTEXT_KEY]["cinematography.lighting_mood=BLUE_HOUR"]
    assert migrated.alpha == 2.0
    assert migrated.beta == 3.0


def test_detect_ood_enum_threshold_is_tunable_per_call():
    fresh_arm = EnumArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=1.0, beta=3.0)

    assert detect_ood_enum(fresh_arm, "BLUE_HOUR", min_observations=10) is True
    assert detect_ood_enum(fresh_arm, "BLUE_HOUR", min_observations=2) is False

    confident_arm = EnumArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=3.0, beta=3.0)
    expected_variance = (3.0 * 3.0) / ((6.0 ** 2) * 7.0)
    assert math.isclose(confident_arm.posterior_variance, expected_variance)
    assert detect_ood_enum(confident_arm, "BLUE_HOUR", variance_threshold=expected_variance + 0.01) is False
    assert detect_ood_enum(confident_arm, "BLUE_HOUR", variance_threshold=expected_variance - 0.01) is True
