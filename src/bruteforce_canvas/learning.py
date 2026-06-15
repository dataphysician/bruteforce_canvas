from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from bruteforce_canvas.evaluation import CoordinateEvaluationAggregate
from bruteforce_canvas.shared import CanonicalStatus, CoordinateId, StrictModel


class EnumArmState(StrictModel):
    axis: str
    value: str
    alpha: float = 1.0
    beta: float = 1.0
    suppressed_until: str | None = None
    context_key: str | None = None
    locked_reliability_observations: int = 0

    @computed_field
    @property
    def observations(self) -> int:
        return max(0, int(round(self.alpha + self.beta - 2.0)))

    @computed_field
    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class ComboAffinityState(StrictModel):
    combo_signature: str
    gp_mean: float = 0.0
    gp_uncertainty: float = 1.0
    observations: int = 0
    last_failure_type: str | None = None


class EnumSuppressionPolicy(StrictModel):
    min_observations: int = 10
    pass_rate_floor: float = 0.20
    stable_failure_family_ratio: float = 0.60
    cooldown_generated_candidates: int = 500
    min_exploration_probability: float = 0.01

    @property
    def cooldown_label(self) -> str:
        return f"cooldown:{self.cooldown_generated_candidates}_generated_candidates"


class LearningState(StrictModel):
    enum_arms: dict[str, EnumArmState] = Field(default_factory=dict)
    combo_affinities: dict[str, ComboAffinityState] = Field(default_factory=dict)
    applied_event_ids: set[str] = Field(default_factory=set)


class LearningEvent(StrictModel):
    event_id: str
    coordinate_id: CoordinateId
    sampled_arms: dict[str, str]
    locked_arms: dict[str, str] = Field(default_factory=dict)
    combo_signature: str
    aggregate: CoordinateEvaluationAggregate


class SuppressionDecision(StrictModel):
    suppress: bool
    state: str | None = None
    reason: str


class QuarantineDecision(StrictModel):
    quarantine: bool
    reason: str


INFRASTRUCTURE_FAILURE_TYPES = {
    "evaluator_unavailable",
    "evaluator_timeout",
    "evaluator_malformed_output",
    "gpu_memory_failure",
    "impact_unavailable",
    "impact_unlicensed",
}


def _arm_key(axis: str, value: str) -> str:
    return f"{axis}={value}"


def apply_coordinate_learning(state: LearningState, event: LearningEvent) -> LearningState:
    if event.event_id in state.applied_event_ids:
        return state

    enum_arms = dict(state.enum_arms)
    combo_affinities = dict(state.combo_affinities)
    alpha_delta = event.aggregate.update_signal.thompson_alpha_delta
    beta_delta = event.aggregate.update_signal.thompson_beta_delta

    for axis, value in event.sampled_arms.items():
        key = _arm_key(axis, value)
        current = enum_arms.get(key, EnumArmState(axis=axis, value=value))
        enum_arms[key] = current.model_copy(
            update={"alpha": current.alpha + alpha_delta, "beta": current.beta + beta_delta}
        )

    for axis, value in event.locked_arms.items():
        key = _arm_key(axis, value)
        current = enum_arms.get(key, EnumArmState(axis=axis, value=value))
        enum_arms[key] = current.model_copy(
            update={"locked_reliability_observations": current.locked_reliability_observations + 1}
        )

    combo = combo_affinities.get(
        event.combo_signature,
        ComboAffinityState(combo_signature=event.combo_signature),
    )
    observations = combo.observations + 1
    gp_mean = combo.gp_mean + (event.aggregate.update_signal.gp_affinity_delta - combo.gp_mean) / observations
    combo_affinities[event.combo_signature] = combo.model_copy(
        update={
            "gp_mean": gp_mean,
            "gp_uncertainty": max(0.05, 1.0 / observations),
            "observations": observations,
            "last_failure_type": event.aggregate.aggregate_failure_types[0]
            if event.aggregate.aggregate_failure_types
            else combo.last_failure_type,
        }
    )

    return LearningState(
        enum_arms=enum_arms,
        combo_affinities=combo_affinities,
        applied_event_ids={*state.applied_event_ids, event.event_id},
    )


def enum_suppression_decision(
    arm: EnumArmState,
    *,
    repeated_failure_types: list[str],
    user_authored_locked: bool,
    min_observations: int = 10,
    pass_rate_floor: float = 0.20,
    stable_failure_family_ratio: float = 0.60,
) -> SuppressionDecision:
    if user_authored_locked:
        return SuppressionDecision(
            suppress=False,
            state="locked_reliability_warning",
            reason="user_authored_locked",
        )
    if arm.observations < min_observations:
        return SuppressionDecision(suppress=False, reason="insufficient_observations")
    semantic_failures = [failure for failure in repeated_failure_types if failure not in INFRASTRUCTURE_FAILURE_TYPES]
    if not semantic_failures:
        return SuppressionDecision(suppress=False, reason="infrastructure_only")
    dominant_count = max(semantic_failures.count(failure) for failure in set(semantic_failures))
    if dominant_count / len(semantic_failures) < stable_failure_family_ratio:
        return SuppressionDecision(suppress=False, reason="failure_family_not_stable")
    if arm.posterior_mean >= pass_rate_floor:
        return SuppressionDecision(suppress=False, reason="posterior_above_floor")
    return SuppressionDecision(
        suppress=True,
        state=CanonicalStatus.MATCHED_SUPPRESSED.value,
        reason="repeated_failures_below_floor",
    )


def coordinate_quarantine_decision(
    aggregate: CoordinateEvaluationAggregate,
    combo: ComboAffinityState,
    *,
    combo_floor: float = -0.35,
    min_combo_observations: int = 10,
) -> QuarantineDecision:
    if aggregate.outcome == "blocked" or (
        aggregate.aggregate_failure_types
        and all(failure in INFRASTRUCTURE_FAILURE_TYPES for failure in aggregate.aggregate_failure_types)
    ):
        return QuarantineDecision(quarantine=False, reason="infrastructure_only")
    if aggregate.promoted_count == 0 and aggregate.evaluated_count >= 5:
        return QuarantineDecision(quarantine=True, reason="zero_pass_seed_sweep")
    if combo.observations >= min_combo_observations and combo.gp_mean <= combo_floor:
        return QuarantineDecision(quarantine=True, reason="combo_affinity_floor")
    return QuarantineDecision(quarantine=False, reason="insufficient_evidence")
