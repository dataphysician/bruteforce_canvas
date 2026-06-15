from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, computed_field, model_validator

from bruteforce_canvas.evaluation import CoordinateEvaluationAggregate
from bruteforce_canvas.gp import encode_combo_signature, gp_posterior_update
from bruteforce_canvas.shared import CanonicalStatus, CoordinateId, StrictModel


DEFAULT_CONTEXT_KEY = "default"


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

    @computed_field
    @property
    def posterior_variance(self) -> float:
        """Variance of the Beta(alpha, beta) posterior.

        Closed form: ``alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))``.
        Used by :func:`detect_ood_enum` as an OOD signal.
        """
        denominator = (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1.0)
        if denominator <= 0:
            return 0.0
        return (self.alpha * self.beta) / denominator


class ComboAffinityState(StrictModel):
    combo_signature: str
    gp_mean: float = 0.0
    gp_uncertainty: float = 1.0
    observations: int = 0
    train_x: list[list[float]] = Field(default_factory=list)
    train_y: list[float] = Field(default_factory=list)
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
    context_arms: dict[str, dict[str, EnumArmState]] = Field(default_factory=dict)
    combo_affinities: dict[str, ComboAffinityState] = Field(default_factory=dict)
    applied_event_ids: set[str] = Field(default_factory=set)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_enum_arms(cls, data: Any) -> Any:
        """Migrate legacy ``enum_arms`` payloads into the context-keyed store.

        When loading data persisted before the context-keyed Thompson state was
        introduced, the flat ``enum_arms`` dict may carry entries that have not
        yet been mirrored into ``context_arms``. Move them into the
        ``"default"`` context family so downstream code can read everything
        through the nested store without losing historical evidence.
        """
        if not isinstance(data, dict):
            return data
        enum_arms = data.get("enum_arms") or {}
        context_arms = data.get("context_arms")
        if enum_arms and not context_arms:
            data = dict(data)
            data["context_arms"] = {DEFAULT_CONTEXT_KEY: dict(enum_arms)}
        return data


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


def detect_ood_enum(
    arm: EnumArmState,
    value: str,
    *,
    min_observations: int = 4,
    variance_threshold: float = 0.05,
) -> bool:
    """Return True when ``arm`` shows out-of-distribution evidence.

    The arm is treated as OOD when either signal trips:
    - the posterior concentrates too little evidence (``alpha + beta <
      min_observations``) — the arm has not been observed enough times to be
      trusted in a new context.
    - the Beta posterior variance exceeds ``variance_threshold`` — the arm is
      still close to a uniform prior and not a reliable recommendation.

    ``value`` is the candidate value being considered for the arm and is
    accepted so future extensions can cross-check value-specific OOD signals
    (e.g., concept-drift evidence keyed on the rendered value itself).
    """
    del value  # accepted for future OOD signal extensions
    total = arm.alpha + arm.beta
    if total < min_observations:
        return True
    return arm.posterior_variance > variance_threshold


def apply_coordinate_learning(
    state: LearningState,
    event: LearningEvent,
    *,
    context_key: str | None = None,
) -> LearningState:
    if event.event_id in state.applied_event_ids:
        return state

    enum_arms = dict(state.enum_arms)
    context_arms: dict[str, dict[str, EnumArmState]] = {
        ctx: dict(arms) for ctx, arms in state.context_arms.items()
    }
    combo_affinities = dict(state.combo_affinities)
    alpha_delta = event.aggregate.update_signal.thompson_alpha_delta
    beta_delta = event.aggregate.update_signal.thompson_beta_delta
    effective_context = context_key or DEFAULT_CONTEXT_KEY

    def _store(key: str, axis: str, value: str, update: dict[str, object]) -> EnumArmState:
        current = enum_arms.get(key, EnumArmState(axis=axis, value=value, context_key=effective_context))
        # When the same key is written from multiple contexts we merge updates
        # back into the flat enum_arms entry to preserve backward-compatible
        # read-only access.
        return current.model_copy(update=update)

    for axis, value in event.sampled_arms.items():
        key = _arm_key(axis, value)
        existing = enum_arms.get(key)
        base_alpha = existing.alpha if existing is not None else 1.0
        base_beta = existing.beta if existing is not None else 1.0
        updated = _store(
            key,
            axis,
            value,
            {
                "alpha": base_alpha + alpha_delta,
                "beta": base_beta + beta_delta,
                "context_key": effective_context,
            },
        )
        enum_arms[key] = updated
        context_arms.setdefault(effective_context, {})[key] = updated

    for axis, value in event.locked_arms.items():
        key = _arm_key(axis, value)
        existing = enum_arms.get(key)
        if existing is None:
            updated = EnumArmState(axis=axis, value=value, context_key=effective_context).model_copy(
                update={"locked_reliability_observations": 1}
            )
        else:
            updated = existing.model_copy(
                update={
                    "locked_reliability_observations": existing.locked_reliability_observations + 1,
                    "context_key": effective_context,
                }
            )
        enum_arms[key] = updated
        context_arms.setdefault(effective_context, {})[key] = updated

    combo = combo_affinities.get(
        event.combo_signature,
        ComboAffinityState(combo_signature=event.combo_signature),
    )
    observations = combo.observations + 1
    test_x = encode_combo_signature(event.combo_signature)
    train_x = [*combo.train_x, test_x]
    train_y = [*combo.train_y, float(event.aggregate.update_signal.gp_affinity_delta)]
    gp_mean, gp_uncertainty = gp_posterior_update(train_x, train_y, [test_x])
    combo_affinities[event.combo_signature] = combo.model_copy(
        update={
            "gp_mean": gp_mean,
            "gp_uncertainty": gp_uncertainty,
            "observations": observations,
            "train_x": train_x,
            "train_y": train_y,
            "last_failure_type": event.aggregate.aggregate_failure_types[0]
            if event.aggregate.aggregate_failure_types
            else combo.last_failure_type,
        }
    )

    return LearningState(
        enum_arms=enum_arms,
        context_arms=context_arms,
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
    ood_pass_rate_floor: float = 0.60,
) -> SuppressionDecision:
    """Decide whether to suppress a Thompson-sampled enum arm.

    OOD-aware behaviour: when :func:`detect_ood_enum` flags the arm, the
    ``posterior_above_floor`` guard becomes stricter (default
    ``ood_pass_rate_floor=0.60``) so under-observed or high-variance arms need
    much stronger evidence to stay active.
    """
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
    is_ood = detect_ood_enum(arm, arm.value)
    effective_floor = ood_pass_rate_floor if is_ood else pass_rate_floor
    if arm.posterior_mean >= effective_floor:
        if is_ood:
            return SuppressionDecision(suppress=False, reason="posterior_above_ood_floor")
        return SuppressionDecision(suppress=False, reason="posterior_above_floor")
    if is_ood:
        return SuppressionDecision(
            suppress=True,
            state=CanonicalStatus.MATCHED_DIAGNOSTIC_HOLD.value,
            reason="ood_evidence_below_floor",
        )
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
