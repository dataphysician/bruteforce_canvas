from __future__ import annotations

import random
from enum import StrEnum

from pydantic import Field

from bruteforce_canvas.shared import (
    CandidateLifecycle,
    CoordinateId,
    DocId,
    RunId,
    StrictModel,
    TargetManifestId,
    stable_id,
)


class FieldState(StrEnum):
    EXPLICIT_RAW = "explicit_raw"
    EXPLICIT_LOCKED = "explicit_locked"
    EXPLICIT_LOCKED_SUPPRESSED = "explicit_locked_suppressed"
    ENTAILED_LOCKED = "entailed_locked"
    ENTAILED_LOCKED_SUPPRESSED = "entailed_locked_suppressed"
    MISSING_SAMPLEABLE = "missing_sampleable"
    WEAK_SAMPLEABLE = "weak_sampleable"
    SUPPRESSED_SAMPLEABLE = "suppressed_sampleable"
    BLOCKED = "blocked"
    CONFLICT = "conflict"


class AxisDomain(StrictModel):
    value: str
    state: FieldState
    source: str


class ThompsonArmState(StrictModel):
    axis: str
    value: str
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)
    suppressed: bool = False


class CompatibilitySeverity(StrEnum):
    HARD_REJECT = "hard_reject"
    STRONG_DOWNRANK = "strong_downrank"
    SOFT_DOWNRANK = "soft_downrank"
    BOOST = "boost"


class CompatibilityMatrixRule(StrictModel):
    left_field: str
    left_value: str
    right_field: str
    right_value: str
    severity: CompatibilitySeverity
    weight: float
    reason: str

    def matches(self, coordinate: dict[str, str]) -> bool:
        return (
            coordinate.get(self.left_field) == self.left_value
            and coordinate.get(self.right_field) == self.right_value
        ) or (
            coordinate.get(self.left_field) == self.right_value
            and coordinate.get(self.right_field) == self.left_value
        )


class CompatibilityTraceEntry(StrictModel):
    fields: list[str]
    values: list[str]
    severity: str
    weight: float
    reason: str


class CompatibilityTrace(StrictModel):
    hard_rejects: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score: float = 1.0
    prior_score: float = 1.0
    min_pair_score: float = 1.0
    mean_pair_score: float = 1.0
    downranks: list[CompatibilityTraceEntry] = Field(default_factory=list)
    boosts: list[CompatibilityTraceEntry] = Field(default_factory=list)


class CompatibilityPrior(StrictModel):
    hard_rejected_arms: dict[str, set[str]] = Field(default_factory=dict)
    hard_rejected_combos: set[str] = Field(default_factory=set)
    pair_rules: list[CompatibilityMatrixRule] = Field(default_factory=list)

    def allowed_arms(self, axis: str, arms: list[ThompsonArmState]) -> tuple[list[ThompsonArmState], list[str]]:
        rejected_values = self.hard_rejected_arms.get(axis, set())
        warnings = [f"rejected arm {axis}={arm.value}" for arm in arms if arm.value in rejected_values]
        return [arm for arm in arms if arm.value not in rejected_values], warnings

    def reject_combo_reason(self, combo_signature: str) -> str | None:
        if combo_signature in self.hard_rejected_combos:
            return f"rejected combo {combo_signature}"
        return None

    def score_coordinate(self, coordinate: dict[str, str]) -> CompatibilityTrace:
        pair_scores: list[float] = []
        downranks: list[CompatibilityTraceEntry] = []
        boosts: list[CompatibilityTraceEntry] = []
        hard_rejects: list[str] = []
        for rule in self.pair_rules:
            if not rule.matches(coordinate):
                continue
            entry = CompatibilityTraceEntry(
                fields=[rule.left_field, rule.right_field],
                values=[rule.left_value, rule.right_value],
                severity=rule.severity,
                weight=rule.weight,
                reason=rule.reason,
            )
            if rule.severity == CompatibilitySeverity.HARD_REJECT:
                hard_rejects.append(rule.reason)
                pair_scores.append(0.0)
            elif rule.severity == CompatibilitySeverity.STRONG_DOWNRANK:
                downranks.append(entry)
                pair_scores.append(max(0.10, 0.50 + rule.weight))
            elif rule.severity == CompatibilitySeverity.SOFT_DOWNRANK:
                downranks.append(entry)
                pair_scores.append(max(0.35, 0.75 + rule.weight))
            elif rule.severity == CompatibilitySeverity.BOOST:
                boosts.append(entry)
                pair_scores.append(min(1.0, 0.80 + rule.weight))
        if not pair_scores:
            return CompatibilityTrace()
        min_pair = min(pair_scores)
        mean_pair = sum(pair_scores) / len(pair_scores)
        prior_score = min_pair * 0.50 + mean_pair * 0.30 + 1.0 * 0.20
        return CompatibilityTrace(
            hard_rejects=hard_rejects,
            score=prior_score,
            prior_score=prior_score,
            min_pair_score=min_pair,
            mean_pair_score=mean_pair,
            downranks=downranks,
            boosts=boosts,
        )


class RouterInput(StrictModel):
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    fixed_arms: dict[str, AxisDomain] = Field(default_factory=dict)
    fixed_context: dict[str, str] = Field(default_factory=dict)
    sampleable_axes: dict[str, list[ThompsonArmState]] = Field(default_factory=dict)
    count: int = Field(default=8, gt=0)


class CandidateCoordinate(StrictModel):
    candidate_id: str
    coordinate_id: CoordinateId
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    enum_coordinate: dict[str, AxisDomain]
    fixed_arms: dict[str, str]
    sampled_arms: dict[str, str]
    lhs_row: dict[str, int]
    compatibility_trace: CompatibilityTrace
    bayesian_score: float
    combo_signature: str
    lifecycle: CandidateLifecycle = CandidateLifecycle.PROPOSED


class CandidateCoordinateBatch(StrictModel):
    coordinates: list[CandidateCoordinate]
    rejected_traces: list[CompatibilityTrace] = Field(default_factory=list)


class LHSRouter:
    def __init__(
        self,
        seed: int | None = None,
        *,
        compatibility_prior: CompatibilityPrior | None = None,
        suppressed_exploration_floor: float = 0.0,
        compatibility_prior_weight: float = 0.0,
    ) -> None:
        self._rng = random.Random(seed)
        self.compatibility_prior = compatibility_prior or CompatibilityPrior()
        self.suppressed_exploration_floor = suppressed_exploration_floor
        self.compatibility_prior_weight = compatibility_prior_weight

    def propose(self, router_input: RouterInput) -> CandidateCoordinateBatch:
        axes = list(router_input.sampleable_axes)
        coordinates: list[CandidateCoordinate] = []
        rejected_traces: list[CompatibilityTrace] = []
        for index in range(router_input.count):
            enum_coordinate = dict(router_input.fixed_arms)
            sampled_arms: dict[str, str] = {}
            lhs_row: dict[str, int] = {}
            scores: list[float] = []
            warnings: list[str] = []
            for axis in axes:
                arms = [
                    arm
                    for arm in router_input.sampleable_axes[axis]
                    if not arm.suppressed or self._rng.random() < self.suppressed_exploration_floor
                ]
                arms, arm_warnings = self.compatibility_prior.allowed_arms(axis, arms)
                warnings.extend(arm_warnings)
                if not arms:
                    continue
                stratum = index % len(arms)
                ordered = sorted(
                    arms,
                    key=lambda arm: self._rng.betavariate(arm.alpha, arm.beta),
                    reverse=True,
                )
                arm = ordered[stratum % len(ordered)]
                enum_coordinate[axis] = AxisDomain(
                    value=arm.value,
                    state=FieldState.MISSING_SAMPLEABLE,
                    source="lhs_router",
                )
                sampled_arms[axis] = arm.value
                lhs_row[axis] = stratum
                scores.append(arm.alpha / (arm.alpha + arm.beta))
            fixed_arms = {axis: domain.value for axis, domain in router_input.fixed_arms.items()}
            coordinate_values = {**router_input.fixed_context, **fixed_arms, **sampled_arms}
            combo_signature = "|".join(f"{axis}={value}" for axis, value in sorted({**fixed_arms, **sampled_arms}.items()))
            combo_reject = self.compatibility_prior.reject_combo_reason(combo_signature)
            if combo_reject is not None:
                rejected_traces.append(CompatibilityTrace(hard_rejects=[combo_reject], warnings=warnings, score=0.0))
                continue
            compatibility_trace = self.compatibility_prior.score_coordinate(coordinate_values)
            compatibility_trace = compatibility_trace.model_copy(update={"warnings": [*compatibility_trace.warnings, *warnings]})
            if compatibility_trace.hard_rejects:
                rejected_traces.append(compatibility_trace)
                continue
            arm_score = sum(scores) / len(scores) if scores else 1.0
            bayesian_score = (
                arm_score * (1.0 - self.compatibility_prior_weight)
                + compatibility_trace.prior_score * self.compatibility_prior_weight
            )
            coordinate_number = index + 1
            coordinates.append(
                CandidateCoordinate(
                    candidate_id=stable_id("cand", coordinate_number),
                    coordinate_id=stable_id("coord", coordinate_number),
                    run_id=router_input.run_id,
                    prompt_document_id=router_input.prompt_document_id,
                    target_manifest_id=router_input.target_manifest_id,
                    enum_coordinate=enum_coordinate,
                    fixed_arms=fixed_arms,
                    sampled_arms=sampled_arms,
                    lhs_row=lhs_row,
                    compatibility_trace=compatibility_trace,
                    bayesian_score=bayesian_score,
                    combo_signature=combo_signature,
                )
            )
        return CandidateCoordinateBatch(coordinates=coordinates, rejected_traces=rejected_traces)
