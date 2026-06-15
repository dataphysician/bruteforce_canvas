from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from bruteforce_canvas.evaluation import CoordinateEvaluationAggregate
from bruteforce_canvas.learning import (
    DEFAULT_CONTEXT_KEY,
    ComboAffinityState,
    EnumArmState,
    EnumSuppressionPolicy,
    LearningEvent,
    LearningState,
    QuarantineDecision,
    SuppressionDecision,
    apply_coordinate_learning,
    coordinate_quarantine_decision,
    detect_ood_enum,
    enum_suppression_decision,
)
from bruteforce_canvas.router import CandidateCoordinateBatch, CompatibilityPrior, LHSRouter, RouterInput
from bruteforce_canvas.shared import StrictModel


class BalancerSnapshot(StrictModel):
    learning_state: LearningState
    enum_suppression_policy: EnumSuppressionPolicy
    suppression_checked_count: int
    suppressed_count: int
    quarantine_checked_count: int
    quarantined_count: int
    snapshot_at: str


class BayesianBalancer:
    def __init__(
        self,
        learning_state: LearningState,
        enum_suppression_policy: EnumSuppressionPolicy,
        *,
        coordinate_quarantine_fn: Callable[
            [CoordinateEvaluationAggregate, ComboAffinityState], QuarantineDecision
        ]
        | None = None,
        context_key: str | None = None,
    ) -> None:
        self._learning_state = learning_state
        self._enum_suppression_policy = enum_suppression_policy
        self._coordinate_quarantine_fn = coordinate_quarantine_fn or coordinate_quarantine_decision
        self._context_key = context_key or DEFAULT_CONTEXT_KEY
        self._suppression_results: list[tuple[EnumArmState, SuppressionDecision]] = []
        self._quarantine_decision: QuarantineDecision | None = None
        self._suppression_checked_count = 0
        self._suppressed_count = 0
        self._quarantine_checked_count = 0
        self._quarantined_count = 0

    @property
    def suppression_results(self) -> list[tuple[EnumArmState, SuppressionDecision]]:
        return list(self._suppression_results)

    @property
    def quarantine_decision(self) -> QuarantineDecision | None:
        return self._quarantine_decision

    def update(self, event: LearningEvent) -> LearningState:
        updated_state = apply_coordinate_learning(
            self._learning_state,
            event,
            context_key=self._context_key,
        )
        self._learning_state = updated_state
        self._suppression_results = []
        policy = self._enum_suppression_policy
        for arm in updated_state.enum_arms.values():
            if arm.locked_reliability_observations > 0:
                continue
            decision = enum_suppression_decision(
                arm,
                repeated_failure_types=[str(failure) for failure in event.aggregate.aggregate_failure_types],
                user_authored_locked=False,
                min_observations=policy.min_observations,
                pass_rate_floor=policy.pass_rate_floor,
                stable_failure_family_ratio=policy.stable_failure_family_ratio,
            )
            self._suppression_results.append((arm, decision))
        combo = updated_state.combo_affinities.get(
            event.combo_signature,
            ComboAffinityState(combo_signature=event.combo_signature),
        )
        quarantine_decision = self._coordinate_quarantine_fn(event.aggregate, combo)
        self._quarantine_decision = quarantine_decision
        self._suppression_checked_count = len(self._suppression_results)
        self._suppressed_count = sum(1 for _arm, decision in self._suppression_results if decision.suppress)
        self._quarantine_checked_count = 1
        self._quarantined_count = int(quarantine_decision.quarantine)
        return updated_state

    def propose_coordinate(self, router_input: RouterInput) -> CandidateCoordinateBatch:
        router = LHSRouter(
            seed=None,
            compatibility_prior=CompatibilityPrior(),
            suppressed_exploration_floor=self._enum_suppression_policy.min_exploration_probability,
        )
        return router.propose(router_input)

    def is_ood(self, arm: EnumArmState) -> bool:
        return detect_ood_enum(arm, arm.value)

    def snapshot(self) -> BalancerSnapshot:
        return BalancerSnapshot(
            learning_state=self._learning_state,
            enum_suppression_policy=self._enum_suppression_policy,
            suppression_checked_count=self._suppression_checked_count,
            suppressed_count=self._suppressed_count,
            quarantine_checked_count=self._quarantine_checked_count,
            quarantined_count=self._quarantined_count,
            snapshot_at=datetime.now(timezone.utc).isoformat(),
        )
