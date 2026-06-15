from __future__ import annotations

from typing import Literal, cast

from bruteforce_canvas.balancer import BalancerSnapshot, BayesianBalancer
from bruteforce_canvas.evaluation import CoordinateEvaluationAggregate, FailureType, LearningUpdateSignal
from bruteforce_canvas.learning import EnumArmState, EnumSuppressionPolicy, LearningEvent, LearningState
from bruteforce_canvas.router import CandidateCoordinateBatch, RouterInput, ThompsonArmState


def aggregate(
    *,
    outcome: Literal["strong", "viable", "fragile", "failed", "blocked"] = "failed",
    promoted_count: int = 0,
    pass_rate: float = 0.0,
    failure_types: list[FailureType] | None = None,
    alpha_delta: float | None = None,
    beta_delta: float | None = None,
    gp_delta: float | None = None,
) -> CoordinateEvaluationAggregate:
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
        aggregate_failure_types=failure_types or ["alignment_below_cutoff"],
        aggregate_blame=[],
        update_signal=LearningUpdateSignal(
            thompson_alpha_delta=float(promoted_count) if alpha_delta is None else alpha_delta,
            thompson_beta_delta=float(5 - promoted_count) if beta_delta is None else beta_delta,
            gp_affinity_delta=(pass_rate - 0.5) if gp_delta is None else gp_delta,
        ),
    )


def event(
    *,
    event_id: str = "eval:coord_001",
    sampled_arms: dict[str, str] | None = None,
    combo_signature: str = "shot=MEDIUM_SHOT",
    aggregate_value: CoordinateEvaluationAggregate | None = None,
) -> LearningEvent:
    return LearningEvent(
        event_id=event_id,
        coordinate_id="coord_001",
        sampled_arms=sampled_arms or {"cinematography.shot_size": "MEDIUM_SHOT"},
        locked_arms={},
        combo_signature=combo_signature,
        aggregate=aggregate_value or aggregate(),
    )


def router_input(*, count: int) -> RouterInput:
    return RouterInput(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        sampleable_axes={
            "cinematography.shot_size": [
                ThompsonArmState(axis="cinematography.shot_size", value="MEDIUM_SHOT", alpha=3.0, beta=1.0),
                ThompsonArmState(axis="cinematography.shot_size", value="WIDE_SHOT", alpha=1.0, beta=3.0),
            ],
            "cinematography.camera_angle": [
                ThompsonArmState(axis="cinematography.camera_angle", value="EYE_LEVEL", alpha=2.0, beta=1.0),
                ThompsonArmState(axis="cinematography.camera_angle", value="LOW_ANGLE", alpha=1.0, beta=2.0),
            ],
        },
        count=count,
    )


def test_update_is_idempotent_for_same_event() -> None:
    balancer = BayesianBalancer(LearningState(), EnumSuppressionPolicy())
    learning_event = event()

    first = balancer.update(learning_event)
    second = balancer.update(learning_event)

    assert second == first
    assert first.enum_arms["cinematography.shot_size=MEDIUM_SHOT"].beta == 6.0
    assert first.combo_affinities["shot=MEDIUM_SHOT"].observations == 1


def test_update_increments_thompson_alpha_beta_and_gp_combo_affinity() -> None:
    balancer = BayesianBalancer(LearningState(), EnumSuppressionPolicy())
    learning_event = event(
        aggregate_value=aggregate(
            outcome="fragile",
            promoted_count=2,
            pass_rate=0.4,
            alpha_delta=2.0,
            beta_delta=3.0,
            gp_delta=-0.25,
        )
    )

    state = balancer.update(learning_event)

    arm = state.enum_arms["cinematography.shot_size=MEDIUM_SHOT"]
    combo = state.combo_affinities["shot=MEDIUM_SHOT"]
    assert arm.alpha == 3.0
    assert arm.beta == 4.0
    assert combo.observations == 1
    assert combo.train_y == [-0.25]


def test_is_ood_flags_fresh_high_variance_but_not_well_observed_arm() -> None:
    balancer = BayesianBalancer(LearningState(), EnumSuppressionPolicy())
    fresh = EnumArmState(axis="cinematography.shot_size", value="MEDIUM_SHOT")
    observed = EnumArmState(axis="cinematography.shot_size", value="MEDIUM_SHOT", alpha=50.0, beta=5.0)

    assert balancer.is_ood(fresh) is True
    assert balancer.is_ood(observed) is False


def test_snapshot_returns_serializable_balancer_snapshot() -> None:
    weak_state = LearningState(
        enum_arms={
            "cinematography.shot_size=MEDIUM_SHOT": EnumArmState(
                axis="cinematography.shot_size",
                value="MEDIUM_SHOT",
                alpha=1.0,
                beta=11.0,
            )
        }
    )
    balancer = BayesianBalancer(weak_state, EnumSuppressionPolicy())
    balancer.update(
        event(
            sampled_arms={},
            aggregate_value=aggregate(failure_types=cast(list[FailureType], ["wrong_shot_size"] * 10)),
        )
    )

    snapshot = balancer.snapshot()

    assert isinstance(snapshot, BalancerSnapshot)
    assert snapshot.suppression_checked_count == 1
    assert snapshot.suppressed_count == 1
    assert snapshot.quarantine_checked_count == 1
    assert snapshot.model_dump_json()


def test_propose_coordinate_returns_requested_candidate_batch_count() -> None:
    balancer = BayesianBalancer(LearningState(), EnumSuppressionPolicy())

    batch = balancer.propose_coordinate(router_input(count=3))

    assert isinstance(batch, CandidateCoordinateBatch)
    assert len(batch.coordinates) == 3


def test_full_cycle_update_snapshot_and_propose_on_populated_state() -> None:
    populated_state = LearningState(
        enum_arms={
            "cinematography.shot_size=MEDIUM_SHOT": EnumArmState(
                axis="cinematography.shot_size",
                value="MEDIUM_SHOT",
                alpha=4.0,
                beta=2.0,
            )
        }
    )
    balancer = BayesianBalancer(populated_state, EnumSuppressionPolicy(min_exploration_probability=1.0))

    updated = balancer.update(event(event_id="eval:coord_002"))
    snapshot = balancer.snapshot()
    batch = balancer.propose_coordinate(router_input(count=2))

    assert "eval:coord_002" in updated.applied_event_ids
    assert snapshot.learning_state == updated
    assert snapshot.suppression_checked_count >= 1
    assert len(batch.coordinates) == 2
