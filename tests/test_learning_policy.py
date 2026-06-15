from bruteforce_canvas.evaluation import (
    CoordinateEvaluationAggregate,
    LearningUpdateSignal,
)
from bruteforce_canvas.learning import (
    ComboAffinityState,
    EnumArmState,
    LearningEvent,
    LearningState,
    apply_coordinate_learning,
    coordinate_quarantine_decision,
    enum_suppression_decision,
)


def aggregate(*, outcome: str, promoted_count: int, pass_rate: float) -> CoordinateEvaluationAggregate:
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


def test_coordinate_learning_updates_sampled_arms_and_combo_once():
    state = LearningState()
    event = LearningEvent(
        event_id="eval:coord_001",
        coordinate_id="coord_001",
        sampled_arms={"cinematography.lighting_mood": "BLUE_HOUR"},
        locked_arms={"object.material.object_01": "CERAMIC"},
        combo_signature="lighting=BLUE_HOUR|material=CERAMIC",
        aggregate=aggregate(outcome="failed", promoted_count=0, pass_rate=0.0),
    )

    first = apply_coordinate_learning(state, event)
    second = apply_coordinate_learning(first, event)

    arm = first.enum_arms["cinematography.lighting_mood=BLUE_HOUR"]
    locked = first.enum_arms["object.material.object_01=CERAMIC"]
    combo = first.combo_affinities["lighting=BLUE_HOUR|material=CERAMIC"]
    assert arm.alpha == 1.0
    assert arm.beta == 6.0
    assert locked.locked_reliability_observations == 1
    assert combo.observations == 1
    assert second == first


def test_learning_update_alone_does_not_suppress_without_policy_thresholds():
    arm = EnumArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=1.0, beta=6.0)
    decision = enum_suppression_decision(
        arm,
        repeated_failure_types=["alignment_below_cutoff"],
        user_authored_locked=False,
    )

    assert decision.suppress is False
    assert decision.reason == "insufficient_observations"


def test_enum_suppression_requires_repeated_evidence_and_never_erases_user_locked_intent():
    weak_arm = EnumArmState(axis="cinematography.lighting_mood", value="BLUE_HOUR", alpha=2.0, beta=10.0)
    suppress = enum_suppression_decision(
        weak_arm,
        repeated_failure_types=["alignment_below_cutoff"] * 6 + ["wrong_lighting"] * 4,
        user_authored_locked=False,
    )
    locked = enum_suppression_decision(
        weak_arm,
        repeated_failure_types=["alignment_below_cutoff"] * 6 + ["wrong_lighting"] * 4,
        user_authored_locked=True,
    )

    assert suppress.suppress is True
    assert suppress.state == "matched_suppressed"
    assert locked.suppress is False
    assert locked.state == "locked_reliability_warning"


def test_enum_suppression_accepts_one_stable_failure_family_but_not_infrastructure_failures():
    weak_arm = EnumArmState(axis="cinematography.camera_angle", value="DUTCH_ANGLE", alpha=1.0, beta=11.0)
    stable = enum_suppression_decision(
        weak_arm,
        repeated_failure_types=["wrong_camera_angle"] * 10,
        user_authored_locked=False,
    )
    infrastructure = enum_suppression_decision(
        weak_arm,
        repeated_failure_types=["evaluator_unavailable"] * 10,
        user_authored_locked=False,
    )

    assert stable.suppress is True
    assert stable.reason == "repeated_failures_below_floor"
    assert infrastructure.suppress is False
    assert infrastructure.reason == "infrastructure_only"


def test_coordinate_quarantine_uses_seed_sweep_failure_or_combo_floor():
    failed = coordinate_quarantine_decision(
        aggregate(outcome="failed", promoted_count=0, pass_rate=0.0),
        ComboAffinityState(combo_signature="bad_combo", gp_mean=0.0, observations=1),
    )
    sparse_bad_combo = coordinate_quarantine_decision(
        aggregate(outcome="viable", promoted_count=2, pass_rate=0.4),
        ComboAffinityState(combo_signature="bad_combo", gp_mean=-0.8, observations=2),
    )
    repeated_bad_combo = coordinate_quarantine_decision(
        aggregate(outcome="viable", promoted_count=2, pass_rate=0.4),
        ComboAffinityState(combo_signature="bad_combo", gp_mean=-0.8, observations=10),
    )

    assert failed.quarantine is True
    assert failed.reason == "zero_pass_seed_sweep"
    assert sparse_bad_combo.quarantine is False
    assert repeated_bad_combo.quarantine is True
    assert repeated_bad_combo.reason == "combo_affinity_floor"


def test_coordinate_quarantine_ignores_blocked_infrastructure_batches():
    blocked = aggregate(outcome="blocked", promoted_count=0, pass_rate=0.0).model_copy(
        update={"aggregate_failure_types": ["evaluator_unavailable"]}
    )

    decision = coordinate_quarantine_decision(
        blocked,
        ComboAffinityState(combo_signature="infra_combo", gp_mean=-0.8, observations=10),
    )

    assert decision.quarantine is False
    assert decision.reason == "infrastructure_only"
