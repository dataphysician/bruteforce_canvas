from bruteforce_canvas.evaluation import CoordinateEvaluationAggregate, LearningUpdateSignal
from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE
from bruteforce_canvas.seed_surfing import SeedSurfPolicy, enqueue_seed_surf_bundle


def aggregate(outcome: str, promoted: int) -> CoordinateEvaluationAggregate:
    return CoordinateEvaluationAggregate(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seeds=DEFAULT_SEED_BUNDLE,
        generated_count=5,
        evaluated_count=5,
        promoted_count=promoted,
        quality_pass_count=promoted,
        alignment_pass_count=promoted,
        full_pass_count=promoted,
        mean_quality=0.8,
        mean_alignment=0.8,
        best_quality=0.9,
        best_alignment=0.9,
        pass_rate=promoted / 5,
        outcome=outcome,
        aggregate_failure_types=[],
        aggregate_blame=[],
        update_signal=LearningUpdateSignal(
            thompson_alpha_delta=float(promoted),
            thompson_beta_delta=float(5 - promoted),
            gp_affinity_delta=promoted / 5 - 0.5,
        ),
    )


def test_strong_coordinate_enqueues_seed_surf_bundle_with_parent_provenance():
    bundle = enqueue_seed_surf_bundle(
        aggregate("strong", 4),
        SeedSurfPolicy(enabled=True, surf_seed_count=4, seed_start=1000),
    )

    assert bundle is not None
    assert bundle.parent_coordinate_id == "coord_001"
    assert bundle.run_id == "run_001"
    assert bundle.seeds == [1000, 1001, 1002, 1003]
    assert not set(bundle.seeds).intersection(DEFAULT_SEED_BUNDLE)


def test_fragile_failed_and_disabled_policy_do_not_enqueue_seed_surf():
    assert enqueue_seed_surf_bundle(aggregate("fragile", 1), SeedSurfPolicy(enabled=True)) is None
    assert enqueue_seed_surf_bundle(aggregate("failed", 0), SeedSurfPolicy(enabled=True)) is None
    assert enqueue_seed_surf_bundle(aggregate("strong", 4), SeedSurfPolicy(enabled=False)) is None


def test_policy_can_require_higher_pass_rate_than_strong_label():
    bundle = enqueue_seed_surf_bundle(
        aggregate("strong", 3),
        SeedSurfPolicy(enabled=True, min_pass_rate=0.8),
    )

    assert bundle is None
