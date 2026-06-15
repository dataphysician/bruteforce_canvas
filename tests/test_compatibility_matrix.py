from bruteforce_canvas.router import (
    CompatibilityMatrixRule,
    CompatibilityPrior,
    CompatibilitySeverity,
    FieldState,
    LHSRouter,
    RouterInput,
    ThompsonArmState,
)


def test_pairwise_compatibility_downrank_and_boost_affect_prior_trace():
    prior = CompatibilityPrior(
        pair_rules=[
            CompatibilityMatrixRule(
                left_field="cinematography.shot_size",
                left_value="EXTREME_CLOSE_UP",
                right_field="scene_density",
                right_value="DENSE",
                severity=CompatibilitySeverity.STRONG_DOWNRANK,
                weight=-0.55,
                reason="Tight crop may hide required relation targets.",
            ),
            CompatibilityMatrixRule(
                left_field="cinematography.lens",
                left_value="MACRO_LENS",
                right_field="object_scale",
                right_value="TINY_PRODUCT",
                severity=CompatibilitySeverity.BOOST,
                weight=0.20,
                reason="Macro lens fits tiny product detail.",
            ),
        ]
    )

    trace = prior.score_coordinate(
        {
            "cinematography.shot_size": "EXTREME_CLOSE_UP",
            "scene_density": "DENSE",
            "cinematography.lens": "MACRO_LENS",
            "object_scale": "TINY_PRODUCT",
        }
    )

    assert trace.hard_rejects == []
    assert trace.downranks[0].severity == CompatibilitySeverity.STRONG_DOWNRANK
    assert trace.boosts[0].severity == CompatibilitySeverity.BOOST
    assert trace.min_pair_score < trace.mean_pair_score
    assert trace.prior_score < 1.0


def test_pairwise_hard_reject_removes_candidate_from_router_batch():
    prior = CompatibilityPrior(
        pair_rules=[
            CompatibilityMatrixRule(
                left_field="cinematography.camera_angle",
                left_value="FLAT_LAY",
                right_field="cinematography.camera_angle_secondary",
                right_value="LOW_ANGLE",
                severity=CompatibilitySeverity.HARD_REJECT,
                weight=0.0,
                reason="Flat lay and low angle are mutually exclusive.",
            )
        ]
    )

    batch = LHSRouter(seed=1, compatibility_prior=prior).propose(
        RouterInput(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            sampleable_axes={
                "cinematography.camera_angle": [
                    ThompsonArmState(axis="cinematography.camera_angle", value="FLAT_LAY", alpha=2, beta=1)
                ],
                "cinematography.camera_angle_secondary": [
                    ThompsonArmState(axis="cinematography.camera_angle_secondary", value="LOW_ANGLE", alpha=2, beta=1)
                ],
            },
            count=1,
        )
    )

    assert batch.coordinates == []
    assert batch.rejected_traces[0].hard_rejects == ["Flat lay and low angle are mutually exclusive."]


def test_router_bayesian_score_includes_compatibility_prior_weight():
    neutral = LHSRouter(seed=2, compatibility_prior=CompatibilityPrior(), compatibility_prior_weight=0.5).propose(
        RouterInput(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            sampleable_axes={
                "cinematography.shot_size": [
                    ThompsonArmState(axis="cinematography.shot_size", value="MEDIUM_SHOT", alpha=9, beta=1)
                ]
            },
            count=1,
        )
    )
    downranked = LHSRouter(
        seed=2,
        compatibility_prior=CompatibilityPrior(
            pair_rules=[
                CompatibilityMatrixRule(
                    left_field="cinematography.shot_size",
                    left_value="MEDIUM_SHOT",
                    right_field="scene_density",
                    right_value="DENSE",
                    severity=CompatibilitySeverity.SOFT_DOWNRANK,
                    weight=-0.25,
                    reason="medium shot with dense scene has mild visibility risk",
                )
            ]
        ),
        compatibility_prior_weight=0.5,
    ).propose(
        RouterInput(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            fixed_context={"scene_density": "DENSE"},
            sampleable_axes={
                "cinematography.shot_size": [
                    ThompsonArmState(axis="cinematography.shot_size", value="MEDIUM_SHOT", alpha=9, beta=1)
                ]
            },
            count=1,
        )
    )

    assert downranked.coordinates[0].compatibility_trace.prior_score < 1.0
    assert downranked.coordinates[0].bayesian_score < neutral.coordinates[0].bayesian_score
