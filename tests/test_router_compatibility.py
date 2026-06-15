from bruteforce_canvas.router import (
    AxisDomain,
    CompatibilityPrior,
    FieldState,
    LHSRouter,
    RouterInput,
    ThompsonArmState,
)


def test_router_excludes_hard_rejected_arm_and_records_trace_warning():
    prior = CompatibilityPrior(
        hard_rejected_arms={"cinematography.camera_angle": {"DUTCH_ANGLE"}},
    )
    batch = LHSRouter(seed=3, compatibility_prior=prior).propose(
        RouterInput(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            fixed_arms={},
            sampleable_axes={
                "cinematography.camera_angle": [
                    ThompsonArmState(axis="cinematography.camera_angle", value="DUTCH_ANGLE", alpha=10, beta=1),
                    ThompsonArmState(axis="cinematography.camera_angle", value="EYE_LEVEL", alpha=1, beta=1),
                ]
            },
            count=2,
        )
    )

    assert {coord.sampled_arms["cinematography.camera_angle"] for coord in batch.coordinates} == {"EYE_LEVEL"}
    assert all("rejected arm cinematography.camera_angle=DUTCH_ANGLE" in coord.compatibility_trace.warnings for coord in batch.coordinates)


def test_router_skips_full_coordinate_when_combo_is_hard_rejected():
    prior = CompatibilityPrior(
        hard_rejected_combos={
            "cinematography.shot_size=EXTREME_CLOSE_UP|object.material.object_01=CERAMIC"
        }
    )
    batch = LHSRouter(seed=3, compatibility_prior=prior).propose(
        RouterInput(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            fixed_arms={
                "object.material.object_01": AxisDomain(
                    value="CERAMIC",
                    state=FieldState.EXPLICIT_LOCKED,
                    source="llm_canonicalizer",
                )
            },
            sampleable_axes={
                "cinematography.shot_size": [
                    ThompsonArmState(axis="cinematography.shot_size", value="EXTREME_CLOSE_UP", alpha=10, beta=1),
                ]
            },
            count=1,
        )
    )

    assert batch.coordinates == []
    assert batch.rejected_traces[0].hard_rejects == [
        "rejected combo cinematography.shot_size=EXTREME_CLOSE_UP|object.material.object_01=CERAMIC"
    ]


def test_router_can_allow_suppressed_arm_only_through_exploration_floor():
    suppressed_arm = ThompsonArmState(
        axis="cinematography.lighting_mood",
        value="BLUE_HOUR",
        alpha=1,
        beta=20,
        suppressed=True,
    )
    active_arm = ThompsonArmState(
        axis="cinematography.lighting_mood",
        value="SOFT_WINDOW_LIGHT",
        alpha=2,
        beta=1,
    )

    no_floor = LHSRouter(seed=5).propose(
        RouterInput(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            sampleable_axes={"cinematography.lighting_mood": [suppressed_arm, active_arm]},
            count=4,
        )
    )
    with_floor = LHSRouter(seed=5, suppressed_exploration_floor=1.0).propose(
        RouterInput(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            sampleable_axes={"cinematography.lighting_mood": [suppressed_arm, active_arm]},
            count=4,
        )
    )

    assert {coord.sampled_arms["cinematography.lighting_mood"] for coord in no_floor.coordinates} == {
        "SOFT_WINDOW_LIGHT"
    }
    assert "BLUE_HOUR" in {coord.sampled_arms["cinematography.lighting_mood"] for coord in with_floor.coordinates}


def test_router_rejects_pair_family_hard_rejects():
    prior = CompatibilityPrior()
    batch = LHSRouter(seed=7, compatibility_prior=prior).propose(
        RouterInput(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            sampleable_axes={
                "cinematography.shot_size": [
                    ThompsonArmState(axis="cinematography.shot_size", value="MACRO", alpha=10, beta=1),
                    ThompsonArmState(axis="cinematography.shot_size", value="MEDIUM_SHOT", alpha=10, beta=1),
                ],
                "cinematography.camera_angle": [
                    ThompsonArmState(axis="cinematography.camera_angle", value="WIDE_SHOT", alpha=10, beta=1),
                    ThompsonArmState(axis="cinematography.camera_angle", value="EYE_LEVEL", alpha=10, beta=1),
                ],
            },
            count=1,
        )
    )

    assert all(
        "macro captures only a tiny patch; wide_shot framing is physically impossible"
        not in coord.compatibility_trace.hard_rejects
        for coord in batch.coordinates
    )
