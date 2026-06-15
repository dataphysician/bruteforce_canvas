from bruteforce_canvas.locking import (
    LockOverride,
    LockState,
    apply_lock_overrides,
    build_default_lock_config,
)
from bruteforce_canvas.prompt import (
    CanonicalEnum,
    EvidenceCategory,
    EvidenceSpan,
    ConstraintLane,
    ObjectLane,
    PromptDocumentSpec,
    SceneGraphDraft,
    VerificationReport,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Guardrail, Importance, LightingMood
from bruteforce_canvas.prompt_models import CinematographyLane, Element, ObjectDescriptor
from bruteforce_canvas.shared import CanonicalStatus


def document_with_canonical_fields() -> PromptDocumentSpec:
    return PromptDocumentSpec(
        raw_user_prompt="a red ceramic bowl, no extra people",
        graph=SceneGraphDraft(
            seed_prompt="red ceramic bowl",
            elements=[
                Element(
                    id="object_01",
                    label="bowl",
                    entity_type=EntityType.PRODUCT,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="bowl", category=EvidenceCategory.EXPLICIT),
                )
            ]
        ),
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="object_01", color="red")]),
        cinematography_lane=CinematographyLane(lighting_mood=LightingMood.BLUE_HOUR_TWILIGHT),
        canonical_metadata={
            "cinematography.lighting_mood": CanonicalEnum(
                raw_value="blue hour",
                enum_value="BLUE_HOUR",
                status=CanonicalStatus.MATCHED_SUPPRESSED,
                confidence="high",
                reason="suppressed for active model context",
            )
        },
        constraint_lane=ConstraintLane(guardrails=[Guardrail.NO_EXTRA_PEOPLE]),
        verification=VerificationReport(approved=True, issues=[]),
    )


def test_default_lock_config_locks_user_facts_and_negative_guards():
    config = build_default_lock_config(document_with_canonical_fields())

    color = config.by_field_path["object.color.object_01"]
    guard = config.by_field_path["constraint.no_extra_people"]

    assert color.lock_state == LockState.LOCKED
    assert color.lhs_policy == "fixed"
    assert color.evaluation_policy == "must_match"
    assert color.learning_policy == "track_locked_reliability"
    assert guard.lock_state == LockState.LOCKED
    assert guard.priority == "negative_guard"
    assert guard.evaluation_policy == "must_not_appear"


def test_default_lock_config_keeps_suppressed_user_match_fixed_raw_preserving():
    config = build_default_lock_config(document_with_canonical_fields())

    lighting = config.by_field_path["cinematography.lighting_mood"]

    assert lighting.lock_state == LockState.LOCKED
    assert lighting.canonical_status == CanonicalStatus.MATCHED_SUPPRESSED
    assert lighting.lhs_policy == "fixed"
    assert lighting.render_policy == "use_raw_or_safe_raw_preserving_phrase"


def test_default_lock_config_adds_missing_presentation_axes_as_sampleable():
    config = build_default_lock_config(document_with_canonical_fields())

    shot = config.by_field_path["cinematography.shot_size"]

    assert shot.lock_state == LockState.UNLOCKED
    assert shot.lhs_policy == "sampleable_if_missing"
    assert shot.priority == "sampled"


def test_lock_overrides_preserve_default_and_effective_audit_state():
    default = build_default_lock_config(document_with_canonical_fields())
    effective = apply_lock_overrides(
        default,
        [
            LockOverride(
                field_path="cinematography.shot_size",
                lock_state=LockState.LOCKED,
                override_source="user_pre_run",
            )
        ],
    )

    assert default.by_field_path["cinematography.shot_size"].lock_state == LockState.UNLOCKED
    assert effective.by_field_path["cinematography.shot_size"].lock_state == LockState.LOCKED
    assert effective.by_field_path["cinematography.shot_size"].lock_source == "user_pre_run"
