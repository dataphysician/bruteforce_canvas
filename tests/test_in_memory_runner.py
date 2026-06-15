from pathlib import Path

from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE
from bruteforce_canvas.prompt import (
    EvidenceCategory,
    EvidenceSpan,
    ObjectLane,
    PromptDocumentSpec,
    SceneGraphDraft,
    VerificationIssue,
    VerificationReport,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Importance
from bruteforce_canvas.prompt_models import Element, ObjectDescriptor
from bruteforce_canvas.runner import InMemoryRunEngine


def approved_document() -> PromptDocumentSpec:
    return PromptDocumentSpec(
        raw_user_prompt="a ceramic bowl on a wooden table",
        graph=SceneGraphDraft(
            seed_prompt="ceramic bowl on wooden table",
            elements=[
                Element(
                    id="object_01",
                    label="bowl",
                    entity_type=EntityType.PRODUCT,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="ceramic bowl", category=EvidenceCategory.EXPLICIT),
                ),
                Element(
                    id="object_02",
                    label="table",
                    entity_type=EntityType.FURNITURE,
                    role=ElementRole.SUPPORTING,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="wooden table", category=EvidenceCategory.EXPLICIT),
                ),
            ]
        ),
        object_lane=ObjectLane(
            objects=[
                ObjectDescriptor(target_id="object_01", material="ceramic"),
                ObjectDescriptor(target_id="object_02", material="wooden"),
            ]
        ),
        verification=VerificationReport(approved=True, issues=[]),
    )


def blocked_document() -> PromptDocumentSpec:
    return PromptDocumentSpec(
        raw_user_prompt="person throwing something",
        graph=SceneGraphDraft(
            seed_prompt="person throwing something",
            elements=[
                Element(
                    id="person_01",
                    label="person",
                    entity_type=EntityType.PERSON,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="person", category=EvidenceCategory.EXPLICIT),
                )
            ]
        ),
        verification=VerificationReport(
            approved=False,
            issues=[
                VerificationIssue(
                    issue_type="unresolved_action_target",
                    repair_scope="prompt_improvement",
                    blocking=True,
                    message="Specify the thrown object.",
                )
            ],
        ),
    )


def test_in_memory_runner_executes_verified_prompt_through_generate_evaluate_learn(tmp_path: Path):
    engine = InMemoryRunEngine(output_dir=tmp_path)
    result = engine.run_once(
        run_id="run_001",
        document=approved_document(),
        quality_scores=[0.9, 0.8, 0.7, 0.2, 0.1],
        alignment_scores=[0.9, 0.8, 0.7, 0.9, 0.9],
    )

    assert result.generated_seeds == DEFAULT_SEED_BUNDLE
    assert result.curated_count == 3
    assert result.aggregate.outcome == "strong"
    assert result.learning_state.combo_affinities
    assert all(path.exists() for path in result.generated_paths)
    assert [record.record_type for record in result.persisted_records].count("candidate_record") == 5
    record_types = [record.record_type for record in result.persisted_records]
    assert "default_lock_config" in record_types
    assert "effective_lock_config" in record_types
    default_lock = next(record for record in result.persisted_records if record.record_type == "default_lock_config")
    effective_lock = next(record for record in result.persisted_records if record.record_type == "effective_lock_config")
    assert default_lock.record_id != effective_lock.record_id
    assert default_lock.payload["entries"] == effective_lock.payload["entries"]
    assert "evaluation_aggregate" in [record.record_type for record in result.persisted_records]


def test_in_memory_runner_blocks_generation_when_prompt_verification_fails(tmp_path: Path):
    engine = InMemoryRunEngine(output_dir=tmp_path)
    result = engine.run_once(
        run_id="run_001",
        document=blocked_document(),
        quality_scores=[0.9, 0.8, 0.7, 0.2, 0.1],
        alignment_scores=[0.9, 0.8, 0.7, 0.9, 0.9],
    )

    assert result.generated_seeds == []
    assert result.curated_count == 0
    assert [record.record_type for record in result.persisted_records] == ["run_config", "prompt_blocked"]
