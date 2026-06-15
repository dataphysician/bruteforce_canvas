from pathlib import Path

from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE
from bruteforce_canvas.prompt import (
    Element,
    Evidence,
    EvidenceCategory,
    Graph,
    ObjectDescriptor,
    PromptDocument,
    VerificationIssue,
    VerificationReport,
)
from bruteforce_canvas.runner import InMemoryRunEngine


def approved_document() -> PromptDocument:
    return PromptDocument(
        prompt_document_id="doc_001",
        raw_user_prompt="a ceramic bowl on a wooden table",
        seed_prompt="ceramic bowl on wooden table",
        graph=Graph(
            elements=[
                Element(
                    element_id="object_01",
                    label="bowl",
                    entity_type="object",
                    importance="primary",
                    evidence=Evidence(text="ceramic bowl", category=EvidenceCategory.EXPLICIT),
                ),
                Element(
                    element_id="object_02",
                    label="table",
                    entity_type="object",
                    importance="supporting",
                    evidence=Evidence(text="wooden table", category=EvidenceCategory.EXPLICIT),
                ),
            ]
        ),
        objects=[
            ObjectDescriptor(element_id="object_01", field_name="material", raw_value="ceramic"),
            ObjectDescriptor(element_id="object_02", field_name="material", raw_value="wooden"),
        ],
        verification=VerificationReport(approved=True, issues=[]),
    )


def blocked_document() -> PromptDocument:
    return PromptDocument(
        prompt_document_id="doc_001",
        raw_user_prompt="person throwing something",
        seed_prompt="person throwing something",
        graph=Graph(
            elements=[
                Element(
                    element_id="person_01",
                    label="person",
                    entity_type="person",
                    importance="primary",
                    evidence=Evidence(text="person", category=EvidenceCategory.EXPLICIT),
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
