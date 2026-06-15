from pathlib import Path

from bruteforce_canvas.evaluation import (
    CoordinateEvaluationAggregate,
    DispositionSignal,
    EvaluationPlan,
    LearningUpdateSignal,
    StaticIQAAdapter,
    StaticVLMAdapter,
)
from bruteforce_canvas.generation import (
    CandidateRecord,
    GenerationRequest,
    GenerationResult,
    GenerationSettings,
    StubGeneratorAdapter,
    generation_timestamp,
    seed_sweep_requests,
)
from bruteforce_canvas.learning import EnumArmState, EnumSuppressionPolicy, LearningState
from bruteforce_canvas.persistence import JsonlEventStore, reconstruct_run_state
from bruteforce_canvas.router import CompatibilityTrace, CompatibilityTraceEntry
from bruteforce_canvas.seed_surfing import SeedSurfPolicy
from bruteforce_canvas.worker import PersistentSeedSweepWorker, SeedSweepWorkItem


class MixedValidityGenerator:
    model_version = "1"

    def __init__(self, *, invalid_seeds: set[int]) -> None:
        self.invalid_seeds = invalid_seeds

    def generate(self, request: GenerationRequest) -> GenerationResult:
        path = Path(request.image_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        invalid = request.seed in self.invalid_seeds
        path.write_bytes(b"not a png" if invalid else b"\x89PNG\r\n\x1a\n")
        candidate = CandidateRecord(
            candidate_id=f"cand_{request.seed}",
            run_id=request.run_id,
            prompt_document_id=request.prompt_document_id,
            target_manifest_id=request.target_manifest_id,
            coordinate_id=request.coordinate_id,
            seed=request.seed,
            rendered_prompt=request.rendered_prompt,
            generator_model_id=request.generator_model_id,
            generator_backend=request.generator_backend,
            generation_settings=request.generation_settings.model_dump(),
            image_path=request.image_path,
            file_valid=not invalid,
            timestamp=generation_timestamp(),
            generation_elapsed_ms=0,
        )
        signal = DispositionSignal(
            class_name="hard_purge_invalid_artifact" if invalid else "fail_persist_for_learning",
            confidence="high" if invalid else "medium",
            reasons=["not a png header"] if invalid else ["generated but not evaluated"],
        )
        return GenerationResult(candidate=candidate, infrastructure_blocked=False, disposition_signal=signal)


def work_item(
    tmp_path: Path,
    *,
    compatibility_trace: CompatibilityTrace | None = None,
    bayesian_score_before_generation: float = 0.73,
) -> SeedSweepWorkItem:
    requests = seed_sweep_requests(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        rendered_prompt="Generate a ceramic bowl on wooden table",
        generation_settings=GenerationSettings(),
        output_dir=tmp_path,
        generator_model_id="stub-generator",
        generator_backend="stub",
    )
    return SeedSweepWorkItem(
        run_id="run_001",
        raw_user_prompt="a ceramic bowl on wooden table",
        coordinate_id="coord_001",
        rendered_prompt="Generate a ceramic bowl on wooden table",
        target_manifest={},
        generation_requests=requests,
        evaluation_plan=EvaluationPlan(quality_cutoff=0.55, alignment_cutoff=0.25),
        sampled_arms={"cinematography.shot_size": "MEDIUM_SHOT"},
        locked_arms={"object.material.object_01": "CERAMIC"},
        lhs_row={"cinematography.shot_size": 0},
        lock_configuration={
            "entries": [
                {
                    "field_path": "object.material.object_01",
                    "lock_state": "locked",
                    "lhs_policy": "fixed",
                }
            ]
        },
        default_lock_configuration={
            "entries": [
                {
                    "field_path": "cinematography.shot_size",
                    "lock_state": "unlocked",
                    "lhs_policy": "sampleable_if_missing",
                }
            ]
        },
        effective_lock_configuration={
            "entries": [
                {
                    "field_path": "cinematography.shot_size",
                    "lock_state": "locked",
                    "lhs_policy": "fixed",
                    "override_source": "user_pre_run",
                }
            ]
        },
        verifier_result={"approved": True, "issues": []},
        compatibility_trace=compatibility_trace
        or CompatibilityTrace(
            score=0.88,
            prior_score=0.88,
            min_pair_score=0.75,
            mean_pair_score=0.92,
            downranks=[
                CompatibilityTraceEntry(
                    fields=["object.material.object_01", "cinematography.lighting_mood"],
                    values=["CERAMIC", "BLUE_HOUR"],
                    severity="soft_downrank",
                    weight=-0.1,
                    reason="cool lighting weakens ceramic warmth",
                )
            ],
        ),
        bayesian_score_before_generation=bayesian_score_before_generation,
        combo_signature="shot=MEDIUM_SHOT|material=CERAMIC",
    )


def test_persistent_worker_persists_each_candidate_before_evaluation(tmp_path: Path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=StubGeneratorAdapter(),
        iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2, 0.1]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
    )

    result = worker.run_seed_sweep(work_item(tmp_path))
    records = store.replay()

    assert result.aggregate.promoted_count == 3
    run_config = next(record for record in records if record.record_type == "run_config")
    assert run_config.payload["raw_user_prompt"] == "a ceramic bowl on wooden table"
    assert run_config.payload["prompt_document_id"] == "doc_001"
    assert run_config.payload["target_manifest_id"] == "eval_manifest_001"
    assert run_config.payload["seed_bundle"] == [7, 42, 156, 8888, 42069]
    assert run_config.payload["thresholds"] == {
        "quality_cutoff": 0.55,
        "alignment_cutoff": 0.25,
        "human_quality_cutoff": None,
        "impact_cutoff": None,
    }
    assert run_config.payload["model_versions"]["generator"] == {
        "model_id": "stub-generator",
        "backend": "stub",
        "model_version": "1",
    }
    assert run_config.payload["model_versions"]["iqa"] == {
        "model_id": "static-quality-evaluator",
        "model_version": "1",
    }
    assert run_config.payload["model_versions"]["vlm"] == {
        "model_id": "static-alignment-evaluator",
        "model_version": "1",
    }
    assert run_config.payload["lock_configuration"]["entries"][0]["field_path"] == "object.material.object_01"
    assert run_config.payload["default_lock_configuration"]["entries"][0] == {
        "field_path": "cinematography.shot_size",
        "lock_state": "unlocked",
        "lhs_policy": "sampleable_if_missing",
    }
    assert run_config.payload["effective_lock_configuration"]["entries"][0] == {
        "field_path": "cinematography.shot_size",
        "lock_state": "locked",
        "lhs_policy": "fixed",
        "override_source": "user_pre_run",
    }
    assert run_config.payload["verifier_result"] == {"approved": True, "issues": []}
    coordinate = next(record for record in records if record.record_type == "coordinate_record")
    assert coordinate.record_id == "coordinate:coord_001"
    assert coordinate.idempotency_key == "coordinate:coord_001"
    assert coordinate.payload["rendered_prompt"] == "Generate a ceramic bowl on wooden table"
    assert coordinate.payload["fixed_arms"] == {"object.material.object_01": "CERAMIC"}
    assert coordinate.payload["sampled_arms"] == {"cinematography.shot_size": "MEDIUM_SHOT"}
    assert coordinate.payload["lhs_row"] == {"cinematography.shot_size": 0}
    assert coordinate.payload["combo_signature"] == "shot=MEDIUM_SHOT|material=CERAMIC"
    assert coordinate.payload["compatibility_trace"]["prior_score"] == 0.88
    assert coordinate.payload["bayesian_score"] == 0.73
    assert coordinate.payload["parent_coordinate_id"] is None
    assert coordinate.payload["lifecycle_state"] == "proposed"
    assert [record.record_type for record in records].count("candidate_record") == 5
    assert [record.record_type for record in records].count("image_evaluation") == 5
    candidate = next(record for record in records if record.record_type == "candidate_record")
    assert candidate.payload["timestamp"].endswith("Z")
    assert candidate.payload["timestamp"] != "1970-01-01T00:00:00Z"
    assert candidate.payload["generation_elapsed_ms"] >= 0
    assert candidate.payload["raw_user_prompt"] == "a ceramic bowl on wooden table"
    assert candidate.payload["prompt_document_version"] == "1"
    assert candidate.payload["promotion_thresholds"] == {
        "quality_cutoff": 0.55,
        "alignment_cutoff": 0.25,
        "human_quality_cutoff": None,
        "impact_cutoff": None,
    }
    assert candidate.payload["coordinate_enum_json"] == {
        "locked_arms": {"object.material.object_01": "CERAMIC"},
        "sampled_arms": {"cinematography.shot_size": "MEDIUM_SHOT"},
        "combo_signature": "shot=MEDIUM_SHOT|material=CERAMIC",
    }
    assert candidate.payload["compatibility_trace"]["prior_score"] == 0.88
    assert candidate.payload["compatibility_trace"]["downranks"][0]["reason"] == "cool lighting weakens ceramic warmth"
    assert candidate.payload["bayesian_score_before_generation"] == 0.73
    image_actions = [record for record in records if record.record_type == "system_action" and record.candidate_id]
    assert [record.payload["action_name"] for record in image_actions] == [
        "promote_curate",
        "promote_curate",
        "promote_curate",
        "persist_for_learning",
        "persist_for_learning",
    ]
    assert {record.payload["persistence_version"] for record in image_actions} == {"1"}
    promoted = [
        record
        for record in records
        if record.record_type == "image_evaluation" and record.payload["pass_flags"]["full"]
    ]
    assert [record.candidate_id for record in promoted] == ["cand_7", "cand_42", "cand_156"]
    assert promoted[0].payload["evaluator_request_id"] == "batch:coord_001"
    assert promoted[0].payload["evaluator_telemetry"]["iqa"]["execution_mode"] == "batch"
    assert promoted[0].payload["evaluator_versions"]["iqa"] == {
        "model_id": "static-quality-evaluator",
        "model_version": "1",
    }
    assert promoted[0].payload["evaluator_versions"]["vlm"] == {
        "model_id": "static-alignment-evaluator",
        "model_version": "1",
    }
    assert any(record.record_type == "evaluation_aggregate" for record in records)
    learning = next(record for record in records if record.record_type == "learning_delta")
    assert learning.payload["persistence_version"] == "1"
    assert learning.payload["learning_signal_source"] == "automated_evaluation"
    assert learning.payload["locked_reliability_records"] == [
        {
            "field_path": "object.material.object_01",
            "enum_value": "CERAMIC",
            "observations_delta": 1,
            "total_observations": 1,
            "outcome": "strong",
            "failure_types": ["quality_below_cutoff"],
        }
    ]
    state = reconstruct_run_state(records)
    assert state.generated_count == 5
    assert state.promoted_curated_count == 3


def test_persistent_worker_persists_coordinate_retirement_action_for_failed_coordinate(tmp_path: Path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=StubGeneratorAdapter(),
        iqa=StaticIQAAdapter(scores=[0.1, 0.1, 0.1, 0.1, 0.1]),
        vlm=StaticVLMAdapter(scores=[]),
    )

    result = worker.run_seed_sweep(work_item(tmp_path))

    coordinate_actions = [
        record
        for record in store.replay()
        if record.record_type == "system_action" and record.coordinate_id == result.aggregate.coordinate_id and record.candidate_id is None
    ]
    learning = next(record for record in store.replay() if record.record_type == "learning_delta")
    assert result.aggregate.outcome == "failed"
    assert [record.payload["action_name"] for record in coordinate_actions] == [
        "retire_coordinate",
        "quarantine_coordinate",
    ]
    assert coordinate_actions[0].payload["persistence_version"] == "1"
    assert learning.payload["promotion_curation_state"] == {
        "promoted_count": 0,
        "curated_count": 0,
        "pass_rate": 0.0,
    }
    assert learning.payload["thompson_delta"] == {"alpha": 0.0, "beta": 5.0}
    assert learning.payload["gp_combo_affinity_delta"] == -0.5
    assert learning.payload["coordinate_lifecycle_update"] == {
        "from": "evaluated",
        "to": "retired",
        "reason": "failed",
    }
    assert learning.payload["suppression_counters"] == {"checked": 1, "suppressed": 0}
    assert learning.payload["quarantine_counters"] == {"checked": 1, "quarantined": 1}
    assert learning.payload["suppression_records"] == [
        {
            "field_path": "cinematography.shot_size",
            "enum_value": "MEDIUM_SHOT",
            "context_key": None,
            "reason": "insufficient_observations",
            "suppressed": False,
            "suppression_state": None,
            "suppressed_until": None,
            "min_exploration_probability": 0.01,
        }
    ]
    assert learning.payload["quarantine_records"] == [
        {
            "coordinate_id": "coord_001",
            "combo_signature": "shot=MEDIUM_SHOT|material=CERAMIC",
            "reason": "zero_pass_seed_sweep",
            "quarantined": True,
        }
    ]


def test_persistent_worker_uses_configured_enum_suppression_policy_in_learning_records(tmp_path: Path):
    worker = PersistentSeedSweepWorker(
        store=JsonlEventStore(tmp_path / "events.jsonl"),
        generator=StubGeneratorAdapter(),
        iqa=StaticIQAAdapter(scores=[]),
        vlm=StaticVLMAdapter(scores=[]),
        enum_suppression_policy=EnumSuppressionPolicy(
            cooldown_generated_candidates=250,
            min_exploration_probability=0.05,
        ),
    )
    aggregate = CoordinateEvaluationAggregate(
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        coordinate_id="coord_001",
        seeds=[7, 42, 156, 8888, 42069],
        generated_count=5,
        evaluated_count=5,
        promoted_count=0,
        quality_pass_count=0,
        alignment_pass_count=0,
        full_pass_count=0,
        mean_quality=0.2,
        mean_alignment=0.2,
        best_quality=0.3,
        best_alignment=0.3,
        pass_rate=0.0,
        outcome="failed",
        aggregate_failure_types=["wrong_shot_size"] * 10,
        aggregate_blame=[],
        update_signal=LearningUpdateSignal(
            thompson_alpha_delta=0.0,
            thompson_beta_delta=5.0,
            gp_affinity_delta=-0.5,
        ),
    )
    learning_state = LearningState(
        enum_arms={
            "cinematography.shot_size=MEDIUM_SHOT": EnumArmState(
                axis="cinematography.shot_size",
                value="MEDIUM_SHOT",
                alpha=1.0,
                beta=11.0,
            )
        }
    )

    summary = worker._learning_summary(aggregate, learning_state, "shot=MEDIUM_SHOT", {})

    assert summary["suppression_records"][0]["suppressed"] is True
    assert summary["suppression_records"][0]["suppressed_until"] == "cooldown:250_generated_candidates"
    assert summary["suppression_records"][0]["min_exploration_probability"] == 0.05


def test_persistent_worker_records_seed_surf_bundle_for_strong_coordinate_when_enabled(tmp_path: Path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=StubGeneratorAdapter(),
        iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2, 0.1]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
        seed_surf_policy=SeedSurfPolicy(enabled=True, surf_seed_count=5, seed_start=1000),
    )

    result = worker.run_seed_sweep(work_item(tmp_path))

    surf = next(record for record in store.replay() if record.record_type == "seed_surf_bundle")
    assert result.aggregate.outcome == "strong"
    assert surf.record_id == "seed_surf_bundle:coord_001"
    assert surf.idempotency_key == "seed_surf_bundle:coord_001"
    assert surf.coordinate_id == "coord_001"
    assert surf.payload["parent_coordinate_id"] == "coord_001"
    assert surf.payload["seeds"] == [1000, 1001, 1002, 1003, 1004]
    assert surf.payload["rendered_prompt"] == "Generate a ceramic bowl on wooden table"
    assert surf.payload["fixed_arms"] == {"object.material.object_01": "CERAMIC"}
    assert surf.payload["sampled_arms"] == {"cinematography.shot_size": "MEDIUM_SHOT"}
    assert surf.payload["reason"] == "strong_coordinate_seed_surf"


def test_persistent_worker_keeps_infrastructure_blocked_seed_in_five_seed_audit_without_semantic_penalty(tmp_path: Path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=StubGeneratorAdapter(blocked_seeds={42069}),
        iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
    )

    result = worker.run_seed_sweep(work_item(tmp_path))
    records = store.replay()
    retry_actions = [
        record
        for record in records
        if record.record_type == "system_action"
        and record.candidate_id == "cand_42069"
        and record.payload["action_name"] == "infrastructure_retry"
    ]
    learning = next(record for record in records if record.record_type == "learning_delta")
    sampled_arm = learning.payload["enum_arms"]["cinematography.shot_size=MEDIUM_SHOT"]

    assert result.aggregate.outcome == "blocked"
    assert result.aggregate.generated_count == 5
    assert result.aggregate.evaluated_count == 5
    assert [record.record_type for record in records].count("candidate_record") == 5
    assert [record.record_type for record in records].count("image_evaluation") == 5
    assert retry_actions[0].payload["semantic_penalty"] is False
    assert sampled_arm["beta"] == 2.0
    assert any(
        record.record_type == "image_evaluation"
        and record.candidate_id == "cand_42069"
        and record.payload["failure_types"] == ["gpu_memory_failure"]
        for record in records
    )


def test_persistent_worker_marks_invalid_artifact_for_purge_without_semantic_penalty(tmp_path: Path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=MixedValidityGenerator(invalid_seeds={42069}),
        iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.6]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7, 0.6]),
    )

    result = worker.run_seed_sweep(work_item(tmp_path))
    records = store.replay()
    purge_action = next(
        record
        for record in records
        if record.record_type == "system_action"
        and record.candidate_id == "cand_42069"
        and record.payload["action_name"] == "hard_purge_invalid_artifact"
    )

    assert result.aggregate.promoted_count == 4
    assert purge_action.payload["semantic_penalty"] is False
    assert any(
        record.record_type == "image_evaluation"
        and record.candidate_id == "cand_42069"
        and record.payload["failure_types"] == ["invalid_image_file"]
        for record in records
    )


def test_persistent_worker_replay_prevents_duplicate_learning_on_repeated_run(tmp_path: Path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    item = work_item(tmp_path)
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=StubGeneratorAdapter(),
        iqa=StaticIQAAdapter(scores=[0.9, 0.8, 0.7, 0.2, 0.1]),
        vlm=StaticVLMAdapter(scores=[0.9, 0.8, 0.7]),
    )

    worker.run_seed_sweep(item)
    worker.run_seed_sweep(item)

    records = store.replay()
    assert [record.record_type for record in records].count("learning_delta") == 1
    assert [record.record_type for record in records].count("evaluation_aggregate") == 1
    assert [record.record_type for record in records].count("image_evaluation") == 5
    assert [record.record_type for record in records].count("system_action") == 5
