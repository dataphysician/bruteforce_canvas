from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bruteforce_canvas.app_config import AppConfig
from bruteforce_canvas.app_factory import build_evaluation_plan, build_run_service
from bruteforce_canvas.evaluation import AlignmentEvaluation, EvaluationImageInput, QualityEvaluation
from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE, GenerationSettings, seed_sweep_requests
from bruteforce_canvas.loop import LoopDecision
from bruteforce_canvas.orchestration import RunConfig, RunRuntimeState
from bruteforce_canvas.persistence import PersistenceRecord, reconstruct_run_state
from bruteforce_canvas.prompt import (
    CanonicalEnum,
    EvidenceCategory,
    EvidenceSpan,
    PromptDocument,
    VerificationIssue,
    VerificationReport,
    target_manifest_from_prompt_spec,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Importance
from bruteforce_canvas.prompt_models import (
    ConstraintLane,
    Element,
    ObjectDescriptor,
    ObjectLane,
    PromptDocumentSpec,
    RelationDescriptor,
    SceneGraphDraft,
)
from bruteforce_canvas.prompt_pipeline import PromptPipeline
from bruteforce_canvas.router import AxisDomain, FieldState, LHSRouter, RouterInput, ThompsonArmState
from bruteforce_canvas.scheduler import HardwareTier
from bruteforce_canvas.shared import CanonicalStatus
from bruteforce_canvas.validation import RetryRequest
from bruteforce_canvas.worker import SeedSweepWorkItem


real_adapters = pytest.importorskip("bruteforce_canvas.real_adapters")
JoyQualityAdapter = real_adapters.JoyQualityAdapter
MiniCPMVAdapter = real_adapters.MiniCPMVAdapter
TRIBEv2Adapter = real_adapters.TRIBEv2Adapter


def _opted_in(config: pytest.Config) -> bool:
    mark_expression = config.option.markexpr
    direct_file_selection = any(Path(argument).name == "test_e2e_real_adapters.py" for argument in config.args)
    return "slow" in mark_expression or "requires_real_models" in mark_expression or direct_file_selection


class _Extractor:
    def extract(self, raw_prompt: str) -> PromptDocumentSpec:
        return _prompt_document(raw_prompt)


class _Canonicalizer:
    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        return CanonicalEnum(
            raw_value=raw_value,
            enum_value=raw_value.upper().replace(" ", "_"),
            status=CanonicalStatus.MATCHED_ACTIVE,
            confidence="high",
            reason=f"deterministic test canonicalization for {field_path}",
        )


class _Verifier:
    def verify(self, document: PromptDocument | PromptDocumentSpec) -> VerificationReport:
        _ = document
        return VerificationReport(approved=True, issues=[])


class _Repairer:
    def repair(
        self,
        document: PromptDocument | PromptDocumentSpec,
        issue: RetryRequest | VerificationIssue,
    ) -> PromptDocument | PromptDocumentSpec:
        _ = issue
        return document


def _prompt_document(raw_prompt: str) -> PromptDocumentSpec:
    return PromptDocumentSpec(
        raw_user_prompt=raw_prompt,
        graph=SceneGraphDraft(
            seed_prompt="red bowl on wooden table",
            elements=[
                Element(
                    id="object_01",
                    label="bowl",
                    entity_type=EntityType.PRODUCT,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="red bowl", category=EvidenceCategory.EXPLICIT),
                ),
                Element(
                    id="object_02",
                    label="table",
                    entity_type=EntityType.FURNITURE,
                    role=ElementRole.SUPPORTING,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="wooden table", category=EvidenceCategory.EXPLICIT),
                ),
            ],
            relations=[
                RelationDescriptor(
                    id="rel_01",
                    source_id="object_01",
                    target_id="object_02",
                    relation_raw="on",
                    evidence=EvidenceSpan(text="bowl on a wooden table", category=EvidenceCategory.EXPLICIT),
                )
            ],
        ),
        object_lane=ObjectLane(
            objects=[
                ObjectDescriptor(target_id="object_01", color="red"),
                ObjectDescriptor(target_id="object_02", material="wooden"),
            ]
        ),
        constraint_lane=ConstraintLane(),
        verification=VerificationReport(
            approved=True,
            issues=[
                VerificationIssue(
                    issue_type="none",
                    repair_scope="document",
                    blocking=False,
                    message="pre-approved deterministic fixture",
                )
            ],
        ),
    )


def _with_score_methods(
    *,
    iqa: Any,
    vlm: Any,
    impact: Any,
    rendered_prompt: str,
    target_manifest: object,
) -> tuple[Any, Any, Any]:
    def score_iqa(images: list[EvaluationImageInput]) -> list[QualityEvaluation]:
        return iqa.evaluate([image.image_path for image in images])

    def score_vlm(images: list[EvaluationImageInput]) -> list[AlignmentEvaluation]:
        return vlm.evaluate(
            [image.image_path for image in images],
            prompt=rendered_prompt,
            manifest=target_manifest,
        )

    def score_impact(images: list[EvaluationImageInput]) -> list[dict[str, Any]]:
        return [result.model_dump() for result in impact.evaluate([image.image_path for image in images])]

    iqa.score = score_iqa
    vlm.score = score_vlm
    impact.score = score_impact
    return iqa, vlm, impact


@pytest.mark.slow
@pytest.mark.requires_real_models
def test_end_to_end_with_real_adapters_cpu_mode(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    if not _opted_in(request.config):
        pytest.skip("opt-in slow real-adapter E2E; run with -m 'slow or requires_real_models'")

    raw_prompt = "a red bowl on a wooden table"
    pipeline_result = PromptPipeline(_Extractor(), _Canonicalizer(), _Verifier(), _Repairer()).run_spec(raw_prompt)
    assert pipeline_result.approved is True
    assert pipeline_result.rendered_prompt is not None

    rendered = pipeline_result.rendered_prompt
    target_manifest = target_manifest_from_prompt_spec(pipeline_result.document)
    router_batch = LHSRouter(seed=7).propose(
        RouterInput(
            run_id="run_101",
            prompt_document_id=rendered.prompt_document_id,
            target_manifest_id=target_manifest.manifest_id,
            fixed_arms={
                "object.color.object_01": AxisDomain(
                    value="RED",
                    state=FieldState.EXPLICIT_LOCKED,
                    source="prompt_pipeline",
                ),
                "object.material.object_02": AxisDomain(
                    value="WOODEN",
                    state=FieldState.EXPLICIT_LOCKED,
                    source="prompt_pipeline",
                ),
            },
            sampleable_axes={
                "cinematography.shot_size": [
                    ThompsonArmState(axis="cinematography.shot_size", value="MEDIUM_SHOT", alpha=3.0, beta=1.0),
                    ThompsonArmState(axis="cinematography.shot_size", value="WIDE_SHOT", alpha=1.0, beta=3.0),
                ],
            },
            count=1,
        )
    )
    coordinate = router_batch.coordinates[0]
    target_manifest = target_manifest.model_copy(update={"run_id": "run_101", "coordinate_id": coordinate.coordinate_id})

    app_config = AppConfig(
        event_store_path=tmp_path / "events.jsonl",
        run=RunConfig(
            run_id="run_101",
            raw_user_prompt=raw_prompt,
            iqa_cutoff=0.0,
            alignment_cutoff=0.0,
            human_iqa_cutoff=0.0,
            metacognitive_impact_enabled=True,
            metacognitive_min_vram_gib=0,
            seed_bundle=list(DEFAULT_SEED_BUNDLE),
            vram_sample_interval_ticks=1,
        ),
        hardware=HardwareTier(vram_gib=0, cuda_available=False),
    )
    evaluation_plan = build_evaluation_plan(app_config).model_copy(
        update={
            "quality_cutoff": 0.0,
            "alignment_cutoff": 0.0,
            "human_quality_cutoff": 0.0,
            "metacognitive_impact": True,
        }
    )
    iqa, vlm, impact = _with_score_methods(
        iqa=JoyQualityAdapter(mode="static", device="cpu"),
        vlm=MiniCPMVAdapter(mode="static", device="cpu"),
        impact=TRIBEv2Adapter(enabled=True, mode="static", device="cpu"),
        rendered_prompt=rendered.rendered_prompt,
        target_manifest=target_manifest,
    )
    service = build_run_service(app_config, iqa=iqa, vlm=vlm, impact=impact)

    service.store.append(
        PersistenceRecord(
            record_id=f"prompt_document:{rendered.prompt_document_id}",
            record_type="prompt_document",
            run_id="run_101",
            prompt_document_id=rendered.prompt_document_id,
            payload=pipeline_result.document.model_dump(mode="json"),
        )
    )
    service.store.append(
        PersistenceRecord(
            record_id=f"target_manifest:{target_manifest.manifest_id}",
            record_type="target_manifest",
            run_id="run_101",
            prompt_document_id=rendered.prompt_document_id,
            target_manifest_id=target_manifest.manifest_id,
            coordinate_id=coordinate.coordinate_id,
            payload=target_manifest.model_dump(mode="json"),
        )
    )
    service.enqueue(
        SeedSweepWorkItem(
            run_id="run_101",
            raw_user_prompt=raw_prompt,
            prompt_document_version="1",
            coordinate_id=coordinate.coordinate_id,
            rendered_prompt=rendered.rendered_prompt,
            target_manifest=target_manifest.model_dump(mode="json"),
            generation_requests=seed_sweep_requests(
                run_id="run_101",
                prompt_document_id=rendered.prompt_document_id,
                target_manifest_id=target_manifest.manifest_id,
                coordinate_id=coordinate.coordinate_id,
                rendered_prompt=rendered.rendered_prompt,
                generation_settings=GenerationSettings(steps=1, height=64, width=64, backend="stub"),
                output_dir=tmp_path,
                generator_model_id="stub-generator",
                generator_backend="stub",
            ),
            evaluation_plan=evaluation_plan,
            sampled_arms=coordinate.sampled_arms,
            locked_arms=coordinate.fixed_arms,
            lhs_row=coordinate.lhs_row,
            compatibility_trace=coordinate.compatibility_trace,
            bayesian_score_before_generation=coordinate.bayesian_score,
            combo_signature=coordinate.combo_signature,
        )
    )

    decisions: list[LoopDecision] = [service.tick(), service.tick(), service.tick()]
    assert decisions[0].reason == "pending_coordinates"

    records = service.store.replay()
    record_types = {record.record_type for record in records}
    candidate_records = [record for record in records if record.record_type == "candidate_record"]
    evaluation_records = [record for record in records if record.record_type == "image_evaluation"]

    assert candidate_records
    assert len(evaluation_records) >= 3
    for record in evaluation_records:
        assert 0.0 <= float(record.payload["quality"]["score"]) <= 1.0
        assert 0.0 <= float(record.payload["alignment"]["score"]) <= 1.0
        if record.payload.get("impact") is not None:
            assert 0.0 <= float(record.payload["impact"]["score"]) <= 1.0

    assert {
        "prompt_document",
        "target_manifest",
        "coordinate_record",
        "candidate_record",
        "image_evaluation",
        "evaluation_aggregate",
        "learning_delta",
    }.issubset(record_types)
    assert reconstruct_run_state(records).learning_update_count >= 1
    assert service.state != RunRuntimeState.BLOCKED
    assert not any(record.record_type == "gate_blocked" for record in records)
    assert not any("error_state" in record.payload for record in records)
