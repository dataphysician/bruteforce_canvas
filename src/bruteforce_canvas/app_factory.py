from __future__ import annotations

from bruteforce_canvas.app_config import AppConfig, GeneratorKind
from bruteforce_canvas.evaluation import EvaluationPlan, StaticIQAAdapter, StaticImpactAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import BonsaiTernaryAdapter, BonsaiTernaryConfig, StubGeneratorAdapter
from bruteforce_canvas.persistence import JsonlEventStore
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.scheduler import EvaluatorStagePlan, plan_evaluator_stages
from bruteforce_canvas.worker import PersistentSeedSweepWorker


def build_generator_adapter(config: AppConfig) -> StubGeneratorAdapter | BonsaiTernaryAdapter:
    if config.generator.kind == GeneratorKind.BONSAI.value:
        return BonsaiTernaryAdapter(
            config=BonsaiTernaryConfig(
                model_root=config.generator.bonsai_model_root,
                triton_cache_dir=config.generator.bonsai_triton_cache_dir,
            )
        )
    return StubGeneratorAdapter()


def build_event_store(config: AppConfig) -> JsonlEventStore:
    return JsonlEventStore(config.event_store_path)


def build_stage_plan(config: AppConfig) -> EvaluatorStagePlan:
    return plan_evaluator_stages(config.run, config.hardware)


def build_evaluation_plan(config: AppConfig) -> EvaluationPlan:
    stage_plan = build_stage_plan(config)
    return EvaluationPlan(
        quality_cutoff=config.run.iqa_cutoff,
        alignment_cutoff=config.run.alignment_cutoff,
        human_quality_cutoff=config.run.human_iqa_cutoff,
        metacognitive_impact=stage_plan.impact,
        execution_preference="auto",
    )


def build_run_service(
    config: AppConfig,
    *,
    iqa: StaticIQAAdapter,
    vlm: StaticVLMAdapter,
    impact: StaticImpactAdapter | None = None,
) -> RunService:
    store = build_event_store(config)
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=build_generator_adapter(config),
        iqa=iqa,
        vlm=vlm,
        impact=impact,
    )
    return RunService(config=config.run, store=store, worker=worker)
