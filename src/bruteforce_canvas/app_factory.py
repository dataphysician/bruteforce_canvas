from __future__ import annotations

from typing import Any

from bruteforce_canvas.app_config import AppConfig, CanonicalizerProvider, GeneratorKind, LLMProvider, VLMProvider
from bruteforce_canvas.canonicalizers import (
    EmbeddingCanonicalizerAdapter,
    FallbackCanonicalizerAdapter,
    default_embedding_enum_contexts,
)
from bruteforce_canvas.evaluation import EvaluationPlan, StaticIQAAdapter, StaticImpactAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import BonsaiHttpAdapter, BonsaiTernaryAdapter, StubGeneratorAdapter
from bruteforce_canvas.generator_registry import GENERATOR_REGISTRY
from bruteforce_canvas.llm_adapters import (
    LLMCanonicalizerAdapter,
    LLMPromptExtractionAdapter,
    LLMRepairAdapter,
    LLMVerificationAdapter,
)
from bruteforce_canvas.llm_clients import OpenAICompatibleServerJsonLLMClient
from bruteforce_canvas.persistence import JsonlEventStore
from bruteforce_canvas.prompt_models import PromptDocumentSpec
from bruteforce_canvas.prompt_pipeline import PromptPipeline
from bruteforce_canvas.real_adapters import MiniCPMVAdapter, OpenAICompatibleVLMAlignmentAdapter
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.scheduler import EvaluatorStagePlan, plan_evaluator_stages
from bruteforce_canvas.worker import PersistentSeedSweepWorker


def build_generator_adapter(config: AppConfig) -> StubGeneratorAdapter | BonsaiTernaryAdapter | BonsaiHttpAdapter:
    factory = GENERATOR_REGISTRY.get(config.generator.kind, GENERATOR_REGISTRY["stub"])
    return factory(config)


def build_event_store(config: AppConfig) -> JsonlEventStore:
    return JsonlEventStore(config.event_store_path)


def build_json_llm_client(config: AppConfig) -> OpenAICompatibleServerJsonLLMClient:
    if config.llm.provider != LLMProvider.OPENAI_COMPATIBLE_SERVER.value:
        raise ValueError(f"unsupported llm provider: {config.llm.provider}")
    return OpenAICompatibleServerJsonLLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        api_key=config.llm.api_key,
        timeout_seconds=config.llm.timeout_seconds,
        max_completion_tokens=config.llm.max_completion_tokens,
        temperature=config.llm.temperature,
        structured_decoding=config.llm.structured_decoding,
    )


def prewarm_json_llm(config: AppConfig) -> dict:
    """Run one low-stakes prompt-extraction inference against the configured LLM.

    Runtime Gradio uses this before binding the UI so a Modal/vLLM cold start,
    model load, kernel setup, and PromptDocumentSpec JSON-schema setup happen
    before the first user submit.
    """

    warm_llm = config.llm.model_copy(
        update={
            "max_completion_tokens": min(config.llm.max_completion_tokens, 1024),
            "temperature": 0.0,
        }
    )
    warm_config = config.model_copy(update={"llm": warm_llm})
    client = build_json_llm_client(warm_config)
    payload = client.generate_json(
        system=(
            "Warm up prompt extraction for the runtime UI. Extract one graph-first PromptDocumentSpec from "
            "the raw prompt with grounded evidence spans and no hidden reasoning."
        ),
        user={
            "raw_prompt": (
                "Generate a simple red ceramic cube resting on a matte wooden table, "
                "studio lighting, centered composition."
            )
        },
        schema_name=PromptDocumentSpec.__name__,
    )
    if not isinstance(payload, dict):
        raise ValueError("LLM prewarm response must be a JSON object")
    return payload


def build_canonicalizer_adapter(
    config: AppConfig,
    *,
    fallback_client: OpenAICompatibleServerJsonLLMClient | None = None,
) -> object:
    enum_contexts = default_embedding_enum_contexts()
    llm_fallback = LLMCanonicalizerAdapter(
        fallback_client or build_json_llm_client(config),
        enum_contexts=enum_contexts,
    )
    if config.canonicalizer.provider == CanonicalizerProvider.LLM.value:
        return llm_fallback
    if config.canonicalizer.provider != CanonicalizerProvider.EMBEDDING.value:
        raise ValueError(f"unsupported canonicalizer provider: {config.canonicalizer.provider}")

    embedding = EmbeddingCanonicalizerAdapter(
        model_id=config.canonicalizer.embedding_model,
        device=config.device.device,
        match_threshold=config.canonicalizer.match_threshold,
        enum_contexts=enum_contexts,
    )
    if config.canonicalizer.llm_fallback:
        return FallbackCanonicalizerAdapter(embedding, llm_fallback)
    return embedding


def build_prompt_pipeline(
    config: AppConfig,
    *,
    extraction_validation_retries: int = 1,
    max_repairs: int = 2,
    max_semantic_repairs: int | None = None,
    run_semantic_validation: bool = True,
    run_verifier: bool = True,
) -> PromptPipeline:
    client = build_json_llm_client(config)
    return PromptPipeline(
        LLMPromptExtractionAdapter(client, max_validation_retries=extraction_validation_retries),
        build_canonicalizer_adapter(config, fallback_client=client),
        LLMVerificationAdapter(client),
        LLMRepairAdapter(client),
        max_repairs=max_repairs,
        max_semantic_repairs=max_semantic_repairs,
        run_semantic_validation=run_semantic_validation,
        run_verifier=run_verifier,
    )


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


def build_vlm_adapter(config: AppConfig) -> MiniCPMVAdapter | OpenAICompatibleVLMAlignmentAdapter:
    if config.vlm.provider == VLMProvider.LOCAL_MINICPM.value:
        return MiniCPMVAdapter(mode="real", device=config.device.device)
    if config.vlm.provider == VLMProvider.OPENAI_COMPATIBLE_SERVER.value:
        return OpenAICompatibleVLMAlignmentAdapter(
            base_url=config.vlm.base_url,
            model=config.vlm.model,
            api_key=config.vlm.api_key,
            timeout_seconds=config.vlm.timeout_seconds,
            max_completion_tokens=config.vlm.max_completion_tokens,
            temperature=config.vlm.temperature,
            structured_decoding=config.vlm.structured_decoding,
        )
    raise ValueError(f"unsupported vlm provider: {config.vlm.provider}")


def build_run_service(
    config: AppConfig,
    *,
    iqa: StaticIQAAdapter,
    vlm: StaticVLMAdapter,
    impact: StaticImpactAdapter | None = None,
) -> RunService:
    store = build_event_store(config)
    generator = build_generator_adapter(config)
    if config.device.prewarm or config.generator.kind in {GeneratorKind.BONSAI.value, GeneratorKind.BONSAI_HTTP.value}:
        prewarm_all(generator=generator, iqa=iqa, vlm=vlm, impact=impact)
    worker = PersistentSeedSweepWorker(
        store=store,
        generator=generator,
        iqa=iqa,
        vlm=vlm,
        impact=impact,
    )
    return RunService(config=config.run, store=store, worker=worker)


def prewarm_all(
    *,
    generator: Any,
    iqa: Any | None = None,
    vlm: Any | None = None,
    impact: Any | None = None,
) -> None:
    """Invoke ``prewarm()`` on the generator and any real evaluator adapters.

    Adapters that do not implement ``prewarm`` are silently skipped, so
    the factory can call this unconditionally without coupling to the
    G2-G4 real adapter rollout.  Static ``*Adapter`` doubles are valid
    inputs because they all expose the ``prewarm`` method (or a no-op
    equivalent).
    """

    for adapter in (generator, iqa, vlm, impact):
        if adapter is None:
            continue
        prewarm = getattr(adapter, "prewarm", None)
        if prewarm is None:
            continue
        prewarm()
