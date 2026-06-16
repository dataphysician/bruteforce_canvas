from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from pydantic import Field, model_validator

from bruteforce_canvas.canonicalizers import BGE_SMALL_EN_MODEL_ID, DEFAULT_EMBEDDING_MATCH_THRESHOLD
from bruteforce_canvas.llm_clients import (
    OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL,
    OPENAI_COMPATIBLE_SERVER_DEFAULT_MODEL,
    OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS,
)
from bruteforce_canvas.orchestration import RunConfig
from bruteforce_canvas.scheduler import HardwareTier
from bruteforce_canvas.shared import StrictModel


class GeneratorKind(StrEnum):
    STUB = "stub"
    BONSAI = "bonsai"
    BONSAI_HTTP = "bonsai-http"


class DeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class LLMProvider(StrEnum):
    OPENAI_COMPATIBLE_SERVER = "openai-compatible-server"


class VLMProvider(StrEnum):
    LOCAL_MINICPM = "local-minicpm"
    OPENAI_COMPATIBLE_SERVER = "openai-compatible-server"


class CanonicalizerProvider(StrEnum):
    EMBEDDING = "embedding"
    LLM = "llm"


class GeneratorConfig(StrictModel):
    kind: str = GeneratorKind.STUB.value
    bonsai_backend_root: Path | None = None
    bonsai_model_root: Path = Path("runtime/models/bonsai-image-4B-ternary-gemlite")
    bonsai_triton_cache_dir: Path = Path("runtime/.triton_cache")
    bonsai_kernel_warmup: bool = True
    bonsai_warmup_prompt: str = "Generate a neutral gray ceramic sphere on a matte table"
    bonsai_warmup_steps: int = Field(default=4, ge=1)
    bonsai_warmup_height: int = Field(default=512, ge=16)
    bonsai_warmup_width: int = Field(default=512, ge=16)
    bonsai_http_url: str = "http://127.0.0.1:7950"
    bonsai_http_token: str | None = None

    @model_validator(mode="after")
    def validate_kind(self) -> GeneratorConfig:
        allowed = {kind.value for kind in GeneratorKind}
        if self.kind not in allowed:
            raise ValueError(f"generator kind must be one of {sorted(allowed)}")
        return self


class DeviceConfig(StrictModel):
    device: str = DeviceKind.AUTO.value
    prewarm: bool = True

    @model_validator(mode="after")
    def validate_device(self) -> DeviceConfig:
        allowed = {kind.value for kind in DeviceKind}
        if self.device not in allowed:
            raise ValueError(f"device must be one of {sorted(allowed)}")
        return self


class LLMConfig(StrictModel):
    provider: str = LLMProvider.OPENAI_COMPATIBLE_SERVER.value
    base_url: str = OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL
    model: str = OPENAI_COMPATIBLE_SERVER_DEFAULT_MODEL
    api_key: str | None = None
    timeout_seconds: float = OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS
    max_completion_tokens: int = 2048
    temperature: float = 0.0
    structured_decoding: bool = True

    @model_validator(mode="after")
    def validate_provider(self) -> LLMConfig:
        allowed = {provider.value for provider in LLMProvider}
        if self.provider not in allowed:
            raise ValueError(f"llm provider must be one of {sorted(allowed)}")
        if not self.base_url.strip():
            raise ValueError("llm base_url must not be empty")
        return self


class VLMConfig(StrictModel):
    provider: str = VLMProvider.LOCAL_MINICPM.value
    base_url: str = OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL
    model: str = "openbmb/MiniCPM-V-4.6"
    api_key: str | None = None
    timeout_seconds: float = OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS
    max_completion_tokens: int = 512
    temperature: float = 0.0
    structured_decoding: bool = True

    @model_validator(mode="after")
    def validate_provider(self) -> VLMConfig:
        allowed = {provider.value for provider in VLMProvider}
        if self.provider not in allowed:
            raise ValueError(f"vlm provider must be one of {sorted(allowed)}")
        if self.provider == VLMProvider.OPENAI_COMPATIBLE_SERVER.value and not self.base_url.strip():
            raise ValueError("vlm base_url must not be empty")
        if not self.model.strip():
            raise ValueError("vlm model must not be empty")
        return self


class CanonicalizerConfig(StrictModel):
    provider: str = CanonicalizerProvider.EMBEDDING.value
    embedding_model: str = BGE_SMALL_EN_MODEL_ID
    match_threshold: float = DEFAULT_EMBEDDING_MATCH_THRESHOLD
    llm_fallback: bool = True

    @model_validator(mode="after")
    def validate_provider(self) -> CanonicalizerConfig:
        allowed = {provider.value for provider in CanonicalizerProvider}
        if self.provider not in allowed:
            raise ValueError(f"canonicalizer provider must be one of {sorted(allowed)}")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("canonicalizer match_threshold must be in [0, 1]")
        if not self.embedding_model.strip():
            raise ValueError("canonicalizer embedding_model must not be empty")
        return self


class AppConfig(StrictModel):
    event_store_path: Path = Path("runtime/events.jsonl")
    openai_model: str = "gpt-4.1-mini"
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    canonicalizer: CanonicalizerConfig = Field(default_factory=CanonicalizerConfig)
    run: RunConfig = Field(
        default_factory=lambda: RunConfig(run_id="run_001", raw_user_prompt="configured run")
    )
    hardware: HardwareTier = Field(default_factory=lambda: HardwareTier(vram_gib=0, cuda_available=False))

    @model_validator(mode="after")
    def validate_bonsai_runtime_device(self) -> AppConfig:
        if self.generator.kind == GeneratorKind.BONSAI.value and self.device.device != DeviceKind.CUDA.value:
            raise ValueError("BC_GENERATOR=bonsai requires BC_DEVICE=cuda")
        return self


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    return default if value is None else float(value)


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    return default if value is None else int(value)


def _optional_int(env: Mapping[str, str], key: str) -> int | None:
    value = env.get(key)
    return None if value is None else int(value)


def _optional_str(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    return None if value is None or value == "" else value


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    return default if value is None else value.lower() == "true"


def load_app_config(env: Mapping[str, str] | None = None) -> AppConfig:
    source = os.environ if env is None else env
    bonsai_backend_root = _optional_str(source, "BC_BONSAI_BACKEND_ROOT")
    generator = GeneratorConfig(
        kind=source.get("BC_GENERATOR", GeneratorKind.STUB.value),
        bonsai_backend_root=Path(bonsai_backend_root) if bonsai_backend_root is not None else None,
        bonsai_model_root=Path(source.get("BC_BONSAI_MODEL_ROOT", "runtime/models/bonsai-image-4B-ternary-gemlite")),
        bonsai_triton_cache_dir=Path(source.get("BC_BONSAI_TRITON_CACHE", "runtime/.triton_cache")),
        bonsai_kernel_warmup=_bool(source, "BC_BONSAI_KERNEL_WARMUP", True),
        bonsai_warmup_prompt=source.get(
            "BC_BONSAI_WARMUP_PROMPT",
            "Generate a neutral gray ceramic sphere on a matte table",
        ),
        bonsai_warmup_steps=_int(source, "BC_BONSAI_WARMUP_STEPS", 4),
        bonsai_warmup_height=_int(source, "BC_BONSAI_WARMUP_HEIGHT", 512),
        bonsai_warmup_width=_int(source, "BC_BONSAI_WARMUP_WIDTH", 512),
        bonsai_http_url=source.get("BC_BONSAI_HTTP_URL", "http://127.0.0.1:7950"),
        bonsai_http_token=_optional_str(source, "BC_BONSAI_HTTP_TOKEN"),
    )
    device = DeviceConfig(
        device=source.get("BC_DEVICE", DeviceKind.AUTO.value),
        prewarm=_bool(source, "BC_DEVICE_PREWARM", True),
    )
    llm = LLMConfig(
        provider=source.get("BC_LLM_PROVIDER", LLMProvider.OPENAI_COMPATIBLE_SERVER.value),
        base_url=source.get("BC_LLM_BASE_URL", OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL),
        model=source.get("BC_LLM_MODEL", OPENAI_COMPATIBLE_SERVER_DEFAULT_MODEL),
        api_key=_optional_str(source, "BC_LLM_API_KEY"),
        timeout_seconds=_float(
            source,
            "BC_LLM_TIMEOUT_SECONDS",
            OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS,
        ),
        max_completion_tokens=_int(source, "BC_LLM_MAX_COMPLETION_TOKENS", 2048),
        temperature=_float(source, "BC_LLM_TEMPERATURE", 0.0),
        structured_decoding=_bool(source, "BC_LLM_STRUCTURED_DECODING", True),
    )
    vlm = VLMConfig(
        provider=source.get("BC_VLM_PROVIDER", VLMProvider.LOCAL_MINICPM.value),
        base_url=source.get("BC_VLM_BASE_URL", OPENAI_COMPATIBLE_SERVER_DEFAULT_BASE_URL),
        model=source.get("BC_VLM_MODEL", "openbmb/MiniCPM-V-4.6"),
        api_key=_optional_str(source, "BC_VLM_API_KEY"),
        timeout_seconds=_float(
            source,
            "BC_VLM_TIMEOUT_SECONDS",
            OPENAI_COMPATIBLE_SERVER_DEFAULT_TIMEOUT_SECONDS,
        ),
        max_completion_tokens=_int(source, "BC_VLM_MAX_COMPLETION_TOKENS", 512),
        temperature=_float(source, "BC_VLM_TEMPERATURE", 0.0),
        structured_decoding=_bool(source, "BC_VLM_STRUCTURED_DECODING", True),
    )
    canonicalizer = CanonicalizerConfig(
        provider=source.get("BC_CANONICALIZER_PROVIDER", CanonicalizerProvider.EMBEDDING.value),
        embedding_model=source.get("BC_CANONICALIZER_EMBEDDING_MODEL", BGE_SMALL_EN_MODEL_ID),
        match_threshold=_float(source, "BC_CANONICALIZER_MATCH_THRESHOLD", DEFAULT_EMBEDDING_MATCH_THRESHOLD),
        llm_fallback=_bool(source, "BC_CANONICALIZER_LLM_FALLBACK", True),
    )
    run = RunConfig(
        run_id=source.get("BC_RUN_ID", "run_001"),
        raw_user_prompt=source.get("BC_RAW_PROMPT", "configured run"),
        iqa_cutoff=_float(source, "BC_IQA_CUTOFF", 0.55),
        alignment_cutoff=_float(source, "BC_ALIGNMENT_CUTOFF", 0.25),
        human_iqa_cutoff=_float(source, "BC_HUMAN_IQA_CUTOFF", 0.70),
        stall_window_seconds=_int(source, "BC_STALL_WINDOW_SECONDS", 1800),
        stall_min_promoted=_int(source, "BC_STALL_MIN_PROMOTED", 10),
        promoted_high_watermark=_optional_int(source, "BC_PROMOTED_HIGH_WATERMARK"),
        promoted_low_watermark=_optional_int(source, "BC_PROMOTED_LOW_WATERMARK"),
        metacognitive_impact_enabled=source.get("BC_IMPACT_ENABLED", "false").lower() == "true",
        metacognitive_min_vram_gib=_int(source, "BC_IMPACT_MIN_VRAM_GIB", 24),
    )
    return AppConfig(
        event_store_path=Path(source.get("BC_EVENT_STORE", "runtime/events.jsonl")),
        openai_model=source.get("BC_OPENAI_MODEL", "gpt-4.1-mini"),
        generator=generator,
        device=device,
        llm=llm,
        vlm=vlm,
        canonicalizer=canonicalizer,
        run=run,
        hardware=HardwareTier(
            vram_gib=_int(source, "BC_VRAM_GIB", 0),
            cuda_available=_bool(source, "BC_CUDA_AVAILABLE", False),
        ),
    )
