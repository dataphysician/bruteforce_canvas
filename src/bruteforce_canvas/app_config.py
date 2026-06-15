from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from pydantic import Field, model_validator

from bruteforce_canvas.orchestration import RunConfig
from bruteforce_canvas.scheduler import HardwareTier
from bruteforce_canvas.shared import StrictModel


class GeneratorKind(StrEnum):
    STUB = "stub"
    BONSAI = "bonsai"


class GeneratorConfig(StrictModel):
    kind: str = GeneratorKind.STUB.value
    bonsai_model_root: Path = Path("runtime/models/bonsai-image-4B-ternary-gemlite")
    bonsai_triton_cache_dir: Path = Path("runtime/.triton_cache")

    @model_validator(mode="after")
    def validate_kind(self) -> GeneratorConfig:
        allowed = {kind.value for kind in GeneratorKind}
        if self.kind not in allowed:
            raise ValueError(f"generator kind must be one of {sorted(allowed)}")
        return self


class AppConfig(StrictModel):
    event_store_path: Path = Path("runtime/events.jsonl")
    openai_model: str = "gpt-4.1-mini"
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    run: RunConfig = Field(
        default_factory=lambda: RunConfig(run_id="run_001", raw_user_prompt="configured run")
    )
    hardware: HardwareTier = Field(default_factory=lambda: HardwareTier(vram_gib=0, cuda_available=False))


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    return default if value is None else float(value)


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    return default if value is None else int(value)


def _optional_int(env: Mapping[str, str], key: str) -> int | None:
    value = env.get(key)
    return None if value is None else int(value)


def load_app_config(env: Mapping[str, str] | None = None) -> AppConfig:
    source = os.environ if env is None else env
    generator = GeneratorConfig(
        kind=source.get("BC_GENERATOR", GeneratorKind.STUB.value),
        bonsai_model_root=Path(source.get("BC_BONSAI_MODEL_ROOT", "runtime/models/bonsai-image-4B-ternary-gemlite")),
        bonsai_triton_cache_dir=Path(source.get("BC_BONSAI_TRITON_CACHE", "runtime/.triton_cache")),
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
        run=run,
        hardware=HardwareTier(
            vram_gib=_int(source, "BC_VRAM_GIB", 0),
            cuda_available=source.get("BC_CUDA_AVAILABLE", "false").lower() == "true",
        ),
    )
