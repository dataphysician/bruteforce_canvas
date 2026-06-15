from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field, model_validator

from bruteforce_canvas.generator_registry import register_generator
from bruteforce_canvas.shared import CoordinateId, DocId, RunId, StrictModel, TargetManifestId


DEFAULT_SEED_BUNDLE = [7, 42, 156, 8888, 42069]


def generation_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def elapsed_ms(start: float) -> int:
    return max(0, int(round((perf_counter() - start) * 1000)))


class BonsaiTernaryConfig(StrictModel):
    model_root: Path = Path("runtime/models/bonsai-image-4B-ternary-gemlite")
    stage_name: str = "fast_image_generation"
    model_id: str = "prism-ml/bonsai-image-ternary-4B-gemlite-2bit"
    backend: str = "bonsai-ternary-gemlite"
    triton_cache_dir: Path = Path("runtime/.triton_cache")

    @property
    def text_encoder_path(self) -> Path:
        return self.model_root / "text_encoder"

    @property
    def transformer_path(self) -> Path:
        return self.model_root / "transformer"

    @property
    def vae_path(self) -> Path:
        return self.model_root / "vae"

    @property
    def tokenizer_path(self) -> Path:
        return self.model_root / "tokenizer"

    @property
    def binary_transformer_path(self) -> Path:
        return self.transformer_path

    @property
    def ternary_transformer_path(self) -> Path:
        return self.transformer_path


class GenerationSettings(StrictModel):
    steps: int = 4
    height: int = 512
    width: int = 512
    backend: str = "bonsai-ternary-gemlite"


class GenerationRequest(StrictModel):
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    coordinate_id: CoordinateId
    seed: int
    rendered_prompt: str
    generation_settings: GenerationSettings
    image_path: str
    generator_model_id: str
    generator_backend: str

    @model_validator(mode="after")
    def validate_rendered_prompt(self) -> "GenerationRequest":
        if not self.rendered_prompt.startswith("Generate "):
            raise ValueError("generation requests require a rendered prompt beginning with 'Generate '")
        return self


class CandidateRecord(StrictModel):
    candidate_id: str
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    coordinate_id: CoordinateId
    seed: int
    rendered_prompt: str
    generator_model_id: str
    generator_backend: str
    generation_settings: dict
    image_path: str
    file_valid: bool
    timestamp: str
    generation_elapsed_ms: int = Field(ge=0)


class FileValidationResult(StrictModel):
    valid: bool
    failure_type: Literal["invalid_image_file", "image_decode_failed"] | None = None
    reason: str | None = None


class GenerationResult(StrictModel):
    candidate: CandidateRecord
    infrastructure_blocked: bool = False
    disposition_signal: object


class SeedSweepAggregate(StrictModel):
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    coordinate_id: CoordinateId
    seeds: list[int] = Field(default_factory=lambda: list(DEFAULT_SEED_BUNDLE))
    generated: int
    promoted: int = 0
    pass_rate: float = 0.0
    coordinate_status: str = "generated"


class ResidentGeneratorWorker:
    def __init__(self, adapter: object) -> None:
        self.adapter = adapter
        self._prewarmed = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not request.rendered_prompt.startswith("Generate "):
            raise ValueError("generation requires a rendered prompt beginning with 'Generate'")
        if not self._prewarmed:
            prewarm = getattr(self.adapter, "prewarm", None)
            if prewarm is not None:
                prewarm()
            self._prewarmed = True
        return self.adapter.generate(request)


class BonsaiTernaryAdapter:
    def __init__(self, *, config: BonsaiTernaryConfig | None = None, pipeline_factory: object | None = None) -> None:
        self.config = config or BonsaiTernaryConfig()
        self.pipeline_factory = pipeline_factory
        self._pipeline: object | None = None
        self._prewarmed = False

    def _default_pipeline_factory(self):
        from backend_gpu.pipeline_gpu import GpuPipeline

        return GpuPipeline

    def _load_pipeline(self) -> object:
        if self._pipeline is None:
            os.environ["TRITON_CACHE_DIR"] = str(self.config.triton_cache_dir)
            factory = self.pipeline_factory or self._default_pipeline_factory()
            self._pipeline = factory(
                text_encoder_path=str(self.config.text_encoder_path),
                transformer_path=str(self.config.transformer_path),
                vae_path=str(self.config.vae_path),
                tokenizer_path=str(self.config.tokenizer_path),
                binary_transformer_path=str(self.config.binary_transformer_path),
                ternary_transformer_path=str(self.config.ternary_transformer_path),
                backend=self.config.backend,
            )
        return self._pipeline

    def prewarm(self) -> None:
        pipeline = self._load_pipeline()
        if not self._prewarmed:
            prewarm = getattr(pipeline, "prewarm", None)
            if prewarm is not None:
                prewarm()
            self._prewarmed = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        from bruteforce_canvas.evaluation import DispositionSignal

        if not request.rendered_prompt.startswith("Generate "):
            raise ValueError("generation requires a rendered prompt beginning with 'Generate'")
        started = perf_counter()
        self.prewarm()
        pipeline = self._load_pipeline()
        generate = getattr(pipeline, "generate")
        generate(
            prompt=request.rendered_prompt,
            seed=request.seed,
            steps=request.generation_settings.steps,
            height=request.generation_settings.height,
            width=request.generation_settings.width,
            output_path=request.image_path,
        )
        validation = validate_image_file(Path(request.image_path))
        generated_at = generation_timestamp()
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
            file_valid=validation.valid,
            timestamp=generated_at,
            generation_elapsed_ms=elapsed_ms(started),
        )
        signal = (
            DispositionSignal(
                class_name="fail_persist_for_learning",
                confidence="medium",
                reasons=["generated but not evaluated"],
            )
            if validation.valid
            else DispositionSignal(
                class_name="hard_purge_invalid_artifact",
                confidence="high",
                reasons=[validation.reason or "invalid image"],
            )
        )
        return GenerationResult(candidate=candidate, infrastructure_blocked=False, disposition_signal=signal)


def seed_sweep_requests(
    *,
    run_id: str,
    prompt_document_id: str,
    target_manifest_id: str,
    coordinate_id: str,
    rendered_prompt: str,
    generation_settings: GenerationSettings,
    output_dir: Path,
    generator_model_id: str,
    generator_backend: str,
) -> list[GenerationRequest]:
    return [
        GenerationRequest(
            run_id=run_id,
            prompt_document_id=prompt_document_id,
            target_manifest_id=target_manifest_id,
            coordinate_id=coordinate_id,
            seed=seed,
            rendered_prompt=rendered_prompt,
            generation_settings=generation_settings,
            image_path=str(output_dir / coordinate_id / f"seed_{seed}.png"),
            generator_model_id=generator_model_id,
            generator_backend=generator_backend,
        )
        for seed in DEFAULT_SEED_BUNDLE
    ]


def validate_image_file(path: Path) -> FileValidationResult:
    if not path.exists() or not path.is_file():
        return FileValidationResult(valid=False, failure_type="invalid_image_file", reason="file missing")
    try:
        header = path.read_bytes()[:8]
    except OSError as exc:
        return FileValidationResult(valid=False, failure_type="image_decode_failed", reason=str(exc))
    if header != b"\x89PNG\r\n\x1a\n":
        return FileValidationResult(valid=False, failure_type="invalid_image_file", reason="not a png header")
    return FileValidationResult(valid=True)


@register_generator("stub")
def build_stub_generator(config: object) -> StubGeneratorAdapter:
    return StubGeneratorAdapter()


@register_generator("bonsai")
def build_bonsai_generator(config: object) -> BonsaiTernaryAdapter:
    return BonsaiTernaryAdapter(
        config=BonsaiTernaryConfig(
            model_root=config.generator.bonsai_model_root,
            triton_cache_dir=config.generator.bonsai_triton_cache_dir,
        )
    )


class StubGeneratorAdapter:
    def __init__(self, *, blocked_seeds: set[int] | None = None) -> None:
        self.blocked_seeds = blocked_seeds or set()
        self.prewarm_count = 0
        self.generate_count = 0

    def prewarm(self) -> None:
        self.prewarm_count += 1

    def generate(self, request: GenerationRequest) -> GenerationResult:
        from bruteforce_canvas.evaluation import DispositionSignal

        started = perf_counter()
        self.generate_count += 1
        blocked = request.seed in self.blocked_seeds
        path = Path(request.image_path)
        if not blocked:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
        validation = validate_image_file(path)
        generated_at = generation_timestamp()
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
            file_valid=validation.valid,
            timestamp=generated_at,
            generation_elapsed_ms=elapsed_ms(started),
        )
        if blocked:
            signal = DispositionSignal(
                class_name="infrastructure_retry_no_semantic_penalty",
                confidence="high",
                reasons=["stubbed infrastructure block"],
            )
        elif validation.valid:
            signal = DispositionSignal(
                class_name="fail_persist_for_learning",
                confidence="medium",
                reasons=["generated but not evaluated"],
            )
        else:
            signal = DispositionSignal(
                class_name="hard_purge_invalid_artifact",
                confidence="high",
                reasons=[validation.reason or "invalid image"],
            )
        return GenerationResult(candidate=candidate, infrastructure_blocked=blocked, disposition_signal=signal)
