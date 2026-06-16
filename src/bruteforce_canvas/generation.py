from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
import urllib.error
import urllib.request

from pydantic import Field, model_validator

from bruteforce_canvas.generator_registry import register_generator
from bruteforce_canvas.shared import CoordinateId, DocId, RunId, StrictModel, TargetManifestId


DEFAULT_SEED_BUNDLE = [7, 42, 156, 8888, 42069]
MIN_SEED_BUNDLE_SIZE = 3
BonsaiDevice = Literal["cpu", "cuda", "auto"]


class BonsaiRuntimeError(RuntimeError):
    """Raised when the external Bonsai runtime is not correctly configured."""


def generation_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def elapsed_ms(start: float) -> int:
    return max(0, int(round((perf_counter() - start) * 1000)))


class BonsaiTernaryConfig(StrictModel):
    model_root: Path = Path("runtime/models/bonsai-image-4B-ternary-gemlite")
    backend_root: Path | None = None
    stage_name: str = "fast_image_generation"
    model_id: str = "prism-ml/bonsai-image-ternary-4B-gemlite-2bit"
    backend: str = "bonsai-ternary-gemlite"
    triton_cache_dir: Path = Path("runtime/.triton_cache")
    device: BonsaiDevice = "auto"
    smoke_output_dir: Path = Path("runtime/bonsai_smoke")
    kernel_warmup: bool = True
    warmup_prompt: str = "Generate a neutral gray ceramic sphere on a matte table"
    warmup_steps: int = Field(default=4, ge=1)
    warmup_height: int = Field(default=512, ge=16)
    warmup_width: int = Field(default=512, ge=16)

    @model_validator(mode="after")
    def validate_device(self) -> "BonsaiTernaryConfig":
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("bonsai device must be one of ['auto', 'cpu', 'cuda']")
        return self

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


class BonsaiHttpConfig(StrictModel):
    base_url: str = "http://127.0.0.1:7950"
    token: str | None = None
    backend: str = "bonsai-ternary-gemlite"
    model_id: str = "bonsai-http"
    kernel_warmup: bool = True
    warmup_prompt: str = "Generate a neutral gray ceramic sphere on a matte table"
    warmup_steps: int = Field(default=4, ge=1)
    warmup_height: int = Field(default=512, ge=16)
    warmup_width: int = Field(default=512, ge=16)
    timeout_seconds: float = Field(default=300.0, gt=0)

    @model_validator(mode="after")
    def validate_base_url(self) -> "BonsaiHttpConfig":
        if not self.base_url.strip():
            raise ValueError("Bonsai HTTP base_url must not be empty")
        return self


class GenerationRequest(StrictModel):
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    coordinate_id: CoordinateId
    candidate_id: str | None = None
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


class BonsaiSmokeResult(StrictModel):
    output_path: Path
    file_valid: bool
    elapsed_ms: int = Field(ge=0)
    gpu_memory_before_bytes: int | None = None
    gpu_memory_after_bytes: int | None = None


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
        self._kernel_warmed = False

    def _validated_backend_root(self) -> Path:
        root = self.config.backend_root
        if root is None:
            raise BonsaiRuntimeError(
                "Bonsai backend_gpu is not importable and BC_BONSAI_BACKEND_ROOT is not set. "
                "Set BC_BONSAI_BACKEND_ROOT to the external Bonsai Image Demo root that contains "
                "backend_gpu/server.py."
            )
        root = Path(root)
        expected = root / "backend_gpu" / "server.py"
        if not root.exists() or not root.is_dir():
            raise BonsaiRuntimeError(
                f"BC_BONSAI_BACKEND_ROOT={root} is not a directory. "
                "Set it to the external Bonsai Image Demo root that contains backend_gpu/server.py."
            )
        if not expected.is_file():
            raise BonsaiRuntimeError(
                f"BC_BONSAI_BACKEND_ROOT={root} does not contain backend_gpu/server.py. "
                "Set it to the external Bonsai Image Demo repository root, not this app repository."
            )
        return root

    def _default_pipeline_factory(self):
        try:
            server = importlib.import_module("backend_gpu.server")
        except ModuleNotFoundError as error:
            missing_name = getattr(error, "name", "") or ""
            if missing_name not in {"backend_gpu", "backend_gpu.server"}:
                raise BonsaiRuntimeError(
                    f"Bonsai backend import failed because dependency {missing_name!r} is missing. "
                    "Install the external Bonsai backend dependencies in this virtual environment."
                ) from error
            root = self._validated_backend_root()
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            try:
                server = importlib.import_module("backend_gpu.server")
            except ModuleNotFoundError as second_error:
                raise BonsaiRuntimeError(
                    f"BC_BONSAI_BACKEND_ROOT={root} was added to sys.path, but backend_gpu.server "
                    "still could not be imported. Confirm the external backend package is complete."
                ) from second_error

        build_pipeline = getattr(server, "build_pipeline", None)
        if not callable(build_pipeline):
            raise BonsaiRuntimeError(
                "Bonsai backend module backend_gpu.server must expose a callable build_pipeline(model_id=...)."
            )

        return build_pipeline

    def _pipeline_model_id(self) -> str:
        if self.config.model_root.exists():
            return str(self.config.model_root)
        return self.config.model_id

    def _assert_cuda_available(self) -> None:
        if self.config.device != "cuda":
            return
        try:
            import torch
        except Exception as error:  # pragma: no cover - depends on local ML stack
            raise BonsaiRuntimeError(
                "BC_DEVICE=cuda was requested for Bonsai, but torch is not importable in this environment."
            ) from error
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as error:  # pragma: no cover - depends on local ML stack
            raise BonsaiRuntimeError("Unable to query torch.cuda.is_available() for Bonsai startup.") from error
        if not cuda_available:
            raise BonsaiRuntimeError("BC_DEVICE=cuda was requested for Bonsai, but CUDA is not available.")

    def _assert_output_directory_access(self, output_dir: Path) -> None:
        path = Path(output_dir)
        if path.exists() and not path.is_dir():
            raise BonsaiRuntimeError(f"Bonsai smoke output path is not a directory: {path}")
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".bonsai_write_test"
            probe.write_bytes(b"ok")
            probe.unlink(missing_ok=True)
        except OSError as error:
            raise BonsaiRuntimeError(f"Bonsai smoke output directory is not writable: {path}") from error

    def _load_pipeline(self, *, factory: Any | None = None) -> object:
        if self._pipeline is None:
            os.environ["TRITON_CACHE_DIR"] = str(self.config.triton_cache_dir)
            self.config.triton_cache_dir.mkdir(parents=True, exist_ok=True)
            selected_factory = factory or self.pipeline_factory or self._default_pipeline_factory()
            self._pipeline = selected_factory(model_id=self._pipeline_model_id())
        return self._pipeline

    def assert_available(self, *, output_dir: Path | None = None) -> None:
        self._assert_output_directory_access(output_dir or self.config.smoke_output_dir)
        factory = self.pipeline_factory or self._default_pipeline_factory()
        self._assert_cuda_available()
        self._load_pipeline(factory=factory)

    def is_available(self, *, output_dir: Path | None = None) -> bool:
        try:
            self.assert_available(output_dir=output_dir)
        except Exception:
            return False
        return True

    def prewarm(self) -> None:
        if not self._prewarmed:
            self.assert_available()
            pipeline = self._load_pipeline()
            prewarm = getattr(pipeline, "prewarm", None)
            if prewarm is not None:
                prewarm()
            self._warm_generation_kernels(pipeline)
            self._prewarmed = True

    def _warm_generation_kernels(self, pipeline: object) -> None:
        if self._kernel_warmed or not self.config.kernel_warmup:
            return
        prompt = (
            self.config.warmup_prompt
            if self.config.warmup_prompt.startswith("Generate ")
            else f"Generate {self.config.warmup_prompt}"
        )
        result = pipeline(
            prompt=prompt,
            num_inference_steps=self.config.warmup_steps,
            guidance_scale=1.0,
            height=self.config.warmup_height,
            width=self.config.warmup_width,
        )
        image = getattr(result, "images", [None])[0]
        if image is None:
            raise BonsaiRuntimeError("Bonsai kernel warmup did not return an image")
        self._kernel_warmed = True

    def _cuda_memory_allocated_bytes(self) -> int | None:
        try:
            import torch
        except Exception:
            return None
        try:
            if not torch.cuda.is_available():
                return None
            return int(torch.cuda.memory_allocated())
        except Exception:
            return None

    def smoke_test(
        self,
        *,
        output_dir: Path | None = None,
        prompt: str = "Generate a ceramic bowl on wooden table",
        steps: int = 4,
        height: int = 512,
        width: int = 512,
    ) -> BonsaiSmokeResult:
        smoke_dir = output_dir or self.config.smoke_output_dir
        rendered_prompt = prompt if prompt.startswith("Generate ") else f"Generate {prompt}"
        image_path = smoke_dir / "bonsai_smoke.png"
        memory_before = self._cuda_memory_allocated_bytes()
        request = GenerationRequest(
            run_id="run_001",
            prompt_document_id="doc_001",
            target_manifest_id="eval_manifest_001",
            coordinate_id="coord_001",
            seed=7,
            rendered_prompt=rendered_prompt,
            generation_settings=GenerationSettings(steps=steps, height=height, width=width, backend=self.config.backend),
            image_path=str(image_path),
            generator_model_id=self._pipeline_model_id(),
            generator_backend=self.config.backend,
        )
        result = self.generate(request)
        memory_after = self._cuda_memory_allocated_bytes()
        if not result.candidate.file_valid:
            raise BonsaiRuntimeError(f"Bonsai smoke generation did not produce a valid PNG at {image_path}")
        return BonsaiSmokeResult(
            output_path=image_path,
            file_valid=True,
            elapsed_ms=result.candidate.generation_elapsed_ms,
            gpu_memory_before_bytes=memory_before,
            gpu_memory_after_bytes=memory_after,
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        from bruteforce_canvas.evaluation import DispositionSignal

        if not request.rendered_prompt.startswith("Generate "):
            raise ValueError("generation requires a rendered prompt beginning with 'Generate'")
        started = perf_counter()
        self.prewarm()
        pipeline = self._load_pipeline()
        result = pipeline(
            prompt=request.rendered_prompt,
            num_inference_steps=request.generation_settings.steps,
            guidance_scale=1.0,
            height=request.generation_settings.height,
            width=request.generation_settings.width,
        )
        image = getattr(result, "images", [None])[0]
        if image is None:
            raise RuntimeError("Bonsai pipeline did not return an image")
        Path(request.image_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(request.image_path)
        validation = validate_image_file(Path(request.image_path))
        generated_at = generation_timestamp()
        candidate = CandidateRecord(
            candidate_id=request.candidate_id or f"cand_{request.seed}",
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


class BonsaiHttpAdapter:
    def __init__(self, *, config: BonsaiHttpConfig | None = None) -> None:
        self.config = config or BonsaiHttpConfig()
        self._prewarmed = False
        self._kernel_warmed = False

    def _url(self, path: str) -> str:
        return self.config.base_url.rstrip("/") + path

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        headers = {"Accept": accept}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        return headers

    def _open(self, request: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if error.code in {401, 403}:
                raise BonsaiRuntimeError(
                    "Bonsai HTTP backend rejected the request. Set BC_BONSAI_HTTP_TOKEN to the bearer token "
                    "for the running backend service."
                ) from error
            raise BonsaiRuntimeError(
                f"Bonsai HTTP backend returned HTTP {error.code}: {detail or error.reason}"
            ) from error
        except urllib.error.URLError as error:
            raise BonsaiRuntimeError(
                f"Bonsai HTTP backend is unreachable at {self.config.base_url}: {error.reason}"
            ) from error

    def _post_generate(self, *, prompt: str, seed: int, steps: int, height: int, width: int, backend: str) -> bytes:
        payload = {
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "guidance": 1.0,
            "backend": backend,
            "height": height,
            "width": width,
        }
        request = urllib.request.Request(
            self._url("/generate"),
            data=json.dumps(payload).encode("utf-8"),
            headers={**self._headers(accept="image/png"), "Content-Type": "application/json"},
            method="POST",
        )
        return self._open(request)

    def assert_available(self) -> None:
        request = urllib.request.Request(self._url("/healthz"), headers=self._headers(), method="GET")
        payload = self._open(request)
        try:
            health = json.loads(payload.decode("utf-8"))
        except Exception as error:
            raise BonsaiRuntimeError("Bonsai HTTP backend health endpoint returned malformed JSON.") from error
        if str(health.get("status")) != "ok":
            raise BonsaiRuntimeError(f"Bonsai HTTP backend health is not ok: {health}")

    def is_available(self) -> bool:
        try:
            self.assert_available()
        except Exception:
            return False
        return True

    def prewarm(self) -> None:
        if self._prewarmed:
            return
        self.assert_available()
        if self.config.kernel_warmup and not self._kernel_warmed:
            prompt = (
                self.config.warmup_prompt
                if self.config.warmup_prompt.startswith("Generate ")
                else f"Generate {self.config.warmup_prompt}"
            )
            data = self._post_generate(
                prompt=prompt,
                seed=0,
                steps=self.config.warmup_steps,
                height=self.config.warmup_height,
                width=self.config.warmup_width,
                backend=self.config.backend,
            )
            if data[:8] != b"\x89PNG\r\n\x1a\n":
                raise BonsaiRuntimeError("Bonsai HTTP warmup did not return a PNG image.")
            self._kernel_warmed = True
        self._prewarmed = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        from bruteforce_canvas.evaluation import DispositionSignal

        if not request.rendered_prompt.startswith("Generate "):
            raise ValueError("generation requires a rendered prompt beginning with 'Generate'")
        started = perf_counter()
        self.prewarm()
        data = self._post_generate(
            prompt=request.rendered_prompt,
            seed=request.seed,
            steps=request.generation_settings.steps,
            height=request.generation_settings.height,
            width=request.generation_settings.width,
            backend=request.generation_settings.backend or self.config.backend,
        )
        path = Path(request.image_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        validation = validate_image_file(path)
        generated_at = generation_timestamp()
        candidate = CandidateRecord(
            candidate_id=request.candidate_id or f"cand_{request.seed}",
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
    candidate_id_prefix: str | None = None,
) -> list[GenerationRequest]:
    return [
        GenerationRequest(
            run_id=run_id,
            prompt_document_id=prompt_document_id,
            target_manifest_id=target_manifest_id,
            coordinate_id=coordinate_id,
            candidate_id=f"{candidate_id_prefix}_{seed}" if candidate_id_prefix is not None else None,
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
            backend_root=config.generator.bonsai_backend_root,
            triton_cache_dir=config.generator.bonsai_triton_cache_dir,
            device=config.device.device,
            kernel_warmup=config.generator.bonsai_kernel_warmup,
            warmup_prompt=config.generator.bonsai_warmup_prompt,
            warmup_steps=config.generator.bonsai_warmup_steps,
            warmup_height=config.generator.bonsai_warmup_height,
            warmup_width=config.generator.bonsai_warmup_width,
        )
    )


@register_generator("bonsai-http")
def build_bonsai_http_generator(config: object) -> BonsaiHttpAdapter:
    return BonsaiHttpAdapter(
        config=BonsaiHttpConfig(
            base_url=config.generator.bonsai_http_url,
            token=config.generator.bonsai_http_token,
            backend="bonsai-ternary-gemlite",
            model_id="bonsai-http",
            kernel_warmup=config.generator.bonsai_kernel_warmup,
            warmup_prompt=config.generator.bonsai_warmup_prompt,
            warmup_steps=config.generator.bonsai_warmup_steps,
            warmup_height=config.generator.bonsai_warmup_height,
            warmup_width=config.generator.bonsai_warmup_width,
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
            candidate_id=request.candidate_id or f"cand_{request.seed}",
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
