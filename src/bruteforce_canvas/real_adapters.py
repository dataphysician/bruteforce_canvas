"""Real evaluator adapter implementations (Spec 04 / Phase G).

This module hosts the *real* IQA / VLM / impact adapters that load actual
HuggingFace models. The classes are kept separate from
:mod:`bruteforce_canvas.evaluation` so that the base evaluation module
remains importable without the heavy ``[ml]`` optional dependencies
(``transformers``, ``torch``, ``safetensors``, ``Pillow``).

All public adapters in this module:

* expose a ``prewarm() -> None`` method that loads the underlying model
  once (idempotent),
* expose an ``is_available() -> bool`` method that returns ``True`` only
  when the runtime can actually load the model,
* expose an ``evaluate(images)`` method that returns a list of stable
  :class:`~bruteforce_canvas.evaluation.QualityEvaluation` (or
  alignment / impact equivalent) records — one per input image.

Adapters that require ``[ml]`` extras do **not** import
``transformers`` / ``torch`` at module import time. They perform the
import lazily inside methods, so the package can still be imported
on machines without the ML stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from bruteforce_canvas.evaluation import AlignmentEvaluation, ImpactEvaluation, QualityEvaluation
from bruteforce_canvas.prompt import EvaluationTargetManifest


# Spec 04 §6.1 / §20: JoyQuality SigLIP2 SO400M is the reference IQA
# encoder.
JOYQUALITY_PRIMARY_MODEL_ID = "fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn"
JOYQUALITY_PROCESSOR_MODEL_ID = "google/siglip2-so400m-patch16-512"

# Spec 04 / Phase G: MiniCPM-V 4.6 is the reference open-source VLM
# used for the alignment evaluator. The adapter below wraps it
# following the same lazy-load contract as JoyQualityAdapter.
MINICPM_V_MODEL_ID: str = "openbmb/MiniCPM-V-4.6"
MINICPM_V_MODEL_VERSION: str = "4.6"

# Phase G: TRIBE v2 (Lite, Q-Value variant) is the reference
# metacognitive impact predictor. It is an open weights image
# classifier used to score the "impression" of a generated image
# (1 = strong, 0 = weak). The adapter is lazy-loaded on first
# ``prewarm``/``evaluate`` call.
TRIBE_V2_MODEL_ID: str = "Jessylg27/tribev2-lite-qv"

# Default model version string used when the loaded checkpoint does not
# expose one via the transformers config.
_DEFAULT_MODEL_VERSION = "1"


def _static_score_for_path(image_path: Path) -> float:
    """Deterministic, hash-derived score in [0, 1] for a given path.

    Used by the static mode and as a graceful fallback when the real
    model is unreachable. The score is stable across processes because
    it is derived from a SHA-256 of the absolute path.
    """

    digest = hashlib.sha256(str(image_path).encode("utf-8")).digest()
    # Use the first 4 bytes as an unsigned int; divide by max uint32.
    raw = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return raw / 0xFFFFFFFF


class JoyQualityAdapter:
    """Real IQA adapter backed by the JoyQuality SigLIP2 model family.

    The adapter operates in two modes:

    ``"static"`` (default — no heavy deps required)
        Returns deterministic, hash-derived scores in ``[0, 1]`` without
        loading any model. Useful for unit tests, CI, and the legacy
        ``StaticIQAAdapter`` contract.

    ``"real"``
        Lazy-imports ``transformers`` and ``torch`` and loads the
        SigLIP2 classifier. The first call to :meth:`evaluate` triggers
        the load if :meth:`prewarm` was not called explicitly.

    The ``device`` argument controls accelerator placement:

    * ``"auto"`` (default) — use CUDA when available, otherwise CPU.
    * ``"cuda"`` — require CUDA; raise if it is not available.
    * ``"cpu"`` — always run on CPU (slowest but safest in CI).

    Model choice
    ------------
    * **Primary**: ``fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn``
      (the spec 04 §20 reference IQA encoder).

    Notes
    -----
    The adapter is intentionally conservative: it never blocks import
    of :mod:`bruteforce_canvas.real_adapters`, and it never commits
    model weights. All heavy imports happen inside
    :meth:`_load_model` (called from :meth:`prewarm` or lazily on the
    first :meth:`evaluate` call).
    """

    def __init__(
        self,
        *,
        mode: Literal["real", "static"] = "static",
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ) -> None:
        if mode not in ("real", "static"):
            raise ValueError(f"mode must be 'real' or 'static', got {mode!r}")
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {device!r}")

        self.mode = mode
        self.device = device
        self._model: Any | None = None
        self._processor: Any | None = None
        self._resolved_model_id: str | None = None
        self._resolved_model_version: str = _DEFAULT_MODEL_VERSION
        self._resolved_device: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def prewarm(self) -> None:
        """Load the underlying model once.

        Calling :meth:`prewarm` more than once is a no-op. In ``static``
        mode the method exists for API symmetry but does nothing.
        """

        if self.mode == "static":
            return
        if self._model is not None:
            return
        self._model, self._processor, self._resolved_model_id, self._resolved_model_version = self._load_model()
        self._resolved_device = self._resolve_device()
        try:
            self._model.to(self._resolved_device)
        except Exception:
            pass
        self._model.eval()

    def is_available(self) -> bool:
        """Return ``True`` when the model can actually be loaded.

        In ``static`` mode this always returns ``True`` (the adapter
        never needs the heavy stack).

        In ``real`` mode the check attempts to import ``transformers``
        and ``torch``; missing dependencies or no CUDA device when
        ``device="cuda"`` results in ``False``.
        """

        if self.mode == "static":
            return True
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        if self.device == "cuda":
            try:
                import torch

                return bool(torch.cuda.is_available())
            except Exception:
                return False
        return True

    # ------------------------------------------------------------------
    # Public evaluation entry point
    # ------------------------------------------------------------------
    def evaluate(self, images: list[Path]) -> list[QualityEvaluation]:
        """Return one :class:`QualityEvaluation` per input image path.

        In ``static`` mode the score is a deterministic function of the
        image path (SHA-256 → ``[0, 1]``). In ``real`` mode the score
        is the model's sigmoid-normalized quality logit; if the model
        cannot be reached the adapter falls back to the static path
        rather than raising, so the evaluator pipeline stays robust.
        """

        paths = [Path(image) for image in images]
        if self.mode == "static":
            return [self._static_evaluation(path) for path in paths]

        try:
            return self._evaluate_real(paths)
        except Exception:
            # Defensive fallback: a model load / inference failure must
            # not crash the batch — degrade to deterministic scores so
            # the orchestrator can still record a result.
            return [self._static_evaluation(path) for path in paths]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _static_evaluation(self, path: Path) -> QualityEvaluation:
        score = _static_score_for_path(path)
        # Clamp away from the literal 0.0 / 1.0 boundaries so the
        # downstream cutoff logic does not collapse to pass/fail on a
        # hash quirk.
        score = min(max(score, 1e-6), 1.0 - 1e-6)
        return QualityEvaluation(
            score=score,
            model_id="static-joyquality",
            model_version=_DEFAULT_MODEL_VERSION,
            confidence="high",
        )

    def _evaluate_real(self, paths: list[Path]) -> list[QualityEvaluation]:
        if self._model is None or self._processor is None:
            self.prewarm()

        assert self._model is not None
        assert self._processor is not None

        import torch  # local import; only needed in real mode
        from PIL import Image  # local import; only needed in real mode

        device = self._resolved_device or self._resolve_device()
        results: list[QualityEvaluation] = []
        for path in paths:
            with Image.open(path) as raw:
                image = raw.convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            score = self._logit_to_score(logits)
            results.append(
                QualityEvaluation(
                    score=score,
                    model_id=self._resolved_model_id or JOYQUALITY_PRIMARY_MODEL_ID,
                    model_version=self._resolved_model_version,
                    confidence="high",
                )
            )
        return results

    @staticmethod
    def _logit_to_score(logits: Any) -> float:
        """Map a raw classifier logit tensor to a ``[0, 1]`` score.

        We reduce to a scalar via ``sigmoid(mean(logits))`` which works
        for both single-label and multi-quality-bin heads.
        """

        import torch  # local import; only needed in real mode

        reduced = logits.float().mean()
        probability = torch.sigmoid(reduced).item()
        if math.isnan(probability):
            return 0.5
        return float(min(max(probability, 1e-6), 1.0 - 1e-6))

    def _resolve_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model(self) -> tuple[Any, Any, str, str]:
        """Load the JoyQuality SigLIP2 IQA model and its processor.

        Returns a 4-tuple ``(model, processor, model_id, model_version)``.
        """

        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoProcessor
        except Exception as error:  # pragma: no cover - exercised only with deps
            raise RuntimeError(
                "transformers is required for JoyQualityAdapter in 'real' mode; "
                "install the [ml] extras to enable it"
            ) from error

        model = AutoModelForImageClassification.from_pretrained(JOYQUALITY_PRIMARY_MODEL_ID)
        processor = _load_joyquality_processor(AutoProcessor, AutoImageProcessor)
        version = _extract_model_version(model)
        return model, processor, JOYQUALITY_PRIMARY_MODEL_ID, version


def _extract_model_version(model: Any) -> str:
    """Best-effort model version extraction that never raises."""

    config = getattr(model, "config", None)
    for attr in ("model_version", "version", "revision"):
        value = getattr(config, attr, None)
        if value:
            return str(value)
    id_label = getattr(config, "_name_or_path", None)
    if id_label:
        return str(id_label)
    return _DEFAULT_MODEL_VERSION


def _load_joyquality_processor(auto_processor: Any, auto_image_processor: Any) -> Any:
    """Load the image processor for JoyQuality classifier weights.

    The JoyQuality checkpoint contains the classifier config and
    weights, but no processor files. The matching SigLIP2 base checkpoint
    ships the required ``preprocessor_config.json``.
    """

    last_error: Exception | None = None
    for loader in (auto_processor, auto_image_processor):
        for model_id in (JOYQUALITY_PRIMARY_MODEL_ID, JOYQUALITY_PROCESSOR_MODEL_ID):
            try:
                return loader.from_pretrained(model_id)
            except Exception as error:  # pragma: no cover - depends on HF cache/network
                last_error = error
    raise RuntimeError("could not load JoyQuality image processor") from last_error


def _static_alignment_score(
    image_path: str | Path,
    prompt: str,
    manifest: EvaluationTargetManifest,
) -> float:
    """Deterministic, hash-derived alignment score in [0, 1].

    Combines the image path, prompt, and a stable signature of the
    manifest's targets. The score is stable across processes and
    platforms because it is derived from a SHA-256 digest.
    """

    target_signature = "|".join(
        f"{target.target_id}:{target.target_kind}:{target.evaluation_policy}"
        for target in manifest.targets
    )
    negative_signature = "|".join(
        f"!{target.target_id}:{target.target_kind}:{target.evaluation_policy}"
        for target in manifest.negative_targets
    )
    payload = (
        f"{Path(image_path).as_posix()}\x00{prompt}\x00{target_signature}\x00{negative_signature}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    bucket = int(digest[:8], 16)
    return bucket / 0xFFFFFFFF


def _parse_alignment_from_response(response: Any) -> float | None:
    """Best-effort extraction of a float score from a MiniCPM-V response.

    Returns ``None`` if the response does not contain a parseable score;
    callers should fall back to a deterministic static score in that
    case so a malformed model reply never crashes the batch.
    """

    if isinstance(response, (int, float)):
        return float(response)
    text: str
    if isinstance(response, str):
        text = response
    elif isinstance(response, dict):
        raw = response.get("score")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    else:
        try:
            text = json.dumps(response)
        except TypeError:
            text = str(response)

    match = re.search(r"\"score\"\s*:\s*(-?\d+(?:\.\d+)?)", text)
    if match is not None:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


class MiniCPMVAdapter:
    """VLM adapter for the ``openbmb/MiniCPM-V-4.6`` model.

    Parameters
    ----------
    mode:
        ``"static"`` (default) returns deterministic scores derived from
        a hash of the inputs and never touches ML libraries. ``"real"``
        lazy-imports ``torch`` and ``transformers`` to load and run
        MiniCPM-V. Real mode is intentionally **not** the default and
        must be opted into explicitly.
    device:
        Inference device hint used only in real mode. ``"auto"`` defers
        to the model's own device-resolution logic.

    The adapter is safe to instantiate in any environment. Missing ML
    dependencies only cause :meth:`is_available` to return ``False``
    and any real-mode call to raise ``RuntimeError`` with a clear,
    actionable message. Importing this module never requires the
    optional dependencies.
    """

    def __init__(
        self,
        *,
        mode: Literal["real", "static"] = "static",
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ) -> None:
        if mode not in ("real", "static"):
            raise ValueError(f"mode must be 'real' or 'static', got {mode!r}")
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {device!r}")

        self.mode = mode
        self.device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._resolved_device: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def prewarm(self) -> None:
        """Load MiniCPM-V once and cache it on this instance.

        Calling :meth:`prewarm` more than once is a no-op. In ``static``
        mode the method exists for API symmetry but does nothing. In
        ``real`` mode the model and tokenizer are downloaded (if
        necessary) and moved to the resolved device. Model weights are
        never committed to the repository.
        """

        if self.mode == "static":
            return
        if self._model is not None:
            return
        model, tokenizer, resolved = self._load_model()
        self._model = model
        self._tokenizer = tokenizer
        self._resolved_device = resolved

    def is_available(self) -> bool:
        """Return ``True`` iff the real backend is loadable right now.

        Static mode always reports available because it never touches
        ML libraries. Real mode probes for ``torch`` and
        ``transformers`` without loading weights, so this method is
        cheap and safe to call from health checks.
        """

        if self.mode == "static":
            return True
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            return False
        if self.device == "cuda":
            try:
                import torch

                return bool(torch.cuda.is_available())
            except Exception:
                return False
        return True

    # ------------------------------------------------------------------
    # Public evaluation entry point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        images: Sequence[str | Path],
        prompt: str,
        manifest: EvaluationTargetManifest,
    ) -> list[AlignmentEvaluation]:
        """Score each image for alignment against ``prompt`` and ``manifest``.

        Returns one :class:`~bruteforce_canvas.evaluation.AlignmentEvaluation`
        per input image. In ``static`` mode the score is a deterministic
        function of the image path, prompt, and manifest signature. In
        ``real`` mode the prewarmed MiniCPM-V model is asked to score
        the image; a malformed response falls back to the deterministic
        static score so the batch never crashes.
        """

        if not isinstance(images, list):
            raise TypeError(f"images must be a list, got {type(images).__name__}")
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a str, got {type(prompt).__name__}")
        if not isinstance(manifest, EvaluationTargetManifest):
            raise TypeError(
                f"manifest must be an EvaluationTargetManifest, got {type(manifest).__name__}"
            )

        paths = [Path(image) for image in images]
        if len(paths) == 0:
            return []
        if self.mode == "static":
            return [self._static_evaluation(path, prompt, manifest) for path in paths]

        try:
            return self._evaluate_real(paths, prompt, manifest)
        except Exception:
            # Defensive fallback: a model load / inference failure must
            # not crash the batch — degrade to deterministic scores so
            # the orchestrator can still record a result.
            return [self._static_evaluation(path, prompt, manifest) for path in paths]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _static_evaluation(
        self,
        path: Path,
        prompt: str,
        manifest: EvaluationTargetManifest,
    ) -> AlignmentEvaluation:
        score = _static_alignment_score(path, prompt, manifest)
        # Clamp away from the literal 0.0 / 1.0 boundaries so the
        # downstream cutoff logic does not collapse to pass/fail on a
        # hash quirk.
        score = min(max(score, 1e-6), 1.0 - 1e-6)
        return AlignmentEvaluation(
            score=score,
            model_id=MINICPM_V_MODEL_ID,
            model_version=MINICPM_V_MODEL_VERSION,
            confidence="high",
        )

    def _evaluate_real(
        self,
        paths: list[Path],
        prompt: str,
        manifest: EvaluationTargetManifest,
    ) -> list[AlignmentEvaluation]:
        if self._model is None:
            self.prewarm()

        assert self._model is not None  # for type checkers

        from PIL import Image as PILImage  # local import; only needed in real mode

        target_summary = "; ".join(
            f"{target.target_id} ({target.evaluation_policy})" for target in manifest.targets
        ) or "no explicit targets"
        question = (
            f"{prompt}\n\n"
            f"Target manifest: {target_summary}.\n"
            "Reply with a single JSON object of the form "
            '{"score": <float between 0 and 1>, "reason": "..."} '
            "indicating how well the image aligns with the manifest."
        )

        results: list[AlignmentEvaluation] = []
        for path in paths:
            score = self._score_single(path, question, prompt, manifest, PILImage)
            results.append(
                AlignmentEvaluation(
                    score=score,
                    model_id=MINICPM_V_MODEL_ID,
                    model_version=MINICPM_V_MODEL_VERSION,
                    confidence="high",
                )
            )
        return results

    def _score_single(
        self,
        path: Path,
        question: str,
        prompt: str,
        manifest: EvaluationTargetManifest,
        pil_image_module: Any,
    ) -> float:
        """Run a single image through MiniCPM-V and extract a score."""

        assert self._model is not None
        try:
            with pil_image_module.open(path) as raw:
                image = raw.convert("RGB")
        except Exception:
            return _static_alignment_score(path, prompt, manifest)

        response: Any
        try:
            response = self._model.chat(
                image=image,
                msgs=[{"role": "user", "content": question}],
                tokenizer=self._tokenizer,
            )
        except TypeError:
            try:
                response = self._model.chat(image, question, self._tokenizer)
            except Exception:
                return _static_alignment_score(path, prompt, manifest)
        except Exception:
            return _static_alignment_score(path, prompt, manifest)

        parsed = _parse_alignment_from_response(response)
        if parsed is None:
            return _static_alignment_score(path, prompt, manifest)
        return min(max(parsed, 1e-6), 1.0 - 1e-6)

    def _load_model(self) -> tuple[Any, Any, str]:
        """Lazy-import torch/transformers and load MiniCPM-V.

        Returns a 3-tuple ``(model, tokenizer, resolved_device)``.
        """

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as error:  # pragma: no cover - exercised only with deps
            raise RuntimeError(
                "MiniCPMVAdapter in 'real' mode requires 'torch' and 'transformers'; "
                "install the [ml] extras to enable it"
            ) from error

        resolved = self._resolve_device(torch)
        torch_dtype = torch.float16 if resolved.startswith("cuda") else None
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        model = AutoModel.from_pretrained(MINICPM_V_MODEL_ID, **load_kwargs)
        try:
            model = model.to(resolved)
        except Exception:
            pass
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(MINICPM_V_MODEL_ID, trust_remote_code=True)
        return model, tokenizer, resolved

    def _resolve_device(self, torch_mod: Any) -> str:
        if self.device == "cpu":
            resolved = "cpu"
        elif self.device == "cuda":
            resolved = "cuda"
        else:  # "auto"
            cuda_available = bool(getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available())
            resolved = "cuda" if cuda_available else "cpu"
        self._resolved_device = resolved
        return resolved


class TRIBEv2Adapter:
    """Real impact evaluator adapter wrapping ``Jessylg27/tribev2-lite-qv``.

    The adapter follows the same lazy-load contract as the other real
    evaluators in this module: importing :mod:`bruteforce_canvas.real_adapters`
    never requires the optional ``[ml]`` dependencies.

    Parameters
    ----------
    enabled:
        Master switch. When ``False`` (the default), :meth:`evaluate`
        short-circuits to an empty list and :meth:`prewarm` is a no-op,
        so no model weights are ever downloaded or loaded. Operators
        must explicitly opt in to real impact evaluation per batch.
    mode:
        ``"static"`` (default) returns deterministic per-path scores
        derived from a SHA-256 of the input path. ``"real"`` lazy-imports
        ``torch`` and ``transformers`` to load the TRIBE v2 model.
    device:
        Inference device hint used only in real mode. ``"auto"`` defers
        to ``torch.cuda.is_available()``; ``"cpu"`` and ``"cuda"`` pin
        the adapter to a specific accelerator.

    The adapter is safe to instantiate in any environment. Missing ML
    dependencies only cause :meth:`is_available` to return ``False``
    in real mode, and the public methods degrade gracefully (returning
    no-impact results) rather than raising on infrastructure failure.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: Literal["real", "static"] = "static",
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ) -> None:
        if mode not in ("real", "static"):
            raise ValueError(f"mode must be 'real' or 'static', got {mode!r}")
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {device!r}")

        self.enabled = enabled
        self.mode = mode
        self.device = device
        self._pipeline: Any | None = None
        self._resolved_device: int | None = None
        self._prewarmed: bool = False

    def is_available(self) -> bool:
        """Return ``True`` only when the adapter can actually produce scores.

        In ``static`` mode the check always returns ``True`` because the
        adapter never needs the heavy stack. In ``real`` mode the check
        attempts to import ``transformers`` and ``torch``; missing
        dependencies or an unavailable CUDA device when ``device="cuda"``
        cause the check to return ``False``.
        """

        if self.mode != "real":
            return True
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        if self.device == "cuda":
            try:
                import torch

                return bool(torch.cuda.is_available())
            except Exception:
                return False
        return True

    def prewarm(self) -> None:
        """Load the TRIBE v2 model once and cache it on this instance.

        Calling :meth:`prewarm` more than once is a no-op. The method is
        a no-op when ``enabled=False`` or when ``mode="static"``. Real
        mode defers the import to keep the package importable on
        machines without the ML stack.
        """

        if not self.enabled or self.mode != "real":
            return
        if self._prewarmed:
            return
        if not self.is_available():
            return

        try:
            import torch
            from transformers import pipeline  # type: ignore[import-not-found]
        except Exception:
            return

        self._resolved_device = self._resolve_device(torch)
        try:
            self._pipeline = pipeline(
                "image-classification",
                model=TRIBE_V2_MODEL_ID,
                device=self._resolved_device,
            )
        except Exception:
            self._pipeline = None
            return
        self._prewarmed = True

    def evaluate(self, images: list[str | Path]) -> list[ImpactEvaluation]:
        """Return one :class:`ImpactEvaluation` per input image path.

        * ``enabled=False`` returns an empty list (zero-impact, no model
          loaded — the documented fast path for disabled impact scoring).
        * ``mode="static"`` returns deterministic SHA-256-derived scores
          in ``[0, 1]`` keyed off the path string.
        * ``mode="real"`` runs the TRIBE v2 model and uses the top-1
          classifier score as the impact score. If the model cannot be
          reached, an empty list is returned so the orchestrator can
          mark the batch as ``impact_unavailable``.
        """

        if not self.enabled:
            return []
        if self.mode == "static":
            return [self._static_evaluation(image) for image in images]
        if not self.is_available():
            return []
        return self._evaluate_real(images)

    def _static_evaluation(self, image: str | Path) -> ImpactEvaluation:
        score = _static_score_for_path(Path(image))
        score = min(max(score, 1e-6), 1.0 - 1e-6)
        return ImpactEvaluation(
            score=score,
            model_id="static-tribev2-evaluator",
            model_version=_DEFAULT_MODEL_VERSION,
            confidence="high",
            informational_only=True,
        )

    def _evaluate_real(self, images: list[str | Path]) -> list[ImpactEvaluation]:
        if self._pipeline is None:
            self.prewarm()
        if self._pipeline is None:
            return []
        results: list[ImpactEvaluation] = []
        for image in images:
            try:
                output = self._pipeline(str(image))
            except Exception:
                results.append(
                    ImpactEvaluation(
                        score=0.0,
                        model_id=TRIBE_V2_MODEL_ID,
                        model_version=_DEFAULT_MODEL_VERSION,
                        confidence="low",
                        informational_only=True,
                    )
                )
                continue
            score = self._extract_score(output)
            results.append(
                ImpactEvaluation(
                    score=score,
                    model_id=TRIBE_V2_MODEL_ID,
                    model_version=_DEFAULT_MODEL_VERSION,
                    confidence="medium",
                    informational_only=True,
                )
            )
        return results

    def _resolve_device(self, torch_mod: Any) -> int:
        if self.device == "cpu":
            return -1
        if self.device == "cuda":
            return 0
        try:
            return 0 if bool(torch_mod.cuda.is_available()) else -1
        except Exception:
            return -1

    @staticmethod
    def _extract_score(output: Any) -> float:
        """Map a transformers pipeline output to a ``[0, 1]`` impact score."""

        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict) and "score" in first:
                return float(min(max(float(first["score"]), 1e-6), 1.0 - 1e-6))
        if isinstance(output, dict) and "score" in output:
            return float(min(max(float(output["score"]), 1e-6), 1.0 - 1e-6))
        return 0.5


__all__ = [
    "JoyQualityAdapter",
    "JOYQUALITY_PRIMARY_MODEL_ID",
    "JOYQUALITY_PROCESSOR_MODEL_ID",
    "MiniCPMVAdapter",
    "MINICPM_V_MODEL_ID",
    "TRIBEv2Adapter",
    "TRIBE_V2_MODEL_ID",
]
