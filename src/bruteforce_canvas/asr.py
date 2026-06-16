from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field

from bruteforce_canvas.shared import StrictModel


COHERE_TRANSCRIBE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_ASR_LANGUAGE = "en"
DEFAULT_ASR_MAX_NEW_TOKENS = 256
TARGET_ASR_SAMPLE_RATE = 16000


class ASRConfig(StrictModel):
    model_id: str = COHERE_TRANSCRIBE_MODEL_ID
    cache_dir: str | None = None
    language: str = DEFAULT_ASR_LANGUAGE
    punctuation: bool = True
    max_new_tokens: int = Field(default=DEFAULT_ASR_MAX_NEW_TOKENS, ge=1)
    device_map: str = "auto"
    require_cuda: bool = True
    local_files_only: bool = False
    token: str | None = None

    @classmethod
    def from_env(cls) -> "ASRConfig":
        return cls(
            model_id=os.getenv("BC_ASR_MODEL_ID", COHERE_TRANSCRIBE_MODEL_ID),
            cache_dir=os.getenv("BC_ASR_CACHE_DIR") or None,
            language=os.getenv("BC_ASR_LANGUAGE", DEFAULT_ASR_LANGUAGE),
            punctuation=os.getenv("BC_ASR_PUNCTUATION", "true").lower() not in {"0", "false", "no"},
            max_new_tokens=int(os.getenv("BC_ASR_MAX_NEW_TOKENS", str(DEFAULT_ASR_MAX_NEW_TOKENS))),
            device_map=os.getenv("BC_ASR_DEVICE_MAP", "auto"),
            require_cuda=os.getenv("BC_ASR_REQUIRE_CUDA", "true").lower() not in {"0", "false", "no"},
            local_files_only=os.getenv("BC_ASR_LOCAL_FILES_ONLY", "false").lower() in {"1", "true", "yes"},
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None,
        )


def cache_asr_weights(config: ASRConfig | None = None) -> str:
    from huggingface_hub import snapshot_download

    config = config or ASRConfig.from_env()
    return snapshot_download(
        repo_id=config.model_id,
        cache_dir=config.cache_dir,
        token=config.token,
        local_files_only=config.local_files_only,
        allow_patterns=[
            "*.bin",
            "*.json",
            "*.model",
            "*.py",
            "*.safetensors",
            "*.txt",
            "*.yaml",
            "*.yml",
            "*.tiktoken",
        ],
        ignore_patterns=[".eval_results/*", "*.md"],
    )


def normalize_audio_input(audio: object) -> tuple[int, object]:
    if audio is None:
        raise ValueError("no audio was provided")
    if isinstance(audio, str | Path):
        from transformers.audio_utils import load_audio

        return TARGET_ASR_SAMPLE_RATE, load_audio(str(audio), sampling_rate=TARGET_ASR_SAMPLE_RATE)
    if not isinstance(audio, tuple) or len(audio) != 2:
        raise TypeError("expected Gradio audio as (sample_rate, samples) or an audio filepath")

    sample_rate, samples = audio
    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover - numpy is provided by Gradio/Transformers.
        raise RuntimeError("numpy is required for microphone audio normalization") from error

    array = np.asarray(samples)
    if np.issubdtype(array.dtype, np.integer):
        max_value = max(1, np.iinfo(array.dtype).max)
        array = array.astype("float32") / max_value
    else:
        array = array.astype("float32")
    if array.ndim == 2:
        array = array.mean(axis=1)
    sample_rate = int(sample_rate)
    if sample_rate != TARGET_ASR_SAMPLE_RATE:
        array = _resample_audio(array, sample_rate, TARGET_ASR_SAMPLE_RATE)
    array = np.clip(array, -1.0, 1.0).astype("float32")
    return TARGET_ASR_SAMPLE_RATE, array


def _resample_audio(array: object, source_rate: int, target_rate: int) -> object:
    try:
        import librosa

        return librosa.resample(array, orig_sr=source_rate, target_sr=target_rate).astype("float32")
    except ModuleNotFoundError:
        pass

    import numpy as np

    source = np.asarray(array, dtype="float32")
    if source.size == 0:
        return source
    target_size = max(1, int(round(source.size * target_rate / source_rate)))
    source_positions = np.linspace(0.0, 1.0, num=source.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, source).astype("float32")


class LocalCohereTranscriber:
    def __init__(self, config: ASRConfig | None = None) -> None:
        self.config = config or ASRConfig.from_env()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def transcribe(self, audio: object) -> str:
        sample_rate, audio_array = normalize_audio_input(audio)
        processor, model, torch = self._load()
        inputs = processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language=self.config.language,
            punctuation=self.config.punctuation,
        )
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs.to(device=model.device, dtype=model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        return self._decode(processor, output_ids, audio_chunk_index).strip()

    def _load(self) -> tuple[Any, Any, Any]:
        if self._processor is not None and self._model is not None and self._torch is not None:
            return self._processor, self._model, self._torch

        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        if self.config.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("Cohere Transcribe ASR requires CUDA; set BC_ASR_REQUIRE_CUDA=false to override")

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
            token=self.config.token,
        )
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            self.config.model_id,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
            token=self.config.token,
            device_map=self.config.device_map,
        )
        self._model.eval()
        self._torch = torch
        return self._processor, self._model, self._torch

    def _decode(self, processor: Any, output_ids: Any, audio_chunk_index: Any) -> str:
        kwargs: dict[str, object] = {
            "skip_special_tokens": True,
            "language": self.config.language,
        }
        if audio_chunk_index is not None:
            kwargs["audio_chunk_index"] = audio_chunk_index
        decoded = processor.decode(output_ids, **kwargs)
        if isinstance(decoded, str):
            return decoded
        if decoded:
            return str(decoded[0])
        return ""

_DEFAULT_TRANSCRIBER: LocalCohereTranscriber | None = None


def default_transcriber() -> LocalCohereTranscriber:
    global _DEFAULT_TRANSCRIBER
    if _DEFAULT_TRANSCRIBER is None:
        _DEFAULT_TRANSCRIBER = LocalCohereTranscriber()
    return _DEFAULT_TRANSCRIBER
