from contextlib import nullcontext

import numpy as np

from bruteforce_canvas.asr import (
    ASRConfig,
    COHERE_TRANSCRIBE_MODEL_ID,
    LocalCohereTranscriber,
    normalize_audio_input,
)


def test_default_asr_config_targets_cohere_transcribe():
    config = ASRConfig.from_env()

    assert config.model_id == COHERE_TRANSCRIBE_MODEL_ID
    assert config.require_cuda is True


def test_normalize_audio_input_converts_stereo_int16_microphone_audio():
    left = np.linspace(0, 32767, num=4410, dtype=np.int16)
    right = np.full(4410, 32767, dtype=np.int16)
    samples = np.column_stack([left, right])

    sample_rate, normalized = normalize_audio_input((44100, samples))

    assert sample_rate == 16000
    assert normalized.dtype == np.float32
    assert normalized.shape == (1600,)
    assert 0.0 <= normalized.min() <= normalized.max() <= 1.0


def test_transcriber_uses_native_processor_generate_decode_flow():
    class FakeInputs(dict):
        def to(self, **kwargs):
            assert kwargs == {"device": "cuda:0", "dtype": "bf16"}
            return self

    class FakeProcessor:
        def __call__(self, audio_array, **kwargs):
            assert audio_array.shape == (1600,)
            assert kwargs["sampling_rate"] == 16000
            assert kwargs["return_tensors"] == "pt"
            assert kwargs["language"] == "en"
            assert kwargs["punctuation"] is True
            return FakeInputs({"input_features": "features", "audio_chunk_index": [(0, None)]})

        def decode(self, output_ids, **kwargs):
            assert output_ids == ["token_ids"]
            assert kwargs["skip_special_tokens"] is True
            assert kwargs["language"] == "en"
            assert kwargs["audio_chunk_index"] == [(0, None)]
            return ["transcribed prompt"]

    class FakeModel:
        device = "cuda:0"
        dtype = "bf16"

        def generate(self, **kwargs):
            assert kwargs["input_features"] == "features"
            assert kwargs["audio_chunk_index"] == [(0, None)]
            assert kwargs["max_new_tokens"] == 256
            return ["token_ids"]

    class FakeTorch:
        @staticmethod
        def device(value):
            return value

        @staticmethod
        def inference_mode():
            return nullcontext()

    transcriber = LocalCohereTranscriber()
    transcriber._load = lambda: (FakeProcessor(), FakeModel(), FakeTorch())  # type: ignore[method-assign]

    assert transcriber.transcribe((16000, np.zeros(1600, dtype=np.float32))) == "transcribed prompt"
