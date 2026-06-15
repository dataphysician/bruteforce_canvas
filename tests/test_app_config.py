from pathlib import Path

from bruteforce_canvas.app_config import AppConfig, GeneratorKind, load_app_config


def test_load_app_config_defaults_are_safe_for_local_tdd(monkeypatch):
    monkeypatch.delenv("BC_GENERATOR", raising=False)
    monkeypatch.delenv("BC_EVENT_STORE", raising=False)
    monkeypatch.delenv("BC_OPENAI_MODEL", raising=False)

    config = load_app_config({})

    assert config.generator.kind == GeneratorKind.STUB
    assert config.event_store_path == Path("runtime/events.jsonl")
    assert config.openai_model == "gpt-4.1-mini"
    assert config.run.iqa_cutoff == 0.55
    assert config.hardware.vram_gib == 0


def test_load_app_config_reads_environment_overrides():
    config = load_app_config(
        {
            "BC_GENERATOR": "bonsai",
            "BC_EVENT_STORE": "/tmp/bruteforce/events.jsonl",
            "BC_OPENAI_MODEL": "gpt-4.1",
            "BC_IQA_CUTOFF": "0.72",
            "BC_ALIGNMENT_CUTOFF": "0.31",
            "BC_VRAM_GIB": "24",
            "BC_BONSAI_MODEL_ROOT": "/models/bonsai",
            "BC_PROMOTED_HIGH_WATERMARK": "500",
            "BC_PROMOTED_LOW_WATERMARK": "120",
        }
    )

    assert config.generator.kind == GeneratorKind.BONSAI
    assert config.event_store_path == Path("/tmp/bruteforce/events.jsonl")
    assert config.openai_model == "gpt-4.1"
    assert config.run.iqa_cutoff == 0.72
    assert config.run.alignment_cutoff == 0.31
    assert config.hardware.vram_gib == 24
    assert config.generator.bonsai_model_root == Path("/models/bonsai")
    assert config.run.promoted_high_watermark == 500
    assert config.run.promoted_low_watermark == 120


def test_app_config_rejects_invalid_generator_kind():
    try:
        AppConfig.model_validate({"generator": {"kind": "diffusers"}})
    except ValueError as error:
        assert "generator" in str(error)
    else:
        raise AssertionError("expected validation error")
