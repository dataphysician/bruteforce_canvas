from __future__ import annotations

import importlib
import sys
from pathlib import Path

from bruteforce_canvas.app_config import DeviceConfig, DeviceKind
from bruteforce_canvas.app_factory import prewarm_all
from bruteforce_canvas.evaluation import StaticIQAAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import StubGeneratorAdapter
from bruteforce_canvas.orchestration import RunConfig
from bruteforce_canvas.persistence import JsonlEventStore
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.telemetry import VRAMTelemetry, collect_vram_telemetry, measure_vram_gib
from bruteforce_canvas.worker import PersistentSeedSweepWorker


def test_measure_vram_gib_returns_zero_when_pynvml_unavailable(monkeypatch):
    for module_name in [name for name in list(sys.modules) if name == "pynvml" or name.startswith("pynvml.")]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "pynvml" or name.startswith("pynvml."):
            raise ImportError("pynvml is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    importlib.reload(importlib.import_module("bruteforce_canvas.telemetry"))

    assert measure_vram_gib() == 0.0
    snapshot = collect_vram_telemetry()
    assert snapshot.used_gib == 0.0
    assert snapshot.total_gib == 0.0
    assert snapshot.free_gib == 0.0


def test_vram_telemetry_model_has_required_shape():
    sample = VRAMTelemetry(total_gib=24.0, used_gib=12.5, free_gib=11.5, timestamp=123.456)

    dumped = sample.model_dump()
    assert set(dumped) == {"total_gib", "used_gib", "free_gib", "timestamp"}
    assert dumped["total_gib"] == 24.0
    assert dumped["used_gib"] == 12.5
    assert dumped["free_gib"] == 11.5
    assert dumped["timestamp"] == 123.456

    rebuilt = VRAMTelemetry.model_validate(dumped)
    assert rebuilt == sample


def test_run_service_snapshot_includes_vram_telemetry(tmp_path: Path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    service = RunService(
        config=RunConfig(run_id="run_001", raw_user_prompt="telemetry snapshot test"),
        store=store,
        worker=PersistentSeedSweepWorker(
            store=store,
            generator=StubGeneratorAdapter(),
            iqa=StaticIQAAdapter(scores=[0.9]),
            vlm=StaticVLMAdapter(scores=[0.9]),
        ),
    )

    snap = service.snapshot()

    assert snap.vram_gib == measure_vram_gib()
    assert snap.vram_gib >= 0.0
    assert hasattr(snap, "state")
    assert hasattr(snap, "counters")
    assert hasattr(snap, "pending_count")
    assert hasattr(snap, "snapshot_at")


def test_device_config_exposes_cpu_cuda_and_auto():
    for value in ("cpu", "cuda", "auto"):
        config = DeviceConfig(device=value)
        assert config.device == value

    try:
        DeviceConfig.model_validate({"device": "tpu"})
    except ValueError as error:
        assert "device" in str(error)
    else:
        raise AssertionError("expected validation error for unknown device")


def test_device_kind_enum_lists_supported_devices():
    assert DeviceKind.CPU.value == "cpu"
    assert DeviceKind.CUDA.value == "cuda"
    assert DeviceKind.AUTO.value == "auto"


def test_prewarm_all_invokes_prewarm_on_each_adapter():
    class Recorder:
        def __init__(self) -> None:
            self.calls = 0

        def prewarm(self) -> None:
            self.calls += 1

    generator = Recorder()
    iqa = Recorder()
    vlm = Recorder()
    impact = Recorder()

    prewarm_all(generator=generator, iqa=iqa, vlm=vlm, impact=impact)

    assert generator.calls == 1
    assert iqa.calls == 1
    assert vlm.calls == 1
    assert impact.calls == 1


def test_prewarm_all_skips_adapters_without_prewarm():
    class NoPrewarm:
        pass

    generator = StubGeneratorAdapter()
    prewarm_all(generator=generator, iqa=None, vlm=None, impact=NoPrewarm())
    assert generator.prewarm_count == 1
