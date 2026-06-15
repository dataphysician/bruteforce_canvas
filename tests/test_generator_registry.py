from __future__ import annotations

from pathlib import Path

import pytest

from bruteforce_canvas.app_config import AppConfig, GeneratorConfig
from bruteforce_canvas.app_factory import build_generator_adapter
from bruteforce_canvas.generation import BonsaiTernaryAdapter, StubGeneratorAdapter
from bruteforce_canvas.generator_registry import (
    BUILDER_INCLUDES,
    GENERATOR_REGISTRY,
    register,
    register_generator,
)


def test_registration_via_decorator() -> None:
    @register_generator("fancy")
    def build_fancy(config: object) -> str:
        return "fancy"

    assert "fancy" in GENERATOR_REGISTRY
    assert GENERATOR_REGISTRY["fancy"] is build_fancy


def test_lookup_returns_registered_factory() -> None:
    factory = GENERATOR_REGISTRY.get("stub")
    assert factory is not None
    assert callable(factory)


def test_unknown_kind_falls_back_to_stub() -> None:
    factory = GENERATOR_REGISTRY.get("totally_unknown", GENERATOR_REGISTRY["stub"])
    adapter = factory(None)
    assert isinstance(adapter, StubGeneratorAdapter)


def test_stub_factory_works() -> None:
    factory = GENERATOR_REGISTRY["stub"]
    adapter = factory(None)
    assert isinstance(adapter, StubGeneratorAdapter)


def test_bonsai_factory_works(tmp_path: Path) -> None:
    config = AppConfig(
        generator=GeneratorConfig(
            kind="bonsai",
            bonsai_model_root=tmp_path / "bonsai",
            bonsai_triton_cache_dir=tmp_path / ".triton",
        )
    )
    adapter = build_generator_adapter(config)
    assert isinstance(adapter, BonsaiTernaryAdapter)
    assert adapter.config.model_root == tmp_path / "bonsai"


def test_custom_external_registration_works() -> None:
    def my_factory(config: object) -> str:
        return "custom"

    register("custom", my_factory)
    assert GENERATOR_REGISTRY["custom"] is my_factory
