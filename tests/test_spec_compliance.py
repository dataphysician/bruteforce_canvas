import importlib.util
import sys
import types
from typing import Any

import pytest

from bruteforce_canvas.spec_compliance import (
    check_all,
    check_spec_01,
    check_spec_02,
    check_spec_03,
    check_spec_04,
    check_spec_05,
    check_spec_06,
)


def test_check_all_returns_expected_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    result = check_all()
    assert set(result.keys()) == {"compliant", "missing", "phases"}
    assert isinstance(result["compliant"], bool)
    assert isinstance(result["missing"], list)
    assert all(isinstance(item, str) for item in result["missing"])
    assert result["phases"] == "A-L"


def test_check_spec_01_structure_and_render_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = check_spec_01()
    assert isinstance(missing, list)
    assert all(isinstance(item, str) for item in missing)
    assert "module:bruteforce_canvas.prompt_models" not in missing
    assert "module:bruteforce_canvas.prompt_render" not in missing


def test_check_spec_02_passes_with_existing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = check_spec_02()
    assert isinstance(missing, list)
    assert all(isinstance(item, str) for item in missing)
    assert "module:bruteforce_canvas.compatibility_pairs" not in missing
    assert "module:bruteforce_canvas.gp" not in missing
    assert "learning:detect_ood_enum_missing" not in missing


def test_check_spec_03_passes_with_existing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = check_spec_03()
    assert isinstance(missing, list)
    assert all(isinstance(item, str) for item in missing)
    assert "module:bruteforce_canvas.generator_registry" not in missing


def test_check_spec_04_passes_with_existing_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = check_spec_04()
    assert isinstance(missing, list)
    assert all(isinstance(item, str) for item in missing)
    for adapter in ["JoyQualityAdapter", "MiniCPMVAdapter", "TRIBEv2Adapter"]:
        assert f"adapter:{adapter}" not in missing


def test_check_spec_05_reflects_balancer_location(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = check_spec_05()
    assert isinstance(missing, list)
    assert all(isinstance(item, str) for item in missing)
    assert "loop:AsyncRunDriver_missing" not in missing
    assert "RunService:tick_missing" not in missing
    assert "RunService:_run_gate_chain_missing" not in missing


def test_check_spec_06_passes_with_cli_and_static_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = check_spec_06()
    assert isinstance(missing, list)
    assert all(isinstance(item, str) for item in missing)
    assert "module:bruteforce_canvas.cli" not in missing
    assert "module:bruteforce_canvas.static_ui" not in missing
    assert "static_ui:_advanced_view_missing" not in missing
    assert "static_ui:_error_state_missing" not in missing
    assert "static_ui:render_workspace_html_missing" not in missing
