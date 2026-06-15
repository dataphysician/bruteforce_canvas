from __future__ import annotations

import importlib.util
from typing import Any


def _module_exists(module_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _symbol_in_module(module_name: str, attribute_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        return hasattr(module, attribute_name)
    except (ImportError, ModuleNotFoundError, Exception):
        return False


def _count_class_defs(module_name: str) -> int:
    try:
        module = importlib.import_module(module_name)
        return sum(
            1
            for name in dir(module)
            if isinstance(getattr(module, name), type) and getattr(module, name).__module__ == module_name
        )
    except (ImportError, ModuleNotFoundError, Exception):
        return 0


def check_spec_01() -> list[str]:
    missing: list[str] = []

    expected_enums = [
        "EntityType",
        "ElementRole",
        "RelationType",
        "Importance",
        "Finish",
        "Condition",
        "Pattern",
        "MovementType",
        "MotionIntensity",
        "EnumMatchConfidence",
        "ActionSupportStatus",
        "ShotSize",
        "CameraAngle",
        "OpticCharacter",
        "LightingMood",
        "ColorTreatment",
        "Framing",
        "Guardrail",
        "CanonicalStatus",
        "EvidenceCategory",
    ]
    for enum_name in expected_enums:
        if not _symbol_in_module("bruteforce_canvas.prompt_enums", enum_name) and not _symbol_in_module(
            "bruteforce_canvas.prompt", enum_name
        ):
            missing.append(f"enum:{enum_name}")

    if not _module_exists("bruteforce_canvas.prompt_models"):
        missing.append("module:bruteforce_canvas.prompt_models")
    else:
        model_count = _count_class_defs("bruteforce_canvas.prompt_models")
        if model_count < 5:
            missing.append(f"prompt_models:expected>=5_models_found_{model_count}")

    expected_validators = [
        "validate_relation_compatibility",
        "validate_object_ownership",
        "validate_action_support",
        "validate_evidence_and_placeholders",
        "validate_cross_lane_coherence",
    ]
    for validator_name in expected_validators:
        if not _symbol_in_module("bruteforce_canvas.validators", validator_name):
            missing.append(f"validator:{validator_name}")

    expected_render_helpers = [
        "object_phrase",
        "relation_phrase",
        "action_phrase",
        "compile_positive_prompt",
        "render_cinematography",
        "compile_negative_prompt",
        "compile_prompt",
    ]
    if not _module_exists("bruteforce_canvas.prompt_render"):
        missing.append("module:bruteforce_canvas.prompt_render")
    else:
        for helper_name in expected_render_helpers:
            if not _symbol_in_module("bruteforce_canvas.prompt_render", helper_name):
                missing.append(f"render_helper:{helper_name}")

    return missing


def check_spec_02() -> list[str]:
    missing: list[str] = []

    try:
        from bruteforce_canvas.compatibility_pairs import PAIR_FAMILY_RULES

        if not PAIR_FAMILY_RULES:
            missing.append("compatibility_pairs:PAIR_FAMILY_RULES_empty")
    except (ImportError, ModuleNotFoundError, Exception):
        missing.append("module:bruteforce_canvas.compatibility_pairs")

    if not _module_exists("bruteforce_canvas.gp"):
        missing.append("module:bruteforce_canvas.gp")

    if not _symbol_in_module("bruteforce_canvas.learning", "detect_ood_enum"):
        missing.append("learning:detect_ood_enum_missing")

    return missing


def check_spec_03() -> list[str]:
    missing: list[str] = []

    if not _module_exists("bruteforce_canvas.generator_registry"):
        missing.append("module:bruteforce_canvas.generator_registry")
    else:
        if not _symbol_in_module("bruteforce_canvas.generator_registry", "GENERATOR_REGISTRY"):
            missing.append("generator_registry:GENERATOR_REGISTRY_missing")
        if not _symbol_in_module("bruteforce_canvas.generator_registry", "register"):
            missing.append("generator_registry:register_missing")
        if not _symbol_in_module("bruteforce_canvas.generator_registry", "register_generator"):
            missing.append("generator_registry:register_generator_missing")

    return missing


def check_spec_04() -> list[str]:
    missing: list[str] = []

    expected_adapters = [
        "JoyQualityAdapter",
        "MiniCPMVAdapter",
        "TRIBEv2Adapter",
    ]
    for adapter_name in expected_adapters:
        if not _symbol_in_module("bruteforce_canvas.real_adapters", adapter_name):
            missing.append(f"adapter:{adapter_name}")

    return missing


def check_spec_05() -> list[str]:
    missing: list[str] = []

    if not _symbol_in_module("bruteforce_canvas.balancer", "BayesianBalancer"):
        missing.append("balancer:BayesianBalancer_missing")

    if not _symbol_in_module("bruteforce_canvas.loop", "AsyncRunDriver"):
        missing.append("loop:AsyncRunDriver_missing")

    if not _module_exists("bruteforce_canvas.run_service"):
        missing.append("module:bruteforce_canvas.run_service")
    else:
        run_service = importlib.import_module("bruteforce_canvas.run_service")
        if not hasattr(run_service.RunService, "tick"):
            missing.append("RunService:tick_missing")
        if not hasattr(run_service.RunService, "_run_gate_chain"):
            missing.append("RunService:_run_gate_chain_missing")

    return missing


def check_spec_06() -> list[str]:
    missing: list[str] = []

    if not _module_exists("bruteforce_canvas.cli"):
        missing.append("module:bruteforce_canvas.cli")
    else:
        cli = importlib.import_module("bruteforce_canvas.cli")
        if not hasattr(cli, "main"):
            missing.append("cli:main_missing")
        has_stream = (
            hasattr(cli, "_run_stream_command") or hasattr(cli, "_make_stream_server") or hasattr(cli, "event_stream_from_records")
        )
        if not has_stream:
            missing.append("cli:stream_subcommand_missing")

    if not _module_exists("bruteforce_canvas.static_ui"):
        missing.append("module:bruteforce_canvas.static_ui")
    else:
        static_ui = importlib.import_module("bruteforce_canvas.static_ui")
        if not hasattr(static_ui, "_advanced_view"):
            missing.append("static_ui:_advanced_view_missing")
        if not hasattr(static_ui, "_error_state"):
            missing.append("static_ui:_error_state_missing")
        if not hasattr(static_ui, "render_workspace_html"):
            missing.append("static_ui:render_workspace_html_missing")

    return missing


def check_all() -> dict[str, Any]:
    missing: list[str] = []
    missing.extend(check_spec_01())
    missing.extend(check_spec_02())
    missing.extend(check_spec_03())
    missing.extend(check_spec_04())
    missing.extend(check_spec_05())
    missing.extend(check_spec_06())

    return {
        "compliant": len(missing) == 0,
        "missing": missing,
        "phases": "A-L",
    }

