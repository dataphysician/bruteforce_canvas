from bruteforce_canvas.orchestration import RunConfig
from bruteforce_canvas.scheduler import HardwareTier, plan_evaluator_stages


def test_base_tier_disables_optional_impact_even_if_requested():
    config = RunConfig(
        run_id="run_001",
        raw_user_prompt="a bowl on a table",
        metacognitive_impact_enabled=True,
        metacognitive_min_vram_gib=24,
    )

    plan = plan_evaluator_stages(config, HardwareTier(vram_gib=12))

    assert plan.iqa is True
    assert plan.vlm is True
    assert plan.impact is False
    assert plan.reasons["impact"] == "insufficient_vram"


def test_impact_stage_enabled_only_when_policy_and_hardware_allow():
    enabled = RunConfig(
        run_id="run_001",
        raw_user_prompt="a bowl on a table",
        metacognitive_impact_enabled=True,
        metacognitive_min_vram_gib=24,
    )
    disabled = enabled.model_copy(update={"metacognitive_impact_enabled": False})

    enabled_plan = plan_evaluator_stages(enabled, HardwareTier(vram_gib=24))
    disabled_plan = plan_evaluator_stages(disabled, HardwareTier(vram_gib=48))

    assert enabled_plan.impact is True
    assert disabled_plan.impact is False
    assert disabled_plan.reasons["impact"] == "disabled_by_run_config"


def test_hardware_plan_reports_batch_size_by_tier():
    base = plan_evaluator_stages(
        RunConfig(run_id="run_001", raw_user_prompt="a bowl on a table"),
        HardwareTier(vram_gib=12),
    )
    high = plan_evaluator_stages(
        RunConfig(run_id="run_001", raw_user_prompt="a bowl on a table"),
        HardwareTier(vram_gib=48),
    )

    assert base.max_iqa_batch_size < high.max_iqa_batch_size
    assert base.max_vlm_batch_size < high.max_vlm_batch_size
