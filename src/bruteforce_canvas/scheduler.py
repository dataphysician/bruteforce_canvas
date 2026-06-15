from __future__ import annotations

from pydantic import Field

from bruteforce_canvas.orchestration import RunConfig
from bruteforce_canvas.shared import StrictModel


class HardwareTier(StrictModel):
    vram_gib: int
    cuda_available: bool = True


class EvaluatorStagePlan(StrictModel):
    iqa: bool = True
    vlm: bool = True
    impact: bool = False
    max_iqa_batch_size: int
    max_vlm_batch_size: int
    reasons: dict[str, str] = Field(default_factory=dict)


def plan_evaluator_stages(config: RunConfig, hardware: HardwareTier) -> EvaluatorStagePlan:
    high_tier = hardware.cuda_available and hardware.vram_gib >= config.metacognitive_min_vram_gib
    if not config.metacognitive_impact_enabled:
        impact = False
        impact_reason = "disabled_by_run_config"
    elif not high_tier:
        impact = False
        impact_reason = "insufficient_vram"
    else:
        impact = True
        impact_reason = "enabled"

    if hardware.vram_gib >= 24:
        iqa_batch = 32
        vlm_batch = 8
    else:
        iqa_batch = 8
        vlm_batch = 2

    return EvaluatorStagePlan(
        iqa=True,
        vlm=True,
        impact=impact,
        max_iqa_batch_size=iqa_batch,
        max_vlm_batch_size=vlm_batch,
        reasons={"impact": impact_reason},
    )
