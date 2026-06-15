from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from bruteforce_canvas.evaluation import ImageEvaluationResult
from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE
from bruteforce_canvas.shared import CandidateId, CoordinateId, DocId, FeedbackAction, RunId, StrictModel, TargetManifestId


class RunConfig(StrictModel):
    run_id: RunId
    raw_user_prompt: str = Field(min_length=1)
    mode: Literal["continuous", "single_batch", "diagnostic"] = "continuous"
    iqa_cutoff: float = 0.55
    alignment_cutoff: float = 0.25
    human_iqa_cutoff: float = 0.70
    seed_bundle: list[int] = Field(default_factory=lambda: list(DEFAULT_SEED_BUNDLE))
    stall_window_seconds: int = 1800
    stall_min_promoted: int = 10
    promoted_high_watermark: int | None = None
    promoted_low_watermark: int | None = None
    metacognitive_impact_enabled: bool = False
    metacognitive_min_vram_gib: int = 24


class RunRuntimeState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    PAUSED_HIGH_WATERMARK = "paused_high_watermark"
    STOPPED = "stopped"
    BLOCKED = "blocked"


class RunCounters(StrictModel):
    generated_count: int = 0
    iqa_evaluated_count: int = 0
    vlm_evaluated_count: int = 0
    promoted_curated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    shredded_count: int = 0
    elapsed_seconds: int = 0


class RuntimeDecision(StrictModel):
    stop: bool = False
    next_state: RunRuntimeState | None = None
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class StallDiagnostic(StrictModel):
    run_id: RunId
    elapsed_seconds: int
    generated_count: int
    iqa_pass_count: int
    vlm_pass_count: int
    promoted_curated_count: int
    dominant_failure_types: list[str]
    most_penalized_enum_arms: list[str]
    most_penalized_combinations: list[str]
    restart_hints: list[str]
    threshold_changes_applied: bool = False


class CandidateState(StrictModel):
    candidate_id: CandidateId
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    coordinate_id: CoordinateId
    seed: int
    promoted: bool = False
    curated: bool = False
    demoted: bool = False
    blocked: bool = False
    disposition_history: list[str] = Field(default_factory=list)


class FeedbackResult(StrictModel):
    state: dict[str, Any]
    applied: bool
    candidate_id: CandidateId
    action: FeedbackAction
    learning_delta: dict[str, float]


class FeedbackPolicyError(ValueError):
    pass


class CandidateFeedbackResult(StrictModel):
    candidate: CandidateState
    applied: bool
    action: FeedbackAction
    signal_source: Literal["swipe_feedback"] = "swipe_feedback"
    automated_status: Literal["promoted_curated"] = "promoted_curated"
    effective_status: str
    learning_delta: dict[str, float]
    include_in_iqa_negative_dataset: bool


def _learning_delta(action: FeedbackAction) -> dict[str, float]:
    if action == FeedbackAction.ACCEPT:
        return {"alpha": 1.0, "beta": 0.0, "gp_affinity": 0.2}
    if action == FeedbackAction.REJECT:
        return {"alpha": 0.0, "beta": 1.0, "gp_affinity": -0.2}
    return {"alpha": 0.0, "beta": 2.0, "gp_affinity": -0.5}


def apply_feedback(state: dict[str, Any], *, candidate_id: str, action: FeedbackAction) -> FeedbackResult:
    feedback = dict(state.get("feedback", {}))
    key = f"{candidate_id}:{action.value}"
    delta = _learning_delta(action)
    if key in feedback:
        return FeedbackResult(
            state=state,
            applied=False,
            candidate_id=candidate_id,
            action=action,
            learning_delta=delta,
        )
    feedback[key] = {"candidate_id": candidate_id, "action": action.value, "learning_delta": delta}
    next_state = dict(state)
    next_state["feedback"] = feedback
    return FeedbackResult(
        state=next_state,
        applied=True,
        candidate_id=candidate_id,
        action=action,
        learning_delta=delta,
    )


def apply_candidate_feedback(candidate: CandidateState, action: FeedbackAction) -> CandidateFeedbackResult:
    if not candidate.promoted or not candidate.curated:
        raise FeedbackPolicyError("feedback is only accepted for promoted and curated candidates")

    delta = _learning_delta(action)
    if action == FeedbackAction.ACCEPT:
        return CandidateFeedbackResult(
            candidate=candidate.model_copy(update={"promoted": True, "curated": True}),
            applied=True,
            action=action,
            effective_status="accepted",
            learning_delta=delta,
            include_in_iqa_negative_dataset=False,
        )
    demoted = candidate.model_copy(update={"promoted": False, "curated": False, "demoted": True})
    return CandidateFeedbackResult(
        candidate=demoted,
        applied=True,
        action=action,
        effective_status="demoted_false_positive" if action == FeedbackAction.REJECT else "shredded_false_positive",
        learning_delta=delta,
        include_in_iqa_negative_dataset=action == FeedbackAction.REJECT,
    )


def apply_evaluation_disposition(candidate: CandidateState, result: ImageEvaluationResult) -> CandidateState:
    signal = result.disposition_signal.class_name
    history = [*candidate.disposition_history, signal]
    if signal == "passes_thresholds" and result.pass_flags.get("full", False):
        return candidate.model_copy(update={"promoted": True, "curated": True, "disposition_history": history})
    if signal == "demote_candidate":
        return candidate.model_copy(update={"promoted": False, "curated": False, "demoted": True, "disposition_history": history})
    if signal in {"hard_purge_invalid_artifact", "infrastructure_retry_no_semantic_penalty"}:
        return candidate.model_copy(update={"blocked": True, "disposition_history": history})
    return candidate.model_copy(update={"disposition_history": history})


def stall_guard_decision(config: RunConfig, counters: RunCounters) -> RuntimeDecision:
    if counters.elapsed_seconds < config.stall_window_seconds:
        return RuntimeDecision(reason="stall_window_open")
    if counters.promoted_curated_count >= config.stall_min_promoted:
        return RuntimeDecision(reason="stall_minimum_met")
    return RuntimeDecision(
        stop=True,
        next_state=RunRuntimeState.STOPPED,
        reason="stall_guard",
        details={
            "curated_count": counters.promoted_curated_count,
            "minimum_required_promoted": config.stall_min_promoted,
            "elapsed_seconds": counters.elapsed_seconds,
            "generated_count": counters.generated_count,
            "run_id": config.run_id,
        },
    )


def build_stall_diagnostic(
    config: RunConfig,
    counters: RunCounters,
    *,
    failure_types: list[str],
    penalized_enum_arms: dict[str, float],
    penalized_combos: dict[str, float],
) -> StallDiagnostic:
    dominant = [item for item, _count in Counter(failure_types).most_common()]
    enum_arms = [item for item, _score in sorted(penalized_enum_arms.items(), key=lambda pair: pair[1])]
    combos = [item for item, _score in sorted(penalized_combos.items(), key=lambda pair: pair[1])]
    hints: list[str] = []
    if "quality_below_cutoff" in dominant:
        hints.append("consider_lowering_iqa_cutoff")
    if "alignment_below_cutoff" in dominant:
        hints.append("consider_lowering_alignment_cutoff")
    if any(item in dominant for item in {"missing_locked_element", "missing_locked_relation", "missing_action_target"}):
        hints.append("clarify_or_rehash_prompt")
    if enum_arms or combos or any(item.startswith("missing_") for item in dominant):
        hints.append("narrow_lhs_enum_space")
    if not hints:
        hints.append("review_prompt_and_thresholds")

    return StallDiagnostic(
        run_id=config.run_id,
        elapsed_seconds=counters.elapsed_seconds,
        generated_count=counters.generated_count,
        iqa_pass_count=counters.vlm_evaluated_count,
        vlm_pass_count=counters.promoted_curated_count,
        promoted_curated_count=counters.promoted_curated_count,
        dominant_failure_types=dominant,
        most_penalized_enum_arms=enum_arms,
        most_penalized_combinations=combos,
        restart_hints=hints,
        threshold_changes_applied=False,
    )


def watermark_decision(
    config: RunConfig,
    counters: RunCounters,
    current_state: RunRuntimeState,
) -> RuntimeDecision:
    if (
        config.promoted_high_watermark is not None
        and current_state == RunRuntimeState.RUNNING
        and counters.promoted_curated_count >= config.promoted_high_watermark
    ):
        return RuntimeDecision(
            next_state=RunRuntimeState.PAUSED_HIGH_WATERMARK,
            reason="high_watermark_reached",
            details={"promoted_curated_count": counters.promoted_curated_count},
        )
    if (
        config.promoted_low_watermark is not None
        and current_state == RunRuntimeState.PAUSED_HIGH_WATERMARK
        and counters.promoted_curated_count < config.promoted_low_watermark
    ):
        return RuntimeDecision(
            next_state=RunRuntimeState.RUNNING,
            reason="low_watermark_reached",
            details={"promoted_curated_count": counters.promoted_curated_count},
        )
    return RuntimeDecision(next_state=current_state, reason="no_watermark_transition")
