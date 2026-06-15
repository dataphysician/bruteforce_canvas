from __future__ import annotations

import time
from collections import deque

from bruteforce_canvas.evaluation import ImageEvaluationResult
from bruteforce_canvas.loop import LoopAction, LoopDecision, next_loop_action
from bruteforce_canvas.orchestration import (
    CandidateFeedbackResult,
    CandidateState,
    RunConfig,
    RunCounters,
    RunRuntimeState,
    RuntimeSnapshot,
    apply_candidate_feedback,
    build_stall_diagnostic,
)
from bruteforce_canvas.persistence import PERSISTENCE_VERSION, JsonlEventStore, PersistenceRecord, reconstruct_run_state
from bruteforce_canvas.shared import FeedbackAction
from bruteforce_canvas.telemetry import measure_vram_gib
from bruteforce_canvas.ui import UIEvent
from bruteforce_canvas.worker import PersistentSeedSweepWorker, SeedSweepWorkItem


class RunService:
    def __init__(
        self,
        *,
        config: RunConfig,
        store: JsonlEventStore,
        worker: PersistentSeedSweepWorker,
    ) -> None:
        self.config = config
        self.store = store
        self.worker = worker
        self._pending: deque[SeedSweepWorkItem] = deque()
        self._state = RunRuntimeState.RUNNING
        self._stop_requested = False
        self._transition_counter = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def state(self) -> RunRuntimeState:
        return self._state

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            state=self._state,
            counters=self._counters_from_store(),
            pending_count=len(self._pending),
            vram_gib=measure_vram_gib(),
            snapshot_at=time.time(),
        )

    def enqueue(self, item: SeedSweepWorkItem) -> None:
        self._pending.append(item)

    def request_stop(self) -> None:
        self._stop_requested = True

    def request_pause(self) -> None:
        if self._state != RunRuntimeState.STOPPED:
            self._state = RunRuntimeState.PAUSED

    def resume(self) -> None:
        if self._state == RunRuntimeState.PAUSED:
            self._state = RunRuntimeState.RUNNING

    def handle_ui_event(self, event: UIEvent) -> object | None:
        if event.run_id != self.config.run_id:
            raise ValueError(f"event run_id {event.run_id} does not match service run_id {self.config.run_id}")
        if event.event_type == "run_pause_intent":
            self.request_pause()
            self._persist_ui_event(event)
            return None
        if event.event_type == "run_stop_intent":
            self.request_stop()
            self._persist_ui_event(event)
            return None
        if event.event_type == "pre_run_begin":
            self.resume()
            self._persist_ui_event(event)
            return None
        if event.event_type == "pre_run_cancel":
            self.request_pause()
            self._persist_ui_event(event)
            return None
        if event.event_type == "feedback_submitted":
            self._persist_ui_event(event)
            return self.submit_feedback(
                candidate_id=str(event.payload["candidate_id"]),
                action=FeedbackAction(event.payload["action"]),
            )
        self._persist_ui_event(event)
        return None

    def submit_feedback(self, *, candidate_id: str, action: FeedbackAction) -> CandidateFeedbackResult:
        evaluation = self._evaluation_for_candidate(candidate_id)
        candidate = CandidateState(
            candidate_id=candidate_id,
            run_id=evaluation.run_id,
            prompt_document_id=evaluation.prompt_document_id,
            target_manifest_id=evaluation.target_manifest_id,
            coordinate_id=evaluation.coordinate_id,
            seed=evaluation.seed,
            promoted=evaluation.pass_flags.get("full", False),
            curated=evaluation.disposition_signal.class_name == "passes_thresholds"
            and evaluation.pass_flags.get("full", False),
        )
        result = apply_candidate_feedback(candidate, action)
        if self._has_feedback_for_candidate(candidate_id):
            return result.model_copy(update={"applied": False})
        key = f"feedback:{candidate_id}:{action.value}"
        feedback_append = self.store.append(
            PersistenceRecord(
                record_id=key,
                record_type="feedback",
                run_id=evaluation.run_id,
                prompt_document_id=evaluation.prompt_document_id,
                target_manifest_id=evaluation.target_manifest_id,
                coordinate_id=evaluation.coordinate_id,
                candidate_id=candidate_id,
                seed=evaluation.seed,
                idempotency_key=key,
                payload={
                    "feedback_action": action.value,
                    "feedback_scope": "pre_curated_candidate",
                    "signal_source": result.signal_source,
                    "automated_status": result.automated_status,
                    "effective_status": result.effective_status,
                    "learning_delta": result.learning_delta,
                    "include_in_iqa_negative_dataset": result.include_in_iqa_negative_dataset,
                    "persistence_version": PERSISTENCE_VERSION,
                },
            )
        )
        self.store.append(
            PersistenceRecord(
                record_id=f"feedback_learning_delta:{candidate_id}:{action.value}",
                record_type="feedback_learning_delta",
                run_id=evaluation.run_id,
                prompt_document_id=evaluation.prompt_document_id,
                target_manifest_id=evaluation.target_manifest_id,
                coordinate_id=evaluation.coordinate_id,
                candidate_id=candidate_id,
                seed=evaluation.seed,
                idempotency_key=f"feedback_learning_delta:{candidate_id}:{action.value}",
                payload={
                    "feedback_action": action.value,
                    "learning_delta": result.learning_delta,
                    "source": result.signal_source,
                    "learning_signal_source": result.signal_source,
                    "persistence_version": PERSISTENCE_VERSION,
                },
            )
        )
        return result.model_copy(update={"applied": feedback_append.written})

    def _has_feedback_for_candidate(self, candidate_id: str) -> bool:
        return any(
            record.record_type == "feedback" and record.candidate_id == candidate_id
            for record in self.store.replay()
        )

    def tick(self) -> LoopDecision:
        counters = self._counters_from_store()
        decision = next_loop_action(
            self.config,
            counters,
            self._state,
            stop_requested=self._stop_requested,
            has_pending_coordinates=bool(self._pending),
            has_pending_candidates=False,
        )
        self._persist_transition(decision)
        if decision.reason == "stall_guard":
            self._persist_stall_diagnostic(counters)
        self._state = decision.next_state
        if decision.action == LoopAction.GENERATE_PENDING_COORDINATE and self._pending:
            item = self._pending.popleft()
            self.worker.run_seed_sweep(item)
        return decision

    def _counters_from_store(self) -> RunCounters:
        records = self.store.replay()
        if not records:
            return RunCounters()
        state = reconstruct_run_state(records)
        return RunCounters(
            generated_count=state.generated_count,
            iqa_evaluated_count=state.iqa_evaluated_count,
            vlm_evaluated_count=state.vlm_evaluated_count,
            promoted_curated_count=state.promoted_curated_count,
            accepted_count=state.accepted_count,
            rejected_count=state.rejected_count,
            shredded_count=state.shredded_count,
            elapsed_seconds=state.elapsed_seconds,
        )

    def _evaluation_for_candidate(self, candidate_id: str) -> ImageEvaluationResult:
        for record in reversed(self.store.replay()):
            if record.record_type == "image_evaluation" and record.candidate_id == candidate_id:
                return ImageEvaluationResult.model_validate(record.payload)
        candidate = next(
            (
                record
                for record in reversed(self.store.replay())
                if record.record_type == "candidate_record" and record.candidate_id == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"candidate {candidate_id} has not been generated")
        return ImageEvaluationResult(
            candidate_id=candidate.candidate_id,
            image_path=str(candidate.payload["image_path"]),
            seed=int(candidate.seed or candidate.payload["seed"]),
            coordinate_id=candidate.coordinate_id,
            run_id=candidate.run_id,
            prompt_document_id=candidate.prompt_document_id,
            target_manifest_id=candidate.target_manifest_id,
            file_valid=bool(candidate.payload.get("file_valid", False)),
            quality={"score": 0.0, "confidence": "low"},
            alignment={"score": 0.0, "confidence": "low"},
            pass_flags={"quality": False, "alignment": False, "full": False},
            failure_types=["evaluator_unavailable"],
            localized_blame=[],
            disposition_signal={
                "class_name": "fail_persist_for_learning",
                "confidence": "high",
                "reasons": ["candidate has not passed evaluation"],
            },
            confidence="low",
        )

    def _persist_transition(self, decision: LoopDecision) -> None:
        self._transition_counter += 1
        self.store.append(
            PersistenceRecord(
                record_id=f"loop_transition:{self._transition_counter}",
                record_type="loop_transition",
                run_id=self.config.run_id,
                payload={
                    "action": str(decision.action),
                    "reason": decision.reason,
                    "next_state": str(decision.next_state),
                },
            )
        )

    def _persist_stall_diagnostic(self, counters: RunCounters) -> None:
        records = self.store.replay()
        failure_types: list[str] = []
        penalized_enum_arms: dict[str, float] = {}
        penalized_combos: dict[str, float] = {}
        for record in records:
            if record.record_type == "image_evaluation":
                failure_types.extend(str(item) for item in record.payload.get("failure_types", []))
            elif record.record_type == "learning_delta":
                for key, value in dict(record.payload.get("enum_arms", {})).items():
                    alpha = float(value.get("alpha", 1.0))
                    beta = float(value.get("beta", 1.0))
                    penalized_enum_arms[str(key)] = alpha / (alpha + beta)
                for key, value in dict(record.payload.get("combo_affinities", {})).items():
                    penalized_combos[str(key)] = float(value.get("gp_mean", 0.0))

        diagnostic = build_stall_diagnostic(
            self.config,
            counters,
            failure_types=failure_types,
            penalized_enum_arms=penalized_enum_arms,
            penalized_combos=penalized_combos,
        )
        self.store.append(
            PersistenceRecord(
                record_id=f"stall_diagnostic:{self.config.run_id}",
                record_type="stall_diagnostic",
                run_id=self.config.run_id,
                idempotency_key=f"stall_diagnostic:{self.config.run_id}",
                payload=diagnostic.model_dump(),
            )
        )

    def _persist_ui_event(self, event: UIEvent) -> None:
        self._transition_counter += 1
        self.store.append(
            PersistenceRecord(
                record_id=f"ui_event:{event.event_id}",
                record_type="ui_event",
                run_id=self.config.run_id,
                idempotency_key=event.event_id,
                payload={
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    **event.payload,
                },
            )
        )
