from __future__ import annotations

from pydantic import Field

from bruteforce_canvas.cli import candidate_cards_from_records, detail_report_from_records
from bruteforce_canvas.orchestration import RunRuntimeState
from bruteforce_canvas.persistence import PersistenceRecord, reconstruct_run_state
from bruteforce_canvas.run_service import RunService
from bruteforce_canvas.shared import StrictModel
from bruteforce_canvas.ui import (
    CandidateCard,
    DetailReport,
    DiagnosticsReadModel,
    RunWorkspaceReadModel,
    UIEvent,
    catalogue_default_items,
)


class WorkspaceBundle(StrictModel):
    workspace: RunWorkspaceReadModel
    catalogue: list[CandidateCard] = Field(default_factory=list)
    selected: DetailReport | None = None


class RunAppController:
    def __init__(self, service: RunService) -> None:
        self.service = service

    def workspace(self) -> WorkspaceBundle:
        records = self.service.store.replay()
        if not records:
            return WorkspaceBundle(
                workspace=RunWorkspaceReadModel(
                    run_id=self.service.config.run_id,
                    raw_user_prompt=self.service.config.raw_user_prompt,
                    run_state=str(self.service.state),
                    generated_count=0,
                    iqa_evaluated_count=0,
                    vlm_evaluated_count=0,
                    promoted_curated_count=0,
                    accepted_count=0,
                    rejected_count=0,
                    shredded_count=0,
                    stall_guard_state="inactive",
                    notification="Waiting for pre-run confirmation.",
                )
            )
        state = reconstruct_run_state(records)
        workspace = RunWorkspaceReadModel(
            run_id=state.run_id,
            raw_user_prompt=state.raw_user_prompt or self.service.config.raw_user_prompt,
            run_state=str(self.service.state),
            generated_count=state.generated_count,
            iqa_evaluated_count=state.iqa_evaluated_count,
            vlm_evaluated_count=state.vlm_evaluated_count,
            promoted_curated_count=state.promoted_curated_count,
            accepted_count=state.accepted_count,
            rejected_count=state.rejected_count,
            shredded_count=state.shredded_count,
            stall_guard_state="active",
            notification=self._notification(records),
            elapsed_seconds=state.elapsed_seconds,
        )
        catalogue = catalogue_default_items(candidate_cards_from_records(records))
        selected = detail_report_from_records(records, catalogue[0].candidate_id) if catalogue else None
        return WorkspaceBundle(workspace=workspace, catalogue=catalogue, selected=selected)

    def handle_event(self, event: UIEvent) -> object | None:
        return self.service.handle_ui_event(event)

    def diagnostics(self) -> DiagnosticsReadModel:
        records = self.service.store.replay()
        record_counts: dict[str, int] = {}
        for record in records:
            record_counts[record.record_type] = record_counts.get(record.record_type, 0) + 1
        system_actions = [record for record in records if record.record_type == "system_action"]
        recent_actions = [
            {
                "action_name": record.payload.get("action_name"),
                "candidate_id": record.candidate_id,
                "coordinate_id": record.coordinate_id,
                "semantic_penalty": record.payload.get("semantic_penalty", True),
                "reasons": record.payload.get("reasons", []),
            }
            for record in system_actions[-10:]
        ]
        infrastructure_retries = [
            action for action in recent_actions if action["action_name"] == "infrastructure_retry"
        ]
        return DiagnosticsReadModel(
            record_counts=record_counts,
            system_action_count=len(system_actions),
            infrastructure_retry_count=sum(
                1 for record in system_actions if record.payload.get("action_name") == "infrastructure_retry"
            ),
            infrastructure_retries=infrastructure_retries,
            recent_system_actions=recent_actions,
        )

    def _notification(self, records: list[PersistenceRecord]) -> str:
        if self.service.state == RunRuntimeState.PAUSED:
            ui_event = self._latest_ui_event(records)
            if ui_event is not None and ui_event.payload.get("event_type") == "pre_run_cancel":
                return "Pre-run canceled. Revise the prompt or start again."
            return "Run paused."
        if self.service.state == RunRuntimeState.STOPPED:
            transition = self._latest_loop_transition(records)
            if transition is not None and transition.payload.get("reason") == "stall_guard":
                return "Run stopped by stall guard."
            return "Run stopped."
        if self.service.state == RunRuntimeState.PAUSED_HIGH_WATERMARK:
            return "Paused at promoted image watermark."
        transition = self._latest_loop_transition(records)
        if transition is None:
            return "Loaded live run state."
        reason = transition.payload.get("reason")
        if reason == "pending_coordinates":
            return "Generating pending coordinate."
        if reason == "coordinate_budget_available":
            return "Preparing new prompt coordinates."
        return "Loaded live run state."

    def _latest_loop_transition(self, records: list[PersistenceRecord]) -> PersistenceRecord | None:
        return next((record for record in reversed(records) if record.record_type == "loop_transition"), None)

    def _latest_ui_event(self, records: list[PersistenceRecord]) -> PersistenceRecord | None:
        return next((record for record in reversed(records) if record.record_type == "ui_event"), None)
