from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, computed_field

from bruteforce_canvas.shared import (
    CandidateId,
    CoordinateId,
    DocId,
    RunId,
    StrictModel,
    TargetManifestId,
)


PERSISTENCE_VERSION = "1"


class PersistenceRecord(StrictModel):
    record_id: str
    record_type: str
    run_id: RunId
    prompt_document_id: DocId | None = None
    target_manifest_id: TargetManifestId | None = None
    coordinate_id: CoordinateId | None = None
    candidate_id: CandidateId | None = None
    seed: int | None = None
    idempotency_key: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def traceability_key(self) -> str:
        parts = [
            self.run_id,
            self.prompt_document_id,
            self.target_manifest_id,
            self.coordinate_id,
            self.candidate_id,
            str(self.seed) if self.seed is not None else None,
        ]
        return "/".join(part for part in parts if part is not None)


class AppendResult(StrictModel):
    written: bool
    record_id: str


class ReconstructedRunState(StrictModel):
    run_id: RunId
    raw_user_prompt: str | None = None
    generated_count: int = 0
    iqa_evaluated_count: int = 0
    vlm_evaluated_count: int = 0
    promoted_curated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    shredded_count: int = 0
    learning_update_count: int = 0
    elapsed_seconds: int = 0
    coordinate_ids: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)


class JsonlEventStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def replay(self) -> list[PersistenceRecord]:
        if not self.path.exists():
            return []
        records: list[PersistenceRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(PersistenceRecord.model_validate_json(line))
        return records

    def append(self, record: PersistenceRecord) -> AppendResult:
        existing = self.replay()
        record_keys = {item.record_id for item in existing}
        idempotency_keys = {item.idempotency_key for item in existing if item.idempotency_key is not None}
        if record.record_id in record_keys or (
            record.idempotency_key is not None and record.idempotency_key in idempotency_keys
        ):
            return AppendResult(written=False, record_id=record.record_id)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json(exclude={"traceability_key"}) + "\n")
        return AppendResult(written=True, record_id=record.record_id)


def _unique_records(records: list[PersistenceRecord]) -> list[PersistenceRecord]:
    seen_record_ids: set[str] = set()
    seen_idempotency_keys: set[str] = set()
    unique: list[PersistenceRecord] = []
    for record in records:
        if record.record_id in seen_record_ids:
            continue
        if record.idempotency_key is not None and record.idempotency_key in seen_idempotency_keys:
            continue
        seen_record_ids.add(record.record_id)
        if record.idempotency_key is not None:
            seen_idempotency_keys.add(record.idempotency_key)
        unique.append(record)
    return unique


def reconstruct_run_state(records: list[PersistenceRecord]) -> ReconstructedRunState:
    unique = _unique_records(records)
    if not unique:
        raise ValueError("cannot reconstruct run state from empty records")
    run_id = unique[0].run_id
    raw_prompt: str | None = None
    candidate_ids: list[str] = []
    coordinate_ids: list[str] = []
    generated_count = 0
    iqa_evaluated = 0
    vlm_evaluated = 0
    promoted = 0
    accepted = 0
    rejected = 0
    shredded = 0
    learning_updates = 0
    elapsed_seconds = 0

    for record in unique:
        elapsed_seconds = max(elapsed_seconds, int(record.payload.get("elapsed_seconds", 0)))
        if record.record_type == "run_config":
            raw_prompt = record.payload.get("raw_user_prompt", raw_prompt)
        if record.coordinate_id and record.coordinate_id not in coordinate_ids:
            coordinate_ids.append(record.coordinate_id)
        if record.candidate_id and record.candidate_id not in candidate_ids:
            candidate_ids.append(record.candidate_id)
        if record.record_type == "candidate_record":
            generated_count += 1
        elif record.record_type == "evaluation_aggregate":
            iqa_evaluated += int(record.payload.get("evaluated_count", 0))
            vlm_evaluated += int(record.payload.get("quality_pass_count", 0))
            promoted += int(record.payload.get("promoted_count", 0))
        elif record.record_type == "feedback":
            action = record.payload.get("feedback_action")
            if action == "accept":
                accepted += 1
            elif action == "reject":
                rejected += 1
            elif action == "shred":
                shredded += 1
        elif record.record_type in {"learning_delta", "feedback_learning_delta"}:
            learning_updates += 1

    return ReconstructedRunState(
        run_id=run_id,
        raw_user_prompt=raw_prompt,
        generated_count=generated_count,
        iqa_evaluated_count=iqa_evaluated,
        vlm_evaluated_count=vlm_evaluated,
        promoted_curated_count=promoted,
        accepted_count=accepted,
        rejected_count=rejected,
        shredded_count=shredded,
        learning_update_count=learning_updates,
        elapsed_seconds=elapsed_seconds,
        coordinate_ids=coordinate_ids,
        candidate_ids=candidate_ids,
    )
