from __future__ import annotations

from pydantic import Field

from bruteforce_canvas.evaluation import CoordinateEvaluationAggregate
from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE
from bruteforce_canvas.shared import CoordinateId, DocId, RunId, StrictModel, TargetManifestId


class SeedSurfPolicy(StrictModel):
    enabled: bool = True
    surf_seed_count: int = Field(default=5, gt=0)
    seed_start: int = 100000
    min_pass_rate: float = 0.60


class SeedSurfBundle(StrictModel):
    run_id: RunId
    prompt_document_id: DocId
    target_manifest_id: TargetManifestId
    parent_coordinate_id: CoordinateId
    seeds: list[int]
    reason: str


def enqueue_seed_surf_bundle(
    aggregate: CoordinateEvaluationAggregate,
    policy: SeedSurfPolicy,
) -> SeedSurfBundle | None:
    if not policy.enabled:
        return None
    if aggregate.outcome != "strong":
        return None
    if aggregate.pass_rate < policy.min_pass_rate:
        return None

    seeds: list[int] = []
    candidate = policy.seed_start
    while len(seeds) < policy.surf_seed_count:
        if candidate not in DEFAULT_SEED_BUNDLE:
            seeds.append(candidate)
        candidate += 1

    return SeedSurfBundle(
        run_id=aggregate.run_id,
        prompt_document_id=aggregate.prompt_document_id,
        target_manifest_id=aggregate.target_manifest_id,
        parent_coordinate_id=aggregate.coordinate_id,
        seeds=seeds,
        reason="strong_coordinate_seed_surf",
    )
