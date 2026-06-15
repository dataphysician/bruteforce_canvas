"""Seed-sweep shape validator tests for ``EvaluationBatchRequest``.

Phase B6 fixes ``validate_seed_sweep_shape`` to require a minimum of
``MIN_SEED_BUNDLE_SIZE`` (3) images instead of an exact five-seed match
against ``DEFAULT_SEED_BUNDLE``. The tests below pin both the relaxed
acceptance boundary and the rejection boundary.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from bruteforce_canvas.evaluation import (
    EvaluationBatchRequest,
    EvaluationImageInput,
    EvaluationPlan,
)
from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE, MIN_SEED_BUNDLE_SIZE


def _image(seed: int) -> EvaluationImageInput:
    return EvaluationImageInput(
        candidate_id=f"cand_{seed}",
        image_path=f"runtime/images/cand_{seed}.png",
        seed=seed,
        coordinate_id="coord_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        generation_settings={"steps": 4},
    )


def _request(*, seeds: list[int], batch_kind: str = "seed_sweep") -> EvaluationBatchRequest:
    return EvaluationBatchRequest(
        batch_id="batch_001",
        run_id="run_001",
        prompt_document_id="doc_001",
        target_manifest_id="eval_manifest_001",
        batch_kind=batch_kind,
        coordinate_id="coord_001",
        rendered_prompt="Generate a handbag",
        target_manifest={"targets": []},
        images=[_image(seed) for seed in seeds],
        evaluator_plan=EvaluationPlan(quality_cutoff=0.5, alignment_cutoff=0.5),
    )


def test_seed_sweep_accepts_three_seed_minimum() -> None:
    request = _request(seeds=DEFAULT_SEED_BUNDLE[:MIN_SEED_BUNDLE_SIZE])
    assert len(request.images) == MIN_SEED_BUNDLE_SIZE


def test_seed_sweep_accepts_five_seed_default_bundle() -> None:
    request = _request(seeds=DEFAULT_SEED_BUNDLE)
    assert len(request.images) == len(DEFAULT_SEED_BUNDLE)


def test_seed_sweep_accepts_seven_seed_bundle() -> None:
    request = _request(seeds=DEFAULT_SEED_BUNDLE + [11, 22])
    assert len(request.images) == 7


def test_seed_sweep_rejects_two_image_bundle() -> None:
    with pytest.raises(ValidationError) as exc:
        _request(seeds=DEFAULT_SEED_BUNDLE[: MIN_SEED_BUNDLE_SIZE - 1])
    assert "at least 3 images" in str(exc.value)


def test_seed_sweep_rejects_empty_image_list() -> None:
    with pytest.raises(ValidationError) as exc:
        _request(seeds=[])
    assert "at least 3 images" in str(exc.value)


def test_mixed_image_batch_does_not_apply_seed_validator() -> None:
    request = _request(seeds=[7, 42], batch_kind="mixed_image_batch")
    assert request.batch_kind == "mixed_image_batch"
    assert len(request.images) == 2
