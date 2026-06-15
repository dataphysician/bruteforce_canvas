from __future__ import annotations

from bruteforce_canvas.evaluation import ImageEvaluationResult
from bruteforce_canvas.generation import MIN_SEED_BUNDLE_SIZE, CandidateRecord
from bruteforce_canvas.prompt import PromptDocument, RenderedPrompt
from bruteforce_canvas.prompt_models import PromptDocumentSpec
from bruteforce_canvas.router import CandidateCoordinateBatch


class GateError(ValueError):
    pass


class StageGate:
    @staticmethod
    def prompt(document: PromptDocument | PromptDocumentSpec) -> PromptDocument | PromptDocumentSpec:
        if not document.verification.approved or any(issue.blocking for issue in document.verification.issues):
            raise GateError("PromptDocument verification did not pass")
        return document

    @staticmethod
    def router(batch: CandidateCoordinateBatch) -> CandidateCoordinateBatch:
        if any(trace.hard_rejects for trace in batch.rejected_traces):
            raise GateError("router batch contains a hard compatibility reject")
        if not batch.coordinates:
            raise GateError("router batch contains no coordinates")
        return batch

    @staticmethod
    def rendering(rendered: RenderedPrompt) -> RenderedPrompt:
        if not rendered.rendered_prompt or not rendered.rendered_prompt.startswith("Generate "):
            raise GateError("rendered prompt must begin with Generate")
        return rendered

    @staticmethod
    def generation(
        candidates: list[CandidateRecord],
        *,
        infrastructure_blocked_candidate_ids: set[str] | None = None,
    ) -> list[CandidateRecord]:
        blocked_ids = infrastructure_blocked_candidate_ids or set()
        seeds = [candidate.seed for candidate in candidates]
        if len(seeds) < MIN_SEED_BUNDLE_SIZE:
            raise GateError("seed bundle must contain at least 3 seeds")
        for candidate in candidates:
            if not candidate.file_valid and candidate.candidate_id not in blocked_ids:
                raise GateError("generated artifact is not valid and not infrastructure-blocked")
        return candidates

    @staticmethod
    def evaluation(results: list[ImageEvaluationResult]) -> list[ImageEvaluationResult]:
        seeds = [result.seed for result in results]
        if len(seeds) < MIN_SEED_BUNDLE_SIZE:
            raise GateError("seed bundle must contain at least 3 seeds")
        required_ids = {
            (result.run_id, result.prompt_document_id, result.target_manifest_id, result.coordinate_id)
            for result in results
        }
        if len(required_ids) != 1:
            raise GateError("evaluation results must belong to one coordinate seed bundle")
        return results
