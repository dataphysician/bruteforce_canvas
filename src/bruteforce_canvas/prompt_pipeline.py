from __future__ import annotations

from typing import Protocol

from pydantic import Field

from bruteforce_canvas.prompt import (
    CanonicalEnum,
    PromptDocument,
    RenderedPrompt,
    VerificationIssue,
    VerificationReport,
    render_prompt,
)
from bruteforce_canvas.shared import StrictModel


class ExtractionAdapter(Protocol):
    def extract(self, raw_prompt: str) -> PromptDocument:
        """Return a draft PromptDocument from one raw user prompt."""


class CanonicalizerAdapter(Protocol):
    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        """Canonicalize one field-scoped raw value without rewriting the raw phrase."""


class VerificationAdapter(Protocol):
    def verify(self, document: PromptDocument) -> VerificationReport:
        """Verify graph linkage, lane ownership, enum fit, and renderability."""


class RepairAdapter(Protocol):
    def repair(self, document: PromptDocument, issue: VerificationIssue) -> PromptDocument:
        """Repair only the slice identified by the verifier issue."""


class PromptPipelineResult(StrictModel):
    approved: bool
    document: PromptDocument
    verifier_report: VerificationReport
    rendered_prompt: RenderedPrompt | None = None
    prompt_improvement_feedback: list[str] = Field(default_factory=list)


class PromptPipeline:
    def __init__(
        self,
        extractor: ExtractionAdapter,
        canonicalizer: CanonicalizerAdapter,
        verifier: VerificationAdapter,
        repairer: RepairAdapter,
        *,
        max_repairs: int = 2,
    ) -> None:
        self.extractor = extractor
        self.canonicalizer = canonicalizer
        self.verifier = verifier
        self.repairer = repairer
        self.max_repairs = max_repairs

    def run(self, raw_prompt: str) -> PromptPipelineResult:
        document = self._canonicalize_document(self.extractor.extract(raw_prompt))
        report = self.verifier.verify(document)
        repairs_used = 0
        while not report.approved and repairs_used < self.max_repairs:
            blocking = [issue for issue in report.issues if issue.blocking]
            if not blocking:
                break
            for issue in blocking:
                document = self.repairer.repair(document, issue)
            document = self._canonicalize_document(document)
            repairs_used += 1
            report = self.verifier.verify(document)

        document = document.model_copy(update={"verification": report})
        if report.approved:
            return PromptPipelineResult(
                approved=True,
                document=document,
                verifier_report=report,
                rendered_prompt=render_prompt(document),
            )

        return PromptPipelineResult(
            approved=False,
            document=document,
            verifier_report=report,
            prompt_improvement_feedback=[issue.message for issue in report.issues if issue.blocking],
        )

    def _canonicalize_document(self, document: PromptDocument) -> PromptDocument:
        canonical_metadata: dict[str, CanonicalEnum] = dict(document.canonical_metadata)

        relations = []
        for relation in document.graph.relations:
            canonical = self.canonicalizer.canonicalize(
                field_path=f"relation.{relation.relation_id}",
                raw_value=relation.relation_raw,
            )
            canonical_metadata[f"relation.{relation.relation_id}"] = canonical
            relations.append(relation.model_copy(update={"canonical": canonical}))

        objects = []
        for descriptor in document.objects:
            field_path = f"object.{descriptor.field_name}.{descriptor.element_id}"
            canonical = self.canonicalizer.canonicalize(field_path=field_path, raw_value=descriptor.raw_value)
            canonical_metadata[field_path] = canonical
            objects.append(descriptor.model_copy(update={"canonical": canonical}))

        actions = []
        for action in document.actions:
            field_path = f"action.{action.actor_id}"
            canonical = self.canonicalizer.canonicalize(field_path=field_path, raw_value=action.action_raw)
            canonical_metadata[field_path] = canonical
            actions.append(action.model_copy(update={"canonical": canonical}))

        graph = document.graph.model_copy(update={"relations": relations})
        return document.model_copy(
            update={
                "graph": graph,
                "objects": objects,
                "actions": actions,
                "canonical_metadata": canonical_metadata,
            }
        )
