from bruteforce_canvas.prompt import (
    CanonicalEnum,
    EvidenceCategory,
    EvidenceSpan,
    ObjectLane,
    PromptDocument,
    PromptDocumentSpec,
    SceneGraphDraft,
    VerificationIssue,
    VerificationReport,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Importance
from bruteforce_canvas.prompt_models import Element, ObjectDescriptor, RelationDescriptor
from bruteforce_canvas.prompt_pipeline import (
    CanonicalizerAdapter,
    ExtractionAdapter,
    PromptPipeline,
    RepairAdapter,
    VerificationAdapter,
)
from bruteforce_canvas.shared import CanonicalStatus
from bruteforce_canvas.validation import RetryRequest


class RecordingExtractor(ExtractionAdapter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, raw_prompt: str) -> PromptDocumentSpec:
        self.calls.append(raw_prompt)
        return PromptDocumentSpec(
            raw_user_prompt=raw_prompt,
            graph=SceneGraphDraft(
                seed_prompt="red ceramic bowl on wooden table",
                elements=[
                    Element(
                        id="object_01",
                        label="bowl",
                        entity_type=EntityType.PRODUCT,
                        role=ElementRole.PRIMARY_SUBJECT,
                        importance=Importance.REQUIRED,
                        evidence=EvidenceSpan(text="bowl", category=EvidenceCategory.EXPLICIT),
                    ),
                    Element(
                        id="object_02",
                        label="table",
                        entity_type=EntityType.FURNITURE,
                        role=ElementRole.SUPPORTING,
                        importance=Importance.REQUIRED,
                        evidence=EvidenceSpan(text="table", category=EvidenceCategory.EXPLICIT),
                    ),
                ],
                relations=[
                    RelationDescriptor(
                        id="rel_01",
                        source_id="object_01",
                        target_id="object_02",
                        relation_raw="on",
                        evidence=EvidenceSpan(text="bowl on table", category=EvidenceCategory.EXPLICIT),
                    )
                ],
            ),
            object_lane=ObjectLane(
                objects=[
                    ObjectDescriptor(target_id="object_01", color="red", material="ceramic"),
                    ObjectDescriptor(target_id="object_02", material="wooden"),
                ]
            ),
            verification=VerificationReport(approved=False, issues=[]),
        )


class RecordingCanonicalizer(CanonicalizerAdapter):
    def __init__(self) -> None:
        self.field_paths: list[str] = []

    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        self.field_paths.append(field_path)
        enum_value = raw_value.upper().replace(" ", "_")
        return CanonicalEnum(
            raw_value=raw_value,
            enum_value=enum_value,
            status=CanonicalStatus.MATCHED_ACTIVE,
            confidence="high",
            reason="test canonicalization",
        )


class RecordingVerifier(VerificationAdapter):
    def __init__(self, reports: list[VerificationReport]) -> None:
        self.reports = reports
        self.calls = 0

    def verify(self, document: PromptDocument | PromptDocumentSpec) -> VerificationReport:
        report = self.reports[min(self.calls, len(self.reports) - 1)]
        self.calls += 1
        return report


class RecordingRepairer(RepairAdapter):
    def __init__(self) -> None:
        self.scopes: list[str] = []

    def repair(
        self,
        document: PromptDocument | PromptDocumentSpec,
        issue: RetryRequest | VerificationIssue,
    ) -> PromptDocument | PromptDocumentSpec:
        repair_scope = issue.issues[0].retry_scope if isinstance(issue, RetryRequest) else issue.repair_scope
        self.scopes.append(repair_scope)
        if not isinstance(document, PromptDocumentSpec):
            return document
        if repair_scope == "object_descriptor":
            repaired = [
                descriptor.model_copy(update={"material": None})
                if descriptor.target_id == "object_01"
                else descriptor
                for descriptor in document.object_lane.objects
            ]
            return document.model_copy(update={"object_lane": ObjectLane(objects=repaired)})
        return document


def test_prompt_pipeline_runs_extract_canonicalize_verify_and_render():
    extractor = RecordingExtractor()
    canonicalizer = RecordingCanonicalizer()
    verifier = RecordingVerifier([VerificationReport(approved=True, issues=[])])
    repairer = RecordingRepairer()

    result = PromptPipeline(extractor, canonicalizer, verifier, repairer).run_spec(
        "a red ceramic bowl on a wooden table"
    )

    assert result.approved is True
    assert result.rendered_prompt is not None
    assert result.rendered_prompt.rendered_prompt.startswith("Generate ")
    assert extractor.calls == ["a red ceramic bowl on a wooden table"]
    assert "relation.rel_01" in canonicalizer.field_paths
    assert "object.color.object_01" in canonicalizer.field_paths
    assert result.document.canonical_metadata["relation.rel_01"].enum_value == "ON"
    assert result.document.canonical_metadata["object.color.object_01"].enum_value == "RED"


def test_prompt_pipeline_repairs_only_blocking_issue_scope_before_reverify():
    issue = VerificationIssue(
        issue_type="descriptor_wrong_owner",
        repair_scope="object_descriptor",
        blocking=True,
        message="wooden belongs to table, not bowl",
    )
    verifier = RecordingVerifier(
        [
            VerificationReport(approved=False, issues=[issue]),
            VerificationReport(approved=True, issues=[]),
        ]
    )
    repairer = RecordingRepairer()

    result = PromptPipeline(
        RecordingExtractor(),
        RecordingCanonicalizer(),
        verifier,
        repairer,
        max_repairs=1,
    ).run_spec("a red ceramic bowl on a wooden table")

    assert result.approved is True
    assert verifier.calls == 2
    assert repairer.scopes == ["object_descriptor"]


def test_prompt_pipeline_returns_blocked_result_without_rendering_when_repair_budget_exhausted():
    issue = VerificationIssue(
        issue_type="unresolved_action_target",
        repair_scope="prompt_improvement",
        blocking=True,
        message="Specify what is being thrown.",
    )
    result = PromptPipeline(
        RecordingExtractor(),
        RecordingCanonicalizer(),
        RecordingVerifier([VerificationReport(approved=False, issues=[issue])]),
        RecordingRepairer(),
        max_repairs=0,
    ).run_spec("person throwing something")

    assert result.approved is False
    assert result.rendered_prompt is None
    assert result.prompt_improvement_feedback == ["Specify what is being thrown."]


def test_prompt_pipeline_does_not_force_action_triplet_for_static_scene():
    result = PromptPipeline(
        RecordingExtractor(),
        RecordingCanonicalizer(),
        RecordingVerifier([VerificationReport(approved=True, issues=[])]),
        RecordingRepairer(),
    ).run_spec("a red ceramic bowl on a wooden table")

    assert result.document.action_lane.actions == []
