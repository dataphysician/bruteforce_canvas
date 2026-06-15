from bruteforce_canvas.prompt import (
    ActionDescriptor,
    CanonicalEnum,
    Element,
    Evidence,
    EvidenceCategory,
    Graph,
    ObjectDescriptor,
    PromptDocument,
    Relation,
    VerificationIssue,
    VerificationReport,
)
from bruteforce_canvas.prompt_pipeline import (
    CanonicalizerAdapter,
    ExtractionAdapter,
    PromptPipeline,
    RepairAdapter,
    VerificationAdapter,
)
from bruteforce_canvas.shared import CanonicalStatus


class RecordingExtractor(ExtractionAdapter):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, raw_prompt: str) -> PromptDocument:
        self.calls.append(raw_prompt)
        return PromptDocument(
            prompt_document_id="doc_001",
            raw_user_prompt=raw_prompt,
            seed_prompt="red ceramic bowl on wooden table",
            graph=Graph(
                elements=[
                    Element(
                        element_id="object_01",
                        label="bowl",
                        entity_type="object",
                        importance="primary",
                        evidence=Evidence(text="bowl", category=EvidenceCategory.EXPLICIT),
                    ),
                    Element(
                        element_id="object_02",
                        label="table",
                        entity_type="object",
                        importance="supporting",
                        evidence=Evidence(text="table", category=EvidenceCategory.EXPLICIT),
                    ),
                ],
                relations=[
                    Relation(
                        relation_id="rel_01",
                        source_id="object_01",
                        target_id="object_02",
                        relation_raw="on",
                        evidence=Evidence(text="bowl on table", category=EvidenceCategory.EXPLICIT),
                    )
                ],
            ),
            objects=[
                ObjectDescriptor(element_id="object_01", field_name="color", raw_value="red"),
                ObjectDescriptor(element_id="object_01", field_name="material", raw_value="ceramic"),
                ObjectDescriptor(element_id="object_02", field_name="material", raw_value="wooden"),
            ],
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

    def verify(self, document: PromptDocument) -> VerificationReport:
        report = self.reports[min(self.calls, len(self.reports) - 1)]
        self.calls += 1
        return report


class RecordingRepairer(RepairAdapter):
    def __init__(self) -> None:
        self.scopes: list[str] = []

    def repair(self, document: PromptDocument, issue: VerificationIssue) -> PromptDocument:
        self.scopes.append(issue.repair_scope)
        if issue.repair_scope == "object_descriptor":
            repaired = [
                descriptor
                for descriptor in document.objects
                if not (descriptor.element_id == "object_01" and descriptor.field_name == "material")
            ]
            repaired.append(ObjectDescriptor(element_id="object_02", field_name="material", raw_value="wooden"))
            return document.model_copy(update={"objects": repaired})
        return document


def test_prompt_pipeline_runs_extract_canonicalize_verify_and_render():
    extractor = RecordingExtractor()
    canonicalizer = RecordingCanonicalizer()
    verifier = RecordingVerifier([VerificationReport(approved=True, issues=[])])
    repairer = RecordingRepairer()

    result = PromptPipeline(extractor, canonicalizer, verifier, repairer).run(
        "a red ceramic bowl on a wooden table"
    )

    assert result.approved is True
    assert result.rendered_prompt.rendered_prompt.startswith("Generate ")
    assert extractor.calls == ["a red ceramic bowl on a wooden table"]
    assert "relation.rel_01" in canonicalizer.field_paths
    assert "object.color.object_01" in canonicalizer.field_paths
    assert result.document.graph.relations[0].canonical.enum_value == "ON"
    assert result.document.objects[0].canonical.enum_value == "RED"


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
    ).run("a red ceramic bowl on a wooden table")

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
    ).run("person throwing something")

    assert result.approved is False
    assert result.rendered_prompt is None
    assert result.prompt_improvement_feedback == ["Specify what is being thrown."]


def test_prompt_pipeline_does_not_force_action_triplet_for_static_scene():
    result = PromptPipeline(
        RecordingExtractor(),
        RecordingCanonicalizer(),
        RecordingVerifier([VerificationReport(approved=True, issues=[])]),
        RecordingRepairer(),
    ).run("a red ceramic bowl on a wooden table")

    assert result.document.actions == []
