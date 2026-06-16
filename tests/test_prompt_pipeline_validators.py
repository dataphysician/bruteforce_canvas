import pytest

import bruteforce_canvas.prompt_pipeline as pipeline_module
from bruteforce_canvas.prompt import (
    CanonicalEnum,
    EvidenceCategory,
    EvidenceSpan,
    VerificationIssue,
    VerificationReport,
)
from bruteforce_canvas.prompt_enums import ElementRole, EntityType, Importance
from bruteforce_canvas.prompt_models import Element, ObjectDescriptor, ObjectLane, PromptDocumentSpec, SceneGraphDraft
from bruteforce_canvas.prompt_pipeline import PromptPipeline
from bruteforce_canvas.shared import CanonicalStatus
from bruteforce_canvas.validation import RetryRequest


VALIDATOR_NAMES = (
    "validate_relation_compatibility",
    "validate_object_ownership",
    "validate_action_support",
    "validate_evidence_and_placeholders",
    "validate_cross_lane_coherence",
)


class StaticExtractor:
    def __init__(self, document: PromptDocumentSpec) -> None:
        self.document = document

    def extract(self, raw_prompt: str) -> PromptDocumentSpec:
        return self.document


class StaticCanonicalizer:
    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        return CanonicalEnum(
            raw_value=raw_value,
            enum_value=raw_value.upper().replace(" ", "_"),
            status=CanonicalStatus.MATCHED_ACTIVE,
            confidence="high",
            reason=f"canonicalized {field_path}",
        )


class RecordingVerifier:
    def __init__(self, reports: list[VerificationReport] | None = None) -> None:
        self.reports = reports or [VerificationReport(approved=True, issues=[])]
        self.calls = 0

    def verify(self, document: PromptDocumentSpec) -> VerificationReport:
        report = self.reports[min(self.calls, len(self.reports) - 1)]
        self.calls += 1
        return report


class RecordingRepairer:
    def __init__(self) -> None:
        self.requests: list[RetryRequest | VerificationIssue] = []

    def repair(
        self,
        document: PromptDocumentSpec,
        issue: RetryRequest | VerificationIssue,
    ) -> PromptDocumentSpec:
        self.requests.append(issue)
        return document


def spec_document(*, material: str | None = "ceramic") -> PromptDocumentSpec:
    return PromptDocumentSpec(
        raw_user_prompt="a red ceramic bowl",
        graph=SceneGraphDraft(
            seed_prompt="red ceramic bowl",
            elements=[
                Element(
                    id="object_01",
                    label="bowl",
                    entity_type=EntityType.PRODUCT,
                    role=ElementRole.PRIMARY_SUBJECT,
                    importance=Importance.REQUIRED,
                    evidence=EvidenceSpan(text="bowl", category=EvidenceCategory.EXPLICIT),
                )
            ],
        ),
        object_lane=ObjectLane(objects=[ObjectDescriptor(target_id="object_01", material=material, color="red")]),
        verification=VerificationReport(approved=False, issues=[]),
    )


def patch_validators(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object] | None = None) -> None:
    overrides = overrides or {}
    for name in VALIDATOR_NAMES:
        monkeypatch.setattr(pipeline_module, name, overrides.get(name, lambda document: []))


def test_all_five_validators_are_called_during_run_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def traced(name: str):
        def validator(document: object) -> list[str]:
            called.append(name)
            return []

        return validator

    patch_validators(monkeypatch, {name: traced(name) for name in VALIDATOR_NAMES})

    result = PromptPipeline(
        StaticExtractor(spec_document()),
        StaticCanonicalizer(),
        RecordingVerifier(),
        RecordingRepairer(),
    ).run_spec("a red ceramic bowl")

    assert result.approved is True
    assert called == list(VALIDATOR_NAMES)


def test_validator_issues_cause_unapproved_result(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_validators(
        monkeypatch,
        {"validate_relation_compatibility": lambda graph: ["rel_01 invalid relation"]},
    )

    result = PromptPipeline(
        StaticExtractor(spec_document()),
        StaticCanonicalizer(),
        RecordingVerifier(),
        RecordingRepairer(),
    ).run_spec("a red ceramic bowl")

    assert result.approved is False
    assert result.rendered_prompt is None
    assert result.prompt_improvement_feedback == ["rel_01 invalid relation"]
    assert result.verifier_report.issues[0].repair_scope == "relation"


def test_retry_loop_repairs_fixable_validator_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    def object_validator(document: PromptDocumentSpec) -> list[str]:
        if document.object_lane.objects[0].material == "wrong":
            return ["object_01 material is wrong"]
        return []

    patch_validators(monkeypatch, {"validate_object_ownership": object_validator})

    class FixingRepairer(RecordingRepairer):
        def repair(
            self,
            document: PromptDocumentSpec,
            issue: RetryRequest | VerificationIssue,
        ) -> PromptDocumentSpec:
            self.requests.append(issue)
            assert isinstance(issue, RetryRequest)
            fixed_objects = [
                descriptor.model_copy(update={"material": None})
                for descriptor in document.object_lane.objects
            ]
            return document.model_copy(update={"object_lane": ObjectLane(objects=fixed_objects)})

    repairer = FixingRepairer()
    verifier = RecordingVerifier()

    result = PromptPipeline(
        StaticExtractor(spec_document(material="wrong")),
        StaticCanonicalizer(),
        verifier,
        repairer,
    ).run_spec("a red ceramic bowl")

    assert result.approved is True
    assert len(repairer.requests) == 1
    assert verifier.calls == 1


def test_validator_retry_loop_respects_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_validators(
        monkeypatch,
        {"validate_relation_compatibility": lambda graph: ["rel_01 still invalid"]},
    )
    repairer = RecordingRepairer()

    result = PromptPipeline(
        StaticExtractor(spec_document()),
        StaticCanonicalizer(),
        RecordingVerifier(),
        repairer,
    ).run_spec("a red ceramic bowl")

    assert result.approved is False
    assert len(repairer.requests) == 2
    assert all(isinstance(request, RetryRequest) for request in repairer.requests)
    assert result.prompt_improvement_feedback == ["rel_01 still invalid"]


def test_retry_request_shape_passed_to_repairer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def action_validator(document: PromptDocumentSpec) -> list[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ["supported action lacks relation preconditions"]
        return []

    patch_validators(monkeypatch, {"validate_action_support": action_validator})
    repairer = RecordingRepairer()

    result = PromptPipeline(
        StaticExtractor(spec_document()),
        StaticCanonicalizer(),
        RecordingVerifier(),
        repairer,
    ).run_spec("a red ceramic bowl")

    assert result.approved is True
    request = repairer.requests[0]
    assert isinstance(request, RetryRequest)
    assert request.failed_stage == "action_lane"
    assert request.frozen_graph == result.document.graph
    assert request.invalid_payload["issues"][0]["message"] == "supported action lacks relation preconditions"
    assert request.issues[0].retry_scope == "action_descriptor"
    assert "action descriptor" in request.instruction
