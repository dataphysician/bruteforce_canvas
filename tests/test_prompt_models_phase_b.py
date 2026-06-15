"""Phase B2 spec-compliance tests for the five new ``prompt`` models.

Source of truth: ``specs/01-bruteforce-canvas_DAG_prompt.md`` sections
321-354 (EvidenceSpan, RelationEnumMatch, ProposedRelationEnum) and
518-529 (InferredGraphSupport, PromptImprovementHint).

The new models are *additive*: they do not modify any pre-existing
``prompt.py`` behaviour. Tests therefore only assert spec-locked shapes
and a couple of derived constraints (default values, ``max_length``
on list fields, ``frozen=True`` behaviour inherited from
``StrictModel``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bruteforce_canvas.prompt import (
    EvidenceCategory,
    EvidenceSpan,
    InferredGraphSupport,
    PromptImprovementHint,
    ProposedRelationEnum,
    RelationEnumMatch,
)
from bruteforce_canvas.prompt_enums import EntityType, RelationType
from bruteforce_canvas.shared import (
    CanonicalStatus,
    ResidualDescription,
    SeedPrompt,
    ShortText,
    StrictModel,
)


def test_evidence_span_minimal_required_fields() -> None:
    """Spec §321-324: ``text`` required, ``category`` defaults to EXPLICIT,
    ``reason`` defaults to ``None``."""
    span = EvidenceSpan(text="red ceramic bowl")

    assert span.text == "red ceramic bowl"
    assert span.category == EvidenceCategory.EXPLICIT
    assert span.reason is None


def test_evidence_span_with_non_default_category_and_reason() -> None:
    span = EvidenceSpan(
        text="the figure appears to reach for something",
        category=EvidenceCategory.ENTAILED,
        reason="inferred from posture and gaze direction",
    )

    assert span.text == "the figure appears to reach for something"
    assert span.category == EvidenceCategory.ENTAILED
    assert span.reason == "inferred from posture and gaze direction"


def test_evidence_span_accepts_all_evidence_categories() -> None:
    for category in EvidenceCategory:
        span = EvidenceSpan(text="any short text", category=category)
        assert span.category == category


def test_evidence_span_rejects_empty_text() -> None:
    """``text: ShortText`` enforces min_length=1."""
    with pytest.raises(ValidationError):
        EvidenceSpan(text="")


def test_evidence_span_is_strict_model() -> None:
    """``EvidenceSpan`` must inherit from ``StrictModel`` (per spec)."""
    assert issubclass(EvidenceSpan, StrictModel)


def test_relation_enum_match_minimal_required_fields() -> None:
    """Spec §336-344: ``raw``, ``status``, ``confidence`` are required;
    ``enum_value`` and ``reason`` default to ``None``."""
    match = RelationEnumMatch(
        raw="resting on",
        status=CanonicalStatus.MATCHED_ACTIVE,
        confidence="clear",
    )

    assert match.raw == "resting on"
    assert match.enum_value is None
    assert match.status == CanonicalStatus.MATCHED_ACTIVE
    assert match.confidence == "clear"
    assert match.reason is None


def test_relation_enum_match_with_full_fields() -> None:
    match = RelationEnumMatch(
        raw="draped over",
        enum_value=RelationType.COVERING,
        status=CanonicalStatus.MATCHED_ACTIVE,
        confidence="clear",
        reason="clear covering relation",
    )

    assert match.raw == "draped over"
    assert match.enum_value == RelationType.COVERING
    assert match.status == CanonicalStatus.MATCHED_ACTIVE
    assert match.confidence == "clear"
    assert match.reason == "clear covering relation"


def test_relation_enum_match_enum_value_uses_string_value() -> None:
    """``StrictModel`` has ``use_enum_values=True`` so enum is stored as str."""
    match = RelationEnumMatch(
        raw="holding",
        enum_value=RelationType.HOLDING,
        status=CanonicalStatus.MATCHED_ACTIVE,
        confidence="clear",
    )
    assert match.enum_value == "holding"


def test_relation_enum_match_accepts_all_canonical_statuses() -> None:
    for status in CanonicalStatus:
        match = RelationEnumMatch(raw="x", status=status, confidence="clear")
        assert match.status == status


def test_proposed_relation_enum_minimal_required_fields() -> None:
    """Spec §347-353: ``code``, ``label``, ``description`` are required;
    list fields default to empty."""
    candidate = ProposedRelationEnum(
        code="LEANING_AGAINST",
        label="leaning against",
        description="a body supported at an angle by a surface or object",
    )

    assert candidate.code == "LEANING_AGAINST"
    assert candidate.label == "leaning against"
    assert candidate.aliases == []
    assert candidate.source_type_examples == []
    assert candidate.target_type_examples == []


def test_proposed_relation_enum_with_full_fields() -> None:
    candidate = ProposedRelationEnum(
        code="LEANING_AGAINST",
        label="leaning against",
        aliases=["lean on", "laying against", "propped against"],
        description="a body supported at an angle by a surface or object",
        source_type_examples=[EntityType.PERSON, EntityType.TOOL],
        target_type_examples=[EntityType.ARCHITECTURE, EntityType.FURNITURE],
    )

    assert candidate.aliases == ["lean on", "laying against", "propped against"]
    assert candidate.source_type_examples == [EntityType.PERSON, EntityType.TOOL]
    assert candidate.target_type_examples == [
        EntityType.ARCHITECTURE,
        EntityType.FURNITURE,
    ]


def test_proposed_relation_enum_aliases_max_length_12() -> None:
    """Spec: ``aliases: list[ShortText] = Field(default_factory=list, max_length=12)``."""
    ProposedRelationEnum(
        code="X",
        label="y",
        description="d",
        aliases=["a"] * 12,
    )

    with pytest.raises(ValidationError):
        ProposedRelationEnum(
            code="X",
            label="y",
            description="d",
            aliases=["a"] * 13,
        )


def test_proposed_relation_enum_examples_max_length_8() -> None:
    """Spec: source/target_type_examples capped at 8."""
    ProposedRelationEnum(
        code="X",
        label="y",
        description="d",
        source_type_examples=[EntityType.PERSON] * 8,
        target_type_examples=[EntityType.PERSON] * 8,
    )

    with pytest.raises(ValidationError):
        ProposedRelationEnum(
            code="X",
            label="y",
            description="d",
            source_type_examples=[EntityType.PERSON] * 9,
        )

    with pytest.raises(ValidationError):
        ProposedRelationEnum(
            code="X",
            label="y",
            description="d",
            target_type_examples=[EntityType.PERSON] * 9,
        )


def test_inferred_graph_support_minimal_required_fields() -> None:
    """Spec §518-523: ``reason``, ``source_action``, ``evidence`` are required.
    ``implied_element_label`` and ``implied_relation_type`` default to ``None``."""
    support = InferredGraphSupport(
        reason="inferred presence of a hand-held object",
        source_action="the figure extends an arm forward",
        evidence=EvidenceSpan(text="extended arm pose"),
    )

    assert support.reason == "inferred presence of a hand-held object"
    assert support.implied_element_label is None
    assert support.implied_relation_type is None
    assert support.source_action == "the figure extends an arm forward"
    assert isinstance(support.evidence, EvidenceSpan)
    assert support.evidence.text == "extended arm pose"


def test_inferred_graph_support_with_implied_element_and_relation() -> None:
    support = InferredGraphSupport(
        reason="implied prop from action",
        implied_element_label="umbrella",
        implied_relation_type=RelationType.HOLDING,
        source_action="the figure shelters from the rain",
        evidence=EvidenceSpan(
            text="the figure shelters from the rain",
            category=EvidenceCategory.ENTAILED,
        ),
    )

    assert support.implied_element_label == "umbrella"
    assert support.implied_relation_type == RelationType.HOLDING
    assert support.evidence.category == EvidenceCategory.ENTAILED


def test_prompt_improvement_hint_minimal_required_fields() -> None:
    """Spec §526-529: only ``issue`` is required; ``suggested_rewrites``
    defaults to empty list, ``safe_downgrade`` defaults to ``None``."""
    hint = PromptImprovementHint(issue="ambiguous lighting description")

    assert hint.issue == "ambiguous lighting description"
    assert hint.suggested_rewrites == []
    assert hint.safe_downgrade is None


def test_prompt_improvement_hint_with_full_fields() -> None:
    hint = PromptImprovementHint(
        issue="ambiguous lighting description",
        suggested_rewrites=["soft window light", "warm rim light"],
        safe_downgrade="remove lighting description",
    )

    assert hint.issue == "ambiguous lighting description"
    assert hint.suggested_rewrites == ["soft window light", "warm rim light"]
    assert hint.safe_downgrade == "remove lighting description"


def test_prompt_improvement_hint_suggested_rewrites_max_length_4() -> None:
    """Spec: ``suggested_rewrites: list[ShortText] = Field(default_factory=list, max_length=4)``."""
    PromptImprovementHint(issue="x", suggested_rewrites=["a", "b", "c", "d"])

    with pytest.raises(ValidationError):
        PromptImprovementHint(issue="x", suggested_rewrites=["a"] * 5)


def test_residual_description_and_seed_prompt_aliases_exist() -> None:
    """Both aliases must be importable from ``bruteforce_canvas.shared``."""
    assert ResidualDescription is not None
    assert SeedPrompt is not None
    assert getattr(ResidualDescription, "__metadata__", None) is not None
    assert getattr(SeedPrompt, "__metadata__", None) is not None


@pytest.mark.parametrize(
    "model_cls",
    [
        EvidenceSpan,
        RelationEnumMatch,
        ProposedRelationEnum,
        InferredGraphSupport,
        PromptImprovementHint,
    ],
)
def test_all_five_models_subclass_strict_model(model_cls: type) -> None:
    """Spec: every model in this module subclasses ``StrictModel``."""
    assert issubclass(model_cls, StrictModel)


@pytest.mark.parametrize(
    "model_cls",
    [
        EvidenceSpan,
        RelationEnumMatch,
        ProposedRelationEnum,
        InferredGraphSupport,
        PromptImprovementHint,
    ],
)
def test_all_five_models_are_frozen_and_extra_forbid(model_cls: type) -> None:
    """``StrictModel`` config: ``frozen=True, extra='forbid'``."""
    config = model_cls.model_config
    assert config.get("frozen") is True
    assert config.get("extra") == "forbid"
    assert config.get("use_enum_values") is True


def test_all_five_models_reject_unknown_fields() -> None:
    """``extra='forbid'`` inherited from ``StrictModel``."""
    with pytest.raises(ValidationError):
        EvidenceSpan(text="x", unknown_field="y")
    with pytest.raises(ValidationError):
        RelationEnumMatch(raw="x", status=CanonicalStatus.MATCHED_ACTIVE, unknown_field="y")
    with pytest.raises(ValidationError):
        ProposedRelationEnum(code="X", label="y", description="d", unknown_field="z")
    with pytest.raises(ValidationError):
        InferredGraphSupport(
            reason="r",
            source_action="s",
            evidence=EvidenceSpan(text="t"),
            unknown_field="u",
        )
    with pytest.raises(ValidationError):
        PromptImprovementHint(issue="i", unknown_field="j")


def test_shorttext_aliases_available_for_reference() -> None:
    """Sanity check that ``ShortText`` is still exported from ``shared``."""
    assert ShortText is not None
