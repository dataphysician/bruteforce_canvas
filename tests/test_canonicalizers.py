import pytest

from bruteforce_canvas.canonicalizers import (
    EmbeddingCanonicalizerAdapter,
    EmbeddingUnavailableError,
    FallbackCanonicalizerAdapter,
)
from bruteforce_canvas.llm_adapters import FieldEnumContext
from bruteforce_canvas.prompt import CanonicalEnum
from bruteforce_canvas.shared import CanonicalStatus


class FakeEmbeddingModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise EmbeddingUnavailableError("embedding model unavailable")
        self.calls.append(texts)
        return [_vector(text) for text in texts]


class RecordingCanonicalizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def canonicalize(self, *, field_path: str, raw_value: str):
        self.calls.append((field_path, raw_value))
        return CanonicalEnum(
            raw_value=raw_value,
            enum_value="LLM_VALUE",
            status=CanonicalStatus.MATCHED_ACTIVE,
            confidence="medium",
            reason="llm fallback",
        )


def _vector(text: str) -> list[float]:
    normalized = text.lower()
    if "on top" in normalized or "on the table" in normalized:
        return [1.0, 0.0, 0.0]
    if "inside" in normalized:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def test_embedding_canonicalizer_matches_known_enum_context() -> None:
    adapter = EmbeddingCanonicalizerAdapter(
        embedding_model=FakeEmbeddingModel(),
        match_threshold=0.9,
        enum_contexts={
            "relation.": FieldEnumContext(
                field_name="relation",
                semantic_role="relation type",
                enum_values={
                    "ON_TOP_OF": "on top of",
                    "INSIDE": "inside",
                },
            )
        },
    )

    result = adapter.canonicalize(field_path="relation.rel_01", raw_value="on the table")

    assert result.enum_value == "ON_TOP_OF"
    assert result.status == CanonicalStatus.MATCHED_ACTIVE
    assert result.reason.startswith("embedding match ON_TOP_OF")


def test_embedding_canonicalizer_preserves_raw_when_no_enum_context() -> None:
    adapter = EmbeddingCanonicalizerAdapter(
        embedding_model=FakeEmbeddingModel(),
        enum_contexts={},
    )

    result = adapter.canonicalize(field_path="object.color.object_01", raw_value="warm red")

    assert result.enum_value is None
    assert result.status == CanonicalStatus.UNMATCHED_RAW_ONLY
    assert "no embedding enum context" in result.reason


def test_fallback_canonicalizer_uses_llm_only_when_embedding_unavailable() -> None:
    primary = EmbeddingCanonicalizerAdapter(
        embedding_model=FakeEmbeddingModel(fail=True),
        enum_contexts={
            "relation.": FieldEnumContext(
                field_name="relation",
                semantic_role="relation type",
                enum_values={"ON_TOP_OF": "on top of"},
            )
        },
    )
    fallback = RecordingCanonicalizer()
    adapter = FallbackCanonicalizerAdapter(primary, fallback)

    result = adapter.canonicalize(field_path="relation.rel_01", raw_value="on the table")

    assert result.enum_value == "LLM_VALUE"
    assert fallback.calls == [("relation.rel_01", "on the table")]


def test_fallback_canonicalizer_uses_llm_when_embedding_is_raw_only() -> None:
    primary = EmbeddingCanonicalizerAdapter(
        embedding_model=FakeEmbeddingModel(),
        match_threshold=0.99,
        enum_contexts={
            "relation.": FieldEnumContext(
                field_name="relation",
                semantic_role="relation type",
                enum_values={"INSIDE": "inside"},
            )
        },
    )
    fallback = RecordingCanonicalizer()
    adapter = FallbackCanonicalizerAdapter(primary, fallback)

    result = adapter.canonicalize(field_path="relation.rel_01", raw_value="on the table")

    assert result.enum_value == "LLM_VALUE"
    assert fallback.calls == [("relation.rel_01", "on the table")]


def test_embedding_canonicalizer_propagates_non_availability_errors() -> None:
    class BrokenCanonicalizer:
        def canonicalize(self, *, field_path: str, raw_value: str):
            raise ValueError("bad context")

    adapter = FallbackCanonicalizerAdapter(BrokenCanonicalizer(), RecordingCanonicalizer())

    with pytest.raises(ValueError, match="bad context"):
        adapter.canonicalize(field_path="relation.rel_01", raw_value="on")
