from __future__ import annotations

import math
from enum import StrEnum
from typing import Protocol

from bruteforce_canvas.llm_adapters import FieldEnumContext
from bruteforce_canvas.prompt import CanonicalEnum
from bruteforce_canvas.prompt_enums import (
    Condition,
    Finish,
    MovementType,
    Pattern,
    RelationType,
)
from bruteforce_canvas.shared import CanonicalStatus


BGE_SMALL_EN_MODEL_ID = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_MATCH_THRESHOLD = 0.62


class EmbeddingUnavailableError(RuntimeError):
    pass


class TextEmbeddingModel(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class TransformersTextEmbeddingModel:
    """Lazy BGE encoder backed by transformers.

    The implementation avoids making ``sentence-transformers`` a hard
    dependency. BGE small en is a normal encoder model, so mean pooling
    over the last hidden state is sufficient for this canonicalizer.
    """

    def __init__(self, model_id: str = BGE_SMALL_EN_MODEL_ID, *, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self._tokenizer: object | None = None
        self._model: object | None = None
        self._resolved_device: str | None = None

    def prewarm(self) -> None:
        self.encode(["warmup"])

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        try:
            import torch
            import torch.nn.functional as functional
        except Exception as error:  # pragma: no cover - depends on optional deps
            raise EmbeddingUnavailableError("torch is required for BGE canonicalization") from error

        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise EmbeddingUnavailableError("BGE embedding model is not loaded")

        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        device = self._resolved_device or "cpu"
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            normalized = functional.normalize(pooled, p=2, dim=1)
        return normalized.detach().cpu().tolist()

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as error:  # pragma: no cover - depends on optional deps
            raise EmbeddingUnavailableError(
                "transformers and torch are required for BGE canonicalization"
            ) from error

        if self.device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = self.device
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModel.from_pretrained(self.model_id)
            model.to(resolved)
            model.eval()
        except Exception as error:  # pragma: no cover - depends on HF cache/network
            raise EmbeddingUnavailableError(f"could not load embedding model {self.model_id}") from error
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = resolved


class EmbeddingCanonicalizerAdapter:
    def __init__(
        self,
        *,
        embedding_model: TextEmbeddingModel | None = None,
        model_id: str = BGE_SMALL_EN_MODEL_ID,
        device: str = "auto",
        match_threshold: float = DEFAULT_EMBEDDING_MATCH_THRESHOLD,
        enum_contexts: dict[str, FieldEnumContext] | None = None,
    ) -> None:
        self.embedding_model = embedding_model or TransformersTextEmbeddingModel(model_id=model_id, device=device)
        self.model_id = model_id
        self.match_threshold = match_threshold
        self.enum_contexts = enum_contexts or default_embedding_enum_contexts()
        self._encoded_contexts: dict[str, list[list[float]]] = {}

    def prewarm(self) -> None:
        context = next(iter(self.enum_contexts.values()))
        self.embedding_model.encode([*context.enum_values.values(), "warmup"])

    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        context_key, context = self._context_for_field_path(field_path)
        if context is None:
            return _raw_only(raw_value, f"no embedding enum context for {field_path}")
        labels = list(context.enum_values)
        if not labels:
            return _raw_only(raw_value, f"empty embedding enum context for {field_path}")

        candidate_texts = [context.enum_values[label] for label in labels]
        candidate_vectors = self._encoded_contexts.get(context_key)
        if candidate_vectors is None:
            candidate_vectors = self.embedding_model.encode(candidate_texts)
            self._encoded_contexts[context_key] = candidate_vectors
        query_vector = self.embedding_model.encode([_query_text(field_path, raw_value)])[0]
        best_index, best_score = _best_cosine(query_vector, candidate_vectors)
        best_label = labels[best_index]
        if best_score < self.match_threshold:
            return _raw_only(
                raw_value,
                f"best embedding match {best_label} scored {best_score:.3f} below {self.match_threshold:.3f}",
            )
        return CanonicalEnum(
            raw_value=raw_value,
            enum_value=best_label,
            status=CanonicalStatus.MATCHED_ACTIVE,
            confidence=_confidence(best_score),
            reason=f"embedding match {best_label} scored {best_score:.3f}",
        )

    def _context_for_field_path(self, field_path: str) -> tuple[str, FieldEnumContext | None]:
        for prefix, context in self.enum_contexts.items():
            if field_path.startswith(prefix):
                return prefix, context
        return field_path, None


class FallbackCanonicalizerAdapter:
    def __init__(self, primary: object, fallback: object) -> None:
        self.primary = primary
        self.fallback = fallback

    def canonicalize(self, *, field_path: str, raw_value: str) -> CanonicalEnum:
        try:
            result = self.primary.canonicalize(field_path=field_path, raw_value=raw_value)
        except EmbeddingUnavailableError:
            return self.fallback.canonicalize(field_path=field_path, raw_value=raw_value)
        if result.status != CanonicalStatus.MATCHED_ACTIVE:
            return self.fallback.canonicalize(field_path=field_path, raw_value=raw_value)
        return result

    def prewarm(self) -> None:
        prewarm = getattr(self.primary, "prewarm", None)
        if prewarm is not None:
            try:
                prewarm()
            except EmbeddingUnavailableError:
                fallback_prewarm = getattr(self.fallback, "prewarm", None)
                if fallback_prewarm is not None:
                    fallback_prewarm()


def default_embedding_enum_contexts() -> dict[str, FieldEnumContext]:
    return {
        "relation.": _enum_context("relation", "scene graph relation", RelationType),
        "object.finish.": _enum_context("finish", "object surface finish", Finish),
        "object.condition.": _enum_context("condition", "object condition", Condition),
        "object.pattern.": _enum_context("pattern", "object visual pattern", Pattern),
        "action.": _enum_context("movement_type", "action movement category", MovementType),
    }


def _enum_context(field_name: str, semantic_role: str, enum_type: type[StrEnum]) -> FieldEnumContext:
    return FieldEnumContext(
        field_name=field_name,
        semantic_role=semantic_role,
        enum_values={member.name: _enum_text(member) for member in enum_type},
    )


def _enum_text(member: StrEnum) -> str:
    return str(member.value).replace("_", " ")


def _query_text(field_path: str, raw_value: str) -> str:
    field_hint = field_path.split(".", 1)[0].replace("_", " ")
    return f"{field_hint}: {raw_value.replace('_', ' ')}"


def _best_cosine(query: list[float], candidates: list[list[float]]) -> tuple[int, float]:
    if not candidates:
        raise ValueError("candidate embedding list must not be empty")
    scores = [_cosine(query, candidate) for candidate in candidates]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return best_index, scores[best_index]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _confidence(score: float) -> str:
    if score >= 0.82:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


def _raw_only(raw_value: str, reason: str) -> CanonicalEnum:
    return CanonicalEnum(
        raw_value=raw_value,
        enum_value=None,
        status=CanonicalStatus.UNMATCHED_RAW_ONLY,
        confidence="low",
        reason=reason,
    )
