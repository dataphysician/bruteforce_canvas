from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

import gradio as gr
from PIL import Image, ImageDraw, ImageEnhance
from pydantic import Field

from bruteforce_canvas.app_config import AppConfig, GeneratorKind, load_app_config
from bruteforce_canvas.app_controller import RunAppController
from bruteforce_canvas.app_factory import build_evaluation_plan, build_prompt_pipeline, build_run_service, build_vlm_adapter
from bruteforce_canvas.asr import default_transcriber
from bruteforce_canvas.evaluation import EvaluationPlan, StaticIQAAdapter, StaticImpactAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE, GenerationSettings, seed_sweep_requests
from bruteforce_canvas.orchestration import RunConfig, RunRuntimeState
from bruteforce_canvas.persistence import PERSISTENCE_VERSION, PersistenceRecord, reconstruct_run_state
from bruteforce_canvas.prompt import (
    EvidenceCategory,
    EvidenceSpan,
    EvaluationTargetManifest,
    RenderedPrompt,
    RelationEnumMatch,
    VerificationIssue,
    VerificationReport,
    render_prompt_spec,
    target_manifest_from_prompt_spec,
)
from bruteforce_canvas.prompt_enums import (
    ElementRole,
    EntityType,
    Finish,
    Framing,
    Guardrail,
    Importance,
    LightingMood,
    OpticCharacter,
    RelationType,
    ShotSize,
)
from bruteforce_canvas.prompt_models import (
    CinematographyLane,
    ConstraintLane,
    Element,
    ObjectDescriptor,
    ObjectLane,
    PromptDocumentSpec,
    RelationDescriptor,
    SceneGraphDraft,
)
from bruteforce_canvas.router import AxisDomain, FieldState, LHSRouter, RouterInput
from bruteforce_canvas.shared import FeedbackAction, StrictModel
from bruteforce_canvas.ui import (
    CandidateCard,
    DetailReport,
    PreRunModalReadModel,
    RunWorkspaceReadModel,
    pre_run_modal_from_prompt,
    submit_feedback_event,
)
from bruteforce_canvas.worker import SeedSweepWorkItem


ASSET_DIR = Path(tempfile.gettempdir()) / "bruteforce_canvas_gradio_sim"
RUNTIME_RUN_ROOT = Path("runtime/gradio_runs")
RUN_ID = "run_001"
PROMPT_DOCUMENT_ID = "doc_001"
TARGET_MANIFEST_ID = "eval_manifest_001"
COORDINATE_ID = "coord_001"
RUNTIME_LOOP_LIMIT_SECONDS = 900
RUNTIME_STALL_WINDOW_SECONDS = 600
RUNTIME_STALL_MIN_PROMOTED = 10
LOCK_TABLE_HEADERS = ["Locked", "Field", "Raw", "Enum", "LHS policy", "Status"]
GradioMode = Literal["simulation", "runtime"]
WORKFLOW_MERMAID_CODE = """%%{init: {"theme": "base", "flowchart": {"htmlLabels": false}, "themeVariables": {"primaryColor": "#eef8f6", "primaryTextColor": "#17211f", "primaryBorderColor": "#0f766e", "secondaryColor": "#fff8e6", "tertiaryColor": "#ffffff", "lineColor": "#17211f", "fontFamily": "Inter, ui-sans-serif, system-ui, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
  Mic["Microphone audio"] --> ASR["ASR transcript<br/>Cohere Transcribe path"]
  Typed["Typed prompt"] --> Prompt["Prompt text"]
  ASR --> Prompt
  Prompt --> Mellum["Mellum2 Thinking 12B<br/>via Modal Cloud<br/>structured JSON schemas"]
  Mellum --> Parse["PromptDocument<br/>extraction<br/>canonicalize + verify"]
  Parse --> Locks["Pre-run locks<br/>thresholds<br/>IQA >= 0.55<br/>Alignment >= 0.25<br/>Human IQA >= 0.70"]
  Locks --> Coord["LHS coordinate<br/>coord_###<br/>compatibility + Bayesian<br/>score"]
  Coord --> Seeds["5-seed batch<br/>[7, 42, 156]<br/>[8888, 42069]<br/>minimum bundle: 3"]
  Seeds --> IQA{"JoyQuality IQA<br/>score >= 0.55?"}
  IQA -- "fail" --> Persist["Persist candidate + failure evidence"]
  IQA -- "pass" --> VLM{"MiniCPM-V alignment<br/>score >= 0.25?"}
  VLM -- "fail" --> Persist
  VLM -- "pass" --> Impact{"TRIBE impact enabled?<br/>optional cutoff"}
  Impact -- "disabled or pass" --> Curated["Curated catalog<br/>fragile: 1 promoted<br/>viable: 2 promoted<br/>strong: >= 3 promoted"]
  Curated --> Feedback["Accept / reject / shred feedback"]
  Feedback --> Priors["Update priors<br/>enum arms alpha/beta<br/>enum-combo GP affinity"]
  Priors --> Persist
  Persist --> Stop{"Stop rule?"}
  Stop -- "Gradio runtime cap: 15 minutes" --> End["Stop run"]
  Stop -- "stall: fewer than 10 curated after 10 minutes" --> End
  Stop -- "backend/requested stop" --> End
  Stop -- "continue with updated priors" --> Coord
  classDef step fill:#eef8f6,stroke:#0f766e,color:#17211f
  classDef gate fill:#fff8e6,stroke:#a16207,color:#17211f
  class Mic,ASR,Typed,Prompt,Mellum,Parse,Locks,Coord,Seeds,Persist,Curated,Feedback,Priors,End step
  class IQA,VLM,Impact,Stop gate"""
WORKFLOW_MERMAID_MARKDOWN = f"```mermaid\n{WORKFLOW_MERMAID_CODE}\n```"
WORKFLOW_EXPLANATION_MARKDOWN = """### Workflow Steps

1. **Decomposition** extracts objects, relations, constraints, and cinematography lanes from typed text or ASR output.
2. **Mellum2 Thinking 12B via Modal Cloud** supplies the structured JSON reasoning path for prompt extraction, repair, and verification.
3. **Repair/Verify** checks blocking issues, unresolved targets, and threshold readiness before generation is allowed.
4. **Canonicalization** maps raw prompt values to project enums and exposes lock or unlock controls for pre-run review.
5. **LHS** proposes coverage-oriented coordinate rows across unlocked enum arms while preserving locked prompt evidence.
6. **Thompson Sampling/GP** ranks sampled arms with Bayesian feedback state and can use GP-style coordinate scoring as evaluation evidence accumulates.
7. **IQA** filters each 5-seed batch by JoyQuality against the configured quality cutoff.
8. **VLM alignment** scores surviving images against the compiled prompt and target manifest.
9. **TRIBE v2** remains disabled for now, but the impact gate is reserved for optional metacognitive scoring before catalog promotion.
10. **Prior updates** write evaluation and feedback evidence back into enum-arm alpha/beta priors and enum-combination GP affinity for the next coordinate.
"""


@dataclass
class RuntimeUISession:
    config: AppConfig
    service: Any
    controller: RunAppController
    document: PromptDocumentSpec
    rendered_prompt: RenderedPrompt
    target_manifest: EvaluationTargetManifest
    output_dir: Path


_RUNTIME_SESSIONS: dict[str, RuntimeUISession] = {}


class SimCandidate(StrictModel):
    candidate_id: str
    coordinate_id: str = COORDINATE_ID
    seed: int
    rendered_prompt: str
    image_path: str
    preview_path: str
    display_path: str
    quality_score: float | None = None
    alignment_score: float | None = None
    pass_iqa: bool | None = None
    pass_alignment: bool | None = None
    promoted: bool = False
    outcome: Literal["pending", "failed", "fragile", "viable", "strong"] = "pending"
    failure_reasons: list[str] = Field(default_factory=list)
    feedback_state: str | None = None


class GradioSimulationState(StrictModel):
    run_id: str = RUN_ID
    raw_prompt: str = ""
    rendered_prompt: str = ""
    prompt_document_id: str = PROMPT_DOCUMENT_ID
    batch_index: int = 0
    review: PreRunModalReadModel | None = None
    current_batch: list[SimCandidate] = Field(default_factory=list)
    curated: list[SimCandidate] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    generated_count: int = 0
    iqa_evaluated_count: int = 0
    vlm_evaluated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    shredded_count: int = 0
    notification: str = "Waiting for prompt."


def initial_state() -> GradioSimulationState:
    return GradioSimulationState()


def _coerce_state(value: GradioSimulationState | dict[str, Any] | None) -> GradioSimulationState:
    if value is None:
        return initial_state()
    if isinstance(value, GradioSimulationState):
        return value
    return GradioSimulationState.model_validate(value)


def _short_text(value: str, *, limit: int = 180, fallback: str = "subject") -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        normalized = fallback
    return normalized[:limit].strip() or fallback


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _evidence(text: str, reason: str | None = None) -> EvidenceSpan:
    return EvidenceSpan(
        text=_short_text(text),
        category=EvidenceCategory.EXPLICIT,
        reason=reason,
    )


def _relation_match(raw: str, enum_value: RelationType, reason: str) -> RelationEnumMatch:
    from bruteforce_canvas.shared import CanonicalStatus

    return RelationEnumMatch(
        raw=raw,
        enum_value=enum_value,
        status=CanonicalStatus.MATCHED_ACTIVE,
        confidence="clear",
        reason=reason,
    )


def build_prompt_document_for_demo(raw_prompt: str) -> PromptDocumentSpec:
    prompt = _short_text(raw_prompt, limit=500, fallback="image subject")
    lower = prompt.lower()
    elements: list[Element] = []
    descriptors: list[ObjectDescriptor] = []
    relations: list[RelationDescriptor] = []

    if "rose" in lower:
        elements.append(
            Element(
                id="object_01",
                label="rose",
                entity_type=EntityType.PRODUCT,
                role=ElementRole.PRIMARY_SUBJECT,
                importance=Importance.REQUIRED,
                evidence=_evidence("rose"),
            )
        )
        descriptors.append(
            ObjectDescriptor(
                target_id="object_01",
                color="red" if "red" in lower or "rose" in lower else None,
                description="magical glowing" if _has_any(lower, ("magical", "glowing", "glow")) else None,
            )
        )
        if "glass" in lower or "case" in lower or "encased" in lower:
            elements.append(
                Element(
                    id="object_02",
                    label="glass case",
                    entity_type=EntityType.CONTAINER,
                    role=ElementRole.SUPPORTING,
                    importance=Importance.REQUIRED,
                    evidence=_evidence("glass case"),
                )
            )
            descriptors.append(
                ObjectDescriptor(
                    target_id="object_02",
                    material="glass",
                    finish=Finish.GLOSSY,
                )
            )
            relations.append(
                RelationDescriptor(
                    id="rel_01",
                    source_id="object_01",
                    target_id="object_02",
                    relation_raw="inside",
                    relation_match=_relation_match("inside", RelationType.INSIDE, "rose is encased by the glass case"),
                    evidence=_evidence("rose encased in glass"),
                )
            )
    elif "bowl" in lower:
        elements.append(
            Element(
                id="object_01",
                label="bowl",
                entity_type=EntityType.PRODUCT,
                role=ElementRole.PRIMARY_SUBJECT,
                importance=Importance.REQUIRED,
                evidence=_evidence("bowl"),
            )
        )
        descriptors.append(
            ObjectDescriptor(
                target_id="object_01",
                material="ceramic" if "ceramic" in lower else None,
                color="red" if "red" in lower else "blue" if "blue" in lower else None,
            )
        )
        if "table" in lower:
            elements.append(
                Element(
                    id="object_02",
                    label="table",
                    entity_type=EntityType.FURNITURE,
                    role=ElementRole.SUPPORTING,
                    importance=Importance.REQUIRED,
                    evidence=_evidence("table"),
                )
            )
            descriptors.append(
                ObjectDescriptor(
                    target_id="object_02",
                    material="wooden" if "wood" in lower or "wooden" in lower else None,
                )
            )
            relations.append(
                RelationDescriptor(
                    id="rel_01",
                    source_id="object_01",
                    target_id="object_02",
                    relation_raw="on",
                    relation_match=_relation_match("on", RelationType.ON_TOP_OF, "bowl rests on the table"),
                    evidence=_evidence("bowl on table"),
                )
            )
    else:
        label = _short_text(prompt.split(",")[0], limit=40, fallback="subject")
        if len(label.split()) > 4:
            label = " ".join(label.split()[:4])
        elements.append(
            Element(
                id="object_01",
                label=label,
                entity_type=EntityType.UNKNOWN_SLOT,
                role=ElementRole.PRIMARY_SUBJECT,
                importance=Importance.REQUIRED,
                evidence=_evidence(label),
            )
        )
        descriptors.append(ObjectDescriptor(target_id="object_01", description=label))

    shot_size = ShotSize.CLOSE_UP if _has_any(lower, ("close-up", "close up", "macro")) else None
    optic = OpticCharacter.DREAM_GLOW if _has_any(lower, ("magical", "glowing", "glow", "dream")) else None
    lighting = None
    if "blue hour" in lower:
        lighting = LightingMood.BLUE_HOUR_TWILIGHT
    elif "soft" in lower:
        lighting = LightingMood.SOFT_NATURAL
    elif "neon" in lower:
        lighting = LightingMood.NEON_NIGHT

    guardrails = []
    if _has_any(lower, ("no people", "no extra people")):
        guardrails.append(Guardrail.NO_EXTRA_PEOPLE)
    if "no text" in lower:
        guardrails.append(Guardrail.NO_TEXT)

    issues: list[VerificationIssue] = []
    approved = True
    if "something" in lower or "unknown object" in lower:
        approved = False
        issues.append(
            VerificationIssue(
                issue_type="unresolved_action_target",
                repair_scope="prompt_improvement",
                blocking=True,
                message="Specify the unresolved object before generation.",
            )
        )

    return PromptDocumentSpec(
        prompt_document_id=PROMPT_DOCUMENT_ID,
        raw_user_prompt=_short_text(prompt),
        graph=SceneGraphDraft(seed_prompt=prompt, elements=elements, relations=relations),
        object_lane=ObjectLane(objects=descriptors),
        cinematography_lane=CinematographyLane(
            shot_size=shot_size,
            optic_character=optic,
            lighting_mood=lighting,
            framing=Framing.CENTERED if _has_any(lower, ("centered", "symmetrical")) else None,
        ),
        constraint_lane=ConstraintLane(guardrails=guardrails),
        verification=VerificationReport(approved=approved, issues=issues),
    )


def _render_prompt_for_demo(document: PromptDocumentSpec) -> str:
    if not document.verification.approved:
        return ""
    return render_prompt_spec(document).rendered_prompt


def _lock_table_from_review(review: PreRunModalReadModel) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for entry in review.lock_entries:
        rows.append(
            [
                str(entry.get("lock_state")) == "locked",
                str(entry.get("field_path", "")),
                str(entry.get("raw_value") or ""),
                str(entry.get("enum_value") or ""),
                str(entry.get("lhs_policy", "")),
                str(entry.get("canonical_status", "")),
            ]
        )
    return rows


def _normal_lock_rows(lock_rows: Any) -> list[list[Any]]:
    if lock_rows is None:
        return []
    if hasattr(lock_rows, "values"):
        return [list(row) for row in lock_rows.values.tolist()]
    if isinstance(lock_rows, dict):
        return [list(row) for row in lock_rows.get("data", [])]
    return [list(row) for row in lock_rows]


def _locked_field_count(lock_rows: Any) -> int:
    return sum(1 for row in _normal_lock_rows(lock_rows) if row and bool(row[0]))


_PROMPT_RETRY_HINT = "Try adding that detail, then submit the prompt again."
_PROMPT_INTERNAL_EVIDENCE_RETRY_HINT = (
    "Try rephrasing with the main objects, action, and relationship named explicitly, then submit again."
)


def _friendly_prompt_block_reason(raw_reason: str) -> str:
    normalized = raw_reason.lower()
    evidence_markers = (
        "evidence.text",
        "evidence.reason",
        "unresolved evidence",
        "non-explicit evidence requires text and reason",
        "requires unresolved evidence",
    )
    if any(marker in normalized for marker in evidence_markers):
        return "The parser could not trace some parsed objects or relationships back to exact words in your prompt."
    return raw_reason


def _prompt_block_reasons(review: PreRunModalReadModel) -> list[str]:
    seen: set[str] = set()
    reasons: list[str] = []
    for item in review.prompt_improvement_feedback:
        reason = _friendly_prompt_block_reason(str(item).strip())
        if reason and reason not in seen:
            seen.add(reason)
            reasons.append(reason)
    return reasons


def _prompt_retry_hint_for_reasons(reasons: list[str]) -> str:
    if any("could not trace" in reason.lower() for reason in reasons):
        return _PROMPT_INTERNAL_EVIDENCE_RETRY_HINT
    return _PROMPT_RETRY_HINT


def _needs_parseable_prompt_suggestion(reasons: list[str]) -> bool:
    return any("could not trace" in reason.lower() for reason in reasons)


def _prompt_fragment(text: str) -> str:
    fragment = _short_text(text.strip().rstrip("."), limit=90, fallback="a simple studio scene")
    if fragment[:2].lower() == "a ":
        return "a " + fragment[2:]
    if fragment[:3].lower() == "an ":
        return "an " + fragment[3:]
    if fragment[:4].lower() == "the ":
        return "the " + fragment[4:]
    return fragment[:1].lower() + fragment[1:]


def _parseable_prompt_suggestion(raw_prompt: str, document: PromptDocumentSpec) -> str:
    label_by_id = {
        element.id: _prompt_fragment(str(element.label))
        for element in document.graph.elements
        if str(element.label).strip()
    }
    for relation in document.graph.relations:
        source = label_by_id.get(relation.source_id)
        target = label_by_id.get(relation.target_id)
        if source and target and relation.relation_raw:
            return (
                f"Create a clear image of {source} {relation.relation_raw} {target}. "
                f"Make {source} the main subject and {target} clearly visible."
            )

    labels = list(label_by_id.values())
    if len(labels) >= 2:
        return (
            f"Create a clear image of {labels[0]} next to {labels[1]}. "
            f"Make {labels[0]} the main subject and {labels[1]} a supporting object."
        )
    if labels:
        label = labels[0]
        if len(label.split()) >= 3 or any(word in label.lower() for word in ("scene", "studio", "setting")):
            return (
                f"Create a clear image of {label} featuring a red glass sphere on a black pedestal. "
                "The red glass sphere is the main subject, and the black pedestal supports it."
            )
        return (
            f"Create a clear image of {label} on a wooden table. "
            f"Make {label} the main subject and the wooden table the supporting surface."
        )

    scene = _prompt_fragment(raw_prompt)
    return (
        f"Create a clear image of {scene}, featuring a red glass sphere on a black pedestal. "
        "The red glass sphere is the main subject, and the black pedestal supports it."
    )


def _parseable_prompt_suggestion_for_block(
    raw_prompt: str,
    document: PromptDocumentSpec,
    review: PreRunModalReadModel,
) -> str | None:
    reasons = _prompt_block_reasons(review)
    if not _needs_parseable_prompt_suggestion(reasons):
        return None
    return _parseable_prompt_suggestion(raw_prompt, document)


def _prompt_blocked_notification(
    review: PreRunModalReadModel,
    *,
    ready_message: str,
    parseable_prompt: str | None = None,
) -> str:
    if review.can_begin_generation:
        return ready_message
    reasons = _prompt_block_reasons(review)
    retry_hint = _prompt_retry_hint_for_reasons(reasons)
    suggestion = f' Try this prompt: "{parseable_prompt}"' if parseable_prompt else ""
    if not reasons:
        return f"Prompt parse blocked. {retry_hint}{suggestion}"
    return f"Prompt parse blocked: {reasons[0]} {retry_hint}{suggestion}"


def _compiled_prompt_blocked_markup(raw_prompt: str, document: PromptDocumentSpec, review: PreRunModalReadModel) -> str:
    if review.can_begin_generation:
        return ""
    reasons = _prompt_block_reasons(review)
    reason = reasons[0] if reasons else "The verifier needs one more concrete prompt detail before generation can start."
    retry_hint = _prompt_retry_hint_for_reasons(reasons)
    parseable_prompt = _parseable_prompt_suggestion_for_block(raw_prompt, document, review)
    suggestion = ""
    if parseable_prompt is not None:
        suggestion = (
            '<div class="bc-suggested-prompt">'
            '<span>Try this prompt:</span>'
            f"<code>{escape(parseable_prompt)}</code>"
            "</div>"
        )
    return (
        '<div class="bc-blocked-explainer">'
        f"<strong>Compiled prompt blocked.</strong> {escape(reason)} "
        f"{escape(retry_hint)}"
        f"{suggestion}"
        "</div>"
    )


def _review_markdown(raw_prompt: str, document: PromptDocumentSpec, review: PreRunModalReadModel, rendered: str) -> str:
    descriptor_by_target = {
        descriptor.target_id: descriptor.model_dump(exclude_none=True, mode="json")
        for descriptor in document.object_lane.objects
    }
    element_cards = []
    for element in document.graph.elements:
        descriptor = descriptor_by_target.get(element.id, {})
        descriptor_items = [
            f"{key.replace('_', ' ')}: {value}"
            for key, value in descriptor.items()
            if key != "target_id" and value is not None and value != "" and value != []
        ]
        descriptor_markup = "".join(f'<span class="bc-token">{escape(str(item))}</span>' for item in descriptor_items)
        if not descriptor_markup:
            descriptor_markup = '<span class="bc-token bc-token-muted">no extra descriptors</span>'
        element_cards.append(
            '<article class="bc-object-card">'
            f'<div class="bc-object-id">{escape(element.id)}: {escape(element.label)}</div>'
            '<div class="bc-object-tags">'
            f'<span>{escape(str(element.entity_type))}</span>'
            f'<span>{escape(str(element.role))}</span>'
            f'<span>{escape(str(element.importance))}</span>'
            "</div>"
            f'<div class="bc-object-meta">{descriptor_markup}</div>'
            "</article>"
        )
    element_markup = "".join(element_cards) or '<p class="bc-muted">No parsed objects.</p>'

    relation_cards = []
    for relation in document.graph.relations:
        line = f"{relation.source_id} {relation.relation_raw} {relation.target_id}"
        enum_value = relation.relation_match.enum_value if relation.relation_match else "unmatched"
        relation_cards.append(
            '<article class="bc-relation-card">'
            f'<div class="bc-relation-line">{escape(line)}</div>'
            f'<div class="bc-relation-meta">canonical: {escape(str(enum_value))}</div>'
            "</article>"
        )
    relation_markup = "".join(relation_cards) or '<p class="bc-muted">No explicit relations.</p>'

    editable = "".join(f'<span class="bc-token">{escape(str(item))}</span>' for item in review.editable_fields)
    if not editable:
        editable = '<span class="bc-token bc-token-muted">none</span>'
    feedback_items = _prompt_block_reasons(review) if not review.can_begin_generation else []
    if not feedback_items:
        feedback_items = ["clear"]
    feedback = "".join(f"<li>{escape(str(item))}</li>" for item in feedback_items)
    rendered_line = rendered or "blocked"
    state_class = "ready" if review.can_begin_generation else "blocked"
    return (
        '<div class="bc-review">'
        '<section class="bc-review-hero">'
        '<div>'
        '<div class="bc-eyebrow">Pre-run parse</div>'
        f'<h2>{escape(raw_prompt)}</h2>'
        f'<p>Document <code>{escape(document.prompt_document_id)}</code></p>'
        "</div>"
        f'<span class="bc-state-pill bc-state-{state_class}">{escape(str(review.state))}</span>'
        "</section>"
        '<section class="bc-review-grid">'
        '<article class="bc-review-block bc-review-block-wide">'
        '<h3>Objects</h3>'
        f'<div class="bc-object-grid">{element_markup}</div>'
        "</article>"
        '<article class="bc-review-block">'
        '<h3>Relations</h3>'
        f"{relation_markup}"
        "</article>"
        '<article class="bc-review-block">'
        '<h3>Generation controls</h3>'
        f'<div class="bc-token-stack">{editable}</div>'
        "</article>"
        '<article class="bc-review-block bc-review-block-wide">'
        '<h3>Validation</h3>'
        f'<ul class="bc-validation-list">{feedback}</ul>'
        '<h3>Compiled prompt</h3>'
        f"{_compiled_prompt_blocked_markup(raw_prompt, document, review)}"
        f'<pre class="bc-compiled-prompt">{escape(rendered_line)}</pre>'
        "</article>"
        "</section>"
        "</div>"
    )


def _candidate_digest(raw_prompt: str, seed: int, batch_index: int, salt: str) -> float:
    payload = f"{raw_prompt}|{seed}|{batch_index}|{salt}".encode("utf-8")
    value = int(hashlib.sha256(payload).hexdigest()[:8], 16)
    return value / 0xFFFFFFFF


def _score(raw_prompt: str, seed: int, batch_index: int, salt: str, *, floor: float, span: float) -> float:
    return round(min(0.99, floor + _candidate_digest(raw_prompt, seed, batch_index, salt) * span), 3)


def _outcome(quality: float, alignment: float, iqa_cutoff: float, alignment_cutoff: float) -> str:
    if quality < iqa_cutoff or alignment < alignment_cutoff:
        return "failed"
    margin = min(quality - iqa_cutoff, alignment - alignment_cutoff)
    if margin >= 0.28:
        return "strong"
    if margin >= 0.12:
        return "viable"
    return "fragile"


def _base_color(seed: int) -> tuple[int, int, int]:
    palette = [
        (47, 111, 122),
        (122, 79, 47),
        (74, 120, 84),
        (136, 92, 38),
        (78, 89, 133),
    ]
    return palette[seed % len(palette)]


def _draw_prompt_scene(draw: ImageDraw.ImageDraw, prompt: str, seed: int) -> None:
    lower = prompt.lower()
    if "rose" in lower:
        draw.ellipse((288, 120, 480, 480), outline=(180, 215, 220), width=10)
        draw.rectangle((270, 460, 498, 500), fill=(180, 215, 220))
        draw.line((384, 430, 384, 260), fill=(53, 119, 79), width=10)
        for offset in (-42, -18, 18, 42):
            draw.ellipse((342 + offset, 205, 420 + offset, 285), fill=(172, 37, 58))
        draw.ellipse((340, 210, 428, 300), fill=(203, 49, 74))
    elif "bowl" in lower:
        draw.ellipse((240, 220, 528, 405), fill=(225, 231, 222), outline=(55, 67, 74), width=8)
        draw.arc((235, 180, 533, 410), 0, 180, fill=(55, 67, 74), width=8)
        draw.rectangle((160, 430, 608, 470), fill=(134, 93, 58))
    else:
        draw.ellipse((230, 155, 360, 285), fill=(230, 237, 229))
        draw.polygon([(430, 150), (555, 330), (345, 330)], fill=(213, 227, 218))
        draw.rectangle((295, 375, 510, 500), fill=(225, 231, 222))
    draw.text((32, 34), f"seed {seed}", fill=(245, 246, 244))


def _write_candidate_image(
    *,
    path: Path,
    prompt: str,
    seed: int,
    outcome: str,
    quality: float | None = None,
    alignment: float | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = _base_color(seed)
    image = Image.new("RGB", (768, 640), base)
    draw = ImageDraw.Draw(image)
    for index in range(0, 768, 48):
        draw.line((index, 0, index - 180, 640), fill=tuple(max(0, channel - 22) for channel in base), width=3)
    _draw_prompt_scene(draw, prompt, seed)
    if quality is not None and alignment is not None:
        draw.rounded_rectangle((28, 530, 740, 605), radius=12, fill=(245, 246, 244))
        draw.text(
            (52, 552),
            f"IQA {quality:.2f}  ALIGN {alignment:.2f}  {outcome.upper()}",
            fill=(23, 31, 36),
        )
    if outcome == "failed":
        image = ImageEnhance.Color(image).enhance(0.08)
        image = ImageEnhance.Brightness(image).enhance(0.42)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 760, 632), outline=(211, 47, 47), width=18)
        draw.rounded_rectangle((502, 30, 720, 100), radius=10, fill=(211, 47, 47))
        draw.text((536, 54), "FAILED", fill=(255, 255, 255))
    elif outcome in {"fragile", "viable", "strong"}:
        color = {
            "fragile": (207, 140, 39),
            "viable": (45, 131, 117),
            "strong": (46, 130, 72),
        }[outcome]
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 760, 632), outline=color, width=12)
    image.save(path)
    return str(path)


def _make_pending_batch(state: GradioSimulationState, lock_rows: Any) -> list[SimCandidate]:
    batch_index = state.batch_index + 1
    coordinate_id = f"coord_{batch_index:03d}"
    prompt = state.rendered_prompt or f"Generate {state.raw_prompt}"
    locked_count = _locked_field_count(lock_rows)
    candidates = []
    for seed in DEFAULT_SEED_BUNDLE:
        candidate_id = f"cand_{batch_index}_{seed}"
        output_dir = ASSET_DIR / state.run_id / f"batch_{batch_index}"
        preview_path = output_dir / f"seed_{seed}_preview.png"
        path = _write_candidate_image(path=preview_path, prompt=prompt, seed=seed, outcome="pending")
        candidates.append(
            SimCandidate(
                candidate_id=candidate_id,
                coordinate_id=coordinate_id,
                seed=seed,
                rendered_prompt=f"{prompt}. locked_fields={locked_count}; candidate_seed={seed}",
                image_path=path,
                preview_path=path,
                display_path=path,
            )
        )
    return candidates


def _evaluate_batch(
    state: GradioSimulationState,
    candidates: list[SimCandidate],
    *,
    iqa_cutoff: float,
    alignment_cutoff: float,
) -> list[SimCandidate]:
    evaluated = []
    for candidate in candidates:
        quality = _score(state.raw_prompt, candidate.seed, state.batch_index + 1, "quality", floor=0.35, span=0.60)
        alignment = _score(state.raw_prompt, candidate.seed, state.batch_index + 1, "alignment", floor=0.18, span=0.75)
        outcome = _outcome(quality, alignment, iqa_cutoff, alignment_cutoff)
        reasons = []
        if quality < iqa_cutoff:
            reasons.append("quality_below_cutoff")
        if alignment < alignment_cutoff:
            reasons.append("alignment_below_cutoff")
        display_path = _write_candidate_image(
            path=ASSET_DIR / state.run_id / f"batch_{state.batch_index + 1}" / f"seed_{candidate.seed}_evaluated.png",
            prompt=state.rendered_prompt or state.raw_prompt,
            seed=candidate.seed,
            outcome=outcome,
            quality=quality,
            alignment=alignment,
        )
        evaluated.append(
            candidate.model_copy(
                update={
                    "display_path": display_path,
                    "quality_score": quality,
                    "alignment_score": alignment,
                    "pass_iqa": quality >= iqa_cutoff,
                    "pass_alignment": alignment >= alignment_cutoff,
                    "promoted": outcome != "failed",
                    "outcome": outcome,
                    "failure_reasons": reasons,
                }
            )
        )
    return evaluated


def _preview_gallery(candidates: list[SimCandidate]) -> list[tuple[str, str]]:
    gallery = []
    for candidate in candidates:
        if candidate.outcome == "pending":
            caption = f"seed {candidate.seed} | pending"
        elif candidate.outcome == "failed":
            reasons = ", ".join(candidate.failure_reasons) or "failed"
            caption = f"seed {candidate.seed} | failed | {reasons}"
        else:
            caption = f"seed {candidate.seed} | {candidate.outcome} | Q {candidate.quality_score:.2f} A {candidate.alignment_score:.2f}"
        gallery.append((candidate.display_path, caption))
    return gallery


def _visible_catalog(state: GradioSimulationState) -> list[SimCandidate]:
    return [candidate for candidate in state.curated if candidate.feedback_state not in {"reject", "shred"}]


def _catalog_gallery(state: GradioSimulationState) -> list[tuple[str, str]]:
    items = []
    for candidate in _visible_catalog(state):
        accepted = " | accepted" if candidate.feedback_state == "accept" else ""
        items.append(
            (
                candidate.display_path,
                f"{candidate.outcome} | seed {candidate.seed} | Q {candidate.quality_score:.2f} A {candidate.alignment_score:.2f}{accepted}",
            )
        )
    return items


def _workspace_model(state: GradioSimulationState) -> RunWorkspaceReadModel:
    return RunWorkspaceReadModel(
        run_id=state.run_id,
        raw_user_prompt=state.raw_prompt or "",
        run_state="running" if state.current_batch else "waiting_for_pre_run_confirmation",
        generated_count=state.generated_count,
        iqa_evaluated_count=state.iqa_evaluated_count,
        vlm_evaluated_count=state.vlm_evaluated_count,
        promoted_curated_count=len(_visible_catalog(state)),
        accepted_count=state.accepted_count,
        rejected_count=state.rejected_count,
        shredded_count=state.shredded_count,
        stall_guard_state="healthy" if state.current_batch else "inactive",
        notification=state.notification,
    )


_STATUS_VALUE_LABELS = {
    "waiting_for_pre_run_confirmation": "waiting for pre-run confirmation",
    "inactive": "inactive",
    "healthy": "healthy",
    "running": "running",
}


def _status_label(value: str) -> str:
    return value.replace("_", " ")


def _status_value(value: Any) -> str:
    text = str(value)
    return _STATUS_VALUE_LABELS.get(text, text.replace("_", " "))


def _status_html(state: GradioSimulationState) -> str:
    heartbeat = _workspace_model(state).progress_heartbeat
    chips = "".join(
        '<span class="bc-chip">'
        f'<span class="bc-chip-key">{escape(_status_label(key))}</span>'
        f'<span class="bc-chip-value">{escape(_status_value(value))}</span>'
        "</span>"
        for key, value in heartbeat.items()
        if key != "vram_telemetry"
    )
    return f'<div class="bc-status" aria-live="polite">{chips}<span class="bc-note">{escape(state.notification)}</span></div>'


def _candidate_card(candidate: SimCandidate) -> CandidateCard:
    return CandidateCard(
        candidate_id=candidate.candidate_id,
        promoted=candidate.promoted,
        curated=candidate.promoted,
        thumbnail_path=candidate.display_path,
        seed=candidate.seed,
        optional_tags=[candidate.outcome],
        feedback_action=FeedbackAction(candidate.feedback_state) if candidate.feedback_state else None,
        accepted=candidate.feedback_state == "accept",
    )


def _detail_report(state: GradioSimulationState, candidate: SimCandidate) -> DetailReport:
    return DetailReport.from_candidate_card(
        _candidate_card(candidate),
        run_id=state.run_id,
        raw_user_prompt=state.raw_prompt,
        prompt_document_id=state.prompt_document_id,
        target_manifest_id=TARGET_MANIFEST_ID,
        coordinate_id=candidate.coordinate_id,
        rendered_prompt=candidate.rendered_prompt,
        generator_model_id="gradio-sim-generator",
        generator_backend="simulated-pil-raster",
        generation_settings={"steps": 4, "seed_bundle": list(DEFAULT_SEED_BUNDLE)},
        coordinate_enum_json={
            "candidate_seed": candidate.seed,
            "outcome": candidate.outcome,
            "tribe_metacognitive_score": "disabled",
        },
        compatibility_trace={"simulation": True},
        bayesian_score_before_generation=_candidate_digest(state.raw_prompt, candidate.seed, state.batch_index, "gp"),
        quality_score=candidate.quality_score or 0.0,
        alignment_score=candidate.alignment_score or 0.0,
        promotion_thresholds={"quality_cutoff": "configured", "alignment_cutoff": "configured"},
        promotion_gate_reasons=candidate.failure_reasons or ["quality and alignment passed"],
        image_path=candidate.display_path,
    )


def _detail_markdown(state: GradioSimulationState, candidate: SimCandidate | None) -> str:
    if candidate is None:
        return ""
    report = _detail_report(state, candidate)
    reasons = ", ".join(report.promotion_gate_reasons) or "none"
    feedback = candidate.feedback_state or "unreviewed"
    return (
        f"**Candidate** `{report.candidate_id}`\n\n"
        f"**Compiled prompt**\n\n{report.rendered_prompt}\n\n"
        f"**Seed** `{report.seed}`\n\n"
        f"**Scores** IQA `{report.quality_score:.3f}` | Alignment `{report.alignment_score:.3f}` | TRIBE `disabled`\n\n"
        f"**Metadata** run `{report.run_id}` | prompt `{report.prompt_document_id}` | coordinate `{report.coordinate_id}`\n\n"
        f"**Gate reasons** {reasons}\n\n"
        f"**Feedback** `{feedback}`"
    )


def _resolve_gradio_mode(mode: GradioMode | str | None = None) -> GradioMode:
    requested = mode or os.environ.get("BC_GRADIO_MODE") or os.environ.get("BC_GRADIO_BACKEND")
    if requested is None:
        generator = os.environ.get("BC_GENERATOR", GeneratorKind.STUB.value)
        return "runtime" if generator != GeneratorKind.STUB.value else "simulation"
    if requested not in {"simulation", "runtime"}:
        raise ValueError("Gradio mode must be 'simulation' or 'runtime'")
    return requested  # type: ignore[return-value]


def _new_runtime_run_id() -> str:
    return f"run_{int(time.time() * 1000)}"


def _runtime_output_dir(run_id: str) -> Path:
    return RUNTIME_RUN_ROOT / run_id / "images"


def _runtime_config_for_prompt(raw_prompt: str) -> AppConfig:
    base = load_app_config()
    run_id = _new_runtime_run_id()
    run_root = RUNTIME_RUN_ROOT / run_id
    run = base.run.model_copy(
        update={
            "run_id": run_id,
            "raw_user_prompt": raw_prompt,
            "mode": "continuous",
            "stall_window_seconds": RUNTIME_STALL_WINDOW_SECONDS,
            "stall_min_promoted": RUNTIME_STALL_MIN_PROMOTED,
        }
    )
    return base.model_copy(update={"event_store_path": run_root / "events.jsonl", "run": run})


def _runtime_real_eval_default(config: AppConfig) -> bool:
    return config.generator.kind != GeneratorKind.STUB.value


def _runtime_device(config: AppConfig) -> Literal["cpu", "cuda", "auto"]:
    device = config.device.device
    if device in {"cpu", "cuda", "auto"}:
        return device  # type: ignore[return-value]
    return "auto"


def _build_runtime_iqa(config: AppConfig) -> object:
    mode = os.environ.get("BC_IQA_MODE")
    if mode is None:
        mode = "real" if _runtime_real_eval_default(config) else "static"
    if mode == "static":
        return StaticIQAAdapter(scores=[0.85 for _seed in DEFAULT_SEED_BUNDLE], model_id="static-runtime-quality")
    if mode != "real":
        raise ValueError("BC_IQA_MODE must be 'real' or 'static'")
    from bruteforce_canvas.real_adapters import JoyQualityAdapter

    return JoyQualityAdapter(mode="real", device=_runtime_device(config))


def _build_runtime_vlm(config: AppConfig) -> object:
    mode = os.environ.get("BC_VLM_MODE")
    if mode is None:
        mode = "real" if _runtime_real_eval_default(config) else "static"
    if mode == "static":
        return StaticVLMAdapter(scores=[0.90 for _seed in DEFAULT_SEED_BUNDLE], model_id="static-runtime-alignment")
    if mode != "real":
        raise ValueError("BC_VLM_MODE must be 'real' or 'static'")
    return build_vlm_adapter(config)


def _build_runtime_impact(config: AppConfig) -> object | None:
    if not config.run.metacognitive_impact_enabled:
        return None
    mode = os.environ.get("BC_IMPACT_MODE", "real" if _runtime_real_eval_default(config) else "static")
    if mode == "static":
        return StaticImpactAdapter(
            scores=[0.50 for _seed in DEFAULT_SEED_BUNDLE],
            enabled=True,
            model_id="static-runtime-impact",
        )
    if mode != "real":
        raise ValueError("BC_IMPACT_MODE must be 'real' or 'static'")
    from bruteforce_canvas.real_adapters import TRIBEv2Adapter

    return TRIBEv2Adapter(enabled=True, mode="real", device=_runtime_device(config))


def _build_runtime_service(config: AppConfig) -> Any:
    iqa = _build_runtime_iqa(config)
    vlm = _build_runtime_vlm(config)
    impact = _build_runtime_impact(config)
    return build_run_service(config, iqa=iqa, vlm=vlm, impact=impact)


def _runtime_generator_backend(config: AppConfig) -> str:
    if config.generator.kind in {GeneratorKind.BONSAI.value, GeneratorKind.BONSAI_HTTP.value}:
        return "bonsai-ternary-gemlite"
    return str(config.generator.kind)


def _runtime_generator_model_id(config: AppConfig) -> str:
    if config.generator.kind == GeneratorKind.BONSAI.value:
        if config.generator.bonsai_model_root.exists():
            return str(config.generator.bonsai_model_root)
        return "prism-ml/bonsai-image-ternary-4B-gemlite-2bit"
    if config.generator.kind == GeneratorKind.BONSAI_HTTP.value:
        return f"bonsai-http:{config.generator.bonsai_http_url}"
    return f"{config.generator.kind}-generator"


def _runtime_fixed_arms(lock_rows: Any) -> dict[str, AxisDomain]:
    fixed: dict[str, AxisDomain] = {}
    for row in _normal_lock_rows(lock_rows):
        if len(row) < 4 or not bool(row[0]):
            continue
        field_path = str(row[1]).strip()
        raw_value = str(row[2]).strip()
        enum_value = str(row[3]).strip()
        value = enum_value or raw_value
        if not field_path or not value:
            continue
        fixed[field_path] = AxisDomain(
            value=value,
            state=FieldState.EXPLICIT_LOCKED,
            source="gradio_lock_table",
        )
    return fixed


def _runtime_target_manifest(
    session: RuntimeUISession,
    *,
    coordinate_id: str | None = None,
) -> EvaluationTargetManifest:
    return session.target_manifest.model_copy(
        update={
            "run_id": session.config.run.run_id,
            "prompt_document_id": session.document.prompt_document_id,
            "coordinate_id": coordinate_id,
            "rendered_prompt": session.rendered_prompt.rendered_prompt,
        }
    )


def _runtime_evaluation_plan(config: AppConfig, *, iqa_cutoff: float, alignment_cutoff: float) -> EvaluationPlan:
    plan = build_evaluation_plan(config)
    return plan.model_copy(
        update={
            "quality_cutoff": float(iqa_cutoff),
            "alignment_cutoff": float(alignment_cutoff),
            "human_quality_cutoff": max(float(iqa_cutoff), float(config.run.human_iqa_cutoff)),
            "execution_preference": "serialized",
        }
    )


def _persist_runtime_inputs(session: RuntimeUISession, target_manifest: EvaluationTargetManifest) -> None:
    run_id = session.config.run.run_id
    document = session.document.model_copy(update={"raw_user_prompt": session.config.run.raw_user_prompt})
    session.service.store.append(
        PersistenceRecord(
            record_id=f"prompt_document:{document.prompt_document_id}",
            record_type="prompt_document",
            run_id=run_id,
            prompt_document_id=document.prompt_document_id,
            idempotency_key=f"prompt_document:{document.prompt_document_id}",
            payload={
                **document.model_dump(mode="json"),
                "persistence_version": PERSISTENCE_VERSION,
            },
        )
    )
    session.service.store.append(
        PersistenceRecord(
            record_id=f"target_manifest:{target_manifest.manifest_id}:{target_manifest.coordinate_id}",
            record_type="target_manifest",
            run_id=run_id,
            prompt_document_id=document.prompt_document_id,
            target_manifest_id=target_manifest.manifest_id,
            coordinate_id=target_manifest.coordinate_id,
            idempotency_key=f"target_manifest:{target_manifest.manifest_id}:{target_manifest.coordinate_id}",
            payload={
                **target_manifest.model_dump(mode="json"),
                "persistence_version": PERSISTENCE_VERSION,
            },
        )
    )


def _persist_runtime_batch_summary(
    session: RuntimeUISession,
    *,
    coordinate_id: str,
    batch_index: int,
    elapsed_seconds: int,
) -> None:
    try:
        counts = reconstruct_run_state(session.service.store.replay())
    except ValueError:
        counts = None
    session.service.store.append(
        PersistenceRecord(
            record_id=f"runtime_batch_summary:{coordinate_id}",
            record_type="runtime_batch_summary",
            run_id=session.config.run.run_id,
            prompt_document_id=session.document.prompt_document_id,
            target_manifest_id=session.target_manifest.manifest_id,
            coordinate_id=coordinate_id,
            idempotency_key=f"runtime_batch_summary:{coordinate_id}",
            payload={
                "batch_index": batch_index,
                "elapsed_seconds": int(elapsed_seconds),
                "generated_count": counts.generated_count if counts is not None else 0,
                "promoted_curated_count": counts.promoted_curated_count if counts is not None else 0,
                "persistence_version": PERSISTENCE_VERSION,
            },
        )
    )


def _runtime_work_item_for_batch(
    session: RuntimeUISession,
    *,
    lock_rows: Any,
    fixed_arms: dict[str, AxisDomain],
    evaluation_plan: EvaluationPlan,
    batch_index: int,
) -> SeedSweepWorkItem:
    router_batch = LHSRouter(seed=7).propose(
        RouterInput(
            run_id=session.config.run.run_id,
            prompt_document_id=session.document.prompt_document_id,
            target_manifest_id=session.target_manifest.manifest_id,
            fixed_arms=fixed_arms,
            sampleable_axes={},
            count=1,
        )
    )
    if not router_batch.coordinates:
        raise gr.Error("Router rejected the locked coordinate configuration.")
    coordinate_id = f"coord_{batch_index:03d}"
    coordinate = router_batch.coordinates[0].model_copy(update={"coordinate_id": coordinate_id})
    target_manifest = _runtime_target_manifest(session, coordinate_id=coordinate.coordinate_id)
    generation_settings = GenerationSettings(
        steps=int(os.environ.get("BC_GRADIO_GENERATION_STEPS", "4")),
        height=int(os.environ.get("BC_GRADIO_IMAGE_HEIGHT", "512")),
        width=int(os.environ.get("BC_GRADIO_IMAGE_WIDTH", "512")),
        backend=_runtime_generator_backend(session.config),
    )
    requests = seed_sweep_requests(
        run_id=session.config.run.run_id,
        prompt_document_id=session.document.prompt_document_id,
        target_manifest_id=target_manifest.manifest_id,
        coordinate_id=coordinate.coordinate_id,
        rendered_prompt=session.rendered_prompt.rendered_prompt,
        generation_settings=generation_settings,
        output_dir=session.output_dir,
        generator_model_id=_runtime_generator_model_id(session.config),
        generator_backend=_runtime_generator_backend(session.config),
        candidate_id_prefix=f"cand_{coordinate.coordinate_id}",
    )
    _persist_runtime_inputs(session, target_manifest)
    return SeedSweepWorkItem(
        run_id=session.config.run.run_id,
        raw_user_prompt=session.config.run.raw_user_prompt,
        prompt_document_version=session.document.prompt_document_version,
        coordinate_id=coordinate.coordinate_id,
        rendered_prompt=session.rendered_prompt.rendered_prompt,
        target_manifest=target_manifest.model_dump(mode="json"),
        generation_requests=requests,
        evaluation_plan=evaluation_plan,
        sampled_arms=coordinate.sampled_arms,
        locked_arms=coordinate.fixed_arms,
        lhs_row=coordinate.lhs_row,
        lock_configuration={"rows": _normal_lock_rows(lock_rows)},
        effective_lock_configuration=coordinate.fixed_arms,
        compatibility_trace=coordinate.compatibility_trace,
        bayesian_score_before_generation=coordinate.bayesian_score,
        combo_signature=coordinate.combo_signature or f"fixed_only:{coordinate.coordinate_id}",
    )


def _format_runtime_elapsed(seconds: int) -> str:
    minutes, remainder = divmod(max(0, int(seconds)), 60)
    return f"{minutes}m {remainder:02d}s"


def _runtime_latest_coordinate_id(records: list[PersistenceRecord]) -> str | None:
    for record in reversed(records):
        if record.coordinate_id and record.record_type in {
            "evaluation_aggregate",
            "runtime_batch_summary",
            "candidate_record",
            "coordinate_record",
        }:
            return str(record.coordinate_id)
    return None


def _runtime_progress_notification(records: list[PersistenceRecord], *, batch_index: int, elapsed_seconds: int) -> str:
    try:
        counts = reconstruct_run_state(records)
    except ValueError:
        return f"Batch {batch_index} complete: 0 curated, 0 generated, elapsed {_format_runtime_elapsed(elapsed_seconds)}."
    return (
        f"Batch {batch_index} complete: {counts.promoted_curated_count} curated, "
        f"{counts.generated_count} generated, elapsed {_format_runtime_elapsed(elapsed_seconds)}."
    )


def _runtime_batch_notification(records: list[PersistenceRecord]) -> str:
    aggregate = next((record for record in reversed(records) if record.record_type == "evaluation_aggregate"), None)
    if aggregate is None:
        return "Backend run complete."
    promoted = int(aggregate.payload.get("promoted_count", 0))
    generated = int(aggregate.payload.get("generated_count", 0))
    outcome = str(aggregate.payload.get("outcome", "complete"))
    return f"Backend evaluation complete: {promoted}/{generated} promoted ({outcome})."


def _runtime_candidates_from_records(records: list[PersistenceRecord]) -> list[SimCandidate]:
    evaluations = {
        str(record.candidate_id): record
        for record in records
        if record.record_type == "image_evaluation" and record.candidate_id is not None
    }
    feedback = {
        str(record.candidate_id): str(record.payload.get("feedback_action"))
        for record in records
        if record.record_type == "feedback" and record.candidate_id is not None
    }
    candidates: list[SimCandidate] = []
    for record in records:
        if record.record_type != "candidate_record" or record.candidate_id is None:
            continue
        payload = record.payload
        evaluation = evaluations.get(str(record.candidate_id))
        quality: float | None = None
        alignment: float | None = None
        pass_iqa: bool | None = None
        pass_alignment: bool | None = None
        promoted = False
        failure_reasons: list[str] = []
        outcome: Literal["pending", "failed", "fragile", "viable", "strong"] = "pending"
        if evaluation is not None:
            eval_payload = evaluation.payload
            quality = float(eval_payload.get("quality", {}).get("score", 0.0))
            alignment = float(eval_payload.get("alignment", {}).get("score", 0.0))
            flags = dict(eval_payload.get("pass_flags", {}))
            pass_iqa = bool(flags.get("quality", False))
            pass_alignment = bool(flags.get("alignment", False))
            disposition = dict(eval_payload.get("disposition_signal", {}))
            promoted = bool(flags.get("full", False)) and disposition.get("class_name") == "passes_thresholds"
            failure_reasons = [str(reason) for reason in disposition.get("reasons", [])]
            failure_reasons.extend(str(item) for item in eval_payload.get("failure_types", []))
            if promoted:
                plan = dict(eval_payload.get("evaluator_plan", {}) or {})
                outcome = _outcome(
                    quality,
                    alignment,
                    float(plan.get("quality_cutoff", 0.55)),
                    float(plan.get("alignment_cutoff", 0.25)),
                )
            else:
                outcome = "failed"
        image_path = str(payload.get("image_path", ""))
        candidates.append(
            SimCandidate(
                candidate_id=str(record.candidate_id),
                coordinate_id=str(record.coordinate_id or payload.get("coordinate_id") or COORDINATE_ID),
                seed=int(record.seed or payload.get("seed", 0)),
                rendered_prompt=str(payload.get("rendered_prompt", "")),
                image_path=image_path,
                preview_path=image_path,
                display_path=image_path,
                quality_score=quality,
                alignment_score=alignment,
                pass_iqa=pass_iqa,
                pass_alignment=pass_alignment,
                promoted=promoted,
                outcome=outcome,
                failure_reasons=sorted(set(reason for reason in failure_reasons if reason)),
                feedback_state=feedback.get(str(record.candidate_id)),
            )
        )
    return candidates


def _runtime_state_from_store(
    session: RuntimeUISession,
    state: GradioSimulationState,
    *,
    current_coordinate_id: str | None = None,
    notification: str | None = None,
) -> GradioSimulationState:
    records = session.service.store.replay()
    candidates = _runtime_candidates_from_records(records)
    curated = [candidate for candidate in candidates if candidate.promoted]
    current_coordinate_id = current_coordinate_id or _runtime_latest_coordinate_id(records)
    current_batch = (
        [candidate for candidate in candidates if candidate.coordinate_id == current_coordinate_id]
        if current_coordinate_id is not None
        else candidates
    )
    selected_id = state.selected_candidate_id
    visible_ids = {candidate.candidate_id for candidate in curated if candidate.feedback_state not in {"reject", "shred"}}
    if selected_id not in visible_ids:
        selected_id = next(iter(visible_ids), None)

    counts = None
    if records:
        try:
            counts = reconstruct_run_state(records)
        except ValueError:
            counts = None

    updates: dict[str, Any] = {
        "run_id": session.config.run.run_id,
        "raw_prompt": session.config.run.raw_user_prompt,
        "rendered_prompt": session.rendered_prompt.rendered_prompt,
        "prompt_document_id": session.document.prompt_document_id,
        "current_batch": current_batch,
        "curated": curated,
        "selected_candidate_id": selected_id,
        "notification": notification or _runtime_batch_notification(records),
    }
    if counts is not None:
        updates.update(
            {
                "generated_count": counts.generated_count,
                "iqa_evaluated_count": counts.iqa_evaluated_count,
                "vlm_evaluated_count": counts.vlm_evaluated_count,
                "accepted_count": counts.accepted_count,
                "rejected_count": counts.rejected_count,
                "shredded_count": counts.shredded_count,
                "batch_index": max(state.batch_index, len(counts.coordinate_ids)),
            }
        )
    return state.model_copy(update=updates)


def _selected_candidate(state: GradioSimulationState) -> SimCandidate | None:
    if state.selected_candidate_id is None:
        return None
    return next((candidate for candidate in _visible_catalog(state) if candidate.candidate_id == state.selected_candidate_id), None)


def start_pre_run(raw_prompt: str, state_value: GradioSimulationState | dict[str, Any] | None):
    state = _coerce_state(state_value)
    if not raw_prompt or not raw_prompt.strip():
        raise gr.Error("Prompt required.")
    document = build_prompt_document_for_demo(raw_prompt)
    review = pre_run_modal_from_prompt(document)
    rendered = _render_prompt_for_demo(document)
    parseable_prompt = _parseable_prompt_suggestion_for_block(raw_prompt, document, review)
    state = state.model_copy(
        update={
            "raw_prompt": raw_prompt.strip(),
            "rendered_prompt": rendered,
            "review": review,
            "notification": _prompt_blocked_notification(
                review,
                ready_message="Pre-run parse ready.",
                parseable_prompt=parseable_prompt,
            ),
        }
    )
    return (
        state,
        gr.update(visible=True),
        _review_markdown(raw_prompt, document, review, rendered),
        _lock_table_from_review(review),
        gr.update(interactive=review.can_begin_generation),
        _status_html(state),
    )


def cancel_pre_run(state_value: GradioSimulationState | dict[str, Any] | None):
    state = _coerce_state(state_value).model_copy(update={"notification": "Pre-run canceled."})
    return state, gr.update(visible=False), gr.update(interactive=False), _status_html(state)


def start_pre_run_runtime(raw_prompt: str, state_value: GradioSimulationState | dict[str, Any] | None):
    state = _coerce_state(state_value)
    prompt = raw_prompt.strip() if raw_prompt else ""
    if not prompt:
        raise gr.Error("Prompt required.")

    try:
        config = _runtime_config_for_prompt(prompt)
        pipeline = build_prompt_pipeline(config)
        result = pipeline.run_spec(prompt)
    except Exception as error:
        raise gr.Error(f"Backend pre-run startup failed: {error}") from error

    document = result.document.model_copy(update={"raw_user_prompt": prompt})
    review = pre_run_modal_from_prompt(document)
    rendered_prompt = result.rendered_prompt
    rendered = rendered_prompt.rendered_prompt if rendered_prompt is not None else ""
    if review.can_begin_generation and rendered_prompt is None:
        try:
            rendered_prompt = render_prompt_spec(document)
            rendered = rendered_prompt.rendered_prompt
        except Exception as error:
            raise gr.Error(f"Prompt rendered invalidly after approval: {error}") from error

    if review.can_begin_generation and rendered_prompt is not None:
        try:
            target_manifest = target_manifest_from_prompt_spec(document).model_copy(
                update={
                    "run_id": config.run.run_id,
                    "prompt_document_id": document.prompt_document_id,
                    "rendered_prompt": rendered_prompt.rendered_prompt,
                }
            )
            rendered_prompt = rendered_prompt.model_copy(
                update={
                    "run_id": config.run.run_id,
                    "target_manifest_id": target_manifest.manifest_id,
                }
            )
            service = _build_runtime_service(config)
            session = RuntimeUISession(
                config=config,
                service=service,
                controller=RunAppController(service),
                document=document,
                rendered_prompt=rendered_prompt,
                target_manifest=target_manifest,
                output_dir=_runtime_output_dir(config.run.run_id),
            )
            _RUNTIME_SESSIONS[config.run.run_id] = session
        except Exception as error:
            raise gr.Error(f"Backend runtime prewarm failed: {error}") from error

    state = state.model_copy(
        update={
            "run_id": config.run.run_id,
            "raw_prompt": prompt,
            "rendered_prompt": rendered,
            "prompt_document_id": document.prompt_document_id,
            "batch_index": 0,
            "review": review,
            "current_batch": [],
            "curated": [],
            "selected_candidate_id": None,
            "generated_count": 0,
            "iqa_evaluated_count": 0,
            "vlm_evaluated_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "shredded_count": 0,
            "notification": _prompt_blocked_notification(
                review,
                ready_message="Backend pre-run ready.",
                parseable_prompt=_parseable_prompt_suggestion_for_block(prompt, document, review),
            ),
        }
    )
    return (
        state,
        gr.update(visible=True),
        _review_markdown(prompt, document, review, rendered),
        _lock_table_from_review(review),
        gr.update(interactive=review.can_begin_generation),
        _status_html(state),
    )


def cancel_pre_run_runtime(state_value: GradioSimulationState | dict[str, Any] | None):
    state = _coerce_state(state_value)
    session = _RUNTIME_SESSIONS.pop(state.run_id, None)
    if session is not None:
        session.service.request_pause()
    state = state.model_copy(update={"notification": "Backend pre-run canceled."})
    return state, gr.update(visible=False), gr.update(interactive=False), _status_html(state)


def generate_seed_sweep(
    state_value: GradioSimulationState | dict[str, Any] | None,
    lock_rows: Any,
    iqa_cutoff: float,
    alignment_cutoff: float,
):
    state = _coerce_state(state_value)
    if state.review is None or not state.review.can_begin_generation:
        raise gr.Error("Pre-run review is not ready.")
    pending = _make_pending_batch(state, lock_rows)
    generated_state = state.model_copy(
        update={
            "current_batch": pending,
            "batch_index": state.batch_index + 1,
            "generated_count": state.generated_count + len(pending),
            "notification": "5-seed preview generated.",
        }
    )
    selected = _selected_candidate(generated_state)
    yield (
        generated_state,
        gr.update(visible=False),
        gr.update(visible=True),
        _preview_gallery(pending),
        _catalog_gallery(generated_state),
        gr.update(visible=selected is not None),
        selected.display_path if selected else None,
        _detail_markdown(generated_state, selected),
        _status_html(generated_state),
    )
    time.sleep(0.6)
    evaluated = _evaluate_batch(state, pending, iqa_cutoff=iqa_cutoff, alignment_cutoff=alignment_cutoff)
    promoted = [candidate for candidate in evaluated if candidate.promoted]
    selected_id = state.selected_candidate_id
    if selected_id is None and promoted:
        selected_id = promoted[0].candidate_id
    evaluated_state = generated_state.model_copy(
        update={
            "current_batch": evaluated,
            "curated": [*state.curated, *promoted],
            "selected_candidate_id": selected_id,
            "iqa_evaluated_count": state.iqa_evaluated_count + len(evaluated),
            "vlm_evaluated_count": state.vlm_evaluated_count + sum(1 for item in evaluated if item.pass_iqa),
            "notification": f"Evaluation complete: {len(promoted)} promoted.",
        }
    )
    selected = _selected_candidate(evaluated_state)
    yield (
        evaluated_state,
        gr.update(visible=False),
        gr.update(visible=True),
        _preview_gallery(evaluated),
        _catalog_gallery(evaluated_state),
        gr.update(visible=selected is not None),
        selected.display_path if selected else None,
        _detail_markdown(evaluated_state, selected),
        _status_html(evaluated_state),
    )


def generate_seed_sweep_runtime(
    state_value: GradioSimulationState | dict[str, Any] | None,
    lock_rows: Any,
    iqa_cutoff: float,
    alignment_cutoff: float,
):
    state = _coerce_state(state_value)
    if state.review is None or not state.review.can_begin_generation:
        raise gr.Error("Pre-run review is not ready.")
    session = _RUNTIME_SESSIONS.get(state.run_id)
    if session is None:
        raise gr.Error("Runtime backend session is missing. Submit the prompt again to rebuild it.")

    active_state = state.model_copy(
        update={
            "notification": "Backend run started.",
        }
    )
    yield (
        active_state,
        gr.update(visible=False),
        gr.update(visible=True),
        [],
        _catalog_gallery(active_state),
        gr.update(visible=False),
        None,
        "",
        _status_html(active_state),
    )

    loop_started_at = time.monotonic()
    fixed_arms = _runtime_fixed_arms(lock_rows)
    evaluation_plan = _runtime_evaluation_plan(
        session.config,
        iqa_cutoff=float(iqa_cutoff),
        alignment_cutoff=float(alignment_cutoff),
    )
    working_state = active_state
    while True:
        elapsed_seconds = int(time.monotonic() - loop_started_at)
        if elapsed_seconds >= RUNTIME_LOOP_LIMIT_SECONDS:
            session.service.stop_with_reason(
                "gradio_runtime_time_limit",
                details={
                    "elapsed_seconds": elapsed_seconds,
                    "limit_seconds": RUNTIME_LOOP_LIMIT_SECONDS,
                    "run_id": session.config.run.run_id,
                },
            )
            stopped_state = _runtime_state_from_store(
                session,
                working_state,
                notification="Stopped at 15-minute time limit.",
            )
            selected = _selected_candidate(stopped_state)
            yield (
                stopped_state,
                gr.update(visible=False),
                gr.update(visible=True),
                _preview_gallery(stopped_state.current_batch),
                _catalog_gallery(stopped_state),
                gr.update(visible=selected is not None),
                selected.display_path if selected else None,
                _detail_markdown(stopped_state, selected),
                _status_html(stopped_state),
            )
            return

        batch_index = working_state.batch_index + 1
        item = _runtime_work_item_for_batch(
            session,
            lock_rows=lock_rows,
            fixed_arms=fixed_arms,
            evaluation_plan=evaluation_plan,
            batch_index=batch_index,
        )
        session.service.enqueue(item)
        decision = session.service.tick()
        elapsed_seconds = int(time.monotonic() - loop_started_at)

        if decision.next_state == RunRuntimeState.STOPPED and decision.reason != "pending_coordinates":
            stopped_state = _runtime_state_from_store(
                session,
                working_state,
                current_coordinate_id=item.coordinate_id,
                notification="Stopped by backend/requested stop.",
            )
            selected = _selected_candidate(stopped_state)
            yield (
                stopped_state,
                gr.update(visible=False),
                gr.update(visible=True),
                _preview_gallery(stopped_state.current_batch),
                _catalog_gallery(stopped_state),
                gr.update(visible=selected is not None),
                selected.display_path if selected else None,
                _detail_markdown(stopped_state, selected),
                _status_html(stopped_state),
            )
            return

        _persist_runtime_batch_summary(
            session,
            coordinate_id=item.coordinate_id,
            batch_index=batch_index,
            elapsed_seconds=elapsed_seconds,
        )
        records = session.service.store.replay()
        notification = _runtime_progress_notification(
            records,
            batch_index=batch_index,
            elapsed_seconds=elapsed_seconds,
        )
        next_state = _runtime_state_from_store(
            session,
            working_state,
            current_coordinate_id=item.coordinate_id,
            notification=notification,
        )

        if elapsed_seconds >= RUNTIME_LOOP_LIMIT_SECONDS:
            session.service.stop_with_reason(
                "gradio_runtime_time_limit",
                details={
                    "elapsed_seconds": elapsed_seconds,
                    "limit_seconds": RUNTIME_LOOP_LIMIT_SECONDS,
                    "run_id": session.config.run.run_id,
                },
            )
            next_state = _runtime_state_from_store(
                session,
                next_state,
                current_coordinate_id=item.coordinate_id,
                notification="Stopped at 15-minute time limit.",
            )
        else:
            stall_decision = session.service.stop_for_stall_guard_if_needed()
            if stall_decision is not None:
                next_state = _runtime_state_from_store(
                    session,
                    next_state,
                    current_coordinate_id=item.coordinate_id,
                    notification="Stopped by stall guard: fewer than 10 curated images after 10 minutes.",
                )
            elif session.service.state == RunRuntimeState.STOPPED:
                next_state = _runtime_state_from_store(
                    session,
                    next_state,
                    current_coordinate_id=item.coordinate_id,
                    notification="Stopped by backend/requested stop.",
                )

        selected = _selected_candidate(next_state)
        yield (
            next_state,
            gr.update(visible=False),
            gr.update(visible=True),
            _preview_gallery(next_state.current_batch),
            _catalog_gallery(next_state),
            gr.update(visible=selected is not None),
            selected.display_path if selected else None,
            _detail_markdown(next_state, selected),
            _status_html(next_state),
        )
        if session.service.state == RunRuntimeState.STOPPED:
            return
        working_state = next_state


def select_curated(evt: gr.SelectData, state_value: GradioSimulationState | dict[str, Any] | None):
    state = _coerce_state(state_value)
    index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    visible = _visible_catalog(state)
    if not isinstance(index, int) or index < 0 or index >= len(visible):
        return state, gr.update(visible=False), None, "", _status_html(state)
    candidate = visible[index]
    state = state.model_copy(
        update={
            "selected_candidate_id": candidate.candidate_id,
            "notification": f"Selected seed {candidate.seed}.",
        }
    )
    return state, gr.update(visible=True), candidate.display_path, _detail_markdown(state, candidate), _status_html(state)


def move_selection(state_value: GradioSimulationState | dict[str, Any] | None, delta: int):
    state = _coerce_state(state_value)
    visible = _visible_catalog(state)
    if not visible:
        return state, gr.update(visible=False), None, "", _status_html(state)
    ids = [candidate.candidate_id for candidate in visible]
    if state.selected_candidate_id in ids:
        next_index = (ids.index(state.selected_candidate_id) + delta) % len(visible)
    else:
        next_index = 0
    candidate = visible[next_index]
    state = state.model_copy(
        update={
            "selected_candidate_id": candidate.candidate_id,
            "notification": f"Selected seed {candidate.seed}.",
        }
    )
    return state, gr.update(visible=True), candidate.display_path, _detail_markdown(state, candidate), _status_html(state)


def submit_feedback(
    state_value: GradioSimulationState | dict[str, Any] | None,
    action_value: str,
):
    state = _coerce_state(state_value)
    selected = _selected_candidate(state)
    if selected is None:
        return state, _catalog_gallery(state), gr.update(visible=False), None, "", _status_html(state)
    action = FeedbackAction(action_value)
    submit_feedback_event(run_id=state.run_id, candidate_id=selected.candidate_id, action=action)
    updated_curated = [
        candidate.model_copy(update={"feedback_state": action.value})
        if candidate.candidate_id == selected.candidate_id
        else candidate
        for candidate in state.curated
    ]
    next_state = state.model_copy(
        update={
            "curated": updated_curated,
            "accepted_count": state.accepted_count + (1 if action == FeedbackAction.ACCEPT else 0),
            "rejected_count": state.rejected_count + (1 if action == FeedbackAction.REJECT else 0),
            "shredded_count": state.shredded_count + (1 if action == FeedbackAction.SHRED else 0),
            "notification": f"Feedback recorded: {action.value}.",
        }
    )
    if action in {FeedbackAction.REJECT, FeedbackAction.SHRED}:
        visible = _visible_catalog(next_state)
        next_state = next_state.model_copy(
            update={"selected_candidate_id": visible[0].candidate_id if visible else None}
        )
    selected_after = _selected_candidate(next_state)
    return (
        next_state,
        _catalog_gallery(next_state),
        gr.update(visible=selected_after is not None),
        selected_after.display_path if selected_after else None,
        _detail_markdown(next_state, selected_after),
        _status_html(next_state),
    )


def submit_feedback_runtime(
    state_value: GradioSimulationState | dict[str, Any] | None,
    action_value: str,
):
    state = _coerce_state(state_value)
    selected = _selected_candidate(state)
    if selected is None:
        return state, _catalog_gallery(state), gr.update(visible=False), None, "", _status_html(state)
    session = _RUNTIME_SESSIONS.get(state.run_id)
    if session is None:
        raise gr.Error("Runtime backend session is missing. Submit the prompt again to rebuild it.")
    action = FeedbackAction(action_value)
    try:
        session.controller.handle_event(
            submit_feedback_event(run_id=state.run_id, candidate_id=selected.candidate_id, action=action)
        )
    except Exception as error:
        raise gr.Error(f"Feedback rejected by backend: {error}") from error
    next_state = _runtime_state_from_store(
        session,
        state,
        notification=f"Feedback recorded: {action.value}.",
    )
    selected_after = _selected_candidate(next_state)
    return (
        next_state,
        _catalog_gallery(next_state),
        gr.update(visible=selected_after is not None),
        selected_after.display_path if selected_after else None,
        _detail_markdown(next_state, selected_after),
        _status_html(next_state),
    )


def transcribe_microphone_to_prompt(
    audio: object,
    current_prompt: str,
    state_value: GradioSimulationState | dict[str, Any] | None,
):
    state = _coerce_state(state_value)
    if audio is None:
        return current_prompt, state, _status_html(state)
    try:
        transcript = default_transcriber().transcribe(audio)
    except Exception as error:
        state = state.model_copy(update={"notification": f"ASR failed: {error}"})
        return current_prompt, state, _status_html(state)
    prompt = transcript.strip()
    if not prompt:
        state = state.model_copy(update={"notification": "ASR returned an empty transcript."})
        return current_prompt, state, _status_html(state)
    state = state.model_copy(update={"raw_prompt": prompt, "notification": "ASR transcript inserted."})
    return prompt, state, _status_html(state)


def transcribe_microphone_to_prompt_steps(
    audio: object,
    current_prompt: str,
    state_value: GradioSimulationState | dict[str, Any] | None,
):
    state = _coerce_state(state_value)
    if audio is None:
        yield gr.update(value=current_prompt, interactive=True), state, _status_html(state)
        return
    processing_state = state.model_copy(update={"notification": "Transcribing microphone audio."})
    yield gr.update(value="Transcribing audio...", interactive=False), processing_state, _status_html(processing_state)
    prompt, next_state, status = transcribe_microphone_to_prompt(audio, current_prompt, state)
    yield gr.update(value=prompt, interactive=True), next_state, status


CSS = """
:root {
    --bc-ink: #17211f;
    --bc-muted: #586760;
    --bc-page: #eef3f0;
    --bc-surface: #ffffff;
    --bc-surface-soft: #f4f7f5;
    --bc-surface-raised: #fbfdfc;
    --bc-line: #c7d1cc;
    --bc-line-strong: #87938e;
    --bc-teal: #0f766e;
    --bc-teal-dark: #0f3d36;
    --bc-green: #2f7d4f;
    --bc-amber: #a16207;
    --bc-red: #b42318;
}
.gradio-container {
    background: var(--bc-page) !important;
    color: var(--bc-ink);
}
.gradio-container * {
    box-sizing: border-box;
}
.gradio-container footer {
    display: none !important;
}
#bc-app {
    max-width: 1440px;
    margin: 0 auto;
    padding: 18px 20px 32px;
}
#bc-title {
    border-bottom: 2px solid var(--bc-line);
    padding: 8px 0 16px;
    margin-bottom: 16px;
}
#bc-title h1 {
    margin: 0;
    font-size: 32px;
    line-height: 1.05;
    letter-spacing: 0;
    font-weight: 720;
    color: var(--bc-ink);
}
#workflow-accordion {
    border: 1px solid var(--bc-line);
    border-radius: 8px;
    background: var(--bc-surface);
    margin: 0 0 16px;
    box-shadow: 0 1px 0 rgba(23, 33, 31, 0.04);
}
#workflow-accordion .label-wrap,
#workflow-accordion [data-testid="block-label"] {
    background: #f8faf9 !important;
    color: var(--bc-teal-dark) !important;
    border-bottom: 1px solid var(--bc-line) !important;
    font-weight: 780 !important;
}
.bc-workflow-row {
    align-items: stretch;
    gap: 14px;
    padding: 12px;
}
.bc-workflow-diagram,
.bc-workflow-explanation {
    border: 1px solid var(--bc-line);
    border-radius: 8px;
    background: var(--bc-surface-raised);
    padding: 12px;
    min-height: 100%;
    overflow: auto;
}
#workflow-mermaid,
#workflow-explanation {
    color: var(--bc-ink) !important;
}
#workflow-explanation,
#workflow-explanation *,
#workflow-mermaid,
#workflow-mermaid * {
    opacity: 1 !important;
}
#workflow-explanation p,
#workflow-explanation li,
#workflow-explanation span {
    color: var(--bc-ink) !important;
}
#workflow-mermaid pre {
    background: #fbfdfc !important;
    border: 1px solid #d9e2de !important;
    border-radius: 8px !important;
}
#workflow-mermaid svg text,
#workflow-mermaid svg tspan,
#workflow-mermaid svg foreignObject,
#workflow-mermaid svg foreignObject div,
#workflow-mermaid svg foreignObject span,
#workflow-mermaid svg foreignObject p,
#workflow-mermaid .nodeLabel,
#workflow-mermaid .nodeLabel p,
#workflow-mermaid .edgeLabel,
#workflow-mermaid .label {
    color: var(--bc-ink) !important;
    fill: var(--bc-ink) !important;
    -webkit-text-fill-color: var(--bc-ink) !important;
    font-size: 14px !important;
}
#workflow-mermaid svg foreignObject,
#workflow-mermaid svg foreignObject div,
#workflow-mermaid svg foreignObject p,
#workflow-mermaid .label {
    overflow: visible !important;
}
#workflow-mermaid .node rect,
#workflow-mermaid .node polygon,
#workflow-mermaid .node path {
    fill: #eef8f6 !important;
    stroke: #0f766e !important;
}
#workflow-mermaid .edgePath path {
    stroke: var(--bc-ink) !important;
}
#workflow-explanation h3 {
    margin: 0 0 10px;
    color: var(--bc-teal-dark) !important;
    font-size: 17px;
    line-height: 1.25;
    letter-spacing: 0;
}
#workflow-explanation ol {
    margin: 0;
    padding-left: 22px;
}
#workflow-explanation li {
    margin: 0 0 8px;
    line-height: 1.45;
}
#workflow-explanation li:last-child {
    margin-bottom: 0;
}
#workflow-explanation strong {
    color: var(--bc-ink) !important;
}
.bc-prompt-row {
    --bc-recorder-width: clamp(11rem, 14vw, 14rem);
    align-items: stretch;
    gap: 12px;
    border: 1px solid var(--bc-line);
    border-radius: 8px;
    background: var(--bc-surface);
    padding: 12px;
    margin-bottom: 16px;
    box-shadow: 0 1px 0 rgba(23, 33, 31, 0.04);
}
#prompt-input,
#prompt-microphone {
    margin: 0 !important;
}
.bc-prompt-row > .form {
    align-self: stretch;
    display: flex !important;
}
#prompt-input,
#prompt-input > div,
#prompt-input .wrap,
#prompt-input textarea {
    background: var(--bc-surface-raised) !important;
    color: var(--bc-ink) !important;
}
#prompt-input {
    flex: 1 1 auto;
    height: 100% !important;
}
#prompt-input .wrap {
    min-height: 100%;
    height: 100%;
    border: 1px solid var(--bc-line-strong) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
#prompt-input textarea {
    min-height: 100%;
    height: 100% !important;
    font-size: 16px;
    border: 0 !important;
    padding: 15px 14px !important;
    box-shadow: none !important;
}
#prompt-input textarea:focus {
    outline: 2px solid rgba(15, 118, 110, 0.24) !important;
    outline-offset: -2px;
}
#prompt-input textarea:disabled,
#prompt-input textarea[disabled],
#prompt-input [aria-disabled="true"] textarea {
    opacity: 1 !important;
    background: #fffdf7 !important;
    color: var(--bc-ink) !important;
    -webkit-text-fill-color: var(--bc-ink) !important;
}
#prompt-input .wrap:has(textarea:disabled),
#prompt-input:has(textarea:disabled) {
    background: #fffdf7 !important;
    border-color: #d1b36d !important;
}
#prompt-microphone {
    flex: 0 0 var(--bc-recorder-width) !important;
    width: var(--bc-recorder-width) !important;
    min-width: var(--bc-recorder-width) !important;
    max-width: var(--bc-recorder-width) !important;
    aspect-ratio: 4 / 3;
    align-self: stretch;
    overflow: hidden !important;
}
#prompt-microphone,
#prompt-microphone *,
#prompt-microphone *::before,
#prompt-microphone *::after {
    animation: none !important;
    transition: none !important;
}
#prompt-microphone,
#prompt-microphone > div,
#prompt-microphone .wrap {
    background: var(--bc-surface-raised) !important;
    color: var(--bc-ink) !important;
}
#prompt-microphone [class*="container"],
#prompt-microphone [class*="source"],
#prompt-microphone [class*="waveform"],
#prompt-microphone [class*="empty"],
#prompt-microphone [data-testid] {
    background: var(--bc-surface-raised) !important;
    color: var(--bc-ink) !important;
    border-color: var(--bc-line) !important;
}
#prompt-microphone .audio-container,
#prompt-microphone .component-wrapper,
#prompt-microphone .microphone,
#prompt-microphone .controls,
#prompt-microphone .controls .wrapper,
#prompt-microphone .mic-select,
#prompt-microphone select {
    background: var(--bc-surface-raised) !important;
    color: var(--bc-ink) !important;
    border-color: var(--bc-line) !important;
}
#prompt-microphone .mic-select,
#prompt-microphone select {
    width: 100% !important;
    min-height: 1.65rem !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
    font-size: 11px !important;
}
#prompt-microphone .mic-select:disabled,
#prompt-microphone select:disabled {
    opacity: 1 !important;
    background: #f4f7f5 !important;
    color: #586760 !important;
}
#prompt-microphone .audio-container {
    min-height: 100% !important;
    height: 100% !important;
    aspect-ratio: 4 / 3;
    overflow: hidden !important;
}
#prompt-microphone .wrap {
    min-height: 100%;
    height: 100%;
    aspect-ratio: 4 / 3;
    border: 1px solid var(--bc-line) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    overflow: hidden !important;
}
#prompt-microphone .component-wrapper {
    min-height: 100% !important;
    height: 100% !important;
    display: grid !important;
    grid-template-rows: 1fr auto !important;
    align-items: end !important;
    gap: 0.35rem !important;
    padding: 1.75rem 0.5rem 0.5rem !important;
    overflow: hidden !important;
}
#prompt-microphone .microphone,
#prompt-microphone [data-testid="recording-waveform"] {
    display: none !important;
}
#prompt-microphone .controls {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 0.4rem !important;
    overflow: hidden !important;
}
#prompt-microphone .controls .wrapper {
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 0.35rem !important;
    min-height: 2.5rem !important;
    overflow: hidden !important;
}
#prompt-microphone * {
    color: var(--bc-ink) !important;
}
#prompt-microphone button {
    border-radius: 6px !important;
    border-color: var(--bc-line) !important;
    background: #ffffff !important;
    color: var(--bc-ink) !important;
    box-shadow: none !important;
}
#prompt-microphone button:disabled {
    opacity: 1 !important;
}
#prompt-microphone .controls button,
#prompt-microphone .controls button * {
    color: inherit !important;
    -webkit-text-fill-color: currentColor !important;
}
#prompt-microphone button:hover {
    background: #eef8f6 !important;
    border-color: #9fb9b2 !important;
}
#prompt-microphone .icon-button {
    width: 22px !important;
    height: 22px !important;
    min-width: 22px !important;
    min-height: 22px !important;
    padding: 2px !important;
}
#prompt-microphone .record-button,
#prompt-microphone .stop-button,
#prompt-microphone .stop-button-paused,
#prompt-microphone .resume-button {
    min-width: 6.7rem !important;
    min-height: 2.35rem !important;
    padding: 0.55rem 0.75rem !important;
    font-size: 13px !important;
    line-height: 1.1 !important;
}
#prompt-microphone .record-button {
    border-color: #9fb9b2 !important;
    background: #ffffff !important;
    color: var(--bc-teal-dark) !important;
    font-weight: 760 !important;
}
#prompt-microphone .stop-button,
#prompt-microphone .stop-button-paused {
    border-color: #d92d20 !important;
    background: #fff1f0 !important;
    color: #7a271a !important;
    font-weight: 780 !important;
}
#prompt-microphone .resume-button {
    border-color: #9fb9b2 !important;
    background: #eef8f6 !important;
    color: var(--bc-teal-dark) !important;
    font-weight: 760 !important;
}
#prompt-microphone .pause-button {
    width: 2.35rem !important;
    height: 2.35rem !important;
    min-width: 2.35rem !important;
    min-height: 2.35rem !important;
    padding: 0.45rem !important;
    border-color: #d1b36d !important;
    background: #fff8e6 !important;
    color: #553a08 !important;
}
#prompt-submit {
    height: auto !important;
    min-height: 100% !important;
    align-self: stretch;
}
#bc-app button.primary {
    border: 1px solid var(--bc-teal) !important;
    background: var(--bc-teal) !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-weight: 760 !important;
    box-shadow: none !important;
}
#bc-app button.primary:hover {
    background: #0d665f !important;
    border-color: #0d665f !important;
}
#pre-run-panel,
#active-panel,
#detail-panel {
    border: 1px solid var(--bc-line);
    border-radius: 8px;
    padding: 14px;
    background: var(--bc-surface);
    box-shadow: 0 1px 0 rgba(23, 33, 31, 0.04);
}
#pre-run-panel {
    border-left: 5px solid var(--bc-teal);
}
#active-panel {
    border-left: 5px solid var(--bc-amber);
}
#detail-panel {
    border-left: 5px solid var(--bc-green);
}
#seed-gallery,
#catalog-gallery {
    --gallery-gap: 12px;
    border: 1px solid var(--bc-line) !important;
    border-radius: 8px !important;
    background: var(--bc-surface) !important;
    overflow: hidden;
}
#seed-gallery img,
#catalog-gallery img,
#detail-image img {
    border-radius: 6px;
}
#seed-gallery,
#seed-gallery > div,
#seed-gallery .gallery,
#seed-gallery .grid-container,
#seed-gallery .empty,
#catalog-gallery,
#catalog-gallery > div,
#catalog-gallery .gallery {
    background: #f8faf9 !important;
    color: var(--bc-ink) !important;
}
#catalog-gallery .grid-container,
#catalog-gallery .empty,
#catalog-gallery .preview,
#catalog-gallery .thumbnail-lg,
#catalog-gallery .thumbnail-item {
    background: #f8faf9 !important;
    color: var(--bc-ink) !important;
}
#seed-gallery svg,
#catalog-gallery svg {
    color: #789089 !important;
    stroke: #789089 !important;
    opacity: 0.7;
}
#seed-gallery .label-wrap,
#catalog-gallery .label-wrap,
#seed-gallery .block-label,
#catalog-gallery .block-label,
#seed-gallery [data-testid="block-label"],
#catalog-gallery [data-testid="block-label"],
#seed-gallery label,
#catalog-gallery label {
    background: #ffffff !important;
    color: var(--bc-ink) !important;
    border-bottom: 1px solid var(--bc-line) !important;
    border-radius: 0 !important;
}
.bc-review {
    color: var(--bc-ink);
}
.bc-review-hero {
    display: flex;
    gap: 16px;
    align-items: flex-start;
    justify-content: space-between;
    border-bottom: 1px solid var(--bc-line);
    padding-bottom: 12px;
    margin-bottom: 12px;
}
.bc-review-hero h2 {
    margin: 2px 0 6px;
    font-size: 20px;
    line-height: 1.25;
    letter-spacing: 0;
}
.bc-review-hero p {
    margin: 0;
    color: var(--bc-muted);
}
.bc-eyebrow {
    color: var(--bc-teal-dark);
    font-size: 12px;
    font-weight: 780;
    text-transform: uppercase;
    letter-spacing: 0;
}
.bc-state-pill {
    flex: 0 0 auto;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 12px;
    font-weight: 760;
    border: 1px solid var(--bc-line-strong);
    background: #fff;
}
.bc-state-ready {
    color: var(--bc-teal-dark);
    border-color: #72aaa0;
    background: #e7f4f1;
}
.bc-state-blocked {
    color: var(--bc-red);
    border-color: #e19b96;
    background: #fff1f0;
}
.bc-review-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
    gap: 12px;
}
.bc-review-block {
    border: 1px solid var(--bc-line);
    border-radius: 8px;
    background: var(--bc-surface-soft);
    padding: 12px;
}
.bc-review-block-wide {
    grid-column: 1 / -1;
}
.bc-review-block h3 {
    margin: 0 0 9px;
    font-size: 14px;
    line-height: 1.25;
    letter-spacing: 0;
    color: var(--bc-teal-dark);
}
.bc-object-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 10px;
}
.bc-object-card,
.bc-relation-card {
    border: 1px solid #b9c8c1;
    border-radius: 8px;
    background: var(--bc-surface);
    padding: 10px;
}
.bc-object-id,
.bc-relation-line {
    font-weight: 780;
    line-height: 1.3;
}
.bc-object-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
}
.bc-object-tags span {
    border: 1px solid #d1b36d;
    background: #fff8e6;
    color: #553a08;
    border-radius: 6px;
    padding: 3px 7px;
    font-size: 12px;
    line-height: 1.25;
}
.bc-object-meta,
.bc-token-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.bc-token {
    display: inline-flex;
    align-items: center;
    min-height: 24px;
    border: 1px solid #9fb9b2;
    background: #eef8f6;
    color: var(--bc-teal-dark);
    border-radius: 6px;
    padding: 3px 7px;
    font-size: 12px;
    line-height: 1.25;
    overflow-wrap: anywhere;
}
.bc-token-muted,
.bc-muted {
    color: var(--bc-muted);
}
.bc-relation-card + .bc-relation-card {
    margin-top: 8px;
}
.bc-relation-meta {
    margin-top: 5px;
    color: var(--bc-muted);
    font-size: 12px;
}
.bc-validation-list {
    margin: 0 0 12px;
    padding-left: 18px;
}
.bc-validation-list li {
    margin: 4px 0;
}
.bc-blocked-explainer {
    margin: 0 0 10px;
    border: 1px solid #e19b96;
    border-radius: 8px;
    background: #fff1f0;
    color: #7f1d1d;
    padding: 9px 10px;
    line-height: 1.4;
    overflow-wrap: anywhere;
}
.bc-suggested-prompt {
    display: grid;
    gap: 5px;
    margin-top: 8px;
}
.bc-suggested-prompt span {
    font-weight: 760;
}
.bc-suggested-prompt code {
    display: block;
    border: 1px solid #e8b8b4;
    border-radius: 7px;
    background: #fffafa;
    color: #5f1717;
    padding: 8px;
    white-space: normal;
    overflow-wrap: anywhere;
}
.bc-compiled-prompt {
    white-space: pre-wrap;
    margin: 0;
    border: 1px solid #b9c8c1;
    border-radius: 8px;
    background: #111827;
    color: #f8fafc;
    padding: 10px;
    font-size: 13px;
    line-height: 1.45;
    overflow-wrap: anywhere;
}
.bc-status {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    border: 1px solid #87938e;
    border-radius: 8px;
    padding: 10px;
    background: #ffffff;
    color: #111827;
    margin-bottom: 18px;
}
.bc-chip {
    display: inline-grid;
    grid-template-columns: auto auto;
    font-size: 12px;
    line-height: 1.35;
    border: 1px solid #5f6f68;
    border-radius: 6px;
    padding: 0;
    background: #ffffff;
    color: #111827;
    font-weight: 500;
    overflow-wrap: anywhere;
    overflow: hidden;
}
.bc-chip b,
.bc-chip-key {
    color: #0f3d36;
    background: #eef8f6;
    font-weight: 750;
    padding: 5px 7px;
    border-right: 1px solid #c7d1cc;
    white-space: nowrap;
}
.bc-chip-value {
    padding: 5px 7px;
    color: #111827;
    min-width: 22px;
}
.bc-note {
    font-size: 13px;
    margin-left: auto;
    color: #12352f;
    font-weight: 600;
}
@media (max-width: 760px) {
    #bc-title h1 {
        font-size: 28px;
    }
    #bc-app {
        padding: 10px;
    }
    .bc-review-hero {
        flex-direction: column;
    }
    .bc-review-grid {
        grid-template-columns: 1fr;
    }
    .bc-review-block-wide {
        grid-column: auto;
    }
    .bc-workflow-row {
        flex-direction: column;
    }
    .bc-note {
        width: 100%;
        margin-left: 0;
    }
    .bc-prompt-row {
        flex-direction: column;
    }
    #prompt-microphone {
        max-width: none;
    }
    .bc-chip {
        grid-template-columns: 1fr;
    }
    .bc-chip-key {
        border-right: 0;
        border-bottom: 1px solid #c7d1cc;
    }
}
"""


def _build_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="teal",
        secondary_hue="green",
        neutral_hue="gray",
        text_size="sm",
        spacing_size="sm",
        radius_size="sm",
    )


def build_demo(mode: GradioMode | str | None = None) -> gr.Blocks:
    resolved_mode = _resolve_gradio_mode(mode)
    start_handler = start_pre_run_runtime if resolved_mode == "runtime" else start_pre_run
    cancel_handler = cancel_pre_run_runtime if resolved_mode == "runtime" else cancel_pre_run
    generate_handler = generate_seed_sweep_runtime if resolved_mode == "runtime" else generate_seed_sweep
    feedback_handler = submit_feedback_runtime if resolved_mode == "runtime" else submit_feedback

    with gr.Blocks(
        title="Bruteforce Canvas",
        fill_width=True,
        analytics_enabled=False,
    ) as demo:
        state = gr.State(initial_state())
        with gr.Column(elem_id="bc-app"):
            gr.HTML('<header id="bc-title"><h1>BRUTEFORCE CANVAS</h1></header>')
            with gr.Accordion("Workflow diagram", open=True, elem_id="workflow-accordion"):
                with gr.Row(elem_classes=["bc-workflow-row"]):
                    with gr.Column(scale=6, min_width=420, elem_classes=["bc-workflow-diagram"]):
                        gr.Markdown(
                            WORKFLOW_MERMAID_MARKDOWN,
                            elem_id="workflow-mermaid",
                            container=False,
                            padding=False,
                        )
                    with gr.Column(scale=4, min_width=320, elem_classes=["bc-workflow-explanation"]):
                        gr.Markdown(
                            WORKFLOW_EXPLANATION_MARKDOWN,
                            elem_id="workflow-explanation",
                            container=False,
                            padding=False,
                        )
            with gr.Row(elem_classes=["bc-prompt-row"]):
                microphone = gr.Microphone(
                    sources=["microphone"],
                    type="numpy",
                    format="wav",
                    show_label=False,
                    label="Record prompt",
                    min_width=180,
                    scale=1,
                    elem_id="prompt-microphone",
                )
                prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Input prompt here.",
                    lines=1,
                    max_lines=5,
                    elem_id="prompt-input",
                    scale=8,
                )
                submit = gr.Button("Submit", variant="primary", scale=1, min_width=120, elem_id="prompt-submit")
            status = gr.HTML(_status_html(initial_state()))

            with gr.Group(visible=False, elem_id="pre-run-panel") as review_panel:
                parsed_report = gr.HTML()
                lock_table = gr.Dataframe(
                    headers=LOCK_TABLE_HEADERS,
                    datatype=["bool", "str", "str", "str", "str", "str"],
                    type="array",
                    interactive=True,
                    label="Enum locks",
                    row_count=8,
                    wrap=True,
                )
                with gr.Row():
                    iqa_cutoff = gr.Slider(0.0, 1.0, value=0.55, step=0.01, label="IQA cutoff")
                    alignment_cutoff = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Alignment cutoff")
                with gr.Row():
                    generate = gr.Button("Generate", variant="primary", interactive=False)
                    cancel = gr.Button("Cancel")

            with gr.Group(visible=False, elem_id="active-panel") as active_panel:
                seed_gallery = gr.Gallery(
                    label="5-seed candidate sweep",
                    columns=5,
                    rows=1,
                    height=320,
                    allow_preview=True,
                    object_fit="cover",
                    elem_id="seed-gallery",
                )

            with gr.Row():
                with gr.Column(scale=3):
                    catalog_gallery = gr.Gallery(
                        label="Curated catalog",
                        columns=4,
                        rows=2,
                        height=430,
                        allow_preview=False,
                        object_fit="cover",
                        elem_id="catalog-gallery",
                    )
                with gr.Column(scale=2, visible=False, elem_id="detail-panel") as detail_panel:
                    detail_image = gr.Image(label="Selected image", elem_id="detail-image", height=430)
                    detail_report = gr.Markdown()
                    with gr.Row():
                        previous_btn = gr.Button("Prev", size="sm")
                        next_btn = gr.Button("Next", size="sm")
                    with gr.Row():
                        up_btn = gr.Button("Thumbs up", size="sm", variant="secondary")
                        down_btn = gr.Button("Thumbs down", size="sm", variant="secondary")
                        trash_btn = gr.Button("Trash", size="sm", variant="stop")

            submit.click(
                start_handler,
                inputs=[prompt, state],
                outputs=[state, review_panel, parsed_report, lock_table, generate, status],
            )
            prompt.submit(
                start_handler,
                inputs=[prompt, state],
                outputs=[state, review_panel, parsed_report, lock_table, generate, status],
            )
            microphone.change(
                transcribe_microphone_to_prompt_steps,
                inputs=[microphone, prompt, state],
                outputs=[prompt, state, status],
                show_progress="minimal",
            )
            cancel.click(cancel_handler, inputs=[state], outputs=[state, review_panel, generate, status])
            generate.click(
                generate_handler,
                inputs=[state, lock_table, iqa_cutoff, alignment_cutoff],
                outputs=[
                    state,
                    review_panel,
                    active_panel,
                    seed_gallery,
                    catalog_gallery,
                    detail_panel,
                    detail_image,
                    detail_report,
                    status,
                ],
                show_progress="minimal",
            )
            catalog_gallery.select(
                select_curated,
                inputs=[state],
                outputs=[state, detail_panel, detail_image, detail_report, status],
            )
            previous_btn.click(
                lambda current_state: move_selection(current_state, -1),
                inputs=[state],
                outputs=[state, detail_panel, detail_image, detail_report, status],
            )
            next_btn.click(
                lambda current_state: move_selection(current_state, 1),
                inputs=[state],
                outputs=[state, detail_panel, detail_image, detail_report, status],
            )
            up_btn.click(
                lambda current_state: feedback_handler(current_state, FeedbackAction.ACCEPT.value),
                inputs=[state],
                outputs=[state, catalog_gallery, detail_panel, detail_image, detail_report, status],
            )
            down_btn.click(
                lambda current_state: feedback_handler(current_state, FeedbackAction.REJECT.value),
                inputs=[state],
                outputs=[state, catalog_gallery, detail_panel, detail_image, detail_report, status],
            )
            trash_btn.click(
                lambda current_state: feedback_handler(current_state, FeedbackAction.SHRED.value),
                inputs=[state],
                outputs=[state, catalog_gallery, detail_panel, detail_image, detail_report, status],
            )
    return demo


def launch(
    *,
    server_name: str | None = None,
    server_port: int | None = None,
    share: bool = False,
    mode: GradioMode | str | None = None,
) -> None:
    demo = build_demo(mode=mode)
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=server_name, server_port=server_port, share=share, css=CSS, theme=_build_theme())


if __name__ == "__main__":
    launch()
