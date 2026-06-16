from __future__ import annotations

import gc
import hashlib
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import escape
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import gradio as gr
from PIL import Image, ImageDraw, ImageEnhance
from pydantic import Field

from bruteforce_canvas.app_config import AppConfig, GeneratorKind, load_app_config
from bruteforce_canvas.app_controller import RunAppController
from bruteforce_canvas.app_factory import (
    build_evaluation_plan,
    build_prompt_pipeline,
    build_run_service,
    build_vlm_adapter,
    prewarm_json_llm,
)
from bruteforce_canvas.asr import default_transcriber
from bruteforce_canvas.evaluation import EvaluationPlan, StaticIQAAdapter, StaticImpactAdapter, StaticVLMAdapter
from bruteforce_canvas.generation import DEFAULT_SEED_BUNDLE, GenerationSettings, seed_sweep_requests
from bruteforce_canvas.loop import LoopAction, LoopDecision
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
    CameraAngle,
    ColorTreatment,
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
from bruteforce_canvas.router import (
    AxisDomain,
    CompatibilityMatrixRule,
    CompatibilityPrior,
    CompatibilitySeverity,
    FieldState,
    LHSRouter,
    RouterInput,
    ThompsonArmState,
)
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
RUNTIME_STREAM_POLL_SECONDS = 0.5
RUNTIME_FINAL_SEED_REFRESH_SECONDS = 0.35
CATALOG_SLOT_COUNT = 8
LOCK_TABLE_HEADERS = [
    "Locked",
    "Field",
    "Raw",
    "Selected Enum",
    "Top LHS Choices",
    "Arm Prior",
    "Pair Prior",
    "LHS policy",
    "Status",
]
GradioMode = Literal["simulation", "runtime"]
WORKFLOW_MERMAID_CODE = """%%{init: {"theme": "base", "flowchart": {"htmlLabels": false}, "themeVariables": {"primaryColor": "#eef8f6", "primaryTextColor": "#17211f", "primaryBorderColor": "#0f766e", "secondaryColor": "#fff8e6", "tertiaryColor": "#ffffff", "lineColor": "#17211f", "fontFamily": "Inter, ui-sans-serif, system-ui, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
  Mic["Microphone audio"] --> ASR["Cohere Transcribe<br/>03-2026<br/>16 kHz en"]
  Typed["Typed prompt"] --> Prompt["Prompt text"]
  ASR --> Prompt
  Prompt --> Mellum["Mellum2 Thinking 12B<br/>via Modal Cloud<br/>structured JSON schemas"]
  Mellum --> Parse["PromptDocument<br/>extract + repair/verify"]
  Parse --> Canon["BGE enum canonicalizer<br/>bge-small-en-v1.5<br/>threshold 0.62"]
  Canon --> Locks["Pre-run locks<br/>thresholds<br/>IQA >= 0.55<br/>Alignment >= 0.25<br/>Human IQA >= 0.70"]
  Locks --> Coord["LHS coordinate<br/>Thompson + GP<br/>Bayesian score"]
  Coord --> Seeds["Bonsai Ternary 4B<br/>5-seed batch<br/>steps=4, 512x512"]
  Seeds --> IQA{"JoyQuality SigLIP2<br/>score >= 0.55?"}
  IQA -- "fail" --> Persist["Persist candidate + failure evidence"]
  IQA -- "pass" --> VLM{"MiniCPM-V-4.6<br/>alignment >= 0.25?"}
  VLM -- "fail" --> Persist
  VLM -- "pass" --> Impact{"TRIBE v2 lite-qv<br/>disabled by default"}
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
  class Mic,ASR,Typed,Prompt,Mellum,Parse,Canon,Locks,Coord,Seeds,Persist,Curated,Feedback,Priors,End step
  class IQA,VLM,Impact,Stop gate"""
WORKFLOW_MERMAID_MARKDOWN = f"```mermaid\n{WORKFLOW_MERMAID_CODE}\n```"
WORKFLOW_DIAGRAM_HTML = """
<div class="bc-flow" role="img" aria-label="Bruteforce Canvas runtime workflow">
  <svg class="bc-flow-svg" viewBox="0 0 1040 640" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <defs>
      <marker id="bc-flow-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" />
      </marker>
      <marker id="bc-flow-arrow-loop" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" />
      </marker>
      <filter id="bc-flow-shadow" x="-10%" y="-12%" width="120%" height="130%">
        <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#17211f" flood-opacity="0.10" />
      </filter>
    </defs>

    <path class="bc-flow-edge" d="M230 91 H295" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M485 91 H550" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M740 91 H805" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M900 138 V204" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M805 251 H740" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M550 251 H485" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M295 251 H230" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M135 314 V350" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M230 412 H295" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M485 412 H550" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M740 411 H805" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M900 458 V524" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M805 571 H740" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M550 571 H485" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge" d="M295 572 H230" marker-end="url(#bc-flow-arrow)" />
    <path class="bc-flow-edge bc-flow-edge-loop" d="M390 510 C390 450 645 430 645 298" marker-end="url(#bc-flow-arrow-loop)" />
    <text class="bc-flow-edge-label" x="442" y="454">continue</text>

    <g class="bc-flow-node" transform="translate(40 44)">
      <title>CohereLabs/cohere-transcribe-03-2026 ASR path</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">ASR / TEXT</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Audio/Text Prompt</tspan>
        <tspan x="18" dy="18">Cohere Transcribe 03-2026</tspan>
        <tspan x="18" dy="17">16 kHz en, max tokens 256</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(295 44)">
      <title>Mellum2 Thinking 12B via Modal Cloud</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">MELLUM2</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Structured JSON</tspan>
        <tspan x="18" dy="18">Mellum2 Thinking 12B</tspan>
        <tspan x="18" dy="17">temp 0.0, tokens 2048</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(550 44)">
      <title>PromptDocument decomposition, repair, and verification</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">DECOMPOSITION</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">PromptDocument</tspan>
        <tspan x="18" dy="18">Objects, relations, lanes</tspan>
        <tspan x="18" dy="17">repair + verify signals</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(805 44)">
      <title>BAAI/bge-small-en-v1.5 enum canonicalizer</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">CANONICALIZATION</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Enum mapping</tspan>
        <tspan x="18" dy="18">BAAI/bge-small-en-v1.5</tspan>
        <tspan x="18" dy="17">threshold 0.62</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(805 204)">
      <title>Pre-run locks and generation thresholds</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">LOCKS</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Pre-run controls</tspan>
        <tspan x="18" dy="18">Locked enum arms stay fixed</tspan>
        <tspan x="18" dy="17">IQA .55, align .25</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(550 204)">
      <title>Latin hypercube coordinate routing with Thompson sampling and GP affinity</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">LATIN HYPERCUBE SAMPLING</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Coordinate router</tspan>
        <tspan x="18" dy="18">Coverage rows + Thompson</tspan>
        <tspan x="18" dy="17">GP compatibility priors</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(295 204)">
      <title>prism-ml/bonsai-image-ternary-4B-gemlite-2bit generation adapter</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">GENERATION</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Bonsai 5-seed batch</tspan>
        <tspan x="18" dy="18">Ternary 4B, steps 4</tspan>
        <tspan x="18" dy="17">512x512 preview sweep</tspan>
      </text>
    </g>

    <g class="bc-flow-gate" transform="translate(40 190)">
      <title>fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn image quality gate</title>
      <polygon points="95,0 190,62 95,124 0,62" />
      <text class="bc-flow-gate-label" x="95" y="43" text-anchor="middle">
        <tspan class="bc-flow-kicker">IQA</tspan>
        <tspan class="bc-flow-title" x="95" dy="22">JoyQuality</tspan>
        <tspan x="95" dy="18">score &gt;= 0.55</tspan>
      </text>
    </g>

    <g class="bc-flow-gate" transform="translate(40 350)">
      <title>openbmb/MiniCPM-V-4.6 visual alignment gate</title>
      <polygon points="95,0 190,62 95,124 0,62" />
      <text class="bc-flow-gate-label" x="95" y="43" text-anchor="middle">
        <tspan class="bc-flow-kicker">VLM</tspan>
        <tspan class="bc-flow-title" x="95" dy="22">MiniCPM-V-4.6</tspan>
        <tspan x="95" dy="18">alignment &gt;= 0.25</tspan>
      </text>
    </g>

    <g class="bc-flow-gate" transform="translate(295 350)">
      <title>Jessylg27/tribev2-lite-qv optional impact adapter</title>
      <polygon points="95,0 190,62 95,124 0,62" />
      <text class="bc-flow-gate-label" x="95" y="43" text-anchor="middle">
        <tspan class="bc-flow-kicker">TRIBE</tspan>
        <tspan class="bc-flow-title" x="95" dy="22">v2 lite-qv</tspan>
        <tspan x="95" dy="18">optional impact</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(550 364)">
      <title>Curated catalog promotion bands</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">CURATED CATALOG</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Promotion bands</tspan>
        <tspan x="18" dy="18">fragile 1, viable 2</tspan>
        <tspan x="18" dy="17">strong 3+ promoted</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(805 364)">
      <title>Accept, reject, and shred feedback actions</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">FEEDBACK</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Accept / reject</tspan>
        <tspan x="18" dy="18">Human signal updates</tspan>
        <tspan x="18" dy="17">backend feedback policy</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(805 524)">
      <title>Enum-arm alpha/beta priors and enum-combination GP affinity</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">PRIORS</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Learned search</tspan>
        <tspan x="18" dy="18">alpha/beta enum arms</tspan>
        <tspan x="18" dy="17">combo GP affinity</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(550 524)">
      <title>Persistence stores candidate, gate, aggregate, feedback, prior, and stop evidence</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">PERSISTENCE</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">JSONL evidence</tspan>
        <tspan x="18" dy="18">records candidates, gates</tspan>
        <tspan x="18" dy="17">feedback, priors, stops</tspan>
      </text>
    </g>

    <g class="bc-flow-gate" transform="translate(295 510)">
      <title>Stop rules: 15-minute Gradio cap, fewer than 10 curated after 10 minutes, or backend/requested stop</title>
      <polygon points="95,0 190,62 95,124 0,62" />
      <text class="bc-flow-gate-label" x="95" y="43" text-anchor="middle">
        <tspan class="bc-flow-kicker">STOP RULES</tspan>
        <tspan class="bc-flow-title" x="95" dy="22">Runtime limits</tspan>
        <tspan x="95" dy="18">cap / stall / stop</tspan>
      </text>
    </g>

    <g class="bc-flow-node" transform="translate(40 524)">
      <title>End run when a stop rule fires</title>
      <rect width="190" height="94" rx="8" />
      <text class="bc-flow-label" x="18" y="25">
        <tspan class="bc-flow-kicker">END</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Stop run</tspan>
        <tspan x="18" dy="18">Final catalog remains</tspan>
        <tspan x="18" dy="17">replayable from JSONL</tspan>
      </text>
    </g>
  </svg>

  <svg class="bc-flow-svg-mobile" viewBox="40 0 300 1710" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <defs>
      <marker id="bc-flow-arrow-mobile" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" />
      </marker>
      <marker id="bc-flow-arrow-loop-mobile" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" />
      </marker>
      <filter id="bc-flow-shadow-mobile" x="-10%" y="-12%" width="120%" height="130%">
        <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#17211f" flood-opacity="0.10" />
      </filter>
    </defs>

    <path class="bc-flow-edge" d="M190 104 V130" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 204 V230" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 304 V330" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 404 V430" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 504 V530" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 604 V630" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 704 V730" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 826 V850" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 946 V970" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 1066 V1090" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 1164 V1190" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 1264 V1290" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 1364 V1390" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 1464 V1490" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge" d="M190 1586 V1610" marker-end="url(#bc-flow-arrow-mobile)" />
    <path class="bc-flow-edge bc-flow-edge-loop" d="M315 1528 C330 1410 330 600 315 568" marker-end="url(#bc-flow-arrow-loop-mobile)" />
    <text class="bc-flow-edge-label" x="278" y="1218">continue</text>

    <g class="bc-flow-node" transform="translate(65 30)">
      <title>CohereLabs/cohere-transcribe-03-2026 ASR path</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">ASR / TEXT</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Audio/Text Prompt</tspan>
        <tspan x="18" dy="18">Cohere Transcribe 03-2026</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 130)">
      <title>Mellum2 Thinking 12B via Modal Cloud</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">MELLUM2</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Structured JSON</tspan>
        <tspan x="18" dy="18">Mellum2 Thinking 12B</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 230)">
      <title>PromptDocument decomposition, repair, and verification</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">DECOMPOSITION</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">PromptDocument</tspan>
        <tspan x="18" dy="18">objects, relations, lanes</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 330)">
      <title>BAAI/bge-small-en-v1.5 enum canonicalizer</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">CANONICALIZATION</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Enum mapping</tspan>
        <tspan x="18" dy="18">BAAI/bge-small-en-v1.5</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 430)">
      <title>Pre-run locks and generation thresholds</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">LOCKS</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Pre-run controls</tspan>
        <tspan x="18" dy="18">IQA .55, alignment .25</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 530)">
      <title>Latin hypercube coordinate routing with Thompson sampling and GP affinity</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">LATIN HYPERCUBE SAMPLING</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Coordinate router</tspan>
        <tspan x="18" dy="18">Thompson + GP priors</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 630)">
      <title>prism-ml/bonsai-image-ternary-4B-gemlite-2bit generation adapter</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">GENERATION</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Bonsai 5-seed batch</tspan>
        <tspan x="18" dy="18">steps 4, 512x512</tspan>
      </text>
    </g>
    <g class="bc-flow-gate" transform="translate(65 730)">
      <title>fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn image quality gate</title>
      <polygon points="125,0 250,48 125,96 0,48" />
      <text class="bc-flow-gate-label" x="125" y="34" text-anchor="middle">
        <tspan class="bc-flow-kicker">IQA</tspan>
        <tspan class="bc-flow-title" x="125" dy="22">JoyQuality</tspan>
        <tspan x="125" dy="18">score &gt;= 0.55</tspan>
      </text>
    </g>
    <g class="bc-flow-gate" transform="translate(65 850)">
      <title>openbmb/MiniCPM-V-4.6 visual alignment gate</title>
      <polygon points="125,0 250,48 125,96 0,48" />
      <text class="bc-flow-gate-label" x="125" y="34" text-anchor="middle">
        <tspan class="bc-flow-kicker">VLM</tspan>
        <tspan class="bc-flow-title" x="125" dy="22">MiniCPM-V-4.6</tspan>
        <tspan x="125" dy="18">alignment &gt;= 0.25</tspan>
      </text>
    </g>
    <g class="bc-flow-gate" transform="translate(65 970)">
      <title>Jessylg27/tribev2-lite-qv optional impact adapter</title>
      <polygon points="125,0 250,48 125,96 0,48" />
      <text class="bc-flow-gate-label" x="125" y="34" text-anchor="middle">
        <tspan class="bc-flow-kicker">TRIBE</tspan>
        <tspan class="bc-flow-title" x="125" dy="22">v2 lite-qv</tspan>
        <tspan x="125" dy="18">optional impact</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 1090)">
      <title>Curated catalog promotion bands</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">CURATED CATALOG</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Promotion bands</tspan>
        <tspan x="18" dy="18">fragile 1, viable 2, strong 3+</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 1190)">
      <title>Accept, reject, and shred feedback actions</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">FEEDBACK</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Accept / reject / shred</tspan>
        <tspan x="18" dy="18">human signal updates priors</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 1290)">
      <title>Enum-arm alpha/beta priors and enum-combination GP affinity</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">PRIORS</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Learned search</tspan>
        <tspan x="18" dy="18">alpha/beta + combo GP</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 1390)">
      <title>Persistence stores candidate, gate, aggregate, feedback, prior, and stop evidence</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">PERSISTENCE</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">JSONL evidence</tspan>
        <tspan x="18" dy="18">candidates, gates, feedback</tspan>
      </text>
    </g>
    <g class="bc-flow-gate" transform="translate(65 1490)">
      <title>Stop rules: 15-minute Gradio cap, fewer than 10 curated after 10 minutes, or backend/requested stop</title>
      <polygon points="125,0 250,48 125,96 0,48" />
      <text class="bc-flow-gate-label" x="125" y="34" text-anchor="middle">
        <tspan class="bc-flow-kicker">STOP RULES</tspan>
        <tspan class="bc-flow-title" x="125" dy="22">Runtime limits</tspan>
        <tspan x="125" dy="18">cap / stall / stop</tspan>
      </text>
    </g>
    <g class="bc-flow-node" transform="translate(65 1610)">
      <title>End run when a stop rule fires</title>
      <rect width="250" height="74" rx="8" />
      <text class="bc-flow-label" x="18" y="24">
        <tspan class="bc-flow-kicker">END</tspan>
        <tspan class="bc-flow-title" x="18" dy="22">Stop run</tspan>
        <tspan x="18" dy="18">replayable from JSONL</tspan>
      </text>
    </g>
  </svg>
</div>
"""
WORKFLOW_EXPLANATION_MARKDOWN = """### Workflow Steps

1. **ASR / text input** accepts typed prompts or microphone audio. The local ASR path uses `CohereLabs/cohere-transcribe-03-2026`, normalizes to `16 kHz`, defaults to English punctuation, and decodes with `max_new_tokens=256`.
2. **Decomposition** extracts objects, relations, constraints, and cinematography lanes from typed text or ASR output.
3. **Mellum2 Thinking 12B via Modal Cloud** supplies the structured JSON reasoning path for prompt extraction, repair, and verification through an OpenAI-compatible endpoint using `temperature=0.0`, strict schema output, and `max_completion_tokens=2048`.
4. **Repair/Verify** checks blocking issues, unresolved targets, and threshold readiness before generation is allowed.
5. **Canonicalization** maps raw prompt values to project enums with `BAAI/bge-small-en-v1.5`, `match_threshold=0.62`, and LLM fallback for uncertain matches.
6. **Latin Hypercube Sampling (LHS)** proposes coverage-oriented coordinate rows across unlocked enum arms while preserving locked prompt evidence.
7. **Thompson Sampling/GP** ranks sampled arms with Bayesian feedback state, GP-style coordinate scoring, and compatibility priors as evaluation evidence accumulates.
8. **Quick generation** uses `prism-ml/bonsai-image-ternary-4B-gemlite-2bit` through Bonsai local or HTTP adapters, with the fast preview path defaulting to `steps=4`, `512x512`, and seeds `[7, 42, 156, 8888, 42069]`.
9. **IQA** filters each 5-seed batch with `fancyfeast/joyquality-siglip2-so400m-512-16-05k047vn` against the configured quality cutoff.
10. **VLM alignment** scores surviving images with `openbmb/MiniCPM-V-4.6`, `max_new_tokens=128`, and the compiled prompt/target manifest.
11. **TRIBE v2** uses `Jessylg27/tribev2-lite-qv` as an optional metacognitive impact adapter; it is disabled by default and kept downstream of IQA and VLM gates.
12. **Prior updates** write evaluation and feedback evidence back into enum-arm alpha/beta priors and enum-combination GP affinity for the next coordinate.
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


def _clear_worker_runtime_adapters(worker: Any) -> None:
    if worker is None:
        return
    for attr in ("generator", "iqa", "vlm", "impact"):
        if not hasattr(worker, attr):
            continue
        try:
            setattr(worker, attr, None)
        except Exception:
            pass


def _release_runtime_session(session: RuntimeUISession, *, reason: str) -> None:
    try:
        if getattr(session.service, "state", None) != RunRuntimeState.STOPPED:
            session.service.request_stop()
            session.service.stop_with_reason(reason, details={"run_id": session.config.run.run_id})
    except Exception as error:
        print(f"Runtime session stop failed during {reason}: {error}", flush=True)
    _clear_worker_runtime_adapters(getattr(session.service, "worker", None))


def _release_inactive_runtime_sessions(*, keep_run_id: str | None = None, reason: str) -> None:
    stale_run_ids = [run_id for run_id in _RUNTIME_SESSIONS if run_id != keep_run_id]
    if not stale_run_ids:
        return
    for run_id in stale_run_ids:
        session = _RUNTIME_SESSIONS.pop(run_id, None)
        if session is not None:
            _release_runtime_session(session, reason=reason)
    gc.collect()
    try:
        import torch

        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class ASRRuntimeStatus:
    state: Literal["idle", "loading", "ready", "failed"] = "idle"
    message: str = "Cohere Transcribe ASR has not loaded yet."
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


_ASR_RUNTIME_STATUS = ASRRuntimeStatus()
_ASR_RUNTIME_STATUS_LOCK = RLock()


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
    catalog_page_index: int = 0
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


RUNTIME_ENUM_DOMAINS: dict[str, list[str]] = {
    "relation.": [member.name for member in RelationType],
    "cinematography.shot_size": [member.name for member in ShotSize],
    "cinematography.camera_angle": [member.name for member in CameraAngle],
    "cinematography.lens": [member.name for member in OpticCharacter],
    "cinematography.lighting_mood": [member.name for member in LightingMood],
    "cinematography.color_treatment": [member.name for member in ColorTreatment],
    "cinematography.composition": [member.name for member in Framing],
}

RUNTIME_ARM_PRIOR_OVERRIDES: dict[tuple[str, str], tuple[float, float]] = {
    ("relation.", "ON_TOP_OF"): (7.0, 1.4),
    ("relation.", "INSIDE"): (5.0, 1.8),
    ("relation.", "NEXT_TO"): (4.5, 2.0),
    ("cinematography.shot_size", "MEDIUM_SHOT"): (6.0, 1.6),
    ("cinematography.shot_size", "WIDE_SHOT"): (4.2, 2.0),
    ("cinematography.shot_size", "CLOSE_UP"): (3.8, 2.2),
    ("cinematography.shot_size", "MEDIUM_CLOSE_UP"): (3.4, 2.3),
    ("cinematography.camera_angle", "EYE_LEVEL"): (6.0, 1.5),
    ("cinematography.camera_angle", "THREE_QUARTER"): (4.4, 1.9),
    ("cinematography.camera_angle", "HIGH_ANGLE"): (3.2, 2.2),
    ("cinematography.lens", "NATURAL_35MM"): (5.5, 1.6),
    ("cinematography.lens", "PORTRAIT_50MM"): (4.4, 1.9),
    ("cinematography.lens", "MACRO"): (3.5, 2.0),
    ("cinematography.lighting_mood", "STUDIO_SOFTBOX"): (5.2, 1.7),
    ("cinematography.lighting_mood", "SOFT_NATURAL"): (4.8, 1.8),
    ("cinematography.lighting_mood", "GOLDEN_HOUR"): (3.7, 2.0),
    ("cinematography.color_treatment", "NATURAL_COLOR"): (5.8, 1.5),
    ("cinematography.color_treatment", "RICH_SATURATION"): (3.6, 2.2),
    ("cinematography.color_treatment", "FILMIC_CONTRAST"): (3.5, 2.1),
    ("cinematography.composition", "CENTERED"): (5.4, 1.6),
    ("cinematography.composition", "RULE_OF_THIRDS"): (4.6, 1.8),
    ("cinematography.composition", "SYMMETRICAL"): (3.4, 2.1),
}


def _enum_domain_key(field_path: str) -> str | None:
    if field_path.startswith("relation."):
        return "relation."
    return field_path if field_path in RUNTIME_ENUM_DOMAINS else None


def _enum_domain_values(field_path: str) -> list[str]:
    domain_key = _enum_domain_key(field_path)
    return RUNTIME_ENUM_DOMAINS.get(domain_key or "", [])


def _runtime_arm_prior(axis: str, value: str, *, index: int = 0) -> tuple[float, float]:
    domain_key = _enum_domain_key(axis) or axis
    override = RUNTIME_ARM_PRIOR_OVERRIDES.get((domain_key, value)) or RUNTIME_ARM_PRIOR_OVERRIDES.get((axis, value))
    if override is not None:
        return override
    alpha = max(1.2, 2.8 - min(index, 8) * 0.12)
    beta = 2.4 + min(index, 8) * 0.10
    return round(alpha, 2), round(beta, 2)


def _runtime_arm_score(axis: str, value: str, *, index: int = 0) -> float:
    alpha, beta = _runtime_arm_prior(axis, value, index=index)
    return alpha / (alpha + beta)


def _runtime_top_lhs_choices(field_path: str, *, limit: int = 3) -> list[tuple[str, float, float, float]]:
    choices: list[tuple[str, float, float, float]] = []
    for index, value in enumerate(_enum_domain_values(field_path)):
        alpha, beta = _runtime_arm_prior(field_path, value, index=index)
        score = alpha / (alpha + beta)
        choices.append((value, alpha, beta, score))
    return sorted(choices, key=lambda item: item[3], reverse=True)[:limit]


def _format_lhs_choices(field_path: str) -> str:
    choices = _runtime_top_lhs_choices(field_path)
    if not choices:
        return ""
    return "; ".join(f"{value} {score:.0%}" for value, _alpha, _beta, score in choices)


def _format_arm_prior(field_path: str) -> str:
    choices = _runtime_top_lhs_choices(field_path, limit=1)
    if not choices:
        return ""
    value, alpha, beta, score = choices[0]
    return f"{value}: alpha {alpha:g}, beta {beta:g}, mean {score:.0%}"


RUNTIME_PAIR_PRIOR_RULES: list[CompatibilityMatrixRule] = [
    CompatibilityMatrixRule(
        left_field="cinematography.shot_size",
        left_value="CLOSE_UP",
        right_field="cinematography.lens",
        right_value="MACRO",
        severity=CompatibilitySeverity.BOOST,
        weight=0.18,
        reason="close-up framing pairs well with macro detail.",
    ),
    CompatibilityMatrixRule(
        left_field="cinematography.shot_size",
        left_value="WIDE_SHOT",
        right_field="cinematography.lens",
        right_value="WIDE_ANGLE",
        severity=CompatibilitySeverity.BOOST,
        weight=0.14,
        reason="wide shot and wide-angle lens preserve full scene context.",
    ),
    CompatibilityMatrixRule(
        left_field="cinematography.lighting_mood",
        left_value="STUDIO_SOFTBOX",
        right_field="cinematography.color_treatment",
        right_value="NATURAL_COLOR",
        severity=CompatibilitySeverity.BOOST,
        weight=0.12,
        reason="studio softbox and natural color preserve product readability.",
    ),
    CompatibilityMatrixRule(
        left_field="cinematography.camera_angle",
        left_value="LOW_ANGLE",
        right_field="cinematography.shot_size",
        right_value="EXTREME_CLOSE_UP",
        severity=CompatibilitySeverity.SOFT_DOWNRANK,
        weight=-0.22,
        reason="low angle with extreme close-up can hide object relationships.",
    ),
]

OBJECT_LABEL_COLOR_TERMS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "gold",
    "golden",
    "gray",
    "green",
    "grey",
    "magenta",
    "maroon",
    "navy",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "teal",
    "turquoise",
    "violet",
    "white",
    "yellow",
}
OBJECT_LABEL_COLOR_MODIFIERS = {"bright", "dark", "deep", "light", "muted", "pale"}


def _runtime_compatibility_prior() -> CompatibilityPrior:
    default_prior = CompatibilityPrior()
    return CompatibilityPrior(pair_rules=[*(default_prior.pair_rules or []), *RUNTIME_PAIR_PRIOR_RULES])


def _format_pair_prior(field_path: str) -> str:
    rules = [rule for rule in RUNTIME_PAIR_PRIOR_RULES if rule.left_field == field_path or rule.right_field == field_path]
    if not rules:
        return ""
    labels = []
    for rule in rules[:2]:
        other_field = rule.right_field if rule.left_field == field_path else rule.left_field
        other_value = rule.right_value if rule.left_field == field_path else rule.left_value
        labels.append(f"{rule.severity}: {other_field}={other_value}")
    return "; ".join(labels)


def _selected_enum_display(entry: dict[str, object]) -> str:
    enum_value = str(entry.get("enum_value") or "").strip()
    if enum_value:
        status = str(entry.get("canonical_status", ""))
        if status == "matched_active":
            return f"[selected] {enum_value}"
        return enum_value
    field_path = str(entry.get("field_path", ""))
    choices = _runtime_top_lhs_choices(field_path, limit=1)
    if choices:
        return f"[suggested] {choices[0][0]}"
    return ""


def _lock_table_from_review(review: PreRunModalReadModel) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for entry in review.lock_entries:
        field_path = str(entry.get("field_path", ""))
        rows.append(
            [
                str(entry.get("lock_state")) == "locked",
                field_path,
                str(entry.get("raw_value") or ""),
                _selected_enum_display(entry),
                _format_lhs_choices(field_path),
                _format_arm_prior(field_path),
                _format_pair_prior(field_path),
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


def _clean_selected_enum_cell(value: object) -> str:
    text = str(value or "").strip()
    for prefix in ("[selected]", "[suggested]"):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


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


def _short_backend_error(error: BaseException | str, *, limit: int = 220) -> str:
    text = " ".join(str(error).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _runtime_parse_failed_document(raw_prompt: str, error: BaseException) -> PromptDocumentSpec:
    document = build_prompt_document_for_demo(raw_prompt)
    issue = VerificationIssue(
        issue_type="prompt_parse_failed",
        repair_scope="prompt_improvement",
        blocking=True,
        message=f"Prompt parser returned an invalid document shape: {_short_backend_error(error)}",
    )
    return document.model_copy(
        update={
            "raw_user_prompt": raw_prompt,
            "verification": VerificationReport(approved=False, issues=[issue]),
        }
    )


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


def _display_object_label_and_inferred_color(label: str) -> tuple[str, str | None]:
    words = label.strip().split()
    if len(words) < 2:
        return label, None
    normalized = [word.strip(".,;:()[]{}").lower() for word in words]
    if normalized[0] in OBJECT_LABEL_COLOR_TERMS:
        return " ".join(words[1:]), words[0].strip(".,;:()[]{}")
    if (
        len(words) >= 3
        and normalized[0] in OBJECT_LABEL_COLOR_MODIFIERS
        and normalized[1] in OBJECT_LABEL_COLOR_TERMS
    ):
        return " ".join(words[2:]), f"{words[0].strip('.,;:()[]{}')} {words[1].strip('.,;:()[]{}')}"
    return label, None


def _review_markdown(raw_prompt: str, document: PromptDocumentSpec, review: PreRunModalReadModel, rendered: str) -> str:
    descriptor_by_target = {
        descriptor.target_id: descriptor.model_dump(exclude_none=True, mode="json")
        for descriptor in document.object_lane.objects
    }

    element_by_id = {element.id: element for element in document.graph.elements}

    def descriptor_markup_for(element_id: str, *, inferred_color: str | None = None) -> str:
        descriptor = descriptor_by_target.get(element_id, {})
        descriptor_items = [
            f"{key.replace('_', ' ')}: {value}"
            for key, value in descriptor.items()
            if key != "target_id" and value is not None and value != "" and value != []
        ]
        if inferred_color and not str(descriptor.get("color") or "").strip():
            descriptor_items.insert(0, f"color: {inferred_color}")
        descriptor_markup = "".join(f'<span class="bc-token">{escape(str(item))}</span>' for item in descriptor_items)
        if not descriptor_markup:
            descriptor_markup = '<span class="bc-token bc-token-muted">no extra descriptors</span>'
        return descriptor_markup

    def element_box(element_id: str) -> str:
        element = element_by_id.get(element_id)
        if element is None:
            return (
                '<div class="bc-triplet-box bc-triplet-object">'
                '<span class="bc-triplet-kicker">Object</span>'
                f"<strong>{escape(element_id)}</strong>"
                '<small>unresolved reference</small>'
                "</div>"
            )
        display_label, inferred_color = _display_object_label_and_inferred_color(str(element.label))
        return (
            '<div class="bc-triplet-box bc-triplet-object">'
            '<span class="bc-triplet-kicker">Object</span>'
            f"<strong>{escape(display_label)}</strong>"
            f"<small>{escape(element.id)} | {escape(str(element.role))} | {escape(str(element.importance))}</small>"
            f'<div class="bc-object-meta">{descriptor_markup_for(element.id, inferred_color=inferred_color)}</div>'
            "</div>"
        )

    triplet_rows = []
    related_ids: set[str] = set()
    for relation in document.graph.relations:
        enum_value = relation.relation_match.enum_value if relation.relation_match else "unmatched"
        relation_label = str(relation.relation_raw or enum_value or "relation")
        related_ids.update({relation.source_id, relation.target_id})
        triplet_rows.append(
            '<article class="bc-triplet-row">'
            f"{element_box(relation.source_id)}"
            '<div class="bc-triplet-box bc-triplet-relation">'
            '<span class="bc-triplet-kicker">Relation</span>'
            f"<strong>{escape(relation_label)}</strong>"
            f"<small>canonical: {escape(str(enum_value))}</small>"
            "</div>"
            f"{element_box(relation.target_id)}"
            "</article>"
        )
    triplet_markup = "".join(triplet_rows)
    if not triplet_markup:
        triplet_markup = '<p class="bc-muted">No explicit object relation object triplets.</p>'

    standalone_objects = [
        '<article class="bc-standalone-object">'
        f"<strong>{escape(_display_object_label_and_inferred_color(str(element.label))[0])}</strong>"
        f"<small>{escape(element.id)} | {escape(str(element.role))}</small>"
        f'<div class="bc-object-meta">{descriptor_markup_for(element.id, inferred_color=_display_object_label_and_inferred_color(str(element.label))[1])}</div>'
        "</article>"
        for element in document.graph.elements
        if element.id not in related_ids
    ]
    standalone_markup = ""
    if standalone_objects:
        standalone_markup = (
            '<div class="bc-standalone-objects">'
            '<div class="bc-eyebrow">Standalone objects</div>'
            f"{''.join(standalone_objects)}"
            "</div>"
        )

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
        '<h3>Scene graph</h3>'
        f'<div class="bc-triplet-stack">{triplet_markup}</div>'
        f"{standalone_markup}"
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


def _runtime_batch_prompt_markup(session: RuntimeUISession, item: SeedSweepWorkItem, *, batch_index: int) -> str:
    sampled = "".join(
        f'<span class="bc-token">{escape(field_path)}={escape(value)}</span>'
        for field_path, value in sorted(item.sampled_arms.items())
    )
    if not sampled:
        sampled = '<span class="bc-token bc-token-muted">fixed coordinate</span>'
    return (
        '<article class="bc-batch-prompt">'
        '<div class="bc-eyebrow">LHS candidate prompt</div>'
        f"<h3>Batch {batch_index} · {escape(item.coordinate_id)}</h3>"
        f'<pre class="bc-compiled-prompt">{escape(item.rendered_prompt)}</pre>'
        f'<div class="bc-token-stack">{sampled}</div>'
        "</article>"
    )


def _runtime_review_markup(
    session: RuntimeUISession,
    state: GradioSimulationState,
    *,
    item: SeedSweepWorkItem | None = None,
    batch_index: int | None = None,
) -> str:
    review = state.review or pre_run_modal_from_prompt(session.document)
    rendered_prompt = item.rendered_prompt if item is not None else state.rendered_prompt or session.rendered_prompt.rendered_prompt
    base_markup = _review_markdown(session.config.run.raw_user_prompt, session.document, review, rendered_prompt)
    if item is None or batch_index is None:
        return base_markup
    return _runtime_batch_prompt_markup(session, item, batch_index=batch_index) + base_markup


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


def _seed_slot_dir(run_id: str, batch_index: int) -> Path:
    path = RUNTIME_RUN_ROOT / run_id / "images" / "_seed_slots" / f"batch_{batch_index:03d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slot_base_image() -> Image.Image:
    return Image.new("RGB", (768, 1024), (22, 28, 32))


def _draw_slot_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    seed: int,
    status: str,
    border: tuple[int, int, int],
    label_fill: tuple[int, int, int],
) -> None:
    draw.rectangle((0, 0, 767, 1023), outline=border, width=16)
    draw.rounded_rectangle((42, 42, 246, 104), radius=10, fill=label_fill)
    draw.text((66, 62), f"seed {seed}", fill=(255, 255, 255))
    draw.text((54, 140), status.upper(), fill=border)


def _write_seed_slot_placeholder(*, run_id: str, batch_index: int, seed: int) -> str:
    path = _seed_slot_dir(run_id, batch_index) / f"seed_{seed}_pending.png"
    if path.exists():
        return str(path)
    image = _slot_base_image()
    draw = ImageDraw.Draw(image)
    for offset in range(-1024, 1024, 52):
        draw.line((offset, 1024, offset + 1024, 0), fill=(37, 45, 50), width=7)
    draw.rounded_rectangle((64, 330, 704, 602), radius=22, fill=(31, 39, 44), outline=(78, 96, 99), width=4)
    draw.ellipse((318, 414, 450, 546), outline=(126, 146, 149), width=8)
    _draw_slot_chrome(
        draw,
        seed=seed,
        status="waiting for image",
        border=(126, 146, 149),
        label_fill=(69, 84, 88),
    )
    image.save(path)
    return str(path)


def _write_failed_seed_slot(*, run_id: str, batch_index: int, candidate: SimCandidate) -> str:
    path = _seed_slot_dir(run_id, batch_index) / f"{candidate.candidate_id}_failed.png"
    if path.exists():
        return str(path)
    try:
        image = Image.open(candidate.display_path).convert("RGB").resize((768, 1024))
    except Exception:
        image = _slot_base_image()
    image = ImageEnhance.Color(image).enhance(0.10)
    image = ImageEnhance.Brightness(image).enhance(0.28)
    overlay = Image.new("RGBA", image.size, (5, 8, 10, 118))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.line((96, 196, 672, 772), fill=(216, 38, 38), width=30)
    draw.line((672, 196, 96, 772), fill=(216, 38, 38), width=30)
    draw.rounded_rectangle((168, 802, 600, 900), radius=14, fill=(28, 8, 8), outline=(216, 38, 38), width=6)
    draw.text((262, 834), "PURGED", fill=(255, 236, 236))
    _draw_slot_chrome(
        draw,
        seed=candidate.seed,
        status="failed gate",
        border=(216, 38, 38),
        label_fill=(150, 28, 28),
    )
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


def _preview_gallery(
    candidates: list[SimCandidate],
    *,
    expected_seeds: list[int] | None = None,
    run_id: str = RUN_ID,
    batch_index: int = 0,
) -> list[tuple[str, str]]:
    gallery = []
    by_seed = {candidate.seed: candidate for candidate in candidates}
    ordered_seeds = expected_seeds or [candidate.seed for candidate in candidates]
    for seed in ordered_seeds:
        candidate = by_seed.get(seed)
        if candidate is None:
            gallery.append(
                (
                    _write_seed_slot_placeholder(run_id=run_id, batch_index=batch_index, seed=seed),
                    f"seed {seed} | waiting for image",
                )
            )
            continue
        if candidate.outcome == "pending":
            caption = f"seed {candidate.seed} | generated, evaluating"
            image_path = candidate.display_path
        elif candidate.outcome == "failed":
            reasons = ", ".join(candidate.failure_reasons) or "failed"
            caption = f"seed {candidate.seed} | failed | {reasons}"
            image_path = _write_failed_seed_slot(run_id=run_id, batch_index=batch_index, candidate=candidate)
        else:
            caption = f"seed {candidate.seed} | {candidate.outcome} | Q {candidate.quality_score:.2f} A {candidate.alignment_score:.2f}"
            image_path = candidate.display_path
        gallery.append((image_path, caption))
    return gallery


def _preview_slot_paths(state: GradioSimulationState) -> list[str | None]:
    if not state.current_batch and not state.current_batch_expected_seeds:
        return [None for _seed in DEFAULT_SEED_BUNDLE]
    gallery = _preview_gallery(
        state.current_batch,
        expected_seeds=state.current_batch_expected_seeds,
        run_id=state.run_id,
        batch_index=state.batch_index,
    )
    paths = [path for path, _caption in gallery[: len(DEFAULT_SEED_BUNDLE)]]
    if len(paths) < len(DEFAULT_SEED_BUNDLE):
        paths.extend(None for _seed in DEFAULT_SEED_BUNDLE[len(paths) :])
    return paths


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


def _catalog_gallery_update(state: GradioSimulationState, *, clear: bool = False):
    return gr.update(value=[] if clear else _catalog_gallery(state))


def _catalog_page_count(state: GradioSimulationState) -> int:
    visible_count = len(_visible_catalog(state))
    return max(1, (visible_count + CATALOG_SLOT_COUNT - 1) // CATALOG_SLOT_COUNT)


def _clamp_catalog_page_index(state: GradioSimulationState) -> int:
    return min(max(int(state.catalog_page_index), 0), _catalog_page_count(state) - 1)


def _catalog_page_window(state: GradioSimulationState) -> tuple[int, int]:
    page_index = _clamp_catalog_page_index(state)
    start = page_index * CATALOG_SLOT_COUNT
    end = start + CATALOG_SLOT_COUNT
    return start, end


def _catalog_slot_paths(state: GradioSimulationState, *, clear: bool = False) -> list[str | None]:
    if clear:
        return [None for _index in range(CATALOG_SLOT_COUNT)]
    start, end = _catalog_page_window(state)
    paths = [candidate.display_path for candidate in _visible_catalog(state)[start:end]]
    if len(paths) < CATALOG_SLOT_COUNT:
        paths.extend(None for _index in range(CATALOG_SLOT_COUNT - len(paths)))
    return paths


def _catalog_page_label(state: GradioSimulationState) -> str:
    visible_count = len(_visible_catalog(state))
    if visible_count == 0:
        return "Curated catalog: 0 images"
    page_index = _clamp_catalog_page_index(state)
    start = page_index * CATALOG_SLOT_COUNT + 1
    end = min((page_index + 1) * CATALOG_SLOT_COUNT, visible_count)
    return f"Curated catalog: {start}-{end} of {visible_count}"


def _catalog_page_controls(state: GradioSimulationState) -> tuple[Any, Any, str]:
    page_index = _clamp_catalog_page_index(state)
    page_count = _catalog_page_count(state)
    return (
        gr.update(interactive=page_index > 0),
        gr.update(interactive=page_index < page_count - 1),
        _catalog_page_label(state),
    )


def _clamp_catalog_page_state(state: GradioSimulationState) -> GradioSimulationState:
    clamped = _clamp_catalog_page_index(state)
    if clamped == state.catalog_page_index:
        return state
    return state.model_copy(update={"catalog_page_index": clamped})


def _detail_image_update(candidate: SimCandidate | None, *, clear: bool = False):
    return gr.update(value=None if clear or candidate is None else candidate.display_path)


def _catalog_gallery_signature(state: GradioSimulationState) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (candidate.candidate_id, candidate.display_path, candidate.feedback_state or "")
        for candidate in _visible_catalog(state)
    )


def _runtime_catalog_signature_from_records(records: list[PersistenceRecord]) -> tuple[tuple[str, str, str], ...]:
    candidates = _runtime_candidates_from_records(records)
    return tuple(
        (candidate.candidate_id, candidate.display_path, candidate.feedback_state or "")
        for candidate in candidates
        if candidate.promoted and candidate.feedback_state not in {"reject", "shred"}
    )


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


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _short_asr_error(error: BaseException | str, *, limit: int = 260) -> str:
    text = " ".join(str(error).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _set_asr_runtime_status(
    state: Literal["idle", "loading", "ready", "failed"],
    message: str,
    *,
    error: BaseException | str | None = None,
) -> ASRRuntimeStatus:
    now = time.monotonic()
    with _ASR_RUNTIME_STATUS_LOCK:
        _ASR_RUNTIME_STATUS.state = state
        _ASR_RUNTIME_STATUS.message = message
        _ASR_RUNTIME_STATUS.error = _short_asr_error(error) if error is not None else None
        if state == "loading":
            _ASR_RUNTIME_STATUS.started_at = now
            _ASR_RUNTIME_STATUS.completed_at = None
        elif state in {"ready", "failed", "idle"}:
            _ASR_RUNTIME_STATUS.completed_at = now
        return ASRRuntimeStatus(
            state=_ASR_RUNTIME_STATUS.state,
            message=_ASR_RUNTIME_STATUS.message,
            error=_ASR_RUNTIME_STATUS.error,
            started_at=_ASR_RUNTIME_STATUS.started_at,
            completed_at=_ASR_RUNTIME_STATUS.completed_at,
        )


def _asr_runtime_status_snapshot() -> ASRRuntimeStatus:
    with _ASR_RUNTIME_STATUS_LOCK:
        return ASRRuntimeStatus(
            state=_ASR_RUNTIME_STATUS.state,
            message=_ASR_RUNTIME_STATUS.message,
            error=_ASR_RUNTIME_STATUS.error,
            started_at=_ASR_RUNTIME_STATUS.started_at,
            completed_at=_ASR_RUNTIME_STATUS.completed_at,
        )


def _asr_transcription_block_message() -> str | None:
    status = _asr_runtime_status_snapshot()
    if status.state == "loading":
        return "Cohere Transcribe ASR is still warming up. Your recording is kept; try again when ASR is ready."
    if status.state == "failed":
        detail = f" Reason: {status.error or status.message}" if (status.error or status.message) else ""
        return (
            "Cohere Transcribe ASR is unavailable because preload failed."
            f"{detail} You can type the prompt, or restart after freeing GPU memory."
        )
    if (
        status.state == "idle"
        and "released" in status.message.lower()
        and _env_flag("BC_ASR_PREWARM", default=True)
    ):
        return (
            "Cohere Transcribe ASR was released to free GPU memory for generation. "
            "Use the text prompt, or restart the runtime UI before recording again."
        )
    return None


def _asr_release_after_transcribe_enabled() -> bool:
    if "BC_ASR_RELEASE_AFTER_TRANSCRIBE" in os.environ:
        return _env_flag("BC_ASR_RELEASE_AFTER_TRANSCRIBE", default=False)
    if "BC_ASR_KEEP_LOADED_AFTER_TRANSCRIBE" in os.environ:
        return not _env_flag("BC_ASR_KEEP_LOADED_AFTER_TRANSCRIBE", default=True)
    return False


def _asr_release_before_runtime_service_enabled() -> bool:
    return _env_flag("BC_ASR_RELEASE_BEFORE_RUNTIME_SERVICE", default=False) or _env_flag(
        "BC_ASR_RELEASE_BEFORE_GENERATION", default=False
    )


def _unload_runtime_asr(reason: str) -> None:
    status = _asr_runtime_status_snapshot()
    if status.state not in {"ready", "loading"}:
        return
    try:
        unload = getattr(default_transcriber(), "unload", None)
        if unload is not None:
            unload()
    except Exception as error:
        print(f"Cohere Transcribe ASR unload failed after {reason}: {error}", flush=True)
        return
    _set_asr_runtime_status("idle", f"Cohere Transcribe ASR released after {reason}.")
    print(f"Cohere Transcribe ASR released after {reason}.", flush=True)


def _prewarm_runtime_llm_if_enabled(mode: GradioMode | str) -> None:
    if _resolve_gradio_mode(mode) != "runtime":
        return
    if not _env_flag("BC_LLM_PREWARM", default=True):
        return

    config = load_app_config()
    started = time.monotonic()
    print(
        f"Prewarming prompt LLM before runtime UI launch: model={config.llm.model} base_url={config.llm.base_url}",
        flush=True,
    )
    try:
        prewarm_json_llm(config)
    except Exception as error:
        if _env_flag("BC_LLM_PREWARM_REQUIRED", default=False):
            raise
        print(f"Prompt LLM prewarm failed; continuing runtime UI launch: {error}", flush=True)
        return
    elapsed = time.monotonic() - started
    print(f"Prompt LLM prewarm complete in {elapsed:.1f}s.", flush=True)


def _prewarm_runtime_asr_if_enabled(mode: GradioMode | str) -> None:
    if _resolve_gradio_mode(mode) != "runtime":
        return
    if not _env_flag("BC_ASR_PREWARM", default=True):
        _set_asr_runtime_status("idle", "Cohere Transcribe ASR prewarm is disabled.")
        return

    transcriber = default_transcriber()
    started = time.monotonic()
    _set_asr_runtime_status("loading", "Cohere Transcribe ASR is warming up.")
    print(
        f"Prewarming Cohere Transcribe ASR before runtime UI launch: model={transcriber.config.model_id}",
        flush=True,
    )
    try:
        transcriber.prewarm(run_dummy_inference=_env_flag("BC_ASR_PREWARM_INFERENCE", default=True))
    except Exception as error:
        unload = getattr(transcriber, "unload", None)
        if unload is not None:
            unload()
        _set_asr_runtime_status(
            "failed",
            "Cohere Transcribe ASR preload failed.",
            error=error,
        )
        if _env_flag("BC_ASR_PREWARM_REQUIRED", default=False):
            raise
        print(f"Cohere Transcribe ASR prewarm failed; continuing runtime UI launch: {error}", flush=True)
        return
    elapsed = time.monotonic() - started
    _set_asr_runtime_status("ready", f"Cohere Transcribe ASR ready in {elapsed:.1f}s.")
    print(f"Cohere Transcribe ASR prewarm complete in {elapsed:.1f}s.", flush=True)


def _prewarm_runtime_startup_if_enabled(mode: GradioMode | str) -> None:
    if _resolve_gradio_mode(mode) != "runtime":
        return
    prewarmers = [_prewarm_runtime_llm_if_enabled, _prewarm_runtime_asr_if_enabled]
    if not _env_flag("BC_RUNTIME_PREWARM_PARALLEL", default=True):
        for prewarm in prewarmers:
            prewarm(mode)
        return
    with ThreadPoolExecutor(max_workers=len(prewarmers)) as executor:
        futures = [executor.submit(prewarm, mode) for prewarm in prewarmers]
        for future in futures:
            future.result()


def _new_runtime_run_id() -> str:
    return f"run_{int(time.time() * 1000)}"


def _runtime_output_dir(run_id: str) -> Path:
    return RUNTIME_RUN_ROOT / run_id / "images"


def _runtime_fast_parse_enabled() -> bool:
    return _env_flag("BC_RUNTIME_FAST_PARSE", default=True)


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
    updates: dict[str, Any] = {"event_store_path": run_root / "events.jsonl", "run": run}
    if _runtime_fast_parse_enabled() and not _env_flag("BC_RUNTIME_LLM_CANONICALIZER_FALLBACK", default=False):
        updates["canonicalizer"] = base.canonicalizer.model_copy(update={"llm_fallback": False})
    return base.model_copy(update=updates)


def _build_runtime_prompt_pipeline(config: AppConfig) -> Any:
    if not _runtime_fast_parse_enabled():
        return build_prompt_pipeline(config)
    return build_prompt_pipeline(
        config,
        extraction_validation_retries=_env_int("BC_RUNTIME_EXTRACTION_RETRIES", default=0),
        max_repairs=_env_int("BC_RUNTIME_VERIFIER_REPAIRS", default=0),
        max_semantic_repairs=_env_int("BC_RUNTIME_SEMANTIC_REPAIRS", default=0),
        run_semantic_validation=_env_flag("BC_RUNTIME_SEMANTIC_VALIDATION", default=False),
        run_verifier=_env_flag("BC_RUNTIME_VERIFIER", default=False),
    )


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


def _runtime_display_enum_value(value: str) -> str:
    return value.lower().replace("_", " ")


def _runtime_arm_prompt_phrase(field_path: str, value: str) -> str:
    label = _runtime_display_enum_value(value)
    if field_path == "cinematography.shot_size":
        return label
    if field_path == "cinematography.camera_angle":
        return f"{label} camera angle"
    if field_path == "cinematography.lens":
        return f"{label} lens"
    if field_path == "cinematography.lighting_mood":
        return f"{label} lighting"
    if field_path == "cinematography.color_treatment":
        return f"{label} color treatment"
    if field_path == "cinematography.composition":
        return f"{label} composition"
    if field_path.startswith("relation."):
        return f"{label} relation"
    return label


def _runtime_sampled_arm_prompt_suffix(sampled_arms: dict[str, str]) -> str:
    field_order = {
        "cinematography.shot_size": 0,
        "cinematography.camera_angle": 1,
        "cinematography.lens": 2,
        "cinematography.lighting_mood": 3,
        "cinematography.color_treatment": 4,
        "cinematography.composition": 5,
    }
    phrases = [
        _runtime_arm_prompt_phrase(field_path, value)
        for field_path, value in sorted(sampled_arms.items(), key=lambda item: (field_order.get(item[0], 99), item[0]))
        if value
    ]
    return ", ".join(dict.fromkeys(phrases))


def _runtime_prompt_with_sampled_arms(base_prompt: str, sampled_arms: dict[str, str]) -> str:
    suffix = _runtime_sampled_arm_prompt_suffix(sampled_arms)
    if not suffix:
        return base_prompt
    negative_marker = ". Negative prompt: "
    if negative_marker in base_prompt:
        positive, negative = base_prompt.split(negative_marker, 1)
        return f"{positive.rstrip('.')}, {suffix}{negative_marker}{negative}"
    return f"{base_prompt.rstrip('.')}, {suffix}."


def _runtime_fixed_arms(lock_rows: Any) -> dict[str, AxisDomain]:
    fixed: dict[str, AxisDomain] = {}
    for row in _normal_lock_rows(lock_rows):
        if len(row) < 4 or not bool(row[0]):
            continue
        field_path = str(row[1]).strip()
        raw_value = str(row[2]).strip()
        enum_value = _clean_selected_enum_cell(row[3])
        value = enum_value or raw_value
        if not field_path or not value:
            continue
        fixed[field_path] = AxisDomain(
            value=value,
            state=FieldState.EXPLICIT_LOCKED,
            source="gradio_lock_table",
        )
    return fixed


def _runtime_sampleable_axes(lock_rows: Any, fixed_arms: dict[str, AxisDomain]) -> dict[str, list[ThompsonArmState]]:
    axes: dict[str, list[ThompsonArmState]] = {}
    for row in _normal_lock_rows(lock_rows):
        if len(row) < 2:
            continue
        locked = bool(row[0])
        field_path = str(row[1]).strip()
        if locked or field_path in fixed_arms:
            continue
        domain_values = _enum_domain_values(field_path)
        if not domain_values:
            continue
        arms: list[ThompsonArmState] = []
        for index, value in enumerate(domain_values):
            alpha, beta = _runtime_arm_prior(field_path, value, index=index)
            arms.append(ThompsonArmState(axis=field_path, value=value, alpha=alpha, beta=beta))
        axes[field_path] = sorted(
            arms,
            key=lambda arm: arm.alpha / (arm.alpha + arm.beta),
            reverse=True,
        )
    return axes


def _runtime_target_manifest(
    session: RuntimeUISession,
    *,
    coordinate_id: str | None = None,
    rendered_prompt: str | None = None,
) -> EvaluationTargetManifest:
    return session.target_manifest.model_copy(
        update={
            "run_id": session.config.run.run_id,
            "prompt_document_id": session.document.prompt_document_id,
            "coordinate_id": coordinate_id,
            "rendered_prompt": rendered_prompt or session.rendered_prompt.rendered_prompt,
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


def _persist_runtime_batch_prompt(session: RuntimeUISession, item: SeedSweepWorkItem, *, batch_index: int) -> None:
    session.service.store.append(
        PersistenceRecord(
            record_id=f"runtime_batch_prompt:{item.coordinate_id}",
            record_type="runtime_batch_prompt",
            run_id=session.config.run.run_id,
            prompt_document_id=session.document.prompt_document_id,
            target_manifest_id=session.target_manifest.manifest_id,
            coordinate_id=item.coordinate_id,
            idempotency_key=f"runtime_batch_prompt:{item.coordinate_id}",
            payload={
                "batch_index": batch_index,
                "coordinate_id": item.coordinate_id,
                "rendered_prompt": item.rendered_prompt,
                "sampled_arms": item.sampled_arms,
                "locked_arms": item.locked_arms,
                "lhs_row": item.lhs_row,
                "combo_signature": item.combo_signature,
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
    sampleable_axes = _runtime_sampleable_axes(lock_rows, fixed_arms)
    lhs_count = max((len(arms) for arms in sampleable_axes.values()), default=1)
    cycle_index = (batch_index - 1) % lhs_count
    cycle_seed = 7 + ((batch_index - 1) // lhs_count) * 7919
    router_batch = LHSRouter(
        seed=cycle_seed,
        compatibility_prior=_runtime_compatibility_prior(),
        compatibility_prior_weight=0.25,
    ).propose(
        RouterInput(
            run_id=session.config.run.run_id,
            prompt_document_id=session.document.prompt_document_id,
            target_manifest_id=session.target_manifest.manifest_id,
            fixed_arms=fixed_arms,
            sampleable_axes=sampleable_axes,
            count=lhs_count,
        )
    )
    if not router_batch.coordinates:
        raise gr.Error("Router rejected the locked coordinate configuration.")
    coordinate_id = f"coord_{batch_index:03d}"
    coordinate = router_batch.coordinates[cycle_index % len(router_batch.coordinates)].model_copy(
        update={"coordinate_id": coordinate_id}
    )
    rendered_prompt = _runtime_prompt_with_sampled_arms(session.rendered_prompt.rendered_prompt, coordinate.sampled_arms)
    target_manifest = _runtime_target_manifest(
        session,
        coordinate_id=coordinate.coordinate_id,
        rendered_prompt=rendered_prompt,
    )
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
        rendered_prompt=rendered_prompt,
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
        rendered_prompt=rendered_prompt,
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


def _runtime_lhs_prompt_notification(*, batch_index: int, elapsed_seconds: int) -> str:
    return f"Batch {batch_index} LHS prompt ready: 0/5 seeds displayed, elapsed {_format_runtime_elapsed(elapsed_seconds)}."


def _runtime_generation_notification(
    *,
    batch_index: int,
    rendered_count: int,
    total_count: int,
    elapsed_seconds: int,
) -> str:
    return (
        f"Batch {batch_index} rendering: {rendered_count}/{total_count} seeds displayed, "
        f"elapsed {_format_runtime_elapsed(elapsed_seconds)}."
    )


def _runtime_stream_poll_seconds() -> float:
    return max(0.01, float(os.environ.get("BC_GRADIO_STREAM_POLL_SECONDS", str(RUNTIME_STREAM_POLL_SECONDS))))


def _runtime_final_seed_refresh_seconds() -> float:
    return max(
        0.0,
        float(os.environ.get("BC_GRADIO_FINAL_SEED_REFRESH_SECONDS", str(RUNTIME_FINAL_SEED_REFRESH_SECONDS))),
    )


def _runtime_coordinate_candidate_count(records: list[PersistenceRecord], coordinate_id: str) -> int:
    return sum(
        1
        for record in records
        if record.record_type == "candidate_record" and record.coordinate_id == coordinate_id
    )


def _runtime_coordinate_rendered_prompt(
    records: list[PersistenceRecord],
    *,
    coordinate_id: str | None,
    fallback: str,
) -> str:
    if coordinate_id is None:
        return fallback
    for record in reversed(records):
        if record.coordinate_id != coordinate_id:
            continue
        if record.record_type not in {"runtime_batch_prompt", "coordinate_record", "target_manifest"}:
            continue
        rendered_prompt = str(record.payload.get("rendered_prompt") or "").strip()
        if rendered_prompt:
            return rendered_prompt
    return fallback


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
    rendered_prompt: str | None = None,
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
        selected_id = None

    counts = None
    if records:
        try:
            counts = reconstruct_run_state(records)
        except ValueError:
            counts = None

    updates: dict[str, Any] = {
        "run_id": session.config.run.run_id,
        "raw_prompt": session.config.run.raw_user_prompt,
        "rendered_prompt": rendered_prompt
        or _runtime_coordinate_rendered_prompt(
            records,
            coordinate_id=current_coordinate_id,
            fallback=state.rendered_prompt or session.rendered_prompt.rendered_prompt,
        ),
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
    return _clamp_catalog_page_state(state.model_copy(update=updates))


def _runtime_state_from_session(
    session: RuntimeUISession,
    state: GradioSimulationState,
    *,
    notification: str | None = None,
) -> GradioSimulationState:
    review = pre_run_modal_from_prompt(session.document)
    recovered = state.model_copy(
        update={
            "run_id": session.config.run.run_id,
            "raw_prompt": session.config.run.raw_user_prompt,
            "rendered_prompt": session.rendered_prompt.rendered_prompt,
            "prompt_document_id": session.document.prompt_document_id,
            "review": review,
            "notification": notification or state.notification,
        }
    )
    if session.service.store.path.exists():
        return _runtime_state_from_store(session, recovered, notification=notification or recovered.notification)
    return recovered


def _recover_runtime_ready_state(
    state: GradioSimulationState,
    raw_prompt: str | None,
) -> tuple[GradioSimulationState, RuntimeUISession | None]:
    session = _RUNTIME_SESSIONS.get(state.run_id)
    if session is not None:
        recovered = _runtime_state_from_session(session, state)
        return recovered, session

    prompt = (raw_prompt or state.raw_prompt or "").strip()
    if not prompt:
        return state, None

    recovered_state, *_ = start_pre_run_runtime(prompt, state)
    recovered = _coerce_state(recovered_state)
    return recovered, _RUNTIME_SESSIONS.get(recovered.run_id)


def _runtime_generation_started(state: GradioSimulationState) -> bool:
    return state.batch_index > 0 or state.generated_count > 0 or bool(state.current_batch_expected_seeds)


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
            "current_batch": [],
            "current_batch_expected_seeds": [],
            "curated": [],
            "catalog_page_index": 0,
            "selected_candidate_id": None,
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
        pipeline = _build_runtime_prompt_pipeline(config)
    except Exception as error:
        raise gr.Error(f"Backend pre-run startup failed: {error}") from error

    try:
        result = pipeline.run_spec(prompt)
    except Exception as error:
        document = _runtime_parse_failed_document(prompt, error)
        review = pre_run_modal_from_prompt(document)
        state = state.model_copy(
            update={
                "run_id": config.run.run_id,
                "raw_prompt": prompt,
                "rendered_prompt": "",
                "prompt_document_id": document.prompt_document_id,
                "batch_index": 0,
                "review": review,
                "current_batch": [],
                "current_batch_expected_seeds": [],
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
            _review_markdown(prompt, document, review, ""),
            _lock_table_from_review(review),
            gr.update(interactive=False),
            _status_html(state),
        )

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
            if _asr_release_before_runtime_service_enabled():
                _unload_runtime_asr("runtime pre-run service build")
            _release_inactive_runtime_sessions(
                keep_run_id=config.run.run_id,
                reason="gradio_runtime_replaced_by_new_prompt",
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
            "current_batch_expected_seeds": [],
            "curated": [],
            "catalog_page_index": 0,
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
    session = _RUNTIME_SESSIONS.get(state.run_id)
    if session is not None and _runtime_generation_started(state):
        session.service.request_stop()
        state = state.model_copy(
            update={
                "notification": "Stop requested. Finishing the current 5-seed batch before stopping.",
            }
        )
        return state, gr.update(), gr.update(interactive=False), _status_html(state)

    session = _RUNTIME_SESSIONS.pop(state.run_id, None)
    if session is not None:
        session.service.stop_with_reason(
            "pre_run_cancel_requested",
            details={"run_id": session.config.run.run_id},
        )
    state = state.model_copy(update={"notification": "Backend pre-run canceled."})
    return state, gr.update(visible=False), gr.update(interactive=False), _status_html(state)


def generate_seed_sweep(
    state_value: GradioSimulationState | dict[str, Any] | None,
    lock_rows: Any,
    iqa_cutoff: float,
    alignment_cutoff: float,
    raw_prompt: str | None = None,
):
    state = _coerce_state(state_value)
    if state.review is None or not state.review.can_begin_generation:
        raise gr.Error("Pre-run review is not ready. Submit the prompt, wait for pre-run review, then click Generate.")
    pending = _make_pending_batch(state, lock_rows)
    generated_state = state.model_copy(
        update={
            "current_batch": pending,
            "current_batch_expected_seeds": [candidate.seed for candidate in pending],
            "batch_index": state.batch_index + 1,
            "generated_count": state.generated_count + len(pending),
            "notification": "5-seed preview generated.",
        }
    )
    yield (
        generated_state,
        gr.update(visible=True),
        gr.update(),
        gr.update(visible=True),
        *_preview_slot_paths(generated_state),
        _catalog_gallery(generated_state),
        *_catalog_slot_paths(generated_state),
        *_catalog_page_controls(generated_state),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        _status_html(generated_state),
    )
    time.sleep(0.6)
    evaluated = _evaluate_batch(state, pending, iqa_cutoff=iqa_cutoff, alignment_cutoff=alignment_cutoff)
    promoted = [candidate for candidate in evaluated if candidate.promoted]
    visible_ids = {candidate.candidate_id for candidate in [*state.curated, *promoted]}
    selected_id = state.selected_candidate_id if state.selected_candidate_id in visible_ids else None
    evaluated_state = generated_state.model_copy(
        update={
            "current_batch": evaluated,
            "current_batch_expected_seeds": [candidate.seed for candidate in evaluated],
            "curated": [*state.curated, *promoted],
            "selected_candidate_id": selected_id,
            "iqa_evaluated_count": state.iqa_evaluated_count + len(evaluated),
            "vlm_evaluated_count": state.vlm_evaluated_count + sum(1 for item in evaluated if item.pass_iqa),
            "notification": f"Evaluation complete: {len(promoted)} promoted.",
        }
    )
    yield (
        evaluated_state,
        gr.update(visible=True),
        gr.update(),
        gr.update(visible=True),
        *_preview_slot_paths(evaluated_state),
        _catalog_gallery(evaluated_state),
        *_catalog_slot_paths(evaluated_state),
        *_catalog_page_controls(evaluated_state),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        _status_html(evaluated_state),
    )


def generate_seed_sweep_runtime(
    state_value: GradioSimulationState | dict[str, Any] | None,
    lock_rows: Any,
    iqa_cutoff: float,
    alignment_cutoff: float,
    raw_prompt: str | None = None,
):
    state = _coerce_state(state_value)
    session = _RUNTIME_SESSIONS.get(state.run_id)
    if state.review is None or not state.review.can_begin_generation or session is None:
        state, session = _recover_runtime_ready_state(state, raw_prompt)
    if state.review is None or not state.review.can_begin_generation:
        raise gr.Error(
            "Pre-run review is not ready. Submit the prompt, wait for Backend pre-run ready, then click Generate."
        )
    session = _RUNTIME_SESSIONS.get(state.run_id)
    if session is None:
        raise gr.Error("Runtime backend session expired. Submit the prompt again to rebuild it, then click Generate.")

    def output_for(
        next_state: GradioSimulationState,
        review_markup: str | None = None,
        *,
        clear_catalog: bool = False,
    ):
        return (
            next_state,
            gr.update(visible=True),
            review_markup if review_markup is not None else _runtime_review_markup(session, next_state),
            gr.update(visible=True),
            *_preview_slot_paths(next_state),
            _catalog_gallery_update(next_state, clear=clear_catalog),
            *_catalog_slot_paths(next_state, clear=clear_catalog),
            *_catalog_page_controls(next_state),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            _status_html(next_state),
        )

    active_state = state.model_copy(
        update={
            "notification": "Backend run started.",
        }
    )
    yield output_for(active_state)

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
        if session.service.stop_requested:
            if session.service.state != RunRuntimeState.STOPPED:
                session.service.stop_with_reason(
                    "gradio_runtime_cancel_requested",
                    details={
                        "elapsed_seconds": elapsed_seconds,
                        "run_id": session.config.run.run_id,
                    },
                )
            stopped_state = _runtime_state_from_store(
                session,
                working_state,
                notification="Stopped by backend/requested stop.",
            )
            yield output_for(stopped_state)
            return

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
            yield output_for(stopped_state)
            return

        batch_index = working_state.batch_index + 1
        item = _runtime_work_item_for_batch(
            session,
            lock_rows=lock_rows,
            fixed_arms=fixed_arms,
            evaluation_plan=evaluation_plan,
            batch_index=batch_index,
        )
        working_state = working_state.model_copy(
            update={
                "current_batch": [],
                "current_batch_expected_seeds": [request.seed for request in item.generation_requests],
                "batch_index": batch_index,
                "rendered_prompt": item.rendered_prompt,
                "notification": _runtime_lhs_prompt_notification(
                    batch_index=batch_index,
                    elapsed_seconds=elapsed_seconds,
                ),
            }
        )
        _persist_runtime_batch_prompt(session, item, batch_index=batch_index)
        yield output_for(working_state, _runtime_review_markup(session, working_state, item=item, batch_index=batch_index))
        session.service.enqueue(item)
        last_rendered_count = 0
        last_catalog_signature = _catalog_gallery_signature(working_state)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(session.service.tick)
            while not future.done():
                elapsed_seconds = int(time.monotonic() - loop_started_at)
                records = session.service.store.replay()
                rendered_count = _runtime_coordinate_candidate_count(records, item.coordinate_id)
                catalog_signature = _runtime_catalog_signature_from_records(records)
                catalog_changed = catalog_signature != last_catalog_signature
                if rendered_count != last_rendered_count or catalog_changed:
                    last_rendered_count = rendered_count
                    streaming_state = _runtime_state_from_store(
                        session,
                        working_state,
                        current_coordinate_id=item.coordinate_id,
                        notification=_runtime_generation_notification(
                            batch_index=batch_index,
                            rendered_count=rendered_count,
                            total_count=len(item.generation_requests),
                            elapsed_seconds=elapsed_seconds,
                        ),
                        rendered_prompt=item.rendered_prompt,
                    )
                    if catalog_changed:
                        yield output_for(
                            streaming_state,
                            _runtime_review_markup(session, streaming_state, item=item, batch_index=batch_index),
                            clear_catalog=True,
                        )
                        time.sleep(0.05)
                    yield output_for(
                        streaming_state,
                        _runtime_review_markup(session, streaming_state, item=item, batch_index=batch_index),
                    )
                    working_state = streaming_state
                    last_catalog_signature = _catalog_gallery_signature(streaming_state)
                time.sleep(_runtime_stream_poll_seconds())
            decision = future.result()
        elapsed_seconds = int(time.monotonic() - loop_started_at)
        records = session.service.store.replay()
        rendered_count = _runtime_coordinate_candidate_count(records, item.coordinate_id)
        catalog_signature = _runtime_catalog_signature_from_records(records)
        catalog_changed = catalog_signature != last_catalog_signature
        if rendered_count != last_rendered_count or catalog_changed:
            streaming_state = _runtime_state_from_store(
                session,
                working_state,
                current_coordinate_id=item.coordinate_id,
                notification=_runtime_generation_notification(
                    batch_index=batch_index,
                    rendered_count=rendered_count,
                    total_count=len(item.generation_requests),
                    elapsed_seconds=elapsed_seconds,
                ),
                rendered_prompt=item.rendered_prompt,
            )
            if catalog_changed:
                yield output_for(
                    streaming_state,
                    _runtime_review_markup(session, streaming_state, item=item, batch_index=batch_index),
                    clear_catalog=True,
                )
                time.sleep(0.05)
            yield output_for(
                streaming_state,
                _runtime_review_markup(session, streaming_state, item=item, batch_index=batch_index),
            )
            working_state = streaming_state
            last_catalog_signature = _catalog_gallery_signature(streaming_state)
            if rendered_count >= len(item.generation_requests):
                time.sleep(_runtime_final_seed_refresh_seconds())

        if decision.next_state == RunRuntimeState.STOPPED and decision.reason != "pending_coordinates":
            stopped_state = _runtime_state_from_store(
                session,
                working_state,
                current_coordinate_id=item.coordinate_id,
                notification="Stopped by backend/requested stop.",
            )
            yield output_for(stopped_state)
            return

        if decision.action == LoopAction.GATE_BLOCKED:
            session.service.stop_with_reason(
                "backend_gate_blocked",
                details={
                    "reason": decision.reason,
                    "coordinate_id": item.coordinate_id,
                    "batch_index": batch_index,
                },
            )
            stopped_state = _runtime_state_from_store(
                session,
                working_state,
                current_coordinate_id=item.coordinate_id,
                notification=f"Stopped by backend gate: {decision.reason}",
            )
            yield output_for(stopped_state)
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

        if session.service.stop_requested:
            if session.service.state != RunRuntimeState.STOPPED:
                session.service.stop_with_reason(
                    "gradio_runtime_cancel_requested",
                    details={
                        "elapsed_seconds": elapsed_seconds,
                        "run_id": session.config.run.run_id,
                        "coordinate_id": item.coordinate_id,
                        "batch_index": batch_index,
                    },
                )
            next_state = _runtime_state_from_store(
                session,
                next_state,
                current_coordinate_id=item.coordinate_id,
                notification="Stopped by backend/requested stop.",
            )
        elif elapsed_seconds >= RUNTIME_LOOP_LIMIT_SECONDS:
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

        yield output_for(next_state)
        if session.service.state == RunRuntimeState.STOPPED:
            return
        working_state = next_state


def select_curated_index(index: int, state_value: GradioSimulationState | dict[str, Any] | None):
    state = _coerce_state(state_value)
    visible = _visible_catalog(state)
    page_start, _page_end = _catalog_page_window(state)
    absolute_index = page_start + index
    if not isinstance(index, int) or index < 0 or absolute_index >= len(visible):
        return state, gr.update(visible=False), _detail_image_update(None), "", _status_html(state)
    candidate = visible[absolute_index]
    state = state.model_copy(
        update={
            "selected_candidate_id": candidate.candidate_id,
            "notification": f"Selected seed {candidate.seed}.",
        }
    )
    return state, gr.update(visible=True), _detail_image_update(candidate), _detail_markdown(state, candidate), _status_html(state)


def select_curated(evt: gr.SelectData, state_value: GradioSimulationState | dict[str, Any] | None):
    index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    return select_curated_index(index, state_value)


def move_catalog_page(state_value: GradioSimulationState | dict[str, Any] | None, delta: int):
    state = _coerce_state(state_value)
    page_count = _catalog_page_count(state)
    next_index = min(max(state.catalog_page_index + delta, 0), page_count - 1)
    state = state.model_copy(
        update={
            "catalog_page_index": next_index,
            "notification": f"Curated catalog page {next_index + 1} of {page_count}.",
        }
    )
    return (
        state,
        *_catalog_slot_paths(state),
        *_catalog_page_controls(state),
        _status_html(state),
    )


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
    return state, gr.update(visible=True), _detail_image_update(candidate), _detail_markdown(state, candidate), _status_html(state)


def submit_feedback(
    state_value: GradioSimulationState | dict[str, Any] | None,
    action_value: str,
):
    state = _coerce_state(state_value)
    selected = _selected_candidate(state)
    if selected is None:
        return state, _catalog_gallery(state), gr.update(visible=False), _detail_image_update(None), "", _status_html(state)
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
        next_state = next_state.model_copy(update={"selected_candidate_id": None})
    selected_after = _selected_candidate(next_state)
    return (
        next_state,
        _catalog_gallery(next_state),
        gr.update(visible=selected_after is not None),
        _detail_image_update(selected_after),
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
        return state, _catalog_gallery(state), gr.update(visible=False), _detail_image_update(None), "", _status_html(state)
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
    if action in {FeedbackAction.REJECT, FeedbackAction.SHRED}:
        next_state = next_state.model_copy(update={"selected_candidate_id": None})
    selected_after = _selected_candidate(next_state)
    return (
        next_state,
        _catalog_gallery(next_state),
        gr.update(visible=selected_after is not None),
        _detail_image_update(selected_after),
        _detail_markdown(next_state, selected_after),
        _status_html(next_state),
    )


def submit_feedback_with_catalog_slots(
    handler: Any,
    state_value: GradioSimulationState | dict[str, Any] | None,
    action_value: str,
):
    next_state, gallery, detail_panel_update, detail_image, detail_report, status = handler(state_value, action_value)
    state = _clamp_catalog_page_state(_coerce_state(next_state))
    return (
        state,
        gallery,
        *_catalog_slot_paths(state),
        *_catalog_page_controls(state),
        detail_panel_update,
        detail_image,
        detail_report,
        status,
    )


def transcribe_microphone_to_prompt(
    audio: object,
    current_prompt: str,
    state_value: GradioSimulationState | dict[str, Any] | None,
):
    state = _coerce_state(state_value)
    if audio is None:
        return current_prompt, state, _status_html(state)
    blocked_message = _asr_transcription_block_message()
    if blocked_message is not None:
        state = state.model_copy(update={"notification": blocked_message})
        return current_prompt, state, _status_html(state)
    _set_asr_runtime_status("loading", "Cohere Transcribe ASR is transcribing microphone audio.")
    transcriber = default_transcriber()
    try:
        transcript = transcriber.transcribe(audio)
    except Exception as error:
        unload = getattr(transcriber, "unload", None)
        if unload is not None:
            unload()
        _set_asr_runtime_status("failed", "Cohere Transcribe ASR transcription failed.", error=error)
        state = state.model_copy(update={"notification": f"ASR failed: {error}"})
        return current_prompt, state, _status_html(state)
    _set_asr_runtime_status("ready", "Cohere Transcribe ASR transcribed microphone audio.")
    prompt = transcript.strip()
    release_after_transcribe = _asr_release_after_transcribe_enabled()
    if not prompt:
        state = state.model_copy(update={"notification": "ASR returned an empty transcript."})
        return current_prompt, state, _status_html(state)
    notification = "ASR transcript inserted."
    if release_after_transcribe:
        _unload_runtime_asr("microphone transcription")
        notification = "ASR transcript inserted; ASR released by low-VRAM policy."
    state = state.model_copy(update={"raw_prompt": prompt, "notification": notification})
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
    blocked_message = _asr_transcription_block_message()
    if blocked_message is not None:
        blocked_state = state.model_copy(update={"notification": blocked_message})
        yield gr.update(value=current_prompt, interactive=True), blocked_state, _status_html(blocked_state)
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
#workflow-diagram,
#workflow-explanation {
    color: var(--bc-ink) !important;
}
#workflow-explanation,
#workflow-explanation *,
#workflow-diagram,
#workflow-diagram * {
    opacity: 1 !important;
}
#workflow-explanation p,
#workflow-explanation li,
#workflow-explanation span {
    color: var(--bc-ink) !important;
}
#workflow-diagram {
    overflow-x: auto;
}
.bc-flow {
    display: block;
    color: var(--bc-ink);
    overflow-wrap: anywhere;
}
.bc-flow-svg,
.bc-flow-svg-mobile {
    display: block;
    width: 100%;
    height: auto;
    color: var(--bc-ink);
}
.bc-flow-svg {
    min-width: 760px;
}
.bc-flow-svg-mobile {
    display: none;
    min-width: 0;
}
.bc-flow-svg text,
.bc-flow-svg-mobile text {
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    letter-spacing: 0;
    fill: var(--bc-ink);
}
.bc-flow-node rect,
.bc-flow-gate polygon {
    filter: url(#bc-flow-shadow);
    stroke-width: 1.5;
}
.bc-flow-svg-mobile .bc-flow-node rect,
.bc-flow-svg-mobile .bc-flow-gate polygon {
    filter: url(#bc-flow-shadow-mobile);
}
.bc-flow-node rect {
    fill: #eef8f6;
    stroke: #0f766e;
}
.bc-flow-gate polygon {
    fill: #fff8e6;
    stroke: #a16207;
}
.bc-flow-kicker {
    color: var(--bc-teal-dark);
    fill: var(--bc-teal-dark);
    font-size: 11px;
    font-weight: 820;
    text-transform: uppercase;
}
.bc-flow-gate .bc-flow-kicker {
    fill: #6f4400;
}
.bc-flow-label {
    font-size: 13px;
}
.bc-flow-gate-label {
    font-size: 13px;
}
.bc-flow-title {
    fill: var(--bc-ink);
    font-size: 16px;
    font-weight: 780;
}
.bc-flow-label tspan:not(.bc-flow-kicker):not(.bc-flow-title),
.bc-flow-gate-label tspan:not(.bc-flow-kicker):not(.bc-flow-title) {
    fill: #30433d;
    font-size: 12px;
    font-weight: 500;
}
.bc-flow-edge {
    fill: none;
    stroke: #17211f;
    stroke-width: 2.1;
    stroke-linecap: round;
    stroke-linejoin: round;
}
#bc-flow-arrow path {
    fill: #17211f;
}
#bc-flow-arrow-mobile path {
    fill: #17211f;
}
.bc-flow-edge-loop {
    stroke: #0f766e;
    stroke-dasharray: 7 6;
}
#bc-flow-arrow-loop path {
    fill: #0f766e;
}
#bc-flow-arrow-loop-mobile path {
    fill: #0f766e;
}
.bc-flow-edge-label {
    fill: #0f3d36;
    font-size: 12px;
    font-weight: 720;
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
    aspect-ratio: 3 / 4;
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
    aspect-ratio: 3 / 4;
    overflow: hidden !important;
}
#prompt-microphone .wrap {
    min-height: 100%;
    height: 100%;
    aspect-ratio: 3 / 4;
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
    background: #ffffff !important;
    color: var(--bc-ink) !important;
}
#seed-slot-row {
    display: grid !important;
    grid-template-columns: repeat(5, minmax(120px, 1fr));
    gap: 12px;
}
#seed-slot-row .bc-seed-slot {
    min-width: 0 !important;
    border: 1px solid var(--bc-line) !important;
    border-radius: 8px !important;
    background: #f8faf9 !important;
    overflow: hidden !important;
}
#seed-slot-row .bc-seed-slot img {
    width: 100% !important;
    aspect-ratio: 3 / 4;
    object-fit: cover !important;
    border-radius: 6px;
    background: #17211f !important;
    transition: opacity 180ms ease, filter 180ms ease, transform 180ms ease;
}
#seed-slot-row .bc-seed-slot img:hover {
    transform: scale(1.01);
}
#catalog-slot-panel {
    border: 1px solid var(--bc-line);
    border-radius: 8px;
    padding: 12px;
    background: #ffffff;
}
#catalog-slot-title .prose,
#catalog-slot-title .prose p {
    margin: 0 0 10px !important;
    color: var(--bc-ink) !important;
    font-weight: 780 !important;
}
#catalog-pagination {
    display: grid !important;
    grid-template-columns: minmax(110px, auto) minmax(180px, 1fr) minmax(110px, auto);
    gap: 10px;
    align-items: center;
    margin: 0 0 12px !important;
}
#catalog-pagination button {
    border: 1px solid #5f6f68 !important;
    border-radius: 6px !important;
    background: #ffffff !important;
    color: #0f3d36 !important;
    font-weight: 760 !important;
    min-height: 34px !important;
}
#catalog-pagination button:disabled {
    opacity: 0.55 !important;
    color: #475569 !important;
    background: #f1f5f3 !important;
}
#catalog-page-status,
#catalog-page-status .prose,
#catalog-page-status .prose p {
    margin: 0 !important;
    color: var(--bc-ink) !important;
    font-weight: 760 !important;
    text-align: center;
}
#catalog-slot-row {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 12px;
}
#catalog-slot-row .bc-catalog-slot {
    min-width: 0 !important;
    border: 1px solid var(--bc-line) !important;
    border-radius: 8px !important;
    background: #f8faf9 !important;
    overflow: hidden !important;
}
#catalog-slot-row .bc-catalog-slot img {
    display: block !important;
    width: 100% !important;
    aspect-ratio: 1 / 1;
    object-fit: cover !important;
    border-radius: 6px;
    background: #17211f !important;
    transition: opacity 180ms ease, filter 180ms ease, transform 180ms ease;
}
#catalog-slot-row .bc-catalog-slot img:hover {
    transform: scale(1.01);
}
#lock-table table tbody tr td:nth-child(4),
#lock-table table thead tr th:nth-child(4) {
    background: #ecfdf5 !important;
    color: #065f46 !important;
    font-weight: 700 !important;
}
#lock-table table tbody tr td:nth-child(5),
#lock-table table thead tr th:nth-child(5) {
    background: #fff8e6 !important;
    color: #553a08 !important;
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
#catalog-gallery img {
    display: block !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 172px !important;
    object-fit: cover !important;
    background: #17211f !important;
}
#seed-gallery img {
    transition: opacity 180ms ease, filter 180ms ease, transform 180ms ease;
}
#seed-gallery img:hover {
    transform: scale(1.01);
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
#catalog-gallery .thumbnail-item {
    border: 1px solid var(--bc-line) !important;
    border-radius: 8px !important;
    min-height: 184px !important;
    overflow: hidden !important;
}
#catalog-gallery .caption,
#catalog-gallery figcaption,
#catalog-gallery .empty,
#catalog-gallery .empty * {
    color: var(--bc-muted) !important;
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
#detail-image,
#detail-image > div,
#detail-image .image-container,
#detail-image .wrap,
#detail-image .empty {
    background: #f8faf9 !important;
    color: var(--bc-ink) !important;
}
#detail-image .label-wrap,
#detail-image .block-label,
#detail-image [data-testid="block-label"],
#detail-image label {
    background: #17211f !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    border-radius: 0 0 4px 0 !important;
}
#detail-report,
#detail-report > div,
#detail-report .prose {
    background: #f8faf9 !important;
    color: var(--bc-ink) !important;
}
#detail-report {
    border: 1px solid var(--bc-line) !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
#detail-report .prose,
#detail-report .prose p,
#detail-report .prose li,
#detail-report .prose span {
    color: var(--bc-ink) !important;
}
#detail-report .prose strong {
    color: var(--bc-teal-dark) !important;
    font-weight: 780 !important;
}
#detail-report .prose code {
    background: #17211f !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    border-radius: 4px !important;
    padding: 1px 4px !important;
    font-size: 0.92em !important;
}
#detail-panel button {
    background: #475569 !important;
    border-color: #475569 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}
#detail-panel button:hover {
    background: #334155 !important;
    border-color: #334155 !important;
}
.bc-review {
    color: var(--bc-ink);
}
.bc-batch-prompt {
    border: 1px solid #9fb9b2;
    border-left: 5px solid var(--bc-teal);
    border-radius: 8px;
    background: #eef8f6;
    color: var(--bc-ink);
    padding: 12px;
    margin-bottom: 12px;
}
.bc-batch-prompt h3 {
    margin: 4px 0 10px;
    font-size: 15px;
    line-height: 1.25;
    letter-spacing: 0;
}
.bc-batch-prompt .bc-compiled-prompt {
    margin-bottom: 10px;
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
.bc-triplet-stack {
    display: grid;
    gap: 10px;
}
.bc-triplet-row {
    display: grid;
    grid-template-columns: minmax(170px, 1fr) minmax(140px, 0.72fr) minmax(170px, 1fr);
    gap: 10px;
    align-items: stretch;
}
.bc-triplet-box,
.bc-standalone-object {
    border: 1px solid #b9c8c1;
    border-radius: 8px;
    background: var(--bc-surface);
    padding: 10px;
    min-width: 0;
}
.bc-triplet-object {
    border-left: 4px solid var(--bc-teal);
}
.bc-triplet-relation {
    border-left: 4px solid var(--bc-amber);
    background: #fff8e6;
}
.bc-triplet-kicker {
    display: block;
    color: var(--bc-muted);
    font-size: 11px;
    font-weight: 760;
    text-transform: uppercase;
    letter-spacing: 0;
    margin-bottom: 4px;
}
.bc-triplet-box strong,
.bc-standalone-object strong {
    display: block;
    color: var(--bc-ink);
    font-size: 14px;
    line-height: 1.25;
    overflow-wrap: anywhere;
}
.bc-triplet-box small,
.bc-standalone-object small {
    display: block;
    color: var(--bc-muted);
    font-size: 12px;
    line-height: 1.3;
    margin: 4px 0 7px;
    overflow-wrap: anywhere;
}
.bc-standalone-objects {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-top: 12px;
}
.bc-standalone-objects .bc-eyebrow {
    grid-column: 1 / -1;
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
    .bc-triplet-row {
        grid-template-columns: 1fr;
    }
    .bc-review-block-wide {
        grid-column: auto;
    }
    .bc-workflow-row {
        flex-direction: column;
    }
    .bc-flow-svg {
        display: none;
    }
    .bc-flow-svg-mobile {
        display: block;
        min-width: 0;
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
    #seed-slot-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    #catalog-slot-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
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
    start_event_handler = start_handler
    generate_event_handler = generate_handler
    microphone_event_handler = transcribe_microphone_to_prompt_steps

    with gr.Blocks(
        title="Bruteforce Canvas",
        fill_width=True,
        analytics_enabled=False,
    ) as demo:
        state = gr.State(initial_state())
        with gr.Column(elem_id="bc-app"):
            gr.HTML('<header id="bc-title"><h1>BRUTEFORCE CANVAS</h1></header>')
            with gr.Accordion("Workflow diagram", open=True, elem_id="workflow-accordion"):
                with gr.Column(elem_classes=["bc-workflow-row"]):
                    gr.HTML(
                        WORKFLOW_DIAGRAM_HTML,
                        elem_id="workflow-diagram",
                        elem_classes=["bc-workflow-diagram"],
                        container=False,
                        padding=False,
                    )
                    gr.Markdown(
                        WORKFLOW_EXPLANATION_MARKDOWN,
                        elem_id="workflow-explanation",
                        elem_classes=["bc-workflow-explanation"],
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

            with gr.Group(visible=False, elem_id="pre-run-panel") as review_panel:
                parsed_report = gr.HTML()
                with gr.Group(elem_id="lock-table-panel") as lock_table_panel:
                    lock_table = gr.Dataframe(
                        headers=LOCK_TABLE_HEADERS,
                        datatype=["bool", "str", "str", "str", "str", "str", "str", "str", "str"],
                        type="array",
                        interactive=True,
                        label="Enum locks",
                        row_count=10,
                        wrap=True,
                        elem_id="lock-table",
                    )
                with gr.Row():
                    iqa_cutoff = gr.Slider(
                        0.0,
                        1.0,
                        value=0.55,
                        step=0.01,
                        label="Image Quality Assessment cutoff",
                    )
                    alignment_cutoff = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Alignment cutoff")
                with gr.Row():
                    generate = gr.Button("Generate", variant="primary", interactive=False)
                    cancel = gr.Button("Cancel")

            with gr.Group(visible=False, elem_id="active-panel") as active_panel:
                with gr.Row(elem_id="seed-slot-row"):
                    seed_slots = [
                        gr.Image(
                            label=f"seed {seed}",
                            type="filepath",
                            interactive=False,
                            height=320,
                            buttons=["fullscreen"],
                            elem_classes=["bc-seed-slot"],
                            elem_id=f"seed-slot-{seed}",
                        )
                        for seed in DEFAULT_SEED_BUNDLE
                    ]

            status = gr.HTML(_status_html(initial_state()))

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
                        visible=False,
                    )
                    with gr.Group(visible=False, elem_id="catalog-slot-panel") as catalog_panel:
                        gr.Markdown("Curated catalog", elem_id="catalog-slot-title")
                        with gr.Row(elem_id="catalog-pagination"):
                            catalog_prev_btn = gr.Button(
                                "Prev page",
                                size="sm",
                                interactive=False,
                                elem_id="catalog-prev-page",
                            )
                            catalog_page_status = gr.Markdown(
                                _catalog_page_label(initial_state()),
                                elem_id="catalog-page-status",
                            )
                            catalog_next_btn = gr.Button(
                                "Next page",
                                size="sm",
                                interactive=False,
                                elem_id="catalog-next-page",
                            )
                        with gr.Row(elem_id="catalog-slot-row"):
                            catalog_slots = [
                                gr.Image(
                                    label=f"curated {index + 1}",
                                    type="filepath",
                                    interactive=False,
                                    height=205,
                                    buttons=["fullscreen"],
                                    elem_classes=["bc-catalog-slot"],
                                    elem_id=f"catalog-slot-{index + 1}",
                                )
                                for index in range(CATALOG_SLOT_COUNT)
                            ]
                with gr.Column(scale=2, visible=False, elem_id="detail-panel") as detail_panel:
                    detail_image = gr.Image(label="Selected image", elem_id="detail-image", height=430)
                    detail_report = gr.Markdown(elem_id="detail-report")
                    with gr.Row():
                        previous_btn = gr.Button("Prev", size="sm")
                        next_btn = gr.Button("Next", size="sm")
                    with gr.Row():
                        up_btn = gr.Button("Thumbs up", size="sm", variant="secondary")
                        down_btn = gr.Button("Thumbs down", size="sm", variant="secondary")
                        trash_btn = gr.Button("Trash", size="sm", variant="stop")

            def start_with_catalog_hidden(raw_prompt: str, current_state: Any):
                next_state, panel, report, locks, generate_button, next_status = start_event_handler(
                    raw_prompt,
                    current_state,
                )
                return (
                    next_state,
                    panel,
                    report,
                    gr.update(value=locks, visible=True),
                    generate_button,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    next_status,
                )

            def cancel_with_catalog_hidden(current_state: Any):
                next_state, panel, generate_button, next_status = cancel_handler(current_state)
                if _runtime_generation_started(_coerce_state(next_state)):
                    return (
                        next_state,
                        panel,
                        generate_button,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(),
                        next_status,
                    )
                return (
                    next_state,
                    panel,
                    generate_button,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    next_status,
                )

            submit.click(
                start_with_catalog_hidden,
                inputs=[prompt, state],
                outputs=[
                    state,
                    review_panel,
                    parsed_report,
                    lock_table,
                    generate,
                    lock_table_panel,
                    catalog_panel,
                    status,
                ],
            )
            prompt.submit(
                start_with_catalog_hidden,
                inputs=[prompt, state],
                outputs=[
                    state,
                    review_panel,
                    parsed_report,
                    lock_table,
                    generate,
                    lock_table_panel,
                    catalog_panel,
                    status,
                ],
            )
            microphone.change(
                microphone_event_handler,
                inputs=[microphone, prompt, state],
                outputs=[prompt, state, status],
                show_progress="minimal",
            )
            cancel.click(
                cancel_with_catalog_hidden,
                inputs=[state],
                outputs=[state, review_panel, generate, lock_table_panel, lock_table, catalog_panel, status],
                queue=False,
            )
            generate.click(
                generate_event_handler,
                inputs=[state, lock_table, iqa_cutoff, alignment_cutoff, prompt],
                outputs=[
                    state,
                    review_panel,
                    parsed_report,
                    active_panel,
                    *seed_slots,
                    catalog_gallery,
                    *catalog_slots,
                    catalog_prev_btn,
                    catalog_next_btn,
                    catalog_page_status,
                    catalog_panel,
                    lock_table_panel,
                    lock_table,
                    status,
                ],
                show_progress="minimal",
            )
            for slot_index, catalog_slot in enumerate(catalog_slots):
                catalog_slot.select(
                    lambda current_state, index=slot_index: select_curated_index(index, current_state),
                    inputs=[state],
                    outputs=[state, detail_panel, detail_image, detail_report, status],
                )
            catalog_prev_btn.click(
                lambda current_state: move_catalog_page(current_state, -1),
                inputs=[state],
                outputs=[state, *catalog_slots, catalog_prev_btn, catalog_next_btn, catalog_page_status, status],
            )
            catalog_next_btn.click(
                lambda current_state: move_catalog_page(current_state, 1),
                inputs=[state],
                outputs=[state, *catalog_slots, catalog_prev_btn, catalog_next_btn, catalog_page_status, status],
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
                lambda current_state: submit_feedback_with_catalog_slots(
                    feedback_handler,
                    current_state,
                    FeedbackAction.ACCEPT.value,
                ),
                inputs=[state],
                outputs=[
                    state,
                    catalog_gallery,
                    *catalog_slots,
                    catalog_prev_btn,
                    catalog_next_btn,
                    catalog_page_status,
                    detail_panel,
                    detail_image,
                    detail_report,
                    status,
                ],
            )
            down_btn.click(
                lambda current_state: submit_feedback_with_catalog_slots(
                    feedback_handler,
                    current_state,
                    FeedbackAction.REJECT.value,
                ),
                inputs=[state],
                outputs=[
                    state,
                    catalog_gallery,
                    *catalog_slots,
                    catalog_prev_btn,
                    catalog_next_btn,
                    catalog_page_status,
                    detail_panel,
                    detail_image,
                    detail_report,
                    status,
                ],
            )
            trash_btn.click(
                lambda current_state: submit_feedback_with_catalog_slots(
                    feedback_handler,
                    current_state,
                    FeedbackAction.SHRED.value,
                ),
                inputs=[state],
                outputs=[
                    state,
                    catalog_gallery,
                    *catalog_slots,
                    catalog_prev_btn,
                    catalog_next_btn,
                    catalog_page_status,
                    detail_panel,
                    detail_image,
                    detail_report,
                    status,
                ],
            )
    return demo


def launch(
    *,
    server_name: str | None = None,
    server_port: int | None = None,
    share: bool = False,
    mode: GradioMode | str | None = None,
) -> None:
    resolved_mode = _resolve_gradio_mode(mode)
    _prewarm_runtime_startup_if_enabled(resolved_mode)
    demo = build_demo(mode=resolved_mode)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        css=CSS,
        theme=_build_theme(),
        allowed_paths=[str(RUNTIME_RUN_ROOT.resolve())],
    )


if __name__ == "__main__":
    launch()
