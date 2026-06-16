**Graph-First Cinematic Prompt Schema**  
**Specification for Prompt Extraction, Canonicalization, Verification, and Repair**  
Status: design specification  
   
 Audience: prompt pipeline implementers  
   
 Primary goal: transform one user-sourced image or scene-in-motion prompt into a semantically owned, cinema/film/photography-oriented generation prompt without requiring the image or video model to understand highly granular internal fields directly.  
The core design is:  
1. Use a primary prompt-LLM to extract the complete PromptDocument: elements, raw-first relations, lanes, action targets, and evidence.  
2. Use field-scoped canonicalization to attach optional locked enum matches. The default canonicalizer is embedding-first and may call a prompt-LLM fallback only for ambiguous or explicitly configured fields.  
3. Use a prompt-LLM verifier to check graph linkage, lane ownership, enum fit, prompt-faithfulness, and reconstruction risk.  
The JSON schema is an internal control surface. The final image model receives a synthesized prompt, not a raw field dump.  
Runtime generation, evaluation, and learning-loop actions are specified in `bruteforce-canvas_Orchestration.md`; this prompt schema owns only the prompt deconstruction, canonicalization, verification, and rendering contract.  
Build-order position: consume this spec first, then `bruteforce-canvas_LHS_enum_router.md`, then `bruteforce-canvas_fast_image_generation.md`, then `bruteforce-canvas_Evaluator_pipeline.md`, with `bruteforce-canvas_Orchestration.md` as the runtime authority for shared lifecycle, threshold, feedback, and persistence policy.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OMQ0AIAwAwZKQ6kBqjSAOJywYYCIkd9OP36pqRMQMAAB+sfqJfLoBAMCN3NYoAzBA+QG0AAAAAElFTkSuQmCC)  
**1. Design Principles**  
**1.1 Graph Before Details**  
Elements and relations are the semantic skeleton. They decide what exists and who owns, wears, holds, performs, contains, touches, or spatially relates to what. Relations preserve a raw linkage phrase first; locked relation enums are optional affordances, not the only allowed relation language.  
The graph must be valid before object appearance, action, cinematography, and constraints are expanded. This prevents appearance descriptors from being attached to the wrong subject.  
The graph is not required to contain an actor/action/object triplet. Many valid prompts describe still life, geography, architecture, weather, abstract visuals, or product scenes with no agent and no action. The graph only needs the visible or strongly implied elements and the relations that preserve semantic ownership.  
Bad:  
woman: tailored wool coat material; deep charcoal color; matte finish  
   
Good:  
person_01: woman  
 garment_01: coat  
 relation: person_01 -> relation_raw="wearing" -> garment_01; optional enum=WEARING  
 garment_01 descriptor: wool material; deep charcoal color; matte finish  
   
**1.2 Lanes Own Narrow Responsibilities**  
Each lane is allowed to express only one class of information.  
| | | |  
|-|-|-|  
| **Lane** | **Owns** | **Must Not Own** |   
| Elements | identity, entity type, importance | color, material, finish, condition, action, camera |   
| Relations | semantic ownership, contact, containment, spatial links, visible source-to-target links, raw linkage phrases | appearance, lighting treatment prose, camera language |   
| Objects | appearance of a specific element | ownership, action, camera, negative constraints |   
| Actions | dynamic behavior by an actor | persistent wearing/holding, material, color |   
| Cinematography | camera, lens, camera movement, focus behavior, lighting, framing, color treatment, scene atmosphere | subject identity, object ownership |   
| Constraints | exclusions and guardrails | positive visual content |   
   
**1.3 Verification Is Prompt-LLM-Gated**  
Verification is performed by a prompt-LLM verifier after extraction and canonicalization. The verifier checks graph linkage, lane ownership, evidence support, enum fit, unresolved slots, and reconstruction risk. It must return structured issues with a repair scope, but it must not silently rewrite the document.  
**1.4 Retry Is Slice-Scoped**  
If the object lane assigns wool to person_01 instead of garment_01, only that object descriptor slice is repaired. The action lane and cinematography lane do not need to rerun.  
If constraints conflict with the scene graph, only constraints are repaired.  
If a relation points to a missing element, the relation or element slice is repaired before canonicalization continues.  
**1.5 No Hidden Normalization Service**  
This spec does not require a separate semantic remapping service. Normalization is handled through:  
1. Clear primary prompt-LLM extraction instructions.  
2. Embedding-first field-scoped enum canonicalization.  
3. Canonicalization fallback for ambiguous or configured fields.  
4. Prompt-LLM verification over the assembled PromptDocument.  
5. Narrow prompt-LLM repair calls that return corrected JSON slices.  
**1.6 Raw Strings First, Enum Affordances Second**  
LLM-facing schemas should preserve user intent as raw strings. Enums are internal affordances used only after a separate normalization step determines that a raw phrase clearly maps to one of the known enum values.  
  
Do not build large rule-based alias libraries for prompt concepts such as movement, style, material, camera language, relation linkage, or action support. Those libraries can become user-opinionated and can grow without bound. Instead, use field-scoped embedding retrieval with a prompt-LLM fallback to produce an optional enum match with a confidence label and reason.  
  
The normalization rule is:  
- raw string is always preserved.  
- enum match is optional.  
- enum match may be used only when semantically clear.  
- unclear matches remain raw strings and cannot trigger strong behavior such as inferred graph support.  
- the canonicalizer must not rewrite the user's intent or invent missing graph participants.  
  
For relations, the preserved raw value is the linkage itself. A relation may be `(source_id, relation_raw, target_id)` with no canonical enum. If the linkage is reusable but not represented in the enum registry, the relation may carry a proposed enum candidate for the enum candidate queue, such as `relation_raw="leaning against"` with proposed enum `LEANING_AGAINST`.  
**1.7 Scene Frames, Not Universal Action Triplets**  
The graph should represent a prompt as typed scene frames: compact semantic links among elements. A frame may be static, compositional, ownership-based, environmental, or motion-bearing. Only motion-bearing frames need action-lane interpretation.  
  
Common frame shapes include:  
- subject wearing garment.  
- object on_top_of support surface.  
- fog over lake.  
- visible lamp shining on desk.  
- room with warm window light as cinematography.  
- vehicle parked_on street.  
- mountain under sky.  
- person performing action with target.  
- abstract visual occupying space or field.  
  
This prevents the parser from forcing agent/action/object structure onto prompts such as "a snow-covered mountain at sunrise", "a red ceramic vase on a wooden table", or "abstract blue smoke in darkness". For these prompts, the action lane can be empty and the graph is still complete.  
  
**1.8 Evidence-Gated Inference and Unresolved Slots**  
Graph facts must be either explicit in the user prompt or tightly entailed by a low-ambiguity phrase. Plausible defaults are not graph facts.  
  
Every non-explicit graph addition must carry an evidence category:  
- explicit: directly named or directly described by the user.  
- entailed: required by a specific phrase with low ambiguity, such as "swordfighting" implying a sword or weapon-class participant.  
- unresolved: a required role exists syntactically, but its content is unspecified, such as "throwing something".  
- blocked: the prompt cannot be compiled faithfully without clarification or a safe downgrade.  
  
Unresolved slots may be represented in the graph to preserve structure, but they are not normal visual elements. Secondary lanes must not add appearance, camera emphasis, constraints, or concrete identity to unresolved slots. The renderer may suppress the frame, emit neutral motion wording, or produce prompt-improvement feedback, but it must not convert an unresolved slot into an invented prop.  
  
The validator must reject invented graph facts when no evidence span supports them. For example, "person throwing something" may preserve an unresolved thrown-object slot or compile as "throwing pose with no visible thrown object"; it must not silently add "baseball", "rock", or "paper".  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsScYxpg/i2XMYARvRrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA22YBcnkstSpAAAAAElFTkSuQmCC)  
**2. Pipeline Overview**  
The production path is prompt-LLM-based for scene understanding and field-scoped for enum normalization: primary prompt-LLM extraction, embedding-first canonicalization with prompt-LLM fallback, and prompt-LLM verification. There is no rule-based prompt extraction fallback. Code must not discover scene relations such as `fog around wall`, `bowl on table`, or `chair leaning against wall`; those are primary prompt-LLM extraction responsibilities.  
  
User Prompt  
   |  
   v  
 Stage 1: Primary Prompt-LLM Extraction  
   -> PromptDocument graph and raw lane values  
   -> owns element discovery, raw relation discovery, action targets, lane ownership, evidence spans  
   |  
   v  
  Stage 2: Field-Scoped Canonicalization
   -> embedding-first enum matching per locked/standardized field
   -> embeddings for enum matching come from BAAI/bge-small-en-v1.5
   -> prompt-LLM fallback runs only for ambiguous, unavailable, or explicitly configured fields
   -> graph relation_raw values are canonicalized the same way as lane fields
   -> each fallback call receives only the raw field value plus that field's enum RAG context
   -> raw strings preserved, optional enum matches attached
   |  
   v  
 Stage 3: Prompt-LLM Verification  
   -> verifies graph linkage, lane ownership, canonical enum fit, unresolved slots, and reconstruction risk  
   -> returns structured issues or an approval report  
   
The graph is the dependency root for semantic ownership. All later lanes are generated from the raw prompt plus the frozen graph and are forbidden to invent new element IDs unless a separate graph patch flow is explicitly added and validated.  
  
Stage exhibition expected from implementation:  
- `llm_extraction`: primary prompt-LLM returned graph elements, raw relations, lanes, action targets, and evidence.  
- `canonicalization`: embedding-first canonicalization attached enum metadata, with prompt-LLM fallback calls only where configured or needed.  
- `llm_verification`: prompt-LLM verifier checked graph linkage, lane ownership, enum fit, prompt-faithfulness, and reconstruction risk.  

**2.1 Prompt-LLM Server Adapter Assignment**
Prompt extraction, slice repair, verification, and canonicalization fallback share one prompt-LLM service assignment by default.

The service may be a local model adapter or an OpenAI-compatible server endpoint. When a server endpoint is used, it is configured through the prompt-LLM settings:
- `BC_LLM_PROVIDER=openai-compatible-server`.
- `BC_LLM_BASE_URL`.
- `BC_LLM_MODEL`.
- `BC_LLM_API_KEY` when authentication is required.

One configured prompt-LLM model targets all prompt-LLM roles unless a future prompt spec defines role-specific overrides. The role is selected by the schema and system instruction sent by the adapter, not by separate endpoint defaults.

The prompt-LLM server adapter is not an evaluator adapter, generator adapter, IQA adapter, VLM adapter, or metacognitive-impact adapter. It must not be reused to route arbitrary model services outside the prompt pipeline.
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhwgJWEPcbJpnRgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseaIkEMIPgIvAAAAAASUVORK5CYII=)  
**3. Stage Responsibilities**  
**3.1 Stage 1: Primary Prompt-LLM Extraction**  
Input:  
One raw user prompt.  
   
Output:  
PromptDocument draft with graph, raw relations, lanes, evidence, and unresolved slots.  
   
Responsibilities:  
- Create stable element IDs.  
- Use noun-only labels.  
- Separate clothing, accessories, props, locations, and visible entities into distinct elements.  
- Create explicit graph relations with source ID, target ID, and relation_raw. Add a canonical relation enum only when the linkage clearly maps to a locked enum, otherwise preserve `(None, relation_raw)`.  
- Represent static, spatial, environmental, ownership, and motion-support frames without forcing every prompt into an actor/action/object structure.  
- Mark each graph element or relation as explicit, entailed, unresolved, or blocked.  
- Preserve a short seed prompt anchor for user intent.  
- Exclude camera, lighting treatment, lens, and shot-language fields from the graph payload.  
- Populate object, action, cinematography, and constraint lanes from the same prompt pass. Lanes must reference extracted element IDs and preserve target phrases instead of flattening them.  
Rules:  
- label must identify the thing, not describe its appearance.  
- wool coat becomes label="coat" plus later material="wool".  
- burgundy handbag becomes label="handbag" plus later color="burgundy".  
- woman holding handbag becomes a holding relation, not an action unless dynamic motion is implied.  
- woman wearing coat becomes a wearing relation, not an object descriptor on the woman.  
- snow-covered mountain becomes element mountain plus a snow element or descriptor, with an on_top_of or covering relation if snow should be independently preserved.  
- red ceramic vase on a wooden table becomes vase on_top_of table; red and ceramic belong to the vase descriptor, wooden belongs to the table descriptor.  
- fog over lake becomes fog and lake with an over or spatial relation; no actor is required.  
- warm window light, golden hour, soft sun, practical lighting, and blue hour belong to cinematography unless a visible light-emitting object is part of the scene.  
- desk lamp shining on notebook may create a lamp element and a raw relation such as `shining on`, because the lamp is a visible object.  
- leaning against, draped over, wedged between, resting against, or other reusable linkages may remain raw-only relations with proposed enum candidates. Do not force them into the closest bad enum.  
- throwing something may create an unresolved object slot, but not a concrete prop.  
**3.2 Stage 2: Field-Scoped Canonicalization With Prompt-LLM Fallback**  
Each locked or standardized raw field is canonicalized independently because each decision is field-scoped, not scene-understanding. The default path uses BAAI/bge-small-en-v1.5 embeddings to compare the raw field value against that field's enum context. A prompt-LLM fallback may run when the embedding match is ambiguous, unavailable, below confidence threshold, or explicitly configured for that field. Fallback calls may run in parallel because each call receives only field-local context.   
  
Each embedding match or fallback call receives:  
- The raw field value.  
- The field name and expected semantic role.  
- Context containing only that field's locked enum values, aliases, and short definitions.  
- The local evidence span when useful.  
  
The canonicalizer returns an optional enum match, confidence label, and reason. It must preserve the original raw string. If no enum is a good fit, the result stays raw-only as `(None, raw_string)`; reusable field-specific phrases may be emitted as `proposed_new_enum` metadata for the enum candidate queue. Canonicalization results are attached back to the PromptDocument as field metadata so LHS and renderers can consume enum affordances without reparsing the prompt.  
  
Canonicalized fields include relation_raw values, action labels, material, color, finish, lighting, shot size, angle, camera movement, lens, focus, style, quality, and constraint guards when the field has a locked vocabulary.  
  
**3.3 Stage 3: Prompt-LLM Verification**  
The verifier receives the raw prompt, the full extracted PromptDocument, and canonicalization results. It does not rediscover the scene with code. It checks whether the prompt-LLM extraction and enum choices are faithful, connected, and renderable.  
  
Verification checks:  
- Element IDs are unique.  
- Relation IDs are unique.  
- Every relation source and target resolves to an element.  
- Canonical relation types fit the source and target entity types when a canonical enum is present. Raw-only relations must carry relation_raw and evidence.  
- At least one primary or foreground element exists unless the prompt is intentionally abstract or environmental.  
- Element labels are noun-like and do not contain leaked material, color, finish, condition, or action terms.  
- The graph preserves the user prompt's major entities, including static elements with no action.  
- Lane descriptors attach to the right element instead of leaking clothing, props, or setting details onto the subject.  
- Target-bearing action phrases preserve their targets when rendered, such as "walking toward glass door" or "reaching toward ceramic bowl".  
- Cinematography has no subject identity leakage and preserves relevant shot, focus, movement, angle, and lighting language.  
- Lighting treatment is not a graph relation by default. The verifier flags ordinary lighting phrases that were promoted into graph elements without a visible light-emitting object.  
- Constraints do not ban required graph elements or required visual properties.  
- Unresolved slots are not treated as required visible elements unless the renderer can phrase them without inventing identity.  
  
Failure examples:  
| | |  
|-|-|  
| **Issue** | **Repair Scope** |   
| relation.target_id="bag_01" but no bag_01 element exists | extraction repair |   
| label="burgundy leather handbag" | element label repair |   
| person_01 wearing location_01 | relation repair |   
| relation_raw="leaning against" forced into next_to | canonicalization repair |   
| user mentioned handbag but graph omitted it | extraction repair |   
| person throwing something becomes person throwing baseball | extraction inference repair |   
| mountain prompt invents a hiker actor | extraction inference repair |   
  
**3.3.1 Graph/Action Overlap Policy**  
Relations and actions are allowed to overlap when the overlap makes the final prompt more semantically grounded. The prompt-LLM verifier must classify each action into one of four support states:  
- supported: the graph already contains the required participants and relation state for the action.  
- inferred: the action strongly entails a missing participant or relation, so the pipeline may request a bounded graph inference patch.  
- unresolved: the action has a syntactic missing component represented by an unresolved graph slot, so the system may compile neutral wording or return prompt-improvement feedback.  
- indeterminate: the action has no grounded relation and cannot be compiled faithfully without invention, so it should be suppressed or downgraded.  
  
Graph relations are state. They describe where entities are, how they are oriented, what they touch, what they own or wear, what contains them, and what supports the action. Actions are motion. They describe what changes, what is performed, or what movement is implied.  
  
Inferred support must never let the action lane mutate the graph directly. It must create a graph inference patch request with provenance, validate the patch, and re-freeze the graph before merge. If the graph changes, only affected lane slices retry or reconcile.  
  
Examples:  
- A graph containing person_01 holding sword_01 can support an action such as person_01 swordfighting. The holding relation is an action precondition and the renderer may suppress a separate "holding a sword" phrase if swordfighting already carries that visual fact clearly.  
- If the user prompt says swordfighting but the graph omitted the sword, the action may be inferred because swordfighting strongly entails a sword or equivalent weapon. The pipeline may request an inferred sword element plus holding or using relation, then validate and re-freeze the graph.  
- If the action is throwing something and no thrown object is specified, the action is unresolved or indeterminate. The pipeline must not invent the object. It may preserve an unresolved slot, suppress the action, downgrade to "throwing pose with no visible thrown object", or return prompt-improvement feedback.  
  
For image prompts, supported or inferred actions render as pose, gesture, tension, orientation, implied motion, or target-preserving action phrases. For video prompts, they render as temporal motion behavior or target-preserving action phrases without adding a storyboard or timeline. If the raw prompt says "walking toward a glass door" or "reaching toward a ceramic bowl", the renderer should preserve that target phrase rather than flattening it into generic "walking motion" or "reaching pose". This policy does not introduce storyboard modeling, timeline modeling, physics validation, biomechanics validation, arbitrary generation of missing props, or a broad semantic remapping service.  
**3.3.2 Prompt-Improvement Feedback**  
When an unresolved or indeterminate frame would materially affect the visual output, the pipeline may return prompt-improvement feedback rather than forcing a graph patch. This feedback should preserve user authorship and offer precise rewrites.  
  
Example:  
Input: person throwing something in a studio portrait  
Feedback: Specify what the person is throwing, or rephrase as "a person in a throwing pose, no thrown object visible" if the object should not appear.  
  
This feedback is optional for non-blocking ambiguity and required for blocking ambiguity. Batch generation may continue with the safest non-inventive downgrade only if the trace records the unresolved component and the rendered prompt does not invent it.  
**3.4 Post-Verification Rendering**  
Rendering is not a prompt-understanding stage and must not infer missing scene facts. It converts the verified PromptDocument into natural prompt language while preserving graph links, lane ownership, and verifier decisions.  
  
Rendering order:  
1. Primary subject phrase.  
2. Owned/worn/held/attached object phrases.  
3. Supporting elements and spatial relations.  
4. Dynamic actions.  
5. Setting and environment.  
6. Cinematography and lighting.  
7. Style and quality modifiers.  
8. Negative prompt.  
9. Alignment checklist and trace.  
The renderer uses field names semantically. It does not emit raw enum dumps.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OUQmAABBAsSdYxKbXxlpGEAOIFfwTYUuwZWa2ag8AgL841uquzq8nAAC8dj05VAYO3phhoQAAAABJRU5ErkJggg==)  
**4. Pydantic Base Models**  
The following models are intentionally compact enough for LLM JSON output while still enforcing semantic ownership.  
from __future__ import annotations  
   
 from enum import Enum  
 from typing import Annotated, Literal  
   
 from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator  
   
   
 ElementId = Annotated[str, StringConstraints(pattern=r"^[a-z]+_[0-9]{2}$")]  
 RelationId = Annotated[str, StringConstraints(pattern=r"^rel_[0-9]{2}$")]  
 ShortText = Annotated[str, StringConstraints(min_length=1, max_length=80)]  
 ResidualDescription = Annotated[str, StringConstraints(min_length=1, max_length=180)]  
 SeedPrompt = Annotated[str, StringConstraints(min_length=1, max_length=500)]  
   
   
 class StrictModel(BaseModel):  
     model_config = ConfigDict(extra="forbid", frozen=True, strict=True)  
   
   
 class EntityType(str, Enum):  
     PERSON = "person"  
     ANIMAL = "animal"  
     TEXTILE = "textile"  
     ACCESSORY = "accessory"  
     TOOL = "tool"  
     PRODUCT = "product"  
     VEHICLE = "vehicle"  
     ARCHITECTURE = "architecture"  
     LOCATION = "location"  
     ENVIRONMENT = "environment"  
     LIGHT_SOURCE = "light_source"  
     ABSTRACT_VISUAL = "abstract_visual"  
     UNKNOWN_SLOT = "unknown_slot"  
     SURFACE = "surface"  
     CONTAINER = "container"  
     FURNITURE = "furniture"  
   
   
 class ElementRole(str, Enum):  
     PRIMARY_SUBJECT = "primary_subject"  
     FOREGROUND = "foreground"  
     SUPPORTING = "supporting"  
     BACKGROUND = "background"  
     CONTEXT = "context"  
   
   
 class RelationType(str, Enum):  
     WEARING = "wearing"  
     HOLDING = "holding"  
     CARRYING = "carrying"  
     USING = "using"  
     ATTACHED_TO = "attached_to"  
     PART_OF = "part_of"  
     INSIDE = "inside"  
     ON_TOP_OF = "on_top_of"  
     UNDER = "under"  
     OVER = "over"  
     COVERING = "covering"  
     NEXT_TO = "next_to"  
     IN_FRONT_OF = "in_front_of"  
     BEHIND = "behind"  
     LOOKING_AT = "looking_at"  
     ILLUMINATING = "illuminating"  
     PARKED_ON = "parked_on"  
   
   
 class Importance(str, Enum):  
     REQUIRED = "required"  
     OPTIONAL = "optional"  
     AMBIENT = "ambient"  
     UNRESOLVED = "unresolved"  
   
   
 class EvidenceCategory(str, Enum):  
     EXPLICIT = "explicit"  
     ENTAILED = "entailed"  
     UNRESOLVED = "unresolved"  
     BLOCKED = "blocked"  
   
   
 class EvidenceSpan(StrictModel):  
     text: ShortText  
     category: EvidenceCategory = EvidenceCategory.EXPLICIT  
     reason: ResidualDescription | None = None  
   
   
class CanonicalStatus(str, Enum):  
     MATCHED_ACTIVE = "matched_active"  
     MATCHED_SUPPRESSED = "matched_suppressed"  
     MATCHED_DIAGNOSTIC_HOLD = "matched_diagnostic_hold"  
     UNMATCHED_RAW_ONLY = "unmatched_raw_only"  
     PROPOSED_NEW_ENUM = "proposed_new_enum"  
     REJECTED_INVALID = "rejected_invalid"  
   
   
 class RelationEnumMatch(StrictModel):  
     raw: ShortText = Field(description="Raw linkage phrase from the prompt, such as resting on, leaning against, or draped over.")  
     enum_value: RelationType | None = Field(  
         default=None,  
         description="Optional locked relation enum. Null means preserve raw linkage and do not force a bad enum.",  
     )  
     status: CanonicalStatus  
     confidence: ShortText = Field(description="clear, weak, unmatched, or rejected")  
     reason: ResidualDescription | None = None  
   
   
 class ProposedRelationEnum(StrictModel):  
     code: ShortText = Field(description="UPPER_SNAKE_CASE candidate for the enum registry candidate queue.")  
     label: ShortText  
     aliases: list[ShortText] = Field(default_factory=list, max_length=12)  
     description: ResidualDescription  
     source_type_examples: list[EntityType] = Field(default_factory=list, max_length=8)  
     target_type_examples: list[EntityType] = Field(default_factory=list, max_length=8)  
   
   
 class Element(StrictModel):  
     id: ElementId  
     entity_type: EntityType  
     label: ShortText = Field(description="Identity-only noun phrase, such as woman, coat, handbag, alley.")  
     role: ElementRole  
     importance: Importance = Importance.REQUIRED  
     evidence: EvidenceSpan | None = Field(  
         default=None,  
         description="Prompt evidence for this element. Required for entailed, unresolved, and blocked graph facts.",  
     )  
   
   
 class RelationDescriptor(StrictModel):  
     id: RelationId  
     source_id: ElementId  
     target_id: ElementId  
     relation_raw: ShortText = Field(  
         description="Raw linkage text lifted from the prompt. This is required even when a canonical enum is available.",  
     )  
     relation_match: RelationEnumMatch | None = Field(  
         default=None,  
         description="Optional canonical relation enum match. Leave null when the raw linkage should be preserved without forcing an enum.",  
     )  
     proposed_enum: ProposedRelationEnum | None = Field(  
         default=None,  
         description="Set only when relation_raw is reusable and should enter the enum candidate queue.",  
     )  
     importance: Importance = Importance.REQUIRED  
     evidence: EvidenceSpan | None = Field(  
         default=None,  
         description="Prompt evidence for this relation. Required for entailed, unresolved, and blocked graph facts.",  
     )  
     description: ResidualDescription | None = Field(  
         default=None,  
         description="Only relation-specific nuance, not appearance or cinematography.",  
     )  
   
   
 class SceneGraphDraft(StrictModel):  
     seed_prompt: SeedPrompt  
     elements: list[Element] = Field(min_length=1, max_length=24)  
     relations: list[RelationDescriptor] = Field(default_factory=list, max_length=40)  
   
     @model_validator(mode="after")  
     def validate_graph(self) -> "SceneGraphDraft":  
         element_ids = [element.id for element in self.elements]  
         if len(element_ids) != len(set(element_ids)):  
             raise ValueError("element IDs must be unique")  
   
         relation_ids = [relation.id for relation in self.relations]  
         if len(relation_ids) != len(set(relation_ids)):  
             raise ValueError("relation IDs must be unique")  
   
         known = set(element_ids)  
         for relation in self.relations:  
             if relation.source_id not in known:  
                 raise ValueError(f"relation {relation.id} has unknown source_id {relation.source_id}")  
             if relation.target_id not in known:  
                 raise ValueError(f"relation {relation.id} has unknown target_id {relation.target_id}")  
             if relation.source_id == relation.target_id:  
                 raise ValueError(f"relation {relation.id} cannot point to itself")  
             if not relation.relation_raw.strip():  
                 raise ValueError(f"relation {relation.id} must preserve relation_raw")  
             if relation.relation_match and relation.relation_match.enum_value is None and relation.relation_match.status == CanonicalStatus.MATCHED_ACTIVE:  
                 raise ValueError(f"relation {relation.id} cannot be matched with null enum_value")  
             if relation.proposed_enum and relation.relation_match and relation.relation_match.status not in {CanonicalStatus.PROPOSED_NEW_ENUM, CanonicalStatus.UNMATCHED_RAW_ONLY}:  
                 raise ValueError(f"relation {relation.id} proposed_enum is only valid for proposed or unmatched relation matches")  
   
         if not any(element.importance != Importance.UNRESOLVED for element in self.elements):  
             raise ValueError("at least one resolved visible element is required")  
   
         return self  
   
**4.1 Object Lane**  
The object lane describes the visible appearance of specific elements. It does not create relations.  
class Finish(str, Enum):  
     MATTE = "matte"  
     SATIN = "satin"  
     GLOSSY = "glossy"  
     POLISHED = "polished"  
     WEATHERED = "weathered"  
     NATURAL = "natural"  
   
   
 class Condition(str, Enum):  
     PRISTINE = "pristine"  
     NEW = "new"  
     USED = "used"  
     WORN = "worn"  
     AGED = "aged"  
     DISTRESSED = "distressed"  
   
   
 class Pattern(str, Enum):  
     SOLID = "solid"  
     GEOMETRIC = "geometric"  
     STRIPED = "striped"  
     FLORAL = "floral"  
     TEXTURED = "textured"  
     NONE = "none"  
   
   
 class ObjectDescriptor(StrictModel):  
     target_id: ElementId  
     description: ResidualDescription | None = Field(  
         default=None,  
         description=(  
             "Residual form or construction detail not already captured by typed fields. "  
             "Do not restate material, color, finish, condition, or pattern."  
         ),  
     )  
     material: ShortText | None = None  
     color: ShortText | None = None  
     finish: Finish | None = None  
     condition: Condition | None = None  
     pattern: Pattern | None = None  
   
   
 class ObjectLane(StrictModel):  
     objects: list[ObjectDescriptor] = Field(default_factory=list, max_length=32)  
   
**4.2 Action Lane**  
The action lane describes temporal, fluid, dynamic, or implied-motion behavior. Stable spatial, orientational, contact, ownership, containment, and support state belongs in relations. `movement_raw` is free text and remains the render source. `MovementType` is a coarse visual motion class used for validation, analytics, and LHS grouping; it is not a controlled verb vocabulary.  
class MovementType(str, Enum):  
     POSTURE_STANCE = "posture_stance"  
     LOCOMOTION = "locomotion"  
     MANUAL_INTERACTION = "manual_interaction"  
     OBJECT_MANIPULATION = "object_manipulation"  
     GESTURAL = "gestural"  
     GAZE_ATTENTION = "gaze_attention"  
     FACIAL_EXPRESSION = "facial_expression"  
     PERFORMANCE = "performance"  
     COMBAT = "combat"  
     BODY_ORIENTATION = "body_orientation"  
   
   
 class MotionIntensity(str, Enum):  
     STILL = "still"  
     SUBTLE = "subtle"  
     MODERATE = "moderate"  
     ENERGETIC = "energetic"  
   
   
 class EnumMatchConfidence(str, Enum):  
     CLEAR = "clear"  
     UNCLEAR = "unclear"  
   
   
 class EnumMatch(StrictModel):  
     raw: ShortText  
     enum_value: ShortText | None = None  
     confidence: EnumMatchConfidence  
     reason: ResidualDescription  
   
   
 class ActionSupportStatus(str, Enum):  
     SUPPORTED = "supported"  
     INFERRED = "inferred"  
     UNRESOLVED = "unresolved"  
     INDETERMINATE = "indeterminate"  
   
   
 class InferredGraphSupport(StrictModel):  
     reason: ShortText  
     implied_element_label: ShortText | None = None  
     implied_relation_type: RelationType | None = None  
     source_action: ShortText  
     evidence: EvidenceSpan  
   
   
 class PromptImprovementHint(StrictModel):  
     issue: ShortText  
     suggested_rewrites: list[ShortText] = Field(default_factory=list, max_length=4)  
     safe_downgrade: ShortText | None = None  
   
   
 class ActionDescriptor(StrictModel):  
     actor_id: ElementId  
     movement_raw: ShortText  
     movement_match: EnumMatch | None = Field(  
         default=None,  
         description="Optional tiny-LLM match to a motion class. Preserve and render movement_raw even when matched.",  
     )  
     target_id: ElementId | None = None  
     intensity: MotionIntensity = MotionIntensity.SUBTLE  
     support_status: ActionSupportStatus = ActionSupportStatus.SUPPORTED  
     required_relation_types: list[RelationType] = Field(default_factory=list, max_length=8)  
     inferred_support: InferredGraphSupport | None = None  
     prompt_improvement: PromptImprovementHint | None = None  
     description: ResidualDescription | None = Field(  
         default=None,  
         description="Action nuance only. Do not describe clothing, props, material, or camera.",  
     )  
   
   
 class ActionLane(StrictModel):  
     actions: list[ActionDescriptor] = Field(default_factory=list, max_length=16)  
   
**4.2.1 Embedding-First Enum Affordance Matching**  
Enum matching is a secondary affordance, not the source of truth. A primary prompt-LLM performs graph parsing and lane expansion because those tasks require global semantic ownership decisions. Field-level enum matching is narrow: given one raw phrase and a bounded enum list, return an enum only when the match is semantically clear.  
  
The intended implementation is not deterministic alias parsing. The primary prompt-LLM returns a graph/lane document with raw strings preserved. Then every field with known locked or standardized terms is canonicalized independently. The default path uses BAAI/bge-small-en-v1.5 embeddings; a prompt-LLM fallback may run only when the embedding match is ambiguous, unavailable, or explicitly configured. Fallback calls may run in parallel because each call receives only:  
  
- the raw value for one field, such as `"medium close shot"`;  
- the lane and field name, such as `cinematography.shot_size`;  
- the locked enum candidates for that exact field;  
- rules forbidding graph edits, new objects, new relations, support-status changes, or raw text rewrites.  
  
The canonicalizer must choose only from the supplied field-scoped enum list. If no supplied enum fits but the concept is reusable for that exact field, it may return `proposed_new_enum`. Otherwise it returns `unmatched_raw_only` and the raw string remains the source of truth.  
  
Use an Instructor-style Annotated validator when a single field can be judged in isolation. The raw field still returns a string; the enum match is recorded separately so prompted intent is not overwritten:  
from typing import Annotated  
from pydantic import AfterValidator, field_validator  
   
 def preserve_raw_phrase(value: str) -> str:  
     return value.strip()  
   
 def canonical_enum_match(raw: str, enum_name: str, enum_values: list[str]) -> EnumMatch:  
     """Return {raw, enum_value, confidence, reason} from embedding match or prompt-LLM fallback."""  
     ...  
   
 MovementRaw = Annotated[str, AfterValidator(preserve_raw_phrase)]  
   
 class ActionDescriptor(StrictModel):  
     movement_raw: MovementRaw  
     movement_match: EnumMatch | None = None  
   
     @field_validator("movement_match", mode="before")  
     @classmethod  
     def attach_field_level_match(cls, value, info):  
         if value is not None:  
             return value  
         raw = info.data.get("movement_raw")  
         if raw is None:  
             return None  
         return canonical_enum_match(  
             raw=raw,  
             enum_name="MovementType",  
             enum_values=[item.value for item in MovementType],  
         )  
   
Use a model_validator when the enum match depends on graph or lane context, such as actor, target, support status, relations, or raw prompt evidence:  
class ActionDescriptor(StrictModel):  
     actor_id: ElementId  
     movement_raw: MovementRaw  
     movement_match: EnumMatch | None = None  
     target_id: ElementId | None = None  
     support_status: ActionSupportStatus = ActionSupportStatus.SUPPORTED  
   
     @model_validator(mode="after")  
     def attach_movement_match(self) -> "ActionDescriptor":  
         self.movement_match = canonical_enum_match(  
             raw=self.movement_raw,  
             enum_name="MovementType",  
             enum_values=[item.value for item in MovementType],  
         )  
         return self  
   
Canonicalizer output must be conservative:  
{  
   "raw": "swordfighting",  
   "enum_value": "combat",  
   "confidence": "clear",  
   "reason": "The phrase describes visually distinct combat motion. The raw verb remains swordfighting for rendering."  
 }  
   
For unclear or open-slot phrases, preserve the raw phrase and return no enum:  
{  
   "raw": "throwing something",  
   "enum_value": null,  
   "confidence": "unclear",  
   "reason": "The motion has an unspecified target and should not trigger inferred graph support by itself."  
 }  
   
Strong behaviors such as inferred graph support require a clear enum match or another explicitly validated semantic support decision. An unclear enum match must not invent elements, relations, props, or actions.  
   
**4.3 Cinematography Lane**  
The cinematography lane describes the shot, including lighting treatment. It must not introduce new subjects. Lighting phrases such as `warm window light`, `golden hour`, `soft sun`, `practical lighting`, and `blue hour` remain here when they describe the look of the scene. A lamp, neon sign, candle, screen, or other visible emitter may still be a graph element when the object itself matters.  
class ShotSize(str, Enum):  
     EXTREME_CLOSE_UP = "extreme_close_up"  
     BIG_CLOSE_UP = "big_close_up"  
     CLOSE_UP = "close_up"  
     MEDIUM_CLOSE_UP = "medium_close_up"  
     MEDIUM_CLOSE_SHOT = "medium_close_shot"  
     MEDIUM_SHOT = "medium_shot"  
     MEDIUM_LONG_SHOT = "medium_long_shot"  
     FULL_BODY = "full_body"  
     WIDE_SHOT = "wide_shot"  
     ESTABLISHING_SHOT = "establishing_shot"  
     TWO_SHOT = "two_shot"  
     THREE_SHOT = "three_shot"  
     GROUP_SHOT = "group_shot"  
     OVER_THE_SHOULDER_SHOT = "over_the_shoulder_shot"  
     POINT_OF_VIEW_SHOT = "point_of_view_shot"  
     INSERT_SHOT = "insert_shot"  
   
   
 class CameraAngle(str, Enum):  
     EYE_LEVEL = "eye_level"  
     LOW_ANGLE = "low_angle"  
     HIGH_ANGLE = "high_angle"  
     DUTCH_ANGLE = "dutch_angle"  
     OVER_THE_SHOULDER = "over_the_shoulder"  
     BIRDS_EYE = "birds_eye"  
     WORMS_EYE = "worms_eye"  
     CANTED = "canted"  
     PROFILE = "profile"  
     THREE_QUARTER = "three_quarter"  
     OVERHEAD_FLAT_LAY = "overhead_flat_lay"  
   
   
 class OpticCharacter(str, Enum):  
     NATURAL_35MM = "natural_35mm"  
     PORTRAIT_50MM = "portrait_50mm"  
     TELEPHOTO_COMPRESSION = "telephoto_compression"  
     WIDE_ANGLE = "wide_angle"  
     MACRO = "macro"  
     SHALLOW_FOCUS = "shallow_focus"  
     DEEP_FOCUS = "deep_focus"  
     ANAMORPHIC_CINEMATIC = "anamorphic_cinematic"  
     VINTAGE_SOFT = "vintage_soft"  
     TILT_SHIFT_SELECTIVE = "tilt_shift_selective"  
     FISHEYE_DISTORTED = "fisheye_distorted"  
     DREAM_GLOW = "dream_glow"  
   
   
 class LightingMood(str, Enum):  
     SOFT_NATURAL = "soft_natural"  
     GOLDEN_HOUR = "golden_hour"  
     LOW_KEY = "low_key"  
     HIGH_KEY = "high_key"  
     NEON_NOIR = "neon_noir"  
     STUDIO_SOFTBOX = "studio_softbox"  
     PRACTICAL_LIGHTING = "practical_lighting"  
     BLUE_HOUR_TWILIGHT = "blue_hour_twilight"  
     TUNGSTEN_INTERIOR = "tungsten_interior"  
     OVERCAST_SOFT = "overcast_soft"  
     HIGH_KEY_BRIGHT = "high_key_bright"  
     CHIAROSCURO_EXTREME = "chiaroscuro_extreme"  
     NEON_NIGHT = "neon_night"  
     RIM_SILHOUETTE = "rim_silhouette"  
     CANDLELIGHT_INTIMATE = "candlelight_intimate"  
   
   
 class ColorTreatment(str, Enum):  
     NATURAL_COLOR = "natural_color"  
     FILMIC_CONTRAST = "filmic_contrast"  
     MUTED_PALETTE = "muted_palette"  
    RICH_SATURATION = "rich_saturation"  
     MONOCHROME = "monochrome"  
     CINEMATIC_TEAL_ORANGE = "cinematic_teal_orange"  
     CROSS_PROCESSED = "cross_processed"  
     BLEACH_BYPASS = "bleach_bypass"  
     PASTEL_SOFT = "pastel_soft"  
     NEON_SATURATED = "neon_saturated"  
     EARTHY_ORGANIC = "earthy_organic"  
     MONOCHROMATIC_SEPIA = "monochromatic_sepia"  
     NOCTURNAL_BLUE = "nocturnal_blue"  
   
   
 class Framing(str, Enum):  
     CENTERED = "centered"  
     RULE_OF_THIRDS = "rule_of_thirds"  
     SYMMETRICAL = "symmetrical"  
     NEGATIVE_SPACE = "negative_space"  
     LAYERED_DEPTH = "layered_depth"  
     LEADING_LINES = "leading_lines"  
     FRAME_WITHIN_FRAME = "frame_within_frame"  
     DIAGONAL = "diagonal"  
     ASYMMETRICAL_BALANCE = "asymmetrical_balance"  
     S_CURVE = "s_curve"  
     GOLDEN_RATIO = "golden_ratio"  
     OFF_CENTER = "off_center"  
   
   
 class CinematographyLane(StrictModel):  
     shot_size: ShotSize | None = None  
     camera_angle: CameraAngle | None = None  
     optic_character: OpticCharacter | None = None  
     camera_motion: ShortText | None = Field(  
         default=None,  
         description="Camera movement such as locked-off, handheld, tracking, or push-in; no timeline or storyboard beats.",  
     )  
     focus_behavior: ShortText | None = Field(  
         default=None,  
         description="Focus/depth behavior such as shallow depth of field, deep focus, or rack focus.",  
     )  
     lighting_mood: LightingMood | None = None  
     color_treatment: ColorTreatment | None = None  
     framing: Framing | None = None  
     setting_description: ResidualDescription | None = Field(  
         default=None,  
         description="Environment and atmosphere only; do not invent subject identity or object ownership.",  
     )  
   
**4.4 Constraint Lane**  
The constraint lane describes exclusions and guardrails.  
class Guardrail(str, Enum):  
     NO_EXTRA_PEOPLE = "no_extra_people"  
     NO_TEXT = "no_text"  
     NO_LOGOS = "no_logos"  
     NO_DISTORTED_HANDS = "no_distorted_hands"  
     NO_EXTRA_LIMBS = "no_extra_limbs"  
     NO_BLUR = "no_blur"  
     NO_OVEREXPOSURE = "no_overexposure"  
     NO_UNDEREXPOSURE = "no_underexposure"  
   
   
 class ConstraintLane(StrictModel):  
     guardrails: list[Guardrail] = Field(default_factory=list, max_length=16)  
     negative_phrases: list[ShortText] = Field(default_factory=list, max_length=24)  
   
**4.5 Merged Prompt Document**  
class PromptDocument(StrictModel):  
     graph: SceneGraphDraft  
     object_lane: ObjectLane = Field(default_factory=ObjectLane)  
     action_lane: ActionLane = Field(default_factory=ActionLane)  
     cinematography_lane: CinematographyLane = Field(default_factory=CinematographyLane)  
     constraint_lane: ConstraintLane = Field(default_factory=ConstraintLane)  
   
     @model_validator(mode="after")  
     def validate_references(self) -> "PromptDocument":  
         known = {element.id for element in self.graph.elements}  
   
         for obj in self.object_lane.objects:  
             if obj.target_id not in known:  
                 raise ValueError(f"object descriptor has unknown target_id {obj.target_id}")  
   
         for action in self.action_lane.actions:  
             if action.actor_id not in known:  
                 raise ValueError(f"action has unknown actor_id {action.actor_id}")  
             if action.target_id is not None and action.target_id not in known:  
                 raise ValueError(f"action has unknown target_id {action.target_id}")  
   
         return self  
   
   
 class PromptBundle(StrictModel):  
     positive_prompt: str  
     negative_prompt: str  
     alignment_checklist: list[str]  
     render_trace: list[str]  
     prompt_improvement_hints: list[PromptImprovementHint] = Field(default_factory=list)  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsSfYxZo/jVEMYQLPJrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA4rLBc059ysnAAAAAElFTkSuQmCC)  
**5. Semantic Validators**  
Pydantic catches structural failures. Additional semantic validators catch ownership mistakes.  
**5.1 Label Ownership Verification**  
Element labels must be noun-like identity labels, but this cannot be enforced safely with a hardcoded color/material denylist. A phrase such as `black cat` or `white house` may be the entity name, while `burgundy leather handbag` is likely a descriptor dump that should split into label=`handbag` plus object-lane color/material fields.  
  
This check belongs to the prompt-LLM verifier or a scoped prompt-LLM repair call with prompt context. It should flag suspicious labels for repair, not silently rewrite the graph.  
  
Rejected when context indicates descriptor leakage:  
- wool coat -> label `coat`, material `wool`.  
- burgundy leather handbag -> label `handbag`, color `burgundy`, material `leather`.  
- running woman -> label `woman`, action lane `running`.  
  
Accepted when context indicates entity identity:  
- black cat.  
- white house.  
- red panda.  
**5.2 Relation Compatibility Validator**  
Relations should always have valid endpoint IDs and raw linkage text. Canonical relation compatibility is checked only when a locked relation enum has been matched. Raw-only relations remain valid graph edges, but they cannot satisfy enum-specific action preconditions until they are canonicalized or explicitly accepted by a semantic verifier.  
RELATION_COMPATIBILITY: dict[RelationType, set[tuple[EntityType, EntityType]]] = {  
     RelationType.WEARING: {  
         (EntityType.PERSON, EntityType.TEXTILE),  
         (EntityType.PERSON, EntityType.ACCESSORY),  
     },  
     RelationType.HOLDING: {  
         (EntityType.PERSON, EntityType.ACCESSORY),  
         (EntityType.PERSON, EntityType.PRODUCT),  
         (EntityType.PERSON, EntityType.TOOL),  
     },  
     RelationType.CARRYING: {  
         (EntityType.PERSON, EntityType.ACCESSORY),  
         (EntityType.PERSON, EntityType.PRODUCT),  
     },  
     RelationType.ATTACHED_TO: {  
         (EntityType.ACCESSORY, EntityType.TEXTILE),  
         (EntityType.PRODUCT, EntityType.VEHICLE),  
     },  
     RelationType.PART_OF: {  
         (EntityType.PRODUCT, EntityType.PRODUCT),  
         (EntityType.ARCHITECTURE, EntityType.LOCATION),  
     },  
 }  
   
   
 def validate_relation_compatibility(graph: SceneGraphDraft) -> list[str]:  
     elements = {element.id: element for element in graph.elements}  
     issues: list[str] = []  
   
     for relation in graph.relations:  
         relation_type = relation.relation_match.enum_value if relation.relation_match else None  
         if not relation.relation_raw:  
             issues.append(f"{relation.id} missing raw linkage text")  
             continue  
         if relation_type is None:  
             continue  
         allowed = RELATION_COMPATIBILITY.get(relation_type)  
         if allowed is None:  
             continue  
   
         source_type = elements[relation.source_id].entity_type  
         target_type = elements[relation.target_id].entity_type  
         if (source_type, target_type) not in allowed:  
             issues.append(  
                 f"{relation.id} invalid: {source_type.value} "  
                 f"{relation_type.value} {target_type.value}"  
             )  
   
     return issues  
   
**5.3 Object Ownership Validator**  
Appearance descriptors must attach to the thing they describe.  
Rules:  
- Clothing material belongs on the textile element.  
- Handbag material belongs on the accessory element.  
- Person descriptors may describe visible person features or pose-related residual appearance, but should not absorb clothing or accessory fields.  
- If a person has material="wool" and a wearing relation to a coat exists, that is invalid.  
def validate_object_ownership(document: PromptDocument) -> list[str]:  
     element_by_id = {element.id: element for element in document.graph.elements}  
     worn_targets = {  
         relation.target_id  
         for relation in document.graph.relations  
         if relation.relation_match and relation.relation_match.enum_value == RelationType.WEARING  
     }  
     person_sources = {  
         relation.source_id  
         for relation in document.graph.relations  
         if relation.relation_match and relation.relation_match.enum_value == RelationType.WEARING  
     }  
   
     issues: list[str] = []  
     for obj in document.object_lane.objects:  
         element = element_by_id[obj.target_id]  
         if element.entity_type == EntityType.UNKNOWN_SLOT or element.importance == Importance.UNRESOLVED:  
             issues.append(  
                 f"{obj.target_id} is unresolved; object lane must not assign concrete appearance"  
             )  
         if element.entity_type == EntityType.PERSON and obj.material and person_sources:  
             issues.append(  
                 f"{obj.target_id} has material={obj.material!r}; "  
                 "clothing material should attach to the worn garment element"  
             )  
         if obj.target_id in worn_targets and element.entity_type not in {  
             EntityType.TEXTILE,  
             EntityType.ACCESSORY,  
         }:  
             issues.append(f"{obj.target_id} is worn but is not a textile/accessory")  
     return issues  
   
**5.4 Action Support Validator**  
Action validation checks whether a dynamic action is grounded by graph state. It does not perform physics or biomechanics validation.  
Rules:  
- supported actions already have the required actor, target, and relation preconditions in the graph.  
- inferred actions have a strongly entailed missing participant or relation and must request a bounded graph inference patch.  
- unresolved actions preserve an open syntactic slot and must include prompt-improvement feedback or a safe downgrade.  
- indeterminate actions have no faithful grounded rendering and must be suppressed or downgraded.  
- The action lane must never create graph elements directly.  
   
def validate_action_support(document: PromptDocument) -> list[str]:  
     relations = document.graph.relations  
     known = {element.id for element in document.graph.elements}  
     issues: list[str] = []  
   
     for action in document.action_lane.actions:  
         if action.actor_id not in known:  
             issues.append(f"action actor {action.actor_id} is not in graph")  
             continue  
   
         if action.target_id is not None and action.target_id not in known:  
             issues.append(f"action target {action.target_id} is not in graph")  
   
         relation_types = {  
             relation.relation_match.enum_value  
             for relation in relations  
             if relation.relation_match and relation.relation_match.enum_value is not None  
             and relation.relation_match.status == CanonicalStatus.MATCHED_ACTIVE  
             and relation.source_id == action.actor_id  
             and (action.target_id is None or relation.target_id == action.target_id)  
         }  
         missing = [  
             relation_type  
             for relation_type in action.required_relation_types  
             if relation_type not in relation_types  
         ]  
   
         if action.support_status == ActionSupportStatus.SUPPORTED and missing:  
             issues.append(  
                 f"supported action {action.movement_raw} lacks required relation preconditions: {missing}"  
             )  
   
         if action.support_status == ActionSupportStatus.INFERRED and action.inferred_support is None:  
             issues.append("inferred action must include an inferred graph support request")  
   
         if action.support_status == ActionSupportStatus.UNRESOLVED:  
             if action.prompt_improvement is None:  
                 issues.append("unresolved action should include prompt-improvement feedback or a safe downgrade")  
             if action.target_id is not None:  
                 target = next(  
                     (element for element in document.graph.elements if element.id == action.target_id),  
                     None,  
                 )  
                 if target is not None and target.entity_type != EntityType.UNKNOWN_SLOT:  
                     issues.append("unresolved action target must be an unknown_slot or omitted")  
   
        if action.support_status == ActionSupportStatus.INDETERMINATE and action.target_id is not None:  
             issues.append("indeterminate action should not claim a concrete target")  
   
     return issues  
   
Inferred support is not graph fact until the patch validates. The patch must preserve provenance, validate element and relation compatibility, re-freeze the graph, then retry or reconcile only the affected lane slices.  
   
**5.5 Constraint Conflict Validator**  
Constraints must not negate required graph content.  
Examples:  
- NO_EXTRA_PEOPLE conflicts when person_01 is required. Use a narrower negative phrase such as `duplicate person`, `crowd`, or `extra background people` when the intent is to preserve one required subject.  
- negative_phrases=["people"] conflicts when person_01 is required.  
- negative_phrases=["handbag"] conflicts when accessory_01 is a required handbag.  
  
**5.6 Cross-Lane Coherence Validators**  
Cross-lane validators catch contradictions that structural reference checks cannot see.  
  
Required validators:  
| | | |  
|-|-|-|  
| **Validator** | **Prevents** | **Severity** |  
| Guardrail-element conflict | `NO_EXTRA_PEOPLE` while a person element is required | error |  
| Constraint negates required element | `negative_phrases=["handbag"]` while handbag is required | error |  
| Scale coherence | `macro` optic with `wide shot` / establishing framing | warning |  
| Renderable lane presence | graph-only documents with no object/action/cinematography/constraint surface | error |  
  
Scale coherence warnings do not block repair or rendering by default; they tell the verifier or sampler that a candidate may be visually incoherent. Constraint conflicts are hard errors because they ask the generator to remove required content.  
  
**5.7 Evidence and Placeholder Validator**  
Evidence validation prevents graph pollution. Entailed graph additions must point to a concrete phrase that requires the addition; unresolved placeholders must remain non-rendering slots.  
  
Rules:  
- explicit graph facts should preserve a prompt evidence span when available.  
- entailed graph facts must include evidence text and reason.  
- unresolved elements must use entity_type="unknown_slot" and importance="unresolved".  
- unresolved elements must not be primary subjects.  
- unresolved elements must not receive object descriptors, negative constraints, or cinematography emphasis.  
- blocked graph facts cannot proceed to lane expansion without prompt-improvement feedback or user clarification.  
  
def validate_evidence_and_placeholders(document: PromptDocument) -> list[str]:  
     issues: list[str] = []  
     unresolved_ids = set()  
   
     for element in document.graph.elements:  
         evidence = element.evidence  
         if evidence and evidence.category in {EvidenceCategory.ENTAILED, EvidenceCategory.UNRESOLVED, EvidenceCategory.BLOCKED}:  
             if not evidence.text or not evidence.reason:  
                 issues.append(f"{element.id} non-explicit evidence requires text and reason")  
   
         if element.importance == Importance.UNRESOLVED or element.entity_type == EntityType.UNKNOWN_SLOT:  
             unresolved_ids.add(element.id)  
             if element.entity_type != EntityType.UNKNOWN_SLOT:  
                 issues.append(f"{element.id} unresolved element must use entity_type=unknown_slot")  
             if element.role == ElementRole.PRIMARY_SUBJECT:  
                 issues.append(f"{element.id} unresolved element cannot be the primary subject")  
             if not evidence or evidence.category != EvidenceCategory.UNRESOLVED:  
                 issues.append(f"{element.id} unresolved element requires unresolved evidence")  
   
         if evidence and evidence.category == EvidenceCategory.BLOCKED:  
             issues.append(f"{element.id} is blocked and requires prompt-improvement feedback before lane expansion")  
   
     for relation in document.graph.relations:  
         evidence = relation.evidence  
         if relation.source_id in unresolved_ids or relation.target_id in unresolved_ids:  
             if not evidence or evidence.category != EvidenceCategory.UNRESOLVED:  
                 issues.append(f"{relation.id} touching unresolved slot requires unresolved evidence")  
   
     for obj in document.object_lane.objects:  
         if obj.target_id in unresolved_ids:  
             issues.append(f"{obj.target_id} unresolved slot cannot receive object descriptors")  
   
     return issues  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBCUpfDq4wwIAABiywEZJWQZeZ2ao9AAD+4liruzq/ngAA8Nr1ABweBgdur/QFAAAAAElFTkSuQmCC)  
**6. Retry Contracts**  
Validation issues are returned as structured repair requests.  
class ValidationSeverity(str, Enum):  
     ERROR = "error"  
     WARNING = "warning"  
   
   
 class ValidationIssue(StrictModel):  
     stage: Literal["graph", "object_lane", "action_lane", "cinematography_lane", "constraint_lane", "document", "render"]  
     severity: ValidationSeverity  
     path: str  
     message: str  
     retry_scope: str  
   
   
 class RetryRequest(StrictModel):  
     failed_stage: str  
     frozen_graph: SceneGraphDraft | None = None  
     invalid_payload: dict  
     issues: list[ValidationIssue]  
     instruction: str  
   
**6.1 Retry Rules**  
| | | | |  
|-|-|-|-|  
| **Failure Location** | **Retry Unit** | **Max Attempts** | **If Still Failing** |   
| Graph skeleton | full graph or graph slice | 2 | return blocking validation report |   
| Element label leakage | affected element slice | 2 | request an extraction repair that moves the modifier to the correct lane |   
| Relation compatibility | affected relation slice | 2 | remove optional relation or block if required |   
| Object descriptor | affected descriptor slice | 2 | omit optional field, keep element |   
| Action descriptor | affected action slice | 2 | omit action if nonessential |   
| Inferred action support | bounded graph inference patch, then affected lane slices | 2 | suppress or downgrade action if support cannot validate |   
| Unresolved action support | affected action slice or prompt-improvement hint | 1 | compile safe downgrade or return clarification feedback |   
| Indeterminate action support | affected action slice | 1 | suppress action; drop unsupported slots introduced solely for that indeterminate action |   
| Cinematography | whole cinematography lane | 2 | use conservative default shot language |   
| Constraints | whole constraint lane | 2 | remove conflicting negative phrase |   
| Merge | affected lane or relation slice | 2 | compile only validated content |   
| Compiler | prompt bundle | 1 | return validation report and partial trace |   
   
**6.2 Repair Prompt Shape**  
The repair call must be narrow and must return only the corrected JSON slice.  
You are repairing one invalid object descriptor.  
   
 Frozen graph:  
 {graph_json}  
   
 Invalid descriptor:  
 {invalid_descriptor_json}  
   
 Validation issues:  
 {issues_json}  
   
 Return only a corrected ObjectDescriptor JSON object.  
 Do not invent new element IDs.  
 Do not move ownership into description.  
 Do not add cinematography, action, or negative constraints.  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSd4EKxgBjP+Asa0hxW8ibAl2DIzR3UFAMBf3Gu1VefXEwAAXtsfSqwDVbgKngwAAAAASUVORK5CYII=)  
**7. LLM Call Contracts**  
**7.1 Graph Skeleton Call**  
System instruction:  
Extract a scene graph from the user prompt.  
 Elements are identity-only nouns.  
 Relations express ownership, contact, containment, spatial links, gaze, attachment, or visible light-emitter links.  
 Every relation must preserve a raw linkage phrase in relation_raw.  
 A locked relation enum is optional; do not force relation_raw into a bad enum.  
 If relation_raw is reusable but no enum fits, include proposed_enum for the enum candidate queue.  
 Do not put material, color, finish, condition, pattern, action, or camera terms in element labels.  
 Do not output cinematography, lighting treatment, lens, color grading, or shot fields.  
 Return JSON matching SceneGraphDraft.  
   
User input:  
A woman wearing a charcoal wool coat and holding a burgundy leather handbag  
 on a rain-slick city street, shot like a moody film still with soft practical lighting.  
   
Expected graph shape:  
{  
   "seed_prompt": "A woman wearing a charcoal wool coat and holding a burgundy leather handbag on a rain-slick city street, shot like a moody film still with soft practical lighting.",  
   "elements": [  
     {  
       "id": "person_01",  
       "entity_type": "person",  
       "label": "woman",  
       "role": "primary_subject",  
       "importance": "required"  
     },  
     {  
       "id": "garment_01",  
       "entity_type": "garment",  
       "label": "coat",  
       "role": "foreground",  
       "importance": "required"  
     },  
     {  
       "id": "accessory_01",  
       "entity_type": "accessory",  
       "label": "handbag",  
       "role": "foreground",  
       "importance": "required"  
     },  
     {  
       "id": "location_01",  
       "entity_type": "location",  
       "label": "city street",  
       "role": "background",  
       "importance": "required"  
     }  
   ],  
   "relations": [  
     {  
       "id": "rel_01",  
       "source_id": "person_01",  
       "target_id": "garment_01",  
       "relation_raw": "wearing",  
       "relation_match": {  
         "raw": "wearing",  
         "enum_value": "wearing",  
         "status": "matched_active",  
         "confidence": "clear",  
         "reason": "The prompt directly states the woman is wearing the coat."  
       },  
       "proposed_enum": null,  
       "importance": "required",  
       "description": null  
     },  
     {  
       "id": "rel_02",  
       "source_id": "person_01",  
       "target_id": "accessory_01",  
       "relation_raw": "holding",  
       "relation_match": {  
         "raw": "holding",  
         "enum_value": "holding",  
         "status": "matched_active",  
         "confidence": "clear",  
         "reason": "The prompt directly states the woman is holding the handbag."  
       },  
       "proposed_enum": null,  
       "importance": "required",  
       "description": null  
     },  
     {  
       "id": "rel_03",  
       "source_id": "person_01",  
       "target_id": "location_01",  
       "relation_raw": "on",  
       "relation_match": {  
         "raw": "on",  
         "enum_value": "on_top_of",  
         "status": "matched_active",  
         "confidence": "clear",  
         "reason": "The person is positioned on the street surface."  
       },  
       "proposed_enum": null,  
       "importance": "required",  
       "description": "standing on the street"  
     }  
   ]  
 }  
   
**7.2 Object Lane Call**  
System instruction:  
Expand appearance descriptors for existing graph elements.  
 Attach appearance to the element it describes.  
 Do not create relations.  
 Do not invent element IDs.  
 Do not describe actions or camera.  
 Return JSON matching ObjectLane.  
   
Expected object lane:  
{  
   "objects": [  
     {  
       "target_id": "garment_01",  
       "description": null,  
       "material": "wool",  
       "color": "charcoal",  
       "finish": "matte",  
       "condition": null,  
       "pattern": "solid"  
     },  
     {  
       "target_id": "accessory_01",  
       "description": "structured handbag",  
       "material": "leather",  
       "color": "burgundy",  
       "finish": "polished",  
       "condition": null,  
       "pattern": "solid"  
     },  
     {  
       "target_id": "location_01",  
       "description": "rain-slick pavement",  
       "material": null,  
       "color": null,  
       "finish": "glossy",  
       "condition": null,  
       "pattern": null  
     }  
   ]  
 }  
   
**7.3 Action Lane Call**  
For the example above, holding is already a relation. The action lane may stay sparse unless the prompt implies motion.  
{  
   "actions": [  
     {  
       "actor_id": "person_01",  
       "movement_raw": "posing",  
       "movement_match": {  
         "raw": "posing",  
         "enum_value": "posing",  
         "confidence": "clear",  
         "reason": "The raw phrase directly names the posing movement affordance."  
       },  
       "target_id": null,  
       "intensity": "subtle",  
       "support_status": "supported",  
       "required_relation_types": [],  
       "inferred_support": null,  
       "description": "composed stillness suitable for a film still"  
     }  
   ]  
 }  
   
**7.4 Cinematography Lane Call**  
Inputs:  
raw_user_prompt:  
 A woman wearing a charcoal wool coat and holding a burgundy leather handbag  
 on a rain-slick city street, shot like a moody film still with soft practical lighting.  
   
 validated_scene_graph:  
 {SceneGraphDraft JSON}  
   
System instruction:  
Extract only cinematography, lighting, framing, lens, color treatment, and setting atmosphere.  
 Use the raw prompt and graph seed_prompt as evidence for shot language.  
 Use the validated graph only to preserve scene context; do not add, remove, rename, or reassign elements.  
 Do not output object materials, clothing ownership, actions, relations, or negative constraints.  
 Return JSON matching CinematographyLane.  
   
{  
   "shot_size": "medium_shot",  
   "camera_angle": "eye_level",  
   "optic_character": "portrait_50mm",  
   "lighting_mood": "practical_lighting",  
   "color_treatment": "filmic_contrast",  
   "framing": "layered_depth",  
   "setting_description": "rain-slick city street with soft practical lights reflecting on pavement"  
 }  
   
**7.5 Constraint Lane Call**  
{  
   "guardrails": ["no_text", "no_logos", "no_extra_limbs", "no_distorted_hands"],  
   "negative_phrases": ["extra handbag", "duplicate subject", "warped face"]  
 }  
   
**7.6 Stage 2 Parallel Field Normalization**  
Each locked/standardized field is normalized by an independent field-scoped canonicalizer. The default path is embedding-first; prompt-LLM fallback calls are optional per field and should be launched in parallel after the primary graph/lane parse has completed. Fallback calls should not see the full prompt unless a field-specific context dependency is explicitly required. The default input is one raw field value and the enum context for one field.  
   
Input:  
{  
   "lane": "action",  
   "field": "movement_raw",  
   "raw": "swordfighting",  
   "enum_name": "MovementType",  
   "enum_values": ["posture_stance", "locomotion", "manual_interaction", "object_manipulation", "gestural", "gaze_attention", "facial_expression", "performance", "combat", "body_orientation"],  
   "instruction": "Return an enum only if the semantic match is clear. Use only the supplied enum_values. Otherwise return enum_value=null and confidence=unclear."  
 }  
  
Relation input example:  
{  
   "lane": "graph",  
   "field": "relation_type",  
   "raw": "leaning against",  
   "enum_name": "RelationType",  
   "enum_values": ["wearing", "holding", "on_top_of", "under", "over", "next_to", "illuminating"],  
   "instruction": "Return an enum only if the linkage clearly matches a supplied relation. Otherwise preserve raw and optionally propose a reusable new enum."  
 }  
  
Relation proposed-enum output:  
{  
   "raw": "leaning against",  
   "enum_value": null,  
   "confidence": "unmatched",  
   "reason": "The linkage is a stable support/orientation relation but no supplied enum captures it exactly.",  
   "proposed_new_enum": {  
     "code": "LEANING_AGAINST",  
     "label": "leaning against",  
     "aliases": ["propped against", "resting against"],  
     "description": "One element is supported at an angle by another vertical or stable element."  
   }  
 }  
   
Output:  
{  
   "raw": "swordfighting",  
   "enum_value": "combat",  
   "confidence": "clear",  
   "reason": "The phrase describes visually distinct combat motion. The raw phrase remains swordfighting for rendering."  
 }  
   
Unclear output:  
{  
   "raw": "throwing something",  
   "enum_value": null,  
   "confidence": "unclear",  
   "reason": "The action has an unspecified target and should remain raw rather than trigger inferred graph support."  
 }  
   
The canonicalizer must not create elements, add relations, change support_status, or revise the raw phrase. It only attaches an optional enum affordance.  
  
If a prompt-LLM fallback returns an enum outside the supplied field context, the response must be rejected as unmatched. This prevents a lighting enum, shot-size enum, material enum, or graph relation enum from leaking into the wrong field.  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OUQmAABBAsaeI2MKqV8RyJrGCfyJsCbbMzFldAQDwF/dWrdXx9QQAgNf2B/NkAzRb7P0YAAAAAElFTkSuQmCC)  
**8. Post-Verification Rendering**  
The renderer builds semantically sound descriptors by traversing verified graph relations and lane descriptors. It should produce a natural prompt string beginning with `Generate`, not a raw field dump.  
  
There is no final LLM prompt normalizer in the production path. LHS fanout should vary structured fields and enum affordances, then call this renderer cheaply for each candidate. If rendered language is weak, fix renderer templates, ordering, or field ownership upstream rather than adding another LLM pass per variation.  
  
Rendering rules:  
- Required graph elements must survive rendering even when they have no action.  
- Stable relations render before standalone loose elements.  
- Cinematography lighting renders as lighting treatment, such as `warm window light`, `soft sun`, `golden hour`, or `practical lighting`.  
- Visible light-emitting objects render as scene elements and may carry raw relations such as `lamp shining on notebook` or `neon sign glowing behind subject`.  
- Raw-only relations render from relation_raw, such as `chair leaning against wall`.  
- Target-bearing actions preserve their target phrases, such as `woman walking toward glass door` or `chef reaching toward ceramic bowl`. Do not flatten these into generic `walking motion` or `reaching pose`.  
- Cinematography terms such as dutch angle, dolly in, dolly out, handheld tracking, shallow depth of field, and blue hour survive reconstruction as camera/focus/lighting clauses.  
- Lighting phrases must not duplicate words, e.g. render `practical lighting`, not `practical lighting lighting`.  
**8.1 Rendering Helper Surface**

The renderer is implemented in `src/bruteforce_canvas/prompt_render.py`. It exports nine helpers that consume either pydantic models or plain dicts with matching field names: `object_phrase`, `relation_type`, `relation_label`, `relation_phrase`, `action_phrase`, `compile_positive_prompt`, `render_cinematography`, `compile_negative_prompt`, and `compile_prompt`. The signatures below match the production module; callers pass a pre-rendered target string to `relation_phrase`, and `action_phrase` reads actor, movement, and target from the action descriptor.

def object_phrase(element: Element, descriptor: ObjectDescriptor | None) -> str:

The renderer is implemented in `src/bruteforce_canvas/prompt_render.py`. It exports nine helpers that consume either pydantic models or plain dicts with matching field names: `object_phrase`, `relation_type`, `relation_label`, `relation_phrase`, `action_phrase`, `compile_positive_prompt`, `render_cinematography`, `compile_negative_prompt`, and `compile_prompt`. The signatures below match the production module; callers pass a pre-rendered target string to `relation_phrase`, and `action_phrase` reads actor, movement, and target from the action descriptor.

def object_phrase(element: Element, descriptor: ObjectDescriptor | None) -> str:
     parts: list[str] = []  
     if descriptor and descriptor.description:  
         parts.append(descriptor.description)  
     if descriptor and descriptor.color:  
         parts.append(descriptor.color)  
     if descriptor and descriptor.finish:  
         parts.append(descriptor.finish.value.replace("_", " "))  
     if descriptor and descriptor.material:  
         parts.append(descriptor.material)  
     if descriptor and descriptor.pattern and descriptor.pattern != Pattern.NONE:  
         parts.append(descriptor.pattern.value)  
     parts.append(element.label)  
     return " ".join(parts)  
   
   
def relation_type(relation: RelationDescriptor) -> RelationType | None:  
     if relation.relation_match and relation.relation_match.status == CanonicalStatus.MATCHED_ACTIVE:  
         return relation.relation_match.enum_value  
     return None  
   
   
def relation_label(relation: RelationDescriptor) -> str:  
     matched = relation_type(relation)  
     if matched:  
         return matched.value.replace("_", " ")  
     return relation.relation_raw  
   
   
def relation_phrase(
      source: Element,
      relation: RelationDescriptor,
      target: Element,
  ) -> str:
     rendered_target = object_phrase(target, None)  
     rendered_source = object_phrase(source, None)  
     matched = relation_type(relation)  
   
     if matched == RelationType.WEARING:  
         return f"wearing {rendered_target}"  
     if matched == RelationType.HOLDING:  
         return f"holding {rendered_target}"  
     if matched == RelationType.CARRYING:  
         return f"carrying {rendered_target}"  
     if matched == RelationType.ON_TOP_OF:  
         return f"on {rendered_target}"  
     if matched == RelationType.NEXT_TO:  
         return f"next to {rendered_target}"  
     if matched == RelationType.IN_FRONT_OF:  
         return f"in front of {rendered_target}"  
     if matched == RelationType.BEHIND:  
         return f"behind {rendered_target}"  
     if matched == RelationType.ILLUMINATING:  
         return f"lit by {rendered_source}"  
   
     return f"{relation_label(relation)} {rendered_target}"  
   
   
def action_phrase(
    action: ActionDescriptor,
    medium: Literal["image", "video"] = "image",
) -> str | None:
     if action.support_status == ActionSupportStatus.UNRESOLVED:  
         if action.prompt_improvement and action.prompt_improvement.safe_downgrade:  
             return action.prompt_improvement.safe_downgrade  
         return None  
   
     if action.support_status == ActionSupportStatus.INDETERMINATE:  
         return None  
   
     movement = action.movement_raw  
     if action.movement_match and action.movement_match.confidence == EnumMatchConfidence.CLEAR and action.movement_match.enum_value:  
         movement = action.movement_match.enum_value.replace("_", " ")  
     target_phrase = action.target_id  # caller resolves to a rendered target phrase  
     if movement in {"walking", "running"} and target_phrase:  
         return f"{action.actor_id} {movement} toward {target_phrase}"  
     if movement == "reaching" and target_phrase:  
         return f"{action.actor_id} reaching toward {target_phrase}"  
     if medium == "image":  
         phrase = f"{movement} pose"  
         if action.description:  
             phrase = f"{phrase} with {action.description}"  
         return f"{action.actor_id} {phrase}"  
   
     phrase = f"{movement} motion"  
     if action.description:  
         phrase = f"{phrase} with {action.description}"  
     return f"{action.actor_id} {phrase}"  
   
**8.2 Positive Prompt Compiler**  
def compile_positive_prompt(document: PromptDocument) -> str:  
     elements = {element.id: element for element in document.graph.elements}  
     descriptors = {obj.target_id: obj for obj in document.object_lane.objects}  
     trace: list[str] = []  
   
     primary = next(  
         element for element in document.graph.elements  
         if element.role == ElementRole.PRIMARY_SUBJECT  
     )  
   
     primary_phrase = object_phrase(primary, descriptors.get(primary.id))  
     trace.append(f"primary={primary.id}")  
   
     attached_phrases: list[str] = []  
     rendered_element_ids = {primary.id}  
     for relation in document.graph.relations:  
         if relation.source_id != primary.id:  
             continue  
         if relation_type(relation) in {  
             RelationType.WEARING,  
             RelationType.HOLDING,  
             RelationType.CARRYING,  
             RelationType.USING,  
         } or relation_type(relation) is None:  
             target = elements[relation.target_id]  
             attached_phrases.append(  
                 relation_phrase(primary, relation, target, descriptors.get(target.id))  
             )  
             rendered_element_ids.add(target.id)  
             trace.append(f"relation={relation.id}")  
   
     subject_clause = primary_phrase  
     if attached_phrases:  
         subject_clause = f"{primary_phrase} " + " and ".join(attached_phrases)  
   
     action_clauses = []  
     for action in document.action_lane.actions:  
         actor = elements[action.actor_id]  
         target = elements.get(action.target_id) if action.target_id else None  
         phrase = action_phrase(action, actor, target, descriptors.get(action.target_id) if action.target_id else None, medium="image")  
         if phrase is None:  
             trace.append(f"action_suppressed={action.actor_id}:{action.movement_raw}")  
             continue  
         action_clauses.append(phrase)  
         if action.target_id:  
             rendered_element_ids.add(action.target_id)  
         trace.append(f"action={action.actor_id}:{action.movement_raw}:{action.support_status.value}")  
   
     cinema = render_cinematography(document.cinematography_lane)  
     trace.append("cinematography=rendered")  
   
     standalone_clauses = [  
         object_phrase(element, descriptors.get(element.id))  
         for element in document.graph.elements  
         if element.id not in rendered_element_ids and element.importance == Importance.REQUIRED  
     ]  
   
     clauses = [subject_clause]  
     clauses.extend(action_clauses)  
     clauses.extend(standalone_clauses)  
     if document.cinematography_lane.setting_description:  
         clauses.append(document.cinematography_lane.setting_description)  
     if cinema:  
         clauses.append(cinema)  
   
     rendered = ", ".join(clause for clause in clauses if clause)  
     return f"Generate {rendered}" if rendered else "", trace  
   
**8.3 Cinematography Rendering**  
def render_cinematography(cine: CinematographyLane) -> str:  
     parts: list[str] = []  
     if cine.shot_size:  
         parts.append(cine.shot_size.value.replace("_", " "))  
     if cine.camera_angle:  
         parts.append(f"{cine.camera_angle.value.replace('_', ' ')} perspective")  
     if cine.optic_character:  
         parts.append(cine.optic_character.value.replace("_", " "))  
     if cine.camera_motion:  
         parts.append(cine.camera_motion)  
     if cine.focus_behavior:  
         parts.append(cine.focus_behavior)  
     if cine.lighting_mood:  
         lighting = cine.lighting_mood.value.replace("_", " ")  
         parts.append(lighting if "lighting" in lighting else f"{lighting} lighting")  
     if cine.color_treatment:  
         parts.append(cine.color_treatment.value.replace("_", " "))  
     if cine.framing:  
         parts.append(f"{cine.framing.value.replace('_', ' ')} framing")  
     return ", ".join(parts)  
   
**8.4 Negative Prompt Rendering**  
def compile_negative_prompt(constraints: ConstraintLane) -> str:  
     guardrail_text = [  
         guardrail.value.replace("_", " ")  
         for guardrail in constraints.guardrails  
     ]  
     return ", ".join([*guardrail_text, *constraints.negative_phrases])  
   
**8.5 Bundle Compiler**  
def compile_prompt(document: PromptDocument) -> PromptBundle:  
     positive_prompt, trace = compile_positive_prompt(document)  
     negative_prompt = compile_negative_prompt(document.constraint_lane)  
     prompt_hints = [  
         action.prompt_improvement  
         for action in document.action_lane.actions  
         if action.prompt_improvement is not None  
     ]  
   
     checklist = [  
         f"preserve element {element.id}: {element.label}"  
         for element in document.graph.elements  
         if element.importance == Importance.REQUIRED  
     ]  
     checklist.extend(  
         f"preserve relation {relation.source_id} {relation_label(relation)} {relation.target_id}"  
         for relation in document.graph.relations  
         if relation.importance == Importance.REQUIRED  
     )  
   
     return PromptBundle(  
         positive_prompt=positive_prompt,  
         negative_prompt=negative_prompt,  
         alignment_checklist=checklist,  
         render_trace=trace,  
         prompt_improvement_hints=prompt_hints,  
     )  
   
Example positive output:  
Generate woman wearing charcoal matte wool coat and holding structured polished burgundy leather handbag, rain-slick city street, medium shot, eye level perspective, portrait 50mm, practical lighting, filmic contrast, layered depth framing  
   
A production renderer can smooth grammar further:  
Generate a woman wearing a solid charcoal matte wool coat and holding a structured polished burgundy leather handbag on a rain-slick city street, medium shot from an eye-level portrait 50mm perspective, practical film lighting, filmic contrast, layered depth framing.  
   
Both outputs are valid only if they preserve graph ownership.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OYQ1AABSAwc8mi5wvkwZyCKCAACr4Z7a7BLfMzFYdAQDwF+da3dX+9QQAgNeuB6feBdUJcyS2AAAAAElFTkSuQmCC)  

**8.6 Implementation Contract**

The prompt deconstruction and rendering pipeline exposes these contract surfaces:

- `src/bruteforce_canvas/prompt_render.py` (deterministic rendering helpers such as `object_phrase`, `relation_type`, `relation_label`, `relation_phrase`, `action_phrase`, `compile_positive_prompt`, `render_cinematography`, `compile_negative_prompt`, and `compile_prompt`).
- `src/bruteforce_canvas/prompt.py` (Pydantic models for `Element`, `RelationDescriptor`, `ObjectDescriptor`, `ActionDescriptor`, `CinematographyLane`, `ConstraintLane`, `PromptDocument`, `PromptBundle`).
- `src/bruteforce_canvas/prompt_models.py` (external prompt-model import surface).
- `src/bruteforce_canvas/canonicalizers.py` (embedding-first canonicalization and prompt-LLM fallback boundary).
- `src/bruteforce_canvas/llm_clients.py` (OpenAI-compatible prompt-LLM server adapter for extraction, repair, verification, and canonicalization fallback).
- `src/bruteforce_canvas/spec_compliance.py` (per-spec checkers `check_spec_01..06` and the `check_all` aggregator).

**9. Conflict Examples**  
**9.0 Static Scene Without Action Triplet**  
Prompt:  
a snow-covered mountain at sunrise  
   
Preferred graph shape:  
{  
   "elements": [  
     {  
       "id": "environment_01",  
       "entity_type": "environment",  
       "label": "mountain",  
       "role": "primary_subject",  
       "importance": "required",  
       "evidence": {"text": "mountain", "category": "explicit", "reason": null}  
     },  
     {  
       "id": "environment_02",  
       "entity_type": "environment",  
       "label": "snow",  
       "role": "foreground",  
       "importance": "required",  
       "evidence": {"text": "snow-covered", "category": "explicit", "reason": null}  
     }  
   ],  
   "relations": [  
     {  
       "id": "rel_01",  
       "source_id": "environment_02",  
       "target_id": "environment_01",  
       "relation_raw": "covering",  
       "relation_match": {  
         "raw": "covering",  
         "enum_value": "covering",  
         "status": "matched_active",  
         "confidence": "clear",  
         "reason": "Snow-covered indicates snow covering the mountain."  
       },  
       "proposed_enum": null,  
       "importance": "required",  
       "evidence": {"text": "snow-covered mountain", "category": "explicit", "reason": null}  
     }  
   ]  
 }  
  
Preferred cinematography lane:  
{  
   "lighting_mood": "sunrise",  
   "setting_description": "quiet cold alpine atmosphere"  
 }  
   
Why valid:  
- The prompt has no actor, agent, or dynamic action.  
- The graph captures the primary subject and environmental covering relation.  
- The action lane remains empty; cinematography carries sunrise lighting and atmosphere.  
   
**9.1 Woman Incorrectly Owns Coat Material**  
Invalid object lane:  
{  
  "objects": [  
     {  
       "target_id": "person_01",  
       "description": "elegant pose with relaxed posture",  
       "material": "wool",  
       "color": "charcoal",  
       "finish": "matte",  
       "condition": "pristine",  
       "pattern": "solid"  
     }  
   ]  
 }  
   
Why invalid:  
- person_01 is the woman.  
- garment_01 is the coat.  
- person_01 wearing garment_01 already establishes ownership.  
- Coat material and coat color must attach to garment_01.  
Repair target:  
{  
   "objects": [  
     {  
       "target_id": "person_01",  
       "description": "elegant relaxed posture",  
       "material": null,  
       "color": null,  
       "finish": null,  
       "condition": null,  
       "pattern": null  
     },  
     {  
       "target_id": "garment_01",  
       "description": null,  
       "material": "wool",  
       "color": "charcoal",  
       "finish": "matte",  
       "condition": "pristine",  
       "pattern": "solid"  
     }  
   ]  
 }  
   
**9.2 Holding as Relation, Not Action**  
Prompt:  
woman holding a handbag  
   
Preferred graph:  
{  
   "source_id": "person_01",  
   "target_id": "accessory_01",  
   "relation_raw": "holding",  
   "relation_match": {  
     "raw": "holding",  
     "enum_value": "holding",  
     "status": "matched_active",  
     "confidence": "clear",  
     "reason": "The prompt directly states the woman is holding the handbag."  
   },  
   "proposed_enum": null  
 }  
   
Action lane may remain empty or describe pose. It should not duplicate stable holding unless the prompt implies motion:  
woman lifting the handbag  
 woman swinging the handbag  
 woman reaching into the handbag  
   
**9.3 Swordfighting as Supported or Inferred Action**  
Supported prompt:  
man holding a sword while swordfighting  
   
Supported graph/action shape:  
{  
   "relations": [  
     {  
       "source_id": "person_01",  
       "target_id": "prop_01",  
       "relation_raw": "holding",  
       "relation_match": {  
         "raw": "holding",  
         "enum_value": "holding",  
         "status": "matched_active",  
         "confidence": "clear",  
         "reason": "The prompt directly states the man is holding the sword."  
       },  
       "proposed_enum": null  
     }  
   ],  
   "actions": [  
     {  
       "actor_id": "person_01",  
       "movement_raw": "swordfighting",  
       "movement_match": {  
         "raw": "swordfighting",  
         "enum_value": "combat",  
         "confidence": "clear",  
         "reason": "The phrase describes visually distinct combat motion. The raw phrase remains swordfighting for rendering."  
       },  
       "target_id": "prop_01",  
       "support_status": "supported",  
       "required_relation_types": ["holding"],  
       "description": "swordfighting with implied directional force"  
     }  
   ]  
 }  
   
Why valid:  
- The sword exists in the graph.  
- The holding relation grounds the action.  
- For an image prompt, the action compiles as pose, gesture, tension, orientation, or implied motion.  
- For a video prompt, the action compiles as temporal motion behavior without adding a storyboard or timeline.  
   
Inferred prompt:  
man swordfighting  
   
If the graph contains person_01 but omitted a sword, swordfighting may be inferred because the action strongly entails an instrument. The action lane may request a bounded graph inference patch for a sword element and holding or using relation. That patch must validate and the graph must be re-frozen before merge. The action lane must not create prop_01 directly.  
   
**9.4 Throwing Something as Indeterminate**  
Prompt:  
person throwing something  
   
Graph unresolved slot:  
{  
   "id": "unknown_01",  
   "entity_type": "unknown_slot",  
   "label": "unspecified thrown object",  
   "role": "supporting",  
   "importance": "unresolved",  
   "evidence": {  
     "text": "something",  
     "category": "unresolved",  
     "reason": "The prompt indicates a thrown object slot but does not identify the object."  
   }  
 }  
   
Unresolved action shape:  
{  
   "actor_id": "person_01",  
   "movement_raw": "throwing something",  
   "movement_match": {  
     "raw": "throwing something",  
     "enum_value": null,  
     "confidence": "unclear",  
     "reason": "The phrase has an unspecified target and should not be normalized into a confident movement affordance."  
   },  
   "target_id": "unknown_01",  
   "support_status": "unresolved",  
   "required_relation_types": [],  
   "inferred_support": null,  
   "prompt_improvement": {  
     "issue": "The thrown object is unspecified.",  
     "suggested_rewrites": [  
       "person throwing a named object",  
       "person in a throwing pose with no visible thrown object"  
     ],  
     "safe_downgrade": "throwing pose with no visible thrown object"  
   },  
   "description": "open throwing-like motion with no specified object"  
 }  
   
Why unresolved:  
- The prompt does not identify the thrown object.  
- The action has an open missing target.  
- The pipeline must not invent the object.  
- The renderer should either use the safe downgrade or surface the prompt-improvement hint.  
- Any candidate element created only for an unresolved or indeterminate action must remain an unknown_slot or be dropped; directly prompted or already validated graph elements remain.  
   
**9.5 Target-Preserving Actions and Camera Reconstruction**  
Prompt:  
a woman in a black coat walking toward a glass door, camera slowly pushes in, blurry background, low view  
  
Required extraction behavior:  
- Elements: woman, coat, door. Do not create a camera prop from camera-language.  
- Object lane: black attaches to coat; glass attaches to door.  
- Action lane: actor_id=person_01, movement_raw="walking", target_id=door element.  
- Cinematography lane: camera_motion="dolly in", camera_angle="low angle", focus_behavior="shallow depth of field".  
  
Compiled prompt should preserve the directional target:  
Generate woman wearing black coat, woman walking toward glass door, medium shot, low angle perspective, dolly in, shallow depth of field  
  
Bad reconstruction:  
Generate woman wearing black coat, woman walking motion with steady forward movement, glass door, low angle perspective  
  
Why bad:  
- The action target "toward a glass door" was detached from walking.  
- The camera phrase was underused or risked creating a camera object.  
  
**9.6 Raw-Only Relation and Cinematography Pressure Case**  
Prompt:  
a chef reaching toward a ceramic bowl resting on a table while a chair leans against the wall behind it, dutch angle, dolly out, warm practical light  
  
Required extraction behavior:  
- `bowl on table` is a canonical on_top_of relation.  
- `chair leans against wall` is a raw-first relation. If no locked enum exists, preserve relation_raw="leaning against" and propose `LEANING_AGAINST`.  
- Action lane targets the bowl: actor_id=chef, movement_raw="reaching", target_id=bowl.  
- Cinematography preserves dutch angle, dolly out, and practical lighting.  
  
Compiled prompt should preserve target and raw relation:  
Generate chef, ceramic bowl on table, chair leaning against wall, chef reaching toward ceramic bowl, dutch angle perspective, dolly out, practical lighting  
  
Bad reconstruction:  
Generate chef, ceramic bowl on table, chair leaning against wall, chef reaching motion, dutch angle perspective, dolly out, practical lighting lighting  
  
Why bad:  
- "reaching toward ceramic bowl" was flattened into generic motion.  
- lighting was duplicated as "lighting lighting".  
  
**9.7 Lighting Ownership**  
Lighting treatment belongs to cinematography when it describes the look of the scene. Visible light-emitting objects belong to the graph when the object itself is part of the scene.  
  
Prompt:  
a lonely wall with soft sun on it, quiet and wide  
  
Preferred extraction:  
{  
   "graph": {  
     "elements": [  
       {"id": "architecture_01", "entity_type": "architecture", "label": "wall", "role": "primary_subject"}  
     ],  
     "relations": []  
   },  
   "cinematography_lane": {  
     "lighting_mood": "soft sun",  
     "shot_size": "wide shot",  
     "setting_description": "quiet atmosphere"  
   }  
 }  
  
Preferred rendered phrase:  
Generate wall, wide shot, soft sun lighting, quiet atmosphere  
  
Visible emitter prompt:  
a desk lamp shining on an open notebook, warm room  
  
Preferred extraction:  
{  
   "graph": {  
     "elements": [  
       {"id": "prop_01", "entity_type": "prop", "label": "desk lamp", "role": "foreground"},  
       {"id": "prop_02", "entity_type": "prop", "label": "notebook", "role": "primary_subject"}  
     ],  
     "relations": [  
       {"id": "rel_01", "source_id": "prop_01", "target_id": "prop_02", "relation_raw": "shining on"}  
     ]  
   },  
   "cinematography_lane": {  
     "lighting_mood": "warm room light"  
   }  
 }  
  
**9.8 Constraint Conflict**  
Invalid:  
{  
   "guardrails": ["no_extra_people"],  
   "negative_phrases": ["woman", "handbag"]  
 }  
   
Why invalid:  
- no_extra_people conflicts with a required woman. Use `duplicate person` or `crowd` when the intended exclusion is extra background people.  
- woman and handbag as negative phrases conflict with required graph elements.  
Repair:  
{  
   "guardrails": ["no_extra_people"],  
   "negative_phrases": ["duplicate subject", "extra handbag"]  
 }  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSPBCj5fFgpQwYwEZiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AMTRBeEgNK9YAAAAAElFTkSuQmCC)  
**10. End-to-End Walkthrough**  
Input prompt:  
a woman in a muddy green jacket holding a flashlight, pushing open a rusty gate with one hand and looking at a dog behind it, old yard, dark moody light, shaky close camera, no extra people  
  
**10.1 Primary Prompt-LLM Extraction**  
The primary prompt-LLM performs semantic deconstruction. It does not write polished prompt prose. It assigns each phrase to the correct field and preserves raw linkage/action/lighting strings for later canonicalization.  
  
Extracted elements:  
{  
   "elements": [  
     {"id": "person_01", "entity_type": "person", "label": "woman", "role": "primary_subject"},  
     {"id": "garment_01", "entity_type": "garment", "label": "jacket", "role": "foreground"},  
     {"id": "prop_01", "entity_type": "prop", "label": "flashlight", "role": "foreground"},  
     {"id": "architecture_01", "entity_type": "architecture", "label": "gate", "role": "foreground"},  
     {"id": "animal_01", "entity_type": "animal", "label": "dog", "role": "supporting"},  
     {"id": "location_01", "entity_type": "location", "label": "yard", "role": "background"}  
   ]  
 }  
  
Extracted relations:  
{  
   "relations": [  
     {"id": "rel_01", "source_id": "person_01", "relation_raw": "in", "target_id": "garment_01"},  
     {"id": "rel_02", "source_id": "person_01", "relation_raw": "holding", "target_id": "prop_01"},  
     {"id": "rel_03", "source_id": "animal_01", "relation_raw": "behind", "target_id": "architecture_01"},  
     {"id": "rel_04", "source_id": "architecture_01", "relation_raw": "in", "target_id": "location_01"}  
   ]  
 }  
  
Object lane:  
{  
   "object_lane": {  
     "objects": [  
       {"target_id": "garment_01", "color": "muddy green"},  
       {"target_id": "architecture_01", "condition": "rusty"},  
       {"target_id": "location_01", "condition": "old"}  
     ]  
   }  
 }  
  
`muddy green` is a color phrase. It must not also become `condition="muddy"` unless the prompt separately indicates dirt, mud, staining, or wear as a condition, such as `mud-caked green jacket` or `green jacket covered in mud`. One source phrase should not be duplicated into two fields without separate evidence.  
  
Action lane:  
{  
   "action_lane": {  
     "actions": [  
       {  
         "actor_id": "person_01",  
         "movement_raw": "pushing open",  
         "target_id": "architecture_01",  
         "description": "with one hand",  
         "intensity": "moderate"  
       },  
       {  
         "actor_id": "person_01",  
         "movement_raw": "looking at",  
         "target_id": "animal_01",  
         "intensity": "subtle"  
       }  
     ]  
   }  
 }  
  
Cinematography lane:  
{  
   "cinematography_lane": {  
     "lighting_mood": "dark moody light",  
     "camera_motion": "shaky camera",  
     "shot_size": "close camera"  
   }  
 }  
  
Constraint lane:  
{  
   "constraint_lane": {  
     "guardrails": ["no extra people"]  
   }  
 }  
  
Ownership decisions:  
- `flashlight` is a graph element because it is a visible prop.  
- `dark moody light` is cinematography because it is lighting treatment, not a visible light-emitting object.  
- `muddy green` belongs to the jacket color field.  
- `rusty` belongs to the gate condition field.  
- `old` belongs to the yard condition field.  
- `looking at dog` keeps its target and must not flatten to generic looking.  
  
**10.2 Pydantic Validation**  
No LLM is used for this step. The extracted payload is checked structurally before canonicalization continues.  
  
Validation checks:  
- all relation source IDs resolve.  
- all relation target IDs resolve.  
- every relation has `relation_raw`.  
- every object lane target ID resolves to a graph element.  
- every action actor ID and target ID resolves to a graph element.  
- element labels are noun-like and do not carry lane descriptors.  
  
Invalid extraction example:  
{  
   "object_lane": {  
     "objects": [  
       {"target_id": "person_01", "color": "muddy green"}  
     ]  
   }  
 }  
  
Why invalid: the color belongs to `garment_01`, not `person_01`.  
  
**10.3 Parallel Field-Scoped Enum Canonicalization**  
Embedding-first field canonicalizers run independently. Prompt-LLM fallback calls run only for ambiguous or configured fields. Each fallback receives one raw value, the field name, the locked enum context for that field, and only the minimal prompt or graph context required by that field. Canonicalization attaches enum affordances; it does not rewrite the raw field or add graph facts.  
  
Relation canonicalization example:  
{  
   "field": "relation_type",  
   "raw": "holding",  
   "enum_values": ["WEARING", "HOLDING", "ON_TOP_OF", "BEHIND"]  
 }  
  
Output:  
{  
   "raw": "holding",  
   "enum_code": "HOLDING",  
   "confidence": "clear"  
 }  
  
Camera canonicalization example:  
{  
   "field": "camera_motion",  
   "raw": "shaky camera",  
   "enum_values": ["LOCKED_OFF", "HANDHELD", "DOLLY_IN", "DOLLY_OUT"]  
 }  
  
Output:  
{  
   "raw": "shaky camera",  
   "enum_code": "HANDHELD",  
   "confidence": "clear"  
 }  
  
Motion-class example:  
{  
   "field": "movement",  
   "raw": "pushing open",  
   "enum_values": ["POSTURE_STANCE", "LOCOMOTION", "MANUAL_INTERACTION", "OBJECT_MANIPULATION", "GESTURAL", "GAZE_ATTENTION"]  
 }  
  
Output:  
{  
   "raw": "pushing open",  
   "enum_code": "MANUAL_INTERACTION",  
   "confidence": "clear",  
   "reason": "The phrase describes hand-driven interaction with a target. The raw phrase remains pushing open for rendering."  
 }  
   
Unmatched still means preserve raw phrase, do not force a bad enum, and optionally propose a reusable new enum for fields where the missing concept is a reusable category rather than a verb phrase.  
  
**10.4 Canonicalization Attachment**  
No LLM is used for this step. The document still preserves raw values, with enum metadata attached where available.  
  
{  
   "relation_raw": "holding",  
   "relation_match": {  
     "enum": "HOLDING",  
     "confidence": "clear"  
   }  
 }  
  
{  
   "movement_raw": "pushing open",  
   "movement_match": {  
     "enum": "MANUAL_INTERACTION",  
     "confidence": "clear"  
   }  
 }  
  
The raw phrase remains the source of truth. The enum is an internal affordance for validation, variation, and analytics.  
  
**10.5 Prompt-LLM Verification**  
The prompt-LLM verifier checks the assembled document for faithfulness, graph linkage, lane ownership, enum fit, unresolved slots, lighting ownership, and renderability. It returns structured issues; it does not silently rewrite the full prompt.  
  
Expected approval:  
{  
   "approved": true,  
   "issues": []  
 }  
  
Example verifier issue:  
{  
   "approved": false,  
   "issues": [  
     {  
       "path": "graph.elements",  
       "message": "dark moody light is lighting treatment, not a visible scene element",  
       "repair_scope": "graph extraction slice"  
     }  
   ]  
 }  
  
**10.6 Deterministic Rendering**  
No LLM is used for rendering. The renderer walks the verified structure in stable order: primary subject, worn/held objects, target-preserving actions, supporting relations, setting, cinematography, constraints.  
  
Rendered assembly:  
woman wearing muddy green jacket holding flashlight  
+ pushing open rusty gate with one hand  
+ looking at dog behind gate  
+ old yard  
+ close-up  
+ handheld camera  
+ low-key lighting  
+ no extra people  
  
Compiled prompt:  
Generate woman wearing muddy green jacket holding flashlight, pushing open rusty gate with one hand while looking at dog behind it, old yard, close-up, handheld camera, low-key lighting, no extra people  
  
**10.7 LHS Variation**  
No LLM is used by default for LHS fanout. The verified document becomes a stable base. LHS varies structured fields and enum affordances, then calls the deterministic renderer for each candidate.  
Detailed enum routing, field eligibility, Thompson Sampling, GP-style combination memory, and evaluation-feedback policy are specified in `bruteforce-canvas_LHS_enum_router.md`.  
  
Variation axes:  
{  
   "shot_size": ["close-up", "medium shot", "wide shot"],  
   "camera_motion": ["handheld", "locked-off"],  
   "lighting_mood": ["low-key lighting", "warm practical lighting"],  
   "color_treatment": ["filmic contrast", "muted palette"]  
 }  
  
Candidate render:  
Generate woman wearing muddy green jacket holding flashlight, pushing open rusty gate with one hand while looking at dog behind it, old yard, medium shot, locked-off camera, warm practical lighting, muted palette, no extra people  
  
No additional extraction, graph repair, or final LLM prompt normalizer is needed for ordinary LHS candidates.  
  
**11. Implementation Checklist**  
Minimum viable implementation:  
1. Run primary prompt-LLM extraction into PromptDocument.  
2. Preserve raw strings, evidence spans, relation_raw values, unresolved slots, and lane ownership from extraction.  
3. Run embedding-first canonicalization per locked/standardized field using BAAI/bge-small-en-v1.5, with prompt-LLM fallback only where configured or needed.  
4. Attach optional enum matches while preserving the original raw field values.  
5. Run prompt-LLM verification over the raw prompt, extracted document, and canonicalization results.  
6. Classify action support as supported, inferred, unresolved, or indeterminate inside the prompt-LLM verification report.  
7. Route verifier failures to extraction repair, canonicalization repair, prompt-improvement feedback, or safe non-inventive suppression.  
8. Retry only the failed extraction slice or canonicalization field when a repair is possible.  
9. Render PromptBundle only after verifier approval or an explicitly traced safe downgrade.  
10. Validate final prompt strings are non-empty and preserve required graph facts.  
Recommended trace data:  
- Initial raw prompt.  
- Extracted PromptDocument JSON.  
- Per-field canonicalization results.  
- LLM verification issues and approvals.  
- Retry attempts and final repaired slice or field.  
- Final prompt bundle.  
- Alignment checklist.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNhRAF6EPYDLhGADSywEZJWQZeZ2aszAAD+4l6rrTq+ngAA8Nr1AIWsBDYDm5cLAAAAAElFTkSuQmCC)  
**12. Stage Validation and Retry Contract**  
Validation exists at each prompt-pipeline stage, and conflicting issues are repaired at the narrowest reliable scope.  

The retry scope is:  
- Graph conflicts retry graph slices.  
- Object conflicts retry object descriptors.  
- Action conflicts retry actions.  
- Cinematography conflicts retry cinematography.  
- Constraint conflicts retry constraints.  
- Merge conflicts retry the specific lane or relation that caused the contradiction.  
This is the main reason the graph-first layout is better than a single large JSON shot. The first call establishes semantic ownership. Later calls can be parallel and narrow because each lane has a frozen graph contract. When a lane fails, the pipeline knows exactly what to repair without asking the model to regenerate the whole scene.  
