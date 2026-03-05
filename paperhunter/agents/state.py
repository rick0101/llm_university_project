"""
LangGraph shared state for the agentic survey writing pipeline.

Defines the TypedDict state schema used across all agents (Planner,
Writer, Critic, Assembler) and auxiliary Pydantic models for structured
sections, outlines, and verification results.

Usage:
    from paperhunter.agents.state import (
        SurveyState, SectionOutline, SectionDraft, VerificationResult,
        CitationRecord, create_initial_state, state_to_dict, state_from_dict,
    )

    state = create_initial_state(l2_summaries=[...])
"""

from __future__ import annotations

import json
import operator
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field

from paperhunter.models import (
    L2Summary,
    ProvenanceRecord,
    SurveyVersion,
    VerificationStatus,
)


# ============================================================================
# Enums
# ============================================================================

class SectionStatus(str, Enum):
    """Status of a single section in the writing pipeline."""
    PENDING = "pending"           # Non ancora scritta
    DRAFTED = "drafted"           # Prima bozza completata dal Writer
    UNDER_REVIEW = "under_review" # In fase di verifica dal Critic
    NEEDS_REVISION = "needs_revision"  # Citazioni non verificate, da riscrivere
    VERIFIED = "verified"         # Citazioni verificate con successo
    FINALIZED = "finalized"       # Assemblata nella versione finale


class PipelinePhase(str, Enum):
    """Current phase of the overall pipeline."""
    INIT = "init"
    PLANNING = "planning"
    WRITING = "writing"
    VERIFYING = "verifying"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Auxiliary Pydantic models
# ============================================================================

class CitationRecord(BaseModel):
    """A single citation used in a section draft.

    Maps a claim text to its bibtex_key and the chunk_id from which
    the information was retrieved.
    """
    bibtex_key: str = Field(..., description="Chiave BibTeX citata (es. Zou_2023_GCG)")
    claim: str = Field("", description="Testo del claim nel survey che cita questa fonte")
    chunk_id: Optional[str] = Field(None, description="ID del chunk sorgente usato per il claim")
    retrieval_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Score del retrieval che ha fornito questo chunk"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | CitationRecord) -> CitationRecord:
        if isinstance(data, CitationRecord):
            return data
        return cls(**data)


class SectionOutline(BaseModel):
    """Outline of a single section, produced by the Planner agent.

    Defines the structure, target length, and key topics for a section
    of the survey paper.
    """
    key: str = Field(
        ...,
        description="Chiave univoca della sezione (snake_case, es. 'prompt_injection_attacks')"
    )
    title: str = Field(..., description="Titolo della sezione (es. 'Prompt Injection Attacks')")
    description: str = Field(
        "",
        description="Descrizione di cosa deve coprire la sezione"
    )
    target_length_words: int = Field(
        1500,
        ge=100, le=10000,
        description="Lunghezza target in parole"
    )
    key_topics: list[str] = Field(
        default_factory=list,
        description="Topic chiave da coprire nella sezione"
    )
    related_categories: list[str] = Field(
        default_factory=list,
        description="Categorie del corpus correlate (es. ['Prompt Injection', 'Adversarial Attacks'])"
    )
    order: int = Field(
        0, ge=0,
        description="Ordine nella sequenza del paper (0-based)"
    )
    section_type: str = Field(
        "body",
        description="Tipo: 'abstract', 'introduction', 'body', 'testing', 'discussion', 'conclusion'"
    )
    parent_key: Optional[str] = Field(
        None,
        description="Chiave della sezione padre (per sotto-sezioni)"
    )
    subsections: list[str] = Field(
        default_factory=list,
        description="Chiavi delle sotto-sezioni (se presenti)"
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | SectionOutline) -> SectionOutline:
        if isinstance(data, SectionOutline):
            return data
        return cls(**data)


class VerificationDetail(BaseModel):
    """Verification result for a single citation within a section."""
    citation_key: str = Field(..., description="Chiave BibTeX verificata")
    claim_text: str = Field("", description="Testo del claim verificato")
    source_chunk_id: Optional[str] = Field(None, description="Chunk ID sorgente")
    source_text_excerpt: str = Field("", description="Estratto dal testo sorgente")
    status: VerificationStatus = Field(
        VerificationStatus.SOURCE_NOT_FOUND,
        description="Esito della verifica"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidenza della verifica")
    reasoning: str = Field("", description="Spiegazione del Critic per questa verifica")

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["status"] = self.status.value if isinstance(self.status, VerificationStatus) else self.status
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | VerificationDetail) -> VerificationDetail:
        if isinstance(data, VerificationDetail):
            return data
        return cls(**data)


class VerificationResult(BaseModel):
    """Aggregated verification result for an entire section.

    Contains the per-citation verification details and the overall
    pass/fail decision based on the verification_threshold from config.
    """
    section_key: str = Field(..., description="Chiave della sezione verificata")
    total_citations: int = Field(0, description="Numero totale di citazioni nella sezione")
    supported: int = Field(0, description="Citazioni SUPPORTED")
    partially_supported: int = Field(0, description="Citazioni PARTIALLY_SUPPORTED")
    not_supported: int = Field(0, description="Citazioni NOT_SUPPORTED")
    source_not_found: int = Field(0, description="Citazioni SOURCE_NOT_FOUND")
    details: list[VerificationDetail] = Field(
        default_factory=list,
        description="Dettaglio per ogni citazione verificata"
    )
    pass_rate: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Percentuale di citazioni supportate (supported + partially) / total"
    )
    passed: bool = Field(
        False,
        description="Se la sezione supera la soglia di verifica"
    )
    iteration: int = Field(
        1, ge=1,
        description="Numero di iterazione writer↔critic"
    )
    verified_at: Optional[datetime] = Field(None, description="Timestamp della verifica")

    def compute_pass_rate(self, threshold: float = 0.80) -> None:
        """Computes pass_rate and sets passed flag based on threshold."""
        if self.total_citations > 0:
            ok = self.supported + self.partially_supported
            self.pass_rate = ok / self.total_citations
        else:
            # No citations = trivially passes (intro/conclusion may have none)
            self.pass_rate = 1.0
        self.passed = self.pass_rate >= threshold

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["details"] = [d.to_dict() for d in self.details]
        if self.verified_at:
            data["verified_at"] = self.verified_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | VerificationResult) -> VerificationResult:
        if isinstance(data, VerificationResult):
            return data
        if "details" in data and data["details"]:
            data["details"] = [VerificationDetail.from_dict(d) for d in data["details"]]
        if "verified_at" in data and isinstance(data["verified_at"], str):
            data["verified_at"] = datetime.fromisoformat(data["verified_at"])
        return cls(**data)


class SectionDraft(BaseModel):
    """A drafted section produced by the Writer agent.

    Contains the text, the citations used, and the current verification
    status.
    """
    section_key: str = Field(..., description="Chiave della sezione (corrisponde a SectionOutline.key)")
    title: str = Field("", description="Titolo della sezione")
    text: str = Field("", description="Testo completo della sezione in Markdown")
    citations: list[CitationRecord] = Field(
        default_factory=list,
        description="Citazioni usate nella sezione"
    )
    word_count: int = Field(0, description="Conteggio parole")
    status: SectionStatus = Field(
        SectionStatus.PENDING,
        description="Stato corrente della sezione"
    )
    iteration: int = Field(
        0, ge=0,
        description="Numero di iterazione di scrittura (0 = non ancora scritta)"
    )
    rag_queries: list[str] = Field(
        default_factory=list,
        description="Query RAG usate per il retrieval"
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs recuperati dal retriever"
    )
    writer_notes: str = Field(
        "",
        description="Note interne del Writer (es. difficoltà, integrazione mancante)"
    )
    critic_feedback: str = Field(
        "",
        description="Ultimo feedback del Critic"
    )
    drafted_at: Optional[datetime] = Field(None, description="Timestamp dell'ultima bozza")

    def compute_word_count(self) -> int:
        """Ricalcola il conteggio parole dal testo."""
        self.word_count = len(self.text.split()) if self.text else 0
        return self.word_count

    def unique_citation_keys(self) -> set[str]:
        """Ritorna le chiavi BibTeX uniche usate nella sezione."""
        return {c.bibtex_key for c in self.citations}

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["citations"] = [c.to_dict() for c in self.citations]
        data["status"] = self.status.value if isinstance(self.status, SectionStatus) else self.status
        if self.drafted_at:
            data["drafted_at"] = self.drafted_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | SectionDraft) -> SectionDraft:
        if isinstance(data, SectionDraft):
            return data
        if "citations" in data and data["citations"]:
            data["citations"] = [CitationRecord.from_dict(c) for c in data["citations"]]
        if "drafted_at" in data and isinstance(data["drafted_at"], str):
            data["drafted_at"] = datetime.fromisoformat(data["drafted_at"])
        return cls(**data)


# ============================================================================
# Custom reducers for LangGraph Annotated state
# ============================================================================

def _replace_or_default(existing: Any, new: Any) -> Any:
    """Reducer that simply replaces the old value with the new one.

    This is the default LangGraph behaviour for non-Annotated fields,
    but we make it explicit for clarity.
    """
    return new


def _merge_sections(
    existing: dict[str, SectionDraft],
    updates: dict[str, SectionDraft],
) -> dict[str, SectionDraft]:
    """Reducer that merges section drafts by key.

    New drafts overwrite existing ones with the same key.
    Existing keys not in the update are preserved.
    """
    merged = dict(existing) if existing else {}
    if updates:
        merged.update(updates)
    return merged


def _merge_verification(
    existing: dict[str, VerificationResult],
    updates: dict[str, VerificationResult],
) -> dict[str, VerificationResult]:
    """Reducer that merges verification results by section key.

    Newer verifications overwrite older ones for the same section.
    """
    merged = dict(existing) if existing else {}
    if updates:
        merged.update(updates)
    return merged


def _append_unique_errors(existing: list[str], new: list[str]) -> list[str]:
    """Reducer that appends only new unique error messages."""
    combined = list(existing) if existing else []
    seen = set(combined)
    for err in (new or []):
        if err not in seen:
            combined.append(err)
            seen.add(err)
    return combined


def _merge_retrieved_facts(
    existing: dict[str, list[dict]],
    updates: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Reducer that merges retrieved_facts by section key.

    For each section, appends new fact entries to the existing list
    (no deduplication — each tool call is logged individually).
    """
    merged = dict(existing) if existing else {}
    for key, entries in (updates or {}).items():
        if key in merged:
            merged[key] = merged[key] + entries
        else:
            merged[key] = list(entries)
    return merged


def _merge_agentic_stats(
    existing: dict,
    updates: dict,
) -> dict:
    """Reducer that merges agentic_stats (shallow merge, updates overwrite)."""
    merged = dict(existing) if existing else {}
    merged.update(updates or {})
    return merged


# ============================================================================
# LangGraph State — TypedDict with Annotated reducers
# ============================================================================

class SurveyState(TypedDict):
    """Shared state for the LangGraph agentic survey writing pipeline.

    Each field uses an Annotated reducer so that LangGraph can correctly
    merge partial updates returned by individual agent nodes.

    Fields:
        l2_summaries:          L2 summaries (one per category) used as source material.
        outline:               Ordered list of section outlines produced by the Planner.
        sections:              Section drafts keyed by section_key, written by the Writer.
        verification_results:  Verification results keyed by section_key, from the Critic.
        bibliography:          BibTeX keys effectively used in the final paper.
        provenance:            Provenance records for every citation in the paper.
        final_paper:           Assembled final Markdown content.
        version_info:          Versioning metadata for the current run.
        current_phase:         Current pipeline phase.
        current_section_key:   Section key currently being processed (writer/critic loop).
        iteration_count:       Global iteration counter across the pipeline.
        errors:                Error messages collected during execution.
        metadata:              Arbitrary metadata dict for extensibility.
    """

    # --- Source material ---
    l2_summaries: Annotated[list[dict], _replace_or_default]

    # --- Planner output ---
    outline: Annotated[list[dict], _replace_or_default]

    # --- Writer output ---
    sections: Annotated[dict[str, dict], _merge_sections]

    # --- Critic output ---
    verification_results: Annotated[dict[str, dict], _merge_verification]

    # --- Bibliography & provenance ---
    bibliography: Annotated[list[str], _replace_or_default]
    provenance: Annotated[list[dict], operator.add]

    # --- Assembly ---
    final_paper: Annotated[str, _replace_or_default]

    # --- Versioning ---
    version_info: Annotated[dict, _replace_or_default]

    # --- Pipeline control ---
    current_phase: Annotated[str, _replace_or_default]
    current_section_key: Annotated[str, _replace_or_default]
    iteration_count: Annotated[int, _replace_or_default]

    # --- Error tracking (append-only) ---
    errors: Annotated[list[str], _append_unique_errors]

    # --- Agentic scratchpad & tool log ---
    current_scratchpad: Annotated[list[str], _replace_or_default]
    retrieved_facts: Annotated[dict[str, list[dict]], _merge_retrieved_facts]
    agentic_stats: Annotated[dict, _merge_agentic_stats]

    # --- Extensibility ---
    metadata: Annotated[dict, _replace_or_default]


# ============================================================================
# State helper functions
# ============================================================================

def create_initial_state(
    l2_summaries: list[L2Summary] | None = None,
    outline: list[SectionOutline] | None = None,
    version_info: SurveyVersion | None = None,
    metadata: dict | None = None,
) -> SurveyState:
    """Create a fresh initial state for the LangGraph pipeline.

    Args:
        l2_summaries: Pre-computed L2 summaries (optional, can be loaded later).
        outline: Pre-defined outline (optional, Planner will generate one if absent).
        version_info: Version metadata (optional, auto-created if absent).
        metadata: Extra metadata dict.

    Returns:
        A SurveyState TypedDict ready for the pipeline.
    """
    l2_dicts: list[dict] = []
    if l2_summaries:
        l2_dicts = [s.model_dump() for s in l2_summaries]

    outline_dicts: list[dict] = []
    if outline:
        outline_dicts = [o.to_dict() for o in outline]

    version_dict: dict = {}
    if version_info:
        version_dict = version_info.model_dump()
        if "timestamp" in version_dict and isinstance(version_dict["timestamp"], datetime):
            version_dict["timestamp"] = version_dict["timestamp"].isoformat()
    else:
        now = datetime.now()
        version_dict = SurveyVersion(
            version_number="1.0",
            timestamp=now,
            paper_count=0,
            categories=[],
        ).model_dump()
        version_dict["timestamp"] = now.isoformat()

    return SurveyState(
        l2_summaries=l2_dicts,
        outline=outline_dicts,
        sections={},
        verification_results={},
        bibliography=[],
        provenance=[],
        final_paper="",
        version_info=version_dict,
        current_phase=PipelinePhase.INIT.value,
        current_section_key="",
        iteration_count=0,
        errors=[],
        current_scratchpad=[],
        retrieved_facts={},
        agentic_stats={},
        metadata=metadata or {},
    )


def state_to_dict(state: SurveyState) -> dict[str, Any]:
    """Serialize the full state to a plain JSON-serializable dict.

    Useful for checkpoint saving and debugging.
    """
    result: dict[str, Any] = {}
    for key in SurveyState.__annotations__:
        val = state.get(key)  # type: ignore[attr-defined]
        if val is None:
            result[key] = None
        elif isinstance(val, datetime):
            result[key] = val.isoformat()
        else:
            result[key] = val
    return result


def state_from_dict(data: dict[str, Any]) -> SurveyState:
    """Deserialize a plain dict into a SurveyState.

    Restores default values for any missing keys.
    """
    defaults = create_initial_state()
    for key in SurveyState.__annotations__:
        if key not in data or data[key] is None:
            data[key] = defaults[key]  # type: ignore[literal-required]
    return SurveyState(**data)  # type: ignore[typeddict-item]


# ============================================================================
# Convenience — extract typed models from raw state dicts
# ============================================================================

def get_outline_from_state(state: SurveyState) -> list[SectionOutline]:
    """Extract SectionOutline objects from the raw state dict list."""
    return [SectionOutline.from_dict(d) for d in (state.get("outline") or [])]  # type: ignore[attr-defined]


def get_sections_from_state(state: SurveyState) -> dict[str, SectionDraft]:
    """Extract SectionDraft objects from the raw state sections dict."""
    raw = state.get("sections") or {}  # type: ignore[attr-defined]
    return {k: SectionDraft.from_dict(v) if isinstance(v, dict) else v for k, v in raw.items()}


def get_verifications_from_state(state: SurveyState) -> dict[str, VerificationResult]:
    """Extract VerificationResult objects from the raw state dict."""
    raw = state.get("verification_results") or {}  # type: ignore[attr-defined]
    return {k: VerificationResult.from_dict(v) if isinstance(v, dict) else v for k, v in raw.items()}


def get_l2_summaries_from_state(state: SurveyState) -> list[L2Summary]:
    """Extract L2Summary objects from the raw state list."""
    return [L2Summary(**d) for d in (state.get("l2_summaries") or [])]  # type: ignore[attr-defined]


def get_version_from_state(state: SurveyState) -> SurveyVersion:
    """Extract SurveyVersion from the raw state dict."""
    raw = state.get("version_info") or {}  # type: ignore[attr-defined]
    if isinstance(raw, dict):
        if "timestamp" in raw and isinstance(raw["timestamp"], str):
            raw["timestamp"] = datetime.fromisoformat(raw["timestamp"])
        return SurveyVersion(**raw)
    return SurveyVersion()


def get_provenance_from_state(state: SurveyState) -> list[ProvenanceRecord]:
    """Extract ProvenanceRecord objects from the raw state list."""
    return [ProvenanceRecord(**d) for d in (state.get("provenance") or [])]  # type: ignore[attr-defined]


def get_bibliography_keys(state: SurveyState) -> set[str]:
    """Get the set of BibTeX keys used in the final paper."""
    return set(state.get("bibliography") or [])  # type: ignore[attr-defined]


def collect_all_citation_keys(state: SurveyState) -> set[str]:
    """Collect all unique BibTeX keys from all section drafts."""
    keys: set[str] = set()
    sections = get_sections_from_state(state)
    for draft in sections.values():
        keys.update(draft.unique_citation_keys())
    return keys


def reset_scratchpad(
    state: SurveyState,
    section_key: str = "",
) -> dict[str, Any]:
    """Reset the scratchpad and prepare retrieved_facts for a new section.

    Clears ``current_scratchpad`` and ensures ``retrieved_facts`` has an
    empty list for *section_key* (existing facts for other sections are
    preserved thanks to the ``_merge_retrieved_facts`` reducer).

    Returns a partial state dict suitable for LangGraph updates.
    """
    update: dict[str, Any] = {
        "current_scratchpad": [],
    }
    if section_key:
        update["retrieved_facts"] = {section_key: []}
    return update


def get_pipeline_summary(state: SurveyState) -> dict[str, Any]:
    """Generate a summary dict of the pipeline state for logging/display."""
    sections = get_sections_from_state(state)
    verifications = get_verifications_from_state(state)
    outline = get_outline_from_state(state)

    section_statuses: dict[str, str] = {}
    for key, draft in sections.items():
        section_statuses[key] = draft.status.value if isinstance(draft.status, SectionStatus) else draft.status

    verification_summary: dict[str, dict[str, Any]] = {}
    for key, vr in verifications.items():
        verification_summary[key] = {
            "total": vr.total_citations,
            "supported": vr.supported,
            "pass_rate": round(vr.pass_rate, 3),
            "passed": vr.passed,
            "iteration": vr.iteration,
        }

    all_keys = collect_all_citation_keys(state)

    # Agentic stats
    retrieved_facts = state.get("retrieved_facts") or {}  # type: ignore[attr-defined]
    agentic_stats = state.get("agentic_stats") or {}  # type: ignore[attr-defined]
    total_tool_calls = sum(
        len(entries) for entries in retrieved_facts.values()
    )

    return {
        "current_phase": state.get("current_phase", "unknown"),  # type: ignore[attr-defined]
        "current_section": state.get("current_section_key", ""),  # type: ignore[attr-defined]
        "iteration_count": state.get("iteration_count", 0),  # type: ignore[attr-defined]
        "outline_sections": len(outline),
        "drafted_sections": len(sections),
        "section_statuses": section_statuses,
        "verification_summary": verification_summary,
        "total_unique_citations": len(all_keys),
        "provenance_records": len(state.get("provenance") or []),  # type: ignore[attr-defined]
        "has_final_paper": bool(state.get("final_paper")),  # type: ignore[attr-defined]
        "errors": state.get("errors", []),  # type: ignore[attr-defined]
        "agentic_tool_calls": total_tool_calls,
        "agentic_stats": agentic_stats,
    }
