"""
Pydantic models for data validation in PaperHunter.

Modelli di dominio per l'intero workflow: ingestion, summarization,
indexing, agentic writing, post-processing.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


# ============================================================================
# Enums
# ============================================================================

class SourceType(str, Enum):
    """Enum representing the source of a paper."""
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    HUGGINGFACE = "huggingface"


class ChunkType(str, Enum):
    """Tipo di chunk nel vector store."""
    ORIGINAL = "original"          # Testo originale dal PDF
    L1_SUMMARY = "l1_summary"      # Summary L1 singolo paper
    L2_SUMMARY = "l2_summary"      # Meta-summary L2 per categoria


class VerificationStatus(str, Enum):
    """Stato di verifica di una citazione."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"


# ============================================================================
# Paper (esteso per il nuovo workflow)
# ============================================================================

class Paper(BaseModel):
    """Represents a scientific paper with its metadata.

    Esteso con campi per: GROBID processing, S2 enrichment,
    BibTeX generation, categorizzazione, ingestion tracking.
    """

    title: str = Field(..., description="The title of the paper")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field(default="", description="The paper's abstract")
    url: str = Field(..., description="URL to the paper's page")
    pdf_url: Optional[str] = Field(None, description="Direct URL to the PDF")
    source: SourceType = Field(..., description="Source where the paper was found")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    arxiv_id: Optional[str] = Field(None, description="ArXiv ID if available")
    doi: Optional[str] = Field(None, description="DOI if available")
    github_url: Optional[str] = Field(None, description="Associated GitHub repository URL")
    categories: list[str] = Field(default_factory=list, description="Paper categories/tags")
    citations: Optional[int] = Field(None, description="Number of citations")

    # --- Nuovi campi (v2.0) ---
    venue: Optional[str] = Field(None, description="Venue di pubblicazione (es. NeurIPS, ACL)")
    journal: Optional[str] = Field(None, description="Journal se applicabile")
    year: Optional[int] = Field(None, description="Anno di pubblicazione")
    month: Optional[int] = Field(None, ge=1, le=12, description="Mese di pubblicazione")
    citation_velocity: Optional[float] = Field(None, description="Citazioni per anno (calcolato)")
    influential_citation_count: Optional[int] = Field(None, description="Citazioni influenti (S2)")
    fields_of_study: list[str] = Field(default_factory=list, description="Campi di studio (S2)")

    # Semantic Scholar
    semantic_scholar_id: Optional[str] = Field(None, description="S2 paper ID")

    # BibTeX
    bibtex_key: Optional[str] = Field(None, description="Chiave BibTeX (es. Zou_2023_GCG)")
    bibtex_entry: Optional[str] = Field(None, description="Entry BibTeX completa")

    # Ingestion tracking
    pdf_path: Optional[str] = Field(None, description="Percorso locale del PDF")
    pdf_hash: Optional[str] = Field(None, description="SHA-256 hash del PDF")
    category: Optional[str] = Field(None, description="Categoria assegnata (es. prompt_injection)")
    grobid_processed: bool = Field(False, description="Se il PDF è stato processato da GROBID")
    ingested_at: Optional[datetime] = Field(None, description="Timestamp dell'ingestion")

    class Config:
        use_enum_values = True


# ============================================================================
# GROBID Output
# ============================================================================

class GROBIDSection(BaseModel):
    """Una sezione estratta da GROBID."""
    title: str = Field("", description="Titolo della sezione")
    text: str = Field("", description="Testo della sezione")
    level: int = Field(1, ge=1, le=4, description="Livello heading (1=sezione, 2=sotto-sezione)")


class GROBIDReference(BaseModel):
    """Un riferimento bibliografico estratto da GROBID."""
    id: Optional[str] = Field(None, description="ID del riferimento nel TEI-XML")
    title: Optional[str] = Field(None, description="Titolo del paper citato")
    authors: list[str] = Field(default_factory=list, description="Autori del paper citato")
    year: Optional[int] = Field(None, description="Anno di pubblicazione")
    venue: Optional[str] = Field(None, description="Venue/journal")
    doi: Optional[str] = Field(None, description="DOI")
    raw_text: Optional[str] = Field(None, description="Testo grezzo della citazione")


class GROBIDOutput(BaseModel):
    """Output strutturato da GROBID per un singolo PDF.

    Contiene il testo estratto, suddiviso in sezioni, con
    riferimenti bibliografici, figure captions e tabelle.
    """
    paper_id: Optional[str] = Field(None, description="ID del paper associato")
    title: str = Field("", description="Titolo estratto dall'header")
    authors: list[str] = Field(default_factory=list, description="Autori estratti")
    abstract: str = Field("", description="Abstract estratto")
    sections: list[GROBIDSection] = Field(default_factory=list, description="Sezioni del paper")
    references: list[GROBIDReference] = Field(default_factory=list, description="Riferimenti bibliografici")
    figure_captions: list[str] = Field(default_factory=list, description="Didascalie delle figure")
    table_captions: list[str] = Field(default_factory=list, description="Didascalie delle tabelle")
    full_text: str = Field("", description="Testo completo (fallback)")
    processing_time: Optional[float] = Field(None, description="Tempo di processing GROBID (secondi)")
    success: bool = Field(True, description="Se il processing è andato a buon fine")
    error_message: Optional[str] = Field(None, description="Errore se success=False")


# ============================================================================
# PaperChunk (per vector store)
# ============================================================================

class PaperChunk(BaseModel):
    """Chunk di testo indicizzato nel vector store.

    Ogni chunk ha metadata ricchi per il retrieval ibrido
    con filtri per categoria, anno, tipo.
    """
    chunk_id: str = Field(..., description="ID univoco del chunk (hash di bibtex_key::section::index)")
    text: str = Field(..., description="Contenuto testuale del chunk")
    chunk_type: ChunkType = Field(ChunkType.ORIGINAL, description="Tipo: original, l1_summary, l2_summary")

    # Metadata per retrieval
    paper_id: Optional[str] = Field(None, description="ID del paper di origine")
    bibtex_key: Optional[str] = Field(None, description="Chiave BibTeX del paper")
    section_name: Optional[str] = Field(None, description="Nome della sezione di origine")
    category: Optional[str] = Field(None, description="Categoria del paper")
    year: Optional[int] = Field(None, description="Anno di pubblicazione")
    month: Optional[int] = Field(None, description="Mese di pubblicazione")
    chunk_index: int = Field(0, description="Indice del chunk nella sezione")

    # Embedding (opzionale, calcolato separatamente)
    embedding: Optional[list[float]] = Field(None, exclude=True, description="Vettore embedding")

    def to_metadata_dict(self) -> dict[str, Any]:
        """Converte i metadata in un dict per il vector store."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value if isinstance(self.chunk_type, ChunkType) else self.chunk_type,
            "paper_id": self.paper_id or "",
            "bibtex_key": self.bibtex_key or "",
            "section_name": self.section_name or "",
            "category": self.category or "",
            "year": self.year or 0,
            "month": self.month or 0,
            "chunk_index": self.chunk_index,
        }


# ============================================================================
# Summarization (L1 e L2)
# ============================================================================

class L1Summary(BaseModel):
    """Summary L1 — singolo paper.

    Estratto da Gemini Pro con prompt strutturato.
    Contiene obbligatoriamente la testing_methodology.
    """
    paper_id: str = Field(..., description="ID del paper originale")
    bibtex_key: Optional[str] = Field(None, description="Chiave BibTeX")
    category: Optional[str] = Field(None, description="Categoria del paper")
    year: Optional[int] = Field(None, description="Anno di pubblicazione")

    key_contributions: list[str] = Field(
        default_factory=list,
        description="Contributi principali (max 5 bullet points)"
    )
    methodology: str = Field("", description="Approccio tecnico dettagliato")
    testing_methodology: str = Field(
        "",
        description="CAMPO CRITICO: benchmark, dataset, metriche, prompt usati per testare"
    )
    temporal_context: str = Field(
        "",
        description="Quando proposto, differenze col passato, posizione nella timeline SOTA"
    )
    results_summary: str = Field("", description="Risultati quantitativi principali")
    limitations: str = Field("", description="Limiti dichiarati o impliciti")

    generated_at: Optional[datetime] = Field(None, description="Timestamp di generazione")
    model_used: Optional[str] = Field(None, description="Modello Gemini usato (flash/pro)")


class L2Summary(BaseModel):
    """Summary L2 — meta-summary per categoria.

    Sintetizza tutti gli L1 di una categoria, evidenziando
    trend temporali, gap, e convergenze metodologiche.
    """
    category: str = Field(..., description="Categoria (es. prompt_injection)")
    paper_ids: list[str] = Field(default_factory=list, description="ID dei paper sintetizzati")
    paper_count: int = Field(0, description="Numero di paper sintetizzati")
    version: int = Field(1, description="Versione del summary (incrementale)")

    temporal_trends: str = Field("", description="Evoluzione temporale (es. 2019→2026)")
    taxonomy: str = Field("", description="Tassonomia delle sotto-aree")
    gaps_filled: str = Field("", description="Gap colmati dalla letteratura")
    open_gaps: str = Field("", description="Gap ancora aperti")
    methodological_convergences: str = Field("", description="Convergenze e divergenze metodologiche")
    testing_overview: str = Field("", description="Panoramica delle testing methodologies più usate")
    synthesis: str = Field("", description="Sintesi complessiva della categoria")

    invalidated: bool = Field(False, description="Se il summary è stato invalidato da nuovi paper")
    generated_at: Optional[datetime] = Field(None, description="Timestamp di generazione")
    model_used: Optional[str] = Field(None, description="Modello Gemini usato")


# ============================================================================
# Provenance & Verification
# ============================================================================

class ProvenanceRecord(BaseModel):
    """Record di provenance per una singola citazione nel survey.

    Mappa ogni claim nel paper generato alla sua fonte originale
    nel corpus di paper, con stato di verifica.
    """
    section: str = Field("", description="Sezione del survey in cui appare il claim")
    claim_text: str = Field("", description="Testo del claim nel survey")
    citation_key: str = Field("", description="Chiave BibTeX citata (es. Zou_2023_GCG)")
    source_chunk_id: Optional[str] = Field(None, description="ID del chunk sorgente")
    source_text_excerpt: str = Field("", description="Estratto del testo sorgente")
    verification_status: VerificationStatus = Field(
        VerificationStatus.SOURCE_NOT_FOUND,
        description="Stato di verifica della citazione"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Confidenza della verifica (0-1)"
    )


# ============================================================================
# Survey Versioning
# ============================================================================

class SurveyVersion(BaseModel):
    """Metadata di una versione del survey generato."""
    version_number: str = Field("1.0", description="Numero versione (es. 1.0, 1.1)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp di generazione")
    paper_count: int = Field(0, description="Numero totale di paper nel corpus")
    categories: list[str] = Field(default_factory=list, description="Categorie coperte")
    changelog: str = Field("", description="Descrizione delle modifiche dalla versione precedente")
    l1_count: int = Field(0, description="Numero di L1 summary generati")
    l2_count: int = Field(0, description="Numero di L2 summary generati")
    sections_count: int = Field(0, description="Numero di sezioni nel survey")
    total_citations: int = Field(0, description="Numero totale di citazioni nel survey")
    verified_citations: int = Field(0, description="Citazioni verificate con provenance")


# ============================================================================
# Modelli legacy (mantenuti per backward compatibility)
# ============================================================================

class RelevanceResult(BaseModel):
    """Result of LLM relevance analysis."""

    is_relevant: bool = Field(..., description="Whether the paper is relevant to the criteria")
    reasoning: str = Field(..., description="Explanation for the relevance decision")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the relevance decision (0-1)"
    )


class FilteredPaper(BaseModel):
    """A paper that has been processed through the relevance filter."""

    paper: Paper = Field(..., description="The original paper")
    relevance: RelevanceResult = Field(..., description="The relevance analysis result")


class DownloadResult(BaseModel):
    """Result of a download attempt."""

    success: bool = Field(..., description="Whether the download was successful")
    file_path: Optional[str] = Field(None, description="Path to the downloaded file")
    method: Optional[str] = Field(None, description="Method used for download (direct/playwright)")
    error_message: Optional[str] = Field(None, description="Error message if download failed")


class SearchQuery(BaseModel):
    """Search query configuration."""

    keywords: str = Field(..., description="Broad search keywords")
    filtering_criteria: str = Field(..., description="Specific filtering criteria for LLM")
    max_results_per_source: int = Field(default=20, ge=1, le=100, description="Max results per source")
    categories: list[str] = Field(
        default_factory=lambda: ["cs.CL", "cs.AI"],
        description="ArXiv categories to search"
    )


class SearchResult(BaseModel):
    """Aggregated search results from all sources."""

    papers: list[Paper] = Field(default_factory=list, description="List of found papers")
    total_found: int = Field(default=0, description="Total number of papers found")
    sources_queried: list[str] = Field(default_factory=list, description="Sources that were queried")
    errors: list[str] = Field(default_factory=list, description="Errors encountered during search")
