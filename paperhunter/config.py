"""
Configuration settings for PaperHunter / LLM Security Survey Generator.

Architettura a due livelli:
- config.yaml  → tutti i parametri applicativi (modelli, chunk size, batch size, ecc.)
- .env         → solo API keys e segreti (mai committati)

Uso:
    from paperhunter.config import settings
    print(settings.gemini.flash_model)
    print(settings.google_api_key)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# =============================================================================
# Sotto-modelli Pydantic per ogni sezione del config.yaml
# =============================================================================


class GeminiConfig(BaseModel):
    """Configurazione modelli Gemini (LLM + Embedding)."""
    flash_model: str = "gemini-3-flash-preview"
    pro_model: str = "gemini-3.1-pro-preview"
    embedding_model: str = "gemini-embedding-2-preview"
    temperature_flash: float = 0.1
    temperature_pro: float = 0.3
    max_output_tokens_flash: int = 4096
    max_output_tokens_pro: int = 8192
    timeout: int = 120
    max_retries: int = 3
    retry_strategy: str = "autonomous"  # "autonomous" | "interactive"

    @field_validator("retry_strategy")
    @classmethod
    def validate_retry_strategy(cls, v: str) -> str:
        allowed = {"autonomous", "interactive"}
        if v not in allowed:
            raise ValueError(
                f"retry_strategy deve essere uno di {allowed}, ricevuto: {v}"
            )
        return v


class GrobidConfig(BaseModel):
    """Configurazione GROBID Docker per estrazione PDF strutturata."""
    base_url: str = "http://localhost:8070"
    timeout: int = 120
    batch_size: int = 10
    consolidate_header: bool = True
    consolidate_citations: bool = True
    include_raw_citations: bool = True


class Neo4jConfig(BaseModel):
    """Configurazione Neo4j Community Edition (opzionale, per GraphRAG)."""
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class FigureExtractionConfig(BaseModel):
    """Configurazione estrazione e descrizione figure dai PDF."""
    enabled: bool = True
    vision_model: str = "gemini-2.5-flash"  # Modello con supporto Vision
    min_width: int = 150         # Larghezza minima immagine in pixel
    min_height: int = 150        # Altezza minima immagine in pixel
    max_figures_per_paper: int = 10  # Massimo figure processate per paper
    figures_dir: str = "./output/figures"  # Directory per salvare le immagini estratte


class ImageFilteringConfig(BaseModel):
    """Configurazione filtraggio intelligente immagini (pipeline a 3 stadi)."""
    enabled: bool = True
    # --- Filtro Euristico (Stage 1) ---
    min_aspect_ratio: float = 0.3
    max_aspect_ratio: float = 4.0
    min_color_entropy: float = 3.0
    min_edge_density: float = 0.05
    skip_first_page: bool = True
    skip_last_page: bool = True
    relevance_threshold: float = 0.5
    # --- Filtro Embedding Multimodale (Stage 2) ---
    embedding_filter_enabled: bool = True
    embedding_similarity_threshold: float = 0.3

    @field_validator("min_aspect_ratio")
    @classmethod
    def validate_min_aspect_ratio(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"min_aspect_ratio deve essere in (0.0, 1.0], ricevuto: {v}")
        return v

    @field_validator("max_aspect_ratio")
    @classmethod
    def validate_max_aspect_ratio(cls, v: float) -> float:
        if not (1.0 <= v <= 20.0):
            raise ValueError(f"max_aspect_ratio deve essere in [1.0, 20.0], ricevuto: {v}")
        return v

    @field_validator("min_color_entropy")
    @classmethod
    def validate_min_color_entropy(cls, v: float) -> float:
        if not (0.0 <= v <= 8.0):
            raise ValueError(f"min_color_entropy deve essere in [0.0, 8.0], ricevuto: {v}")
        return v

    @field_validator("min_edge_density")
    @classmethod
    def validate_min_edge_density(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_edge_density deve essere in [0.0, 1.0], ricevuto: {v}")
        return v

    @field_validator("relevance_threshold")
    @classmethod
    def validate_relevance_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"relevance_threshold deve essere in [0.0, 1.0], ricevuto: {v}")
        return v

    @field_validator("embedding_similarity_threshold")
    @classmethod
    def validate_embedding_similarity_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"embedding_similarity_threshold deve essere in [0.0, 1.0], ricevuto: {v}")
        return v


class QdrantConfig(BaseModel):
    """Configurazione Qdrant per vector store (locale, senza server)."""
    persist_dir: str = "./qdrant_db"
    collection_paper_chunks: str = "paper_chunks"
    collection_l1_summaries: str = "l1_summaries"
    collection_l2_summaries: str = "l2_summaries"
    collection_l1_5_summaries: str = "l1_5_summaries"
    collection_graph_communities: str = "graph_communities"
    collection_figure_descriptions: str = "figure_descriptions"
    collection_image_embeddings: str = "image_embeddings"
    embedding_dimension: int = 3072
    distance_metric: str = "Cosine"

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        allowed = {"Cosine", "Euclid", "Dot"}
        if v not in allowed:
            raise ValueError(f"distance_metric deve essere uno di {allowed}, ricevuto: {v}")
        return v


class SemanticScholarConfig(BaseModel):
    """Configurazione Semantic Scholar Academic Graph API."""
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    rate_limit_rpm: int = 100
    rate_limit_rpm_authenticated: int = 100
    fields: str = "title,authors,year,venue,journal,citationCount,influentialCitationCount,fieldsOfStudy,publicationDate,externalIds,abstract,citationStyles"
    search_fields: str = "title,authors,year,venue,externalIds,abstract,citationCount"
    max_retries: int = 3
    timeout: int = 30
    cache_dir: str = "./cache/semantic_scholar"


class IngestionConfig(BaseModel):
    """Configurazione pipeline di ingestion incrementale."""
    pdf_base_dir: str = "./scaricati"
    hash_registry_path: str = "./data/hash_registry.json"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    batch_size: int = 10
    fallback_to_pymupdf: bool = True
    use_best_fit_packing: bool = True  # Best-fit bin packing post-segmentazione GROBID
    papers_per_category: int = 100  # Numero target di paper per categoria (usato per budget)
    category_mapping: dict[str, str] = Field(default_factory=lambda: {
        "Prompt_injections1": "Prompt Injection",
        "Hallucinations1": "LLM Hallucinations",
        "Data_and_Model_Poisoning1": "Data and Model Poisoning",
        "Adversarial_Code1": "Adversarial Attacks",
    })


class L2RecencyTiersConfig(BaseModel):
    """Soglie di compressione per recency nei summary L2 (anni fa)."""
    full: int = 2       # Paper ≤ full anni: testo L1 completo
    reduced: int = 4    # Paper full+1 → reduced anni: ~70% del testo
    compressed: int = 999  # Paper > reduced anni: solo key_contributions + testing


class ClusteringConfig(BaseModel):
    """Configurazione clustering semantico RAPTOR (L1 → L1.5)."""
    max_cluster_size: int = 20
    similarity_threshold: float = 0.75

    @field_validator("max_cluster_size")
    @classmethod
    def validate_max_cluster_size(cls, v: int) -> int:
        if not (5 <= v <= 100):
            raise ValueError(
                f"max_cluster_size deve essere tra 5 e 100, ricevuto: {v}"
            )
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(
                f"similarity_threshold deve essere tra 0.0 e 1.0 (esclusi), ricevuto: {v}"
            )
        return v


class SummarizationConfig(BaseModel):
    """Configurazione hierarchical summarization (L1 + L2)."""
    l1_max_tokens: int = 2000
    l2_max_tokens: int = 6000
    l2_invalidation_threshold: int = 1
    use_pro_for_l1: bool = True
    use_pro_for_l2: bool = True
    allow_flash_fallback: bool = False  # Se False, non degrada mai a Flash per task complessi
    pro_max_retries: int = 6           # Retry pazienti per Pro (con delay lunghi)
    pro_base_delay: float = 15.0       # Delay base tra retry Pro (secondi)
    max_context_tokens: int = 150_000  # Budget token massimo per singola chiamata LLM
    l1_target_chars_auto: bool = True  # Se True, calcola target L1 chars in base a papers_per_category
    l1_min_chars: int = 2000           # Floor: un L1 non sarà mai più corto di così
    l2_recency_tiers: L2RecencyTiersConfig = Field(default_factory=L2RecencyTiersConfig)


class VenueRankingConfig(BaseModel):
    """Configurazione venue ranking post-RRF nel HybridRetriever."""
    enabled: bool = True
    tier1_boost: float = 0.15    # Boost per Tier 1 (NeurIPS, ICML, ACL, ...)
    tier2_boost: float = 0.05    # Boost per Tier 2 (workshop, AAAI, ...)
    tier3_penalty: float = 0.05  # Penalità per Tier 3 (arXiv preprint, ...)


class RetrievalConfig(BaseModel):
    """Configurazione hybrid retrieval (vector + BM25 + GraphRAG)."""
    vector_weight: float = 0.6
    bm25_weight: float = 0.3
    graph_weight: float = 0.1
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 10
    rerank_batch_size: int = 20
    bm25_index_path: str = "./data/bm25_index.pkl"
    recency_boost: float = 0.1  # Boost temporale post-RRF (0.0 = disabilitato)
    venue_ranking_enabled: bool = True
    venue_tier1_boost: float = 0.15
    venue_tier2_boost: float = 0.05
    venue_tier3_penalty: float = 0.05
    venue_ranking: VenueRankingConfig = Field(default_factory=VenueRankingConfig)


class AgentsConfig(BaseModel):
    """Configurazione pipeline multi-agente (LangGraph)."""
    max_writer_iterations: int = 3
    min_section_words: int = 500
    max_section_words: int = 3000
    verification_threshold: float = 0.80
    planner_model: str = "pro"
    writer_model: str = "pro"
    critic_model: str = "flash"
    assembler_model: str = "pro"
    writer_max_context_chars: int = 60_000  # Max chars contesto per singola sezione writer
    writer_budget_mode: str = "proportional"  # "fixed" | "proportional"
    writer_mode: str = "agentic"  # "static" (RAG pre-calcolato) | "agentic" (ReAct multi-step)
    force_rewrite: bool = False  # Se True, riscrive tutto ignorando i checkpoint per-sezione

    @field_validator("planner_model", "writer_model", "critic_model", "assembler_model")
    @classmethod
    def validate_model_choice(cls, v: str) -> str:
        allowed = {"flash", "pro"}
        if v not in allowed:
            raise ValueError(f"Modello deve essere 'flash' o 'pro', ricevuto: {v}")
        return v

    @field_validator("writer_budget_mode")
    @classmethod
    def validate_writer_budget_mode(cls, v: str) -> str:
        allowed = {"fixed", "proportional"}
        if v not in allowed:
            raise ValueError(f"writer_budget_mode deve essere uno di {allowed}, ricevuto: {v}")
        return v

    @field_validator("writer_mode")
    @classmethod
    def validate_writer_mode(cls, v: str) -> str:
        allowed = {"static", "agentic", "prefetch_agentic"}
        if v not in allowed:
            raise ValueError(f"writer_mode deve essere 'static' o 'agentic', ricevuto: {v}")
        return v


class OutputConfig(BaseModel):
    """Configurazione output e post-processing."""
    base_dir: str = "./output"
    versions_dir: str = "./output/versions"
    checkpoints_dir: str = "./output/.checkpoints"
    provenance_dir: str = "./output/provenance"
    latex_template: str = "acl"
    generate_latex: bool = True
    generate_bibtex: bool = True
    generate_provenance_json: bool = True

    @field_validator("latex_template")
    @classmethod
    def validate_latex_template(cls, v: str) -> str:
        allowed = {"acl", "ieeetran", "plain"}
        if v not in allowed:
            raise ValueError(f"latex_template deve essere uno di {allowed}, ricevuto: {v}")
        return v


class SearchConfig(BaseModel):
    """Configurazione ricerca paper (legacy PaperHunter)."""
    max_results_per_source: int = 20
    default_categories: list[str] = Field(default_factory=lambda: ["cs.CL", "cs.AI", "cs.CR", "cs.LG"])


class DownloadConfig(BaseModel):
    """Configurazione download PDF."""
    download_dir: str = "./scaricati"
    timeout: int = 30
    headless: bool = True
    request_delay: float = 1.0
    playwright_delay: float = 2.0


class RateLimitingConfig(BaseModel):
    """Configurazione rate limiting globale per tutte le API."""
    gemini_rpm: int = 15
    gemini_daily_limit: int = 1500
    semantic_scholar_rpm: int = 100
    grobid_concurrent: int = 5


class OpenRouterConfig(BaseModel):
    """Configurazione OpenRouter come fallback per Google Gemini.

    OpenRouter è un API gateway OpenAI-compatibile che instrada a vari
    provider (Anthropic, Google, Meta, ecc.). Usato come fallback quando
    tutte le retry con l'API Google diretta falliscono.
    """
    enabled: bool = False  # Se True, fallback a OpenRouter quando Google fallisce
    api_base: str = "https://openrouter.ai/api/v1"
    flash_model: str = "google/gemini-3-flash-preview"  # Equivalente Flash economico
    pro_model: str = "google/gemini-3.1-pro-preview"  # Equivalente Pro potente
    temperature_flash: float = 0.1
    temperature_pro: float = 0.3
    max_output_tokens_flash: int = 4096
    max_output_tokens_pro: int = 8192
    timeout: int = 300  # Timeout in secondi
    max_retries: int = 3  # Retry con OpenRouter


class EvolutionConfig(BaseModel):
    """Configurazione Self-Evolution System (lessons from failures)."""
    enabled: bool = True
    lessons_path: str = "./data/evolution/lessons.jsonl"
    half_life_days: float = 30.0
    expiry_days: float = 90.0
    max_overlay_lessons: int = 5
    max_overlay_chars: int = 2000
    auto_prune: bool = True


class FigureAgentConfig(BaseModel):
    """Configurazione FigureAgent (generazione automatica figure)."""
    enabled: bool = True
    min_figures: int = 3
    max_figures: int = 8
    max_iterations: int = 3
    output_format: str = "markdown"
    dpi: int = 300
    style: str = "seaborn-v0_8-paper"
    figures_dir: str = "./output/generated_figures"


class SkillsConfig(BaseModel):
    """Configurazione Skill Injection (prompt overlay persistenti)."""
    enabled: bool = True
    skills_dir: str = "./data/skills"
    max_total_chars: int = 4000


class LoggingConfig(BaseModel):
    """Configurazione logging (rich + tqdm)."""
    level: str = "INFO"
    rich_enabled: bool = True
    log_file: str = "./logs/paperhunter.log"
    show_progress_bars: bool = True

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Log level deve essere uno di {allowed}, ricevuto: {v}")
        return v_upper


# =============================================================================
# Settings principale — unisce YAML + .env
# =============================================================================

def _find_project_root() -> Path:
    """Risali le directory fino a trovare config.yaml o pyproject.toml."""
    current = Path(__file__).resolve().parent  # paperhunter/
    for parent in [current, current.parent] + list(current.parent.parents):
        if (parent / "config.yaml").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current.parent  # Fallback: directory padre del package


def _load_yaml_config(config_path: Optional[Path] = None) -> dict:
    """Carica la configurazione dal file YAML.
    
    Args:
        config_path: Percorso esplicito al file YAML. Se None, cerca config.yaml
                     nella root del progetto.
    
    Returns:
        Dizionario con la configurazione. Vuoto se il file non esiste.
    """
    if config_path is None:
        config_path = _find_project_root() / "config.yaml"
    
    if not config_path.exists():
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return data if isinstance(data, dict) else {}


class Settings(BaseSettings):
    """Configurazione centralizzata dell'applicazione.
    
    Carica da due fonti:
    - config.yaml  → parametri applicativi (sotto-modelli tipizzati)
    - .env          → API keys e segreti
    
    Uso:
        from paperhunter.config import settings
        
        # Accesso a sotto-sezioni
        settings.gemini.flash_model        # → "gemini-3-flash-preview"
        settings.ingestion.chunk_size      # → 1500
        settings.chroma.persist_dir        # → "./chroma_db"
        
        # API keys (da .env)
        settings.google_api_key            # → "AIza..."
        
        # Backward compatibility
        settings.llm_model                 # → alias per gemini.flash_model
    """
    
    # --- Priorità Recency (globale) ---
    recency_priority: bool = True  # Se True, paper recenti hanno priorità in ingestion, retrieval, L2

    # --- API Keys (da .env) ---
    google_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    semantic_scholar_api_key: str = ""
    
    # --- Sotto-configurazioni (da config.yaml) ---
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    grobid: GrobidConfig = Field(default_factory=GrobidConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    semantic_scholar: SemanticScholarConfig = Field(default_factory=SemanticScholarConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    figure_extraction: FigureExtractionConfig = Field(default_factory=FigureExtractionConfig)
    image_filtering: ImageFilteringConfig = Field(default_factory=ImageFilteringConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    figure_agent: FigureAgentConfig = Field(default_factory=FigureAgentConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # Ignora campi extra nel .env
    }
    
    # --- Proprietà backward-compatible ---
    
    @property
    def llm_provider(self) -> str:
        """Provider LLM (sempre 'google' nel nuovo sistema)."""
        return "google"
    
    @property
    def llm_model(self) -> str:
        """Modello LLM veloce (Flash). Backward compatible."""
        return self.gemini.flash_model
    
    @property
    def heavy_llm_model(self) -> str:
        """Modello LLM potente (Pro). Backward compatible."""
        return self.gemini.pro_model
    
    @property
    def llm_temperature(self) -> float:
        """Temperatura default (Flash). Backward compatible."""
        return self.gemini.temperature_flash
    
    @property
    def max_results_per_source(self) -> int:
        """Max risultati per fonte di ricerca. Backward compatible."""
        return self.search.max_results_per_source
    
    @property
    def default_categories(self) -> list[str]:
        """Categorie ArXiv di default. Backward compatible."""
        return self.search.default_categories
    
    @property
    def request_delay(self) -> float:
        """Delay tra richieste HTTP. Backward compatible."""
        return self.download.request_delay
    
    @property
    def playwright_delay(self) -> float:
        """Delay per operazioni Playwright. Backward compatible."""
        return self.download.playwright_delay
    
    @property
    def download_dir(self) -> Path:
        """Directory download. Backward compatible."""
        return Path(self.download.download_dir)
    
    @property
    def download_timeout(self) -> int:
        """Timeout download. Backward compatible."""
        return self.download.timeout
    
    @property
    def headless(self) -> bool:
        """Playwright headless mode. Backward compatible."""
        return self.download.headless


def _build_settings(config_path: Optional[Path] = None) -> Settings:
    """Costruisce l'istanza Settings unendo YAML + .env.
    
    Args:
        config_path: Percorso esplicito al config.yaml (opzionale).
    
    Returns:
        Istanza Settings completamente inizializzata.
    """
    yaml_data = _load_yaml_config(config_path)
    return Settings(**yaml_data)


# Istanza globale singleton
settings = _build_settings()


def reload_settings(config_path: Optional[Path] = None) -> Settings:
    """Ricarica la configurazione (utile per test o cambio config a runtime).
    
    Args:
        config_path: Percorso esplicito al config.yaml (opzionale).
    
    Returns:
        Nuova istanza Settings.
    """
    global settings
    settings = _build_settings(config_path)
    return settings


def ensure_download_dir() -> Path:
    """Assicura che la directory di download esista. Backward compatible."""
    settings.download_dir.mkdir(parents=True, exist_ok=True)
    return settings.download_dir


def ensure_output_dirs() -> dict[str, Path]:
    """Crea tutte le directory di output necessarie.
    
    Returns:
        Dizionario {nome: Path} delle directory create.
    """
    dirs = {
        "output": Path(settings.output.base_dir),
        "versions": Path(settings.output.versions_dir),
        "checkpoints": Path(settings.output.checkpoints_dir),
        "provenance": Path(settings.output.provenance_dir),
        "qdrant": Path(settings.qdrant.persist_dir),
        "s2_cache": Path(settings.semantic_scholar.cache_dir),
        "data": Path(settings.ingestion.hash_registry_path).parent,
        "logs": Path(settings.logging.log_file).parent if settings.logging.log_file else None,
    }
    
    created = {}
    for name, path in dirs.items():
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            created[name] = path
    
    return created
