"""
Hybrid Retriever per PaperHunter.

Combina ricerca vettoriale (Qdrant), keyword-based (BM25) e metadata
filtering in un unico retriever unificato. Usa Reciprocal Rank Fusion
(RRF) per fondere i ranking dei due sistemi.

Workflow di retrieval:
1. Query espande opzionalmente la query originale
2. Vector search via QdrantStore (embedding Gemini 3072-dim)
3. BM25 search via BM25Index (keyword matching)
4. Reciprocal Rank Fusion (RRF) per combinare i due ranking
5. Metadata filtering post-retrieval (opzionale, extra filtri)
6. Ritorna ``list[RetrievedChunk]`` con score e provenienza

Pesi configurabili da ``config.yaml`` → sezione ``retrieval``:
- ``vector_weight``: peso della ricerca vettoriale (default: 0.6)
- ``bm25_weight``: peso BM25 (default: 0.3)
- ``top_k_vector``: candidati dalla ricerca vettoriale (default: 20)
- ``top_k_bm25``: candidati dal BM25 (default: 20)
- ``top_k_final``: risultati finali dopo fusione (default: 10)

Uso::

    from paperhunter.indexing.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()

    # Ricerca ibrida base
    results = retriever.retrieve("prompt injection defense mechanisms")

    # Ricerca con filtri
    results = retriever.retrieve(
        query="backdoor attack detection",
        category_filter="Data and Model Poisoning",
        year_range=(2023, 2026),
        chunk_types=["original", "l1_summary"],
        n_results=15,
    )

    # Accesso ai risultati
    for r in results:
        print(r.chunk_id, r.score, r.retrieval_method, r.text[:100])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from paperhunter.utils.logging_setup import get_logger

# Anno di riferimento per il calcolo recency (pre-deep-learning era)
_RECENCY_BASELINE_YEAR = 2015

logger = get_logger(__name__)


# ===========================================================================
# Enums & Data Classes
# ===========================================================================

class RetrievalMethod(str, Enum):
    """Metodo di retrieval che ha prodotto il risultato."""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"       # Presente in entrambi i sistemi
    GRAPH = "graph"         # Reserved per GraphRAG futuro


@dataclass
class RetrievedChunk:
    """Chunk recuperato dal retriever ibrido.

    Contiene il testo, lo score combinato, i metadata e l'indicazione
    del metodo di retrieval che ha contribuito al risultato.

    Attributes:
        venue_tier: Tier di autorevolezza della venue (1=top, 2=medium, 3=preprint, 0=sconosciuta).
            Valorizzato durante il venue boost, None se il boost non è stato applicato.
        chunk_id: ID univoco del chunk.
        text: Contenuto testuale.
        score: Score combinato (RRF o singola fonte).
        retrieval_method: Metodo/i di retrieval che hanno trovato il chunk.
        metadata: Metadata completi del chunk (category, year, bibtex_key, ecc.).
        vector_score: Score dalla ricerca vettoriale (None se non presente).
        bm25_score: Score dal BM25 (None se non presente).
        vector_rank: Posizione nel ranking vettoriale (None se non presente).
        bm25_rank: Posizione nel ranking BM25 (None se non presente).
    """
    chunk_id: str
    text: str
    score: float
    retrieval_method: RetrievalMethod
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    venue_tier: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Serializza in dizionario."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "retrieval_method": self.retrieval_method.value,
            "metadata": self.metadata,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "venue_tier": self.venue_tier,
        }


# ===========================================================================
# Reciprocal Rank Fusion
# ===========================================================================

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (RRF) per combinare ranking multipli.

    Formula: RRF_score(d) = Σ_i  w_i / (k + rank_i(d))

    dove rank_i(d) è la posizione del documento d nel ranking i (1-based),
    w_i è il peso del ranking i, e k è il parametro di smoothing (default: 60).

    Args:
        ranked_lists: Lista di liste di risultati, ciascuna ordinata per
            score decrescente. Ogni risultato è un dict con almeno ``chunk_id``.
        weights: Pesi per ciascun ranking (stessa lunghezza di ranked_lists).
        k: Parametro di smoothing RRF (default: 60). Valori più alti
            penalizzano maggiormente i risultati in fondo alla lista.

    Returns:
        Lista di tuple ``(chunk_id, rrf_score)`` ordinate per score decrescente.

    Raises:
        ValueError: Se ``ranked_lists`` e ``weights`` hanno lunghezze diverse.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError(
            f"Numero di ranking ({len(ranked_lists)}) != "
            f"numero di pesi ({len(weights)})"
        )

    scores: dict[str, float] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank_0based, result in enumerate(ranked_list):
            chunk_id = result.get("chunk_id", "")
            if not chunk_id:
                continue
            rank_1based = rank_0based + 1
            rrf_contribution = weight / (k + rank_1based)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_contribution

    # Ordina per score decrescente
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


# ===========================================================================
# HybridRetriever
# ===========================================================================

class HybridRetriever:
    """Retriever ibrido che combina vector search + BM25 + metadata filtering.

    Esegue ricerca parallela su QdrantStore e BM25Index, fonde i risultati
    con Reciprocal Rank Fusion (RRF) e applica filtri metadata.

    Args:
        qdrant_store: Istanza QdrantStore. Se None, creata automaticamente.
        bm25_index: Istanza BM25Index. Se None, caricata da disco.
        vector_weight: Peso della ricerca vettoriale in RRF.
            Default da ``config.yaml`` → ``retrieval.vector_weight``.
        bm25_weight: Peso del BM25 in RRF.
            Default da ``config.yaml`` → ``retrieval.bm25_weight``.
        top_k_vector: Numero di candidati dalla ricerca vettoriale.
            Default da ``config.yaml`` → ``retrieval.top_k_vector``.
        top_k_bm25: Numero di candidati dal BM25.
            Default da ``config.yaml`` → ``retrieval.top_k_bm25``.
        top_k_final: Numero di risultati finali dopo fusione.
            Default da ``config.yaml`` → ``retrieval.top_k_final``.
        rrf_k: Parametro k per Reciprocal Rank Fusion (default: 60).

    Attributes:
        qdrant_store: Istanza QdrantStore in uso.
        bm25_index: Istanza BM25Index in uso.
    """

    def __init__(
        self,
        qdrant_store: Optional[Any] = None,
        bm25_index: Optional[Any] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        top_k_vector: Optional[int] = None,
        top_k_bm25: Optional[int] = None,
        top_k_final: Optional[int] = None,
        rrf_k: int = 60,
    ):
        from paperhunter.config import settings

        cfg = settings.retrieval

        self._vector_weight = vector_weight if vector_weight is not None else cfg.vector_weight
        self._bm25_weight = bm25_weight if bm25_weight is not None else cfg.bm25_weight
        self._top_k_vector = top_k_vector if top_k_vector is not None else cfg.top_k_vector
        self._top_k_bm25 = top_k_bm25 if top_k_bm25 is not None else cfg.top_k_bm25
        self._top_k_final = top_k_final if top_k_final is not None else cfg.top_k_final
        self._rrf_k = rrf_k
        self._recency_boost = cfg.recency_boost if settings.recency_priority else 0.0
        self._venue_ranking_enabled: bool = getattr(cfg, "venue_ranking_enabled", False)

        # Lazy init: store e index
        self._qdrant_store = qdrant_store
        self._qdrant_initialized = qdrant_store is not None
        self._bm25_index = bm25_index
        self._bm25_initialized = bm25_index is not None

        logger.info(
            "HybridRetriever inizializzato: "
            f"vector_weight={self._vector_weight}, "
            f"bm25_weight={self._bm25_weight}, "
            f"top_k_vector={self._top_k_vector}, "
            f"top_k_bm25={self._top_k_bm25}, "
            f"top_k_final={self._top_k_final}, "
            f"rrf_k={self._rrf_k}, "
            f"recency_boost={self._recency_boost}"
        )

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _get_qdrant_store(self):
        """Ottieni QdrantStore (lazy init).

        Returns:
            Istanza QdrantStore.
        """
        if not self._qdrant_initialized:
            from paperhunter.indexing.qdrant_store import QdrantStore
            self._qdrant_store = QdrantStore()
            self._qdrant_initialized = True
            logger.debug("QdrantStore inizializzato (lazy)")
        return self._qdrant_store

    def _get_bm25_index(self):
        """Ottieni BM25Index (lazy init, carica da disco se disponibile).

        Returns:
            Istanza BM25Index, oppure None se non disponibile/vuoto.
        """
        if not self._bm25_initialized:
            from paperhunter.indexing.bm25_index import BM25Index
            self._bm25_index = BM25Index()
            try:
                count = self._bm25_index.load()
                logger.info(f"BM25 indice caricato da disco: {count} chunk")
            except FileNotFoundError:
                logger.warning(
                    "BM25 indice non trovato su disco. "
                    "Retrieval sarà solo vettoriale."
                )
                self._bm25_index = None
            except Exception as e:
                logger.error(f"Errore caricamento BM25 index: {e}")
                self._bm25_index = None
            self._bm25_initialized = True
        return self._bm25_index

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Chiude la connessione QdrantStore, rilasciando il lock su disco."""
        if self._qdrant_store is not None:
            try:
                self._qdrant_store.close()
            except Exception:
                pass
            self._qdrant_store = None
            self._qdrant_initialized = False
            logger.debug("HybridRetriever: QdrantStore chiuso")

    # ------------------------------------------------------------------
    # Costruzione filtri
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where_filter(
        category_filter: Optional[str | list[str]] = None,
        year_range: Optional[tuple[int | None, int | None]] = None,
        chunk_types: Optional[list[str]] = None,
        bibtex_key: Optional[str] = None,
        paper_id: Optional[str] = None,
        extra_filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Costruisce un dizionario filtro unificato per Qdrant e BM25.

        Normalizza i parametri nominati in un dict compatibile con
        ``QdrantStore.query(where_filter=...)`` e
        ``BM25Index.query(where_filter=...)``.

        Args:
            category_filter: Categoria singola o lista di categorie.
            year_range: Tupla (min_year, max_year). None per unbounded.
            chunk_types: Lista di tipi chunk da includere (es. ["original", "l1_summary"]).
            bibtex_key: Filtro per chiave BibTeX specifica.
            paper_id: Filtro per paper_id specifico.
            extra_filters: Filtri aggiuntivi arbitrari.

        Returns:
            Dict di filtri. Vuoto se nessun filtro.
        """
        where: dict[str, Any] = {}

        # Categoria (singola stringa o lista)
        if category_filter is not None:
            if isinstance(category_filter, list):
                if len(category_filter) == 1:
                    where["category"] = category_filter[0]
                elif len(category_filter) > 1:
                    where["category"] = category_filter
            else:
                where["category"] = category_filter

        # Range anni → compatibile sia con Qdrant che BM25
        if year_range is not None:
            year_min, year_max = year_range
            if year_min is not None and year_max is not None:
                where["year"] = (year_min, year_max)
            elif year_min is not None:
                where["year"] = {"gte": year_min}
            elif year_max is not None:
                where["year"] = {"lte": year_max}

        # Tipi chunk
        if chunk_types is not None:
            if len(chunk_types) == 1:
                where["chunk_type"] = chunk_types[0]
            elif len(chunk_types) > 1:
                where["chunk_type"] = chunk_types

        # BibTeX key
        if bibtex_key is not None:
            where["bibtex_key"] = bibtex_key

        # Paper ID
        if paper_id is not None:
            where["paper_id"] = paper_id

        # Extra filtri
        if extra_filters:
            where.update(extra_filters)

        return where

    @staticmethod
    def _where_filter_to_bm25(where_filter: dict[str, Any]) -> dict[str, Any]:
        """Converte un where_filter generico nel formato BM25Index.

        BM25 usa chiavi specifiche (``year_min``, ``year_max``) per range,
        mentre Qdrant usa tuple/dict. Questa funzione adatta il formato.

        Args:
            where_filter: Filtro generico.

        Returns:
            Filtro compatibile con BM25Index.
        """
        if not where_filter:
            return {}

        bm25_filter: dict[str, Any] = {}

        for key, value in where_filter.items():
            if key == "year":
                # Converte formati Qdrant → BM25
                if isinstance(value, tuple) and len(value) == 2:
                    if value[0] is not None:
                        bm25_filter["year_min"] = value[0]
                    if value[1] is not None:
                        bm25_filter["year_max"] = value[1]
                elif isinstance(value, dict):
                    if "gte" in value:
                        bm25_filter["year_min"] = value["gte"]
                    if "gt" in value:
                        bm25_filter["year_min"] = value["gt"] + 1
                    if "lte" in value:
                        bm25_filter["year_max"] = value["lte"]
                    if "lt" in value:
                        bm25_filter["year_max"] = value["lt"] - 1
                elif isinstance(value, int):
                    bm25_filter["year"] = value
                else:
                    bm25_filter["year"] = value
            elif key == "chunk_type":
                # BM25 supporta match esatto o lista
                bm25_filter["chunk_type"] = value
            else:
                # Passa direttamente (category, bibtex_key, paper_id, ecc.)
                bm25_filter[key] = value

        return bm25_filter

    # ------------------------------------------------------------------
    # Retrieval principale
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        *,
        category_filter: Optional[str | list[str]] = None,
        year_range: Optional[tuple[int | None, int | None]] = None,
        chunk_types: Optional[list[str]] = None,
        bibtex_key: Optional[str] = None,
        paper_id: Optional[str] = None,
        extra_filters: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        include_vector_only: bool = False,
        include_bm25_only: bool = False,
        recency_boost: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """Ricerca ibrida: vector + BM25 + RRF + metadata filtering.

        Esegue ricerca parallela su entrambi i sistemi, fonde i risultati
        con Reciprocal Rank Fusion e ritorna i top-k finali.

        Se BM25 non è disponibile (indice non costruito), fallback a
        ricerca puramente vettoriale.

        Args:
            query: Testo della query.
            n_results: Numero di risultati finali. Default da config.
            category_filter: Filtro per categoria o lista di categorie.
            year_range: Tupla (anno_min, anno_max). None per unbounded.
            chunk_types: Tipi di chunk da includere (es. ["original", "l1_summary"]).
            bibtex_key: Filtro per chiave BibTeX.
            paper_id: Filtro per paper_id.
            extra_filters: Filtri aggiuntivi arbitrari.
            collection_name: Collezione Qdrant da interrogare. Default: paper_chunks.
            vector_weight: Override del peso vettoriale per questa query.
            bm25_weight: Override del peso BM25 per questa query.
            include_vector_only: Se True, usa solo ricerca vettoriale.
            include_bm25_only: Se True, usa solo BM25.
            recency_boost: Override del boost temporale per questa query.
                Se None, usa il valore da config. 0.0 per disabilitare.

        Returns:
            Lista di ``RetrievedChunk`` ordinati per score RRF decrescente.
        """
        if not query or not query.strip():
            logger.warning("HybridRetriever.retrieve: query vuota")
            return []

        t0 = time.time()
        final_n = n_results if n_results is not None else self._top_k_final
        v_weight = vector_weight if vector_weight is not None else self._vector_weight
        b_weight = bm25_weight if bm25_weight is not None else self._bm25_weight

        # Costruisci filtro unificato
        where_filter = self._build_where_filter(
            category_filter=category_filter,
            year_range=year_range,
            chunk_types=chunk_types,
            bibtex_key=bibtex_key,
            paper_id=paper_id,
            extra_filters=extra_filters,
        )

        # --- Vector search ---
        vector_results: list[dict[str, Any]] = []
        if not include_bm25_only:
            vector_results = self._vector_search(
                query, where_filter, collection_name
            )

        # --- BM25 search ---
        bm25_results: list[dict[str, Any]] = []
        if not include_vector_only:
            bm25_results = self._bm25_search(query, where_filter)

        # --- Fusione risultati ---
        if include_vector_only or (not bm25_results and vector_results):
            # Solo vettoriale
            fused = self._results_from_single_source(
                vector_results, RetrievalMethod.VECTOR, "vector"
            )
        elif include_bm25_only or (not vector_results and bm25_results):
            # Solo BM25
            fused = self._results_from_single_source(
                bm25_results, RetrievalMethod.BM25, "bm25"
            )
        elif vector_results and bm25_results:
            # Hybrid: RRF fusion
            fused = self._fuse_results(
                vector_results, bm25_results,
                v_weight, b_weight,
            )
        else:
            # Nessun risultato
            fused = []

        # --- Recency boost ---
        r_boost = recency_boost if recency_boost is not None else self._recency_boost
        if r_boost and r_boost > 0.0 and fused:
            fused = _apply_recency_boost(fused, r_boost)

        # --- Venue boost ---
        if self._venue_ranking_enabled and fused:
            fused = _apply_venue_boost(fused)

        # Tronca a final_n
        fused = fused[:final_n]

        elapsed = time.time() - t0
        top_score = fused[0].score if fused else 0.0
        logger.info(
            f"HybridRetriever: query={query[:60]!r}, "
            f"vector={len(vector_results)}, bm25={len(bm25_results)}, "
            f"fused={len(fused)}, top_score={top_score:.4f}, "
            f"recency_boost={r_boost}, "
            f"venue_ranking={self._venue_ranking_enabled}, "
            f"elapsed={elapsed:.3f}s"
        )

        return fused

    def retrieve_multi_collection(
        self,
        query: str,
        collections: Optional[list[str]] = None,
        n_results: Optional[int] = None,
        *,
        category_filter: Optional[str | list[str]] = None,
        year_range: Optional[tuple[int | None, int | None]] = None,
        chunk_types: Optional[list[str]] = None,
        deduplicate: bool = True,
    ) -> list[RetrievedChunk]:
        """Ricerca ibrida su più collezioni Qdrant simultaneamente.

        Utile per cercare in chunk originali + L1 + L2 contemporaneamente.
        I risultati vengono fusi e deduplicati per chunk_id.

        Args:
            query: Testo della query.
            collections: Nomi delle collezioni. Default: tutte e tre.
            n_results: Numero di risultati finali.
            category_filter: Filtro per categoria.
            year_range: Range anni.
            chunk_types: Tipi di chunk.
            deduplicate: Se True, deduplicazione per chunk_id (tiene lo score più alto).

        Returns:
            Lista di ``RetrievedChunk`` ordinati per score decrescente.
        """
        store = self._get_qdrant_store()

        if collections is None:
            collections = [
                store.COLLECTION_CHUNKS,
                store.COLLECTION_L1,
                store.COLLECTION_L2,
            ]

        final_n = n_results if n_results is not None else self._top_k_final

        all_results: list[RetrievedChunk] = []
        for col in collections:
            results = self.retrieve(
                query=query,
                n_results=self._top_k_vector,  # Prendi più candidati da ogni collection
                category_filter=category_filter,
                year_range=year_range,
                chunk_types=chunk_types,
                collection_name=col,
            )
            all_results.extend(results)

        if deduplicate:
            all_results = _deduplicate_by_chunk_id(all_results)

        # Riordina e tronca
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:final_n]

    # ------------------------------------------------------------------
    # Retrieval per singola sorgente
    # ------------------------------------------------------------------

    def retrieve_vector_only(
        self,
        query: str,
        n_results: Optional[int] = None,
        **kwargs,
    ) -> list[RetrievedChunk]:
        """Ricerca puramente vettoriale (Qdrant).

        Shortcut per ``retrieve(..., include_vector_only=True)``.

        Args:
            query: Testo della query.
            n_results: Numero di risultati.
            **kwargs: Parametri filtro passati a ``retrieve()``.

        Returns:
            Lista di ``RetrievedChunk``.
        """
        return self.retrieve(
            query, n_results=n_results, include_vector_only=True, **kwargs
        )

    def retrieve_bm25_only(
        self,
        query: str,
        n_results: Optional[int] = None,
        **kwargs,
    ) -> list[RetrievedChunk]:
        """Ricerca puramente BM25 (keyword).

        Shortcut per ``retrieve(..., include_bm25_only=True)``.

        Args:
            query: Testo della query.
            n_results: Numero di risultati.
            **kwargs: Parametri filtro passati a ``retrieve()``.

        Returns:
            Lista di ``RetrievedChunk``.
        """
        return self.retrieve(
            query, n_results=n_results, include_bm25_only=True, **kwargs
        )

    # ------------------------------------------------------------------
    # Internal: ricerca vettoriale
    # ------------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        where_filter: dict[str, Any],
        collection_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Esegue ricerca vettoriale su QdrantStore.

        Args:
            query: Testo della query.
            where_filter: Filtro metadata.
            collection_name: Collezione target.

        Returns:
            Lista di risultati dal QdrantStore.
        """
        store = self._get_qdrant_store()
        try:
            results = store.query(
                query_text=query,
                n_results=self._top_k_vector,
                where_filter=where_filter if where_filter else None,
                collection_name=collection_name,
            )
            logger.debug(f"Vector search: {len(results)} risultati")
            return results
        except Exception as e:
            logger.error(f"Errore ricerca vettoriale: {e}")
            return []

    # ------------------------------------------------------------------
    # Internal: ricerca BM25
    # ------------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        where_filter: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Esegue ricerca BM25.

        Args:
            query: Testo della query.
            where_filter: Filtro metadata (viene convertito al formato BM25).

        Returns:
            Lista di risultati dal BM25Index.
        """
        bm25 = self._get_bm25_index()
        if bm25 is None or not bm25.is_built:
            logger.debug("BM25 non disponibile, skip ricerca keyword")
            return []

        bm25_filter = self._where_filter_to_bm25(where_filter)

        try:
            results = bm25.query(
                query_text=query,
                n_results=self._top_k_bm25,
                where_filter=bm25_filter if bm25_filter else None,
            )
            logger.debug(f"BM25 search: {len(results)} risultati")
            return results
        except Exception as e:
            logger.error(f"Errore ricerca BM25: {e}")
            return []

    # ------------------------------------------------------------------
    # Internal: fusione RRF
    # ------------------------------------------------------------------

    def _fuse_results(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        vector_weight: float,
        bm25_weight: float,
    ) -> list[RetrievedChunk]:
        """Fonde risultati vettoriali e BM25 con Reciprocal Rank Fusion.

        Crea un indice di lookup per tutti i risultati, calcola RRF
        scores, e costruisce ``RetrievedChunk`` con metadata completi.

        Args:
            vector_results: Risultati dalla ricerca vettoriale.
            bm25_results: Risultati dalla ricerca BM25.
            vector_weight: Peso vettoriale per RRF.
            bm25_weight: Peso BM25 per RRF.

        Returns:
            Lista di ``RetrievedChunk`` ordinati per score RRF decrescente.
        """
        # Indice di lookup: chunk_id → (text, metadata)
        lookup: dict[str, dict[str, Any]] = {}

        # Tracking rank per ogni sorgente
        vector_ranks: dict[str, int] = {}
        vector_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        bm25_scores: dict[str, float] = {}

        for rank, res in enumerate(vector_results):
            cid = res.get("chunk_id", "")
            if not cid:
                continue
            lookup[cid] = {"text": res.get("text", ""), "metadata": res.get("metadata", {})}
            vector_ranks[cid] = rank + 1  # 1-based
            vector_scores[cid] = res.get("score", 0.0)

        for rank, res in enumerate(bm25_results):
            cid = res.get("chunk_id", "")
            if not cid:
                continue
            if cid not in lookup:
                lookup[cid] = {"text": res.get("text", ""), "metadata": res.get("metadata", {})}
            bm25_ranks[cid] = rank + 1  # 1-based
            bm25_scores[cid] = res.get("score", 0.0)

        # Calcola RRF scores
        rrf_results = reciprocal_rank_fusion(
            ranked_lists=[vector_results, bm25_results],
            weights=[vector_weight, bm25_weight],
            k=self._rrf_k,
        )

        # Costruisci RetrievedChunk con dettagli per-source
        chunks: list[RetrievedChunk] = []
        for chunk_id, rrf_score in rrf_results:
            info = lookup.get(chunk_id, {"text": "", "metadata": {}})

            in_vector = chunk_id in vector_ranks
            in_bm25 = chunk_id in bm25_ranks

            if in_vector and in_bm25:
                method = RetrievalMethod.HYBRID
            elif in_vector:
                method = RetrievalMethod.VECTOR
            else:
                method = RetrievalMethod.BM25

            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=info["text"],
                score=rrf_score,
                retrieval_method=method,
                metadata=info["metadata"],
                vector_score=vector_scores.get(chunk_id),
                bm25_score=bm25_scores.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
                bm25_rank=bm25_ranks.get(chunk_id),
            ))

        return chunks

    # ------------------------------------------------------------------
    # Internal: risultati da singola sorgente
    # ------------------------------------------------------------------

    def _results_from_single_source(
        self,
        results: list[dict[str, Any]],
        method: RetrievalMethod,
        score_field: str,
    ) -> list[RetrievedChunk]:
        """Converte risultati da una singola sorgente in RetrievedChunk.

        Args:
            results: Lista di risultati (da Qdrant o BM25).
            method: Metodo di retrieval (VECTOR o BM25).
            score_field: Campo per lo score specifico ("vector" o "bm25").

        Returns:
            Lista di ``RetrievedChunk``.
        """
        chunks: list[RetrievedChunk] = []
        for rank, res in enumerate(results):
            cid = res.get("chunk_id", "")
            if not cid:
                continue

            score = res.get("score", 0.0)

            chunks.append(RetrievedChunk(
                chunk_id=cid,
                text=res.get("text", ""),
                score=score,
                retrieval_method=method,
                metadata=res.get("metadata", {}),
                vector_score=score if score_field == "vector" else None,
                bm25_score=score if score_field == "bm25" else None,
                vector_rank=(rank + 1) if score_field == "vector" else None,
                bm25_rank=(rank + 1) if score_field == "bm25" else None,
            ))

        return chunks

    # ------------------------------------------------------------------
    # Utility: statistiche retriever
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Statistiche aggregate del retriever.

        Ritorna informazioni su entrambe le sorgenti di retrieval
        e la configurazione corrente.

        Returns:
            Dict con statistiche di configurazione e sorgenti.
        """
        stats: dict[str, Any] = {
            "config": {
                "vector_weight": self._vector_weight,
                "bm25_weight": self._bm25_weight,
                "top_k_vector": self._top_k_vector,
                "top_k_bm25": self._top_k_bm25,
                "top_k_final": self._top_k_final,
                "rrf_k": self._rrf_k,
                "recency_boost": self._recency_boost,
            },
            "qdrant_available": False,
            "bm25_available": False,
        }

        # Qdrant stats
        if self._qdrant_initialized and self._qdrant_store is not None:
            try:
                qdrant_stats = self._qdrant_store.get_stats()
                stats["qdrant_available"] = True
                stats["qdrant"] = qdrant_stats
            except Exception as e:
                stats["qdrant_error"] = str(e)

        # BM25 stats
        if self._bm25_initialized and self._bm25_index is not None:
            try:
                bm25_stats = self._bm25_index.get_stats()
                stats["bm25_available"] = True
                stats["bm25"] = bm25_stats
            except Exception as e:
                stats["bm25_error"] = str(e)

        return stats

    def __repr__(self) -> str:
        qdrant_status = "ready" if self._qdrant_initialized else "lazy"
        bm25_status = "ready" if self._bm25_initialized else "lazy"
        return (
            f"HybridRetriever(vector_w={self._vector_weight}, "
            f"bm25_w={self._bm25_weight}, "
            f"qdrant={qdrant_status}, bm25={bm25_status})"
        )


# ===========================================================================
# Helpers
# ===========================================================================

def _apply_recency_boost(
    results: list[RetrievedChunk],
    boost: float,
) -> list[RetrievedChunk]:
    """Applica un boost temporale ai risultati post-RRF.

    Formula: ``final_score = rrf_score * (1 + boost * recency_factor)``
    dove ``recency_factor = (year - baseline) / (max_year - baseline)``
    normalizzato in ``[0, 1]``. Anno baseline = 2015.

    I risultati senza anno (metadata ``year`` assente o None) ricevono
    ``recency_factor = 0.0`` (nessun boost).

    Dopo l'applicazione del boost, i risultati vengono ri-ordinati
    per score decrescente.

    Args:
        results: Lista di ``RetrievedChunk`` da boostare.
        boost: Fattore di boost (es. 0.1 = max +10% per paper più recenti).
            Se 0.0 o negativo, restituisce i risultati invariati.

    Returns:
        Lista di ``RetrievedChunk`` con score aggiornato e ri-ordinata.
    """
    if not results or boost <= 0.0:
        return results

    # Raccogli gli anni disponibili
    years: list[int] = []
    for r in results:
        y = r.metadata.get("year")
        if y is not None:
            try:
                years.append(int(y))
            except (ValueError, TypeError):
                pass

    if not years:
        # Nessun anno disponibile → nessun boost possibile
        logger.debug("Recency boost: nessun anno nei metadata, skip")
        return results

    max_year = max(years)
    baseline = _RECENCY_BASELINE_YEAR
    year_range = max(max_year - baseline, 1)  # Evita divisione per zero

    boosted_count = 0
    for r in results:
        y = r.metadata.get("year")
        if y is not None:
            try:
                y_int = int(y)
            except (ValueError, TypeError):
                continue
            recency_factor = max((y_int - baseline) / year_range, 0.0)
            r.score = r.score * (1.0 + boost * recency_factor)
            boosted_count += 1

    # Ri-ordina per score decrescente
    results.sort(key=lambda r: r.score, reverse=True)

    logger.debug(
        f"Recency boost applicato: boost={boost}, "
        f"year_range={baseline}-{max_year}, "
        f"boosted={boosted_count}/{len(results)}"
    )

    return results


def _apply_venue_boost(
    results: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Applica il boost di autorevolezza della venue ai risultati post-RRF.

    Usa ``VenueRanker`` per calcolare il boost (positivo o negativo) in
    base alla venue nel payload del chunk. Il boost viene sommato allo
    score corrente e clampato in ``[0.0, 1.0]``.

    Per ogni candidato viene impostato ``venue_tier`` per tracciabilità.

    Args:
        results: Lista di ``RetrievedChunk`` dopo RRF (e opzionale recency boost).

    Returns:
        Lista di ``RetrievedChunk`` con score aggiornato e ri-ordinata per score
        decrescente.
    """
    from paperhunter.indexing.venue_ranker import create_venue_ranker

    ranker = create_venue_ranker()

    for r in results:
        venue = r.metadata.get("venue") or r.metadata.get("journal") or None
        if not venue:
            venue = None
        boost = ranker.get_boost(venue)
        tier = ranker.get_tier(venue)
        old_score = r.score
        r.score = max(0.0, r.score + boost)
        r.venue_tier = tier
        logger.debug(
            "Venue boost: chunk_id=%s venue=%r tier=%d boost=%+.2f score %.4f→%.4f",
            r.chunk_id,
            venue,
            tier,
            boost,
            old_score,
            r.score,
        )

    # Ri-ordina per score decrescente
    results.sort(key=lambda r: r.score, reverse=True)

    logger.debug("Venue boost applicato a %d chunk", len(results))
    return results


def _deduplicate_by_chunk_id(
    results: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Deduplicazione per chunk_id, mantiene lo score più alto.

    Args:
        results: Lista di RetrievedChunk (potenzialmente con duplicati).

    Returns:
        Lista deduplicata.
    """
    seen: dict[str, RetrievedChunk] = {}
    for r in results:
        if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
            seen[r.chunk_id] = r
    return list(seen.values())
