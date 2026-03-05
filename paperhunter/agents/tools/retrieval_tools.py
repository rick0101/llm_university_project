"""
Retrieval Tools — Tool invocabili dal Writer ReAct Agent.

Trasforma i retriever esistenti (QdrantStore, BM25Index, graph communities)
in ``FunctionTool`` di LlamaIndex, invocabili autonomamente dall'agente
durante il ReAct loop.

Ogni tool:
- Restituisce una stringa di testo concisa (max ~2000 chars)
- Include ``[bibtex_key]`` per ogni fonte citata
- Gestisce errori gracefully (restituendo un messaggio di errore informativo)
- Logga la chiamata (query, n_risultati, tempo)

Tool disponibili:

- ``search_vector_database``: Ricerca vettoriale su Qdrant (paper chunks, L1, L2)
- ``search_keyword_index``: Ricerca BM25 keyword-based su testo full-text
- ``search_graph_communities``: Ricerca nei community summary del knowledge graph

Uso standalone per test::

    from paperhunter.agents.tools.retrieval_tools import (
        search_vector_database,
        search_keyword_index,
        search_graph_communities,
    )

    result = search_vector_database("prompt injection attacks", category="Prompt Injection")
    print(result)

Uso come FunctionTool per il ReAct Agent::

    from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools

    tools = get_all_retrieval_tools()
    # → [FunctionTool(search_vector_database), ...]
"""

from __future__ import annotations

import time
from typing import Optional

from llama_index.core.tools import FunctionTool

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ============================================================================
# Costanti
# ============================================================================

_MAX_RESULT_CHARS = 2000
"""Massimo numero di caratteri restituiti da ogni tool."""

_DEFAULT_TOP_K = 5
"""Numero di default di risultati per ogni tool."""

_TEXT_TRUNCATE = 350
"""Massimo numero di caratteri per ogni singolo chunk nel risultato."""


# ============================================================================
# Lazy singletons per i backend (evitano ri-inizializzazione ad ogni call)
# ============================================================================

_qdrant_store_singleton: object | None = None
_bm25_index_singleton: object | None = None


def _get_qdrant_store():
    """Ottieni o crea l'istanza singleton di QdrantStore.

    Returns:
        Istanza ``QdrantStore`` pronta all'uso.

    Raises:
        RuntimeError: Se QdrantStore non è inizializzabile.
    """
    global _qdrant_store_singleton
    if _qdrant_store_singleton is None:
        from paperhunter.indexing.qdrant_store import QdrantStore

        _qdrant_store_singleton = QdrantStore()
        logger.debug("QdrantStore singleton inizializzato per retrieval tools")
    return _qdrant_store_singleton


def _get_bm25_index():
    """Ottieni o crea l'istanza singleton di BM25Index.

    Carica l'indice da disco se disponibile. Ritorna None se
    l'indice non è stato ancora costruito.

    Returns:
        Istanza ``BM25Index`` se disponibile, altrimenti None.
    """
    global _bm25_index_singleton
    if _bm25_index_singleton is None:
        try:
            from paperhunter.indexing.bm25_index import BM25Index

            idx = BM25Index()
            if idx.index_path.exists():
                idx.load()
                if idx.is_built:
                    _bm25_index_singleton = idx
                    logger.debug(
                        f"BM25Index singleton caricato ({idx.size} chunk)"
                    )
                else:
                    logger.warning(
                        "BM25Index caricato ma non costruito (vuoto)"
                    )
            else:
                logger.warning(
                    f"BM25Index file non trovato: {idx.index_path}"
                )
        except Exception as e:
            logger.warning(f"Impossibile caricare BM25Index: {e}")
    return _bm25_index_singleton


def reset_singletons() -> None:
    """Reset dei singleton (utile per i test).

    Riporta i backend a None, forzando la ri-inizializzazione
    alla prossima chiamata dei tool.
    """
    global _qdrant_store_singleton, _bm25_index_singleton
    _qdrant_store_singleton = None
    _bm25_index_singleton = None
    logger.debug("Retrieval tools singletons resettati")


# ============================================================================
# Helper: formatting
# ============================================================================

def _truncate_text(text: str, max_chars: int = _TEXT_TRUNCATE) -> str:
    """Tronca un testo a un massimo di caratteri, aggiungendo '...' se necessario."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _format_qdrant_results(
    results: list[dict],
    max_total_chars: int = _MAX_RESULT_CHARS,
) -> str:
    """Formatta i risultati di una query Qdrant come testo citabile.

    Ogni risultato viene formattato con:
    - ``[bibtex_key]`` per la citazione
    - Categoria e anno (se disponibili)
    - Testo del chunk (troncato)

    Args:
        results: Lista di dict dal ``QdrantStore.query()``.
        max_total_chars: Budget massimo per l'output complessivo.

    Returns:
        Stringa formattata pronta per l'agente.
    """
    if not results:
        return "(No results found for this query.)"

    parts: list[str] = []
    total_chars = 0

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        bibtex_key = meta.get("bibtex_key", "unknown")
        category = meta.get("category", "N/A")
        year = meta.get("year", "N/A")
        score = r.get("score", 0.0)
        text = r.get("text", "")

        # Tronca il testo del singolo chunk
        text = _truncate_text(text)

        entry = (
            f"[{i}] [{bibtex_key}] (cat: {category}, year: {year}, "
            f"score: {score:.3f})\n{text}\n"
        )

        if total_chars + len(entry) > max_total_chars:
            remaining = len(results) - i + 1
            parts.append(f"[... {remaining} more results truncated ...]")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


def _format_bm25_results(
    results: list[dict],
    max_total_chars: int = _MAX_RESULT_CHARS,
) -> str:
    """Formatta i risultati di una query BM25 come testo citabile.

    Args:
        results: Lista di dict dal ``BM25Index.query()``.
        max_total_chars: Budget massimo per l'output complessivo.

    Returns:
        Stringa formattata pronta per l'agente.
    """
    if not results:
        return "(No results found for this query.)"

    parts: list[str] = []
    total_chars = 0

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        bibtex_key = meta.get("bibtex_key", "unknown")
        category = meta.get("category", "N/A")
        year = meta.get("year", "N/A")
        score = r.get("score", 0.0)
        text = r.get("text", "")

        text = _truncate_text(text)

        entry = (
            f"[{i}] [{bibtex_key}] (cat: {category}, year: {year}, "
            f"BM25: {score:.3f})\n{text}\n"
        )

        if total_chars + len(entry) > max_total_chars:
            remaining = len(results) - i + 1
            parts.append(f"[... {remaining} more results truncated ...]")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


def _format_community_results(
    results: list[dict],
    max_total_chars: int = _MAX_RESULT_CHARS,
) -> str:
    """Formatta i risultati di una ricerca community come testo informativo.

    I community summary non hanno bibtex_key individuali, ma contengono
    informazioni strutturali sulle macro-tematiche del knowledge graph.

    Args:
        results: Lista di dict dalla query Qdrant sulla collection
            ``graph_communities``.
        max_total_chars: Budget massimo per l'output complessivo.

    Returns:
        Stringa formattata pronta per l'agente.
    """
    if not results:
        return "(No graph community summaries found for this topic.)"

    parts: list[str] = []
    total_chars = 0

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        paper_id = meta.get("paper_id", "unknown")
        score = r.get("score", 0.0)
        text = r.get("text", "")

        # Il paper_id per le community è "community_<id>"
        community_label = paper_id if paper_id.startswith("community_") else f"community_{i}"

        # Tronca il testo (i community summary possono essere lunghi)
        text = _truncate_text(text, max_chars=500)

        entry = (
            f"[{i}] {community_label} (relevance: {score:.3f})\n{text}\n"
        )

        if total_chars + len(entry) > max_total_chars:
            remaining = len(results) - i + 1
            parts.append(f"[... {remaining} more community summaries truncated ...]")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


# ============================================================================
# Tool 1: Vector Database Search (Qdrant)
# ============================================================================

def search_vector_database(
    query: str,
    category: str | None = None,
    year_min: int | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> str:
    """Search the vector database for semantically similar paper chunks.

    Performs a semantic vector search on the Qdrant store (collection:
    paper_chunks) using the query embedding. Returns the most relevant
    chunks with their citations [bibtex_key].

    Use this tool to find papers semantically related to a specific topic
    or research question. You can filter by category and/or minimum year.

    Args:
        query: The search query in natural language (e.g., "adversarial
            prompt injection techniques for GPT-4").
        category: Optional category filter (e.g., "Prompt Injection",
            "Adversarial Code", "Data and Model Poisoning", "Hallucinations").
            If None, searches all categories.
        year_min: Optional minimum publication year filter (e.g., 2023).
            If None, no year filtering is applied.
        top_k: Number of results to return (default: 5, max: 10).

    Returns:
        Formatted text with the top matching chunks, each prefixed with
        [bibtex_key] for citation, including category, year, and
        relevance score. Returns an error message if the search fails.

    Examples:
        >>> search_vector_database("GCG adversarial suffix attack")
        "[1] [Zou_2023_GCG] (cat: Adversarial Code, year: 2023, score: 0.891)\\n..."
        >>> search_vector_database("jailbreak defense", category="Prompt Injection", year_min=2024)
        "[1] [Wei_2024_Jailbreak] ..."
    """
    t0 = time.time()
    top_k = min(max(top_k, 1), 10)

    logger.info(
        f"[tool] search_vector_database: query={query!r}, "
        f"category={category}, year_min={year_min}, top_k={top_k}"
    )

    try:
        store = _get_qdrant_store()

        # Costruisci filtro metadata
        where_filter: dict | None = None
        filter_parts: dict = {}
        if category:
            filter_parts["category"] = category
        if year_min is not None:
            filter_parts["year"] = {"gte": year_min}
        if filter_parts:
            where_filter = filter_parts

        results = store.query(
            query_text=query,
            n_results=top_k,
            where_filter=where_filter,
            collection_name=None,  # default: paper_chunks
        )

        elapsed = time.time() - t0
        logger.info(
            f"[tool] search_vector_database: {len(results)} risultati "
            f"in {elapsed:.2f}s"
        )

        return _format_qdrant_results(results)

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(
            f"[tool] search_vector_database ERRORE: {e} ({elapsed:.2f}s)"
        )
        return (
            f"(Error searching vector database: {type(e).__name__}: {e}. "
            f"The vector store may not be available.)"
        )


# ============================================================================
# Tool 2: BM25 Keyword Index Search
# ============================================================================

def search_keyword_index(
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    category: str | None = None,
) -> str:
    """Search the BM25 keyword index for exact term matches.

    Performs a keyword-based search using the BM25 (Okapi BM25) index.
    Unlike vector search, this tool is best for finding specific terms,
    technique names, or exact phrases mentioned in papers.

    Use this tool when you need to find papers that mention specific
    terminology, model names, dataset names, or benchmark results.

    Args:
        query: The keyword search query (e.g., "GCG suffix",
            "MMLU benchmark", "DPO alignment"). Best with specific terms.
        top_k: Number of results to return (default: 5, max: 10).
        category: Optional category filter (e.g., "Prompt Injection").
            If None, searches all categories.

    Returns:
        Formatted text with the top matching chunks ranked by BM25 score,
        each prefixed with [bibtex_key] for citation. Returns an error
        message if the BM25 index is not available.

    Examples:
        >>> search_keyword_index("GCG adversarial suffix")
        "[1] [Zou_2023_GCG] (cat: Adversarial Code, year: 2023, BM25: 12.34)\\n..."
        >>> search_keyword_index("prompt injection defense", category="Prompt Injection")
        "[1] [Wei_2024_Defense] ..."
    """
    t0 = time.time()
    top_k = min(max(top_k, 1), 10)

    logger.info(
        f"[tool] search_keyword_index: query={query!r}, "
        f"top_k={top_k}, category={category}"
    )

    try:
        bm25 = _get_bm25_index()
        if bm25 is None:
            elapsed = time.time() - t0
            logger.warning(
                f"[tool] search_keyword_index: BM25 non disponibile ({elapsed:.2f}s)"
            )
            return (
                "(BM25 keyword index is not available. "
                "It may not have been built yet. "
                "Try using search_vector_database instead.)"
            )

        # Filtro opzionale per categoria
        where_filter: dict | None = None
        if category:
            where_filter = {"category": category}

        results = bm25.query(
            query_text=query,
            n_results=top_k,
            where_filter=where_filter,
        )

        elapsed = time.time() - t0
        logger.info(
            f"[tool] search_keyword_index: {len(results)} risultati "
            f"in {elapsed:.2f}s"
        )

        return _format_bm25_results(results)

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(
            f"[tool] search_keyword_index ERRORE: {e} ({elapsed:.2f}s)"
        )
        return (
            f"(Error searching keyword index: {type(e).__name__}: {e}. "
            f"The BM25 index may not be built.)"
        )


# ============================================================================
# Tool 3: Graph Communities Search
# ============================================================================

def search_graph_communities(
    topic: str,
    top_k: int = 3,
) -> str:
    """Search graph community summaries for macro-thematic context.

    Searches the knowledge graph community summaries stored in Qdrant
    (collection: graph_communities). These summaries describe macro-themes
    identified by community detection (Louvain/Leiden) on the entity
    relationship graph.

    Use this tool to understand high-level thematic structures and
    cross-paper connections. Community summaries provide a bird's-eye
    view of how research topics relate to each other.

    Args:
        topic: The topic or theme to search for (e.g., "prompt injection
            defense mechanisms", "adversarial robustness testing").
        top_k: Number of community summaries to return (default: 3, max: 5).

    Returns:
        Formatted text with the most relevant community summaries,
        including community ID, relevance score, and narrative text.
        Returns an error message if no community summaries exist.

    Examples:
        >>> search_graph_communities("adversarial attacks on language models")
        "[1] community_3 (relevance: 0.823)\\nThis community centers around..."
        >>> search_graph_communities("hallucination detection and mitigation")
        "[1] community_7 (relevance: 0.756)\\n..."
    """
    t0 = time.time()
    top_k = min(max(top_k, 1), 5)

    logger.info(
        f"[tool] search_graph_communities: topic={topic!r}, top_k={top_k}"
    )

    try:
        store = _get_qdrant_store()

        from paperhunter.config import settings

        collection = settings.qdrant.collection_graph_communities

        # Verifica che la collection esista
        if not store.client.collection_exists(collection):
            elapsed = time.time() - t0
            logger.info(
                f"[tool] search_graph_communities: collection '{collection}' "
                f"non esiste ({elapsed:.2f}s)"
            )
            return (
                "(No graph community summaries available. "
                "Community detection may not have been run yet. "
                "Try using search_vector_database instead.)"
            )

        # Query sulla collection graph_communities
        results = store.query(
            query_text=topic,
            n_results=top_k,
            collection_name=collection,
        )

        elapsed = time.time() - t0
        logger.info(
            f"[tool] search_graph_communities: {len(results)} risultati "
            f"in {elapsed:.2f}s"
        )

        return _format_community_results(results)

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(
            f"[tool] search_graph_communities ERRORE: {e} ({elapsed:.2f}s)"
        )
        return (
            f"(Error searching graph communities: {type(e).__name__}: {e}. "
            f"The community summaries may not be available.)"
        )


# ============================================================================
# Helper: costruzione lista tool per il ReAct Agent
# ============================================================================

def get_all_retrieval_tools() -> list[FunctionTool]:
    """Restituisce tutti i retrieval tool come ``FunctionTool`` di LlamaIndex.

    I tool sono pronti per essere registrati in un ``ReActAgent`` o
    ``AgentRunner`` di LlamaIndex.

    Returns:
        Lista di 3 ``FunctionTool``:
        - ``search_vector_database``: ricerca vettoriale Qdrant
        - ``search_keyword_index``: ricerca BM25 keyword
        - ``search_graph_communities``: ricerca community graph

    Example::

        from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools
        from llama_index.core.agent import ReActAgent

        tools = get_all_retrieval_tools()
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
    """
    tools = [
        FunctionTool.from_defaults(
            fn=search_vector_database,
            name="search_vector_database",
            description=(
                "Search the vector database for semantically similar paper "
                "chunks. Use for finding papers related to a research topic. "
                "Supports optional category and year_min filters. "
                "Returns formatted text with [bibtex_key] citations."
            ),
        ),
        FunctionTool.from_defaults(
            fn=search_keyword_index,
            name="search_keyword_index",
            description=(
                "Search the BM25 keyword index for exact term matches. "
                "Best for finding specific technique names, model names, "
                "dataset names, or benchmark results. "
                "Returns formatted text with [bibtex_key] citations."
            ),
        ),
        FunctionTool.from_defaults(
            fn=search_graph_communities,
            name="search_graph_communities",
            description=(
                "Search knowledge graph community summaries for macro-thematic "
                "context. Use when you need a high-level overview of how "
                "research topics connect across papers. "
                "Returns narrative community descriptions."
            ),
        ),
    ]

    logger.info(
        f"Retrieval tools inizializzati: {len(tools)} tool "
        f"({', '.join(t.metadata.name for t in tools)})"
    )

    return tools
