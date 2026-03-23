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
- ``search_figure_descriptions``: Ricerca combinata su figure_descriptions + image_embeddings

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
    """Reset dei singleton.

    Chiude il client Qdrant e riporta i backend a None, forzando
    la ri-inizializzazione alla prossima chiamata dei tool.

    Include gc.collect() per assicurarsi che il lock del filesystem
    Qdrant (locale/embedded) venga rilasciato prima di riaprire.
    """
    import gc

    global _qdrant_store_singleton, _bm25_index_singleton
    if _qdrant_store_singleton is not None:
        try:
            _qdrant_store_singleton.close()
        except Exception:
            pass
    _qdrant_store_singleton = None
    _bm25_index_singleton = None
    gc.collect()
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
# Tool 4: Figure Descriptions Search
# ============================================================================

def _format_figure_results(
    results: list[dict],
    max_total_chars: int = _MAX_RESULT_CHARS,
) -> str:
    """Formatta i risultati di figure con path immagine e sintassi Markdown.

    Ogni risultato include:
    - ``[bibtex_key]`` per la citazione
    - Fonte: ``[description]`` o ``[image_embedding]``
    - Descrizione testuale della figura
    - Sintassi Markdown ``![caption](path)`` per includere l'immagine

    Args:
        results: Lista di dict dal ``QdrantStore.query()``.
            Ogni dict può contenere un campo ``source`` con valore
            ``"description"`` o ``"image_embedding"``.
        max_total_chars: Budget massimo per l'output complessivo.

    Returns:
        Stringa formattata con figure e path immagini.
    """
    if not results:
        return "(No figure results found for this query.)"

    parts: list[str] = []
    total_chars = 0

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        bibtex_key = meta.get("bibtex_key", "unknown")
        category = meta.get("category", "N/A")
        year = meta.get("year", "N/A")
        score = r.get("score", 0.0)
        text = r.get("text", "")
        image_path = meta.get("image_path", "")
        source = r.get("source", "description")

        text = _truncate_text(text)

        entry = (
            f"[{i}] [{bibtex_key}] [{source}] (cat: {category}, year: {year}, "
            f"score: {score:.3f})\n{text}\n"
        )

        # Se c'è un path immagine, aggiungi la sintassi Markdown
        if image_path:
            # Genera caption breve dal bibtex_key e indice
            caption = f"Figure from [{bibtex_key}]"
            entry += f"IMAGE_PATH: {image_path}\n"
            entry += f"To include this figure, use: ![{caption}]({image_path})\n"

        if total_chars + len(entry) > max_total_chars:
            remaining = len(results) - i + 1
            parts.append(f"[... {remaining} more results truncated ...]")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


def _merge_and_deduplicate_figure_results(
    desc_results: list[dict],
    img_results: list[dict],
    top_k: int,
) -> list[dict]:
    """Merge risultati da figure_descriptions e image_embeddings, deduplicando.

    Deduplicazione per combinazione di ``paper_id`` + ``image_path``:
    se un'immagine appare in entrambe le collection, mantiene lo score
    più alto e annota la fonte come ``"both"``.

    Args:
        desc_results: Risultati dalla collection ``figure_descriptions``.
        img_results: Risultati dalla collection ``image_embeddings``.
        top_k: Numero massimo di risultati finali.

    Returns:
        Lista di risultati deduplicati, ordinati per score decrescente,
        troncati a ``top_k``. Ogni dict ha un campo ``source``.
    """
    # Tag source
    for r in desc_results:
        r["source"] = "description"
    for r in img_results:
        r["source"] = "image_embedding"

    # Dedup map: key = (paper_id, image_path)
    seen: dict[tuple[str, str], dict] = {}

    for r in desc_results + img_results:
        meta = r.get("metadata", {})
        paper_id = meta.get("paper_id", "")
        image_path = meta.get("image_path", "")
        key = (paper_id, image_path)

        if key in seen:
            existing = seen[key]
            if r.get("score", 0.0) > existing.get("score", 0.0):
                old_source = existing["source"]
                seen[key] = r
                seen[key]["source"] = "both"
            else:
                existing["source"] = "both"
        else:
            seen[key] = r

    # Ordina per score decrescente e tronca a top_k
    merged = sorted(seen.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:top_k]


def search_figure_descriptions(
    query: str,
    category: str | None = None,
    top_k: int = _DEFAULT_TOP_K,
    include_image_embeddings: bool = True,
) -> str:
    """Search figure descriptions extracted from paper PDFs.

    Searches the figure_descriptions collection in Qdrant, which contains
    AI-generated semantic descriptions of figures, diagrams, plots, and
    tables extracted from the original PDFs using multimodal vision.

    When ``include_image_embeddings`` is True (default), also searches the
    ``image_embeddings`` collection for cross-modal retrieval (text query
    → visually similar figures). Results from both collections are merged,
    deduplicated by paper_id + image_path, and sorted by score.

    Results include the image file path, so the writer can include the
    actual figure in the survey using Markdown image syntax:
    ``![caption](figures/image_path.png)``

    Use this tool when you need information about:
    - Experimental results shown in plots or charts
    - System architectures or pipeline diagrams
    - Comparison tables or confusion matrices
    - Attack/defense workflow diagrams

    Args:
        query: The search query describing what type of figure or visual
            information you're looking for (e.g., "attack success rate
            comparison chart", "defense architecture diagram").
        category: Optional category filter (e.g., "Prompt Injection").
        top_k: Number of results to return (default: 5, max: 10).
        include_image_embeddings: If True, also search the image_embeddings
            collection for cross-modal retrieval. Default: True.

    Returns:
        Formatted text with the most relevant figure descriptions,
        each prefixed with [bibtex_key] for citation. Includes
        source indicator [description], [image_embedding], or [both].
        Includes IMAGE_PATH and Markdown inclusion syntax for each figure.

    Examples:
        >>> search_figure_descriptions("attack success rate comparison")
        "[1] [Zou_2023_GCG] [description] (cat: Adversarial Code, year: 2023, score: 0.82)\\n..."
        >>> search_figure_descriptions("defense pipeline architecture", category="Prompt Injection")
        "[1] [Wei_2024_Defense] [image_embedding] ..."
    """
    t0 = time.time()
    top_k = min(max(top_k, 1), 10)

    logger.info(
        f"[tool] search_figure_descriptions: query={query!r}, "
        f"category={category}, top_k={top_k}, "
        f"include_image_embeddings={include_image_embeddings}"
    )

    try:
        store = _get_qdrant_store()

        from paperhunter.config import settings
        collection_desc = settings.qdrant.collection_figure_descriptions
        collection_img = settings.qdrant.collection_image_embeddings

        # Costruisci filtro metadata
        where_filter: dict | None = None
        if category:
            where_filter = {"category": category}

        # --- Cerca in figure_descriptions ---
        desc_results: list[dict] = []
        desc_available = store.client.collection_exists(collection_desc)
        if desc_available:
            desc_results = store.query(
                query_text=query,
                n_results=top_k,
                where_filter=where_filter,
                collection_name=collection_desc,
            )
        else:
            logger.info(
                f"[tool] search_figure_descriptions: collection '{collection_desc}' "
                f"non esiste, skip"
            )

        # --- Cerca in image_embeddings (se abilitato) ---
        img_results: list[dict] = []
        img_available = False
        if include_image_embeddings:
            img_available = store.client.collection_exists(collection_img)
            if img_available:
                img_results = store.query_image_embeddings(
                    query_text=query,
                    n_results=top_k,
                    where_filter=where_filter,
                )
            else:
                logger.info(
                    f"[tool] search_figure_descriptions: collection '{collection_img}' "
                    f"non esiste, skip image embeddings"
                )

        # Se nessuna collection disponibile
        if not desc_available and not img_available:
            elapsed = time.time() - t0
            logger.info(
                f"[tool] search_figure_descriptions: nessuna collection figure "
                f"disponibile ({elapsed:.2f}s)"
            )
            return (
                "(No figure descriptions available. "
                "Figure extraction may not have been run during ingestion. "
                "Try using search_vector_database instead.)"
            )

        # Merge e deduplicazione
        if img_results:
            results = _merge_and_deduplicate_figure_results(
                desc_results, img_results, top_k
            )
        else:
            # Solo figure_descriptions, aggiungi source tag
            for r in desc_results:
                r.setdefault("source", "description")
            results = desc_results

        elapsed = time.time() - t0
        n_desc = len(desc_results)
        n_img = len(img_results)
        logger.info(
            f"[tool] search_figure_descriptions: {len(results)} risultati "
            f"(desc: {n_desc}, img_emb: {n_img}) in {elapsed:.2f}s"
        )

        return _format_figure_results(results)

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(
            f"[tool] search_figure_descriptions ERRORE: {e} ({elapsed:.2f}s)"
        )
        return (
            f"(Error searching figure descriptions: {type(e).__name__}: {e}. "
            f"The figure descriptions may not be available.)"
        )


# ============================================================================
# Helper: costruzione lista tool per il ReAct Agent
# ============================================================================


# ============================================================================
# Tool: get_figure_for_section (Task 3.1)
# ============================================================================

_VALID_FIGURE_TYPES = frozenset({
    "any", "results_plot", "architecture_diagram", "flowchart",
    "comparison_table", "confusion_matrix", "attack_diagram", "defense_pipeline",
})


def get_figure_for_section(
    paper_bibtex_key: str,
    figure_type: str = "any",
) -> str:
    """Retrieve a specific figure from a paper for inclusion in the survey.

    Queries Qdrant for figures associated with the given paper. After calling
    this tool and receiving a valid result, insert a figure placeholder tag
    in the text using the format:

        {{FIGURE:bibtex_key:figure_type}}

    Example: {{FIGURE:Zou_2023_GCG:architecture_diagram}}

    Valid figure_types: any, results_plot, architecture_diagram, flowchart,
    comparison_table, confusion_matrix, attack_diagram, defense_pipeline

    Args:
        paper_bibtex_key: The BibTeX key of the paper (e.g., "Zou_2023_GCG").
        figure_type: Type of figure to retrieve. Use "any" to get the best
            available figure regardless of type.

    Returns:
        JSON string with figure metadata (figure_id, paper_key, figure_type,
        description, image_path, caption) or "NO_FIGURE_FOUND" if not available.
    """
    import json as _json

    start = time.time()
    logger.info(
        "get_figure_for_section: paper=%s type=%s", paper_bibtex_key, figure_type
    )

    if not paper_bibtex_key or not paper_bibtex_key.strip():
        return "NO_FIGURE_FOUND"

    if figure_type not in _VALID_FIGURE_TYPES:
        figure_type = "any"

    try:
        store = _get_qdrant_store()
    except Exception as exc:
        logger.warning("QdrantStore non disponibile: %s", exc)
        return "NO_FIGURE_FOUND"

    # ---- 1. Cerca nella collection figure_descriptions ----
    result = _query_figure_collection(
        store=store,
        collection="figure_descriptions",
        bibtex_key=paper_bibtex_key,
        figure_type=figure_type,
    )

    # ---- 2. Fallback: cerca nella collection image_embeddings ----
    if result is None:
        result = _query_figure_collection(
            store=store,
            collection="image_embeddings",
            bibtex_key=paper_bibtex_key,
            figure_type=figure_type,
        )

    elapsed = time.time() - start
    if result is None:
        logger.info(
            "get_figure_for_section: NO_FIGURE_FOUND per %s (%.2fs)",
            paper_bibtex_key, elapsed,
        )
        return "NO_FIGURE_FOUND"

    logger.info(
        "get_figure_for_section: trovata figura %s per %s (%.2fs)",
        result.get("figure_id", "?"), paper_bibtex_key, elapsed,
    )
    return _json.dumps(result, ensure_ascii=False)


def _query_figure_collection(
    store: object,
    collection: str,
    bibtex_key: str,
    figure_type: str,
) -> dict | None:
    """Interroga una collection Qdrant per trovare una figura di un paper.

    Filtra per `bibtex_key` nel payload e opzionalmente per `figure_type`.
    Restituisce il primo risultato come dict, o None se nessuno trovato.
    """
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must_conditions = [
            FieldCondition(key="bibtex_key", match=MatchValue(value=bibtex_key))
        ]
        if figure_type != "any":
            must_conditions.append(
                FieldCondition(key="figure_type", match=MatchValue(value=figure_type))
            )

        qdrant_filter = Filter(must=must_conditions)

        # Usa scroll (non vettoriale) per filtrare per metadata
        client = getattr(store, "client", None) or getattr(store, "_client", None)
        if client is None:
            return None

        records, _ = client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            return None

        payload = records[0].payload or {}
        return {
            "figure_id": str(records[0].id),
            "paper_key": payload.get("bibtex_key", bibtex_key),
            "figure_type": payload.get("figure_type", figure_type),
            "description": payload.get("description", payload.get("text", "")),
            "image_path": payload.get("image_path", ""),
            "caption": payload.get("caption", payload.get("figure_caption", "")),
        }

    except Exception as exc:
        logger.debug(
            "_query_figure_collection(%s, %s): %s", collection, bibtex_key, exc
        )
        return None


def get_all_retrieval_tools(
    default_category: str | None = None,
    light: bool = False,
) -> list[FunctionTool]:
    """Restituisce tutti i retrieval tool come ``FunctionTool`` di LlamaIndex.
    I tool sono pronti per essere registrati in un ``ReActAgent`` o
    ``AgentRunner`` di LlamaIndex.

    Args:
        default_category: Se fornito, i tool di ricerca useranno questa
            categoria come default (il LLM può comunque sovrascriverla).

    Returns:
        Lista di ``FunctionTool``:
        - ``search_vector_database``: ricerca vettoriale Qdrant
        - ``search_keyword_index``: ricerca BM25 keyword
        - ``search_graph_communities``: ricerca community graph
        - ``search_figure_descriptions``: ricerca descrizioni figure
        - ``get_figure_for_section``: recupero figure specifiche

    Example::

        from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools
        from llama_index.core.agent import ReActAgent

        tools = get_all_retrieval_tools(default_category="Prompt Injection")
        agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
    """
    # Se c'è una default_category, crea closure che forzano category
    # nella signature visibile al LLM (partial NON funziona perché
    # il parametro resta nello schema JSON e il LLM può passare None).
    if default_category:
        _cat = default_category  # cattura nella closure

        def vec_fn(query: str, year_min: int | None = None, top_k: int = 5) -> str:
            """Search the vector database (category pre-set)."""
            return search_vector_database(query, category=_cat, year_min=year_min, top_k=top_k)

        def kw_fn(query: str, top_k: int = 5) -> str:
            """Search the keyword index (category pre-set)."""
            return search_keyword_index(query, top_k=top_k, category=_cat)

        def fig_fn(
            query: str, top_k: int = 5, include_image_embeddings: bool = True,
        ) -> str:
            """Search figure descriptions (category pre-set)."""
            return search_figure_descriptions(
                query, category=_cat, top_k=top_k,
                include_image_embeddings=include_image_embeddings,
            )
    else:
        vec_fn = search_vector_database
        kw_fn = search_keyword_index
        fig_fn = search_figure_descriptions

    # Aggiorna description se category è pre-impostata
    cat_note = (
        f" Category is pre-set to '{default_category}'."
        if default_category else ""
    )

    tools = [
        FunctionTool.from_defaults(
            fn=vec_fn,
            name="search_vector_database",
            description=(
                "Search the vector database for semantically similar paper "
                "chunks. Use for finding papers related to a research topic. "
                "Supports optional year_min filter. "
                "Returns formatted text with [bibtex_key] citations."
                + cat_note
            ),
        ),
        FunctionTool.from_defaults(
            fn=kw_fn,
            name="search_keyword_index",
            description=(
                "Search the BM25 keyword index for exact term matches. "
                "Best for finding specific technique names, model names, "
                "dataset names, or benchmark results. "
                "Returns formatted text with [bibtex_key] citations."
                + cat_note
            ),
        ),
    ]

    # In modalità light (prefetch_agentic), ometti graph_communities e figure_descriptions
    # perché il contesto è già pre-caricato. Tieni solo search tools essenziali.
    if not light:
        tools.append(
            FunctionTool.from_defaults(
                fn=search_graph_communities,
                name="search_graph_communities",
                description=(
                    "Search knowledge graph community summaries for macro-thematic "
                    "context. Use when you need a high-level overview of how "
                    "research topics connect across papers. "
                    "Returns narrative community descriptions."
                ),
            )
        )
        tools.append(
            FunctionTool.from_defaults(
                fn=fig_fn,
                name="search_figure_descriptions",
                description=(
                    "Search AI-generated descriptions of figures, diagrams, "
                    "plots, and tables from paper PDFs. Also searches "
                    "multimodal image embeddings for cross-modal retrieval. "
                    "Use when you need information about experimental results, "
                    "architectures, or visual data from papers. "
                    "Returns formatted text with [bibtex_key] citations "
                    "and source indicator [description] or [image_embedding]."
                ),
            )
        )

    tools.append(
        FunctionTool.from_defaults(
            fn=get_figure_for_section,
            name="get_figure_for_section",
            description=(
                "Retrieve a specific figure from a paper to embed in the survey. "
                "Use when you want to include a diagram, plot, or chart from a specific paper. "
                "Call this BEFORE inserting a {{FIGURE:key:type}} tag in the text. "
                "Returns JSON with figure metadata or 'NO_FIGURE_FOUND'. "
                "Valid figure_types: any, results_plot, architecture_diagram, flowchart, "
                "comparison_table, confusion_matrix, attack_diagram, defense_pipeline."
            ),
        )
    )

    logger.info(
        f"Retrieval tools inizializzati: {len(tools)} tool "
        f"({', '.join(t.metadata.name for t in tools)})"
    )

    return tools
