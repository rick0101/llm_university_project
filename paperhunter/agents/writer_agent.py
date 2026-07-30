"""
Writer Agent — Scrive le sezioni del survey paper con RAG e citazioni verificabili.

Agent 2 nella pipeline agentica LangGraph. Supporta due modalità
controllate da ``config.yaml → agents.writer_mode``:

**Modalità "static"** (default — backward compatible):
1. Formula 3-5 query RAG specifiche dalla descrizione della sezione
2. Esegue retrieval ibrido (HybridRetriever) con filtri appropriati
3. (Opzionale) Query al PropertyGraph per contesto relazionale
4. Genera il testo della sezione con Gemini Pro, con citazioni [bibtex_key]
5. Output: SectionDraft con text + CitationRecord list

**Modalità "agentic"** (ReAct multi-step):
1. Inizializza un LlamaIndex ``ReActAgent`` con i retrieval tool
2. L'agente cerca autonomamente le informazioni necessarie
3. Loop ReAct: Thought → Action (tool call) → Observation → Final Answer
4. Fallback automatico a "static" in caso di errore

Produce testo in stile scientifico formale con:
- Ogni claim supportato da citazione [bibtex_key]
- Contesto temporale (evoluzione cronologica)
- Per la sezione Testing Methodologies, tabella riassuntiva

Uso standalone::

    from paperhunter.agents.writer_agent import write_section, write_all_sections
    from paperhunter.agents.state import create_initial_state

    state = create_initial_state(l2_summaries=[...])
    state["outline"] = [...]
    updated = write_section(state, "prompt_injection_attacks")

Uso come nodo LangGraph::

    from langgraph.graph import StateGraph
    from paperhunter.agents.state import SurveyState
    from paperhunter.agents.writer_agent import write

    graph = StateGraph(SurveyState)
    graph.add_node("write", write)
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.agents.state import (
    CitationRecord,
    PipelinePhase,
    SectionDraft,
    SectionOutline,
    SectionStatus,
    SurveyState,
    get_l2_summaries_from_state,
    get_outline_from_state,
    get_sections_from_state,
)
from paperhunter.config import settings
from paperhunter.evolution.lesson_store import LessonStore
from paperhunter.evolution.prompt_overlay import generate_full_overlay
from paperhunter.utils.llm_factory import (
    get_llm,
    llm_call_with_retry,
)
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_success,
    print_table,
    print_warning,
)
from paperhunter.utils.prompt_loader import get_prompt
from paperhunter.utils.token_counter import (
    compute_writer_budget,
    get_writer_max_context_chars,
    get_writer_budget_mode,
)

logger = get_logger(__name__)


# ============================================================================
# Custom exceptions
# ============================================================================

class MaxTokensError(RuntimeError):
    """L'LLM ha raggiunto il limite max_tokens e la risposta è stata troncata."""
    pass


class MaxIterationsError(RuntimeError):
    """Il ReAct agent ha raggiunto il limite massimo di iterazioni."""
    pass


class TransientLLMError(RuntimeError):
    """Errore transitorio dell'API LLM (timeout, 503, rate limit, ecc.)."""
    pass


# Pattern che indicano errori transitori recuperabili
_TRANSIENT_PATTERNS = (
    "timed out",
    "timeout",
    "503",
    "502",
    "500",
    "429",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "overloaded",
    "capacity",
    "rate limit",
    "rate_limit",
    "connection",
    "ConnectionError",
    "ServiceUnavailable",
    "InternalServerError",
)


# ============================================================================
# Tool-call cache for retry resilience
# ============================================================================

def _make_cache_key(name: str, args: tuple, kwargs: dict) -> str:
    """Deterministic cache key from tool name + arguments."""
    raw = json.dumps({"n": name, "a": list(args), "k": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def _wrap_tools_with_cache(
    tools: list,
    cache: dict[str, str],
) -> list:
    """Wrap each FunctionTool so results are cached across retries.

    On the first call with given arguments the real tool runs and the result
    is stored in *cache*.  On subsequent calls with the same arguments (e.g.
    after a TransientLLMError retry) the cached result is returned instantly,
    saving an API / embedding round-trip.
    """
    from llama_index.core.tools import FunctionTool

    wrapped: list = []
    for tool in tools:
        original_fn = tool.fn
        tool_name = tool.metadata.name

        @functools.wraps(original_fn)
        def _cached_fn(*args, _orig=original_fn, _name=tool_name, **kwargs):
            key = _make_cache_key(_name, args, kwargs)
            if key in cache:
                logger.debug("[tool-cache] HIT %s(%s…)", _name, str(args)[:60])
                return cache[key]
            result = _orig(*args, **kwargs)
            cache[key] = result
            logger.debug("[tool-cache] MISS+STORE %s(%s…)", _name, str(args)[:60])
            return result

        wrapped.append(
            FunctionTool.from_defaults(
                fn=_cached_fn,
                name=tool_name,
                description=tool.metadata.description,
            )
        )
    return wrapped


# ============================================================================
# Prompt templates
# ============================================================================

_QUERY_GENERATION_PROMPT = get_prompt("QUERY_GENERATION_PROMPT")

_WRITER_PROMPT_TEMPLATE = get_prompt("WRITER_PROMPT")

_TESTING_SPECIAL_INSTRUCTIONS = """\

### SPECIAL INSTRUCTIONS FOR TESTING METHODOLOGIES SECTION

This is the CRITICAL Testing Methodologies section for the thesis. You MUST include:

1. **Comprehensive Table**: Create a Markdown table summarizing the testing methodologies \
across ALL categories. Columns: | Category | Benchmark/Dataset | Metrics | Tools/Frameworks | Representative Papers |

2. **Per-Category Analysis**: For each security threat category, describe:
   - The most commonly used benchmarks and datasets
   - Standard evaluation metrics (e.g., ASR, F1, BLEU for factuality)
   - Open-source tools and frameworks for testing
   - Gaps in evaluation methodology

3. **Cross-Category Comparison**: Highlight how testing approaches differ \
across threat categories and where standardization is needed.
"""

_INTRO_SPECIAL_INSTRUCTIONS = """\

### SPECIAL INSTRUCTIONS FOR INTRODUCTION

Write a compelling introduction that:
1. Motivates the survey by highlighting the growing deployment of LLMs
2. Identifies the key security challenges (briefly preview each category)
3. States the contributions of this survey
4. Provides a brief overview of the paper structure (referencing section names)
"""

_DISCUSSION_SPECIAL_INSTRUCTIONS = """\

### SPECIAL INSTRUCTIONS FOR DISCUSSION

The discussion should:
1. Synthesize findings across ALL categories
2. Identify cross-cutting themes and open challenges
3. Discuss limitations of current approaches
4. Propose concrete future research directions
5. Address practical implications for practitioners
"""


# ============================================================================
# Helper: Query generation
# ============================================================================

def _generate_rag_queries(
    outline: SectionOutline,
    n_queries: int = 5,
) -> list[str]:
    """Genera query RAG specifiche per una sezione usando Gemini Flash.

    Se la generazione LLM fallisce, produce query di fallback
    dalla descrizione e dai topic della sezione.

    Args:
        outline: L'outline della sezione per cui generare le query.
        n_queries: Numero di query da generare (3-7).

    Returns:
        Lista di query testuali per il retrieval.
    """
    n_queries = max(3, min(7, n_queries))

    prompt = _QUERY_GENERATION_PROMPT.format(
        title=outline.title,
        description=outline.description[:1500],
        key_topics=", ".join(outline.key_topics) if outline.key_topics else "N/A",
        related_categories=", ".join(outline.related_categories) if outline.related_categories else "all",
        n_queries=n_queries,
    )

    try:
        # Query generation is a tiny deterministic JSON task.  Hidden
        # reasoning only consumes the visible-output budget here.
        llm = get_llm(
            "flash",
            max_tokens=2048,
            timeout_seconds=60,
            thinking_enabled=False,
        )
        raw = llm_call_with_retry(
            llm, prompt, max_retries=2, base_delay=5.0, max_delay=30.0,
        )
        queries = _parse_query_json(raw)
        if queries and len(queries) >= 2:
            logger.info(f"Generatee {len(queries)} query RAG per '{outline.title}'")
            return queries[:n_queries]
    except Exception as e:
        logger.warning(f"Fallback query generation per '{outline.title}': {e}")

    # Fallback: costruisci query dai topic e dalla descrizione
    return _build_fallback_queries(outline, n_queries)


def _parse_query_json(raw: str) -> list[str] | None:
    """Parse la risposta LLM come JSON array di stringhe."""
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Tentativo 1: JSON diretto
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(q, str) for q in data):
            return [q.strip() for q in data if q.strip()]
    except json.JSONDecodeError:
        pass

    # Tentativo 2: estrai array
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [str(q).strip() for q in data if str(q).strip()]
        except json.JSONDecodeError:
            pass

    return None


def _build_fallback_queries(outline: SectionOutline, n_queries: int) -> list[str]:
    """Costruisce query di fallback dalla sezione outline."""
    queries: list[str] = []

    # Query dal titolo
    queries.append(f"{outline.title} LLM security survey")

    # Query dai topic
    for topic in outline.key_topics[:n_queries - 1]:
        queries.append(f"{topic} large language models")

    # Query dalla descrizione (prime frasi)
    if outline.description:
        sentences = outline.description.split(". ")
        for s in sentences[:2]:
            if len(s) > 20:
                queries.append(s.strip().rstrip("."))

    # Assicurati di avere almeno n_queries
    while len(queries) < n_queries:
        cats = " ".join(outline.related_categories[:2]) if outline.related_categories else "LLM"
        queries.append(f"{cats} recent advances security")

    return queries[:n_queries]


# ============================================================================
# Helper: Retrieval
# ============================================================================


def _diversify_chunks_by_paper(
    chunks: list[dict],
    *,
    limit: int,
    max_per_paper: int = 2,
) -> list[dict]:
    """Select evidence with broad paper coverage before duplicate passages.

    Hybrid retrieval can return many high-scoring passages from one paper.  A
    writer then sees too few distinct citable sources to satisfy the Critic, no
    matter how good the prose is.  This stable round-robin first takes the best
    chunk from each citable paper, then a second supporting chunk, and only
    afterwards background-only summaries.
    """
    if limit <= 0 or not chunks:
        return []

    def _score(chunk: dict) -> float:
        return float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0)

    ranked = sorted(chunks, key=_score, reverse=True)
    citable = [
        chunk for chunk in ranked
        if (chunk.get("metadata") or {}).get("is_citable", False)
        and (chunk.get("metadata") or {}).get("bibtex_key")
    ]
    background = [chunk for chunk in ranked if chunk not in citable]

    selected: list[dict] = []
    selected_ids: set[str] = set()
    per_paper: dict[str, int] = {}

    # One best evidence unit from every paper first.
    for chunk in citable:
        key = str((chunk.get("metadata") or {}).get("bibtex_key"))
        chunk_id = str(chunk.get("chunk_id") or id(chunk))
        if per_paper.get(key, 0) or chunk_id in selected_ids:
            continue
        selected.append(chunk)
        selected_ids.add(chunk_id)
        per_paper[key] = 1
        if len(selected) >= limit:
            return selected

    # Add detail passages while preventing any paper from dominating.
    for chunk in citable:
        key = str((chunk.get("metadata") or {}).get("bibtex_key"))
        chunk_id = str(chunk.get("chunk_id") or id(chunk))
        if chunk_id in selected_ids or per_paper.get(key, 0) >= max_per_paper:
            continue
        selected.append(chunk)
        selected_ids.add(chunk_id)
        per_paper[key] = per_paper.get(key, 0) + 1
        if len(selected) >= limit:
            return selected

    # L2/background material is useful for synthesis, but never at the expense
    # of citable evidence coverage.
    for chunk in background:
        chunk_id = str(chunk.get("chunk_id") or id(chunk))
        if chunk_id in selected_ids:
            continue
        selected.append(chunk)
        selected_ids.add(chunk_id)
        if len(selected) >= limit:
            break
    return selected

def _retrieve_context_for_section(
    queries: list[str],
    outline: SectionOutline,
    *,
    retriever=None,
    reranker=None,
    graph_builder=None,
    top_k_per_query: int = 30,
    top_k_final: int = 25,
    use_reranker: bool = True,
    use_graph: bool = False,
) -> tuple[list[dict], list[str]]:
    """Esegue retrieval ibrido per una sezione, con reranking e graph opzionali.

    Args:
        queries: Query RAG per questa sezione.
        outline: Outline della sezione (per filtri).
        retriever: HybridRetriever instance (lazy-init se None).
        reranker: GeminiReranker instance (lazy-init se None).
        graph_builder: PropertyGraphBuilder instance (opzionale).
        top_k_per_query: Risultati per singola query.
        top_k_final: Risultati finali dopo dedup + reranking.
        use_reranker: Se usare il reranker Gemini.
        use_graph: Se usare il PropertyGraph per contesto aggiuntivo.

    Returns:
        Tuple di (chunks, chunk_ids) dove ogni chunk è un dict con
        chunk_id, text, score, metadata.
    """
    # Lazy init retriever — track if created locally to close it after use
    _local_retriever = False
    if retriever is None:
        from paperhunter.indexing.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        _local_retriever = True

    # Build category filter
    category_filter = None
    if outline.related_categories:
        category_filter = outline.related_categories

    # Collect results from all queries
    all_results: list[dict] = []
    seen_ids: set[str] = set()

    try:
        for query in queries:
            try:
                results = retriever.retrieve_multi_collection(
                    query,
                    n_results=top_k_per_query,
                    category_filter=category_filter,
                )
                for r in results:
                    rid = r.chunk_id if hasattr(r, "chunk_id") else r.get("chunk_id", "")
                    if rid and rid not in seen_ids:
                        seen_ids.add(rid)
                        # Normalize to dict
                        if hasattr(r, "to_dict"):
                            all_results.append(r.to_dict())
                        elif isinstance(r, dict):
                            all_results.append(r)
                        else:
                            all_results.append({
                                "chunk_id": rid,
                                "text": getattr(r, "text", str(r)),
                                "score": getattr(r, "score", 0.0),
                                "metadata": getattr(r, "metadata", {}),
                            })
            except Exception as e:
                logger.warning(f"Errore retrieval per query '{query[:60]}': {e}")
    finally:
        # Release the Qdrant lock immediately so other callers (ReAct tools, figure
        # search) can open their own client without a "already accessed" conflict.
        if _local_retriever:
            retriever.close()

    logger.info(
        f"Retrieval per '{outline.title}': {len(all_results)} chunk unici "
        f"da {len(queries)} query"
    )

    # Graph context (opzionale)
    graph_context = ""
    if use_graph and graph_builder is not None:
        try:
            graph_query = f"{outline.title}: {', '.join(outline.key_topics[:5])}"
            graph_result = graph_builder.query_graph(graph_query, limit=20)
            if graph_result and graph_result.context_text:
                graph_context = graph_result.context_text
                logger.info(
                    f"Graph context: {len(graph_context)} chars, "
                    f"{len(graph_result.entities_found)} entità"
                )
        except Exception as e:
            logger.warning(f"Errore GraphRAG per '{outline.title}': {e}")

    # Reranking
    candidate_limit = max(top_k_final * 2, top_k_final + 15)
    if use_reranker and len(all_results) > candidate_limit:
        try:
            if reranker is None:
                from paperhunter.indexing.reranker import GeminiReranker
                reranker = GeminiReranker(top_k=candidate_limit)

            rerank_query = f"Write about: {outline.title}. Topics: {', '.join(outline.key_topics[:5])}"
            reranked = reranker.rerank(rerank_query, all_results, top_k=candidate_limit)
            logger.info(f"Reranking: {len(all_results)} → {len(reranked)} chunk")
            all_results = reranked
        except Exception as e:
            logger.warning(f"Reranking fallito per '{outline.title}': {e}. Uso ordine originale.")
            # Sort by score descending e tronca
            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            all_results = all_results[:candidate_limit]
    elif len(all_results) > candidate_limit:
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        all_results = all_results[:candidate_limit]

    all_results = _diversify_chunks_by_paper(
        all_results,
        limit=top_k_final,
        max_per_paper=2,
    )
    distinct_citable = {
        (c.get("metadata") or {}).get("bibtex_key")
        for c in all_results
        if (c.get("metadata") or {}).get("is_citable", False)
    }
    logger.info(
        "Evidence diversification: %d chunk, %d paper citabili distinti",
        len(all_results),
        len(distinct_citable - {None, ""}),
    )

    # Se abbiamo graph context, aggiungilo come pseudo-chunk
    if graph_context:
        all_results.append({
            "chunk_id": f"graph_context_{outline.key}",
            "text": f"[Knowledge Graph Context]\n{graph_context}",
            "score": 0.5,
            "metadata": {
                "chunk_type": "graph",
                "category": "cross-cutting",
                "bibtex_key": "",
                "year": 0,
                "section_name": "graph_context",
            },
        })

    chunk_ids = [c.get("chunk_id", "") for c in all_results if c.get("chunk_id")]
    return all_results, chunk_ids


# Evidence that can be cited must precede background-only synthesis.  Within
# citable material, compact L1 atomic summaries come before original passages.
_CHUNK_TYPE_PRIORITY: dict[str, int] = {
    "L1_SUMMARY": 0,
    "l1_summary": 0,
    "original": 1,
    "L2_SUMMARY": 2,
    "l2_summary": 2,
    "graph": 3,
}


def _compute_section_budget(
    n_total_sections: int | None = None,
    max_chars_override: int | None = None,
) -> int:
    """Calcola il budget in caratteri per una singola sezione del writer.

    Se ``max_chars_override`` è fornito, lo usa direttamente.
    Altrimenti legge la modalità da config:

    - ``"proportional"``: ``min(writer_max_context_chars, global_budget / N_sezioni)``
    - ``"fixed"``: ``writer_max_context_chars``

    Args:
        n_total_sections: Numero totale di sezioni nel paper. Necessario
            solo per la modalità proportional.
        max_chars_override: Override diretto del budget in chars.

    Returns:
        Budget in caratteri per questa sezione.
    """
    if max_chars_override is not None:
        return max_chars_override

    if n_total_sections is None or n_total_sections <= 0:
        n_total_sections = 1

    budget_info = compute_writer_budget(n_sections=n_total_sections)
    budget = budget_info["per_section_chars"]
    logger.debug(
        f"Writer budget: mode={budget_info['mode']}, "
        f"per_section={budget:,} chars, "
        f"n_sections={n_total_sections}"
    )
    return budget


def _sort_chunks_by_priority(chunks: list[dict]) -> list[dict]:
    """Ordina i chunk dando priorità a L1/L2 summary rispetto a original.

    L'ordinamento è stabile: a parità di tipo, l'ordine originale
    (basato sullo score del retrieval/reranker) è preservato.

    Priorità: evidenze citabili (L1/original) > background (L2/graph).

    Args:
        chunks: Lista di chunk dict con metadata.

    Returns:
        Nuova lista ordinata per priorità di tipo (più informativi prima).
    """
    def _priority_key(chunk: dict) -> tuple[int, int]:
        metadata = chunk.get("metadata", {}) or {}
        chunk_type = metadata.get("chunk_type", "original")
        citable_rank = 0 if metadata.get("is_citable", False) else 1
        return citable_rank, _CHUNK_TYPE_PRIORITY.get(chunk_type, 2)

    return sorted(chunks, key=_priority_key)


def _format_context_for_prompt(
    chunks: list[dict],
    max_chars: int | None = None,
    *,
    n_total_sections: int | None = None,
    prefer_summaries: bool = True,
    related_categories: list[str] | None = None,
) -> str:
    """Formatta i chunk recuperati per il prompt del Writer.

    Il budget è determinato in quest'ordine:

    1. Se *max_chars* è fornito esplicitamente, lo usa.
    2. Altrimenti calcola il budget dalla config (``writer_budget_mode``
       + ``writer_max_context_chars``).

    Quando *prefer_summaries* è True (default), le evidenze citabili sono
    posizionate prima del background; all'interno dei due gruppi l'ordine è
    L1_SUMMARY > original > L2_SUMMARY > graph.

    Quando *related_categories* è fornito, i chunk L2/L1 summary
    di categorie non pertinenti vengono deprioritizzati per evitare
    di sprecare budget su contesto irrilevante.

    Args:
        chunks: Chunk recuperati dal retrieval.
        max_chars: Budget diretto (legacy). Se None, calcolato da config.
        n_total_sections: Numero totale di sezioni (per modalità proportional).
        prefer_summaries: Se True, riordina dando priorità a summary chunks.
        related_categories: Categorie pertinenti per la sezione corrente.
            Se fornito, i summary di altre categorie vengono penalizzati.

    Returns:
        Testo formattato pronto per il prompt del Writer.
    """
    if not chunks:
        return "(No source evidence retrieved. Do not draft factual claims.)"

    # --- Filtra summary di categorie irrilevanti ---
    if related_categories:
        _cats_lower = {c.lower() for c in related_categories}

        def _is_relevant(chunk: dict) -> bool:
            meta = chunk.get("metadata", {})
            chunk_type = meta.get("chunk_type", "original")
            # Solo summary L1/L2 vengono filtrati per categoria
            if chunk_type in ("L2_SUMMARY", "l2_summary", "L1_SUMMARY", "l1_summary"):
                cat = (meta.get("category") or "").lower()
                if cat and cat not in _cats_lower:
                    return False
            return True

        n_before = len(chunks)
        chunks = [c for c in chunks if _is_relevant(c)]
        n_filtered = n_before - len(chunks)
        if n_filtered > 0:
            logger.debug(
                f"Category filter: rimossi {n_filtered} summary chunk "
                f"di categorie non pertinenti (kept: {related_categories})"
            )

    # --- Calcola budget ---
    effective_max = _compute_section_budget(
        n_total_sections=n_total_sections,
        max_chars_override=max_chars,
    )

    # --- Riordina per priorità di tipo se richiesto ---
    ordered_chunks = _sort_chunks_by_priority(chunks) if prefer_summaries else chunks

    selected_chunks: list[dict] = []
    selected_entries: list[str] = []
    truncated_count = 0

    def _truncation_notice(count: int) -> str:
        return (
            f"[... {count} more chunks truncated for prompt length "
            f"(budget: {effective_max:,} chars) ...]"
        )

    def _whitelist_block(items: list[dict]) -> str:
        """Build a whitelist only from evidence that will actually be shown."""
        citable_keys = sorted({
            str((item.get("metadata", {}) or {}).get("bibtex_key") or "")
            for item in items
            if (item.get("metadata", {}) or {}).get("is_citable", False)
            and (item.get("metadata", {}) or {}).get("bibtex_key")
        })
        return (
            "=== CITATION WHITELIST (v3.6) ===\n"
            "You MAY ONLY cite the following bibtex_keys (CITABLE):\n  "
            + (", ".join(citable_keys) if citable_keys else "(none)")
            + "\nAll non-listed keys/chunks are BACKGROUND-ONLY: do not cite them."
            + "\n=== END WHITELIST ===\n"
        )

    for i, chunk in enumerate(ordered_chunks):
        meta = chunk.get("metadata", {}) or {}
        bibtex_key = meta.get("bibtex_key", "Unknown")
        category = meta.get("category", "N/A")
        year = meta.get("year", "N/A")
        section = meta.get("section_name", "N/A")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        chunk_type = meta.get("chunk_type", "original")
        text = chunk.get("text", "")
        citable_tag = "CITABLE" if meta.get("is_citable", False) else "BACKGROUND"

        if len(text) > 2000:
            text = text[:2000] + "..."

        entry_number = len(selected_entries) + 1
        entry = (
            f"--- Chunk {entry_number} [chunk_id: {chunk_id}] [{citable_tag}] ---\n"
            f"Paper: [{bibtex_key}] | Category: {category} | "
            f"Year: {year} | Section: {section} | Type: {chunk_type}\n"
            f"{text}\n"
        )
        candidate_chunks = [*selected_chunks, chunk]
        candidate_entries = [*selected_entries, entry]
        candidate_parts = [_whitelist_block(candidate_chunks), *candidate_entries]
        candidate_len = len("\n".join(candidate_parts))
        remaining_after = len(ordered_chunks) - i - 1
        # Reserve a compact truncation marker without exceeding the real prompt
        # budget. This makes budgeting deterministic, including the whitelist.
        truncation_reserve = (
            len("\n" + _truncation_notice(remaining_after))
            if remaining_after
            else 0
        )
        if candidate_len + truncation_reserve > effective_max:
            truncated_count = len(ordered_chunks) - i
            break

        selected_chunks = candidate_chunks
        selected_entries = candidate_entries

    parts = [_whitelist_block(selected_chunks), *selected_entries]
    if truncated_count:
        notice = _truncation_notice(truncated_count)
        if len("\n".join([*parts, notice])) <= effective_max:
            parts.append(notice)

    formatted = "\n".join(parts)
    total_chars = len(formatted)
    logger.info(
        f"Context formattato: {len(selected_chunks)} chunk inclusi, "
        f"{total_chars:,}/{effective_max:,} chars usati "
        f"({total_chars * 100 // effective_max if effective_max else 0}%)"
    )

    return formatted


def _extract_citable_keys_from_formatted_context(context: str) -> list[str]:
    """Read the exact citation whitelist embedded in the final context."""
    match = re.search(
        r"You MAY ONLY cite the following bibtex_keys \(CITABLE\):\n\s*(.*?)\n"
        r"All non-listed",
        context,
        re.DOTALL,
    )
    if not match or match.group(1).strip() == "(none)":
        return []
    return [key.strip() for key in match.group(1).split(",") if key.strip()]


def _get_special_instructions(section_type: str) -> str:
    """Ritorna istruzioni speciali per tipi di sezione specifici."""
    if section_type == "testing":
        return _TESTING_SPECIAL_INSTRUCTIONS
    elif section_type == "introduction":
        return _INTRO_SPECIAL_INSTRUCTIONS
    elif section_type in ("discussion",):
        return _DISCUSSION_SPECIAL_INSTRUCTIONS
    return ""


# ============================================================================
# Helper: Response parsing
# ============================================================================

# Patterns for meta-commentary preambles that the LLM may prepend to actual content
_META_PREAMBLE_RE = re.compile(
    r"^("
    r"Here(?:'s| is)[^\n]*(?:section|draft|text)[^\n]*[.!:]\s*"
    r"|Below (?:is|you)[^\n]*[.!:]\s*"
    r"|I (?:have |'ve )?(?:drafted|written|composed|prepared|created)[^\n]*[.!:]\s*"
    r"|Following is[^\n]*[.!:]\s*"
    r"|This section (?:covers|addresses|discusses|provides|focuses)[^\n]*[.!:]\s*"
    r")(?:\n+\*{3}\s*\n+)?",
    re.IGNORECASE,
)

# Patterns for trailing meta-commentary after the actual content
_META_EPILOGUE_RE = re.compile(
    r"\n+\*{0,3}(?:"
    r"(?:This (?:section|draft|text|content) (?:covers|addresses|discusses|provides|includes)[^.]*\.)"
    r"|(?:(?:I hope|Let me know|Feel free|Please (?:let|note))[^.]*\.)"
    r"|(?:The (?:above|section) (?:covers|discusses|addresses)[^.]*\.)"
    r")\s*\*{0,3}\s*$",
    re.IGNORECASE,
)


def _strip_meta_commentary(text: str) -> str:
    """Remove LLM meta-commentary preambles and epilogues from section text.

    The writer sometimes prepends "Here is the drafted section..." or appends
    "This section covers..." — this strips those without touching actual content.
    """
    # Strip preamble
    text = _META_PREAMBLE_RE.sub("", text, count=1).lstrip()

    # Strip epilogue
    text = _META_EPILOGUE_RE.sub("", text).rstrip()

    # Also strip horizontal rules used as separators before/after meta text
    text = re.sub(r"^\*{3}\s*\n+", "", text)
    text = re.sub(r"\n+\*{3}\s*$", "", text)

    return text.strip()


def _parse_writer_response(
    raw_response: str,
    section_key: str,
) -> dict | None:
    """Parse la risposta JSON del Writer.

    Ritorna dict con {text, citations, notes} o None se parsing fallisce.
    """
    if not raw_response or not raw_response.strip():
        return None

    text = raw_response.strip()

    # Rimuovi code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Tentativo 1: JSON diretto
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "text" in data:
            return _normalize_writer_output(data)
    except json.JSONDecodeError:
        pass

    # Tentativo 2: estrai il primo oggetto JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "text" in data:
                return _normalize_writer_output(data)
        except json.JSONDecodeError:
            pass

    # Tentativo 3: fix trailing commas e riprova
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "text" in data:
                return _normalize_writer_output(data)
        except json.JSONDecodeError:
            pass

    # Tentativo 4: se non è JSON, tratta la risposta come testo puro
    # ed estrai citazioni con regex
    logger.warning(
        f"Risposta per '{section_key}' non è JSON valido. "
        f"Tentativo estrazione testo + citazioni con regex."
    )

    # Strip meta-commentary preambles (e.g. "Here is the drafted section...")
    clean_text = _strip_meta_commentary(raw_response.strip())

    citations = _extract_citations_from_text(clean_text)
    return {
        "text": clean_text,
        "citations": citations,
        "notes": "Response was not valid JSON. Citations extracted via regex.",
    }


def _normalize_writer_output(data: dict) -> dict:
    """Normalizza l'output del Writer."""
    text = data.get("text", "")
    citations_raw = data.get("citations", [])
    notes = data.get("notes", "")

    # Normalizza citazioni
    citations: list[dict] = []
    if isinstance(citations_raw, list):
        for c in citations_raw:
            if isinstance(c, dict) and "bibtex_key" in c:
                citations.append({
                    "bibtex_key": c["bibtex_key"],
                    "claim": c.get("claim", ""),
                    "chunk_id": c.get("chunk_id"),
                    "retrieval_score": c.get("retrieval_score", 0.0),
                })

    return {"text": text, "citations": citations, "notes": notes}


def _extract_citations_from_text(text: str) -> list[dict]:
    """Estrai citazioni [bibtex_key] dal testo con regex."""
    matches: list[str] = []
    for group in re.findall(r"\[([^\]]+)\]", text):
        for raw_key in group.split(","):
            key = raw_key.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", key):
                matches.append(key)
    seen: set[str] = set()
    citations: list[dict] = []
    for key in matches:
        if re.match(r"^(Table|Figure|Section|Appendix|Algorithm|Equation)_\d+$", key):
            continue
        if key not in seen:
            seen.add(key)
            citations.append({
                "bibtex_key": key,
                "claim": "",
                "chunk_id": None,
                "retrieval_score": 0.0,
            })
    return citations


def _count_words(text: str) -> int:
    """Conta le parole in un testo."""
    return len(text.split()) if text else 0


# ============================================================================
# Agentic ReAct Writer
# ============================================================================

_REACT_SYSTEM_PROMPT = get_prompt("REACT_SYSTEM_PROMPT")

_REACT_MAX_ITERATIONS = 15
"""Numero massimo di iterazioni ReAct (Thought→Action→Observation)."""

_PREFETCH_MAX_ITERATIONS = 12
"""Iterazioni ReAct per modalità prefetch (contesto già caricato, ma serve margine per scrivere)."""

# --- Retry con backoff incrementale per errori transitori ---
_TRANSIENT_MAX_RETRIES = 4
"""Massimo numero di retry per errori transitori (timeout, 503, ecc.)."""
_TRANSIENT_BASE_DELAY = 15.0
"""Delay iniziale (secondi) prima del primo retry."""
_TRANSIENT_MAX_DELAY = 120.0
"""Delay massimo (secondi) tra retry."""
_TRANSIENT_TIMEOUT_INCREMENT = 120.0
"""Incremento del timeout del workflow ad ogni retry (secondi)."""
_TRANSIENT_BASE_TIMEOUT = 360.0
"""Timeout base del workflow ReAct (secondi)."""

_PREFETCH_REACT_SYSTEM_PROMPT = get_prompt("PREFETCH_REACT_SYSTEM_PROMPT")


def _write_section_prefetch_agentic(
    outline: SectionOutline,
    *,
    force_model: str | None = None,
    critic_feedback: str = "",
    retriever=None,
    reranker=None,
    graph_builder=None,
    n_rag_queries: int = 5,
    top_k_retrieval: int = 25,
    use_reranker: bool = True,
    use_graph: bool = False,
    n_total_sections: int | None = None,
    max_context_chars: int | None = None,
) -> dict | None:
    """Scrive una sezione con modalità ibrida: pre-fetch + ReAct leggero.

    Fase 1 (0 Pro calls): Genera query con Flash → retrieval ibrido → formatta contesto.
    Fase 2 (~3-5 Pro calls): ReAct agent con contesto pre-caricato e tool limitati.

    Args:
        outline: Outline della sezione da scrivere.
        force_model: Override del modello LLM ('flash' o 'pro').
        critic_feedback: Feedback dal Critic Agent (per riscrittura).
        retriever: HybridRetriever (lazy-init se None).
        reranker: GeminiReranker (lazy-init se None).
        graph_builder: PropertyGraphBuilder (opzionale).
        n_rag_queries: Numero query RAG da generare.
        top_k_retrieval: Risultati finali retrieval.
        use_reranker: Se usare il reranker.
        use_graph: Se usare il PropertyGraph.
        n_total_sections: Numero totale sezioni (per budget).
        max_context_chars: Override budget chars.

    Returns:
        Dict con {text, citations, notes} o None se fallisce.
    """
    from llama_index.core.agent import ReActAgent
    from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools

    t0 = time.time()
    tier = force_model or settings.agents.writer_model

    logger.info(
        f"[prefetch_agentic] Avvio per '{outline.title}' con {tier.upper()}"
    )

    # === FASE 1: Pre-fetch contesto (0 Pro calls) ===

    # Step 1a: Genera query RAG con Flash
    queries = _generate_rag_queries(outline, n_queries=n_rag_queries)
    if critic_feedback:
        queries.append(f"evidence for claims in {outline.title}")
        queries = queries[:n_rag_queries + 2]

    logger.info(
        f"[prefetch_agentic] Query RAG ({len(queries)}): "
        f"{[q[:50] for q in queries]}"
    )

    # Step 1b: Retrieval ibrido (0 LLM calls)
    chunks, chunk_ids = _retrieve_context_for_section(
        queries,
        outline,
        retriever=retriever,
        reranker=reranker,
        graph_builder=graph_builder,
        top_k_final=top_k_retrieval,
        use_reranker=use_reranker,
        use_graph=use_graph,
    )

    # Step 1c: Formatta contesto
    if n_total_sections is None:
        n_total_sections = 10  # default ragionevole
    context_text = _format_context_for_prompt(
        chunks,
        max_chars=max_context_chars,
        n_total_sections=n_total_sections,
        prefer_summaries=True,
        related_categories=outline.related_categories or None,
    )

    # Step 1d: Pre-fetch figure descriptions per la sezione
    figure_context = ""
    try:
        from paperhunter.agents.tools.retrieval_tools import search_figure_descriptions
        fig_query = f"{outline.title} {' '.join(outline.key_topics or [])}"
        fig_result = search_figure_descriptions(
            query=fig_query,
            category=outline.related_categories[0] if outline.related_categories else None,
            top_k=10,
        )
        if fig_result and "NO_RESULTS" not in fig_result and "Error" not in fig_result:
            # Parse bibtex_keys and figure_types to provide ready-to-use tags
            import re as _re
            _fig_entries = _re.findall(
                r'\[(\w+_\d{4}_\w+)\].*?IMAGE_PATH:\s*figures/\S+',
                fig_result, _re.DOTALL
            )
            _fig_types = _re.findall(
                r'\*\*Figure Type\*\*:\s*(\w+)',
                fig_result,
            )
            ready_tags = []
            for idx, bk in enumerate(_fig_entries):
                ft = _fig_types[idx] if idx < len(_fig_types) else "any"
                ready_tags.append(f"  {{{{FIGURE:{bk}:{ft}}}}}")

            tag_hint = ""
            if ready_tags:
                tag_hint = (
                    "\n\n**Ready-to-use tags** (copy-paste directly into your text, "
                    "no tool call needed — figures are already confirmed):\n"
                    + "\n".join(ready_tags)
                    + "\n"
                )

            figure_context = (
                "\n\n## AVAILABLE FIGURES FROM PAPERS\n"
                "The following figures were found in papers cited in this section. "
                "Use your scientific judgment to decide which figures genuinely strengthen "
                "the narrative — include as many as you find valuable (there is no hard limit). "
                "Place the corresponding tag on its own line after the paragraph it illustrates. "
                "These figures have been pre-validated — you can insert the tags directly "
                "without calling get_figure_for_section.\n\n"
                f"{fig_result}\n"
                f"{tag_hint}"
            )
            logger.info(
                f"[prefetch_agentic] Pre-fetched figure descriptions "
                f"({len(fig_result):,} chars, {len(ready_tags)} ready tags) "
                f"for '{outline.title}'"
            )
    except Exception as e:
        logger.warning(f"[prefetch_agentic] Figure pre-fetch failed: {e}")

    prefetch_elapsed = time.time() - t0
    logger.info(
        f"[prefetch_agentic] Pre-fetch: {len(chunks)} chunk, "
        f"{len(context_text):,} chars contesto in {prefetch_elapsed:.1f}s"
    )

    # === FASE 2: ReAct leggero con contesto pre-caricato ===

    special_instructions = _get_special_instructions(outline.section_type)
    critic_section = ""
    if critic_feedback:
        critic_section = (
            "## CRITIC FEEDBACK (from previous iteration)\n\n"
            "Address these issues in your rewrite:\n\n"
            f"{critic_feedback}\n"
        )

    system_prompt = _PREFETCH_REACT_SYSTEM_PROMPT.format(
        title=outline.title,
        description=outline.description[:2000],
        key_topics=", ".join(outline.key_topics) if outline.key_topics else "N/A",
        related_categories=", ".join(outline.related_categories) if outline.related_categories else "all",
        target_words=outline.target_length_words,
        special_instructions=special_instructions,
        prefetched_context=context_text + figure_context,
        critic_feedback_section=critic_section,
    )

    # --- Self-evolution: inject skills + lessons overlay ---
    if settings.evolution.enabled or settings.skills.enabled:
        try:
            evo_store = LessonStore(
                path=settings.evolution.lessons_path,
                half_life_days=settings.evolution.half_life_days,
                expiry_days=settings.evolution.expiry_days,
            )
            overlay = generate_full_overlay(
                evo_store,
                section_key=outline.key,
                agent_type="writer",
                skills_dir=settings.skills.skills_dir if settings.skills.enabled else None,
                skills_enabled=settings.skills.enabled,
                max_skill_chars=settings.skills.max_total_chars,
                max_lessons=settings.evolution.max_overlay_lessons,
                max_lesson_chars=settings.evolution.max_overlay_chars,
            )
            if overlay:
                system_prompt = f"{system_prompt}\n\n{overlay}"
                logger.info("[evolution] Injected %d-char overlay for '%s'", len(overlay), outline.key)
        except Exception as e:
            logger.warning("[evolution] Failed to load overlay: %s", e)

    # Tool ridotti: solo vector + keyword + figure tools (no graph_communities)
    default_cat = outline.related_categories[0] if outline.related_categories else None
    tools = get_all_retrieval_tools(default_category=default_cat, light=True)

    user_msg = (
        f"Write the section '{outline.title}' using the pre-fetched context. "
        f"Cover these topics: {', '.join(outline.key_topics[:7])}. "
        f"Target approximately {outline.target_length_words} words with [bibtex_key] citations. "
        f"Use at least {settings.agents.min_citations_per_section} distinct CITABLE papers; "
        f"prefer broader evidence coverage when it remains claim-accurate. The pre-fetched "
        f"context already contains relevant papers — use up to 7 tool calls for gaps, then WRITE. "
        f"PRIORITY: produce the full written text, not more searches. "
    )
    if figure_context:
        user_msg += (
            "Figures from relevant papers are available in the AVAILABLE FIGURES section. "
            "Use your judgment to include any that genuinely strengthen the narrative — "
            "place {{FIGURE:bibtex_key:figure_type}} tags on their own line after "
            "the relevant paragraph. The figures are pre-validated — insert tags directly. "
            "You may also search for additional figures via search_figure_descriptions if needed."
        )
    else:
        user_msg += (
            "If relevant figures are available via search_figure_descriptions, "
            "consider including them using {{FIGURE:bibtex_key:type}} tags after "
            "confirming with get_figure_for_section."
        )

    # Esegui ReAct con retry incrementale per errori transitori
    max_iter = _PREFETCH_MAX_ITERATIONS
    result = None

    # Cache tool-call results across retries to avoid repeating successful searches
    tool_cache: dict[str, str] = {}

    for attempt in range(_TRANSIENT_MAX_RETRIES + 1):
        current_timeout = _TRANSIENT_BASE_TIMEOUT + attempt * _TRANSIENT_TIMEOUT_INCREMENT
        llm = get_llm(tier, max_tokens=16384, timeout_seconds=current_timeout)

        cached_tools = _wrap_tools_with_cache(tools, tool_cache)

        agent = ReActAgent(
            name="SurveyWriterAgent",
            description="Academic survey section writer with pre-fetched context",
            system_prompt=system_prompt,
            tools=cached_tools,
            llm=llm,
            timeout=current_timeout,
            verbose=logger.isEnabledFor(10),
        )

        try:
            result = asyncio.run(
                _run_react_agent(agent, user_msg, max_iterations=max_iter)
            )
            break  # successo

        except TransientLLMError as te:
            if attempt < _TRANSIENT_MAX_RETRIES:
                delay = min(
                    _TRANSIENT_BASE_DELAY * (2 ** attempt),
                    _TRANSIENT_MAX_DELAY,
                )
                next_timeout = _TRANSIENT_BASE_TIMEOUT + (attempt + 1) * _TRANSIENT_TIMEOUT_INCREMENT
                logger.warning(
                    f"[prefetch_agentic] Transient error (attempt {attempt + 1}/"
                    f"{_TRANSIENT_MAX_RETRIES + 1}) per '{outline.title}': "
                    f"{str(te)[:120]} — retry in {delay:.0f}s "
                    f"(timeout {current_timeout:.0f}→{next_timeout:.0f}s, "
                    f"tool_cache={len(tool_cache)} entries)"
                )
                print_warning(
                    f"⏳ Transient error per '{outline.title}' "
                    f"(attempt {attempt + 1}/{_TRANSIENT_MAX_RETRIES + 1}), "
                    f"retry in {delay:.0f}s..."
                )
                time.sleep(delay)
                continue
            else:
                logger.error(
                    f"[prefetch_agentic] Transient error esaurito tutti i retry "
                    f"({_TRANSIENT_MAX_RETRIES + 1} tentativi) per '{outline.title}'"
                )
                return None

        except MaxIterationsError:
            logger.warning(
                f"[prefetch_agentic] Max iterations ({max_iter}) per '{outline.title}' "
                f"— tentativo di recupero risultato parziale"
            )
            result = None
            break
        except MaxTokensError:
            logger.warning(
                f"[prefetch_agentic] MAX_TOKENS per '{outline.title}' — fallback a static"
            )
            return None
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    result = asyncio.run(
                        _run_react_agent(agent, user_msg, max_iterations=max_iter)
                    )
                    break
                except TransientLLMError as te2:
                    if attempt < _TRANSIENT_MAX_RETRIES:
                        delay = min(
                            _TRANSIENT_BASE_DELAY * (2 ** attempt),
                            _TRANSIENT_MAX_DELAY,
                        )
                        logger.warning(
                            f"[prefetch_agentic] Transient after nest_asyncio "
                            f"(attempt {attempt + 1}) per '{outline.title}': "
                            f"{str(te2)[:120]} — retry in {delay:.0f}s"
                        )
                        time.sleep(delay)
                        continue
                except Exception:
                    logger.error(f"[prefetch_agentic] nest_asyncio fallback failed")
                    return None
            else:
                logger.error(f"[prefetch_agentic] RuntimeError: {e}")
                return None
        except Exception as e:
            logger.error(f"[prefetch_agentic] ReAct agent failed: {e}", exc_info=True)
            return None

    elapsed = time.time() - t0

    if result is None:
        logger.warning(
            f"[prefetch_agentic] ReAct returned None per '{outline.title}' "
            f"dopo {elapsed:.1f}s"
        )
        return None

    raw_text = result.response.content if result.response else ""
    logger.info(
        f"[prefetch_agentic] Completato per '{outline.title}': "
        f"{len(raw_text)} chars in {elapsed:.1f}s "
        f"(prefetch: {prefetch_elapsed:.1f}s)"
    )

    if not raw_text or len(raw_text.strip()) < 100:
        logger.warning(
            f"[prefetch_agentic] Risposta troppo corta ({len(raw_text)} chars)"
        )
        return None

    # --- Parse output ---
    parsed = _parse_writer_response(raw_text, outline.key)
    if parsed is None:
        citations = _extract_citations_from_text(raw_text)
        parsed = {
            "text": raw_text.strip(),
            "citations": citations,
            "notes": f"Produced by prefetch_agentic ({elapsed:.1f}s).",
        }

    if "notes" in parsed:
        parsed["notes"] = (
            f"[prefetch_agentic] {parsed['notes']}"
            if not parsed["notes"].startswith("[prefetch_agentic]")
            else parsed["notes"]
        )
    else:
        parsed["notes"] = (
            f"[prefetch_agentic] Completed in {elapsed:.1f}s "
            f"(prefetch: {prefetch_elapsed:.1f}s)."
        )

    return parsed


def _write_section_agentic(
    outline: "SectionOutline",
    force_model: str | None = None,
    critic_feedback: str | None = None,
    max_iterations: int | None = None,
) -> dict | None:
    """Scrive una sezione usando il ReAct Agent multi-step.

    L'agente usa i retrieval tool per cercare autonomamente le
    informazioni necessarie durante la scrittura.

    Args:
        outline: Outline della sezione da scrivere.
        force_model: Override del modello LLM ('flash' o 'pro').
        critic_feedback: Feedback dal Critic Agent (per riscrittura).
        max_iterations: Override del max iterazioni ReAct.

    Returns:
        Dict con {text, citations, notes} o None se il ReAct fallisce.
    """
    from llama_index.core.agent import ReActAgent
    from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools

    tier = force_model or settings.agents.writer_model
    max_iter = max_iterations or _REACT_MAX_ITERATIONS

    logger.info(
        f"[agentic] Avvio ReAct Agent per '{outline.title}' "
        f"con {tier.upper()}, max_iter={max_iter}"
    )

    # --- Costruisci system prompt ---
    special_instructions = _get_special_instructions(outline.section_type)
    critic_section = ""
    if critic_feedback:
        critic_section = (
            "## CRITIC FEEDBACK (from previous iteration)\n\n"
            "Address these issues in your rewrite:\n\n"
            f"{critic_feedback}\n"
        )

    system_prompt = _REACT_SYSTEM_PROMPT.format(
        title=outline.title,
        description=outline.description[:2000],
        key_topics=", ".join(outline.key_topics) if outline.key_topics else "N/A",
        related_categories=", ".join(outline.related_categories) if outline.related_categories else "all",
        target_words=outline.target_length_words,
        special_instructions=special_instructions,
        critic_feedback_section=critic_section,
    )

    # --- Self-evolution: inject skills + lessons overlay ---
    if settings.evolution.enabled or settings.skills.enabled:
        try:
            evo_store = LessonStore(
                path=settings.evolution.lessons_path,
                half_life_days=settings.evolution.half_life_days,
                expiry_days=settings.evolution.expiry_days,
            )
            overlay = generate_full_overlay(
                evo_store,
                section_key=outline.key,
                agent_type="writer",
                skills_dir=settings.skills.skills_dir if settings.skills.enabled else None,
                skills_enabled=settings.skills.enabled,
                max_skill_chars=settings.skills.max_total_chars,
                max_lessons=settings.evolution.max_overlay_lessons,
                max_lesson_chars=settings.evolution.max_overlay_chars,
            )
            if overlay:
                system_prompt = f"{system_prompt}\n\n{overlay}"
                logger.info("[evolution] Injected %d-char overlay for '%s'", len(overlay), outline.key)
        except Exception as e:
            logger.warning("[evolution] Failed to load overlay: %s", e)

    # --- Inizializza tool e agente ---
    default_cat = outline.related_categories[0] if outline.related_categories else None
    tools = get_all_retrieval_tools(default_category=default_cat)
    logger.info(
        f"[agentic] default_category={default_cat!r} "
        f"(related_categories={outline.related_categories})"
    )

    # --- User message: trigger per l'agente ---
    user_msg = (
        f"Search for relevant papers and write the section '{outline.title}'. "
        f"Cover these topics: {', '.join(outline.key_topics[:7])}. "
        f"Target approximately {outline.target_length_words} words with [bibtex_key] citations."
    )

    # --- Esegui il ReAct loop con retry ---
    # Retry limitato: 1 retry per MaxIterationsError con +5 iter,
    # 2 retry per MaxTokensError con -4 iter.
    # Retry con backoff per errori transitori (timeout/503/rate-limit).
    max_token_retries = 2
    max_token_attempt = 0
    max_iter_retried = False
    transient_attempt = 0
    result = None

    # Cache tool-call results across retries to avoid repeating successful searches
    tool_cache: dict[str, str] = {}

    while True:
        current_timeout = _TRANSIENT_BASE_TIMEOUT + transient_attempt * _TRANSIENT_TIMEOUT_INCREMENT
        llm = get_llm(tier, max_tokens=16384, timeout_seconds=current_timeout)

        cached_tools = _wrap_tools_with_cache(tools, tool_cache)

        agent = ReActAgent(
            name="SurveyWriterAgent",
            description="Academic survey section writer with RAG tools",
            system_prompt=system_prompt,
            tools=cached_tools,
            llm=llm,
            timeout=current_timeout,
            verbose=logger.isEnabledFor(10),
        )

        t0 = time.time()
        try:
            result = asyncio.run(
                _run_react_agent(agent, user_msg, max_iterations=max_iter)
            )
            break  # Successo, esci dal loop

        except TransientLLMError as te:
            transient_attempt += 1
            elapsed_attempt = time.time() - t0
            if transient_attempt <= _TRANSIENT_MAX_RETRIES:
                delay = min(
                    _TRANSIENT_BASE_DELAY * (2 ** (transient_attempt - 1)),
                    _TRANSIENT_MAX_DELAY,
                )
                next_timeout = _TRANSIENT_BASE_TIMEOUT + transient_attempt * _TRANSIENT_TIMEOUT_INCREMENT
                logger.warning(
                    f"[agentic] Transient error (attempt {transient_attempt}/"
                    f"{_TRANSIENT_MAX_RETRIES}) per '{outline.title}' dopo "
                    f"{elapsed_attempt:.0f}s: {str(te)[:120]} — retry in {delay:.0f}s "
                    f"(timeout {current_timeout:.0f}→{next_timeout:.0f}s)"
                )
                print_warning(
                    f"⏳ Transient error per '{outline.title}' "
                    f"(attempt {transient_attempt}/{_TRANSIENT_MAX_RETRIES}), "
                    f"retry in {delay:.0f}s..."
                )
                time.sleep(delay)
                continue
            else:
                logger.error(
                    f"[agentic] Transient error esaurito tutti i retry "
                    f"({_TRANSIENT_MAX_RETRIES} tentativi) per '{outline.title}'"
                )
                return None

        except MaxIterationsError as mie:
            elapsed_attempt = time.time() - t0
            # Prova a estrarre risultato parziale dall'agente
            partial = getattr(mie, "partial_result", None) or ""
            if isinstance(partial, str) and len(partial.strip()) >= 100:
                logger.info(
                    f"[agentic] MaxIterations per '{outline.title}' dopo "
                    f"{elapsed_attempt:.0f}s — accetto risultato parziale "
                    f"({len(partial)} chars)"
                )
                # Crea un result mock con il testo parziale
                from types import SimpleNamespace
                result = SimpleNamespace(
                    response=SimpleNamespace(content=partial.strip())
                )
                break

            if not max_iter_retried:
                max_iter_retried = True
                old_iter = max_iter
                max_iter = max_iter + 5
                logger.warning(
                    f"[agentic] Max iterations per '{outline.title}' "
                    f"dopo {elapsed_attempt:.0f}s — unico retry "
                    f"con max_iter {old_iter}→{max_iter}"
                )
                print_warning(
                    f"Max iterations per '{outline.title}', "
                    f"unico retry con più iterazioni ({max_iter})..."
                )
                continue
            else:
                logger.error(
                    f"[agentic] MaxIterations anche al retry per "
                    f"'{outline.title}' — rinuncio"
                )
                return None

        except MaxTokensError:
            max_token_attempt += 1
            elapsed_attempt = time.time() - t0
            if max_token_attempt < max_token_retries:
                old_iter = max_iter
                max_iter = max(5, max_iter - 4)
                logger.warning(
                    f"[agentic] MAX_TOKENS al tentativo {max_token_attempt}/{max_token_retries} "
                    f"dopo {elapsed_attempt:.0f}s — retry con max_iter {old_iter}→{max_iter}"
                )
                print_warning(
                    f"MAX_TOKENS per '{outline.title}' (tentativo {max_token_attempt}/{max_token_retries}), "
                    f"retry con meno iterazioni ({max_iter})..."
                )
                continue
            else:
                logger.error(
                    f"[agentic] MAX_TOKENS dopo {max_token_retries} tentativi "
                    f"per '{outline.title}' — rinuncio"
                )
                return None

        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                logger.warning(
                    "[agentic] Detected running event loop. "
                    "Attempting nest_asyncio fallback."
                )
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    result = asyncio.run(
                        _run_react_agent(agent, user_msg, max_iterations=max_iter)
                    )
                    break
                except (MaxTokensError, MaxIterationsError) as inner:
                    logger.error(
                        f"[agentic] {type(inner).__name__} in nest_asyncio fallback"
                    )
                    return None
                except Exception as nested_e:
                    logger.error(f"[agentic] nest_asyncio fallback failed: {nested_e}")
                    return None
            else:
                logger.error(f"[agentic] RuntimeError: {e}")
                return None

        except Exception as e:
            logger.error(f"[agentic] ReAct agent failed: {e}", exc_info=True)
            return None

    elapsed = time.time() - t0
    if result is None:
        logger.warning(f"[agentic] ReAct agent returned None after {elapsed:.1f}s")
        return None

    raw_text = result.response.content if result.response else ""
    logger.info(
        f"[agentic] ReAct completato per '{outline.title}': "
        f"{len(raw_text)} chars in {elapsed:.1f}s"
    )

    if not raw_text or len(raw_text.strip()) < 100:
        logger.warning(
            f"[agentic] Risposta troppo corta ({len(raw_text)} chars), "
            f"fallback a static"
        )
        return None

    # --- Parse: prova JSON, altrimenti testo puro ---
    parsed = _parse_writer_response(raw_text, outline.key)
    if parsed is None:
        # Il ReAct agent tipicamente produce direttamente il testo,
        # non JSON. Tratta come testo puro.
        citations = _extract_citations_from_text(raw_text)
        parsed = {
            "text": raw_text.strip(),
            "citations": citations,
            "notes": f"Produced by ReAct agent ({elapsed:.1f}s, agentic mode).",
        }

    # Aggiungi nota sulla modalità
    if "notes" in parsed:
        parsed["notes"] = (
            f"[agentic] {parsed['notes']}"
            if not parsed["notes"].startswith("[agentic]")
            else parsed["notes"]
        )
    else:
        parsed["notes"] = f"[agentic] ReAct agent completed in {elapsed:.1f}s."

    return parsed


async def _run_react_agent(agent, user_msg: str, max_iterations: int = 7):
    """Esegue il ReAct agent in modo asincrono.

    Args:
        agent: Istanza ``ReActAgent`` configurata.
        user_msg: Messaggio utente per l'agente.
        max_iterations: Massimo numero di iterazioni ReAct.

    Returns:
        AgentOutput con la risposta finale, o None se fallisce.

    Raises:
        MaxTokensError: Se la risposta è stata troncata per MAX_TOKENS.
    """
    try:
        handler = agent.run(user_msg=user_msg, max_iterations=max_iterations)
        result = await handler
        return result
    except (RuntimeError, Exception) as e:
        err_str = str(e)
        if "MAX_TOKENS" in err_str:
            logger.warning(f"[agentic] _run_react_agent: MAX_TOKENS hit — will retry")
            raise MaxTokensError(err_str) from e
        if "Max iterations" in err_str:
            logger.warning(
                f"[agentic] _run_react_agent: max iterations ({max_iterations}) "
                f"raggiunto — will retry con più iterazioni"
            )
            raise MaxIterationsError(err_str) from e
        err_lower = err_str.lower()
        if any(p.lower() in err_lower for p in _TRANSIENT_PATTERNS):
            logger.warning(
                f"[agentic] _run_react_agent: transient error detected — {err_str[:200]}"
            )
            raise TransientLLMError(err_str) from e
        logger.error(f"[agentic] _run_react_agent error: {e}", exc_info=True)
        return None


def _canonicalize_citation_key(
    key: str,
    whitelist: set[str],
) -> str | None:
    """Resolve an omitted manifest hash only when the match is unambiguous.

    DBLP-backed keys end with an eight-hex digest, for example
    ``Abbas_2025_Small_ea340ecf``. Long generations sometimes copy the human
    readable prefix but omit that digest. Recovering a unique exact prefix is
    safe because the candidate still comes from the strict whitelist; multiple
    candidates remain rejected so citation handling stays fail-closed.
    """
    if key in whitelist:
        return key
    prefix = f"{key}_"
    candidates = [
        candidate
        for candidate in whitelist
        if candidate.startswith(prefix)
        and re.fullmatch(r"[0-9a-fA-F]{8}", candidate[len(prefix):])
    ]
    return candidates[0] if len(candidates) == 1 else None


def _enforce_citation_whitelist(
    text: str,
    whitelist: set[str] | None,
) -> tuple[str, list[str]]:
    """Rimuove dal testo le citazioni di paper non in whitelist (v3.6 C3).

    Cerca pattern ``[bibtex_key]`` (con almeno un underscore + cifra, come
    nelle convenzioni del progetto) e:
    - se ``bibtex_key`` è nella whitelist → lascia;
    - se manca solo l'hash DBLP e il prefisso identifica una singola chiave
      verificata → ripristina la chiave completa;
    - altrimenti → rimuove la citazione (mantiene il testo circostante)
      e logga il caso per audit.

    Se ``whitelist`` è None non applica il filtro (back-compat). Un set vuoto
    significa invece che non esiste alcuna fonte citabile e rimuove quindi
    ogni chiave bibliografica, così il Critic può bloccare subito la sezione.

    Returns:
        (clean_text, removed_keys)
    """
    if whitelist is None:
        return text, []

    removed: list[str] = []

    def _replace(match: "re.Match[str]") -> str:
        content = match.group(1)
        raw_keys = [key.strip() for key in content.split(",")]
        is_citation_group = raw_keys and all(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", key)
            and any(char.isdigit() for char in key)
            for key in raw_keys
        )
        if not is_citation_group:
            return match.group(0)
        kept: list[str] = []
        for key in raw_keys:
            canonical = _canonicalize_citation_key(key, whitelist)
            if canonical is None:
                removed.append(key)
            else:
                kept.append(canonical)
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = re.sub(r"\[([^\]]+)\]", _replace, text)
    # Compatta doppi spazi/spazi prima di punteggiatura
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned, removed


def _build_citable_keys_from_chunks(chunks: list[dict] | None) -> set[str]:
    """Estrae i bibtex_keys citable dai chunk recuperati."""
    if not chunks:
        return set()
    out: set[str] = set()
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        if meta.get("is_citable") and meta.get("bibtex_key"):
            out.add(meta["bibtex_key"])
    # I payload Qdrant possono essere antecedenti all'ultimo audit DBLP:
    # interseca sempre il contesto con il master BibTeX fail-closed.
    return out & _load_global_citable_keys()


_CITABLE_KEYS_CACHE: set[str] | None = None


def _load_global_citable_keys() -> set[str]:
    """Carica con cache le chiavi del master BibTeX verificato.

    Il master ``references.bib`` è la fonte di verità dell'export fail-closed:
    Qdrant può contenere payload ``is_citable`` obsoleti rispetto all'ultimo
    audit DBLP. Usare il master evita sia citazioni non esportabili sia l'apertura
    concorrente del database Qdrant locale durante Writer e Critic.
    """
    global _CITABLE_KEYS_CACHE
    if _CITABLE_KEYS_CACHE is not None:
        return _CITABLE_KEYS_CACHE

    try:
        from paperhunter.ingestion.bibtex_generator import load_master_bib

        master_path = Path(settings.output.base_dir) / "references.bib"
        _CITABLE_KEYS_CACHE = set(load_master_bib(master_path))
        logger.info(
            "Citation whitelist globale caricata dal master BibTeX: %d chiavi verificate",
            len(_CITABLE_KEYS_CACHE),
        )
        return _CITABLE_KEYS_CACHE
    except Exception as exc:
        logger.error("Impossibile caricare il master BibTeX per la whitelist: %s", exc)
        _CITABLE_KEYS_CACHE = set()
        return _CITABLE_KEYS_CACHE


def _build_section_result(
    parsed: dict,
    outline: SectionOutline,
    section_key: str,
    iteration: int,
    elapsed: float,
    errors: list[str],
    *,
    mode: str = "static",
    queries: list[str] | None = None,
    chunk_ids: list[str] | None = None,
    citable_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Costruisce il risultato di una sezione scritta da qualsiasi modalità.

    Prende l'output parsed (text + citations + notes) e costruisce il
    SectionDraft e il dict di aggiornamento stato compatibile con LangGraph.

    Args:
        parsed: Dict con chiavi 'text', 'citations', 'notes'.
        outline: Outline della sezione.
        section_key: Chiave della sezione.
        iteration: Numero di iterazione corrente.
        elapsed: Tempo trascorso in secondi.
        errors: Lista errori accumulati.
        mode: Modalità utilizzata ('static' o 'agentic').
        queries: Query RAG usate (solo modalità static).
        chunk_ids: Chunk ID recuperati (solo modalità static).

    Returns:
        Dict parziale per aggiornamento stato LangGraph.
    """
    section_text = parsed["text"]

    # --- v3.6: enforcement citation whitelist (strict mode) ---
    if (
        getattr(settings.agents, "citation_whitelist_strict", False)
        and citable_keys is not None
    ):
        section_text, removed = _enforce_citation_whitelist(section_text, citable_keys)
        if removed:
            unique_removed = sorted(set(removed))
            logger.warning(
                "Citation whitelist: rimosse %d citazioni non-citable da '%s': %s",
                len(removed),
                section_key,
                unique_removed[:10],
            )
            errors.append(
                f"citation_whitelist: stripped non-citable refs: {unique_removed[:10]}"
            )

    word_count = _count_words(section_text)

    # La fonte di verità del draft è il testo effettivamente conservato dopo
    # l'enforcement. I record del JSON LLM possono ancora contenere citazioni
    # appena rimosse dalla whitelist: salvarle renderebbe incoerenti Critic,
    # provenance e bibliografia finale.
    text_citations: set[str] = set()
    for group in re.findall(r"\[([^\]]+)\]", section_text):
        for raw_key in group.split(","):
            key = raw_key.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", key):
                text_citations.add(key)

    parsed_citations = parsed.get("citations", [])
    if (
        getattr(settings.agents, "citation_whitelist_strict", False)
        and citable_keys is not None
    ):
        canonicalized_citations: list[dict] = []
        for citation in parsed_citations:
            canonical = _canonicalize_citation_key(
                str(citation.get("bibtex_key") or ""),
                citable_keys,
            )
            if canonical is not None:
                canonicalized_citations.append(
                    {**citation, "bibtex_key": canonical}
                )
        parsed_citations = canonicalized_citations
    parsed_citations = [
        citation
        for citation in parsed_citations
        if citation.get("bibtex_key") in text_citations
    ]

    citation_records = [
        CitationRecord(
            bibtex_key=c["bibtex_key"],
            claim=c.get("claim", ""),
            chunk_id=c.get("chunk_id"),
            retrieval_score=c.get("retrieval_score", 0.0),
        )
        for c in parsed_citations
    ]

    # Verifica che tutte le citazioni nel testo siano tracciate.
    tracked_keys = {c.bibtex_key for c in citation_records}
    untracked = {
        k for k in text_citations
        if "_" in k and any(c.isdigit() for c in k) and k not in tracked_keys
    }
    if untracked:
        logger.warning(f"Citazioni nel testo non tracciate: {untracked}")
        for key in untracked:
            citation_records.append(CitationRecord(
                bibtex_key=key,
                claim="(auto-detected from text)",
                chunk_id=None,
                retrieval_score=0.0,
            ))

    writer_notes = parsed.get("notes", "")
    if mode == "agentic" and not writer_notes.startswith("[agentic]"):
        writer_notes = f"[agentic] {writer_notes}"

    draft = SectionDraft(
        section_key=section_key,
        title=outline.title,
        text=section_text,
        citations=citation_records,
        word_count=word_count,
        status=SectionStatus.DRAFTED,
        iteration=iteration,
        rag_queries=queries or [],
        retrieved_chunk_ids=chunk_ids or [],
        writer_notes=writer_notes,
        drafted_at=datetime.now(),
    )

    # Logga warning per lunghezze anomale
    min_words = settings.agents.min_section_words
    max_words = settings.agents.max_section_words
    if word_count < min_words and outline.section_type not in ("abstract", "conclusion"):
        logger.warning(
            f"Sezione '{section_key}' troppo corta: {word_count} parole (min: {min_words})"
        )
    if word_count > max_words:
        logger.warning(
            f"Sezione '{section_key}' troppo lunga: {word_count} parole (max: {max_words})"
        )

    logger.info(
        f"Sezione '{section_key}' completata [{mode}]: {word_count} parole, "
        f"{len(citation_records)} citazioni, {elapsed:.1f}s"
    )

    return {
        "sections": {section_key: draft.to_dict()},
        "current_section_key": section_key,
        "errors": errors if errors else [],
    }


def _minimum_acceptable_revision_words(
    existing_draft: SectionDraft,
    outline: SectionOutline,
) -> int:
    """Prevent legacy drafts with a stale zero count from being shortened."""
    existing_words = existing_draft.word_count or _count_words(existing_draft.text or "")
    return max(
        settings.agents.min_section_words,
        int(outline.target_length_words * 0.80),
        existing_words,
    )


def _run_agentic_writer_with_clean_retrieval(runner, /, *args, **kwargs):
    """Esegue un writer ReAct isolando i singleton Qdrant locali.

    Qdrant in modalità embedded mantiene un lock esclusivo sulla directory.
    Un agente ReAct può aprire il singleton dei retrieval tool mentre la
    pipeline poi deve creare un HybridRetriever o il Critic. Reset esplicito
    prima e dopo ogni sezione evita che il lock attraversi i nodi.
    """
    from paperhunter.agents.tools.retrieval_tools import reset_singletons

    reset_singletons()
    try:
        return runner(*args, **kwargs)
    finally:
        reset_singletons()


# ============================================================================
# Core: write_section
# ============================================================================

def write_section(
    state: SurveyState,
    section_key: str,
    *,
    retriever=None,
    reranker=None,
    graph_builder=None,
    force_model: str | None = None,
    use_reranker: bool = True,
    use_graph: bool = False,
    top_k_retrieval: int = 25,
    n_rag_queries: int = 5,
    n_total_sections: int | None = None,
    max_context_chars: int | None = None,
) -> dict[str, Any]:
    """Scrive una singola sezione del survey.

    Funzione core del Writer Agent. Può essere usata standalone
    o come parte del nodo LangGraph `write`.

    Args:
        state: Stato corrente della pipeline.
        section_key: Chiave della sezione da scrivere (dall'outline).
        retriever: HybridRetriever pre-inizializzato (opzionale).
        reranker: GeminiReranker pre-inizializzato (opzionale).
        graph_builder: PropertyGraphBuilder pre-inizializzato (opzionale).
        force_model: Override del modello LLM ('flash' o 'pro').
        use_reranker: Se usare il reranker Gemini dopo il retrieval.
        use_graph: Se usare il PropertyGraph per contesto aggiuntivo.
        top_k_retrieval: Numero massimo di chunk da recuperare.
        n_rag_queries: Numero di query RAG da generare.
        n_total_sections: Numero totale di sezioni nel paper (per budget proportional).
        max_context_chars: Override diretto del budget chars per questa sezione.

    Returns:
        Dict parziale per aggiornamento dello stato LangGraph con
        ``sections``, ``current_section_key``, ``errors``.
    """
    errors: list[str] = []

    # --- Trova l'outline per questa sezione ---
    outline_list = get_outline_from_state(state)
    target_outline: SectionOutline | None = None
    for o in outline_list:
        if o.key == section_key:
            target_outline = o
            break

    if target_outline is None:
        err = f"Sezione '{section_key}' non trovata nell'outline."
        logger.error(err)
        return {"sections": {}, "provenance": [], "errors": [err]}

    # --- Controlla se la sezione è già stata scritta (skip abstract/conclusion) ---
    existing_sections = get_sections_from_state(state)
    existing_draft = existing_sections.get(section_key)
    if existing_draft and existing_draft.status in (SectionStatus.VERIFIED, SectionStatus.FINALIZED):
        logger.info(f"Sezione '{section_key}' già verificata/finalizzata, skip.")
        return {}

    # Determina il numero di iterazione
    current_iteration = (existing_draft.iteration + 1) if existing_draft else 1
    max_iterations = settings.agents.max_writer_iterations

    if current_iteration > max_iterations:
        logger.warning(
            f"Sezione '{section_key}' ha raggiunto il max di {max_iterations} iterazioni."
        )
        return {}

    logger.info(
        f"Writing sezione '{section_key}' (iter {current_iteration}/{max_iterations}): "
        f"{target_outline.title}"
    )

    t0 = time.time()

    # --- Modalità duale: agentic vs static ---
    writer_mode = settings.agents.writer_mode
    critic_feedback = existing_draft.critic_feedback if existing_draft else ""

    if writer_mode == "agentic":
        logger.info(f"[agentic] Modalità ReAct attiva per '{section_key}'")
        parsed = _run_agentic_writer_with_clean_retrieval(
            _write_section_agentic,
            target_outline,
            force_model=force_model,
            critic_feedback=critic_feedback,
        )
        if parsed is not None:
            elapsed = time.time() - t0
            return _build_section_result(
                parsed, target_outline, section_key,
                current_iteration, elapsed, errors,
                mode="agentic",
                citable_keys=_load_global_citable_keys(),
            )
        else:
            logger.warning(
                f"[agentic] Fallback a modalità static per '{section_key}'"
            )
            print_warning(
                f"ReAct agent fallito per '{section_key}', fallback a static"
            )

    elif writer_mode == "prefetch_agentic":
        logger.info(f"[prefetch_agentic] Modalità ibrida per '{section_key}'")
        parsed = _run_agentic_writer_with_clean_retrieval(
            _write_section_prefetch_agentic,
            target_outline,
            force_model=force_model,
            critic_feedback=critic_feedback,
            retriever=retriever,
            reranker=reranker,
            graph_builder=graph_builder,
            n_rag_queries=n_rag_queries,
            top_k_retrieval=top_k_retrieval,
            use_reranker=use_reranker,
            use_graph=use_graph,
            n_total_sections=n_total_sections,
            max_context_chars=max_context_chars,
        )
        if parsed is not None:
            elapsed = time.time() - t0
            return _build_section_result(
                parsed, target_outline, section_key,
                current_iteration, elapsed, errors,
                mode="prefetch_agentic",
                citable_keys=_load_global_citable_keys(),
            )
        else:
            logger.warning(
                f"[prefetch_agentic] Fallback a modalità static per '{section_key}'"
            )
            print_warning(
                f"Prefetch-agentic fallito per '{section_key}', fallback a static"
            )

    # --- Static mode (o fallback da agentic/prefetch_agentic) ---

    # --- Step 1: Genera query RAG ---
    if critic_feedback:
        # Se c'è feedback dal Critic, integra nelle query
        logger.info(f"Integrazione feedback Critic per riscrittura: {critic_feedback[:100]}...")

    queries = _generate_rag_queries(target_outline, n_queries=n_rag_queries)
    if critic_feedback:
        # Aggiungi query specifiche dal feedback
        queries.append(f"evidence for claims in {target_outline.title}")
        queries = queries[:n_rag_queries + 2]

    logger.info(f"Query RAG generate ({len(queries)}): {[q[:60] for q in queries]}")

    # --- Step 2: Retrieval ---
    chunks, chunk_ids = _retrieve_context_for_section(
        queries,
        target_outline,
        retriever=retriever,
        reranker=reranker,
        graph_builder=graph_builder,
        top_k_final=top_k_retrieval,
        use_reranker=use_reranker,
        use_graph=use_graph,
    )

    if not chunks:
        err = f"Nessun chunk recuperato per sezione '{section_key}'."
        logger.warning(err)
        return {"sections": {}, "provenance": [], "errors": [err]}

    # --- Step 3: Formatta contesto e prompt (budget-aware) ---
    # Se n_total_sections non fornito, calcolalo dall'outline
    if n_total_sections is None:
        n_total_sections = len(outline_list)
    context_text = _format_context_for_prompt(
        chunks,
        max_chars=max_context_chars,
        n_total_sections=n_total_sections,
        prefer_summaries=True,
        related_categories=target_outline.related_categories or None,
    )
    citable_keys = set(_extract_citable_keys_from_formatted_context(context_text))
    minimum_sources = settings.agents.min_citations_per_section
    if len(citable_keys) < minimum_sources:
        err = (
            f"Evidence coverage insufficiente per '{section_key}' dopo il "
            f"budgeting del prompt: {len(citable_keys)} fonti citabili distinte, "
            f"minimo {minimum_sources}. Writer non chiamato: la sezione non "
            "potrebbe superare il Critic."
        )
        logger.warning(err)
        return {"sections": {}, "provenance": [], "errors": [err]}

    special_instructions = _get_special_instructions(target_outline.section_type)

    # Se c'è feedback del Critic, aggiungilo al prompt
    if critic_feedback:
        special_instructions += (
            f"\n\n### CRITIC FEEDBACK (from previous iteration)\n\n"
            f"The following feedback was provided by the Critic agent. "
            f"Address these issues in your rewrite:\n\n{critic_feedback}\n"
        )

    prompt = _WRITER_PROMPT_TEMPLATE.format(
        title=target_outline.title,
        description=target_outline.description,
        key_topics=", ".join(target_outline.key_topics) if target_outline.key_topics else "N/A",
        target_words=target_outline.target_length_words,
        retrieved_context=context_text,
        special_instructions=special_instructions,
    )

    # --- Self-evolution: inject skills + lessons overlay (static mode) ---
    if settings.evolution.enabled or settings.skills.enabled:
        try:
            evo_store = LessonStore(
                path=settings.evolution.lessons_path,
                half_life_days=settings.evolution.half_life_days,
                expiry_days=settings.evolution.expiry_days,
            )
            overlay = generate_full_overlay(
                evo_store,
                section_key=section_key,
                agent_type="writer",
                skills_dir=settings.skills.skills_dir if settings.skills.enabled else None,
                skills_enabled=settings.skills.enabled,
                max_skill_chars=settings.skills.max_total_chars,
                max_lessons=settings.evolution.max_overlay_lessons,
                max_lesson_chars=settings.evolution.max_overlay_chars,
            )
            if overlay:
                prompt = f"{prompt}\n\n{overlay}"
                logger.info("[evolution] Injected %d-char overlay for '%s' (static)", len(overlay), section_key)
        except Exception as e:
            logger.warning("[evolution] Failed to load overlay: %s", e)

    # --- Step 4: Genera testo con LLM ---
    tier = force_model or settings.agents.writer_model
    logger.info(
        f"Generazione testo per '{section_key}' con {tier.upper()} "
        f"({len(chunks)} chunk di contesto, ~{target_outline.target_length_words} parole target)"
    )

    max_retries = settings.summarization.pro_max_retries
    base_delay = settings.summarization.pro_base_delay

    # A 1,000-3,000 word section plus hidden reasoning does not fit reliably in
    # 16K shared completion tokens.  High (not max) reasoning and 32K leave a
    # safe visible-prose budget; the adapter rejects any remaining truncation.
    llm = get_llm(
        tier,
        max_tokens=32768,
        reasoning_effort="high",
    )

    try:
        raw_response = llm_call_with_retry(
            llm, prompt,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=120.0,
        )
    except Exception as e:
        err = f"Errore LLM per sezione '{section_key}': {e}"
        logger.error(err)
        errors.append(err)
        return {"sections": {}, "provenance": [], "errors": errors}

    elapsed = time.time() - t0
    logger.info(f"Risposta LLM per '{section_key}': {len(raw_response)} chars in {elapsed:.1f}s")

    # --- Step 5: Parse risposta ---
    parsed = _parse_writer_response(raw_response, section_key)
    if parsed is None:
        err = f"Impossibile parsare la risposta per '{section_key}'."
        logger.error(err)
        errors.append(err)
        return {"sections": {}, "provenance": [], "errors": errors}

    elapsed = time.time() - t0
    result = _build_section_result(
        parsed, target_outline, section_key,
        current_iteration, elapsed, errors,
        mode="static",
        queries=queries,
        chunk_ids=chunk_ids,
        citable_keys=citable_keys,
    )

    if existing_draft is not None:
        candidate_data = result.get("sections", {}).get(section_key)
        if candidate_data:
            candidate = SectionDraft.from_dict(candidate_data)
            required_words = _minimum_acceptable_revision_words(
                existing_draft,
                target_outline,
            )
            if candidate.word_count < required_words:
                message = (
                    f"Revision for '{section_key}' rejected: {candidate.word_count} "
                    f"words would regress below the preserved quality floor of "
                    f"{required_words}."
                )
                logger.warning(message)
                return {
                    "sections": {},
                    "provenance": [],
                    "current_phase": PipelinePhase.WRITING.value,
                    "errors": [message],
                    "metadata": {
                        "last_writer_time": round(elapsed, 1),
                        "last_writer_mode": "static",
                        "revision_preserved": section_key,
                    },
                }

    return result


# ============================================================================
# Writer section checkpoints (per-section granularity)
# ============================================================================

_WRITER_CHECKPOINT_SUBDIR = "writer_sections"


def _get_writer_checkpoint_dir() -> Path:
    """Directory per i checkpoint per-sezione del writer."""
    base = Path(settings.output.checkpoints_dir)
    d = base / _WRITER_CHECKPOINT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_section_checkpoint(section_key: str, section_data: dict) -> Path:
    """Salva un checkpoint per una singola sezione completata.

    Il file è ``<section_key>.json`` e contiene i dati della SectionDraft
    serializzati, più un timestamp.
    """
    cp_dir = _get_writer_checkpoint_dir()
    filepath = cp_dir / f"{section_key}.json"
    payload = {
        "section_key": section_key,
        "saved_at": datetime.now().isoformat(),
        "data": section_data,
    }
    filepath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info(f"[checkpoint] Sezione '{section_key}' salvata → {filepath.name}")
    return filepath


def _load_section_checkpoints() -> dict[str, dict]:
    """Carica tutti i checkpoint per-sezione esistenti.

    Returns:
        Dict {section_key: section_data} per ogni sezione trovata.
    """
    cp_dir = _get_writer_checkpoint_dir()
    if not cp_dir.exists():
        return {}

    loaded: dict[str, dict] = {}
    for fp in sorted(cp_dir.glob("*.json")):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            key = payload.get("section_key", fp.stem)
            loaded[key] = payload["data"]
            logger.debug(
                f"[checkpoint] Caricata sezione '{key}' "
                f"(salvata: {payload.get('saved_at', '?')})"
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[checkpoint] Impossibile caricare {fp.name}: {e}")
    return loaded


def delete_section_checkpoint(section_key: str) -> bool:
    """Elimina il checkpoint di una singola sezione.

    Args:
        section_key: Chiave della sezione da eliminare.

    Returns:
        True se il file è stato eliminato, False se non esisteva.
    """
    cp_dir = _get_writer_checkpoint_dir()
    fp = cp_dir / f"{section_key}.json"
    if fp.exists():
        fp.unlink()
        logger.info(f"[checkpoint] Eliminato checkpoint sezione '{section_key}'")
        return True
    return False


def clean_section_checkpoints() -> int:
    """Elimina tutti i checkpoint per-sezione del writer.

    Returns:
        Numero di file eliminati.
    """
    cp_dir = _get_writer_checkpoint_dir()
    if not cp_dir.exists():
        return 0
    removed = 0
    for fp in cp_dir.glob("*.json"):
        fp.unlink()
        removed += 1
    if removed:
        logger.info(f"[checkpoint] Eliminati {removed} checkpoint sezione writer")
    return removed


def list_section_checkpoints() -> list[dict[str, Any]]:
    """Lista tutti i checkpoint per-sezione esistenti.

    Returns:
        Lista di dict con section_key, saved_at, word_count, title.
    """
    cp_dir = _get_writer_checkpoint_dir()
    if not cp_dir.exists():
        return []

    result = []
    for fp in sorted(cp_dir.glob("*.json")):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            data = payload.get("data", {})
            result.append({
                "section_key": payload.get("section_key", fp.stem),
                "saved_at": payload.get("saved_at", "?"),
                "title": data.get("title", "?"),
                "word_count": data.get("word_count", 0),
                "citations": len(data.get("citations", [])),
                "status": data.get("status", "?"),
                "file": fp.name,
            })
        except (json.JSONDecodeError, KeyError):
            result.append({
                "section_key": fp.stem,
                "saved_at": "?",
                "title": "?",
                "word_count": 0,
                "citations": 0,
                "status": "corrupted",
                "file": fp.name,
            })
    return result


# ============================================================================
# Write all sections
# ============================================================================

def write_all_sections(
    state: SurveyState,
    *,
    retriever=None,
    reranker=None,
    graph_builder=None,
    force_model: str | None = None,
    use_reranker: bool = True,
    use_graph: bool = False,
    skip_types: set[str] | None = None,
    max_context_chars: int | None = None,
) -> dict[str, Any]:
    """Scrive tutte le sezioni dell'outline che non sono ancora drafted.

    Itera su tutte le sezioni dell'outline in ordine, saltando
    abstract e conclusion (che saranno generati dall'Assembler).

    Args:
        state: Stato corrente.
        retriever: HybridRetriever pre-inizializzato.
        reranker: GeminiReranker pre-inizializzato.
        graph_builder: PropertyGraphBuilder pre-inizializzato.
        force_model: Override del modello LLM.
        use_reranker: Se usare il reranker.
        use_graph: Se usare il PropertyGraph.
        skip_types: Tipi di sezione da saltare (default: {"abstract"}).
        max_context_chars: Override diretto del budget chars per sezione.

    Returns:
        Dict parziale con tutte le sezioni scritte e eventuali errori.
    """
    if skip_types is None:
        skip_types = {"abstract"}  # Abstract generated by Assembler

    outline = get_outline_from_state(state)
    existing_sections = get_sections_from_state(state)
    errors: list[str] = []
    all_sections: dict[str, dict] = {}

    # --- Carica checkpoint per-sezione esistenti (resume) ---
    resumed_count = 0
    if not settings.agents.force_rewrite:
        cached_sections = _load_section_checkpoints()
        if cached_sections:
            for key, data in cached_sections.items():
                if key not in existing_sections:
                    all_sections[key] = data
                    resumed_count += 1
            if resumed_count:
                logger.info(
                    f"[checkpoint] Ripristinate {resumed_count} sezioni da checkpoint: "
                    f"{list(cached_sections.keys())}"
                )
                print_success(
                    f"Ripristinate {resumed_count} sezioni da checkpoint precedente"
                )
    else:
        logger.info("[force_rewrite] Checkpoint sezione ignorati — riscrittura forzata")

    # Filtra sezioni da scrivere (escludi già presenti nello stato E nei checkpoint)
    sections_to_write = [
        o for o in outline
        if o.section_type not in skip_types
        and o.key not in existing_sections
        and o.key not in all_sections
    ]

    if not sections_to_write:
        logger.info("Nessuna nuova sezione da scrivere.")
        if all_sections:
            return {
                "sections": all_sections,
                "current_phase": PipelinePhase.WRITING.value,
            }
        return {"current_phase": PipelinePhase.WRITING.value}

    # Calcola numero totale sezioni per budget proportional
    n_total_sections = len(outline)

    # Log budget per sezione
    section_budget = _compute_section_budget(
        n_total_sections=n_total_sections,
        max_chars_override=max_context_chars,
    )
    logger.info(
        f"Sezioni da scrivere: {len(sections_to_write)}/{n_total_sections}, "
        f"budget contesto per sezione: {section_budget:,} chars"
    )

    # Log tabella sezioni
    data = [
        {"#": str(o.order), "Key": o.key, "Title": o.title,
         "Type": o.section_type, "Words": str(o.target_length_words)}
        for o in sections_to_write
    ]
    print_table("Sezioni da scrivere", data)

    t0 = time.time()
    written_count = 0
    failed_count = 0

    for i, section_outline in enumerate(sections_to_write):
        # Close any Qdrant singleton left open by the previous section's ReAct agent
        # so that _retrieve_context_for_section() can open its own client cleanly.
        try:
            from paperhunter.agents.tools.retrieval_tools import reset_singletons
            reset_singletons()
        except Exception:
            pass

        logger.info(
            f"\n{'=' * 60}\n"
            f"Scrittura sezione {i + 1}/{len(sections_to_write)}: "
            f"{section_outline.title}\n"
            f"{'=' * 60}"
        )

        try:
            result = write_section(
                # Merge current state with already-written sections
                {**state, "sections": {**state.get("sections", {}), **all_sections}},
                section_outline.key,
                retriever=retriever,
                reranker=reranker,
                graph_builder=graph_builder,
                force_model=force_model,
                use_reranker=use_reranker,
                use_graph=use_graph,
                n_total_sections=n_total_sections,
                max_context_chars=max_context_chars,
            )

            if "sections" in result:
                all_sections.update(result["sections"])
                written_count += 1
                draft_data = result["sections"].get(section_outline.key, {})
                wc = draft_data.get("word_count", 0) if isinstance(draft_data, dict) else 0
                nc = len(draft_data.get("citations", [])) if isinstance(draft_data, dict) else 0
                print_success(
                    f"Sezione '{section_outline.key}': {wc} parole, {nc} citazioni"
                )
                # --- Checkpoint per-sezione ---
                if isinstance(draft_data, dict) and draft_data:
                    _save_section_checkpoint(section_outline.key, draft_data)
            if "errors" in result and result["errors"]:
                errors.extend(result["errors"])

        except Exception as e:
            err = f"Errore critico scrivendo '{section_outline.key}': {e}"
            logger.error(err, exc_info=True)
            print_error(err)
            errors.append(err)
            failed_count += 1

    # Rilascia il lock Qdrant solo alla fine di tutto il ciclo di scrittura
    # (non tra una sezione e l'altra, per evitare lock contention su local storage)
    try:
        from paperhunter.agents.tools.retrieval_tools import reset_singletons
        reset_singletons()
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info(
        f"\nScrittura completata: {written_count} sezioni scritte, "
        f"{resumed_count} ripristinate da checkpoint, "
        f"{failed_count} fallite, {elapsed:.1f}s totali"
    )

    return {
        "sections": all_sections,
        "current_phase": PipelinePhase.WRITING.value,
        "errors": errors if errors else [],
        "metadata": {
            **state.get("metadata", {}),
            "writer_sections_written": written_count,
            "writer_sections_resumed": resumed_count,
            "writer_sections_failed": failed_count,
            "writer_total_time": round(elapsed, 1),
            "writer_model": force_model or settings.agents.writer_model,
        },
    }


# ============================================================================
# LangGraph node function
# ============================================================================

def write(state: SurveyState) -> dict[str, Any]:
    """Nodo LangGraph: Writer Agent.

    Scrive tutte le sezioni dell'outline non ancora presenti nello stato.
    Salta lo abstract (generato dall'Assembler). Se una sezione è in
    stato NEEDS_REVISION, la riscrive con il feedback del Critic.

    Args:
        state: Stato corrente della pipeline LangGraph.

    Returns:
        Dict parziale con ``sections``, ``current_phase``, ``errors``, ``metadata``.
    """
    logger.info("=" * 60)
    logger.info("WRITER AGENT — Inizio scrittura sezioni")
    logger.info(f"Modalità: {settings.agents.writer_mode}")
    logger.info("=" * 60)

    # Verifica prerequisiti
    outline = get_outline_from_state(state)
    if not outline:
        err = "Nessun outline disponibile. Eseguire prima il Planner."
        logger.error(err)
        print_error(err)
        return {
            "current_phase": PipelinePhase.FAILED.value,
            "errors": [err],
        }

    l2_summaries = get_l2_summaries_from_state(state)
    logger.info(
        f"Outline: {len(outline)} sezioni, "
        f"L2: {len(l2_summaries)} categorie"
    )

    # Identifica sezioni che necessitano riscrittura (feedback dal Critic)
    existing = get_sections_from_state(state)
    needs_revision = {
        k: v for k, v in existing.items()
        if v.status == SectionStatus.NEEDS_REVISION
    }

    if needs_revision:
        logger.info(
            f"{len(needs_revision)} sezioni da riscrivere: "
            f"{list(needs_revision.keys())}"
        )

    # Configura reranker (una volta per tutto il batch).
    # HybridRetriever NON viene condiviso: ogni sezione crea il proprio e lo
    # chiude subito dopo il retrieval, così il lock su qdrant_db viene rilasciato
    # prima che il ReAct agent apra il suo singleton in retrieval_tools.
    retriever = None
    reranker = None
    graph_builder = None

    try:
        from paperhunter.indexing.reranker import GeminiReranker
        reranker = GeminiReranker()
    except Exception as e:
        logger.warning(f"Impossibile inizializzare GeminiReranker: {e}")

    # GraphRAG opzionale
    use_graph = settings.neo4j.enabled if hasattr(settings, 'neo4j') else False
    if use_graph:
        try:
            from paperhunter.graphrag.property_graph import PropertyGraphBuilder
            graph_builder = PropertyGraphBuilder()
        except Exception as e:
            logger.warning(f"Impossibile inizializzare PropertyGraphBuilder: {e}")
            use_graph = False

    # --- Calcola budget per sezione ---
    n_total_sections = len(outline)
    section_budget = _compute_section_budget(n_total_sections=n_total_sections)
    budget_mode = get_writer_budget_mode()
    logger.info(
        f"Writer budget: mode={budget_mode}, "
        f"per_section={section_budget:,} chars, "
        f"n_total_sections={n_total_sections}"
    )

    # --- Scrivi sezioni che necessitano revisione ---
    all_sections: dict[str, dict] = {}
    errors: list[str] = []

    for section_key, draft in needs_revision.items():
        logger.info(f"Riscrittura sezione '{section_key}' (iter {draft.iteration + 1})")
        try:
            result = write_section(
                state,
                section_key,
                retriever=retriever,
                reranker=reranker,
                graph_builder=graph_builder,
                use_graph=use_graph,
                n_total_sections=n_total_sections,
            )
            if "sections" in result:
                all_sections.update(result["sections"])
            if "errors" in result:
                errors.extend(result["errors"])
        except Exception as e:
            err = f"Errore riscrittura '{section_key}': {e}"
            logger.error(err, exc_info=True)
            errors.append(err)

    # --- Scrivi nuove sezioni ---
    new_result = write_all_sections(
        {**state, "sections": {**state.get("sections", {}), **all_sections}},
        retriever=retriever,
        reranker=reranker,
        graph_builder=graph_builder,
        use_graph=use_graph,
    )

    if "sections" in new_result:
        all_sections.update(new_result["sections"])
    if "errors" in new_result and new_result["errors"]:
        errors.extend(new_result["errors"])

    total_sections = len(all_sections) + len(
        {k: v for k, v in existing.items()
         if k not in all_sections and v.status != SectionStatus.NEEDS_REVISION}
    )

    logger.info(
        f"Writer completato: {len(all_sections)} sezioni scritte/riscritte, "
        f"{total_sections} totali"
    )
    print_success(f"Writer Agent: {len(all_sections)} sezioni completate.")

    # Rilascia il lock Qdrant prima che il nodo successivo (Critic) lo richieda
    if retriever is not None:
        try:
            retriever.close()
        except Exception:
            pass

    return {
        "sections": all_sections,
        "current_phase": PipelinePhase.WRITING.value,
        "errors": errors if errors else [],
        "metadata": {
            **state.get("metadata", {}),
            **(new_result.get("metadata", {})),
        },
    }


# ============================================================================
# Convenience functions
# ============================================================================

def write_sections_from_outline(
    outline: list[SectionOutline],
    l2_summaries: list = None,
    *,
    force_model: str | None = None,
    skip_types: set[str] | None = None,
) -> dict[str, SectionDraft]:
    """Scrive tutte le sezioni partendo da un outline e L2 summaries.

    Funzione di convenienza che non richiede il SurveyState completo.

    Args:
        outline: Lista di SectionOutline.
        l2_summaries: Lista di L2Summary (opzionale).
        force_model: Override modello LLM.
        skip_types: Tipi di sezione da saltare.

    Returns:
        Dict di SectionDraft keyed by section_key.
    """
    from paperhunter.agents.state import create_initial_state

    state = create_initial_state(
        l2_summaries=l2_summaries or [],
        outline=outline,
    )

    result = write_all_sections(state, force_model=force_model, skip_types=skip_types)
    sections_raw = result.get("sections", {})

    return {
        k: SectionDraft.from_dict(v) if isinstance(v, dict) else v
        for k, v in sections_raw.items()
    }


def get_section_stats(sections: dict[str, SectionDraft]) -> dict[str, Any]:
    """Calcola statistiche sulle sezioni scritte.

    Args:
        sections: Dict di SectionDraft.

    Returns:
        Dict con statistiche aggregate.
    """
    if not sections:
        return {"total_sections": 0, "total_words": 0, "total_citations": 0}

    total_words = 0
    total_citations = 0
    all_keys: set[str] = set()
    statuses: dict[str, int] = {}
    per_section: list[dict] = []

    for key, draft in sections.items():
        wc = draft.word_count or _count_words(draft.text)
        nc = len(draft.citations)
        uk = draft.unique_citation_keys()

        total_words += wc
        total_citations += nc
        all_keys.update(uk)

        status = draft.status.value if isinstance(draft.status, SectionStatus) else draft.status
        statuses[status] = statuses.get(status, 0) + 1

        per_section.append({
            "key": key,
            "title": draft.title,
            "words": wc,
            "citations": nc,
            "unique_keys": len(uk),
            "status": status,
            "iteration": draft.iteration,
        })

    return {
        "total_sections": len(sections),
        "total_words": total_words,
        "total_citations": total_citations,
        "unique_citation_keys": len(all_keys),
        "statuses": statuses,
        "avg_words_per_section": round(total_words / len(sections)) if sections else 0,
        "per_section": per_section,
    }
