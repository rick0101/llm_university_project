"""
Writer Agent — Scrive le sezioni del survey paper con RAG e citazioni verificabili.

Agent 2 nella pipeline agentica LangGraph. Per ogni sezione dell'outline:
1. Formula 3-5 query RAG specifiche dalla descrizione della sezione
2. Esegue retrieval ibrido (HybridRetriever) con filtri appropriati
3. (Opzionale) Query al PropertyGraph per contesto relazionale
4. Genera il testo della sezione con Gemini Pro, con citazioni [bibtex_key]
5. Output: SectionDraft con text + CitationRecord list

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

import json
import re
import time
from datetime import datetime
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

logger = get_logger(__name__)


# ============================================================================
# Prompt templates
# ============================================================================

_QUERY_GENERATION_PROMPT = """\
You are an expert academic researcher writing a survey paper on LLM Security.

You need to write the section: **{title}**

Section description:
{description}

Key topics to cover:
{key_topics}

Related categories: {related_categories}

Generate exactly {n_queries} specific search queries to retrieve the most \
relevant research papers and information for writing this section. \
Each query should target a different aspect or subtopic of the section.

Output ONLY a JSON array of strings, no other text:
["query 1", "query 2", "query 3"]
"""

_WRITER_PROMPT_TEMPLATE = """\
You are an expert academic researcher writing a section for a comprehensive \
survey paper on **LLM Security** targeting a top venue (Nature/ACL-level).

## Section Details

**Title:** {title}
**Description:** {description}
**Key Topics:** {key_topics}
**Target Length:** approximately {target_words} words

## Retrieved Context

The following chunks were retrieved from the research paper corpus. \
Use ONLY information from these chunks to write the section. \
Each chunk includes metadata (paper key, category, year, section).

{retrieved_context}

## Writing Instructions

1. **Style**: Formal academic prose. Third person. Present tense for established knowledge, \
past tense for specific study results.

2. **Citations**: Every factual claim MUST include a citation in the format [bibtex_key]. \
Example: "Recent work has demonstrated that adversarial suffixes can transfer across \
LLM families [Zou_2023_GCG]."

3. **Temporal Context**: When discussing methods, explicitly state when they were \
proposed and how they relate to prior work. Use phrases like "Building on earlier \
work by [Author_YYYY]..." or "In YYYY, [Author] introduced..."

4. **Structure**: Use clear paragraphs with topic sentences. You may use Markdown \
subsection headers (###) if the section is long enough. Use bullet lists sparingly \
and only for enumerations.

5. **Completeness**: Cover ALL key topics listed above. If a topic is not covered \
by the retrieved context, state this gap explicitly rather than making claims \
without sources.

6. **No Fabrication**: Do NOT invent citations or facts. If a bibtex_key is [Unknown] \
or missing, do not cite it. Only cite keys that appear in the retrieved context.

{special_instructions}

## Output Format

Respond with a valid JSON object:

```json
{{
  "text": "The full section text in Markdown with [bibtex_key] citations...",
  "citations": [
    {{
      "bibtex_key": "Author_YYYY_Keyword",
      "claim": "The specific claim being cited",
      "chunk_id": "The chunk_id from which this information was derived"
    }}
  ],
  "notes": "Any notes about the writing process, difficulties, or gaps found"
}}
```

IMPORTANT:
- The "text" field should contain the COMPLETE section text.
- Every [bibtex_key] in the text MUST have a corresponding entry in "citations".
- The "chunk_id" in citations should reference the actual chunk from the context.
- Do NOT wrap the JSON output in markdown code fences.
- Target approximately {target_words} words in the text.
"""

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
        llm = get_llm("flash", max_tokens=2048, timeout_seconds=60)
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

def _retrieve_context_for_section(
    queries: list[str],
    outline: SectionOutline,
    *,
    retriever=None,
    reranker=None,
    graph_builder=None,
    top_k_per_query: int = 15,
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
    # Lazy init retriever
    if retriever is None:
        from paperhunter.indexing.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()

    # Build category filter
    category_filter = None
    if outline.related_categories:
        category_filter = outline.related_categories

    # Collect results from all queries
    all_results: list[dict] = []
    seen_ids: set[str] = set()

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
    if use_reranker and len(all_results) > top_k_final:
        try:
            if reranker is None:
                from paperhunter.indexing.reranker import GeminiReranker
                reranker = GeminiReranker(top_k=top_k_final)

            rerank_query = f"Write about: {outline.title}. Topics: {', '.join(outline.key_topics[:5])}"
            reranked = reranker.rerank(rerank_query, all_results, top_k=top_k_final)
            logger.info(f"Reranking: {len(all_results)} → {len(reranked)} chunk")
            all_results = reranked
        except Exception as e:
            logger.warning(f"Reranking fallito per '{outline.title}': {e}. Uso ordine originale.")
            # Sort by score descending e tronca
            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            all_results = all_results[:top_k_final]
    elif len(all_results) > top_k_final:
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        all_results = all_results[:top_k_final]

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


def _format_context_for_prompt(chunks: list[dict], max_chars: int = 60000) -> str:
    """Formatta i chunk recuperati per il prompt del Writer.

    Ogni chunk include metadata chiave (paper, categoria, anno, sezione)
    e il testo troncato se necessario.
    """
    if not chunks:
        return "(No relevant context retrieved. Write based on general knowledge.)"

    parts: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        bibtex_key = meta.get("bibtex_key", "Unknown")
        category = meta.get("category", "N/A")
        year = meta.get("year", "N/A")
        section = meta.get("section_name", "N/A")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        chunk_type = meta.get("chunk_type", "original")
        text = chunk.get("text", "")

        # Tronca testi molto lunghi
        if len(text) > 2000:
            text = text[:2000] + "..."

        header = (
            f"--- Chunk {i + 1} [chunk_id: {chunk_id}] ---\n"
            f"Paper: [{bibtex_key}] | Category: {category} | "
            f"Year: {year} | Section: {section} | Type: {chunk_type}"
        )
        entry = f"{header}\n{text}\n"

        if total_chars + len(entry) > max_chars:
            parts.append(f"\n[... {len(chunks) - i} more chunks truncated for prompt length ...]")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


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
    citations = _extract_citations_from_text(raw_response)
    return {
        "text": raw_response.strip(),
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
    pattern = r"\[([A-Z][a-z]+(?:_[A-Z]?[a-z]*)*_\d{4}_[A-Za-z]+(?:_[A-Za-z]+)*)\]"
    matches = re.findall(pattern, text)
    seen: set[str] = set()
    citations: list[dict] = []
    for key in matches:
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
        return {"errors": [err]}

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

    # --- Step 1: Genera query RAG ---
    critic_feedback = existing_draft.critic_feedback if existing_draft else ""
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
        err = f"Nessun chunk recuperato per sezione '{section_key}'. Scrittura tentata comunque."
        logger.warning(err)
        errors.append(err)

    # --- Step 3: Formatta contesto e prompt ---
    context_text = _format_context_for_prompt(chunks)
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

    # --- Step 4: Genera testo con LLM ---
    tier = force_model or settings.agents.writer_model
    logger.info(
        f"Generazione testo per '{section_key}' con {tier.upper()} "
        f"({len(chunks)} chunk di contesto, ~{target_outline.target_length_words} parole target)"
    )

    max_retries = settings.summarization.pro_max_retries
    base_delay = settings.summarization.pro_base_delay

    llm = get_llm(tier, max_tokens=16384)

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
        return {"errors": errors}

    elapsed = time.time() - t0
    logger.info(f"Risposta LLM per '{section_key}': {len(raw_response)} chars in {elapsed:.1f}s")

    # --- Step 5: Parse risposta ---
    parsed = _parse_writer_response(raw_response, section_key)
    if parsed is None:
        err = f"Impossibile parsare la risposta per '{section_key}'."
        logger.error(err)
        errors.append(err)
        return {"errors": errors}

    section_text = parsed["text"]
    word_count = _count_words(section_text)
    citation_records = [
        CitationRecord(
            bibtex_key=c["bibtex_key"],
            claim=c.get("claim", ""),
            chunk_id=c.get("chunk_id"),
            retrieval_score=c.get("retrieval_score", 0.0),
        )
        for c in parsed.get("citations", [])
    ]

    # Verifica che tutte le citazioni nel testo siano tracciate
    text_citations = set(re.findall(r"\[([^\]]+)\]", section_text))
    tracked_keys = {c.bibtex_key for c in citation_records}
    # Filtra solo potenziali bibtex keys (contengono _ e numeri)
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

    # --- Costruisci SectionDraft ---
    draft = SectionDraft(
        section_key=section_key,
        title=target_outline.title,
        text=section_text,
        citations=citation_records,
        word_count=word_count,
        status=SectionStatus.DRAFTED,
        iteration=current_iteration,
        rag_queries=queries,
        retrieved_chunk_ids=chunk_ids,
        writer_notes=parsed.get("notes", ""),
        drafted_at=datetime.now(),
    )

    # Controlla se la sezione è troppo corta e logga warning
    min_words = settings.agents.min_section_words
    max_words = settings.agents.max_section_words
    if word_count < min_words and target_outline.section_type not in ("abstract", "conclusion"):
        logger.warning(
            f"Sezione '{section_key}' troppo corta: {word_count} parole "
            f"(min: {min_words})"
        )
    if word_count > max_words:
        logger.warning(
            f"Sezione '{section_key}' troppo lunga: {word_count} parole "
            f"(max: {max_words})"
        )

    logger.info(
        f"Sezione '{section_key}' completata: {word_count} parole, "
        f"{len(citation_records)} citazioni, {elapsed:.1f}s"
    )

    return {
        "sections": {section_key: draft.to_dict()},
        "current_section_key": section_key,
        "errors": errors if errors else [],
    }


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

    Returns:
        Dict parziale con tutte le sezioni scritte e eventuali errori.
    """
    if skip_types is None:
        skip_types = {"abstract"}  # Abstract generated by Assembler

    outline = get_outline_from_state(state)
    existing_sections = get_sections_from_state(state)
    errors: list[str] = []
    all_sections: dict[str, dict] = {}

    # Filtra sezioni da scrivere
    sections_to_write = [
        o for o in outline
        if o.section_type not in skip_types
        and o.key not in existing_sections
    ]

    if not sections_to_write:
        logger.info("Nessuna nuova sezione da scrivere.")
        return {"current_phase": PipelinePhase.WRITING.value}

    logger.info(f"Sezioni da scrivere: {len(sections_to_write)}/{len(outline)}")

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
            if "errors" in result and result["errors"]:
                errors.extend(result["errors"])

        except Exception as e:
            err = f"Errore critico scrivendo '{section_outline.key}': {e}"
            logger.error(err, exc_info=True)
            print_error(err)
            errors.append(err)
            failed_count += 1

    elapsed = time.time() - t0
    logger.info(
        f"\nScrittura completata: {written_count} sezioni scritte, "
        f"{failed_count} fallite, {elapsed:.1f}s totali"
    )

    return {
        "sections": all_sections,
        "current_phase": PipelinePhase.WRITING.value,
        "errors": errors if errors else [],
        "metadata": {
            **state.get("metadata", {}),
            "writer_sections_written": written_count,
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

    # Configura retriever e reranker (una volta per tutto il batch)
    retriever = None
    reranker = None
    graph_builder = None

    try:
        from paperhunter.indexing.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
    except Exception as e:
        logger.warning(f"Impossibile inizializzare HybridRetriever: {e}")

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
