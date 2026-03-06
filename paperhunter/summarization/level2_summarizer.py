"""
Level 2 Summarizer — Meta-summary per categoria.

Genera un meta-summary L2 di 4-6k token per ogni categoria,
sintetizzando tutti i summary L1 dei paper appartenenti a
quella categoria.

Il summary L2 include:
- ``temporal_trends``: evoluzione temporale (es. 2019→2026)
- ``taxonomy``: tassonomia delle sotto-aree
- ``gaps_filled``: gap colmati dalla letteratura
- ``open_gaps``: gap ancora aperti
- ``methodological_convergences``: convergenze/divergenze metodologiche
- ``testing_overview``: panoramica delle testing methodologies più usate
- ``synthesis``: sintesi complessiva della categoria

Il summary viene anche salvato come chunk nel QdrantStore
(collection ``l2_summaries``) per il retrieval successivo.

Uso::

    from paperhunter.summarization.level2_summarizer import (
        generate_l2_summary,
        generate_l2_for_all_categories,
        load_l1_summaries_for_category,
    )
    from paperhunter.models import L1Summary

    # Da lista di L1Summary
    l2 = generate_l2_summary("Prompt Injection", l1_summaries)

    # Carica L1 dal QdrantStore e genera L2
    l2 = generate_l2_summary("Prompt Injection")

    # Genera L2 per tutte le categorie con almeno 1 L1
    l2s = generate_l2_for_all_categories(save_to_store=True)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.models import ChunkType, L1Summary, L1_5Summary, L2Summary, PaperChunk
from paperhunter.utils.llm_factory import (
    get_flash_llm,
    get_pro_llm,
    get_pro_tracker,
    llm_call_with_retry,
)
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_success,
    print_table,
    print_warning,
    progress_bar,
)
from paperhunter.utils.rate_limiter import get_gemini_limiter

logger = get_logger(__name__)


# ============================================================================
# Prompt template L2
# ============================================================================

_L2_PROMPT_TEMPLATE = """\
You are an expert academic researcher writing a comprehensive survey on LLM security. \
You are synthesizing research for the **{category}** sub-field.

Below are structured summaries (Level 1) of {paper_count} individual papers in this category, \
ordered chronologically by publication year.

**NOTE ON DETAIL LEVELS:** Summaries are formatted at different detail levels \
based on recency-weighted importance:
- Papers from the last {recency_full_years} years appear with **FULL** detail (all fields).
- Papers from {recency_full_years}-{recency_reduced_years} years ago appear with **REDUCED** detail \
(~70%, marked [REDUCED]) — limitations removed, some fields truncated.
- Papers older than {recency_reduced_years} years appear **COMPRESSED** to key contributions \
and testing methodology only (marked [COMPRESSED]).

Give analytical priority to recent papers with full details, as they represent the \
most current state of the field. However, still cover older compressed papers to \
provide historical context and trace the evolution of ideas.

{l1_texts}

---

## Instructions

Based ONLY on the summaries above, produce a **meta-summary** of this research category. \
This will be used as a building block for a Nature/ACL-style survey paper.

The meta-summary should be 4000-6000 tokens long, comprehensive, and well-structured. \
Use the actual paper IDs (bibtex_key) as citations in square brackets, e.g., [Zou_2023_GCG].

Respond with a valid JSON object (no markdown fences, no comments):

{{
  "temporal_trends": "A detailed narrative (800-1500 tokens) describing the chronological evolution of this research area. Identify key phases, turning points, and how the field has progressed from earliest to latest papers. Use citations and years. Example: 'Early work in 2020-2021 focused on X [Author_2020_Key], while 2023 saw a shift toward Y [Author_2023_Key]...'",

  "taxonomy": "A structured taxonomy (500-1000 tokens) of sub-areas within this category. Group the papers into logical clusters (e.g., by method type, target model, threat scenario). For each cluster, list the relevant papers by bibtex_key.",

  "gaps_filled": "What gaps in the literature have been filled by the papers in this collection? (400-800 tokens) Reference specific papers that addressed previously open problems.",

  "open_gaps": "What significant gaps or open problems remain? (400-800 tokens) Identify questions not yet answered, limitations shared across multiple papers, and promising but unexplored directions.",

  "methodological_convergences": "Convergences and divergences in methodology across papers. (400-800 tokens) What approaches are most common? What are the dominant paradigms? Where do authors disagree on methodology?",

  "testing_overview": "A comprehensive overview (600-1000 tokens) of the testing methodologies used across papers in this category. Aggregate: (1) most commonly used benchmarks/datasets, (2) dominant evaluation metrics, (3) standard models tested on, (4) any shared open-source tools or frameworks. Highlight which testing approaches are most rigorous and which are lacking.",

  "synthesis": "A cohesive synthesis (600-1000 tokens) integrating all the above. This should read as a polished paragraph(s) suitable for a survey paper introduction to this category. Include temporal perspective and cite key papers."
}}

IMPORTANT:
- Be exhaustive: cover ALL {paper_count} papers, not just a subset.
- Use square bracket citations with bibtex_key: [Author_YYYY_Key].
- Do NOT hallucinate facts not present in the L1 summaries.
- If information is insufficient for a field, explicitly state so.
- The response must be valid JSON with all 7 fields.
- Aim for 4000-6000 total tokens across all fields combined.
- Papers marked [COMPRESSED] have less detail — do not penalize them, but give \
more analytical depth to recent papers with full details when building the narrative.
"""


# ============================================================================
# Prompt template L2 — from L1.5 cluster summaries (RAPTOR path)
# ============================================================================

_L2_FROM_L1_5_PROMPT_TEMPLATE = """\
You are an expert academic researcher writing a comprehensive survey on LLM security. \
You are synthesizing research for the **{category}** sub-field.

Below are **{cluster_count} thematic cluster summaries** (Level 1.5), each synthesizing \
a group of ~{avg_papers_per_cluster} related papers. These clusters were generated by an \
intermediate RAPTOR layer that semantically grouped {total_paper_count} individual paper \
summaries into coherent thematic clusters.

**IMPORTANT:** Each cluster summary already aggregates multiple papers. \
Citations in square brackets [bibtex_key] refer to individual papers within clusters. \
Look for cross-cluster patterns, convergences, and divergences when building the narrative.

{l1_5_texts}

---

## Instructions

Based ONLY on the cluster summaries above, produce a **meta-summary** of this research category. \
This will be used as a building block for a Nature/ACL-style survey paper.

The meta-summary should be 4000-6000 tokens long, comprehensive, and well-structured. \
Use the actual paper IDs (bibtex_key) as citations in square brackets, e.g., [Zou_2023_GCG].

Respond with a valid JSON object (no markdown fences, no comments):

{{
  "temporal_trends": "A detailed narrative (800-1500 tokens) describing the chronological evolution of this research area. \
Identify key phases, turning points, and how the field has progressed. Use citations and years. \
Since input comes from cluster summaries, synthesize temporal information across clusters.",

  "taxonomy": "A structured taxonomy (500-1000 tokens) of sub-areas within this category. \
The clusters already represent thematic groups — use them as a starting point for the taxonomy, \
but feel free to reorganize or split/merge if the research landscape warrants it.",

  "gaps_filled": "What gaps in the literature have been filled by the papers in this collection? \
(400-800 tokens) Reference specific papers that addressed previously open problems.",

  "open_gaps": "What significant gaps or open problems remain? (400-800 tokens) \
Identify questions not yet answered, limitations shared across clusters, and promising but unexplored directions.",

  "methodological_convergences": "Convergences and divergences in methodology across clusters. \
(400-800 tokens) What approaches are most common? What are the dominant paradigms? \
Where do different clusters' papers disagree on methodology?",

  "testing_overview": "A comprehensive overview (600-1000 tokens) of the testing methodologies \
used across all clusters. Aggregate: (1) most commonly used benchmarks/datasets, \
(2) dominant evaluation metrics, (3) standard models tested on, (4) any shared open-source tools \
or frameworks. Highlight which testing approaches are most rigorous and which are lacking.",

  "synthesis": "A cohesive synthesis (600-1000 tokens) integrating all the above. \
This should read as a polished paragraph(s) suitable for a survey paper introduction to this category. \
Include temporal perspective and cite key papers from across all clusters."
}}

IMPORTANT:
- Be exhaustive: cover ALL {cluster_count} clusters and reference papers from each.
- Use square bracket citations with bibtex_key: [Author_YYYY_Key].
- Do NOT hallucinate facts not present in the cluster summaries.
- If information is insufficient for a field, explicitly state so.
- The response must be valid JSON with all 7 fields.
- Aim for 4000-6000 total tokens across all fields combined.
- Look for **cross-cluster patterns**: themes, methods, or findings that span multiple clusters \
represent particularly important trends in the field.
"""


# ============================================================================
# Core function
# ============================================================================

def generate_l2_summary(
    category: str,
    l1_summaries: list[L1Summary] | None = None,
    *,
    force_model: str | None = None,
    save_to_store: bool = False,
    store: Any = None,
    version: int | None = None,
) -> L2Summary:
    """Genera un meta-summary L2 per una categoria.

    **RAPTOR path (preferito):** Se esistono summary L1.5 per la
    categoria nella collection Qdrant ``l1_5_summaries``, vengono usati
    quelli come input (~50 cluster summary). Questo permette di gestire
    1.000+ paper senza sfondare il budget token.

    **Fallback path:** Se non esistono L1.5, usa la logica legacy di
    compressione recency-tier sugli L1 (backward compatibility).

    Se ``l1_summaries`` non è fornito e non ci sono L1.5 disponibili,
    tenta di caricare gli L1 dal QdrantStore (collection ``l1_summaries``).

    Args:
        category: Nome della categoria (es. "Prompt Injection").
        l1_summaries: Lista di L1Summary della categoria.
            Se None, vengono caricati dal vector store.
            Se vengono trovati L1.5, questo parametro viene ignorato.
        force_model: Override tier modello (``'flash'`` o ``'pro'``).
            Default: da ``config.yaml`` → ``summarization.use_pro_for_l2``.
        save_to_store: Se True, salva il summary L2 come chunk nella
            collection ``l2_summaries`` del QdrantStore.
        store: Istanza ``QdrantStore`` opzionale. Se serve caricare L1
            o salvare L2 e ``store`` è None, ne crea una nuova.
        version: Versione del L2. Se None, viene calcolata
            automaticamente (incrementale).

    Returns:
        Istanza ``L2Summary`` con tutti i campi.

    Raises:
        ValueError: Se non ci sono né L1.5 né L1 summaries per la
            categoria.
        RuntimeError: Se la chiamata LLM fallisce dopo tutti i retry.
    """
    # --- Inizializzazione store ---
    if store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    # ================================================================
    # RAPTOR path: prova a caricare L1.5 summary (cluster) dalla Qdrant
    # ================================================================
    from paperhunter.summarization.level1_5_summarizer import (
        load_l1_5_summaries_for_category,
    )

    l1_5_summaries = load_l1_5_summaries_for_category(category, store=store)
    use_l1_5_path = len(l1_5_summaries) > 0

    if use_l1_5_path:
        return _generate_l2_from_l1_5(
            category=category,
            l1_5_summaries=l1_5_summaries,
            force_model=force_model,
            save_to_store=save_to_store,
            store=store,
            version=version,
        )

    # ================================================================
    # Fallback path: usa i singoli L1 summary (logica legacy)
    # ================================================================
    logger.info(
        f"Nessun L1.5 trovato per '{category}', "
        f"fallback alla logica legacy (L1 diretti con recency tiers)"
    )

    # --- Caricamento L1 se non forniti ---
    if l1_summaries is None:
        l1_summaries = load_l1_summaries_for_category(category, store=store)

    if not l1_summaries:
        raise ValueError(
            f"Nessun L1 summary (né L1.5) trovato per la categoria '{category}'. "
            f"Esegui prima la generazione L1."
        )

    # --- Ordinamento cronologico ---
    l1_summaries = sorted(
        l1_summaries,
        key=lambda s: (s.year or 0, s.paper_id),
    )

    # --- Deduplica per paper_id (tieni il più recente) ---
    seen_ids: dict[str, L1Summary] = {}
    for s in l1_summaries:
        seen_ids[s.paper_id] = s
    l1_summaries = sorted(
        seen_ids.values(),
        key=lambda s: (s.year or 0, s.paper_id),
    )

    paper_count = len(l1_summaries)
    paper_ids = [s.paper_id for s in l1_summaries]

    logger.info(
        f"Generazione L2 summary per '{category}' "
        f"({paper_count} paper L1, path legacy)"
    )

    # --- Selezione modello ---
    pro_tracker = get_pro_tracker()
    if force_model:
        tier = force_model
    else:
        tier = "pro" if settings.summarization.use_pro_for_l2 else "flash"

    llm = get_pro_llm() if tier == "pro" else get_flash_llm()

    # --- Preparazione testo L1 con recency weighting ---
    l1_texts = _format_l1_summaries_for_prompt(l1_summaries)

    # --- Computa statistiche recency tier ---
    recency_stats = _compute_recency_tier_stats(l1_summaries)
    import paperhunter.config as _cfg
    recency_enabled = _cfg.settings.recency_priority

    if recency_enabled:
        logger.info(
            f"Recency tiers per '{category}': "
            f"full={recency_stats['full']} ({recency_stats['full_pct']:.0f}%), "
            f"reduced={recency_stats['reduced']} ({recency_stats['reduced_pct']:.0f}%), "
            f"compressed={recency_stats['compressed']} ({recency_stats['compressed_pct']:.0f}%)"
        )
    else:
        logger.info(
            f"Recency priority DISABILITATA per '{category}': "
            f"tutti i {paper_count} paper trattati a dettaglio completo"
        )

    # --- Leggi soglie recency per il prompt ---
    tiers_cfg = settings.summarization.l2_recency_tiers

    # --- Costruzione prompt ---
    prompt = _L2_PROMPT_TEMPLATE.format(
        category=category,
        paper_count=paper_count,
        l1_texts=l1_texts,
        recency_full_years=tiers_cfg.full,
        recency_reduced_years=tiers_cfg.reduced,
    )

    logger.info(
        f"Prompt L2 per '{category}': {len(prompt)} chars, "
        f"tier={tier}, {paper_count} paper"
    )

    # --- Chiamata LLM con rate limiting ---
    limiter = get_gemini_limiter()
    limiter.wait()

    # Retry strategy (stessa del L1)
    sum_cfg = settings.summarization
    pro_retries = sum_cfg.pro_max_retries
    pro_base_delay = sum_cfg.pro_base_delay

    start_time = time.time()
    try:
        raw_response = llm_call_with_retry(
            llm, prompt,
            max_retries=pro_retries if tier == "pro" else 3,
            base_delay=pro_base_delay if tier == "pro" else 3.0,
            max_delay=120.0,
        )
        if tier == "pro":
            pro_tracker.record_success()
    except Exception as e:
        error_str = str(e).lower()
        is_transient = any(kw in error_str for kw in (
            "503", "504", "unavailable", "overloaded", "deadline",
        ))

        if tier == "pro" and is_transient:
            pro_tracker.record_failure()

            if sum_cfg.allow_flash_fallback:
                logger.warning(
                    f"Pro esaurito per L2 '{category}', "
                    f"fallback a Flash (allow_flash_fallback=true)"
                )
                tier = "flash"
                llm = get_flash_llm()
                limiter.wait()
                raw_response = llm_call_with_retry(
                    llm, prompt, max_retries=3, base_delay=3.0, max_delay=60.0,
                )
            else:
                logger.error(
                    f"Pro non disponibile per L2 '{category}' dopo "
                    f"{pro_retries + 1} tentativi. Flash fallback disabilitato."
                )
                raise
        else:
            raise

    elapsed = time.time() - start_time
    logger.debug(f"LLM response L2 ricevuta in {elapsed:.1f}s ({len(raw_response)} chars)")

    # --- Parsing JSON response ---
    parsed = _parse_l2_json_response(raw_response, category)

    # --- Calcolo versione ---
    if version is None:
        version = _get_next_l2_version(category, store)

    # --- Costruzione L2Summary ---
    summary = L2Summary(
        category=category,
        paper_ids=paper_ids,
        paper_count=paper_count,
        version=version,
        temporal_trends=parsed.get("temporal_trends", ""),
        taxonomy=parsed.get("taxonomy", ""),
        gaps_filled=parsed.get("gaps_filled", ""),
        open_gaps=parsed.get("open_gaps", ""),
        methodological_convergences=parsed.get("methodological_convergences", ""),
        testing_overview=parsed.get("testing_overview", ""),
        synthesis=parsed.get("synthesis", ""),
        invalidated=False,
        generated_at=datetime.now(),
        model_used=tier,
    )

    # --- Validazione qualità ---
    _validate_l2_summary(summary)

    # --- Salvataggio opzionale nel vector store ---
    if save_to_store:
        _save_l2_to_store(summary, store)

    logger.info(
        f"L2 summary generato per '{category}' v{version} "
        f"({elapsed:.1f}s, model={tier}, {paper_count} paper)"
    )

    # --- Salva metadata con recency stats ---
    _save_l2_metadata_to_disk(summary, recency_tier_stats=recency_stats)

    return summary


# ============================================================================
# RAPTOR path: genera L2 da cluster L1.5
# ============================================================================

def _generate_l2_from_l1_5(
    category: str,
    l1_5_summaries: list[L1_5Summary],
    *,
    force_model: str | None = None,
    save_to_store: bool = False,
    store: Any = None,
    version: int | None = None,
) -> L2Summary:
    """Genera un meta-summary L2 a partire da cluster L1.5 (RAPTOR).

    Questo è il path preferito quando esistono summary L1.5 per la
    categoria. Invece di comprimere 1.000 singoli L1, l'L2 viene
    generato dai ~50 cluster summary L1.5 che sono già tematicamente
    aggregati.

    Args:
        category: Nome della categoria.
        l1_5_summaries: Lista di ``L1_5Summary`` caricati dal Qdrant.
        force_model: Override tier modello.
        save_to_store: Se True, salva l'L2 nel QdrantStore.
        store: Istanza ``QdrantStore``.
        version: Versione dell'L2. Se None, auto-incrementale.

    Returns:
        Istanza ``L2Summary`` con tutti i campi.

    Raises:
        RuntimeError: Se la chiamata LLM fallisce dopo tutti i retry.
    """
    from paperhunter.utils.token_counter import compute_l2_budget

    cluster_count = len(l1_5_summaries)

    # Raccogli tutti i paper_ids dai cluster
    all_paper_ids: list[str] = []
    total_paper_count = 0
    for s in l1_5_summaries:
        all_paper_ids.extend(s.paper_ids)
        total_paper_count += s.paper_count

    # Deduplica paper_ids (un paper potrebbe apparire in cluster diversi)
    unique_paper_ids = list(dict.fromkeys(all_paper_ids))
    total_paper_count = max(total_paper_count, len(unique_paper_ids))

    avg_papers_per_cluster = (
        total_paper_count / cluster_count if cluster_count > 0 else 0
    )

    logger.info(
        f"Generazione L2 summary per '{category}' via RAPTOR path: "
        f"{cluster_count} cluster L1.5, ~{total_paper_count} paper totali "
        f"(media {avg_papers_per_cluster:.1f} paper/cluster)"
    )

    # --- Formattazione L1.5 per il prompt ---
    l1_5_texts = _format_l1_5_summaries_for_prompt(l1_5_summaries)

    # --- Budget check ---
    budget_info = compute_l2_budget()
    available_chars = budget_info["available_chars"]
    _TEMPLATE_OVERHEAD_CHARS = 4000  # Overhead prompt L1.5 (leggermente più alto)
    usable_chars = available_chars - _TEMPLATE_OVERHEAD_CHARS
    l1_5_chars = len(l1_5_texts)

    budget_pct = l1_5_chars * 100 // usable_chars if usable_chars > 0 else 0
    logger.info(
        f"L2 budget (RAPTOR path): L1.5 text={l1_5_chars:,} chars, "
        f"budget={usable_chars:,} chars ({budget_pct}% utilizzato)"
    )

    if l1_5_chars > usable_chars:
        logger.warning(
            f"L1.5 text ({l1_5_chars:,} chars) supera il budget "
            f"({usable_chars:,} chars) per '{category}'. "
            f"Applicazione truncation proporzionale."
        )
        l1_5_texts = _truncate_l1_5_for_budget(
            l1_5_summaries, usable_chars,
        )
        logger.info(
            f"L1.5 troncato a {len(l1_5_texts):,} chars per '{category}'"
        )

    # --- Selezione modello ---
    pro_tracker = get_pro_tracker()
    if force_model:
        tier = force_model
    else:
        tier = "pro" if settings.summarization.use_pro_for_l2 else "flash"

    llm = get_pro_llm() if tier == "pro" else get_flash_llm()

    # --- Costruzione prompt ---
    prompt = _L2_FROM_L1_5_PROMPT_TEMPLATE.format(
        category=category,
        cluster_count=cluster_count,
        avg_papers_per_cluster=int(round(avg_papers_per_cluster)),
        total_paper_count=total_paper_count,
        l1_5_texts=l1_5_texts,
    )

    logger.info(
        f"Prompt L2 (RAPTOR) per '{category}': {len(prompt):,} chars, "
        f"tier={tier}, {cluster_count} cluster, ~{total_paper_count} paper"
    )

    # --- Chiamata LLM con rate limiting ---
    limiter = get_gemini_limiter()
    limiter.wait()

    sum_cfg = settings.summarization
    pro_retries = sum_cfg.pro_max_retries
    pro_base_delay = sum_cfg.pro_base_delay

    start_time = time.time()
    try:
        raw_response = llm_call_with_retry(
            llm, prompt,
            max_retries=pro_retries if tier == "pro" else 3,
            base_delay=pro_base_delay if tier == "pro" else 3.0,
            max_delay=120.0,
        )
        if tier == "pro":
            pro_tracker.record_success()
    except Exception as e:
        error_str = str(e).lower()
        is_transient = any(kw in error_str for kw in (
            "503", "504", "unavailable", "overloaded", "deadline",
        ))

        if tier == "pro" and is_transient:
            pro_tracker.record_failure()

            if sum_cfg.allow_flash_fallback:
                logger.warning(
                    f"Pro esaurito per L2 (RAPTOR) '{category}', "
                    f"fallback a Flash"
                )
                tier = "flash"
                llm = get_flash_llm()
                limiter.wait()
                raw_response = llm_call_with_retry(
                    llm, prompt, max_retries=3, base_delay=3.0, max_delay=60.0,
                )
            else:
                logger.error(
                    f"Pro non disponibile per L2 (RAPTOR) '{category}' "
                    f"dopo {pro_retries + 1} tentativi."
                )
                raise
        else:
            raise

    elapsed = time.time() - start_time
    logger.debug(
        f"LLM response L2 (RAPTOR) ricevuta in {elapsed:.1f}s "
        f"({len(raw_response)} chars)"
    )

    # --- Parsing JSON response ---
    parsed = _parse_l2_json_response(raw_response, category)

    # --- Calcolo versione ---
    if version is None:
        version = _get_next_l2_version(category, store)

    # --- Costruzione L2Summary ---
    summary = L2Summary(
        category=category,
        paper_ids=unique_paper_ids,
        paper_count=total_paper_count,
        version=version,
        temporal_trends=parsed.get("temporal_trends", ""),
        taxonomy=parsed.get("taxonomy", ""),
        gaps_filled=parsed.get("gaps_filled", ""),
        open_gaps=parsed.get("open_gaps", ""),
        methodological_convergences=parsed.get("methodological_convergences", ""),
        testing_overview=parsed.get("testing_overview", ""),
        synthesis=parsed.get("synthesis", ""),
        invalidated=False,
        generated_at=datetime.now(),
        model_used=tier,
    )

    # --- Validazione qualità ---
    _validate_l2_summary(summary)

    # --- Salvataggio opzionale ---
    if save_to_store:
        _save_l2_to_store(summary, store)

    logger.info(
        f"L2 summary (RAPTOR) generato per '{category}' v{version} "
        f"({elapsed:.1f}s, model={tier}, {cluster_count} cluster, "
        f"~{total_paper_count} paper)"
    )

    # --- Salva metadata ---
    raptor_stats = {
        "path": "raptor_l1_5",
        "cluster_count": cluster_count,
        "total_paper_count": total_paper_count,
        "avg_papers_per_cluster": round(avg_papers_per_cluster, 1),
        "l1_5_chars": l1_5_chars,
        "budget_pct": budget_pct,
    }
    _save_l2_metadata_to_disk(summary, recency_tier_stats=raptor_stats)

    return summary


# ============================================================================
# L1.5 formatting for L2 prompt
# ============================================================================

def _format_l1_5_summaries_for_prompt(
    l1_5_summaries: list[L1_5Summary],
) -> str:
    """Formatta i cluster summary L1.5 per il prompt dell'L2.

    Ogni cluster viene presentato come blocco strutturato con:
    - ID cluster, numero di paper, temporal span
    - Tema comune
    - Metodologie condivise
    - Risultati chiave
    - Panoramica testing
    - Lista dei paper inclusi

    I cluster sono ordinati per temporal_span (cluster con paper più
    recenti prima), in modo che l'LLM dia priorità ai dati più attuali.

    Args:
        l1_5_summaries: Lista di ``L1_5Summary`` da formattare.

    Returns:
        Testo formattato per il prompt L2.
    """
    if not l1_5_summaries:
        return ""

    # Ordina per temporal_span (anno massimo più recente prima)
    import re as _re

    def _sort_key(s: L1_5Summary) -> tuple[int, int]:
        """(max_year DESC, cluster_id ASC)."""
        years = _re.findall(r"(\d{4})", s.temporal_span or "")
        max_year = max(int(y) for y in years) if years else 0
        return (-max_year, s.cluster_id)

    sorted_summaries = sorted(l1_5_summaries, key=_sort_key)

    total_clusters = len(sorted_summaries)
    parts: list[str] = []

    for i, s in enumerate(sorted_summaries, 1):
        paper_ids_str = ", ".join(
            f"[{pid}]" for pid in s.paper_ids[:15]
        )
        if len(s.paper_ids) > 15:
            paper_ids_str += f", ... (+{len(s.paper_ids) - 15} more)"

        block = (
            f"=== Cluster {i}/{total_clusters} "
            f"(ID: {s.cluster_id}, {s.paper_count} papers, "
            f"Temporal Span: {s.temporal_span or 'N/A'}) ===\n\n"
            f"**Common Theme:**\n{s.common_theme or 'N/A'}\n\n"
            f"**Shared Methodologies:**\n{s.shared_methodologies or 'N/A'}\n\n"
            f"**Key Findings:**\n{s.key_findings or 'N/A'}\n\n"
            f"**Testing Overview:**\n{s.testing_overview or 'N/A'}\n\n"
            f"**Papers in this cluster:** {paper_ids_str}\n"
        )
        parts.append(block)

    return "\n---\n\n".join(parts)


def _truncate_l1_5_for_budget(
    l1_5_summaries: list[L1_5Summary],
    max_chars: int,
) -> str:
    """Tronca i testi L1.5 per rientrare nel budget.

    Strategia: riduce progressivamente i campi di ogni cluster
    partendo dai meno recenti, fino a rientrare nel budget.

    1. Prima: tronca ``shared_methodologies`` e ``testing_overview``
       dei cluster più vecchi a 200 chars
    2. Poi: tronca tutti i campi secondari a 150 chars
    3. Ultima risorsa: tronca proporzionalmente tutto

    Args:
        l1_5_summaries: Lista di L1_5Summary.
        max_chars: Budget massimo in caratteri.

    Returns:
        Testo formattato troncato.
    """
    import re as _re

    def _max_year(s: L1_5Summary) -> int:
        years = _re.findall(r"(\d{4})", s.temporal_span or "")
        return max(int(y) for y in years) if years else 0

    # Ordina per anno recente (da preservare) → vecchio (da troncare)
    by_recency = sorted(l1_5_summaries, key=_max_year, reverse=True)

    # Fase 1: tronca campi secondari dei cluster più vecchi
    truncated = []
    for s in by_recency:
        truncated.append(L1_5Summary(
            cluster_id=s.cluster_id,
            category=s.category,
            paper_ids=s.paper_ids,
            paper_count=s.paper_count,
            common_theme=s.common_theme,
            shared_methodologies=_truncate_text(s.shared_methodologies, 200),
            key_findings=s.key_findings,
            testing_overview=_truncate_text(s.testing_overview, 200),
            temporal_span=s.temporal_span,
            generated_at=s.generated_at,
            model_used=s.model_used,
        ))

    result = _format_l1_5_summaries_for_prompt(truncated)
    if len(result) <= max_chars:
        return result

    # Fase 2: tronca tutto a 150 chars
    truncated_2 = []
    for s in by_recency:
        truncated_2.append(L1_5Summary(
            cluster_id=s.cluster_id,
            category=s.category,
            paper_ids=s.paper_ids[:10],
            paper_count=s.paper_count,
            common_theme=_truncate_text(s.common_theme, 150),
            shared_methodologies=_truncate_text(s.shared_methodologies, 150),
            key_findings=_truncate_text(s.key_findings, 150),
            testing_overview=_truncate_text(s.testing_overview, 150),
            temporal_span=s.temporal_span,
            generated_at=s.generated_at,
            model_used=s.model_used,
        ))

    result = _format_l1_5_summaries_for_prompt(truncated_2)
    if len(result) <= max_chars:
        return result

    # Fase 3: truncation proporzionale hard
    ratio = max_chars / max(len(result), 1)
    if ratio < 1.0:
        result = result[:max_chars - 50] + "\n\n[...truncated for budget...]"

    return result


# ============================================================================
# Batch: genera L2 per tutte le categorie
# ============================================================================

def generate_l2_for_all_categories(
    *,
    force_model: str | None = None,
    save_to_store: bool = False,
    store: Any = None,
    skip_existing: bool = False,
    categories: list[str] | None = None,
) -> list[L2Summary]:
    """Genera summary L2 per tutte le categorie con almeno 1 L1.

    Args:
        force_model: Override tier modello.
        save_to_store: Salva ogni L2 nel QdrantStore.
        store: Istanza ``QdrantStore``.
        skip_existing: Se True, salta categorie che hanno già un L2
            non-invalidato nel vector store.
        categories: Lista di categorie da processare. Se None,
            scopre automaticamente le categorie dal vector store.

    Returns:
        Lista di ``L2Summary`` generati con successo.
    """
    # --- Inizializzazione store ---
    if store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    # --- Scoperta categorie ---
    if categories is None:
        categories = _discover_categories_from_store(store)

    if not categories:
        logger.warning("Nessuna categoria trovata nel vector store per L2.")
        return []

    logger.info(f"Generazione L2 per {len(categories)} categorie: {categories}")

    summaries: list[L2Summary] = []
    errors: dict[str, str] = {}
    skipped = 0

    for category in progress_bar(categories, description="L2 Summaries"):
        # Skip se già presente e non invalidato
        if skip_existing and _l2_exists_and_valid(category, store):
            logger.debug(f"L2 già esistente per '{category}', skip")
            skipped += 1
            continue

        try:
            l2 = generate_l2_summary(
                category,
                force_model=force_model,
                save_to_store=save_to_store,
                store=store,
            )
            summaries.append(l2)
        except ValueError as e:
            # Nessun L1 per questa categoria
            logger.warning(f"Skip L2 per '{category}': {e}")
            skipped += 1
        except Exception as e:
            logger.error(f"Errore L2 per '{category}': {e}")
            errors[category] = str(e)

    # --- Report ---
    _print_batch_report(
        total=len(categories),
        generated=len(summaries),
        skipped=skipped,
        errors=len(errors),
        error_details=errors,
    )

    return summaries


# ============================================================================
# Caricamento L1 dal QdrantStore
# ============================================================================

def load_l1_summaries_for_category(
    category: str,
    *,
    store: Any = None,
) -> list[L1Summary]:
    """Carica tutti i summary L1 di una categoria dal QdrantStore.

    Esegue uno scroll sulla collection ``l1_summaries`` filtrando
    per ``category``, poi ricostruisce gli oggetti ``L1Summary``
    dal testo e dai metadata.

    Args:
        category: Nome della categoria.
        store: Istanza ``QdrantStore``. Se None, ne crea una nuova.

    Returns:
        Lista di ``L1Summary``.
    """
    if store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    collection = settings.qdrant.collection_l1_summaries

    # Scroll con filtro per categoria
    from qdrant_client import models

    all_points: list[Any] = []
    offset = None

    while True:
        try:
            result, next_offset = store.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category",
                            match=models.MatchValue(value=category),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.error(
                f"Errore caricamento L1 per categoria '{category}': {e}"
            )
            break

        all_points.extend(result)

        if next_offset is None:
            break
        offset = next_offset

    # Ricostruisci L1Summary dal payload
    summaries: list[L1Summary] = []
    for point in all_points:
        payload = point.payload or {}
        l1 = _reconstruct_l1_from_payload(payload)
        if l1 is not None:
            summaries.append(l1)

    logger.info(
        f"Caricati {len(summaries)} L1 per categoria '{category}' "
        f"da {collection}"
    )
    return summaries


def _reconstruct_l1_from_payload(payload: dict) -> L1Summary | None:
    """Ricostruisce un L1Summary dal payload di un punto Qdrant.

    Il payload contiene il testo dell'L1 summary in formato narrativo
    (generato da ``_l1_summary_to_text``). Estraiamo i metadata e
    il testo grezzo per ricostruire un oggetto L1Summary.

    Args:
        payload: Payload del punto Qdrant.

    Returns:
        L1Summary ricostruito, o None se il payload è insufficiente.
    """
    text = payload.get("text", "")
    if not text:
        return None

    paper_id = payload.get("paper_id", "") or payload.get("bibtex_key", "")
    if not paper_id:
        return None

    # Parsing del testo narrativo
    # Il testo è stato generato da _l1_summary_to_text con sezioni note
    sections = _parse_l1_text_sections(text)

    return L1Summary(
        paper_id=paper_id,
        bibtex_key=payload.get("bibtex_key"),
        category=payload.get("category"),
        year=payload.get("year"),
        key_contributions=sections.get("key_contributions", []),
        methodology=sections.get("methodology", ""),
        testing_methodology=sections.get("testing_methodology", ""),
        temporal_context=sections.get("temporal_context", ""),
        results_summary=sections.get("results_summary", ""),
        limitations=sections.get("limitations", ""),
        citation_count=payload.get("citation_count"),
    )


def _parse_l1_text_sections(text: str) -> dict[str, Any]:
    """Estrae i campi dal testo narrativo di un L1 summary.

    Il testo è generato da ``_l1_summary_to_text`` con il formato:

        Paper: ...
        Key Contributions:
          1. ...
        Methodology: ...
        Testing Methodology: ...
        Temporal Context: ...
        Results: ...
        Limitations: ...

    Args:
        text: Testo narrativo dell'L1 summary.

    Returns:
        Dict con i campi estratti.
    """
    result: dict[str, Any] = {}

    # Key Contributions (lista numerata)
    contrib_match = re.search(
        r"Key Contributions:\s*\n((?:\s+\d+\..+\n?)+)",
        text,
        re.IGNORECASE,
    )
    if contrib_match:
        contribs = re.findall(r"\d+\.\s*(.+)", contrib_match.group(1))
        result["key_contributions"] = [c.strip() for c in contribs]
    else:
        result["key_contributions"] = []

    # Campi stringa: estraiamo il testo tra una label e la successiva
    field_map = {
        "methodology": r"Methodology:\s*(.+?)(?=\n\s*(?:Testing Methodology:|Temporal Context:|Results:|Limitations:)|\Z)",
        "testing_methodology": r"Testing Methodology:\s*(.+?)(?=\n\s*(?:Temporal Context:|Results:|Limitations:)|\Z)",
        "temporal_context": r"Temporal Context:\s*(.+?)(?=\n\s*(?:Results:|Limitations:)|\Z)",
        "results_summary": r"Results:\s*(.+?)(?=\n\s*(?:Limitations:)|\Z)",
        "limitations": r"Limitations:\s*(.+?)(?=\Z)",
    }

    for field, pattern in field_map.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result[field] = match.group(1).strip()
        else:
            result[field] = ""

    return result


# ============================================================================
# Salvataggio nel QdrantStore
# ============================================================================

def _save_l2_to_store(
    summary: L2Summary,
    store: Any = None,
) -> bool:
    """Salva un L2 summary come chunk nella collection l2_summaries.

    Il summary viene convertito in un singolo ``PaperChunk`` con
    ``chunk_type=L2_SUMMARY`` e indicizzato nel vector store.

    Prima di salvare, elimina eventuali L2 precedenti per la stessa
    categoria (per gestire aggiornamenti).

    Args:
        summary: L2Summary da salvare.
        store: QdrantStore. Se None, ne crea una nuova.

    Returns:
        True se il salvataggio è andato a buon fine.
    """
    try:
        if store is None:
            from paperhunter.indexing.qdrant_store import QdrantStore
            store = QdrantStore()

        collection = settings.qdrant.collection_l2_summaries

        # Elimina eventuali L2 precedenti per questa categoria
        _delete_existing_l2(summary.category, store)

        # Converti in testo per embedding
        summary_text = _l2_summary_to_text(summary)

        # Crea chunk ID deterministico
        chunk_id = _make_l2_chunk_id(summary.category, summary.version)

        # Crea PaperChunk con metadata estesi
        chunk = PaperChunk(
            chunk_id=chunk_id,
            text=summary_text,
            chunk_type=ChunkType.L2_SUMMARY,
            paper_id=f"l2_{summary.category}",
            bibtex_key=f"l2_{summary.category}_v{summary.version}",
            section_name="l2_summary",
            category=summary.category,
            year=0,  # L2 copre più anni
            month=0,
            chunk_index=0,
        )

        # Aggiungi metadata extra al payload (non nel modello PaperChunk)
        # Li gestiamo tramite il payload completo nel vector store

        n = store.add_chunks([chunk], collection_name=collection, show_progress=False)

        if n > 0:
            logger.debug(
                f"L2 salvato in Qdrant per '{summary.category}' v{summary.version}"
            )
            return True
        else:
            logger.warning(f"L2 non salvato per '{summary.category}'")
            return False

    except Exception as e:
        logger.error(f"Errore salvataggio L2 per '{summary.category}': {e}")
        return False


def _delete_existing_l2(category: str, store: Any) -> int:
    """Elimina eventuali L2 summary precedenti per una categoria.

    Args:
        category: Nome della categoria.
        store: QdrantStore.

    Returns:
        Numero di punti eliminati.
    """
    try:
        collection = settings.qdrant.collection_l2_summaries
        return store.delete_by_category(category, collection_name=collection)
    except Exception as e:
        logger.warning(f"Errore eliminazione L2 precedente per '{category}': {e}")
        return 0


def _count_l1_for_category(category: str, store: Any) -> int:
    """Conta gli L1 summary nel vector store per una categoria."""
    try:
        from qdrant_client import models
        result = store.client.count(
            collection_name=settings.qdrant.collection_l1_summaries,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category),
                    )
                ]
            ),
        )
        return result.count
    except Exception:
        return 0


def _l2_exists_and_valid(category: str, store: Any) -> bool:
    """Verifica se un L2 summary esiste ed è non-invalidato e non-stale.

    Controlla:
    1. Esistenza in Qdrant
    2. Flag invalidated nel metadata su disco
    3. Staleness: se il paper_count memorizzato nel metadata è inferiore
       al numero attuale di L1 per la categoria, il L2 è obsoleto.

    Args:
        category: Nome della categoria.
        store: QdrantStore.

    Returns:
        True se un L2 valido e aggiornato esiste già.
    """
    try:
        collection = settings.qdrant.collection_l2_summaries
        from qdrant_client import models

        result, _ = store.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not result:
            return False

        # Controlla se è invalidato (dal metadata su disco)
        metadata = _load_l2_metadata_from_disk(category)
        if metadata and metadata.get("invalidated", False):
            return False

        # Controlla staleness: se ci sono più L1 di quanti il L2 copra
        if metadata:
            stored_count = metadata.get("paper_count", 0)
            actual_l1_count = _count_l1_for_category(category, store)
            if actual_l1_count > 0 and actual_l1_count > stored_count:
                logger.info(
                    f"L2 '{category}' stale: copre {stored_count} paper "
                    f"ma ci sono {actual_l1_count} L1 attuali → rigenerazione"
                )
                return False

        return True

    except Exception:
        return False


# ============================================================================
# Helpers: recency tier stats
# ============================================================================


def _compute_recency_tier_stats(
    l1_summaries: list[L1Summary],
    current_year: int | None = None,
) -> dict:
    """Computa le statistiche di distribuzione per recency tier.

    Classifica ogni paper in full/reduced/compressed in base all'anno
    e restituisce conteggi e percentuali.

    Args:
        l1_summaries: Lista di L1Summary.
        current_year: Anno corrente. Se None, usa ``datetime.now().year``.

    Returns:
        Dict con chiavi: ``full``, ``reduced``, ``compressed`` (conteggi),
        ``full_pct``, ``reduced_pct``, ``compressed_pct`` (percentuali),
        ``total``, ``year_range`` (min-max anno).
    """
    if current_year is None:
        current_year = datetime.now().year

    n = len(l1_summaries)
    if n == 0:
        return {
            "full": 0, "reduced": 0, "compressed": 0,
            "full_pct": 0.0, "reduced_pct": 0.0, "compressed_pct": 0.0,
            "total": 0, "year_range": None,
        }

    counts = {"full": 0, "reduced": 0, "compressed": 0}
    years = []

    for s in l1_summaries:
        tier = _get_paper_recency_tier(s.year, current_year)
        counts[tier] += 1
        if s.year is not None:
            years.append(s.year)

    year_range = f"{min(years)}-{max(years)}" if years else None

    return {
        "full": counts["full"],
        "reduced": counts["reduced"],
        "compressed": counts["compressed"],
        "full_pct": counts["full"] * 100 / n,
        "reduced_pct": counts["reduced"] * 100 / n,
        "compressed_pct": counts["compressed"] * 100 / n,
        "total": n,
        "year_range": year_range,
    }


# ============================================================================
# Helpers: formatting — recency tiers & budget
# ============================================================================


def _get_paper_recency_tier(
    year: int | None,
    current_year: int | None = None,
) -> str:
    """Determina il tier di recency per un paper in base al suo anno.

    Usa le soglie configurate in ``config.yaml`` →
    ``summarization.l2_recency_tiers``:

    - ``full`` (≤N anni): testo L1 completo
    - ``reduced`` (≤M anni): ~70% del testo
    - ``compressed`` (>M anni): solo key_contributions + testing

    Args:
        year: Anno di pubblicazione. Se None, trattato come
            "molto vecchio" → compressed.
        current_year: Anno corrente. Se None, usa
            ``datetime.now().year``.

    Returns:
        ``"full"``, ``"reduced"`` o ``"compressed"``.
    """
    if current_year is None:
        current_year = datetime.now().year

    tiers = settings.summarization.l2_recency_tiers

    if year is None:
        return "compressed"

    years_ago = current_year - year

    if years_ago <= tiers.full:
        return "full"
    elif years_ago <= tiers.reduced:
        return "reduced"
    else:
        return "compressed"


def _compute_paper_importance(
    s: L1Summary,
    current_year: int | None = None,
) -> float:
    """Calcola un punteggio di importanza per un paper.

    Combina recency e citation count per determinare la
    "protezione" dalla compressione (paper più importanti
    vengono compressi per ultimi durante il downgrade).

    ``score = 0.7 × recency_factor + 0.3 × citation_factor``

    - recency_factor: ``(year − 2015) / (current_year − 2015)``,
      normalizzato in [0, 1]
    - citation_factor: ``log₁₀(1 + citation_count) / 5``,
      cappato a 1.0

    Args:
        s: L1Summary del paper.
        current_year: Anno corrente.

    Returns:
        Score float ≥ 0. Più alto = meno comprimibile.
    """
    if current_year is None:
        current_year = datetime.now().year

    base_year = 2015
    year = s.year or base_year
    recency_factor = max(
        0.0, (year - base_year) / max(1, current_year - base_year)
    )

    citation_count = getattr(s, "citation_count", None) or 0
    citation_factor = math.log10(1 + citation_count) / 5.0

    return 0.7 * recency_factor + 0.3 * min(citation_factor, 1.0)


def _truncate_text(
    text: str, max_chars: int, suffix: str = "...",
) -> str:
    """Tronca un testo a ``max_chars`` preservando parole intere.

    Args:
        text: Testo da troncare.
        max_chars: Lunghezza massima (incluso suffix).
        suffix: Suffisso da aggiungere se troncato.

    Returns:
        Testo eventualmente troncato.
    """
    if not text or len(text) <= max_chars:
        return text
    cutoff = max_chars - len(suffix)
    if cutoff <= 0:
        return suffix
    space_pos = text.rfind(" ", 0, cutoff)
    if space_pos > cutoff // 2:
        cutoff = space_pos
    return text[:cutoff] + suffix


def _format_l1_at_tier(
    s: L1Summary, tier: str, index: int, total: int,
) -> str:
    """Formatta un singolo L1 summary al livello di dettaglio del tier.

    Args:
        s: L1Summary da formattare.
        tier: ``"full"``, ``"reduced"`` o ``"compressed"``.
        index: Indice 1-based del paper nella lista.
        total: Numero totale di paper.

    Returns:
        Testo formattato.
    """
    if tier == "full":
        return _format_l1_full(s, index, total)
    elif tier == "reduced":
        return _format_l1_reduced(s, index, total)
    else:
        return _format_l1_compressed(s, index, total)


def _format_l1_full(s: L1Summary, index: int, total: int) -> str:
    """Formatta L1 a dettaglio completo (tutti i campi).

    Equivalente al formato originale pre-budget-aware.
    """
    bibtex = s.bibtex_key or s.paper_id
    year_str = str(s.year) if s.year else "Unknown"

    return f"""--- Paper {index}/{total}: [{bibtex}] (Year: {year_str}) ---

Key Contributions:
{_format_contributions(s.key_contributions)}

Methodology:
{s.methodology or "Not available."}

Testing Methodology:
{s.testing_methodology or "Not available."}

Temporal Context:
{s.temporal_context or "Not available."}

Results:
{s.results_summary or "Not available."}

Limitations:
{s.limitations or "Not available."}
"""


def _format_l1_reduced(s: L1Summary, index: int, total: int) -> str:
    """Formatta L1 a ~70% di dettaglio.

    Mantiene: key_contributions, methodology, testing_methodology,
    temporal_context (troncato a 200 chars), results (troncati a 300 chars).
    Rimuove: limitations.
    """
    bibtex = s.bibtex_key or s.paper_id
    year_str = str(s.year) if s.year else "Unknown"

    temporal = _truncate_text(s.temporal_context or "", 200)
    results = _truncate_text(s.results_summary or "", 300)

    return f"""--- Paper {index}/{total}: [{bibtex}] (Year: {year_str}) [REDUCED] ---

Key Contributions:
{_format_contributions(s.key_contributions)}

Methodology:
{s.methodology or "Not available."}

Testing Methodology:
{s.testing_methodology or "Not available."}

Temporal Context:
{temporal or "Not available."}

Results:
{results or "Not available."}
"""


def _format_l1_compressed(s: L1Summary, index: int, total: int) -> str:
    """Formatta L1 al livello minimale.

    Mantiene solo: key_contributions (max 3) + testing_methodology.
    """
    bibtex = s.bibtex_key or s.paper_id
    year_str = str(s.year) if s.year else "Unknown"

    contribs = s.key_contributions[:3] if s.key_contributions else []

    return f"""--- Paper {index}/{total}: [{bibtex}] (Year: {year_str}) [COMPRESSED] ---

Key Contributions:
{_format_contributions(contribs)}

Testing Methodology:
{s.testing_methodology or "Not available."}
"""


def _format_all_at_tiers(
    paper_entries: list[dict], total: int,
) -> list[str]:
    """Formatta tutti i paper ai tier attualmente assegnati.

    Args:
        paper_entries: Lista di dict con ``summary``, ``tier``, ``index``.
        total: Numero totale di paper.

    Returns:
        Lista di stringhe formattate (nell'ordine originale).
    """
    return [
        _format_l1_at_tier(
            entry["summary"],
            entry["tier"],
            entry["index"] + 1,
            total,
        )
        for entry in paper_entries
    ]


# ============================================================================
# Helpers: formatting — main budget-aware entry point
# ============================================================================


def _format_l1_summaries_for_prompt(l1_summaries: list[L1Summary]) -> str:
    """Formatta i summary L1 per il prompt L2 rispettando il budget token.

    Applica compressione progressiva multi-fase basata su:

    1. **Recency tiers** (da ``config.yaml`` →
       ``summarization.l2_recency_tiers``): paper recenti ricevono
       più spazio, paper vecchi vengono compressi automaticamente.
    2. **Citation weighting**: a parità di tier, paper con più
       citazioni sono meno comprimibili (downgraded per ultimi).
    3. **Progressive downgrade**: se il totale supera il budget,
       i paper vengono progressivamente downgraded
       (reduced→compressed, poi full→reduced, poi all-compressed)
       partendo dai meno importanti.
    4. **Hard truncation**: ultima risorsa se all-compressed
       ancora sfora.

    Il budget è derivato da ``summarization.max_context_tokens``
    sottraendo l'overhead del prompt template L2.

    Args:
        l1_summaries: Lista di L1Summary (già ordinati
            cronologicamente).

    Returns:
        Testo formattato entro il budget token configurato.
    """
    from paperhunter.utils.token_counter import compute_l2_budget

    n = len(l1_summaries)
    if n == 0:
        return ""

    # ------------------------------------------------------------------
    # 1) Calcolo budget disponibile
    # ------------------------------------------------------------------
    budget_info = compute_l2_budget()
    available_chars = budget_info["available_chars"]

    # Sottrai overhead stimato del prompt template L2 (istruzioni,
    # formattazione JSON, separatori, ecc.)
    _TEMPLATE_OVERHEAD_CHARS = 3500
    available_chars -= _TEMPLATE_OVERHEAD_CHARS
    available_chars = max(available_chars, 10_000)  # Floor di sicurezza

    current_year = datetime.now().year

    # ------------------------------------------------------------------
    # 2) Assegna recency tier e importanza a ogni paper
    # ------------------------------------------------------------------
    import paperhunter.config as _cfg
    recency_enabled = _cfg.settings.recency_priority

    paper_entries: list[dict] = []
    for i, s in enumerate(l1_summaries):
        if recency_enabled:
            tier = _get_paper_recency_tier(s.year, current_year)
        else:
            tier = "full"  # Nessuna compressione per recency
        importance = _compute_paper_importance(s, current_year)
        paper_entries.append({
            "summary": s,
            "tier": tier,
            "original_tier": tier,
            "importance": importance,
            "index": i,
        })

    tier_counts = {"full": 0, "reduced": 0, "compressed": 0}
    for entry in paper_entries:
        tier_counts[entry["tier"]] += 1

    logger.info(
        f"L2 recency weighting: {n} paper (recency_priority={'ON' if recency_enabled else 'OFF'}). "
        f"Tier: full={tier_counts['full']}, "
        f"reduced={tier_counts['reduced']}, "
        f"compressed={tier_counts['compressed']}. "
        f"Budget: {available_chars:,} chars"
    )

    # ------------------------------------------------------------------
    # 2b) Costruisci preamble con statistiche recency
    # ------------------------------------------------------------------
    if recency_enabled:
        tiers_cfg = _cfg.settings.summarization.l2_recency_tiers
        preamble_lines = [
            f"=== Recency-Weighted Paper Summaries ({n} papers) ===",
            f"Detail levels: FULL ({tier_counts['full']} papers, ≤{tiers_cfg.full}y) | "
            f"REDUCED ({tier_counts['reduced']} papers, {tiers_cfg.full+1}-{tiers_cfg.reduced}y) | "
            f"COMPRESSED ({tier_counts['compressed']} papers, >{tiers_cfg.reduced}y or unknown year)",
            "",
        ]
        preamble = "\n".join(preamble_lines)
    else:
        preamble = f"=== Paper Summaries ({n} papers, uniform detail) ===\n\n"

    # ------------------------------------------------------------------
    # 3) Fase 1: Formatta al tier assegnato (recency-based)
    # ------------------------------------------------------------------
    parts = _format_all_at_tiers(paper_entries, n)
    total_chars = sum(len(p) for p in parts)

    if total_chars <= available_chars:
        pct = total_chars * 100 // available_chars if available_chars else 0
        logger.info(
            f"L2 budget OK: {total_chars:,}/{available_chars:,} chars "
            f"({pct}%)"
        )
        return preamble + "\n".join(parts)

    # ------------------------------------------------------------------
    # 4) Fase 2: Downgrade reduced → compressed (meno importanti prima)
    # ------------------------------------------------------------------
    logger.info(
        f"L2 over budget: {total_chars:,}/{available_chars:,}. "
        f"Downgrading reduced→compressed..."
    )

    reduced_indices = sorted(
        (i for i, e in enumerate(paper_entries) if e["tier"] == "reduced"),
        key=lambda i: paper_entries[i]["importance"],
    )

    for idx in reduced_indices:
        paper_entries[idx]["tier"] = "compressed"
        parts = _format_all_at_tiers(paper_entries, n)
        total_chars = sum(len(p) for p in parts)
        if total_chars <= available_chars:
            break

    if total_chars <= available_chars:
        downgraded = sum(
            1 for e in paper_entries if e["tier"] != e["original_tier"]
        )
        logger.info(
            f"L2 budget OK dopo downgrade reduced→compressed: "
            f"{total_chars:,}/{available_chars:,} ({downgraded} downgraded)"
        )
        return preamble + "\n".join(parts)

    # ------------------------------------------------------------------
    # 5) Fase 3: Downgrade full → reduced (meno importanti prima)
    # ------------------------------------------------------------------
    logger.info(
        f"L2 ancora over: {total_chars:,}/{available_chars:,}. "
        f"Downgrading full→reduced..."
    )

    full_indices = sorted(
        (i for i, e in enumerate(paper_entries) if e["tier"] == "full"),
        key=lambda i: paper_entries[i]["importance"],
    )

    for idx in full_indices:
        paper_entries[idx]["tier"] = "reduced"
        parts = _format_all_at_tiers(paper_entries, n)
        total_chars = sum(len(p) for p in parts)
        if total_chars <= available_chars:
            break

    if total_chars <= available_chars:
        downgraded = sum(
            1 for e in paper_entries if e["tier"] != e["original_tier"]
        )
        logger.info(
            f"L2 budget OK dopo downgrade full→reduced: "
            f"{total_chars:,}/{available_chars:,} ({downgraded} downgraded)"
        )
        return preamble + "\n".join(parts)

    # ------------------------------------------------------------------
    # 6) Fase 4: All-compressed
    # ------------------------------------------------------------------
    logger.warning(
        f"L2 ancora over: {total_chars:,}/{available_chars:,}. "
        f"Forzando tutti i paper a tier compressed."
    )

    for entry in paper_entries:
        entry["tier"] = "compressed"

    parts = _format_all_at_tiers(paper_entries, n)
    total_chars = sum(len(p) for p in parts)

    if total_chars <= available_chars:
        logger.info(
            f"L2 budget OK a all-compressed: "
            f"{total_chars:,}/{available_chars:,}"
        )
        return preamble + "\n".join(parts)

    # ------------------------------------------------------------------
    # 7) Fase 5: Hard truncation proporzionale (ultima risorsa)
    # ------------------------------------------------------------------
    logger.warning(
        f"L2 ANCORA over a all-compressed: "
        f"{total_chars:,}/{available_chars:,}. "
        f"Applicando truncation proporzionale."
    )

    ratio = available_chars / max(total_chars, 1)
    truncated_parts: list[str] = []
    for p in parts:
        max_len = max(100, int(len(p) * ratio))
        if len(p) > max_len:
            truncated_parts.append(
                p[:max_len] + "\n[...truncated for budget...]"
            )
        else:
            truncated_parts.append(p)

    result = preamble + "\n".join(truncated_parts)
    logger.info(
        f"L2 hard truncation: {len(result):,}/{available_chars:,} chars "
        f"(ratio={ratio:.2f})"
    )
    return result


def _format_contributions(contributions: list[str]) -> str:
    """Formatta la lista di contributi come bullet points.

    Args:
        contributions: Lista di contributi.

    Returns:
        Testo formattato.
    """
    if not contributions:
        return "  - Not available."
    return "\n".join(f"  - {c}" for c in contributions)


def _l2_summary_to_text(summary: L2Summary) -> str:
    """Converte un L2Summary in testo per l'embedding nel vector store.

    Produce un testo sintetico che cattura tutti i campi del summary
    in forma narrativa, ottimale per il retrieval vettoriale.

    Args:
        summary: L2Summary.

    Returns:
        Testo del summary per embedding.
    """
    parts = [
        f"Category: {summary.category}",
        f"Papers Covered: {summary.paper_count}",
        f"Version: {summary.version}",
        "",
        f"Temporal Trends:\n{summary.temporal_trends}",
        "",
        f"Taxonomy:\n{summary.taxonomy}",
        "",
        f"Gaps Filled:\n{summary.gaps_filled}",
        "",
        f"Open Gaps:\n{summary.open_gaps}",
        "",
        f"Methodological Convergences:\n{summary.methodological_convergences}",
        "",
        f"Testing Overview:\n{summary.testing_overview}",
        "",
        f"Synthesis:\n{summary.synthesis}",
    ]

    return "\n".join(parts)


def _make_l2_chunk_id(category: str, version: int) -> str:
    """Genera un chunk_id deterministico per un L2 summary.

    Args:
        category: Nome della categoria.
        version: Versione del summary.

    Returns:
        Chunk ID di 16 caratteri hex.
    """
    raw = f"l2_summary::{category}::v{version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ============================================================================
# Helpers: parsing
# ============================================================================

def _parse_l2_json_response(raw_response: str, category: str) -> dict:
    """Estrae e parsa il JSON dalla risposta LLM per L2.

    Gestisce varie forme di output (stessa logica di L1):
    - JSON puro
    - JSON dentro code fence ```json ... ```
    - JSON con testo attorno

    Args:
        raw_response: Testo grezzo dalla risposta LLM.
        category: Nome della categoria (per logging).

    Returns:
        Dizionario parsato.

    Raises:
        ValueError: Se il JSON non può essere estratto.
    """
    text = raw_response.strip()

    # 1. Rimuovi code fence
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fence_match = re.search(fence_pattern, text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Trova il primo { e l'ultimo }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    # 3. Prova a parsare
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 4. Tentativo con fix comuni
    try:
        # Rimuovi trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        # Rimuovi commenti //
        fixed = re.sub(r"//[^\n]*", "", fixed)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict):
            logger.warning(
                f"JSON L2 riparato per '{category}' "
                f"(trailing commas o commenti rimossi)"
            )
            return parsed
    except json.JSONDecodeError:
        pass

    # 5. Fallback: estrai campi con regex
    logger.warning(
        f"JSON L2 non parsabile per '{category}'. "
        f"Tentativo estrazione regex. Response: {text[:300]}..."
    )
    return _regex_fallback_l2(text)


def _regex_fallback_l2(text: str) -> dict:
    """Fallback: estrai i campi L2 con regex dal testo.

    Args:
        text: Testo della risposta LLM.

    Returns:
        Dict con i campi estratti (potenzialmente incompleto).
    """
    result: dict[str, str] = {}

    fields = [
        "temporal_trends", "taxonomy", "gaps_filled", "open_gaps",
        "methodological_convergences", "testing_overview", "synthesis",
    ]

    for field in fields:
        pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}}]'
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            result[field] = m.group(1).strip().replace('\\"', '"').replace('\\n', '\n')

    if result:
        logger.info(f"Regex fallback L2: estratti {len(result)}/{len(fields)} campi")
    else:
        logger.error("Regex fallback L2: nessun campo estratto")

    return result


# ============================================================================
# Helpers: validazione
# ============================================================================

def _validate_l2_summary(summary: L2Summary) -> None:
    """Valida la qualità del summary L2 generato.

    Emette warning per campi vuoti o troppo brevi.

    Args:
        summary: L2Summary da validare.
    """
    issues: list[str] = []
    category = summary.category

    field_min_lengths = {
        "temporal_trends": 200,
        "taxonomy": 100,
        "gaps_filled": 100,
        "open_gaps": 100,
        "methodological_convergences": 100,
        "testing_overview": 150,
        "synthesis": 200,
    }

    for field_name, min_len in field_min_lengths.items():
        value = getattr(summary, field_name, "")
        if not value or len(value) < min_len:
            issues.append(
                f"{field_name} vuoto o troppo breve ({len(value or '')} chars, "
                f"min {min_len})"
            )

    if issues:
        for issue in issues:
            print_warning(f"L2 '{category}': {issue}")
        logger.warning(
            f"L2 summary per '{category}' ha {len(issues)} issue di qualità"
        )
    else:
        logger.debug(f"L2 summary per '{category}' passa la validazione")


# ============================================================================
# Helpers: versioning e discovery
# ============================================================================

def _get_next_l2_version(category: str, store: Any = None) -> int:
    """Calcola la prossima versione per un L2 summary.

    Legge i metadata su disco per trovare l'ultima versione.

    Args:
        category: Nome della categoria.
        store: QdrantStore (non usato al momento, per future estensioni).

    Returns:
        Numero di versione successivo (parte da 1).
    """
    metadata = _load_l2_metadata_from_disk(category)
    if metadata and "version" in metadata:
        return metadata["version"] + 1
    return 1


def _discover_categories_from_store(store: Any) -> list[str]:
    """Scopre le categorie uniche nel vector store.

    Scansiona la collection ``l1_summaries`` per estrarre
    le categorie con almeno un L1 summary.

    Args:
        store: QdrantStore.

    Returns:
        Lista di nomi di categorie.
    """
    try:
        collection = settings.qdrant.collection_l1_summaries
        stats = store.get_stats(collection_name=collection)
        categories = list(stats.get("categories", {}).keys())
        return sorted(categories)
    except Exception as e:
        logger.warning(f"Errore discovery categorie: {e}")
        return []


# ============================================================================
# Persistenza metadata L2 su disco
# ============================================================================

_L2_METADATA_DIR = Path(settings.output.base_dir) / "l2_metadata"


def _save_l2_metadata_to_disk(
    summary: L2Summary,
    *,
    recency_tier_stats: dict | None = None,
) -> None:
    """Salva i metadata di un L2 summary su disco come JSON.

    Usato per recovery, tracking versioni, e check invalidazione.

    Args:
        summary: L2Summary da salvare.
        recency_tier_stats: Statistiche distribuzione recency tier
            (da ``_compute_recency_tier_stats``). Se fornite, vengono
            incluse nel file di metadata.
    """
    try:
        _L2_METADATA_DIR.mkdir(parents=True, exist_ok=True)

        # Nome file: normalizzazione della categoria
        safe_name = re.sub(r"[^a-z0-9_]", "_", summary.category.lower())
        filepath = _L2_METADATA_DIR / f"{safe_name}.json"

        metadata = {
            "category": summary.category,
            "version": summary.version,
            "paper_count": summary.paper_count,
            "paper_ids": summary.paper_ids,
            "invalidated": summary.invalidated,
            "generated_at": summary.generated_at.isoformat() if summary.generated_at else None,
            "model_used": summary.model_used,
        }

        if recency_tier_stats is not None:
            metadata["recency_tier_stats"] = recency_tier_stats

        filepath.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        logger.debug(f"L2 metadata salvati: {filepath}")

    except Exception as e:
        logger.warning(f"Errore salvataggio metadata L2 su disco: {e}")


def _load_l2_metadata_from_disk(category: str) -> dict | None:
    """Carica i metadata di un L2 summary da disco.

    Args:
        category: Nome della categoria.

    Returns:
        Dict con i metadata, o None se non trovato.
    """
    try:
        safe_name = re.sub(r"[^a-z0-9_]", "_", category.lower())
        filepath = _L2_METADATA_DIR / f"{safe_name}.json"

        if not filepath.exists():
            return None

        return json.loads(filepath.read_text())

    except Exception as e:
        logger.warning(f"Errore caricamento metadata L2 per '{category}': {e}")
        return None


def load_all_l2_metadata() -> dict[str, dict]:
    """Carica tutti i metadata L2 da disco.

    Returns:
        Dict categoria → metadata.
    """
    result: dict[str, dict] = {}

    try:
        if not _L2_METADATA_DIR.exists():
            return result

        for filepath in _L2_METADATA_DIR.glob("*.json"):
            try:
                metadata = json.loads(filepath.read_text())
                category = metadata.get("category", filepath.stem)
                result[category] = metadata
            except Exception as e:
                logger.warning(f"Errore lettura {filepath}: {e}")

    except Exception as e:
        logger.warning(f"Errore caricamento metadata L2: {e}")

    return result


# ============================================================================
# Report e output
# ============================================================================

def _print_batch_report(
    total: int,
    generated: int,
    skipped: int,
    errors: int,
    error_details: dict[str, str],
) -> None:
    """Stampa report riassuntivo del batch L2.

    Args:
        total: Numero totale di categorie.
        generated: Numero di L2 generati con successo.
        skipped: Categorie saltate.
        errors: Numero di errori.
        error_details: Dict categoria → messaggio errore.
    """
    print_table("L2 Summary Batch Report", {
        "Totale categorie": total,
        "L2 generati": generated,
        "Saltati": skipped,
        "Errori": errors,
    })

    if error_details:
        print_warning("Errori durante la generazione L2:")
        for cat, err in error_details.items():
            logger.error(f"  {cat}: {err}")

    if generated > 0:
        print_success(
            f"L2 batch completato: {generated}/{total} summary generati"
        )
