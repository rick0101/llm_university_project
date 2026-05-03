"""
Level 1 Summarizer — Summary strutturato per singolo paper.

Genera un summary L1 ultra-specifico per ogni paper usando Gemini Pro,
con estrazione obbligatoria della ``testing_methodology`` (benchmark,
dataset, metriche, prompt usati dagli autori per testare).

Output strutturato in JSON con i campi del modello ``L1Summary``:
- key_contributions (max 5 bullet points)
- methodology (approccio tecnico dettagliato)
- testing_methodology (**CAMPO CRITICO**)
- temporal_context (posizione nella timeline SOTA)
- results_summary (risultati quantitativi)
- limitations (limiti dichiarati o impliciti)

Il summary viene anche salvato come chunk nel QdrantStore
(collection ``l1_summaries``) per il retrieval successivo.

Uso::

    from paperhunter.summarization.level1_summarizer import (
        generate_l1_summary,
        generate_l1_batch,
    )
    from paperhunter.models import Paper, PaperChunk

    summary = generate_l1_summary(paper, chunks)
    print(summary.testing_methodology)

    # Batch
    summaries = generate_l1_batch(papers_with_chunks)
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.models import ChunkType, ClaimAtomic, L1Summary, Paper, PaperChunk
from paperhunter.summarization.taxonomy import normalize_list, vocab_for_prompt
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
from paperhunter.utils.token_counter import (
    compute_l1_target_chars,
    get_l1_compression_tier,
    get_papers_per_category,
)

logger = get_logger(__name__)


# ============================================================================
# Prompt template
# ============================================================================

_L1_PROMPT_TEMPLATE = """\
You are an expert academic researcher specializing in LLM security. \
Analyze the following paper and produce a **structured summary** in JSON format.

## Paper Metadata
- **Title:** {title}
- **Authors:** {authors}
- **Year:** {year}
- **Venue:** {venue}
- **Category:** {category}
- **Citations:** {citations}

## Paper Content (chunks)
{chunks_text}

---

## Controlled Vocabularies (use ONLY these values when applicable)

- **attack_vector** — pick from: {attack_vectors_vocab}
- **defense_type** — pick from: {defense_types_vocab}
- **evaluation_metrics** — pick from: {evaluation_metrics_vocab}
- **datasets** — pick from: {datasets_vocab}

If the paper uses a sub-domain not covered by the vocabularies, leave the
list empty for that field and put the free-form term in ``subcategory_freeform``.

## Instructions

Based ONLY on the paper content above, extract the following fields. \
Be precise, specific, and use the actual terminology from the paper. \
If a field cannot be determined from the content, write "Not explicitly stated in the paper."

Respond with a valid JSON object (no markdown fences, no comments):

{{
  "key_contributions": [
    "Contribution 1 (one concise sentence)",
    "Contribution 2"
  ],
  "methodology": "Detailed description of the technical approach: what model/system/attack/defense is proposed, how it works step by step, key algorithmic or architectural choices.",
  "testing_methodology": "CRITICAL FIELD — Extract ALL of the following if available: (1) benchmark datasets used (name, size, source), (2) evaluation metrics (exact names like ASR, BLEU, F1, etc.), (3) baseline methods compared against, (4) models tested on (GPT-4, LLaMA, etc.), (5) any open-source prompts, tools, or frameworks used for testing, (6) experimental setup details (hardware, API version, etc.). Be exhaustive.",
  "temporal_context": "When was this method proposed? How does it relate to prior work? What gap does it address?",
  "results_summary": "Key quantitative results: accuracy/success rates, performance numbers, comparison with baselines. Use actual numbers from the paper.",
  "limitations": "Limitations explicitly stated by the authors AND any implicit limitations.",

  "threat_model": "One concise sentence describing WHO attacks WHAT and under WHICH access (e.g., 'External attacker, black-box access to LLM API, no fine-tuning'). Empty string if not an offensive paper.",
  "attack_vector": ["use canonical values from the vocabulary above"],
  "defense_type": ["use canonical values from the vocabulary above"],
  "target_models": ["gpt-4", "llama-2-7b", "..."],
  "evaluation_metrics": ["ASR", "F1", "..."],
  "datasets": ["AdvBench", "..."],
  "subcategory_freeform": "free-form sub-category if controlled vocab is insufficient, else empty",

  "claims": [
    {{
      "text": "Atomic, fact-checkable single sentence (e.g., 'Method X achieves 87.3% ASR on AdvBench against LLaMA-2-7B').",
      "claim_type": "empirical|methodological|theoretical",
      "evidence_chunk_ids": ["chunk_id_1"]
    }}
  ]
}}

IMPORTANT:
- key_contributions must have 1-5 entries maximum.
- testing_methodology is critical — be extremely thorough.
- ``claims`` must be ATOMIC: ONE fact per item. Numbers must come from the paper.
- ``evidence_chunk_ids``: cite the chunk IDs you can see in "Paper Content" above (the chunk identifier shown before each chunk). Use [] if uncertain.
- Use the controlled vocabularies for attack_vector / defense_type / evaluation_metrics / datasets. Lowercase snake_case for the first two; original case for metrics/datasets.
- Do NOT hallucinate or invent data not present in the source chunks.
"""

# --- Prompt «concise»: per categorie con 31-70 paper ---
# Output target: ~1.5k chars. Campi ridotti, meno dettaglio narrativo.
_L1_PROMPT_CONCISE = """\
You are an expert academic researcher specializing in LLM security. \
Analyze the following paper and produce a **concise structured summary** in JSON format.

## Paper Metadata
- **Title:** {title}
- **Authors:** {authors}
- **Year:** {year}
- **Venue:** {venue}
- **Category:** {category}
- **Citations:** {citations}

## Paper Content (chunks)
{chunks_text}

---

## Controlled Vocabularies (use ONLY these values when applicable)

- **attack_vector** — pick from: {attack_vectors_vocab}
- **defense_type** — pick from: {defense_types_vocab}
- **evaluation_metrics** — pick from: {evaluation_metrics_vocab}
- **datasets** — pick from: {datasets_vocab}

## Instructions

Based ONLY on the paper content above, extract the following fields. \
Be concise but precise. Target total output length: ~1800 characters.

Respond with a valid JSON object (no markdown fences, no comments):

{{
  "key_contributions": ["Contribution 1", "Contribution 2"],
  "methodology": "Brief description (2-3 sentences max).",
  "testing_methodology": "CRITICAL — Concisely list: benchmark datasets, evaluation metrics, models tested on, key baselines.",
  "temporal_context": "One sentence positioning this work.",
  "results_summary": "Key quantitative results only.",
  "limitations": "Main limitations in 1-2 sentences.",

  "threat_model": "One sentence (or empty for defense papers).",
  "attack_vector": ["..."],
  "defense_type": ["..."],
  "target_models": ["..."],
  "evaluation_metrics": ["..."],
  "datasets": ["..."],
  "subcategory_freeform": "",

  "claims": [
    {{"text": "Atomic claim (1 fact)", "claim_type": "empirical", "evidence_chunk_ids": []}}
  ]
}}

IMPORTANT:
- key_contributions: max 3 entries.
- claims: 2-5 atomic, fact-checkable items maximum.
- Use the controlled vocabularies above.
- Do NOT hallucinate.
"""

# --- Prompt «compressed»: per categorie con >70 paper ---
# Output target: ~1.2k chars. Solo campi essenziali.
_L1_PROMPT_COMPRESSED = """\
You are an expert academic researcher specializing in LLM security. \
Analyze the following paper and produce a **highly compressed summary** in JSON format.

## Paper Metadata
- **Title:** {title}
- **Authors:** {authors}
- **Year:** {year}
- **Venue:** {venue}
- **Category:** {category}
- **Citations:** {citations}

## Paper Content (chunks)
{chunks_text}

---

## Controlled Vocabularies (use ONLY these values when applicable)

- **attack_vector** — pick from: {attack_vectors_vocab}
- **defense_type** — pick from: {defense_types_vocab}
- **evaluation_metrics** — pick from: {evaluation_metrics_vocab}
- **datasets** — pick from: {datasets_vocab}

## Instructions

Extract ONLY the most essential information. Target total output: ~1200 characters.

Respond with a valid JSON object (no markdown fences, no comments):

{{
  "key_contributions": ["Main contribution"],
  "methodology": "Core technique in 1-2 sentences.",
  "testing_methodology": "Datasets, metrics, models tested.",
  "temporal_context": "",
  "results_summary": "Top result numbers only.",
  "limitations": "Primary limitation.",

  "threat_model": "",
  "attack_vector": ["..."],
  "defense_type": ["..."],
  "target_models": ["..."],
  "evaluation_metrics": ["..."],
  "datasets": ["..."],
  "subcategory_freeform": "",

  "claims": [
    {{"text": "Top atomic claim (numerical preferred)", "claim_type": "empirical", "evidence_chunk_ids": []}}
  ]
}}

IMPORTANT:
- key_contributions: max 2 entries.
- claims: 1-3 atomic items maximum.
- temporal_context can be left empty.
- Do NOT hallucinate.
"""


# ============================================================================
# Core function
# ============================================================================

# ---------------------------------------------------------------------------
# Selezione prompt per tier di compressione
# ---------------------------------------------------------------------------

def _get_l1_prompt_template(tier: str) -> str:
    """Restituisce il prompt template L1 adatto al tier di compressione.

    Args:
        tier: ``"standard"``, ``"concise"`` o ``"compressed"``.

    Returns:
        Template string con placeholder per ``.format()``.
    """
    if tier == "concise":
        return _L1_PROMPT_CONCISE
    elif tier == "compressed":
        return _L1_PROMPT_COMPRESSED
    else:
        return _L1_PROMPT_TEMPLATE


def _compute_l1_char_count(summary: L1Summary) -> int:
    """Calcola il numero totale di caratteri dei campi testuali di un L1Summary.

    Conta solo i campi che verranno inclusi nel prompt L2.

    Args:
        summary: L1Summary da misurare.

    Returns:
        Numero totale di caratteri.
    """
    parts = [
        "; ".join(summary.key_contributions),
        summary.methodology,
        summary.testing_methodology,
        summary.temporal_context,
        summary.results_summary,
        summary.limitations,
    ]
    return sum(len(p) for p in parts if p)


def generate_l1_summary(
    paper: Paper,
    chunks: list[PaperChunk],
    *,
    force_model: str | None = None,
    max_chunk_chars: int = 50_000,
    save_to_store: bool = False,
    store: Any = None,
    compression_tier: str | None = None,
    n_papers_in_category: int | None = None,
) -> L1Summary:
    """Genera un summary L1 strutturato per un singolo paper.

    Quando ``l1_target_chars_auto`` è abilitato in config, il prompt 
    viene adattato automaticamente in base al numero di paper nella
    categoria, per garantire che la fase L2 a valle resti entro il
    budget token.

    Tier di compressione:
    - ``"standard"`` (N ≤ 30): prompt completo, ~2-3k chars/L1
    - ``"concise"`` (30 < N ≤ 70): prompt ridotto, ~1.5k chars/L1
    - ``"compressed"`` (N > 70): prompt minimale, ~1k chars/L1

    Args:
        paper: Oggetto ``Paper`` con metadata completi.
        chunks: Lista di ``PaperChunk`` del paper (testo originale).
        force_model: Override tier modello (``'flash'`` o ``'pro'``).
            Default: da ``config.yaml`` → ``summarization.use_pro_for_l1``.
        max_chunk_chars: Massimo numero di caratteri dal testo dei chunk
            da includere nel prompt (per restare nel context window).
        save_to_store: Se True, salva il summary come chunk nella
            collection ``l1_summaries`` del QdrantStore.
        store: Istanza ``QdrantStore`` opzionale. Se ``save_to_store=True``
            e ``store`` è None, ne crea una nuova.
        compression_tier: Override tier di compressione 
            (``"standard"``, ``"concise"``, ``"compressed"``).
            Se None, viene calcolato automaticamente da config.
        n_papers_in_category: Numero di paper nella categoria.
            Se None, usa ``papers_per_category`` da config.

    Returns:
        Istanza ``L1Summary`` con tutti i campi popolati + ``l1_char_count``.

    Raises:
        ValueError: Se non ci sono chunk da processare.
        RuntimeError: Se la chiamata LLM fallisce dopo tutti i retry.
    """
    if not chunks:
        raise ValueError(
            f"Nessun chunk fornito per il paper '{paper.title}'. "
            f"Esegui prima l'ingestion pipeline."
        )

    # --- Determinazione tier di compressione budget-aware ---
    sum_cfg = settings.summarization
    if compression_tier is not None:
        c_tier = compression_tier
    elif sum_cfg.l1_target_chars_auto:
        n_papers = n_papers_in_category or get_papers_per_category()
        c_tier = get_l1_compression_tier(n_papers)
        target_chars = compute_l1_target_chars(n_papers)
        logger.debug(
            f"Budget-aware L1: n_papers={n_papers}, tier={c_tier}, "
            f"target_chars={target_chars}"
        )
    else:
        c_tier = "standard"

    # --- Selezione modello ---
    pro_tracker = get_pro_tracker()
    if force_model:
        tier = force_model
    else:
        tier = "pro" if sum_cfg.use_pro_for_l1 else "flash"

    llm = get_pro_llm() if tier == "pro" else get_flash_llm()

    # --- Preparazione testo chunk (con filtro rilevanza) ---
    ref_text = f"{paper.title or ''} {paper.abstract or ''}"
    chunks_text = _prepare_chunks_text(
        chunks, max_chars=max_chunk_chars, reference_text=ref_text,
    )

    # --- Costruzione prompt con template adatto al tier ---
    prompt_template = _get_l1_prompt_template(c_tier)
    prompt = prompt_template.format(
        title=paper.title or "Unknown",
        authors=", ".join(paper.authors[:10]) if paper.authors else "Unknown",
        year=paper.year or "Unknown",
        venue=paper.venue or paper.journal or "Unknown",
        category=paper.category or "Unknown",
        citations=paper.citations or "N/A",
        chunks_text=chunks_text,
        # v3.6 — vocabolari controllati iniettati
        attack_vectors_vocab=vocab_for_prompt("attack_vector"),
        defense_types_vocab=vocab_for_prompt("defense_type"),
        evaluation_metrics_vocab=vocab_for_prompt("evaluation_metrics"),
        datasets_vocab=vocab_for_prompt("datasets"),
    )

    # --- Chiamata LLM con rate limiting ---
    logger.info(
        f"Generazione L1 summary per '{paper.title[:60]}...' "
        f"(model={tier}, compression={c_tier}, {len(chunks)} chunk, "
        f"{len(chunks_text)} chars)"
    )

    limiter = get_gemini_limiter()
    limiter.wait()

    # --- Retry strategy basata su config ---
    # Pro: retry pazienti con delay lunghi (15s base, cap 120s, 6 retry)
    # Flash: retry standard (3s base, cap 60s, 3 retry)
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
        # Se Pro ha risposto, resetta il tracker
        if tier == "pro":
            pro_tracker.record_success()
    except Exception as e:
        error_str = str(e).lower()
        is_transient = any(kw in error_str for kw in (
            "503", "504", "unavailable", "overloaded", "deadline",
        ))

        if tier == "pro" and is_transient:
            pro_tracker.record_failure()

            # Fallback a Flash SOLO se esplicitamente consentito in config
            if sum_cfg.allow_flash_fallback:
                logger.warning(
                    f"Pro esaurito per '{paper.title[:40]}...', "
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
                    f"Pro non disponibile per '{paper.title[:40]}...' dopo "
                    f"{pro_retries + 1} tentativi. Flash fallback disabilitato. "
                    f"Riprova più tardi o abilita allow_flash_fallback in config.yaml"
                )
                raise
        else:
            raise
    elapsed = time.time() - start_time

    logger.debug(f"LLM response ricevuta in {elapsed:.1f}s ({len(raw_response)} chars)")

    # --- Parsing JSON response ---
    parsed = _parse_json_response(raw_response, paper.title)

    # --- Costruzione L1Summary ---
    paper_id = paper.bibtex_key or paper.pdf_hash or paper.title[:30]

    # Max key_contributions basato sul tier di compressione
    max_contribs = {"standard": 5, "concise": 3, "compressed": 2}.get(c_tier, 5)

    def _to_str(v) -> str:
        """Coerce dict/list restituiti dal modello in stringa JSON."""
        if isinstance(v, str):
            return v
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v) if v is not None else ""

    # --- Parsing e normalizzazione campi tassonomici (v3.6) ---
    raw_attack = parsed.get("attack_vector") or []
    raw_defense = parsed.get("defense_type") or []
    raw_metrics = parsed.get("evaluation_metrics") or []
    raw_datasets = parsed.get("datasets") or []
    raw_models = parsed.get("target_models") or []
    if not isinstance(raw_attack, list):
        raw_attack = [str(raw_attack)]
    if not isinstance(raw_defense, list):
        raw_defense = [str(raw_defense)]
    if not isinstance(raw_metrics, list):
        raw_metrics = [str(raw_metrics)]
    if not isinstance(raw_datasets, list):
        raw_datasets = [str(raw_datasets)]
    if not isinstance(raw_models, list):
        raw_models = [str(raw_models)]

    attack_vec = normalize_list("attack_vector", raw_attack)
    defense_t = normalize_list("defense_type", raw_defense)
    eval_m = normalize_list("evaluation_metrics", raw_metrics)
    datasets_n = normalize_list("datasets", raw_datasets)
    # target_models: normalizza solo se matcha gli hint, altrimenti lowercase strip
    target_m_norm = normalize_list("target_models", raw_models)
    if not target_m_norm and raw_models:
        target_m_norm = [
            str(m).strip().lower() for m in raw_models
            if isinstance(m, (str, int, float)) and str(m).strip()
        ]

    # Parsing claims
    raw_claims = parsed.get("claims") or []
    claims_list: list[ClaimAtomic] = []
    if isinstance(raw_claims, list):
        for c in raw_claims:
            if not isinstance(c, dict):
                continue
            try:
                ctype = str(c.get("claim_type", "empirical")).strip().lower()
                if ctype not in {"empirical", "methodological", "theoretical"}:
                    ctype = "empirical"
                evidence = c.get("evidence_chunk_ids") or []
                if not isinstance(evidence, list):
                    evidence = [str(evidence)]
                evidence = [str(e) for e in evidence if e]
                txt = str(c.get("text", "")).strip()
                if not txt:
                    continue
                claims_list.append(
                    ClaimAtomic(text=txt, claim_type=ctype, evidence_chunk_ids=evidence)
                )
            except Exception:
                continue

    summary = L1Summary(
        paper_id=paper_id,
        bibtex_key=paper.bibtex_key,
        category=paper.category,
        year=paper.year,
        key_contributions=parsed.get("key_contributions", [])[:max_contribs],
        methodology=_to_str(parsed.get("methodology", "")),
        testing_methodology=_to_str(parsed.get("testing_methodology", "")),
        temporal_context=_to_str(parsed.get("temporal_context", "")),
        results_summary=_to_str(parsed.get("results_summary", "")),
        limitations=_to_str(parsed.get("limitations", "")),
        # v3.6 — tassonomia + claims + citability passthrough
        threat_model=_to_str(parsed.get("threat_model", "")),
        attack_vector=attack_vec,
        defense_type=defense_t,
        target_models=target_m_norm,
        evaluation_metrics=eval_m,
        datasets=datasets_n,
        subcategory_freeform=_to_str(parsed.get("subcategory_freeform", "")),
        claims=claims_list,
        is_citable=bool(getattr(paper, "is_citable", False)),
        citation_tier=getattr(paper, "citation_tier", "uncited_context"),
        compression_tier=c_tier,
        generated_at=datetime.now(),
        model_used=tier,
    )

    # --- Calcolo e tracking l1_char_count ---
    summary.l1_char_count = _compute_l1_char_count(summary)

    # --- Validazione qualità ---
    _validate_l1_summary(summary, paper.title)

    # --- Salvataggio opzionale nel vector store ---
    if save_to_store:
        _save_l1_to_store(summary, paper, store)

    logger.info(
        f"L1 summary generato per '{paper.title[:50]}...' "
        f"({elapsed:.1f}s, model={tier}, compression={c_tier}, "
        f"{summary.l1_char_count} chars)"
    )

    return summary


# ============================================================================
# Batch processing
# ============================================================================

def generate_l1_batch(
    papers_with_chunks: list[tuple[Paper, list[PaperChunk]]],
    *,
    force_model: str | None = None,
    save_to_store: bool = False,
    store: Any = None,
    skip_existing: bool = True,
    compression_tier: str | None = None,
    n_papers_in_category: int | None = None,
) -> list[L1Summary]:
    """Genera summary L1 per un batch di paper.

    Il tier di compressione viene determinato automaticamente dal numero
    di paper nel batch (se ``l1_target_chars_auto=true`` in config),
    oppure può essere forzato con ``compression_tier``.

    Args:
        papers_with_chunks: Lista di tuple ``(Paper, list[PaperChunk])``.
        force_model: Override tier modello.
        save_to_store: Salva ogni summary nel QdrantStore.
        store: Istanza ``QdrantStore``.
        skip_existing: Se True e ``store`` fornito, salta paper
            che hanno già un L1 nel vector store.
        compression_tier: Override tier di compressione per tutto il batch
            (``"standard"``, ``"concise"``, ``"compressed"``).
        n_papers_in_category: Numero totale di paper nella categoria.
            Se None e ``compression_tier`` è None, usa ``len(papers_with_chunks)``
            o ``papers_per_category`` da config (il maggiore dei due).

    Returns:
        Lista di ``L1Summary`` generati con successo.
    """
    if not papers_with_chunks:
        logger.warning("generate_l1_batch: lista vuota")
        return []

    # --- Determina tier di compressione per il batch ---
    if compression_tier is None and n_papers_in_category is None:
        # Stima: usa il massimo tra la dimensione del batch e papers_per_category
        n_papers_in_category = max(
            len(papers_with_chunks),
            get_papers_per_category(),
        )

    if compression_tier is None:
        batch_tier = get_l1_compression_tier(n_papers_in_category)
    else:
        batch_tier = compression_tier

    batch_target_chars = compute_l1_target_chars(
        n_papers=n_papers_in_category or len(papers_with_chunks)
    )
    logger.info(
        f"L1 batch: {len(papers_with_chunks)} paper, "
        f"compression_tier={batch_tier}, target_chars={batch_target_chars}"
    )

    # --- Inizializzazione store se necessario ---
    if save_to_store and store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    summaries: list[L1Summary] = []
    errors: dict[str, str] = {}
    skipped = 0
    total_chars = 0

    logger.info(f"Generazione L1 batch per {len(papers_with_chunks)} paper...")

    gc.disable()
    try:
        for paper, chunks in progress_bar(
            papers_with_chunks,
            description="L1 Summaries",
        ):
            paper_id = paper.bibtex_key or paper.pdf_hash or paper.title[:30]

            # Skip se già presente
            if skip_existing and store is not None:
                if _l1_exists_in_store(paper_id, store):
                    logger.debug(f"L1 già esistente per '{paper_id}', skip")
                    skipped += 1
                    continue

            try:
                summary = generate_l1_summary(
                    paper,
                    chunks,
                    force_model=force_model,
                    save_to_store=save_to_store,
                    store=store,
                    compression_tier=batch_tier,
                    n_papers_in_category=n_papers_in_category,
                )
                summaries.append(summary)
                total_chars += summary.l1_char_count or 0
                del summary
            except Exception as e:
                logger.error(f"Errore L1 per '{paper.title[:50]}...': {e}")
                errors[paper_id] = str(e)
    finally:
        gc.enable()
        gc.collect()

    # --- Report con statistiche budget ---
    avg_chars = total_chars // len(summaries) if summaries else 0
    _print_batch_report(
        total=len(papers_with_chunks),
        generated=len(summaries),
        skipped=skipped,
        errors=len(errors),
        error_details=errors,
        compression_tier=batch_tier,
        avg_l1_chars=avg_chars,
        total_l1_chars=total_chars,
        target_chars=batch_target_chars,
    )

    return summaries


# ============================================================================
# Salvataggio nel QdrantStore
# ============================================================================

def _save_l1_to_store(
    summary: L1Summary,
    paper: Paper,
    store: Any = None,
) -> bool:
    """Salva un L1 summary come chunk nella collection l1_summaries.

    Il summary viene convertito in un singolo ``PaperChunk`` con
    ``chunk_type=L1_SUMMARY`` e indicizzato nel vector store.

    Args:
        summary: L1Summary da salvare.
        paper: Paper associato.
        store: QdrantStore. Se None, ne crea una nuova.

    Returns:
        True se il salvataggio è andato a buon fine.
    """
    try:
        if store is None:
            from paperhunter.indexing.qdrant_store import QdrantStore
            store = QdrantStore()

        # Componi un testo sintetico per l'embedding
        summary_text = _l1_summary_to_text(summary, paper)

        # Crea chunk ID deterministico
        chunk_id = _make_l1_chunk_id(summary.paper_id)

        # Crea PaperChunk (v3.6 — propaga citability + tassonomia ai metadata Qdrant)
        chunk = PaperChunk(
            chunk_id=chunk_id,
            text=summary_text,
            chunk_type=ChunkType.L1_SUMMARY,
            paper_id=summary.paper_id,
            bibtex_key=summary.bibtex_key,
            section_name="l1_summary",
            category=summary.category,
            year=summary.year,
            month=paper.month,
            chunk_index=0,
            is_citable=bool(summary.is_citable),
            citation_tier=summary.citation_tier or "uncited_context",
            attack_vector=list(summary.attack_vector),
            defense_type=list(summary.defense_type),
            target_models=list(summary.target_models),
            evaluation_metrics=list(summary.evaluation_metrics),
            datasets=list(summary.datasets),
            threat_model=summary.threat_model or None,
        )

        # Salva nella collection l1_summaries
        collection = settings.qdrant.collection_l1_summaries
        n = store.add_chunks([chunk], collection_name=collection, show_progress=False)

        if n > 0:
            logger.debug(f"L1 salvato in Qdrant per '{summary.paper_id}'")
            return True
        else:
            logger.warning(f"L1 non salvato per '{summary.paper_id}'")
            return False

    except Exception as e:
        logger.error(f"Errore salvataggio L1 per '{summary.paper_id}': {e}")
        return False


def _l1_exists_in_store(paper_id: str, store: Any) -> bool:
    """Verifica se un L1 summary esiste già nel vector store.

    Args:
        paper_id: ID del paper (bibtex_key).
        store: QdrantStore.

    Returns:
        True se il L1 è già presente.
    """
    try:
        collection = settings.qdrant.collection_l1_summaries
        return store.has_paper(paper_id, collection_name=collection)
    except Exception:
        return False


# ============================================================================
# Helpers
# ============================================================================

def _prepare_chunks_text(
    chunks: list[PaperChunk],
    max_chars: int = 50_000,
    reference_text: str = "",
) -> str:
    """Prepara il testo dei chunk per il prompt LLM.

    Se ``reference_text`` è fornito (es. titolo + abstract del paper),
    i chunk vengono filtrati per rilevanza TF-IDF prima della selezione,
    evitando di sprecare token su appendici, ringraziamenti e sezioni
    poco informative.

    I chunk di tipo abstract/introduction sono sempre inclusi per primi
    (priority 0/1), indipendentemente dal punteggio di rilevanza.

    Args:
        chunks: Lista di PaperChunk.
        max_chars: Limite massimo di caratteri.
        reference_text: Testo di riferimento per il filtraggio
            (tipicamente titolo + abstract). Se vuoto, nessun filtro.

    Returns:
        Testo formattato per il prompt.
    """
    # --- Sezioni a bassa informazione da penalizzare ---
    _LOW_VALUE_SECTIONS = {
        "acknowledgement", "acknowledgements", "acknowledgment",
        "references", "bibliography", "appendix", "appendices",
        "supplementary", "supplementary material",
        "author contributions", "funding", "declaration",
        "conflict of interest", "data availability",
    }

    def _section_priority(section_name: str) -> int:
        lower = (section_name or "").lower()
        if "abstract" in lower:
            return 0
        if "introduction" in lower or "intro" in lower:
            return 1
        if any(lv in lower for lv in _LOW_VALUE_SECTIONS):
            return 4  # bassa priorità
        return 2  # sezioni normali

    # --- Scoring TF-IDF leggero per rilevanza ---
    relevance_scores: dict[int, float] = {}
    if reference_text and len(chunks) > 15:
        try:
            from collections import Counter
            import math as _math

            def _tokenize(text: str) -> list[str]:
                return [w.lower() for w in text.split() if len(w) > 2]

            ref_tokens = _tokenize(reference_text)
            ref_counter = Counter(ref_tokens)

            for idx, chunk in enumerate(chunks):
                chunk_tokens = _tokenize(chunk.text)
                if not chunk_tokens:
                    relevance_scores[idx] = 0.0
                    continue
                chunk_counter = Counter(chunk_tokens)
                # Cosine-like overlap con IDF weighting
                shared = set(ref_counter) & set(chunk_counter)
                if not shared:
                    relevance_scores[idx] = 0.0
                    continue
                score = sum(
                    ref_counter[w] * chunk_counter[w]
                    for w in shared
                ) / (_math.sqrt(sum(v**2 for v in ref_counter.values()))
                     * _math.sqrt(sum(v**2 for v in chunk_counter.values())))
                relevance_scores[idx] = score
        except Exception:
            relevance_scores = {}

    # --- Sort: priority, poi rilevanza (decrescente), poi posizione ---
    def sort_key(item: tuple[int, PaperChunk]) -> tuple:
        idx, c = item
        priority = _section_priority(c.section_name or "")
        rel = -(relevance_scores.get(idx, 0.5))  # negato per desc
        return (priority, rel, c.chunk_index)

    indexed_chunks = sorted(enumerate(chunks), key=sort_key)

    parts: list[str] = []
    total_chars = 0
    current_section = None

    for _idx, chunk in indexed_chunks:
        # Header di sezione
        section = chunk.section_name or "Unknown Section"
        if section != current_section:
            header = f"\n### [{section}]\n"
            if total_chars + len(header) > max_chars:
                break
            parts.append(header)
            total_chars += len(header)
            current_section = section

        # Testo del chunk
        text = chunk.text.strip()
        if not text:
            continue

        if total_chars + len(text) + 2 > max_chars:
            # Tronca l'ultimo chunk se necessario
            remaining = max_chars - total_chars - 10
            if remaining > 200:
                parts.append(text[:remaining] + "\n[...truncated]")
                total_chars += remaining + 15
            break

        parts.append(text)
        total_chars += len(text) + 1

    result = "\n".join(parts)

    # Log stats
    n_included = sum(1 for p in parts if not p.startswith("\n### ["))
    logger.debug(
        f"Chunks preparati: {n_included}/{len(chunks)} inclusi, "
        f"{total_chars}/{max_chars} chars"
        + (f", relevance filter attivo" if relevance_scores else "")
    )

    return result


def _parse_json_response(raw_response: str, paper_title: str) -> dict:
    """Estrae e parsa il JSON dalla risposta LLM.

    Gestisce varie forme di output:
    - JSON puro
    - JSON dentro code fence ```json ... ```
    - JSON con testo attorno

    Args:
        raw_response: Testo grezzo dalla risposta LLM.
        paper_title: Titolo del paper (per logging).

    Returns:
        Dizionario parsato.

    Raises:
        ValueError: Se il JSON non può essere estratto.
    """
    text = raw_response.strip()

    # Tentativi in ordine di probabilità
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
                f"JSON riparato per '{paper_title[:40]}...' "
                f"(trailing commas o commenti rimossi)"
            )
            return parsed
    except json.JSONDecodeError:
        pass

    # 5. Fallback: estrai campi con regex
    logger.warning(
        f"JSON non parsabile per '{paper_title[:40]}...'. "
        f"Tentativo estrazione regex. Response: {text[:200]}..."
    )
    return _regex_fallback_extraction(text)


def _regex_fallback_extraction(text: str) -> dict:
    """Fallback: estrai i campi principali con regex dal testo.

    Usato quando il JSON non è parsabile. Tenta di estrarre
    almeno i campi più importanti.

    Args:
        text: Testo della risposta LLM.

    Returns:
        Dict con i campi estratti (potenzialmente incompleto).
    """
    result: dict[str, Any] = {}

    # testing_methodology (campo più importante)
    for pattern in [
        r'"testing_methodology"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]',
        r'testing_methodology[:\s]+(.*?)(?:\n\n|\Z)',
    ]:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            result["testing_methodology"] = m.group(1).strip().replace('\\"', '"')
            break

    # methodology
    for pattern in [
        r'"methodology"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]',
    ]:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            result["methodology"] = m.group(1).strip().replace('\\"', '"')
            break

    # key_contributions (array)
    m = re.search(
        r'"key_contributions"\s*:\s*\[(.*?)\]',
        text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        result["key_contributions"] = items[:5]

    # Campi stringa semplici
    for field in ["temporal_context", "results_summary", "limitations"]:
        m = re.search(
            rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}}]',
            text, re.DOTALL | re.IGNORECASE,
        )
        if m:
            result[field] = m.group(1).strip().replace('\\"', '"')

    if result:
        logger.info(f"Regex fallback: estratti {len(result)} campi")
    else:
        logger.error("Regex fallback: nessun campo estratto")

    return result


def _validate_l1_summary(summary: L1Summary, paper_title: str) -> None:
    """Valida la qualità del summary L1 generato.

    Emette warning per campi vuoti o sospetti, in particolare
    per ``testing_methodology`` che è il campo critico.

    Args:
        summary: L1Summary da validare.
        paper_title: Titolo del paper (per logging).
    """
    short_title = paper_title[:50]
    issues: list[str] = []

    if not summary.testing_methodology or len(summary.testing_methodology) < 30:
        issues.append(
            "testing_methodology vuoto o troppo breve "
            f"({len(summary.testing_methodology)} chars)"
        )

    if not summary.key_contributions:
        issues.append("key_contributions vuoto")

    if not summary.methodology or len(summary.methodology) < 30:
        issues.append(
            "methodology vuoto o troppo breve "
            f"({len(summary.methodology)} chars)"
        )

    if not summary.results_summary or len(summary.results_summary) < 20:
        issues.append(
            "results_summary vuoto o troppo breve "
            f"({len(summary.results_summary)} chars)"
        )

    if not summary.temporal_context:
        issues.append("temporal_context vuoto")

    if issues:
        for issue in issues:
            print_warning(f"L1 '{short_title}...': {issue}")
        logger.warning(
            f"L1 summary per '{short_title}...' ha {len(issues)} issue di qualità"
        )
    else:
        logger.debug(f"L1 summary per '{short_title}...' passa la validazione")


def _l1_summary_to_text(summary: L1Summary, paper: Paper) -> str:
    """Converte un L1Summary in testo per l'embedding nel vector store.

    Produce un testo sintetico che cattura tutti i campi del summary
    in forma narrativa, ottimale per il retrieval vettoriale.

    Args:
        summary: L1Summary.
        paper: Paper associato.

    Returns:
        Testo del summary per embedding.
    """
    parts = [
        f"Paper: {paper.title}",
        f"Authors: {', '.join(paper.authors[:5])}" if paper.authors else "",
        f"Year: {paper.year}" if paper.year else "",
        f"Venue: {paper.venue or paper.journal or 'N/A'}",
        f"Category: {summary.category or 'N/A'}",
        "",
        "Key Contributions:",
    ]

    for i, contrib in enumerate(summary.key_contributions, 1):
        parts.append(f"  {i}. {contrib}")

    parts.extend([
        "",
        f"Methodology: {summary.methodology}",
        "",
        f"Testing Methodology: {summary.testing_methodology}",
        "",
        f"Temporal Context: {summary.temporal_context}",
        "",
        f"Results: {summary.results_summary}",
        "",
        f"Limitations: {summary.limitations}",
    ])

    # v3.6 — Atomic claims (B4): rendere visibili al retrieval E al critic
    if summary.claims:
        parts.extend(["", "Atomic Claims (fact-checkable):"])
        for i, claim in enumerate(summary.claims, 1):
            ctype = getattr(claim, "claim_type", "") or ""
            evidence = getattr(claim, "evidence_chunk_ids", []) or []
            ev_str = f" [evidence: {', '.join(evidence[:3])}]" if evidence else ""
            type_str = f" ({ctype})" if ctype else ""
            parts.append(f"  C{i}{type_str}: {claim.text}{ev_str}")

    # v3.6 — Tassonomia: assi semantici espliciti per il retrieval
    tax_lines = []
    if summary.threat_model:
        tax_lines.append(f"Threat Model: {summary.threat_model}")
    if summary.attack_vector:
        tax_lines.append(f"Attack Vector: {', '.join(summary.attack_vector)}")
    if summary.defense_type:
        tax_lines.append(f"Defense Type: {', '.join(summary.defense_type)}")
    if summary.target_models:
        tax_lines.append(f"Target Models: {', '.join(summary.target_models)}")
    if summary.evaluation_metrics:
        tax_lines.append(f"Evaluation Metrics: {', '.join(summary.evaluation_metrics)}")
    if summary.datasets:
        tax_lines.append(f"Datasets: {', '.join(summary.datasets)}")
    if tax_lines:
        parts.append("")
        parts.extend(tax_lines)

    return "\n".join(p for p in parts if p is not None)


def _make_l1_chunk_id(paper_id: str) -> str:
    """Genera un chunk_id deterministico per un L1 summary.

    Args:
        paper_id: ID del paper (bibtex_key).

    Returns:
        Chunk ID di 16 caratteri hex.
    """
    raw = f"l1_summary::{paper_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _print_batch_report(
    total: int,
    generated: int,
    skipped: int,
    errors: int,
    error_details: dict[str, str],
    compression_tier: str = "standard",
    avg_l1_chars: int = 0,
    total_l1_chars: int = 0,
    target_chars: int = 0,
) -> None:
    """Stampa report riassuntivo del batch L1 con statistiche budget.

    Args:
        total: Numero totale di paper nel batch.
        generated: Numero di L1 generati con successo.
        skipped: Paper saltati (già esistenti).
        errors: Numero di errori.
        error_details: Dict paper_id → messaggio errore.
        compression_tier: Tier di compressione usato.
        avg_l1_chars: Media caratteri per L1.
        total_l1_chars: Totale caratteri di tutti gli L1.
        target_chars: Target caratteri per L1 (da budget).
    """
    report_data = {
        "Totale paper": total,
        "Generati": generated,
        "Saltati (già esistenti)": skipped,
        "Errori": errors,
        "Compression tier": compression_tier,
    }
    if generated > 0:
        report_data["Avg chars/L1"] = f"{avg_l1_chars:,}"
        report_data["Target chars/L1"] = f"{target_chars:,}"
        report_data["Totale chars L1"] = f"{total_l1_chars:,}"
        if target_chars > 0 and avg_l1_chars > 0:
            ratio = avg_l1_chars / target_chars
            status = "✅ OK" if ratio <= 1.2 else "⚠️ sopra target"
            report_data["Budget status"] = f"{ratio:.1%} del target ({status})"

    print_table("L1 Summary Batch Report", report_data)

    if error_details:
        print_warning("Errori durante la generazione:")
        for pid, err in error_details.items():
            logger.error(f"  {pid}: {err}")

    if generated > 0:
        print_success(
            f"L1 batch completato: {generated}/{total} summary generati "
            f"(tier={compression_tier}, avg={avg_l1_chars} chars)"
        )


# ============================================================================
# Utility: caricamento paper + chunk per testing
# ============================================================================

def load_paper_and_chunks_from_ingestion(
    pdf_path: str | Path,
    *,
    skip_s2: bool = False,
) -> tuple[Paper, list[PaperChunk]]:
    """Processa un singolo PDF attraverso la pipeline di ingestion
    e ritorna il Paper + i chunk, senza salvare nel registry/store.

    Utility per testing e uso standalone del summarizer.

    Args:
        pdf_path: Percorso al file PDF.
        skip_s2: Se True, salta l'enrichment Semantic Scholar.

    Returns:
        Tupla ``(Paper, list[PaperChunk])``.

    Raises:
        FileNotFoundError: Se il PDF non esiste.
        RuntimeError: Se GROBID fallisce.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF non trovato: {pdf_path}")

    from paperhunter.ingestion.grobid_client import GrobidClient
    from paperhunter.ingestion.chunker import chunk_grobid_output
    from paperhunter.ingestion.pdf_hasher import compute_pdf_hash
    from paperhunter.ingestion.bibtex_generator import enrich_paper_bibtex

    # Step 1: GROBID
    grobid = GrobidClient()
    logger.info(f"Processing con GROBID: {pdf_path.name}")

    paper_id = pdf_path.stem
    grobid_output = grobid.process_pdf(str(pdf_path), paper_id=paper_id)

    # Step 2: Crea Paper
    from paperhunter.ingestion.pdf_hasher import _get_category_from_path

    pdf_hash = compute_pdf_hash(pdf_path)
    category = _get_category_from_path(pdf_path, Path(settings.ingestion.pdf_base_dir))

    paper = Paper(
        title=grobid_output.title or pdf_path.stem,
        authors=grobid_output.authors,
        abstract=grobid_output.abstract,
        url=f"file://{pdf_path}",
        source="arxiv",
        pdf_path=str(pdf_path),
        pdf_hash=pdf_hash,
        category=category,
        grobid_processed=grobid_output.success,
    )

    # Step 3: S2 enrichment (opzionale)
    if not skip_s2:
        try:
            from paperhunter.ingestion.semantic_scholar import SemanticScholarEnricher
            enricher = SemanticScholarEnricher()
            paper = enricher.enrich_paper(paper)
        except Exception as e:
            logger.warning(f"S2 enrichment fallito per '{paper.title[:40]}...': {e}")

    # Step 4: BibTeX
    paper = enrich_paper_bibtex(paper)

    # Step 5: Chunking
    chunks = chunk_grobid_output(grobid_output, paper)

    logger.info(
        f"Paper caricato: '{paper.title[:50]}...' — "
        f"{len(chunks)} chunk, bibtex_key={paper.bibtex_key}"
    )

    return paper, chunks
