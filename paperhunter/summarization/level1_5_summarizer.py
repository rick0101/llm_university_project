"""
Level 1.5 Summarizer — Meta-summary intermedio per cluster RAPTOR.

Genera un meta-summary per ogni cluster semantico di L1 Summary (~20 paper
ciascuno), riducendo da ~1.000 L1 a ~50 cluster summary. Questo livello
intermedio permette all'L2 di rientrare nel budget di 150k token anche
con 1.000+ paper per categoria.

Pipeline::

    L1 Summary (1.000+)
        ↓ clustering_utils.py (AgglomerativeClustering)
    Cluster semantici (~50)
        ↓ level1_5_summarizer.py (questo modulo)
    L1.5 Summary (~50)
        ↓ level2_summarizer.py
    L2 Summary (1 per categoria)

Il summary L1.5 viene salvato nella collection Qdrant ``l1_5_summaries``
per il retrieval e per l'input all'L2.

Uso::

    from paperhunter.summarization.level1_5_summarizer import (
        generate_cluster_summary,
        generate_all_cluster_summaries,
    )
    from paperhunter.models import L1Summary

    # clusters: dict[int, list[L1Summary]] (output di clustering_utils)
    l1_5_summaries = generate_all_cluster_summaries(clusters, category="Prompt Injection")

    # Singolo cluster
    l1_5 = generate_cluster_summary(cluster_id=0, summaries=cluster_0_list, category="...")
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.models import ChunkType, L1Summary, L1_5Summary, PaperChunk
from paperhunter.utils.llm_factory import (
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
# Prompt template per L1.5
# ============================================================================

_L1_5_PROMPT_TEMPLATE = """\
You are an expert academic researcher specializing in LLM security. \
You are given a **cluster of {paper_count} related research papers** from the \
category "{category}". Each paper has been individually summarized (L1 summary). \
Your task is to synthesize these summaries into a coherent **cluster meta-summary**.

## Individual Paper Summaries (L1)

{l1_summaries_text}

---

## Instructions

Analyze ALL the L1 summaries above and produce a **cohesive meta-summary** \
for this cluster of papers. Focus on identifying:

1. **Common Theme**: What unifying topic or research question connects these papers?
2. **Shared Methodologies**: What approaches, techniques, or frameworks are shared \
   across papers? Where do they diverge?
3. **Key Findings**: What are the most important aggregated results? \
   Look for converging evidence across papers.
4. **Testing Overview**: Summarize the testing methodologies used across the cluster: \
   common benchmarks, datasets, metrics, models tested.

IMPORTANT RULES:
- Always cite individual papers using their **[bibtex_key]** in square brackets \
  (e.g., [Zou_2023_GCG], [Liu_2024_Jailbreak]).
- Every claim must be traceable to at least one [bibtex_key].
- Do NOT invent information not present in the L1 summaries.
- Identify both **convergences** (what papers agree on) and **divergences** \
  (where papers disagree or take different approaches).
- The output should be a coherent narrative, not a list of individual paper summaries.

Respond with a valid JSON object (no markdown fences, no comments):

{{
  "common_theme": "A 2-4 sentence description of the unifying theme of this cluster. \
Mention the key research question or problem space that connects all papers.",
  "shared_methodologies": "A 3-5 sentence description of shared and divergent \
methodological approaches. Cite [bibtex_keys] for each approach mentioned.",
  "key_findings": "A 4-8 sentence synthesis of the most important results across \
the cluster. Include quantitative results where available. Cite [bibtex_keys].",
  "testing_overview": "A 2-4 sentence overview of testing setups: common benchmarks, \
datasets, metrics (ASR, F1, etc.), and models tested. Cite [bibtex_keys]."
}}
"""


# ============================================================================
# Formatting L1 summaries for the prompt
# ============================================================================

def _format_l1_for_cluster_prompt(summaries: list[L1Summary]) -> str:
    """Formatta le L1 Summary di un cluster per il prompt LLM.

    Ogni L1 viene formattato come blocco strutturato con bibtex_key,
    year, e i campi principali.

    Args:
        summaries: Lista di L1 Summary del cluster.

    Returns:
        Testo formattato per il prompt.
    """
    parts: list[str] = []

    # Ordina per anno (più recenti prima)
    sorted_summaries = sorted(
        summaries,
        key=lambda s: s.year or 0,
        reverse=True,
    )

    for i, s in enumerate(sorted_summaries, 1):
        bibtex = s.bibtex_key or s.paper_id
        year = s.year or "N/A"
        contributions = "; ".join(s.key_contributions) if s.key_contributions else "N/A"

        block = (
            f"### Paper {i}: [{bibtex}] (Year: {year})\n"
            f"- **Key Contributions:** {contributions}\n"
            f"- **Methodology:** {s.methodology or 'N/A'}\n"
            f"- **Testing:** {s.testing_methodology or 'N/A'}\n"
            f"- **Results:** {s.results_summary or 'N/A'}\n"
            f"- **Temporal Context:** {s.temporal_context or 'N/A'}\n"
            f"- **Limitations:** {s.limitations or 'N/A'}\n"
        )
        parts.append(block)

    return "\n".join(parts)


def _compute_temporal_span(summaries: list[L1Summary]) -> str:
    """Calcola il range temporale dei paper nel cluster.

    Args:
        summaries: Lista di L1 Summary.

    Returns:
        Stringa con il range temporale (es. "2022-2025") o "N/A".
    """
    years = [s.year for s in summaries if s.year]
    if not years:
        return "N/A"
    min_year = min(years)
    max_year = max(years)
    if min_year == max_year:
        return str(min_year)
    return f"{min_year}-{max_year}"


def _extract_paper_ids(summaries: list[L1Summary]) -> list[str]:
    """Estrae i paper_id (o bibtex_key) da una lista di L1 Summary.

    Args:
        summaries: Lista di L1 Summary.

    Returns:
        Lista di identificativi paper.
    """
    ids: list[str] = []
    for s in summaries:
        pid = s.bibtex_key or s.paper_id
        if pid:
            ids.append(pid)
    return ids


# ============================================================================
# JSON parsing
# ============================================================================

def _parse_l1_5_json_response(raw_response: str, cluster_id: int) -> dict:
    """Estrae e parsa il JSON dalla risposta LLM per un L1.5 summary.

    Gestisce varie forme di output:
    - JSON puro
    - JSON dentro code fence ```json ... ```
    - JSON con testo attorno

    Args:
        raw_response: Testo grezzo dalla risposta LLM.
        cluster_id: ID del cluster (per logging).

    Returns:
        Dizionario parsato.

    Raises:
        ValueError: Se il JSON non può essere estratto.
    """
    text = raw_response.strip()

    # Tentativo 1: JSON puro
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tentativo 2: code fence ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Tentativo 3: trova il primo { ... } nel testo
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Impossibile estrarre JSON dalla risposta L1.5 per cluster {cluster_id}. "
        f"Risposta (primi 500 char): {text[:500]}"
    )


# ============================================================================
# Core: genera un singolo L1.5 summary
# ============================================================================

def generate_cluster_summary(
    cluster_id: int,
    summaries: list[L1Summary],
    category: str,
    *,
    force_model: str | None = None,
    save_to_store: bool = False,
    store: Any = None,
) -> L1_5Summary:
    """Genera un meta-summary L1.5 per un singolo cluster di L1 Summary.

    Invia al LLM (Gemini Pro) tutte le L1 del cluster e produce
    un summary coeso con tema comune, metodologie condivise,
    risultati chiave e panoramica testing.

    Args:
        cluster_id: ID numerico del cluster.
        summaries: Lista di ``L1Summary`` appartenenti a questo cluster.
        category: Nome della categoria (es. ``"Prompt Injection"``).
        force_model: Override tier modello (``'flash'`` o ``'pro'``).
            Default: Gemini Pro (da config).
        save_to_store: Se True, salva il summary nella collection
            ``l1_5_summaries`` del Qdrant.
        store: Istanza ``QdrantStore`` opzionale.

    Returns:
        Istanza ``L1_5Summary`` con tutti i campi popolati.

    Raises:
        ValueError: Se la lista di summaries è vuota.
        Exception: Se la chiamata LLM fallisce dopo tutti i retry.
    """
    if not summaries:
        raise ValueError(f"Cluster {cluster_id}: lista di L1 Summary vuota")

    start_time = time.time()

    # Prepara il testo delle L1 per il prompt
    l1_text = _format_l1_for_cluster_prompt(summaries)
    paper_ids = _extract_paper_ids(summaries)
    temporal_span = _compute_temporal_span(summaries)

    # Costruisci il prompt
    prompt = _L1_5_PROMPT_TEMPLATE.format(
        paper_count=len(summaries),
        category=category,
        l1_summaries_text=l1_text,
    )

    # Seleziona il modello
    tier = force_model or "pro"
    if tier == "pro" and settings.summarization.use_pro_for_l2:
        # L1.5 usa lo stesso modello dell'L2 (Pro) poiché è una sintesi complessa
        llm = get_pro_llm()
    else:
        from paperhunter.utils.llm_factory import get_flash_llm
        llm = get_flash_llm()

    # Rate limiting
    limiter = get_gemini_limiter()
    limiter.wait()

    # Chiamata LLM con retry
    logger.info(
        f"Generazione L1.5 per cluster {cluster_id} "
        f"({len(summaries)} paper, categoria: {category})"
    )

    try:
        raw_response = llm_call_with_retry(
            llm,
            prompt,
            max_retries=settings.summarization.pro_max_retries,
            base_delay=settings.summarization.pro_base_delay,
        )
    except Exception as e:
        logger.error(
            f"Errore LLM per cluster {cluster_id}: {e}"
        )
        raise

    # Parse JSON dalla risposta
    parsed = _parse_l1_5_json_response(raw_response, cluster_id)

    elapsed = time.time() - start_time

    # Costruisci L1_5Summary
    summary = L1_5Summary(
        cluster_id=cluster_id,
        category=category,
        paper_ids=paper_ids,
        paper_count=len(summaries),
        common_theme=parsed.get("common_theme", ""),
        shared_methodologies=parsed.get("shared_methodologies", ""),
        key_findings=parsed.get("key_findings", ""),
        testing_overview=parsed.get("testing_overview", ""),
        temporal_span=temporal_span,
        generated_at=datetime.now().isoformat(),
        model_used=tier,
    )

    # Validazione qualità
    _validate_l1_5_summary(summary, cluster_id)

    # Salvataggio opzionale nel vector store
    if save_to_store:
        _save_l1_5_to_store(summary, store)

    logger.info(
        f"L1.5 summary generato per cluster {cluster_id} "
        f"({len(summaries)} paper, {elapsed:.1f}s, model={tier})"
    )

    return summary


# ============================================================================
# Batch processing: tutti i cluster
# ============================================================================

def generate_all_cluster_summaries(
    clusters: dict[int, list[L1Summary]],
    category: str,
    *,
    force_model: str | None = None,
    save_to_store: bool = True,
    store: Any = None,
    skip_existing: bool = True,
) -> list[L1_5Summary]:
    """Genera summary L1.5 per tutti i cluster di una categoria.

    Iterazione batch con progress bar Rich e rate limiting Gemini.

    Args:
        clusters: Dizionario ``{cluster_id: [L1Summary, ...]}``
            (output di ``clustering_utils.cluster_summaries``).
        category: Nome della categoria.
        force_model: Override tier modello.
        save_to_store: Se True, salva ogni L1.5 nella collection Qdrant.
        store: Istanza ``QdrantStore`` opzionale.
        skip_existing: Se True e ``store`` fornito, salta cluster
            che hanno già un L1.5 nel vector store.

    Returns:
        Lista di ``L1_5Summary`` generati con successo.
    """
    if not clusters:
        logger.warning("generate_all_cluster_summaries: nessun cluster fornito")
        return []

    # Inizializzazione store se necessario
    if save_to_store and store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    total_clusters = len(clusters)
    total_papers = sum(len(v) for v in clusters.values())

    logger.info(
        f"Generazione L1.5 summaries: {total_clusters} cluster, "
        f"{total_papers} paper totali, categoria: {category}"
    )

    results: list[L1_5Summary] = []
    errors: dict[int, str] = {}
    skipped = 0

    # Ordina i cluster per ID per consistenza
    sorted_cluster_ids = sorted(clusters.keys())

    for cluster_id in progress_bar(
        sorted_cluster_ids,
        description="L1.5 Summaries",
    ):
        summaries = clusters[cluster_id]

        # Skip se già presente nel vector store
        if skip_existing and store is not None:
            if _l1_5_exists_in_store(cluster_id, category, store):
                logger.debug(
                    f"L1.5 già esistente per cluster {cluster_id} "
                    f"(categoria: {category}), skip"
                )
                skipped += 1
                continue

        try:
            l1_5 = generate_cluster_summary(
                cluster_id=cluster_id,
                summaries=summaries,
                category=category,
                force_model=force_model,
                save_to_store=save_to_store,
                store=store,
            )
            results.append(l1_5)
        except Exception as e:
            logger.error(
                f"Errore L1.5 per cluster {cluster_id} ({len(summaries)} paper): {e}"
            )
            errors[cluster_id] = str(e)

    # Report finale
    _print_batch_report(
        total=total_clusters,
        generated=len(results),
        skipped=skipped,
        errors=len(errors),
        error_details=errors,
        category=category,
        total_papers=total_papers,
    )

    return results


# ============================================================================
# Salvataggio nel QdrantStore
# ============================================================================

def _make_l1_5_chunk_id(cluster_id: int, category: str) -> str:
    """Genera un chunk_id deterministico per un L1.5 summary.

    Args:
        cluster_id: ID del cluster.
        category: Categoria di appartenenza.

    Returns:
        Stringa hash di 16 caratteri.
    """
    raw = f"l1.5::{category}::{cluster_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _l1_5_summary_to_text(summary: L1_5Summary) -> str:
    """Converte un L1.5 summary in testo per l'embedding.

    Crea un testo coeso che cattura tutte le informazioni del
    cluster meta-summary, adatto per la vettorizzazione.

    Args:
        summary: L1_5Summary da convertire.

    Returns:
        Testo narrativo per l'embedding.
    """
    parts = [
        f"Cluster Meta-Summary — {summary.category} (Cluster {summary.cluster_id})",
        f"Papers: {summary.paper_count} ({summary.temporal_span})",
        f"Paper IDs: {', '.join(summary.paper_ids[:10])}",  # Max 10 per leggibilità
        "",
        f"Theme: {summary.common_theme}",
        "",
        f"Methodologies: {summary.shared_methodologies}",
        "",
        f"Key Findings: {summary.key_findings}",
        "",
        f"Testing: {summary.testing_overview}",
    ]
    return "\n".join(parts)


def _save_l1_5_to_store(
    summary: L1_5Summary,
    store: Any = None,
) -> bool:
    """Salva un L1.5 summary come chunk nella collection l1_5_summaries.

    Il summary viene convertito in un ``PaperChunk`` con
    ``chunk_type=L1_SUMMARY`` e indicizzato nel vector store con
    metadata specifici per il cluster.

    Args:
        summary: L1_5Summary da salvare.
        store: QdrantStore. Se None, ne crea una nuova.

    Returns:
        True se il salvataggio è andato a buon fine.
    """
    try:
        if store is None:
            from paperhunter.indexing.qdrant_store import QdrantStore
            store = QdrantStore()

        # Componi testo per embedding
        summary_text = _l1_5_summary_to_text(summary)

        # Chunk ID deterministico
        chunk_id = _make_l1_5_chunk_id(summary.cluster_id, summary.category)

        # Crea PaperChunk (usiamo L1_SUMMARY come tipo più vicino)
        chunk = PaperChunk(
            chunk_id=chunk_id,
            text=summary_text,
            chunk_type=ChunkType.L1_SUMMARY,
            paper_id=f"cluster_{summary.cluster_id}",
            bibtex_key=f"cluster_{summary.cluster_id}",
            section_name="l1_5_summary",
            category=summary.category,
            year=_extract_max_year(summary.temporal_span),
            chunk_index=0,
        )

        # Salva nella collection l1_5_summaries
        collection = settings.qdrant.collection_l1_5_summaries
        n = store.add_chunks(
            [chunk],
            collection_name=collection,
            show_progress=False,
        )

        if n > 0:
            logger.debug(
                f"L1.5 salvato in Qdrant per cluster {summary.cluster_id} "
                f"(categoria: {summary.category})"
            )
            return True
        else:
            logger.warning(
                f"L1.5 non salvato per cluster {summary.cluster_id}"
            )
            return False

    except Exception as e:
        logger.error(
            f"Errore salvataggio L1.5 per cluster {summary.cluster_id}: {e}"
        )
        return False


def _l1_5_exists_in_store(
    cluster_id: int,
    category: str,
    store: Any,
) -> bool:
    """Verifica se un L1.5 summary esiste già nel vector store.

    Args:
        cluster_id: ID del cluster.
        category: Categoria.
        store: QdrantStore.

    Returns:
        True se il L1.5 è già presente.
    """
    try:
        collection = settings.qdrant.collection_l1_5_summaries
        paper_id = f"cluster_{cluster_id}"
        return store.has_paper(paper_id, collection_name=collection)
    except Exception:
        return False


# ============================================================================
# Helpers
# ============================================================================

def _extract_max_year(temporal_span: str) -> int | None:
    """Estrae l'anno massimo da un temporal_span (es. "2022-2025" → 2025).

    Args:
        temporal_span: Stringa con range temporale.

    Returns:
        Anno massimo come int, o None se non estraibile.
    """
    years = re.findall(r"(\d{4})", temporal_span)
    if years:
        return max(int(y) for y in years)
    return None


def _validate_l1_5_summary(summary: L1_5Summary, cluster_id: int) -> None:
    """Validazione qualità di un L1.5 summary.

    Controlla che i campi principali siano non vuoti e che contengano
    citazioni [bibtex_key].

    Args:
        summary: L1_5Summary da validare.
        cluster_id: ID del cluster (per logging).
    """
    warnings: list[str] = []

    if not summary.common_theme or len(summary.common_theme.strip()) < 20:
        warnings.append("common_theme troppo corto o vuoto")

    if not summary.shared_methodologies or len(summary.shared_methodologies.strip()) < 20:
        warnings.append("shared_methodologies troppo corto o vuoto")

    if not summary.key_findings or len(summary.key_findings.strip()) < 20:
        warnings.append("key_findings troppo corto o vuoto")

    if not summary.testing_overview or len(summary.testing_overview.strip()) < 10:
        warnings.append("testing_overview troppo corto o vuoto")

    # Verifica la presenza di almeno una citazione [bibtex_key]
    all_text = " ".join([
        summary.common_theme,
        summary.shared_methodologies,
        summary.key_findings,
        summary.testing_overview,
    ])
    citation_pattern = re.compile(r"\[[\w_]+\]")
    citations_found = citation_pattern.findall(all_text)

    if not citations_found:
        warnings.append("nessuna citazione [bibtex_key] trovata nel testo")

    if warnings:
        for w in warnings:
            print_warning(f"L1.5 cluster {cluster_id}: {w}")
        logger.warning(
            f"L1.5 cluster {cluster_id}: {len(warnings)} warning di qualità"
        )
    else:
        logger.debug(
            f"L1.5 cluster {cluster_id}: validazione OK "
            f"({len(citations_found)} citazioni trovate)"
        )


def _print_batch_report(
    total: int,
    generated: int,
    skipped: int,
    errors: int,
    error_details: dict[int, str],
    category: str,
    total_papers: int,
) -> None:
    """Stampa un report riassuntivo dopo il batch processing L1.5.

    Args:
        total: Numero totale di cluster.
        generated: Cluster generati con successo.
        skipped: Cluster saltati (già esistenti).
        errors: Numero di errori.
        error_details: Dizionario {cluster_id: errore}.
        category: Nome della categoria.
        total_papers: Numero totale di paper nei cluster.
    """
    print_success(
        f"L1.5 batch completato per '{category}': "
        f"{generated}/{total} generati, {skipped} skipped, {errors} errori "
        f"({total_papers} paper totali)"
    )

    if error_details:
        print_error(f"Errori L1.5 ({len(error_details)} cluster):")
        for cid, err in list(error_details.items())[:5]:
            print_error(f"  Cluster {cid}: {err[:100]}")
        if len(error_details) > 5:
            print_warning(f"  ... e altri {len(error_details) - 5} errori")

    # Tabella riepilogativa
    data = {
        "Categoria": category,
        "Cluster totali": str(total),
        "L1.5 generati": str(generated),
        "Cluster skipped": str(skipped),
        "Errori": str(errors),
        "Paper totali": str(total_papers),
        "Media paper/cluster": f"{total_papers / total:.1f}" if total > 0 else "N/A",
    }
    print_table(
        title="L1.5 Summary Batch Report",
        data=data,
    )


# ============================================================================
# Loading utilities
# ============================================================================

def load_l1_5_summaries_for_category(
    category: str,
    store: Any = None,
) -> list[L1_5Summary]:
    """Carica tutti gli L1.5 summary di una categoria dal Qdrant store.

    Esegue uno scroll sulla collection ``l1_5_summaries`` filtrando
    per categoria, e ricostruisce gli oggetti ``L1_5Summary`` dal payload.

    Args:
        category: Nome della categoria da caricare.
        store: Istanza ``QdrantStore``. Se None, ne crea una nuova.

    Returns:
        Lista di ``L1_5Summary`` ordinati per ``cluster_id``.
    """
    if store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    collection = settings.qdrant.collection_l1_5_summaries

    # Verifica che la collection esista
    if not store.client.collection_exists(collection):
        logger.info(f"Collection '{collection}' non esiste: nessun L1.5 disponibile")
        return []

    # Scroll tutti i punti della collection filtrando per categoria
    from qdrant_client import models as qdrant_models

    category_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="category",
                match=qdrant_models.MatchValue(value=category),
            )
        ]
    )

    all_points: list = []
    offset = None

    while True:
        try:
            points, next_offset = store.client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=category_filter,
            )
        except Exception as e:
            logger.error(f"Errore scroll L1.5 per '{category}': {e}")
            break

        all_points.extend(points)

        if next_offset is None:
            break
        offset = next_offset

    if not all_points:
        logger.info(f"Nessun L1.5 trovato per categoria '{category}'")
        return []

    # Ricostruisci L1_5Summary dal testo e payload
    summaries: list[L1_5Summary] = []
    for point in all_points:
        payload = point.payload or {}
        text = payload.get("text", "")

        # Estrai i campi dal testo strutturato
        l1_5 = _reconstruct_l1_5_from_payload(payload, text)
        if l1_5 is not None:
            summaries.append(l1_5)

    # Ordina per cluster_id
    summaries.sort(key=lambda s: s.cluster_id)

    logger.info(
        f"Caricati {len(summaries)} L1.5 summaries per '{category}'"
    )

    return summaries


def _reconstruct_l1_5_from_payload(
    payload: dict,
    text: str,
) -> L1_5Summary | None:
    """Ricostruisce un L1_5Summary dal payload Qdrant.

    Poiché il payload contiene i metadati del PaperChunk (non direttamente
    i campi L1_5Summary), ricostruiamo i campi dal testo memorizzato.

    Args:
        payload: Payload del punto Qdrant.
        text: Testo del chunk (generato da ``_l1_5_summary_to_text``).

    Returns:
        L1_5Summary ricostruito, o None se impossibile.
    """
    try:
        # Estrai cluster_id dal paper_id (formato: "cluster_{id}")
        paper_id = payload.get("paper_id", "")
        cluster_id_match = re.match(r"cluster_(\d+)", paper_id)
        if not cluster_id_match:
            return None
        cluster_id = int(cluster_id_match.group(1))

        category = payload.get("category", "")
        year = payload.get("year", None)

        # Parse del testo strutturato
        # Il testo ha formato:
        # Cluster Meta-Summary — {category} (Cluster {id})
        # Papers: {count} ({span})
        # Paper IDs: id1, id2, ...
        #
        # Theme: ...
        # Methodologies: ...
        # Key Findings: ...
        # Testing: ...

        sections = {
            "theme": "",
            "methodologies": "",
            "key_findings": "",
            "testing": "",
        }

        # Estrai paper_count e temporal_span dalla riga Papers
        paper_count = 0
        temporal_span = ""
        paper_ids: list[str] = []

        for line in text.split("\n"):
            line_stripped = line.strip()

            if line_stripped.startswith("Papers:"):
                papers_match = re.match(r"Papers:\s*(\d+)\s*\(([^)]+)\)", line_stripped)
                if papers_match:
                    paper_count = int(papers_match.group(1))
                    temporal_span = papers_match.group(2)
            elif line_stripped.startswith("Paper IDs:"):
                ids_str = line_stripped.replace("Paper IDs:", "").strip()
                paper_ids = [pid.strip() for pid in ids_str.split(",") if pid.strip()]
            elif line_stripped.startswith("Theme:"):
                sections["theme"] = line_stripped.replace("Theme:", "").strip()
            elif line_stripped.startswith("Methodologies:"):
                sections["methodologies"] = line_stripped.replace("Methodologies:", "").strip()
            elif line_stripped.startswith("Key Findings:"):
                sections["key_findings"] = line_stripped.replace("Key Findings:", "").strip()
            elif line_stripped.startswith("Testing:"):
                sections["testing"] = line_stripped.replace("Testing:", "").strip()

        return L1_5Summary(
            cluster_id=cluster_id,
            category=category,
            paper_ids=paper_ids,
            paper_count=paper_count,
            common_theme=sections["theme"],
            shared_methodologies=sections["methodologies"],
            key_findings=sections["key_findings"],
            testing_overview=sections["testing"],
            temporal_span=temporal_span,
            generated_at="",  # Non disponibile dal payload
            model_used=None,
        )

    except Exception as e:
        logger.warning(f"Impossibile ricostruire L1.5 dal payload: {e}")
        return None


def count_l1_5_for_category(
    category: str,
    store: Any = None,
) -> int:
    """Conta quanti L1.5 summary esistono per una data categoria.

    Utile per verificare rapidamente se la generazione L1.5 è necessaria.

    Args:
        category: Nome della categoria.
        store: Istanza ``QdrantStore``.

    Returns:
        Numero di L1.5 summary esistenti.
    """
    if store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    collection = settings.qdrant.collection_l1_5_summaries

    if not store.client.collection_exists(collection):
        return 0

    try:
        from qdrant_client import models as qdrant_models

        result = store.client.count(
            collection_name=collection,
            count_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="category",
                        match=qdrant_models.MatchValue(value=category),
                    )
                ]
            ),
        )
        return result.count
    except Exception as e:
        logger.error(f"Errore conteggio L1.5 per '{category}': {e}")
        return 0
