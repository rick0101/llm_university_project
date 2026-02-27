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

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.models import ChunkType, L1Summary, Paper, PaperChunk
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

## Instructions

Based ONLY on the paper content above, extract the following fields. \
Be precise, specific, and use the actual terminology from the paper. \
If a field cannot be determined from the content, write "Not explicitly stated in the paper."

Respond with a valid JSON object (no markdown fences, no comments):

{{
  "key_contributions": [
    "Contribution 1 (one concise sentence)",
    "Contribution 2",
    ...
  ],
  "methodology": "Detailed description of the technical approach: what model/system/attack/defense is proposed, how it works step by step, key algorithmic or architectural choices.",
  "testing_methodology": "CRITICAL FIELD — Extract ALL of the following if available: (1) benchmark datasets used (name, size, source), (2) evaluation metrics (exact names like ASR, BLEU, F1, etc.), (3) baseline methods compared against, (4) models tested on (GPT-4, LLaMA, etc.), (5) any open-source prompts, tools, or frameworks used for testing, (6) experimental setup details (hardware, API version, etc.). Be exhaustive.",
  "temporal_context": "When was this method proposed? How does it relate to prior work? What gap in the existing literature does it address? Position it in the timeline of the field (e.g., 'Proposed in 2023, building on [prior work], this was among the first to...').",
  "results_summary": "Key quantitative results: accuracy/success rates, performance numbers, comparison with baselines. Use actual numbers from the paper.",
  "limitations": "Limitations explicitly stated by the authors AND any implicit limitations you can identify from the methodology or experimental setup."
}}

IMPORTANT:
- key_contributions must have 1-5 entries maximum.
- testing_methodology is the MOST IMPORTANT field. Be extremely thorough.
- Use the paper's own terminology and citation keys where possible.
- Do NOT hallucinate or invent data not present in the source chunks.
- If the paper content is insufficient to fill a field, explicitly state so.
"""


# ============================================================================
# Core function
# ============================================================================

def generate_l1_summary(
    paper: Paper,
    chunks: list[PaperChunk],
    *,
    force_model: str | None = None,
    max_chunk_chars: int = 50_000,
    save_to_store: bool = False,
    store: Any = None,
) -> L1Summary:
    """Genera un summary L1 strutturato per un singolo paper.

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

    Returns:
        Istanza ``L1Summary`` con tutti i campi popolati.

    Raises:
        ValueError: Se non ci sono chunk da processare.
        RuntimeError: Se la chiamata LLM fallisce dopo tutti i retry.
    """
    if not chunks:
        raise ValueError(
            f"Nessun chunk fornito per il paper '{paper.title}'. "
            f"Esegui prima l'ingestion pipeline."
        )

    # --- Selezione modello ---
    pro_tracker = get_pro_tracker()
    if force_model:
        tier = force_model
    else:
        tier = "pro" if settings.summarization.use_pro_for_l1 else "flash"

    llm = get_pro_llm() if tier == "pro" else get_flash_llm()

    # --- Preparazione testo chunk ---
    chunks_text = _prepare_chunks_text(chunks, max_chars=max_chunk_chars)

    # --- Costruzione prompt ---
    prompt = _L1_PROMPT_TEMPLATE.format(
        title=paper.title or "Unknown",
        authors=", ".join(paper.authors[:10]) if paper.authors else "Unknown",
        year=paper.year or "Unknown",
        venue=paper.venue or paper.journal or "Unknown",
        category=paper.category or "Unknown",
        citations=paper.citations or "N/A",
        chunks_text=chunks_text,
    )

    # --- Chiamata LLM con rate limiting ---
    logger.info(
        f"Generazione L1 summary per '{paper.title[:60]}...' "
        f"(tier={tier}, {len(chunks)} chunk, {len(chunks_text)} chars)"
    )

    limiter = get_gemini_limiter()
    limiter.wait()

    # --- Retry strategy basata su config ---
    # Pro: retry pazienti con delay lunghi (15s base, cap 120s, 6 retry)
    # Flash: retry standard (3s base, cap 60s, 3 retry)
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

    summary = L1Summary(
        paper_id=paper_id,
        bibtex_key=paper.bibtex_key,
        category=paper.category,
        year=paper.year,
        key_contributions=parsed.get("key_contributions", [])[:5],
        methodology=parsed.get("methodology", ""),
        testing_methodology=parsed.get("testing_methodology", ""),
        temporal_context=parsed.get("temporal_context", ""),
        results_summary=parsed.get("results_summary", ""),
        limitations=parsed.get("limitations", ""),
        generated_at=datetime.now(),
        model_used=tier,
    )

    # --- Validazione qualità ---
    _validate_l1_summary(summary, paper.title)

    # --- Salvataggio opzionale nel vector store ---
    if save_to_store:
        _save_l1_to_store(summary, paper, store)

    logger.info(
        f"L1 summary generato per '{paper.title[:50]}...' "
        f"({elapsed:.1f}s, model={tier})"
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
) -> list[L1Summary]:
    """Genera summary L1 per un batch di paper.

    Args:
        papers_with_chunks: Lista di tuple ``(Paper, list[PaperChunk])``.
        force_model: Override tier modello.
        save_to_store: Salva ogni summary nel QdrantStore.
        store: Istanza ``QdrantStore``.
        skip_existing: Se True e ``store`` fornito, salta paper
            che hanno già un L1 nel vector store.

    Returns:
        Lista di ``L1Summary`` generati con successo.
    """
    if not papers_with_chunks:
        logger.warning("generate_l1_batch: lista vuota")
        return []

    # --- Inizializzazione store se necessario ---
    if save_to_store and store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    summaries: list[L1Summary] = []
    errors: dict[str, str] = {}
    skipped = 0

    logger.info(f"Generazione L1 batch per {len(papers_with_chunks)} paper...")

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
            )
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Errore L1 per '{paper.title[:50]}...': {e}")
            errors[paper_id] = str(e)

    # --- Report ---
    _print_batch_report(
        total=len(papers_with_chunks),
        generated=len(summaries),
        skipped=skipped,
        errors=len(errors),
        error_details=errors,
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

        # Crea PaperChunk
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
) -> str:
    """Prepara il testo dei chunk per il prompt LLM.

    Ordina i chunk per section_name e chunk_index, poi li concatena
    fino al limite di caratteri. Aggiunge header per sezione.

    Args:
        chunks: Lista di PaperChunk.
        max_chars: Limite massimo di caratteri.

    Returns:
        Testo formattato per il prompt.
    """
    # Ordina: prima per section_name (abstract prima), poi per chunk_index
    def sort_key(c: PaperChunk) -> tuple:
        section = c.section_name or ""
        # Abstract e Introduction prima di tutto
        priority = 2
        lower = section.lower()
        if "abstract" in lower:
            priority = 0
        elif "introduction" in lower or "intro" in lower:
            priority = 1
        return (priority, section, c.chunk_index)

    sorted_chunks = sorted(chunks, key=sort_key)

    parts: list[str] = []
    total_chars = 0
    current_section = None

    for chunk in sorted_chunks:
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
) -> None:
    """Stampa report riassuntivo del batch L1.

    Args:
        total: Numero totale di paper nel batch.
        generated: Numero di L1 generati con successo.
        skipped: Paper saltati (già esistenti).
        errors: Numero di errori.
        error_details: Dict paper_id → messaggio errore.
    """
    print_table("L1 Summary Batch Report", {
        "Totale paper": total,
        "Generati": generated,
        "Saltati (già esistenti)": skipped,
        "Errori": errors,
    })

    if error_details:
        print_warning("Errori durante la generazione:")
        for pid, err in error_details.items():
            logger.error(f"  {pid}: {err}")

    if generated > 0:
        print_success(
            f"L1 batch completato: {generated}/{total} summary generati"
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
