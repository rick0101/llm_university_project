"""
Critic Agent — Verifica la provenance delle citazioni nelle sezioni del survey.

Agent 3 nella pipeline agentica LangGraph. Per ogni sezione drafted:
1. Estrae tutte le citazioni [bibtex_key] dal testo
2. Per ogni citazione, recupera il chunk originale dal QdrantStore
3. Usa Gemini Flash per verificare il claim vs il chunk sorgente
4. Produce un VerificationResult aggregato per la sezione
5. Se > 20% citazioni non supportate → flag NEEDS_REVISION

Il Critic supporta max 3 iterazioni writer↔critic per sezione.
Dopo la terza iterazione, la sezione viene accettata comunque con warning.

Uso standalone::

    from paperhunter.agents.critic_agent import verify_section, verify_all_sections
    from paperhunter.agents.state import create_initial_state

    state = create_initial_state(...)
    # ... dopo che il Writer ha scritto le sezioni
    updated = verify_section(state, "prompt_injection_attacks")

Uso come nodo LangGraph::

    from langgraph.graph import StateGraph
    from paperhunter.agents.state import SurveyState
    from paperhunter.agents.critic_agent import verify

    graph = StateGraph(SurveyState)
    graph.add_node("verify", verify)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Sequence

from paperhunter.agents.state import (
    PipelinePhase,
    SectionDraft,
    SectionOutline,
    SectionStatus,
    SurveyState,
    VerificationDetail,
    VerificationResult,
    get_outline_from_state,
    get_sections_from_state,
    get_verifications_from_state,
)
from paperhunter.config import settings
from paperhunter.evolution.lesson_extractor import extract_from_verification
from paperhunter.evolution.lesson_store import LessonStore
from paperhunter.models import VerificationStatus
from paperhunter.utils.llm_factory import (
    get_llm,
    llm_call_with_retry,
)
from paperhunter.utils.prompt_loader import get_prompt
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

_VERIFICATION_PROMPT_TEMPLATE = get_prompt("VERIFICATION_PROMPT")

_BATCH_VERIFICATION_PROMPT_TEMPLATE = get_prompt("BATCH_VERIFICATION_PROMPT")

_FEEDBACK_GENERATION_PROMPT_TEMPLATE = get_prompt("FEEDBACK_GENERATION_PROMPT")


# ============================================================================
# Helper: Extract citations from section text
# ============================================================================

def _extract_citations_from_text(text: str) -> list[str]:
    """Estrae tutte le chiavi BibTeX citate nel testo con [bibtex_key].

    Riconosce pattern come [Author_YYYY_Keyword],
    [Author_YYYY_Multi_Word], ecc.

    Args:
        text: Testo della sezione.

    Returns:
        Lista di chiavi BibTeX uniche, nell'ordine di prima apparizione.
    """
    if not text:
        return []

    # Chiavi singole e gruppi: [Key_2024_A, Other_2025_B].
    matches: list[str] = []
    for group in re.findall(r"\[([^\]]+)\]", text):
        for raw_key in re.split(r"[,;]", group):
            key = raw_key.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", key):
                matches.append(key)

    # Dedup mantenendo ordine
    seen: set[str] = set()
    unique: list[str] = []
    for key in matches:
        # Filtra falsi positivi (es. [Table_1], [Figure_2], [Section_3])
        if re.match(r"^(Table|Figure|Section|Appendix|Algorithm|Equation|Step)_\d+$", key):
            continue
        if key not in seen:
            seen.add(key)
            unique.append(key)

    return unique


def _extract_claims_for_citation(
    text: str,
    bibtex_key: str,
    context_chars: int = 300,
) -> list[str]:
    """Estrae i claim che citano un dato bibtex_key dal testo.

    Per ogni occorrenza di [bibtex_key], estrae la frase o il
    contesto circostante come claim.

    Args:
        text: Testo della sezione.
        bibtex_key: Chiave BibTeX da cercare.
        context_chars: Caratteri di contesto da estrarre.

    Returns:
        Lista di claim (frasi che citano il bibtex_key).
    """
    if not text or not bibtex_key:
        return [f"(Citation [{bibtex_key}] used in the section)"]

    claims: list[str] = []
    citation_groups = [
        match
        for match in re.finditer(r"\[([^\]]+)\]", text)
        if bibtex_key in {
            key.strip() for key in re.split(r"[,;]", match.group(1))
        }
    ]

    for match in citation_groups:
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        context = text[start:end].strip()

        # Cerca di estrarre la frase completa
        sentences = re.split(r'(?<=[.!?])\s+', context)
        for sentence in sentences:
            if bibtex_key in sentence:
                claims.append(sentence.strip())
                break
        else:
            # Se non troviamo una frase contenente la citazione, usa il contesto
            claims.append(context)

    return claims if claims else [f"(Citation [{bibtex_key}] used in the section)"]


def _extract_citation_occurrences(text: str, context_chars: int = 300) -> list[dict[str, str]]:
    """Return every claim-source occurrence in textual order.

    ``_extract_citations_from_text`` intentionally deduplicates keys for
    bibliography coverage.  Citation verification must not: the same paper can
    support one sentence and fail to support a later one.  Each adjacent claim
    is therefore an independent verification unit.
    """
    if not text:
        return []
    occurrences: list[dict[str, str]] = []
    false_positive = re.compile(
        r"^(Table|Figure|Section|Appendix|Algorithm|Equation|Step)_\d+$"
    )
    for match in re.finditer(r"\[([^\]]+)\]", text):
        keys = [key.strip() for key in re.split(r"[,;]", match.group(1))]
        keys = [
            key for key in keys
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", key)
            and not false_positive.match(key)
        ]
        if not keys:
            continue

        # Slice the sentence by offsets rather than searching for the citation
        # string: the same key can appear in adjacent sentences.
        sentence_start = 0
        for separator in (". ", "! ", "? ", "\n"):
            pos = text.rfind(separator, 0, match.start())
            if pos >= 0:
                sentence_start = max(sentence_start, pos + len(separator))
        sentence_end = len(text)
        ending = re.search(r"[.!?](?=\s|$)", text[match.end():])
        if ending:
            sentence_end = match.end() + ending.end()
        sentence = text[sentence_start:sentence_end].strip()
        for key in keys:
            occurrences.append({"bibtex_key": key, "claim": sentence})
    return occurrences


def _check_section_length(
    draft: SectionDraft,
    outline: SectionOutline | None,
    *,
    strict: bool,
) -> tuple[bool, str]:
    """Deterministic completeness gate based on the declared outline target."""
    if not strict:
        return True, ""
    actual = len((draft.text or "").split())
    target = outline.target_length_words if outline else settings.agents.min_section_words
    required = max(settings.agents.min_section_words, int(target * 0.80))
    if actual >= required:
        return True, ""
    return False, (
        f"Section is too short or truncated: {actual} words; minimum {required} "
        f"words (80% of the {target}-word outline target)."
    )


def _build_strict_supported_only_fallback(
    draft: SectionDraft,
    details: Sequence[VerificationDetail | dict],
    outline: SectionOutline | None,
    *,
    allow_incomplete_length: bool = False,
) -> tuple[SectionDraft, list[VerificationDetail]] | None:
    """Prune failed cited sentences at the final attempt, without inventing text.

    This fallback is deliberately fail-closed.  It succeeds only when every
    failed claim can be removed exactly, every remaining citation occurrence has
    an already-SUPPORTED detail, source diversity is still sufficient, and the
    pruned draft remains above the strict length floor.
    """
    normalized_details = [
        detail
        if isinstance(detail, VerificationDetail)
        else VerificationDetail.from_dict(detail)
        for detail in details
    ]
    failed_claims: list[str] = []
    for detail in normalized_details:
        if detail.status != VerificationStatus.SUPPORTED:
            claim = (detail.claim_text or "").strip()
            if claim:
                failed_claims.append(claim)
    if not failed_claims:
        return None

    pruned_text = draft.text
    for claim in failed_claims:
        if claim not in pruned_text:
            # The detail can be stale after a preserved/reducer-merged state.
            # Remaining occurrences are validated below against a bounded
            # multiset of SUPPORTED occurrence records, so skipping it cannot
            # retain an unsupported citation.
            continue
        pruned_text = pruned_text.replace(claim, "", 1)
    pruned_text = re.sub(r"(?<=\S)[ \t]{2,}", " ", pruned_text)
    pruned_text = re.sub(r"[ \t]+\n", "\n", pruned_text)
    pruned_text = re.sub(r"\n{3,}", "\n\n", pruned_text).strip()

    supported_by_occurrence: dict[tuple[str, str], list[VerificationDetail]] = {}
    for detail in normalized_details:
        if detail.status != VerificationStatus.SUPPORTED:
            continue
        identity = (
            detail.citation_key,
            re.sub(r"\s+", " ", detail.claim_text or "").strip().lower(),
        )
        supported_by_occurrence.setdefault(identity, []).append(detail)

    kept_details: list[VerificationDetail] = []
    for occurrence in _extract_citation_occurrences(pruned_text):
        identity = (
            occurrence["bibtex_key"],
            re.sub(r"\s+", " ", occurrence["claim"]).strip().lower(),
        )
        matching_details = supported_by_occurrence.get(identity)
        if not matching_details:
            return None
        kept_details.append(matching_details.pop(0))

    distinct_keys = {detail.citation_key for detail in kept_details}
    if len(distinct_keys) < settings.agents.min_citations_per_section:
        return None

    pruned = SectionDraft.from_dict(draft.to_dict())
    pruned.text = pruned_text
    pruned.word_count = len(pruned_text.split())
    length_ok, _ = _check_section_length(pruned, outline, strict=True)
    if not length_ok and not allow_incomplete_length:
        return None
    pruned.citations = [
        record for record in pruned.citations
        if record.bibtex_key in distinct_keys
    ]
    pruned.status = SectionStatus.DRAFTED
    note = "Strict final fallback pruned unsupported cited sentences."
    pruned.writer_notes = f"{pruned.writer_notes.strip()} {note}".strip()
    return pruned, kept_details


# ============================================================================
# Helper: Retrieve source chunks for a citation
# ============================================================================

def _retrieve_source_chunks(
    bibtex_key: str,
    store=None,
    max_chunks: int | None = None,
) -> list[dict]:
    """Retrieve all paper chunks before claim-specific ranking.

    Cerca nelle collezioni paper_chunks e l1_summaries.

    Args:
        bibtex_key: Chiave BibTeX del paper.
        store: QdrantStore instance (lazy-init se None).
        max_chunks: Optional emergency cap. ``None`` keeps all chunks so a
            relevant methods/results passage is not discarded before ranking.

    Returns:
        Lista di dict con chunk_id, text e metadata.
    """
    if store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    all_chunks: list[dict] = []

    # Cerca nella collezione paper_chunks
    try:
        chunks = store.get_paper_chunks(
            bibtex_key,
            collection_name=settings.qdrant.collection_paper_chunks,
        )
        all_chunks.extend(chunks)
    except Exception as e:
        logger.debug(f"Errore recupero chunk per '{bibtex_key}' da paper_chunks: {e}")

    # Cerca anche nella collezione l1_summaries (contiene summary strutturati)
    try:
        l1_chunks = store.get_paper_chunks(
            bibtex_key,
            collection_name=settings.qdrant.collection_l1_summaries,
        )
        all_chunks.extend(l1_chunks)
    except Exception as e:
        logger.debug(f"Errore recupero chunk per '{bibtex_key}' da l1_summaries: {e}")

    # Ordina per rilevanza: L1 summaries prima, poi chunk originali
    def _sort_key(c: dict) -> int:
        ct = str(c.get("metadata", {}).get("chunk_type", "original")).lower()
        if ct == "l1_summary":
            return 0
        section = c.get("metadata", {}).get("section_name", "").lower()
        if "abstract" in section:
            return 1
        if "introduction" in section or "intro" in section:
            return 2
        if "conclusion" in section:
            return 3
        if "result" in section:
            return 4
        return 5

    all_chunks.sort(key=_sort_key)
    return all_chunks if max_chunks is None else all_chunks[:max_chunks]


def _select_tracked_citation_record(
    records: list[dict],
    bibtex_key: str,
    claim: str,
) -> dict | None:
    """Match one citation occurrence to its writer-provided provenance record.

    A dict keyed only by ``bibtex_key`` collapsed repeated uses of the same paper
    to the final record.  Earlier occurrences were then verified against the wrong
    chunk.  Claim overlap retains occurrence-level provenance without trusting the
    Writer's support judgment.
    """
    candidates = [
        record for record in records
        if str(record.get("bibtex_key") or "") == bibtex_key
    ]
    if not candidates:
        return None

    claim_terms = set(re.findall(r"[a-z0-9]+", claim.lower()))

    def _score(record: dict) -> tuple[float, int]:
        tracked_claim = str(record.get("claim") or "")
        tracked_terms = set(re.findall(r"[a-z0-9]+", tracked_claim.lower()))
        union = claim_terms | tracked_terms
        similarity = len(claim_terms & tracked_terms) / max(1, len(union))
        return similarity, int(bool(record.get("chunk_id")))

    best = max(candidates, key=_score)
    if len(candidates) > 1 and _score(best)[0] == 0:
        return None
    return best


def _verification_cache_key(bibtex_key: str, claim: str) -> tuple[str, str]:
    """Create a stable key for a verbatim citation occurrence."""
    normalized_claim = re.sub(r"\s+", " ", claim).strip().lower()
    return bibtex_key, normalized_claim


def _build_supported_verification_cache(
    verification: VerificationResult | dict | None,
) -> dict[tuple[str, str], dict]:
    """Cache only prior SUPPORTED judgments for unchanged claim/source pairs.

    Failed claims are always rechecked after repair.  A supported sentence kept
    verbatim, however, has the same paper and evidence; asking a stochastic judge
    again can turn progress backwards without any source change.
    """
    if verification is None:
        return {}
    if isinstance(verification, VerificationResult):
        verification = verification.to_dict()
    cache: dict[tuple[str, str], dict] = {}
    for raw_detail in verification.get("details", []):
        detail = (
            raw_detail.to_dict()
            if isinstance(raw_detail, VerificationDetail)
            else dict(raw_detail)
        )
        status = str(detail.get("status") or "").split(".")[-1].upper()
        if status != VerificationStatus.SUPPORTED.value:
            continue
        key = str(detail.get("citation_key") or "")
        claim = str(detail.get("claim_text") or "")
        if key and claim:
            cache[_verification_cache_key(key, claim)] = detail
    return cache


def _select_source_chunks_for_claim(
    chunks: list[dict],
    claim: str,
    *,
    tracked_chunk_id: str | None = None,
    max_chunks: int = 6,
) -> list[dict]:
    """Choose paper passages relevant to this exact citation occurrence.

    Taking the first five chunks of a paper systematically favored abstract
    and introduction text, producing false negatives for method/result claims.
    Lexical claim matching is deterministic and keeps the verifier grounded in
    source text; a writer-tracked chunk receives highest priority.
    """
    if not chunks:
        return []
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "with",
        "is", "are", "was", "were", "this", "that", "these", "those", "paper",
    }
    claim_terms = {
        token for token in re.findall(r"[a-z0-9]+", claim.lower())
        if len(token) > 2 and token not in stopwords
    }

    def _score(chunk: dict) -> tuple[float, float]:
        text = str(chunk.get("text") or "")
        terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        overlap = len(claim_terms & terms) / max(1, len(claim_terms))
        chunk_id = str(chunk.get("chunk_id") or "")
        tracked_bonus = 10.0 if tracked_chunk_id and chunk_id == tracked_chunk_id else 0.0
        chunk_type = str((chunk.get("metadata") or {}).get("chunk_type", "")).lower()
        summary_bonus = 0.10 if chunk_type == "l1_summary" else 0.0
        return tracked_bonus + overlap + summary_bonus, float(len(text))

    return sorted(chunks, key=_score, reverse=True)[:max_chunks]


def _source_chunks_are_citable(chunks: list[dict]) -> bool:
    """Return True only when the source belongs to the DBLP-backed corpus."""
    return any(
        bool((chunk.get("metadata") or {}).get("is_citable", False))
        for chunk in chunks
    )


def _format_source_text(chunks: list[dict], max_chars: int = 4000) -> str:
    """Formatta i chunk sorgente in testo per il prompt di verifica.

    Args:
        chunks: Lista di chunk dict.
        max_chars: Limite caratteri totali.

    Returns:
        Testo formattato con separatori per ogni chunk.
    """
    if not chunks:
        return "(No source text found for this citation.)"

    parts: list[str] = []
    total = 0
    has_content = False

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        has_content = True

        meta = chunk.get("metadata", {})
        section = meta.get("section_name", "N/A")
        chunk_type = meta.get("chunk_type", "original")

        header = f"[Source {i + 1} — Section: {section}, Type: {chunk_type}]"
        entry = f"{header}\n{text}\n"

        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                entry = f"{header}\n{text[:remaining - len(header) - 20]}...\n"
                parts.append(entry)
            break

        parts.append(entry)
        total += len(entry)

    if not has_content:
        return "(No source text found for this citation.)"
    return "\n".join(parts)


# ============================================================================
# v3.6 — Atomic claim selection + holistic coherence (B4 + C4)
# ============================================================================

_ATOMIC_CLAIM_LINE_RE = re.compile(r"^\s*C\d+(?:\s*\([^)]+\))?:\s*(.+?)$", re.MULTILINE)


def _extract_atomic_claims_from_source(source_text: str) -> list[str]:
    """Estrae le righe di 'Atomic Claims' dal testo dell'L1 summary.

    Cerca pattern del tipo ``C1 (empirical): ... [evidence: ...]`` prodotti
    da ``_l1_summary_to_text`` in v3.6.
    """
    if not source_text or "Atomic Claims" not in source_text:
        return []
    out: list[str] = []
    for m in _ATOMIC_CLAIM_LINE_RE.finditer(source_text):
        line = m.group(1).strip()
        # Rimuovi suffisso "[evidence: ...]" se presente
        line = re.sub(r"\s*\[evidence:[^\]]*\]\s*$", "", line)
        if line:
            out.append(line)
    return out


def _select_best_atomic_claim(claim_text: str, candidates: list[str]) -> str | None:
    """Sceglie l'atomic claim più simile al claim del writer, via overlap di token."""
    if not claim_text or not candidates:
        return None
    claim_tokens = set(re.findall(r"\w+", claim_text.lower()))
    if not claim_tokens:
        return None
    best: tuple[float, str] | None = None
    for cand in candidates:
        cand_tokens = set(re.findall(r"\w+", cand.lower()))
        if not cand_tokens:
            continue
        overlap = len(claim_tokens & cand_tokens)
        score = overlap / max(1, len(cand_tokens | claim_tokens))
        if best is None or score > best[0]:
            best = (score, cand)
    if best and best[0] >= 0.20:
        return best[1]
    return None


def _augment_source_with_atomic_claim(
    source_text: str,
    claim_text: str,
) -> str:
    """Se il testo sorgente contiene atomic claims, ne mette in cima il più
    rilevante per il claim del writer. Aiuta il critic a fare verifica chirurgica.
    """
    candidates = _extract_atomic_claims_from_source(source_text)
    if not candidates:
        return source_text
    best = _select_best_atomic_claim(claim_text, candidates)
    if not best:
        return source_text
    header = (
        "[ATOMIC CLAIM MATCH — verify the writer's claim primarily against this "
        f"author-stated atomic fact]\n→ {best}\n\n"
    )
    return header + source_text


_COHERENCE_PROMPT = """You are a meticulous scientific editor performing a HOLISTIC coherence review of one section of an academic survey.

Section title: {section_title}
Outline focus: {section_focus}

Section text:
\"\"\"
{section_text}
\"\"\"

Check ONLY for the following high-impact issues:
1. Internal contradictions (the section contradicts itself).
2. Outline drift (the section talks about topics outside its declared focus).
3. Unsupported strong claims with no citation (sweeping statements lacking [bibtex_key]).
4. Logical incoherence between paragraphs.

Respond ONLY with a compact JSON object:
{{
  "passed": true|false,
  "issues": ["short bullet 1", "short bullet 2"],
  "suggested_revisions": "1-3 sentences of actionable advice for the writer, or empty string"
}}

If everything is fine, return passed=true with empty issues and empty suggested_revisions.
Do NOT comment on style, grammar, or word choice.
"""


def _run_coherence_pass(
    section_text: str,
    outline: SectionOutline | None,
    llm=None,
) -> dict | None:
    """Pass 2 olistico (Flash) — ritorna dict con passed/issues/suggested_revisions, o None."""
    if not section_text:
        return None
    if llm is None:
        llm = get_llm(
            "flash", max_tokens=2048, timeout_seconds=60, thinking_enabled=False,
        )
    section_title = outline.title if outline else "(unknown)"
    section_focus = (outline.description if outline and getattr(outline, "description", None) else section_title)
    prompt = _COHERENCE_PROMPT.format(
        section_title=section_title,
        section_focus=section_focus,
        section_text=section_text[:30000],
    )
    try:
        raw = llm_call_with_retry(llm, prompt, max_retries=1, base_delay=4.0, max_delay=20.0)
    except Exception as exc:
        logger.warning("Coherence pass fallita: %s", exc)
        return None

    # Estrai JSON
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    return {
        "passed": bool(data.get("passed", True)),
        "issues": list(data.get("issues") or [])[:5],
        "suggested_revisions": str(data.get("suggested_revisions") or "").strip(),
    }


# ============================================================================
# Helper: Parse LLM verification response
# ============================================================================

def _parse_verification_response(raw: str) -> dict | None:
    """Parse la risposta JSON del LLM di verifica.

    Gestisce code fences, trailing commas, testo extra.

    Args:
        raw: Risposta grezza del LLM.

    Returns:
        Dict con status, confidence, reasoning; o None.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Rimuovi code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Tentativo 1: JSON diretto
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "status" in data:
            return _normalize_verification(data)
    except json.JSONDecodeError:
        pass

    # Tentativo 2: estrai oggetto JSON
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "status" in data:
                return _normalize_verification(data)
        except json.JSONDecodeError:
            pass

    # Tentativo 3: fix trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "status" in data:
                return _normalize_verification(data)
        except json.JSONDecodeError:
            pass

    # Tentativo 4: regex fallback
    return _regex_parse_verification(raw)


def _parse_batch_verification_response(
    raw: str,
    expected_count: int,
) -> list[dict] | None:
    """Parse la risposta JSON array del batch verification.

    Args:
        raw: Risposta grezza del LLM.
        expected_count: Numero di verifiche attese.

    Returns:
        Lista di dict con status, confidence, reasoning; o None.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Tentativo 1: JSON diretto (array)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [_normalize_verification(d) for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass

    # Tentativo 2: estrai array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [_normalize_verification(d) for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

    # Tentativo 3: fix trailing commas + riprova
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [_normalize_verification(d) for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

    # Tentativo 4: cerca oggetti JSON individuali nella risposta
    objects = re.findall(r"\{[^{}]+\}", text)
    if objects:
        results = []
        for obj_str in objects:
            try:
                d = json.loads(obj_str)
                if isinstance(d, dict) and "status" in d:
                    results.append(_normalize_verification(d))
            except json.JSONDecodeError:
                continue
        if results:
            return results

    return None


def _normalize_verification(data: dict) -> dict:
    """Normalizza un risultato di verifica singolo.

    Mappa stringhe di status al VerificationStatus enum, clamp confidence
    a [0.0, 1.0], e assicura che reasoning sia una stringa.
    """
    status_raw = data.get("status", "SOURCE_NOT_FOUND").upper().strip()

    # Mappa varianti allo status canonico
    status_map = {
        "SUPPORTED": "SUPPORTED",
        "PARTIALLY_SUPPORTED": "PARTIALLY_SUPPORTED",
        "PARTIAL": "PARTIALLY_SUPPORTED",
        "PARTIALLY": "PARTIALLY_SUPPORTED",
        "NOT_SUPPORTED": "NOT_SUPPORTED",
        "UNSUPPORTED": "NOT_SUPPORTED",
        "NOT SUPPORTED": "NOT_SUPPORTED",
        "SOURCE_NOT_FOUND": "SOURCE_NOT_FOUND",
        "NOT_FOUND": "SOURCE_NOT_FOUND",
        "NOT FOUND": "SOURCE_NOT_FOUND",
        "MISSING": "SOURCE_NOT_FOUND",
    }
    status = status_map.get(status_raw, "SOURCE_NOT_FOUND")

    confidence = data.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(data.get("reasoning", "")).strip()

    return {
        "status": status,
        "confidence": confidence,
        "reasoning": reasoning,
        "bibtex_key": data.get("bibtex_key", ""),
        "citation_index": data.get("citation_index", -1),
    }


def _regex_parse_verification(raw: str) -> dict | None:
    """Fallback regex per parsare la risposta di verifica.

    Cerca pattern come: "status": "SUPPORTED"
    """
    if not raw:
        return None

    # Cerca status
    status_match = re.search(
        r'"?status"?\s*[:=]\s*"?(SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED|SOURCE_NOT_FOUND)"?',
        raw, re.IGNORECASE,
    )
    if not status_match:
        return None

    status = status_match.group(1).upper()

    # Cerca confidence
    conf_match = re.search(r'"?confidence"?\s*[:=]\s*([\d.]+)', raw)
    confidence = float(conf_match.group(1)) if conf_match else 0.0
    confidence = max(0.0, min(1.0, confidence))

    # Cerca reasoning
    reason_match = re.search(r'"?reasoning"?\s*[:=]\s*"([^"]*)"', raw)
    reasoning = reason_match.group(1) if reason_match else ""

    return {
        "status": status,
        "confidence": confidence,
        "reasoning": reasoning,
        "bibtex_key": "",
        "citation_index": -1,
    }


# ============================================================================
# Helper: Generate feedback for Writer
# ============================================================================

def _generate_critic_feedback(
    section_title: str,
    verification: VerificationResult,
) -> str:
    """Genera feedback strutturato per il Writer da un VerificationResult.

    Se la verifica è passata, ritorna stringa vuota (nessun feedback).
    Altrimenti, usa Gemini Flash per generare feedback azionabile.

    Args:
        section_title: Titolo della sezione.
        verification: Risultato della verifica.

    Returns:
        Testo del feedback, o stringa vuota se non necessario.
    """
    if verification.passed:
        return ""

    # Costruisci sommario delle verifiche
    summary_lines: list[str] = []
    for d in verification.details:
        status_icon = {
            VerificationStatus.SUPPORTED: "✓",
            VerificationStatus.PARTIALLY_SUPPORTED: "~",
            VerificationStatus.NOT_SUPPORTED: "✗",
            VerificationStatus.SOURCE_NOT_FOUND: "?",
        }.get(d.status, "?")

        summary_lines.append(
            f"  {status_icon} [{d.citation_key}]: {d.status.value} "
            f"(conf={d.confidence:.2f}) — {d.reasoning[:120]}"
        )

    verification_summary = "\n".join(summary_lines)

    prompt = _FEEDBACK_GENERATION_PROMPT_TEMPLATE.format(
        title=section_title,
        verification_summary=verification_summary,
        total_citations=verification.total_citations,
        supported=verification.supported,
        partially_supported=verification.partially_supported,
        not_supported=verification.not_supported,
        source_not_found=verification.source_not_found,
        pass_rate=verification.pass_rate,
    )

    try:
        llm = get_llm(
            "flash", max_tokens=2048, timeout_seconds=60, thinking_enabled=False,
        )
        feedback = llm_call_with_retry(
            llm, prompt, max_retries=2, base_delay=5.0, max_delay=30.0,
        )
        feedback = feedback.strip()
        logger.info(f"Feedback generato per '{section_title}': {len(feedback)} chars")
        return feedback
    except Exception as e:
        logger.warning(f"Errore generazione feedback per '{section_title}': {e}")
        # Fallback: feedback basato sulle verifiche fallite
        failed = [
            d for d in verification.details
            if d.status in (VerificationStatus.NOT_SUPPORTED, VerificationStatus.SOURCE_NOT_FOUND)
        ]
        lines = [
            f"The following citations failed verification and need to be "
            f"corrected or replaced:\n"
        ]
        for d in failed:
            lines.append(
                f"- [{d.citation_key}]: {d.status.value}. "
                f"{d.reasoning}"
            )
        lines.append(
            f"\nOverall pass rate: {verification.pass_rate:.1%}. "
            f"Target: {settings.agents.verification_threshold:.0%}."
        )
        return "\n".join(lines)


# ============================================================================
# Core: verify a single citation
# ============================================================================

def _verify_single_citation(
    bibtex_key: str,
    claim: str,
    source_text: str,
    llm=None,
) -> dict:
    """Verifica un singolo claim contro il testo sorgente.

    Args:
        bibtex_key: Chiave BibTeX del paper citato.
        claim: Testo del claim nella sezione del survey.
        source_text: Testo sorgente dal paper citato.
        llm: Istanza LLM (default: Flash).

    Returns:
        Dict con status, confidence, reasoning.
    """
    if llm is None:
        llm = get_llm(
            "flash", max_tokens=8192, timeout_seconds=90, reasoning_effort="high",
        )

    prompt = _VERIFICATION_PROMPT_TEMPLATE.format(
        claim=claim[:1000],  # Tronca claim molto lunghi
        bibtex_key=bibtex_key,
        source_text=source_text[:4000],
    )

    try:
        raw = llm_call_with_retry(
            llm, prompt, max_retries=2, base_delay=5.0, max_delay=30.0,
        )
        parsed = _parse_verification_response(raw)
        if parsed:
            parsed["bibtex_key"] = bibtex_key
            return parsed
    except Exception as e:
        logger.warning(f"Errore verifica [{bibtex_key}]: {e}")

    # Fallback: SOURCE_NOT_FOUND se non riesci a verificare
    return {
        "status": "SOURCE_NOT_FOUND",
        "confidence": 0.0,
        "reasoning": "Verification failed due to LLM error.",
        "bibtex_key": bibtex_key,
        "citation_index": -1,
    }


def _verify_citation_batch(
    citations_data: list[dict],
    llm=None,
) -> list[dict]:
    """Verifica un batch di citazioni in una singola chiamata LLM.

    Più efficiente di verifiche singole quando ci sono molte citazioni.

    Args:
        citations_data: Lista di dict con bibtex_key, claim, source_text.
        llm: Istanza LLM (default: Flash).

    Returns:
        Lista di dict di verifica, uno per citazione.
    """
    if not citations_data:
        return []

    if llm is None:
        llm = get_llm(
            "flash", max_tokens=16384, timeout_seconds=120, reasoning_effort="high",
        )

    # Costruisci il blocco citazioni
    blocks: list[str] = []
    for i, cd in enumerate(citations_data):
        block = (
            f"### Citation {i}\n"
            f"**Bibtex Key:** [{cd['bibtex_key']}]\n"
            f"**Claim:** \"{cd['claim'][:500]}\"\n"
            f"**Source Text:**\n{cd['source_text'][:8000]}\n"
        )
        blocks.append(block)

    prompt = _BATCH_VERIFICATION_PROMPT_TEMPLATE.format(
        citations_block="\n".join(blocks),
    )

    try:
        raw = llm_call_with_retry(
            llm, prompt, max_retries=2, base_delay=5.0, max_delay=60.0,
        )
        parsed = _parse_batch_verification_response(raw, len(citations_data))
        if parsed and len(parsed) >= 1:
            # Assicura che ogni risultato abbia il bibtex_key corretto
            for i, result in enumerate(parsed):
                if i < len(citations_data):
                    result["bibtex_key"] = citations_data[i]["bibtex_key"]
                    result["citation_index"] = i
            return parsed
    except Exception as e:
        logger.warning(f"Errore batch verification ({len(citations_data)} citazioni): {e}")

    # Fallback: verifica singola per ogni citazione
    logger.info("Fallback a verifica singola per citazione...")
    results = []
    for cd in citations_data:
        result = _verify_single_citation(
            cd["bibtex_key"], cd["claim"], cd["source_text"], llm=llm,
        )
        results.append(result)
    return results


# ============================================================================
# Core: verify_section
# ============================================================================

def verify_section(
    state: SurveyState,
    section_key: str,
    *,
    store=None,
    llm=None,
    coherence_llm=None,
    batch_size: int = 5,
    generate_feedback: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Verifica tutte le citazioni in una singola sezione.

    Per ogni citazione [bibtex_key] nella sezione:
    1. Recupera i chunk originali dal QdrantStore
    2. Estrae il claim (frase che cita il paper)
    3. Usa Gemini Flash per verificare claim vs chunk
    4. Produce VerificationResult aggregato

    Se > verification_threshold citazioni non supportate → NEEDS_REVISION.
    Max 3 iterazioni writer↔critic per sezione.

    Args:
        state: Stato corrente della pipeline.
        section_key: Chiave della sezione da verificare.
        store: QdrantStore pre-inizializzato (opzionale).
        llm: LLM pre-inizializzato (opzionale, default Flash).
        batch_size: Numero di citazioni per batch LLM.
        generate_feedback: Se generare feedback per il Writer.

    Returns:
        Dict parziale per aggiornamento dello stato LangGraph con
        ``verification_results``, ``sections``, ``errors``.
    """
    errors: list[str] = []
    t0 = time.time()

    # --- Recupera la sezione drafted ---
    sections = get_sections_from_state(state)
    draft = sections.get(section_key)

    if draft is None:
        err = f"Sezione '{section_key}' non trovata nello stato. Niente da verificare."
        logger.warning(err)
        return {"errors": [err]}

    if draft.status in (SectionStatus.VERIFIED, SectionStatus.FINALIZED):
        logger.info(f"Sezione '{section_key}' già verificata/finalizzata, skip.")
        return {}

    logger.info(
        f"Verifica sezione '{section_key}': {draft.word_count} parole, "
        f"iterazione {draft.iteration}"
    )

    outline_list = get_outline_from_state(state)
    target_outline = next((o for o in outline_list if o.key == section_key), None)
    length_passed, length_message = _check_section_length(
        draft,
        target_outline,
        strict=strict,
    )

    # Compatibilità per l'API helper isolata. La pipeline completa passa
    # sempre strict=True e non accetta mai sezioni imperfette al limite.
    max_iters = settings.agents.max_writer_iterations
    if not strict and draft.iteration >= max_iters:
        updated_draft = SectionDraft.from_dict(draft.to_dict())
        updated_draft.status = SectionStatus.VERIFIED
        updated_draft.critic_feedback = (
            f"Accepted after {max_iters} iterations (max reached)."
        )
        return {
            "sections": {section_key: updated_draft.to_dict()},
            "errors": [f"Accepted after {max_iters} iterations (legacy helper mode)."],
        }

    # --- Estrai citazioni dal testo ---
    text_citations = _extract_citations_from_text(draft.text)
    citation_occurrences = _extract_citation_occurrences(draft.text)

    # Keep occurrence-level provenance.  Repeated uses of one paper can point to
    # different chunks and must not collapse into a dict keyed only by bibtex key.
    tracked_citations = [c.to_dict() for c in draft.citations]
    previous_verification = (state.get("verification_results") or {}).get(section_key)
    supported_cache = _build_supported_verification_cache(previous_verification)

    if not text_citations:
        logger.warning("Nessuna citazione trovata in '%s'.", section_key)
        no_citation_passed = not strict
        vr = VerificationResult(
            section_key=section_key,
            total_citations=0,
            supported=0,
            partially_supported=0,
            not_supported=0,
            source_not_found=0,
            details=[],
            pass_rate=1.0 if no_citation_passed else 0.0,
            passed=no_citation_passed,
            iteration=draft.iteration,
            verified_at=datetime.now(),
        )
        updated_draft = SectionDraft.from_dict(draft.to_dict())
        if no_citation_passed:
            updated_draft.status = SectionStatus.VERIFIED
        else:
            updated_draft.status = SectionStatus.NEEDS_REVISION
            updated_draft.critic_feedback = (
                f"Add at least {settings.agents.min_citations_per_section} distinct, "
                "DBLP-verified citations supported by retrieved source text."
            )
            if not length_passed:
                updated_draft.critic_feedback += f"\n\n[LENGTH]\n{length_message}"

        return {
            "verification_results": {section_key: vr.to_dict()},
            "sections": {section_key: updated_draft.to_dict()},
        }

    logger.info(
        "Citazioni da verificare in '%s': %d occorrenze, %d fonti distinte",
        section_key,
        len(citation_occurrences),
        len(text_citations),
    )

    # --- Lazy init store e LLM ---
    if store is None:
        try:
            from paperhunter.indexing.qdrant_store import QdrantStore
            store = QdrantStore()
        except Exception as e:
            err = f"Impossibile inizializzare QdrantStore: {e}"
            logger.error(err)
            errors.append(err)
            # Senza store, tutte le citazioni → SOURCE_NOT_FOUND
            details = [
                VerificationDetail(
                    citation_key=occurrence["bibtex_key"],
                    claim_text=occurrence["claim"],
                    source_chunk_id=None,
                    source_text_excerpt="",
                    status=VerificationStatus.SOURCE_NOT_FOUND,
                    confidence=0.0,
                    reasoning="QdrantStore not available for verification.",
                )
                for occurrence in citation_occurrences
            ]
            vr = _build_verification_result(
                section_key, details, draft.iteration, strict=strict,
            )
            return {
                "verification_results": {section_key: vr.to_dict()},
                "errors": errors,
            }

    if llm is None:
        tier = settings.agents.critic_model
        llm = get_llm(
            tier,
            max_tokens=16384,
            timeout_seconds=120,
            reasoning_effort="high",
        )

    # --- Prepara dati per verifica ---
    citations_to_verify: list[dict] = []
    verification_details: list[VerificationDetail] = []
    source_cache: dict[str, list[dict]] = {}

    for occurrence in citation_occurrences:
        bibtex_key = occurrence["bibtex_key"]
        claim_text = occurrence["claim"]

        tracked = _select_tracked_citation_record(
            tracked_citations,
            bibtex_key,
            claim_text,
        )
        specific_chunk_id = tracked.get("chunk_id") if tracked else None

        cached = supported_cache.get(
            _verification_cache_key(bibtex_key, claim_text)
        )
        cached_chunk_id = cached.get("source_chunk_id") if cached else None
        if (
            cached is not None
            and specific_chunk_id
            and cached_chunk_id == specific_chunk_id
        ):
            cached_detail = VerificationDetail.from_dict(cached)
            cached_detail.claim_text = claim_text[:500]
            verification_details.append(cached_detail)
            continue

        # Multiple claim checks for the same paper share one Qdrant read.
        if bibtex_key not in source_cache:
            source_cache[bibtex_key] = _retrieve_source_chunks(bibtex_key, store=store)
        all_source_chunks = source_cache[bibtex_key]

        if not all_source_chunks:
            verification_details.append(VerificationDetail(
                citation_key=bibtex_key,
                claim_text=claim_text[:500],
                source_chunk_id=None,
                source_text_excerpt="",
                status=VerificationStatus.SOURCE_NOT_FOUND,
                confidence=0.0,
                reasoning="No source chunks were found for this citation key.",
            ))
            continue

        # Semantic support is irrelevant if the paper is not in the strict
        # DBLP-backed corpus. Reject it before spending a verifier call.
        if all_source_chunks and not _source_chunks_are_citable(all_source_chunks):
            verification_details.append(VerificationDetail(
                citation_key=bibtex_key,
                claim_text=claim_text[:500],
                source_chunk_id=None,
                source_text_excerpt="",
                status=VerificationStatus.SOURCE_NOT_FOUND,
                confidence=0.0,
                reasoning=(
                    "Source exists only in the background corpus and is not "
                    "DBLP-verified/citable."
                ),
            ))
            continue
        # Cerca anche per chunk_id specifico (se il Writer l'ha tracciato)
        source_chunks = _select_source_chunks_for_claim(
            all_source_chunks,
            claim_text,
            tracked_chunk_id=specific_chunk_id,
        )

        source_text = _format_source_text(source_chunks, max_chars=8000)

        # v3.6 — Atomic verification: prepone l'atomic claim più simile (B4)
        if getattr(settings.agents, "critic_atomic_verification", False):
            source_text = _augment_source_with_atomic_claim(source_text, claim_text)

        citations_to_verify.append({
            "bibtex_key": bibtex_key,
            "claim": claim_text,
            "source_text": source_text,
            "source_chunks": source_chunks,
            "chunk_id": specific_chunk_id,
        })

    # --- Verifica in batch ---
    # Splitta in batch
    for batch_start in range(0, len(citations_to_verify), batch_size):
        batch = citations_to_verify[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(citations_to_verify) + batch_size - 1) // batch_size

        logger.info(
            f"Verifica batch {batch_num}/{total_batches}: "
            f"{len(batch)} citazioni"
        )

        if len(batch) == 1:
            # Singola citazione → verifica singola
            result = _verify_single_citation(
                batch[0]["bibtex_key"],
                batch[0]["claim"],
                batch[0]["source_text"],
                llm=llm,
            )
            results = [result]
        else:
            results = _verify_citation_batch(batch, llm=llm)

        # Mappa risultati ai citation data
        for i, cd in enumerate(batch):
            if i < len(results):
                r = results[i]
            else:
                r = {
                    "status": "SOURCE_NOT_FOUND",
                    "confidence": 0.0,
                    "reasoning": "Verification result missing from batch response.",
                }

            # Determina il chunk_id sorgente
            source_chunk_id = cd.get("chunk_id")
            if not source_chunk_id and cd.get("source_chunks"):
                source_chunk_id = cd["source_chunks"][0].get("chunk_id", "")

            # Excerpt dal testo sorgente
            source_excerpt = ""
            if cd.get("source_chunks"):
                source_excerpt = cd["source_chunks"][0].get("text", "")[:300]

            detail = VerificationDetail(
                citation_key=cd["bibtex_key"],
                claim_text=cd["claim"][:500],
                source_chunk_id=source_chunk_id,
                source_text_excerpt=source_excerpt,
                status=VerificationStatus(r.get("status", "SOURCE_NOT_FOUND")),
                confidence=r.get("confidence", 0.0),
                reasoning=r.get("reasoning", ""),
            )
            verification_details.append(detail)

    # --- Construisci il VerificationResult ---
    vr = _build_verification_result(
        section_key, verification_details, draft.iteration, strict=strict,
    )
    coverage_notes = ""
    distinct_citations = len(set(text_citations))
    if strict and distinct_citations < settings.agents.min_citations_per_section:
        vr.passed = False
        coverage_notes = (
            "\n\n[CITATION COVERAGE]\n"
            f"Found {distinct_citations} distinct sources; add at least "
            f"{settings.agents.min_citations_per_section - distinct_citations} more "
            "DBLP-verified, claim-supported sources."
        )

    length_notes = ""
    if strict and not length_passed:
        vr.passed = False
        length_notes = f"\n\n[LENGTH / COMPLETENESS]\n{length_message}"

    if (
        strict
        and draft.iteration >= settings.agents.max_writer_iterations
        and not vr.passed
    ):
        fallback = _build_strict_supported_only_fallback(
            draft,
            verification_details,
            target_outline,
            allow_incomplete_length=True,
        )
        if fallback is not None:
            draft, verification_details = fallback
            text_citations = _extract_citations_from_text(draft.text)
            distinct_citations = len(set(text_citations))
            vr = _build_verification_result(
                section_key,
                verification_details,
                draft.iteration,
                strict=True,
            )
            coverage_notes = ""
            if distinct_citations < settings.agents.min_citations_per_section:
                vr.passed = False
                coverage_notes = (
                    "\n\n[CITATION COVERAGE]\n"
                    f"Found {distinct_citations} distinct sources after strict "
                    "pruning."
                )
            length_passed, length_message = _check_section_length(
                draft,
                target_outline,
                strict=True,
            )
            length_notes = ""
            if not length_passed:
                vr.passed = False
                length_notes = f"\n\n[LENGTH / COMPLETENESS]\n{length_message}"
            logger.info(
                "Strict final fallback for '%s': %d supported occurrences, "
                "%d words",
                section_key,
                len(verification_details),
                draft.word_count,
            )

    # --- v3.6 Pass 2: Holistic coherence (C4) ---
    coherence_notes = ""
    if getattr(settings.agents, "critic_two_pass", False):
        # Verification reasoning and holistic JSON editing have different token
        # profiles; production passes a compact non-thinking LLM for this pass.
        coh = _run_coherence_pass(
            draft.text,
            target_outline,
            llm=coherence_llm or llm,
        )
        if coh is not None:
            if not coh["passed"] and (coh["issues"] or coh["suggested_revisions"]):
                # Marca la verifica come non passata se la coerenza è gravemente compromessa
                logger.warning(
                    "[two-pass] Coerenza compromessa per '%s': %s",
                    section_key,
                    coh["issues"],
                )
                vr.passed = False
                bullets = "\n".join(f"  - {x}" for x in coh["issues"])
                coherence_notes = (
                    "\n\n[COHERENCE PASS — issues to fix]\n"
                    f"{bullets}\n"
                    f"Suggested revisions: {coh['suggested_revisions']}"
                )
            else:
                logger.info("[two-pass] Coerenza OK per '%s'", section_key)

    elapsed = time.time() - t0

    # --- Log risultati ---
    _log_verification_result(section_key, vr, elapsed)

    # --- Genera feedback se la verifica non è passata ---
    critic_feedback = ""
    if not vr.passed and generate_feedback:
        # Trova il titolo della sezione dall'outline
        outline_list = get_outline_from_state(state)
        section_title = section_key
        for o in outline_list:
            if o.key == section_key:
                section_title = o.title
                break

        critic_feedback = _generate_critic_feedback(section_title, vr)
        if coverage_notes:
            critic_feedback = (critic_feedback or "") + coverage_notes
        if length_notes:
            critic_feedback = (critic_feedback or "") + length_notes
        if coherence_notes:
            critic_feedback = (critic_feedback or "") + coherence_notes

    # --- Aggiorna la sezione ---
    updated_draft = SectionDraft.from_dict(draft.to_dict())
    if vr.passed:
        updated_draft.status = SectionStatus.VERIFIED
        logger.info(f"Sezione '{section_key}' VERIFICATA (pass rate: {vr.pass_rate:.1%})")
        print_success(f"Sezione '{section_key}' verificata: {vr.pass_rate:.1%}")
    else:
        updated_draft.status = SectionStatus.NEEDS_REVISION
        updated_draft.critic_feedback = critic_feedback
        logger.warning(
            f"Sezione '{section_key}' DA RISCRIVERE (pass rate: {vr.pass_rate:.1%})"
        )
        print_warning(
            f"Sezione '{section_key}' non verificata: {vr.pass_rate:.1%}. "
            f"Feedback generato per il Writer."
        )

    # --- Provenance records ---
    provenance_records = _build_provenance_records(section_key, verification_details)

    # --- Self-evolution: record lessons from failed verifications ---
    if settings.evolution.enabled and not vr.passed:
        try:
            evo_store = LessonStore(
                path=settings.evolution.lessons_path,
                half_life_days=settings.evolution.half_life_days,
                expiry_days=settings.evolution.expiry_days,
            )
            lessons = extract_from_verification(
                section_key=section_key,
                verification_result=vr,
                threshold=settings.agents.verification_threshold,
            )
            for lesson in lessons:
                evo_store.record(lesson)
            if lessons:
                logger.info("[evolution] Recorded %d lesson(s) from '%s' verification", len(lessons), section_key)
        except Exception as e:
            logger.warning("[evolution] Failed to record lessons: %s", e)

    return {
        "verification_results": {section_key: vr.to_dict()},
        "sections": {section_key: updated_draft.to_dict()},
        "provenance": provenance_records,
        "errors": errors if errors else [],
    }


# ============================================================================
# Verify all sections
# ============================================================================

def verify_all_sections(
    state: SurveyState,
    *,
    store=None,
    llm=None,
    coherence_llm=None,
    batch_size: int = 5,
    skip_types: set[str] | None = None,
) -> dict[str, Any]:
    """Verifica tutte le sezioni drafted nello stato.

    Itera su tutte le sezioni con status DRAFTED e le verifica.
    Salta sezioni già verificate/finalizzate.

    Args:
        state: Stato corrente.
        store: QdrantStore pre-inizializzato.
        llm: LLM pre-inizializzato.
        batch_size: Citazioni per batch LLM.
        skip_types: Tipi di sezione da saltare.

    Returns:
        Dict parziale con verification_results, sections, provenance, errors.
    """
    if skip_types is None:
        skip_types = set()

    sections = get_sections_from_state(state)
    outline = get_outline_from_state(state)
    outline_map = {o.key: o for o in outline}

    errors: list[str] = []
    all_verifications: dict[str, dict] = {}
    all_sections: dict[str, dict] = {}
    all_provenance: list[dict] = []

    # Filtra sezioni da verificare
    sections_to_verify = [
        (k, v) for k, v in sections.items()
        if k in outline_map
        and v.status == SectionStatus.DRAFTED
        and outline_map[k].section_type not in skip_types
    ]

    if not sections_to_verify:
        logger.info("Nessuna sezione da verificare.")
        return {"current_phase": PipelinePhase.VERIFYING.value}

    logger.info(f"Sezioni da verificare: {len(sections_to_verify)}/{len(sections)}")

    # Log tabella sezioni
    data = [
        {"Key": k, "Title": v.title, "Words": str(v.word_count),
         "Citations": str(len(v.citations)), "Iter": str(v.iteration)}
        for k, v in sections_to_verify
    ]
    print_table("Sezioni da verificare", data)

    # Inizializza store e LLM una sola volta
    if store is None:
        try:
            from paperhunter.indexing.qdrant_store import QdrantStore
            store = QdrantStore()
        except Exception as e:
            logger.error(f"Impossibile inizializzare QdrantStore: {e}")
            errors.append(f"QdrantStore init failed: {e}")

    llm_was_provided = llm is not None
    if llm is None:
        tier = settings.agents.critic_model
        llm = get_llm(
            tier,
            max_tokens=16384,
            timeout_seconds=120,
            reasoning_effort="high",
        )
    if coherence_llm is None:
        # Preserve offline test injection; production gets a purpose-sized model.
        coherence_llm = llm if llm_was_provided else get_llm(
            "flash",
            max_tokens=2048,
            timeout_seconds=60,
            thinking_enabled=False,
        )

    t0 = time.time()
    verified_count = 0
    needs_revision_count = 0
    failed_count = 0

    for i, (section_key, draft) in enumerate(sections_to_verify):
        logger.info(
            f"\n{'=' * 60}\n"
            f"Verifica sezione {i + 1}/{len(sections_to_verify)}: "
            f"{draft.title}\n"
            f"{'=' * 60}"
        )

        try:
            result = verify_section(
                {**state, "sections": {**state.get("sections", {}), **all_sections}},
                section_key,
                store=store,
                llm=llm,
                coherence_llm=coherence_llm,
                batch_size=batch_size,
                strict=True,
            )

            if "verification_results" in result:
                all_verifications.update(result["verification_results"])
            if "sections" in result:
                all_sections.update(result["sections"])
            if "provenance" in result:
                all_provenance.extend(result["provenance"])
            if result.get("errors"):
                errors.extend(result["errors"])

            # Aggiorna contatori
            vr_data = result.get("verification_results", {}).get(section_key, {})
            if isinstance(vr_data, dict):
                if vr_data.get("passed", False):
                    verified_count += 1
                else:
                    needs_revision_count += 1
            else:
                verified_count += 1

        except Exception as e:
            err = f"Errore critico verificando '{section_key}': {e}"
            logger.error(err, exc_info=True)
            print_error(err)
            errors.append(err)
            failed_count += 1

    elapsed = time.time() - t0
    logger.info(
        f"\nVerifica completata: {verified_count} verificate, "
        f"{needs_revision_count} da riscrivere, "
        f"{failed_count} fallite, {elapsed:.1f}s totali"
    )

    # Rilascia il lock Qdrant prima che il nodo successivo (Writer) lo richieda
    if store is not None:
        try:
            store.close()
        except Exception:
            pass

    return {
        "verification_results": all_verifications,
        "sections": all_sections,
        "provenance": all_provenance,
        "current_phase": PipelinePhase.VERIFYING.value,
        "errors": errors if errors else [],
        "metadata": {
            **state.get("metadata", {}),
            "critic_verified_count": verified_count,
            "critic_needs_revision_count": needs_revision_count,
            "critic_failed_count": failed_count,
            "critic_total_time": round(elapsed, 1),
            "critic_model": settings.agents.critic_model,
        },
    }


# ============================================================================
# LangGraph node function
# ============================================================================

def verify(state: SurveyState) -> dict[str, Any]:
    """Nodo LangGraph: Critic Agent.

    Verifica tutte le sezioni con status DRAFTED nello stato.
    Genera feedback per le sezioni che non passano la verifica.

    Args:
        state: Stato corrente della pipeline LangGraph.

    Returns:
        Dict parziale con ``verification_results``, ``sections``,
        ``provenance``, ``current_phase``, ``errors``, ``metadata``.
    """
    logger.info("=" * 60)
    logger.info("CRITIC AGENT — Inizio verifica citazioni")
    logger.info("=" * 60)

    # Verifica prerequisiti
    sections = get_sections_from_state(state)
    drafted = {
        k: v for k, v in sections.items()
        if v.status == SectionStatus.DRAFTED
    }

    if not drafted:
        logger.info("Nessuna sezione drafted da verificare.")
        return {
            "current_phase": PipelinePhase.VERIFYING.value,
        }

    logger.info(f"Sezioni drafted da verificare: {len(drafted)}")

    # Esegui verifica di tutte le sezioni
    result = verify_all_sections(state)

    # Aggiorna la fase
    result["current_phase"] = PipelinePhase.VERIFYING.value

    # Sommario
    verifications = result.get("verification_results", {})
    needs_revision = sum(
        1 for vr_data in verifications.values()
        if isinstance(vr_data, dict) and not vr_data.get("passed", False)
    )
    verified = sum(
        1 for vr_data in verifications.values()
        if isinstance(vr_data, dict) and vr_data.get("passed", False)
    )

    logger.info(
        f"Critic completato: {verified} verificate, {needs_revision} da riscrivere"
    )
    if needs_revision > 0:
        print_warning(
            f"Critic Agent: {needs_revision} sezioni necessitano riscrittura."
        )
    else:
        print_success(f"Critic Agent: tutte le {verified} sezioni verificate!")

    return result


# ============================================================================
# Internal helpers
# ============================================================================

def _build_verification_result(
    section_key: str,
    details: list[VerificationDetail],
    iteration: int,
    *,
    strict: bool = False,
) -> VerificationResult:
    """Costruisce il VerificationResult aggregato dai dettagli."""
    total = len(details)
    supported = sum(1 for d in details if d.status == VerificationStatus.SUPPORTED)
    partially = sum(1 for d in details if d.status == VerificationStatus.PARTIALLY_SUPPORTED)
    not_supported = sum(1 for d in details if d.status == VerificationStatus.NOT_SUPPORTED)
    not_found = sum(1 for d in details if d.status == VerificationStatus.SOURCE_NOT_FOUND)

    vr = VerificationResult(
        section_key=section_key,
        total_citations=total,
        supported=supported,
        partially_supported=partially,
        not_supported=not_supported,
        source_not_found=not_found,
        details=details,
        iteration=iteration,
        verified_at=datetime.now(),
    )

    threshold = settings.agents.verification_threshold
    vr.compute_pass_rate(threshold, require_full_support=strict)

    return vr


def _log_verification_result(
    section_key: str,
    vr: VerificationResult,
    elapsed: float,
) -> None:
    """Logga il risultato della verifica con tabella Rich."""
    status_str = "PASSED ✓" if vr.passed else "FAILED ✗"
    logger.info(
        f"Verifica '{section_key}': {status_str} "
        f"({vr.pass_rate:.1%}, {vr.total_citations} citazioni, {elapsed:.1f}s)"
    )

    # Dettagli per citazione
    if vr.details:
        data = [
            {
                "Key": d.citation_key[:25],
                "Status": d.status.value,
                "Conf": f"{d.confidence:.2f}",
                "Reasoning": d.reasoning[:60],
            }
            for d in vr.details
        ]
        print_table(f"Verifica '{section_key}' — {status_str}", data)


def _build_provenance_records(
    section_key: str,
    details: list[VerificationDetail],
) -> list[dict]:
    """Costruisce record di provenance dai dettagli di verifica.

    Ogni record mappa un claim alla sua fonte e allo stato di verifica.

    Args:
        section_key: Chiave della sezione.
        details: Dettagli di verifica per ogni citazione.

    Returns:
        Lista di dict serializzabili come ProvenanceRecord.
    """
    records: list[dict] = []
    for d in details:
        records.append({
            "section_key": section_key,
            "citation_key": d.citation_key,
            "claim_text": d.claim_text,
            "source_chunk_id": d.source_chunk_id or "",
            "source_text_excerpt": d.source_text_excerpt[:300],
            "verification_status": d.status.value,
            "confidence": d.confidence,
            "reasoning": d.reasoning,
            "verified_at": datetime.now().isoformat(),
        })
    return records


# ============================================================================
# Convenience functions
# ============================================================================

def get_verification_stats(
    verifications: dict[str, VerificationResult],
) -> dict[str, Any]:
    """Calcola statistiche aggregate sulle verifiche.

    Args:
        verifications: Dict di VerificationResult per sezione.

    Returns:
        Dict con statistiche aggregate.
    """
    if not verifications:
        return {
            "total_sections": 0,
            "verified_sections": 0,
            "needs_revision_sections": 0,
            "total_citations": 0,
            "overall_pass_rate": 0.0,
        }

    total_sections = len(verifications)
    verified = sum(1 for vr in verifications.values() if vr.passed)
    needs_revision = total_sections - verified
    total_citations = sum(vr.total_citations for vr in verifications.values())
    total_supported = sum(vr.supported + vr.partially_supported for vr in verifications.values())
    overall_pass_rate = total_supported / total_citations if total_citations > 0 else 0.0

    per_section: list[dict] = []
    for key, vr in verifications.items():
        per_section.append({
            "key": key,
            "total": vr.total_citations,
            "supported": vr.supported,
            "partially": vr.partially_supported,
            "not_supported": vr.not_supported,
            "not_found": vr.source_not_found,
            "pass_rate": round(vr.pass_rate, 3),
            "passed": vr.passed,
            "iteration": vr.iteration,
        })

    return {
        "total_sections": total_sections,
        "verified_sections": verified,
        "needs_revision_sections": needs_revision,
        "total_citations": total_citations,
        "total_supported": total_supported,
        "overall_pass_rate": round(overall_pass_rate, 3),
        "per_section": per_section,
    }
