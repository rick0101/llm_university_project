"""
Test suite per paperhunter.agents.critic_agent

Test offline (no LLM, no QdrantStore) + struttura per test LIVE opzionali.
"""

import json
import sys
import os
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

# Assicurati che il progetto sia nel path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperhunter.agents.state import (
    CitationRecord,
    PipelinePhase,
    SectionDraft,
    SectionOutline,
    SectionStatus,
    SurveyState,
    VerificationDetail,
    VerificationResult,
    create_initial_state,
    get_sections_from_state,
    get_verifications_from_state,
)
from paperhunter.models import VerificationStatus


# ============================================================================
# Test helpers
# ============================================================================

def _make_draft(
    section_key: str = "test_section",
    title: str = "Test Section",
    text: str = "",
    citations: list[dict] | None = None,
    status: SectionStatus = SectionStatus.DRAFTED,
    iteration: int = 1,
) -> SectionDraft:
    """Crea un SectionDraft di test."""
    cits = [
        CitationRecord(**c) for c in (citations or [])
    ]
    draft = SectionDraft(
        section_key=section_key,
        title=title,
        text=text,
        citations=cits,
        status=status,
        iteration=iteration,
        drafted_at=datetime.now(),
    )
    draft.compute_word_count()
    return draft


def _make_state_with_draft(
    section_key: str = "test_section",
    text: str = "",
    citations: list[dict] | None = None,
    status: SectionStatus = SectionStatus.DRAFTED,
    iteration: int = 1,
    outline: list[dict] | None = None,
) -> SurveyState:
    """Crea uno stato con una sezione drafted."""
    draft = _make_draft(
        section_key=section_key,
        text=text,
        citations=citations,
        status=status,
        iteration=iteration,
    )
    state = create_initial_state()
    state["sections"] = {section_key: draft.to_dict()}
    if outline:
        state["outline"] = outline
    else:
        state["outline"] = [
            SectionOutline(
                key=section_key,
                title="Test Section",
                section_type="body",
                order=0,
            ).to_dict()
        ]
    return state


# ============================================================================
# Test 1: Estrazione citazioni dal testo
# ============================================================================

def test_extract_citations_from_text():
    """Testa l'estrazione di chiavi BibTeX dal testo."""
    from paperhunter.agents.critic_agent import _extract_citations_from_text

    # Caso 1: testo con citazioni
    text = (
        "Recent work has shown that adversarial suffixes can transfer "
        "across LLM families [Zou_2023_GCG]. Building on this, "
        "[Greshake_2023_LLM] demonstrated indirect prompt injection "
        "vulnerabilities. The evaluation framework [Li_2023_Halueval] "
        "provides benchmarks for hallucination detection. "
        "See also [Zou_2023_GCG] for more details."
    )
    keys = _extract_citations_from_text(text)
    assert len(keys) == 3, f"Expected 3 unique keys, got {len(keys)}: {keys}"
    assert keys[0] == "Zou_2023_GCG"
    assert keys[1] == "Greshake_2023_LLM"
    assert keys[2] == "Li_2023_Halueval"

    # Caso 2: testo senza citazioni
    assert _extract_citations_from_text("No citations here.") == []
    assert _extract_citations_from_text("") == []

    # Caso 3: falsi positivi filtrati
    text_fp = "See [Table_1] and [Figure_2] and [Section_3] and [Zou_2023_GCG]."
    keys_fp = _extract_citations_from_text(text_fp)
    assert len(keys_fp) == 1
    assert keys_fp[0] == "Zou_2023_GCG"

    # Caso 4: chiavi multi-word
    text_mw = "As shown in [Lee_2025_PromptShield] and [Wang_2024_Multi_Word_Key]."
    keys_mw = _extract_citations_from_text(text_mw)
    assert len(keys_mw) == 2
    assert "Lee_2025_PromptShield" in keys_mw
    assert "Wang_2024_Multi_Word_Key" in keys_mw

    print("  ✓ test_extract_citations_from_text: 4 casi passati")


# ============================================================================
# Test 2: Estrazione claim per citazione
# ============================================================================

def test_extract_claims_for_citation():
    """Testa l'estrazione dei claim per una citazione specifica."""
    from paperhunter.agents.critic_agent import _extract_claims_for_citation

    text = (
        "Adversarial suffixes can transfer across LLM families [Zou_2023_GCG]. "
        "This approach was later extended by several studies. "
        "The original GCG attack remains the baseline [Zou_2023_GCG]."
    )

    claims = _extract_claims_for_citation(text, "Zou_2023_GCG")
    assert len(claims) == 2, f"Expected 2 claims, got {len(claims)}"
    assert any("transfer" in c.lower() or "gcg" in c.lower() for c in claims)

    # Citazione inesistente
    claims_none = _extract_claims_for_citation(text, "Unknown_2024_Key")
    assert len(claims_none) == 1  # fallback
    assert "Unknown_2024_Key" in claims_none[0]

    # Testo vuoto
    assert len(_extract_claims_for_citation("", "Key")) == 1

    print("  ✓ test_extract_claims_for_citation: 3 casi passati")


# ============================================================================
# Test 3: Parsing risposta verifica singola
# ============================================================================

def test_parse_verification_response():
    """Testa il parsing delle risposte JSON del LLM di verifica."""
    from paperhunter.agents.critic_agent import _parse_verification_response

    # Caso 1: JSON pulito
    raw = json.dumps({
        "status": "SUPPORTED",
        "confidence": 0.95,
        "reasoning": "The claim matches the source text."
    })
    parsed = _parse_verification_response(raw)
    assert parsed is not None
    assert parsed["status"] == "SUPPORTED"
    assert parsed["confidence"] == 0.95
    assert "matches" in parsed["reasoning"]

    # Caso 2: con code fences
    raw_fenced = '```json\n{"status": "NOT_SUPPORTED", "confidence": 0.2, "reasoning": "Contradicts"}\n```'
    parsed2 = _parse_verification_response(raw_fenced)
    assert parsed2 is not None
    assert parsed2["status"] == "NOT_SUPPORTED"

    # Caso 3: testo extra attorno al JSON
    raw_extra = 'Here is my analysis:\n{"status": "PARTIALLY_SUPPORTED", "confidence": 0.7, "reasoning": "Partially matches"}\nEnd.'
    parsed3 = _parse_verification_response(raw_extra)
    assert parsed3 is not None
    assert parsed3["status"] == "PARTIALLY_SUPPORTED"

    # Caso 4: status con varianti
    raw_variant = '{"status": "PARTIAL", "confidence": 0.6, "reasoning": ""}'
    parsed4 = _parse_verification_response(raw_variant)
    assert parsed4 is not None
    assert parsed4["status"] == "PARTIALLY_SUPPORTED"

    # Caso 5: regex fallback
    raw_regex = 'The status is "status": "SUPPORTED" with "confidence": 0.8 and "reasoning": "Good match"'
    parsed5 = _parse_verification_response(raw_regex)
    assert parsed5 is not None
    assert parsed5["status"] == "SUPPORTED"

    # Caso 6: vuoto → None
    assert _parse_verification_response("") is None
    assert _parse_verification_response("   ") is None

    # Caso 7: confidence out of range → clamped
    raw_clamp = '{"status": "SUPPORTED", "confidence": 1.5, "reasoning": "Sure"}'
    parsed7 = _parse_verification_response(raw_clamp)
    assert parsed7 is not None
    assert parsed7["confidence"] == 1.0

    # Caso 8: trailing comma
    raw_tc = '{"status": "SUPPORTED", "confidence": 0.9, "reasoning": "Test",}'
    parsed8 = _parse_verification_response(raw_tc)
    assert parsed8 is not None
    assert parsed8["status"] == "SUPPORTED"

    print("  ✓ test_parse_verification_response: 8 casi passati")


# ============================================================================
# Test 4: Parsing risposta batch verification
# ============================================================================

def test_parse_batch_verification_response():
    """Testa il parsing delle risposte batch (JSON array)."""
    from paperhunter.agents.critic_agent import _parse_batch_verification_response

    # Caso 1: JSON array pulito
    raw = json.dumps([
        {"citation_index": 0, "bibtex_key": "Zou_2023_GCG", "status": "SUPPORTED", "confidence": 0.95, "reasoning": "Match"},
        {"citation_index": 1, "bibtex_key": "Li_2023_Halueval", "status": "NOT_SUPPORTED", "confidence": 0.3, "reasoning": "No match"},
    ])
    parsed = _parse_batch_verification_response(raw, 2)
    assert parsed is not None
    assert len(parsed) == 2
    assert parsed[0]["status"] == "SUPPORTED"
    assert parsed[1]["status"] == "NOT_SUPPORTED"

    # Caso 2: con code fences
    raw_fenced = '```json\n' + raw + '\n```'
    parsed2 = _parse_batch_verification_response(raw_fenced, 2)
    assert parsed2 is not None
    assert len(parsed2) == 2

    # Caso 3: oggetti JSON separati (fallback)
    raw_sep = 'Result 1: {"status": "SUPPORTED", "confidence": 0.9, "reasoning": "OK"}\n\nResult 2: {"status": "PARTIALLY_SUPPORTED", "confidence": 0.6, "reasoning": "Partial"}'
    parsed3 = _parse_batch_verification_response(raw_sep, 2)
    assert parsed3 is not None
    assert len(parsed3) == 2

    # Caso 4: vuoto → None
    assert _parse_batch_verification_response("", 2) is None

    print("  ✓ test_parse_batch_verification_response: 4 casi passati")


# ============================================================================
# Test 5: Normalizzazione verifica
# ============================================================================

def test_normalize_verification():
    """Testa la normalizzazione dei risultati di verifica."""
    from paperhunter.agents.critic_agent import _normalize_verification

    # Status varianti
    assert _normalize_verification({"status": "PARTIAL"})["status"] == "PARTIALLY_SUPPORTED"
    assert _normalize_verification({"status": "UNSUPPORTED"})["status"] == "NOT_SUPPORTED"
    assert _normalize_verification({"status": "NOT FOUND"})["status"] == "SOURCE_NOT_FOUND"
    assert _normalize_verification({"status": "MISSING"})["status"] == "SOURCE_NOT_FOUND"
    assert _normalize_verification({"status": "unknown_status"})["status"] == "SOURCE_NOT_FOUND"

    # Confidence clamping
    assert _normalize_verification({"status": "SUPPORTED", "confidence": -0.5})["confidence"] == 0.0
    assert _normalize_verification({"status": "SUPPORTED", "confidence": 2.0})["confidence"] == 1.0
    assert _normalize_verification({"status": "SUPPORTED", "confidence": "0.7"})["confidence"] == 0.7
    assert _normalize_verification({"status": "SUPPORTED", "confidence": "invalid"})["confidence"] == 0.0

    print("  ✓ test_normalize_verification: 9 casi passati")


# ============================================================================
# Test 6: Format source text
# ============================================================================

def test_format_source_text():
    """Testa la formattazione del testo sorgente per il prompt."""
    from paperhunter.agents.critic_agent import _format_source_text

    # Case 1: vuoto
    assert "No source" in _format_source_text([])
    assert "No source" in _format_source_text([{"text": ""}])

    # Case 2: con chunk
    chunks = [
        {
            "text": "This paper proposes GCG attack...",
            "metadata": {"section_name": "Introduction", "chunk_type": "original"},
        },
        {
            "text": "Results show 98% ASR...",
            "metadata": {"section_name": "Results", "chunk_type": "original"},
        },
    ]
    formatted = _format_source_text(chunks)
    assert "Source 1" in formatted
    assert "Source 2" in formatted
    assert "GCG attack" in formatted
    assert "Introduction" in formatted

    # Case 3: troncamento
    big_chunks = [{"text": "x" * 5000, "metadata": {"section_name": f"s{i}", "chunk_type": "o"}} for i in range(10)]
    formatted_big = _format_source_text(big_chunks, max_chars=500)
    assert len(formatted_big) < 1000  # Dovrebbe essere troncato

    print("  ✓ test_format_source_text: 3 casi passati")


# ============================================================================
# Test 7: Build VerificationResult
# ============================================================================

def test_build_verification_result():
    """Testa la costruzione del VerificationResult aggregato."""
    from paperhunter.agents.critic_agent import _build_verification_result

    details = [
        VerificationDetail(
            citation_key="Zou_2023_GCG",
            status=VerificationStatus.SUPPORTED,
            confidence=0.95,
            reasoning="Match",
        ),
        VerificationDetail(
            citation_key="Li_2023_Halueval",
            status=VerificationStatus.SUPPORTED,
            confidence=0.90,
            reasoning="Match",
        ),
        VerificationDetail(
            citation_key="Greshake_2023_LLM",
            status=VerificationStatus.PARTIALLY_SUPPORTED,
            confidence=0.7,
            reasoning="Partial",
        ),
        VerificationDetail(
            citation_key="Unknown_2024_Key",
            status=VerificationStatus.NOT_SUPPORTED,
            confidence=0.2,
            reasoning="No match",
        ),
    ]

    vr = _build_verification_result("test_section", details, iteration=1)
    assert vr.section_key == "test_section"
    assert vr.total_citations == 4
    assert vr.supported == 2
    assert vr.partially_supported == 1
    assert vr.not_supported == 1
    assert vr.source_not_found == 0
    assert vr.iteration == 1
    assert vr.verified_at is not None

    # Pass rate: (2 supported + 1 partial) / 4 = 0.75
    assert abs(vr.pass_rate - 0.75) < 0.01
    assert not vr.passed  # 0.75 < 0.80 threshold

    # Caso: tutte supportate → passed
    details_ok = [
        VerificationDetail(citation_key="A", status=VerificationStatus.SUPPORTED, confidence=1.0),
        VerificationDetail(citation_key="B", status=VerificationStatus.SUPPORTED, confidence=0.9),
    ]
    vr_ok = _build_verification_result("ok_section", details_ok, iteration=1)
    assert vr_ok.passed
    assert vr_ok.pass_rate == 1.0

    # Caso: nessuna citazione → trivialmente passed
    vr_empty = _build_verification_result("empty_section", [], iteration=1)
    assert vr_empty.passed
    assert vr_empty.pass_rate == 1.0

    print("  ✓ test_build_verification_result: 3 casi passati")


# ============================================================================
# Test 8: Build provenance records
# ============================================================================

def test_build_provenance_records():
    """Testa la costruzione dei record di provenance."""
    from paperhunter.agents.critic_agent import _build_provenance_records

    details = [
        VerificationDetail(
            citation_key="Zou_2023_GCG",
            claim_text="GCG attack transfers across models",
            source_chunk_id="chunk_abc",
            source_text_excerpt="The GCG attack...",
            status=VerificationStatus.SUPPORTED,
            confidence=0.95,
            reasoning="Direct match",
        ),
        VerificationDetail(
            citation_key="Li_2023_Halueval",
            claim_text="HaluEval benchmarks hallucination",
            source_chunk_id=None,
            source_text_excerpt="",
            status=VerificationStatus.SOURCE_NOT_FOUND,
            confidence=0.0,
            reasoning="No source found",
        ),
    ]

    records = _build_provenance_records("test_section", details)
    assert len(records) == 2

    r0 = records[0]
    assert r0["section_key"] == "test_section"
    assert r0["citation_key"] == "Zou_2023_GCG"
    assert r0["verification_status"] == "SUPPORTED"
    assert r0["confidence"] == 0.95
    assert "verified_at" in r0

    r1 = records[1]
    assert r1["verification_status"] == "SOURCE_NOT_FOUND"
    assert r1["source_chunk_id"] == ""

    print("  ✓ test_build_provenance_records: 2 record costruiti correttamente")


# ============================================================================
# Test 9: verify_section con mock LLM e mock store
# ============================================================================

def test_verify_section_mock():
    """Testa verify_section con mock LLM e store."""
    from paperhunter.agents.critic_agent import verify_section

    text = (
        "Adversarial suffixes can transfer across LLM families [Zou_2023_GCG]. "
        "The GCG attack was evaluated on multiple models [Zou_2023_GCG]. "
        "Prompt injection represents a significant threat [Greshake_2023_LLM]. "
        "HaluEval provides benchmarks for hallucination [Li_2023_Halueval]."
    )
    citations = [
        {"bibtex_key": "Zou_2023_GCG", "claim": "transfer across models", "chunk_id": "c1"},
        {"bibtex_key": "Greshake_2023_LLM", "claim": "prompt injection threat", "chunk_id": "c2"},
        {"bibtex_key": "Li_2023_Halueval", "claim": "hallucination benchmarks", "chunk_id": "c3"},
    ]

    state = _make_state_with_draft(text=text, citations=citations)

    # Mock store che ritorna chunk per ogni paper
    mock_store = MagicMock()
    mock_store.get_paper_chunks.return_value = [
        {
            "chunk_id": "c1",
            "text": "We propose the GCG attack, which generates adversarial suffixes "
                    "that transfer across multiple LLM families including GPT-4 and LLaMA-2.",
            "metadata": {
                "section_name": "Abstract",
                "chunk_type": "original",
                "bibtex_key": "Zou_2023_GCG",
                "is_citable": True,
            },
        },
    ]

    # Mock LLM che risponde sempre SUPPORTED (batch di 3)
    mock_llm = MagicMock()
    def mock_complete_batch(prompt):
        resp = MagicMock()
        # Determina se è un batch prompt o singolo
        if "Citation 0" in prompt and "Citation 1" in prompt:
            # Batch: ritorna array
            resp.__str__ = lambda self: json.dumps([
                {"citation_index": 0, "bibtex_key": "Zou_2023_GCG", "status": "SUPPORTED", "confidence": 0.9, "reasoning": "Match"},
                {"citation_index": 1, "bibtex_key": "Zou_2023_GCG", "status": "SUPPORTED", "confidence": 0.90, "reasoning": "Second claim match"},
                {"citation_index": 2, "bibtex_key": "Greshake_2023_LLM", "status": "SUPPORTED", "confidence": 0.85, "reasoning": "Match"},
                {"citation_index": 3, "bibtex_key": "Li_2023_Halueval", "status": "SUPPORTED", "confidence": 0.88, "reasoning": "Match"},
            ])
        else:
            resp.__str__ = lambda self: json.dumps({
                "status": "SUPPORTED", "confidence": 0.9, "reasoning": "The claim is supported by the source."
            })
        return resp
    mock_llm.complete.side_effect = mock_complete_batch

    result = verify_section(
        state, "test_section",
        store=mock_store, llm=mock_llm,
        generate_feedback=False,
    )

    assert "verification_results" in result
    assert "test_section" in result["verification_results"]

    vr_data = result["verification_results"]["test_section"]
    assert isinstance(vr_data, dict)
    assert vr_data["total_citations"] == 4
    assert vr_data["passed"]

    # Sezione aggiornata
    assert "sections" in result
    section_data = result["sections"]["test_section"]
    assert section_data["status"] == SectionStatus.VERIFIED.value

    # Provenance
    assert "provenance" in result
    assert len(result["provenance"]) == 4

    print("  ✓ test_verify_section_mock: verifica con mock passata")


# ============================================================================
# Test 10: verify_section — sezione senza citazioni
# ============================================================================

def test_verify_section_no_citations():
    """Testa verify_section su una sezione senza citazioni."""
    from paperhunter.agents.critic_agent import verify_section

    state = _make_state_with_draft(
        text="This section has no citations at all. Just plain text."
    )

    result = verify_section(state, "test_section", generate_feedback=False)

    vr_data = result.get("verification_results", {}).get("test_section", {})
    assert vr_data.get("total_citations") == 0
    assert vr_data.get("passed")
    assert vr_data.get("pass_rate") == 1.0

    section_data = result.get("sections", {}).get("test_section", {})
    assert section_data.get("status") == SectionStatus.VERIFIED.value

    print("  ✓ test_verify_section_no_citations: sezione senza citazioni verificata")


# ============================================================================
# Test 11: verify_section — max iterazioni raggiunto
# ============================================================================

def test_verify_section_max_iterations():
    """Testa che al max di iterazioni la sezione venga accettata con warning."""
    from paperhunter.agents.critic_agent import verify_section
    from paperhunter.config import settings

    max_iters = settings.agents.max_writer_iterations
    state = _make_state_with_draft(
        text="Some text with [Zou_2023_GCG] citation.",
        citations=[{"bibtex_key": "Zou_2023_GCG", "claim": "test"}],
        iteration=max_iters,  # Al max
    )

    result = verify_section(state, "test_section", generate_feedback=False)

    section_data = result.get("sections", {}).get("test_section", {})
    assert section_data.get("status") == SectionStatus.VERIFIED.value
    assert "max" in section_data.get("critic_feedback", "").lower() or \
           "accepted" in section_data.get("critic_feedback", "").lower()

    # Non dovrebbe avere verification_results (skip verifica)
    assert "verification_results" not in result or "test_section" not in result.get("verification_results", {})

    print("  ✓ test_verify_section_max_iterations: accettata con warning")


# ============================================================================
# Test 12: verify_section — sezione NEEDS_REVISION
# ============================================================================

def test_verify_section_needs_revision():
    """Testa che sezioni con troppe citazioni non supportate vengano flaggate."""
    from paperhunter.agents.critic_agent import verify_section

    text = (
        "Claim one [Zou_2023_GCG]. Claim two [Greshake_2023_LLM]. "
        "Claim three [Li_2023_Halueval]. Claim four [Wang_2024_Adversarial]. "
        "Claim five [Lee_2025_PromptShield]."
    )

    state = _make_state_with_draft(text=text, iteration=1)

    # Mock store che ritorna chunk vuoti (nessun chunk trovato)
    mock_store = MagicMock()
    mock_store.get_paper_chunks.return_value = []

    # Mock LLM: 3 NOT_SUPPORTED su 5 (60% fail → NEEDS_REVISION)
    call_count = [0]
    def mock_complete(prompt):
        call_count[0] += 1
        resp = MagicMock()
        # Prima 2 SUPPORTED, poi tutto NOT_SUPPORTED
        # Ma come batch response, facciamo un batch di 5
        if "[Source" not in prompt and "Citation" in prompt:
            # Batch prompt
            resp.__str__ = lambda self: json.dumps([
                {"citation_index": 0, "status": "SUPPORTED", "confidence": 0.9, "reasoning": "OK"},
                {"citation_index": 1, "status": "NOT_SUPPORTED", "confidence": 0.2, "reasoning": "No match"},
                {"citation_index": 2, "status": "NOT_SUPPORTED", "confidence": 0.1, "reasoning": "Wrong"},
                {"citation_index": 3, "status": "NOT_SUPPORTED", "confidence": 0.2, "reasoning": "Missing"},
                {"citation_index": 4, "status": "SUPPORTED", "confidence": 0.85, "reasoning": "OK"},
            ])
        else:
            resp.__str__ = lambda self: json.dumps({
                "status": "NOT_SUPPORTED", "confidence": 0.2, "reasoning": "No source"
            })
        return resp

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = mock_complete

    result = verify_section(
        state, "test_section",
        store=mock_store, llm=mock_llm,
        generate_feedback=False,
    )

    vr_data = result.get("verification_results", {}).get("test_section", {})
    assert vr_data is not None
    # 2 supported / 5 total = 0.4 → lower than 0.80 threshold
    assert not vr_data.get("passed")

    section_data = result.get("sections", {}).get("test_section", {})
    assert section_data.get("status") == SectionStatus.NEEDS_REVISION.value

    print("  ✓ test_verify_section_needs_revision: sezione flaggata per riscrittura")


# ============================================================================
# Test 13: verify_section — sezione già verificata (skip)
# ============================================================================

def test_verify_section_already_verified():
    """Testa che sezioni già verificate vengano skippate."""
    from paperhunter.agents.critic_agent import verify_section

    state = _make_state_with_draft(
        text="Text [Zou_2023_GCG].",
        status=SectionStatus.VERIFIED,
    )

    result = verify_section(state, "test_section")
    assert result == {}  # Skip, nessun aggiornamento

    print("  ✓ test_verify_section_already_verified: skip corretto")


# ============================================================================
# Test 14: verify_section — sezione non trovata
# ============================================================================

def test_verify_section_not_found():
    """Testa che una sezione non esistente produca un errore."""
    from paperhunter.agents.critic_agent import verify_section

    state = create_initial_state()
    result = verify_section(state, "nonexistent_section")
    assert result.get("errors")
    assert any("non trovata" in e or "not" in e.lower() for e in result["errors"])

    print("  ✓ test_verify_section_not_found: errore corretto")


# ============================================================================
# Test 15: get_verification_stats
# ============================================================================

def test_get_verification_stats():
    """Testa le statistiche aggregate delle verifiche."""
    from paperhunter.agents.critic_agent import get_verification_stats

    # Caso vuoto
    stats_empty = get_verification_stats({})
    assert stats_empty["total_sections"] == 0
    assert stats_empty["total_citations"] == 0

    # Con dati
    verifications = {
        "section_a": VerificationResult(
            section_key="section_a",
            total_citations=5,
            supported=4,
            partially_supported=1,
            not_supported=0,
            source_not_found=0,
            pass_rate=1.0,
            passed=True,
            iteration=1,
        ),
        "section_b": VerificationResult(
            section_key="section_b",
            total_citations=3,
            supported=1,
            partially_supported=0,
            not_supported=2,
            source_not_found=0,
            pass_rate=0.33,
            passed=False,
            iteration=2,
        ),
    }

    stats = get_verification_stats(verifications)
    assert stats["total_sections"] == 2
    assert stats["verified_sections"] == 1
    assert stats["needs_revision_sections"] == 1
    assert stats["total_citations"] == 8
    assert stats["total_supported"] == 6  # (4+1) + (1+0)
    assert len(stats["per_section"]) == 2

    print("  ✓ test_get_verification_stats: statistiche corrette")


# ============================================================================
# Test 16: verify (nodo LangGraph)
# ============================================================================

def test_verify_langgraph_node():
    """Testa il nodo LangGraph `verify` con mock."""
    from paperhunter.agents.critic_agent import verify

    # Stato senza sezioni drafted → niente da fare
    state = create_initial_state()
    result = verify(state)
    assert result.get("current_phase") == PipelinePhase.VERIFYING.value

    print("  ✓ test_verify_langgraph_node: nodo LangGraph testato")


# ============================================================================
# Test 17: verify_all_sections con mock
# ============================================================================

def test_verify_all_sections_mock():
    """Testa verify_all_sections con più sezioni."""
    from paperhunter.agents.critic_agent import verify_all_sections

    state = create_initial_state()

    # Due sezioni drafted
    draft_a = _make_draft(
        section_key="section_a", title="Section A",
        text="Text with [Zou_2023_GCG] citation.",
        citations=[{"bibtex_key": "Zou_2023_GCG", "claim": "test"}],
    )
    draft_b = _make_draft(
        section_key="section_b", title="Section B",
        text="Another text with [Li_2023_Halueval].",
        citations=[{"bibtex_key": "Li_2023_Halueval", "claim": "test2"}],
    )
    # Una sezione già verificata (dovrebbe essere skippata)
    draft_c = _make_draft(
        section_key="section_c", title="Section C",
        text="Verified text.", status=SectionStatus.VERIFIED,
    )

    state["sections"] = {
        "section_a": draft_a.to_dict(),
        "section_b": draft_b.to_dict(),
        "section_c": draft_c.to_dict(),
    }
    state["outline"] = [
        SectionOutline(key="section_a", title="Section A", section_type="body", order=0).to_dict(),
        SectionOutline(key="section_b", title="Section B", section_type="body", order=1).to_dict(),
        SectionOutline(key="section_c", title="Section C", section_type="body", order=2).to_dict(),
    ]

    # Mock store e LLM
    mock_store = MagicMock()
    mock_store.get_paper_chunks.return_value = [
        {"chunk_id": "c1", "text": "Source text...", "metadata": {"section_name": "Intro", "chunk_type": "original"}},
    ]

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: json.dumps({
        "status": "SUPPORTED", "confidence": 0.9, "reasoning": "OK"
    })
    mock_llm.complete.return_value = mock_response

    result = verify_all_sections(state, store=mock_store, llm=mock_llm)

    assert "verification_results" in result
    # Solo section_a e section_b devono essere verificate (section_c è già VERIFIED)
    assert "section_a" in result["verification_results"]
    assert "section_b" in result["verification_results"]

    assert result.get("current_phase") == PipelinePhase.VERIFYING.value

    # Provenance
    assert "provenance" in result
    assert len(result["provenance"]) >= 2  # Almeno 1 per sezione

    print("  ✓ test_verify_all_sections_mock: 2 sezioni verificate, 1 skippata")


# ============================================================================
# Test 18: Retrieve source chunks
# ============================================================================

def test_retrieve_source_chunks():
    """Testa il recupero dei chunk sorgente con mock store."""
    from paperhunter.agents.critic_agent import _retrieve_source_chunks

    mock_store = MagicMock()

    # Ritorna chunk diversi per ogni collection
    def mock_get_chunks(bibtex_key, collection_name=None):
        if collection_name and "l1" in collection_name:
            return [{
                "chunk_id": "l1_c",
                "text": "L1 summary text",
                "metadata": {"section_name": "summary", "chunk_type": "l1_summary"},
            }]
        return [
            {
                "chunk_id": "c1",
                "text": "Abstract text",
                "metadata": {"section_name": "abstract", "chunk_type": "original"},
            },
            {
                "chunk_id": "c2",
                "text": "Conclusion text",
                "metadata": {"section_name": "conclusion", "chunk_type": "original"},
            },
        ]

    mock_store.get_paper_chunks.side_effect = mock_get_chunks

    chunks = _retrieve_source_chunks("Zou_2023_GCG", store=mock_store, max_chunks=3)
    assert len(chunks) == 3

    # L1 summary dovrebbe essere prima (sort_key=0)
    assert chunks[0]["metadata"]["chunk_type"] == "l1_summary"
    # Poi abstract (sort_key=1)
    assert "abstract" in chunks[1]["metadata"]["section_name"]

    print("  ✓ test_retrieve_source_chunks: ordinamento corretto")


# ============================================================================
# Test 19: Package imports
# ============================================================================

def test_package_imports():
    """Verifica che tutti gli export dal package funzionino."""
    from paperhunter.agents import (
        verify,
        verify_section,
        verify_all_sections,
        get_verification_stats,
    )
    assert callable(verify)
    assert callable(verify_section)
    assert callable(verify_all_sections)
    assert callable(get_verification_stats)

    print("  ✓ test_package_imports: tutti gli export funzionano")


# ============================================================================
# Main
# ============================================================================

def main():
    """Esegue tutti i test."""
    print("=" * 60)
    print("TEST CRITIC AGENT — paperhunter.agents.critic_agent")
    print("=" * 60)

    tests = [
        test_extract_citations_from_text,
        test_extract_claims_for_citation,
        test_parse_verification_response,
        test_parse_batch_verification_response,
        test_normalize_verification,
        test_format_source_text,
        test_build_verification_result,
        test_build_provenance_records,
        test_verify_section_mock,
        test_verify_section_no_citations,
        test_verify_section_max_iterations,
        test_verify_section_needs_revision,
        test_verify_section_already_verified,
        test_verify_section_not_found,
        test_get_verification_stats,
        test_verify_langgraph_node,
        test_verify_all_sections_mock,
        test_retrieve_source_chunks,
        test_package_imports,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: FALLITO — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RISULTATI: {passed}/{passed + failed} test passati")
    if failed > 0:
        print(f"ATTENZIONE: {failed} test falliti!")
    else:
        print("Tutti i test passati! ✓")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
