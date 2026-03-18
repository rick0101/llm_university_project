"""Test per postprocessing/provenance_tracker.py

Tutti i test sono offline (nessuna API call).
Verifica: creazione record, import da stato, report JSON, statistiche,
lookup, persistenza, markdown, merge, edge cases.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from paperhunter.postprocessing.provenance_tracker import (
    ProvenanceTracker,
    _compute_section_stats,
    _normalize_status,
    _record_to_dict,
    create_provenance_from_state,
    load_provenance,
)
from paperhunter.models import ProvenanceRecord, VerificationStatus


# ============================================================================
# Helper: sample records
# ============================================================================

def _make_sample_records() -> list[dict]:
    """Crea 10 record di esempio simili a quelli prodotti dal Critic Agent."""
    return [
        # Sezione 1: Prompt Injection — 4 record
        {
            "section_key": "prompt_injection_attacks",
            "citation_key": "Zou_2023_GCG",
            "claim_text": "GCG generates adversarial suffixes that transfer across models.",
            "source_chunk_id": "chunk_abc123",
            "source_text_excerpt": "We propose Greedy Coordinate Gradient (GCG), an optimization-based approach...",
            "verification_status": "SUPPORTED",
            "confidence": 0.95,
            "reasoning": "The claim matches the source description of GCG.",
            "verified_at": "2026-02-26T10:00:00",
        },
        {
            "section_key": "prompt_injection_attacks",
            "citation_key": "Greshake_2023_LLM",
            "claim_text": "Indirect prompt injection exploits LLM-integrated applications.",
            "source_chunk_id": "chunk_def456",
            "source_text_excerpt": "We demonstrate how an attacker can control an LLM...",
            "verification_status": "SUPPORTED",
            "confidence": 0.90,
            "reasoning": "Directly supported by the source.",
        },
        {
            "section_key": "prompt_injection_attacks",
            "citation_key": "Lee_2025_PromptShield",
            "claim_text": "PromptShield achieves 99% detection rate on known attacks.",
            "source_chunk_id": "chunk_ghi789",
            "source_text_excerpt": "PromptShield detects prompt injections with high accuracy...",
            "verification_status": "PARTIALLY_SUPPORTED",
            "confidence": 0.65,
            "reasoning": "The exact 99% figure is not stated in the source, but high accuracy is confirmed.",
        },
        {
            "section_key": "prompt_injection_attacks",
            "citation_key": "FakeRef_2024_NotReal",
            "claim_text": "All LLMs are vulnerable to prompt injection.",
            "source_chunk_id": None,
            "source_text_excerpt": "",
            "verification_status": "SOURCE_NOT_FOUND",
            "confidence": 0.0,
            "reasoning": "No source chunk found for this citation.",
        },
        # Sezione 2: Data Poisoning — 3 record
        {
            "section_key": "data_poisoning",
            "citation_key": "Wan_2023_Poisoning",
            "claim_text": "Language model fine-tuning can be compromised via data poisoning.",
            "source_chunk_id": "chunk_jkl012",
            "source_text_excerpt": "We show that poisoning the fine-tuning data...",
            "verification_status": "SUPPORTED",
            "confidence": 0.92,
            "reasoning": "Directly supported.",
        },
        {
            "section_key": "data_poisoning",
            "citation_key": "Li_2023_Backdoorllm",
            "claim_text": "BackdoorLLM provides a comprehensive benchmark for backdoor attacks.",
            "source_chunk_id": "chunk_mno345",
            "source_text_excerpt": "BackdoorLLM includes a standardized evaluation framework...",
            "verification_status": "SUPPORTED",
            "confidence": 0.88,
            "reasoning": "The benchmark description matches.",
        },
        {
            "section_key": "data_poisoning",
            "citation_key": "Unknown_2024_Invented",
            "claim_text": "Poisoning attacks are trivially detectable.",
            "source_chunk_id": "chunk_pqr678",
            "source_text_excerpt": "Detection of data poisoning remains challenging...",
            "verification_status": "NOT_SUPPORTED",
            "confidence": 0.15,
            "reasoning": "The source contradicts the claim.",
        },
        # Sezione 3: Hallucinations — 3 record
        {
            "section_key": "llm_hallucinations",
            "citation_key": "Li_2023_Halueval",
            "claim_text": "HaluEval provides a large-scale hallucination evaluation benchmark.",
            "source_chunk_id": "chunk_stu901",
            "source_text_excerpt": "HaluEval contains 35,000 hallucinated samples...",
            "verification_status": "SUPPORTED",
            "confidence": 0.93,
            "reasoning": "Directly supported with quantitative details.",
        },
        {
            "section_key": "llm_hallucinations",
            "citation_key": "Zhang_2024_AI",
            "claim_text": "LLM hallucinations are a critical safety concern in medical domains.",
            "source_chunk_id": "chunk_vwx234",
            "source_text_excerpt": "Hallucinations in medical AI systems can lead to...",
            "verification_status": "PARTIALLY_SUPPORTED",
            "confidence": 0.70,
            "reasoning": "Medical context mentioned but safety framing is broader.",
        },
        {
            "section_key": "llm_hallucinations",
            "citation_key": "Zou_2023_GCG",
            "claim_text": "GCG has been shown to trigger hallucinations in some models.",
            "source_chunk_id": "chunk_yza567",
            "source_text_excerpt": "GCG primarily targets safety alignment...",
            "verification_status": "NOT_SUPPORTED",
            "confidence": 0.20,
            "reasoning": "GCG focuses on jailbreaking, not hallucination triggering.",
        },
    ]


# ============================================================================
# Test 1: _normalize_status
# ============================================================================

class TestNormalizeStatus:
    """Test della normalizzazione status."""

    def test_standard_values(self):
        assert _normalize_status("SUPPORTED") == VerificationStatus.SUPPORTED
        assert _normalize_status("PARTIALLY_SUPPORTED") == VerificationStatus.PARTIALLY_SUPPORTED
        assert _normalize_status("NOT_SUPPORTED") == VerificationStatus.NOT_SUPPORTED
        assert _normalize_status("SOURCE_NOT_FOUND") == VerificationStatus.SOURCE_NOT_FOUND

    def test_case_insensitive(self):
        assert _normalize_status("supported") == VerificationStatus.SUPPORTED
        assert _normalize_status("Supported") == VerificationStatus.SUPPORTED

    def test_variants(self):
        assert _normalize_status("partial") == VerificationStatus.PARTIALLY_SUPPORTED
        assert _normalize_status("partially supported") == VerificationStatus.PARTIALLY_SUPPORTED
        assert _normalize_status("unsupported") == VerificationStatus.NOT_SUPPORTED
        assert _normalize_status("not found") == VerificationStatus.SOURCE_NOT_FOUND
        assert _normalize_status("missing") == VerificationStatus.SOURCE_NOT_FOUND

    def test_unknown_defaults_to_not_found(self):
        assert _normalize_status("garbage") == VerificationStatus.SOURCE_NOT_FOUND
        assert _normalize_status("") == VerificationStatus.SOURCE_NOT_FOUND


# ============================================================================
# Test 2: _record_to_dict
# ============================================================================

class TestRecordToDict:
    """Test serializzazione record."""

    def test_basic_serialization(self):
        record = ProvenanceRecord(
            section="test_section",
            claim_text="A test claim.",
            citation_key="Author_2023_Paper",
            source_chunk_id="chunk_123",
            source_text_excerpt="Some source text...",
            verification_status=VerificationStatus.SUPPORTED,
            confidence=0.85,
        )
        d = _record_to_dict(record)
        assert d["section"] == "test_section"
        assert d["claim"] == "A test claim."
        assert d["citation_key"] == "Author_2023_Paper"
        assert d["source_chunk_id"] == "chunk_123"
        assert d["verification"] == "SUPPORTED"
        assert d["confidence"] == 0.85

    def test_none_chunk_id(self):
        record = ProvenanceRecord(
            section="sec",
            citation_key="key",
        )
        d = _record_to_dict(record)
        assert d["source_chunk_id"] == ""


# ============================================================================
# Test 3: _compute_section_stats
# ============================================================================

class TestComputeSectionStats:
    """Test statistiche per sezione."""

    def test_empty_list(self):
        stats = _compute_section_stats([])
        assert stats["total"] == 0
        assert stats["coverage_rate"] == 0.0

    def test_mixed_statuses(self):
        records = [
            ProvenanceRecord(
                section="s", citation_key="k1",
                verification_status=VerificationStatus.SUPPORTED,
                confidence=0.9,
            ),
            ProvenanceRecord(
                section="s", citation_key="k2",
                verification_status=VerificationStatus.PARTIALLY_SUPPORTED,
                confidence=0.6,
            ),
            ProvenanceRecord(
                section="s", citation_key="k3",
                verification_status=VerificationStatus.NOT_SUPPORTED,
                confidence=0.2,
            ),
            ProvenanceRecord(
                section="s", citation_key="k4",
                verification_status=VerificationStatus.SOURCE_NOT_FOUND,
                confidence=0.0,
            ),
        ]
        stats = _compute_section_stats(records)
        assert stats["total"] == 4
        assert stats["supported"] == 1
        assert stats["partially_supported"] == 1
        assert stats["not_supported"] == 1
        assert stats["source_not_found"] == 1
        assert stats["coverage_rate"] == 0.5  # 2/4
        assert stats["support_rate"] == 0.25  # 1/4
        assert stats["unique_citations"] == 4
        assert stats["avg_confidence"] == pytest.approx(0.425, abs=0.01)


# ============================================================================
# Test 4: ProvenanceTracker — add_record
# ============================================================================

class TestAddRecord:
    """Test aggiunta record singoli."""

    def test_add_single(self):
        tracker = ProvenanceTracker(version="1.0")
        r = tracker.add_record(
            section="intro",
            claim="LLMs are vulnerable.",
            citation_key="Smith_2023_Survey",
            chunk_id="c1",
            source_text="Source text here.",
            verification_status="SUPPORTED",
            confidence=0.88,
        )
        assert tracker.record_count == 1
        assert r.section == "intro"
        assert r.citation_key == "Smith_2023_Survey"
        assert r.verification_status == VerificationStatus.SUPPORTED
        assert r.confidence == 0.88

    def test_add_with_enum_status(self):
        tracker = ProvenanceTracker()
        r = tracker.add_record(
            section="s",
            claim="claim",
            citation_key="k",
            verification_status=VerificationStatus.NOT_SUPPORTED,
            confidence=0.1,
        )
        assert r.verification_status == VerificationStatus.NOT_SUPPORTED

    def test_confidence_clamping(self):
        tracker = ProvenanceTracker()
        r1 = tracker.add_record(
            section="s", claim="c", citation_key="k",
            confidence=1.5,
        )
        r2 = tracker.add_record(
            section="s", claim="c", citation_key="k",
            confidence=-0.5,
        )
        assert r1.confidence == 1.0
        assert r2.confidence == 0.0

    def test_claim_truncation(self):
        tracker = ProvenanceTracker()
        long_claim = "x" * 2000
        r = tracker.add_record(
            section="s", claim=long_claim, citation_key="k",
        )
        assert len(r.claim_text) <= 1000


# ============================================================================
# Test 5: ProvenanceTracker — add_records_batch
# ============================================================================

class TestAddRecordsBatch:
    """Test aggiunta batch di record."""

    def test_batch_from_critic_format(self):
        tracker = ProvenanceTracker()
        records = _make_sample_records()
        added = tracker.add_records_batch(records)
        assert added == 10
        assert tracker.record_count == 10

    def test_batch_with_invalid_records(self):
        tracker = ProvenanceTracker()
        records = [
            {"section": "s", "claim": "c", "citation_key": "k"},
            {"bad_format": True},  # Questo dovrebbe essere gestito
        ]
        added = tracker.add_records_batch(records)
        # Entrambi dovrebbero funzionare (il secondo ha default)
        assert added >= 1


# ============================================================================
# Test 6: ProvenanceTracker — import_from_state
# ============================================================================

class TestImportFromState:
    """Test import da SurveyState."""

    def test_import_with_provenance(self):
        state = {
            "provenance": _make_sample_records(),
            "version_info": {"version_number": "2.0"},
        }
        tracker = ProvenanceTracker(version="2.0")
        imported = tracker.import_from_state(state)
        assert imported == 10
        assert tracker.record_count == 10

    def test_import_empty_state(self):
        state = {"provenance": []}
        tracker = ProvenanceTracker()
        imported = tracker.import_from_state(state)
        assert imported == 0

    def test_import_missing_provenance(self):
        state = {}
        tracker = ProvenanceTracker()
        imported = tracker.import_from_state(state)
        assert imported == 0


# ============================================================================
# Test 7: ProvenanceTracker — get_stats
# ============================================================================

class TestGetStats:
    """Test statistiche aggregate."""

    def test_empty_tracker(self):
        tracker = ProvenanceTracker()
        stats = tracker.get_stats()
        assert stats["total_claims"] == 0
        assert stats["coverage_rate"] == 0.0
        assert stats["top_cited_papers"] == []

    def test_full_stats(self):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())
        stats = tracker.get_stats()

        assert stats["total_claims"] == 10
        # SUPPORTED: Zou, Greshake, Wan, Li_Backdoorllm, Li_Halueval = 5
        assert stats["supported"] == 5
        # PARTIALLY: Lee, Zhang = 2
        assert stats["partially_supported"] == 2
        # NOT_SUPPORTED: Unknown_2024, Zou (in hallucinations) = 2
        assert stats["not_supported"] == 2
        # SOURCE_NOT_FOUND: FakeRef = 1
        assert stats["source_not_found"] == 1
        # verified = supported + partially = 7
        assert stats["verified_claims"] == 7
        assert stats["unverified_claims"] == 3
        assert stats["coverage_rate"] == pytest.approx(0.7, abs=0.01)
        assert stats["unique_sections"] == 3
        # Zou_2023_GCG appare 2 volte → deve essere nel top
        assert any(
            p["citation_key"] == "Zou_2023_GCG" and p["claim_count"] == 2
            for p in stats["top_cited_papers"]
        )

    def test_section_stats(self):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())
        pi_stats = tracker.get_section_stats("prompt_injection_attacks")
        assert pi_stats["total"] == 4
        assert pi_stats["supported"] == 2
        assert pi_stats["partially_supported"] == 1
        assert pi_stats["source_not_found"] == 1


# ============================================================================
# Test 8: ProvenanceTracker — lookup e query
# ============================================================================

class TestLookup:
    """Test funzioni di lookup."""

    def test_get_records_by_section(self):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())
        pi_records = tracker.get_records_by_section("prompt_injection_attacks")
        assert len(pi_records) == 4
        assert all(r.section == "prompt_injection_attacks" for r in pi_records)

    def test_get_records_by_citation(self):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())
        zou_records = tracker.get_records_by_citation("Zou_2023_GCG")
        assert len(zou_records) == 2  # Appare in PI e Hallucinations

    def test_get_records_by_status(self):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())

        supported = tracker.get_records_by_status("SUPPORTED")
        assert len(supported) == 5

        supported_enum = tracker.get_records_by_status(VerificationStatus.SUPPORTED)
        assert len(supported_enum) == 5

    def test_get_unsupported_claims(self):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())
        unsupported = tracker.get_unsupported_claims()
        # NOT_SUPPORTED: 2 + SOURCE_NOT_FOUND: 1 = 3
        assert len(unsupported) == 3

    def test_get_records_nonexistent_section(self):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())
        records = tracker.get_records_by_section("nonexistent")
        assert records == []


# ============================================================================
# Test 9: ProvenanceTracker — generate_report
# ============================================================================

class TestGenerateReport:
    """Test generazione report completo."""

    def test_report_structure(self):
        tracker = ProvenanceTracker(version="1.5")
        tracker.add_records_batch(_make_sample_records())
        report = tracker.generate_report()

        assert report["schema_version"] == "1.0"
        assert report["survey_version"] == "1.5"
        assert "generated_at" in report
        assert "created_at" in report
        assert report["total_claims"] == 10
        assert report["verified_claims"] == 7
        assert report["unverified_claims"] == 3
        assert "statistics" in report
        assert "sections" in report
        assert len(report["sections"]) == 3  # 3 sezioni

    def test_report_sections_content(self):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())
        report = tracker.generate_report()

        section_keys = [s["section"] for s in report["sections"]]
        assert "prompt_injection_attacks" in section_keys
        assert "data_poisoning" in section_keys
        assert "llm_hallucinations" in section_keys

        # Verifica che ogni sezione abbia records e statistics
        for section in report["sections"]:
            assert "statistics" in section
            assert "records" in section
            assert len(section["records"]) > 0

    def test_empty_report(self):
        tracker = ProvenanceTracker()
        report = tracker.generate_report()
        assert report["total_claims"] == 0
        assert report["sections"] == []


# ============================================================================
# Test 10: ProvenanceTracker — save_json e load_json
# ============================================================================

class TestPersistenceJson:
    """Test persistenza JSON (save + load round-trip)."""

    def test_save_and_load_roundtrip(self):
        tracker = ProvenanceTracker(version="2.0")
        tracker.add_records_batch(_make_sample_records())

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "provenance.json"
            saved_path = tracker.save_json(json_path)
            assert saved_path.exists()

            # Verifica contenuto JSON
            data = json.loads(saved_path.read_text(encoding="utf-8"))
            assert data["total_claims"] == 10
            assert data["survey_version"] == "2.0"
            assert len(data["sections"]) == 3

            # Ricarica
            loaded = ProvenanceTracker.load_json(json_path)
            assert loaded.record_count == 10
            assert loaded.version == "2.0"

            # Statistiche devono corrispondere
            orig_stats = tracker.get_stats()
            loaded_stats = loaded.get_stats()
            assert orig_stats["total_claims"] == loaded_stats["total_claims"]
            assert orig_stats["supported"] == loaded_stats["supported"]
            assert orig_stats["not_supported"] == loaded_stats["not_supported"]

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            ProvenanceTracker.load_json("/tmp/nonexistent_provenance.json")

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("[1, 2, 3]")
            f.flush()
            with pytest.raises(ValueError, match="atteso dict"):
                ProvenanceTracker.load_json(f.name)

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "a" / "b" / "c" / "provenance.json"
            tracker = ProvenanceTracker()
            tracker.add_record(
                section="s", claim="c", citation_key="k",
                verification_status="SUPPORTED",
            )
            saved = tracker.save_json(deep_path)
            assert saved.exists()


# ============================================================================
# Test 11: ProvenanceTracker — save_markdown
# ============================================================================

class TestPersistenceMarkdown:
    """Test generazione report Markdown."""

    def test_markdown_content(self):
        tracker = ProvenanceTracker(version="1.0")
        tracker.add_records_batch(_make_sample_records())

        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "report.md"
            saved = tracker.save_markdown(md_path)
            assert saved.exists()

            content = saved.read_text(encoding="utf-8")
            assert "# Provenance Report" in content
            assert "v1.0" in content
            assert "SUPPORTED" in content
            assert "prompt_injection_attacks" in content
            assert "Zou_2023_GCG" in content
            assert "Claim Problematici" in content

    def test_empty_markdown(self):
        tracker = ProvenanceTracker()
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "empty.md"
            saved = tracker.save_markdown(md_path)
            content = saved.read_text(encoding="utf-8")
            assert "Claim totali:** 0" in content


# ============================================================================
# Test 12: ProvenanceTracker — merge
# ============================================================================

class TestMerge:
    """Test merge di due tracker."""

    def test_merge_new_records(self):
        t1 = ProvenanceTracker()
        t1.add_record(
            section="s1", claim="claim A", citation_key="k1",
            verification_status="SUPPORTED", confidence=0.9,
        )

        t2 = ProvenanceTracker()
        t2.add_record(
            section="s2", claim="claim B", citation_key="k2",
            verification_status="SUPPORTED", confidence=0.8,
        )

        added = t1.merge(t2)
        assert added == 1
        assert t1.record_count == 2

    def test_merge_duplicates_skipped(self):
        t1 = ProvenanceTracker()
        t1.add_record(
            section="s1", claim="same claim", citation_key="k1",
            verification_status="SUPPORTED", confidence=0.5,
        )

        t2 = ProvenanceTracker()
        t2.add_record(
            section="s1", claim="same claim", citation_key="k1",
            verification_status="SUPPORTED", confidence=0.9,
        )

        added = t1.merge(t2)
        assert added == 0  # Duplicato
        assert t1.record_count == 1
        # Confidence aggiornata al più alto
        assert t1.records[0].confidence == 0.9

    def test_merge_empty(self):
        t1 = ProvenanceTracker()
        t1.add_record(
            section="s", claim="c", citation_key="k",
        )
        t2 = ProvenanceTracker()
        added = t1.merge(t2)
        assert added == 0
        assert t1.record_count == 1


# ============================================================================
# Test 13: ProvenanceTracker — clear
# ============================================================================

class TestClear:
    """Test pulizia record."""

    def test_clear(self):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())
        assert tracker.record_count == 10
        tracker.clear()
        assert tracker.record_count == 0
        assert tracker.records == []


# ============================================================================
# Test 14: create_provenance_from_state
# ============================================================================

class TestCreateFromState:
    """Test convenience function create_provenance_from_state."""

    def test_from_state_with_version(self):
        state = {
            "provenance": _make_sample_records()[:3],
            "version_info": {"version_number": "3.0"},
        }
        tracker = create_provenance_from_state(state)
        assert tracker.record_count == 3
        assert tracker.version == "3.0"

    def test_from_state_with_explicit_version(self):
        state = {
            "provenance": _make_sample_records()[:2],
            "version_info": {"version_number": "1.0"},
        }
        tracker = create_provenance_from_state(state, version="5.0")
        assert tracker.version == "5.0"

    def test_from_empty_state(self):
        tracker = create_provenance_from_state({})
        assert tracker.record_count == 0
        assert tracker.version == "1.0"


# ============================================================================
# Test 15: load_provenance shortcut
# ============================================================================

class TestLoadProvenance:
    """Test shortcut load_provenance."""

    def test_load_roundtrip(self):
        tracker = ProvenanceTracker(version="1.0")
        tracker.add_records_batch(_make_sample_records()[:5])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prov.json"
            tracker.save_json(path)

            loaded = load_provenance(path)
            assert loaded.record_count == 5
            assert loaded.version == "1.0"


# ============================================================================
# Test 16: __repr__ e __len__
# ============================================================================

class TestReprLen:
    """Test metodi speciali."""

    def test_repr(self):
        tracker = ProvenanceTracker(version="2.5")
        tracker.add_record(section="s", claim="c", citation_key="k")
        r = repr(tracker)
        assert "2.5" in r
        assert "1" in r

    def test_len(self):
        tracker = ProvenanceTracker()
        assert len(tracker) == 0
        tracker.add_record(section="s", claim="c", citation_key="k")
        assert len(tracker) == 1


# ============================================================================
# Test 17: print_summary (smoke test — non verifica output, solo no crash)
# ============================================================================

class TestPrintSummary:
    """Smoke test per print_summary (verificare che non crashi)."""

    def test_print_with_data(self, capsys):
        tracker = ProvenanceTracker()
        tracker.add_records_batch(_make_sample_records())
        tracker.print_summary()  # Non deve crashare

    def test_print_empty(self, capsys):
        tracker = ProvenanceTracker()
        tracker.print_summary()  # Non deve crashare


# ============================================================================
# Test 18: Edge cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases vari."""

    def test_default_status_is_source_not_found(self):
        tracker = ProvenanceTracker()
        r = tracker.add_record(section="s", claim="c", citation_key="k")
        assert r.verification_status == VerificationStatus.SOURCE_NOT_FOUND

    def test_records_property_returns_copy(self):
        tracker = ProvenanceTracker()
        tracker.add_record(section="s", claim="c", citation_key="k")
        records = tracker.records
        records.clear()
        assert tracker.record_count == 1  # Interno non modificato

    def test_version_property(self):
        tracker = ProvenanceTracker(version="99.9")
        assert tracker.version == "99.9"

    def test_multiple_sections_report(self):
        """Verifica che il report gestisca correttamente 5+ sezioni."""
        tracker = ProvenanceTracker()
        for i in range(5):
            tracker.add_record(
                section=f"section_{i}",
                claim=f"claim {i}",
                citation_key=f"key_{i}",
                verification_status="SUPPORTED",
                confidence=0.9,
            )
        stats = tracker.get_stats()
        assert stats["unique_sections"] == 5
        report = tracker.generate_report()
        assert len(report["sections"]) == 5


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
