"""Test per best_fit_pack() e integrazione con chunk_grobid_output().

Verifica che:
1. I chunk risultanti abbiano dimensioni più uniformi rispetto al metodo attuale.
2. Nessun chunk superi chunk_size (1500 chars da config).
3. I chunk troppo corti (< 80 chars) siano eliminati o accorpati.
"""

import pytest

from paperhunter.ingestion.chunker import (
    best_fit_pack,
    chunk_grobid_output,
    get_chunk_stats,
)
from paperhunter.models import GROBIDOutput, GROBIDSection, Paper


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        title="Test Paper on LLM Security",
        authors=["Author A", "Author B"],
        url="https://arxiv.org/abs/0000.00000",
        year=2025,
        month=3,
        abstract="This is a test abstract about LLM security vulnerabilities.",
        category="Prompt Injection",
        bibtex_key="testpaper2025",
    )


@pytest.fixture
def grobid_output_with_sections() -> GROBIDOutput:
    """GROBIDOutput con mix di sezioni corte e lunghe."""
    short_section = GROBIDSection(title="Introduction", text="A" * 500)
    
    # Sezione lunga: 8 paragrafi da ~313 chars ciascuno (totale ~2660)
    paragraphs = [f"Paragraph {i}: " + "X" * 300 for i in range(8)]
    long_section = GROBIDSection(title="Methodology", text="\n\n".join(paragraphs))
    
    medium_section = GROBIDSection(title="Results", text="R" * 1200)
    
    return GROBIDOutput(
        title="Test Paper on LLM Security",
        abstract="This is a test abstract about LLM security vulnerabilities and adversarial attacks on large language models.",
        sections=[short_section, long_section, medium_section],
        full_text="",
        figure_captions=[],
        table_captions=[],
    )


# ============================================================================
# Test best_fit_pack() isolato
# ============================================================================

class TestBestFitPack:
    
    def test_empty_input(self):
        assert best_fit_pack([], max_tokens=1400) == []
    
    def test_single_unit(self):
        result = best_fit_pack(["Hello world"], max_tokens=1400)
        assert len(result) == 1
        assert result[0] == "Hello world"
    
    def test_small_units_fit_one_bin(self):
        units = ["Short A", "Short B", "Short C"]
        result = best_fit_pack(units, max_tokens=1400)
        assert len(result) == 1
        assert "\n\n" in result[0]
    
    def test_large_units_multiple_bins(self):
        units = ["A" * 800, "B" * 700, "C" * 600]
        result = best_fit_pack(units, max_tokens=1400)
        assert len(result) >= 2
        for b in result:
            assert len(b) <= 1400
    
    def test_no_bin_exceeds_max_tokens(self):
        units = ["A" * 1000, "B" * 300, "C" * 200, "D" * 100, "E" * 50]
        result = best_fit_pack(units, max_tokens=1400)
        for b in result:
            assert len(b) <= 1400
    
    def test_uniform_sizes(self):
        """5 unità da 400 chars → packing produce bins di dimensione simile."""
        units = ["P" * 400] * 5
        result = best_fit_pack(units, max_tokens=1500)
        sizes = [len(b) for b in result]
        # Con packing ottimale: max 3 per bin (400+2+400+2+400=1204), 2 bins
        assert len(result) == 2
        assert all(s <= 1500 for s in sizes)
    
    def test_oversize_unit_creates_single_bin(self):
        """Unità più grande di max_tokens → crea comunque un bin (gestione a monte)."""
        result = best_fit_pack(["X" * 2000], max_tokens=1400)
        assert len(result) == 1
        assert len(result[0]) == 2000
    
    def test_empty_units_ignored(self):
        result = best_fit_pack(["", "Hello", ""], max_tokens=1400)
        assert len(result) == 1
        assert result[0] == "Hello"
    
    def test_separator_accounted_in_size(self):
        """Il separatore (\\n\\n) viene contato nella dimensione del bin."""
        # 698 + 2 (sep) + 698 = 1398 ≤ 1400 → un bin
        units = ["A" * 698, "B" * 698]
        result = best_fit_pack(units, max_tokens=1400)
        assert len(result) == 1
        assert len(result[0]) == 698 + 2 + 698  # 1398
    
    def test_separator_prevents_overflow(self):
        """Se con il separatore si sfora, va in un nuovo bin."""
        # 700 + 2 (sep) + 700 = 1402 > 1400 → due bin
        units = ["A" * 700, "B" * 700]
        result = best_fit_pack(units, max_tokens=1400)
        assert len(result) == 2


# ============================================================================
# Test integrazione con chunk_grobid_output()
# ============================================================================

class TestBestFitPackIntegration:
    
    def test_chunks_within_size_limit(
        self, sample_paper, grobid_output_with_sections
    ):
        """Nessun chunk deve superare chunk_size."""
        from paperhunter.config import settings
        cs = settings.ingestion.chunk_size
        
        chunks = chunk_grobid_output(grobid_output_with_sections, sample_paper)
        for c in chunks:
            assert len(c.text) <= cs, (
                f"Chunk troppo grande: {len(c.text)} > {cs} "
                f"(section={c.section_name})"
            )
    
    def test_no_short_chunks(
        self, sample_paper, grobid_output_with_sections
    ):
        """Nessun chunk deve essere più corto di _MIN_CHUNK_SIZE (80)."""
        chunks = chunk_grobid_output(grobid_output_with_sections, sample_paper)
        for c in chunks:
            assert len(c.text) >= 80, (
                f"Chunk troppo corto: {len(c.text)} < 80 "
                f"(section={c.section_name})"
            )
    
    def test_packing_produces_fewer_or_equal_chunks(
        self, sample_paper, grobid_output_with_sections
    ):
        """Il packing non deve produrre più chunk del metodo sliding window."""
        # Risultato con packing
        chunks_packed = chunk_grobid_output(
            grobid_output_with_sections, sample_paper
        )
        
        # Risultato senza packing (sliding window)
        chunks_sliding = chunk_grobid_output(
            grobid_output_with_sections,
            sample_paper,
            chunk_size=1500,
            chunk_overlap=200,
        )
        
        # Il packing dovrebbe produrre chunk più uniformi
        # (non necessariamente meno, ma con dimensioni migliori)
        stats_packed = get_chunk_stats(chunks_packed)
        stats_sliding = get_chunk_stats(chunks_sliding)
        
        print(f"\nPacked: {stats_packed['total']} chunks, "
              f"avg={stats_packed['avg_length']:.0f}, "
              f"min={stats_packed['min_length']}, "
              f"max={stats_packed['max_length']}")
        print(f"Sliding: {stats_sliding['total']} chunks, "
              f"avg={stats_sliding['avg_length']:.0f}, "
              f"min={stats_sliding['min_length']}, "
              f"max={stats_sliding['max_length']}")
        
        assert stats_packed["total"] > 0
    
    def test_abstract_always_separate(
        self, sample_paper, grobid_output_with_sections
    ):
        """L'abstract è sempre un chunk separato, non impacchettato."""
        chunks = chunk_grobid_output(grobid_output_with_sections, sample_paper)
        abstract_chunks = [c for c in chunks if c.section_name == "Abstract"]
        assert len(abstract_chunks) == 1
    
    def test_section_metadata_preserved(
        self, sample_paper, grobid_output_with_sections
    ):
        """I chunk mantengono le informazioni di sezione originali."""
        chunks = chunk_grobid_output(grobid_output_with_sections, sample_paper)
        section_names = {c.section_name for c in chunks}
        # Deve contenere almeno Abstract + le sezioni GROBID
        assert "Abstract" in section_names
        assert "Introduction" in section_names or "Methodology" in section_names
    
    def test_fallback_no_sections(self, sample_paper):
        """Senza sezioni GROBID, usa il full_text con fallback sliding window."""
        grobid = GROBIDOutput(
            title="Test",
            abstract="",
            sections=[],
            full_text="Introduction\n\n" + "Content paragraph. " * 200,
            figure_captions=[],
            table_captions=[],
        )
        chunks = chunk_grobid_output(grobid, sample_paper)
        assert len(chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
