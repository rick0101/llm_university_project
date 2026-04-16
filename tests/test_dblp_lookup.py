"""Test per il modulo DBLP lookup e l'integrazione con bibtex_generator."""
import re
import pytest
from unittest.mock import patch, MagicMock

from paperhunter.ingestion.dblp_lookup import (
    DblpLookup,
    DblpResult,
    _normalize_title,
    _title_similarity,
    _strip_dblp_fields,
    _cache_key,
)
from paperhunter.ingestion.bibtex_generator import (
    generate_bibtex_entry,
    enrich_paper_bibtex,
    ensure_complete_bibtex,
)
from paperhunter.models import Paper


# ---------------------------------------------------------------------------
# Unit test: normalizzazione e similarità titoli
# ---------------------------------------------------------------------------

class TestTitleNormalization:
    def test_normalize_basic(self):
        assert _normalize_title("Attention Is All You Need") == "attention is all you need"

    def test_normalize_punctuation(self):
        assert _normalize_title("Hello: World! (2023)") == "hello world 2023"

    def test_normalize_accents(self):
        assert _normalize_title("Résumé über Naïveté") == "resume uber naivete"

    def test_normalize_extra_spaces(self):
        assert _normalize_title("  too   many   spaces  ") == "too many spaces"

    def test_similarity_identical(self):
        assert _title_similarity("Hello World", "Hello World") == 1.0

    def test_similarity_close(self):
        sim = _title_similarity(
            "Attention Is All You Need",
            "Attention is All you Need.",
        )
        assert sim > 0.95

    def test_similarity_different(self):
        sim = _title_similarity(
            "Attention Is All You Need",
            "Completely Different Paper Title",
        )
        assert sim < 0.5


# ---------------------------------------------------------------------------
# Unit test: strip campi DBLP
# ---------------------------------------------------------------------------

class TestStripDblpFields:
    def test_strip_fields(self):
        bibtex = """@inproceedings{key,
  title = {Test},
  author = {Author},
  timestamp = {Thu, 21 Jan 2021 15:15:21 +0100},
  biburl = {https://dblp.org/rec/conf/test.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}"""
        result = _strip_dblp_fields(bibtex)
        assert "timestamp" not in result
        assert "biburl" not in result
        assert "bibsource" not in result
        assert "title = {Test}" in result
        assert "author = {Author}" in result


# ---------------------------------------------------------------------------
# Unit test: cache key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_same_inputs_same_key(self):
        k1 = _cache_key("Test Title", 2023, "Smith")
        k2 = _cache_key("Test Title", 2023, "Smith")
        assert k1 == k2

    def test_different_year_different_key(self):
        k1 = _cache_key("Test Title", 2023, "Smith")
        k2 = _cache_key("Test Title", 2024, "Smith")
        assert k1 != k2

    def test_different_author_different_key(self):
        k1 = _cache_key("Test Title", 2023, "Smith")
        k2 = _cache_key("Test Title", 2023, "Jones")
        assert k1 != k2


# ---------------------------------------------------------------------------
# Integration: DBLP non sovrascrive @inproceedings con @misc
# ---------------------------------------------------------------------------

class TestDblpPreservation:
    """Verifica che entry DBLP @inproceedings non vengano degradate a @misc."""

    def test_dblp_inproceedings_preserved_with_arxiv_id(self):
        """Paper pubblicato in conferenza con arxiv_id deve restare @inproceedings."""
        dblp_entry = """@inproceedings{Vaswani_2017_Attention,
  author = {Ashish Vaswani and Noam Shazeer},
  title = {Attention is All you Need},
  booktitle = {Advances in Neural Information Processing Systems 30},
  pages = {5998--6008},
  year = {2017},
  url = {https://proceedings.neurips.cc/paper/2017/hash/abc-Abstract.html}
}"""
        paper = Paper(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            url="https://arxiv.org/abs/1706.03762",
            year=2017,
            arxiv_id="1706.03762",
            bibtex_key="Vaswani_2017_Attention",
            bibtex_entry=dblp_entry,
            bibtex_source="dblp",
        )
        result = ensure_complete_bibtex(paper)
        assert "@inproceedings" in result.bibtex_entry
        assert "@misc" not in result.bibtex_entry

    def test_local_entry_with_arxiv_becomes_misc(self):
        """Paper locale con arxiv_id e senza DOI deve diventare @misc."""
        paper = Paper(
            title="Some Preprint Paper",
            authors=["Test Author"],
            url="https://arxiv.org/abs/2301.12345",
            year=2023,
            arxiv_id="2301.12345",
            bibtex_key="Author_2023_Some",
            bibtex_source="local",
        )
        result = ensure_complete_bibtex(paper) if paper.bibtex_entry else enrich_paper_bibtex(paper)
        # Con arxiv_id e senza DOI, dovrebbe essere @misc
        assert "@misc" in result.bibtex_entry


# ---------------------------------------------------------------------------
# Integration: generate_bibtex_entry con mock DBLP
# ---------------------------------------------------------------------------

class TestGenerateBibtexWithDblp:
    """Test generate_bibtex_entry con DBLP mockato."""

    @patch("paperhunter.ingestion.bibtex_generator.get_dblp_lookup")
    def test_uses_dblp_when_available(self, mock_get_dblp):
        """Se DBLP trova il paper, usa il BibTeX ufficiale."""
        mock_lookup = MagicMock()
        mock_lookup.lookup_bibtex.return_value = DblpResult(
            dblp_key="conf/nips/VaswaniSPUJGKP17",
            bibtex='@inproceedings{DBLP:conf/nips/VaswaniSPUJGKP17,\n  author = {Ashish Vaswani},\n  title = {Attention is All you Need},\n  booktitle = {NeurIPS 2017},\n  year = {2017}\n}',
            dblp_type="Conference and Workshop Papers",
            title="Attention is All you Need",
            similarity=0.98,
        )
        mock_get_dblp.return_value = mock_lookup

        paper = Paper(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            url="https://arxiv.org/abs/1706.03762",
            year=2017,
        )

        entry = generate_bibtex_entry(paper)
        assert "@inproceedings" in entry
        assert "NeurIPS 2017" in entry
        # La chiave deve essere stata sostituita con la nostra
        assert "DBLP:conf/nips" not in entry

    @patch("paperhunter.ingestion.bibtex_generator.get_dblp_lookup")
    def test_falls_back_when_dblp_fails(self, mock_get_dblp):
        """Se DBLP non trova nulla, usa il fallback locale."""
        mock_lookup = MagicMock()
        mock_lookup.lookup_bibtex.return_value = None
        mock_get_dblp.return_value = mock_lookup

        paper = Paper(
            title="My Unpublished Paper",
            authors=["Test Author"],
            url="https://arxiv.org/abs/2301.99999",
            year=2023,
            arxiv_id="2301.99999",
        )

        entry = generate_bibtex_entry(paper)
        assert entry  # Non vuoto
        assert "My Unpublished Paper" in entry

    @patch("paperhunter.ingestion.bibtex_generator.get_dblp_lookup")
    def test_falls_back_on_exception(self, mock_get_dblp):
        """Se DBLP lancia eccezione, usa il fallback locale."""
        mock_lookup = MagicMock()
        mock_lookup.lookup_bibtex.side_effect = Exception("Network error")
        mock_get_dblp.return_value = mock_lookup

        paper = Paper(
            title="Test Paper",
            authors=["Author"],
            url="https://example.com",
            year=2023,
        )

        entry = generate_bibtex_entry(paper)
        assert entry  # Non vuoto, fallback ha funzionato
        assert "Test Paper" in entry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
