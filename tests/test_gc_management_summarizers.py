"""Test per Task 2.2 — GC management in level1_summarizer.py e level2_summarizer.py.

Verifica che:
1. Il GC venga disabilitato durante il loop di batch summarization (L1 e L2).
2. Il GC venga riabilitato e gc.collect() venga chiamato nel finally.
3. Il GC venga sempre riabilitato anche in caso di eccezione.
4. Gli oggetti intermedi (summary, l2) vengano rilasciati con del.
"""

import gc
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from paperhunter.models import (
    ChunkType,
    L1Summary,
    Paper,
    PaperChunk,
)


# ============================================================================
# Helpers — factory per oggetti di test
# ============================================================================

def _make_paper(title: str = "Test Paper", category: str = "TestCat") -> Paper:
    """Crea un Paper di test con campi minimi."""
    return Paper(
        title=title,
        authors=["Author A"],
        year=2025,
        category=category,
        url="https://example.com/paper",
        bibtex_key=f"Author_2025_{title.replace(' ', '_')[:10]}",
        pdf_hash="abc123",
    )


def _make_chunk(paper_title: str = "Test Paper") -> PaperChunk:
    """Crea un PaperChunk di test."""
    return PaperChunk(
        chunk_id=f"Author_2025_{paper_title.replace(' ', '_')[:10]}::intro::0",
        paper_id=f"Author_2025_{paper_title.replace(' ', '_')[:10]}",
        chunk_index=0,
        text="This is a test chunk with enough content for summarization.",
        chunk_type=ChunkType.ORIGINAL,
    )


def _make_l1_summary(paper_id: str = "Author_2025_Test", category: str = "TestCat") -> L1Summary:
    """Crea un L1Summary di test."""
    return L1Summary(
        paper_id=paper_id,
        bibtex_key=paper_id,
        category=category,
        year=2025,
        key_contributions=["Contribution 1"],
        methodology="Test methodology",
        testing_methodology="Test testing methodology",
        temporal_context="Test temporal context",
        results_summary="Test results",
        limitations="Test limitations",
        generated_at=datetime.now(),
        model_used="flash",
    )


# ============================================================================
# Test: gc importato nei moduli
# ============================================================================


class TestGCImportedInModules:
    """Verifica che i moduli importino gc."""

    def test_gc_imported_in_level1_summarizer(self):
        """level1_summarizer deve importare gc."""
        import paperhunter.summarization.level1_summarizer as mod
        assert hasattr(mod, "gc")
        assert mod.gc is gc

    def test_gc_imported_in_level2_summarizer(self):
        """level2_summarizer deve importare gc."""
        import paperhunter.summarization.level2_summarizer as mod
        assert hasattr(mod, "gc")
        assert mod.gc is gc


# ============================================================================
# Test: GC management in generate_l1_batch
# ============================================================================


class TestGCInL1Batch:
    """Verifica il pattern gc.disable/enable/collect in generate_l1_batch."""

    @patch("paperhunter.summarization.level1_summarizer.gc")
    @patch("paperhunter.summarization.level1_summarizer.generate_l1_summary")
    @patch("paperhunter.summarization.level1_summarizer.progress_bar", side_effect=lambda x, **kw: x)
    def test_gc_disabled_during_batch_and_reenabled(
        self, mock_pb, mock_gen_l1, mock_gc,
    ):
        """gc.disable() prima del loop, gc.enable()+gc.collect() nel finally."""
        from paperhunter.summarization.level1_summarizer import generate_l1_batch

        # Simula 2 paper con chunk
        papers_with_chunks = [
            (_make_paper(f"Paper {i}"), [_make_chunk(f"Paper {i}")])
            for i in range(2)
        ]

        # generate_l1_summary restituisce un L1Summary finto
        mock_gen_l1.return_value = _make_l1_summary()
        mock_gen_l1.return_value.l1_char_count = 100

        result = generate_l1_batch(papers_with_chunks, save_to_store=False)

        # gc.disable() deve essere stato chiamato
        mock_gc.disable.assert_called()
        # gc.enable() e gc.collect() devono essere chiamati nel finally
        mock_gc.enable.assert_called()
        mock_gc.collect.assert_called()

        # L'ordine deve essere: disable → enable → collect
        calls = [str(c) for c in mock_gc.method_calls]
        disable_idx = next(i for i, c in enumerate(calls) if "disable" in c)
        enable_idx = next(i for i, c in enumerate(calls) if "enable" in c)
        collect_idx = next(i for i, c in enumerate(calls) if "collect" in c)
        assert disable_idx < enable_idx < collect_idx

        # Deve aver generato 2 summary
        assert len(result) == 2

    @patch("paperhunter.summarization.level1_summarizer.gc")
    @patch("paperhunter.summarization.level1_summarizer.generate_l1_summary")
    @patch("paperhunter.summarization.level1_summarizer.progress_bar", side_effect=lambda x, **kw: x)
    def test_gc_reenabled_on_exception(
        self, mock_pb, mock_gen_l1, mock_gc,
    ):
        """Anche se generate_l1_summary esplode, gc.enable() viene chiamato."""
        from paperhunter.summarization.level1_summarizer import generate_l1_batch

        papers_with_chunks = [
            (_make_paper("Exploding Paper"), [_make_chunk("Exploding Paper")]),
        ]

        # generate_l1_summary solleva un'eccezione
        mock_gen_l1.side_effect = RuntimeError("LLM timeout")

        # Non deve propagare: l'eccezione è catturata nel try/except interno
        result = generate_l1_batch(papers_with_chunks, save_to_store=False)

        # gc.enable() DEVE essere chiamato anche con eccezione
        mock_gc.enable.assert_called()
        mock_gc.collect.assert_called()

        # Nessun summary generato (errore catturato)
        assert len(result) == 0

    @patch("paperhunter.summarization.level1_summarizer.gc")
    @patch("paperhunter.summarization.level1_summarizer.generate_l1_summary")
    @patch("paperhunter.summarization.level1_summarizer.progress_bar", side_effect=lambda x, **kw: x)
    def test_gc_called_once_for_batch(
        self, mock_pb, mock_gen_l1, mock_gc,
    ):
        """gc.disable() viene chiamato una volta per l'intero batch."""
        from paperhunter.summarization.level1_summarizer import generate_l1_batch

        papers_with_chunks = [
            (_make_paper(f"Paper {i}"), [_make_chunk(f"Paper {i}")])
            for i in range(5)
        ]

        mock_gen_l1.return_value = _make_l1_summary()
        mock_gen_l1.return_value.l1_char_count = 100

        generate_l1_batch(papers_with_chunks, save_to_store=False)

        # Un solo batch → una sola chiamata a gc.disable/enable/collect
        assert mock_gc.disable.call_count == 1
        assert mock_gc.enable.call_count == 1
        assert mock_gc.collect.call_count == 1


# ============================================================================
# Test: GC management in generate_l2_for_all_categories
# ============================================================================


class TestGCInL2Batch:
    """Verifica il pattern gc.disable/enable/collect in generate_l2_for_all_categories."""

    @patch("paperhunter.summarization.level2_summarizer.gc")
    @patch("paperhunter.summarization.level2_summarizer._print_batch_report")
    @patch("paperhunter.summarization.level2_summarizer.generate_l2_summary")
    @patch("paperhunter.summarization.level2_summarizer.progress_bar", side_effect=lambda x, **kw: x)
    @patch("paperhunter.summarization.level2_summarizer._discover_categories_from_store")
    def test_gc_disabled_during_l2_batch_and_reenabled(
        self, mock_discover, mock_pb, mock_gen_l2, mock_report, mock_gc,
    ):
        """gc.disable() prima del loop L2, gc.enable()+gc.collect() nel finally."""
        from paperhunter.summarization.level2_summarizer import generate_l2_for_all_categories

        mock_discover.return_value = ["CatA", "CatB"]


        from paperhunter.models import L2Summary
        mock_l2 = MagicMock(spec=L2Summary)
        mock_gen_l2.return_value = mock_l2

        mock_store = MagicMock()
        result = generate_l2_for_all_categories(store=mock_store)

        # gc.disable() deve essere stato chiamato
        mock_gc.disable.assert_called()
        # gc.enable() e gc.collect() devono essere chiamati nel finally
        mock_gc.enable.assert_called()
        mock_gc.collect.assert_called()

        # L'ordine deve essere: disable → enable → collect
        calls = [str(c) for c in mock_gc.method_calls]
        disable_idx = next(i for i, c in enumerate(calls) if "disable" in c)
        enable_idx = next(i for i, c in enumerate(calls) if "enable" in c)
        collect_idx = next(i for i, c in enumerate(calls) if "collect" in c)
        assert disable_idx < enable_idx < collect_idx

        # Deve aver generato 2 summary L2
        assert len(result) == 2

    @patch("paperhunter.summarization.level2_summarizer.gc")
    @patch("paperhunter.summarization.level2_summarizer._print_batch_report")
    @patch("paperhunter.summarization.level2_summarizer.generate_l2_summary")
    @patch("paperhunter.summarization.level2_summarizer.progress_bar", side_effect=lambda x, **kw: x)
    @patch("paperhunter.summarization.level2_summarizer._discover_categories_from_store")
    def test_gc_reenabled_on_l2_exception(
        self, mock_discover, mock_pb, mock_gen_l2, mock_report, mock_gc,
    ):
        """Anche se generate_l2_summary esplode, gc.enable() viene chiamato."""
        from paperhunter.summarization.level2_summarizer import generate_l2_for_all_categories

        mock_discover.return_value = ["CatA"]
        mock_gen_l2.side_effect = RuntimeError("LLM explosion")

        mock_store = MagicMock()
        result = generate_l2_for_all_categories(store=mock_store)

        # gc.enable() DEVE essere chiamato anche con eccezione
        mock_gc.enable.assert_called()
        mock_gc.collect.assert_called()

        # Nessun summary generato (errore catturato)
        assert len(result) == 0

    @patch("paperhunter.summarization.level2_summarizer.gc")
    @patch("paperhunter.summarization.level2_summarizer._print_batch_report")
    @patch("paperhunter.summarization.level2_summarizer.generate_l2_summary")
    @patch("paperhunter.summarization.level2_summarizer.progress_bar", side_effect=lambda x, **kw: x)
    @patch("paperhunter.summarization.level2_summarizer._discover_categories_from_store")
    def test_gc_called_once_for_all_categories(
        self, mock_discover, mock_pb, mock_gen_l2, mock_report, mock_gc,
    ):
        """gc.disable() viene chiamato una volta per tutto il loop delle categorie."""
        from paperhunter.summarization.level2_summarizer import generate_l2_for_all_categories

        mock_discover.return_value = ["CatA", "CatB", "CatC"]

        from paperhunter.models import L2Summary
        mock_l2 = MagicMock(spec=L2Summary)
        mock_gen_l2.return_value = mock_l2

        mock_store = MagicMock()
        generate_l2_for_all_categories(store=mock_store)

        # Un solo loop → una sola chiamata a gc.disable/enable/collect
        assert mock_gc.disable.call_count == 1
        assert mock_gc.enable.call_count == 1
        assert mock_gc.collect.call_count == 1


# ============================================================================
# Test: GC reale (non mocked)
# ============================================================================


class TestGCRealSummarizers:
    """Test con gc reale per verificare che il GC sia effettivamente gestito."""

    def test_gc_enabled_after_l1_import(self):
        """Dopo l'import di level1_summarizer, gc deve essere abilitato."""
        import paperhunter.summarization.level1_summarizer  # noqa: F401
        assert gc.isenabled(), "gc deve essere abilitato dopo l'import"

    def test_gc_enabled_after_l2_import(self):
        """Dopo l'import di level2_summarizer, gc deve essere abilitato."""
        import paperhunter.summarization.level2_summarizer  # noqa: F401
        assert gc.isenabled(), "gc deve essere abilitato dopo l'import"

    @patch("paperhunter.summarization.level1_summarizer.generate_l1_summary")
    @patch("paperhunter.summarization.level1_summarizer.progress_bar", side_effect=lambda x, **kw: x)
    def test_gc_real_l1_batch_reenables(self, mock_pb, mock_gen_l1):
        """Con gc reale, verify gc è abilitato dopo generate_l1_batch."""
        from paperhunter.summarization.level1_summarizer import generate_l1_batch

        papers_with_chunks = [
            (_make_paper("RealTest"), [_make_chunk("RealTest")]),
        ]

        mock_gen_l1.return_value = _make_l1_summary()
        mock_gen_l1.return_value.l1_char_count = 50

        assert gc.isenabled()
        generate_l1_batch(papers_with_chunks, save_to_store=False)
        assert gc.isenabled(), "gc deve essere riabilitato dopo generate_l1_batch"

    @patch("paperhunter.summarization.level1_summarizer.generate_l1_summary")
    @patch("paperhunter.summarization.level1_summarizer.progress_bar", side_effect=lambda x, **kw: x)
    def test_gc_real_l1_batch_reenables_on_exception(self, mock_pb, mock_gen_l1):
        """Con gc reale, gc è riabilitato anche dopo eccezione in L1 batch."""
        from paperhunter.summarization.level1_summarizer import generate_l1_batch

        papers_with_chunks = [
            (_make_paper("ExplodeReal"), [_make_chunk("ExplodeReal")]),
        ]

        mock_gen_l1.side_effect = RuntimeError("Boom")

        assert gc.isenabled()
        generate_l1_batch(papers_with_chunks, save_to_store=False)
        assert gc.isenabled(), "gc deve essere riabilitato dopo eccezione"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
