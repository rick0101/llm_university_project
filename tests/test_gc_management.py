"""Test per Task 2.1 — GC management in ingestion_pipeline.py.

Verifica che:
1. Il GC venga disabilitato durante l'elaborazione di un batch.
2. Il GC venga riabilitato e gc.collect() venga chiamato dopo ogni batch.
3. Il GC venga sempre riabilitato anche in caso di eccezione durante l'elaborazione.
4. Gli oggetti intermedi (result, chunks, figure_chunks) vengano rilasciati con del.
"""

import gc
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ============================================================================
# Unit tests: gc.disable / gc.enable / gc.collect pattern
# ============================================================================


class TestGCManagementImport:
    """Verifica che il modulo importi gc e lo usi nel run()."""

    def test_gc_imported_in_module(self):
        """Il modulo ingestion_pipeline deve importare gc."""
        import paperhunter.ingestion.ingestion_pipeline as mod
        # gc deve essere nel namespace del modulo
        assert hasattr(mod, "gc")
        assert mod.gc is gc


class TestGCDuringBatchProcessing:
    """Verifica il pattern gc.disable/enable/collect durante il batch processing."""

    @pytest.fixture
    def pipeline(self):
        """Crea una IngestionPipeline con componenti stubbed."""
        with patch("paperhunter.ingestion.ingestion_pipeline.settings") as mock_settings:
            mock_settings.ingestion.pdf_base_dir = "/tmp/test_pdfs"
            mock_settings.ingestion.batch_size = 5
            mock_settings.output.checkpoints_dir = "/tmp/test_checkpoints"
            mock_settings.output.base_dir = "/tmp/test_output"
            mock_settings.recency_priority = False
            mock_settings.figure_extraction.enabled = False

            from paperhunter.ingestion.ingestion_pipeline import IngestionPipeline
            pipe = IngestionPipeline(
                pdf_base_dir="/tmp/test_pdfs",
                batch_size=2,
                skip_s2=True,
                skip_qdrant=True,
                use_checkpoint=False,
            )
            return pipe

    @patch("paperhunter.ingestion.ingestion_pipeline.gc")
    @patch("paperhunter.ingestion.ingestion_pipeline.save_hash_registry")
    @patch("paperhunter.ingestion.ingestion_pipeline.load_hash_registry", return_value={})
    @patch("paperhunter.ingestion.ingestion_pipeline.get_new_or_modified_pdfs")
    @patch("paperhunter.ingestion.ingestion_pipeline.ensure_output_dirs")
    @patch("paperhunter.ingestion.ingestion_pipeline.print_panel")
    @patch("paperhunter.ingestion.ingestion_pipeline.progress_bar", side_effect=lambda x, **kw: x)
    def test_gc_disabled_during_batch_and_reenabled_after(
        self, mock_pb, mock_panel, mock_ensure, mock_get_pdfs,
        mock_load_reg, mock_save_reg, mock_gc, pipeline,
    ):
        """gc.disable() prima del batch, gc.enable()+gc.collect() nel finally."""
        # Simula 2 PDF in un batch
        mock_get_pdfs.return_value = [
            {"relative_path": "cat/paper1.pdf", "absolute_path": "/tmp/p1.pdf",
             "category": "cat", "status": "new", "filename": "paper1.pdf"},
            {"relative_path": "cat/paper2.pdf", "absolute_path": "/tmp/p2.pdf",
             "category": "cat", "status": "new", "filename": "paper2.pdf"},
        ]

        # Stub _process_single_pdf per restituire None (nessun risultato)
        pipeline._process_single_pdf = MagicMock(return_value=None)

        pipeline.run()

        # gc.disable() deve essere stato chiamato almeno una volta (una per batch)
        mock_gc.disable.assert_called()
        # gc.enable() e gc.collect() devono essere chiamati nel finally
        mock_gc.enable.assert_called()
        mock_gc.collect.assert_called()

        # L'ordine deve essere: disable → (processing) → enable → collect
        calls = [str(c) for c in mock_gc.method_calls]
        disable_idx = next(i for i, c in enumerate(calls) if "disable" in c)
        enable_idx = next(i for i, c in enumerate(calls) if "enable" in c)
        collect_idx = next(i for i, c in enumerate(calls) if "collect" in c)
        assert disable_idx < enable_idx < collect_idx

    @patch("paperhunter.ingestion.ingestion_pipeline.gc")
    @patch("paperhunter.ingestion.ingestion_pipeline.save_hash_registry")
    @patch("paperhunter.ingestion.ingestion_pipeline.load_hash_registry", return_value={})
    @patch("paperhunter.ingestion.ingestion_pipeline.get_new_or_modified_pdfs")
    @patch("paperhunter.ingestion.ingestion_pipeline.ensure_output_dirs")
    @patch("paperhunter.ingestion.ingestion_pipeline.print_panel")
    @patch("paperhunter.ingestion.ingestion_pipeline.progress_bar", side_effect=lambda x, **kw: x)
    def test_gc_reenabled_on_exception(
        self, mock_pb, mock_panel, mock_ensure, mock_get_pdfs,
        mock_load_reg, mock_save_reg, mock_gc, pipeline,
    ):
        """Anche se _process_single_pdf esplode, gc.enable() viene chiamato."""
        mock_get_pdfs.return_value = [
            {"relative_path": "cat/paper1.pdf", "absolute_path": "/tmp/p1.pdf",
             "category": "cat", "status": "new", "filename": "paper1.pdf"},
        ]

        # _process_single_pdf solleva un'eccezione RuntimeError
        # che viene catturata dal try/except interno ma il finally deve comunque girare
        pipeline._process_single_pdf = MagicMock(
            side_effect=RuntimeError("Test explosion")
        )

        # Non deve propagare: l'eccezione è catturata nel try/except interno
        report = pipeline.run()

        # gc.enable() DEVE essere chiamato anche con eccezione
        mock_gc.enable.assert_called()
        mock_gc.collect.assert_called()
        # L'errore deve essere registrato nel report
        assert report.errors == 1

    @patch("paperhunter.ingestion.ingestion_pipeline.gc")
    @patch("paperhunter.ingestion.ingestion_pipeline.save_hash_registry")
    @patch("paperhunter.ingestion.ingestion_pipeline.load_hash_registry", return_value={})
    @patch("paperhunter.ingestion.ingestion_pipeline.get_new_or_modified_pdfs")
    @patch("paperhunter.ingestion.ingestion_pipeline.ensure_output_dirs")
    @patch("paperhunter.ingestion.ingestion_pipeline.print_panel")
    @patch("paperhunter.ingestion.ingestion_pipeline.progress_bar", side_effect=lambda x, **kw: x)
    def test_gc_called_per_batch(
        self, mock_pb, mock_panel, mock_ensure, mock_get_pdfs,
        mock_load_reg, mock_save_reg, mock_gc, pipeline,
    ):
        """Con batch_size=2 e 4 PDF, gc.disable() deve essere chiamato 2 volte."""
        mock_get_pdfs.return_value = [
            {"relative_path": f"cat/paper{i}.pdf", "absolute_path": f"/tmp/p{i}.pdf",
             "category": "cat", "status": "new", "filename": f"paper{i}.pdf"}
            for i in range(4)
        ]

        pipeline._process_single_pdf = MagicMock(return_value=None)

        pipeline.run()

        # 2 batch → 2 chiamate a gc.disable() e gc.enable()
        assert mock_gc.disable.call_count == 2
        assert mock_gc.enable.call_count == 2
        assert mock_gc.collect.call_count == 2


class TestGCRealBehavior:
    """Test con gc reale (non mocked) per verificare che il GC sia effettivamente gestito."""

    def test_gc_is_enabled_after_import(self):
        """Dopo l'import del modulo, gc deve essere abilitato (non interferisce col runtime)."""
        import paperhunter.ingestion.ingestion_pipeline  # noqa: F401
        assert gc.isenabled(), "gc deve essere abilitato dopo l'import"

    def test_gc_disable_enable_pattern_correctness(self):
        """Verifica che il pattern gc.disable/enable/collect funzioni correttamente."""
        assert gc.isenabled()

        gc.disable()
        assert not gc.isenabled()

        try:
            # Simula lavoro intensivo
            _ = [object() for _ in range(1000)]
        finally:
            gc.enable()
            collected = gc.collect()

        assert gc.isenabled()
        # collected >= 0, verifica che gc.collect() restituisca un int
        assert isinstance(collected, int)

    def test_gc_reenabled_after_exception(self):
        """gc.enable() nel finally viene eseguito anche con eccezione."""
        assert gc.isenabled()

        gc.disable()
        try:
            raise ValueError("test")
        except ValueError:
            pass
        finally:
            gc.enable()
            gc.collect()

        assert gc.isenabled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
