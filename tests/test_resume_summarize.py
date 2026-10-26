"""Test per verificare che `paperhunter summarize` riprenda correttamente
la generazione L1.5 parziale, rilevi L1 stale, e gestisca L2 correttamente.
"""

import argparse
import logging
from unittest.mock import MagicMock, patch

import pytest


# ─── Helpers per costruire args finti ───────────────────────────────────

def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        l1=False, l1_5=False, l2=False, invalidate=False,
        force=False, category=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ─── Test: _run_l1_5_summarization ──────────────────────────────────────

class TestL15Resume:
    """Verifica che la logica di resume L1.5 confronti cluster attesi vs esistenti."""

    def _call_l1_5(self, existing_count, expected_clusters, force=False,
                   l1_summaries=None, clusters=None, l1_stale=False):
        """Chiama _run_l1_5_summarization con mock controllati."""
        from paperhunter.main import _run_l1_5_summarization

        store = MagicMock()
        args = _make_args(force=force)
        logger = logging.getLogger("test_resume")

        # Mock scroll per restituire una categoria finta
        fake_point = MagicMock()
        fake_point.payload = {"category": "TestCategory"}
        store.client.scroll.return_value = ([fake_point], None)

        # Mock L1 summaries
        if l1_summaries is None:
            l1_summaries = [MagicMock() for _ in range(10)]

        # Mock cluster results
        if clusters is None:
            clusters = [["c"] for _ in range(expected_clusters)]

        with patch("paperhunter.summarization.load_l1_summaries_for_category", return_value=l1_summaries), \
             patch("paperhunter.summarization.clustering_utils.fetch_l1_embeddings_from_qdrant", return_value={}), \
             patch("paperhunter.summarization.clustering_utils.cluster_summaries", return_value=clusters), \
             patch("paperhunter.summarization.count_l1_5_for_category", return_value=existing_count), \
             patch("paperhunter.main._is_l1_5_stale", return_value=l1_stale), \
             patch("paperhunter.main._save_l1_5_state"), \
             patch("paperhunter.main._invalidate_l2_metadata"), \
             patch("paperhunter.summarization.generate_all_cluster_summaries", return_value=[]) as mock_gen:
            errors, new_generated = _run_l1_5_summarization(store, args, logger)
            return errors, new_generated, mock_gen

    def test_all_complete_skips_generation(self):
        """Se existing >= expected_clusters e L1 non stale, NON deve chiamare generate."""
        errors, new_gen, mock_gen = self._call_l1_5(
            existing_count=5, expected_clusters=5, l1_stale=False
        )
        assert errors == 0
        assert new_gen is False
        mock_gen.assert_not_called()

    def test_partial_resumes_generation(self):
        """Se existing < expected_clusters (parziale), DEVE chiamare generate."""
        errors, new_gen, mock_gen = self._call_l1_5(
            existing_count=3, expected_clusters=5
        )
        assert errors == 0
        mock_gen.assert_called_once()
        # Deve passare skip_existing=True (non force, non stale)
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("skip_existing") is True

    def test_zero_existing_generates_all(self):
        """Se existing == 0, genera tutti i cluster."""
        errors, new_gen, mock_gen = self._call_l1_5(
            existing_count=0, expected_clusters=5
        )
        assert errors == 0
        mock_gen.assert_called_once()

    def test_force_regenerates_even_if_complete(self):
        """Con --force, deve rigenerare anche se tutti già presenti."""
        errors, new_gen, mock_gen = self._call_l1_5(
            existing_count=5, expected_clusters=5, force=True
        )
        assert errors == 0
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("skip_existing") is False

    def test_stale_l1_forces_regeneration(self):
        """Se L1 è cambiato (stale), deve rigenerare anche se count ok."""
        errors, new_gen, mock_gen = self._call_l1_5(
            existing_count=5, expected_clusters=5, l1_stale=True
        )
        assert errors == 0
        mock_gen.assert_called_once()
        # skip_existing=False perché l1_stale=True
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("skip_existing") is False


# ─── Test: cmd_summarize (L2 sempre chiamato) ──────────────────────────

class TestCmdSummarizeResume:
    """Verifica che cmd_summarize chiami sempre L2 (staleness rilevata da _l2_exists_and_valid)."""

    def test_l2_always_called_no_force(self):
        """Senza --force, L2 deve essere comunque chiamato (skip_existing default)."""
        from paperhunter.main import cmd_summarize

        args = _make_args()  # Nessun flag → genera tutti

        with patch("paperhunter.indexing.QdrantStore"), \
             patch("paperhunter.main._get_logger"), \
             patch("paperhunter.main._run_l1_summarization", return_value=0), \
             patch("paperhunter.main._run_l1_5_summarization", return_value=(0, True)), \
             patch("paperhunter.main._run_l2_summarization", return_value=0) as mock_l2:

            result = cmd_summarize(args)

            assert result == 0
            mock_l2.assert_called_once()

    def test_l2_called_even_without_new_l1_5(self):
        """L2 deve essere chiamato anche se nessun nuovo L1.5 (staleness check in _l2_exists_and_valid)."""
        from paperhunter.main import cmd_summarize

        args = _make_args()

        with patch("paperhunter.indexing.QdrantStore"), \
             patch("paperhunter.main._get_logger"), \
             patch("paperhunter.main._run_l1_summarization", return_value=0), \
             patch("paperhunter.main._run_l1_5_summarization", return_value=(0, False)), \
             patch("paperhunter.main._run_l2_summarization", return_value=0) as mock_l2:

            result = cmd_summarize(args)

            assert result == 0
            mock_l2.assert_called_once()

    def test_force_overrides_everything(self):
        """Con --force, cmd_summarize chiama L2 normalmente."""
        from paperhunter.main import cmd_summarize

        args = _make_args(force=True)

        with patch("paperhunter.indexing.QdrantStore"), \
             patch("paperhunter.main._get_logger"), \
             patch("paperhunter.main._run_l1_summarization", return_value=0), \
             patch("paperhunter.main._run_l1_5_summarization", return_value=(0, False)), \
             patch("paperhunter.main._run_l2_summarization", return_value=0) as mock_l2:

            result = cmd_summarize(args)

            assert result == 0
            mock_l2.assert_called_once()


# ─── Test: _run_l2_summarization skip_existing ──────────────────────────

class TestL2SkipExisting:
    """Verifica che _run_l2_summarization onori il parametro skip_existing."""

    def test_skip_existing_default_from_force(self):
        """Senza skip_existing esplicito, usa not force."""
        from paperhunter.main import _run_l2_summarization
        from paperhunter.summarization import generate_l2_for_all_categories

        store = MagicMock()
        args = _make_args(force=False)
        logger = logging.getLogger("test")

        with patch("paperhunter.summarization.generate_l2_for_all_categories", return_value=[]) as mock_gen:
            _run_l2_summarization(store, args, logger)
            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["skip_existing"] is True

    def test_skip_existing_override_false(self):
        """Con skip_existing=False, passa False anche senza --force."""
        from paperhunter.main import _run_l2_summarization

        store = MagicMock()
        args = _make_args(force=False)
        logger = logging.getLogger("test")

        with patch("paperhunter.summarization.generate_l2_for_all_categories", return_value=[]) as mock_gen:
            _run_l2_summarization(store, args, logger, skip_existing=False)
            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["skip_existing"] is False
