"""
Test suite per paperhunter.agents.tools.retrieval_tools

Copre:
- Ogni tool (search_vector_database, search_keyword_index, search_graph_communities)
  con mock backend
- Formattazione output (testo con [bibtex_key], troncamento)
- Gestione errori (backend non disponibile, eccezioni)
- Helper get_all_retrieval_tools() produce FunctionTool validi
- Singleton reset
- Limiti top_k (clamping a range valido)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ============================================================================
# Fixtures: mock data
# ============================================================================

MOCK_QDRANT_RESULTS = [
    {
        "chunk_id": "abc123",
        "text": "This paper introduces GCG, an adversarial suffix attack that generates universal adversarial prompts to bypass aligned language models.",
        "score": 0.891,
        "metadata": {
            "bibtex_key": "Zou_2023_GCG",
            "category": "Adversarial Code",
            "year": 2023,
            "section_name": "abstract",
            "chunk_type": "original",
            "paper_id": "paper_001",
        },
    },
    {
        "chunk_id": "def456",
        "text": "We propose AutoDAN, an automatic jailbreak method using hierarchical genetic algorithms.",
        "score": 0.845,
        "metadata": {
            "bibtex_key": "Liu_2024_AutoDAN",
            "category": "Adversarial Code",
            "year": 2024,
            "section_name": "introduction",
            "chunk_type": "original",
            "paper_id": "paper_002",
        },
    },
    {
        "chunk_id": "ghi789",
        "text": "A comprehensive survey on prompt injection vulnerabilities in large language models.",
        "score": 0.780,
        "metadata": {
            "bibtex_key": "Wei_2024_Survey",
            "category": "Prompt Injection",
            "year": 2024,
            "section_name": "abstract",
            "chunk_type": "l1_summary",
            "paper_id": "paper_003",
        },
    },
]

MOCK_BM25_RESULTS = [
    {
        "chunk_id": "bm25_001",
        "text": "The GCG suffix is optimized through greedy coordinate gradient descent on the token logits.",
        "score": 15.432,
        "metadata": {
            "bibtex_key": "Zou_2023_GCG",
            "category": "Adversarial Code",
            "year": 2023,
        },
    },
    {
        "chunk_id": "bm25_002",
        "text": "Experiments show that adversarial suffixes transfer across different models including GPT-4.",
        "score": 12.100,
        "metadata": {
            "bibtex_key": "Zou_2023_GCG",
            "category": "Adversarial Code",
            "year": 2023,
        },
    },
]

MOCK_COMMUNITY_RESULTS = [
    {
        "chunk_id": "comm_001",
        "text": "Community Summary: This community centers around adversarial attacks and prompt injection techniques. Key entities include GCG, AutoDAN, and jailbreak prompts.",
        "score": 0.823,
        "metadata": {
            "paper_id": "community_3",
            "category": "graph_community",
        },
    },
    {
        "chunk_id": "comm_002",
        "text": "Community Summary: This community focuses on defense mechanisms against adversarial inputs, including input sanitization and perplexity-based detection.",
        "score": 0.756,
        "metadata": {
            "paper_id": "community_7",
            "category": "graph_community",
        },
    },
]


# ============================================================================
# Fixture: auto-reset singleton
# ============================================================================

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset i singleton prima e dopo ogni test."""
    from paperhunter.agents.tools.retrieval_tools import reset_singletons
    reset_singletons()
    yield
    reset_singletons()


# ============================================================================
# Test: search_vector_database
# ============================================================================

class TestSearchVectorDatabase:
    """Test per il tool search_vector_database."""

    def test_basic_search(self):
        """Cerca nel vector database e verifica il formato output."""
        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_QDRANT_RESULTS

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_vector_database

            result = search_vector_database("adversarial prompt injection")

        assert "[Zou_2023_GCG]" in result
        assert "[Liu_2024_AutoDAN]" in result
        assert "[Wei_2024_Survey]" in result
        assert "score:" in result
        mock_store.query.assert_called_once()

    def test_with_category_filter(self):
        """Verifica che il filtro categoria venga passato correttamente."""
        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_QDRANT_RESULTS[:1]

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_vector_database

            result = search_vector_database(
                "GCG attack", category="Adversarial Code"
            )

        call_kwargs = mock_store.query.call_args
        assert call_kwargs.kwargs.get("where_filter") is not None or \
               (call_kwargs[1] if len(call_kwargs) > 1 else {}).get("where_filter") is not None
        assert "[Zou_2023_GCG]" in result

    def test_with_year_filter(self):
        """Verifica che il filtro anno minimo venga passato."""
        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_QDRANT_RESULTS[1:2]

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_vector_database

            result = search_vector_database(
                "jailbreak", year_min=2024
            )

        call_kwargs = mock_store.query.call_args
        where = call_kwargs.kwargs.get("where_filter")
        assert where is not None
        assert "year" in where
        assert "[Liu_2024_AutoDAN]" in result

    def test_with_combined_filters(self):
        """Verifica filtro categoria + anno simultaneo."""
        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_QDRANT_RESULTS[1:2]

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_vector_database

            result = search_vector_database(
                "automated jailbreak",
                category="Adversarial Code",
                year_min=2024,
            )

        where = mock_store.query.call_args.kwargs.get("where_filter")
        assert where is not None
        assert "category" in where
        assert "year" in where

    def test_empty_results(self):
        """Verifica gestione risultati vuoti."""
        mock_store = MagicMock()
        mock_store.query.return_value = []

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_vector_database

            result = search_vector_database("nonexistent topic xyz")

        assert "No results found" in result

    def test_error_handling(self):
        """Verifica gestione errori (Qdrant non disponibile)."""
        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            side_effect=RuntimeError("Qdrant connection failed"),
        ):
            from paperhunter.agents.tools.retrieval_tools import search_vector_database

            result = search_vector_database("test query")

        assert "Error" in result
        assert "RuntimeError" in result

    def test_top_k_clamping_max(self):
        """Verifica che top_k venga limitato a 10."""
        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_QDRANT_RESULTS

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_vector_database

            search_vector_database("test", top_k=50)

        call_kwargs = mock_store.query.call_args.kwargs
        assert call_kwargs["n_results"] <= 10

    def test_top_k_clamping_min(self):
        """Verifica che top_k venga limitato ad almeno 1."""
        mock_store = MagicMock()
        mock_store.query.return_value = MOCK_QDRANT_RESULTS[:1]

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_vector_database

            search_vector_database("test", top_k=-5)

        call_kwargs = mock_store.query.call_args.kwargs
        assert call_kwargs["n_results"] >= 1


# ============================================================================
# Test: search_keyword_index
# ============================================================================

class TestSearchKeywordIndex:
    """Test per il tool search_keyword_index."""

    def test_basic_search(self):
        """Ricerca BM25 con risultati validi."""
        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = MOCK_BM25_RESULTS

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_bm25_index",
            return_value=mock_bm25,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_keyword_index

            result = search_keyword_index("GCG adversarial suffix")

        assert "[Zou_2023_GCG]" in result
        assert "BM25:" in result
        mock_bm25.query.assert_called_once()

    def test_with_category_filter(self):
        """Verifica passaggio filtro categoria al BM25."""
        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = MOCK_BM25_RESULTS[:1]

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_bm25_index",
            return_value=mock_bm25,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_keyword_index

            result = search_keyword_index(
                "GCG suffix", category="Adversarial Code"
            )

        call_kwargs = mock_bm25.query.call_args
        where = call_kwargs.kwargs.get("where_filter")
        assert where is not None
        assert where.get("category") == "Adversarial Code"

    def test_bm25_not_available(self):
        """Verifica messaggio quando BM25 non è costruito."""
        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_bm25_index",
            return_value=None,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_keyword_index

            result = search_keyword_index("test query")

        assert "not available" in result

    def test_bm25_empty_results(self):
        """Verifica gestione risultati vuoti dal BM25."""
        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = []

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_bm25_index",
            return_value=mock_bm25,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_keyword_index

            result = search_keyword_index("nonexistent term")

        assert "No results found" in result

    def test_bm25_error_handling(self):
        """Verifica gestione errori dal BM25."""
        mock_bm25 = MagicMock()
        mock_bm25.query.side_effect = RuntimeError("Index corrupted")

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_bm25_index",
            return_value=mock_bm25,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_keyword_index

            result = search_keyword_index("test query")

        assert "Error" in result
        assert "RuntimeError" in result


# ============================================================================
# Test: search_graph_communities
# ============================================================================

class TestSearchGraphCommunities:
    """Test per il tool search_graph_communities."""

    def test_basic_search(self):
        """Ricerca community summary con risultati validi."""
        mock_store = MagicMock()
        mock_store.client.collection_exists.return_value = True
        mock_store.query.return_value = MOCK_COMMUNITY_RESULTS

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_graph_communities

            result = search_graph_communities("adversarial attacks")

        assert "community_3" in result
        assert "community_7" in result
        assert "relevance:" in result

    def test_collection_not_exists(self):
        """Verifica messaggio quando la collection graph_communities non esiste."""
        mock_store = MagicMock()
        mock_store.client.collection_exists.return_value = False

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_graph_communities

            result = search_graph_communities("test topic")

        assert "No graph community summaries available" in result
        mock_store.query.assert_not_called()

    def test_empty_results(self):
        """Verifica gestione risultati vuoti."""
        mock_store = MagicMock()
        mock_store.client.collection_exists.return_value = True
        mock_store.query.return_value = []

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_graph_communities

            result = search_graph_communities("nonexistent topic")

        assert "No graph community summaries found" in result

    def test_error_handling(self):
        """Verifica gestione errori nella ricerca community."""
        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            side_effect=ConnectionError("Qdrant unavailable"),
        ):
            from paperhunter.agents.tools.retrieval_tools import search_graph_communities

            result = search_graph_communities("test")

        assert "Error" in result
        assert "ConnectionError" in result

    def test_top_k_clamping(self):
        """Verifica che top_k sia limitato a 5."""
        mock_store = MagicMock()
        mock_store.client.collection_exists.return_value = True
        mock_store.query.return_value = MOCK_COMMUNITY_RESULTS

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ):
            from paperhunter.agents.tools.retrieval_tools import search_graph_communities

            search_graph_communities("test", top_k=20)

        call_kwargs = mock_store.query.call_args.kwargs
        assert call_kwargs["n_results"] <= 5


# ============================================================================
# Test: formatting helpers
# ============================================================================

class TestFormattingHelpers:
    """Test per le funzioni helper di formattazione."""

    def test_truncate_text_short(self):
        """Testo corto non viene troncato."""
        from paperhunter.agents.tools.retrieval_tools import _truncate_text

        text = "Short text"
        assert _truncate_text(text) == text

    def test_truncate_text_long(self):
        """Testo lungo viene troncato con '...'."""
        from paperhunter.agents.tools.retrieval_tools import _truncate_text

        text = "A" * 500
        result = _truncate_text(text, max_chars=100)
        assert len(result) <= 103  # 100 + "..."
        assert result.endswith("...")

    def test_format_qdrant_results_with_bibtex(self):
        """Verifica che il formato includa [bibtex_key]."""
        from paperhunter.agents.tools.retrieval_tools import _format_qdrant_results

        result = _format_qdrant_results(MOCK_QDRANT_RESULTS)
        assert "[Zou_2023_GCG]" in result
        assert "[Liu_2024_AutoDAN]" in result
        assert "score:" in result

    def test_format_qdrant_results_empty(self):
        """Risultati vuoti producono messaggio informativo."""
        from paperhunter.agents.tools.retrieval_tools import _format_qdrant_results

        result = _format_qdrant_results([])
        assert "No results found" in result

    def test_format_qdrant_results_truncation(self):
        """Verifica troncamento quando i risultati superano il budget."""
        from paperhunter.agents.tools.retrieval_tools import _format_qdrant_results

        # Crea risultati con testo molto lungo
        long_results = []
        for i in range(10):
            long_results.append({
                "chunk_id": f"chunk_{i}",
                "text": "A" * 1000,
                "score": 0.9 - i * 0.05,
                "metadata": {
                    "bibtex_key": f"Paper_{i}",
                    "category": "Test",
                    "year": 2024,
                },
            })

        result = _format_qdrant_results(long_results, max_total_chars=500)
        assert "truncated" in result

    def test_format_bm25_results_with_score(self):
        """Verifica che il formato BM25 includa lo score BM25."""
        from paperhunter.agents.tools.retrieval_tools import _format_bm25_results

        result = _format_bm25_results(MOCK_BM25_RESULTS)
        assert "BM25:" in result
        assert "[Zou_2023_GCG]" in result

    def test_format_community_results(self):
        """Verifica formattazione community summary."""
        from paperhunter.agents.tools.retrieval_tools import _format_community_results

        result = _format_community_results(MOCK_COMMUNITY_RESULTS)
        assert "community_3" in result
        assert "community_7" in result
        assert "relevance:" in result


# ============================================================================
# Test: get_all_retrieval_tools
# ============================================================================

class TestGetAllRetrievalTools:
    """Test per get_all_retrieval_tools()."""

    def test_returns_three_tools(self):
        """Verifica che vengano restituiti esattamente 3 tool."""
        from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools

        tools = get_all_retrieval_tools()
        assert len(tools) == 3

    def test_tools_are_function_tools(self):
        """Verifica che ogni tool sia un FunctionTool di LlamaIndex."""
        from llama_index.core.tools import FunctionTool
        from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools

        tools = get_all_retrieval_tools()
        for tool in tools:
            assert isinstance(tool, FunctionTool)

    def test_tool_names(self):
        """Verifica i nomi dei tool."""
        from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools

        tools = get_all_retrieval_tools()
        names = {t.metadata.name for t in tools}
        assert "search_vector_database" in names
        assert "search_keyword_index" in names
        assert "search_graph_communities" in names

    def test_tool_descriptions_not_empty(self):
        """Verifica che ogni tool abbia una description non vuota."""
        from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools

        tools = get_all_retrieval_tools()
        for tool in tools:
            assert tool.metadata.description
            assert len(tool.metadata.description) > 20

    def test_tools_have_callable_fn(self):
        """Verifica che ogni tool abbia una fn callable."""
        from paperhunter.agents.tools.retrieval_tools import get_all_retrieval_tools

        tools = get_all_retrieval_tools()
        for tool in tools:
            assert callable(tool.fn)


# ============================================================================
# Test: singleton management
# ============================================================================

class TestSingletons:
    """Test per la gestione dei singleton."""

    def test_reset_singletons(self):
        """Verifica che reset_singletons azzeri entrambi i backend."""
        import paperhunter.agents.tools.retrieval_tools as mod

        mod._qdrant_store_singleton = "fake_store"
        mod._bm25_index_singleton = "fake_bm25"

        mod.reset_singletons()

        assert mod._qdrant_store_singleton is None
        assert mod._bm25_index_singleton is None

    def test_qdrant_singleton_lazy_init(self):
        """Verifica che _get_qdrant_store() inizializzi il singleton lazy."""
        import paperhunter.agents.tools.retrieval_tools as mod

        mock_store_class = MagicMock()
        mock_store_instance = MagicMock()
        mock_store_class.return_value = mock_store_instance

        with patch(
            "paperhunter.agents.tools.retrieval_tools.QdrantStore",
            mock_store_class,
            create=True,
        ):
            with patch.dict(
                "sys.modules",
                {"paperhunter.indexing.qdrant_store": MagicMock(QdrantStore=mock_store_class)},
            ):
                store = mod._get_qdrant_store()

        assert store is mock_store_instance
        assert mod._qdrant_store_singleton is mock_store_instance


# ============================================================================
# Test: import smoke test
# ============================================================================

class TestImports:
    """Test di importazione del modulo."""

    def test_import_module(self):
        """Verifica che il modulo si importi correttamente."""
        import paperhunter.agents.tools.retrieval_tools as mod

        assert hasattr(mod, "search_vector_database")
        assert hasattr(mod, "search_keyword_index")
        assert hasattr(mod, "search_graph_communities")
        assert hasattr(mod, "get_all_retrieval_tools")
        assert hasattr(mod, "reset_singletons")

    def test_import_from_package(self):
        """Verifica importazione dal package tools."""
        from paperhunter.agents.tools import (
            get_all_retrieval_tools,
            search_graph_communities,
            search_keyword_index,
            search_vector_database,
        )

        assert callable(search_vector_database)
        assert callable(search_keyword_index)
        assert callable(search_graph_communities)
        assert callable(get_all_retrieval_tools)
