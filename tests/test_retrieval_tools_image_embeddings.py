"""
Test suite per Task 3.1: Integrazione image_embeddings in search_figure_descriptions.

Copre:
- Ricerca combinata su figure_descriptions + image_embeddings
- Deduplicazione per paper_id + image_path (merge con score più alto)
- Source indicator [description], [image_embedding], [both]
- Fallback graceful se image_embeddings collection non esiste
- Parametro include_image_embeddings=True/False
- Logging statistiche (n_desc, n_img)
- Formattazione output con fonte
- _merge_and_deduplicate_figure_results helper
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Fixtures: mock data
# ============================================================================

MOCK_DESC_RESULTS = [
    {
        "chunk_id": "fig_desc_001",
        "text": "Bar chart showing attack success rate comparison across 5 models.",
        "score": 0.85,
        "metadata": {
            "bibtex_key": "Zou_2023_GCG",
            "category": "Adversarial Code",
            "year": 2023,
            "paper_id": "paper_001",
            "image_path": "output/figures/paper_001_fig1.png",
        },
    },
    {
        "chunk_id": "fig_desc_002",
        "text": "Architecture diagram of the defense pipeline with input sanitizer.",
        "score": 0.78,
        "metadata": {
            "bibtex_key": "Wei_2024_Defense",
            "category": "Prompt Injection",
            "year": 2024,
            "paper_id": "paper_002",
            "image_path": "output/figures/paper_002_fig3.png",
        },
    },
    {
        "chunk_id": "fig_desc_003",
        "text": "Confusion matrix for prompt classification results.",
        "score": 0.72,
        "metadata": {
            "bibtex_key": "Liu_2024_AutoDAN",
            "category": "Adversarial Code",
            "year": 2024,
            "paper_id": "paper_003",
            "image_path": "output/figures/paper_003_fig2.png",
        },
    },
]

MOCK_IMG_RESULTS = [
    {
        "chunk_id": "img_emb_001",
        "text": "Figure from paper_001",
        "score": 0.90,
        "metadata": {
            "bibtex_key": "Zou_2023_GCG",
            "category": "Adversarial Code",
            "year": 2023,
            "paper_id": "paper_001",
            "image_path": "output/figures/paper_001_fig1.png",
        },
    },
    {
        "chunk_id": "img_emb_004",
        "text": "Figure from paper_004",
        "score": 0.82,
        "metadata": {
            "bibtex_key": "Chen_2024_Poison",
            "category": "Data and Model Poisoning",
            "year": 2024,
            "paper_id": "paper_004",
            "image_path": "output/figures/paper_004_fig1.png",
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


def _make_mock_store(
    desc_exists: bool = True,
    img_exists: bool = True,
    desc_results: list | None = None,
    img_results: list | None = None,
):
    """Crea un mock QdrantStore con collection_exists e query configurabili."""
    mock_store = MagicMock()

    if desc_results is None:
        desc_results = MOCK_DESC_RESULTS
    if img_results is None:
        img_results = MOCK_IMG_RESULTS

    def mock_collection_exists(name):
        if name == "figure_descriptions":
            return desc_exists
        if name == "image_embeddings":
            return img_exists
        return False

    mock_store.client.collection_exists.side_effect = mock_collection_exists

    def mock_query(query_text, n_results=10, where_filter=None,
                   collection_name=None, score_threshold=None):
        if collection_name == "figure_descriptions":
            return desc_results[:n_results]
        return []

    mock_store.query.side_effect = mock_query

    def mock_query_img(query_text, n_results=10, where_filter=None,
                       score_threshold=None):
        return img_results[:n_results]

    mock_store.query_image_embeddings.side_effect = mock_query_img

    return mock_store


def _make_mock_settings():
    """Crea mock settings con collection names."""
    mock_settings = MagicMock()
    mock_settings.qdrant.collection_figure_descriptions = "figure_descriptions"
    mock_settings.qdrant.collection_image_embeddings = "image_embeddings"
    return mock_settings


# ============================================================================
# Test: _merge_and_deduplicate_figure_results
# ============================================================================

class TestMergeAndDeduplicate:
    """Test per l'helper _merge_and_deduplicate_figure_results."""

    def test_no_overlap(self):
        """Risultati senza overlap vengono tutti inclusi."""
        from paperhunter.agents.tools.retrieval_tools import (
            _merge_and_deduplicate_figure_results,
        )

        desc = [
            {"text": "desc A", "score": 0.8, "metadata": {"paper_id": "p1", "image_path": "a.png"}},
        ]
        img = [
            {"text": "img B", "score": 0.75, "metadata": {"paper_id": "p2", "image_path": "b.png"}},
        ]

        result = _merge_and_deduplicate_figure_results(desc, img, top_k=10)

        assert len(result) == 2
        # Ordinati per score
        assert result[0]["score"] == 0.8
        assert result[1]["score"] == 0.75
        assert result[0]["source"] == "description"
        assert result[1]["source"] == "image_embedding"

    def test_overlap_keeps_higher_score(self):
        """Per stessa immagine, mantiene lo score più alto e segna 'both'."""
        from paperhunter.agents.tools.retrieval_tools import (
            _merge_and_deduplicate_figure_results,
        )

        desc = [
            {"text": "desc A", "score": 0.7, "metadata": {"paper_id": "p1", "image_path": "a.png"}},
        ]
        img = [
            {"text": "img A", "score": 0.9, "metadata": {"paper_id": "p1", "image_path": "a.png"}},
        ]

        result = _merge_and_deduplicate_figure_results(desc, img, top_k=10)

        assert len(result) == 1
        assert result[0]["score"] == 0.9
        assert result[0]["source"] == "both"

    def test_overlap_desc_higher(self):
        """Se description ha score più alto, mantiene quello e segna 'both'."""
        from paperhunter.agents.tools.retrieval_tools import (
            _merge_and_deduplicate_figure_results,
        )

        desc = [
            {"text": "desc A", "score": 0.95, "metadata": {"paper_id": "p1", "image_path": "a.png"}},
        ]
        img = [
            {"text": "img A", "score": 0.6, "metadata": {"paper_id": "p1", "image_path": "a.png"}},
        ]

        result = _merge_and_deduplicate_figure_results(desc, img, top_k=10)

        assert len(result) == 1
        assert result[0]["score"] == 0.95
        assert result[0]["source"] == "both"

    def test_top_k_limit(self):
        """Risultati vengono troncati a top_k."""
        from paperhunter.agents.tools.retrieval_tools import (
            _merge_and_deduplicate_figure_results,
        )

        desc = [
            {"text": f"d{i}", "score": 0.9 - i * 0.1,
             "metadata": {"paper_id": f"p{i}", "image_path": f"{i}.png"}}
            for i in range(5)
        ]
        img = [
            {"text": f"i{i}", "score": 0.85 - i * 0.1,
             "metadata": {"paper_id": f"q{i}", "image_path": f"q{i}.png"}}
            for i in range(5)
        ]

        result = _merge_and_deduplicate_figure_results(desc, img, top_k=3)

        assert len(result) == 3
        # Ordinato per score decrescente
        assert result[0]["score"] >= result[1]["score"] >= result[2]["score"]

    def test_empty_desc(self):
        """Se desc vuoto, restituisce solo risultati da img."""
        from paperhunter.agents.tools.retrieval_tools import (
            _merge_and_deduplicate_figure_results,
        )

        img = [
            {"text": "img A", "score": 0.8, "metadata": {"paper_id": "p1", "image_path": "a.png"}},
        ]

        result = _merge_and_deduplicate_figure_results([], img, top_k=10)

        assert len(result) == 1
        assert result[0]["source"] == "image_embedding"

    def test_empty_img(self):
        """Se img vuoto, restituisce solo risultati da desc."""
        from paperhunter.agents.tools.retrieval_tools import (
            _merge_and_deduplicate_figure_results,
        )

        desc = [
            {"text": "desc A", "score": 0.8, "metadata": {"paper_id": "p1", "image_path": "a.png"}},
        ]

        result = _merge_and_deduplicate_figure_results(desc, [], top_k=10)

        assert len(result) == 1
        assert result[0]["source"] == "description"

    def test_both_empty(self):
        """Se entrambi vuoti, restituisce lista vuota."""
        from paperhunter.agents.tools.retrieval_tools import (
            _merge_and_deduplicate_figure_results,
        )

        result = _merge_and_deduplicate_figure_results([], [], top_k=10)
        assert len(result) == 0

    def test_mixed_overlap_and_unique(self):
        """Mix di risultati duplicati e unici: dedup corretto."""
        from paperhunter.agents.tools.retrieval_tools import (
            _merge_and_deduplicate_figure_results,
        )

        desc = [
            {"text": "shared", "score": 0.7, "metadata": {"paper_id": "p1", "image_path": "a.png"}},
            {"text": "unique desc", "score": 0.6, "metadata": {"paper_id": "p2", "image_path": "b.png"}},
        ]
        img = [
            {"text": "shared img", "score": 0.9, "metadata": {"paper_id": "p1", "image_path": "a.png"}},
            {"text": "unique img", "score": 0.5, "metadata": {"paper_id": "p3", "image_path": "c.png"}},
        ]

        result = _merge_and_deduplicate_figure_results(desc, img, top_k=10)

        assert len(result) == 3  # p1 dedup (both), p2, p3
        # p1 ha score 0.9 (il più alto)
        assert result[0]["score"] == 0.9
        assert result[0]["source"] == "both"

        sources = {r["source"] for r in result}
        assert "both" in sources


# ============================================================================
# Test: search_figure_descriptions con image embeddings
# ============================================================================

class TestSearchFigureDescriptionsWithImageEmbeddings:
    """Test per search_figure_descriptions con dual-collection search."""

    def test_combined_search_both_collections(self):
        """Cerca in entrambe le collection e restituisce risultati merged."""
        mock_store = _make_mock_store()

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.agents.tools.retrieval_tools.settings",
            _make_mock_settings(),
            create=True,
        ):
            # Patch anche l'import di settings dentro la funzione
            with patch(
                "paperhunter.config.settings",
                _make_mock_settings(),
            ):
                from paperhunter.agents.tools.retrieval_tools import (
                    search_figure_descriptions,
                )

                result = search_figure_descriptions(
                    "attack success rate", top_k=5
                )

        # Verifico che risultati da entrambe le collection siano presenti
        assert "[Zou_2023_GCG]" in result
        # paper_004 è solo in image_embeddings
        assert "[Chen_2024_Poison]" in result
        # Source indicators devono essere presenti
        assert "[both]" in result or "[description]" in result or "[image_embedding]" in result

    def test_include_image_embeddings_false(self):
        """Con include_image_embeddings=False, cerca solo in figure_descriptions."""
        mock_store = _make_mock_store()

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            result = search_figure_descriptions(
                "attack success rate",
                include_image_embeddings=False,
            )

        # Non deve cercare in image_embeddings
        mock_store.query_image_embeddings.assert_not_called()
        assert "[Zou_2023_GCG]" in result
        # paper_004 è solo in image_embeddings, non deve comparire
        assert "Chen_2024_Poison" not in result

    def test_fallback_no_image_embeddings_collection(self):
        """Se image_embeddings non esiste, restituisce solo figure_descriptions."""
        mock_store = _make_mock_store(img_exists=False)

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            result = search_figure_descriptions("architecture diagram")

        # Non deve crashare, restituisce solo desc
        mock_store.query_image_embeddings.assert_not_called()
        assert "[Wei_2024_Defense]" in result
        assert "[description]" in result

    def test_fallback_no_figure_descriptions_collection(self):
        """Se figure_descriptions non esiste ma image_embeddings sì, restituisce img."""
        mock_store = _make_mock_store(desc_exists=False)

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            result = search_figure_descriptions("attack plot")

        # Solo risultati da image_embeddings
        assert "[Chen_2024_Poison]" in result
        assert "[image_embedding]" in result

    def test_fallback_no_collections(self):
        """Se nessuna collection esiste, restituisce messaggio informativo."""
        mock_store = _make_mock_store(desc_exists=False, img_exists=False)

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            result = search_figure_descriptions("anything")

        assert "No figure descriptions available" in result

    def test_deduplication_in_combined_search(self):
        """Paper_001 appare in entrambe le collection: deduplicato con 'both'."""
        mock_store = _make_mock_store()

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            result = search_figure_descriptions("attack success rate")

        # paper_001 è in entrambe → [both]
        assert "[both]" in result
        # Il bibtex_key deve comparire esattamente una volta come citazione [N]
        # (può comparire anche in IMAGE_PATH/caption, ma come entry [N] solo 1 volta)
        lines_with_zou = [
            line for line in result.split("\n")
            if line.strip().startswith("[") and "[Zou_2023_GCG]" in line
        ]
        assert len(lines_with_zou) == 1

    def test_category_filter_passed_to_both(self):
        """Il filtro categoria viene passato a entrambe le collection."""
        mock_store = _make_mock_store()

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            search_figure_descriptions(
                "test", category="Adversarial Code"
            )

        # query per figure_descriptions
        desc_call = mock_store.query.call_args
        assert desc_call.kwargs.get("where_filter") == {"category": "Adversarial Code"}

        # query_image_embeddings
        img_call = mock_store.query_image_embeddings.call_args
        assert img_call.kwargs.get("where_filter") == {"category": "Adversarial Code"}

    def test_empty_results_both_collections(self):
        """Entrambe le collection esistono ma non hanno risultati."""
        mock_store = _make_mock_store(desc_results=[], img_results=[])

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            result = search_figure_descriptions("nonexistent figure")

        assert "No figure results found" in result

    def test_error_handling(self):
        """Errori durante la ricerca restituiscono messaggio informativo."""
        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            side_effect=ConnectionError("Qdrant unavailable"),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            result = search_figure_descriptions("test")

        assert "Error" in result
        assert "ConnectionError" in result

    def test_top_k_clamping(self):
        """top_k viene limitato al range [1, 10]."""
        mock_store = _make_mock_store()

        with patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            search_figure_descriptions("test", top_k=20)

        desc_call = mock_store.query.call_args
        assert desc_call.kwargs["n_results"] <= 10


# ============================================================================
# Test: _format_figure_results con source indicator
# ============================================================================

class TestFormatFigureResultsWithSource:
    """Test per _format_figure_results con campo source."""

    def test_source_description(self):
        """Risultato con source=description mostra [description]."""
        from paperhunter.agents.tools.retrieval_tools import _format_figure_results

        results = [{
            "text": "A chart of results.",
            "score": 0.85,
            "source": "description",
            "metadata": {
                "bibtex_key": "Test_2024",
                "category": "Test",
                "year": 2024,
                "image_path": "test.png",
            },
        }]

        output = _format_figure_results(results)
        assert "[description]" in output
        assert "[Test_2024]" in output

    def test_source_image_embedding(self):
        """Risultato con source=image_embedding mostra [image_embedding]."""
        from paperhunter.agents.tools.retrieval_tools import _format_figure_results

        results = [{
            "text": "Image embedding figure.",
            "score": 0.75,
            "source": "image_embedding",
            "metadata": {
                "bibtex_key": "Paper_2024",
                "category": "Test",
                "year": 2024,
                "image_path": "fig.png",
            },
        }]

        output = _format_figure_results(results)
        assert "[image_embedding]" in output

    def test_source_both(self):
        """Risultato con source=both mostra [both]."""
        from paperhunter.agents.tools.retrieval_tools import _format_figure_results

        results = [{
            "text": "Deduplicated result.",
            "score": 0.90,
            "source": "both",
            "metadata": {
                "bibtex_key": "Merged_2024",
                "category": "Test",
                "year": 2024,
                "image_path": "merged.png",
            },
        }]

        output = _format_figure_results(results)
        assert "[both]" in output

    def test_default_source_description(self):
        """Risultato senza campo source usa 'description' come default."""
        from paperhunter.agents.tools.retrieval_tools import _format_figure_results

        results = [{
            "text": "No source field.",
            "score": 0.70,
            "metadata": {
                "bibtex_key": "NoSource_2024",
                "category": "Test",
                "year": 2024,
            },
        }]

        output = _format_figure_results(results)
        assert "[description]" in output

    def test_image_path_included(self):
        """IMAGE_PATH e sintassi Markdown presenti nell'output."""
        from paperhunter.agents.tools.retrieval_tools import _format_figure_results

        results = [{
            "text": "Result with image path.",
            "score": 0.80,
            "source": "description",
            "metadata": {
                "bibtex_key": "Fig_2024",
                "category": "Test",
                "year": 2024,
                "image_path": "output/figures/fig1.png",
            },
        }]

        output = _format_figure_results(results)
        assert "IMAGE_PATH: output/figures/fig1.png" in output
        assert "![Figure from [Fig_2024]](output/figures/fig1.png)" in output

    def test_empty_results(self):
        """Risultati vuoti restituiscono messaggio."""
        from paperhunter.agents.tools.retrieval_tools import _format_figure_results

        output = _format_figure_results([])
        assert "No figure results found" in output


# ============================================================================
# Test: Logging
# ============================================================================

class TestSearchFigureDescriptionsLogging:
    """Test per il logging delle statistiche di ricerca."""

    def test_logs_collection_counts(self, caplog):
        """Verifica che il log contenga conteggi per desc e img_emb."""
        import logging

        mock_store = _make_mock_store()

        with caplog.at_level(logging.INFO), patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            search_figure_descriptions("test query")

        log_text = caplog.text
        assert "desc:" in log_text
        assert "img_emb:" in log_text

    def test_logs_skip_nonexistent_collection(self, caplog):
        """Verifica log quando image_embeddings non esiste."""
        import logging

        mock_store = _make_mock_store(img_exists=False)

        with caplog.at_level(logging.INFO), patch(
            "paperhunter.agents.tools.retrieval_tools._get_qdrant_store",
            return_value=mock_store,
        ), patch(
            "paperhunter.config.settings",
            _make_mock_settings(),
        ):
            from paperhunter.agents.tools.retrieval_tools import (
                search_figure_descriptions,
            )

            search_figure_descriptions("test query")

        assert "skip image embeddings" in caplog.text.lower() or \
               "non esiste" in caplog.text.lower()
