"""
Test suite per graphrag/community_summarizer.py — Task 2.2

Testa:
1.  CommunitySummarizer — inizializzazione e attributi
2.  summarize_community() — singola comunità con mock LLM
3.  summarize_all_communities() — batch con progress bar e skip existing
4.  save_to_qdrant() — salvataggio in collection graph_communities
5.  Formatting: _format_entities_for_prompt, _format_relationships_for_prompt,
    _format_key_entities_for_prompt
6.  Cleaning: _clean_llm_response
7.  Validation: _validate_community_summary
8.  Qdrant helpers: _make_community_chunk_id, _community_summary_to_embed_text,
    _community_exists_in_store, _ensure_community_collection
9.  Loading: load_community_summaries, count_community_summaries,
    _reconstruct_community_from_payload
10. Standalone: summarize_graph_communities
11. Config: collection_graph_communities in config.yaml / config.py / qdrant_store.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch, call

import pytest

# Patch config BEFORE importing modules
import paperhunter.config as cfg

cfg.settings.neo4j.enabled = True
cfg.settings.neo4j.uri = "bolt://localhost:7687"
cfg.settings.neo4j.username = "neo4j"
cfg.settings.neo4j.password = "password"
cfg.settings.neo4j.database = "neo4j"

from paperhunter.graphrag.community_summarizer import (
    CommunitySummarizer,
    _clean_llm_response,
    _community_exists_in_store,
    _community_summary_to_embed_text,
    _ensure_community_collection,
    _format_entities_for_prompt,
    _format_key_entities_for_prompt,
    _format_relationships_for_prompt,
    _make_community_chunk_id,
    _print_batch_report,
    _reconstruct_community_from_payload,
    _validate_community_summary,
    _word_count,
    count_community_summaries,
    load_community_summaries,
    summarize_graph_communities,
)


# ============================================================================
# Fixtures & helpers
# ============================================================================

def make_mock_nodes(n: int = 20, community_id: int = 0) -> list[dict]:
    """Crea una lista di nodi mock per una comunità."""
    names = [
        "GCG", "AutoDAN", "SmoothLLM", "Perplexity Filter",
        "GPT-4", "LLaMA-2", "AdvBench", "ASR",
        "Prompt Injection", "Backdoor Attack",
        "HarmBench", "Jailbreak", "RLHF", "Red Teaming",
        "DAN", "PAIR", "Claude", "Mistral", "ToxiGen", "NeurIPS",
    ]
    entity_types = ["Attack", "Defense", "Model", "Dataset"]
    nodes = []
    for i in range(n):
        nodes.append({
            "entity_id": f"eid_{community_id}_{i:03d}",
            "canonical_name": names[i % len(names)],
            "entity_type": entity_types[i % len(entity_types)],
            "description": f"Test entity {i} for community {community_id}",
            "paper_count": (i + 1) * 2,
        })
    return nodes


def make_mock_edges(nodes: list[dict], n_edges: int = 30) -> list[dict]:
    """Crea relazioni mock tra i nodi."""
    edges = []
    rel_types = ["TARGETS", "DEFENDS_AGAINST", "TESTED_ON", "EXTENDS", "RELATED_TO"]
    for i in range(min(n_edges, len(nodes) * 2)):
        src_idx = i % len(nodes)
        tgt_idx = (i + 1 + i // 3) % len(nodes)
        if src_idx == tgt_idx:
            tgt_idx = (tgt_idx + 1) % len(nodes)
        edges.append({
            "source": nodes[src_idx]["canonical_name"],
            "target": nodes[tgt_idx]["canonical_name"],
            "rel_type": rel_types[i % len(rel_types)],
            "weight": 1 + (i % 3),
        })
    return edges


def make_mock_community(community_id: int = 0, n_nodes: int = 20, n_edges: int = 30) -> dict:
    """Crea un dizionario comunità completo (come output di CommunityDetector)."""
    nodes = make_mock_nodes(n_nodes, community_id)
    edges = make_mock_edges(nodes, n_edges)
    key_entities = [n["canonical_name"] for n in nodes[:5]]
    return {
        "community_id": community_id,
        "nodes": nodes,
        "edges": edges,
        "size": len(nodes),
        "key_entities": key_entities,
    }


MOCK_SUMMARY_TEXT = (
    "This community represents a significant cluster of research on "
    "adversarial attacks against large language models. At its core, "
    "**GCG** (Greedy Coordinate Gradient) and **AutoDAN** represent "
    "pioneering optimization-based jailbreak techniques that exploit "
    "gradient information to craft adversarial suffixes. **SmoothLLM** "
    "and **Perplexity Filter** serve as defensive countermeasures, "
    "employing input perturbation and statistical detection respectively. "
    "The community's structure reveals a classic attack-defense dynamic: "
    "**GPT-4** and **LLaMA-2** appear as primary target models, while "
    "**AdvBench** and **ASR** (Attack Success Rate) serve as the "
    "standardized evaluation framework connecting attackers and defenders. "
    "This interconnected landscape illustrates how the field has evolved "
    "from simple prompt injection techniques toward more sophisticated "
    "optimization-based approaches and their corresponding defenses."
)


def make_mock_qdrant_store() -> MagicMock:
    """Crea un mock di QdrantStore con i metodi necessari."""
    store = MagicMock()
    store.client = MagicMock()
    store._embedding_dim = 3072
    store._distance = "Cosine"
    store.add_chunks = MagicMock(return_value=1)
    store.has_paper = MagicMock(return_value=False)
    store.client.collection_exists = MagicMock(return_value=True)
    return store


def make_mock_detector() -> MagicMock:
    """Crea un mock di CommunityDetector."""
    detector = MagicMock()
    return detector


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestConfig:
    """Verifica che la configurazione includa la collection graph_communities."""

    def test_config_yaml_has_collection(self):
        """collection_graph_communities presente in config."""
        assert hasattr(cfg.settings.qdrant, "collection_graph_communities")
        assert cfg.settings.qdrant.collection_graph_communities == "graph_communities"

    def test_qdrant_store_has_collection_constant(self):
        """QdrantStore ha la costante COLLECTION_GRAPH_COMMUNITIES."""
        from paperhunter.indexing.qdrant_store import QdrantStore
        assert hasattr(QdrantStore, "COLLECTION_GRAPH_COMMUNITIES")
        assert QdrantStore.COLLECTION_GRAPH_COMMUNITIES == "graph_communities"


# ============================================================================
# FORMATTING TESTS
# ============================================================================

class TestFormatEntities:
    """Testa _format_entities_for_prompt."""

    def test_basic_formatting(self):
        nodes = make_mock_nodes(3)
        result = _format_entities_for_prompt(nodes)
        assert "**GCG**" in result
        assert "**AutoDAN**" in result
        assert "**SmoothLLM**" in result
        assert "type:" in result

    def test_empty_nodes(self):
        result = _format_entities_for_prompt([])
        assert "nessuna" in result.lower()

    def test_includes_description(self):
        nodes = [{"canonical_name": "GCG", "entity_type": "Attack", "description": "An optimization attack", "paper_count": 5}]
        result = _format_entities_for_prompt(nodes)
        assert "An optimization attack" in result
        assert "5 papers" in result

    def test_long_description_truncated(self):
        nodes = [{"canonical_name": "A", "entity_type": "T", "description": "x" * 200}]
        result = _format_entities_for_prompt(nodes)
        # Descrizione troncata a 150 chars
        assert len(result) < 300


class TestFormatRelationships:
    """Testa _format_relationships_for_prompt."""

    def test_basic_formatting(self):
        nodes = make_mock_nodes(5)
        edges = make_mock_edges(nodes, 5)
        result = _format_relationships_for_prompt(edges)
        assert "—[" in result
        assert "]→" in result

    def test_empty_edges(self):
        result = _format_relationships_for_prompt([])
        assert "nessuna" in result.lower()

    def test_dedup_bidirectional(self):
        edges = [
            {"source": "A", "target": "B", "rel_type": "R"},
            {"source": "B", "target": "A", "rel_type": "R"},  # duplicate
        ]
        result = _format_relationships_for_prompt(edges)
        lines = [l for l in result.split("\n") if l.startswith("-")]
        assert len(lines) == 1

    def test_truncation_over_50(self):
        nodes = make_mock_nodes(20)
        edges = []
        for i in range(60):
            edges.append({
                "source": f"Entity{i}",
                "target": f"Entity{i+1}",
                "rel_type": f"REL_{i}",
            })
        result = _format_relationships_for_prompt(edges)
        assert "altre" in result

    def test_weight_displayed(self):
        edges = [{"source": "A", "target": "B", "rel_type": "R", "weight": 5}]
        result = _format_relationships_for_prompt(edges)
        assert "weight: 5" in result


class TestFormatKeyEntities:
    """Testa _format_key_entities_for_prompt."""

    def test_basic(self):
        result = _format_key_entities_for_prompt(["GCG", "AutoDAN", "SmoothLLM"])
        assert "**GCG**" in result
        assert "**AutoDAN**" in result
        assert ", " in result

    def test_empty(self):
        result = _format_key_entities_for_prompt([])
        assert "nessuna" in result.lower()


# ============================================================================
# CLEANING & VALIDATION TESTS
# ============================================================================

class TestCleanLlmResponse:
    """Testa _clean_llm_response."""

    def test_plain_text(self):
        result = _clean_llm_response("This is a summary about GCG attacks.")
        assert result == "This is a summary about GCG attacks."

    def test_strips_whitespace(self):
        result = _clean_llm_response("  text  \n  ")
        assert result == "text"

    def test_removes_code_fence(self):
        input_text = "```text\nSome summary\n```"
        result = _clean_llm_response(input_text)
        assert result == "Some summary"

    def test_removes_markdown_fence(self):
        input_text = "```markdown\nSome summary\n```"
        result = _clean_llm_response(input_text)
        assert result == "Some summary"

    def test_removes_prefix(self):
        result = _clean_llm_response("Summary: This is the actual text.")
        assert result == "This is the actual text."

    def test_removes_community_summary_prefix(self):
        result = _clean_llm_response("Community Summary: Actual text here.")
        assert result == "Actual text here."


class TestValidateCommunitySummary:
    """Testa _validate_community_summary."""

    def test_valid_summary_no_warnings(self, capsys):
        # A summary of ~150 words should pass without warnings
        _validate_community_summary(
            MOCK_SUMMARY_TEXT,
            community_id=0,
            key_entities=["GCG", "AutoDAN", "SmoothLLM"],
        )
        # Should not print any warnings (but logger may emit debug)

    def test_short_summary_warns(self):
        with patch("paperhunter.graphrag.community_summarizer.print_warning") as pw:
            _validate_community_summary(
                "Too short.",
                community_id=0,
                key_entities=["GCG"],
            )
            assert pw.called

    def test_long_summary_warns(self):
        long_text = ("word " * 900).strip()
        with patch("paperhunter.graphrag.community_summarizer.print_warning") as pw:
            _validate_community_summary(
                long_text,
                community_id=0,
                key_entities=[],
            )
            assert pw.called

    def test_missing_key_entities_warns(self):
        summary = ("This is a fairly long summary " * 20).strip()
        with patch("paperhunter.graphrag.community_summarizer.print_warning") as pw:
            _validate_community_summary(
                summary,
                community_id=0,
                key_entities=["EntityNotInText"],
            )
            assert pw.called


class TestWordCount:
    """Testa _word_count."""

    def test_basic(self):
        assert _word_count("one two three") == 3

    def test_empty(self):
        assert _word_count("") == 0

    def test_multiline(self):
        assert _word_count("one\ntwo\nthree") == 3


# ============================================================================
# CHUNK ID & EMBED TEXT TESTS
# ============================================================================

class TestMakeCommunityChunkId:
    """Testa _make_community_chunk_id."""

    def test_deterministic(self):
        id1 = _make_community_chunk_id(0)
        id2 = _make_community_chunk_id(0)
        assert id1 == id2

    def test_different_ids(self):
        id1 = _make_community_chunk_id(0)
        id2 = _make_community_chunk_id(1)
        assert id1 != id2

    def test_length(self):
        cid = _make_community_chunk_id(42)
        assert len(cid) == 16

    def test_hex_chars(self):
        cid = _make_community_chunk_id(99)
        assert all(c in "0123456789abcdef" for c in cid)


class TestCommunitySummaryToEmbedText:
    """Testa _community_summary_to_embed_text."""

    def test_structure(self):
        text = _community_summary_to_embed_text(
            community_id=0,
            summary="Test summary about GCG.",
            key_entities=["GCG", "AutoDAN"],
            node_count=15,
        )
        assert "Graph Community 0" in text
        assert "Entities: 15" in text
        assert "Key Entities: GCG, AutoDAN" in text
        assert "Test summary about GCG." in text


# ============================================================================
# QDRANT HELPER TESTS
# ============================================================================

class TestCommunityExistsInStore:
    """Testa _community_exists_in_store."""

    def test_exists_true(self):
        store = MagicMock()
        store.has_paper = MagicMock(return_value=True)
        assert _community_exists_in_store(0, store) is True
        store.has_paper.assert_called_once_with("community_0", collection_name="graph_communities")

    def test_exists_false(self):
        store = MagicMock()
        store.has_paper = MagicMock(return_value=False)
        assert _community_exists_in_store(5, store) is False

    def test_exception_returns_false(self):
        store = MagicMock()
        store.has_paper = MagicMock(side_effect=Exception("DB error"))
        assert _community_exists_in_store(0, store) is False


class TestEnsureCommunityCollection:
    """Testa _ensure_community_collection."""

    def test_creates_if_not_exists(self):
        store = MagicMock()
        store.client.collection_exists = MagicMock(return_value=False)
        store._embedding_dim = 3072
        store._distance = "Cosine"
        _ensure_community_collection(store, "graph_communities")
        store.client.create_collection.assert_called_once()

    def test_no_op_if_exists(self):
        store = MagicMock()
        store.client.collection_exists = MagicMock(return_value=True)
        _ensure_community_collection(store, "graph_communities")
        store.client.create_collection.assert_not_called()


# ============================================================================
# RECONSTRUCT FROM PAYLOAD TESTS
# ============================================================================

class TestReconstructCommunityFromPayload:
    """Testa _reconstruct_community_from_payload."""

    def test_valid_reconstruction(self):
        text = (
            "Graph Community 0 — Knowledge Graph Community Summary\n"
            "Entities: 15\n"
            "Key Entities: GCG, AutoDAN, SmoothLLM\n"
            "\n"
            "This is the narrative summary about the community."
        )
        result = _reconstruct_community_from_payload("community_0", text)
        assert result is not None
        assert result["community_id"] == 0
        assert result["node_count"] == 15
        assert result["key_entities"] == ["GCG", "AutoDAN", "SmoothLLM"]
        assert "narrative summary" in result["summary"]

    def test_invalid_paper_id(self):
        result = _reconstruct_community_from_payload("not_community_id", "text")
        assert result is None

    def test_malformed_text(self):
        result = _reconstruct_community_from_payload("community_0", "")
        assert result is not None  # Should still parse (empty summary)
        assert result["community_id"] == 0
        assert result["summary"] == ""

    def test_multiline_summary(self):
        text = (
            "Graph Community 3 — Knowledge Graph Community Summary\n"
            "Entities: 8\n"
            "Key Entities: PAIR, DAN\n"
            "\n"
            "First paragraph of the summary.\n"
            "\n"
            "Second paragraph with more details."
        )
        result = _reconstruct_community_from_payload("community_3", text)
        assert result is not None
        assert result["community_id"] == 3
        assert "First paragraph" in result["summary"]
        assert "Second paragraph" in result["summary"]


# ============================================================================
# COMMUNITY SUMMARIZER CLASS TESTS
# ============================================================================

class TestCommunitySummarizer:
    """Testa la classe CommunitySummarizer."""

    def test_init(self):
        detector = make_mock_detector()
        store = make_mock_qdrant_store()
        summarizer = CommunitySummarizer(detector, store)
        assert summarizer.detector is detector
        assert summarizer.store is store

    def test_repr(self):
        detector = make_mock_detector()
        store = make_mock_qdrant_store()
        summarizer = CommunitySummarizer(detector, store)
        r = repr(summarizer)
        assert "CommunitySummarizer" in r

    @patch("paperhunter.graphrag.community_summarizer.llm_call_with_retry")
    @patch("paperhunter.graphrag.community_summarizer.get_pro_llm")
    @patch("paperhunter.graphrag.community_summarizer.get_gemini_limiter")
    def test_summarize_community_basic(self, mock_limiter, mock_llm_factory, mock_llm_call):
        """Testa che summarize_community generi un summary valido."""
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm
        mock_limiter_instance = MagicMock()
        mock_limiter.return_value = mock_limiter_instance
        mock_llm_call.return_value = MOCK_SUMMARY_TEXT

        detector = make_mock_detector()
        store = make_mock_qdrant_store()
        summarizer = CommunitySummarizer(detector, store)

        community = make_mock_community(community_id=0)
        result = summarizer.summarize_community(
            community_id=0,
            subgraph=community,
        )

        assert isinstance(result, str)
        assert len(result) > 50
        assert "GCG" in result
        mock_llm_call.assert_called_once()

    def test_summarize_community_empty_raises(self):
        """summarize_community con sotto-grafo vuoto deve lanciare ValueError."""
        summarizer = CommunitySummarizer(make_mock_detector(), make_mock_qdrant_store())
        with pytest.raises(ValueError, match="vuoto"):
            summarizer.summarize_community(
                community_id=0,
                subgraph={"nodes": [], "edges": [], "key_entities": []},
            )

    @patch("paperhunter.graphrag.community_summarizer.llm_call_with_retry")
    @patch("paperhunter.graphrag.community_summarizer.get_pro_llm")
    @patch("paperhunter.graphraph.community_summarizer.get_gemini_limiter" if False else "paperhunter.graphrag.community_summarizer.get_gemini_limiter")
    def test_summarize_community_llm_error(self, mock_limiter, mock_llm_factory, mock_llm_call):
        """Se il LLM fallisce, l'eccezione viene propagata."""
        mock_llm_factory.return_value = MagicMock()
        mock_limiter.return_value = MagicMock()
        mock_llm_call.side_effect = RuntimeError("LLM timeout")

        summarizer = CommunitySummarizer(make_mock_detector(), make_mock_qdrant_store())
        community = make_mock_community(community_id=0)

        with pytest.raises(RuntimeError, match="LLM timeout"):
            summarizer.summarize_community(0, community)

    @patch("paperhunter.graphrag.community_summarizer.llm_call_with_retry")
    @patch("paperhunter.graphrag.community_summarizer.get_pro_llm")
    @patch("paperhunter.graphrag.community_summarizer.get_gemini_limiter")
    def test_summarize_community_with_flash(self, mock_limiter, mock_llm_factory, mock_llm_call):
        """force_model='flash' usa Flash LLM."""
        mock_limiter.return_value = MagicMock()
        mock_llm_call.return_value = MOCK_SUMMARY_TEXT

        with patch("paperhunter.graphrag.community_summarizer.get_pro_llm") as mock_pro, \
             patch("paperhunter.utils.llm_factory.get_flash_llm") as mock_flash_ext:
            mock_flash = MagicMock()
            # The import inside the function
            with patch("paperhunter.graphrag.community_summarizer.get_pro_llm") as mock_pro2:
                summarizer = CommunitySummarizer(make_mock_detector(), make_mock_qdrant_store())
                community = make_mock_community(community_id=0)
                result = summarizer.summarize_community(0, community, force_model="flash")
                assert isinstance(result, str)


class TestSummarizeAllCommunities:
    """Testa summarize_all_communities."""

    @patch("paperhunter.graphrag.community_summarizer.llm_call_with_retry")
    @patch("paperhunter.graphrag.community_summarizer.get_pro_llm")
    @patch("paperhunter.graphrag.community_summarizer.get_gemini_limiter")
    @patch("paperhunter.graphrag.community_summarizer._community_exists_in_store")
    def test_batch_processing(self, mock_exists, mock_limiter, mock_llm_factory, mock_llm_call):
        """Batch processing di 3 comunità."""
        mock_exists.return_value = False
        mock_llm_factory.return_value = MagicMock()
        mock_limiter.return_value = MagicMock()
        mock_llm_call.return_value = MOCK_SUMMARY_TEXT

        summarizer = CommunitySummarizer(make_mock_detector(), make_mock_qdrant_store())
        communities = [make_mock_community(i) for i in range(3)]

        results = summarizer.summarize_all_communities(communities)

        assert len(results) == 3
        for r in results:
            assert "community_id" in r
            assert "summary" in r
            assert "node_count" in r
            assert "key_entities" in r
            assert len(r["summary"]) > 50

    @patch("paperhunter.graphrag.community_summarizer._community_exists_in_store")
    def test_skip_existing(self, mock_exists):
        """Comunità già esistenti in Qdrant vengono saltate."""
        mock_exists.return_value = True

        summarizer = CommunitySummarizer(make_mock_detector(), make_mock_qdrant_store())
        communities = [make_mock_community(i) for i in range(3)]

        results = summarizer.summarize_all_communities(communities, skip_existing=True)
        assert len(results) == 0

    def test_empty_communities(self):
        """Lista vuota restituisce lista vuota."""
        summarizer = CommunitySummarizer(make_mock_detector(), make_mock_qdrant_store())
        results = summarizer.summarize_all_communities([])
        assert results == []

    @patch("paperhunter.graphrag.community_summarizer.llm_call_with_retry")
    @patch("paperhunter.graphrag.community_summarizer.get_pro_llm")
    @patch("paperhunter.graphrag.community_summarizer.get_gemini_limiter")
    @patch("paperhunter.graphrag.community_summarizer._community_exists_in_store")
    def test_partial_errors(self, mock_exists, mock_limiter, mock_llm_factory, mock_llm_call):
        """Se un comunità fallisce, le altre vengono comunque processate."""
        mock_exists.return_value = False
        mock_llm_factory.return_value = MagicMock()
        mock_limiter.return_value = MagicMock()

        # First call succeeds, second fails, third succeeds
        mock_llm_call.side_effect = [
            MOCK_SUMMARY_TEXT,
            RuntimeError("LLM error"),
            MOCK_SUMMARY_TEXT,
        ]

        summarizer = CommunitySummarizer(make_mock_detector(), make_mock_qdrant_store())
        communities = [make_mock_community(i) for i in range(3)]

        results = summarizer.summarize_all_communities(communities)
        assert len(results) == 2  # 2 successi, 1 errore


# ============================================================================
# SAVE TO QDRANT TESTS
# ============================================================================

class TestSaveToQdrant:
    """Testa save_to_qdrant."""

    def test_saves_summaries(self):
        store = make_mock_qdrant_store()
        store.add_chunks.return_value = 3

        summarizer = CommunitySummarizer(make_mock_detector(), store)
        summaries = [
            {"community_id": 0, "summary": "Summary 0", "node_count": 10, "key_entities": ["A", "B"]},
            {"community_id": 1, "summary": "Summary 1", "node_count": 8, "key_entities": ["C"]},
            {"community_id": 2, "summary": "Summary 2", "node_count": 15, "key_entities": ["D", "E"]},
        ]

        count = summarizer.save_to_qdrant(summaries)
        assert count == 3
        store.add_chunks.assert_called_once()

        # Verify the chunks have correct paper_ids
        args = store.add_chunks.call_args
        chunks = args[0][0]  # first positional arg
        assert len(chunks) == 3
        assert chunks[0].paper_id == "community_0"
        assert chunks[1].paper_id == "community_1"
        assert chunks[2].paper_id == "community_2"

    def test_empty_summaries(self):
        store = make_mock_qdrant_store()
        summarizer = CommunitySummarizer(make_mock_detector(), store)
        count = summarizer.save_to_qdrant([])
        assert count == 0
        store.add_chunks.assert_not_called()

    def test_save_error_returns_zero(self):
        store = make_mock_qdrant_store()
        store.add_chunks.side_effect = Exception("Qdrant error")

        summarizer = CommunitySummarizer(make_mock_detector(), store)
        summaries = [{"community_id": 0, "summary": "s", "node_count": 5, "key_entities": []}]

        count = summarizer.save_to_qdrant(summaries)
        assert count == 0


# ============================================================================
# LOADING UTILITIES TESTS
# ============================================================================

class TestLoadCommunitySummaries:
    """Testa load_community_summaries."""

    def test_load_from_store(self):
        """Mock scroll that returns community summary points."""
        store = MagicMock()
        store.client.collection_exists.return_value = True

        # Simulate scroll returning points
        point1 = MagicMock()
        point1.payload = {
            "paper_id": "community_0",
            "text": (
                "Graph Community 0 — Knowledge Graph Community Summary\n"
                "Entities: 15\n"
                "Key Entities: GCG, AutoDAN\n"
                "\n"
                "Narrative about GCG and AutoDAN attacks."
            ),
        }
        point2 = MagicMock()
        point2.payload = {
            "paper_id": "community_1",
            "text": (
                "Graph Community 1 — Knowledge Graph Community Summary\n"
                "Entities: 8\n"
                "Key Entities: SmoothLLM\n"
                "\n"
                "Narrative about defenses."
            ),
        }
        store.client.scroll.return_value = ([point1, point2], None)

        results = load_community_summaries(store=store)
        assert len(results) == 2
        assert results[0]["community_id"] == 0
        assert results[1]["community_id"] == 1
        assert "GCG" in results[0]["key_entities"]

    def test_no_collection_returns_empty(self):
        store = MagicMock()
        store.client.collection_exists.return_value = False

        results = load_community_summaries(store=store)
        assert results == []


class TestCountCommunitySummaries:
    """Testa count_community_summaries."""

    def test_count(self):
        store = MagicMock()
        store.client.collection_exists.return_value = True
        info = MagicMock()
        info.points_count = 5
        store.client.get_collection.return_value = info

        count = count_community_summaries(store=store)
        assert count == 5

    def test_no_collection(self):
        store = MagicMock()
        store.client.collection_exists.return_value = False

        count = count_community_summaries(store=store)
        assert count == 0

    def test_error_returns_zero(self):
        store = MagicMock()
        store.client.collection_exists.return_value = True
        store.client.get_collection.side_effect = Exception("err")

        count = count_community_summaries(store=store)
        assert count == 0


# ============================================================================
# BATCH REPORT TESTS
# ============================================================================

class TestPrintBatchReport:
    """Testa _print_batch_report."""

    @patch("paperhunter.graphrag.community_summarizer.print_table")
    @patch("paperhunter.graphrag.community_summarizer.print_success")
    def test_success_report(self, mock_success, mock_table):
        _print_batch_report(
            total=5, generated=4, skipped=1, errors=0, error_details={}
        )
        mock_success.assert_called_once()
        mock_table.assert_called_once()

    @patch("paperhunter.graphrag.community_summarizer.print_table")
    @patch("paperhunter.graphrag.community_summarizer.print_error")
    @patch("paperhunter.graphrag.community_summarizer.print_success")
    def test_error_report(self, mock_success, mock_error, mock_table):
        _print_batch_report(
            total=3, generated=1, skipped=0, errors=2,
            error_details={0: "err0", 1: "err1"}
        )
        mock_success.assert_called_once()
        assert mock_error.call_count >= 1


# ============================================================================
# INTEGRATION-STYLE TESTS (with mocks)
# ============================================================================

class TestFullPipeline:
    """Test integrazione: detection mock → summarization → save to Qdrant."""

    @patch("paperhunter.graphrag.community_summarizer.llm_call_with_retry")
    @patch("paperhunter.graphrag.community_summarizer.get_pro_llm")
    @patch("paperhunter.graphrag.community_summarizer.get_gemini_limiter")
    @patch("paperhunter.graphrag.community_summarizer._community_exists_in_store")
    def test_full_pipeline(self, mock_exists, mock_limiter, mock_llm_factory, mock_llm_call):
        """Pipeline completa: community dict → summarize → save."""
        mock_exists.return_value = False
        mock_llm_factory.return_value = MagicMock()
        mock_limiter.return_value = MagicMock()
        mock_llm_call.return_value = MOCK_SUMMARY_TEXT

        store = make_mock_qdrant_store()
        store.add_chunks.return_value = 2

        summarizer = CommunitySummarizer(make_mock_detector(), store)

        # Simula 2 comunità con sotto-grafi realistici
        communities = [
            make_mock_community(0, n_nodes=20, n_edges=30),
            make_mock_community(1, n_nodes=10, n_edges=15),
        ]

        # Step 1: genera i summary
        results = summarizer.summarize_all_communities(communities)
        assert len(results) == 2
        assert all(r["summary"] == MOCK_SUMMARY_TEXT for r in results)

        # Step 2: salva in Qdrant
        saved = summarizer.save_to_qdrant(results)
        assert saved == 2
        store.add_chunks.assert_called_once()

    @patch("paperhunter.graphrag.community_summarizer.llm_call_with_retry")
    @patch("paperhunter.graphrag.community_summarizer.get_pro_llm")
    @patch("paperhunter.graphrag.community_summarizer.get_gemini_limiter")
    def test_many_communities(self, mock_limiter, mock_llm_factory, mock_llm_call):
        """Stress test con 10 comunità mock."""
        mock_llm_factory.return_value = MagicMock()
        mock_limiter.return_value = MagicMock()
        mock_llm_call.return_value = MOCK_SUMMARY_TEXT

        store = make_mock_qdrant_store()
        store.add_chunks.return_value = 10
        store.has_paper.return_value = False

        summarizer = CommunitySummarizer(make_mock_detector(), store)
        communities = [make_mock_community(i, n_nodes=5 + i, n_edges=8 + i) for i in range(10)]

        results = summarizer.summarize_all_communities(communities, skip_existing=False)
        assert len(results) == 10
        assert mock_llm_call.call_count == 10

        saved = summarizer.save_to_qdrant(results)
        assert saved == 10


# ============================================================================
# IMPORTS / EXPORTS
# ============================================================================

class TestExports:
    """Verifica che gli export nel __init__.py siano corretti."""

    def test_module_exports(self):
        from paperhunter.graphrag import (
            CommunitySummarizer,
            count_community_summaries,
            load_community_summaries,
            summarize_graph_communities,
        )
        assert CommunitySummarizer is not None
        assert callable(count_community_summaries)
        assert callable(load_community_summaries)
        assert callable(summarize_graph_communities)
