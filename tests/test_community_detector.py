"""
Test suite per graphrag/community_detector.py — Task 2.1

Testa:
1.  CommunityDetector — inizializzazione e validazione
2.  ensure_gds() — verifica GDS disponibile/non disponibile
3.  detect_communities() — pipeline completa con mock Neo4j
4.  get_community_subgraph() — estrazione sotto-grafo singola comunità
5.  get_all_communities() — recupero comunità già rilevate (senza re-detection)
6.  get_community_stats() — statistiche aggregate
7.  clear_community_labels() — pulizia proprietà community_id
8.  GDS methods su Neo4jManager — ensure_gds_plugin, project_graph,
    drop_graph_projection, run_community_detection, get_nodes_by_community,
    get_edges_within_community, get_all_community_ids
9.  Helper _extract_key_entities — entità chiave per connessioni
10. detect_graph_communities() — convenience standalone
11. Test con grafo reale (Neo4j, skip se non disponibile)

Prerequisiti per i test di integrazione (opzionali):
    docker run --rm -d --name neo4j_test -p 7474:7474 -p 7687:7687 \\
        -e NEO4J_AUTH=neo4j/password \\
        -e NEO4J_PLUGINS='["graph-data-science"]' \\
        neo4j:5-community
"""

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

from paperhunter.graphrag.community_detector import (
    CommunityDetectionError,
    CommunityDetector,
    GDSNotAvailableError,
    _extract_key_entities,
    _log_community_detection_results,
    detect_graph_communities,
    get_community_stats_standalone,
)
from paperhunter.graphrag.entity_extractor import (
    Entity,
    EntityType,
    Relation,
    RelationType,
)


# ============================================================================
# Fixtures & helpers
# ============================================================================


def make_mock_neo4j_manager(is_connected: bool = True) -> MagicMock:
    """Crea un mock di Neo4jManager con i metodi GDS necessari."""
    mgr = MagicMock()
    type(mgr).is_connected = PropertyMock(return_value=is_connected)
    return mgr


def make_mock_nodes(n: int = 10, community_id: int = 0) -> list[dict]:
    """Crea una lista di nodi mock con community_id assegnato."""
    nodes = []
    entity_types = ["Attack", "Defense", "Model", "Dataset"]
    names = [
        "GCG", "AutoDAN", "SmoothLLM", "Perplexity Filter",
        "GPT-4", "LLaMA-2", "AdvBench", "ASR",
        "Prompt Injection", "Backdoor Attack",
        "HarmBench", "Jailbreak", "RLHF", "Red Teaming",
        "DAN", "PAIR", "Claude", "Mistral", "ToxiGen", "NeurIPS",
    ]
    for i in range(n):
        nodes.append({
            "entity_id": f"eid_{community_id}_{i:03d}",
            "canonical_name": names[i % len(names)],
            "entity_type": entity_types[i % len(entity_types)],
            "labels": ["Entity", entity_types[i % len(entity_types)]],
            "description": f"Test entity {i}",
            "year": 2022 + (i % 4),
            "community_id": community_id,
        })
    return nodes


def make_mock_edges(nodes: list[dict], n_edges: int = 15) -> list[dict]:
    """Crea relazioni mock tra i nodi."""
    edges = []
    rel_types = ["TARGETS", "DEFENDS_AGAINST", "TESTED_ON", "EXTENDS", "UPDATES"]
    for i in range(min(n_edges, len(nodes) * 2)):
        src = nodes[i % len(nodes)]
        tgt = nodes[(i + 1) % len(nodes)]
        if src["canonical_name"] != tgt["canonical_name"]:
            edges.append({
                "source": src["canonical_name"],
                "source_id": src["entity_id"],
                "rel_type": rel_types[i % len(rel_types)],
                "description": f"Relation {i}",
                "year": 2023,
                "source_bibtex_key": f"paper_{i:03d}",
                "target": tgt["canonical_name"],
                "target_id": tgt["entity_id"],
            })
    return edges


# ============================================================================
# Test 1: Inizializzazione
# ============================================================================


class TestCommunityDetectorInit:
    """Test inizializzazione e validazione."""

    def test_init_ok(self):
        """CommunityDetector si inizializza con un manager connesso."""
        mgr = make_mock_neo4j_manager(is_connected=True)
        detector = CommunityDetector(mgr)
        assert detector._mgr is mgr
        assert detector._graph_name == "community_graph"
        assert detector._community_property == "community_id"

    def test_init_custom_params(self):
        """CommunityDetector accetta parametri custom."""
        mgr = make_mock_neo4j_manager()
        detector = CommunityDetector(
            mgr,
            graph_name="test_graph",
            community_property="cluster_label",
        )
        assert detector._graph_name == "test_graph"
        assert detector._community_property == "cluster_label"

    def test_init_disconnected_raises(self):
        """ValueError se il manager non è connesso."""
        mgr = make_mock_neo4j_manager(is_connected=False)
        with pytest.raises(ValueError, match="non è connesso"):
            CommunityDetector(mgr)

    def test_repr(self):
        """Verifica la rappresentazione string."""
        mgr = make_mock_neo4j_manager()
        detector = CommunityDetector(mgr)
        assert "CommunityDetector" in repr(detector)
        assert "community_graph" in repr(detector)


# ============================================================================
# Test 2: GDS availability
# ============================================================================


class TestEnsureGDS:
    """Test verifica disponibilità GDS."""

    def test_ensure_gds_available(self):
        """ensure_gds() ritorna True se GDS è disponibile."""
        mgr = make_mock_neo4j_manager()
        mgr.ensure_gds_plugin.return_value = True
        detector = CommunityDetector(mgr)
        assert detector.ensure_gds() is True
        assert detector._gds_available is True

    def test_ensure_gds_not_available(self):
        """ensure_gds() lancia GDSNotAvailableError se GDS manca."""
        mgr = make_mock_neo4j_manager()
        mgr.ensure_gds_plugin.return_value = False
        detector = CommunityDetector(mgr)
        with pytest.raises(GDSNotAvailableError, match="non è installato"):
            detector.ensure_gds()

    def test_ensure_gds_caches_result(self):
        """ensure_gds() chiama ensure_gds_plugin solo una volta (caching)."""
        mgr = make_mock_neo4j_manager()
        mgr.ensure_gds_plugin.return_value = True
        detector = CommunityDetector(mgr)
        detector.ensure_gds()
        detector.ensure_gds()
        detector.ensure_gds()
        # Solo una chiamata al manager
        assert mgr.ensure_gds_plugin.call_count == 1


# ============================================================================
# Test 3: detect_communities (pipeline completa, mock)
# ============================================================================


class TestDetectCommunities:
    """Test pipeline completa di community detection con mock."""

    def _setup_mock_mgr(self, n_communities: int = 3, nodes_per: int = 10):
        """Setup mock manager con n comunità."""
        mgr = make_mock_neo4j_manager()
        mgr.ensure_gds_plugin.return_value = True

        # project_graph
        mgr.project_graph.return_value = {
            "graph_name": "community_graph",
            "node_count": n_communities * nodes_per,
            "relationship_count": n_communities * 15,
        }

        # run_community_detection
        mgr.run_community_detection.return_value = {
            "community_count": n_communities,
            "modularity": 0.65,
            "node_properties_written": n_communities * nodes_per,
            "algorithm": "louvain",
            "write_property": "community_id",
        }

        # get_all_community_ids
        community_ids = [
            {"community_id": i, "size": nodes_per}
            for i in range(n_communities)
        ]
        mgr.get_all_community_ids.return_value = community_ids

        # get_nodes_by_community & get_edges_within_community
        def mock_nodes(community_id, community_property="community_id"):
            return make_mock_nodes(n=nodes_per, community_id=community_id)

        def mock_edges(community_id, community_property="community_id"):
            nodes = make_mock_nodes(n=nodes_per, community_id=community_id)
            return make_mock_edges(nodes, n_edges=15)

        mgr.get_nodes_by_community.side_effect = mock_nodes
        mgr.get_edges_within_community.side_effect = mock_edges

        # drop_graph_projection
        mgr.drop_graph_projection.return_value = True

        return mgr

    def test_detect_returns_communities(self):
        """detect_communities() restituisce una lista di comunità."""
        mgr = self._setup_mock_mgr(n_communities=3)
        detector = CommunityDetector(mgr)
        communities = detector.detect_communities(algorithm="louvain", min_size=3)

        assert len(communities) == 3
        for c in communities:
            assert "community_id" in c
            assert "nodes" in c
            assert "edges" in c
            assert "size" in c
            assert "key_entities" in c
            assert c["size"] == 10

    def test_detect_calls_gds_methods_in_order(self):
        """detect_communities() chiama i metodi GDS nell'ordine corretto."""
        mgr = self._setup_mock_mgr(n_communities=2)
        detector = CommunityDetector(mgr)
        detector.detect_communities()

        # Ordine: ensure_gds → project → run_detection → get_ids → get_nodes/edges → drop
        mgr.ensure_gds_plugin.assert_called_once()
        mgr.project_graph.assert_called_once()
        mgr.run_community_detection.assert_called_once()
        mgr.get_all_community_ids.assert_called_once()
        assert mgr.get_nodes_by_community.call_count == 2
        assert mgr.get_edges_within_community.call_count == 2
        mgr.drop_graph_projection.assert_called()  # In finally

    def test_detect_leiden_algorithm(self):
        """detect_communities supporta algoritmo leiden."""
        mgr = self._setup_mock_mgr(n_communities=2)
        detector = CommunityDetector(mgr)
        detector.detect_communities(algorithm="leiden")

        mgr.run_community_detection.assert_called_once_with(
            graph_name="community_graph",
            algorithm="leiden",
            write_property="community_id",
        )

    def test_detect_empty_graph(self):
        """detect_communities() con grafo vuoto restituisce lista vuota."""
        mgr = make_mock_neo4j_manager()
        mgr.ensure_gds_plugin.return_value = True
        mgr.project_graph.return_value = {
            "graph_name": "community_graph",
            "node_count": 0,
            "relationship_count": 0,
        }
        mgr.drop_graph_projection.return_value = True

        detector = CommunityDetector(mgr)
        result = detector.detect_communities()
        assert result == []

    def test_detect_projection_error_raises(self):
        """detect_communities() lancia CommunityDetectionError se proiezione fallisce."""
        mgr = make_mock_neo4j_manager()
        mgr.ensure_gds_plugin.return_value = True
        mgr.project_graph.side_effect = RuntimeError("Projection failed")

        detector = CommunityDetector(mgr)
        with pytest.raises(CommunityDetectionError, match="proiezione"):
            detector.detect_communities()

    def test_detect_algorithm_error_drops_projection(self):
        """Se l'algoritmo fallisce, la proiezione viene comunque droppata."""
        mgr = make_mock_neo4j_manager()
        mgr.ensure_gds_plugin.return_value = True
        mgr.project_graph.return_value = {"node_count": 10, "relationship_count": 5}
        mgr.run_community_detection.side_effect = RuntimeError("Algorithm crash")
        mgr.drop_graph_projection.return_value = True

        detector = CommunityDetector(mgr)
        with pytest.raises(CommunityDetectionError):
            detector.detect_communities()

        # Drop deve essere stato chiamato
        mgr.drop_graph_projection.assert_called_with("community_graph")

    def test_detect_min_size_filter(self):
        """min_size filtra le comunità troppo piccole."""
        mgr = make_mock_neo4j_manager()
        mgr.ensure_gds_plugin.return_value = True
        mgr.project_graph.return_value = {"node_count": 30, "relationship_count": 20}
        mgr.run_community_detection.return_value = {
            "community_count": 5, "modularity": 0.5,
            "node_properties_written": 30, "algorithm": "louvain",
        }
        # 3 comunità con size >= 5, 2 con size < 5
        mgr.get_all_community_ids.return_value = [
            {"community_id": 0, "size": 15},
            {"community_id": 1, "size": 10},
            {"community_id": 2, "size": 5},
        ]
        mgr.get_nodes_by_community.return_value = make_mock_nodes(5)
        mgr.get_edges_within_community.return_value = []
        mgr.drop_graph_projection.return_value = True

        detector = CommunityDetector(mgr)
        communities = detector.detect_communities(min_size=5)
        assert len(communities) == 3

    def test_detect_stores_last_result(self):
        """detect_communities() memorizza l'ultimo risultato di detection."""
        mgr = self._setup_mock_mgr(n_communities=2)
        detector = CommunityDetector(mgr)
        detector.detect_communities()

        assert detector._last_detection_result is not None
        assert detector._last_detection_result["modularity"] == 0.65

    def test_detect_custom_relationship_types(self):
        """detect_communities() supporta relationship_types custom."""
        mgr = self._setup_mock_mgr(n_communities=2)
        detector = CommunityDetector(mgr)
        detector.detect_communities(
            relationship_types=["TARGETS", "DEFENDS_AGAINST"]
        )

        mgr.project_graph.assert_called_once_with(
            graph_name="community_graph",
            relationship_types=["TARGETS", "DEFENDS_AGAINST"],
        )


# ============================================================================
# Test 4: get_community_subgraph
# ============================================================================


class TestGetCommunitySubgraph:
    """Test estrazione sotto-grafo di una comunità."""

    def test_subgraph_structure(self):
        """Il sotto-grafo ha la struttura corretta."""
        mgr = make_mock_neo4j_manager()
        nodes = make_mock_nodes(8, community_id=5)
        edges = make_mock_edges(nodes, n_edges=10)
        mgr.get_nodes_by_community.return_value = nodes
        mgr.get_edges_within_community.return_value = edges

        detector = CommunityDetector(mgr)
        subgraph = detector.get_community_subgraph(community_id=5)

        assert subgraph["community_id"] == 5
        assert subgraph["size"] == 8
        assert len(subgraph["nodes"]) == 8
        assert len(subgraph["edges"]) == 10
        assert isinstance(subgraph["key_entities"], list)
        assert len(subgraph["key_entities"]) <= 5

    def test_subgraph_empty_community(self):
        """Sotto-grafo di comunità vuota."""
        mgr = make_mock_neo4j_manager()
        mgr.get_nodes_by_community.return_value = []
        mgr.get_edges_within_community.return_value = []

        detector = CommunityDetector(mgr)
        subgraph = detector.get_community_subgraph(community_id=99)

        assert subgraph["size"] == 0
        assert subgraph["nodes"] == []
        assert subgraph["edges"] == []
        assert subgraph["key_entities"] == []


# ============================================================================
# Test 5: get_all_communities
# ============================================================================


class TestGetAllCommunities:
    """Test recupero comunità già rilevate (senza re-detection)."""

    def test_get_all_returns_communities(self):
        """get_all_communities() carica comunità già esistenti."""
        mgr = make_mock_neo4j_manager()
        mgr.get_all_community_ids.return_value = [
            {"community_id": 0, "size": 10},
            {"community_id": 1, "size": 8},
        ]
        mgr.get_nodes_by_community.return_value = make_mock_nodes(8)
        mgr.get_edges_within_community.return_value = []

        detector = CommunityDetector(mgr)
        communities = detector.get_all_communities(min_size=3)

        assert len(communities) == 2
        mgr.get_all_community_ids.assert_called_once_with(
            community_property="community_id",
            min_size=3,
        )

    def test_get_all_no_communities(self):
        """get_all_communities() con grafo senza community_id."""
        mgr = make_mock_neo4j_manager()
        mgr.get_all_community_ids.return_value = []

        detector = CommunityDetector(mgr)
        result = detector.get_all_communities()
        assert result == []


# ============================================================================
# Test 6: get_community_stats
# ============================================================================


class TestGetCommunityStats:
    """Test statistiche aggregate."""

    def test_stats_structure(self):
        """get_community_stats() restituisce struttura corretta."""
        mgr = make_mock_neo4j_manager()
        mgr.get_all_community_ids.return_value = [
            {"community_id": 0, "size": 15},
            {"community_id": 1, "size": 10},
            {"community_id": 2, "size": 5},
        ]
        mgr.query_cypher.return_value = [{"c": 3}]
        mgr.get_nodes_by_community.return_value = make_mock_nodes(10)
        mgr.get_edges_within_community.return_value = make_mock_edges(
            make_mock_nodes(10), n_edges=5
        )

        detector = CommunityDetector(mgr)
        # Simula detection precedente
        detector._last_detection_result = {
            "modularity": 0.65,
            "algorithm": "louvain",
        }

        stats = detector.get_community_stats()

        assert stats["total_communities"] == 3
        assert stats["total_nodes_assigned"] == 30
        assert stats["modularity"] == 0.65
        assert stats["algorithm"] == "louvain"
        assert "size_distribution" in stats
        assert stats["size_distribution"]["min"] == 5
        assert stats["size_distribution"]["max"] == 15
        assert stats["size_distribution"]["mean"] == 10.0
        assert "top_5_communities" in stats
        assert len(stats["top_5_communities"]) == 3

    def test_stats_no_communities(self):
        """Statistiche con zero comunità."""
        mgr = make_mock_neo4j_manager()
        mgr.get_all_community_ids.return_value = []
        mgr.query_cypher.return_value = [{"c": 50}]

        detector = CommunityDetector(mgr)
        stats = detector.get_community_stats()

        assert stats["total_communities"] == 0
        assert stats["total_nodes_assigned"] == 0
        assert stats["total_nodes_unassigned"] == 50
        assert stats["size_distribution"]["min"] == 0
        assert stats["top_5_communities"] == []

    def test_stats_without_prior_detection(self):
        """Statistiche senza detection precedente (modularity=None)."""
        mgr = make_mock_neo4j_manager()
        mgr.get_all_community_ids.return_value = [
            {"community_id": 0, "size": 5},
        ]
        mgr.query_cypher.return_value = [{"c": 0}]
        mgr.get_nodes_by_community.return_value = make_mock_nodes(5)
        mgr.get_edges_within_community.return_value = []

        detector = CommunityDetector(mgr)
        stats = detector.get_community_stats()

        assert stats["modularity"] is None
        assert stats["algorithm"] is None


# ============================================================================
# Test 7: clear_community_labels
# ============================================================================


class TestClearCommunityLabels:
    """Test pulizia proprietà community_id."""

    def test_clear_labels(self):
        """clear_community_labels() rimuove la proprietà da tutti i nodi."""
        mgr = make_mock_neo4j_manager()
        mgr.query_cypher.return_value = [{"updated": 42}]

        detector = CommunityDetector(mgr)
        count = detector.clear_community_labels()

        assert count == 42
        mgr.query_cypher.assert_called_once()
        query_arg = mgr.query_cypher.call_args[0][0]
        assert "REMOVE" in query_arg
        assert "community_id" in query_arg

    def test_clear_labels_error(self):
        """Se la query fallisce, ritorna 0."""
        mgr = make_mock_neo4j_manager()
        mgr.query_cypher.side_effect = Exception("DB error")

        detector = CommunityDetector(mgr)
        count = detector.clear_community_labels()
        assert count == 0


# ============================================================================
# Test 8: Neo4jManager GDS methods (unit tests with mock)
# ============================================================================


class TestNeo4jManagerGDSMethods:
    """Test dei nuovi metodi GDS aggiunti a Neo4jManager (mock)."""

    def test_ensure_gds_plugin_available(self):
        """ensure_gds_plugin() con GDS presente."""
        from paperhunter.graphrag.neo4j_manager import Neo4jManager

        mgr = MagicMock(spec=Neo4jManager)
        mgr._run = MagicMock(return_value=[{"version": "2.6.0"}])
        # Chiama il metodo reale
        result = Neo4jManager.ensure_gds_plugin(mgr)
        assert result is True

    def test_ensure_gds_plugin_not_available(self):
        """ensure_gds_plugin() senza GDS."""
        from paperhunter.graphrag.neo4j_manager import Neo4jManager

        mgr = MagicMock(spec=Neo4jManager)
        mgr._run = MagicMock(side_effect=Exception("Unknown function gds.version"))
        result = Neo4jManager.ensure_gds_plugin(mgr)
        assert result is False

    def test_drop_graph_projection_exists(self):
        """drop_graph_projection() con proiezione esistente."""
        from paperhunter.graphrag.neo4j_manager import Neo4jManager

        mgr = MagicMock(spec=Neo4jManager)
        mgr._run = MagicMock(side_effect=[
            [{"exists": True}],   # gds.graph.exists
            [{"graphName": "test"}],  # gds.graph.drop
        ])
        result = Neo4jManager.drop_graph_projection(mgr, "test")
        assert result is True

    def test_drop_graph_projection_not_exists(self):
        """drop_graph_projection() con proiezione inesistente."""
        from paperhunter.graphrag.neo4j_manager import Neo4jManager

        mgr = MagicMock(spec=Neo4jManager)
        mgr._run = MagicMock(return_value=[{"exists": False}])
        result = Neo4jManager.drop_graph_projection(mgr, "test")
        assert result is False

    def test_run_community_detection_invalid_algorithm(self):
        """run_community_detection() con algoritmo non supportato."""
        from paperhunter.graphrag.neo4j_manager import Neo4jManager

        mgr = MagicMock(spec=Neo4jManager)
        with pytest.raises(ValueError, match="non supportato"):
            Neo4jManager.run_community_detection(mgr, algorithm="spectral")

    def test_get_all_community_ids(self):
        """get_all_community_ids() filtra per min_size."""
        from paperhunter.graphrag.neo4j_manager import Neo4jManager

        mgr = MagicMock(spec=Neo4jManager)
        mgr._run = MagicMock(return_value=[
            {"community_id": 0, "size": 15},
            {"community_id": 1, "size": 10},
            {"community_id": 2, "size": 2},
        ])
        result = Neo4jManager.get_all_community_ids(mgr, min_size=3)
        assert len(result) == 2
        assert result[0]["community_id"] == 0
        assert result[1]["community_id"] == 1


# ============================================================================
# Test 9: _extract_key_entities helper
# ============================================================================


class TestExtractKeyEntities:
    """Test funzione helper per entità chiave."""

    def test_key_entities_by_connections(self):
        """Le entità con più connessioni sono restituite per prime."""
        nodes = [
            {"entity_id": "e1", "canonical_name": "GCG"},
            {"entity_id": "e2", "canonical_name": "AutoDAN"},
            {"entity_id": "e3", "canonical_name": "SmoothLLM"},
        ]
        edges = [
            {"source": "GCG", "target": "AutoDAN"},
            {"source": "GCG", "target": "SmoothLLM"},
            {"source": "AutoDAN", "target": "SmoothLLM"},
            {"source": "GCG", "target": "GCG"},  # self-loop (conta)
        ]
        key = _extract_key_entities(nodes, edges, top_n=2)
        # GCG ha 4 connessioni, è il più connesso
        assert key[0] == "GCG"
        assert len(key) == 2

    def test_key_entities_empty_nodes(self):
        """Con nodi vuoti, restituisce lista vuota."""
        result = _extract_key_entities([], [], top_n=5)
        assert result == []

    def test_key_entities_no_edges(self):
        """Senza edges, restituisce i primi nodi per nome."""
        nodes = [
            {"entity_id": "e1", "canonical_name": "Alpha"},
            {"entity_id": "e2", "canonical_name": "Beta"},
            {"entity_id": "e3", "canonical_name": "Gamma"},
        ]
        key = _extract_key_entities(nodes, [], top_n=2)
        assert len(key) == 2
        assert key[0] == "Alpha"
        assert key[1] == "Beta"

    def test_key_entities_top_n_limit(self):
        """Rispetta il limite top_n."""
        nodes = make_mock_nodes(10, community_id=0)
        edges = make_mock_edges(nodes, n_edges=20)
        key = _extract_key_entities(nodes, edges, top_n=3)
        assert len(key) <= 3


# ============================================================================
# Test 10: Convenience standalone functions
# ============================================================================


class TestConvenienceFunctions:
    """Test funzioni standalone."""

    @patch("paperhunter.graphrag.neo4j_manager.Neo4jManager")
    def test_detect_graph_communities_standalone(self, MockManager):
        """detect_graph_communities() crea manager e detector internamente."""
        mock_mgr = make_mock_neo4j_manager()
        mock_mgr.ensure_gds_plugin.return_value = True
        mock_mgr.project_graph.return_value = {
            "node_count": 20, "relationship_count": 15,
        }
        mock_mgr.run_community_detection.return_value = {
            "community_count": 2, "modularity": 0.5,
            "node_properties_written": 20, "algorithm": "louvain",
        }
        mock_mgr.get_all_community_ids.return_value = [
            {"community_id": 0, "size": 12},
            {"community_id": 1, "size": 8},
        ]
        mock_mgr.get_nodes_by_community.return_value = make_mock_nodes(10)
        mock_mgr.get_edges_within_community.return_value = []
        mock_mgr.drop_graph_projection.return_value = True

        MockManager.return_value.__enter__ = MagicMock(return_value=mock_mgr)
        MockManager.return_value.__exit__ = MagicMock(return_value=False)

        communities = detect_graph_communities(algorithm="louvain", min_size=3)
        assert len(communities) == 2

    @patch("paperhunter.graphrag.neo4j_manager.Neo4jManager")
    def test_get_community_stats_standalone(self, MockManager):
        """get_community_stats_standalone() recupera statistiche."""
        mock_mgr = make_mock_neo4j_manager()
        mock_mgr.get_all_community_ids.return_value = [
            {"community_id": 0, "size": 5},
        ]
        mock_mgr.query_cypher.return_value = [{"c": 0}]
        mock_mgr.get_nodes_by_community.return_value = make_mock_nodes(5)
        mock_mgr.get_edges_within_community.return_value = []

        MockManager.return_value.__enter__ = MagicMock(return_value=mock_mgr)
        MockManager.return_value.__exit__ = MagicMock(return_value=False)

        stats = get_community_stats_standalone()
        assert stats["total_communities"] == 1


# ============================================================================
# Test 11: Logging helper
# ============================================================================


class TestLoggingHelper:
    """Test funzione di logging."""

    def test_log_does_not_raise(self):
        """Il logging non lancia eccezioni."""
        communities = [
            {"community_id": 0, "size": 10, "edges": [1, 2, 3], "key_entities": ["A", "B"]},
            {"community_id": 1, "size": 5, "edges": [1], "key_entities": ["C"]},
        ]
        # Non deve lanciare
        _log_community_detection_results(
            communities=communities,
            algorithm="louvain",
            modularity=0.65,
            elapsed=1.5,
            min_size=3,
            total_raw=5,
        )

    def test_log_empty_communities(self):
        """Logging con lista vuota non lancia."""
        _log_community_detection_results(
            communities=[],
            algorithm="leiden",
            modularity=0.0,
            elapsed=0.1,
            min_size=3,
            total_raw=0,
        )


# ============================================================================
# Test 12: Integration test con Neo4j reale (skip se non disponibile)
# ============================================================================


def _neo4j_available() -> bool:
    """Controlla se Neo4j è raggiungibile."""
    try:
        from paperhunter.graphrag.neo4j_manager import Neo4jManager
        mgr = Neo4jManager(auto_connect=True)
        mgr.close()
        return True
    except Exception:
        return False


def _gds_available() -> bool:
    """Controlla se Neo4j GDS è disponibile."""
    try:
        from paperhunter.graphrag.neo4j_manager import Neo4jManager
        with Neo4jManager() as mgr:
            return mgr.ensure_gds_plugin()
    except Exception:
        return False


@pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j non disponibile (docker non attivo?)",
)
class TestIntegrationNeo4j:
    """Test di integrazione con Neo4j reale."""

    def test_ensure_gds_real(self):
        """Verifica GDS sul server reale."""
        from paperhunter.graphrag.neo4j_manager import Neo4jManager

        with Neo4jManager() as mgr:
            result = mgr.ensure_gds_plugin()
            # Può essere True o False, non deve lanciare
            assert isinstance(result, bool)

    @pytest.mark.skipif(
        not _gds_available(),
        reason="Neo4j GDS plugin non installato",
    )
    def test_full_pipeline_real(self):
        """Pipeline completa su un grafo di test reale."""
        from paperhunter.graphrag.neo4j_manager import Neo4jManager

        entities = [
            Entity(type=EntityType.ATTACK, canonical_name=f"TestAttack_{i}",
                   year=2023, source_bibtex_key=f"test_{i}")
            for i in range(20)
        ] + [
            Entity(type=EntityType.DEFENSE, canonical_name=f"TestDefense_{i}",
                   year=2024, source_bibtex_key=f"test_def_{i}")
            for i in range(10)
        ]
        relations = [
            Relation(
                type=RelationType.TARGETS,
                source=f"TestAttack_{i}",
                source_type=EntityType.ATTACK,
                target=f"TestDefense_{i % 10}",
                target_type=EntityType.DEFENSE,
                source_bibtex_key=f"test_rel_{i}",
            )
            for i in range(20)
        ]

        with Neo4jManager() as mgr:
            # Importa entità di test
            mgr.import_entities_and_relations(
                entities, relations, show_progress=False, clear_before=True
            )

            # Esegui detection
            detector = CommunityDetector(mgr)
            try:
                communities = detector.detect_communities(
                    algorithm="louvain", min_size=2
                )
                assert isinstance(communities, list)

                # Verifica stats
                stats = detector.get_community_stats()
                assert stats["total_communities"] >= 0
                assert "size_distribution" in stats

            except GDSNotAvailableError:
                pytest.skip("GDS non disponibile")
            finally:
                # Cleanup
                mgr.clear_graph()


# ============================================================================
# Entrypoint
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
