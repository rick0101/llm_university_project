"""
GraphRAG Community Detector — Rilevamento comunità tematiche nel grafo Neo4j.

Identifica macro-temi strutturali tra le entità estratte, partizionando
il grafo in comunità tematiche tramite algoritmi Louvain o Leiden
(via Neo4j GDS plugin).

Le comunità così identificate possono essere:
- Riassunte in testi narrativi (``community_summarizer.py``, Task 2.2)
- Usate come fonte aggiuntiva per il writer agent (Task 3.x)
- Immagazzinate in Qdrant per il retrieval

Prerequisiti:
    - Neo4j con plugin GDS (Graph Data Science) installato::

        docker run --rm -d \\
            -p 7474:7474 -p 7687:7687 \\
            -e NEO4J_AUTH=neo4j/password \\
            -e NEO4J_PLUGINS='["graph-data-science"]' \\
            neo4j:5-community

    - ``neo4j.enabled: true`` in ``config.yaml``

Uso::

    from paperhunter.graphrag.community_detector import CommunityDetector
    from paperhunter.graphrag.neo4j_manager import Neo4jManager

    with Neo4jManager() as mgr:
        detector = CommunityDetector(mgr)

        # Esegui la detection completa
        communities = detector.detect_communities(algorithm="louvain")
        # → [{"community_id": 0, "nodes": [...], "edges": [...], "size": 15}, ...]

        # Singolo sotto-grafo
        subgraph = detector.get_community_subgraph(community_id=0)

        # Tutte le comunità con almeno 3 nodi
        all_coms = detector.get_all_communities(min_size=3)

        # Statistiche
        stats = detector.get_community_stats()
        print(f"Comunità: {stats['total_communities']}, Modularity: {stats['modularity']}")
"""

from __future__ import annotations

import time
from typing import Any, Optional

from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_success,
    print_table,
    print_warning,
)

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class GDSNotAvailableError(RuntimeError):
    """Raised when Neo4j GDS plugin is not installed/available."""
    pass


class CommunityDetectionError(RuntimeError):
    """Raised when community detection fails."""
    pass


# ============================================================================
# Community Detector
# ============================================================================

class CommunityDetector:
    """Rilevamento di comunità tematiche nel grafo Neo4j via GDS.

    Utilizza il ``Neo4jManager`` per interfacciarsi con Neo4j e il plugin
    GDS (Graph Data Science) per eseguire algoritmi di community detection
    (Louvain o Leiden).

    Workflow tipico:
    1. Verifica disponibilità GDS (``ensure_gds()``)
    2. Crea proiezione GDS del grafo
    3. Esegui l'algoritmo Louvain/Leiden
    4. Recupera le comunità con sotto-grafi
    5. Cleanup della proiezione

    Args:
        neo4j_manager: Istanza di ``Neo4jManager`` già connessa.
        graph_name: Nome della proiezione GDS (default: ``community_graph``).
        community_property: Nome della proprietà Neo4j per il community ID
            (default: ``community_id``).

    Raises:
        ValueError: Se ``neo4j_manager`` non è connesso.
    """

    def __init__(
        self,
        neo4j_manager: Any,
        graph_name: str = "community_graph",
        community_property: str = "community_id",
    ):
        if not neo4j_manager.is_connected:
            raise ValueError(
                "Il Neo4jManager non è connesso. Usa il context manager o chiama connect()."
            )
        self._mgr = neo4j_manager
        self._graph_name = graph_name
        self._community_property = community_property
        self._gds_available: bool | None = None
        self._last_detection_result: dict | None = None

    # ------------------------------------------------------------------
    # GDS availability
    # ------------------------------------------------------------------

    def ensure_gds(self) -> bool:
        """Verifica che il plugin GDS sia disponibile in Neo4j.

        Caching: il risultato viene memorizzato dopo la prima chiamata.

        Returns:
            True se GDS è disponibile.

        Raises:
            GDSNotAvailableError: Se GDS non è disponibile.
        """
        if self._gds_available is None:
            self._gds_available = self._mgr.ensure_gds_plugin()

        if not self._gds_available:
            raise GDSNotAvailableError(
                "Il plugin Neo4j GDS non è installato. "
                "Avvia Neo4j con: NEO4J_PLUGINS='[\"graph-data-science\"]' "
                "oppure installa il plugin manualmente."
            )
        return True

    # ------------------------------------------------------------------
    # Main detection workflow
    # ------------------------------------------------------------------

    def detect_communities(
        self,
        algorithm: str = "louvain",
        min_size: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[dict]:
        """Esegue la pipeline completa di community detection.

        Workflow:
        1. Verifica GDS
        2. Crea proiezione del grafo
        3. Esegue l'algoritmo Louvain/Leiden
        4. Recupera tutti i sotto-grafi delle comunità
        5. Cleanup della proiezione

        Args:
            algorithm: ``"louvain"`` o ``"leiden"``. Default ``"louvain"``.
            min_size: Dimensione minima di una comunità (escluse comunità
                con meno nodi). Default 3.
            relationship_types: Tipi di relazione da includere nella proiezione.
                Se None, include tutti i tipi.

        Returns:
            Lista di dizionari, uno per comunità, ordinati per dimensione
            decrescente::

                [
                    {
                        "community_id": 0,
                        "nodes": [{"entity_id": ..., "canonical_name": ..., ...}, ...],
                        "edges": [{"source": ..., "target": ..., "rel_type": ..., ...}, ...],
                        "size": 15,
                        "key_entities": ["GCG", "AutoDAN", "Adversarial Suffix"],
                    },
                    ...
                ]

        Raises:
            GDSNotAvailableError: Se il plugin GDS non è disponibile.
            CommunityDetectionError: Se la detection fallisce.
        """
        t0 = time.time()

        # 1. Verifica GDS
        self.ensure_gds()

        # 2. Proiezione
        try:
            projection_info = self._mgr.project_graph(
                graph_name=self._graph_name,
                relationship_types=relationship_types,
            )
        except RuntimeError as e:
            raise CommunityDetectionError(
                f"Impossibile creare la proiezione GDS: {e}"
            ) from e

        if projection_info.get("node_count", 0) == 0:
            logger.warning("Il grafo è vuoto, nessuna comunità da rilevare.")
            self._mgr.drop_graph_projection(self._graph_name)
            return []

        # 3. Esecuzione algoritmo
        try:
            detection_result = self._mgr.run_community_detection(
                graph_name=self._graph_name,
                algorithm=algorithm,
                write_property=self._community_property,
            )
            self._last_detection_result = detection_result
        except (RuntimeError, ValueError) as e:
            self._mgr.drop_graph_projection(self._graph_name)
            raise CommunityDetectionError(
                f"Community detection ({algorithm}) fallita: {e}"
            ) from e

        # 4. Recupera le comunità
        try:
            community_ids = self._mgr.get_all_community_ids(
                community_property=self._community_property,
                min_size=min_size,
            )

            communities: list[dict] = []
            for cinfo in community_ids:
                cid = cinfo["community_id"]
                subgraph = self.get_community_subgraph(cid)
                communities.append(subgraph)

        except Exception as e:
            raise CommunityDetectionError(
                f"Errore nel recupero delle comunità: {e}"
            ) from e
        finally:
            # 5. Cleanup
            self._mgr.drop_graph_projection(self._graph_name)

        elapsed = time.time() - t0

        # Logging con Rich
        _log_community_detection_results(
            communities=communities,
            algorithm=algorithm,
            modularity=detection_result.get("modularity", 0.0),
            elapsed=elapsed,
            min_size=min_size,
            total_raw=detection_result.get("community_count", 0),
        )

        return communities

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def get_community_subgraph(self, community_id: int) -> dict:
        """Estrae il sotto-grafo completo di una singola comunità.

        Recupera tutti i nodi e le relazioni interne alla comunità
        identificata dal ``community_id`` salvato come proprietà Neo4j.

        Args:
            community_id: ID della comunità (assegnato dall'algoritmo GDS).

        Returns:
            Dizionario con:
            - ``community_id``: int
            - ``nodes``: lista di dizionari nodo
            - ``edges``: lista di dizionari relazione
            - ``size``: numero di nodi
            - ``key_entities``: lista dei nomi delle entità chiave
              (top N per connessioni)
        """
        nodes = self._mgr.get_nodes_by_community(
            community_id=community_id,
            community_property=self._community_property,
        )

        edges = self._mgr.get_edges_within_community(
            community_id=community_id,
            community_property=self._community_property,
        )

        # Identifica le entità chiave (top 5 per numero di connessioni)
        key_entities = _extract_key_entities(nodes, edges, top_n=5)

        return {
            "community_id": community_id,
            "nodes": nodes,
            "edges": edges,
            "size": len(nodes),
            "key_entities": key_entities,
        }

    # ------------------------------------------------------------------
    # Community listing
    # ------------------------------------------------------------------

    def get_all_communities(self, min_size: int = 3) -> list[dict]:
        """Restituisce tutte le comunità con sotto-grafi completi.

        A differenza di ``detect_communities()``, questa funzione non esegue
        l'algoritmo di detection ma si limita a leggere i ``community_id``
        già assegnati ai nodi nel grafo Neo4j.

        Utile per rileggere le comunità dopo una detection precedente
        senza rieseguire l'algoritmo.

        Args:
            min_size: Dimensione minima di una comunità.

        Returns:
            Lista di dizionari comunità, ordinati per dimensione decrescente.
        """
        community_ids = self._mgr.get_all_community_ids(
            community_property=self._community_property,
            min_size=min_size,
        )

        communities: list[dict] = []
        for cinfo in community_ids:
            cid = cinfo["community_id"]
            subgraph = self.get_community_subgraph(cid)
            communities.append(subgraph)

        logger.info(
            f"Caricate {len(communities)} comunità (min_size={min_size}) "
            f"dal grafo Neo4j"
        )

        return communities

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_community_stats(self) -> dict[str, Any]:
        """Restituisce statistiche aggregate sulle comunità nel grafo.

        Le statistiche si basano sui ``community_id`` già assegnati ai nodi
        (dopo una detection). Include distribuzione dimensioni e top-5.

        Returns:
            Dizionario con:
            - ``total_communities``: conteggio comunità (con ≥1 nodo)
            - ``total_nodes_assigned``: nodi con community_id
            - ``total_nodes_unassigned``: nodi senza community_id
            - ``modularity``: modularity dall'ultima detection (se disponibile)
            - ``algorithm``: algoritmo usato nell'ultima detection
            - ``size_distribution``: ``{"min": int, "max": int, "mean": float, "median": float}``
            - ``top_5_communities``: lista dei 5 cluster più grandi con
              ``{"community_id": int, "size": int, "key_entities": list[str]}``
        """
        community_ids = self._mgr.get_all_community_ids(
            community_property=self._community_property,
            min_size=1,
        )

        sizes = [c["size"] for c in community_ids]

        stats: dict[str, Any] = {
            "total_communities": len(community_ids),
            "total_nodes_assigned": sum(sizes) if sizes else 0,
        }

        # Nodi senza community_id
        try:
            result = self._mgr.query_cypher(
                f"MATCH (n:Entity) WHERE n.{self._community_property} IS NULL "
                f"RETURN count(n) AS c"
            )
            stats["total_nodes_unassigned"] = result[0]["c"] if result else 0
        except Exception:
            stats["total_nodes_unassigned"] = 0

        # Modularity e algoritmo dall'ultima detection
        if self._last_detection_result:
            stats["modularity"] = self._last_detection_result.get("modularity", 0.0)
            stats["algorithm"] = self._last_detection_result.get("algorithm", "unknown")
        else:
            stats["modularity"] = None
            stats["algorithm"] = None

        # Distribuzione dimensioni
        if sizes:
            sorted_sizes = sorted(sizes)
            n = len(sorted_sizes)
            median = (
                sorted_sizes[n // 2]
                if n % 2 == 1
                else (sorted_sizes[n // 2 - 1] + sorted_sizes[n // 2]) / 2
            )
            stats["size_distribution"] = {
                "min": min(sizes),
                "max": max(sizes),
                "mean": round(sum(sizes) / len(sizes), 2),
                "median": median,
            }
        else:
            stats["size_distribution"] = {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}

        # Top 5 comunità
        top_5: list[dict] = []
        for cinfo in community_ids[:5]:
            cid = cinfo["community_id"]
            nodes = self._mgr.get_nodes_by_community(
                community_id=cid,
                community_property=self._community_property,
            )
            edges = self._mgr.get_edges_within_community(
                community_id=cid,
                community_property=self._community_property,
            )
            key_ents = _extract_key_entities(nodes, edges, top_n=3)
            top_5.append({
                "community_id": cid,
                "size": cinfo["size"],
                "key_entities": key_ents,
            })
        stats["top_5_communities"] = top_5

        return stats

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_community_labels(self) -> int:
        """Rimuove la proprietà ``community_id`` da tutti i nodi Entity.

        Utile per re-eseguire la detection da zero.

        Returns:
            Numero di nodi aggiornati.
        """
        try:
            result = self._mgr.query_cypher(
                f"MATCH (n:Entity) WHERE n.{self._community_property} IS NOT NULL "
                f"REMOVE n.{self._community_property} "
                f"RETURN count(n) AS updated"
            )
            count = result[0]["updated"] if result else 0
            logger.info(f"Rimossa proprietà '{self._community_property}' da {count} nodi.")
            return count
        except Exception as e:
            logger.error(f"Errore pulizia community labels: {e}")
            return 0

    def __repr__(self) -> str:
        return (
            f"CommunityDetector(graph_name={self._graph_name!r}, "
            f"property={self._community_property!r}, "
            f"gds={'available' if self._gds_available else 'unknown'})"
        )


# ============================================================================
# Helper functions
# ============================================================================

def _extract_key_entities(
    nodes: list[dict],
    edges: list[dict],
    top_n: int = 5,
) -> list[str]:
    """Identifica le entità chiave di una comunità per numero di connessioni.

    Le entità con più relazioni (sia in ingresso che in uscita) sono
    considerate più centrali / importanti nella comunità.

    Args:
        nodes: Lista di nodi della comunità.
        edges: Lista di relazioni interne.
        top_n: Numero massimo di entità chiave da restituire.

    Returns:
        Lista dei ``canonical_name`` delle entità più connesse.
    """
    if not nodes:
        return []

    # Conta le connessioni per ogni nodo
    connection_count: dict[str, int] = {}
    node_names: dict[str, str] = {}  # entity_id → canonical_name

    for node in nodes:
        eid = node.get("entity_id", "")
        name = node.get("canonical_name", eid)
        node_names[eid] = name
        connection_count[name] = 0

    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src in connection_count:
            connection_count[src] += 1
        if tgt in connection_count:
            connection_count[tgt] += 1

    # Se non ci sono edges, restituisci i primi nodi per nome
    if not edges:
        return [n.get("canonical_name", "") for n in nodes[:top_n]]

    # Ordina per numero di connessioni (decrescente)
    sorted_entities = sorted(
        connection_count.items(), key=lambda x: x[1], reverse=True
    )

    return [name for name, _ in sorted_entities[:top_n]]


def _log_community_detection_results(
    communities: list[dict],
    algorithm: str,
    modularity: float,
    elapsed: float,
    min_size: int,
    total_raw: int,
) -> None:
    """Logga i risultati della community detection con formattazione Rich.

    Args:
        communities: Comunità trovate (filtrate per min_size).
        algorithm: Algoritmo usato.
        modularity: Score di modularity.
        elapsed: Tempo di esecuzione in secondi.
        min_size: Soglia minima usata.
        total_raw: Totale comunità prima del filtro.
    """
    sizes = [c["size"] for c in communities]

    logger.info(
        f"Community detection completata in {elapsed:.1f}s — "
        f"Algoritmo: {algorithm}, Modularity: {modularity:.4f}"
    )
    logger.info(
        f"Comunità: {len(communities)} (filtrate da {total_raw}, min_size={min_size})"
    )

    if sizes:
        logger.info(
            f"Distribuzione dimensioni: "
            f"min={min(sizes)}, max={max(sizes)}, "
            f"media={sum(sizes)/len(sizes):.1f}"
        )

        # Top-5 comunità
        for i, c in enumerate(communities[:5]):
            key_ents = ", ".join(c.get("key_entities", [])[:3])
            logger.info(
                f"  Top-{i+1}: Community {c['community_id']} "
                f"({c['size']} nodi, {len(c.get('edges', []))} relazioni) "
                f"— Key: {key_ents}"
            )

    try:
        summary = {
            "Algoritmo": algorithm,
            "Modularity": f"{modularity:.4f}",
            "Comunità (filtrate)": str(len(communities)),
            "Comunità (totale raw)": str(total_raw),
            "Min size": str(min_size),
            "Nodi totali": str(sum(sizes)) if sizes else "0",
            "Tempo": f"{elapsed:.1f}s",
        }
        print_table("Community Detection Results", summary)
    except Exception:
        pass


# ============================================================================
# Standalone convenience functions
# ============================================================================

def detect_graph_communities(
    algorithm: str = "louvain",
    min_size: int = 3,
) -> list[dict]:
    """Convenience: esegue la community detection completa.

    Crea un ``Neo4jManager`` e un ``CommunityDetector``, esegue
    la detection e restituisce le comunità. Gestisce la connessione
    automaticamente.

    Args:
        algorithm: ``"louvain"`` o ``"leiden"``.
        min_size: Dimensione minima delle comunità.

    Returns:
        Lista di comunità con sotto-grafi.

    Raises:
        GDSNotAvailableError: Se GDS non è installato.
        CommunityDetectionError: Se la detection fallisce.

    Uso::

        from paperhunter.graphrag.community_detector import detect_graph_communities

        communities = detect_graph_communities(algorithm="louvain", min_size=3)
        for c in communities:
            print(f"Community {c['community_id']}: {c['size']} nodi")
    """
    from paperhunter.graphrag.neo4j_manager import Neo4jManager

    with Neo4jManager() as mgr:
        detector = CommunityDetector(mgr)
        return detector.detect_communities(
            algorithm=algorithm,
            min_size=min_size,
        )


def get_community_stats_standalone() -> dict:
    """Convenience: recupera le statistiche delle comunità già rilevate.

    Returns:
        Dizionario con statistiche aggregate.
    """
    from paperhunter.graphrag.neo4j_manager import Neo4jManager

    with Neo4jManager() as mgr:
        detector = CommunityDetector(mgr)
        return detector.get_community_stats()
