"""
GraphRAG Neo4j Manager — Connessione e gestione Neo4j Community Edition.

Gestisce la connessione a un'istanza Neo4j locale (Community Edition via Docker)
e fornisce operazioni CRUD per entità e relazioni del knowledge graph.

Setup Docker::

    docker run --rm \\
        -p 7474:7474 -p 7687:7687 \\
        -e NEO4J_AUTH=neo4j/password \\
        -v $PWD/neo4j_data:/data \\
        neo4j:5-community

Uso::

    from paperhunter.graphrag.neo4j_manager import Neo4jManager

    with Neo4jManager() as mgr:
        # Connessione automatica
        mgr.create_constraints()
        mgr.upsert_entity(entity)
        mgr.upsert_relation(relation)

        # Query
        neighbors = mgr.query_neighbors("GCG", depth=2)
        path = mgr.query_path("GCG", "GPT-4")
        timeline = mgr.get_temporal_chain("Attack")

        # Bulk import
        mgr.import_entities_and_relations(entities, relations)

        # Statistiche
        stats = mgr.get_stats()

Note:
    Il modulo è opzionale. Neo4j deve essere abilitato in ``config.yaml`` → ``neo4j.enabled: true``.
    Se disabilitato, le operazioni lanciano ``Neo4jNotEnabledError``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.graphrag.entity_extractor import (
    Entity,
    EntityType,
    Relation,
    RelationType,
)
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_success,
    print_table,
    print_warning,
    progress_bar,
)

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class Neo4jNotEnabledError(RuntimeError):
    """Raised when Neo4j operations are attempted but neo4j.enabled is False."""
    pass


class Neo4jConnectionError(RuntimeError):
    """Raised when connection to Neo4j fails."""
    pass


# ============================================================================
# Neo4j Manager
# ============================================================================

class Neo4jManager:
    """Gestore della connessione e delle operazioni CRUD su Neo4j.

    Legge la configurazione da ``config.yaml`` → sezione ``neo4j``.
    Supporta context manager per apertura/chiusura automatica.

    Args:
        uri: URI di connessione (default da config.yaml).
        username: Username (default da config.yaml).
        password: Password (default da config.yaml).
        database: Nome del database (default da config.yaml).
        auto_connect: Se True, connette nel costruttore.

    Raises:
        Neo4jNotEnabledError: Se ``neo4j.enabled`` è False in config.yaml.
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        auto_connect: bool = True,
    ):
        from paperhunter.config import settings
        cfg = settings.neo4j

        if not cfg.enabled:
            raise Neo4jNotEnabledError(
                "Neo4j è disabilitato in config.yaml. "
                "Imposta 'neo4j.enabled: true' e avvia il container Docker."
            )

        self._uri = uri or cfg.uri
        self._username = username or cfg.username
        self._password = password or cfg.password
        self._database = database or cfg.database
        self._driver = None

        if auto_connect:
            self.connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Stabilisce la connessione a Neo4j.

        Raises:
            Neo4jConnectionError: Se la connessione fallisce.
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "Il pacchetto 'neo4j' non è installato. "
                "Installalo con: pip install neo4j  oppure  pip install -e '.[graphrag]'"
            )

        try:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._username, self._password),
            )
            # Verifica connessione
            self._driver.verify_connectivity()
            logger.info(f"Connesso a Neo4j: {self._uri} (database: {self._database})")
        except Exception as e:
            self._driver = None
            raise Neo4jConnectionError(
                f"Impossibile connettersi a Neo4j ({self._uri}): {e}"
            ) from e

    def close(self) -> None:
        """Chiude la connessione a Neo4j."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Connessione Neo4j chiusa.")

    @property
    def is_connected(self) -> bool:
        """True se il driver è attivo."""
        return self._driver is not None

    def _ensure_connected(self) -> None:
        """Verifica che la connessione sia attiva."""
        if not self._driver:
            raise Neo4jConnectionError(
                "Non connesso a Neo4j. Chiama connect() o usa il context manager."
            )

    def _run(self, query: str, parameters: dict | None = None) -> list[dict]:
        """Esegue una query Cypher e ritorna i risultati come lista di dizionari.

        Args:
            query: Query Cypher.
            parameters: Parametri della query.

        Returns:
            Lista di dizionari con i risultati.
        """
        self._ensure_connected()
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def _write(self, query: str, parameters: dict | None = None) -> Any:
        """Esegue una query di scrittura in una transazione.

        Args:
            query: Query Cypher di scrittura.
            parameters: Parametri della query.

        Returns:
            Summary della transazione.
        """
        self._ensure_connected()
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters or {})
            summary = result.consume()
            return summary

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Neo4jManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"Neo4jManager(uri={self._uri!r}, database={self._database!r}, status={status})"

    # ------------------------------------------------------------------
    # Schema & Constraints
    # ------------------------------------------------------------------

    def create_constraints(self) -> None:
        """Crea vincoli di unicità e indici per il grafo.

        Vincoli:
        - Unicità su ``entity_id`` per ogni tipo di entità.
        - Indici su ``canonical_name``, ``year``.
        """
        logger.info("Creazione vincoli e indici Neo4j...")

        # Vincolo unicità su entity_id per ogni label EntityType
        for et in EntityType:
            label = et.value  # es. "Attack", "Defense", ...
            try:
                self._write(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.entity_id IS UNIQUE"
                )
                logger.debug(f"Vincolo unicità creato per :{label}.entity_id")
            except Exception as e:
                logger.warning(f"Vincolo per :{label} non creato: {e}")

        # Indici su canonical_name e year
        try:
            self._write(
                "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.canonical_name)"
            )
            self._write(
                "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.year)"
            )
            logger.debug("Indici su Entity.canonical_name e Entity.year creati.")
        except Exception as e:
            logger.warning(f"Creazione indici Entity fallita: {e}")

        logger.info("Vincoli e indici Neo4j creati con successo.")

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def create_entity(self, entity: Entity) -> bool:
        """Crea una singola entità nel grafo (CREATE, non upsert).

        Args:
            entity: Entità da creare.

        Returns:
            True se creata, False se già esiste.
        """
        return self._upsert_entity_internal(entity, mode="create")

    def upsert_entity(self, entity: Entity) -> bool:
        """Crea o aggiorna un'entità nel grafo (MERGE).

        L'entità viene matchata per ``entity_id``.
        Se esiste, i campi vengono aggiornati; gli alias vengono uniti.

        Args:
            entity: Entità da inserire/aggiornare.

        Returns:
            True se operazione riuscita.
        """
        return self._upsert_entity_internal(entity, mode="merge")

    def _upsert_entity_internal(self, entity: Entity, mode: str = "merge") -> bool:
        """Implementazione interna per create/upsert entity.

        Il nodo ha due label:
        - ``Entity`` (label generica per query cross-type)
        - Il tipo specifico (es. ``Attack``, ``Defense``, ecc.)

        Properties:
        - entity_id, canonical_name, aliases (JSON array), description,
          year, source_paper_id, source_bibtex_key, properties (JSON),
          updated_at
        """
        label = entity.type.value if isinstance(entity.type, EntityType) else entity.type

        props = {
            "entity_id": entity.id,
            "canonical_name": entity.canonical_name,
            "aliases": json.dumps(entity.aliases, ensure_ascii=False),
            "description": entity.description or "",
            "year": entity.year,
            "source_paper_id": entity.source_paper_id,
            "source_bibtex_key": entity.source_bibtex_key,
            "properties": json.dumps(entity.properties, ensure_ascii=False) if entity.properties else "{}",
            "updated_at": datetime.now().isoformat(),
        }

        if mode == "merge":
            query = (
                f"MERGE (n:Entity:{label} {{entity_id: $entity_id}}) "
                f"ON CREATE SET n += $props "
                f"ON MATCH SET n += $props "
                f"RETURN n.entity_id AS eid"
            )
        else:
            # CREATE — fallisce se esiste duplicato (se vincolo attivo)
            query = (
                f"CREATE (n:Entity:{label} $props) "
                f"RETURN n.entity_id AS eid"
            )

        try:
            self._run(query, {"entity_id": entity.id, "props": props})
            logger.debug(f"Entity upserted: [{label}] {entity.canonical_name}")
            return True
        except Exception as e:
            logger.error(f"Errore upsert entity {entity.canonical_name}: {e}")
            return False

    def get_entity(self, entity_id: str) -> dict | None:
        """Recupera un'entità per ID.

        Args:
            entity_id: ID dell'entità.

        Returns:
            Dizionario con le proprietà dell'entità, o None se non trovata.
        """
        result = self._run(
            "MATCH (n:Entity {entity_id: $eid}) "
            "RETURN n, labels(n) AS labels",
            {"eid": entity_id},
        )
        if not result:
            return None

        node = result[0]["n"]
        labels = result[0]["labels"]

        return _node_to_dict(node, labels)

    def get_entity_by_name(
        self,
        canonical_name: str,
        entity_type: EntityType | str | None = None,
    ) -> dict | None:
        """Recupera un'entità per nome canonico (e opzionalmente tipo).

        Args:
            canonical_name: Nome canonico dell'entità.
            entity_type: Tipo opzionale per disambiguare nomi duplicati.

        Returns:
            Dizionario con le proprietà o None.
        """
        if entity_type:
            label = entity_type.value if isinstance(entity_type, EntityType) else entity_type
            query = (
                f"MATCH (n:{label}) "
                f"WHERE n.canonical_name = $name "
                f"RETURN n, labels(n) AS labels LIMIT 1"
            )
        else:
            query = (
                "MATCH (n:Entity) "
                "WHERE n.canonical_name = $name "
                "RETURN n, labels(n) AS labels LIMIT 1"
            )

        result = self._run(query, {"name": canonical_name})
        if not result:
            return None
        return _node_to_dict(result[0]["n"], result[0]["labels"])

    def delete_entity(self, entity_id: str) -> bool:
        """Elimina un'entità e tutte le sue relazioni.

        Args:
            entity_id: ID dell'entità.

        Returns:
            True se l'entità è stata eliminata.
        """
        result = self._run(
            "MATCH (n:Entity {entity_id: $eid}) "
            "DETACH DELETE n "
            "RETURN count(n) AS deleted",
            {"eid": entity_id},
        )
        deleted = result[0]["deleted"] if result else 0
        if deleted > 0:
            logger.debug(f"Entity eliminata: {entity_id}")
        return deleted > 0

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def create_relation(self, relation: Relation) -> bool:
        """Crea una singola relazione nel grafo (CREATE).

        Args:
            relation: Relazione da creare.

        Returns:
            True se creata.
        """
        return self._upsert_relation_internal(relation, mode="create")

    def upsert_relation(self, relation: Relation) -> bool:
        """Crea o aggiorna una relazione nel grafo (MERGE).

        Le entità sorgente e target devono già esistere.
        La relazione viene matchata per ``source.entity_id`` + ``target.entity_id`` + tipo.

        Args:
            relation: Relazione da inserire/aggiornare.

        Returns:
            True se operazione riuscita.
        """
        return self._upsert_relation_internal(relation, mode="merge")

    def _upsert_relation_internal(self, relation: Relation, mode: str = "merge") -> bool:
        """Implementazione interna per create/upsert relation.

        Il match è basato su entity_id delle entità sorgente e target,
        non sul nome canonico, per evitare ambiguità.
        """
        from paperhunter.graphrag.entity_extractor import _make_entity_id

        # Calcola entity_id di source e target
        source_eid = _make_entity_id(relation.source_type, relation.source)
        target_eid = _make_entity_id(relation.target_type, relation.target)
        rel_type = relation.type.value if isinstance(relation.type, RelationType) else relation.type

        props = {
            "relation_id": relation.id,
            "description": relation.description or "",
            "year": relation.year,
            "source_paper_id": relation.source_paper_id,
            "source_bibtex_key": relation.source_bibtex_key,
            "properties": json.dumps(relation.properties, ensure_ascii=False) if relation.properties else "{}",
            "updated_at": datetime.now().isoformat(),
        }

        if mode == "merge":
            query = (
                f"MATCH (s:Entity {{entity_id: $source_eid}}) "
                f"MATCH (t:Entity {{entity_id: $target_eid}}) "
                f"MERGE (s)-[r:{rel_type}]->(t) "
                f"ON CREATE SET r += $props "
                f"ON MATCH SET r += $props "
                f"RETURN r.relation_id AS rid"
            )
        else:
            query = (
                f"MATCH (s:Entity {{entity_id: $source_eid}}) "
                f"MATCH (t:Entity {{entity_id: $target_eid}}) "
                f"CREATE (s)-[r:{rel_type} $props]->(t) "
                f"RETURN r.relation_id AS rid"
            )

        try:
            result = self._run(query, {
                "source_eid": source_eid,
                "target_eid": target_eid,
                "props": props,
            })
            if result:
                logger.debug(
                    f"Relation upserted: {relation.source} -[{rel_type}]-> {relation.target}"
                )
                return True
            else:
                logger.warning(
                    f"Relation non creata (entità mancanti?): "
                    f"{relation.source} -[{rel_type}]-> {relation.target}"
                )
                return False
        except Exception as e:
            logger.error(f"Errore upsert relation {relation.source} → {relation.target}: {e}")
            return False

    def delete_relation(self, relation_id: str) -> bool:
        """Elimina una relazione per ID.

        Args:
            relation_id: ID della relazione.

        Returns:
            True se eliminata.
        """
        result = self._run(
            "MATCH ()-[r]->() WHERE r.relation_id = $rid "
            "DELETE r RETURN count(r) AS deleted",
            {"rid": relation_id},
        )
        deleted = result[0]["deleted"] if result else 0
        return deleted > 0

    # ------------------------------------------------------------------
    # Bulk import
    # ------------------------------------------------------------------

    def import_entities_and_relations(
        self,
        entities: list[Entity],
        relations: list[Relation],
        *,
        show_progress: bool = True,
        clear_before: bool = False,
    ) -> dict[str, int]:
        """Import bulk di entità e relazioni nel grafo.

        Import workflow:
        1. (Opzionale) Pulisci il grafo
        2. Crea vincoli/indici
        3. Upsert tutte le entità
        4. Upsert tutte le relazioni (le entità devono già esistere)

        Args:
            entities: Lista di entità da importare.
            relations: Lista di relazioni da importare.
            show_progress: Se True, mostra progress bar.
            clear_before: Se True, pulisce il grafo prima dell'import.

        Returns:
            Dizionario con statistiche: entities_created, entities_failed,
            relations_created, relations_failed.
        """
        t0 = time.time()
        stats = {
            "entities_created": 0,
            "entities_failed": 0,
            "relations_created": 0,
            "relations_failed": 0,
        }

        if clear_before:
            self.clear_graph()

        # Assicura che vincoli e indici esistano
        self.create_constraints()

        # --- Entità ---
        entity_iter = progress_bar(entities, "Importing entities") if show_progress else entities
        for entity in entity_iter:
            if self.upsert_entity(entity):
                stats["entities_created"] += 1
            else:
                stats["entities_failed"] += 1

        # --- Relazioni ---
        relation_iter = progress_bar(relations, "Importing relations") if show_progress else relations
        for relation in relation_iter:
            if self.upsert_relation(relation):
                stats["relations_created"] += 1
            else:
                stats["relations_failed"] += 1

        elapsed = time.time() - t0
        stats["elapsed_seconds"] = round(elapsed, 2)

        logger.info(
            f"Import completato in {elapsed:.1f}s: "
            f"{stats['entities_created']} entità, {stats['relations_created']} relazioni "
            f"({stats['entities_failed']} err entity, {stats['relations_failed']} err rel)"
        )

        if show_progress:
            print_table("Import Results", stats)

        return stats

    def import_from_json(
        self,
        json_path: str | Path,
        *,
        show_progress: bool = True,
        clear_before: bool = False,
    ) -> dict[str, int]:
        """Import entità e relazioni da un file JSON (formato save_extraction_results).

        Args:
            json_path: Percorso al file JSON.
            show_progress: Se True, mostra progress bar.
            clear_before: Se True, pulisce il grafo prima dell'import.

        Returns:
            Statistiche dell'import.

        Raises:
            FileNotFoundError: Se il file non esiste.
            ValueError: Se il formato JSON non è valido.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        if "entities" not in data or "relations" not in data:
            raise ValueError(
                f"Formato JSON non valido: mancano 'entities' o 'relations' in {path}"
            )

        # Ricostruisci oggetti Entity e Relation
        entities = []
        for raw_e in data["entities"]:
            try:
                entities.append(Entity(
                    id=raw_e.get("id", ""),
                    type=EntityType(raw_e["type"]),
                    canonical_name=raw_e["canonical_name"],
                    aliases=raw_e.get("aliases", []),
                    description=raw_e.get("description", ""),
                    year=raw_e.get("year"),
                    source_paper_id=raw_e.get("source_paper_id"),
                    source_bibtex_key=raw_e.get("source_bibtex_key"),
                    properties=raw_e.get("properties", {}),
                ))
            except Exception as e:
                logger.warning(f"Entità non valida nel JSON, skipping: {e}")

        relations = []
        for raw_r in data["relations"]:
            try:
                relations.append(Relation(
                    id=raw_r.get("id", ""),
                    type=RelationType(raw_r["type"]),
                    source=raw_r["source"],
                    source_type=EntityType(raw_r["source_type"]),
                    target=raw_r["target"],
                    target_type=EntityType(raw_r["target_type"]),
                    description=raw_r.get("description", ""),
                    year=raw_r.get("year"),
                    properties=raw_r.get("properties", {}),
                    source_paper_id=raw_r.get("source_paper_id"),
                    source_bibtex_key=raw_r.get("source_bibtex_key"),
                ))
            except Exception as e:
                logger.warning(f"Relazione non valida nel JSON, skipping: {e}")

        logger.info(f"Caricati {len(entities)} entità e {len(relations)} relazioni da {path}")
        return self.import_entities_and_relations(
            entities, relations,
            show_progress=show_progress,
            clear_before=clear_before,
        )

    # ------------------------------------------------------------------
    # Query: neighbors
    # ------------------------------------------------------------------

    def query_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
        entity_type: EntityType | str | None = None,
    ) -> dict[str, Any]:
        """Recupera il sottografo locale (neighbors) di un'entità.

        Args:
            entity_name: Nome canonico dell'entità.
            depth: Profondità di ricerca (1 = diretti, 2 = amici di amici, ecc.).
            entity_type: Tipo opzionale per disambiguare.

        Returns:
            Dizionario con ``center``, ``entities`` (list), ``relations`` (list).
        """
        # Trova il nodo centrale
        center = self.get_entity_by_name(entity_name, entity_type)
        if not center:
            logger.warning(f"Entità non trovata: {entity_name}")
            return {"center": None, "entities": [], "relations": []}

        center_eid = center["entity_id"]

        # Query 1: neighbor entities
        entity_query = (
            "MATCH (center:Entity {entity_id: $eid})-[*1.." + str(depth) + "]-(neighbor:Entity) "
            "WHERE neighbor.entity_id <> $eid "
            "RETURN DISTINCT neighbor, labels(neighbor) AS labels"
        )
        entity_result = self._run(entity_query, {"eid": center_eid})

        # Query 2: relationships (returned as individual rows with properties)
        rel_query = (
            "MATCH (center:Entity {entity_id: $eid})-[*1.." + str(depth) + "]-(neighbor:Entity) "
            "WITH center, neighbor "
            "MATCH (center)-[r]-(neighbor) "
            "RETURN DISTINCT type(r) AS rel_type, r.relation_id AS relation_id, "
            "r.description AS description, r.year AS year, r.properties AS properties, "
            "r.source_bibtex_key AS source_bibtex_key"
        )
        # For depth > 1, also get intermediate relations
        if depth > 1:
            rel_query = (
                "MATCH path = (center:Entity {entity_id: $eid})-[*1.." + str(depth) + "]-(neighbor:Entity) "
                "WHERE neighbor.entity_id <> $eid "
                "UNWIND range(0, length(path)-1) AS i "
                "WITH relationships(path)[i] AS r "
                "RETURN DISTINCT type(r) AS rel_type, r.relation_id AS relation_id, "
                "r.description AS description, r.year AS year, r.properties AS properties, "
                "r.source_bibtex_key AS source_bibtex_key"
            )
        else:
            rel_query = (
                "MATCH (center:Entity {entity_id: $eid})-[r]-(neighbor:Entity) "
                "RETURN DISTINCT type(r) AS rel_type, r.relation_id AS relation_id, "
                "r.description AS description, r.year AS year, r.properties AS properties, "
                "r.source_bibtex_key AS source_bibtex_key"
            )

        rel_result = self._run(rel_query, {"eid": center_eid})

        # Raccogli entità
        entities_map: dict[str, dict] = {}
        for record in entity_result:
            n_dict = _node_to_dict(record["neighbor"], record["labels"])
            if n_dict and n_dict["entity_id"] not in entities_map:
                entities_map[n_dict["entity_id"]] = n_dict

        # Raccogli relazioni
        relations_list: list[dict] = []
        seen_rels: set[str] = set()
        for record in rel_result:
            rid = record.get("relation_id", "")
            if rid and rid not in seen_rels:
                seen_rels.add(rid)
                relations_list.append({
                    "type": record.get("rel_type", ""),
                    "relation_id": rid,
                    "description": record.get("description", ""),
                    "year": record.get("year"),
                    "properties": record.get("properties", "{}"),
                    "source_bibtex_key": record.get("source_bibtex_key"),
                })

        logger.info(
            f"Neighbors di '{entity_name}' (depth={depth}): "
            f"{len(entities_map)} entità, {len(relations_list)} relazioni"
        )

        return {
            "center": center,
            "entities": list(entities_map.values()),
            "relations": relations_list,
        }

    # ------------------------------------------------------------------
    # Query: shortest path
    # ------------------------------------------------------------------

    def query_path(
        self,
        entity_a: str,
        entity_b: str,
        entity_type_a: EntityType | str | None = None,
        entity_type_b: EntityType | str | None = None,
        max_depth: int = 5,
    ) -> list[dict]:
        """Trova il cammino più breve tra due entità.

        Args:
            entity_a: Nome canonico dell'entità sorgente.
            entity_b: Nome canonico dell'entità target.
            entity_type_a: Tipo opzionale per A.
            entity_type_b: Tipo opzionale per B.
            max_depth: Profondità massima.

        Returns:
            Lista dei nodi/relazioni nel cammino. Lista vuota se non esiste un cammino.
        """
        # Risolvi IDs
        a = self.get_entity_by_name(entity_a, entity_type_a)
        b = self.get_entity_by_name(entity_b, entity_type_b)

        if not a or not b:
            missing = []
            if not a:
                missing.append(entity_a)
            if not b:
                missing.append(entity_b)
            logger.warning(f"Entità non trovate: {missing}")
            return []

        # Use _run_raw to preserve Neo4j types for relationships
        self._ensure_connected()
        query = (
            f"MATCH path = shortestPath("
            f"  (a:Entity {{entity_id: $eid_a}})"
            f"  -[*1..{max_depth}]-"
            f"  (b:Entity {{entity_id: $eid_b}})"
            f") "
            f"RETURN nodes(path) AS nodes, relationships(path) AS rels"
        )

        params = {"eid_a": a["entity_id"], "eid_b": b["entity_id"]}
        with self._driver.session(database=self._database) as session:
            raw_result = session.run(query, params)
            records = list(raw_result)

        if not records:
            logger.info(f"Nessun cammino tra '{entity_a}' e '{entity_b}' (max depth={max_depth})")
            return []

        record = records[0]
        nodes = record["nodes"]
        rels = record["rels"]

        # Costruisci il percorso
        path_elements: list[dict] = []

        for i, node in enumerate(nodes):
            # Node — Neo4j Node objects support dict() and have .labels
            node_labels = list(node.labels) if hasattr(node, "labels") else []
            path_elements.append({
                "type": "entity",
                "entity_id": node.get("entity_id", ""),
                "canonical_name": node.get("canonical_name", ""),
                "entity_type": _extract_entity_type_from_labels(node_labels),
            })
            # Relazione verso il nodo successivo
            if i < len(rels):
                rel = rels[i]
                path_elements.append({
                    "type": "relation",
                    "relation_type": rel.type,
                    "relation_id": rel.get("relation_id", ""),
                    "description": rel.get("description", ""),
                    "year": rel.get("year"),
                })

        logger.info(
            f"Cammino trovato: {entity_a} → {entity_b} "
            f"({len(nodes)} nodi, {len(rels)} relazioni)"
        )

        return path_elements

    # ------------------------------------------------------------------
    # Query: temporal chain
    # ------------------------------------------------------------------

    def get_temporal_chain(
        self,
        entity_type: EntityType | str = EntityType.ATTACK,
        limit: int = 50,
    ) -> list[dict]:
        """Recupera la timeline temporale delle entità di un dato tipo.

        Ordina le entità per anno di introduzione e include le relazioni
        temporali (UPDATES, EXTENDS, INTRODUCED_IN).

        Args:
            entity_type: Tipo di entità da filtrare.
            limit: Numero massimo di entità.

        Returns:
            Lista ordinata per anno con entità e relazioni temporali.
        """
        label = entity_type.value if isinstance(entity_type, EntityType) else entity_type

        # Recupera entità con anno
        query = (
            f"MATCH (n:{label}) "
            f"WHERE n.year IS NOT NULL "
            f"OPTIONAL MATCH (n)-[r:UPDATES|EXTENDS|INTRODUCED_IN]->(m) "
            f"RETURN n, labels(n) AS labels, "
            f"collect(DISTINCT {{type: type(r), target: m.canonical_name, "
            f"target_id: m.entity_id, year: r.year}}) AS temporal_rels "
            f"ORDER BY n.year ASC, n.canonical_name ASC "
            f"LIMIT {int(limit)}"
        )

        result = self._run(query)

        timeline: list[dict] = []
        for record in result:
            node_dict = _node_to_dict(record["n"], record["labels"])
            if node_dict:
                # Filtra relazioni temporali valide (non null)
                temporal_rels = [
                    r for r in (record.get("temporal_rels") or [])
                    if r.get("type") is not None
                ]
                node_dict["temporal_relations"] = temporal_rels
                timeline.append(node_dict)

        logger.info(
            f"Timeline {label}: {len(timeline)} entità "
            f"({timeline[0]['year'] if timeline else '?'}–{timeline[-1]['year'] if timeline else '?'})"
        )

        return timeline

    # ------------------------------------------------------------------
    # Query: custom Cypher
    # ------------------------------------------------------------------

    def query_cypher(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Esegue una query Cypher personalizzata.

        Args:
            cypher: Query Cypher.
            parameters: Parametri della query.

        Returns:
            Lista di dizionari con i risultati.
        """
        return self._run(cypher, parameters)

    # ------------------------------------------------------------------
    # Graph management
    # ------------------------------------------------------------------

    def clear_graph(self) -> int:
        """Elimina tutti i nodi e le relazioni dal grafo.

        Returns:
            Numero di nodi eliminati.
        """
        result = self._run(
            "MATCH (n) DETACH DELETE n RETURN count(n) AS deleted"
        )
        deleted = result[0]["deleted"] if result else 0
        logger.info(f"Grafo pulito: {deleted} nodi eliminati.")
        return deleted

    def clear_by_paper(self, bibtex_key: str) -> dict[str, int]:
        """Elimina tutte le entità e relazioni originarie di un paper specifico.

        Utile per re-ingestion: rimuove tutto ciò che è stato estratto
        da un paper prima di re-importare.

        Args:
            bibtex_key: BibTeX key del paper.

        Returns:
            Dizionario con entities_deleted e relations_deleted.
        """
        # Prima elimina relazioni
        result_rels = self._run(
            "MATCH ()-[r]->() WHERE r.source_bibtex_key = $key "
            "DELETE r RETURN count(r) AS deleted",
            {"key": bibtex_key},
        )
        rels_deleted = result_rels[0]["deleted"] if result_rels else 0

        # Poi entità (solo quelle create da questo paper e senza altre relazioni)
        result_ents = self._run(
            "MATCH (n:Entity) WHERE n.source_bibtex_key = $key "
            "AND NOT exists { (n)--() } "
            "DELETE n RETURN count(n) AS deleted",
            {"key": bibtex_key},
        )
        ents_deleted = result_ents[0]["deleted"] if result_ents else 0

        logger.info(
            f"Pulito grafo per paper '{bibtex_key}': "
            f"{ents_deleted} entità, {rels_deleted} relazioni eliminate"
        )

        return {"entities_deleted": ents_deleted, "relations_deleted": rels_deleted}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Restituisce statistiche complete del grafo.

        Returns:
            Dizionario con: total_nodes, total_relationships,
            nodes_by_type, relationships_by_type, years_range.
        """
        stats: dict[str, Any] = {}

        # Conteggio totale nodi
        result = self._run("MATCH (n) RETURN count(n) AS c")
        stats["total_nodes"] = result[0]["c"] if result else 0

        # Conteggio totale relazioni
        result = self._run("MATCH ()-[r]->() RETURN count(r) AS c")
        stats["total_relationships"] = result[0]["c"] if result else 0

        # Nodi per tipo
        nodes_by_type: dict[str, int] = {}
        for et in EntityType:
            result = self._run(f"MATCH (n:{et.value}) RETURN count(n) AS c")
            count = result[0]["c"] if result else 0
            if count > 0:
                nodes_by_type[et.value] = count
        stats["nodes_by_type"] = nodes_by_type

        # Relazioni per tipo
        rels_by_type: dict[str, int] = {}
        for rt in RelationType:
            result = self._run(f"MATCH ()-[r:{rt.value}]->() RETURN count(r) AS c")
            count = result[0]["c"] if result else 0
            if count > 0:
                rels_by_type[rt.value] = count
        stats["relationships_by_type"] = rels_by_type

        # Range anni
        result = self._run(
            "MATCH (n:Entity) WHERE n.year IS NOT NULL "
            "RETURN min(n.year) AS min_year, max(n.year) AS max_year"
        )
        if result and result[0]["min_year"] is not None:
            stats["year_range"] = {
                "min": result[0]["min_year"],
                "max": result[0]["max_year"],
            }
        else:
            stats["year_range"] = None

        # Paper unici
        result = self._run(
            "MATCH (n:Entity) WHERE n.source_bibtex_key IS NOT NULL "
            "RETURN count(DISTINCT n.source_bibtex_key) AS c"
        )
        stats["unique_papers"] = result[0]["c"] if result else 0

        return stats

    # ------------------------------------------------------------------
    # GDS (Graph Data Science) — Community Detection support
    # ------------------------------------------------------------------

    def ensure_gds_plugin(self) -> bool:
        """Verifica che il plugin GDS (Graph Data Science) sia disponibile.

        Tenta di chiamare ``gds.version()`` via Cypher. Se il plugin non è
        installato, restituisce False senza lanciare eccezioni.

        Returns:
            True se GDS è disponibile, False altrimenti.
        """
        try:
            result = self._run("RETURN gds.version() AS version")
            version = result[0]["version"] if result else None
            if version:
                logger.info(f"Neo4j GDS disponibile: v{version}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Neo4j GDS non disponibile: {e}")
            return False

    def project_graph(
        self,
        graph_name: str = "community_graph",
        node_label: str = "Entity",
        relationship_types: list[str] | None = None,
        orientation: str = "UNDIRECTED",
    ) -> dict:
        """Crea una proiezione GDS del grafo per gli algoritmi di community detection.

        La proiezione include tutti i nodi ``Entity`` e le relazioni specificate
        (o tutte se non filtra). I pesi sulle relazioni non sono obbligatori
        ma, se presenti nella property ``weight``, vengono inclusi.

        Args:
            graph_name: Nome della proiezione GDS (default: ``community_graph``).
            node_label: Label dei nodi da proiettare (default: ``Entity``).
            relationship_types: Lista dei tipi di relazione da includere.
                Se None, include tutti i tipi relazione noti.
            orientation: Orientamento dei relationship: ``UNDIRECTED``, ``NATURAL``,
                ``REVERSE``. Default ``UNDIRECTED`` per community detection.

        Returns:
            Dizionario con metadati della proiezione:
            ``graph_name``, ``node_count``, ``relationship_count``.

        Raises:
            RuntimeError: Se la proiezione fallisce.
        """
        # Drop della proiezione esistente se presente
        self.drop_graph_projection(graph_name)

        if relationship_types is None:
            from paperhunter.graphrag.entity_extractor import RelationType
            relationship_types = [rt.value for rt in RelationType]

        # Costruisci la configurazione delle relazioni per GDS
        rel_config = {
            rt: {"orientation": orientation}
            for rt in relationship_types
        }

        # Usa gds.graph.project (GDS ≥ 2.x)
        # Fallback a gds.graph.project.cypher se la sintassi standard non funziona
        try:
            # Sintassi GDS ≥ 2.x: gds.graph.project(graphName, nodeProjection, relProjection)
            query = (
                "CALL gds.graph.project("
                "  $graph_name,"
                "  $node_label,"
                "  $rel_config"
                ") YIELD graphName, nodeCount, relationshipCount"
            )
            result = self._run(query, {
                "graph_name": graph_name,
                "node_label": node_label,
                "rel_config": rel_config,
            })

            if result:
                info = {
                    "graph_name": result[0].get("graphName", graph_name),
                    "node_count": result[0].get("nodeCount", 0),
                    "relationship_count": result[0].get("relationshipCount", 0),
                }
                logger.info(
                    f"Proiezione GDS '{graph_name}' creata: "
                    f"{info['node_count']} nodi, {info['relationship_count']} relazioni"
                )
                return info

        except Exception as e:
            logger.debug(f"gds.graph.project standard fallito: {e}")

            # Fallback: Cypher projection per versioni GDS diverse
            try:
                node_query = f"MATCH (n:{node_label}) RETURN id(n) AS id"
                rel_clauses = " OR ".join(f"type(r) = '{rt}'" for rt in relationship_types)
                rel_query = (
                    f"MATCH (s:{node_label})-[r]->(t:{node_label}) "
                    f"WHERE {rel_clauses} "
                    f"RETURN id(s) AS source, id(t) AS target, type(r) AS type"
                )

                query_cypher = (
                    "CALL gds.graph.project.cypher("
                    "  $graph_name,"
                    "  $node_query,"
                    "  $rel_query"
                    ") YIELD graphName, nodeCount, relationshipCount"
                )
                result = self._run(query_cypher, {
                    "graph_name": graph_name,
                    "node_query": node_query,
                    "rel_query": rel_query,
                })

                if result:
                    info = {
                        "graph_name": result[0].get("graphName", graph_name),
                        "node_count": result[0].get("nodeCount", 0),
                        "relationship_count": result[0].get("relationshipCount", 0),
                    }
                    logger.info(
                        f"Proiezione GDS (Cypher) '{graph_name}' creata: "
                        f"{info['node_count']} nodi, {info['relationship_count']} relazioni"
                    )
                    return info

            except Exception as e2:
                logger.error(f"Fallback Cypher projection fallito: {e2}")

        raise RuntimeError(
            f"Impossibile creare la proiezione GDS '{graph_name}'. "
            "Verificare che il plugin GDS sia installato correttamente."
        )

    def drop_graph_projection(self, graph_name: str = "community_graph") -> bool:
        """Elimina una proiezione GDS esistente (cleanup).

        Ignora silenziosamente se la proiezione non esiste.

        Args:
            graph_name: Nome della proiezione da eliminare.

        Returns:
            True se la proiezione è stata eliminata, False se non esisteva.
        """
        try:
            # Verifica se la proiezione esiste
            result = self._run(
                "CALL gds.graph.exists($graph_name) YIELD exists",
                {"graph_name": graph_name},
            )
            exists = result[0]["exists"] if result else False

            if exists:
                self._run(
                    "CALL gds.graph.drop($graph_name) YIELD graphName",
                    {"graph_name": graph_name},
                )
                logger.info(f"Proiezione GDS '{graph_name}' eliminata.")
                return True
            else:
                logger.debug(f"Proiezione GDS '{graph_name}' non esistente, skip drop.")
                return False

        except Exception as e:
            logger.debug(f"Drop proiezione GDS '{graph_name}' fallito (probabilmente non esiste): {e}")
            return False

    def run_community_detection(
        self,
        graph_name: str = "community_graph",
        algorithm: str = "louvain",
        write_property: str = "community_id",
    ) -> dict:
        """Esegue un algoritmo di community detection sulla proiezione GDS.

        Supporta Louvain e Leiden. Scrive il ``community_id`` come proprietà
        su ogni nodo del grafo Neo4j.

        Args:
            graph_name: Nome della proiezione GDS.
            algorithm: ``"louvain"`` o ``"leiden"``. Default ``"louvain"``.
            write_property: Nome della proprietà in cui scrivere il community ID.

        Returns:
            Dizionario con: ``community_count``, ``modularity``, ``node_properties_written``.

        Raises:
            ValueError: Se l'algoritmo non è supportato.
            RuntimeError: Se l'esecuzione fallisce.
        """
        algorithm = algorithm.lower()
        if algorithm not in ("louvain", "leiden"):
            raise ValueError(
                f"Algoritmo non supportato: '{algorithm}'. Usa 'louvain' o 'leiden'."
            )

        try:
            query = (
                f"CALL gds.{algorithm}.write("
                f"  $graph_name,"
                f"  {{writeProperty: $write_property}}"
                f") YIELD communityCount, modularity, nodePropertiesWritten"
            )
            result = self._run(query, {
                "graph_name": graph_name,
                "write_property": write_property,
            })

            if result:
                info = {
                    "community_count": result[0].get("communityCount", 0),
                    "modularity": result[0].get("modularity", 0.0),
                    "node_properties_written": result[0].get("nodePropertiesWritten", 0),
                    "algorithm": algorithm,
                    "write_property": write_property,
                }
                logger.info(
                    f"Community detection ({algorithm}): "
                    f"{info['community_count']} comunità trovate, "
                    f"modularity={info['modularity']:.4f}, "
                    f"{info['node_properties_written']} proprietà scritte"
                )
                return info
            else:
                raise RuntimeError(f"Nessun risultato da gds.{algorithm}.write")

        except Exception as e:
            raise RuntimeError(
                f"Community detection ({algorithm}) fallita: {e}"
            ) from e

    def get_nodes_by_community(
        self,
        community_id: int,
        community_property: str = "community_id",
    ) -> list[dict]:
        """Recupera tutti i nodi appartenenti a una specifica comunità.

        Args:
            community_id: ID della comunità.
            community_property: Nome della proprietà community (default: ``community_id``).

        Returns:
            Lista di dizionari con le proprietà dei nodi.
        """
        query = (
            f"MATCH (n:Entity) "
            f"WHERE n.{community_property} = $cid "
            f"RETURN n, labels(n) AS labels "
            f"ORDER BY n.canonical_name"
        )
        result = self._run(query, {"cid": community_id})
        nodes = []
        for record in result:
            node = _node_to_dict(record["n"], record["labels"])
            if node:
                nodes.append(node)
        return nodes

    def get_edges_within_community(
        self,
        community_id: int,
        community_property: str = "community_id",
    ) -> list[dict]:
        """Recupera tutte le relazioni interne a una comunità.

        Args:
            community_id: ID della comunità.
            community_property: Nome della proprietà community.

        Returns:
            Lista di dizionari con le proprietà delle relazioni.
        """
        query = (
            f"MATCH (s:Entity)-[r]->(t:Entity) "
            f"WHERE s.{community_property} = $cid AND t.{community_property} = $cid "
            f"RETURN s.canonical_name AS source, s.entity_id AS source_id, "
            f"type(r) AS rel_type, r.description AS description, "
            f"r.year AS year, r.source_bibtex_key AS source_bibtex_key, "
            f"t.canonical_name AS target, t.entity_id AS target_id"
        )
        result = self._run(query, {"cid": community_id})
        edges = []
        for record in result:
            edges.append({
                "source": record.get("source", ""),
                "source_id": record.get("source_id", ""),
                "rel_type": record.get("rel_type", ""),
                "description": record.get("description", ""),
                "year": record.get("year"),
                "source_bibtex_key": record.get("source_bibtex_key"),
                "target": record.get("target", ""),
                "target_id": record.get("target_id", ""),
            })
        return edges

    def get_all_community_ids(
        self,
        community_property: str = "community_id",
        min_size: int = 1,
    ) -> list[dict]:
        """Recupera tutti i community ID distinti con il conteggio dei nodi.

        Args:
            community_property: Nome della proprietà community.
            min_size: Dimensione minima della comunità (default: 1).

        Returns:
            Lista di ``{"community_id": int, "size": int}``, ordinata per size decrescente.
        """
        query = (
            f"MATCH (n:Entity) "
            f"WHERE n.{community_property} IS NOT NULL "
            f"RETURN n.{community_property} AS community_id, count(n) AS size "
            f"ORDER BY size DESC"
        )
        result = self._run(query)
        communities = [
            {"community_id": r["community_id"], "size": r["size"]}
            for r in result
            if r["size"] >= min_size
        ]
        return communities


# ============================================================================
# Helper functions
# ============================================================================

def _node_to_dict(node: Any, labels: list[str] | None = None) -> dict | None:
    """Converte un nodo Neo4j in dizionario Python.

    Args:
        node: Node result dal driver Neo4j.
        labels: Lista di label del nodo.

    Returns:
        Dizionario con proprietà del nodo, incluse labels e tipo estratto.
    """
    if node is None:
        return None

    # Proprietà base
    d = dict(node)

    # Deserializza JSON fields
    if "aliases" in d and isinstance(d["aliases"], str):
        try:
            d["aliases"] = json.loads(d["aliases"])
        except (json.JSONDecodeError, TypeError):
            d["aliases"] = []

    if "properties" in d and isinstance(d["properties"], str):
        try:
            d["properties"] = json.loads(d["properties"])
        except (json.JSONDecodeError, TypeError):
            d["properties"] = {}

    # Aggiungi labels e tipo
    if labels:
        d["labels"] = labels
        d["entity_type"] = _extract_entity_type_from_labels(labels)

    return d


def _extract_entity_type_from_labels(labels: list) -> str | None:
    """Estrae il tipo EntityType dalle labels Neo4j.

    Le labels sono es. ["Entity", "Attack"]. Il tipo è la label
    che corrisponde a un valore di EntityType.
    """
    entity_type_values = {et.value for et in EntityType}
    for label in labels:
        lbl = str(label)
        if lbl in entity_type_values:
            return lbl
    return None


# ============================================================================
# Standalone convenience functions
# ============================================================================

def test_neo4j_connection() -> bool:
    """Testa la connessione a Neo4j.

    Returns:
        True se la connessione è riuscita.
    """
    try:
        with Neo4jManager() as mgr:
            stats = mgr.get_stats()
            print_success(
                f"Connessione Neo4j OK — "
                f"{stats['total_nodes']} nodi, {stats['total_relationships']} relazioni"
            )
            return True
    except Neo4jNotEnabledError:
        print_warning("Neo4j è disabilitato in config.yaml")
        return False
    except Neo4jConnectionError as e:
        print_error(f"Connessione Neo4j fallita: {e}")
        return False
    except Exception as e:
        print_error(f"Errore test Neo4j: {e}")
        return False
