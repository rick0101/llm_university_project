"""
GraphRAG PropertyGraph Builder — Costruisce e indicizza un PropertyGraph con LlamaIndex.

Converte le entità e relazioni estratte dall'``entity_extractor`` in un
PropertyGraph indicizzato da LlamaIndex, consentendo retrieval graph-based
(LLMSynonymRetriever, traversal, ecc.) integrato con il ``HybridRetriever``.

Usa ``SimplePropertyGraphStore`` come backend locale (JSON su disco),
senza richiedere un database esterno. Opzionalmente, può essere collegato
a Neo4j tramite il ``Neo4jManager`` per la persistenza.

Workflow tipico::

    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    builder = PropertyGraphBuilder()

    # Costruzione da L1 summaries (estrazione + build in un passo)
    builder.build_from_summaries(l1_summaries)

    # Oppure da entità/relazioni pre-estratte
    builder.build_from_entities(entities, relations)

    # Aggiornamento incrementale
    builder.update_graph(new_l1_summaries)

    # Query graph-based
    context = builder.query_graph("evolution of prompt injection defenses")

    # Persistenza
    builder.save()
    builder.load()

Note:
    Il modulo è opzionale ma fortemente consigliato. Arricchisce il
    retrieval con contesto relazionale (es. "GCG extends AutoDAN",
    "SmoothLLM defends against GCG") che il vector search da solo
    non cattura adeguatamente.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.graphrag.entity_extractor import (
    Entity,
    EntityType,
    ExtractionResult,
    Relation as PaperhunterRelation,
    RelationType,
    extract_batch,
    merge_entities,
    merge_relations,
    save_extraction_results,
    _extract_with_result,
)
from paperhunter.models import L1Summary
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
# Constants
# ============================================================================

_DEFAULT_PERSIST_DIR = "./data/property_graph"
_DEFAULT_GRAPH_FILE = "graph_store.json"
_DEFAULT_METADATA_FILE = "graph_metadata.json"
_DEFAULT_EXTRACTION_CHECKPOINT = "extraction_checkpoint.json"


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class GraphQueryResult:
    """Risultato di una query al PropertyGraph.

    Attributes:
        context_text: Testo contestuale assemblato per RAG.
        entities_found: Entità trovate nel grafo.
        relations_found: Relazioni trovate.
        paths: Cammini trovati (liste di nodi/relazioni alternati).
        query: Query originale.
        retrieval_method: Metodo usato (synonym_llm, traversal, direct).
        elapsed_seconds: Tempo di esecuzione.
    """
    context_text: str = ""
    entities_found: list[dict] = field(default_factory=list)
    relations_found: list[dict] = field(default_factory=list)
    paths: list[list[dict]] = field(default_factory=list)
    query: str = ""
    retrieval_method: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serializza il risultato."""
        return {
            "context_text": self.context_text,
            "entities_found": self.entities_found,
            "relations_found": self.relations_found,
            "paths_count": len(self.paths),
            "query": self.query,
            "retrieval_method": self.retrieval_method,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class BuildReport:
    """Report della costruzione del grafo.

    Attributes:
        entities_added: Entità inserite nel grafo.
        relations_added: Relazioni inserite.
        entities_extracted: Entità estratte (prima del merge).
        relations_extracted: Relazioni estratte (prima del merge).
        entities: Lista entità finale (post-merge).
        relations: Lista relazioni finale (post-merge).
        extraction_results: Risultati per singolo paper.
        elapsed_seconds: Tempo totale.
        errors: Errori incontrati.
    """
    entities_added: int = 0
    relations_added: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    entities: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    extraction_results: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities_added": self.entities_added,
            "relations_added": self.relations_added,
            "entities_extracted": self.entities_extracted,
            "relations_extracted": self.relations_extracted,
            "extraction_results_count": len(self.extraction_results),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "errors": self.errors,
        }


# ============================================================================
# PropertyGraph Builder
# ============================================================================

class PropertyGraphBuilder:
    """Costruisce e gestisce un LlamaIndex PropertyGraph per il knowledge graph.

    Converte entità e relazioni dall'``entity_extractor`` nel formato
    LlamaIndex (``EntityNode``, ``Relation``) e li popola in un
    ``SimplePropertyGraphStore`` locale con persistenza JSON.

    Supporta:
    - Costruzione da L1 summaries (estrazione automatica entità/relazioni)
    - Costruzione da entità/relazioni pre-estratte
    - Aggiornamento incrementale
    - Query graph-based con LLM synonym expansion
    - Query diretta per traversal del grafo
    - Persistenza su disco (load/save)

    Args:
        persist_dir: Directory per la persistenza del grafo.
            Default: ``./data/property_graph``.
        llm: Istanza LLM per query (expansion sinonimi).
            Default: Gemini Flash.
        embed_model: Modello embedding per vector context retrieval.
            Default: None (disabilitato — usa solo synonym retriever).
        auto_load: Se True, carica il grafo da disco se disponibile.

    Uso::

        builder = PropertyGraphBuilder()

        # Build completo da L1
        report = builder.build_from_summaries(l1_summaries)

        # Query
        result = builder.query_graph("prompt injection defense timeline")
        print(result.context_text)
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        llm=None,
        embed_model=None,
        auto_load: bool = True,
    ):
        self._persist_dir = Path(persist_dir or _DEFAULT_PERSIST_DIR)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._llm = llm
        self._embed_model = embed_model
        self._graph_store = None
        self._index = None
        self._metadata: dict[str, Any] = {
            "entity_count": 0,
            "relation_count": 0,
            "paper_ids": [],
            "last_updated": None,
            "build_count": 0,
        }

        # Lazy init — crea o carica
        if auto_load and self._graph_persist_path.exists():
            self.load()
        else:
            self._init_empty_graph()

    # ------------------------------------------------------------------
    # Properties & paths
    # ------------------------------------------------------------------

    @property
    def _graph_persist_path(self) -> Path:
        return self._persist_dir / _DEFAULT_GRAPH_FILE

    @property
    def _metadata_path(self) -> Path:
        return self._persist_dir / _DEFAULT_METADATA_FILE

    @property
    def entity_count(self) -> int:
        """Numero di entità nel grafo."""
        return self._metadata.get("entity_count", 0)

    @property
    def relation_count(self) -> int:
        """Numero di relazioni nel grafo."""
        return self._metadata.get("relation_count", 0)

    @property
    def paper_ids(self) -> list[str]:
        """Paper IDs già presenti nel grafo."""
        return self._metadata.get("paper_ids", [])

    # ------------------------------------------------------------------
    # Lazy-init helpers
    # ------------------------------------------------------------------

    def _get_llm(self):
        """Lazy init del LLM Flash per query."""
        if self._llm is None:
            from paperhunter.utils.llm_factory import get_flash_llm
            self._llm = get_flash_llm(timeout_seconds=90, max_tokens=4096)
        return self._llm

    def _get_embed_model(self):
        """Lazy init del modello embedding (opzionale)."""
        if self._embed_model is None:
            try:
                from paperhunter.utils.llm_factory import get_embedding_model
                self._embed_model = get_embedding_model()
            except Exception as e:
                logger.debug(f"Embedding model non disponibile: {e}")
                return None
        return self._embed_model

    def _init_empty_graph(self) -> None:
        """Inizializza un grafo vuoto."""
        from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
        self._graph_store = SimplePropertyGraphStore()
        self._index = None
        logger.debug("PropertyGraph vuoto inizializzato.")

    # ------------------------------------------------------------------
    # Extraction checkpoint helpers
    # ------------------------------------------------------------------

    @property
    def _extraction_checkpoint_path(self) -> Path:
        return self._persist_dir / _DEFAULT_EXTRACTION_CHECKPOINT

    def _load_extraction_checkpoint(self) -> dict[str, Any]:
        """Carica il checkpoint di estrazione se esiste.

        Returns:
            Dict con 'processed_paper_ids', 'entities', 'relations'.
            Se il checkpoint non esiste, restituisce struttura vuota.
        """
        path = self._extraction_checkpoint_path
        if not path.exists():
            return {"processed_paper_ids": [], "entities": [], "relations": []}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Ricostruisci Entity e Relation da dicts
            entities = [Entity(**e) for e in data.get("entities", [])]
            relations = [PaperhunterRelation(**r) for r in data.get("relations", [])]
            processed = data.get("processed_paper_ids", [])
            logger.info(
                f"Checkpoint estrazione caricato: {len(processed)} paper già estratti, "
                f"{len(entities)} entità, {len(relations)} relazioni"
            )
            return {
                "processed_paper_ids": processed,
                "entities": entities,
                "relations": relations,
            }
        except Exception as e:
            logger.warning(f"Errore caricamento checkpoint estrazione: {e}. Ripartenza da zero.")
            return {"processed_paper_ids": [], "entities": [], "relations": []}

    def _save_extraction_checkpoint(
        self,
        processed_paper_ids: list[str],
        entities: list[Entity],
        relations: list[PaperhunterRelation],
    ) -> None:
        """Salva il checkpoint di estrazione su disco."""
        path = self._extraction_checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "saved_at": datetime.now().isoformat(),
            "processed_paper_ids": processed_paper_ids,
            "entities": [e.to_dict() for e in entities],
            "relations": [r.to_dict() for r in relations],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug(f"Checkpoint estrazione salvato: {len(processed_paper_ids)} paper")

    def _clear_extraction_checkpoint(self) -> None:
        """Rimuove il checkpoint di estrazione (dopo completamento)."""
        path = self._extraction_checkpoint_path
        if path.exists():
            path.unlink()
            logger.debug("Checkpoint estrazione rimosso (build completato).")

    # ------------------------------------------------------------------
    # Build: da L1 summaries
    # ------------------------------------------------------------------

    def build_from_summaries(
        self,
        l1_summaries: list[L1Summary],
        *,
        clear_before: bool = False,
        skip_existing: bool = True,
        merge: bool = True,
        show_progress: bool = True,
    ) -> BuildReport:
        """Costruisce il PropertyGraph a partire da L1 summaries.

        Workflow:
        1. Filtra summaries già nel grafo (se skip_existing=True)
        2. Carica checkpoint estrazione (se esiste) per riprendere
        3. Estrai entità/relazioni incrementalmente con checkpoint
        4. Converti in formato LlamaIndex e popola il PropertyGraph
        5. Salva su disco e rimuovi checkpoint

        Args:
            l1_summaries: Lista di summary L1 da processare.
            clear_before: Se True, ricrea il grafo da zero.
            skip_existing: Se True, ignora paper già nel grafo.
            merge: Se True, merge entità/relazioni duplicate.
            show_progress: Se True, mostra progress bar.

        Returns:
            BuildReport con statistiche dell'operazione.
        """
        from paperhunter.utils.llm_factory import get_flash_llm

        t0 = time.monotonic()
        report = BuildReport()

        if not l1_summaries:
            logger.warning("Nessun L1 summary fornito per build_from_summaries.")
            return report

        if clear_before:
            self._init_empty_graph()
            self._metadata = {
                "entity_count": 0,
                "relation_count": 0,
                "paper_ids": [],
                "last_updated": None,
                "build_count": 0,
            }
            self._clear_extraction_checkpoint()
            logger.info("Grafo svuotato (clear_before=True).")

        # Filtra paper già presenti nel grafo
        if skip_existing:
            existing_ids = set(self.paper_ids)
            new_summaries = [
                s for s in l1_summaries
                if s.paper_id not in existing_ids
            ]
            skipped = len(l1_summaries) - len(new_summaries)
            if skipped > 0:
                logger.info(f"Skippati {skipped} paper già nel grafo.")
            l1_summaries = new_summaries

        if not l1_summaries:
            logger.info("Nessun nuovo L1 da processare (tutti già nel grafo).")
            report.elapsed_seconds = time.monotonic() - t0
            return report

        # Step 1: Carica checkpoint estrazione (se esiste, riprende da dove si era fermato)
        checkpoint = self._load_extraction_checkpoint()
        checkpoint_paper_ids = set(checkpoint["processed_paper_ids"])
        all_entities: list[Entity] = list(checkpoint["entities"])
        all_relations: list[PaperhunterRelation] = list(checkpoint["relations"])
        all_results: list[ExtractionResult] = []

        # Filtra paper già estratti nel checkpoint
        remaining_summaries = [
            s for s in l1_summaries
            if s.paper_id not in checkpoint_paper_ids
        ]

        if checkpoint_paper_ids:
            logger.info(
                f"Ripresa da checkpoint: {len(checkpoint_paper_ids)} paper già estratti, "
                f"{len(remaining_summaries)} rimanenti"
            )

        # Step 2: Estrazione incrementale con checkpoint
        if remaining_summaries:
            logger.info(f"Estraendo entità/relazioni da {len(remaining_summaries)} L1 summaries...")
            llm = get_flash_llm(timeout_seconds=120, max_tokens=65536)
            processed_ids = list(checkpoint["processed_paper_ids"])

            iterable = remaining_summaries
            if show_progress:
                iterable = progress_bar(
                    remaining_summaries,
                    description="Extracting entities & relations",
                    total=len(remaining_summaries),
                )

            for summary in iterable:
                try:
                    result = _extract_with_result(summary, llm=llm)
                    all_results.append(result)

                    if result.error:
                        logger.error(f"Estrazione fallita per {summary.paper_id}: {result.error}")
                    else:
                        all_entities.extend(result.entities)
                        all_relations.extend(result.relations)

                except Exception as exc:
                    logger.error(f"Errore estrazione per {summary.paper_id}: {exc}")
                    all_results.append(ExtractionResult(
                        paper_id=summary.paper_id,
                        bibtex_key=summary.bibtex_key,
                        error=str(exc),
                        extracted_at=datetime.now(),
                    ))

                # Salva checkpoint dopo ogni paper
                processed_ids.append(summary.paper_id)
                self._save_extraction_checkpoint(
                    processed_ids, all_entities, all_relations,
                )

        # Merge se richiesto
        if merge:
            all_entities = merge_entities(all_entities)
            all_relations = merge_relations(all_relations)

        report.entities_extracted = len(all_entities)
        report.relations_extracted = len(all_relations)
        report.entities = all_entities
        report.relations = all_relations
        report.extraction_results = [r.to_dict() for r in all_results]

        # Step 3: Popola il PropertyGraph
        logger.info(f"Popolando PropertyGraph con {len(all_entities)} entità e {len(all_relations)} relazioni...")
        added_e, added_r = self._populate_graph(all_entities, all_relations, show_progress=show_progress)
        report.entities_added = added_e
        report.relations_added = added_r

        # Step 4: Aggiorna metadata
        new_paper_ids = [s.paper_id for s in l1_summaries]
        self._metadata["paper_ids"] = list(set(self.paper_ids + new_paper_ids))
        self._metadata["entity_count"] = self._count_entities()
        self._metadata["relation_count"] = self._count_relations()
        self._metadata["last_updated"] = datetime.now().isoformat()
        self._metadata["build_count"] = self._metadata.get("build_count", 0) + 1

        # Step 5: Salva e rimuovi checkpoint
        self.save()
        self._clear_extraction_checkpoint()

        report.elapsed_seconds = time.monotonic() - t0
        logger.info(
            f"Build completato in {report.elapsed_seconds:.1f}s: "
            f"+{added_e} entità, +{added_r} relazioni "
            f"(totale: {self.entity_count} entità, {self.relation_count} relazioni)"
        )

        if show_progress:
            print_table("PropertyGraph Build Report", report.to_dict())

        return report

    # ------------------------------------------------------------------
    # Build: da entità/relazioni pre-estratte
    # ------------------------------------------------------------------

    def build_from_entities(
        self,
        entities: list[Entity],
        relations: list[PaperhunterRelation],
        *,
        clear_before: bool = False,
        paper_ids: list[str] | None = None,
        show_progress: bool = True,
    ) -> BuildReport:
        """Costruisce il PropertyGraph da entità e relazioni pre-estratte.

        Utile quando le entità sono già state estratte (es. da
        ``extract_batch()``) e non serve rieseguire l'estrazione.

        Args:
            entities: Lista di entità già estratte/merged.
            relations: Lista di relazioni già estratte/merged.
            clear_before: Se True, ricrea il grafo da zero.
            paper_ids: Paper IDs da registrare nei metadata.
            show_progress: Se True, mostra progress bar.

        Returns:
            BuildReport con statistiche.
        """
        t0 = time.monotonic()
        report = BuildReport()

        if clear_before:
            self._init_empty_graph()
            self._metadata = {
                "entity_count": 0,
                "relation_count": 0,
                "paper_ids": [],
                "last_updated": None,
                "build_count": 0,
            }

        report.entities_extracted = len(entities)
        report.relations_extracted = len(relations)

        added_e, added_r = self._populate_graph(entities, relations, show_progress=show_progress)
        report.entities_added = added_e
        report.relations_added = added_r

        # Aggiorna metadata
        if paper_ids:
            self._metadata["paper_ids"] = list(set(self.paper_ids + paper_ids))
        self._metadata["entity_count"] = self._count_entities()
        self._metadata["relation_count"] = self._count_relations()
        self._metadata["last_updated"] = datetime.now().isoformat()
        self._metadata["build_count"] = self._metadata.get("build_count", 0) + 1

        self.save()

        report.elapsed_seconds = time.monotonic() - t0
        logger.info(
            f"Build da entità completato in {report.elapsed_seconds:.1f}s: "
            f"+{added_e} entità, +{added_r} relazioni"
        )
        return report

    # ------------------------------------------------------------------
    # Update incrementale
    # ------------------------------------------------------------------

    def update_graph(
        self,
        new_l1_summaries: list[L1Summary],
        *,
        merge: bool = True,
        show_progress: bool = True,
    ) -> BuildReport:
        """Aggiornamento incrementale del grafo con nuovi L1 summaries.

        Equivale a ``build_from_summaries`` con ``skip_existing=True``
        e ``clear_before=False``. Processa solo paper non ancora nel grafo.

        Args:
            new_l1_summaries: Nuovi L1 summaries da processare.
            merge: Se True, merge entità/relazioni duplicate.
            show_progress: Se True, mostra progress bar.

        Returns:
            BuildReport con statistiche dell'aggiornamento.
        """
        return self.build_from_summaries(
            new_l1_summaries,
            clear_before=False,
            skip_existing=True,
            merge=merge,
            show_progress=show_progress,
        )

    # ------------------------------------------------------------------
    # Remove paper
    # ------------------------------------------------------------------

    def remove_paper(self, paper_id: str) -> int:
        """Rimuove tutte le entità e relazioni di un paper dal grafo.

        Args:
            paper_id: ID del paper da rimuovere.

        Returns:
            Numero di nodi rimossi.
        """
        if self._graph_store is None:
            return 0

        # Trova nodi associati al paper
        nodes_to_remove = []
        try:
            all_nodes = self._graph_store.get(properties=None)
            for node in all_nodes:
                props = getattr(node, "properties", {})
                if props.get("source_paper_id") == paper_id:
                    nodes_to_remove.append(node)
        except Exception as e:
            logger.warning(f"Errore durante ricerca nodi per paper {paper_id}: {e}")
            return 0

        if not nodes_to_remove:
            logger.debug(f"Nessun nodo trovato per paper_id={paper_id}")
            return 0

        # Rimuovi per nome entità
        entity_names = []
        for node in nodes_to_remove:
            name = getattr(node, "name", None)
            if name:
                entity_names.append(name)

        if entity_names:
            try:
                self._graph_store.delete(entity_names=entity_names)
                logger.info(f"Rimossi {len(entity_names)} nodi per paper {paper_id}")
            except Exception as e:
                logger.error(f"Errore rimozione nodi per paper {paper_id}: {e}")
                return 0

        # Aggiorna metadata
        if paper_id in self._metadata.get("paper_ids", []):
            self._metadata["paper_ids"].remove(paper_id)
        self._metadata["entity_count"] = self._count_entities()
        self._metadata["relation_count"] = self._count_relations()
        self.save()

        return len(entity_names)

    # ------------------------------------------------------------------
    # Query: graph-based retrieval
    # ------------------------------------------------------------------

    def query_graph(
        self,
        question: str,
        *,
        max_keywords: int = 10,
        path_depth: int = 2,
        limit: int = 30,
        include_text: bool = True,
    ) -> GraphQueryResult:
        """Query il PropertyGraph per contesto relazionale.

        Usa ``LLMSynonymRetriever`` per espandere la query con sinonimi
        e trovare entità/relazioni rilevanti nel grafo.

        Args:
            question: Domanda in linguaggio naturale.
            max_keywords: Massimo numero di keyword da estrarre.
            path_depth: Profondità di traversal nel grafo.
            limit: Massimo numero di triplet da ritornare.
            include_text: Se True, include proprietà testuali dei nodi.

        Returns:
            GraphQueryResult con contesto testuale e strutturato.
        """
        t0 = time.monotonic()
        result = GraphQueryResult(query=question)

        if self._graph_store is None or self.entity_count == 0:
            logger.warning("PropertyGraph vuoto o non inizializzato, query saltata.")
            result.elapsed_seconds = time.monotonic() - t0
            return result

        try:
            result = self._query_via_synonym_retriever(
                question,
                max_keywords=max_keywords,
                path_depth=path_depth,
                limit=limit,
                include_text=include_text,
            )
        except Exception as e:
            logger.warning(f"LLMSynonymRetriever fallito: {e}. Tentativo con traversal diretto...")
            try:
                result = self._query_via_direct_traversal(question, limit=limit)
            except Exception as e2:
                logger.error(f"Anche il traversal diretto fallito: {e2}")
                result = GraphQueryResult(query=question)

        result.elapsed_seconds = time.monotonic() - t0
        logger.info(
            f"Graph query completata in {result.elapsed_seconds:.1f}s: "
            f"{len(result.entities_found)} entità, "
            f"{len(result.relations_found)} relazioni, "
            f"{len(result.context_text)} chars di contesto"
        )
        return result

    def query_entity_neighborhood(
        self,
        entity_name: str,
        *,
        depth: int = 1,
        limit: int = 20,
    ) -> GraphQueryResult:
        """Query il vicinato di una specifica entità nel grafo.

        Args:
            entity_name: Nome dell'entità (canonico o alias).
            depth: Profondità di traversal.
            limit: Massimo numero di risultati.

        Returns:
            GraphQueryResult con il sottografo locale.
        """
        t0 = time.monotonic()
        result = GraphQueryResult(query=f"neighborhood of {entity_name}")

        if self._graph_store is None:
            result.elapsed_seconds = time.monotonic() - t0
            return result

        try:
            from llama_index.core.graph_stores.types import EntityNode

            # Cerca il nodo nel grafo
            matched_nodes = self._graph_store.get(properties=None)
            target_node = None
            for node in matched_nodes:
                if isinstance(node, EntityNode):
                    node_name = getattr(node, "name", "")
                    aliases = node.properties.get("aliases", [])
                    name_lower = entity_name.lower()
                    if (
                        node_name.lower() == name_lower
                        or name_lower in [a.lower() for a in aliases]
                    ):
                        target_node = node
                        break

            if target_node is None:
                logger.info(f"Entità '{entity_name}' non trovata nel grafo.")
                result.elapsed_seconds = time.monotonic() - t0
                return result

            # Get rel_map per il vicinato
            rel_map = self._graph_store.get_rel_map(
                graph_nodes=[target_node],
                depth=depth,
                limit=limit,
            )

            # Assembla risultato.
            # get_rel_map() restituisce una lista di triplet:
            #   [(EntityNode, Relation, EntityNode), ...]
            entities = []
            relations = []
            seen_entity_ids: set[str] = set()

            for item in rel_map:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    src_node, rel_obj, tgt_node = item
                    rel_label = getattr(rel_obj, "label", str(rel_obj))
                    src_name = getattr(src_node, "name", str(src_node))
                    tgt_name = getattr(tgt_node, "name", str(tgt_node))
                    relations.append({
                        "source": src_name,
                        "type": rel_label,
                        "target": tgt_name,
                    })
                    for n in (src_node, tgt_node):
                        nid = getattr(n, "id", None) or getattr(n, "name", "")
                        if nid and nid not in seen_entity_ids:
                            seen_entity_ids.add(nid)
                            entities.append({
                                "name": getattr(n, "name", ""),
                                "label": getattr(n, "label", ""),
                                "properties": getattr(n, "properties", {}),
                            })

            # Aggiungi il nodo centrale in testa se non già presente
            central_id = getattr(target_node, "id", None) or entity_name
            if central_id not in seen_entity_ids:
                entities.insert(0, {
                    "name": getattr(target_node, "name", entity_name),
                    "label": getattr(target_node, "label", "entity"),
                    "properties": target_node.properties,
                })

            result.entities_found = entities
            result.relations_found = relations
            result.retrieval_method = "neighborhood_traversal"
            result.context_text = self._format_context(entities, relations)

        except Exception as e:
            logger.error(f"Errore query neighborhood per '{entity_name}': {e}")

        result.elapsed_seconds = time.monotonic() - t0
        return result

    def get_temporal_chain(
        self,
        entity_type: str = "Attack",
        *,
        relation_types: list[str] | None = None,
    ) -> GraphQueryResult:
        """Recupera catene temporali dal grafo (UPDATES, EXTENDS, INTRODUCED_IN).

        Utile per ricostruire l'evoluzione di attacchi o difese nel tempo.

        Args:
            entity_type: Tipo di entità da cercare (es. "Attack", "Defense").
            relation_types: Tipi di relazione da includere.
                Default: UPDATES, EXTENDS, INTRODUCED_IN.

        Returns:
            GraphQueryResult con la catena temporale.
        """
        t0 = time.monotonic()
        result = GraphQueryResult(query=f"temporal chain for {entity_type}")

        if relation_types is None:
            relation_types = ["UPDATES", "EXTENDS", "INTRODUCED_IN"]

        if self._graph_store is None:
            result.elapsed_seconds = time.monotonic() - t0
            return result

        try:
            # Get all triplets with matching relation types
            all_entities = []
            all_relations = []

            for rel_type in relation_types:
                triplets = self._graph_store.get_triplets(
                    entity_names=None,
                    relation_names=[rel_type],
                )
                for triplet in triplets:
                    # Triplet: (source_node, relation, target_node)
                    if len(triplet) >= 3:
                        source_node, relation, target_node = triplet[0], triplet[1], triplet[2]

                        source_label = getattr(source_node, "label", "")
                        target_label = getattr(target_node, "label", "")

                        # Filtra per entity_type
                        if entity_type.lower() not in (source_label.lower(), target_label.lower()):
                            continue

                        source_name = getattr(source_node, "name", "?")
                        target_name = getattr(target_node, "name", "?")
                        rel_label = getattr(relation, "label", rel_type)

                        all_entities.append({
                            "name": source_name,
                            "label": source_label,
                            "year": source_node.properties.get("year"),
                        })
                        all_entities.append({
                            "name": target_name,
                            "label": target_label,
                            "year": target_node.properties.get("year"),
                        })
                        all_relations.append({
                            "source": source_name,
                            "type": rel_label,
                            "target": target_name,
                            "year": getattr(relation, "properties", {}).get("year"),
                        })

            # Deduplica entità per nome
            seen_names = set()
            deduped_entities = []
            for e in all_entities:
                if e["name"] not in seen_names:
                    seen_names.add(e["name"])
                    deduped_entities.append(e)

            # Ordina per anno
            deduped_entities.sort(key=lambda x: x.get("year") or 9999)

            result.entities_found = deduped_entities
            result.relations_found = all_relations
            result.retrieval_method = "temporal_chain"
            result.context_text = self._format_temporal_context(deduped_entities, all_relations)

        except Exception as e:
            logger.error(f"Errore temporal chain per {entity_type}: {e}")

        result.elapsed_seconds = time.monotonic() - t0
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        """Salva il PropertyGraph e i metadata su disco.

        Returns:
            Path della directory di persistenza.
        """
        if self._graph_store is None:
            logger.warning("Nessun grafo da salvare.")
            return self._persist_dir

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # Salva graph store
        try:
            self._graph_store.persist(persist_path=str(self._graph_persist_path))
            logger.debug(f"Graph store salvato in: {self._graph_persist_path}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio graph store: {e}")

        # Salva metadata
        try:
            self._metadata_path.write_text(
                json.dumps(self._metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(f"Metadata salvati in: {self._metadata_path}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio metadata: {e}")

        return self._persist_dir

    def load(self) -> bool:
        """Carica il PropertyGraph da disco.

        Returns:
            True se il caricamento è riuscito.
        """
        from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore

        if not self._graph_persist_path.exists():
            logger.info(f"Nessun grafo salvato in {self._graph_persist_path}")
            return False

        try:
            self._graph_store = SimplePropertyGraphStore.from_persist_path(
                str(self._graph_persist_path)
            )
            logger.info(f"Graph store caricato da: {self._graph_persist_path}")
        except Exception as e:
            logger.error(f"Errore caricamento graph store: {e}")
            self._init_empty_graph()
            return False

        # Carica metadata
        if self._metadata_path.exists():
            try:
                self._metadata = json.loads(
                    self._metadata_path.read_text(encoding="utf-8")
                )
                logger.debug(f"Metadata caricati: {self._metadata}")
            except Exception as e:
                logger.warning(f"Errore caricamento metadata: {e}")
                # Ricostruisci metadata dal grafo
                self._metadata["entity_count"] = self._count_entities()
                self._metadata["relation_count"] = self._count_relations()

        logger.info(
            f"PropertyGraph caricato: {self.entity_count} entità, "
            f"{self.relation_count} relazioni, "
            f"{len(self.paper_ids)} paper"
        )
        return True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Statistiche del PropertyGraph.

        Returns:
            Dizionario con statistiche aggregate.
        """
        stats = {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "paper_count": len(self.paper_ids),
            "persist_dir": str(self._persist_dir),
            "last_updated": self._metadata.get("last_updated"),
            "build_count": self._metadata.get("build_count", 0),
        }

        if self._graph_store is not None:
            try:
                # Conta entità per tipo
                type_counts: dict[str, int] = {}
                rel_type_counts: dict[str, int] = {}

                all_nodes = self._graph_store.get(properties=None)
                for node in all_nodes:
                    label = getattr(node, "label", "unknown")
                    type_counts[label] = type_counts.get(label, 0) + 1

                # Conta relazioni per tipo via get_triplets
                for rel_type in [rt.value for rt in RelationType]:
                    try:
                        triplets = self._graph_store.get_triplets(
                            entity_names=None,
                            relation_names=[rel_type],
                        )
                        if triplets:
                            rel_type_counts[rel_type] = len(triplets)
                    except Exception:
                        pass

                stats["entities_by_type"] = type_counts
                stats["relations_by_type"] = {k: v for k, v in rel_type_counts.items() if v > 0}

            except Exception as e:
                logger.debug(f"Errore nel conteggio dettagliato: {e}")

        return stats

    # ------------------------------------------------------------------
    # Internal: Graph population
    # ------------------------------------------------------------------

    def _populate_graph(
        self,
        entities: list[Entity],
        relations: list[PaperhunterRelation],
        *,
        show_progress: bool = True,
    ) -> tuple[int, int]:
        """Popola il PropertyGraph con entità e relazioni.

        Converte dal formato paperhunter al formato LlamaIndex e
        li inserisce nel graph store.

        Returns:
            Tuple (entità_aggiunte, relazioni_aggiunte).
        """
        from llama_index.core.graph_stores.types import EntityNode
        from llama_index.core.graph_stores.types import Relation as LIRelation

        if self._graph_store is None:
            self._init_empty_graph()

        # -- Converti e inserisci entità --
        li_nodes: list[EntityNode] = []
        entity_id_map: dict[str, str] = {}  # paperhunter_id → LlamaIndex node id

        entity_iter = entities
        if show_progress and len(entities) > 5:
            entity_iter = progress_bar(entities, "Adding entities to graph")

        for entity in entity_iter:
            li_node = self._entity_to_li_node(entity)
            li_nodes.append(li_node)
            entity_id_map[entity.id] = li_node.id

        if li_nodes:
            try:
                self._graph_store.upsert_nodes(li_nodes)
                logger.debug(f"Upserted {len(li_nodes)} EntityNode nel graph store.")
            except Exception as e:
                logger.error(f"Errore upsert nodes: {e}")
                return 0, 0

        # -- Converti e inserisci relazioni --
        li_relations: list[LIRelation] = []

        rel_iter = relations
        if show_progress and len(relations) > 5:
            rel_iter = progress_bar(relations, "Adding relations to graph")

        for relation in rel_iter:
            li_rel = self._relation_to_li_relation(relation, entity_id_map)
            if li_rel is not None:
                li_relations.append(li_rel)

        if li_relations:
            try:
                self._graph_store.upsert_relations(li_relations)
                logger.debug(f"Upserted {len(li_relations)} Relation nel graph store.")
            except Exception as e:
                logger.error(f"Errore upsert relations: {e}")
                return len(li_nodes), 0

        return len(li_nodes), len(li_relations)

    def _entity_to_li_node(self, entity: Entity) -> "EntityNode":
        """Converte un'entità paperhunter in un LlamaIndex EntityNode.

        Il mapping è:
        - ``name`` ← ``entity.canonical_name``  (usato anche come ID dal nodo)
        - ``label`` ← ``entity.type.value`` (es. "Attack", "Defense")
        - ``properties`` ← dizionario con tutti i metadati dell'entità

        Nota: EntityNode.id è *sempre* uguale a EntityNode.name.
        Non è possibile impostarlo separatamente.
        """
        from llama_index.core.graph_stores.types import EntityNode

        props = {
            "entity_type": entity.type.value if isinstance(entity.type, EntityType) else entity.type,
            "aliases": entity.aliases if entity.aliases else [],
            "description": entity.description or "",
            "source_paper_id": entity.source_paper_id or "",
            "source_bibtex_key": entity.source_bibtex_key or "",
            "paperhunter_id": entity.id,  # conserva l'ID originale nei properties
        }
        if entity.year is not None:
            props["year"] = entity.year
        if entity.properties:
            props.update(entity.properties)

        return EntityNode(
            name=entity.canonical_name,
            label=entity.type.value if isinstance(entity.type, EntityType) else entity.type,
            properties=props,
        )

    def _relation_to_li_relation(
        self,
        relation: PaperhunterRelation,
        entity_id_map: dict[str, str],
    ) -> Optional["LIRelation"]:
        """Converte una relazione paperhunter in una LlamaIndex Relation.

        Il mapping è:
        - ``source_id`` ← ID dell'entità sorgente
        - ``target_id`` ← ID dell'entità target
        - ``label`` ← ``relation.type.value`` (es. "DEFENDS_AGAINST")
        - ``properties`` ← dizionario con metadati della relazione

        Returns:
            LlamaIndex Relation, o None se source/target non trovati.
        """
        from llama_index.core.graph_stores.types import Relation as LIRelation
        from paperhunter.graphrag.entity_extractor import _make_entity_id

        # Trova gli ID LlamaIndex delle entità.
        # EntityNode.id == EntityNode.name == canonical_name, quindi
        # usiamo i canonical_name come fallback (non gli hash paperhunter).
        source_ph_id = _make_entity_id(relation.source_type, relation.source)
        target_ph_id = _make_entity_id(relation.target_type, relation.target)

        source_li_id = entity_id_map.get(source_ph_id, relation.source)
        target_li_id = entity_id_map.get(target_ph_id, relation.target)

        props = {
            "description": relation.description or "",
            "source_paper_id": relation.source_paper_id or "",
            "source_bibtex_key": relation.source_bibtex_key or "",
        }
        if relation.year is not None:
            props["year"] = relation.year
        if relation.properties:
            props.update(relation.properties)

        rel_label = relation.type.value if isinstance(relation.type, RelationType) else relation.type

        return LIRelation(
            source_id=source_li_id,
            target_id=target_li_id,
            label=rel_label,
            properties=props,
        )

    # ------------------------------------------------------------------
    # Internal: Query implementations
    # ------------------------------------------------------------------

    def _query_via_synonym_retriever(
        self,
        question: str,
        *,
        max_keywords: int = 10,
        path_depth: int = 2,
        limit: int = 30,
        include_text: bool = True,
    ) -> GraphQueryResult:
        """Query usando LLMSynonymRetriever (LLM espande la query in entità/sinonimi)."""
        from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import (
            LLMSynonymRetriever,
        )

        retriever = LLMSynonymRetriever(
            graph_store=self._graph_store,
            llm=self._get_llm(),
            max_keywords=max_keywords,
            path_depth=path_depth,
            limit=limit,
            include_text=include_text,
        )

        # Usa retrieve() per ottenere NodeWithScore
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=question)
        nodes = retriever.retrieve(query_bundle)

        # Assembla risultato
        result = GraphQueryResult(query=question, retrieval_method="llm_synonym")

        context_parts = []
        seen_entities: set[str] = set()
        seen_relations: set[str] = set()

        for node_with_score in nodes:
            node = node_with_score.node
            text = node.get_content()
            if text:
                context_parts.append(text)

            # Estrai info strutturata dal testo del nodo
            metadata = node.metadata or {}
            if "triplet_source_id" in metadata:
                source = metadata.get("triplet_source_id", "")
                rel_type = metadata.get("triplet_rel", "")
                target = metadata.get("triplet_target_id", "")

                if source and source not in seen_entities:
                    seen_entities.add(source)
                    result.entities_found.append({"name": source})
                if target and target not in seen_entities:
                    seen_entities.add(target)
                    result.entities_found.append({"name": target})

                rel_key = f"{source}-{rel_type}-{target}"
                if rel_key not in seen_relations and rel_type:
                    seen_relations.add(rel_key)
                    result.relations_found.append({
                        "source": source,
                        "type": rel_type,
                        "target": target,
                    })

        result.context_text = "\n\n".join(context_parts) if context_parts else ""
        return result

    def _query_via_direct_traversal(
        self,
        question: str,
        *,
        limit: int = 30,
    ) -> GraphQueryResult:
        """Fallback: query diretta al graph store senza LLM.

        Estrae keyword dalla domanda e cerca entità che matchano.
        """
        result = GraphQueryResult(query=question, retrieval_method="direct_traversal")

        # Estrarre keyword dalla domanda (semplice tokenizzazione)
        keywords = _extract_keywords_simple(question)

        if not keywords:
            return result

        # Cerca entità con nomi che matchano le keyword
        try:
            all_nodes = self._graph_store.get(properties=None)
        except Exception:
            return result

        from llama_index.core.graph_stores.types import EntityNode

        matching_nodes = []
        for node in all_nodes:
            if not isinstance(node, EntityNode):
                continue
            node_name = getattr(node, "name", "").lower()
            aliases = [a.lower() for a in node.properties.get("aliases", [])]
            description = node.properties.get("description", "").lower()

            for kw in keywords:
                if (
                    kw in node_name
                    or any(kw in alias for alias in aliases)
                    or kw in description
                ):
                    matching_nodes.append(node)
                    break

        if not matching_nodes:
            return result

        # Get relazioni per i nodi trovati
        try:
            rel_map = self._graph_store.get_rel_map(
                graph_nodes=matching_nodes[:limit],
                depth=1,
                limit=limit,
            )
        except Exception as e:
            logger.debug(f"get_rel_map fallito: {e}")
            rel_map = {}

        entities = []
        relations = []

        for node in matching_nodes[:limit]:
            entities.append({
                "name": getattr(node, "name", ""),
                "label": getattr(node, "label", ""),
                "properties": node.properties,
            })

        # rel_map è una lista di triplet: [(EntityNode, Relation, EntityNode), ...]
        for item in rel_map:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                src_node, rel_obj, tgt_node = item
                relations.append({
                    "source": getattr(src_node, "name", str(src_node)),
                    "type": getattr(rel_obj, "label", str(rel_obj)),
                    "target": getattr(tgt_node, "name", str(tgt_node)),
                })

        result.entities_found = entities
        result.relations_found = relations
        result.context_text = self._format_context(entities, relations)
        return result

    # ------------------------------------------------------------------
    # Internal: counting
    # ------------------------------------------------------------------

    def _count_entities(self) -> int:
        """Conta le entità nel graph store."""
        if self._graph_store is None:
            return 0
        try:
            nodes = self._graph_store.get(properties=None)
            from llama_index.core.graph_stores.types import EntityNode
            return sum(1 for n in nodes if isinstance(n, EntityNode))
        except Exception:
            return 0

    def _count_relations(self) -> int:
        """Conta le relazioni nel graph store."""
        if self._graph_store is None:
            return 0
        try:
            count = 0
            for rel_type in [rt.value for rt in RelationType]:
                triplets = self._graph_store.get_triplets(
                    entity_names=None,
                    relation_names=[rel_type],
                )
                count += len(triplets)
            return count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Internal: formatting
    # ------------------------------------------------------------------

    def _format_context(
        self,
        entities: list[dict],
        relations: list[dict],
    ) -> str:
        """Formatta entità e relazioni in un testo per RAG context."""
        parts = []

        if entities:
            parts.append("### Relevant Entities\n")
            for e in entities:
                name = e.get("name", "?")
                label = e.get("label", "")
                props = e.get("properties", {})
                year = props.get("year", "")
                desc = props.get("description", "")

                line = f"- **{name}**"
                if label:
                    line += f" [{label}]"
                if year:
                    line += f" ({year})"
                if desc:
                    line += f": {desc}"
                parts.append(line)

        if relations:
            parts.append("\n### Relationships\n")
            for r in relations:
                source = r.get("source", "?")
                rel_type = r.get("type", "?")
                target = r.get("target", "?")
                year = r.get("year", "")

                line = f"- {source} --[{rel_type}]--> {target}"
                if year:
                    line += f" ({year})"
                parts.append(line)

        return "\n".join(parts)

    def _format_temporal_context(
        self,
        entities: list[dict],
        relations: list[dict],
    ) -> str:
        """Formatta una catena temporale in testo per RAG."""
        parts = ["### Temporal Evolution\n"]

        # Raggruppa per anno
        by_year: dict[int, list[dict]] = {}
        for e in entities:
            year = e.get("year")
            if year:
                by_year.setdefault(year, []).append(e)

        for year in sorted(by_year.keys()):
            entities_in_year = by_year[year]
            names = [e["name"] for e in entities_in_year]
            parts.append(f"**{year}:** {', '.join(names)}")

        if relations:
            parts.append("\n### Temporal Relationships\n")
            for r in relations:
                source = r.get("source", "?")
                rel_type = r.get("type", "?")
                target = r.get("target", "?")
                year = r.get("year", "")
                line = f"- {source} --[{rel_type}]--> {target}"
                if year:
                    line += f" ({year})"
                parts.append(line)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PropertyGraphBuilder("
            f"entities={self.entity_count}, "
            f"relations={self.relation_count}, "
            f"papers={len(self.paper_ids)}, "
            f"persist_dir={str(self._persist_dir)!r})"
        )


# ============================================================================
# Standalone helpers
# ============================================================================

def _extract_keywords_simple(question: str) -> list[str]:
    """Estrai keyword semplici da una domanda (senza LLM).

    Rimuove stopwords e restituisce token significativi.
    """
    import re

    # Stopwords comuni
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "can", "could", "may", "might",
        "must", "of", "in", "to", "for", "with", "on", "at", "from",
        "by", "about", "as", "into", "through", "during", "before",
        "after", "and", "but", "or", "nor", "not", "so", "yet",
        "both", "either", "neither", "each", "every", "all", "any",
        "few", "more", "most", "other", "some", "such", "no",
        "this", "that", "these", "those", "what", "which", "who",
        "how", "when", "where", "why", "their", "its",
        "evolution", "overview", "history", "describe", "explain",
    }

    tokens = re.findall(r"[a-zA-Z0-9_-]+", question.lower())
    keywords = [t for t in tokens if t not in stopwords and len(t) >= 3]
    return keywords


def build_property_graph(
    l1_summaries: list[L1Summary],
    *,
    persist_dir: str | None = None,
    clear_before: bool = False,
    show_progress: bool = True,
) -> PropertyGraphBuilder:
    """Convenience function per costruire un PropertyGraph da L1 summaries.

    Args:
        l1_summaries: L1 summaries da processare.
        persist_dir: Directory di persistenza.
        clear_before: Se True, ricrea il grafo da zero.
        show_progress: Se True, mostra progress bar.

    Returns:
        PropertyGraphBuilder pronto per query.
    """
    builder = PropertyGraphBuilder(persist_dir=persist_dir, auto_load=not clear_before)
    builder.build_from_summaries(
        l1_summaries,
        clear_before=clear_before,
        show_progress=show_progress,
    )
    return builder


def query_property_graph(
    question: str,
    *,
    persist_dir: str | None = None,
) -> GraphQueryResult:
    """Convenience function per query al PropertyGraph.

    Args:
        question: Domanda in linguaggio naturale.
        persist_dir: Directory del grafo persistito.

    Returns:
        GraphQueryResult con contesto e struttura.
    """
    builder = PropertyGraphBuilder(persist_dir=persist_dir, auto_load=True)
    return builder.query_graph(question)
