"""
GraphRAG Community Summarizer — Generazione summary testuali per comunità.

Trasforma il sotto-grafo di ciascuna comunità (rilevata da ``community_detector.py``)
in un testo narrativo comprensibile dall'agente scrittore. I summary vengono
salvati in Qdrant (collection ``graph_communities``) per il retrieval.

Pipeline::

    Grafo Neo4j (entità + relazioni)
        ↓ community_detector.py (Louvain/Leiden)
    Comunità con sotto-grafi
        ↓ community_summarizer.py (questo modulo)
    Community Summaries testuali → Qdrant "graph_communities"
        ↓ retrieval / writer agent
    Writer Agent usa i community summaries nel contesto

Ogni community summary è un testo narrativo di 200-500 parole che descrive:
- Il macro-tema della comunità
- Le entità chiave e le loro relazioni
- Il pattern di connessioni tra le entità

Uso::

    from paperhunter.graphrag.community_summarizer import CommunitySummarizer
    from paperhunter.graphrag.community_detector import CommunityDetector
    from paperhunter.graphrag.neo4j_manager import Neo4jManager
    from paperhunter.indexing.qdrant_store import QdrantStore

    with Neo4jManager() as mgr:
        detector = CommunityDetector(mgr)
        communities = detector.detect_communities(algorithm="louvain", min_size=3)

        store = QdrantStore()
        summarizer = CommunitySummarizer(detector, store)

        # Genera summary per tutte le comunità
        results = summarizer.summarize_all_communities(communities)
        # → [{"community_id": 0, "summary": "...", "node_count": 15, "key_entities": [...]}, ...]

        # Salva in Qdrant
        summarizer.save_to_qdrant(results)

    # Oppure: singola comunità
    text = summarizer.summarize_community(community_id=0, subgraph=communities[0])
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.models import ChunkType, PaperChunk
from paperhunter.utils.llm_factory import (
    get_pro_llm,
    llm_call_with_retry,
)
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_success,
    print_table,
    print_warning,
    progress_bar,
)
from paperhunter.utils.rate_limiter import get_gemini_limiter

logger = get_logger(__name__)


# ============================================================================
# Prompt template per Community Summary
# ============================================================================

_COMMUNITY_SUMMARY_PROMPT_TEMPLATE = """\
You are an expert academic researcher specializing in LLM security. \
You are given a **community sub-graph** extracted from a knowledge graph \
of LLM security research entities and their relationships.

This community contains {node_count} entities and {edge_count} relationships.

## Entities in this Community

{entities_text}

## Relationships in this Community

{relationships_text}

## Key Entities (most connected)

{key_entities_text}

---

## Instructions

Write a **coherent narrative summary** (200-500 words) that describes this community \
of research concepts. Your summary should:

1. **Identify the macro-theme**: What overarching topic or research area does this \
   community represent? What ties these entities together?
2. **Describe key entities**: Explain the most important entities and why they are \
   central to this community.
3. **Explain relationships**: Describe how the entities are connected and what these \
   connections reveal about the field.
4. **Highlight patterns**: Identify any interesting structural patterns, such as \
   clusters of related attacks/defenses, temporal evolution, or methodological connections.
5. **Provide research context**: Place this community in the broader context of \
   LLM security research.

IMPORTANT RULES:
- Write in a formal academic style suitable for a survey paper.
- Mention entity names in **bold** (e.g., **GCG**, **AutoDAN**).
- Focus on the narrative — do NOT produce a list of entities.
- The text should be self-contained and understandable without seeing the graph.
- Length: 200-500 words, as a coherent paragraph or short set of paragraphs.

Write ONLY the narrative text. Do NOT wrap it in JSON or markdown fences.
"""


# ============================================================================
# Formatting helpers
# ============================================================================

def _format_entities_for_prompt(nodes: list[dict]) -> str:
    """Formatta la lista dei nodi di una comunità per il prompt LLM.

    Args:
        nodes: Lista di dizionari nodo dal sotto-grafo.

    Returns:
        Testo formattato con le entità.
    """
    if not nodes:
        return "(nessuna entità)"

    lines: list[str] = []
    for node in nodes:
        name = node.get("canonical_name", node.get("entity_id", "Unknown"))
        entity_type = node.get("entity_type", "Unknown")
        description = node.get("description", "")
        paper_count = node.get("paper_count", 0)

        line = f"- **{name}** (type: {entity_type})"
        if description:
            line += f" — {description[:150]}"
        if paper_count:
            line += f" [mentioned in {paper_count} papers]"
        lines.append(line)

    return "\n".join(lines)


def _format_relationships_for_prompt(edges: list[dict]) -> str:
    """Formatta la lista delle relazioni di una comunità per il prompt LLM.

    Args:
        edges: Lista di dizionari relazione dal sotto-grafo.

    Returns:
        Testo formattato con le relazioni.
    """
    if not edges:
        return "(nessuna relazione)"

    lines: list[str] = []
    seen: set[str] = set()

    for edge in edges:
        source = edge.get("source", "?")
        target = edge.get("target", "?")
        rel_type = edge.get("rel_type", edge.get("type", "RELATED_TO"))
        weight = edge.get("weight", None)

        # Dedup: evita duplicati bidirezionali
        edge_key = tuple(sorted([source, target])) + (rel_type,)
        if edge_key in seen:
            continue
        seen.add(edge_key)

        line = f"- {source} —[{rel_type}]→ {target}"
        if weight and weight > 1:
            line += f" (weight: {weight})"
        lines.append(line)

    # Limita il numero di relazioni nel prompt (max 50 per evitare prompt troppo grandi)
    if len(lines) > 50:
        lines = lines[:50]
        lines.append(f"  ... e altre {len(edges) - 50} relazioni")

    return "\n".join(lines)


def _format_key_entities_for_prompt(key_entities: list[str]) -> str:
    """Formatta la lista delle entità chiave per il prompt LLM.

    Args:
        key_entities: Lista dei nomi delle entità più connesse.

    Returns:
        Testo formattato.
    """
    if not key_entities:
        return "(nessuna entità chiave identificata)"
    return ", ".join(f"**{name}**" for name in key_entities)


# ============================================================================
# Community Summarizer
# ============================================================================

class CommunitySummarizer:
    """Genera summary testuali narrativi per le comunità del grafo.

    Prende le comunità identificate dal ``CommunityDetector`` e, per ciascuna,
    genera un testo narrativo di 200-500 parole tramite LLM (Gemini Pro).
    I summary sono poi salvabili in Qdrant per il retrieval.

    Args:
        community_detector: Istanza di ``CommunityDetector`` connessa.
        qdrant_store: Istanza di ``QdrantStore`` per il salvataggio.

    Attributes:
        detector: Riferimento al ``CommunityDetector``.
        store: Riferimento al ``QdrantStore``.

    Esempio::

        summarizer = CommunitySummarizer(detector, store)
        results = summarizer.summarize_all_communities(communities)
        summarizer.save_to_qdrant(results)
    """

    def __init__(
        self,
        community_detector: Any,
        qdrant_store: Any,
    ):
        self._detector = community_detector
        self._store = qdrant_store

    @property
    def detector(self) -> Any:
        """Riferimento al CommunityDetector."""
        return self._detector

    @property
    def store(self) -> Any:
        """Riferimento al QdrantStore."""
        return self._store

    # ------------------------------------------------------------------
    # Core: singola comunità
    # ------------------------------------------------------------------

    def summarize_community(
        self,
        community_id: int,
        subgraph: dict,
        *,
        force_model: str | None = None,
    ) -> str:
        """Genera un summary narrativo per una singola comunità.

        Formatta il sotto-grafo come lista di entità + relazioni in testo
        strutturato, poi chiede al LLM di produrre un testo narrativo
        di 200-500 parole con menzione delle entità chiave.

        Args:
            community_id: ID della comunità.
            subgraph: Dizionario sotto-grafo con chiavi ``nodes``, ``edges``,
                ``size``, ``key_entities`` (output di
                ``CommunityDetector.get_community_subgraph()``).
            force_model: Override tier modello (``'flash'`` o ``'pro'``).
                Default: Gemini Pro.

        Returns:
            Testo narrativo del summary (200-500 parole).

        Raises:
            ValueError: Se il sotto-grafo è vuoto.
            Exception: Se la chiamata LLM fallisce dopo tutti i retry.
        """
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        key_entities = subgraph.get("key_entities", [])

        if not nodes:
            raise ValueError(
                f"Community {community_id}: sotto-grafo vuoto (nessun nodo)"
            )

        start_time = time.time()

        # Formatta le componenti per il prompt
        entities_text = _format_entities_for_prompt(nodes)
        relationships_text = _format_relationships_for_prompt(edges)
        key_entities_text = _format_key_entities_for_prompt(key_entities)

        # Costruisci il prompt
        prompt = _COMMUNITY_SUMMARY_PROMPT_TEMPLATE.format(
            node_count=len(nodes),
            edge_count=len(edges),
            entities_text=entities_text,
            relationships_text=relationships_text,
            key_entities_text=key_entities_text,
        )

        # Seleziona il modello
        tier = force_model or "pro"
        if tier == "pro":
            llm = get_pro_llm()
        else:
            from paperhunter.utils.llm_factory import get_flash_llm
            llm = get_flash_llm()

        # Rate limiting
        limiter = get_gemini_limiter()
        limiter.acquire()

        # Chiamata LLM con retry
        logger.info(
            f"Generazione community summary per community {community_id} "
            f"({len(nodes)} nodi, {len(edges)} relazioni)"
        )

        try:
            raw_response = llm_call_with_retry(
                llm,
                prompt,
                max_retries=settings.summarization.pro_max_retries,
                base_delay=settings.summarization.pro_base_delay,
            )
        except Exception as e:
            logger.error(
                f"Errore LLM per community {community_id}: {e}"
            )
            raise

        # Pulisci la risposta (rimuovi eventuali fence markdown)
        summary_text = _clean_llm_response(raw_response)

        elapsed = time.time() - start_time

        # Validazione qualità
        _validate_community_summary(
            summary_text,
            community_id=community_id,
            key_entities=key_entities,
        )

        logger.info(
            f"Community summary generato per community {community_id} "
            f"({len(summary_text)} chars, {_word_count(summary_text)} parole, "
            f"{elapsed:.1f}s, model={tier})"
        )

        return summary_text

    # ------------------------------------------------------------------
    # Batch: tutte le comunità
    # ------------------------------------------------------------------

    def summarize_all_communities(
        self,
        communities: list[dict],
        *,
        force_model: str | None = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        """Genera summary per tutte le comunità con progress bar e rate limiting.

        Args:
            communities: Lista di dizionari comunità (output di
                ``CommunityDetector.detect_communities()``).
            force_model: Override tier modello.
            skip_existing: Se True, salta comunità già presenti in Qdrant.

        Returns:
            Lista di dizionari per ogni comunità processata::

                [
                    {
                        "community_id": int,
                        "summary": str,
                        "node_count": int,
                        "key_entities": list[str],
                    },
                    ...
                ]
        """
        if not communities:
            logger.warning("summarize_all_communities: nessuna comunità fornita")
            return []

        total = len(communities)
        results: list[dict] = []
        errors: dict[int, str] = {}
        skipped = 0

        logger.info(
            f"Generazione community summaries: {total} comunità"
        )

        for community in progress_bar(communities, description="Community Summaries"):
            cid = community.get("community_id", -1)

            # Skip se già presente
            if skip_existing and _community_exists_in_store(cid, self._store):
                logger.debug(f"Community {cid} già presente in Qdrant, skip")
                skipped += 1
                continue

            try:
                summary_text = self.summarize_community(
                    community_id=cid,
                    subgraph=community,
                    force_model=force_model,
                )
                results.append({
                    "community_id": cid,
                    "summary": summary_text,
                    "node_count": community.get("size", len(community.get("nodes", []))),
                    "key_entities": community.get("key_entities", []),
                })
            except Exception as e:
                logger.error(
                    f"Errore summary per community {cid}: {e}"
                )
                errors[cid] = str(e)

        # Report finale
        _print_batch_report(
            total=total,
            generated=len(results),
            skipped=skipped,
            errors=len(errors),
            error_details=errors,
        )

        return results

    # ------------------------------------------------------------------
    # Salvataggio in Qdrant
    # ------------------------------------------------------------------

    def save_to_qdrant(
        self,
        summaries: list[dict],
        *,
        batch_size: int = 10,
    ) -> int:
        """Salva i community summary nella collection Qdrant ``graph_communities``.

        Ogni summary viene convertito in un ``PaperChunk`` e indicizzato
        nel vector store con metadata specifici per la comunità.

        Args:
            summaries: Lista di dizionari con ``community_id``, ``summary``,
                ``node_count``, ``key_entities``.
            batch_size: Numero di summary per batch di upsert.

        Returns:
            Numero di summary salvati con successo.
        """
        if not summaries:
            logger.warning("save_to_qdrant: nessun summary da salvare")
            return 0

        collection = settings.qdrant.collection_graph_communities

        # Assicurati che la collection esista
        _ensure_community_collection(self._store, collection)

        total_saved = 0
        start_time = time.time()

        chunks: list[PaperChunk] = []
        for entry in summaries:
            cid = entry["community_id"]
            summary_text = entry["summary"]
            node_count = entry.get("node_count", 0)
            key_entities = entry.get("key_entities", [])

            # Chunk ID deterministico
            chunk_id = _make_community_chunk_id(cid)

            # Testo per embedding: combina summary + metadata
            embed_text = _community_summary_to_embed_text(
                community_id=cid,
                summary=summary_text,
                key_entities=key_entities,
                node_count=node_count,
            )

            # Crea PaperChunk
            chunk = PaperChunk(
                chunk_id=chunk_id,
                text=embed_text,
                chunk_type=ChunkType.L1_SUMMARY,  # Tipo più prossimo
                paper_id=f"community_{cid}",
                bibtex_key=f"community_{cid}",
                section_name="community_summary",
                category="graph_community",
                year=None,
                chunk_index=0,
            )
            chunks.append(chunk)

        if not chunks:
            return 0

        # Salva con embedding automatico
        try:
            total_saved = self._store.add_chunks(
                chunks,
                collection_name=collection,
                batch_size=batch_size,
                show_progress=True,
            )
        except Exception as e:
            logger.error(f"Errore salvataggio community summaries in Qdrant: {e}")
            return 0

        elapsed = time.time() - start_time
        logger.info(
            f"Salvati {total_saved}/{len(summaries)} community summaries "
            f"in '{collection}' ({elapsed:.1f}s)"
        )

        return total_saved

    def __repr__(self) -> str:
        return (
            f"CommunitySummarizer("
            f"detector={self._detector!r}, "
            f"store={type(self._store).__name__})"
        )


# ============================================================================
# Qdrant helpers
# ============================================================================

def _ensure_community_collection(store: Any, collection_name: str) -> None:
    """Assicura che la collection per i community summary esista in Qdrant.

    Args:
        store: Istanza QdrantStore.
        collection_name: Nome della collection.
    """
    from qdrant_client import models as qdrant_models

    if not store.client.collection_exists(collection_name):
        store.client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=store._embedding_dim,
                distance=store._distance,
            ),
        )
        logger.info(f"Collection '{collection_name}' creata per community summaries")


def _make_community_chunk_id(community_id: int) -> str:
    """Genera un chunk_id deterministico per un community summary.

    Args:
        community_id: ID della comunità.

    Returns:
        Stringa hash di 16 caratteri.
    """
    raw = f"community_summary::{community_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _community_summary_to_embed_text(
    community_id: int,
    summary: str,
    key_entities: list[str],
    node_count: int,
) -> str:
    """Converte un community summary in testo per l'embedding e il retrieval.

    Crea un testo strutturato che combina il summary narrativo con
    metadata utili per il retrieval, adatto alla vettorizzazione.

    Args:
        community_id: ID della comunità.
        summary: Testo narrativo del summary.
        key_entities: Entità chiave della comunità.
        node_count: Numero di nodi nella comunità.

    Returns:
        Testo combinato per embedding.
    """
    parts = [
        f"Graph Community {community_id} — Knowledge Graph Community Summary",
        f"Entities: {node_count}",
        f"Key Entities: {', '.join(key_entities)}",
        "",
        summary,
    ]
    return "\n".join(parts)


def _community_exists_in_store(
    community_id: int,
    store: Any,
) -> bool:
    """Verifica se un community summary esiste già nel vector store.

    Args:
        community_id: ID della comunità.
        store: QdrantStore.

    Returns:
        True se il community summary è già presente.
    """
    try:
        collection = settings.qdrant.collection_graph_communities
        paper_id = f"community_{community_id}"
        return store.has_paper(paper_id, collection_name=collection)
    except Exception:
        return False


# ============================================================================
# Response cleaning & validation
# ============================================================================

def _clean_llm_response(raw_response: str) -> str:
    """Pulisce la risposta LLM da eventuali artefatti.

    Rimuove code fence markdown, spazi extra, e garantisce
    che il risultato sia un testo puro.

    Args:
        raw_response: Testo grezzo dalla risposta LLM.

    Returns:
        Testo pulito.
    """
    text = raw_response.strip()

    # Rimuovi code fence markdown (```...```)
    fence_match = re.match(r"^```(?:text|markdown)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Rimuovi eventuale prefisso "Summary:" o simili
    for prefix in ("Summary:", "Community Summary:", "Narrative:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text


def _validate_community_summary(
    summary: str,
    community_id: int,
    key_entities: list[str],
) -> None:
    """Validazione qualità di un community summary.

    Controlla lunghezza e presenza di entità chiave.

    Args:
        summary: Testo del summary.
        community_id: ID della comunità (per logging).
        key_entities: Entità chiave attese nel testo.
    """
    warnings: list[str] = []

    word_count = _word_count(summary)

    if word_count < 50:
        warnings.append(f"summary troppo corto ({word_count} parole, min atteso: ~200)")

    if word_count > 800:
        warnings.append(f"summary troppo lungo ({word_count} parole, max atteso: ~500)")

    # Verifica che almeno una delle key_entities sia menzionata
    if key_entities:
        mentioned = 0
        summary_lower = summary.lower()
        for entity in key_entities:
            if entity.lower() in summary_lower:
                mentioned += 1
        if mentioned == 0:
            warnings.append(
                f"nessuna delle {len(key_entities)} key_entities menzionata nel summary"
            )
        elif mentioned < len(key_entities) // 2:
            warnings.append(
                f"solo {mentioned}/{len(key_entities)} key_entities menzionate nel summary"
            )

    if warnings:
        for w in warnings:
            print_warning(f"Community {community_id}: {w}")
        logger.warning(
            f"Community {community_id}: {len(warnings)} warning di qualità"
        )
    else:
        logger.debug(
            f"Community {community_id}: validazione OK "
            f"({word_count} parole)"
        )


def _word_count(text: str) -> int:
    """Conta le parole in un testo.

    Args:
        text: Testo da analizzare.

    Returns:
        Numero di parole.
    """
    return len(text.split())


# ============================================================================
# Batch report
# ============================================================================

def _print_batch_report(
    total: int,
    generated: int,
    skipped: int,
    errors: int,
    error_details: dict[int, str],
) -> None:
    """Stampa un report riassuntivo dopo il batch processing.

    Args:
        total: Numero totale di comunità.
        generated: Comunità processate con successo.
        skipped: Comunità saltate (già esistenti).
        errors: Numero di errori.
        error_details: Dizionario {community_id: errore}.
    """
    print_success(
        f"Community summaries batch completato: "
        f"{generated}/{total} generati, {skipped} skipped, {errors} errori"
    )

    if error_details:
        print_error(f"Errori ({len(error_details)} comunità):")
        for cid, err in list(error_details.items())[:5]:
            print_error(f"  Community {cid}: {err[:100]}")
        if len(error_details) > 5:
            print_warning(f"  ... e altri {len(error_details) - 5} errori")

    # Tabella riepilogativa
    data = {
        "Comunità totali": str(total),
        "Summaries generati": str(generated),
        "Skipped (esistenti)": str(skipped),
        "Errori": str(errors),
    }
    print_table(
        title="Community Summaries Batch Report",
        data=data,
    )


# ============================================================================
# Loading utilities
# ============================================================================

def load_community_summaries(
    store: Any = None,
) -> list[dict]:
    """Carica tutti i community summary dal Qdrant store.

    Esegue uno scroll sulla collection ``graph_communities`` e
    restituisce i summary con metadata.

    Args:
        store: Istanza ``QdrantStore``. Se None, ne crea una nuova.

    Returns:
        Lista di dizionari::

            [
                {
                    "community_id": int,
                    "summary": str,
                    "node_count": int (se disponibile),
                    "key_entities": list[str],
                },
                ...
            ]
    """
    if store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    collection = settings.qdrant.collection_graph_communities

    # Verifica che la collection esista
    if not store.client.collection_exists(collection):
        logger.info(
            f"Collection '{collection}' non esiste: "
            f"nessun community summary disponibile"
        )
        return []

    # Scroll tutti i punti
    all_points: list = []
    offset = None

    while True:
        try:
            points, next_offset = store.client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.error(f"Errore scroll community summaries: {e}")
            break

        all_points.extend(points)

        if next_offset is None:
            break
        offset = next_offset

    if not all_points:
        logger.info("Nessun community summary trovato in Qdrant")
        return []

    # Ricostruisci i dizionari dal testo
    results: list[dict] = []
    for point in all_points:
        payload = point.payload or {}
        text = payload.get("text", "")
        paper_id = payload.get("paper_id", "")

        entry = _reconstruct_community_from_payload(paper_id, text)
        if entry is not None:
            results.append(entry)

    # Ordina per community_id
    results.sort(key=lambda x: x.get("community_id", 0))

    logger.info(f"Caricati {len(results)} community summaries da Qdrant")
    return results


def _reconstruct_community_from_payload(
    paper_id: str,
    text: str,
) -> dict | None:
    """Ricostruisce un dizionario community summary dal payload Qdrant.

    Il testo ha il formato prodotto da ``_community_summary_to_embed_text``:

        Graph Community {id} — Knowledge Graph Community Summary
        Entities: {count}
        Key Entities: ent1, ent2, ...

        {narrative summary}

    Args:
        paper_id: ID del paper (formato: ``community_{id}``).
        text: Testo del chunk.

    Returns:
        Dizionario con ``community_id``, ``summary``, ``node_count``,
        ``key_entities``, oppure None se impossibile ricostruire.
    """
    try:
        # Estrai community_id da paper_id
        cid_match = re.match(r"community_(\d+)", paper_id)
        if not cid_match:
            return None
        community_id = int(cid_match.group(1))

        # Parse del testo strutturato
        lines = text.split("\n")
        node_count = 0
        key_entities: list[str] = []
        summary_start = 0
        found_separator = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("Entities:"):
                try:
                    node_count = int(stripped.replace("Entities:", "").strip())
                except ValueError:
                    pass
            elif stripped.startswith("Key Entities:"):
                ents_str = stripped.replace("Key Entities:", "").strip()
                key_entities = [e.strip() for e in ents_str.split(",") if e.strip()]
            elif stripped == "" and i > 0 and not found_separator:
                # La prima riga vuota dopo le metadata segna l'inizio del summary
                summary_start = i + 1
                found_separator = True

        # Il summary è tutto ciò che viene dopo la riga vuota
        summary = "\n".join(lines[summary_start:]).strip()

        return {
            "community_id": community_id,
            "summary": summary,
            "node_count": node_count,
            "key_entities": key_entities,
        }

    except Exception as e:
        logger.warning(f"Impossibile ricostruire community summary: {e}")
        return None


def count_community_summaries(
    store: Any = None,
) -> int:
    """Conta quanti community summary esistono nel Qdrant store.

    Args:
        store: Istanza ``QdrantStore``.

    Returns:
        Numero di community summary presenti.
    """
    if store is None:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()

    collection = settings.qdrant.collection_graph_communities

    if not store.client.collection_exists(collection):
        return 0

    try:
        info = store.client.get_collection(collection)
        return info.points_count or 0
    except Exception as e:
        logger.error(f"Errore conteggio community summaries: {e}")
        return 0


# ============================================================================
# Standalone convenience functions
# ============================================================================

def summarize_graph_communities(
    algorithm: str = "louvain",
    min_size: int = 3,
    save_to_qdrant: bool = True,
    force_model: str | None = None,
) -> list[dict]:
    """Convenience: pipeline completa detection → summarization → save.

    One-shot function che esegue l'intera pipeline:
    1. Rileva le comunità nel grafo Neo4j
    2. Genera summary testuali per ciascuna
    3. Salva i summary in Qdrant

    Args:
        algorithm: ``"louvain"`` o ``"leiden"``.
        min_size: Dimensione minima delle comunità.
        save_to_qdrant: Se True, salva i summary in Qdrant.
        force_model: Override tier modello LLM.

    Returns:
        Lista di dizionari community summary.

    Raises:
        GDSNotAvailableError: Se GDS non è installato.
        CommunityDetectionError: Se la detection fallisce.

    Uso::

        from paperhunter.graphrag.community_summarizer import summarize_graph_communities

        results = summarize_graph_communities(algorithm="louvain", min_size=3)
        for r in results:
            print(f"Community {r['community_id']}: {r['summary'][:100]}...")
    """
    from paperhunter.graphrag.community_detector import CommunityDetector
    from paperhunter.graphrag.neo4j_manager import Neo4jManager
    from paperhunter.indexing.qdrant_store import QdrantStore

    with Neo4jManager() as mgr:
        detector = CommunityDetector(mgr)
        communities = detector.detect_communities(
            algorithm=algorithm,
            min_size=min_size,
        )

        if not communities:
            logger.info("Nessuna comunità trovata, nessun summary da generare")
            return []

        store = QdrantStore()
        summarizer = CommunitySummarizer(detector, store)

        # Genera i summary
        results = summarizer.summarize_all_communities(
            communities,
            force_model=force_model,
        )

        # Salva in Qdrant
        if save_to_qdrant and results:
            summarizer.save_to_qdrant(results)

    return results
