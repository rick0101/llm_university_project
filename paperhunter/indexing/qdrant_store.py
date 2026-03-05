"""
Qdrant Vector Store per PaperHunter.

Storage vettoriale persistente con metadata filtering nativo.
Usa Qdrant in modalità locale (senza server) per zero-cost deployment.

Gestisce tre collezioni separate:
- ``paper_chunks``  — chunk originali estratti dai PDF
- ``l1_summaries``  — summary L1 (singolo paper)
- ``l2_summaries``  — meta-summary L2 (per categoria)

Ogni punto nel vector store contiene:
- **vector**: embedding Gemini 3072-dim
- **payload**: metadata completi (chunk_id, bibtex_key, category, year, ecc.)

Uso::

    from paperhunter.indexing.qdrant_store import QdrantStore

    store = QdrantStore()

    # Aggiungere chunk con embedding automatico
    store.add_chunks(chunks)  # list[PaperChunk]

    # Ricerca vettoriale con filtro per categoria
    results = store.query(
        query_text="prompt injection defense mechanisms",
        n_results=10,
        where_filter={"category": "Prompt Injection"},
    )

    # Verificare se un paper è già indicizzato
    store.has_paper("Zou_2023_GCG")  # → True/False

    # Statistiche
    store.get_stats()  # → {total_chunks: 1200, categories: {...}, ...}
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient, models

from paperhunter.models import ChunkType, PaperChunk
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Converte un chunk_id stringa in UUID deterministico per Qdrant.

    Qdrant in modalità locale richiede UUID validi come ID dei punti.
    Usa UUID v5 (namespace-based) per garantire determinismo:
    stesso chunk_id → stesso UUID → upsert idempotente.

    Args:
        chunk_id: ID del chunk (es. hash hex di 16 caratteri).

    Returns:
        Stringa UUID valida.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _distance_from_config(metric_name: str) -> models.Distance:
    """Converte il nome della metrica dalla config in enum Qdrant.

    Args:
        metric_name: "Cosine", "Euclid" o "Dot" (da config.yaml).

    Returns:
        models.Distance enum.

    Raises:
        ValueError: Se il nome non è riconosciuto.
    """
    mapping = {
        "Cosine": models.Distance.COSINE,
        "Euclid": models.Distance.EUCLID,
        "Dot": models.Distance.DOT,
    }
    if metric_name not in mapping:
        raise ValueError(
            f"Metrica distanza non valida: {metric_name!r}. "
            f"Usa: {list(mapping.keys())}"
        )
    return mapping[metric_name]


def _build_qdrant_filter(where_filter: dict[str, Any]) -> models.Filter:
    """Costruisce un filtro Qdrant da un dict semplificato.

    Supporta:
    - Stringa: match esatto  → ``{"category": "Prompt Injection"}``
    - Int/Float: match esatto → ``{"year": 2024}``
    - List[str]: match any    → ``{"category": ["Prompt Injection", "Adversarial Attacks"]}``
    - Tuple (min, max): range → ``{"year": (2022, 2025)}``
    - Dict con operatori:     → ``{"year": {"gte": 2022, "lte": 2025}}``

    Args:
        where_filter: Dizionario {campo: valore/condizione}.

    Returns:
        models.Filter per Qdrant.
    """
    must_conditions: list[models.Condition] = []

    for key, value in where_filter.items():
        if value is None:
            continue

        if isinstance(value, str):
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
        elif isinstance(value, list):
            # Match any of the values
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value),
                )
            )
        elif isinstance(value, tuple) and len(value) == 2:
            # Range: (min, max)
            range_kwargs: dict[str, Any] = {}
            if value[0] is not None:
                range_kwargs["gte"] = value[0]
            if value[1] is not None:
                range_kwargs["lte"] = value[1]
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    range=models.Range(**range_kwargs),
                )
            )
        elif isinstance(value, dict):
            # Dict con operatori espliciti: {"gte": 2022, "lte": 2025}
            range_kwargs = {}
            for op in ("gt", "gte", "lt", "lte"):
                if op in value:
                    range_kwargs[op] = value[op]
            if range_kwargs:
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        range=models.Range(**range_kwargs),
                    )
                )
        else:
            logger.warning(f"Tipo filtro non supportato per '{key}': {type(value)}")

    return models.Filter(must=must_conditions) if must_conditions else models.Filter()


# ===========================================================================
# QdrantStore
# ===========================================================================

class QdrantStore:
    """Vector store basato su Qdrant per indicizzazione e retrieval di chunk.

    Opera in modalità locale (su disco, senza server) per zero costi.
    Supporta tre collezioni separate per chunk originali, L1 e L2 summaries.

    Args:
        persist_dir: Directory per lo storage persistente su disco.
            Default da ``config.yaml`` → ``qdrant.persist_dir``.
        embedding_model: Istanza embedding model (LlamaIndex).
            Default: ``get_embedding_model()`` da llm_factory.
        collection_name: Nome della collezione di default.
            Default da ``config.yaml`` → ``qdrant.collection_paper_chunks``.

    Attributes:
        client: Istanza ``QdrantClient`` connessa.
        embedding_model: Modello di embedding per vettorizzazione.
        default_collection: Nome della collezione di default.
    """

    # Nomi collezione di default (sovrascrivibili da config.yaml)
    COLLECTION_CHUNKS = "paper_chunks"
    COLLECTION_L1 = "l1_summaries"
    COLLECTION_L1_5 = "l1_5_summaries"
    COLLECTION_L2 = "l2_summaries"
    COLLECTION_GRAPH_COMMUNITIES = "graph_communities"

    # Campi payload con indice (no-op in locale, utile se si migra a server)
    _INDEXED_FIELDS = {
        "category": models.PayloadSchemaType.KEYWORD,
        "chunk_type": models.PayloadSchemaType.KEYWORD,
        "bibtex_key": models.PayloadSchemaType.KEYWORD,
        "paper_id": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
        "month": models.PayloadSchemaType.INTEGER,
    }

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ):
        from paperhunter.config import settings

        cfg = settings.qdrant

        # Directory persistenza
        self._persist_dir = Path(persist_dir or cfg.persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # Nomi collezioni da config
        self.COLLECTION_CHUNKS = cfg.collection_paper_chunks
        self.COLLECTION_L1 = cfg.collection_l1_summaries
        self.COLLECTION_L1_5 = cfg.collection_l1_5_summaries
        self.COLLECTION_L2 = cfg.collection_l2_summaries
        self.COLLECTION_GRAPH_COMMUNITIES = cfg.collection_graph_communities

        # Collezione di default
        self.default_collection = collection_name or self.COLLECTION_CHUNKS

        # Parametri vettoriali
        self._embedding_dim = cfg.embedding_dimension
        self._distance = _distance_from_config(cfg.distance_metric)

        # Inizializza client Qdrant (locale, su disco)
        self.client = QdrantClient(path=str(self._persist_dir))
        logger.info(
            f"QdrantStore inizializzato: {self._persist_dir} "
            f"(dim={self._embedding_dim}, distance={cfg.distance_metric})"
        )

        # Embedding model (lazy: creato al primo uso se non fornito)
        self._embedding_model = embedding_model
        self._embedding_model_initialized = embedding_model is not None

        # Assicura che le collezioni esistano
        self._ensure_collections()

    # ------------------------------------------------------------------
    # Gestione collezioni
    # ------------------------------------------------------------------

    def _ensure_collections(self) -> None:
        """Crea le collezioni se non esistono già."""
        for col_name in (self.COLLECTION_CHUNKS, self.COLLECTION_L1, self.COLLECTION_L1_5, self.COLLECTION_L2, self.COLLECTION_GRAPH_COMMUNITIES):
            if not self.client.collection_exists(col_name):
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=models.VectorParams(
                        size=self._embedding_dim,
                        distance=self._distance,
                    ),
                )
                # Crea indici payload (no-op in locale, utile per migrazione a server)
                self._create_payload_indexes(col_name)
                logger.info(f"Collezione '{col_name}' creata (dim={self._embedding_dim})")
            else:
                logger.debug(f"Collezione '{col_name}' già esistente")

    def _create_payload_indexes(self, collection_name: str) -> None:
        """Crea indici sui campi payload per filtering veloce.

        In modalità locale Qdrant non usa realmente gli indici,
        ma averli dichiarati facilita la migrazione a server Qdrant.
        """
        for field_name, field_schema in self._INDEXED_FIELDS.items():
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                pass  # Ignora errori (es. indice già esistente)

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """Elimina una collezione e tutti i suoi dati.

        Args:
            collection_name: Nome della collezione da eliminare.
                Default: ``default_collection``.
        """
        col = collection_name or self.default_collection
        if self.client.collection_exists(col):
            self.client.delete_collection(col)
            logger.warning(f"Collezione '{col}' eliminata")
        else:
            logger.debug(f"Collezione '{col}' non trovata per eliminazione")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_embedding_model(self):
        """Ottieni il modello di embedding (lazy initialization).

        Returns:
            Istanza GoogleGenAIEmbedding.
        """
        if not self._embedding_model_initialized:
            from paperhunter.utils.llm_factory import get_embedding_model
            self._embedding_model = get_embedding_model()
            self._embedding_model_initialized = True
            logger.debug("Embedding model inizializzato (lazy)")
        return self._embedding_model

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Genera embedding per una lista di testi.

        Usa il rate limiter Gemini e processa in batch.

        Args:
            texts: Lista di stringhe da vettorizzare.

        Returns:
            Lista di vettori embedding (list[float]).
        """
        from paperhunter.utils.rate_limiter import get_gemini_limiter

        embed_model = self._get_embedding_model()
        limiter = get_gemini_limiter()

        embeddings: list[list[float]] = []
        batch_size = getattr(embed_model, "embed_batch_size", 10)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            limiter.wait()  # Rispetta rate limit Gemini
            try:
                batch_embeddings = embed_model.get_text_embedding_batch(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Errore embedding batch {i//batch_size}: {e}")
                # Fallback: prova uno alla volta
                for text in batch:
                    try:
                        limiter.wait()
                        emb = embed_model.get_text_embedding(text)
                        embeddings.append(emb)
                    except Exception as e2:
                        logger.error(f"Errore embedding singolo: {e2}")
                        # Vettore zero come fallback estremo
                        embeddings.append([0.0] * self._embedding_dim)

        return embeddings

    def _embed_query(self, query_text: str) -> list[float]:
        """Genera embedding per una singola query.

        Usa ``get_query_embedding`` se disponibile (ottimizzato per query),
        altrimenti ``get_text_embedding``.

        Args:
            query_text: Testo della query.

        Returns:
            Vettore embedding.
        """
        from paperhunter.utils.rate_limiter import get_gemini_limiter

        embed_model = self._get_embedding_model()
        limiter = get_gemini_limiter()

        limiter.wait()

        # Preferisci get_query_embedding se disponibile
        if hasattr(embed_model, "get_query_embedding"):
            return embed_model.get_query_embedding(query_text)
        return embed_model.get_text_embedding(query_text)

    # ------------------------------------------------------------------
    # CRUD — Add / Upsert
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[PaperChunk],
        collection_name: Optional[str] = None,
        batch_size: int = 50,
        show_progress: bool = True,
    ) -> int:
        """Aggiunge chunk al vector store con upsert (deduplicazione per chunk_id).

        Genera embedding automaticamente e carica i punti in batch.
        Se un chunk con lo stesso ID esiste già, viene aggiornato.

        Args:
            chunks: Lista di ``PaperChunk`` da indicizzare.
            collection_name: Collezione target. Default: ``default_collection``.
            batch_size: Dimensione batch per upsert Qdrant.
            show_progress: Mostra progress bar.

        Returns:
            Numero di chunk inseriti/aggiornati.
        """
        if not chunks:
            logger.warning("add_chunks chiamato con lista vuota")
            return 0

        col = collection_name or self.default_collection
        start_time = time.time()

        # Prepara testi per embedding
        texts = [chunk.text for chunk in chunks]
        logger.info(f"Generazione embedding per {len(texts)} chunk...")

        # Genera embedding in batch
        embeddings = self._embed_texts(texts)

        if len(embeddings) != len(chunks):
            logger.error(
                f"Mismatch: {len(chunks)} chunk ma {len(embeddings)} embedding. "
                f"Abbandono operazione."
            )
            return 0

        # Prepara punti Qdrant
        total_upserted = 0

        # Processa in batch
        if show_progress:
            from paperhunter.utils.logging_setup import progress_bar
            batch_iter = progress_bar(
                range(0, len(chunks), batch_size),
                description=f"Upsert in '{col}'",
                total=(len(chunks) + batch_size - 1) // batch_size,
            )
        else:
            batch_iter = range(0, len(chunks), batch_size)

        for batch_start in batch_iter:
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]

            points = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                point_id = _chunk_id_to_uuid(chunk.chunk_id)
                payload = chunk.to_metadata_dict()
                payload["text"] = chunk.text  # Salva il testo nel payload per retrieval

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            try:
                self.client.upsert(
                    collection_name=col,
                    points=points,
                )
                total_upserted += len(points)
            except Exception as e:
                logger.error(
                    f"Errore upsert batch {batch_start}-{batch_end} "
                    f"in '{col}': {e}"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"Upsert completato: {total_upserted}/{len(chunks)} chunk "
            f"in '{col}' ({elapsed:.1f}s)"
        )
        return total_upserted

    def add_chunks_with_embeddings(
        self,
        chunks: list[PaperChunk],
        embeddings: list[list[float]],
        collection_name: Optional[str] = None,
        batch_size: int = 50,
    ) -> int:
        """Aggiunge chunk con embedding pre-calcolati (skip embedding generation).

        Utile quando gli embedding sono già disponibili (es. da cache o
        da un passo precedente della pipeline).

        Args:
            chunks: Lista di ``PaperChunk``.
            embeddings: Lista di vettori embedding corrispondenti.
            collection_name: Collezione target.
            batch_size: Dimensione batch.

        Returns:
            Numero di chunk inseriti/aggiornati.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunk ma {len(embeddings)} embedding"
            )

        col = collection_name or self.default_collection
        total_upserted = 0

        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            points = []

            for chunk, embedding in zip(
                chunks[batch_start:batch_end],
                embeddings[batch_start:batch_end],
            ):
                point_id = _chunk_id_to_uuid(chunk.chunk_id)
                payload = chunk.to_metadata_dict()
                payload["text"] = chunk.text

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            try:
                self.client.upsert(collection_name=col, points=points)
                total_upserted += len(points)
            except Exception as e:
                logger.error(f"Errore upsert batch pre-embedded: {e}")

        return total_upserted

    # ------------------------------------------------------------------
    # CRUD — Query / Search
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where_filter: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Ricerca vettoriale con filtro metadata opzionale.

        Genera l'embedding della query, cerca i punti più vicini
        nella collezione, e ritorna i risultati con payload completo.

        Args:
            query_text: Testo della query.
            n_results: Numero massimo di risultati.
            where_filter: Filtro metadata (vedi ``_build_qdrant_filter``).
                Esempi:
                - ``{"category": "Prompt Injection"}``
                - ``{"year": (2022, 2025)}``
                - ``{"category": ["Prompt Injection", "Adversarial Attacks"], "year": {"gte": 2023}}``
            collection_name: Collezione in cui cercare.
            score_threshold: Soglia minima di similarità (0-1 per Cosine).

        Returns:
            Lista di dict con chiavi: ``chunk_id``, ``text``, ``score``,
            ``metadata`` (tutti i campi payload).
        """
        col = collection_name or self.default_collection
        start_time = time.time()

        # Genera embedding della query
        query_vector = self._embed_query(query_text)

        # Costruisci filtro
        qdrant_filter = None
        if where_filter:
            qdrant_filter = _build_qdrant_filter(where_filter)

        # Esegui query
        try:
            results = self.client.query_points(
                collection_name=col,
                query=query_vector,
                limit=n_results,
                with_payload=True,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logger.error(f"Errore query in '{col}': {e}")
            return []

        # Formatta risultati
        formatted: list[dict[str, Any]] = []
        for point in results.points:
            payload = point.payload or {}
            formatted.append({
                "chunk_id": payload.get("chunk_id", ""),
                "text": payload.get("text", ""),
                "score": point.score,
                "metadata": {k: v for k, v in payload.items() if k != "text"},
                "qdrant_id": str(point.id),
            })

        elapsed = time.time() - start_time
        logger.debug(
            f"Query in '{col}': {len(formatted)} risultati "
            f"(filtro={where_filter}, {elapsed:.3f}s)"
        )
        return formatted

    def query_by_vector(
        self,
        query_vector: list[float],
        n_results: int = 10,
        where_filter: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Ricerca per vettore pre-calcolato (skip embedding generation).

        Identico a ``query()`` ma accetta un vettore già calcolato.

        Args:
            query_vector: Vettore embedding della query.
            n_results: Numero massimo di risultati.
            where_filter: Filtro metadata.
            collection_name: Collezione target.
            score_threshold: Soglia minima di similarità.

        Returns:
            Lista di risultati (stessa struttura di ``query()``).
        """
        col = collection_name or self.default_collection
        qdrant_filter = _build_qdrant_filter(where_filter) if where_filter else None

        try:
            results = self.client.query_points(
                collection_name=col,
                query=query_vector,
                limit=n_results,
                with_payload=True,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logger.error(f"Errore query_by_vector in '{col}': {e}")
            return []

        formatted: list[dict[str, Any]] = []
        for point in results.points:
            payload = point.payload or {}
            formatted.append({
                "chunk_id": payload.get("chunk_id", ""),
                "text": payload.get("text", ""),
                "score": point.score,
                "metadata": {k: v for k, v in payload.items() if k != "text"},
                "qdrant_id": str(point.id),
            })
        return formatted

    # ------------------------------------------------------------------
    # CRUD — Delete
    # ------------------------------------------------------------------

    def delete_by_paper(
        self,
        bibtex_key: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """Rimuove tutti i chunk di un paper (per bibtex_key).

        Args:
            bibtex_key: Chiave BibTeX del paper (es. "Zou_2023_GCG").
            collection_name: Collezione target.

        Returns:
            Numero approssimativo di punti eliminati.
        """
        col = collection_name or self.default_collection

        # Conta prima per logging
        pre_count = self._count_by_filter(col, {"bibtex_key": bibtex_key})

        try:
            self.client.delete(
                collection_name=col,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="bibtex_key",
                                match=models.MatchValue(value=bibtex_key),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            logger.error(f"Errore delete per bibtex_key={bibtex_key!r}: {e}")
            return 0

        logger.info(f"Eliminati ~{pre_count} chunk per paper '{bibtex_key}' da '{col}'")
        return pre_count

    def delete_by_paper_id(
        self,
        paper_id: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """Rimuove tutti i chunk di un paper (per paper_id).

        Args:
            paper_id: ID del paper.
            collection_name: Collezione target.

        Returns:
            Numero approssimativo di punti eliminati.
        """
        col = collection_name or self.default_collection
        pre_count = self._count_by_filter(col, {"paper_id": paper_id})

        try:
            self.client.delete(
                collection_name=col,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="paper_id",
                                match=models.MatchValue(value=paper_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            logger.error(f"Errore delete per paper_id={paper_id!r}: {e}")
            return 0

        logger.info(f"Eliminati ~{pre_count} chunk per paper_id '{paper_id}' da '{col}'")
        return pre_count

    def delete_by_category(
        self,
        category: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """Rimuove tutti i chunk di una categoria.

        Args:
            category: Nome della categoria (es. "Prompt Injection").
            collection_name: Collezione target.

        Returns:
            Numero approssimativo di punti eliminati.
        """
        col = collection_name or self.default_collection
        pre_count = self._count_by_filter(col, {"category": category})

        try:
            self.client.delete(
                collection_name=col,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="category",
                                match=models.MatchValue(value=category),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            logger.error(f"Errore delete per category={category!r}: {e}")
            return 0

        logger.info(f"Eliminati ~{pre_count} chunk di '{category}' da '{col}'")
        return pre_count

    # ------------------------------------------------------------------
    # CRUD — Lookup / Check
    # ------------------------------------------------------------------

    def has_paper(
        self,
        bibtex_key: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Verifica se un paper è già indicizzato (per bibtex_key).

        Args:
            bibtex_key: Chiave BibTeX del paper.
            collection_name: Collezione da controllare.

        Returns:
            True se almeno un chunk del paper esiste nella collezione.
        """
        col = collection_name or self.default_collection
        try:
            result = self.client.scroll(
                collection_name=col,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="bibtex_key",
                            match=models.MatchValue(value=bibtex_key),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(result[0]) > 0
        except Exception as e:
            logger.error(f"Errore has_paper({bibtex_key!r}): {e}")
            return False

    def has_paper_id(
        self,
        paper_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Verifica se un paper_id è già indicizzato.

        Args:
            paper_id: ID del paper.
            collection_name: Collezione da controllare.

        Returns:
            True se almeno un chunk del paper esiste.
        """
        col = collection_name or self.default_collection
        try:
            result = self.client.scroll(
                collection_name=col,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="paper_id",
                            match=models.MatchValue(value=paper_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(result[0]) > 0
        except Exception as e:
            logger.error(f"Errore has_paper_id({paper_id!r}): {e}")
            return False

    def get_paper_chunks(
        self,
        bibtex_key: str,
        collection_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Recupera tutti i chunk di un paper (senza calcolo similarità).

        Args:
            bibtex_key: Chiave BibTeX del paper.
            collection_name: Collezione.

        Returns:
            Lista di dict con payload di ogni chunk.
        """
        col = collection_name or self.default_collection
        all_chunks: list[dict[str, Any]] = []
        offset = None

        while True:
            try:
                result, next_offset = self.client.scroll(
                    collection_name=col,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="bibtex_key",
                                match=models.MatchValue(value=bibtex_key),
                            )
                        ]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                logger.error(f"Errore get_paper_chunks({bibtex_key!r}): {e}")
                break

            for point in result:
                payload = point.payload or {}
                all_chunks.append({
                    "chunk_id": payload.get("chunk_id", ""),
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                    "qdrant_id": str(point.id),
                })

            if next_offset is None:
                break
            offset = next_offset

        return all_chunks

    # ------------------------------------------------------------------
    # Statistiche
    # ------------------------------------------------------------------

    def get_stats(
        self,
        collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Statistiche dettagliate della collezione.

        Args:
            collection_name: Collezione da analizzare. Se None,
                ritorna stats per tutte le collezioni.

        Returns:
            Dict con: total_points, categories, chunk_types, years,
            bibtex_keys_count, collection_info.
        """
        if collection_name:
            return self._get_collection_stats(collection_name)

        # Stats per tutte le collezioni
        all_stats: dict[str, Any] = {}
        for col_name in (self.COLLECTION_CHUNKS, self.COLLECTION_L1, self.COLLECTION_L1_5, self.COLLECTION_L2, self.COLLECTION_GRAPH_COMMUNITIES):
            if self.client.collection_exists(col_name):
                all_stats[col_name] = self._get_collection_stats(col_name)
            else:
                all_stats[col_name] = {"total_points": 0, "exists": False}

        # Totali aggregati
        total = sum(s.get("total_points", 0) for s in all_stats.values())
        all_stats["_total"] = total

        return all_stats

    def _get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Statistiche per una singola collezione.

        Args:
            collection_name: Nome collezione.

        Returns:
            Dict con statistiche dettagliate.
        """
        try:
            info = self.client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Errore get_stats per '{collection_name}': {e}")
            return {"total_points": 0, "error": str(e)}

        total_points = info.points_count or 0

        # Raccogli statistiche per categoria, tipo, anno
        categories: dict[str, int] = {}
        chunk_types: dict[str, int] = {}
        years: dict[int, int] = {}
        bibtex_keys: set[str] = set()

        # Scroll tutti i punti (solo metadata leggeri)
        offset = None
        while True:
            try:
                points, next_offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=500,
                    offset=offset,
                    with_payload=["category", "chunk_type", "year", "bibtex_key"],
                    with_vectors=False,
                )
            except Exception:
                break

            for p in points:
                payload = p.payload or {}

                cat = payload.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1

                ct = payload.get("chunk_type", "unknown")
                chunk_types[ct] = chunk_types.get(ct, 0) + 1

                yr = payload.get("year", 0)
                if yr and yr > 0:
                    years[yr] = years.get(yr, 0) + 1

                bk = payload.get("bibtex_key", "")
                if bk:
                    bibtex_keys.add(bk)

            if next_offset is None:
                break
            offset = next_offset

        return {
            "exists": True,
            "total_points": total_points,
            "categories": dict(sorted(categories.items())),
            "chunk_types": dict(sorted(chunk_types.items())),
            "years": dict(sorted(years.items())),
            "unique_papers": len(bibtex_keys),
            "distance_metric": str(self._distance),
            "embedding_dim": self._embedding_dim,
        }

    def get_collection_count(
        self,
        collection_name: Optional[str] = None,
    ) -> int:
        """Conta il numero totale di punti in una collezione.

        Args:
            collection_name: Nome collezione.

        Returns:
            Numero di punti.
        """
        col = collection_name or self.default_collection
        try:
            info = self.client.get_collection(col)
            return info.points_count or 0
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Utility private
    # ------------------------------------------------------------------

    def _count_by_filter(
        self,
        collection_name: str,
        filter_dict: dict[str, Any],
    ) -> int:
        """Conta punti che corrispondono a un filtro.

        Args:
            collection_name: Collezione.
            filter_dict: Filtro semplificato.

        Returns:
            Numero di punti corrispondenti.
        """
        try:
            result = self.client.count(
                collection_name=collection_name,
                count_filter=_build_qdrant_filter(filter_dict),
                exact=True,
            )
            return result.count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Chiude la connessione al client Qdrant."""
        try:
            self.client.close()
            logger.debug("QdrantStore: connessione chiusa")
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"QdrantStore(persist_dir={self._persist_dir!r}, "
            f"dim={self._embedding_dim}, distance={self._distance})"
        )
