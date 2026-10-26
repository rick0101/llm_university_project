"""
BM25 Keyword Index per PaperHunter.

Indice BM25 (Okapi BM25) complementare alla ricerca vettoriale Qdrant.
Fornisce retrieval keyword-based per il sistema di hybrid retrieval.

Funzionalità:
- Tokenizzazione con lowercase, rimozione stopwords, stemming opzionale
- Indicizzazione di PaperChunk con metadata preservati
- Query con ranking BM25 e filtri per metadata
- Persistenza su disco (pickle)
- Aggiornamento incrementale (rebuild automatico)

Uso::

    from paperhunter.indexing.bm25_index import BM25Index
    from paperhunter.models import PaperChunk

    index = BM25Index()

    # Costruire l'indice da chunk
    index.build_index(chunks)  # list[PaperChunk]

    # Ricerca keyword
    results = index.query("prompt injection defense", n_results=10)
    # → [{"chunk_id": "abc123", "score": 2.45, "text": "...", "metadata": {...}}, ...]

    # Ricerca con filtro
    results = index.query(
        "backdoor attack",
        n_results=5,
        where_filter={"category": "Data and Model Poisoning"},
    )

    # Persistenza
    index.save()          # salva su data/bm25_index.pkl (default da config.yaml)
    index.load()          # ricarica

    # Aggiornamento incrementale
    index.update(new_chunks)  # rebuild con i nuovi chunk aggiunti
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from paperhunter.models import PaperChunk
from paperhunter.utils.logging_setup import get_logger, progress_bar

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stopwords — lista estesa per testi accademici in inglese
# ---------------------------------------------------------------------------

_STOPWORDS: set[str] = {
    # Articoli / pronomi / ausiliari
    "a", "an", "the", "and", "or", "but", "not", "no", "nor", "is", "are",
    "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing",
    "has", "have", "had", "having", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could",
    # Preposizioni / congiunzioni
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "about", "up", "down", "here", "there", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "because", "so",
    "while", "if", "although", "though", "also", "however", "thus",
    "therefore", "hence", "moreover", "furthermore", "nevertheless",
    "nonetheless", "meanwhile", "otherwise", "instead", "yet", "still",
    # Pronomi
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    # Parole molto comuni in paper accademici (troppo generiche per BM25)
    "paper", "section", "figure", "table", "proposed", "results", "method",
    "approach", "show", "shows", "shown", "demonstrated", "using", "used",
    "based", "et", "al", "respectively", "ie", "eg", "etc", "vs",
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Regex per split: sequenze di caratteri alfanumerici (inclusi trattini interni)
_TOKEN_RE = re.compile(r"[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?", re.IGNORECASE)

# Suffissi per stemming naïve (Porter-like molto semplificato)
_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("ation", "ate"),
    ("ness", ""),
    ("ment", ""),
    ("ful", ""),
    ("ous", ""),
    ("ive", ""),
    ("ize", "ize"),
    ("ise", "ise"),
    ("ing", ""),
    ("ies", "y"),
    ("ied", "y"),
    ("ely", ""),
    ("ally", "al"),
    ("ling", ""),
    ("able", ""),
    ("ible", ""),
    ("ly", ""),
    ("ed", ""),
    ("er", ""),
    ("es", ""),
    ("s", ""),
]


def _naive_stem(word: str) -> str:
    """Stemming naïve: rimuove suffissi comuni dall'inglese.

    Non è sofisticato come Porter/Snowball, ma evita dipendenze
    aggiuntive (NLTK) e copre i casi più frequenti.

    Args:
        word: Parola in minuscolo.

    Returns:
        Parola dopo rimozione suffisso.
    """
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = True,
    stem: bool = True,
    min_token_length: int = 2,
) -> list[str]:
    """Tokenizza testo per indicizzazione BM25.

    Pipeline: lowercase → regex split → filtro lunghezza →
    stopword removal → stemming opzionale.

    Args:
        text: Testo da tokenizzare.
        remove_stopwords: Rimuovi stopwords inglesi.
        stem: Applica stemming naïve.
        min_token_length: Lunghezza minima token dopo processing.

    Returns:
        Lista di token processati.
    """
    if not text:
        return []

    # Lowercase e find all token
    tokens = _TOKEN_RE.findall(text.lower())

    # Filtro lunghezza minima
    tokens = [t for t in tokens if len(t) >= min_token_length]

    # Rimozione stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    # Stemming
    if stem:
        tokens = [_naive_stem(t) for t in tokens]

    return tokens


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class BM25Index:
    """Indice BM25 (Okapi BM25) per retrieval keyword-based.

    Indicizza oggetti ``PaperChunk`` mantenendo metadata completi
    per il filtering post-retrieval. Supporta persistenza su disco
    e aggiornamento incrementale.

    Attributes:
        index_path: Percorso file pickle per persistenza.
        chunk_ids: Lista ordinata di chunk_id (parallela all'indice BM25).
        chunks_metadata: Metadata per ogni chunk (dict chunk_id → metadata).
        chunks_text: Testo originale per ogni chunk (dict chunk_id → text).
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        *,
        remove_stopwords: bool = True,
        stem: bool = True,
        min_token_length: int = 2,
    ) -> None:
        """Inizializza l'indice BM25.

        Args:
            index_path: Percorso su disco per save/load. Se None, usa
                il default da ``config.yaml`` → ``retrieval.bm25_index_path``.
            remove_stopwords: Rimuovi stopwords durante tokenizzazione.
            stem: Applica stemming naïve.
            min_token_length: Lunghezza minima dei token.
        """
        if index_path is None:
            try:
                from paperhunter.config import get_settings
                settings = get_settings()
                index_path = settings.retrieval.bm25_index_path
            except Exception:
                index_path = "./data/bm25_index.pkl"

        self.index_path = Path(index_path)
        self._remove_stopwords = remove_stopwords
        self._stem = stem
        self._min_token_length = min_token_length

        # Stato interno
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._chunks_metadata: dict[str, dict[str, Any]] = {}
        self._chunks_text: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Proprietà
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """True se l'indice è stato costruito o caricato."""
        return self._bm25 is not None and len(self._chunk_ids) > 0

    @property
    def size(self) -> int:
        """Numero di documenti nell'indice."""
        return len(self._chunk_ids)

    @property
    def chunk_ids(self) -> list[str]:
        """Lista dei chunk_id indicizzati (ordine di inserimento)."""
        return list(self._chunk_ids)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_index(
        self,
        chunks: list[PaperChunk],
        *,
        show_progress: bool = True,
    ) -> int:
        """Costruisce l'indice BM25 da una lista di PaperChunk.

        Sovrascrive l'indice esistente se presente. Per aggiornamento
        incrementale, usare ``update()``.

        Args:
            chunks: Lista di PaperChunk da indicizzare.
            show_progress: Mostra progress bar durante tokenizzazione.

        Returns:
            Numero di chunk indicizzati.

        Raises:
            ValueError: Se la lista è vuota.
        """
        if not chunks:
            raise ValueError("Impossibile costruire indice BM25 da lista vuota")

        t0 = time.time()
        logger.info("BM25 build_index: tokenizzazione di %d chunk...", len(chunks))

        # Tokenizzazione batch
        self._chunk_ids = []
        self._corpus_tokens = []
        self._chunks_metadata = {}
        self._chunks_text = {}

        seen_ids: set[str] = set()
        skipped_empty = 0
        skipped_duplicate = 0

        iterable = progress_bar(chunks, description="BM25 tokenize") if show_progress else chunks

        for chunk in iterable:
            # Deduplicazione per chunk_id
            if chunk.chunk_id in seen_ids:
                skipped_duplicate += 1
                continue
            seen_ids.add(chunk.chunk_id)

            # Tokenizza
            tokens = self._tokenize(chunk.text)

            # Skip chunk con meno di 2 token
            if len(tokens) < 2:
                skipped_empty += 1
                continue

            self._chunk_ids.append(chunk.chunk_id)
            self._corpus_tokens.append(tokens)
            self._chunks_text[chunk.chunk_id] = chunk.text
            self._chunks_metadata[chunk.chunk_id] = chunk.to_metadata_dict()

        # Costruisci BM25Okapi
        if not self._corpus_tokens:
            raise ValueError("Nessun chunk valido dopo tokenizzazione (tutti vuoti o troppo corti)")

        self._bm25 = BM25Okapi(self._corpus_tokens)

        elapsed = time.time() - t0
        logger.info(
            "BM25 indice costruito: %d chunk in %.2fs (skipped: %d vuoti, %d duplicati)",
            len(self._chunk_ids),
            elapsed,
            skipped_empty,
            skipped_duplicate,
        )

        return len(self._chunk_ids)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        *,
        where_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Esegue ricerca BM25 con ranking e filtri opzionali.

        Args:
            query_text: Testo della query (verrà tokenizzato).
            n_results: Numero massimo di risultati da ritornare.
            where_filter: Filtro metadata opzionale. Chiavi supportate:
                - ``"category"`` (str): match esatto sulla categoria
                - ``"year"`` (int): match esatto sull'anno
                - ``"year_min"`` (int): anno >= valore
                - ``"year_max"`` (int): anno <= valore
                - ``"chunk_type"`` (str): match esatto sul tipo
                - ``"bibtex_key"`` (str): match esatto sulla chiave BibTeX
                - ``"paper_id"`` (str): match esatto sul paper_id
            score_threshold: Score minimo per includere un risultato (default: 0.0).

        Returns:
            Lista di dict ordinati per score decrescente. Ogni dict ha:
            ``chunk_id``, ``score``, ``text``, ``metadata``.

        Raises:
            RuntimeError: Se l'indice non è stato costruito/caricato.
        """
        if not self.is_built:
            raise RuntimeError(
                "Indice BM25 non costruito. Chiamare build_index() o load() prima."
            )

        if not query_text.strip():
            return []

        # Tokenizza la query
        query_tokens = self._tokenize(query_text)

        if not query_tokens:
            logger.warning("BM25 query: nessun token utile dalla query '%s'", query_text)
            return []

        # Calcola score BM25 per tutto il corpus
        scores = self._bm25.get_scores(query_tokens)

        # Costruisci risultati con score > threshold
        results: list[dict[str, Any]] = []
        for idx, score_val in enumerate(scores):
            if score_val <= score_threshold:
                continue

            chunk_id = self._chunk_ids[idx]
            metadata = self._chunks_metadata.get(chunk_id, {})

            # Applica filtro metadata
            if where_filter and not self._matches_filter(metadata, where_filter):
                continue

            results.append({
                "chunk_id": chunk_id,
                "score": float(score_val),
                "text": self._chunks_text.get(chunk_id, ""),
                "metadata": metadata,
            })

        # Ordina per score decrescente e tronca
        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:n_results]

        logger.debug(
            "BM25 query '%s': %d risultati (top score=%.3f)",
            query_text[:60],
            len(results),
            results[0]["score"] if results else 0.0,
        )

        return results

    def query_batch(
        self,
        queries: list[str],
        n_results: int = 10,
        *,
        where_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[list[dict[str, Any]]]:
        """Esegue più query BM25 in batch.

        Args:
            queries: Lista di testi query.
            n_results: Risultati per query.
            where_filter: Filtro condiviso tra tutte le query.
            score_threshold: Score minimo.

        Returns:
            Lista di liste di risultati (una per query).
        """
        return [
            self.query(
                q,
                n_results=n_results,
                where_filter=where_filter,
                score_threshold=score_threshold,
            )
            for q in queries
        ]

    # ------------------------------------------------------------------
    # Update (incrementale)
    # ------------------------------------------------------------------

    def update(
        self,
        new_chunks: list[PaperChunk],
        *,
        show_progress: bool = True,
    ) -> int:
        """Aggiorna l'indice aggiungendo nuovi chunk (rebuild).

        BM25Okapi non supporta aggiornamento in-place: ricostruisce
        l'intero indice includendo sia i chunk esistenti sia i nuovi.
        I chunk con chunk_id già presente vengono aggiornati (il testo
        e i metadata vengono sostituiti).

        Args:
            new_chunks: Nuovi chunk da aggiungere.
            show_progress: Mostra progress bar.

        Returns:
            Dimensione totale dell'indice dopo l'aggiornamento.

        Raises:
            ValueError: Se i nuovi chunk sono vuoti e l'indice è vuoto.
        """
        if not new_chunks:
            logger.warning("BM25 update: nessun chunk fornito, indice invariato")
            return self.size

        t0 = time.time()
        logger.info("BM25 update: aggiunta di %d chunk all'indice...", len(new_chunks))

        # Ricostruisci: chunk esistenti (non sovrascritti) + nuovi
        existing_ids = set(self._chunk_ids)
        new_ids_set: set[str] = set()
        updated_count = 0
        added_count = 0

        for chunk in new_chunks:
            new_ids_set.add(chunk.chunk_id)
            if chunk.chunk_id in existing_ids:
                updated_count += 1
            else:
                added_count += 1

        # Prendi i vecchi chunk che NON vengono sovrascritti dai nuovi
        preserved_tokens: list[list[str]] = []
        preserved_ids: list[str] = []
        for i, cid in enumerate(self._chunk_ids):
            if cid not in new_ids_set:
                preserved_tokens.append(self._corpus_tokens[i])
                preserved_ids.append(cid)

        # Preserva metadata e testo dei chunk non sovrascritti
        new_metadata = dict(self._chunks_metadata)
        new_text = dict(self._chunks_text)

        # Tokenizza e aggiungi i nuovi
        skipped_empty = 0
        iterable = progress_bar(new_chunks, description="BM25 update") if show_progress else new_chunks

        for chunk in iterable:
            tokens = self._tokenize(chunk.text)
            if len(tokens) < 2:
                skipped_empty += 1
                continue

            preserved_ids.append(chunk.chunk_id)
            preserved_tokens.append(tokens)
            new_metadata[chunk.chunk_id] = chunk.to_metadata_dict()
            new_text[chunk.chunk_id] = chunk.text

        # Rebuild
        self._chunk_ids = preserved_ids
        self._corpus_tokens = preserved_tokens
        self._chunks_metadata = {cid: new_metadata[cid] for cid in preserved_ids if cid in new_metadata}
        self._chunks_text = {cid: new_text[cid] for cid in preserved_ids if cid in new_text}

        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

        elapsed = time.time() - t0
        logger.info(
            "BM25 update: totale %d chunk (aggiunti=%d, aggiornati=%d, vuoti=%d) in %.2fs",
            len(self._chunk_ids),
            added_count,
            updated_count,
            skipped_empty,
            elapsed,
        )

        return self.size

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_paper(self, bibtex_key: str) -> int:
        """Elimina tutti i chunk di un paper per bibtex_key.

        Ricostruisce l'indice BM25 senza i chunk rimossi.

        Args:
            bibtex_key: Chiave BibTeX del paper da rimuovere.

        Returns:
            Numero di chunk rimossi.
        """
        if not self.is_built:
            return 0

        ids_to_remove = {
            cid
            for cid, meta in self._chunks_metadata.items()
            if meta.get("bibtex_key") == bibtex_key
        }

        if not ids_to_remove:
            return 0

        return self._remove_chunk_ids(ids_to_remove)

    def delete_by_category(self, category: str) -> int:
        """Elimina tutti i chunk di una categoria.

        Args:
            category: Nome della categoria da rimuovere.

        Returns:
            Numero di chunk rimossi.
        """
        if not self.is_built:
            return 0

        ids_to_remove = {
            cid
            for cid, meta in self._chunks_metadata.items()
            if meta.get("category") == category
        }

        if not ids_to_remove:
            return 0

        return self._remove_chunk_ids(ids_to_remove)

    def delete_by_paper_id(self, paper_id: str) -> int:
        """Elimina tutti i chunk di un paper per paper_id.

        Args:
            paper_id: ID del paper da rimuovere.

        Returns:
            Numero di chunk rimossi.
        """
        if not self.is_built:
            return 0

        ids_to_remove = {
            cid
            for cid, meta in self._chunks_metadata.items()
            if meta.get("paper_id") == paper_id
        }

        if not ids_to_remove:
            return 0

        return self._remove_chunk_ids(ids_to_remove)

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Salva l'indice BM25 su disco (pickle).

        Salva tutto il necessario per ricaricare l'indice senza
        dover ri-tokenizzare: corpus_tokens, chunk_ids, metadata, testi.

        Args:
            path: Percorso file. Se None, usa ``self.index_path``.

        Returns:
            Path del file salvato.

        Raises:
            RuntimeError: Se l'indice non è stato costruito.
        """
        if not self.is_built:
            raise RuntimeError("Impossibile salvare: indice non costruito")

        save_path = Path(path) if path else self.index_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "chunk_ids": self._chunk_ids,
            "corpus_tokens": self._corpus_tokens,
            "chunks_metadata": self._chunks_metadata,
            "chunks_text": self._chunks_text,
            "config": {
                "remove_stopwords": self._remove_stopwords,
                "stem": self._stem,
                "min_token_length": self._min_token_length,
            },
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = save_path.stat().st_size / (1024 * 1024)
        logger.info(
            "BM25 indice salvato: %s (%.2f MB, %d chunk)",
            save_path,
            size_mb,
            len(self._chunk_ids),
        )

        return save_path

    def load(self, path: str | Path | None = None) -> int:
        """Carica l'indice BM25 da disco.

        Ricostruisce l'oggetto ``BM25Okapi`` dal corpus_tokens salvato.

        Args:
            path: Percorso file. Se None, usa ``self.index_path``.

        Returns:
            Numero di chunk caricati.

        Raises:
            FileNotFoundError: Se il file non esiste.
            ValueError: Se il formato è incompatibile.
        """
        load_path = Path(path) if path else self.index_path

        if not load_path.exists():
            raise FileNotFoundError(f"File indice BM25 non trovato: {load_path}")

        t0 = time.time()

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        # Validazione formato
        if not isinstance(data, dict) or "version" not in data:
            raise ValueError(f"Formato file indice BM25 non valido: {load_path}")

        if data["version"] != 1:
            raise ValueError(
                f"Versione indice BM25 non supportata: {data['version']} (attesa: 1)"
            )

        self._chunk_ids = data["chunk_ids"]
        self._corpus_tokens = data["corpus_tokens"]
        self._chunks_metadata = data["chunks_metadata"]
        self._chunks_text = data["chunks_text"]

        # Ricostruisci BM25Okapi
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

        # Aggiorna config tokenizzazione se presente nel file
        cfg = data.get("config", {})
        if cfg:
            self._remove_stopwords = cfg.get("remove_stopwords", self._remove_stopwords)
            self._stem = cfg.get("stem", self._stem)
            self._min_token_length = cfg.get("min_token_length", self._min_token_length)

        elapsed = time.time() - t0
        logger.info(
            "BM25 indice caricato: %s (%d chunk in %.3fs)",
            load_path,
            len(self._chunk_ids),
            elapsed,
        )

        return len(self._chunk_ids)

    # ------------------------------------------------------------------
    # Statistiche / Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Statistiche dell'indice BM25.

        Returns:
            Dict con: total_chunks, categories (dict con conteggi),
            chunk_types, years, avg_tokens_per_doc, unique_papers.
        """
        if not self.is_built:
            return {
                "total_chunks": 0,
                "is_built": False,
            }

        # Conteggi per metadata
        categories: dict[str, int] = {}
        chunk_types: dict[str, int] = {}
        years: dict[int, int] = {}
        paper_ids: set[str] = set()

        for meta in self._chunks_metadata.values():
            cat = meta.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

            ct = meta.get("chunk_type", "unknown")
            chunk_types[ct] = chunk_types.get(ct, 0) + 1

            yr = meta.get("year", 0)
            if yr:
                years[yr] = years.get(yr, 0) + 1

            pid = meta.get("paper_id", "")
            if pid:
                paper_ids.add(pid)

        # Token stats
        token_counts = [len(tokens) for tokens in self._corpus_tokens]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0

        return {
            "total_chunks": len(self._chunk_ids),
            "is_built": True,
            "unique_papers": len(paper_ids),
            "categories": dict(sorted(categories.items())),
            "chunk_types": dict(sorted(chunk_types.items())),
            "years": dict(sorted(years.items())),
            "avg_tokens_per_doc": round(avg_tokens, 1),
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "index_path": str(self.index_path),
        }

    def has_paper(self, bibtex_key: str) -> bool:
        """Verifica se un paper (per bibtex_key) è nell'indice.

        Args:
            bibtex_key: Chiave BibTeX del paper.

        Returns:
            True se almeno un chunk del paper è indicizzato.
        """
        return any(
            meta.get("bibtex_key") == bibtex_key
            for meta in self._chunks_metadata.values()
        )

    def has_paper_id(self, paper_id: str) -> bool:
        """Verifica se un paper (per paper_id) è nell'indice.

        Args:
            paper_id: ID del paper.

        Returns:
            True se almeno un chunk del paper è indicizzato.
        """
        return any(
            meta.get("paper_id") == paper_id
            for meta in self._chunks_metadata.values()
        )

    def get_paper_chunks(self, bibtex_key: str) -> list[dict[str, Any]]:
        """Recupera tutti i chunk di un paper.

        Args:
            bibtex_key: Chiave BibTeX del paper.

        Returns:
            Lista di dict con chunk_id, text, metadata.
        """
        results = []
        for cid, meta in self._chunks_metadata.items():
            if meta.get("bibtex_key") == bibtex_key:
                results.append({
                    "chunk_id": cid,
                    "text": self._chunks_text.get(cid, ""),
                    "metadata": meta,
                })
        return results

    def clear(self) -> None:
        """Resetta completamente l'indice in memoria.

        Non elimina il file su disco — per quello usare
        ``clear(delete_file=True)`` o rimuovere manualmente.
        """
        self._bm25 = None
        self._chunk_ids = []
        self._corpus_tokens = []
        self._chunks_metadata = {}
        self._chunks_text = {}
        logger.info("BM25 indice resettato")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Tokenizza un singolo testo usando le impostazioni dell'istanza.

        Args:
            text: Testo da tokenizzare.

        Returns:
            Lista di token.
        """
        return tokenize(
            text,
            remove_stopwords=self._remove_stopwords,
            stem=self._stem,
            min_token_length=self._min_token_length,
        )

    def _matches_filter(
        self,
        metadata: dict[str, Any],
        where_filter: dict[str, Any],
    ) -> bool:
        """Verifica se un chunk soddisfa il filtro metadata.

        Supporta filtri semplici per i campi più comuni.

        Args:
            metadata: Metadata del chunk.
            where_filter: Filtro da applicare.

        Returns:
            True se il chunk soddisfa tutti i criteri del filtro.
        """
        for key, value in where_filter.items():
            if key == "year_min":
                yr = metadata.get("year", 0)
                if not yr or yr < value:
                    return False
            elif key == "year_max":
                yr = metadata.get("year", 0)
                if not yr or yr > value:
                    return False
            elif key == "category":
                if metadata.get("category") != value:
                    return False
            elif key == "chunk_type":
                if metadata.get("chunk_type") != value:
                    return False
            elif key == "bibtex_key":
                if metadata.get("bibtex_key") != value:
                    return False
            elif key == "paper_id":
                if metadata.get("paper_id") != value:
                    return False
            elif key == "year":
                if metadata.get("year") != value:
                    return False
            else:
                # Filtro generico: match esatto
                if metadata.get(key) != value:
                    return False

        return True

    def _remove_chunk_ids(self, ids_to_remove: set[str]) -> int:
        """Rimuove un set di chunk_id e ricostruisce l'indice BM25.

        Args:
            ids_to_remove: Set di chunk_id da eliminare.

        Returns:
            Numero di chunk rimossi.
        """
        # Filtra
        new_ids = []
        new_tokens = []
        for i, cid in enumerate(self._chunk_ids):
            if cid not in ids_to_remove:
                new_ids.append(cid)
                new_tokens.append(self._corpus_tokens[i])

        removed = len(self._chunk_ids) - len(new_ids)

        self._chunk_ids = new_ids
        self._corpus_tokens = new_tokens

        # Pulisci metadata e testi
        for cid in ids_to_remove:
            self._chunks_metadata.pop(cid, None)
            self._chunks_text.pop(cid, None)

        # Rebuild BM25
        if new_tokens:
            self._bm25 = BM25Okapi(new_tokens)
        else:
            self._bm25 = None

        logger.info("BM25 rimossi %d chunk (restanti: %d)", removed, len(new_ids))
        return removed

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "built" if self.is_built else "empty"
        return f"BM25Index({status}, size={self.size}, path={self.index_path})"

    def __len__(self) -> int:
        return self.size

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks_metadata
