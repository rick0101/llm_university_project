"""
Vector Store - FAISS + Gemini Embeddings for paper chunks.

Uses FAISS for fast similarity search and Google's gemini-embedding-001
model for embeddings.  Both the index and metadata are persisted to disk.

The gemini-embedding-001 model is free-tier and produces
3072-dimensional embeddings, making it ideal for this use case.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from google import genai

from .chunker import Chunk

logger = logging.getLogger(__name__)

# Embedding dimension for gemini-embedding-001
_EMBED_DIM = 3072
_EMBED_MODEL = "gemini-embedding-001"
_BATCH_SIZE = 100        # Gemini allows up to 100 texts per batch request
_RATE_LIMIT_DELAY = 0.5  # seconds between batches to stay within free tier


class VectorStore:
    """FAISS vector store with Gemini embeddings."""

    _META_FILE = "store_meta.json"
    _INDEX_FILE = "store.faiss"

    def __init__(
        self,
        collection_name: str = "papers",
        persist_dir: str = "./vector_db",
        api_key: Optional[str] = None,
    ):
        """
        Args:
            collection_name: Logical name (used as file prefix).
            persist_dir: Directory for on-disk persistence.
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._index_path = self.persist_dir / f"{collection_name}_{self._INDEX_FILE}"
        self._meta_path = self.persist_dir / f"{collection_name}_{self._META_FILE}"

        # Configure Gemini client
        key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self._client = genai.Client(api_key=key)

        # Internal state
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._index: faiss.IndexFlatIP = None  # cosine sim after L2-norm

        self._load_or_create()

        logger.info(
            f"Vector store ready — collection='{collection_name}', "
            f"docs={len(self._ids)}"
        )

    # ── Persistence ──────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._ids = data["ids"]
            self._documents = data["documents"]
            self._metadatas = data["metadatas"]
            logger.info(f"Loaded existing index with {len(self._ids)} documents")
        else:
            self._index = faiss.IndexFlatIP(_EMBED_DIM)
            logger.info("Created new empty FAISS index")

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        data = {
            "ids": self._ids,
            "documents": self._documents,
            "metadatas": self._metadatas,
        }
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # ── Embedding ────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using Gemini text-embedding-004."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            result = self._client.models.embed_content(
                model=_EMBED_MODEL,
                contents=batch,
            )
            for emb in result.embeddings:
                all_embeddings.append(emb.values)

            if i + _BATCH_SIZE < len(texts):
                time.sleep(_RATE_LIMIT_DELAY)

        arr = np.array(all_embeddings, dtype="float32")
        # L2-normalise so inner-product == cosine similarity
        faiss.normalize_L2(arr)
        return arr

    # ── Ingestion ────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        """
        Add chunks to the store (skips duplicates by chunk_id).

        Returns:
            Number of newly added chunks.
        """
        if not chunks:
            return 0

        existing_ids = set(self._ids)
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already in store — nothing to add")
            return 0

        logger.info(f"Embedding {len(new_chunks)} new chunks …")
        texts = [c.text for c in new_chunks]
        embeddings = self._embed_texts(texts)

        self._index.add(embeddings)

        for c in new_chunks:
            self._ids.append(c.chunk_id)
            self._documents.append(c.text)
            self._metadatas.append(c.metadata)

        self._persist()
        logger.info(
            f"Ingestion done — {len(new_chunks)} new chunks. "
            f"Total: {len(self._ids)}"
        )
        return len(new_chunks)

    # ── Retrieval ────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Returns:
            List[dict] with keys: chunk_id, text, metadata, score.
        """
        if len(self._ids) == 0:
            return []

        q_emb = self._embed_texts([query_text])

        search_k = min(n_results * 3 if where else n_results, len(self._ids))
        scores, indices = self._index.search(q_emb, search_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadatas[idx]

            if where and not self._match_where(meta, where):
                continue

            results.append(
                {
                    "chunk_id": self._ids[idx],
                    "text": self._documents[idx],
                    "metadata": meta,
                    "score": float(score),
                }
            )
            if len(results) >= n_results:
                break

        return results

    @staticmethod
    def _match_where(meta: dict, where: dict) -> bool:
        for k, v in where.items():
            if meta.get(k) != v:
                return False
        return True

    # ── Utilities ────────────────────────────────────────────────────────

    def get_all_paper_titles(self) -> list[str]:
        return sorted({m.get("paper_title", "?") for m in self._metadatas})

    def get_collection_stats(self) -> dict:
        titles: set[str] = set()
        filenames: set[str] = set()
        for m in self._metadatas:
            titles.add(m.get("paper_title", "?"))
            filenames.add(m.get("filename", "?"))
        return {
            "total_chunks": len(self._ids),
            "total_papers": len(titles),
            "total_files": len(filenames),
            "paper_titles": sorted(titles),
        }

    def clear(self) -> None:
        self._ids.clear()
        self._documents.clear()
        self._metadatas.clear()
        self._index = faiss.IndexFlatIP(_EMBED_DIM)
        self._persist()
        logger.info("Collection cleared")

    def delete_by_filename(self, filename: str) -> None:
        """Remove all chunks originating from a file (rebuilds the index)."""
        keep = [
            i for i, m in enumerate(self._metadatas)
            if m.get("filename") != filename
        ]
        removed = len(self._ids) - len(keep)
        if removed == 0:
            return

        new_ids = [self._ids[i] for i in keep]
        new_docs = [self._documents[i] for i in keep]
        new_meta = [self._metadatas[i] for i in keep]

        if new_docs:
            embeddings = self._embed_texts(new_docs)
            new_index = faiss.IndexFlatIP(_EMBED_DIM)
            new_index.add(embeddings)
        else:
            new_index = faiss.IndexFlatIP(_EMBED_DIM)

        self._ids = new_ids
        self._documents = new_docs
        self._metadatas = new_meta
        self._index = new_index
        self._persist()
        logger.info(f"Deleted {removed} chunks from '{filename}'")
