"""
Modulo Indexing — Vector store, BM25 index, hybrid retrieval, reranking.

Componenti:
- ``qdrant_store.QdrantStore`` — Storage vettoriale Qdrant (locale)
- ``bm25_index.BM25Index`` — Indice BM25 keyword-based
- ``bm25_index.tokenize`` — Tokenizer per BM25 (utility pubblica)
- ``hybrid_retriever.HybridRetriever`` — Retrieval ibrido (vector + BM25 + RRF)
- ``hybrid_retriever.RetrievedChunk`` — Risultato di retrieval con score e provenienza
- ``hybrid_retriever.RetrievalMethod`` — Enum metodo di retrieval
- ``hybrid_retriever.reciprocal_rank_fusion`` — Funzione RRF standalone
- ``reranker.GeminiReranker`` — Reranking con Gemini Flash (cross-encoder)
- ``reranker.rerank`` — Funzione standalone di reranking
- ``reranker.rerank_with_details`` — Reranking con statistiche dettagliate
- ``reranker.RerankResult`` — Risultato di reranking con metadati
"""

from paperhunter.indexing.bm25_index import BM25Index, tokenize
from paperhunter.indexing.hybrid_retriever import (
    HybridRetriever,
    RetrievalMethod,
    RetrievedChunk,
    reciprocal_rank_fusion,
)
from paperhunter.indexing.qdrant_store import QdrantStore
from paperhunter.indexing.reranker import (
    GeminiReranker,
    RerankResult,
    rerank,
    rerank_with_details,
)

__all__ = [
    "BM25Index",
    "GeminiReranker",
    "HybridRetriever",
    "QdrantStore",
    "RerankResult",
    "RetrievalMethod",
    "RetrievedChunk",
    "reciprocal_rank_fusion",
    "rerank",
    "rerank_with_details",
    "tokenize",
]
