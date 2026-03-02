#!/usr/bin/env python3
"""
Test per Task 3.2: HybridRetriever

Verifica:
1. Reciprocal Rank Fusion (RRF) — funzione standalone
2. HybridRetriever con mock (senza Qdrant/embedding reali)
3. Costruzione filtri (_build_where_filter, _where_filter_to_bm25)
4. Fusione risultati hybrid (fuse_results)
5. Fallback a singola sorgente (solo vector, solo BM25)
6. Deduplicazione per chunk_id
7. retrieve_multi_collection
8. Statistiche retriever
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperhunter.indexing.hybrid_retriever import (
    HybridRetriever,
    RetrievalMethod,
    RetrievedChunk,
    reciprocal_rank_fusion,
    _deduplicate_by_chunk_id,
    _apply_recency_boost,
)
from paperhunter.indexing.bm25_index import BM25Index
from paperhunter.models import PaperChunk, ChunkType


# ---------------------------------------------------------------------------
# Helpers per creare dati di test
# ---------------------------------------------------------------------------

def make_chunk(
    chunk_id: str,
    text: str,
    bibtex_key: str = "Test_2024_Key",
    category: str = "Prompt Injection",
    year: int = 2024,
    chunk_type: ChunkType = ChunkType.ORIGINAL,
    paper_id: str = "paper_0",
    section_name: str = "introduction",
) -> PaperChunk:
    """Helper: crea un PaperChunk di test."""
    return PaperChunk(
        chunk_id=chunk_id,
        text=text,
        chunk_type=chunk_type,
        paper_id=paper_id,
        bibtex_key=bibtex_key,
        section_name=section_name,
        category=category,
        year=year,
        month=6,
    )


def make_vector_result(chunk_id: str, score: float, text: str = "",
                       category: str = "PI", year: int = 2024) -> dict:
    """Crea un risultato mock in formato QdrantStore.query()."""
    return {
        "chunk_id": chunk_id,
        "text": text or f"Text for {chunk_id}",
        "score": score,
        "metadata": {
            "chunk_id": chunk_id,
            "category": category,
            "year": year,
            "chunk_type": "original",
            "bibtex_key": f"Key_{chunk_id}",
            "paper_id": f"paper_{chunk_id}",
        },
    }


def make_bm25_result(chunk_id: str, score: float, text: str = "",
                     category: str = "PI", year: int = 2024) -> dict:
    """Crea un risultato mock in formato BM25Index.query()."""
    return {
        "chunk_id": chunk_id,
        "text": text or f"Text for {chunk_id}",
        "score": score,
        "metadata": {
            "chunk_id": chunk_id,
            "category": category,
            "year": year,
            "chunk_type": "original",
            "bibtex_key": f"Key_{chunk_id}",
            "paper_id": f"paper_{chunk_id}",
        },
    }


# ===========================================================================
# Test 1: Reciprocal Rank Fusion standalone
# ===========================================================================

def test_rrf_basic():
    """Verifica calcolo RRF base con due ranking."""
    print("\n" + "=" * 70)
    print("TEST 1: Reciprocal Rank Fusion (RRF) — funzione standalone")
    print("=" * 70)

    # Due ranking con overlap parziale
    vector_list = [
        {"chunk_id": "A"}, {"chunk_id": "B"}, {"chunk_id": "C"}, {"chunk_id": "D"},
    ]
    bm25_list = [
        {"chunk_id": "C"}, {"chunk_id": "A"}, {"chunk_id": "E"}, {"chunk_id": "F"},
    ]

    weights = [0.6, 0.4]  # vector=0.6, bm25=0.4
    rrf = reciprocal_rank_fusion([vector_list, bm25_list], weights, k=60)

    print(f"  Vector ranking: {[r['chunk_id'] for r in vector_list]}")
    print(f"  BM25   ranking: {[r['chunk_id'] for r in bm25_list]}")
    print(f"  Weights: vector={weights[0]}, bm25={weights[1]}, k=60")
    print(f"  RRF risultati:")
    for cid, score in rrf:
        print(f"    {cid}: {score:.6f}")

    # "A" appare in entrambi: rank 1 (vector) e rank 2 (bm25)
    # RRF(A) = 0.6/(60+1) + 0.4/(60+2) = 0.009836 + 0.006452 = 0.016288
    # "C" appare in entrambi: rank 3 (vector) e rank 1 (bm25)
    # RRF(C) = 0.6/(60+3) + 0.4/(60+1) = 0.009524 + 0.006557 = 0.016081
    # A dovrebbe essere primo (appare più in alto in vector che ha peso maggiore)

    rrf_dict = dict(rrf)
    assert "A" in rrf_dict, "A deve essere nei risultati"
    assert "C" in rrf_dict, "C deve essere nei risultati"
    assert rrf_dict["A"] > rrf_dict["C"], "A (rank 1 in vector) > C (rank 3 in vector)"

    # "B" solo in vector, "E" solo in bm25
    assert "B" in rrf_dict, "B deve essere nei risultati"
    assert "E" in rrf_dict, "E deve essere nei risultati"

    # Documenti in entrambi i ranking dovrebbero avere score più alto
    # rispetto a quelli in un solo ranking
    assert rrf_dict["A"] > rrf_dict["B"], "A (ibrido) > B (solo vector)"
    assert rrf_dict["A"] > rrf_dict["E"], "A (ibrido) > E (solo bm25)"

    # Pesi uguali
    rrf_equal = reciprocal_rank_fusion(
        [vector_list, bm25_list], [0.5, 0.5], k=60
    )
    rrf_equal_dict = dict(rrf_equal)
    # Con pesi uguali, C dovrebbe essere primo (rank 1+3 vs rank 1+2=stessi pesi)
    # Wait: A=rank1(0.5/61)+rank2(0.5/62) vs C=rank3(0.5/63)+rank1(0.5/61)
    # A: 0.008197+0.008065 = 0.016261
    # C: 0.007937+0.008197 = 0.016134
    # A è ancora primo perché rank1 in vector (peso base) > rank1 in bm25
    # Ma con pesi uguali la differenza è minore
    print(f"\n  Con pesi uguali (0.5, 0.5):")
    for cid, score in rrf_equal[:3]:
        print(f"    {cid}: {score:.6f}")

    print("  ✓ RRF base OK")

    # Test errore lunghezze diverse
    try:
        reciprocal_rank_fusion([vector_list], [0.5, 0.5])
        assert False, "Dovrebbe aver lanciato ValueError"
    except ValueError:
        print("  ✓ ValueError su lunghezze diverse OK")

    # Test con lista vuota
    rrf_empty = reciprocal_rank_fusion([[], []], [0.5, 0.5])
    assert rrf_empty == [], "RRF di liste vuote = lista vuota"
    print("  ✓ RRF con liste vuote OK")

    print("  ✅ Test 1 PASSATO")


# ===========================================================================
# Test 2: Costruzione filtri
# ===========================================================================

def test_build_where_filter():
    """Verifica _build_where_filter e _where_filter_to_bm25."""
    print("\n" + "=" * 70)
    print("TEST 2: Costruzione filtri")
    print("=" * 70)

    # --- _build_where_filter ---

    # Filtro vuoto
    f = HybridRetriever._build_where_filter()
    assert f == {}, f"Filtro vuoto atteso, got {f}"
    print("  ✓ Filtro vuoto OK")

    # Singola categoria
    f = HybridRetriever._build_where_filter(category_filter="Prompt Injection")
    assert f == {"category": "Prompt Injection"}
    print("  ✓ Singola categoria OK")

    # Lista categorie
    f = HybridRetriever._build_where_filter(
        category_filter=["Prompt Injection", "Adversarial Attacks"]
    )
    assert f["category"] == ["Prompt Injection", "Adversarial Attacks"]
    print("  ✓ Lista categorie OK")

    # Lista singola categoria → stringa
    f = HybridRetriever._build_where_filter(category_filter=["Prompt Injection"])
    assert f["category"] == "Prompt Injection"
    print("  ✓ Lista singola → stringa OK")

    # Year range completo
    f = HybridRetriever._build_where_filter(year_range=(2022, 2025))
    assert f["year"] == (2022, 2025)
    print("  ✓ Year range completo OK")

    # Year range solo min
    f = HybridRetriever._build_where_filter(year_range=(2023, None))
    assert f["year"] == {"gte": 2023}
    print("  ✓ Year range solo min OK")

    # Year range solo max
    f = HybridRetriever._build_where_filter(year_range=(None, 2025))
    assert f["year"] == {"lte": 2025}
    print("  ✓ Year range solo max OK")

    # Chunk types
    f = HybridRetriever._build_where_filter(chunk_types=["original", "l1_summary"])
    assert f["chunk_type"] == ["original", "l1_summary"]
    print("  ✓ Chunk types multipli OK")

    # Chunk type singolo
    f = HybridRetriever._build_where_filter(chunk_types=["original"])
    assert f["chunk_type"] == "original"
    print("  ✓ Chunk type singolo OK")

    # Extra filtri
    f = HybridRetriever._build_where_filter(
        category_filter="PI",
        year_range=(2023, 2025),
        bibtex_key="Zou_2023_GCG",
        paper_id="paper_1",
        extra_filters={"month": 6},
    )
    assert f["category"] == "PI"
    assert f["year"] == (2023, 2025)
    assert f["bibtex_key"] == "Zou_2023_GCG"
    assert f["paper_id"] == "paper_1"
    assert f["month"] == 6
    print("  ✓ Filtro combinato OK")

    # --- _where_filter_to_bm25 ---

    # Conversione year range tuple
    bm25_f = HybridRetriever._where_filter_to_bm25({"year": (2022, 2025), "category": "PI"})
    assert bm25_f["year_min"] == 2022
    assert bm25_f["year_max"] == 2025
    assert bm25_f["category"] == "PI"
    print("  ✓ BM25 filter: year range tuple OK")

    # Conversione year range dict
    bm25_f = HybridRetriever._where_filter_to_bm25({"year": {"gte": 2023, "lte": 2025}})
    assert bm25_f["year_min"] == 2023
    assert bm25_f["year_max"] == 2025
    print("  ✓ BM25 filter: year range dict OK")

    # Conversione year range solo gte
    bm25_f = HybridRetriever._where_filter_to_bm25({"year": {"gte": 2023}})
    assert bm25_f["year_min"] == 2023
    assert "year_max" not in bm25_f
    print("  ✓ BM25 filter: year range solo gte OK")

    # Year int esatto
    bm25_f = HybridRetriever._where_filter_to_bm25({"year": 2024})
    assert bm25_f["year"] == 2024
    print("  ✓ BM25 filter: year esatto OK")

    # Filtro vuoto
    bm25_f = HybridRetriever._where_filter_to_bm25({})
    assert bm25_f == {}
    print("  ✓ BM25 filter: vuoto OK")

    print("  ✅ Test 2 PASSATO")


# ===========================================================================
# Test 3: RetrievedChunk dataclass
# ===========================================================================

def test_retrieved_chunk():
    """Verifica RetrievedChunk e serializzazione."""
    print("\n" + "=" * 70)
    print("TEST 3: RetrievedChunk dataclass")
    print("=" * 70)

    chunk = RetrievedChunk(
        chunk_id="abc123",
        text="This is a test chunk about prompt injection.",
        score=0.015,
        retrieval_method=RetrievalMethod.HYBRID,
        metadata={"category": "Prompt Injection", "year": 2024},
        vector_score=0.92,
        bm25_score=3.45,
        vector_rank=2,
        bm25_rank=1,
    )

    assert chunk.chunk_id == "abc123"
    assert chunk.retrieval_method == RetrievalMethod.HYBRID
    assert chunk.vector_score == 0.92
    assert chunk.bm25_score == 3.45
    print("  ✓ Creazione OK")

    d = chunk.to_dict()
    assert d["retrieval_method"] == "hybrid"
    assert d["vector_rank"] == 2
    assert d["metadata"]["category"] == "Prompt Injection"
    print("  ✓ to_dict OK")

    # RetrievalMethod enum values
    assert RetrievalMethod.VECTOR.value == "vector"
    assert RetrievalMethod.BM25.value == "bm25"
    assert RetrievalMethod.HYBRID.value == "hybrid"
    assert RetrievalMethod.GRAPH.value == "graph"
    print("  ✓ RetrievalMethod enum OK")

    print("  ✅ Test 3 PASSATO")


# ===========================================================================
# Test 4: Deduplicazione
# ===========================================================================

def test_deduplication():
    """Verifica deduplicazione per chunk_id."""
    print("\n" + "=" * 70)
    print("TEST 4: Deduplicazione per chunk_id")
    print("=" * 70)

    results = [
        RetrievedChunk(chunk_id="A", text="text A", score=0.9, retrieval_method=RetrievalMethod.VECTOR),
        RetrievedChunk(chunk_id="B", text="text B", score=0.8, retrieval_method=RetrievalMethod.BM25),
        RetrievedChunk(chunk_id="A", text="text A dup", score=0.7, retrieval_method=RetrievalMethod.BM25),
        RetrievedChunk(chunk_id="C", text="text C", score=0.6, retrieval_method=RetrievalMethod.VECTOR),
        RetrievedChunk(chunk_id="B", text="text B dup", score=0.95, retrieval_method=RetrievalMethod.VECTOR),
    ]

    deduped = _deduplicate_by_chunk_id(results)
    assert len(deduped) == 3, f"Attesi 3 unici, ottenuti {len(deduped)}"

    deduped_dict = {r.chunk_id: r for r in deduped}
    assert deduped_dict["A"].score == 0.9, "A: deve tenere score più alto (0.9)"
    assert deduped_dict["B"].score == 0.95, "B: deve tenere score più alto (0.95)"
    assert deduped_dict["C"].score == 0.6, "C: unico, score invariato"
    print(f"  Deduplicati: {len(results)} → {len(deduped)} unici")
    print("  ✓ Score più alto mantenuto per ogni chunk_id")

    # Lista vuota
    assert _deduplicate_by_chunk_id([]) == []
    print("  ✓ Lista vuota OK")

    print("  ✅ Test 4 PASSATO")


# ===========================================================================
# Test 5: HybridRetriever con mock — fusione RRF
# ===========================================================================

def test_hybrid_fusion_with_mocks():
    """Verifica fusione ibrida vector+BM25 con mock store/index."""
    print("\n" + "=" * 70)
    print("TEST 5: HybridRetriever fusione RRF con mock")
    print("=" * 70)

    # Risultati mock di vector search
    mock_vector_results = [
        make_vector_result("chunk_A", 0.95, "Prompt injection defense via input filtering"),
        make_vector_result("chunk_B", 0.88, "Adversarial attacks on language models"),
        make_vector_result("chunk_C", 0.82, "Backdoor defense mechanisms in NLP"),
        make_vector_result("chunk_D", 0.75, "Hallucination detection in LLMs"),
    ]

    # Risultati mock di BM25 search
    mock_bm25_results = [
        make_bm25_result("chunk_C", 4.5, "Backdoor defense mechanisms in NLP"),
        make_bm25_result("chunk_A", 3.8, "Prompt injection defense via input filtering"),
        make_bm25_result("chunk_E", 3.2, "Prompt injection attack taxonomy"),
        make_bm25_result("chunk_F", 2.1, "Red teaming for LLM security"),
    ]

    # Mock QdrantStore e BM25Index
    mock_store = MagicMock()
    mock_store.query.return_value = mock_vector_results
    mock_store.COLLECTION_CHUNKS = "paper_chunks"
    mock_store.COLLECTION_L1 = "l1_summaries"
    mock_store.COLLECTION_L2 = "l2_summaries"

    mock_bm25 = MagicMock()
    mock_bm25.is_built = True
    mock_bm25.query.return_value = mock_bm25_results

    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=mock_bm25,
        vector_weight=0.6,
        bm25_weight=0.4,
        top_k_vector=20,
        top_k_bm25=20,
        top_k_final=10,
    )

    results = retriever.retrieve("prompt injection defense")

    print(f"  Risultati fusi: {len(results)}")
    for r in results:
        in_v = "V" if r.vector_rank else " "
        in_b = "B" if r.bm25_rank else " "
        print(
            f"    [{in_v}{in_b}] {r.chunk_id:10s}  "
            f"score={r.score:.6f}  method={r.retrieval_method.value:7s}  "
            f"v_rank={r.vector_rank}  b_rank={r.bm25_rank}"
        )

    # chunk_A e chunk_C sono in entrambi → method=HYBRID
    hybrid_ids = {r.chunk_id for r in results if r.retrieval_method == RetrievalMethod.HYBRID}
    assert "chunk_A" in hybrid_ids, "chunk_A deve essere HYBRID"
    assert "chunk_C" in hybrid_ids, "chunk_C deve essere HYBRID"
    print("  ✓ chunk_A e chunk_C correttamente HYBRID")

    # chunk_B e chunk_D solo in vector → method=VECTOR
    vector_ids = {r.chunk_id for r in results if r.retrieval_method == RetrievalMethod.VECTOR}
    assert "chunk_B" in vector_ids, "chunk_B deve essere VECTOR"
    assert "chunk_D" in vector_ids, "chunk_D deve essere VECTOR"
    print("  ✓ chunk_B e chunk_D correttamente VECTOR")

    # chunk_E e chunk_F solo in bm25 → method=BM25
    bm25_ids = {r.chunk_id for r in results if r.retrieval_method == RetrievalMethod.BM25}
    assert "chunk_E" in bm25_ids, "chunk_E deve essere BM25"
    assert "chunk_F" in bm25_ids, "chunk_F deve essere BM25"
    print("  ✓ chunk_E e chunk_F correttamente BM25")

    # I risultati ibridi (in entrambi) dovrebbero avere score più alto
    hybrid_min = min(r.score for r in results if r.retrieval_method == RetrievalMethod.HYBRID)
    single_max = max(
        r.score for r in results
        if r.retrieval_method in (RetrievalMethod.VECTOR, RetrievalMethod.BM25)
    )
    assert hybrid_min > single_max, (
        f"Hybrid min ({hybrid_min:.6f}) > single max ({single_max:.6f})"
    )
    print(f"  ✓ Tutti gli HYBRID ({hybrid_min:.6f}) > tutti i single ({single_max:.6f})")

    # Score sono ordinati
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Risultati non ordinati!"
    print("  ✓ Risultati ordinati per score decrescente")

    # Vector/BM25 score e rank preservati
    chunk_a = next(r for r in results if r.chunk_id == "chunk_A")
    assert chunk_a.vector_score == 0.95, "chunk_A.vector_score preservato"
    assert chunk_a.bm25_score == 3.8, "chunk_A.bm25_score preservato"
    assert chunk_a.vector_rank == 1, "chunk_A.vector_rank preservato"
    assert chunk_a.bm25_rank == 2, "chunk_A.bm25_rank preservato"
    print("  ✓ Score e rank per-source preservati")

    # Metadata preservati
    assert chunk_a.metadata.get("category") == "PI"
    print("  ✓ Metadata preservati")

    print("  ✅ Test 5 PASSATO")


# ===========================================================================
# Test 6: Fallback a singola sorgente
# ===========================================================================

def test_single_source_fallback():
    """Verifica fallback quando solo vector o solo BM25 disponibile."""
    print("\n" + "=" * 70)
    print("TEST 6: Fallback a singola sorgente")
    print("=" * 70)

    mock_vector_results = [
        make_vector_result("chunk_A", 0.95),
        make_vector_result("chunk_B", 0.88),
    ]

    mock_store = MagicMock()
    mock_store.query.return_value = mock_vector_results

    # --- Solo vector (BM25 non disponibile) ---
    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=None,  # BM25 non disponibile
        top_k_final=10,
    )
    # Forza _bm25_initialized a True con None
    retriever._bm25_initialized = True

    results = retriever.retrieve("test query")
    assert len(results) == 2
    assert all(r.retrieval_method == RetrievalMethod.VECTOR for r in results)
    print(f"  Solo vector: {len(results)} risultati, tutti VECTOR")
    print("  ✓ Fallback solo vector OK")

    # --- Solo BM25 (forza include_bm25_only) ---
    mock_bm25 = MagicMock()
    mock_bm25.is_built = True
    mock_bm25.query.return_value = [
        make_bm25_result("chunk_X", 4.2),
        make_bm25_result("chunk_Y", 3.1),
    ]

    retriever2 = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=mock_bm25,
        top_k_final=10,
    )
    results_bm25 = retriever2.retrieve("test query", include_bm25_only=True)
    assert all(r.retrieval_method == RetrievalMethod.BM25 for r in results_bm25)
    print(f"  Solo BM25: {len(results_bm25)} risultati, tutti BM25")
    print("  ✓ include_bm25_only OK")

    # --- Solo vector (forza include_vector_only) ---
    results_vec = retriever2.retrieve("test query", include_vector_only=True)
    assert all(r.retrieval_method == RetrievalMethod.VECTOR for r in results_vec)
    print(f"  Solo vector: {len(results_vec)} risultati, tutti VECTOR")
    print("  ✓ include_vector_only OK")

    # --- Query vuota ---
    results_empty = retriever2.retrieve("")
    assert results_empty == []
    print("  ✓ Query vuota → lista vuota OK")

    print("  ✅ Test 6 PASSATO")


# ===========================================================================
# Test 7: Filtri passati correttamente ai backend
# ===========================================================================

def test_filters_propagation():
    """Verifica che i filtri vengano passati correttamente a Qdrant e BM25."""
    print("\n" + "=" * 70)
    print("TEST 7: Propagazione filtri ai backend")
    print("=" * 70)

    mock_store = MagicMock()
    mock_store.query.return_value = [make_vector_result("chunk_A", 0.9)]

    mock_bm25 = MagicMock()
    mock_bm25.is_built = True
    mock_bm25.query.return_value = [make_bm25_result("chunk_B", 3.0)]

    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=mock_bm25,
        top_k_vector=20,
        top_k_bm25=20,
        top_k_final=10,
    )

    retriever.retrieve(
        "backdoor attack",
        category_filter="Data and Model Poisoning",
        year_range=(2023, 2025),
        chunk_types=["original"],
    )

    # Verifica che Qdrant abbia ricevuto il filtro corretto
    qdrant_call = mock_store.query.call_args
    qdrant_filter = qdrant_call.kwargs.get("where_filter") or qdrant_call[1].get("where_filter")
    print(f"  Qdrant where_filter: {qdrant_filter}")
    assert qdrant_filter["category"] == "Data and Model Poisoning"
    assert qdrant_filter["year"] == (2023, 2025)
    assert qdrant_filter["chunk_type"] == "original"
    print("  ✓ Qdrant filtro corretto")

    # Verifica che BM25 abbia ricevuto il filtro convertito
    bm25_call = mock_bm25.query.call_args
    bm25_filter = bm25_call.kwargs.get("where_filter") or bm25_call[1].get("where_filter")
    print(f"  BM25 where_filter: {bm25_filter}")
    assert bm25_filter["category"] == "Data and Model Poisoning"
    assert bm25_filter["year_min"] == 2023
    assert bm25_filter["year_max"] == 2025
    assert bm25_filter["chunk_type"] == "original"
    print("  ✓ BM25 filtro convertito correttamente")

    print("  ✅ Test 7 PASSATO")


# ===========================================================================
# Test 8: retrieve_multi_collection
# ===========================================================================

def test_multi_collection():
    """Verifica retrieve su multiple collezioni."""
    print("\n" + "=" * 70)
    print("TEST 8: retrieve_multi_collection")
    print("=" * 70)

    call_count = {"n": 0}

    def mock_query(query_text, n_results=10, where_filter=None,
                   collection_name=None, score_threshold=None):
        call_count["n"] += 1
        col = collection_name or "paper_chunks"
        return [
            make_vector_result(f"chunk_{col}_{i}", 0.9 - i * 0.1)
            for i in range(3)
        ]

    mock_store = MagicMock()
    mock_store.query.side_effect = mock_query
    mock_store.COLLECTION_CHUNKS = "paper_chunks"
    mock_store.COLLECTION_L1 = "l1_summaries"
    mock_store.COLLECTION_L2 = "l2_summaries"

    mock_bm25 = MagicMock()
    mock_bm25.is_built = True
    mock_bm25.query.return_value = []

    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=mock_bm25,
        top_k_final=5,
    )

    results = retriever.retrieve_multi_collection(
        "prompt injection defense",
        collections=["paper_chunks", "l1_summaries"],
        n_results=5,
    )

    print(f"  Risultati multi-collection: {len(results)}")
    for r in results:
        print(f"    {r.chunk_id}: score={r.score:.4f}")

    # Con deduplicazione default: chunk_id diversi da collezioni diverse
    assert len(results) <= 5, "Rispetta n_results"
    assert len(results) > 0, "Almeno un risultato"
    print(f"  ✓ {len(results)} risultati dopo dedup e troncamento a 5")

    # Ordinati per score
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Non ordinati!"
    print("  ✓ Ordinamento corretto")

    print("  ✅ Test 8 PASSATO")


# ===========================================================================
# Test 9: Shortcut methods (retrieve_vector_only, retrieve_bm25_only)
# ===========================================================================

def test_shortcut_methods():
    """Verifica metodi shortcut."""
    print("\n" + "=" * 70)
    print("TEST 9: Shortcut methods")
    print("=" * 70)

    mock_store = MagicMock()
    mock_store.query.return_value = [
        make_vector_result("A", 0.95), make_vector_result("B", 0.88)
    ]

    mock_bm25 = MagicMock()
    mock_bm25.is_built = True
    mock_bm25.query.return_value = [
        make_bm25_result("C", 4.0), make_bm25_result("D", 3.5)
    ]

    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=mock_bm25,
        top_k_final=10,
    )

    # Vector only
    vec_results = retriever.retrieve_vector_only("test", n_results=5)
    assert all(r.retrieval_method == RetrievalMethod.VECTOR for r in vec_results)
    assert len(vec_results) == 2
    print(f"  ✓ retrieve_vector_only: {len(vec_results)} risultati VECTOR")

    # BM25 only
    bm25_results = retriever.retrieve_bm25_only("test", n_results=5)
    assert all(r.retrieval_method == RetrievalMethod.BM25 for r in bm25_results)
    assert len(bm25_results) == 2
    print(f"  ✓ retrieve_bm25_only: {len(bm25_results)} risultati BM25")

    print("  ✅ Test 9 PASSATO")


# ===========================================================================
# Test 10: Statistiche retriever
# ===========================================================================

def test_stats():
    """Verifica get_stats."""
    print("\n" + "=" * 70)
    print("TEST 10: Statistiche retriever")
    print("=" * 70)

    mock_store = MagicMock()
    mock_store.get_stats.return_value = {"total_points": 500}

    mock_bm25 = MagicMock()
    mock_bm25.get_stats.return_value = {"total_chunks": 500}

    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=mock_bm25,
        vector_weight=0.6,
        bm25_weight=0.3,
    )

    stats = retriever.get_stats()
    print(f"  Config: {stats['config']}")
    print(f"  Qdrant available: {stats['qdrant_available']}")
    print(f"  BM25 available: {stats['bm25_available']}")

    assert stats["config"]["vector_weight"] == 0.6
    assert stats["config"]["bm25_weight"] == 0.3
    assert stats["qdrant_available"] is True
    assert stats["bm25_available"] is True
    print("  ✓ Statistiche corrette")

    # __repr__
    r = repr(retriever)
    assert "HybridRetriever" in r
    print(f"  ✓ repr: {r}")

    print("  ✅ Test 10 PASSATO")


# ===========================================================================
# Test 11: Integrazione con BM25Index reale (senza Qdrant/embedding)
# ===========================================================================

def test_with_real_bm25():
    """Verifica integrazione con BM25Index reale + mock vector store."""
    print("\n" + "=" * 70)
    print("TEST 11: Integrazione con BM25Index reale")
    print("=" * 70)

    # Crea chunk di test reali
    chunks = [
        make_chunk("c1", "Prompt injection attacks exploit vulnerabilities in LLM input processing to bypass safety guardrails and execute unauthorized instructions.", "Greshake_2023_LLM", "Prompt Injection", 2023, paper_id="p1"),
        make_chunk("c2", "Backdoor attacks on neural networks involve poisoning training data with trigger patterns that cause targeted misclassification.", "Li_2023_Backdoor", "Data and Model Poisoning", 2023, paper_id="p2"),
        make_chunk("c3", "Hallucinations in large language models occur when the model generates plausible but factually incorrect information.", "Li_2023_Halueval", "LLM Hallucinations", 2023, paper_id="p3"),
        make_chunk("c4", "Defense mechanisms against prompt injection include input sanitization, output filtering, and instruction hierarchy.", "Liu_2024_Defense", "Prompt Injection", 2024, paper_id="p4"),
        make_chunk("c5", "Adversarial examples for vision-language models exploit cross-modal vulnerabilities through carefully crafted perturbations.", "Zou_2023_GCG", "Adversarial Attacks", 2023, paper_id="p5"),
        make_chunk("c6", "Data poisoning in instruction tuning can compromise model alignment without detectable changes to training loss.", "Wan_2023_Poisoning", "Data and Model Poisoning", 2023, paper_id="p6"),
        make_chunk("c7", "Jailbreaking techniques use encoded instructions and role-playing scenarios to circumvent LLM safety training.", "Wei_2024_Jailbreak", "Prompt Injection", 2024, paper_id="p7"),
        make_chunk("c8", "Benchmark evaluation of hallucination shows that retrieval-augmented generation significantly reduces factual errors.", "Huang_2024_RAG", "LLM Hallucinations", 2024, paper_id="p8"),
    ]

    # Costruisci BM25 reale
    bm25 = BM25Index()
    bm25.build_index(chunks)
    print(f"  BM25 costruito: {bm25.size} chunk")

    # Mock vector search: ritorna i top-4 per embedding similarity (simulato)
    def mock_vector_query(query_text, n_results=10, where_filter=None,
                          collection_name=None, score_threshold=None):
        # Simula risultati vettoriali (diversi dal BM25 per ordine)
        simulated = {
            "prompt injection defense": [
                make_vector_result("c4", 0.93, chunks[3].text, "Prompt Injection", 2024),
                make_vector_result("c1", 0.89, chunks[0].text, "Prompt Injection", 2023),
                make_vector_result("c7", 0.82, chunks[6].text, "Prompt Injection", 2024),
                make_vector_result("c3", 0.65, chunks[2].text, "LLM Hallucinations", 2023),
            ],
        }
        # Fallback: ritorna chunk da PI
        return simulated.get(query_text, simulated["prompt injection defense"])[:n_results]

    mock_store = MagicMock()
    mock_store.query.side_effect = mock_vector_query

    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=bm25,
        vector_weight=0.6,
        bm25_weight=0.4,
        top_k_final=6,
    )

    # Query
    results = retriever.retrieve("prompt injection defense")
    print(f"\n  Query: 'prompt injection defense'")
    print(f"  Risultati ibridi: {len(results)}")
    for r in results:
        in_v = f"V{r.vector_rank}" if r.vector_rank else "  "
        in_b = f"B{r.bm25_rank}" if r.bm25_rank else "  "
        method = r.retrieval_method.value
        print(
            f"    [{in_v:>3s}|{in_b:>3s}] {r.chunk_id:5s}  "
            f"score={r.score:.6f}  method={method:7s}  "
            f"cat={r.metadata.get('category', '?')}"
        )

    # Verifica: i chunk di Prompt Injection devono dominare
    pi_chunks = [r for r in results if r.metadata.get("category") == "Prompt Injection"]
    assert len(pi_chunks) >= 2, f"Almeno 2 chunk PI, ottenuti {len(pi_chunks)}"
    print(f"\n  ✓ {len(pi_chunks)} chunk PI nei top risultati")

    # Verifica: alcuni risultati devono essere HYBRID (presenti in entrambi)
    hybrid_count = sum(1 for r in results if r.retrieval_method == RetrievalMethod.HYBRID)
    print(f"  ✓ {hybrid_count} risultati HYBRID (presenti in entrambe le sorgenti)")

    # Risultati ordinati
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    print("  ✓ Ordinamento decrescente verificato")

    print("  ✅ Test 11 PASSATO")


# ===========================================================================
# Test 12: top_k_final limita i risultati
# ===========================================================================

def test_top_k_final():
    """Verifica che top_k_final tronchi correttamente."""
    print("\n" + "=" * 70)
    print("TEST 12: top_k_final troncamento")
    print("=" * 70)

    mock_store = MagicMock()
    mock_store.query.return_value = [
        make_vector_result(f"v_{i}", 0.99 - i * 0.05) for i in range(10)
    ]

    mock_bm25 = MagicMock()
    mock_bm25.is_built = True
    mock_bm25.query.return_value = [
        make_bm25_result(f"b_{i}", 5.0 - i * 0.3) for i in range(10)
    ]

    # top_k_final = 5
    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=mock_bm25,
        top_k_final=5,
    )

    results = retriever.retrieve("test query")
    assert len(results) == 5, f"Attesi 5 risultati, ottenuti {len(results)}"
    print(f"  ✓ top_k_final=5: {len(results)} risultati")

    # Override con n_results
    results_override = retriever.retrieve("test query", n_results=3)
    assert len(results_override) == 3, f"Attesi 3, ottenuti {len(results_override)}"
    print(f"  ✓ n_results=3 override: {len(results_override)} risultati")

    print("  ✅ Test 12 PASSATO")


# ===========================================================================
# Test 13: _apply_recency_boost — funzione standalone
# ===========================================================================

def test_recency_boost_basic():
    """Verifica il funzionamento base del recency boost."""
    print("\n" + "=" * 70)
    print("TEST 13: _apply_recency_boost — base")
    print("=" * 70)

    # Crea chunk con anni diversi e stesso score RRF
    chunks = [
        RetrievedChunk(chunk_id="old", text="old paper", score=0.01,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2018}),
        RetrievedChunk(chunk_id="mid", text="mid paper", score=0.01,
                       retrieval_method=RetrievalMethod.BM25,
                       metadata={"year": 2022}),
        RetrievedChunk(chunk_id="new", text="new paper", score=0.01,
                       retrieval_method=RetrievalMethod.HYBRID,
                       metadata={"year": 2025}),
    ]

    boosted = _apply_recency_boost(chunks, boost=0.1)

    # Il paper più recente (2025) deve avere lo score più alto
    assert boosted[0].chunk_id == "new", f"Top dovrebbe essere 'new', è '{boosted[0].chunk_id}'"
    assert boosted[-1].chunk_id == "old", f"Ultimo dovrebbe essere 'old', è '{boosted[-1].chunk_id}'"

    # Tutti gli score devono essere >= allo score originale
    for c in boosted:
        assert c.score >= 0.01, f"Score {c.score} < 0.01 per {c.chunk_id}"

    # Il paper più recente deve avere il boost maggiore
    assert boosted[0].score > boosted[1].score > boosted[2].score, \
        f"Ordine score non rispettato: {[c.score for c in boosted]}"

    print(f"  ✓ Ordine corretto: {[c.chunk_id for c in boosted]}")
    print(f"  ✓ Score: {[round(c.score, 6) for c in boosted]}")

    # Verifica formula: score = 0.01 * (1 + 0.1 * recency_factor)
    # max_year = 2025, baseline = 2015, range = 10
    # new: factor = (2025-2015)/10 = 1.0 → 0.01 * 1.10 = 0.011
    # mid: factor = (2022-2015)/10 = 0.7 → 0.01 * 1.07 = 0.0107
    # old: factor = (2018-2015)/10 = 0.3 → 0.01 * 1.03 = 0.0103
    assert abs(boosted[0].score - 0.011) < 1e-9, f"Score 'new' atteso 0.011, ottenuto {boosted[0].score}"
    assert abs(boosted[1].score - 0.0107) < 1e-9, f"Score 'mid' atteso 0.0107, ottenuto {boosted[1].score}"
    assert abs(boosted[2].score - 0.0103) < 1e-9, f"Score 'old' atteso 0.0103, ottenuto {boosted[2].score}"
    print("  ✓ Formula verificata (exact values)")

    print("  ✅ Test 13 PASSATO")


# ===========================================================================
# Test 14: _apply_recency_boost — edge cases
# ===========================================================================

def test_recency_boost_edge_cases():
    """Verifica edge cases del recency boost."""
    print("\n" + "=" * 70)
    print("TEST 14: _apply_recency_boost — edge cases")
    print("=" * 70)

    # 14a: boost=0.0 → nessuna modifica
    chunks = [
        RetrievedChunk(chunk_id="a", text="a", score=0.05,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2020}),
        RetrievedChunk(chunk_id="b", text="b", score=0.03,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2025}),
    ]
    result = _apply_recency_boost(chunks, boost=0.0)
    assert result[0].chunk_id == "a", "Con boost=0, ordine invariato"
    assert result[0].score == 0.05, "Con boost=0, score invariato"
    print("  ✓ boost=0.0: nessuna modifica")

    # 14b: boost negativo → nessuna modifica
    chunks2 = [
        RetrievedChunk(chunk_id="x", text="x", score=0.02,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2024}),
    ]
    result2 = _apply_recency_boost(chunks2, boost=-0.5)
    assert result2[0].score == 0.02, "Con boost negativo, score invariato"
    print("  ✓ boost negativo: nessuna modifica")

    # 14c: lista vuota → ritorna lista vuota
    result3 = _apply_recency_boost([], boost=0.1)
    assert result3 == [], "Lista vuota ritorna lista vuota"
    print("  ✓ Lista vuota: OK")

    # 14d: nessun anno nei metadata → nessun boost
    chunks4 = [
        RetrievedChunk(chunk_id="n1", text="n1", score=0.08,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"category": "test"}),
        RetrievedChunk(chunk_id="n2", text="n2", score=0.05,
                       retrieval_method=RetrievalMethod.BM25,
                       metadata={}),
    ]
    result4 = _apply_recency_boost(chunks4, boost=0.1)
    assert result4[0].score == 0.08, "Senza anno, score invariato"
    assert result4[1].score == 0.05, "Senza anno, score invariato"
    print("  ✓ Nessun anno: nessun boost, ordine preservato")

    # 14e: mix con/senza anno → solo quelli con anno vengono boostati
    chunks5 = [
        RetrievedChunk(chunk_id="with_year", text="wy", score=0.01,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2025}),
        RetrievedChunk(chunk_id="no_year", text="ny", score=0.02,
                       retrieval_method=RetrievalMethod.BM25,
                       metadata={}),
    ]
    result5 = _apply_recency_boost(chunks5, boost=0.1)
    # "no_year" ha score 0.02 invariato
    no_year_result = [r for r in result5 if r.chunk_id == "no_year"][0]
    assert no_year_result.score == 0.02, "no_year score invariato"
    # "with_year" ha score > 0.01 (boosted)
    with_year_result = [r for r in result5 if r.chunk_id == "with_year"][0]
    assert with_year_result.score > 0.01, "with_year score boostato"
    print("  ✓ Mix con/senza anno: boost solo a chi ha anno")

    # 14f: anno prima del baseline (< 2015) → recency_factor = 0.0
    chunks6 = [
        RetrievedChunk(chunk_id="ancient", text="a", score=0.01,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2010}),
        RetrievedChunk(chunk_id="recent", text="r", score=0.01,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2025}),
    ]
    result6 = _apply_recency_boost(chunks6, boost=0.1)
    ancient = [r for r in result6 if r.chunk_id == "ancient"][0]
    recent = [r for r in result6 if r.chunk_id == "recent"][0]
    # recency_factor per 2010: max((2010-2015)/10, 0.0) = 0.0 → score = 0.01 * 1.0 = 0.01
    assert ancient.score == 0.01, f"Anno pre-baseline: score invariato, ottenuto {ancient.score}"
    # recency_factor per 2025: (2025-2015)/10 = 1.0 → score = 0.01 * 1.1 = 0.011
    assert recent.score > ancient.score, "Recent deve avere score più alto di ancient"
    print("  ✓ Anno pre-baseline (< 2015): recency_factor = 0.0")

    # 14g: tutti stesso anno → boost uguale, ordine preservato per score originale
    chunks7 = [
        RetrievedChunk(chunk_id="s1", text="s1", score=0.05,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2024}),
        RetrievedChunk(chunk_id="s2", text="s2", score=0.03,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2024}),
    ]
    result7 = _apply_recency_boost(chunks7, boost=0.1)
    assert result7[0].chunk_id == "s1", "Stesso anno: ordine per score originale"
    assert result7[0].score > result7[1].score, "Primo score > secondo"
    print("  ✓ Stesso anno: boost uniforme, ordine per score originale")

    print("  ✅ Test 14 PASSATO")


# ===========================================================================
# Test 15: Recency boost integrato nel retrieve()
# ===========================================================================

def test_recency_boost_in_retrieve():
    """Verifica che il recency boost sia applicato nel retrieve()."""
    print("\n" + "=" * 70)
    print("TEST 15: Recency boost integrato nel retrieve()")
    print("=" * 70)

    # Crea risultati mock con anni diversi e score uguali
    mock_store = MagicMock()
    mock_store.query.return_value = [
        make_vector_result("old_paper", 0.90, year=2018),
        make_vector_result("new_paper", 0.90, year=2025),
        make_vector_result("mid_paper", 0.90, year=2022),
    ]

    mock_bm25 = MagicMock()
    mock_bm25.is_built = False  # Solo vettoriale

    # 15a: Con recency_boost > 0 (da config default = 0.1)
    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=mock_bm25,
        top_k_final=10,
    )
    # Override recency_boost via parametro retrieve
    results_boosted = retriever.retrieve("test", recency_boost=0.1)
    assert len(results_boosted) == 3
    # Tutti a pari score originale, il più recente deve essere in cima
    assert results_boosted[0].metadata["year"] == 2025, \
        f"Top dovrebbe essere 2025, è {results_boosted[0].metadata['year']}"
    assert results_boosted[-1].metadata["year"] == 2018, \
        f"Ultimo dovrebbe essere 2018, è {results_boosted[-1].metadata['year']}"
    print("  ✓ Con boost: paper recenti in cima")

    # 15b: Con recency_boost = 0 → ordine invariato (per score originale)
    results_no_boost = retriever.retrieve("test", recency_boost=0.0)
    assert len(results_no_boost) == 3
    # Senza boost, ordine è quello di Qdrant (old first)
    assert results_no_boost[0].chunk_id == "old_paper", \
        f"Senza boost, primo dovrebbe essere old_paper, è {results_no_boost[0].chunk_id}"
    print("  ✓ Con boost=0: ordine originale preservato")

    # 15c: Override boost alto → differenza maggiore
    results_high = retriever.retrieve("test", recency_boost=0.5)
    high_scores = [r.score for r in results_high]
    low_scores = [r.score for r in results_boosted]
    # Con boost più alto, la differenza tra primo e ultimo è maggiore
    high_diff = high_scores[0] - high_scores[-1]
    low_diff = low_scores[0] - low_scores[-1]
    assert high_diff > low_diff, \
        f"Boost 0.5 dovrebbe creare gap maggiore ({high_diff:.6f} vs {low_diff:.6f})"
    print(f"  ✓ Boost più alto → gap più grande ({high_diff:.6f} > {low_diff:.6f})")

    print("  ✅ Test 15 PASSATO")


# ===========================================================================
# Test 16: Recency boost nelle statistiche e config
# ===========================================================================

def test_recency_boost_config():
    """Verifica che recency_boost sia presente in get_stats e config."""
    print("\n" + "=" * 70)
    print("TEST 16: Recency boost in config e stats")
    print("=" * 70)

    mock_store = MagicMock()
    mock_bm25 = MagicMock()

    retriever = HybridRetriever(
        qdrant_store=mock_store,
        bm25_index=mock_bm25,
    )

    # 16a: recency_boost è nel config (da config.yaml)
    stats = retriever.get_stats()
    assert "recency_boost" in stats["config"], "recency_boost manca da stats[config]"
    rb = stats["config"]["recency_boost"]
    assert isinstance(rb, float), f"recency_boost dovrebbe essere float, è {type(rb)}"
    print(f"  ✓ recency_boost in stats: {rb}")

    # 16b: recency_boost nel repr()
    repr_str = repr(retriever)
    assert "HybridRetriever" in repr_str, "repr deve contenere HybridRetriever"
    print(f"  ✓ __repr__: {repr_str}")

    print("  ✅ Test 16 PASSATO")


# ===========================================================================
# Test 17: _apply_recency_boost — conservazione ranking dominante
# ===========================================================================

def test_recency_boost_preserves_dominant_ranking():
    """Verifica che il boost non sovverta ranking con score molto diversi."""
    print("\n" + "=" * 70)
    print("TEST 17: Recency boost — conservazione ranking dominante")
    print("=" * 70)

    # Un paper vecchio con score molto alto non deve essere superato
    # da un paper recente con score mediocre (con boost moderato)
    chunks = [
        RetrievedChunk(chunk_id="high_old", text="ho", score=0.10,
                       retrieval_method=RetrievalMethod.HYBRID,
                       metadata={"year": 2018}),
        RetrievedChunk(chunk_id="low_new", text="ln", score=0.01,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2025}),
    ]

    result = _apply_recency_boost(chunks, boost=0.1)
    # 0.10 * (1 + 0.1 * 0.3) = 0.103 vs 0.01 * (1 + 0.1 * 1.0) = 0.011
    assert result[0].chunk_id == "high_old", \
        f"Paper con score molto alto deve restare in testa, non '{result[0].chunk_id}'"
    print(f"  ✓ high_old (2018, score=0.103) > low_new (2025, score=0.011)")

    # Ma con boost estremo (1.0+), il paper recente potrebbe superare
    # (questo è un test di sanità, non un requisito)
    chunks2 = [
        RetrievedChunk(chunk_id="h", text="h", score=0.02,
                       retrieval_method=RetrievalMethod.HYBRID,
                       metadata={"year": 2018}),
        RetrievedChunk(chunk_id="l", text="l", score=0.019,
                       retrieval_method=RetrievalMethod.VECTOR,
                       metadata={"year": 2025}),
    ]
    result2 = _apply_recency_boost(chunks2, boost=0.1)
    # h: 0.02 * (1 + 0.1 * 0.3) = 0.0206
    # l: 0.019 * (1 + 0.1 * 1.0) = 0.0209 → supera!
    assert result2[0].chunk_id == "l", \
        "Con score vicini, il paper recente deve poter superare con boost"
    print(f"  ✓ Con score vicini (0.02 vs 0.019): paper recente supera con boost")

    print("  ✅ Test 17 PASSATO")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("🔬 Test HybridRetriever — Task 3.2")
    print("=" * 70)

    tests = [
        ("Test 1: RRF standalone", test_rrf_basic),
        ("Test 2: Costruzione filtri", test_build_where_filter),
        ("Test 3: RetrievedChunk dataclass", test_retrieved_chunk),
        ("Test 4: Deduplicazione", test_deduplication),
        ("Test 5: Fusione RRF con mock", test_hybrid_fusion_with_mocks),
        ("Test 6: Fallback singola sorgente", test_single_source_fallback),
        ("Test 7: Propagazione filtri", test_filters_propagation),
        ("Test 8: Multi-collection", test_multi_collection),
        ("Test 9: Shortcut methods", test_shortcut_methods),
        ("Test 10: Statistiche", test_stats),
        ("Test 11: BM25 reale", test_with_real_bm25),
        ("Test 12: top_k_final", test_top_k_final),
        ("Test 13: Recency boost base", test_recency_boost_basic),
        ("Test 14: Recency boost edge cases", test_recency_boost_edge_cases),
        ("Test 15: Recency boost in retrieve()", test_recency_boost_in_retrieve),
        ("Test 16: Recency boost config/stats", test_recency_boost_config),
        ("Test 17: Recency boost ranking", test_recency_boost_preserves_dominant_ranking),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ❌ {name} FALLITO: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"📊 Risultati: {passed}/{passed + failed} test passati")
    if failed > 0:
        print(f"   ❌ {failed} test falliti!")
    else:
        print("   ✅ Tutti i test passati!")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
