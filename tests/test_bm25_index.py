#!/usr/bin/env python3
"""
Test per Task 3.1: BM25Index

Verifica:
1. Tokenizzazione (stopwords, stemming, edge cases)
2. Build index da PaperChunk
3. Query con e senza filtri
4. Update incrementale
5. Delete per paper/categoria
6. Save/Load (persistenza pickle)
7. Statistiche
8. Rilevanza dei risultati su query reali
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperhunter.indexing.bm25_index import BM25Index, tokenize, _naive_stem
from paperhunter.models import PaperChunk, ChunkType


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


# =====================================================================
# Test Data
# =====================================================================

SAMPLE_CHUNKS = [
    make_chunk(
        "c001",
        "Prompt injection attacks exploit vulnerabilities in large language models by "
        "inserting malicious instructions into the input context. These attacks can bypass "
        "system prompts and cause the model to perform unintended actions, including data "
        "exfiltration and privilege escalation.",
        bibtex_key="Greshake_2023_LLM",
        category="Prompt Injection",
        year=2023,
        paper_id="p1",
    ),
    make_chunk(
        "c002",
        "Defense mechanisms against indirect prompt injection include input sanitization, "
        "output filtering, and instruction hierarchy enforcement. We evaluate these defenses "
        "on the BIPIA benchmark using attack success rate (ASR) as the primary metric.",
        bibtex_key="Yi_2023_Benchmarking",
        category="Prompt Injection",
        year=2023,
        paper_id="p2",
    ),
    make_chunk(
        "c003",
        "Backdoor attacks on large language models involve poisoning the training data with "
        "trigger patterns that activate malicious behavior at inference time. The attacker "
        "embeds hidden triggers during the fine-tuning phase.",
        bibtex_key="Li_2024_Backdoorllm",
        category="Data and Model Poisoning",
        year=2024,
        paper_id="p3",
    ),
    make_chunk(
        "c004",
        "Hallucination in large language models refers to the generation of factually incorrect "
        "or nonsensical content. HaluEval is a benchmark specifically designed to evaluate "
        "hallucination across different tasks including question answering and summarization.",
        bibtex_key="Li_2023_Halueval",
        category="LLM Hallucinations",
        year=2023,
        paper_id="p4",
    ),
    make_chunk(
        "c005",
        "Adversarial examples for multimodal language models can be crafted by optimizing "
        "perturbations in the image space that cause the model to produce targeted outputs. "
        "Feature alignment techniques provide transferable attacks across different model architectures.",
        bibtex_key="Wang_2024_Advqdet",
        category="Adversarial Attacks",
        year=2024,
        paper_id="p5",
    ),
    make_chunk(
        "c006",
        "We propose an adaptive attack framework that systematically evaluates the robustness "
        "of LLM defenses. Our approach generates adversarial prompts using gradient-based "
        "optimization and tree-of-attack techniques to bypass safety guardrails.",
        bibtex_key="Zou_2023_Universal",
        category="Prompt Injection",
        year=2023,
        paper_id="p6",
    ),
    make_chunk(
        "c007",
        "Data poisoning during instruction tuning can subtly manipulate model behavior without "
        "degrading overall performance. We show that as few as 100 poisoned examples out of "
        "100,000 training samples can introduce targeted vulnerabilities.",
        bibtex_key="Wan_2023_Poisoning",
        category="Data and Model Poisoning",
        year=2023,
        paper_id="p7",
    ),
    make_chunk(
        "c008",
        "Recent advances in chain-of-thought prompting have shown improvements in reasoning "
        "capabilities but also introduced new attack surfaces. Adversarial chain-of-thought "
        "prompts can lead the model through incorrect reasoning paths.",
        bibtex_key="Zhang_2024_CoT",
        category="Prompt Injection",
        year=2024,
        paper_id="p8",
    ),
    make_chunk(
        "c009",
        "We evaluate hallucination detection methods using the FaithDial and TruthfulQA "
        "benchmarks. Our proposed SelfCheckGPT approach achieves state-of-the-art performance "
        "in detecting factual inconsistencies without requiring external knowledge bases.",
        bibtex_key="Manakul_2023_SelfCheck",
        category="LLM Hallucinations",
        year=2023,
        paper_id="p9",
    ),
    make_chunk(
        "c010",
        "Transfer attacks exploit shared vulnerabilities across different language model "
        "architectures. We demonstrate that adversarial suffixes optimized on open-source "
        "models like Llama-2 can successfully attack closed-source models including GPT-4 "
        "and Claude with high success rates.",
        bibtex_key="Zou_2023_GCG",
        category="Adversarial Attacks",
        year=2023,
        paper_id="p10",
    ),
]


def test_tokenizer():
    """Test 1: Tokenizzazione."""
    print("\n=== Test 1: Tokenizer ===")

    # Basics
    tokens = tokenize("Prompt injection attacks on LLMs")
    assert "prompt" in tokens or "prompt" in [t for t in tokens]
    assert "injection" in tokens or "inject" in tokens  # stemmed
    assert "on" not in tokens  # stopword
    print(f"  Tokens: {tokens}")

    # Stopwords removal
    tokens_no_stop = tokenize("this is a test of the emergency system", remove_stopwords=True)
    assert "this" not in tokens_no_stop
    assert "is" not in tokens_no_stop
    assert "a" not in tokens_no_stop
    assert "the" not in tokens_no_stop
    print(f"  Stopwords removed: {tokens_no_stop}")

    # Without stemming
    tokens_no_stem = tokenize("adversarial attacks mechanisms", stem=False)
    assert "adversarial" in tokens_no_stem
    assert "attacks" in tokens_no_stem
    print(f"  No stem: {tokens_no_stem}")

    # Empty input
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("a i") == []  # troppo corti

    # Stemming check
    assert _naive_stem("attacks") == "attack"
    assert _naive_stem("defenses") == "defens"  # "es" rule applied
    assert _naive_stem("vulnerability") == "vulnerability"  # nessun suffisso matcha

    print("  ✓ Tokenizer OK")


def test_build_index():
    """Test 2: Build index."""
    print("\n=== Test 2: Build Index ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    count = idx.build_index(SAMPLE_CHUNKS, show_progress=False)

    assert count == 10
    assert idx.is_built
    assert idx.size == 10

    print(f"  Built: {count} chunks")
    print(f"  ✓ Build OK")


def test_query_basic():
    """Test 3: Query di base."""
    print("\n=== Test 3: Query Basic ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx.build_index(SAMPLE_CHUNKS, show_progress=False)

    # Query: prompt injection
    results = idx.query("prompt injection attacks", n_results=5)
    assert len(results) > 0
    assert results[0]["score"] > 0
    # Il top result dovrebbe essere relativo a prompt injection
    top_meta = results[0]["metadata"]
    print(f"  Query 'prompt injection attacks': {len(results)} results")
    print(f"    Top: {top_meta.get('bibtex_key')} (score={results[0]['score']:.3f})")
    assert top_meta.get("category") == "Prompt Injection"

    # Query: backdoor poisoning
    results2 = idx.query("backdoor poisoning training data", n_results=3)
    assert len(results2) > 0
    top_meta2 = results2[0]["metadata"]
    print(f"  Query 'backdoor poisoning': top={top_meta2.get('bibtex_key')} (score={results2[0]['score']:.3f})")
    assert top_meta2.get("category") == "Data and Model Poisoning"

    # Query: hallucination evaluation
    results3 = idx.query("hallucination evaluation benchmark", n_results=3)
    assert len(results3) > 0
    top_meta3 = results3[0]["metadata"]
    print(f"  Query 'hallucination eval': top={top_meta3.get('bibtex_key')} (score={results3[0]['score']:.3f})")
    assert top_meta3.get("category") == "LLM Hallucinations"

    # Query: adversarial transfer
    results4 = idx.query("adversarial transfer attacks across models", n_results=3)
    assert len(results4) > 0
    print(f"  Query 'adversarial transfer': top={results4[0]['metadata'].get('bibtex_key')} (score={results4[0]['score']:.3f})")

    # Query vuota
    empty = idx.query("", n_results=5)
    assert len(empty) == 0

    print("  ✓ Queries OK")


def test_query_with_filter():
    """Test 4: Query con filtri metadata."""
    print("\n=== Test 4: Query with Filter ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx.build_index(SAMPLE_CHUNKS, show_progress=False)

    # Filtro per categoria
    results = idx.query(
        "attack defense mechanisms",
        n_results=10,
        where_filter={"category": "Prompt Injection"},
    )
    for r in results:
        assert r["metadata"]["category"] == "Prompt Injection"
    print(f"  Filtro category='Prompt Injection': {len(results)} results")

    # Filtro per anno
    results_2024 = idx.query(
        "attack",
        n_results=10,
        where_filter={"year": 2024},
    )
    for r in results_2024:
        assert r["metadata"]["year"] == 2024
    print(f"  Filtro year=2024: {len(results_2024)} results")

    # Filtro year_min
    results_min = idx.query(
        "attack model",
        n_results=10,
        where_filter={"year_min": 2024},
    )
    for r in results_min:
        assert r["metadata"]["year"] >= 2024
    print(f"  Filtro year>=2024: {len(results_min)} results")

    # Filtro combinato
    results_combo = idx.query(
        "attacks",
        n_results=10,
        where_filter={"category": "Data and Model Poisoning", "year_min": 2024},
    )
    for r in results_combo:
        assert r["metadata"]["category"] == "Data and Model Poisoning"
        assert r["metadata"]["year"] >= 2024
    print(f"  Filtro combo (DMP + year>=2024): {len(results_combo)} results")

    # Filtro per bibtex_key
    results_key = idx.query(
        "injection",
        n_results=10,
        where_filter={"bibtex_key": "Greshake_2023_LLM"},
    )
    assert len(results_key) <= 1
    if results_key:
        assert results_key[0]["metadata"]["bibtex_key"] == "Greshake_2023_LLM"
    print(f"  Filtro bibtex_key: {len(results_key)} results")

    # score_threshold
    results_thresh = idx.query("attack", n_results=10, score_threshold=1.0)
    for r in results_thresh:
        assert r["score"] > 1.0
    print(f"  Score threshold > 1.0: {len(results_thresh)} results")

    print("  ✓ Filter queries OK")


def test_update():
    """Test 5: Update incrementale."""
    print("\n=== Test 5: Update ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx.build_index(SAMPLE_CHUNKS[:5], show_progress=False)
    assert idx.size == 5

    # Aggiungi 5 nuovi chunk
    new_size = idx.update(SAMPLE_CHUNKS[5:], show_progress=False)
    assert new_size == 10
    print(f"  After update: {new_size} chunks")

    # Verifica che i nuovi siano cercabili
    results = idx.query("adaptive attack framework", n_results=3)
    assert len(results) > 0
    assert results[0]["metadata"]["bibtex_key"] == "Zou_2023_Universal"
    print(f"  New chunk found: {results[0]['metadata']['bibtex_key']}")

    # Update con chunk già esistente (sovrascrittura)
    updated_chunk = make_chunk(
        "c001",  # stesso ID del primo chunk
        "UPDATED: New text about prompt injection with updated content and methodology.",
        bibtex_key="Greshake_2023_LLM",
        category="Prompt Injection",
        year=2023,
        paper_id="p1",
    )
    new_size2 = idx.update([updated_chunk], show_progress=False)
    assert new_size2 == 10  # L'indice non deve crescere
    # Il testo deve essere aggiornato
    assert idx._chunks_text["c001"].startswith("UPDATED")
    print(f"  After overwrite update: {new_size2} chunks (testo aggiornato: OK)")

    print("  ✓ Update OK")


def test_delete():
    """Test 6: Delete per paper e categoria."""
    print("\n=== Test 6: Delete ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx.build_index(SAMPLE_CHUNKS, show_progress=False)
    assert idx.size == 10

    # Delete per paper bibtex_key
    removed = idx.delete_by_paper("Greshake_2023_LLM")
    assert removed == 1
    assert idx.size == 9
    assert not idx.has_paper("Greshake_2023_LLM")
    print(f"  delete_by_paper: removed {removed}, size={idx.size}")

    # Delete per categoria
    removed_cat = idx.delete_by_category("LLM Hallucinations")
    assert removed_cat == 2
    assert idx.size == 7
    print(f"  delete_by_category: removed {removed_cat}, size={idx.size}")

    # Delete per paper_id
    removed_pid = idx.delete_by_paper_id("p3")
    assert removed_pid == 1
    assert idx.size == 6
    print(f"  delete_by_paper_id: removed {removed_pid}, size={idx.size}")

    # Delete inesistente
    removed_none = idx.delete_by_paper("NonExistent_2025_Key")
    assert removed_none == 0
    print(f"  delete non-existent: removed {removed_none}")

    print("  ✓ Delete OK")


def test_save_load():
    """Test 7: Persistenza save/load."""
    print("\n=== Test 7: Save/Load ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_bm25.pkl"

        # Build e save
        idx = BM25Index(index_path=save_path)
        idx.build_index(SAMPLE_CHUNKS, show_progress=False)
        saved = idx.save()
        assert saved.exists()
        size_kb = saved.stat().st_size / 1024
        print(f"  Saved: {saved} ({size_kb:.1f} KB)")

        # Load in nuova istanza
        idx2 = BM25Index(index_path=save_path)
        loaded = idx2.load()
        assert loaded == 10
        assert idx2.is_built
        assert idx2.size == 10
        print(f"  Loaded: {loaded} chunks")

        # Verifica che le query funzionino identicamente
        r1 = idx.query("prompt injection", n_results=3)
        r2 = idx2.query("prompt injection", n_results=3)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a["chunk_id"] == b["chunk_id"]
            assert abs(a["score"] - b["score"]) < 1e-6
        print("  Queries match after load: OK")

        # File non trovato
        idx3 = BM25Index(index_path=Path(tmpdir) / "nonexistent.pkl")
        try:
            idx3.load()
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("  FileNotFoundError on missing file: OK")

    print("  ✓ Save/Load OK")


def test_stats():
    """Test 8: Statistiche."""
    print("\n=== Test 8: Stats ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx.build_index(SAMPLE_CHUNKS, show_progress=False)
    stats = idx.get_stats()

    assert stats["total_chunks"] == 10
    assert stats["is_built"] is True
    assert stats["unique_papers"] == 10
    assert "Prompt Injection" in stats["categories"]
    assert stats["categories"]["Prompt Injection"] == 4
    assert stats["categories"]["Data and Model Poisoning"] == 2
    assert stats["categories"]["LLM Hallucinations"] == 2
    assert stats["categories"]["Adversarial Attacks"] == 2
    assert stats["avg_tokens_per_doc"] > 0

    print(f"  Total: {stats['total_chunks']}")
    print(f"  Papers: {stats['unique_papers']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Years: {stats['years']}")
    print(f"  Avg tokens/doc: {stats['avg_tokens_per_doc']}")
    print(f"  Min/Max tokens: {stats['min_tokens']}/{stats['max_tokens']}")

    # Empty index stats
    idx_empty = BM25Index()
    stats_empty = idx_empty.get_stats()
    assert stats_empty["total_chunks"] == 0
    assert stats_empty["is_built"] is False
    print("  Empty index stats: OK")

    print("  ✓ Stats OK")


def test_has_paper():
    """Test 9: has_paper e has_paper_id."""
    print("\n=== Test 9: has_paper / has_paper_id ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx.build_index(SAMPLE_CHUNKS, show_progress=False)

    assert idx.has_paper("Greshake_2023_LLM")
    assert idx.has_paper("Li_2024_Backdoorllm")
    assert not idx.has_paper("NonExistent_2025_Key")

    assert idx.has_paper_id("p1")
    assert idx.has_paper_id("p10")
    assert not idx.has_paper_id("p99")

    print("  ✓ has_paper OK")


def test_get_paper_chunks():
    """Test 10: get_paper_chunks."""
    print("\n=== Test 10: get_paper_chunks ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx.build_index(SAMPLE_CHUNKS, show_progress=False)

    chunks = idx.get_paper_chunks("Greshake_2023_LLM")
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "c001"
    assert chunks[0]["metadata"]["bibtex_key"] == "Greshake_2023_LLM"
    print(f"  Chunks for Greshake_2023_LLM: {len(chunks)}")

    empty = idx.get_paper_chunks("NonExistent_2025")
    assert len(empty) == 0
    print("  ✓ get_paper_chunks OK")


def test_clear():
    """Test 11: Clear."""
    print("\n=== Test 11: Clear ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx.build_index(SAMPLE_CHUNKS, show_progress=False)
    assert idx.size == 10

    idx.clear()
    assert idx.size == 0
    assert not idx.is_built

    print("  ✓ Clear OK")


def test_dunder():
    """Test 12: __repr__, __len__, __contains__."""
    print("\n=== Test 12: Dunder methods ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    assert "empty" in repr(idx)
    assert len(idx) == 0

    idx.build_index(SAMPLE_CHUNKS, show_progress=False)
    assert "built" in repr(idx)
    assert len(idx) == 10
    assert "c001" in idx
    assert "nonexistent" not in idx

    print(f"  repr: {repr(idx)}")
    print("  ✓ Dunder OK")


def test_query_batch():
    """Test 13: query_batch."""
    print("\n=== Test 13: Query Batch ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx.build_index(SAMPLE_CHUNKS, show_progress=False)

    queries = [
        "prompt injection defense",
        "backdoor attack training",
        "hallucination detection benchmark",
    ]
    results = idx.query_batch(queries, n_results=3)
    assert len(results) == 3
    for i, (q, r) in enumerate(zip(queries, results)):
        assert len(r) > 0
        print(f"  Query '{q}': {len(r)} results, top score={r[0]['score']:.3f}")

    print("  ✓ Query batch OK")


def test_deduplication():
    """Test 14: Deduplicazione chunk con stesso ID."""
    print("\n=== Test 14: Deduplication ===")

    # Duplica chunk
    duplicated = SAMPLE_CHUNKS + [SAMPLE_CHUNKS[0], SAMPLE_CHUNKS[1]]

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")
    count = idx.build_index(duplicated, show_progress=False)
    assert count == 10  # Solo 10 unici
    print(f"  Built from {len(duplicated)} chunks (with dups): {count} unique")

    print("  ✓ Deduplication OK")


def test_edge_cases():
    """Test 15: Edge cases."""
    print("\n=== Test 15: Edge Cases ===")

    idx = BM25Index(index_path="/tmp/test_bm25.pkl")

    # Build con lista vuota
    try:
        idx.build_index([])
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"  Empty build: ValueError OK — {e}")

    # Query senza indice
    try:
        idx.query("test")
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        print(f"  Query without build: RuntimeError OK — {e}")

    # Save senza indice
    try:
        idx.save()
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        print(f"  Save without build: RuntimeError OK — {e}")

    # Update su indice vuoto con chunk validi
    idx2 = BM25Index(index_path="/tmp/test_bm25.pkl")
    idx2._chunk_ids = []
    idx2._corpus_tokens = []
    idx2._bm25 = None
    size = idx2.update(SAMPLE_CHUNKS[:3], show_progress=False)
    assert size == 3
    print(f"  Update on empty: {size} chunks")

    print("  ✓ Edge Cases OK")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST BM25Index — Task 3.1")
    print("=" * 60)

    test_tokenizer()
    test_build_index()
    test_query_basic()
    test_query_with_filter()
    test_update()
    test_delete()
    test_save_load()
    test_stats()
    test_has_paper()
    test_get_paper_chunks()
    test_clear()
    test_dunder()
    test_query_batch()
    test_deduplication()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("TUTTI I TEST PASSATI ✓")
    print("=" * 60)
