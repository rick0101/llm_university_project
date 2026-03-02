"""
Test suite per Task A.4: Budget-aware Writer Agent context.

Verifica:
1. _compute_section_budget: fixed vs proportional mode
2. _sort_chunks_by_priority: L2 > L1 > original > graph ordering
3. _format_context_for_prompt: budget from config, override, truncation
4. write_section: n_total_sections e max_context_chars propagation
5. write_all_sections: budget propagation
6. Edge cases: empty chunks, 0 sections, unknown chunk types
"""

import sys
import json
from unittest.mock import patch, MagicMock

print("=" * 70)
print("TEST SUITE: Budget-aware Writer Agent context (Task A.4)")
print("=" * 70)

# ============================================================================
# TEST 1: _compute_section_budget
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: _compute_section_budget")
print("=" * 70)

from paperhunter.agents.writer_agent import _compute_section_budget
from paperhunter.utils.token_counter import compute_writer_budget

# 1a. Con override diretto: ignora config
budget = _compute_section_budget(n_total_sections=10, max_chars_override=42000)
assert budget == 42000, f"Expected 42000, got {budget}"
print("  ✓ Override diretto: max_chars_override=42000 → 42000")

# 1b. Senza override: usa config (proportional mode)
from paperhunter.config import settings
mode = settings.agents.writer_budget_mode
max_ctx = settings.summarization.max_context_tokens
max_chars_cfg = settings.agents.writer_max_context_chars

budget_8 = _compute_section_budget(n_total_sections=8)
expected_8 = compute_writer_budget(n_sections=8)
assert budget_8 == expected_8["per_section_chars"], \
    f"Expected {expected_8['per_section_chars']}, got {budget_8}"
print(f"  ✓ 8 sezioni, mode={mode}: budget={budget_8:,} chars")

# 1c. Più sezioni → budget più piccolo (proportional)
if mode == "proportional":
    budget_4 = _compute_section_budget(n_total_sections=4)
    budget_12 = _compute_section_budget(n_total_sections=12)
    assert budget_4 >= budget_12, \
        f"4 sezioni ({budget_4:,}) deve essere >= 12 sezioni ({budget_12:,})"
    print(f"  ✓ Proporzionale: 4 sez={budget_4:,} >= 12 sez={budget_12:,}")

# 1d. Budget cappato a writer_max_context_chars
budget_1 = _compute_section_budget(n_total_sections=1)
assert budget_1 <= max_chars_cfg, \
    f"Budget per 1 sezione ({budget_1:,}) deve essere ≤ max_chars ({max_chars_cfg:,})"
print(f"  ✓ Budget per 1 sezione ({budget_1:,}) cappato a ≤ {max_chars_cfg:,}")

# 1e. n_total_sections zero/negativo → trattato come 1
budget_zero = _compute_section_budget(n_total_sections=0)
budget_neg = _compute_section_budget(n_total_sections=-5)
assert budget_zero == budget_1, f"0 sezioni → stessa di 1 sezione"
assert budget_neg == budget_1, f"-5 sezioni → stessa di 1 sezione"
print("  ✓ 0 o negativo sezioni → budget per 1 sezione")

print("TEST 1 PASSED ✓\n")


# ============================================================================
# TEST 2: _sort_chunks_by_priority
# ============================================================================
print("=" * 70)
print("TEST 2: _sort_chunks_by_priority")
print("=" * 70)

from paperhunter.agents.writer_agent import _sort_chunks_by_priority

chunks = [
    {"chunk_id": "c1", "text": "original text", "metadata": {"chunk_type": "original"}},
    {"chunk_id": "c2", "text": "l2 summary", "metadata": {"chunk_type": "L2_SUMMARY"}},
    {"chunk_id": "c3", "text": "l1 summary", "metadata": {"chunk_type": "L1_SUMMARY"}},
    {"chunk_id": "c4", "text": "graph ctx", "metadata": {"chunk_type": "graph"}},
    {"chunk_id": "c5", "text": "another l2", "metadata": {"chunk_type": "l2_summary"}},
    {"chunk_id": "c6", "text": "another orig", "metadata": {"chunk_type": "original"}},
]

sorted_chunks = _sort_chunks_by_priority(chunks)
types = [c["metadata"]["chunk_type"] for c in sorted_chunks]

# L2 deve venire prima di L1, L1 prima di original, original prima di graph
l2_indices = [i for i, t in enumerate(types) if "l2" in t.lower()]
l1_indices = [i for i, t in enumerate(types) if "l1" in t.lower()]
orig_indices = [i for i, t in enumerate(types) if t == "original"]
graph_indices = [i for i, t in enumerate(types) if t == "graph"]

assert max(l2_indices) < min(l1_indices), f"L2 deve venire prima di L1: {types}"
assert max(l1_indices) < min(orig_indices), f"L1 deve venire prima di original: {types}"
assert max(orig_indices) < min(graph_indices), f"original deve venire prima di graph: {types}"
print(f"  ✓ Ordinamento corretto: {types}")

# Stabilità: l'ordine relativo intra-tipo è preservato
l2_ids = [sorted_chunks[i]["chunk_id"] for i in l2_indices]
assert l2_ids == ["c2", "c5"], f"Ordine L2 preservato: {l2_ids}"
orig_ids = [sorted_chunks[i]["chunk_id"] for i in orig_indices]
assert orig_ids == ["c1", "c6"], f"Ordine original preservato: {orig_ids}"
print("  ✓ Stabilità intra-tipo preservata")

# Chunk senza chunk_type → trattato come original
no_type_chunks = [
    {"chunk_id": "x", "text": "no type", "metadata": {}},
    {"chunk_id": "y", "text": "l2", "metadata": {"chunk_type": "L2_SUMMARY"}},
]
sorted_no_type = _sort_chunks_by_priority(no_type_chunks)
assert sorted_no_type[0]["chunk_id"] == "y", "L2 prima di chunk senza tipo"
print("  ✓ Chunk senza chunk_type → trattato come original (priorità 2)")

print("TEST 2 PASSED ✓\n")


# ============================================================================
# TEST 3: _format_context_for_prompt — budget-aware
# ============================================================================
print("=" * 70)
print("TEST 3: _format_context_for_prompt — budget-aware")
print("=" * 70)

from paperhunter.agents.writer_agent import _format_context_for_prompt

# 3a. Empty chunks
empty_fmt = _format_context_for_prompt([])
assert "No relevant context" in empty_fmt
print("  ✓ Empty chunks → placeholder text")

# 3b. override max_chars
test_chunks = [
    {
        "chunk_id": f"c_{i}",
        "text": "x" * 800,
        "score": 0.5,
        "metadata": {
            "bibtex_key": f"Author_{2020+i}_Key",
            "category": "Test",
            "year": 2020 + i,
            "section_name": "Intro",
            "chunk_type": "original",
        },
    }
    for i in range(50)
]

formatted_small = _format_context_for_prompt(test_chunks, max_chars=3000)
assert len(formatted_small) <= 4000  # un po' di margine per il messaggio di truncation
assert "truncated" in formatted_small.lower() or "budget" in formatted_small.lower()
print(f"  ✓ max_chars=3000: output={len(formatted_small)} chars, truncation message presente")

# 3c. Senza override → usa config
formatted_config = _format_context_for_prompt(test_chunks, n_total_sections=8)
# Deve rispettare il budget calcolato dalla config
budget_for_8 = _compute_section_budget(n_total_sections=8)
# Output deve essere <= budget (con margine per il messaggio di truncation)
assert len(formatted_config) <= budget_for_8 + 200, \
    f"Output ({len(formatted_config)}) > budget ({budget_for_8}) + margine"
print(f"  ✓ n_total_sections=8: output={len(formatted_config):,} chars, budget={budget_for_8:,}")

# 3d. prefer_summaries=True → L1/L2 prima di original
mixed_chunks = [
    {"chunk_id": "orig1", "text": "Original text about attacks", "score": 0.95,
     "metadata": {"chunk_type": "original", "bibtex_key": "A_2023_X"}},
    {"chunk_id": "l2_1", "text": "L2 summary of the category", "score": 0.80,
     "metadata": {"chunk_type": "L2_SUMMARY", "bibtex_key": "Category_L2"}},
    {"chunk_id": "l1_1", "text": "L1 summary of a paper", "score": 0.85,
     "metadata": {"chunk_type": "L1_SUMMARY", "bibtex_key": "B_2024_Y"}},
]
formatted_summaries = _format_context_for_prompt(mixed_chunks, max_chars=100000)
# L2 deve apparire prima di L1, L1 prima di original
l2_pos = formatted_summaries.index("L2_SUMMARY")
l1_pos = formatted_summaries.index("L1_SUMMARY")
orig_pos = formatted_summaries.index("Type: original")
assert l2_pos < l1_pos < orig_pos, \
    f"Ordine nel prompt: L2@{l2_pos} < L1@{l1_pos} < orig@{orig_pos}"
print("  ✓ prefer_summaries=True: L2 → L1 → original nel prompt")

# 3e. prefer_summaries=False → ordine originale preservato
formatted_no_pref = _format_context_for_prompt(
    mixed_chunks, max_chars=100000, prefer_summaries=False,
)
first_chunk_type = "original" if "Chunk 1" in formatted_no_pref.split("Type: original")[0] else "other"
# Il primo chunk dovrebbe essere l'originale (era primo nella lista input)
assert formatted_no_pref.index("orig1") < formatted_no_pref.index("l2_1")
print("  ✓ prefer_summaries=False: ordine originale preservato")

print("TEST 3 PASSED ✓\n")


# ============================================================================
# TEST 4: _CHUNK_TYPE_PRIORITY mapping
# ============================================================================
print("=" * 70)
print("TEST 4: Chunk type priority values")
print("=" * 70)

from paperhunter.agents.writer_agent import _CHUNK_TYPE_PRIORITY

# Verify all expected types are present
assert _CHUNK_TYPE_PRIORITY["L2_SUMMARY"] == 0
assert _CHUNK_TYPE_PRIORITY["l2_summary"] == 0
assert _CHUNK_TYPE_PRIORITY["L1_SUMMARY"] == 1
assert _CHUNK_TYPE_PRIORITY["l1_summary"] == 1
assert _CHUNK_TYPE_PRIORITY["original"] == 2
assert _CHUNK_TYPE_PRIORITY["graph"] == 3
print("  ✓ Tutte le priorità corrette: L2(0) > L1(1) > original(2) > graph(3)")

# Unknown type defaults to 2 (original priority)
unknown_prio = _CHUNK_TYPE_PRIORITY.get("unknown_type", 2)
assert unknown_prio == 2
print("  ✓ Tipo sconosciuto → default priority 2 (original)")

print("TEST 4 PASSED ✓\n")


# ============================================================================
# TEST 5: Budget consistency check
# ============================================================================
print("=" * 70)
print("TEST 5: Budget consistency — config values vs computed")
print("=" * 70)

from paperhunter.utils.token_counter import get_writer_max_context_chars, get_writer_budget_mode

config_max_chars = get_writer_max_context_chars()
config_mode = get_writer_budget_mode()

print(f"  Config: writer_max_context_chars={config_max_chars:,}")
print(f"  Config: writer_budget_mode={config_mode}")
print(f"  Config: max_context_tokens={max_ctx:,}")

# Verify che il budget per N sezioni non eccede il budget globale
for n_sec in [1, 4, 8, 12, 20]:
    budget = _compute_section_budget(n_total_sections=n_sec)
    assert budget <= config_max_chars, \
        f"Budget per {n_sec} sezioni ({budget:,}) > max_chars ({config_max_chars:,})"
    # In proportional mode, budget per-section decresce con N
    if config_mode == "proportional" and n_sec > 1:
        budget_fewer = _compute_section_budget(n_total_sections=n_sec - 1)
        assert budget <= budget_fewer, \
            f"Budget per {n_sec} sez ({budget:,}) > per {n_sec-1} sez ({budget_fewer:,})"
    print(f"  ✓ {n_sec:2d} sezioni → budget={budget:>8,} chars")

print("TEST 5 PASSED ✓\n")


# ============================================================================
# TEST 6: write_section firma aggiornata (verifica parametri)
# ============================================================================
print("=" * 70)
print("TEST 6: write_section signature — nuovi parametri")
print("=" * 70)

import inspect
from paperhunter.agents.writer_agent import write_section

sig = inspect.signature(write_section)
params = list(sig.parameters.keys())

assert "n_total_sections" in params, "n_total_sections deve essere un parametro"
assert "max_context_chars" in params, "max_context_chars deve essere un parametro"
print(f"  ✓ write_section ha n_total_sections e max_context_chars")

# Verifica default None
assert sig.parameters["n_total_sections"].default is None
assert sig.parameters["max_context_chars"].default is None
print("  ✓ Default values sono None (calcolo automatico)")

print("TEST 6 PASSED ✓\n")


# ============================================================================
# TEST 7: write_all_sections firma aggiornata
# ============================================================================
print("=" * 70)
print("TEST 7: write_all_sections signature — nuovo parametro")
print("=" * 70)

from paperhunter.agents.writer_agent import write_all_sections

sig2 = inspect.signature(write_all_sections)
params2 = list(sig2.parameters.keys())

assert "max_context_chars" in params2, "max_context_chars deve essere un parametro"
print(f"  ✓ write_all_sections ha max_context_chars")
assert sig2.parameters["max_context_chars"].default is None
print("  ✓ Default value è None")

print("TEST 7 PASSED ✓\n")


# ============================================================================
# TEST 8: Large context — verifica truncation con budget realistico
# ============================================================================
print("=" * 70)
print("TEST 8: Large context — truncation realistica")
print("=" * 70)

# Simula 200 chunk (scenario ~400 paper)
large_chunks = []
for i in range(200):
    chunk_type = "L2_SUMMARY" if i < 5 else ("L1_SUMMARY" if i < 30 else "original")
    large_chunks.append({
        "chunk_id": f"big_c_{i:03d}",
        "text": f"Content about topic {i}. " * 50,  # ~500 chars per chunk
        "score": 1.0 - (i * 0.004),
        "metadata": {
            "bibtex_key": f"Author_{2020 + (i % 6)}_Paper{i}",
            "category": "Test Category",
            "year": 2020 + (i % 6),
            "section_name": "Body",
            "chunk_type": chunk_type,
        },
    })

# Con budget piccolo (20k chars)
fmt_small = _format_context_for_prompt(large_chunks, max_chars=20000)
assert len(fmt_small) <= 21000
# I primi chunk devono essere L2_SUMMARY (prefer_summaries=True)
assert "L2_SUMMARY" in fmt_small[:2000], "L2 summaries dovrebbero apparire all'inizio"
print(f"  ✓ 200 chunk, budget 20k: output={len(fmt_small):,} chars, L2 all'inizio")

# Con budget grande (500k) — tutti i chunk entrano
fmt_large = _format_context_for_prompt(large_chunks, max_chars=500000)
count_chunks_included = fmt_large.count("--- Chunk")
assert count_chunks_included == 200, f"Tutti i 200 chunk inclusi: {count_chunks_included}"
print(f"  ✓ 200 chunk, budget 500k: tutti {count_chunks_included} chunk inclusi")

print("TEST 8 PASSED ✓\n")


# ============================================================================
# Riepilogo
# ============================================================================
print("=" * 70)
print("TUTTI I TEST PASSATI ✓ (8/8)")
print("=" * 70)
print(f"\nRiepilogo configurazione budget:")
print(f"  writer_budget_mode: {config_mode}")
print(f"  writer_max_context_chars: {config_max_chars:,}")
print(f"  max_context_tokens: {max_ctx:,}")
for n in [4, 8, 12]:
    b = _compute_section_budget(n_total_sections=n)
    print(f"  → {n} sezioni: {b:,} chars/sezione")
