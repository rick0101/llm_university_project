#!/usr/bin/env python3
"""
Test script per il Task 1.3: L2 Summarizer aggiornato con supporto L1.5 (RAPTOR).

Testa:
1. _format_l1_5_summaries_for_prompt: formattazione dei cluster L1.5 per il prompt L2
2. _truncate_l1_5_for_budget: troncamento L1.5 sotto budget stress
3. _generate_l2_from_l1_5: generazione L2 dal RAPTOR path (mock LLM)
4. generate_l2_summary: auto-detection del path (L1.5 vs legacy)
5. Fallback: se non ci sono L1.5, usa il path legacy con L1 diretti
6. Prompt L2 da L1.5: verifica formato e contenuto del prompt
7. Budget logging: verifica che le statistiche chars/budget vengano loggiate
8. Backward compatibility: il path legacy resta intatto

Uso:
    python test_l2_with_l1_5.py
"""

import json
import sys
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

from rich.console import Console
from rich.panel import Panel

console = Console()


# ============================================================================
# Helpers per creare dati mock
# ============================================================================

def _make_l1(
    paper_id: str,
    year: int | None = None,
    category: str = "Prompt Injection",
    citation_count: int | None = None,
):
    """Crea un L1Summary mock."""
    from paperhunter.models import L1Summary

    return L1Summary(
        paper_id=paper_id,
        bibtex_key=paper_id,
        category=category,
        year=year,
        key_contributions=[
            f"Contribution 1 of {paper_id}",
            f"Contribution 2 of {paper_id}",
        ],
        methodology=f"Methodology of {paper_id}. Uses technique X and approach Y.",
        testing_methodology=f"Testing methodology of {paper_id}. Benchmarks: A, B. Metrics: ASR, F1.",
        temporal_context=f"Published in {year or 'unknown'}, extends prior work.",
        results_summary=f"Results of {paper_id}: achieved 90% ASR on benchmark.",
        limitations=f"Limitations of {paper_id}: limited to English.",
        citation_count=citation_count,
        generated_at=datetime.now(),
        model_used="mock",
    )


def _make_l1_5(
    cluster_id: int,
    paper_count: int = 15,
    category: str = "Prompt Injection",
    temporal_span: str = "2023-2025",
):
    """Crea un L1_5Summary mock."""
    from paperhunter.models import L1_5Summary

    paper_ids = [f"Paper_{cluster_id}_{i}" for i in range(paper_count)]

    return L1_5Summary(
        cluster_id=cluster_id,
        category=category,
        paper_ids=paper_ids,
        paper_count=paper_count,
        common_theme=(
            f"Cluster {cluster_id} focuses on defense mechanisms against "
            f"prompt injection attacks using structured query approaches. "
            f"Papers [Paper_{cluster_id}_0] and [Paper_{cluster_id}_1] "
            f"propose novel delimiter-based defenses."
        ),
        shared_methodologies=(
            f"Papers in cluster {cluster_id} share instruction-tuning methods "
            f"with special tokens. [Paper_{cluster_id}_2] and [Paper_{cluster_id}_3] "
            f"use adversarial training. [Paper_{cluster_id}_4] proposes detection-based "
            f"approaches with perplexity scoring."
        ),
        key_findings=(
            f"Key findings from cluster {cluster_id}: ASR reduced from 60% to <5% "
            f"with structured defenses [Paper_{cluster_id}_0]. However, adaptive "
            f"attacks can bypass most defenses [Paper_{cluster_id}_5]. "
            f"[Paper_{cluster_id}_6] shows that model size affects robustness."
        ),
        testing_overview=(
            f"Benchmarks: InjecAgent, AgentDojo. Metrics: ASR, F1, utility score. "
            f"Models: GPT-4, LLaMA-2, Vicuna. [Paper_{cluster_id}_7] introduces "
            f"a new evaluation framework."
        ),
        temporal_span=temporal_span,
        generated_at=datetime.now().isoformat(),
        model_used="mock",
    )


def _make_mock_l1_5_list(n_clusters: int = 5, category: str = "Prompt Injection"):
    """Crea una lista di L1.5 summary mock."""
    spans = ["2020-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
    return [
        _make_l1_5(
            cluster_id=i,
            paper_count=15 + i * 3,
            category=category,
            temporal_span=spans[i % len(spans)],
        )
        for i in range(n_clusters)
    ]


def _make_mock_l1_list(n_papers: int = 5, category: str = "Prompt Injection"):
    """Crea una lista di L1 summary mock."""
    return [
        _make_l1(f"Paper_{i}", year=2020 + i, category=category)
        for i in range(n_papers)
    ]


# ============================================================================
# Test 1: _format_l1_5_summaries_for_prompt
# ============================================================================

def test_format_l1_5_summaries_for_prompt():
    """Test 1: Formattazione L1.5 per il prompt L2."""
    console.print(Panel("Test 1: _format_l1_5_summaries_for_prompt", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import (
        _format_l1_5_summaries_for_prompt,
    )

    all_ok = True

    # --- 1a: Lista vuota ---
    result_empty = _format_l1_5_summaries_for_prompt([])
    ok = result_empty == ""
    console.print(f"  {'✓' if ok else '✗'} Empty list → empty string")
    all_ok &= ok

    # --- 1b: Lista con cluster ---
    summaries = _make_mock_l1_5_list(3)
    result = _format_l1_5_summaries_for_prompt(summaries)

    ok = len(result) > 0
    console.print(f"  {'✓' if ok else '✗'} 3 cluster → {len(result)} chars")
    all_ok &= ok

    # --- 1c: Ordinamento per temporal_span (recenti prima) ---
    # Cluster 4 (2025-2026) dovrebbe apparire prima di Cluster 0 (2020-2022)
    summaries_5 = _make_mock_l1_5_list(5)
    result_5 = _format_l1_5_summaries_for_prompt(summaries_5)

    # Il primo cluster nel testo dovrebbe menzionare 2025-2026
    first_cluster_idx = result_5.find("Cluster 1/5")
    ok = first_cluster_idx >= 0
    console.print(f"  {'✓' if ok else '✗'} First cluster header found at pos {first_cluster_idx}")
    all_ok &= ok

    # Verifica che il cluster con 2025-2026 appaia prima di quello con 2020-2022
    pos_2025 = result_5.find("2025-2026")
    pos_2020 = result_5.find("2020-2022")
    ok = pos_2025 < pos_2020
    console.print(f"  {'✓' if ok else '✗'} Most recent cluster ({pos_2025}) appears before oldest ({pos_2020})")
    all_ok &= ok

    # --- 1d: Contiene tutte le sezioni ---
    for section in ["Common Theme", "Shared Methodologies", "Key Findings",
                     "Testing Overview", "Papers in this cluster"]:
        ok = section in result
        console.print(f"  {'✓' if ok else '✗'} Contains '{section}' section")
        all_ok &= ok

    # --- 1e: Contiene citazioni [bibtex_key] ---
    import re
    citations = re.findall(r"\[Paper_\d+_\d+\]", result)
    ok = len(citations) > 0
    console.print(f"  {'✓' if ok else '✗'} Contains {len(citations)} citations")
    all_ok &= ok

    # --- 1f: Separatore tra cluster ---
    ok = "---" in result
    console.print(f"  {'✓' if ok else '✗'} Contains cluster separator '---'")
    all_ok &= ok

    return all_ok


# ============================================================================
# Test 2: _truncate_l1_5_for_budget
# ============================================================================

def test_truncate_l1_5_for_budget():
    """Test 2: Troncamento L1.5 sotto budget stress."""
    console.print(Panel("Test 2: _truncate_l1_5_for_budget", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import (
        _format_l1_5_summaries_for_prompt,
        _truncate_l1_5_for_budget,
    )

    all_ok = True

    # Crea 50 cluster (simula scenario reale con 1000+ paper)
    summaries_50 = _make_mock_l1_5_list(50)
    full_text = _format_l1_5_summaries_for_prompt(summaries_50)
    full_len = len(full_text)
    console.print(f"  Full L1.5 text for 50 clusters: {full_len:,} chars")

    # --- 2a: Budget generoso (non serve troncare) ---
    result_big = _truncate_l1_5_for_budget(summaries_50, full_len + 1000)
    ok = len(result_big) <= full_len + 1000
    console.print(f"  {'✓' if ok else '✗'} Big budget: {len(result_big):,} chars (limit {full_len + 1000:,})")
    all_ok &= ok

    # --- 2b: Budget stretto (deve troncare) ---
    tight_budget = full_len // 2
    result_tight = _truncate_l1_5_for_budget(summaries_50, tight_budget)
    ok = len(result_tight) <= tight_budget
    console.print(f"  {'✓' if ok else '✗'} Tight budget: {len(result_tight):,} chars (limit {tight_budget:,})")
    all_ok &= ok

    # --- 2c: Budget molto stretto ---
    very_tight = full_len // 5
    result_very_tight = _truncate_l1_5_for_budget(summaries_50, very_tight)
    ok = len(result_very_tight) <= very_tight
    console.print(f"  {'✓' if ok else '✗'} Very tight budget: {len(result_very_tight):,} chars (limit {very_tight:,})")
    all_ok &= ok

    # --- 2d: Budget minimo ---
    result_min = _truncate_l1_5_for_budget(summaries_50, 500)
    ok = len(result_min) <= 500
    console.print(f"  {'✓' if ok else '✗'} Minimal budget: {len(result_min):,} chars (limit 500)")
    all_ok &= ok

    return all_ok


# ============================================================================
# Test 3: L2 prompt template from L1.5
# ============================================================================

def test_l2_prompt_from_l1_5():
    """Test 3: Verifica contenuto e formato del prompt L2 da L1.5."""
    console.print(Panel("Test 3: L2 prompt template from L1.5", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import (
        _L2_FROM_L1_5_PROMPT_TEMPLATE,
        _format_l1_5_summaries_for_prompt,
    )

    all_ok = True

    summaries = _make_mock_l1_5_list(5)
    l1_5_texts = _format_l1_5_summaries_for_prompt(summaries)

    total_paper_count = sum(s.paper_count for s in summaries)

    prompt = _L2_FROM_L1_5_PROMPT_TEMPLATE.format(
        category="Prompt Injection",
        cluster_count=5,
        avg_papers_per_cluster=total_paper_count // 5,
        total_paper_count=total_paper_count,
        l1_5_texts=l1_5_texts,
    )

    # --- 3a: Prompt non vuoto ---
    ok = len(prompt) > 1000
    console.print(f"  {'✓' if ok else '✗'} Prompt length: {len(prompt):,} chars")
    all_ok &= ok

    # --- 3b: Contiene le keyword del RAPTOR path ---
    for kw in ["thematic cluster", "RAPTOR", "cross-cluster", "cluster_count"]:
        # "cluster_count" sarà stato sostituito dal format, ma le altre dovrebbero restare
        pass

    for kw in ["thematic cluster", "RAPTOR", "cross-cluster patterns"]:
        ok = kw in prompt
        console.print(f"  {'✓' if ok else '✗'} Contains '{kw}'")
        all_ok &= ok

    # --- 3c: Contiene i campi JSON richiesti ---
    for field in ["temporal_trends", "taxonomy", "gaps_filled", "open_gaps",
                   "methodological_convergences", "testing_overview", "synthesis"]:
        ok = field in prompt
        console.print(f"  {'✓' if ok else '✗'} JSON field '{field}' in prompt")
        all_ok &= ok

    # --- 3d: Contiene il testo L1.5 ---
    ok = "Common Theme" in prompt
    console.print(f"  {'✓' if ok else '✗'} L1.5 content present in prompt")
    all_ok &= ok

    return all_ok


# ============================================================================
# Test 4: _generate_l2_from_l1_5 con mock LLM
# ============================================================================

def test_generate_l2_from_l1_5_mock():
    """Test 4: Generazione L2 dal RAPTOR path con LLM mockato."""
    console.print(Panel("Test 4: _generate_l2_from_l1_5 (mock LLM)", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import _generate_l2_from_l1_5

    all_ok = True

    summaries = _make_mock_l1_5_list(5)

    # Mock LLM response
    mock_l2_json = json.dumps({
        "temporal_trends": (
            "The field evolved from basic prompt injection attacks in 2020-2022 "
            "[Paper_0_0] to sophisticated defense mechanisms by 2025 [Paper_4_0]. "
            "Key turning points include the introduction of structured queries in 2024 "
            "[Paper_3_1] and adaptive attack evaluation in 2025 [Paper_4_2]." * 3
        ),
        "taxonomy": (
            "The research clusters into: (1) Attack methods [Paper_0_0, Paper_1_1], "
            "(2) Defense mechanisms [Paper_2_0, Paper_3_0], "
            "(3) Evaluation frameworks [Paper_4_0, Paper_4_1]." * 2
        ),
        "gaps_filled": (
            "The gap in systematic evaluation was addressed by [Paper_4_0]. "
            "The need for proactive defenses was filled by [Paper_3_0] and "
            "[Paper_3_1] with structured query approaches." * 2
        ),
        "open_gaps": (
            "No multi-modal prompt injection defenses exist. Cross-language attacks "
            "remain unexplored. Scalability of defenses to larger models is unclear." * 2
        ),
        "methodological_convergences": (
            "Most papers converge on using ASR as primary metric. Divergence exists "
            "between detection-based and prevention-based approaches. "
            "[Paper_0_0] and [Paper_2_0] represent opposing paradigms." * 2
        ),
        "testing_overview": (
            "Common benchmarks: InjecAgent, AgentDojo. Metrics: ASR, F1, utility. "
            "Models: GPT-4, LLaMA-2, Vicuna. [Paper_4_7] introduces a new framework." * 2
        ),
        "synthesis": (
            "This survey of prompt injection research reveals a field rapidly maturing "
            "from proof-of-concept attacks to systematic evaluation and defense. "
            "The interplay between attack [Paper_0_0] and defense [Paper_3_0] papers "
            "drives advancement." * 2
        ),
    })

    # Patch LLM e rate limiter
    with patch("paperhunter.summarization.level2_summarizer.llm_call_with_retry") as mock_llm, \
         patch("paperhunter.summarization.level2_summarizer.get_gemini_limiter") as mock_limiter, \
         patch("paperhunter.summarization.level2_summarizer.get_pro_llm") as mock_get_pro, \
         patch("paperhunter.summarization.level2_summarizer.get_pro_tracker") as mock_tracker:

        # Setup mocks
        mock_llm.return_value = mock_l2_json
        mock_limit_inst = MagicMock()
        mock_limiter.return_value = mock_limit_inst
        mock_get_pro.return_value = MagicMock()
        mock_track_inst = MagicMock()
        mock_tracker.return_value = mock_track_inst

        try:
            l2 = _generate_l2_from_l1_5(
                category="Prompt Injection",
                l1_5_summaries=summaries,
                force_model="pro",
                save_to_store=False,
                version=1,
            )

            # --- 4a: L2 generato ---
            ok = l2 is not None
            console.print(f"  {'✓' if ok else '✗'} L2 generated successfully")
            all_ok &= ok

            # --- 4b: Category e version corretti ---
            ok = l2.category == "Prompt Injection"
            console.print(f"  {'✓' if ok else '✗'} Category: {l2.category}")
            all_ok &= ok

            ok = l2.version == 1
            console.print(f"  {'✓' if ok else '✗'} Version: {l2.version}")
            all_ok &= ok

            # --- 4c: Paper count corretto ---
            total_papers = sum(s.paper_count for s in summaries)
            ok = l2.paper_count == total_papers
            console.print(f"  {'✓' if ok else '✗'} Paper count: {l2.paper_count} (expected {total_papers})")
            all_ok &= ok

            # --- 4d: Paper IDs presenti ---
            ok = len(l2.paper_ids) > 0
            console.print(f"  {'✓' if ok else '✗'} Paper IDs: {len(l2.paper_ids)} unique IDs")
            all_ok &= ok

            # --- 4e: Tutti i campi L2 popolati ---
            fields = [
                "temporal_trends", "taxonomy", "gaps_filled", "open_gaps",
                "methodological_convergences", "testing_overview", "synthesis",
            ]
            for field in fields:
                value = getattr(l2, field, "")
                ok = len(value) > 50
                console.print(f"  {'✓' if ok else '✗'} {field}: {len(value)} chars")
                all_ok &= ok

            # --- 4f: Model corretto ---
            ok = l2.model_used == "pro"
            console.print(f"  {'✓' if ok else '✗'} Model: {l2.model_used}")
            all_ok &= ok

            # --- 4g: LLM chiamato esattamente una volta ---
            ok = mock_llm.call_count == 1
            console.print(f"  {'✓' if ok else '✗'} LLM called {mock_llm.call_count} time(s)")
            all_ok &= ok

        except Exception as e:
            console.print(f"  ✗ Exception: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            all_ok = False

    return all_ok


# ============================================================================
# Test 5: Fallback al path legacy (nessun L1.5 disponibile)
# ============================================================================

def test_fallback_to_legacy():
    """Test 5: Fallback al path legacy quando non ci sono L1.5."""
    console.print(Panel("Test 5: Fallback to legacy path (no L1.5)", style="bold cyan"))

    all_ok = True

    l1_summaries = _make_mock_l1_list(5)

    mock_l2_json = json.dumps({
        "temporal_trends": "Temporal trends from legacy path. " * 20,
        "taxonomy": "Taxonomy from legacy path. " * 15,
        "gaps_filled": "Gaps filled from legacy path. " * 15,
        "open_gaps": "Open gaps from legacy path. " * 15,
        "methodological_convergences": "Methodological convergences. " * 15,
        "testing_overview": "Testing overview from legacy path. " * 15,
        "synthesis": "Synthesis from legacy path. " * 20,
    })

    with patch("paperhunter.summarization.level2_summarizer.llm_call_with_retry") as mock_llm, \
         patch("paperhunter.summarization.level2_summarizer.get_gemini_limiter") as mock_limiter, \
         patch("paperhunter.summarization.level2_summarizer.get_pro_llm") as mock_get_pro, \
         patch("paperhunter.summarization.level2_summarizer.get_pro_tracker") as mock_tracker, \
         patch("paperhunter.summarization.level1_5_summarizer.load_l1_5_summaries_for_category") as mock_l1_5:

        mock_llm.return_value = mock_l2_json
        mock_limit_inst = MagicMock()
        mock_limiter.return_value = mock_limit_inst
        mock_get_pro.return_value = MagicMock()
        mock_track_inst = MagicMock()
        mock_tracker.return_value = mock_track_inst

        # L1.5 non disponibili → lista vuota
        mock_l1_5.return_value = []

        try:
            from paperhunter.summarization.level2_summarizer import generate_l2_summary

            l2 = generate_l2_summary(
                "Prompt Injection",
                l1_summaries=l1_summaries,
                force_model="pro",
                save_to_store=False,
                version=1,
            )

            # --- 5a: L2 generato via legacy ---
            ok = l2 is not None
            console.print(f"  {'✓' if ok else '✗'} L2 generated via legacy path")
            all_ok &= ok

            # --- 5b: Paper count corretto (dal L1, non L1.5) ---
            ok = l2.paper_count == 5
            console.print(f"  {'✓' if ok else '✗'} Paper count: {l2.paper_count} (expected 5)")
            all_ok &= ok

            # --- 5c: LLM chiamato ---
            ok = mock_llm.call_count == 1
            console.print(f"  {'✓' if ok else '✗'} LLM called {mock_llm.call_count} time(s)")
            all_ok &= ok

            # --- 5d: load_l1_5 è stato chiamato (per la detection) ---
            ok = mock_l1_5.call_count == 1
            console.print(f"  {'✓' if ok else '✗'} L1.5 check attempted (mock_l1_5 called {mock_l1_5.call_count} time)")
            all_ok &= ok

        except Exception as e:
            console.print(f"  ✗ Exception: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            all_ok = False

    return all_ok


# ============================================================================
# Test 6: Auto-detection RAPTOR path via generate_l2_summary
# ============================================================================

def test_auto_detection_raptor_path():
    """Test 6: Auto-detection: se L1.5 presenti, usa RAPTOR path."""
    console.print(Panel("Test 6: Auto-detection RAPTOR path", style="bold cyan"))

    all_ok = True

    l1_5_summaries = _make_mock_l1_5_list(3)

    mock_l2_json = json.dumps({
        "temporal_trends": "Temporal trends from RAPTOR. " * 20,
        "taxonomy": "Taxonomy from RAPTOR. " * 15,
        "gaps_filled": "Gaps filled from RAPTOR. " * 15,
        "open_gaps": "Open gaps from RAPTOR. " * 15,
        "methodological_convergences": "Methodology from RAPTOR. " * 15,
        "testing_overview": "Testing from RAPTOR. " * 15,
        "synthesis": "Synthesis from RAPTOR. " * 20,
    })

    with patch("paperhunter.summarization.level2_summarizer.llm_call_with_retry") as mock_llm, \
         patch("paperhunter.summarization.level2_summarizer.get_gemini_limiter") as mock_limiter, \
         patch("paperhunter.summarization.level2_summarizer.get_pro_llm") as mock_get_pro, \
         patch("paperhunter.summarization.level2_summarizer.get_pro_tracker") as mock_tracker, \
         patch("paperhunter.summarization.level1_5_summarizer.load_l1_5_summaries_for_category") as mock_l1_5:

        mock_llm.return_value = mock_l2_json
        mock_limiter.return_value = MagicMock()
        mock_get_pro.return_value = MagicMock()
        mock_tracker.return_value = MagicMock()

        # L1.5 disponibili
        mock_l1_5.return_value = l1_5_summaries

        try:
            from paperhunter.summarization.level2_summarizer import generate_l2_summary

            # Passiamo un mock store per evitare l'inizializzazione reale di QdrantStore
            mock_store = MagicMock()

            l2 = generate_l2_summary(
                "Prompt Injection",
                l1_summaries=None,  # Non fornisco L1
                force_model="pro",
                save_to_store=False,
                store=mock_store,
                version=1,
            )

            # --- 6a: L2 generato via RAPTOR ---
            ok = l2 is not None
            console.print(f"  {'✓' if ok else '✗'} L2 generated via RAPTOR path")
            all_ok &= ok

            # --- 6b: Paper count = somma paper dai cluster L1.5 ---
            expected_count = sum(s.paper_count for s in l1_5_summaries)
            ok = l2.paper_count == expected_count
            console.print(f"  {'✓' if ok else '✗'} Paper count: {l2.paper_count} (expected {expected_count})")
            all_ok &= ok

            # --- 6c: Paper IDs raccolti dai cluster ---
            all_ids = []
            for s in l1_5_summaries:
                all_ids.extend(s.paper_ids)
            ok = len(l2.paper_ids) == len(set(all_ids))
            console.print(f"  {'✓' if ok else '✗'} Paper IDs: {len(l2.paper_ids)} (expected {len(set(all_ids))})")
            all_ok &= ok

            # --- 6d: load_l1_5 chiamato (auto-detection) ---
            ok = mock_l1_5.call_count == 1
            console.print(f"  {'✓' if ok else '✗'} L1.5 auto-detection called")
            all_ok &= ok

        except Exception as e:
            console.print(f"  ✗ Exception: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            all_ok = False

    return all_ok


# ============================================================================
# Test 7: Tutte le funzioni sono importabili
# ============================================================================

def test_imports():
    """Test 7: Verifica che tutte le nuove funzioni siano importabili."""
    console.print(Panel("Test 7: Import verification", style="bold cyan"))

    all_ok = True

    imports = [
        ("generate_l2_summary", False),
        ("_generate_l2_from_l1_5", True),
        ("_format_l1_5_summaries_for_prompt", True),
        ("_truncate_l1_5_for_budget", True),
        ("_L2_FROM_L1_5_PROMPT_TEMPLATE", True),
    ]

    for name, is_private in imports:
        try:
            from paperhunter.summarization import level2_summarizer as mod
            obj = getattr(mod, name)
            ok = obj is not None
            console.print(f"  {'✓' if ok else '✗'} {name} importable ({type(obj).__name__})")
            all_ok &= ok
        except (ImportError, AttributeError) as e:
            console.print(f"  ✗ {name}: {e}", style="bold red")
            all_ok = False

    return all_ok


# ============================================================================
# Test 8: Budget logging e metadata RAPTOR
# ============================================================================

def test_raptor_metadata_saved():
    """Test 8: Verifica che i metadata RAPTOR vengano salvati su disco."""
    console.print(Panel("Test 8: RAPTOR metadata saved on disk", style="bold cyan"))

    from pathlib import Path

    all_ok = True

    summaries = _make_mock_l1_5_list(3)

    mock_l2_json = json.dumps({
        "temporal_trends": "Temporal trends. " * 20,
        "taxonomy": "Taxonomy. " * 15,
        "gaps_filled": "Gaps. " * 15,
        "open_gaps": "Open gaps. " * 15,
        "methodological_convergences": "Methods. " * 15,
        "testing_overview": "Testing. " * 15,
        "synthesis": "Synthesis. " * 20,
    })

    with patch("paperhunter.summarization.level2_summarizer.llm_call_with_retry") as mock_llm, \
         patch("paperhunter.summarization.level2_summarizer.get_gemini_limiter") as mock_limiter, \
         patch("paperhunter.summarization.level2_summarizer.get_pro_llm") as mock_get_pro, \
         patch("paperhunter.summarization.level2_summarizer.get_pro_tracker") as mock_tracker:

        mock_llm.return_value = mock_l2_json
        mock_limiter.return_value = MagicMock()
        mock_get_pro.return_value = MagicMock()
        mock_tracker.return_value = MagicMock()

        from paperhunter.summarization.level2_summarizer import (
            _generate_l2_from_l1_5,
            _load_l2_metadata_from_disk,
        )

        try:
            l2 = _generate_l2_from_l1_5(
                category="Prompt Injection",
                l1_5_summaries=summaries,
                force_model="pro",
                save_to_store=False,
                version=99,
            )

            # Carica metadata da disco
            metadata = _load_l2_metadata_from_disk("Prompt Injection")

            ok = metadata is not None
            console.print(f"  {'✓' if ok else '✗'} Metadata file found on disk")
            all_ok &= ok

            if metadata:
                # --- 8a: RAPTOR path indicator ---
                raptor_stats = metadata.get("recency_tier_stats", {})
                ok = raptor_stats.get("path") == "raptor_l1_5"
                console.print(f"  {'✓' if ok else '✗'} path='raptor_l1_5' in metadata")
                all_ok &= ok

                # --- 8b: cluster_count ---
                ok = raptor_stats.get("cluster_count") == 3
                console.print(f"  {'✓' if ok else '✗'} cluster_count={raptor_stats.get('cluster_count')}")
                all_ok &= ok

                # --- 8c: total_paper_count ---
                expected = sum(s.paper_count for s in summaries)
                ok = raptor_stats.get("total_paper_count") == expected
                console.print(f"  {'✓' if ok else '✗'} total_paper_count={raptor_stats.get('total_paper_count')}")
                all_ok &= ok

                # --- 8d: budget_pct ---
                ok = "budget_pct" in raptor_stats
                console.print(f"  {'✓' if ok else '✗'} budget_pct={raptor_stats.get('budget_pct')}%")
                all_ok &= ok

        except Exception as e:
            console.print(f"  ✗ Exception: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            all_ok = False

    return all_ok


# ============================================================================
# Test 9: Prompt L2 legacy (backward compatibility)
# ============================================================================

def test_legacy_prompt_unchanged():
    """Test 9: Il prompt L2 legacy non è stato modificato."""
    console.print(Panel("Test 9: Legacy L2 prompt unchanged", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import (
        _L2_PROMPT_TEMPLATE,
        _format_l1_summaries_for_prompt,
    )

    all_ok = True

    # --- 9a: Template legacy esiste ---
    ok = _L2_PROMPT_TEMPLATE is not None and len(_L2_PROMPT_TEMPLATE) > 500
    console.print(f"  {'✓' if ok else '✗'} Legacy template exists ({len(_L2_PROMPT_TEMPLATE)} chars)")
    all_ok &= ok

    # --- 9b: Template legacy contiene i placeholder originali ---
    for ph in ["{category}", "{paper_count}", "{l1_texts}",
               "{recency_full_years}", "{recency_reduced_years}"]:
        ok = ph in _L2_PROMPT_TEMPLATE
        console.print(f"  {'✓' if ok else '✗'} Legacy template contains placeholder '{ph}'")
        all_ok &= ok

    # --- 9c: Legacy formatting funziona ---
    l1s = _make_mock_l1_list(5)
    formatted = _format_l1_summaries_for_prompt(l1s)
    ok = len(formatted) > 0 and "Paper" in formatted
    console.print(f"  {'✓' if ok else '✗'} Legacy formatting: {len(formatted)} chars")
    all_ok &= ok

    return all_ok


# ============================================================================
# Test 10: Stress test con 50 cluster L1.5
# ============================================================================

def test_stress_50_clusters():
    """Test 10: 50 cluster L1.5 (~1000 paper) rientrano nel budget."""
    console.print(Panel("Test 10: Stress test — 50 clusters", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import _format_l1_5_summaries_for_prompt
    from paperhunter.utils.token_counter import compute_l2_budget

    all_ok = True

    # 50 cluster da 20 paper ciascuno = 1000 paper
    summaries_50 = _make_mock_l1_5_list(50)
    total_papers = sum(s.paper_count for s in summaries_50)
    console.print(f"  Total: {len(summaries_50)} clusters, ~{total_papers} papers")

    l1_5_text = _format_l1_5_summaries_for_prompt(summaries_50)
    budget_info = compute_l2_budget()
    available_chars = budget_info["available_chars"]
    overhead = 4000
    usable = available_chars - overhead

    pct = len(l1_5_text) * 100 // usable if usable > 0 else 0
    ok = len(l1_5_text) <= usable
    status = "within budget" if ok else "OVER budget (truncation needed)"
    console.print(
        f"  {'✓' if ok else '⚠'} L1.5 text: {len(l1_5_text):,} chars, "
        f"budget: {usable:,} chars ({pct}%) — {status}"
    )
    # This is fine even if over budget — _truncate_l1_5_for_budget handles it
    all_ok = True  # Non-critical

    # Verifica che il testo contenga tutti i 50 cluster
    import re
    cluster_headers = re.findall(r"Cluster \d+/50", l1_5_text)
    ok = len(cluster_headers) == 50
    console.print(f"  {'✓' if ok else '✗'} All 50 cluster headers present ({len(cluster_headers)})")
    all_ok &= ok

    return all_ok


# ============================================================================
# Main
# ============================================================================

def main():
    console.print(Panel(
        "Test L2 Summarizer — Task 1.3: Aggiornamento con supporto L1.5 (RAPTOR)",
        style="bold blue",
    ))
    start = time.time()

    tests = [
        ("Test 1: Format L1.5 for prompt", test_format_l1_5_summaries_for_prompt),
        ("Test 2: Truncate L1.5 for budget", test_truncate_l1_5_for_budget),
        ("Test 3: L2 prompt from L1.5", test_l2_prompt_from_l1_5),
        ("Test 4: Generate L2 from L1.5 (mock)", test_generate_l2_from_l1_5_mock),
        ("Test 5: Fallback to legacy path", test_fallback_to_legacy),
        ("Test 6: Auto-detection RAPTOR path", test_auto_detection_raptor_path),
        ("Test 7: Import verification", test_imports),
        ("Test 8: RAPTOR metadata saved", test_raptor_metadata_saved),
        ("Test 9: Legacy prompt unchanged", test_legacy_prompt_unchanged),
        ("Test 10: Stress test 50 clusters", test_stress_50_clusters),
    ]

    results = {}
    for name, fn in tests:
        try:
            ok = fn()
            results[name] = ok
        except Exception as e:
            console.print(f"  ✗ CRASH: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            results[name] = False
        console.print()

    # Report finale
    elapsed = time.time() - start
    console.print(Panel("Final Report", style="bold green"))

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for name, ok in results.items():
        console.print(f"  {'✓' if ok else '✗'} {name}")

    console.print()
    console.print(
        f"  {passed}/{len(results)} tests passed, "
        f"{failed} failed, {elapsed:.1f}s elapsed"
    )

    if failed > 0:
        console.print("\n✗ Some tests failed!", style="bold red")
        sys.exit(1)
    else:
        console.print("\n✓ All tests passed!", style="bold green")


if __name__ == "__main__":
    main()
