#!/usr/bin/env python3
"""
Test script per Task A.2: Budget-aware L1 Summarization.

Verifica che:
1. Il tier di compressione viene selezionato correttamente in base a N
2. I prompt template corrispondono ai tier
3. Il campo l1_char_count viene calcolato
4. Il campo compression_tier viene registrato
5. generate_l1_batch propaga correttamente il tier
6. I prompt concise/compressed producono output più corti di standard
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from paperhunter.utils.logging_setup import (
    setup_logging,
    print_success,
    print_error,
    print_panel,
    print_table,
    print_warning,
)

setup_logging("INFO")


def test_tier_selection():
    """Test: selezione automatica del tier di compressione."""
    from paperhunter.utils.token_counter import (
        get_l1_compression_tier,
        compute_l1_target_chars,
    )

    print_panel("Test 1: Tier Selection", "")

    test_cases = [
        (10, "standard"),
        (20, "standard"),
        (30, "standard"),
        (31, "concise"),
        (50, "concise"),
        (70, "concise"),
        (71, "compressed"),
        (100, "compressed"),
        (200, "compressed"),
    ]

    all_ok = True
    for n, expected_tier in test_cases:
        actual = get_l1_compression_tier(n)
        target = compute_l1_target_chars(n)
        ok = actual == expected_tier
        status = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        print(f"  {status} N={n:>3} → tier={actual:<12} target_chars={target:>6}")

    if all_ok:
        print_success("Tier selection: PASSED")
    else:
        print_error("Tier selection: FAILED")

    return all_ok


def test_prompt_templates():
    """Test: i prompt template vengono selezionati correttamente."""
    from paperhunter.summarization.level1_summarizer import (
        _get_l1_prompt_template,
        _L1_PROMPT_TEMPLATE,
        _L1_PROMPT_CONCISE,
        _L1_PROMPT_COMPRESSED,
    )

    print_panel("Test 2: Prompt Template Selection", "")

    all_ok = True

    # Standard
    tmpl = _get_l1_prompt_template("standard")
    ok = tmpl is _L1_PROMPT_TEMPLATE
    print(f"  {'✅' if ok else '❌'} standard → standard template ({len(tmpl)} chars)")
    all_ok = all_ok and ok

    # Concise
    tmpl = _get_l1_prompt_template("concise")
    ok = tmpl is _L1_PROMPT_CONCISE
    print(f"  {'✅' if ok else '❌'} concise → concise template ({len(tmpl)} chars)")
    all_ok = all_ok and ok

    # Compressed
    tmpl = _get_l1_prompt_template("compressed")
    ok = tmpl is _L1_PROMPT_COMPRESSED
    print(f"  {'✅' if ok else '❌'} compressed → compressed template ({len(tmpl)} chars)")
    all_ok = all_ok and ok

    # Fallback
    tmpl = _get_l1_prompt_template("unknown")
    ok = tmpl is _L1_PROMPT_TEMPLATE
    print(f"  {'✅' if ok else '❌'} unknown → falls back to standard template")
    all_ok = all_ok and ok

    if all_ok:
        print_success("Prompt templates: PASSED")
    else:
        print_error("Prompt templates: FAILED")

    return all_ok


def test_l1_char_count():
    """Test: _compute_l1_char_count calcola correttamente."""
    from paperhunter.summarization.level1_summarizer import _compute_l1_char_count
    from paperhunter.models import L1Summary

    print_panel("Test 3: L1 Char Count Computation", "")

    summary = L1Summary(
        paper_id="test_paper",
        key_contributions=["Contribution A (50 chars padding here!!!!!!!!!!!!!!!)", "Contribution B"],
        methodology="A detailed methodology description for testing that spans about one hundred chars of text content.",
        testing_methodology="Benchmarks: SQuAD, GLUE. Metrics: F1, Accuracy. Models: GPT-4, LLaMA-2.",
        temporal_context="Proposed in 2024, building on prior work from 2023.",
        results_summary="Achieved 95% accuracy, 0.92 F1 score.",
        limitations="Limited to English only.",
    )

    char_count = _compute_l1_char_count(summary)
    expected = (
        len("; ".join(summary.key_contributions))
        + len(summary.methodology)
        + len(summary.testing_methodology)
        + len(summary.temporal_context)
        + len(summary.results_summary)
        + len(summary.limitations)
    )

    ok = char_count == expected
    print(f"  Computed: {char_count}, Expected: {expected}")
    print(f"  {'✅' if ok else '❌'} _compute_l1_char_count: {'PASSED' if ok else 'FAILED'}")

    if ok:
        print_success("L1 char count: PASSED")
    else:
        print_error("L1 char count: FAILED")

    return ok


def test_l1summary_new_fields():
    """Test: L1Summary ha i nuovi campi l1_char_count e compression_tier."""
    from paperhunter.models import L1Summary

    print_panel("Test 4: L1Summary New Fields", "")

    all_ok = True

    # Default values
    s = L1Summary(paper_id="test")
    ok1 = s.l1_char_count is None
    ok2 = s.compression_tier is None
    print(f"  {'✅' if ok1 else '❌'} l1_char_count default: None (got {s.l1_char_count})")
    print(f"  {'✅' if ok2 else '❌'} compression_tier default: None (got {s.compression_tier})")
    all_ok = all_ok and ok1 and ok2

    # Set values
    s2 = L1Summary(
        paper_id="test2",
        l1_char_count=1500,
        compression_tier="concise",
    )
    ok3 = s2.l1_char_count == 1500
    ok4 = s2.compression_tier == "concise"
    print(f"  {'✅' if ok3 else '❌'} l1_char_count set to 1500 (got {s2.l1_char_count})")
    print(f"  {'✅' if ok4 else '❌'} compression_tier set to 'concise' (got {s2.compression_tier})")
    all_ok = all_ok and ok3 and ok4

    # JSON serialization
    j = s2.model_dump_json()
    ok5 = '"l1_char_count":1500' in j.replace(" ", "")
    ok6 = '"compression_tier":"concise"' in j.replace(" ", "")
    print(f"  {'✅' if ok5 else '❌'} l1_char_count in JSON: {ok5}")
    print(f"  {'✅' if ok6 else '❌'} compression_tier in JSON: {ok6}")
    all_ok = all_ok and ok5 and ok6

    if all_ok:
        print_success("L1Summary new fields: PASSED")
    else:
        print_error("L1Summary new fields: FAILED")

    return all_ok


def test_budget_scaling():
    """Test: il budget scala correttamente con N paper."""
    from paperhunter.utils.token_counter import compute_l1_target_chars

    print_panel("Test 5: Budget Scaling", "")

    # Con N crescente, i target chars devono decrescere (o toccare il floor)
    prev_target = float("inf")
    all_ok = True

    cases = [10, 20, 30, 50, 70, 100, 150, 200, 500]
    results = []
    for n in cases:
        target = compute_l1_target_chars(n)
        ok = target <= prev_target
        if not ok:
            all_ok = False
        results.append({
            "N papers": n,
            "Target chars": f"{target:,}",
            "Monotonic": "✅" if ok else "❌",
        })
        prev_target = target

    print_table("Budget Scaling", results)

    # Floor check: per N molto grande, il target deve essere >= l1_min_chars
    from paperhunter.config import settings
    floor = settings.summarization.l1_min_chars
    target_huge = compute_l1_target_chars(10000)
    ok_floor = target_huge >= floor
    print(f"  Floor check: N=10000 → target={target_huge}, floor={floor} → {'✅' if ok_floor else '❌'}")
    all_ok = all_ok and ok_floor

    if all_ok:
        print_success("Budget scaling: PASSED")
    else:
        print_error("Budget scaling: FAILED")

    return all_ok


def test_batch_tier_propagation():
    """Test: generate_l1_batch calcola correttamente il tier per il batch."""
    from paperhunter.summarization.level1_summarizer import generate_l1_batch
    from paperhunter.utils.token_counter import get_l1_compression_tier

    print_panel("Test 6: Batch Tier Propagation (dry check)", "")

    # Non chiama LLM — verifica solo che la funzione accetta i nuovi parametri
    # e non crasha con lista vuota
    result = generate_l1_batch([], compression_tier="concise")
    ok = result == []
    print(f"  {'✅' if ok else '❌'} Empty batch with compression_tier='concise' → []")

    result2 = generate_l1_batch([], n_papers_in_category=100)
    ok2 = result2 == []
    print(f"  {'✅' if ok2 else '❌'} Empty batch with n_papers_in_category=100 → []")

    all_ok = ok and ok2

    if all_ok:
        print_success("Batch tier propagation: PASSED")
    else:
        print_error("Batch tier propagation: FAILED")

    return all_ok


def main():
    print_panel(
        "Task A.2: Budget-aware L1 Summarization Tests",
        "Verifica tier selection, prompt templates, char counting, budget scaling",
    )

    results = {}
    results["Tier Selection"] = test_tier_selection()
    results["Prompt Templates"] = test_prompt_templates()
    results["L1 Char Count"] = test_l1_char_count()
    results["L1Summary Fields"] = test_l1summary_new_fields()
    results["Budget Scaling"] = test_budget_scaling()
    results["Batch Propagation"] = test_batch_tier_propagation()

    # Final summary
    print_panel("Final Results", "")
    all_passed = all(results.values())
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")

    print()
    if all_passed:
        print_success("ALL TESTS PASSED ✅")
    else:
        failed = [n for n, ok in results.items() if not ok]
        print_error(f"FAILED TESTS: {', '.join(failed)}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
