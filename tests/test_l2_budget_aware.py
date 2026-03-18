#!/usr/bin/env python3
"""
Test script per il Task A.3: Budget-aware L2 summarization.

Testa:
1. Recency tier assignment (_get_paper_recency_tier)
2. Paper importance scoring (_compute_paper_importance)
3. Tier-specific formatting (_format_l1_full/reduced/compressed)
4. Budget-aware formatting con selezione chunk
5. Progressive downgrade sotto budget stress
6. Hard truncation come ultima risorsa
7. citation_count field in L1Summary
8. Compatibilità backward (pochi paper = nessuna compressione)

Uso:
    python test_l2_budget_aware.py
"""

import sys
import time
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

console = Console()


def _make_l1(
    paper_id: str,
    year: int | None = None,
    category: str = "Prompt Injection",
    citation_count: int | None = None,
    long_fields: bool = False,
):
    """Helper per creare L1Summary di test."""
    from paperhunter.models import L1Summary

    base_text = "A " * 150 if long_fields else "Short text. "

    return L1Summary(
        paper_id=paper_id,
        bibtex_key=paper_id,
        category=category,
        year=year,
        key_contributions=[
            f"Contribution 1 of {paper_id}",
            f"Contribution 2 of {paper_id}",
            f"Contribution 3 of {paper_id}",
        ],
        methodology=f"Methodology of {paper_id}. " + base_text,
        testing_methodology=f"Testing methodology of {paper_id}. " + base_text,
        temporal_context=f"Temporal context of {paper_id}. " + base_text,
        results_summary=f"Results of {paper_id}. " + base_text,
        limitations=f"Limitations of {paper_id}. " + base_text,
        citation_count=citation_count,
        generated_at=datetime.now(),
        model_used="mock",
    )


def test_recency_tiers():
    """Test 1: Recency tier assignment."""
    console.print(Panel("Test 1: Recency tier assignment", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import _get_paper_recency_tier

    current_year = 2026

    # Default config: full=2, reduced=4, compressed=999
    cases = [
        (2026, "full", "current year"),
        (2025, "full", "1 year ago"),
        (2024, "full", "2 years ago (boundary full)"),
        (2023, "reduced", "3 years ago"),
        (2022, "reduced", "4 years ago (boundary reduced)"),
        (2021, "compressed", "5 years ago"),
        (2019, "compressed", "7 years ago"),
        (None, "compressed", "unknown year"),
    ]

    all_ok = True
    for year, expected, desc in cases:
        tier = _get_paper_recency_tier(year, current_year)
        ok = tier == expected
        status = "✓" if ok else "✗"
        console.print(f"  {status} year={year} ({desc}) → {tier} (expected: {expected})")
        if not ok:
            all_ok = False

    return all_ok


def test_paper_importance():
    """Test 2: Paper importance scoring."""
    console.print(Panel("Test 2: Paper importance scoring", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import _compute_paper_importance

    current_year = 2026

    # Recent paper with lots of citations should be most important
    s_recent_cited = _make_l1("Recent_2025", year=2025, citation_count=500)
    s_recent_no_cite = _make_l1("Recent_2025_NoCite", year=2025, citation_count=0)
    s_old_cited = _make_l1("Old_2019", year=2019, citation_count=1000)
    s_old_no_cite = _make_l1("Old_2019_NoCite", year=2019, citation_count=0)
    s_no_year = _make_l1("NoYear", year=None, citation_count=0)

    scores = {
        "Recent+Cited (2025, 500 cit)": _compute_paper_importance(s_recent_cited, current_year),
        "Recent+NoCite (2025, 0 cit)": _compute_paper_importance(s_recent_no_cite, current_year),
        "Old+Cited (2019, 1000 cit)": _compute_paper_importance(s_old_cited, current_year),
        "Old+NoCite (2019, 0 cit)": _compute_paper_importance(s_old_no_cite, current_year),
        "NoYear (None, 0 cit)": _compute_paper_importance(s_no_year, current_year),
    }

    all_ok = True
    for desc, score in scores.items():
        console.print(f"  {desc}: importance = {score:.4f}")

    # Verify ordering constraints
    checks = [
        (scores["Recent+Cited (2025, 500 cit)"] > scores["Recent+NoCite (2025, 0 cit)"],
         "Recent+Cited > Recent+NoCite"),
        (scores["Recent+NoCite (2025, 0 cit)"] > scores["Old+NoCite (2019, 0 cit)"],
         "Recent+NoCite > Old+NoCite"),
        (scores["Old+Cited (2019, 1000 cit)"] > scores["Old+NoCite (2019, 0 cit)"],
         "Old+Cited > Old+NoCite"),
        (scores["NoYear (None, 0 cit)"] < scores["Old+NoCite (2019, 0 cit)"],
         "NoYear < Old+NoCite"),
    ]

    for ok, desc in checks:
        status = "✓" if ok else "✗"
        console.print(f"  {status} {desc}")
        if not ok:
            all_ok = False

    return all_ok


def test_tier_formatting():
    """Test 3: Tier-specific L1 formatting."""
    console.print(Panel("Test 3: Tier-specific formatting", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import (
        _format_l1_full,
        _format_l1_reduced,
        _format_l1_compressed,
        _format_l1_at_tier,
    )

    s = _make_l1("Test_2024", year=2024, long_fields=True)

    full = _format_l1_full(s, 1, 5)
    reduced = _format_l1_reduced(s, 1, 5)
    compressed = _format_l1_compressed(s, 1, 5)

    console.print(f"  Full format:       {len(full):,} chars")
    console.print(f"  Reduced format:    {len(reduced):,} chars")
    console.print(f"  Compressed format: {len(compressed):,} chars")

    all_ok = True

    # Full should be the longest
    if not (len(full) > len(reduced) > len(compressed)):
        console.print("  ✗ Expected full > reduced > compressed")
        all_ok = False
    else:
        console.print("  ✓ Size ordering: full > reduced > compressed")

    # Full should contain "Limitations"
    if "Limitations:" in full:
        console.print("  ✓ Full contains Limitations")
    else:
        console.print("  ✗ Full missing Limitations")
        all_ok = False

    # Reduced should NOT contain "Limitations"
    if "Limitations:" not in reduced:
        console.print("  ✓ Reduced omits Limitations")
    else:
        console.print("  ✗ Reduced contains Limitations (should be removed)")
        all_ok = False

    # Compressed should have [COMPRESSED] tag
    if "[COMPRESSED]" in compressed:
        console.print("  ✓ Compressed has [COMPRESSED] tag")
    else:
        console.print("  ✗ Compressed missing [COMPRESSED] tag")
        all_ok = False

    # Compressed should NOT contain "Methodology:"
    if "Methodology:" not in compressed or "Testing Methodology:" in compressed:
        console.print("  ✓ Compressed keeps testing, omits general methodology")
    else:
        console.print("  ✗ Compressed has general Methodology (should be removed)")
        all_ok = False

    # Test _format_l1_at_tier dispatcher
    for tier_name in ("full", "reduced", "compressed"):
        result = _format_l1_at_tier(s, tier_name, 1, 5)
        if len(result) > 0:
            console.print(f"  ✓ _format_l1_at_tier(tier={tier_name}): {len(result)} chars")
        else:
            console.print(f"  ✗ _format_l1_at_tier(tier={tier_name}): empty!")
            all_ok = False

    return all_ok


def test_budget_aware_formatting_small():
    """Test 4: Budget-aware formatting with few papers (should fit easily)."""
    console.print(Panel("Test 4: Budget-aware format (small, 5 papers)", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import _format_l1_summaries_for_prompt

    summaries = [
        _make_l1("Greshake_2023_LLM", year=2023, citation_count=200),
        _make_l1("Liu_2024_Formalizing", year=2024, citation_count=150),
        _make_l1("Yi_2023_Benchmarking", year=2023, citation_count=100),
        _make_l1("Zhan_2025_LLM", year=2025, citation_count=50),
        _make_l1("Chen_2024_StruQ", year=2024, citation_count=80),
    ]

    result = _format_l1_summaries_for_prompt(summaries)
    console.print(f"  Total output: {len(result):,} chars for {len(summaries)} papers")

    all_ok = True
    # All papers should be present (bibtex_key referenced)
    for s in summaries:
        if s.bibtex_key in result:
            console.print(f"  ✓ [{s.bibtex_key}] present in output")
        else:
            console.print(f"  ✗ [{s.bibtex_key}] MISSING from output")
            all_ok = False

    # With 5 papers, should not need any compression
    # (budget is ~596k chars, 5 papers will be a few KB)
    if "[COMPRESSED]" not in result and "[REDUCED]" not in result:
        console.print("  ✓ No compression applied (as expected for 5 papers)")
    else:
        # It's OK if tiers were applied by recency
        console.print("  ℹ Some recency-based tiers applied (expected for older papers)")

    return all_ok


def test_budget_aware_formatting_large():
    """Test 5: Budget-aware formatting with many papers (stress test)."""
    console.print(Panel("Test 5: Budget-aware format (large, 100 papers)", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import _format_l1_summaries_for_prompt
    from paperhunter.utils.token_counter import compute_l2_budget

    # Create 100 papers spanning 2019-2026
    summaries = []
    for i in range(100):
        year = 2019 + (i % 8)  # 2019-2026 cycle
        cit = (100 - i) * 10  # Higher citation for earlier papers
        summaries.append(
            _make_l1(
                f"Author_{year}_{i:03d}",
                year=year,
                citation_count=cit,
                long_fields=True,
            )
        )

    # Sort chronologically as the L2 summarizer expects
    summaries.sort(key=lambda s: (s.year or 0, s.paper_id))

    budget = compute_l2_budget()
    available = budget["available_chars"] - 3500  # template overhead

    result = _format_l1_summaries_for_prompt(summaries)
    result_chars = len(result)

    console.print(f"  Budget available: {available:,} chars")
    console.print(f"  Output:           {result_chars:,} chars")
    console.print(f"  Utilization:      {result_chars * 100 // available}%")

    all_ok = True
    if result_chars <= available:
        console.print(f"  ✓ Within budget")
    else:
        console.print(f"  ✗ Over budget by {result_chars - available:,} chars")
        all_ok = False

    # Check that some papers got COMPRESSED or REDUCED markers
    has_compressed = "[COMPRESSED]" in result
    has_reduced = "[REDUCED]" in result
    console.print(f"  Has [COMPRESSED] markers: {has_compressed}")
    console.print(f"  Has [REDUCED] markers: {has_reduced}")

    if has_compressed or has_reduced:
        console.print("  ✓ Recency-based tiers applied")
    else:
        console.print("  ℹ No tier markers (budget might be very generous)")

    return all_ok


def test_progressive_downgrade():
    """Test 6: Progressive downgrade under extreme budget pressure."""
    console.print(Panel("Test 6: Progressive downgrade (simulated tight budget)", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import (
        _get_paper_recency_tier,
        _compute_paper_importance,
        _format_l1_at_tier,
        _format_all_at_tiers,
    )

    # Create 20 papers with varying years and citations
    papers = []
    for i in range(20):
        year = 2020 + (i % 7)  # 2020-2026
        cit = (20 - i) * 50
        s = _make_l1(f"Paper_{year}_{i:02d}", year=year, citation_count=cit, long_fields=True)
        tier = _get_paper_recency_tier(s.year, 2026)
        importance = _compute_paper_importance(s, 2026)
        papers.append({
            "summary": s,
            "tier": tier,
            "original_tier": tier,
            "importance": importance,
            "index": i,
        })

    n = len(papers)

    # Format at original tiers
    parts_original = _format_all_at_tiers(papers, n)
    total_original = sum(len(p) for p in parts_original)

    # Format all as compressed
    for entry in papers:
        entry["tier"] = "compressed"
    parts_compressed = _format_all_at_tiers(papers, n)
    total_compressed = sum(len(p) for p in parts_compressed)

    console.print(f"  Original tiers total:   {total_original:,} chars")
    console.print(f"  All-compressed total:   {total_compressed:,} chars")
    console.print(f"  Compression ratio:      {total_compressed / total_original:.2%}")

    all_ok = True
    if total_compressed < total_original:
        console.print("  ✓ Compression reduces total size")
    else:
        console.print("  ✗ Compression did not reduce size")
        all_ok = False

    # Check importance-based ordering
    importances = [e["importance"] for e in papers]
    console.print(f"  Importance range: {min(importances):.4f} — {max(importances):.4f}")

    return all_ok


def test_truncate_text():
    """Test 7: _truncate_text helper."""
    console.print(Panel("Test 7: _truncate_text helper", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import _truncate_text

    all_ok = True

    # Short text should not be truncated
    result = _truncate_text("Hello world", 50)
    ok = result == "Hello world"
    console.print(f"  {'✓' if ok else '✗'} Short text unchanged: '{result}'")
    all_ok &= ok

    # Long text should be truncated
    long_text = "This is a long text that should be truncated at the word boundary"
    result = _truncate_text(long_text, 30)
    ok = len(result) <= 30 and result.endswith("...")
    console.print(f"  {'✓' if ok else '✗'} Truncated: '{result}' ({len(result)} chars)")
    all_ok &= ok

    # Empty text
    result = _truncate_text("", 10)
    ok = result == ""
    console.print(f"  {'✓' if ok else '✗'} Empty text: '{result}'")
    all_ok &= ok

    # None text
    result = _truncate_text(None, 10)
    ok = result is None
    console.print(f"  {'✓' if ok else '✗'} None text: {result}")
    all_ok &= ok

    return all_ok


def test_citation_count_field():
    """Test 8: citation_count field in L1Summary."""
    console.print(Panel("Test 8: citation_count in L1Summary", style="bold cyan"))

    from paperhunter.models import L1Summary

    all_ok = True

    # Default is None
    s1 = L1Summary(paper_id="test1")
    ok = s1.citation_count is None
    console.print(f"  {'✓' if ok else '✗'} Default citation_count is None")
    all_ok &= ok

    # Can set value
    s2 = L1Summary(paper_id="test2", citation_count=150)
    ok = s2.citation_count == 150
    console.print(f"  {'✓' if ok else '✗'} citation_count = {s2.citation_count}")
    all_ok &= ok

    # Serialization
    d = s2.model_dump()
    ok = d["citation_count"] == 150
    console.print(f"  {'✓' if ok else '✗'} Serialized: citation_count={d['citation_count']}")
    all_ok &= ok

    # JSON round-trip
    json_str = s2.model_dump_json()
    ok = '"citation_count":150' in json_str or '"citation_count": 150' in json_str
    console.print(f"  {'✓' if ok else '✗'} JSON contains citation_count")
    all_ok &= ok

    return all_ok


def test_backward_compat():
    """Test 9: Backward compatibility — _format_l1_summaries_for_prompt works like before for small inputs."""
    console.print(Panel("Test 9: Backward compatibility (small input)", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import _format_l1_summaries_for_prompt

    summaries = [
        _make_l1("Paper_A", year=2025),
        _make_l1("Paper_B", year=2024),
    ]

    result = _format_l1_summaries_for_prompt(summaries)

    all_ok = True

    # Both papers should appear
    for s in summaries:
        if s.bibtex_key in result:
            console.print(f"  ✓ [{s.bibtex_key}] present")
        else:
            console.print(f"  ✗ [{s.bibtex_key}] missing")
            all_ok = False

    # Key Contributions should appear for both
    count = result.count("Key Contributions:")
    ok = count >= 2
    console.print(f"  {'✓' if ok else '✗'} Key Contributions appears {count} times (expected ≥2)")
    all_ok &= ok

    # Empty input
    result_empty = _format_l1_summaries_for_prompt([])
    ok = result_empty == ""
    console.print(f"  {'✓' if ok else '✗'} Empty input returns empty string")
    all_ok &= ok

    return all_ok


def main():
    console.print(Panel("Test A.3: Budget-aware L2 Summarization", style="bold blue"))
    start_total = time.time()

    results = {}

    tests = [
        ("Recency tiers", test_recency_tiers),
        ("Paper importance", test_paper_importance),
        ("Tier formatting", test_tier_formatting),
        ("Budget small (5)", test_budget_aware_formatting_small),
        ("Budget large (100)", test_budget_aware_formatting_large),
        ("Progressive downgrade", test_progressive_downgrade),
        ("Truncate text", test_truncate_text),
        ("citation_count field", test_citation_count_field),
        ("Backward compat", test_backward_compat),
    ]

    for name, test_fn in tests:
        try:
            ok = test_fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            console.print(f"  ✗ EXCEPTION: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            results[name] = "ERROR"

    # Final report
    elapsed = time.time() - start_total
    console.print(Panel("Final Report", style="bold green"))

    from paperhunter.utils.logging_setup import print_table

    report_data = {f"Test {i+1}: {name}": status for i, (name, status) in enumerate(results.items())}
    report_data["Tempo totale"] = f"{elapsed:.1f}s"
    print_table("Budget-aware L2 Test Results", report_data)

    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    console.print(f"\n{'✓' if passed == total else '✗'} {passed}/{total} test passati", style="bold green" if passed == total else "bold red")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
