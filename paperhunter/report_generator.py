"""
Report Generator for PaperHunter.

Generates a detailed text report of the search session for
reproducibility and statistical purposes.
"""

import platform
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import settings
from .models import DownloadResult, FilteredPaper, Paper, SearchResult


def generate_report(
    keywords: str,
    criteria: str,
    search_result: SearchResult,
    filtered_results: list[FilteredPaper],
    download_results: list[DownloadResult],
    relevant_papers: list[Paper],
    output_dir: Path,
    max_results_per_source: int,
    categories: list[str],
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Path:
    """
    Generate a detailed text report of the paper hunting session.

    Args:
        keywords: Search keywords used.
        criteria: Filtering criteria sent to the LLM.
        search_result: Aggregated search results.
        filtered_results: All papers after LLM filtering.
        download_results: Results of download attempts.
        relevant_papers: Papers judged relevant by the LLM.
        output_dir: Directory where the report will be saved.
        max_results_per_source: Max results requested per source.
        categories: ArXiv categories searched.
        llm_provider: LLM provider used for filtering.
        llm_model: LLM model used for filtering.

    Returns:
        Path to the generated report file.
    """
    timestamp = datetime.now()
    safe_kw = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in keywords)[:60].strip()
    report_name = f"report_{safe_kw}_{timestamp:%Y%m%d_%H%M%S}.txt"
    report_path = output_dir / report_name

    provider = llm_provider or settings.llm_provider
    model = llm_model or settings.llm_model

    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    def kv(key: str, value) -> None:
        lines.append(f"  {key:<35} {value}")

    # ── Header ──────────────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("  PAPERHUNTER — SESSION REPORT")
    lines.append("=" * 70)

    # ── 1. Session metadata ─────────────────────────────────────────────
    section("1. SESSION METADATA")
    kv("Timestamp:", timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    kv("Platform:", platform.platform())
    kv("Python version:", platform.python_version())

    # ── 2. Search parameters (reproducibility) ──────────────────────────
    section("2. SEARCH PARAMETERS")
    kv("Keywords:", keywords)
    kv("Max results per source:", max_results_per_source)
    kv("ArXiv categories:", ", ".join(categories))

    # ── 3. LLM filtering parameters ────────────────────────────────────
    section("3. LLM FILTERING CONFIGURATION")
    kv("LLM provider:", provider)
    kv("LLM model:", model)
    kv("LLM temperature:", settings.llm_temperature)
    lines.append("")
    lines.append("  Filtering criteria (prompt):")
    lines.append("  " + "-" * 66)
    for crit_line in criteria.splitlines():
        lines.append(f"    {crit_line}")
    lines.append("  " + "-" * 66)

    # ── 4. Sources queried ──────────────────────────────────────────────
    section("4. SOURCES QUERIED")
    for src in search_result.sources_queried:
        lines.append(f"    • {src}")
    if search_result.errors:
        lines.append("")
        lines.append("  Errors during search:")
        for err in search_result.errors:
            lines.append(f"    ⚠  {err}")

    # ── 5. Statistics ───────────────────────────────────────────────────
    section("5. STATISTICS")
    total_found = search_result.total_found
    total_relevant = len(relevant_papers)
    total_not_relevant = len(filtered_results) - total_relevant
    total_downloaded = sum(1 for r in download_results if r.success)
    total_download_failed = sum(1 for r in download_results if not r.success)

    kv("Total papers found:", total_found)
    kv("Papers judged relevant:", total_relevant)
    kv("Papers judged NOT relevant:", total_not_relevant)
    kv("Papers downloaded OK:", total_downloaded)
    kv("Papers download FAILED:", total_download_failed)

    # Per-source breakdown
    source_counts: dict[str, int] = {}
    for p in search_result.papers:
        source_counts[p.source] = source_counts.get(p.source, 0) + 1
    if source_counts:
        lines.append("")
        lines.append("  Papers found per source:")
        for src, cnt in sorted(source_counts.items()):
            lines.append(f"    {src:<25} {cnt}")

    source_relevant: dict[str, int] = {}
    for fp in filtered_results:
        if fp.relevance.is_relevant:
            src = fp.paper.source
            source_relevant[src] = source_relevant.get(src, 0) + 1
    if source_relevant:
        lines.append("")
        lines.append("  Relevant papers per source:")
        for src, cnt in sorted(source_relevant.items()):
            lines.append(f"    {src:<25} {cnt}")

    # ── 6. Relevant papers detail ───────────────────────────────────────
    section("6. RELEVANT PAPERS (DETAIL)")
    for i, fp in enumerate(filtered_results, 1):
        if not fp.relevance.is_relevant:
            continue
        paper = fp.paper
        lines.append("")
        lines.append(f"  [{i}] {paper.title}")
        lines.append(f"      Source:       {paper.source}")
        lines.append(f"      URL:          {paper.url}")
        if paper.pdf_url:
            lines.append(f"      PDF URL:      {paper.pdf_url}")
        if paper.authors:
            lines.append(f"      Authors:      {', '.join(paper.authors)}")
        if paper.published_date:
            lines.append(f"      Published:    {paper.published_date.strftime('%Y-%m-%d')}")
        if paper.arxiv_id:
            lines.append(f"      ArXiv ID:     {paper.arxiv_id}")
        if paper.doi:
            lines.append(f"      DOI:          {paper.doi}")
        if paper.github_url:
            lines.append(f"      GitHub:       {paper.github_url}")
        if paper.categories:
            lines.append(f"      Categories:   {', '.join(paper.categories)}")
        if paper.citations is not None:
            lines.append(f"      Citations:    {paper.citations}")
        lines.append(f"      Confidence:   {fp.relevance.confidence:.2f}")
        lines.append(f"      Reasoning:    {fp.relevance.reasoning}")

    # ── 7. Download results ─────────────────────────────────────────────
    if download_results:
        section("7. DOWNLOAD RESULTS")
        for paper, result in zip(relevant_papers, download_results):
            status = "OK" if result.success else "FAILED"
            lines.append("")
            lines.append(f"  [{status}] {paper.title}")
            lines.append(f"      Source:  {paper.source}")
            lines.append(f"      URL:     {paper.url}")
            if result.success:
                lines.append(f"      File:    {result.file_path}")
                if result.method:
                    lines.append(f"      Method:  {result.method}")
            else:
                lines.append(f"      Error:   {result.error_message}")

    # ── 8. Non-relevant papers (summary) ────────────────────────────────
    non_relevant = [fp for fp in filtered_results if not fp.relevance.is_relevant]
    if non_relevant:
        section("8. NON-RELEVANT PAPERS (SUMMARY)")
        for i, fp in enumerate(non_relevant, 1):
            lines.append(f"  [{i}] {fp.paper.title}")
            lines.append(f"      Source:     {fp.paper.source}")
            lines.append(f"      URL:        {fp.paper.url}")
            lines.append(f"      Reasoning:  {fp.relevance.reasoning[:200]}")
            lines.append("")

    # ── Footer ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Write report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
