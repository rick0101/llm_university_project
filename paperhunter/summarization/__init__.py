"""
Modulo Summarization — Hierarchical summarization (L1 + L2) con auto-invalidazione.

- ``level1_summarizer``: Summary strutturato per singolo paper
- ``level2_summarizer``: Meta-summary per categoria
- ``invalidation_manager``: Auto-invalidazione L2 su nuovi paper
"""

from paperhunter.summarization.level1_summarizer import (
    generate_l1_summary,
    generate_l1_batch,
    load_paper_and_chunks_from_ingestion,
)
from paperhunter.summarization.level2_summarizer import (
    generate_l2_summary,
    generate_l2_for_all_categories,
    load_l1_summaries_for_category,
    load_all_l2_metadata,
)
from paperhunter.summarization.invalidation_manager import (
    InvalidationManager,
    InvalidationReport,
)

__all__ = [
    "generate_l1_summary",
    "generate_l1_batch",
    "load_paper_and_chunks_from_ingestion",
    "generate_l2_summary",
    "generate_l2_for_all_categories",
    "load_l1_summaries_for_category",
    "load_all_l2_metadata",
    "InvalidationManager",
    "InvalidationReport",
]
