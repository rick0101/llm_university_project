"""
Modulo Summarization — Hierarchical summarization (L1 + L1.5 + L2) con auto-invalidazione.

- ``level1_summarizer``: Summary strutturato per singolo paper
- ``level1_5_summarizer``: Meta-summary intermedio per cluster RAPTOR
- ``level2_summarizer``: Meta-summary per categoria
- ``clustering_utils``: Clustering semantico degli L1 Summary
- ``invalidation_manager``: Auto-invalidazione L2 su nuovi paper
"""

from paperhunter.summarization.level1_summarizer import (
    generate_l1_summary,
    generate_l1_batch,
    load_paper_and_chunks_from_ingestion,
)
from paperhunter.summarization.level1_5_summarizer import (
    generate_cluster_summary,
    generate_all_cluster_summaries,
    load_l1_5_summaries_for_category,
    count_l1_5_for_category,
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
    "generate_cluster_summary",
    "generate_all_cluster_summaries",
    "load_l1_5_summaries_for_category",
    "count_l1_5_for_category",
    "generate_l2_summary",
    "generate_l2_for_all_categories",
    "load_l1_summaries_for_category",
    "load_all_l2_metadata",
    "InvalidationManager",
    "InvalidationReport",
]
