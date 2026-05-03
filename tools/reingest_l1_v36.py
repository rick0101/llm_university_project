#!/usr/bin/env python
"""
Reingest L1 (v3.6) — rigenera i summary L1 con il nuovo schema tassonomico.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperhunter.indexing.qdrant_store import QdrantStore  # noqa: E402
from paperhunter.ingestion.hash_registry import load_hash_registry  # noqa: E402
from paperhunter.models import Paper, PaperChunk  # noqa: E402
from paperhunter.summarization.level1_summarizer import generate_l1_summary  # noqa: E402
from paperhunter.utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def _load_paper_from_registry(entry: dict) -> Paper | None:
    try:
        return Paper(**entry.get("paper", entry))
    except Exception as exc:
        logger.warning("Skip entry: %s", exc)
        return None


def _load_chunks_for_paper(store: QdrantStore, paper_id: str) -> list[PaperChunk]:
    try:
        return store.get_chunks_by_paper(paper_id)
    except Exception as exc:
        logger.warning("Impossibile recuperare chunk per %s: %s", paper_id, exc)
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-ingestion L1 schema v3.6")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-citable", action="store_true")
    args = parser.parse_args()

    registry = load_hash_registry()
    if not registry:
        print("Registry vuoto. Esegui prima `paperhunter ingest`.")
        return 1

    store = QdrantStore()

    targets: list[Paper] = []
    for _hash, entry in registry.items():
        paper = _load_paper_from_registry(entry)
        if paper is None:
            continue
        if args.category and (paper.category or "").lower() != args.category.lower():
            continue
        if args.only_citable and not paper.is_citable:
            continue
        targets.append(paper)

    if args.limit:
        targets = targets[: args.limit]

    print(f"Paper da rigenerare: {len(targets)}")
    if args.dry_run:
        for p in targets[:20]:
            print(f"  - {p.bibtex_key}  is_citable={p.is_citable}  cat={p.category}")
        if len(targets) > 20:
            print(f"  ... e altri {len(targets) - 20}")
        return 0

    n_ok = 0
    n_fail = 0
    for i, paper in enumerate(targets, 1):
        chunks = _load_chunks_for_paper(store, paper.bibtex_key or paper.pdf_hash)
        if not chunks:
            logger.warning("[%d/%d] Nessun chunk per %s, skip", i, len(targets), paper.bibtex_key)
            n_fail += 1
            continue
        try:
            generate_l1_summary(paper, chunks, store=store, save_to_store=True)
            n_ok += 1
            logger.info("[%d/%d] OK %s", i, len(targets), paper.bibtex_key)
        except Exception as exc:
            logger.error("[%d/%d] FAIL %s: %s", i, len(targets), paper.bibtex_key, exc)
            n_fail += 1

    print(f"\nCompletato: {n_ok} OK, {n_fail} fail")
    print("Esegui ora `paperhunter summarize` per rigenerare L1.5 e L2.")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
