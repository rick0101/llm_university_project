"""Eval Survey Quality — v3.6 (Phase 4.1)

Misura quattro metriche chiave per valutare la qualità di un survey
prodotto da PaperHunter, e la salute del corpus citabile:

1. **bibliography_purity**: % entry in references.bib provenienti da paper
   con ``dblp_verified=True`` (idealmente 100%).
2. **citation_grounding_rate**: % citazioni nel survey con status ``SUPPORTED``
   nel verification report (gate: ``settings.agents.verification_threshold``).
3. **claim_coverage**: % di citazioni nel survey i cui paper hanno almeno
   un atomic claim strutturato in L1 (proxy della verificabilità).
4. **retrieval_precision_at_k**: opzionale, richiede un golden set
   (``--golden tools/golden_set.json``) → per ogni query, % dei paper
   attesi presenti nei top-k del retrieval.

Esempi d'uso::

    # Eval di un survey già generato (con state.json del run)
    python tools/eval_survey_quality.py \\
        --state output/run_20251110/final_state.json

    # Eval del corpus (purity + claim_coverage globale)
    python tools/eval_survey_quality.py --corpus-only

    # Con golden set per retrieval@k
    python tools/eval_survey_quality.py \\
        --state output/run_xxx/final_state.json \\
        --golden tools/golden_set.json --k 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Aggiungi root al path se script lanciato da repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperhunter.config import settings  # noqa: E402
from paperhunter.utils.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric 1: bibliography purity
# ---------------------------------------------------------------------------

_BIBKEY_RE = re.compile(r"^@\w+\s*\{\s*([^,\s]+)", re.MULTILINE)


def _extract_bib_keys(bib_path: Path) -> set[str]:
    if not bib_path.exists():
        return set()
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    return {m.group(1).strip() for m in _BIBKEY_RE.finditer(text)}


def metric_bibliography_purity(output_dir: Path) -> dict[str, Any]:
    """Confronta references.bib (citable) vs references_uncited.bib."""
    citable = _extract_bib_keys(output_dir / "references.bib")
    uncited = _extract_bib_keys(output_dir / "references_uncited.bib")
    total = len(citable) + len(uncited)
    purity = (len(citable) / total) if total > 0 else 1.0
    return {
        "metric": "bibliography_purity",
        "value": round(purity, 4),
        "citable_count": len(citable),
        "uncited_count": len(uncited),
        "total": total,
        "interpretation": (
            "frazione di entry .bib provenienti da paper DBLP-verified; "
            "1.0 = nessun paper hallucinato in bibliografia."
        ),
    }


# ---------------------------------------------------------------------------
# Metric 2: citation grounding rate
# ---------------------------------------------------------------------------

def metric_citation_grounding(state: dict[str, Any]) -> dict[str, Any]:
    """Aggrega tutti i VerificationResult del run."""
    vrs = state.get("verification_results") or {}
    total = 0
    supported = 0
    partial = 0
    not_supp = 0
    not_found = 0
    for vr in vrs.values():
        total += vr.get("total_citations", 0)
        supported += vr.get("supported", 0)
        partial += vr.get("partially_supported", 0)
        not_supp += vr.get("not_supported", 0)
        not_found += vr.get("source_not_found", 0)
    grounded = (supported / total) if total > 0 else 1.0
    return {
        "metric": "citation_grounding_rate",
        "value": round(grounded, 4),
        "supported": supported,
        "partially_supported": partial,
        "not_supported": not_supp,
        "source_not_found": not_found,
        "total_citations": total,
        "threshold": settings.agents.verification_threshold,
        "passes_threshold": grounded >= settings.agents.verification_threshold,
    }


# ---------------------------------------------------------------------------
# Metric 3: claim coverage
# ---------------------------------------------------------------------------

def _papers_with_atomic_claims() -> set[str]:
    """Restituisce il set di bibtex_key che hanno claims in L1."""
    try:
        from paperhunter.indexing.qdrant_store import QdrantStore
        store = QdrantStore()
        keys: set[str] = set()
        offset = None
        coll = store.COLLECTION_L1
        while True:
            pts, offset = store.client.scroll(
                collection_name=coll,
                with_payload=["bibtex_key", "text"],
                with_vectors=False,
                limit=512,
                offset=offset,
            )
            for p in pts:
                payload = p.payload or {}
                txt = payload.get("text") or ""
                bk = payload.get("bibtex_key")
                if bk and "Atomic Claims" in txt:
                    keys.add(bk)
            if offset is None:
                break
        return keys
    except Exception as exc:
        logger.warning("Impossibile caricare papers con atomic claims: %s", exc)
        return set()


def metric_claim_coverage(state: dict[str, Any]) -> dict[str, Any]:
    """% di citazioni nel survey i cui paper hanno almeno 1 atomic claim."""
    cited_keys: set[str] = set()
    sections = state.get("sections") or {}
    for sec in sections.values():
        for c in sec.get("citations") or []:
            bk = c.get("bibtex_key")
            if bk:
                cited_keys.add(bk)
        # Estrai anche dal testo (fallback)
        for m in re.finditer(r"\[([^\]\s]+)\]", sec.get("text", "")):
            k = m.group(1)
            if "_" in k and any(ch.isdigit() for ch in k):
                cited_keys.add(k)

    if not cited_keys:
        return {
            "metric": "claim_coverage",
            "value": 0.0,
            "cited_papers": 0,
            "with_claims": 0,
        }
    with_claims_global = _papers_with_atomic_claims()
    matched = cited_keys & with_claims_global
    coverage = len(matched) / len(cited_keys)
    return {
        "metric": "claim_coverage",
        "value": round(coverage, 4),
        "cited_papers": len(cited_keys),
        "with_claims": len(matched),
        "interpretation": (
            "frazione dei paper citati nel survey che hanno almeno un atomic "
            "claim strutturato in L1 (verificabile chirurgicamente)."
        ),
    }


# ---------------------------------------------------------------------------
# Metric 4: retrieval precision@k (opzionale, richiede golden set)
# ---------------------------------------------------------------------------

def metric_retrieval_precision_at_k(
    golden_path: Path,
    k: int = 10,
) -> dict[str, Any]:
    """Per ogni query del golden set, calcola precision@k del retrieval."""
    try:
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"metric": "retrieval_precision_at_k", "error": str(exc)}

    try:
        from paperhunter.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
    except Exception as exc:
        return {"metric": "retrieval_precision_at_k", "error": f"retriever init failed: {exc}"}

    per_query: list[dict[str, Any]] = []
    precisions: list[float] = []
    for entry in golden.get("queries", []):
        query = entry.get("query")
        expected = set(entry.get("expected_bibtex_keys") or [])
        if not query or not expected:
            continue
        try:
            chunks = retriever.retrieve(query, top_k=k)
        except Exception as exc:
            per_query.append({"query": query, "error": str(exc)})
            continue
        retrieved_keys = set()
        for c in chunks[:k]:
            bk = c.get("metadata", {}).get("bibtex_key")
            if bk:
                retrieved_keys.add(bk)
        hits = expected & retrieved_keys
        prec = len(hits) / min(k, len(expected))
        precisions.append(prec)
        per_query.append({
            "query": query,
            "precision_at_k": round(prec, 4),
            "expected": len(expected),
            "retrieved": len(retrieved_keys),
            "hits": len(hits),
        })

    avg = sum(precisions) / len(precisions) if precisions else 0.0
    return {
        "metric": "retrieval_precision_at_k",
        "value": round(avg, 4),
        "k": k,
        "n_queries": len(precisions),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, help="Path to final_state.json del run survey")
    parser.add_argument("--output-dir", type=Path, help="Cartella output del run (per .bib). Default: parent di --state.")
    parser.add_argument("--corpus-only", action="store_true", help="Esegui solo metriche corpus (purity, claims globali)")
    parser.add_argument("--golden", type=Path, help="Path golden set JSON per retrieval@k")
    parser.add_argument("--k", type=int, default=10, help="K per retrieval precision@k (default 10)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output JSON (default: pretty)")
    args = parser.parse_args()

    results: dict[str, Any] = {"version": "v3.6"}

    output_dir: Path | None = args.output_dir
    state: dict[str, Any] | None = None

    if args.state:
        if not args.state.exists():
            print(f"❌ State file non trovato: {args.state}")
            return 2
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if output_dir is None:
            output_dir = args.state.parent

    if output_dir is None:
        output_dir = Path(settings.output.dir) if hasattr(settings.output, "dir") else Path("output")

    # Metric 1
    results["bibliography_purity"] = metric_bibliography_purity(output_dir)

    if not args.corpus_only and state is not None:
        results["citation_grounding_rate"] = metric_citation_grounding(state)
        results["claim_coverage"] = metric_claim_coverage(state)

    if args.golden:
        results["retrieval_precision_at_k"] = metric_retrieval_precision_at_k(args.golden, k=args.k)

    if args.as_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("\n=== PaperHunter v3.6 — Survey Quality Eval ===\n")
        for name, data in results.items():
            if name == "version":
                print(f"Version: {data}\n")
                continue
            if isinstance(data, dict) and "metric" in data:
                val = data.get("value", "n/a")
                print(f"▸ {data['metric']}: {val}")
                for k, v in data.items():
                    if k in ("metric", "value", "per_query"):
                        continue
                    print(f"    {k}: {v}")
                print()
        print("===============================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
