#!/usr/bin/env python3
"""
Test end-to-end integrazione completa — Task 6.2

Esegue l'intera pipeline su 20 paper di test (5 per categoria):
  ingest → summarize (L1 + L2) → index (BM25 + GraphRAG) → write → export

Output in output/test_run/
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Assicura che .env sia caricato
from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# Setup logging
# ═══════════════════════════════════════════════════════════════════════════
from paperhunter.utils.logging_setup import setup_logging, get_logger, console, print_panel, print_success, print_error, print_warning

setup_logging(level="INFO")
logger = get_logger("test.e2e")

# ═══════════════════════════════════════════════════════════════════════════
# Configurazione test
# ═══════════════════════════════════════════════════════════════════════════
TEST_PDF_DIR = Path("scaricati_test")
TEST_OUTPUT_DIR = Path("output/test_run")
TEST_REPORT_PATH = TEST_OUTPUT_DIR / "e2e_report.json"

# Stato per il report finale
report = {
    "started_at": datetime.now().isoformat(),
    "steps": {},
    "errors": [],
    "warnings": [],
}


def log_step(step_name: str, status: str, details: dict = None, error: str = None):
    """Registra risultato di uno step nel report."""
    report["steps"][step_name] = {
        "status": status,
        "details": details or {},
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    if error:
        report["errors"].append(f"[{step_name}] {error}")


def save_report():
    """Salva il report JSON."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report["finished_at"] = datetime.now().isoformat()
    TEST_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(f"Report salvato: {TEST_REPORT_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: INGESTION
# ═══════════════════════════════════════════════════════════════════════════
def step_ingest() -> bool:
    """Esegue ingestion dei 20 paper di test."""
    logger.info("=" * 60)
    logger.info("STEP 1: INGESTION")
    logger.info("=" * 60)

    from paperhunter.ingestion import IngestionPipeline

    try:
        pipeline = IngestionPipeline(
            pdf_base_dir=TEST_PDF_DIR,
            batch_size=5,
            skip_s2=False,
            skip_qdrant=False,
            use_checkpoint=True,
        )

        ing_report = pipeline.run()

        details = {
            "processed": ing_report.processed,
            "skipped": ing_report.skipped,
            "errors": ing_report.errors,
            "total_chunks": ing_report.total_chunks,
            "time_seconds": round(ing_report.total_time_seconds, 1),
            "grobid_stats": ing_report.grobid_stats,
            "s2_stats": ing_report.s2_stats,
        }

        if ing_report.error_details:
            details["error_details"] = ing_report.error_details
            for ed in ing_report.error_details:
                report["warnings"].append(f"[ingest] {ed}")

        if ing_report.errors > 0 and ing_report.processed == 0:
            log_step("ingest", "FAILED", details, f"Tutti gli ingestion falliti: {ing_report.errors} errori")
            return False

        log_step("ingest", "OK", details)
        logger.info(f"Ingestion: {ing_report.processed} processati, {ing_report.total_chunks} chunk, "
                     f"{ing_report.errors} errori in {ing_report.total_time_seconds:.1f}s")
        return True

    except Exception as e:
        log_step("ingest", "FAILED", error=f"{type(e).__name__}: {e}")
        logger.error(f"Ingestion fallita: {e}", exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: VERIFY INGESTION — Check Qdrant, registry, etc.
# ═══════════════════════════════════════════════════════════════════════════
def step_verify_ingestion() -> bool:
    """Verifica che l'ingestion abbia popolato tutto correttamente."""
    logger.info("=" * 60)
    logger.info("STEP 1b: VERIFICA INGESTION")
    logger.info("=" * 60)

    from paperhunter.indexing import QdrantStore
    from paperhunter.ingestion.pdf_hasher import load_hash_registry, get_registry_stats

    try:
        # Check registry
        registry = load_hash_registry()
        reg_stats = get_registry_stats(registry)
        logger.info(f"Registry: {reg_stats.get('total', 0)} entry, "
                     f"categorie: {reg_stats.get('per_category', {})}")

        # Check Qdrant
        store = QdrantStore()
        qdrant_stats = store.get_stats()

        pc = qdrant_stats.get("paper_chunks", {})
        total_chunks = pc.get("total_points", 0)
        unique_papers = pc.get("unique_papers", 0)
        categories = pc.get("categories", {})

        logger.info(f"Qdrant paper_chunks: {total_chunks} chunk, {unique_papers} paper, "
                     f"categorie: {categories}")

        # Check master BibTeX
        bib_path = Path("output/references.bib")
        bib_exists = bib_path.exists()
        bib_size = bib_path.stat().st_size if bib_exists else 0
        logger.info(f"Master .bib: {'OK' if bib_exists else 'MISSING'} ({bib_size} bytes)")

        details = {
            "registry_total": reg_stats.get("total", 0),
            "qdrant_chunks": total_chunks,
            "qdrant_papers": unique_papers,
            "qdrant_categories": categories,
            "bib_exists": bib_exists,
            "bib_size_bytes": bib_size,
        }

        ok = total_chunks > 0 and unique_papers > 0
        log_step("verify_ingestion", "OK" if ok else "WARNING", details)
        return ok

    except Exception as e:
        log_step("verify_ingestion", "FAILED", error=str(e))
        logger.error(f"Verifica ingestion fallita: {e}", exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: L1 SUMMARIZATION
# ═══════════════════════════════════════════════════════════════════════════
def step_l1_summarization() -> bool:
    """Genera L1 summary per ogni paper nel vector store."""
    logger.info("=" * 60)
    logger.info("STEP 2a: L1 SUMMARIZATION")
    logger.info("=" * 60)

    from paperhunter.indexing import QdrantStore
    from paperhunter.models import Paper, PaperChunk
    from paperhunter.summarization.level1_summarizer import generate_l1_summary

    try:
        store = QdrantStore()

        # Scopri paper nel vector store
        paper_map = {}  # bibtex_key -> {paper_id, metadata}
        offset = None
        while True:
            results = store._client.scroll(
                collection_name="paper_chunks", limit=100, offset=offset,
                with_payload=True, with_vectors=False,
            )
            points, next_offset = results
            if not points:
                break
            for pt in points:
                bk = pt.payload.get("bibtex_key", "")
                if bk and bk not in paper_map:
                    paper_map[bk] = {
                        "paper_id": pt.payload.get("paper_id", ""),
                        "category": pt.payload.get("category", ""),
                        "year": pt.payload.get("year"),
                        "title": pt.payload.get("paper_title", bk),
                    }
            if next_offset is None:
                break
            offset = next_offset

        logger.info(f"Trovati {len(paper_map)} paper unici per L1")

        # Check L1 già esistenti
        existing_l1 = set()
        offset = None
        while True:
            results = store._client.scroll(
                collection_name="l1_summaries", limit=100, offset=offset,
                with_payload=True, with_vectors=False,
            )
            points, next_offset = results
            if not points:
                break
            for pt in points:
                bk = pt.payload.get("bibtex_key", "")
                if bk:
                    existing_l1.add(bk)
            if next_offset is None:
                break
            offset = next_offset

        missing = {bk: info for bk, info in paper_map.items() if bk not in existing_l1}
        logger.info(f"{len(missing)} paper senza L1 (di {len(paper_map)} totali)")

        if not missing:
            log_step("l1_summarization", "OK", {"generated": 0, "total": len(paper_map), "note": "all already exist"})
            return True

        generated = 0
        errors_count = 0
        details_list = []

        for bk, info in missing.items():
            try:
                logger.info(f"  Generando L1 per {bk}...")
                chunks_data = store.get_paper_chunks(bk, collection_name="paper_chunks")
                if not chunks_data:
                    logger.warning(f"  Nessun chunk per {bk}, skip")
                    continue

                meta = chunks_data[0].get("metadata", {})
                paper = Paper(
                    title=meta.get("paper_title", bk),
                    source=meta.get("source", "unknown"),
                    url="",
                    bibtex_key=bk,
                    category=meta.get("category", ""),
                    year=meta.get("year"),
                    pdf_hash=meta.get("pdf_hash", ""),
                )

                chunks = []
                for cd in chunks_data:
                    m = cd.get("metadata", {})
                    chunks.append(PaperChunk(
                        chunk_id=cd.get("chunk_id", ""),
                        text=cd.get("text", ""),
                        paper_id=m.get("paper_id", info["paper_id"]),
                        bibtex_key=bk,
                        section_name=m.get("section_name", ""),
                        chunk_index=m.get("chunk_index", 0),
                        category=m.get("category", ""),
                        year=m.get("year"),
                    ))

                summary = generate_l1_summary(paper, chunks, save_to_store=True, store=store)
                if summary:
                    generated += 1
                    has_testing = bool(summary.testing_methodology and len(summary.testing_methodology) > 50)
                    details_list.append({
                        "bibtex_key": bk,
                        "testing_methodology_length": len(summary.testing_methodology or ""),
                        "has_testing": has_testing,
                    })
                    logger.info(f"  L1 OK per {bk} (testing_methodology: {len(summary.testing_methodology or '')} chars)")

            except Exception as e:
                errors_count += 1
                logger.error(f"  Errore L1 per {bk}: {e}")
                report["warnings"].append(f"[l1] Errore per {bk}: {e}")

        details = {
            "total_papers": len(paper_map),
            "generated": generated,
            "errors": errors_count,
            "papers": details_list,
        }

        ok = generated > 0
        log_step("l1_summarization", "OK" if ok else "WARNING", details)
        return ok

    except Exception as e:
        log_step("l1_summarization", "FAILED", error=str(e))
        logger.error(f"L1 summarization fallita: {e}", exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: L2 SUMMARIZATION
# ═══════════════════════════════════════════════════════════════════════════
def step_l2_summarization() -> bool:
    """Genera L2 meta-summary per ogni categoria."""
    logger.info("=" * 60)
    logger.info("STEP 2b: L2 SUMMARIZATION")
    logger.info("=" * 60)

    from paperhunter.indexing import QdrantStore
    from paperhunter.summarization import generate_l2_for_all_categories

    try:
        store = QdrantStore()
        summaries = generate_l2_for_all_categories(
            save_to_store=True, store=store, skip_existing=False,
        )

        details = {
            "categories_generated": len(summaries),
            "categories": [],
        }
        for s in summaries:
            details["categories"].append({
                "category": s.category,
                "version": s.version,
                "paper_count": s.paper_count,
                "total_chars": len(str(s.temporal_trends or "")) + len(str(s.taxonomy or "")) +
                               len(str(s.testing_overview or "")) + len(str(s.synthesis or "")),
            })

        ok = len(summaries) > 0
        log_step("l2_summarization", "OK" if ok else "WARNING", details)
        logger.info(f"L2: {len(summaries)} categorie generate")
        return ok

    except Exception as e:
        log_step("l2_summarization", "FAILED", error=str(e))
        logger.error(f"L2 summarization fallita: {e}", exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: BM25 INDEX
# ═══════════════════════════════════════════════════════════════════════════
def step_bm25_index() -> bool:
    """Costruisce l'indice BM25."""
    logger.info("=" * 60)
    logger.info("STEP 3a: BM25 INDEX")
    logger.info("=" * 60)

    from paperhunter.indexing import QdrantStore, BM25Index
    from paperhunter.models import PaperChunk

    try:
        store = QdrantStore()
        bm25 = BM25Index()

        all_chunks = []
        for collection in ["paper_chunks", "l1_summaries", "l2_summaries"]:
            offset = None
            while True:
                results = store._client.scroll(
                    collection_name=collection, limit=100, offset=offset,
                    with_payload=True, with_vectors=False,
                )
                points, next_offset = results
                if not points:
                    break
                for pt in points:
                    p = pt.payload
                    chunk = PaperChunk(
                        chunk_id=p.get("chunk_id", str(pt.id)),
                        text=p.get("text", ""),
                        paper_id=p.get("paper_id", ""),
                        bibtex_key=p.get("bibtex_key", ""),
                        section_name=p.get("section_name", ""),
                        chunk_index=p.get("chunk_index", 0),
                        category=p.get("category", ""),
                        year=p.get("year"),
                        month=p.get("month"),
                        chunk_type=p.get("chunk_type", "original"),
                    )
                    all_chunks.append(chunk)
                if next_offset is None:
                    break
                offset = next_offset

        if not all_chunks:
            log_step("bm25_index", "WARNING", {"reason": "no chunks found"})
            return False

        count = bm25.build_index(all_chunks, show_progress=True)
        bm25.save()

        stats = bm25.get_stats()
        details = {
            "total_indexed": count,
            "unique_papers": stats.get("unique_papers", 0),
            "categories": stats.get("categories", {}),
        }

        log_step("bm25_index", "OK", details)
        logger.info(f"BM25: {count} documenti indicizzati")
        return True

    except Exception as e:
        log_step("bm25_index", "FAILED", error=str(e))
        logger.error(f"BM25 index fallito: {e}", exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: GRAPHRAG (PropertyGraph)
# ═══════════════════════════════════════════════════════════════════════════
def step_graphrag() -> bool:
    """Costruisce il PropertyGraph da L1 summaries."""
    logger.info("=" * 60)
    logger.info("STEP 3b: GRAPHRAG (PropertyGraph)")
    logger.info("=" * 60)

    from paperhunter.indexing import QdrantStore
    from paperhunter.graphrag import PropertyGraphBuilder
    from paperhunter.summarization import load_l1_summaries_for_category

    try:
        store = QdrantStore()

        # Scopri categorie con L1
        categories = set()
        offset = None
        while True:
            results = store._client.scroll(
                collection_name="l1_summaries", limit=100, offset=offset,
                with_payload=True, with_vectors=False,
            )
            points, next_offset = results
            if not points:
                break
            for pt in points:
                cat = pt.payload.get("category", "")
                if cat:
                    categories.add(cat)
            if next_offset is None:
                break
            offset = next_offset

        if not categories:
            log_step("graphrag", "WARNING", {"reason": "no L1 found"})
            return False

        all_l1 = []
        for cat in sorted(categories):
            l1s = load_l1_summaries_for_category(cat, store=store)
            all_l1.extend(l1s)
            logger.info(f"  {cat}: {len(l1s)} L1")

        builder = PropertyGraphBuilder()
        build_report = builder.build_from_summaries(all_l1)
        builder.save()

        details = {
            "entities": build_report.entities_created,
            "relations": build_report.relations_created,
            "time_seconds": round(build_report.elapsed_seconds, 1),
            "categories": list(sorted(categories)),
        }

        log_step("graphrag", "OK", details)
        logger.info(f"GraphRAG: {build_report.entities_created} entità, "
                     f"{build_report.relations_created} relazioni")
        return True

    except Exception as e:
        log_step("graphrag", "FAILED", error=str(e))
        logger.error(f"GraphRAG fallito: {e}", exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: AGENTIC WRITING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def step_write() -> dict:
    """Esegue la pipeline agentica (Planner → Writer → Critic → Assembler)."""
    logger.info("=" * 60)
    logger.info("STEP 4: AGENTIC WRITING PIPELINE")
    logger.info("=" * 60)

    from paperhunter.agents import run_pipeline, get_pipeline_summary

    try:
        start = time.time()
        state = run_pipeline(
            save_checkpoints=True,
        )
        elapsed = time.time() - start

        summary = get_pipeline_summary(state)

        details = {
            "phase": summary.get("phase", "UNKNOWN"),
            "total_sections": summary.get("total_sections", 0),
            "sections_verified": summary.get("sections_verified", 0),
            "sections_finalized": summary.get("sections_finalized", 0),
            "total_citations": summary.get("total_citations", 0),
            "iteration_count": summary.get("iteration_count", 0),
            "errors": summary.get("errors", []),
            "time_seconds": round(elapsed, 1),
            "has_final_paper": bool(state.get("final_paper")),
            "final_paper_length": len(state.get("final_paper", "")),
        }

        ok = summary.get("phase") == "COMPLETED" and state.get("final_paper")
        log_step("write", "OK" if ok else "WARNING", details)

        logger.info(f"Pipeline: phase={summary.get('phase')}, "
                     f"sections={summary.get('total_sections')}, "
                     f"citations={summary.get('total_citations')}, "
                     f"{elapsed:.0f}s")

        return state

    except Exception as e:
        log_step("write", "FAILED", error=f"{type(e).__name__}: {e}")
        logger.error(f"Pipeline fallita: {e}", exc_info=True)
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: EXPORT (LaTeX + BibTeX + Provenance + Versioning)
# ═══════════════════════════════════════════════════════════════════════════
def step_export(state: dict) -> bool:
    """Esporta il survey in tutti i formati."""
    logger.info("=" * 60)
    logger.info("STEP 5: EXPORT")
    logger.info("=" * 60)

    details = {}
    errors_count = 0

    # 5a. Versioning
    try:
        from paperhunter.postprocessing import create_version_from_state
        version = create_version_from_state(state)
        details["version"] = version.version_number
        logger.info(f"Versione: {version.version_number}")
    except Exception as e:
        logger.error(f"Versioning fallito: {e}")
        errors_count += 1
        details["version_error"] = str(e)

    # 5b. Provenance
    try:
        from paperhunter.postprocessing import create_provenance_from_state
        tracker = create_provenance_from_state(state)
        prov_path = tracker.save_json()
        tracker.save_markdown()
        prov_stats = tracker.get_stats()
        details["provenance"] = {
            "path": str(prov_path),
            "total_claims": prov_stats.get("total_claims", 0),
            "support_rate": prov_stats.get("support_rate", 0),
        }
        logger.info(f"Provenance: {prov_path} ({prov_stats.get('total_claims', 0)} claims)")
    except Exception as e:
        logger.error(f"Provenance fallita: {e}")
        errors_count += 1
        details["provenance_error"] = str(e)

    # 5c. BibTeX
    try:
        from paperhunter.postprocessing import generate_bib_from_state
        bib_path, bib_report = generate_bib_from_state(state)
        details["bibtex"] = {
            "path": str(bib_path) if bib_path else None,
            "total_filtered": bib_report.get("total_filtered", 0),
            "missing_keys": bib_report.get("total_missing", 0),
        }
        if bib_path:
            logger.info(f"BibTeX: {bib_path} ({bib_report.get('total_filtered', 0)} entry)")
    except Exception as e:
        logger.error(f"BibTeX fallito: {e}")
        errors_count += 1
        details["bibtex_error"] = str(e)

    # 5d. LaTeX
    try:
        from paperhunter.postprocessing import export_latex_from_state, get_export_stats
        tex_path = export_latex_from_state(state, template="acl")
        if tex_path:
            latex_content = tex_path.read_text(encoding="utf-8")
            latex_stats = get_export_stats(latex_content)
            details["latex"] = {
                "path": str(tex_path),
                "lines": latex_stats.get("lines", 0),
                "sections": latex_stats.get("sections", 0),
                "citations": latex_stats.get("unique_cite_keys", 0),
            }
            logger.info(f"LaTeX: {tex_path} ({latex_stats.get('lines', 0)} righe)")
        else:
            details["latex"] = {"path": None}
    except Exception as e:
        logger.error(f"LaTeX fallito: {e}")
        errors_count += 1
        details["latex_error"] = str(e)

    # 5e. Salva survey.md nella directory di test
    try:
        final_paper = state.get("final_paper", "")
        if final_paper:
            md_path = TEST_OUTPUT_DIR / "survey.md"
            TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            md_path.write_text(final_paper, encoding="utf-8")
            details["markdown"] = {
                "path": str(md_path),
                "length_chars": len(final_paper),
            }
            logger.info(f"Markdown: {md_path} ({len(final_paper)} chars)")
    except Exception as e:
        logger.error(f"Markdown save fallito: {e}")

    ok = errors_count == 0
    log_step("export", "OK" if ok else "WARNING", details)
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: FINAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════
def step_final_verification(state: dict) -> bool:
    """Verifica finale di tutti gli artefatti generati."""
    logger.info("=" * 60)
    logger.info("STEP 6: VERIFICA FINALE")
    logger.info("=" * 60)

    checks = {}
    all_ok = True

    # 1. Registry
    try:
        reg_path = Path("data/hash_registry.json")
        reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
        checks["registry"] = {"exists": reg_path.exists(), "entries": len(reg)}
        if len(reg) == 0:
            all_ok = False
    except Exception:
        checks["registry"] = {"exists": False}
        all_ok = False

    # 2. Qdrant collections
    try:
        from paperhunter.indexing import QdrantStore
        store = QdrantStore()
        for coll in ["paper_chunks", "l1_summaries", "l2_summaries"]:
            count = store.get_collection_count(coll)
            checks[f"qdrant_{coll}"] = count
            if count == 0:
                all_ok = False
                report["warnings"].append(f"Qdrant {coll} vuoto")
    except Exception as e:
        checks["qdrant_error"] = str(e)
        all_ok = False

    # 3. BM25
    try:
        bm25_path = Path("data/bm25_index.pkl")
        checks["bm25_exists"] = bm25_path.exists()
        if bm25_path.exists():
            checks["bm25_size_kb"] = round(bm25_path.stat().st_size / 1024, 1)
    except Exception:
        pass

    # 4. PropertyGraph
    try:
        graph_path = Path("data/property_graph.json")
        checks["graph_exists"] = graph_path.exists()
    except Exception:
        pass

    # 5. Master BibTeX
    try:
        bib_path = Path("output/references.bib")
        checks["master_bib_exists"] = bib_path.exists()
        if bib_path.exists():
            content = bib_path.read_text(encoding="utf-8")
            entries = content.count("@")
            checks["master_bib_entries"] = entries
    except Exception:
        pass

    # 6. Final paper
    final_paper = state.get("final_paper", "")
    checks["final_paper_exists"] = bool(final_paper)
    checks["final_paper_chars"] = len(final_paper)
    if final_paper:
        # Count citations in paper
        import re
        citation_re = re.compile(r'\[([A-Z][a-z]+_\d{4}_\w+)\]')
        citations_found = set(citation_re.findall(final_paper))
        checks["citations_in_paper"] = len(citations_found)
        checks["citation_keys"] = sorted(citations_found)[:20]  # Cap at 20

    # 7. Provenance
    prov_dir = Path("output/provenance")
    checks["provenance_dir_exists"] = prov_dir.exists()
    if prov_dir.exists():
        prov_files = list(prov_dir.glob("*.json"))
        checks["provenance_files"] = len(prov_files)

    # 8. Versions
    ver_dir = Path("output/versions")
    checks["versions_dir_exists"] = ver_dir.exists()
    if ver_dir.exists():
        ver_dirs = [d for d in ver_dir.iterdir() if d.is_dir()]
        checks["version_count"] = len(ver_dirs)

    # 9. No critical errors in log
    checks["total_errors"] = len(report.get("errors", []))
    checks["total_warnings"] = len(report.get("warnings", []))

    log_step("final_verification", "OK" if all_ok else "WARNING", checks)

    # Print summary
    logger.info(f"Verifica finale: {'PASS' if all_ok else 'PARTIAL'}")
    for k, v in checks.items():
        logger.info(f"  {k}: {v}")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: RE-INGEST — Verify idempotency
# ═══════════════════════════════════════════════════════════════════════════
def step_reingest_check() -> bool:
    """Verifica che una re-ingestion non processi nulla (idempotenza)."""
    logger.info("=" * 60)
    logger.info("STEP 7: RE-INGEST CHECK (idempotenza)")
    logger.info("=" * 60)

    from paperhunter.ingestion import IngestionPipeline

    try:
        pipeline = IngestionPipeline(
            pdf_base_dir=TEST_PDF_DIR,
            batch_size=5,
            skip_s2=True,
            skip_qdrant=True,
        )
        ing_report = pipeline.run()

        details = {
            "processed": ing_report.processed,
            "skipped": ing_report.skipped,
        }

        ok = ing_report.processed == 0
        log_step("reingest_idempotency", "OK" if ok else "FAILED", details)
        logger.info(f"Re-ingest: {ing_report.processed} processati, {ing_report.skipped} skippati")
        return ok

    except Exception as e:
        log_step("reingest_idempotency", "FAILED", error=str(e))
        return False


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Orchestratore
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print_panel("Test End-to-End Integration", 
                f"20 paper di test, 4 categorie\n"
                f"Pipeline: ingest → summarize → index → write → export\n"
                f"Output: {TEST_OUTPUT_DIR}")

    total_start = time.time()

    # Step 1: Ingestion
    if not step_ingest():
        save_report()
        print_error("Ingestion fallita — abort")
        return 1

    # Step 1b: Verify ingestion
    step_verify_ingestion()

    # Step 2: Re-ingest check (idempotency)
    step_reingest_check()

    # Step 3: L1 Summarization
    if not step_l1_summarization():
        save_report()
        print_warning("L1 summarization ha problemi, continuo comunque...")

    # Step 4: L2 Summarization
    if not step_l2_summarization():
        save_report()
        print_warning("L2 summarization ha problemi, continuo comunque...")

    # Step 5: BM25 Index
    step_bm25_index()

    # Step 6: GraphRAG
    step_graphrag()

    # Step 7: Agentic Writing
    state = step_write()
    if not state or not state.get("final_paper"):
        save_report()
        print_error("Writing pipeline non ha prodotto output — abort export")
        return 1

    # Step 8: Export
    step_export(state)

    # Step 9: Final Verification
    step_final_verification(state)

    # Report finale
    total_elapsed = time.time() - total_start
    report["total_time_seconds"] = round(total_elapsed, 1)

    save_report()

    # Print finale
    print("\n" + "=" * 70)
    print_success(f"Test E2E completato in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    from rich.table import Table
    t = Table(title="Riepilogo Steps", show_lines=True)
    t.add_column("Step", style="bold")
    t.add_column("Status")
    t.add_column("Tempo")
    t.add_column("Note")

    for name, data in report["steps"].items():
        status = data["status"]
        color = "green" if status == "OK" else ("yellow" if status == "WARNING" else "red")
        note = ""
        if data.get("error"):
            note = data["error"][:60]
        elif data.get("details"):
            d = data["details"]
            if "processed" in d:
                note = f"proc={d['processed']}, chunks={d.get('total_chunks', '?')}"
            elif "generated" in d:
                note = f"gen={d['generated']}"
            elif "total_indexed" in d:
                note = f"indexed={d['total_indexed']}"
            elif "entities" in d:
                note = f"ent={d['entities']}, rel={d['relations']}"
            elif "phase" in d:
                note = f"phase={d['phase']}, sections={d.get('total_sections', '?')}"
        t.add_row(name, f"[{color}]{status}[/{color}]", "", note)

    console.print(t)

    if report["errors"]:
        print_warning(f"{len(report['errors'])} errori:")
        for err in report["errors"]:
            print(f"  - {err}")

    print(f"\nReport: {TEST_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
