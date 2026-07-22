"""
PaperHunter — CLI Principale v2.2

Subcommands:
  search      Cerca e scarica nuovi paper (logica legacy PaperHunter)
  ingest      Pipeline di ingestion incrementale (GROBID + S2 + Qdrant)
  summarize   Genera/rigenera summary L1, L1.5 (RAPTOR cluster) e L2
  index       Costruisci/aggiorna indici (BM25 + GraphRAG + Community Detection)
  write       Esegui la pipeline agentica completa (LangGraph, modalità static/agentic)
  query       Query interattiva contro la knowledge base
  stats       Statistiche complete del sistema
  export      Esporta il survey in LaTeX / BibTeX
  cleanup     Elimina paper scaricati per intervallo di date

Novità v2.2:
  - RAPTOR L1.5: clustering gerarchico dei summary L1 con generazione summary di cluster
  - Graph Communities: Louvain/Leiden community detection via Neo4j GDS
  - Agentic RAG: modalità writer ReAct con tool di retrieval (vector, keyword, graph)

Workflow tipico:
  1. paperhunter search -k "prompt injection" -o ./scaricati/Prompt_injections1
  2. paperhunter ingest --pdf-dir ./scaricati
  3. paperhunter summarize                  # L1 + L1.5 + L2
  4. paperhunter index                      # BM25 + Graph + Communities
  5. paperhunter write --writer-mode agentic
  6. paperhunter export --template acl

Esempi:
  paperhunter search -k "prompt injection" -c "Papers about prompt injection attacks" -o ./scaricati/Prompt_injections1
  paperhunter search -k "LLM security" --date-from 2025-10 --date-to 2026-03
  paperhunter ingest
  paperhunter ingest --pdf-dir ./scaricati --batch-size 5
  paperhunter summarize --l1 --l2
  paperhunter summarize --l1-5              # Solo clustering RAPTOR
  paperhunter index --bm25 --graph
  paperhunter index --communities           # Solo community detection
  paperhunter write
  paperhunter write --writer-mode agentic   # ReAct agent con tool di retrieval
  paperhunter write --outline sample_outline.json --resume
  paperhunter query "What defenses exist against prompt injection?"
  paperhunter stats
  paperhunter export --template acl
  paperhunter cleanup --date-from 2023-01 --date-to 2023-12 --dry-run
"""

import argparse
import asyncio
import json
import logging
import re
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Carica .env prima di tutto il resto
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# Lazy imports — caricati solo quando il subcommand corrispondente è invocato,
# per evitare di importare TUTTO all'avvio (pesante).
# ═══════════════════════════════════════════════════════════════════════════

def _init_logging(verbose: bool = False) -> None:
    """Inizializza logging con Rich se disponibile."""
    from paperhunter.config import settings
    level = "DEBUG" if verbose else settings.logging.level
    try:
        from paperhunter.utils.logging_setup import setup_logging
        setup_logging(level=level)
    except Exception:
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        )


def _get_logger(name: str = "paperhunter.cli"):
    """Restituisce un logger, preferendo la versione Rich."""
    try:
        from paperhunter.utils.logging_setup import get_logger
        return get_logger(name)
    except Exception:
        return logging.getLogger(name)


def _get_console():
    """Restituisce la console Rich per output formattato."""
    try:
        from paperhunter.utils.logging_setup import console
        return console
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: search  (logica legacy PaperHunter — asincrona, con checkpoint)
# ═══════════════════════════════════════════════════════════════════════════

def cmd_search(args: argparse.Namespace) -> int:
    """Cerca e scarica paper con checkpoint per ripresa automatica."""
    from paperhunter.config import settings, ensure_download_dir
    from paperhunter.search_engine import PaperAggregator
    from paperhunter.relevance_filter import RelevanceFilter
    from paperhunter.download_manager import Downloader
    from paperhunter.report_generator import generate_report
    from paperhunter.search_checkpoint import SearchCheckpointManager

    logger = _get_logger("paperhunter.search")

    keywords = args.keywords
    criteria = args.criteria
    if not criteria and keywords:
        criteria = f"I want papers related to: {keywords}. Include all relevant research on this topic."
        logger.info(f"Criteria auto-generati: {criteria}")

    max_results = args.max_results or settings.max_results_per_source
    download_dir = Path(args.output_dir) if args.output_dir else ensure_download_dir()
    download_dir.mkdir(parents=True, exist_ok=True)

    # Interattivo se mancano keyword
    if not keywords:
        keywords, criteria = _get_search_input()

    aggregator = PaperAggregator(max_results_per_source=max_results)
    relevance_filter = RelevanceFilter(provider=args.llm_provider)
    downloader = Downloader(download_dir=download_dir)

    # Date range filter (YYYY-MM or YYYY-MM-DD)
    date_from = getattr(args, "date_from", None)
    date_to = getattr(args, "date_to", None)

    # ── Checkpoint setup ──
    ckpt = SearchCheckpointManager()
    search_id = ckpt.compute_search_id(
        keywords, criteria, max_results, str(download_dir), date_from, date_to,
    )

    async def _run():
        data = ckpt.find_checkpoint(search_id)

        # ════════════════════════════════════════════════════════════════
        # Already completed?  Check if there are pending downloads first.
        # ════════════════════════════════════════════════════════════════
        if data and data["phase"] == "complete":
            n_still_relevant = sum(
                1 for p in data["papers"] if p["status"] == "relevant"
            )
            # If there are relevant-but-not-downloaded papers and user
            # did NOT pass --no-download, offer to resume downloads.
            if n_still_relevant > 0 and not args.no_download:
                md_path = ckpt.get_md_path(search_id)
                print("\n" + "═" * 60)
                print(f"  ♻️  Checkpoint trovato (ID: {search_id})")
                print(f"     {n_still_relevant} paper rilevanti non ancora scaricati")
                print(f"     Dettagli: {md_path}")
                print("═" * 60)
                answer = input("\n  Proseguire con il download? [S/n] ").strip().lower()
                if answer in ("n", "no"):
                    return 0
                data["phase"] = "filter_complete"  # reset to allow download
            else:
                n_dl = sum(1 for p in data["papers"] if p["status"] == "downloaded")
                n_dup = sum(1 for p in data["papers"] if p["status"] == "dedup_skipped")
                n_nr = sum(1 for p in data["papers"] if p["status"] == "not_relevant")
                md_path = ckpt.get_md_path(search_id)
                print("\n" + "═" * 60)
                print(f"  ✅ Ricerca già completata (checkpoint: {search_id})")
                print("─" * 60)
                print(f"    Scaricati: {n_dl}  |  Dedup: {n_dup}  |  Non rilevanti: {n_nr}")
                print(f"    Dettagli: {md_path}")
                print("═" * 60)
                print("\n  Per ripetere, elimina il checkpoint e rilancia.")
                return 0

        # ════════════════════════════════════════════════════════════════
        # Resume from existing (incomplete) checkpoint
        # ════════════════════════════════════════════════════════════════
        if data:
            phase = data["phase"]
            md_path = ckpt.get_md_path(search_id)
            n_total = len(data["papers"])

            print("\n" + "═" * 60)
            print(f"  ♻️  Checkpoint trovato per questa ricerca!")
            print(f"     ID: {search_id}  |  {n_total} paper trovati")
            print(f"     File: {md_path}")
            print("═" * 60)

            if phase == "search_complete":
                # Show summary and ask for confirmation (same as fresh)
                _print_source_summary(data)
                if not getattr(args, "yes", False):
                    answer = input("\n  Proseguire con l'analisi LLM e il download? [S/n] ").strip().lower()
                    if answer in ("n", "no"):
                        logger.info("Operazione annullata dall'utente.")
                        return 0
                print()

            elif phase == "filtering":
                done = n_total - len(ckpt.get_pending_filter(data))
                print(f"\n  Ripresa filtro LLM: {done}/{n_total} già analizzati\n")

            elif phase in ("filter_complete", "downloading"):
                pending = ckpt.get_pending_download(data)
                n_rel = sum(1 for p in data["papers"] if p["status"] in (
                    "relevant", "downloaded", "download_failed", "dedup_skipped"))
                done_dl = n_rel - len(pending)
                print(f"\n  Ripresa download: {done_dl}/{n_rel} già completati\n")

        # ════════════════════════════════════════════════════════════════
        # PHASE 1 — Fresh search (only when no checkpoint exists)
        # ════════════════════════════════════════════════════════════════
        else:
            logger.info(f"Ricerca con keyword: '{keywords}'")
            if date_from or date_to:
                logger.info(f"Filtro date: {date_from or '*'} → {date_to or '*'}")
            search_result = await aggregator.search(
                keywords, date_from=date_from, date_to=date_to,
            )
            logger.info(
                f"Trovati {search_result.total_found} paper "
                f"da {len(search_result.sources_queried)} fonti"
            )

            if not search_result.papers:
                logger.warning("Nessun paper trovato. Prova keyword diversi.")
                return 1

            # Build source counts for checkpoint
            from collections import Counter
            source_counts = Counter(
                p.source.value if hasattr(p.source, "value") else str(p.source)
                for p in search_result.papers
            )
            params = {
                "keywords": keywords,
                "criteria": criteria,
                "max_results": max_results,
                "output_dir": str(download_dir),
                "date_from": date_from,
                "date_to": date_to,
            }
            search_meta = {
                "total_found": search_result.total_found,
                "sources_queried": search_result.sources_queried,
                "errors": search_result.errors,
                "source_counts": dict(source_counts),
            }
            data = ckpt.create_from_search(
                search_id, params, search_result.papers, search_meta,
            )
            md_path = ckpt.get_md_path(search_id)
            logger.info(f"💾 Checkpoint salvato: {md_path}")

            # Show source summary and confirm
            _print_source_summary(data)

            if not getattr(args, "yes", False):
                answer = input("\n  Proseguire con l'analisi LLM e il download? [S/n] ").strip().lower()
                if answer in ("n", "no"):
                    logger.info("Operazione annullata dall'utente.")
                    return 0
            print()

        # ════════════════════════════════════════════════════════════════
        # PHASE 2 — LLM Relevance Filtering (per-paper checkpoint)
        # ════════════════════════════════════════════════════════════════
        if data["phase"] in ("search_complete", "filtering"):
            ckpt.set_phase(data, "filtering")
            pending = ckpt.get_pending_filter(data)
            n_total = len(data["papers"])

            if pending:
                done = n_total - len(pending)
                if done > 0:
                    logger.info(f"Filtro LLM: ripresa da {done}/{n_total}")
                logger.info(f"Filtro con criteri: {criteria[:100]}...")

                for i, idx in enumerate(pending):
                    paper = ckpt.reconstruct_paper(data["papers"][idx])
                    result = await relevance_filter.analyze_paper(paper, criteria)

                    if result.relevance.reasoning.startswith("Analysis failed"):
                        status = "filter_error"
                    elif result.relevance.is_relevant:
                        status = "relevant"
                    else:
                        status = "not_relevant"

                    ckpt.update_paper(
                        data, idx, status,
                        relevance_reasoning=result.relevance.reasoning,
                        relevance_confidence=result.relevance.confidence,
                    )
                    tag = (
                        "✓ RILEVANTE" if status == "relevant"
                        else ("✗ non rilevante" if status == "not_relevant"
                              else "⚠ ERRORE")
                    )
                    logger.info(
                        f"  [{done + i + 1}/{n_total}] {tag} — {paper.title[:60]}"
                    )

                    # Rate limiting between LLM calls
                    if i < len(pending) - 1:
                        await asyncio.sleep(1.5)

            ckpt.set_phase(data, "filter_complete")

        # ── Count relevant papers ──
        n_relevant = sum(
            1 for p in data["papers"]
            if p["status"] in ("relevant", "downloaded", "download_failed", "dedup_skipped")
        )
        logger.info(f"{n_relevant} paper rilevanti trovati")

        if n_relevant == 0:
            logger.warning("Nessun paper rilevante. Prova ad aggiustare i criteri.")
            ckpt.set_phase(data, "complete")
            return 1

        # ════════════════════════════════════════════════════════════════
        # PHASE 3 — Download (sequential, per-paper checkpoint)
        # ════════════════════════════════════════════════════════════════
        if not args.no_download and data["phase"] in ("filter_complete", "downloading"):
            ckpt.set_phase(data, "downloading")
            pending_dl = ckpt.get_pending_download(data)

            if pending_dl:
                already_done = n_relevant - len(pending_dl)
                if already_done > 0:
                    logger.info(f"Download: ripresa da {already_done}/{n_relevant}")
                logger.info(f"Download di {len(pending_dl)} paper rilevanti...")

                for i, idx in enumerate(pending_dl):
                    paper = ckpt.reconstruct_paper(data["papers"][idx])
                    result = await downloader.download_paper(paper)

                    if result.success:
                        if result.skipped:
                            new_status = "dedup_skipped"
                        else:
                            new_status = "downloaded"
                        ckpt.update_paper(
                            data, idx, new_status,
                            download_path=result.file_path,
                        )
                    else:
                        ckpt.update_paper(
                            data, idx, "download_failed",
                            download_error=result.error_message,
                        )

                    logger.info(
                        f"  [{already_done + i + 1}/{n_relevant}] "
                        f"{data['papers'][idx]['status'].upper()} — "
                        f"{paper.title[:50]}"
                    )
                    await asyncio.sleep(0.5)

                ok = sum(
                    1 for p in data["papers"]
                    if p["status"] in ("downloaded", "dedup_skipped")
                )
                logger.info(f"Scaricati {ok}/{n_relevant} paper rilevanti")

        # ════════════════════════════════════════════════════════════════
        # PHASE 4 — Report + finalise
        # ════════════════════════════════════════════════════════════════
        ckpt.set_phase(data, "complete")

        # Reconstruct data for standard report
        search_result, filtered, download_results, relevant = (
            ckpt.reconstruct_for_report(data)
        )
        report_path = generate_report(
            keywords=keywords, criteria=criteria,
            search_result=search_result, filtered_results=filtered,
            download_results=download_results, relevant_papers=relevant,
            output_dir=download_dir, max_results_per_source=max_results,
            categories=settings.default_categories,
            llm_provider=args.llm_provider, llm_model=None,
        )
        logger.info(f"Report salvato: {report_path}")

        md_path = ckpt.get_md_path(search_id)
        logger.info(f"📋 Checkpoint finale: {md_path}")
        return 0

    return asyncio.run(_run())


def _get_search_input() -> tuple[str, str]:
    """Input interattivo per la ricerca."""
    print("\n" + "=" * 60)
    print("  PaperHunter — Ricerca Paper Scientifici")
    print("=" * 60)
    print("\nInserisci le keyword di ricerca (es. 'LLM security', 'prompt injection'):")
    keywords = input("  Keywords: ").strip() or "LLM security"
    print("\nInserisci i criteri di filtro LLM:")
    criteria = input("  Criteri: ").strip()
    if not criteria:
        criteria = f"I want papers related to: {keywords}."
    return keywords, criteria


def _print_source_summary(data: dict) -> None:
    """Print per-source paper counts from checkpoint data."""
    source_counts = data.get("search_meta", {}).get("source_counts", {})
    errors = data.get("search_meta", {}).get("errors", [])
    total = len(data["papers"])

    source_labels = {
        "arxiv": "ArXiv",
        "semantic_scholar": "Semantic Scholar",
        "huggingface": "HuggingFace",
    }

    print("\n" + "═" * 60)
    print(f"  📄 Trovati {total} paper unici")
    print("─" * 60)
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        label = source_labels.get(src, src)
        print(f"    • {label:<22s} {count:>4d} paper")
    if errors:
        print("─" * 60)
        for err in errors:
            print(f"    ⚠  {err}")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: ingest
# ═══════════════════════════════════════════════════════════════════════════

def cmd_ingest(args: argparse.Namespace) -> int:
    """Esegue la pipeline di ingestion incrementale."""
    from paperhunter.config import settings
    from paperhunter.ingestion import IngestionPipeline

    logger = _get_logger("paperhunter.ingest")

    pdf_dir = args.pdf_dir or settings.ingestion.pdf_base_dir
    pdf_path = Path(pdf_dir)

    if not pdf_path.is_dir():
        logger.error(f"Directory PDF non trovata: {pdf_path}")
        return 1

    logger.info(f"Ingestion pipeline — directory: {pdf_path}")

    pipeline = IngestionPipeline(
        pdf_base_dir=pdf_path,
        batch_size=args.batch_size or settings.ingestion.batch_size,
        skip_s2=args.skip_s2,
        skip_qdrant=args.skip_qdrant,
        use_checkpoint=not args.no_checkpoint,
        force=args.force,
    )

    report = pipeline.run()

    console = _get_console()
    if console:
        try:
            from paperhunter.utils.logging_setup import print_success
            print_success(
                f"Ingestion completata: {report.processed} processati, "
                f"{report.skipped} skippati, {report.errors} errori, "
                f"{report.total_chunks} chunk generati in {report.total_time_seconds:.1f}s"
            )
        except Exception:
            pass
    else:
        print(f"\nIngestion completata: {report.processed} processati, "
              f"{report.skipped} skippati, {report.errors} errori")

    return 0 if report.errors == 0 else 1


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: summarize
# ═══════════════════════════════════════════════════════════════════════════

def cmd_summarize(args: argparse.Namespace) -> int:
    """Genera/rigenera summary L1, L1.5 (cluster RAPTOR) e/o L2."""
    from paperhunter.indexing import QdrantStore

    logger = _get_logger("paperhunter.summarize")

    has_l1_5 = getattr(args, "l1_5", False)
    # Se nessun flag: genera tutti e tre i livelli
    no_flag = not args.l1 and not has_l1_5 and not args.l2 and not args.invalidate
    do_l1 = args.l1 or no_flag
    do_l1_5 = has_l1_5 or no_flag
    do_l2 = args.l2 or no_flag
    do_invalidate = args.invalidate

    store = QdrantStore()
    errors = 0

    if do_l1:
        logger.info("Generazione summary L1 (singolo paper)...")
        errors += _run_l1_summarization(store, args, logger)

    if do_l1_5:
        logger.info("Generazione summary L1.5 (cluster RAPTOR)...")
        l1_5_errors, _new_l1_5 = _run_l1_5_summarization(store, args, logger)
        errors += l1_5_errors

    if do_l2:
        logger.info("Generazione summary L2 (meta-summary per categoria)...")
        # skip_existing=not force: genera L2 mancanti o stale, salta quelli validi
        errors += _run_l2_summarization(store, args, logger)

    if do_invalidate:
        logger.info("Check invalidazione e rigenerazione L2...")
        errors += _run_invalidation(store, args, logger)

    return 0 if errors == 0 else 1


def _run_l1_summarization(store, args, logger) -> int:
    """Genera L1 per i paper nel vector store che non hanno ancora un L1."""
    from paperhunter.models import Paper, PaperChunk
    from paperhunter.summarization import generate_l1_summary

    try:
        # Paper con chunk nel vector store
        paper_keys = set()
        offset = None
        while True:
            results = store.client.scroll(
                collection_name=store.COLLECTION_CHUNKS, limit=100, offset=offset,
                with_payload=True, with_vectors=False,
            )
            points, next_offset = results
            if not points:
                break
            for pt in points:
                bk = pt.payload.get("bibtex_key", "")
                pid = pt.payload.get("paper_id", "")
                if bk:
                    paper_keys.add((bk, pid))
            if next_offset is None:
                break
            offset = next_offset

        if not paper_keys:
            logger.warning("Nessun paper nel vector store. Esegui prima 'paperhunter ingest'.")
            return 0

        # L1 già esistenti
        existing_l1 = set()
        offset = None
        while True:
            results = store.client.scroll(
                collection_name=store.COLLECTION_L1, limit=100, offset=offset,
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

        missing = [(bk, pid) for bk, pid in paper_keys if bk not in existing_l1]

        if not missing:
            logger.info(f"Tutti {len(paper_keys)} paper hanno già un L1. Niente da fare.")
            return 0

        logger.info(f"{len(missing)} paper senza L1 (di {len(paper_keys)} totali)")

        generated = 0
        errors_count = 0
        for bk, pid in missing:
            try:
                chunks_data = store.get_paper_chunks(bk, collection_name=store.COLLECTION_CHUNKS)
                if not chunks_data:
                    logger.warning(f"Nessun chunk per {bk}, skip")
                    continue

                meta = chunks_data[0].get("metadata", {})
                _source = meta.get("source", "arxiv")
                if _source not in ("arxiv", "semantic_scholar", "huggingface"):
                    _source = "arxiv"
                paper = Paper(
                    title=meta.get("paper_title", bk), source=_source,
                    url="", bibtex_key=bk, category=meta.get("category", ""),
                    year=meta.get("year"), pdf_hash=meta.get("pdf_hash", ""),
                )

                chunks = []
                for cd in chunks_data:
                    m = cd.get("metadata", {})
                    chunks.append(PaperChunk(
                        chunk_id=cd.get("chunk_id", ""), text=cd.get("text", ""),
                        paper_id=m.get("paper_id", pid), bibtex_key=bk,
                        section_name=m.get("section_name", ""),
                        chunk_index=m.get("chunk_index", 0),
                        category=m.get("category", ""), year=m.get("year"),
                    ))

                summary = generate_l1_summary(paper, chunks, save_to_store=True, store=store)
                if summary:
                    generated += 1
                    logger.info(f"  L1 generato per {bk} ({generated}/{len(missing)})")
            except Exception as e:
                logger.error(f"  Errore L1 per {bk}: {e}")
                errors_count += 1

        logger.info(f"L1: {generated} generati, {errors_count} errori")
        return errors_count

    except Exception as e:
        logger.error(f"Errore L1 summarization: {e}")
        return 1


# ── Helpers per staleness detection L1.5/L2 ──

_L1_5_STATE_DIR = Path("output/l1_5_metadata")


def _compute_l1_hash(l1_summaries) -> str:
    """Hash MD5 dei bibtex_key ordinati — rileva cambi nel set L1."""
    import hashlib
    keys = sorted(getattr(s, "bibtex_key", "") or getattr(s, "paper_id", "") for s in l1_summaries)
    return hashlib.md5(",".join(keys).encode()).hexdigest()


def _is_l1_5_stale(category: str, l1_summaries, logger) -> bool:
    """Verifica se gli L1.5 esistenti sono basati su un set L1 diverso dall'attuale."""
    import re as _re
    safe = _re.sub(r"[^a-z0-9_]", "_", category.lower())
    state_file = _L1_5_STATE_DIR / f"{safe}.json"
    if not state_file.exists():
        return False  # nessuno stato salvato → non possiamo dire che è stale
    try:
        state = json.loads(state_file.read_text())
        stored_hash = state.get("l1_hash", "")
        current_hash = _compute_l1_hash(l1_summaries)
        if stored_hash and stored_hash != current_hash:
            logger.debug(
                f"L1.5 stale per '{category}': l1_hash {stored_hash[:8]}→{current_hash[:8]}"
            )
            return True
    except Exception:
        pass
    return False


def _save_l1_5_state(category: str, l1_summaries) -> None:
    """Salva lo stato L1 usato per generare L1.5, per rilevare staleness."""
    import re as _re
    from datetime import datetime
    _L1_5_STATE_DIR.mkdir(parents=True, exist_ok=True)
    safe = _re.sub(r"[^a-z0-9_]", "_", category.lower())
    state = {
        "category": category,
        "l1_count": len(l1_summaries),
        "l1_hash": _compute_l1_hash(l1_summaries),
        "generated_at": datetime.now().isoformat(),
    }
    (_L1_5_STATE_DIR / f"{safe}.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False)
    )


def _invalidate_l2_metadata(category: str, logger) -> None:
    """Marca il metadata L2 su disco come invalidato (L1.5 rigenerati)."""
    import re as _re
    from paperhunter.config import settings
    l2_dir = Path(settings.output.base_dir) / "l2_metadata"
    safe = _re.sub(r"[^a-z0-9_]", "_", category.lower())
    filepath = l2_dir / f"{safe}.json"
    if not filepath.exists():
        return
    try:
        meta = json.loads(filepath.read_text())
        if not meta.get("invalidated", False):
            meta["invalidated"] = True
            filepath.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            logger.info(f"  L2 '{category}' invalidato (L1.5 aggiornati)")
    except Exception as e:
        logger.debug(f"Errore invalidazione L2 metadata per '{category}': {e}")


def _run_l1_5_summarization(store, args, logger) -> tuple[int, bool]:
    """Genera L1.5 (cluster RAPTOR) per le categorie che hanno L1 ma non L1.5.

    Returns:
        Tupla (errori, nuovi_generati): numero errori e flag se nuovi L1.5
        sono stati generati (utile per forzare rigenerazione L2).
    """
    from paperhunter.summarization import (
        load_l1_summaries_for_category,
        generate_all_cluster_summaries,
        count_l1_5_for_category,
    )
    from paperhunter.summarization.clustering_utils import (
        cluster_summaries,
        fetch_l1_embeddings_from_qdrant,
    )

    try:
        # Scopri categorie con L1
        categories = set()
        offset = None
        while True:
            results = store.client.scroll(
                collection_name=store.COLLECTION_L1, limit=100, offset=offset,
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
            logger.warning("Nessun L1 trovato. Esegui prima 'paperhunter summarize --l1'.")
            return 0, False

        category_filter = getattr(args, "category", None)
        if category_filter:
            categories = {c for c in categories if c == category_filter}
            if not categories:
                logger.warning(f"Nessun L1 trovato per la categoria '{category_filter}'.")
                return 0, False

        force = getattr(args, "force", False)
        total_generated = 0
        errors_count = 0

        for cat in sorted(categories):
            try:
                # Carica L1 della categoria
                l1_summaries = load_l1_summaries_for_category(cat, store=store)
                if not l1_summaries:
                    logger.warning(f"  {cat}: nessun L1 ricostruito, skip")
                    continue

                # Recupera embedding e clusterizza (serve per sapere quanti cluster attesi)
                embeddings = fetch_l1_embeddings_from_qdrant(l1_summaries, qdrant_store=store)
                clusters = cluster_summaries(l1_summaries, embeddings)
                expected_clusters = len(clusters)

                # Controlla L1.5 già presenti vs attesi
                existing = count_l1_5_for_category(cat, store=store)

                # Verifica staleness: confronta hash L1 con quello salvato
                l1_stale = _is_l1_5_stale(cat, l1_summaries, logger)

                if existing >= expected_clusters and not force and not l1_stale:
                    logger.info(
                        f"  {cat}: {existing}/{expected_clusters} L1.5 già completi, "
                        f"niente da fare (usa --force per rigenerare)"
                    )
                    continue

                if l1_stale and not force:
                    logger.info(
                        f"  {cat}: L1 cambiati ({len(l1_summaries)} paper) → "
                        f"rigenerazione L1.5 necessaria"
                    )
                elif existing > 0 and not force:
                    logger.info(
                        f"  {cat}: {existing}/{expected_clusters} L1.5 presenti, "
                        f"ripresa generazione cluster mancanti..."
                    )
                else:
                    logger.info(f"  {cat}: {len(l1_summaries)} L1 → {expected_clusters} cluster")

                # Se L1 stale, rigenera tutto (non skip_existing)
                skip_existing_l1_5 = not force and not l1_stale

                # Genera L1.5 per ogni cluster
                l1_5_list = generate_all_cluster_summaries(
                    clusters, category=cat,
                    save_to_store=True, store=store,
                    skip_existing=skip_existing_l1_5,
                )
                total_generated += len(l1_5_list)
                logger.info(f"  {cat}: {len(l1_5_list)} L1.5 generati")

                # Salva stato L1 per rilevare future modifiche
                _save_l1_5_state(cat, l1_summaries)

                # Invalida L2 per questa categoria (L1.5 rinfrescati)
                _invalidate_l2_metadata(cat, logger)

            except Exception as e:
                logger.error(f"  Errore L1.5 per {cat}: {e}")
                errors_count += 1

        logger.info(f"L1.5: {total_generated} generati per {len(categories)} categorie, {errors_count} errori")
        return errors_count, total_generated > 0

    except Exception as e:
        logger.error(f"Errore L1.5 summarization: {e}")
        return 1, False


def _run_l2_summarization(store, args, logger, *, skip_existing: bool | None = None) -> int:
    """Genera L2 per tutte le categorie."""
    from paperhunter.summarization import generate_l2_for_all_categories

    try:
        category = getattr(args, "category", None)
        categories = [category] if category else None

        if skip_existing is None:
            skip_existing = not args.force

        summaries = generate_l2_for_all_categories(
            save_to_store=True, store=store,
            skip_existing=skip_existing, categories=categories,
        )
        logger.info(f"L2: {len(summaries)} categorie generate")
        return 0
    except Exception as e:
        logger.error(f"Errore L2 summarization: {e}")
        return 1


def _run_invalidation(store, args, logger) -> int:
    """Check invalidazione e rigenerazione L2."""
    from paperhunter.summarization import InvalidationManager

    try:
        mgr = InvalidationManager(store=store)
        category = getattr(args, "category", None)
        categories = [category] if category else None

        report = mgr.check_and_regenerate(categories=categories, save_to_store=True)
        logger.info(
            f"Invalidazione: {len(report.invalidated_categories)} invalidate, "
            f"{len(report.regenerated_categories)} rigenerate, "
            f"{len(report.failed_categories)} fallite"
        )
        return len(report.failed_categories)
    except Exception as e:
        logger.error(f"Errore invalidazione: {e}")
        return 1


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: index
# ═══════════════════════════════════════════════════════════════════════════

def cmd_index(args: argparse.Namespace) -> int:
    """Costruisce/aggiorna gli indici (BM25 + GraphRAG + Communities)."""
    logger = _get_logger("paperhunter.index")

    has_communities = getattr(args, "communities", False)
    no_flag = not args.bm25 and not args.graph and not has_communities
    do_bm25 = args.bm25 or no_flag
    do_graph = args.graph or no_flag
    do_communities = has_communities or no_flag
    errors = 0

    if do_bm25:
        errors += _build_bm25_index(logger)
    if do_graph:
        errors += _build_graph_index(args, logger)
    if do_communities:
        errors += _build_communities(args, logger)

    return 0 if errors == 0 else 1


def _build_bm25_index(logger) -> int:
    """Costruisce/ricostruisce l'indice BM25 da tutti i chunk nel QdrantStore."""
    from paperhunter.indexing import QdrantStore, BM25Index
    from paperhunter.models import PaperChunk

    try:
        logger.info("Costruzione indice BM25...")
        store = QdrantStore()
        bm25 = BM25Index()

        all_chunks = []
        for collection in ["paper_chunks", "l1_summaries", "l1_5_summaries", "l2_summaries", "graph_communities"]:
            offset = None
            while True:
                results = store.client.scroll(
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
            logger.warning("Nessun chunk trovato. Esegui prima 'paperhunter ingest'.")
            return 0

        count = bm25.build_index(all_chunks, show_progress=True)
        bm25.save()
        logger.info(f"BM25 costruito: {count} documenti indicizzati")
        return 0

    except Exception as e:
        logger.error(f"Errore costruzione BM25: {e}")
        return 1


def _build_graph_index(args, logger) -> int:
    """Costruisce/aggiorna il PropertyGraph da L1 summaries."""
    from paperhunter.config import settings

    if not settings.neo4j.enabled:
        logger.info("Neo4j disabilitato in config.yaml — solo PropertyGraph locale.")

    try:
        from paperhunter.graphrag import PropertyGraphBuilder
        from paperhunter.summarization import load_l1_summaries_for_category
        from paperhunter.indexing import QdrantStore

        store = QdrantStore()

        # Scopri categorie con L1
        categories = set()
        offset = None
        while True:
            results = store.client.scroll(
                collection_name=store.COLLECTION_L1, limit=100, offset=offset,
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
            logger.warning("Nessun L1 trovato. Esegui prima 'paperhunter summarize --l1'.")
            return 0

        logger.info(f"PropertyGraph — {len(categories)} categorie")

        all_l1 = []
        for cat in sorted(categories):
            l1s = load_l1_summaries_for_category(cat, store=store)
            all_l1.extend(l1s)
            logger.info(f"  {cat}: {len(l1s)} L1")

        if not all_l1:
            logger.warning("Nessun L1 ricostruito.")
            return 0

        builder = PropertyGraphBuilder()
        report = builder.build_from_summaries(all_l1)
        builder.save()
        logger.info(
            f"PropertyGraph: {report.entities_added} entità, "
            f"{report.relations_added} relazioni in {report.elapsed_seconds:.1f}s"
        )

        # Import Neo4j opzionale
        if settings.neo4j.enabled and not args.skip_neo4j:
            try:
                from paperhunter.graphrag import Neo4jManager

                # Se non ci sono nuove entità ma il graph store ha dati,
                # esportiamo dal graph store locale per il bulk import
                entities_to_import = report.entities
                relations_to_import = report.relations
                if not entities_to_import and builder.entity_count > 0:
                    entities_to_import, relations_to_import = (
                        builder.export_entities_and_relations()
                    )
                    logger.info(
                        f"Neo4j import: uso {len(entities_to_import)} entità "
                        f"e {len(relations_to_import)} relazioni da graph store"
                    )

                with Neo4jManager() as mgr:
                    stats_neo = mgr.import_entities_and_relations(
                        entities_to_import, relations_to_import,
                        clear_before=args.clear_graph,
                    )
                    logger.info(
                        f"Neo4j: {stats_neo['entities_created']} entità, "
                        f"{stats_neo['relations_created']} relazioni"
                    )
            except Exception as e:
                logger.warning(f"Import Neo4j fallito (non bloccante): {e}")

        return 0

    except Exception as e:
        logger.error(f"Errore costruzione GraphRAG: {e}")
        return 1


def _build_communities(args, logger) -> int:
    """Rileva comunità nel grafo Neo4j e genera summary testuali."""
    from paperhunter.config import settings

    if not settings.neo4j.enabled:
        logger.info("Neo4j disabilitato — community detection richiede Neo4j con GDS. Skip.")
        return 0

    try:
        from paperhunter.graphrag import (
            Neo4jManager, CommunityDetector, GDSNotAvailableError,
        )
        from paperhunter.graphrag.community_summarizer import (
            CommunitySummarizer, count_community_summaries,
        )
        from paperhunter.indexing import QdrantStore

        algorithm = getattr(args, "community_algorithm", None) or "louvain"
        min_size = getattr(args, "community_min_size", None) or 3

        with Neo4jManager() as mgr:
            detector = CommunityDetector(mgr)

            logger.info(f"Community detection ({algorithm}, min_size={min_size})...")
            communities = detector.detect_communities(
                algorithm=algorithm, min_size=min_size,
            )
            logger.info(f"Trovate {len(communities)} comunità")

            if not communities:
                logger.info("Nessuna comunità trovata. Verifica il grafo con 'paperhunter index --graph'.")
                return 0

            # Genera summary testuali e salva in Qdrant
            store = QdrantStore()
            summarizer = CommunitySummarizer(detector, store)

            logger.info("Generazione community summaries...")
            results = summarizer.summarize_all_communities(communities)

            saved = summarizer.save_to_qdrant(results)
            logger.info(
                f"Community: {len(results)} summary generati, "
                f"{saved} salvati in Qdrant"
            )

            stats = detector.get_community_stats()
            if stats.get("total_communities"):
                logger.info(
                    f"Statistiche: {stats['total_communities']} comunità, "
                    f"modularity={stats.get('modularity', 'N/A')}"
                )

        return 0

    except GDSNotAvailableError:
        logger.warning(
            "Plugin GDS non disponibile in Neo4j. "
            "Installa con NEO4J_PLUGINS='[\"graph-data-science\"]'. Skip."
        )
        return 0
    except Exception as e:
        logger.error(f"Errore community detection: {e}")
        return 1


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: write
# ═══════════════════════════════════════════════════════════════════════════

def cmd_write(args: argparse.Namespace) -> int:
    """Esegue la pipeline agentica completa (LangGraph)."""
    logger = _get_logger("paperhunter.write")

    try:
        if args.resume:
            return _write_resume(args, logger)

        from paperhunter.agents import run_pipeline

        outline_file = getattr(args, "outline", None)
        metadata = {}
        if outline_file:
            p = Path(outline_file)
            if not p.exists():
                logger.error(f"Outline non trovato: {p}")
                return 1
            metadata["outline_file"] = str(p)
            logger.info(f"Outline da file: {p}")

        # Override writer_mode se specificato da CLI
        writer_mode = getattr(args, "writer_mode", None)
        if writer_mode:
            from paperhunter.config import settings
            settings.agents.writer_mode = writer_mode
            logger.info(f"Writer mode override: {writer_mode}")

        # --force: riscrittura forzata da zero
        if getattr(args, "force", False):
            from paperhunter.config import settings
            settings.agents.force_rewrite = True
            from paperhunter.agents.writer_agent import clean_section_checkpoints
            removed = clean_section_checkpoints()
            logger.info(f"Force rewrite: eliminati {removed} checkpoint sezione writer")

        logger.info("Avvio pipeline (Planner → Writer → Critic → Assembler)...")
        start = time.time()

        state = run_pipeline(
            outline_file=outline_file,
            metadata=metadata,
            save_checkpoints=not args.no_checkpoint,
        )
        elapsed = time.time() - start

        from paperhunter.agents import get_pipeline_summary
        summary = get_pipeline_summary(state)

        console = _get_console()
        if console:
            try:
                from paperhunter.utils.logging_setup import print_success, print_panel
                print_success(f"Pipeline completata in {elapsed:.0f}s")
                print_panel("Pipeline Summary", json.dumps(summary, indent=2, default=str))
            except Exception:
                pass
        else:
            logger.info(f"Pipeline completata in {elapsed:.0f}s")

        if not args.no_export:
            _auto_export(state, args, logger)

        return 0 if summary.get("phase") == "COMPLETED" else 1

    except Exception as e:
        logger.error(f"Errore pipeline: {e}", exc_info=True)
        return 1


def _write_resume(args, logger) -> int:
    """Riprende la pipeline da un checkpoint."""
    from paperhunter.agents import (
        run_pipeline_from_checkpoint, get_latest_checkpoint, get_pipeline_summary,
    )

    checkpoint_path = getattr(args, "checkpoint_file", None)
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
    else:
        checkpoint_path = get_latest_checkpoint()

    if checkpoint_path is None:
        logger.error("Nessun checkpoint trovato. Esegui 'write' senza --resume.")
        return 1

    logger.info(f"Resume da checkpoint: {checkpoint_path}")
    start = time.time()
    state = run_pipeline_from_checkpoint(str(checkpoint_path), save_checkpoints=True)
    elapsed = time.time() - start

    summary = get_pipeline_summary(state)
    logger.info(f"Resume completato in {elapsed:.0f}s")

    if not args.no_export:
        _auto_export(state, args, logger)

    return 0 if summary.get("phase") == "COMPLETED" else 1


def _auto_export(state, args, logger) -> None:
    """Export automatico post-pipeline (LaTeX + BibTeX + provenance + versioning)."""
    from paperhunter.config import settings

    try:
        # BibTeX PRIMA di LaTeX: così export_latex_from_state trova
        # references_final.bib aggiornato e non strippa le citazioni valide.
        bib_path = None
        if settings.output.generate_bibtex:
            from paperhunter.postprocessing import generate_bib_from_state
            bib_path, bib_report = generate_bib_from_state(state)
            if bib_path:
                logger.info(f"BibTeX: {bib_path} ({bib_report.get('total_filtered', 0)} entry)")

        if settings.output.generate_latex:
            from paperhunter.postprocessing import export_latex_from_state
            template = getattr(args, "template", None) or settings.output.latex_template
            tex_path = export_latex_from_state(
                state, template=template,
                master_bib_path=bib_path,
                strict_citations=settings.citability.final_latex_fail_on_error,
            )
            if tex_path:
                logger.info(f"LaTeX: {tex_path}")

        if settings.output.generate_provenance_json:
            from paperhunter.postprocessing import create_provenance_from_state
            tracker = create_provenance_from_state(state)
            prov_path = tracker.save_json()
            logger.info(f"Provenance: {prov_path}")

        from paperhunter.postprocessing import create_version_from_state
        version = create_version_from_state(state)
        logger.info(f"Versione: {version.version_number}")

    except Exception as e:
        logger.error(f"Export finale bloccato: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: query
# ═══════════════════════════════════════════════════════════════════════════

def cmd_query(args: argparse.Namespace) -> int:
    """Query interattiva contro la knowledge base."""
    from paperhunter.indexing import HybridRetriever, GeminiReranker

    logger = _get_logger("paperhunter.query")
    retriever = HybridRetriever()
    reranker = GeminiReranker() if not args.no_rerank else None

    question = args.question
    interactive = not question

    def _do_query(q: str, n_results: int = 10) -> None:
        cat_filter = getattr(args, "category", None)
        year_range = None
        yr_min = getattr(args, "year_min", None)
        if yr_min:
            year_range = (yr_min, getattr(args, "year_max", None) or 2030)

        results = retriever.retrieve(
            q, n_results=n_results * 2,
            category_filter=cat_filter, year_range=year_range,
        )

        if reranker and results:
            results = reranker.rerank(q, results, top_k=n_results)
        else:
            results = results[:n_results]

        if not results:
            print("\nNessun risultato trovato.\n")
            return

        console = _get_console()
        if console:
            _print_query_results_rich(console, q, results)
        else:
            _print_query_results_plain(q, results)

    if interactive:
        print("\n  Query interattiva — digita 'quit' per uscire\n")
        try:
            while True:
                q = input("  Domanda: ").strip()
                if q.lower() in ("quit", "exit", "q", ""):
                    break
                _do_query(q, n_results=args.n_results)
        except (KeyboardInterrupt, EOFError):
            print("\n")
    else:
        _do_query(question, n_results=args.n_results)

    return 0


def _print_query_results_rich(console, q: str, results: list) -> None:
    """Stampa risultati query con Rich table."""
    from rich.table import Table
    table = Table(title=f"Risultati per: {q[:80]}", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Paper", style="cyan", max_width=30)
    table.add_column("Sezione", max_width=20)
    table.add_column("Score", style="green", width=6)
    table.add_column("Testo", max_width=60)

    for i, r in enumerate(results, 1):
        meta = r.metadata if hasattr(r, "metadata") else r.get("metadata", {})
        bk = meta.get("bibtex_key", "?")
        sect = meta.get("section_name", "?")
        score = f"{r.score:.3f}" if hasattr(r, "score") else f"{r.get('score', 0):.3f}"
        text = (r.text if hasattr(r, "text") else r.get("text", ""))[:150] + "..."
        table.add_row(str(i), bk, sect, score, text)

    console.print(table)


def _print_query_results_plain(q: str, results: list) -> None:
    """Stampa risultati query in plain text."""
    print(f"\n--- Risultati per: {q[:80]} ---")
    for i, r in enumerate(results, 1):
        meta = r.metadata if hasattr(r, "metadata") else r.get("metadata", {})
        bk = meta.get("bibtex_key", "?")
        score = r.score if hasattr(r, "score") else r.get("score", 0)
        text = (r.text if hasattr(r, "text") else r.get("text", ""))[:200]
        print(f"\n[{i}] [{bk}] (score: {score:.3f})")
        print(f"    {text}...")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: stats
# ═══════════════════════════════════════════════════════════════════════════

def cmd_stats(args: argparse.Namespace) -> int:
    """Mostra statistiche complete del sistema."""
    logger = _get_logger("paperhunter.stats")
    console = _get_console()

    try:
        from rich.table import Table
        use_rich = console is not None
    except ImportError:
        use_rich = False

    stats_data = {}

    # 1. Hash registry
    try:
        from paperhunter.ingestion.pdf_hasher import load_hash_registry, get_registry_stats
        registry = load_hash_registry()
        stats_data["ingestion"] = get_registry_stats(registry)
    except Exception:
        stats_data["ingestion"] = None

    # 2. Qdrant
    try:
        from paperhunter.indexing import QdrantStore
        store = QdrantStore()
        stats_data["qdrant"] = store.get_stats()
    except Exception:
        stats_data["qdrant"] = None

    # 3. BM25
    try:
        from paperhunter.indexing import BM25Index
        bm25 = BM25Index()
        bm25.load()
        stats_data["bm25"] = bm25.get_stats()
    except Exception:
        stats_data["bm25"] = None

    # 4. PropertyGraph
    try:
        from paperhunter.graphrag import PropertyGraphBuilder
        builder = PropertyGraphBuilder()
        builder.load()
        stats_data["graph"] = builder.get_stats()
    except Exception:
        stats_data["graph"] = None

    # 5. Versioning
    try:
        from paperhunter.postprocessing import VersionManager
        stats_data["versions"] = VersionManager().get_stats()
    except Exception:
        stats_data["versions"] = None

    # 6. Checkpoints
    try:
        from paperhunter.postprocessing import get_checkpoint_manager
        stats_data["checkpoints"] = get_checkpoint_manager().get_stats()
    except Exception:
        stats_data["checkpoints"] = None

    if use_rich:
        _print_stats_rich(console, stats_data)
    else:
        _print_stats_plain(stats_data)

    return 0


def _print_stats_rich(console, sd: dict) -> None:
    """Stampa statistiche con Rich."""
    from rich.table import Table

    t = Table(title="Stato del Sistema", show_lines=True)
    t.add_column("Componente", style="bold cyan", width=20)
    t.add_column("Stato", width=15)
    t.add_column("Dettagli", max_width=60)

    # Ingestion
    ing = sd.get("ingestion")
    if ing:
        cats = ", ".join(f"{k}: {v}" for k, v in ing.get("per_category", {}).items())
        t.add_row("Ingestion Registry", f"[green]{ing.get('total', 0)} paper[/green]", cats)
    else:
        t.add_row("Ingestion Registry", "[dim]N/A[/dim]", "")

    # Qdrant
    q = sd.get("qdrant")
    if q:
        parts = []
        for name in ["paper_chunks", "l1_summaries", "l1_5_summaries", "l2_summaries", "graph_communities"]:
            c = q.get(name, {})
            pts = c.get("total_points", 0)
            if pts > 0:
                parts.append(f"{name}: {pts}")
        t.add_row("Qdrant", f"[green]{q.get('total_points', 0)} pts[/green]", " | ".join(parts))
    else:
        t.add_row("Qdrant", "[dim]N/A[/dim]", "")

    # BM25
    bm = sd.get("bm25")
    if bm:
        t.add_row("BM25 Index", f"[green]{bm.get('total_chunks', 0)} docs[/green]",
                   f"Papers: {bm.get('unique_papers', '?')}")
    else:
        t.add_row("BM25 Index", "[yellow]Non costruito[/yellow]", "paperhunter index --bm25")

    # Graph
    gr = sd.get("graph")
    if gr:
        t.add_row("PropertyGraph", f"[green]{gr.get('total_entities', 0)} entità[/green]",
                   f"Relazioni: {gr.get('total_relations', 0)}")
    else:
        t.add_row("PropertyGraph", "[yellow]Non costruito[/yellow]", "paperhunter index --graph")

    # Versions
    ver = sd.get("versions")
    if ver and ver.get("total_versions", 0) > 0:
        t.add_row("Versioning", f"[green]v{ver.get('latest_version', '?')}[/green]",
                   f"{ver.get('total_versions', 0)} versioni")
    else:
        t.add_row("Versioning", "[dim]Nessuna[/dim]", "")

    # Checkpoints
    ck = sd.get("checkpoints")
    if ck and ck.get("total_checkpoints", 0) > 0:
        t.add_row("Checkpoints", f"[green]{ck.get('total_checkpoints', 0)} file[/green]", "")
    else:
        t.add_row("Checkpoints", "[dim]Nessuno[/dim]", "")

    console.print(t)


def _print_stats_plain(sd: dict) -> None:
    """Stampa statistiche in plain text."""
    print("\n" + "=" * 60)
    print("  STATISTICHE DEL SISTEMA")
    print("=" * 60)

    ing = sd.get("ingestion")
    if ing:
        print(f"\n  Ingestion: {ing.get('total', 0)} paper")
        for cat, count in ing.get("per_category", {}).items():
            print(f"    - {cat}: {count}")
    else:
        print("\n  Ingestion: non disponibile")

    q = sd.get("qdrant")
    if q:
        print(f"\n  Qdrant: {q.get('total_points', 0)} punti")
        for coll in ["paper_chunks", "l1_summaries", "l1_5_summaries", "l2_summaries", "graph_communities"]:
            c = q.get(coll, {})
            pts = c.get("total_points", 0)
            if pts > 0:
                print(f"    - {coll}: {pts}")
    else:
        print("\n  Qdrant: non disponibile")

    bm = sd.get("bm25")
    if bm:
        print(f"\n  BM25: {bm.get('total_chunks', 0)} documenti")
    else:
        print("\n  BM25: non costruito")

    gr = sd.get("graph")
    if gr:
        print(f"\n  PropertyGraph: {gr.get('total_entities', 0)} entità, {gr.get('total_relations', 0)} relazioni")
    else:
        print("\n  PropertyGraph: non costruito")

    ver = sd.get("versions")
    if ver and ver.get("total_versions", 0) > 0:
        print(f"\n  Versioni: {ver.get('total_versions', 0)} (ultima: v{ver.get('latest_version', '?')})")
    else:
        print("\n  Versioni: nessuna")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: export
# ═══════════════════════════════════════════════════════════════════════════

def cmd_export(args: argparse.Namespace) -> int:
    """Esporta il survey in LaTeX / BibTeX."""
    from paperhunter.config import settings
    logger = _get_logger("paperhunter.export")

    # Trova il file sorgente Markdown
    source_md = None
    version_dir = None

    if args.input_file:
        source_md = Path(args.input_file)
        if not source_md.exists():
            logger.error(f"File non trovato: {source_md}")
            return 1
        version_dir = source_md.parent
    else:
        # Cerca l'ultimo survey generato
        try:
            from paperhunter.postprocessing import VersionManager
            vm = VersionManager()
            current = vm.get_current_version()
            if current:
                vdir = vm.get_version_dir(current)
                sp = vdir / "survey.md"
                if sp.exists():
                    source_md = sp
                    version_dir = vdir
        except Exception:
            pass

        if source_md is None:
            fallback = Path(settings.output.base_dir) / "survey.md"
            if fallback.exists():
                source_md = fallback
                version_dir = fallback.parent

        if source_md is None:
            logger.error("Nessun survey trovato. Esegui prima 'paperhunter write' o specifica --input.")
            return 1

    logger.info(f"Sorgente: {source_md}")
    md_content = source_md.read_text(encoding="utf-8")

    output_dir = Path(args.output) if args.output else version_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = 0

    # ── Step 1: BibTeX filtrato (PRIMA della sanitization LaTeX, così che
    # _sanitize_latex possa rimuovere le \cite{} con chiavi non presenti) ──
    bib_generated_path: Path | None = None
    if args.bib_only or (not args.latex and not args.bib_only):
        try:
            from paperhunter.postprocessing import generate_final_bib
            from paperhunter.postprocessing.latex_exporter import (
                _extract_markdown_citation_keys,
            )
            used_keys = _extract_markdown_citation_keys(md_content)
            if used_keys:
                bib_path, bib_report = generate_final_bib(
                    used_keys=used_keys,
                    output_path=output_dir / "references_final.bib",
                    survey_text=md_content,
                    strict=settings.citability.final_latex_fail_on_error,
                )
                if bib_path:
                    bib_generated_path = Path(bib_path)
                    logger.info(f"BibTeX: {bib_path} ({bib_report.get('total_filtered', 0)} entry)")
            else:
                logger.warning("Nessuna citazione trovata nel survey.")
        except Exception as e:
            logger.error(f"Errore BibTeX: {e}")
            errors += 1

    # ── Step 2: LaTeX ──
    if args.latex or (not args.latex and not args.bib_only):
        try:
            from paperhunter.postprocessing import export_latex, print_export_summary
            template = args.template or settings.output.latex_template

            # Auto-detect bib_file: usa references_final se è stato generato
            # o se esiste già in output_dir, altrimenti fallback all'arg.
            requested_bib = args.bib_file or "references.bib"
            requested_stem = Path(requested_bib).stem
            final_bib = output_dir / "references_final.bib"
            if bib_generated_path is not None and bib_generated_path.exists():
                effective_bib_stem = bib_generated_path.stem
                effective_bib_source = bib_generated_path
            elif final_bib.exists():
                effective_bib_stem = "references_final"
                effective_bib_source = final_bib
            else:
                effective_bib_stem = requested_stem
                effective_bib_source = (
                    version_dir / requested_bib if version_dir else None
                )

            tex_path = export_latex(
                md_content, output_path=output_dir / "survey.tex",
                bib_file=effective_bib_stem, template=template,
                title=args.title, author=args.author,
                copy_bib=True,
                bib_source_path=effective_bib_source,
                strict_citations=settings.citability.final_latex_fail_on_error,
            )
            logger.info(f"LaTeX: {tex_path} (bib={effective_bib_stem})")
            print_export_summary(tex_path.read_text(encoding="utf-8"), output_path=tex_path)
        except Exception as e:
            logger.error(f"Errore LaTeX: {e}")
            errors += 1

    return 0 if errors == 0 else 1


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: cleanup
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_paper_date(
    pdf_path: Path,
    registry_entry: dict | None = None,
    s2_index: dict[str, tuple[int, int | None]] | None = None,
) -> tuple[int | None, int | None]:
    """Stima anno e mese di un paper dal percorso, bibtex_key, DOI, ArXiv ID o cache S2.

    Restituisce (year, month). month può essere None se non determinabile.
    """
    import unicodedata

    stem = pdf_path.stem

    # 1. Prova a estrarre ArXiv ID dal filename (YYMM.NNNNN)
    m = re.search(r'(\d{2})(\d{2})\.\d{4,5}', stem)
    if m:
        yr = 2000 + int(m.group(1))
        mo = int(m.group(2))
        if 1 <= mo <= 12 and 2000 <= yr <= 2100:
            return yr, mo

    # 2. Da DOI arXiv (10.48550/arXiv.YYMM.NNNNN)
    if registry_entry:
        doi = registry_entry.get("doi", "") or ""
        m = re.search(r'arXiv\.(\d{2})(\d{2})\.\d{4,5}', doi)
        if m:
            yr = 2000 + int(m.group(1))
            mo = int(m.group(2))
            if 1 <= mo <= 12:
                return yr, mo

        # 3. Da bibtex_key (Author_YYYY_Keyword)
        bk = registry_entry.get("bibtex_key", "") or ""
        m = re.search(r'_(\d{4})_', bk)
        if m:
            yr = int(m.group(1))
            if 1990 <= yr <= 2100:
                return yr, None

    # 4. Da cache Semantic Scholar (match per titolo normalizzato)
    if s2_index:
        norm = unicodedata.normalize("NFKD", stem).lower()
        norm = re.sub(r'[^\w\s]', '', norm)
        norm = re.sub(r'_+', ' ', norm)
        norm = ' '.join(norm.split())
        if norm in s2_index:
            return s2_index[norm]
        # Prova partial match (il titolo nel filename potrebbe essere troncato)
        for s2_title, date_tuple in s2_index.items():
            if len(norm) > 20 and norm[:30] in s2_title:
                return date_tuple

    # 5. Fallback: anno dal filename (cerco un 4-digit che sembri un anno)
    for m2 in re.finditer(r'(\d{4})', stem):
        yr = int(m2.group(1))
        if 2015 <= yr <= 2100:
            return yr, None

    return None, None


def _build_s2_year_index() -> dict[str, tuple[int, int | None]]:
    """Costruisce un indice titolo_normalizzato → (year, month) dalla cache S2.

    Scansiona ``cache/semantic_scholar/*.json`` e restituisce un dict
    di titoli normalizzati con anno e mese (se disponibile).
    """
    import json
    import unicodedata

    cache_dir = Path("cache/semantic_scholar")
    if not cache_dir.is_dir():
        return {}

    index: dict[str, tuple[int, int | None]] = {}
    for fp in cache_dir.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = data.get("title", "")
        year = data.get("year")
        if not title or not year:
            continue
        # Normalize title
        norm = unicodedata.normalize("NFKD", title).lower()
        norm = re.sub(r'[^\w\s]', '', norm)
        norm = ' '.join(norm.split())
        # Parse month from publicationDate if available
        month = None
        pd = data.get("publicationDate")
        if pd and len(pd) >= 7:
            try:
                month = int(pd[5:7])
            except ValueError:
                pass
        # Also try to infer from ArXiv ID (YYMM.NNNNN)
        ext = data.get("externalIds", {}) or {}
        arxiv_id = ext.get("ArXiv", "")
        if not month and arxiv_id:
            m = re.match(r'(\d{2})(\d{2})', arxiv_id)
            if m:
                try:
                    mo = int(m.group(2))
                    if 1 <= mo <= 12:
                        month = mo
                except ValueError:
                    pass
        index[norm] = (year, month)
    return index


def _date_in_range(
    year: int | None,
    month: int | None,
    date_from: str | None,
    date_to: str | None,
) -> bool | None:
    """Verifica se (year, month) cade nell'intervallo [date_from, date_to].

    Restituisce None se year è None (impossibile determinare).
    date_from/date_to sono in formato YYYY-MM.
    """
    if year is None:
        return None  # indeterminato

    def _parse_ym(s: str) -> tuple[int, int]:
        parts = s.split("-")
        return int(parts[0]), int(parts[1])

    paper_ym = (year, month or 1)  # se mese ignoto, usa gennaio (conservativo per range)

    if date_from:
        from_y, from_m = _parse_ym(date_from)
        if paper_ym < (from_y, from_m):
            return False

    if date_to:
        to_y, to_m = _parse_ym(date_to)
        # mese ignoto → conservativo: potrebbe essere qualsiasi mese dell'anno
        if month is None:
            # Se l'anno è fuori range sicuro, filtra
            if year < (int(date_from.split("-")[0]) if date_from else 0):
                return False
            if year > int(date_to.split("-")[0]) if date_to else 9999:
                return False
            # Se anno uguale a to_y, lo includiamo perché non sappiamo il mese
            return True
        if paper_ym > (to_y, to_m):
            return False

    return True


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Elimina paper scaricati in base a un intervallo di date."""
    import json

    logger = _get_logger("paperhunter.cleanup")
    console = _get_console()

    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else Path("scaricati")
    date_from = args.date_from
    date_to = args.date_to
    dry_run = args.dry_run
    include_unknown = args.include_unknown

    if not date_from and not date_to:
        logger.error("Specifica almeno --date-from o --date-to.")
        return 1

    if not pdf_dir.is_dir():
        logger.error(f"Directory non trovata: {pdf_dir}")
        return 1

    # Carica il registry (se esiste)
    registry_path = Path("data/hash_registry.json")
    registry: dict = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    # Costruisci indice S2 per match titoli → date
    s2_index = _build_s2_year_index()
    if s2_index:
        logger.info(f"Cache Semantic Scholar: {len(s2_index)} paper con data")

    # Scansiona tutti i PDF
    category_filter = getattr(args, "category", None)
    if category_filter:
        # Cerca sottocartella
        cat_dir = pdf_dir / category_filter
        if cat_dir.is_dir():
            all_pdfs = sorted(cat_dir.rglob("*.pdf"))
        else:
            # Cerca partial match
            candidates = [d for d in pdf_dir.iterdir() if d.is_dir() and category_filter.lower() in d.name.lower()]
            if candidates:
                all_pdfs = []
                for c in candidates:
                    all_pdfs.extend(c.rglob("*.pdf"))
                all_pdfs = sorted(all_pdfs)
            else:
                logger.error(f"Categoria non trovata: {category_filter}")
                return 1
    else:
        all_pdfs = sorted(pdf_dir.rglob("*.pdf"))
    logger.info(f"Trovati {len(all_pdfs)} PDF in {pdf_dir}")

    to_delete: list[tuple[Path, int | None, int | None, str]] = []
    unknown: list[Path] = []

    for pdf in all_pdfs:
        # Cerca nel registry usando path relativo
        rel_key = None
        for k in registry:
            if pdf.name == Path(k).name or str(pdf).endswith(k):
                rel_key = k
                break

        entry = registry.get(rel_key) if rel_key else None
        year, month = _estimate_paper_date(pdf, entry, s2_index=s2_index)

        if year is None:
            unknown.append(pdf)
            if include_unknown:
                to_delete.append((pdf, None, None, "data sconosciuta"))
            continue

        in_range = _date_in_range(year, month, date_from, date_to)
        if in_range is True:
            month_str = f"-{month:02d}" if month else ""
            to_delete.append((pdf, year, month, f"{year}{month_str}"))
        elif in_range is None:
            unknown.append(pdf)
            if include_unknown:
                to_delete.append((pdf, None, None, "data sconosciuta"))

    # Report
    range_str = f"{date_from or '*'} → {date_to or '*'}"
    logger.info(f"Intervallo richiesto: {range_str}")
    logger.info(f"Paper nel range: {len(to_delete)}")
    logger.info(f"Paper con data sconosciuta: {len(unknown)}")

    if not to_delete:
        logger.info("Nessun paper da eliminare.")
        return 0

    # Mostra tabella
    if console:
        try:
            from rich.table import Table
            t = Table(title=f"Paper da eliminare ({range_str})", show_lines=False)
            t.add_column("#", width=4)
            t.add_column("File", max_width=60)
            t.add_column("Data stimata", width=12)
            t.add_column("Categoria", width=20)

            for i, (pdf, yr, mo, date_label) in enumerate(to_delete, 1):
                cat = pdf.parent.name if pdf.parent != pdf_dir else "-"
                t.add_row(str(i), pdf.name[:60], date_label, cat)
            console.print(t)
        except Exception:
            for i, (pdf, yr, mo, date_label) in enumerate(to_delete, 1):
                print(f"  {i:3d}. [{date_label}] {pdf.name}")
    else:
        for i, (pdf, yr, mo, date_label) in enumerate(to_delete, 1):
            print(f"  {i:3d}. [{date_label}] {pdf.name}")

    if dry_run:
        logger.info(f"[DRY RUN] {len(to_delete)} file verrebbero eliminati. Nessuna modifica effettuata.")
        return 0

    # Conferma interattiva (se non --yes)
    if not args.yes:
        answer = input(f"\nEliminare {len(to_delete)} file? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "sì", "si"):
            logger.info("Operazione annullata.")
            return 0

    # Eliminazione
    deleted = 0
    errors = 0
    registry_modified = False

    for pdf, yr, mo, date_label in to_delete:
        try:
            pdf.unlink()
            deleted += 1
            logger.debug(f"Eliminato: {pdf}")

            # Rimuovi dal hash_registry
            for k in list(registry.keys()):
                if pdf.name == Path(k).name or str(pdf).endswith(k):
                    del registry[k]
                    registry_modified = True
                    break

        except OSError as e:
            logger.error(f"Errore eliminando {pdf}: {e}")
            errors += 1

    # Salva registry aggiornato
    if registry_modified:
        registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Hash registry aggiornato ({len(registry)} entry rimanenti)")

    logger.info(f"Cleanup completato: {deleted} eliminati, {errors} errori")
    return 0 if errors == 0 else 1


def cmd_checkpoint(args: argparse.Namespace) -> int:
    """Gestione checkpoint per-sezione del Writer Agent."""
    logger = _get_logger("paperhunter.checkpoint")

    from paperhunter.agents.writer_agent import (
        list_section_checkpoints,
        delete_section_checkpoint,
        clean_section_checkpoints,
    )

    action = getattr(args, "ckpt_action", None)

    if action == "list" or action is None:
        checkpoints = list_section_checkpoints()
        if not checkpoints:
            print("Nessun checkpoint sezione trovato.")
            return 0
        console = _get_console()
        if console:
            try:
                from paperhunter.utils.logging_setup import print_table
                data = [
                    {
                        "Key": cp["section_key"],
                        "Title": cp["title"],
                        "Words": str(cp["word_count"]),
                        "Citations": str(cp["citations"]),
                        "Saved At": cp["saved_at"][:19],
                        "File": cp["file"],
                    }
                    for cp in checkpoints
                ]
                print_table("Writer Section Checkpoints", data)
            except Exception:
                for cp in checkpoints:
                    print(
                        f"  {cp['section_key']:40s} "
                        f"{cp['word_count']:>5d} words  "
                        f"{cp['citations']:>3d} cit  "
                        f"{cp['saved_at'][:19]}"
                    )
        else:
            for cp in checkpoints:
                print(
                    f"  {cp['section_key']:40s} "
                    f"{cp['word_count']:>5d} words  "
                    f"{cp['citations']:>3d} cit  "
                    f"{cp['saved_at'][:19]}"
                )
        print(f"\nTotale: {len(checkpoints)} checkpoint")
        return 0

    elif action == "delete":
        key = args.section_key
        if delete_section_checkpoint(key):
            print(f"Checkpoint sezione '{key}' eliminato.")
        else:
            print(f"Nessun checkpoint trovato per '{key}'.")
            return 1
        return 0

    elif action == "clear":
        removed = clean_section_checkpoints()
        print(f"Eliminati {removed} checkpoint sezione.")
        return 0

    else:
        print("Usa: paperhunter checkpoint [list|delete|clear]")
        return 1


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: figures
# ═══════════════════════════════════════════════════════════════════════════

def cmd_figures(args: argparse.Namespace) -> int:
    """Genera figure scientifiche per il survey paper."""
    import json as _json
    from pathlib import Path as _Path

    from paperhunter.config import settings
    from paperhunter.agents.figure_agent.planner import (
        plan_figures, save_figure_plan, load_figure_plan, FigurePlan,
    )
    from paperhunter.agents.figure_agent.decision import classify_all
    from paperhunter.agents.figure_agent.renderer import render_all_figures
    from paperhunter.agents.figure_agent.integrator import (
        integrate_figures, copy_figures_to_output,
    )

    logger = _get_logger("paperhunter.figures")
    console = _get_console()

    console.print("[bold magenta]🎨 FigureAgent — Generazione Figure Scientifiche[/bold magenta]")
    console.print()

    fig_cfg = settings.figure_agent
    output_dir = _Path(fig_cfg.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load paper markdown ──
    paper_path = args.input_file
    if not paper_path:
        # Cerca l'ultima versione
        latest = _Path("output/latest")
        if latest.exists() and latest.is_symlink():
            target = latest.resolve()
            md_files = list(target.glob("*.md"))
            if md_files:
                paper_path = str(md_files[0])

    if not paper_path or not _Path(paper_path).exists():
        console.print("[red]❌ Nessun paper trovato. Specifica --input PATH[/red]")
        return 1

    paper_markdown = _Path(paper_path).read_text(encoding="utf-8")
    console.print(f"📄 Paper: {paper_path} ({len(paper_markdown):,} chars)")

    # ── Step 2: Load L2 summaries (optional) ──
    l2_summaries = None
    l2_dir = _Path("output/l2_metadata")
    if l2_dir.exists():
        l2_files = list(l2_dir.glob("*.json"))
        if l2_files:
            l2_summaries = []
            for f in l2_files:
                try:
                    data = _json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        l2_summaries.append(data)
                except Exception:
                    pass
            console.print(f"📊 Loaded {len(l2_summaries)} L2 summaries")

    # ── Step 3: Plan or load existing plan ──
    plan_path = output_dir / "figure_plan.json"
    plan: FigurePlan | None = None

    # Auto-invalidate cached plan if source paper is newer
    force = args.force
    if not force and plan_path.exists() and paper_path:
        plan_mtime = plan_path.stat().st_mtime
        paper_mtime = _Path(paper_path).stat().st_mtime
        if paper_mtime > plan_mtime:
            console.print(
                "[yellow]📋 Paper più recente del piano cachato → rigenero il piano[/yellow]"
            )
            force = True

    if args.plan_file:
        plan = load_figure_plan(args.plan_file)
        console.print(f"📋 Loaded plan from {args.plan_file}")
    elif plan_path.exists() and not force:
        plan = load_figure_plan(plan_path)
        console.print(f"📋 Loaded existing plan ({plan.total_figures} figures)")
    else:
        plan = plan_figures(
            paper_markdown=paper_markdown,
            l2_summaries=l2_summaries,
            min_figures=fig_cfg.min_figures,
            max_figures=fig_cfg.max_figures,
            force_model=args.model,
        )
        save_figure_plan(plan, output_dir)

    if not plan or not plan.figures:
        console.print("[yellow]⚠️ Nessuna figura pianificata[/yellow]")
        return 0

    # ── Step 4: Classify figure types ──
    classify_all(plan.figures)

    # ── Step 5: Generate figures with Gemini Image Generation ──
    if not args.plan_only:
        rendered = render_all_figures(
            specs=plan.figures,
            output_dir=output_dir,
            max_iterations=fig_cfg.max_iterations,
            force_model=args.model,
        )

        # ── Step 6: Integrate into paper ──
        if not args.no_integrate and rendered:
            result_paper, insertions = integrate_figures(
                paper_markdown=paper_markdown,
                plan=plan,
                rendered_figures=rendered,
                figures_dir="generated_figures",
                output_format=fig_cfg.output_format,
            )

            # Salva paper con figure
            out_paper = output_dir / "paper_with_figures.md"
            out_paper.write_text(result_paper, encoding="utf-8")
            console.print(f"\n📝 Paper con figure: {out_paper}")

            inserted = sum(1 for ins in insertions if ins.inserted)
            console.print(
                f"[bold green]✅ {inserted} figure integrate nel paper[/bold green]"
            )

            # ── Step 7: Re-export LaTeX con figure ──
            try:
                import shutil
                from paperhunter.postprocessing import export_latex
                base = _Path(settings.output.base_dir)
                tex_out = base / "latest" / "survey.tex"

                # Copy generated figures into output/latest/generated_figures/
                latest_gen = base / "latest" / "generated_figures"
                latest_gen.mkdir(parents=True, exist_ok=True)
                for fig_file in output_dir.glob("*.png"):
                    shutil.copy2(fig_file, latest_gen / fig_file.name)

                bib_src = None
                for cand in [base / "references_final.bib", base / "latest" / "references.bib", base / "references.bib"]:
                    if cand.exists():
                        bib_src = cand
                        break
                export_latex(
                    md_content=result_paper,
                    output_path=tex_out,
                    bib_file="references",
                    copy_bib=True,
                    bib_source_path=bib_src,
                    strict_citations=settings.citability.final_latex_fail_on_error,
                )
                console.print(f"[bold green]✅ LaTeX aggiornato con figure: {tex_out}[/bold green]")
            except Exception as e:
                logger.warning(f"Re-export LaTeX con figure fallito: {e}")
                console.print(f"[yellow]⚠️ Re-export LaTeX fallito: {e}[/yellow]")
        else:
            console.print("📸 Figure renderizzate (--no-integrate: non integrate nel paper)")

    console.print(f"\n📂 Output: {output_dir}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: experiment
# ═══════════════════════════════════════════════════════════════════════════

def cmd_experiment(args: argparse.Namespace) -> int:
    """Esporta dati PaperHunter e lancia pipeline sperimentale via AutoResearchClaw."""
    import shutil
    import subprocess
    from pathlib import Path as _Path

    from paperhunter.config import settings
    from paperhunter.bridge.researchclaw_adapter import ResearchClawAdapter

    logger = _get_logger("paperhunter.experiment")
    console = _get_console()

    console.print(
        "[bold magenta]🔬 PaperHunter → AutoResearchClaw Bridge[/bold magenta]"
    )
    console.print()

    ph_root = _Path(args.project_root or ".")

    # ── Import mode: import results from existing RC run ──
    if args.import_from:
        rc_dir = _Path(args.import_from)
        if not rc_dir.exists():
            console.print(f"[red]❌ Directory non trovata: {rc_dir}[/red]")
            return 1

        adapter = ResearchClawAdapter(
            paperhunter_root=ph_root,
            researchclaw_run_dir=rc_dir,
            run_id=args.run_id or rc_dir.name,
        )
        report = adapter.import_results()

        console.print(f"📂 Import da: {rc_dir}")
        console.print(f"  Analysis: {'✅' if report.analysis_loaded else '❌'}"
                       f" ({report.analysis_chars:,} chars)")
        console.print(f"  Summary:  {'✅' if report.summary_loaded else '❌'}")
        console.print(f"  Charts:   {len(report.charts_imported)} importati")

        if report.section_path:
            console.print(
                f"\n[bold green]✅ Sezione 'Experimental Validation' creata: "
                f"{report.section_path} ({report.section_chars:,} chars)[/bold green]"
            )
        else:
            console.print(
                "\n[yellow]⚠️  Nessun risultato sperimentale trovato. "
                "Verifica che RC abbia completato stage 14+.[/yellow]"
            )
        return 0 if report.success else 1

    # ── Export mode (default) ──
    output_dir = _Path(args.output or "./experiments")
    run_id = args.run_id

    # ── Step 1: Verify survey completeness ──
    versions_dir = ph_root / "output" / "versions"
    if not versions_dir.exists() or not list(versions_dir.glob("Survey_v*")):
        console.print(
            "[red]❌ Nessun survey trovato in output/versions/. "
            "Esegui prima 'paperhunter write'.[/red]"
        )
        return 1

    graph_path = ph_root / "data" / "property_graph" / "graph_store.json"
    if not graph_path.exists():
        console.print(
            "[yellow]⚠️  Property graph non trovato. "
            "Le knowledge cards saranno vuote.[/yellow]"
        )

    bib_found = any(
        (ph_root / "output" / n).exists()
        for n in ("references_final.bib", "references.bib")
    )
    if not bib_found:
        console.print("[yellow]⚠️  Nessun file BibTeX trovato in output/.[/yellow]")

    console.print("[green]✅ Survey verificato — dati pronti per export[/green]")

    # ── Step 2: Create adapter and export ──
    rc_dir = output_dir / (run_id or "run")
    adapter = ResearchClawAdapter(
        paperhunter_root=ph_root,
        researchclaw_run_dir=rc_dir,
        run_id=run_id,
    )

    console.print(f"\n📂 Export directory: {rc_dir}")
    report = adapter.export_all()

    # Print export results
    console.print()
    for exp in report.exports:
        icon = "✅" if exp.success else "❌"
        console.print(f"  {icon} {exp.stage}: {exp.message}")

    ok_count = sum(1 for e in report.exports if e.success)
    total_count = len(report.exports)
    console.print(f"\n📊 Export: {ok_count}/{total_count} fasi completate")

    if ok_count == 0:
        console.print("[red]❌ Nessun export riuscito. Controlla i log.[/red]")
        return 1

    # ── Step 3: Check if ResearchClaw is installed ──
    if args.export_only:
        console.print(
            "\n[cyan]--export-only: dati esportati. "
            "Lancia manualmente:[/cyan]"
        )
        console.print(
            f"  researchclaw run --config config.researchclaw.yaml --output {rc_dir} --resume"
        )
        return 0

    # Verify researchclaw is available
    rc_available = shutil.which("researchclaw") is not None
    if not rc_available:
        try:
            subprocess.run(
                [sys.executable, "-m", "researchclaw", "--version"],
                capture_output=True, timeout=10,
            )
            rc_available = True
        except Exception:
            pass

    if not rc_available:
        console.print(
            "\n[yellow]⚠️  AutoResearchClaw non trovato nel PATH.[/yellow]"
        )
        console.print(
            "Installa con: pip install -e /path/to/AutoResearchClaw"
        )
        console.print(
            f"\nPuoi lanciare manualmente:\n"
            f"  researchclaw run --config config.researchclaw.yaml --output {rc_dir} --resume"
        )
        return 0

    # ── Step 4: Launch ResearchClaw ──
    from_stage = args.from_stage or "EXPERIMENT_DESIGN"

    # Resolve config path (must be absolute for subprocess cwd)
    rc_config = args.rc_config
    if not rc_config:
        default_cfg = ph_root / "config.researchclaw.yaml"
        if default_cfg.exists():
            rc_config = str(default_cfg.resolve())
    elif rc_config and not _Path(rc_config).is_absolute():
        rc_config = str(_Path(rc_config).resolve())

    rc_dir_abs = str(rc_dir.resolve())
    rc_cmd = [
        sys.executable, "-m", "researchclaw", "run",
        "--output", rc_dir_abs,
        "--from-stage", from_stage,
        "--resume",
        "--auto-approve",
        "--skip-preflight",
    ]
    if rc_config:
        rc_cmd.extend(["--config", rc_config])

    console.print(f"\n🚀 Lancio ResearchClaw da stage {from_stage}...")
    console.print(f"   Comando: {' '.join(rc_cmd)}")
    console.print()

    # Prepara env per il subprocess RC
    env = os.environ.copy()
    try:
        import certifi
        env.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    # Raccogli la key del provider selezionato.
    from paperhunter.config import settings as _settings
    api_keys: list[tuple[str, str]] = []
    if _settings.llm_provider == "deepseek":
        if _settings.deepseek_api_key:
            api_keys.append(("DEEPSEEK", _settings.deepseek_api_key))
    else:
        if _settings.google_api_key_free:
            api_keys.append(("FREE", _settings.google_api_key_free))
        if _settings.google_api_key:
            api_keys.append(("PAID", _settings.google_api_key))
    if not api_keys:
        expected = "DEEPSEEK_API_KEY" if _settings.llm_provider == "deepseek" else "GOOGLE_API_KEY"
        console.print(f"[red]❌ Nessuna {expected} configurata nel .env[/red]")
        return 1

    # Leggi quale env var RC usa per la key
    rc_key_env = (
        "DEEPSEEK_API_KEY"
        if _settings.llm_provider == "deepseek"
        else "GOOGLE_API_KEY_FREE"
    )
    try:
        import yaml as _yaml
        with open(rc_config, "r") as _cf:
            _rc_data = _yaml.safe_load(_cf)
        rc_key_env = _rc_data.get("llm", {}).get("api_key_env", rc_key_env)
    except Exception:
        pass

    # Lancia RC con rotazione automatica: FREE → PAID
    for key_idx, (key_label, key_value) in enumerate(api_keys):
        env[rc_key_env] = key_value
        console.print(
            f"   🔑 API key: [bold]{key_label}[/bold] "
            f"({key_idx + 1}/{len(api_keys)})"
        )

        try:
            result = subprocess.run(rc_cmd, env=env)
            if result.returncode == 0:
                console.print(
                    "\n[bold green]✅ Pipeline sperimentale completata![/bold green]"
                )
                return 0

            # RC fallito — se c'è un'altra key, riprova con --resume
            if key_idx + 1 < len(api_keys):
                console.print(
                    f"\n[yellow]⚠️  RC terminato (codice {result.returncode}) "
                    f"con key {key_label}. Passo a key successiva...[/yellow]"
                )
                continue
            else:
                console.print(
                    f"\n[red]❌ ResearchClaw terminato con codice "
                    f"{result.returncode} (tutte le key esaurite)[/red]"
                )
                return result.returncode
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Interrotto dall'utente.[/yellow]")
            console.print(
                f"Riprendi con: researchclaw run --output {rc_dir} --resume"
            )
            return 130

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand: reset
# ═══════════════════════════════════════════════════════════════════════════

def cmd_reset(args: argparse.Namespace) -> int:
    """Elimina tutti i checkpoint salvati per ripartire da zero."""
    from pathlib import Path as _Path
    from paperhunter.config import settings

    logger = _get_logger("paperhunter.reset")
    console = _get_console()
    dry_run = args.dry_run
    yes = getattr(args, "yes", False)
    total_removed = 0

    if not dry_run and not yes:
        print(
            "\nATTENZIONE: verranno eliminati tutti i checkpoint, le sessioni temp_llm\n"
            "e le figure estratte. L'operazione non è reversibile.\n"
        )
        answer = input("Procedere? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "sì", "si"):
            logger.info("Reset annullato dall'utente.")
            return 0

    # 1. Pipeline + ingestion checkpoints (output/.checkpoints/*.json)
    try:
        from paperhunter.postprocessing import get_checkpoint_manager
        mgr = get_checkpoint_manager()
        n_pipeline = len(list(mgr.checkpoint_dir.glob("*.json")))
        if n_pipeline:
            if dry_run:
                logger.info(f"[DRY RUN] {n_pipeline} checkpoint pipeline/ingestion in {mgr.checkpoint_dir}")
            else:
                removed = mgr.clear_all()
                total_removed += removed
                logger.info(f"Rimossi {removed} checkpoint pipeline/ingestion")
        else:
            logger.info("Nessun checkpoint pipeline/ingestion trovato.")
    except Exception as e:
        logger.warning(f"Errore pulizia checkpoint pipeline: {e}")

    # 2. Writer section checkpoints (output/.checkpoints/writer_sections/)
    try:
        from paperhunter.agents.writer_agent import clean_section_checkpoints
        from paperhunter.agents.writer_agent import _get_writer_checkpoint_dir
        cp_dir = _get_writer_checkpoint_dir()
        n_sections = len(list(cp_dir.glob("*.json")))
        if n_sections:
            if dry_run:
                logger.info(f"[DRY RUN] {n_sections} checkpoint sezione writer in {cp_dir}")
            else:
                removed = clean_section_checkpoints()
                total_removed += removed
                logger.info(f"Rimossi {removed} checkpoint sezione writer")
        else:
            logger.info("Nessun checkpoint sezione writer trovato.")
    except Exception as e:
        logger.warning(f"Errore pulizia checkpoint sezione writer: {e}")

    # 3. Search checkpoints (cache/search_checkpoints/)
    try:
        search_ckpt_dir = _Path("cache/search_checkpoints")
        if search_ckpt_dir.is_dir():
            files = list(search_ckpt_dir.glob("*"))
            if files:
                if dry_run:
                    logger.info(f"[DRY RUN] {len(files)} file checkpoint di ricerca in {search_ckpt_dir}")
                else:
                    removed = 0
                    for f in files:
                        if f.is_file():
                            f.unlink()
                            removed += 1
                    total_removed += removed
                    logger.info(f"Rimossi {removed} checkpoint di ricerca")
            else:
                logger.info("Nessun checkpoint di ricerca trovato.")
        else:
            logger.info("Directory checkpoint di ricerca non presente.")
    except Exception as e:
        logger.warning(f"Errore pulizia checkpoint di ricerca: {e}")

    # 4. L1.5 metadata state (output/l1_5_metadata/)
    try:
        l1_5_dir = _L1_5_STATE_DIR
        if l1_5_dir.is_dir():
            files = list(l1_5_dir.glob("*.json"))
            if files:
                if dry_run:
                    logger.info(f"[DRY RUN] {len(files)} file stato L1.5 in {l1_5_dir}")
                else:
                    removed = 0
                    for f in files:
                        f.unlink()
                        removed += 1
                    total_removed += removed
                    logger.info(f"Rimossi {removed} file stato L1.5")
            else:
                logger.info("Nessun file stato L1.5 trovato.")
    except Exception as e:
        logger.warning(f"Errore pulizia stato L1.5: {e}")

    # 5. temp_llm (output/temp_llm/)
    try:
        from paperhunter.config import settings as _settings
        temp_llm_dir = _Path(_settings.output.base_dir) / "temp_llm"
        if temp_llm_dir.is_dir():
            import shutil as _shutil
            n_items = len(list(temp_llm_dir.iterdir()))
            if n_items:
                if dry_run:
                    logger.info(f"[DRY RUN] {n_items} sessioni temp_llm in {temp_llm_dir}")
                else:
                    _shutil.rmtree(temp_llm_dir)
                    temp_llm_dir.mkdir(parents=True, exist_ok=True)
                    total_removed += n_items
                    logger.info(f"Rimossa directory temp_llm ({n_items} sessioni)")
            else:
                logger.info("Nessun file temp_llm trovato.")
        else:
            logger.info("Directory temp_llm non presente.")
    except Exception as e:
        logger.warning(f"Errore pulizia temp_llm: {e}")

    # 6. Figure estratte (output/figures/)
    try:
        from paperhunter.config import settings as _settings
        figures_dir = _Path(_settings.figure_extraction.figures_dir)
        if figures_dir.is_dir():
            fig_files = list(figures_dir.glob("*"))
            if fig_files:
                if dry_run:
                    logger.info(f"[DRY RUN] {len(fig_files)} figure in {figures_dir}")
                else:
                    removed = 0
                    for f in fig_files:
                        if f.is_file():
                            f.unlink()
                            removed += 1
                    total_removed += removed
                    logger.info(f"Rimosse {removed} figure estratte")
            else:
                logger.info("Nessuna figura estratta trovata.")
        else:
            logger.info("Directory figure non presente.")
    except Exception as e:
        logger.warning(f"Errore pulizia figure: {e}")

    # Riepilogo
    if dry_run:
        logger.info("[DRY RUN] Nessun file eliminato.")
    else:
        logger.info(f"Reset completato: {total_removed} file eliminati in totale.")

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Parser principale
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser con tutti i subcommand."""
    parser = argparse.ArgumentParser(
        prog="paperhunter",
        description=(
            "PaperHunter v2.2 — End-to-end LLM Security Survey Generator.\n\n"
            "Pipeline completa: search → ingest → summarize → index → write → export\n"
            "Supporta RAPTOR (clustering L1.5), Graph Communities e Agentic RAG."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Workflow tipico:\n"
            "  1. paperhunter search -k 'prompt injection'   # Cerca e scarica paper\n"
            "  2. paperhunter ingest                          # Processa i PDF\n"
            "  3. paperhunter summarize                       # Genera L1 + L1.5 + L2\n"
            "  4. paperhunter index                           # Costruisci BM25 + Graph + Communities\n"
            "  5. paperhunter write                           # Genera il survey\n"
            "  6. paperhunter write --writer-mode agentic     # Con ReAct agent\n"
            "  7. paperhunter export --template acl           # Esporta in LaTeX\n"
        ),
    )

    # --- Opzioni globali ---
    parser.add_argument("--config", type=str, metavar="PATH",
                        help="Percorso al file config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Abilita output verbose (DEBUG)")
    parser.add_argument("--version", action="version", version="paperhunter 2.2.0")

    sub = parser.add_subparsers(dest="command", help="Comandi disponibili")

    # ── search ──
    p_search = sub.add_parser(
        "search", help="Cerca e scarica nuovi paper scientifici",
        description="Cerca paper su ArXiv, Semantic Scholar e altre fonti, filtra con LLM e scarica i PDF.",
    )
    p_search.add_argument("-k", "--keywords", type=str,
                          help="Keyword di ricerca. Supporta OR per query multiple "
                               "(es. '\"prompt injection\" OR \"jailbreak\" OR \"adversarial attack\"')")
    p_search.add_argument("-c", "--criteria", type=str,
                          help="Criteri di filtro per LLM")
    p_search.add_argument("-n", "--max-results", type=int,
                          help="Max risultati per fonte (default: da config.yaml)")
    p_search.add_argument("--no-download", action="store_true",
                          help="Solo ricerca, nessun download")
    p_search.add_argument("--llm-provider", type=str, choices=["google", "deepseek"],
                          help="Provider LLM per filtro (default: da config, DeepSeek)")
    p_search.add_argument("-o", "--output-dir", type=str,
                          help="Directory output PDF")
    p_search.add_argument("--date-from", type=str, metavar="YYYY-MM",
                          help="Data minima pubblicazione (es. 2025-10)")
    p_search.add_argument("--date-to", type=str, metavar="YYYY-MM",
                          help="Data massima pubblicazione (es. 2026-03)")
    p_search.add_argument("-y", "--yes", action="store_true",
                          help="Salta conferma interattiva, procedi direttamente")

    # ── ingest ──
    p_ingest = sub.add_parser(
        "ingest", help="Pipeline di ingestion incrementale dei PDF",
        description="GROBID → S2 enrichment → BibTeX → chunking → Qdrant.",
    )
    p_ingest.add_argument("--pdf-dir", type=str,
                          help="Directory radice PDF (default: config.yaml)")
    p_ingest.add_argument("--batch-size", type=int,
                          help="PDF per batch (default: config.yaml)")
    p_ingest.add_argument("--skip-s2", action="store_true",
                          help="Salta arricchimento Semantic Scholar")
    p_ingest.add_argument("--skip-qdrant", action="store_true",
                          help="Salta indicizzazione Qdrant")
    p_ingest.add_argument("--no-checkpoint", action="store_true",
                          help="Disabilita checkpoint di resume")
    p_ingest.add_argument("--force", action="store_true",
                          help="Forza re-ingestion di tutti i PDF ignorando il registro hash")

    # ── summarize ──
    p_sum = sub.add_parser(
        "summarize", help="Genera/rigenera summary L1, L1.5 e L2",
        description=(
            "Summary L1 (singolo paper), L1.5 (cluster RAPTOR) e L2 (meta per categoria).\n"
            "Senza flag, genera tutti e tre i livelli. L1.5 richiede L1 già generati."
        ),
    )
    p_sum.add_argument("--l1", action="store_true", help="Solo summary L1")
    p_sum.add_argument("--l1-5", action="store_true", dest="l1_5",
                       help="Solo summary L1.5 (cluster RAPTOR). Richiede L1")
    p_sum.add_argument("--l2", action="store_true", help="Solo summary L2")
    p_sum.add_argument("--invalidate", action="store_true",
                       help="Check invalidazione L2 e rigenera se necessario")
    p_sum.add_argument("--category", type=str,
                       help="Limita a una categoria (es. 'Prompt Injection')")
    p_sum.add_argument("--force", action="store_true",
                       help="Forza rigenerazione anche se già esistenti")

    # ── index ──
    p_index = sub.add_parser(
        "index", help="Costruisci/aggiorna indici (BM25 + GraphRAG + Communities)",
        description=(
            "Costruisce BM25, PropertyGraph e community detection.\n"
            "Senza flag, costruisce tutti gli indici."
        ),
    )
    p_index.add_argument("--bm25", action="store_true", help="Solo BM25")
    p_index.add_argument("--graph", action="store_true", help="Solo PropertyGraph")
    p_index.add_argument("--communities", action="store_true",
                         help="Solo community detection e summary (richiede Neo4j + GDS)")
    p_index.add_argument("--skip-neo4j", action="store_true",
                         help="Salta import Neo4j")
    p_index.add_argument("--clear-graph", action="store_true",
                         help="Pulisci Neo4j prima di importare")
    p_index.add_argument("--community-algorithm", type=str,
                         choices=["louvain", "leiden"], default="louvain",
                         help="Algoritmo community detection (default: louvain)")
    p_index.add_argument("--community-min-size", type=int, default=3,
                         help="Dimensione minima comunità (default: 3)")

    # ── write ──
    p_write = sub.add_parser(
        "write", help="Pipeline agentica completa (LangGraph)",
        description="Planner → Writer → Critic → Assembler. Genera il survey completo.",
    )
    p_write.add_argument("--outline", type=str, metavar="PATH",
                         help="File JSON outline (altrimenti generato dal Planner)")
    p_write.add_argument("--resume", action="store_true",
                         help="Riprendi da checkpoint")
    p_write.add_argument("--checkpoint-file", type=str, metavar="PATH",
                         help="Checkpoint specifico (con --resume)")
    p_write.add_argument("--no-checkpoint", action="store_true",
                         help="Disabilita checkpoint")
    p_write.add_argument("--no-export", action="store_true",
                         help="No export automatico LaTeX/BibTeX")
    p_write.add_argument("--template", type=str, choices=["acl", "ieeetran", "plain"],
                         help="Template LaTeX (default: config.yaml)")
    p_write.add_argument("--writer-mode", type=str, choices=["static", "agentic"],
                         help="Modalità writer: 'static' (RAG) o 'agentic' (ReAct). Override config.yaml")
    p_write.add_argument("--force", action="store_true",
                         help="Riscrive tutte le sezioni da zero, ignorando i checkpoint salvati")

    # ── query ──
    p_query = sub.add_parser(
        "query", help="Query interattiva contro la knowledge base",
        description="Query ibrida (vector + BM25) con reranking. Senza arg, modalità interattiva.",
    )
    p_query.add_argument("question", nargs="?", default=None,
                         help="Domanda (ometti per interattivo)")
    p_query.add_argument("-n", "--n-results", type=int, default=10,
                         help="Numero risultati (default: 10)")
    p_query.add_argument("--category", type=str, help="Filtra per categoria")
    p_query.add_argument("--year-min", type=int, help="Anno minimo")
    p_query.add_argument("--year-max", type=int, help="Anno massimo")
    p_query.add_argument("--no-rerank", action="store_true",
                         help="Disabilita reranking Gemini")

    # ── stats ──
    sub.add_parser("stats", help="Statistiche complete del sistema",
                   description="Stato di ingestion, Qdrant, BM25, GraphRAG, versioni, checkpoint.")

    # ── export ──
    p_export = sub.add_parser(
        "export", help="Esporta survey in LaTeX / BibTeX",
        description="Converte Markdown → LaTeX e genera .bib filtrato.",
    )
    p_export.add_argument("--input", dest="input_file", type=str, metavar="PATH",
                          help="File Markdown survey (default: ultima versione)")
    p_export.add_argument("-o", "--output", type=str, metavar="DIR",
                          help="Directory output")
    p_export.add_argument("--template", type=str, choices=["acl", "ieeetran", "plain"],
                          help="Template LaTeX (default: config.yaml)")
    p_export.add_argument("--title", type=str, help="Titolo override")
    p_export.add_argument("--author", type=str, help="Autore override")
    p_export.add_argument("--bib-file", type=str, default="references.bib",
                          help="Nome .bib (default: references.bib)")
    p_export.add_argument("--latex", action="store_true", help="Solo LaTeX")
    p_export.add_argument("--bib-only", action="store_true", help="Solo BibTeX filtrato")

    # ── cleanup ──
    p_cleanup = sub.add_parser(
        "cleanup", help="Elimina paper scaricati per intervallo di date",
        description=(
            "Elimina PDF scaricati in base alla data di pubblicazione stimata.\n"
            "La data viene estratta dall'ArXiv ID, DOI, bibtex_key o filename.\n\n"
            "Esempi:\n"
            "  paperhunter cleanup --date-from 2023-01 --date-to 2023-12   # Paper del 2023\n"
            "  paperhunter cleanup --date-to 2024-06 --dry-run             # Preview pre-Lug 2024\n"
            "  paperhunter cleanup --date-from 2025-10 --date-to 2026-03 --yes  # Senza conferma\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cleanup.add_argument("--date-from", type=str, metavar="YYYY-MM",
                           help="Data minima (inclusa), es. 2025-10")
    p_cleanup.add_argument("--date-to", type=str, metavar="YYYY-MM",
                           help="Data massima (inclusa), es. 2026-03")
    p_cleanup.add_argument("--pdf-dir", type=str,
                           help="Directory PDF (default: scaricati/)")
    p_cleanup.add_argument("--dry-run", action="store_true",
                           help="Mostra cosa verrebbe eliminato senza cancellare")
    p_cleanup.add_argument("--yes", "-y", action="store_true",
                           help="Salta conferma interattiva")
    p_cleanup.add_argument("--include-unknown", action="store_true",
                           help="Includi paper con data non determinabile")
    p_cleanup.add_argument("--category", type=str,
                           help="Limita a una sotto-cartella/categoria")

    # ── checkpoint ──
    p_ckpt = sub.add_parser(
        "checkpoint", help="Gestione checkpoint writer (list/delete/clear)",
        description=(
            "Gestisce i checkpoint per-sezione salvati dal Writer Agent.\n"
            "Permette di visualizzare, eliminare selettivamente o cancellare\n"
            "tutti i checkpoint per controllare il resume della scrittura.\n\n"
            "Esempi:\n"
            "  paperhunter checkpoint list                       # Mostra checkpoint\n"
            "  paperhunter checkpoint delete introduction        # Elimina una sezione\n"
            "  paperhunter checkpoint clear                      # Elimina tutti\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ckpt_sub = p_ckpt.add_subparsers(dest="ckpt_action", help="Azione")
    ckpt_sub.add_parser("list", help="Lista checkpoint sezioni salvati")
    ckpt_del = ckpt_sub.add_parser("delete", help="Elimina checkpoint di una sezione")
    ckpt_del.add_argument("section_key", type=str, help="Chiave sezione da eliminare")
    ckpt_sub.add_parser("clear", help="Elimina tutti i checkpoint sezione")

    # ── figures ──
    p_figures = sub.add_parser(
        "figures", help="Genera figure scientifiche per il survey",
        description=(
            "Analizza il paper e genera automaticamente figure accademiche.\n\n"
            "Esempi:\n"
            "  paperhunter figures                        # Genera figure per l'ultimo paper\n"
            "  paperhunter figures --input paper.md       # Paper specifico\n"
            "  paperhunter figures --plan-only             # Solo pianificazione\n"
            "  paperhunter figures --force                 # Rigenera piano da zero\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_figures.add_argument("--input", dest="input_file", type=str, metavar="PATH",
                           help="File Markdown del paper (default: ultima versione)")
    p_figures.add_argument("--plan-file", type=str, metavar="PATH",
                           help="Carica piano figure da file JSON esistente")
    p_figures.add_argument("--plan-only", action="store_true",
                           help="Solo pianificazione, non renderizzare")
    p_figures.add_argument("--no-integrate", action="store_true",
                           help="Renderizza ma non integrare nel paper")
    p_figures.add_argument("--model", type=str, choices=["flash", "pro"],
                           help="Override modello LLM (default: config)")
    p_figures.add_argument("--force", action="store_true",
                           help="Rigenera piano anche se esiste già")

    # ── experiment ──
    p_experiment = sub.add_parser(
        "experiment",
        help="Esporta dati e lancia pipeline sperimentale via AutoResearchClaw",
        description=(
            "Bridge PaperHunter → AutoResearchClaw: esporta property graph, L2\n"
            "summaries e bibliografia nel formato ResearchClaw, poi lancia la\n"
            "pipeline sperimentale da stage 9 (EXPERIMENT_DESIGN).\n\n"
            "Esempi:\n"
            "  paperhunter experiment                           # Export + lancio RC\n"
            "  paperhunter experiment --export-only              # Solo export dati\n"
            "  paperhunter experiment --output ./exp/run1        # Directory custom\n"
            "  paperhunter experiment --from-stage PAPER_DRAFT   # Stage diverso\n"
            "  paperhunter experiment --import-from ./exp/run1   # Importa risultati\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_experiment.add_argument(
        "--output", type=str, default="./experiments",
        help="Directory di output per la run ResearchClaw (default: ./experiments)",
    )
    p_experiment.add_argument(
        "--run-id", type=str, default=None,
        help="Identificativo della run (default: auto-generato)",
    )
    p_experiment.add_argument(
        "--project-root", type=str, default=None,
        help="Root del progetto PaperHunter (default: directory corrente)",
    )
    p_experiment.add_argument(
        "--export-only", action="store_true",
        help="Solo export dati, non lanciare ResearchClaw",
    )
    p_experiment.add_argument(
        "--from-stage", type=str, default=None,
        help="Stage da cui partire in ResearchClaw (default: EXPERIMENT_DESIGN)",
    )
    p_experiment.add_argument(
        "--import-from", type=str, default=None, metavar="RC_DIR",
        help="Importa risultati sperimentali da una run RC completata",
    )
    p_experiment.add_argument(
        "--rc-config", type=str, default=None, metavar="YAML",
        help="Config ResearchClaw (default: config.researchclaw.yaml)",
    )

    # ── reset ──
    p_reset = sub.add_parser(
        "reset", help="Elimina tutti i checkpoint per ripartire da zero",
        description=(
            "Elimina tutti i checkpoint salvati (pipeline, ingestion, writer,\n"
            "ricerca e stato L1.5) per poter ripartire da zero.\n\n"
            "Esempi:\n"
            "  paperhunter reset              # Elimina tutto\n"
            "  paperhunter reset --dry-run     # Mostra cosa verrebbe eliminato\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_reset.add_argument("--dry-run", action="store_true",
                         help="Mostra cosa verrebbe eliminato senza cancellare")
    p_reset.add_argument("--yes", "-y", action="store_true",
                         help="Salta conferma interattiva")

    return parser


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def run() -> None:
    """Entry point principale (da pyproject.toml → paperhunter)."""
    parser = build_parser()
    args = parser.parse_args()

    # Config custom
    if args.config:
        from paperhunter.config import reload_settings
        reload_settings(Path(args.config))

    # Logging
    _init_logging(verbose=args.verbose)

    # Dispatch
    if not args.command:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "search": cmd_search,
        "ingest": cmd_ingest,
        "summarize": cmd_summarize,
        "index": cmd_index,
        "write": cmd_write,
        "query": cmd_query,
        "stats": cmd_stats,
        "export": cmd_export,
        "cleanup": cmd_cleanup,
        "checkpoint": cmd_checkpoint,
        "reset": cmd_reset,
        "figures": cmd_figures,
        "experiment": cmd_experiment,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        exit_code = handler(args)
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\n\nInterrotto dall'utente.")
        sys.exit(130)
    except Exception as e:
        log = _get_logger()
        log.error(f"Errore fatale: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    run()
