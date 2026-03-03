"""
PaperHunter — CLI Principale v2.1

Subcommands:
  search      Cerca e scarica nuovi paper (logica legacy PaperHunter)
  ingest      Pipeline di ingestion incrementale (GROBID + S2 + Qdrant)
  summarize   Genera/rigenera summary L1 e L2
  index       Costruisci/aggiorna indici (BM25 + GraphRAG)
  write       Esegui la pipeline agentica completa (LangGraph)
  query       Query interattiva contro la knowledge base
  stats       Statistiche complete del sistema
  export      Esporta il survey in LaTeX / BibTeX
  cleanup     Elimina paper scaricati per intervallo di date

Esempi:
  paperhunter search -k "prompt injection" -c "Papers about prompt injection attacks" -o ./scaricati/Prompt_injections1
  paperhunter search -k "LLM security" --date-from 2025-10 --date-to 2026-03
  paperhunter ingest
  paperhunter ingest --pdf-dir ./scaricati --batch-size 5
  paperhunter summarize --l1 --l2
  paperhunter index --bm25 --graph
  paperhunter write
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
# Subcommand: search  (logica legacy PaperHunter — asincrona)
# ═══════════════════════════════════════════════════════════════════════════

def cmd_search(args: argparse.Namespace) -> int:
    """Cerca e scarica paper (logica legacy PaperHunter)."""
    from paperhunter.config import settings, ensure_download_dir
    from paperhunter.search_engine import PaperAggregator
    from paperhunter.relevance_filter import RelevanceFilter
    from paperhunter.download_manager import Downloader
    from paperhunter.report_generator import generate_report

    logger = _get_logger("paperhunter.search")

    keywords = args.keywords
    criteria = args.criteria
    if not criteria and keywords:
        criteria = f"I want papers related to: {keywords}. Include all relevant research on this topic."
        logger.info(f"Criteria auto-generati: {criteria}")

    max_results = args.max_results or settings.max_results_per_source
    download_dir = Path(args.output_dir) if args.output_dir else ensure_download_dir()

    # Interattivo se mancano keyword
    if not keywords:
        keywords, criteria = _get_search_input()

    aggregator = PaperAggregator(max_results_per_source=max_results)
    relevance_filter = RelevanceFilter(provider=args.llm_provider)
    downloader = Downloader(download_dir=download_dir)

    # Date range filter (YYYY-MM or YYYY-MM-DD)
    date_from = getattr(args, "date_from", None)
    date_to = getattr(args, "date_to", None)

    async def _run():
        logger.info(f"Ricerca con keyword: '{keywords}'")
        if date_from or date_to:
            logger.info(f"Filtro date: {date_from or '*'} → {date_to or '*'}")
        search_result = await aggregator.search(keywords, date_from=date_from, date_to=date_to)
        logger.info(f"Trovati {search_result.total_found} paper da {len(search_result.sources_queried)} fonti")

        if not search_result.papers:
            logger.warning("Nessun paper trovato. Prova keyword diversi.")
            return 1

        logger.info(f"Filtro con criteri: {criteria[:100]}...")
        filtered = await relevance_filter.filter_papers(search_result.papers, criteria)
        relevant = relevance_filter.get_relevant_papers(filtered)
        logger.info(f"{len(relevant)} paper rilevanti trovati")

        if not relevant:
            logger.warning("Nessun paper rilevante. Prova ad aggiustare i criteri.")
            return 1

        download_results = []
        if not args.no_download:
            logger.info(f"Download di {len(relevant)} paper...")
            download_results = await downloader.batch_download(relevant)
            ok = sum(1 for r in download_results if r.success)
            logger.info(f"Scaricati {ok}/{len(relevant)} paper")

        report_path = generate_report(
            keywords=keywords, criteria=criteria,
            search_result=search_result, filtered_results=filtered,
            download_results=download_results, relevant_papers=relevant,
            output_dir=download_dir, max_results_per_source=max_results,
            categories=settings.default_categories,
            llm_provider=args.llm_provider, llm_model=None,
        )
        logger.info(f"Report salvato: {report_path}")
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
    """Genera/rigenera summary L1 e/o L2."""
    from paperhunter.indexing import QdrantStore

    logger = _get_logger("paperhunter.summarize")

    # Se nessun flag: genera entrambi
    do_l1 = args.l1 or (not args.l1 and not args.l2 and not args.invalidate)
    do_l2 = args.l2 or (not args.l1 and not args.l2 and not args.invalidate)
    do_invalidate = args.invalidate

    store = QdrantStore()
    errors = 0

    if do_l1:
        logger.info("Generazione summary L1 (singolo paper)...")
        errors += _run_l1_summarization(store, args, logger)

    if do_l2:
        logger.info("Generazione summary L2 (meta-summary per categoria)...")
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
                collection_name="paper_chunks", limit=100, offset=offset,
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

        missing = [(bk, pid) for bk, pid in paper_keys if bk not in existing_l1]

        if not missing:
            logger.info(f"Tutti {len(paper_keys)} paper hanno già un L1. Niente da fare.")
            return 0

        logger.info(f"{len(missing)} paper senza L1 (di {len(paper_keys)} totali)")

        generated = 0
        errors_count = 0
        for bk, pid in missing:
            try:
                chunks_data = store.get_paper_chunks(bk, collection_name="paper_chunks")
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


def _run_l2_summarization(store, args, logger) -> int:
    """Genera L2 per tutte le categorie."""
    from paperhunter.summarization import generate_l2_for_all_categories

    try:
        category = getattr(args, "category", None)
        categories = [category] if category else None

        summaries = generate_l2_for_all_categories(
            save_to_store=True, store=store,
            skip_existing=not args.force, categories=categories,
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
    """Costruisce/aggiorna gli indici (BM25 + GraphRAG)."""
    logger = _get_logger("paperhunter.index")

    do_bm25 = args.bm25 or (not args.bm25 and not args.graph)
    do_graph = args.graph or (not args.bm25 and not args.graph)
    errors = 0

    if do_bm25:
        errors += _build_bm25_index(logger)
    if do_graph:
        errors += _build_graph_index(args, logger)

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
        for collection in ["paper_chunks", "l1_summaries", "l2_summaries"]:
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
            f"PropertyGraph: {report.entities_created} entità, "
            f"{report.relations_created} relazioni in {report.elapsed_seconds:.1f}s"
        )

        # Import Neo4j opzionale
        if settings.neo4j.enabled and not args.skip_neo4j:
            try:
                from paperhunter.graphrag import Neo4jManager
                with Neo4jManager() as mgr:
                    stats_neo = mgr.import_entities_and_relations(
                        report.entities, report.relations,
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
        if settings.output.generate_latex:
            from paperhunter.postprocessing import export_latex_from_state
            template = getattr(args, "template", None) or settings.output.latex_template
            tex_path = export_latex_from_state(state, template=template)
            if tex_path:
                logger.info(f"LaTeX: {tex_path}")

        if settings.output.generate_bibtex:
            from paperhunter.postprocessing import generate_bib_from_state
            bib_path, bib_report = generate_bib_from_state(state)
            if bib_path:
                logger.info(f"BibTeX: {bib_path} ({bib_report.get('total_filtered', 0)} entry)")

        if settings.output.generate_provenance_json:
            from paperhunter.postprocessing import create_provenance_from_state
            tracker = create_provenance_from_state(state)
            prov_path = tracker.save_json()
            logger.info(f"Provenance: {prov_path}")

        from paperhunter.postprocessing import create_version_from_state
        version = create_version_from_state(state)
        logger.info(f"Versione: {version.version_number}")

    except Exception as e:
        logger.warning(f"Errore export automatico (non bloccante): {e}")


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
        for name in ["paper_chunks", "l1_summaries", "l2_summaries"]:
            c = q.get(name, {})
            parts.append(f"{name}: {c.get('total_points', 0)}")
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
        for coll in ["paper_chunks", "l1_summaries", "l2_summaries"]:
            c = q.get(coll, {})
            print(f"    - {coll}: {c.get('total_points', 0)}")
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

    # LaTeX
    if args.latex or (not args.latex and not args.bib_only):
        try:
            from paperhunter.postprocessing import export_latex, print_export_summary
            template = args.template or settings.output.latex_template
            tex_path = export_latex(
                md_content, output_path=output_dir / "survey.tex",
                bib_file=args.bib_file, template=template,
                title=args.title, author=args.author,
                copy_bib=True,
                bib_source_path=version_dir / args.bib_file if version_dir else None,
            )
            logger.info(f"LaTeX: {tex_path}")
            print_export_summary(tex_path.read_text(encoding="utf-8"), output_path=tex_path)
        except Exception as e:
            logger.error(f"Errore LaTeX: {e}")
            errors += 1

    # BibTeX filtrato
    if args.bib_only or (not args.latex and not args.bib_only):
        try:
            from paperhunter.postprocessing import generate_final_bib
            cite_re = re.compile(r'\[([A-Z][a-z]+_\d{4}_\w+)\]')
            used_keys = set(cite_re.findall(md_content))
            if used_keys:
                bib_path, bib_report = generate_final_bib(
                    used_keys=used_keys,
                    output_path=output_dir / "references_final.bib",
                    survey_text=md_content,
                )
                if bib_path:
                    logger.info(f"BibTeX: {bib_path} ({bib_report.get('total_filtered', 0)} entry)")
            else:
                logger.warning("Nessuna citazione trovata nel survey.")
        except Exception as e:
            logger.error(f"Errore BibTeX: {e}")
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


# ═══════════════════════════════════════════════════════════════════════════
# Parser principale
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser con tutti i subcommand."""
    parser = argparse.ArgumentParser(
        prog="paperhunter",
        description=(
            "PaperHunter v2.0 — End-to-end LLM Security Survey Generator.\n\n"
            "Pipeline completa: search → ingest → summarize → index → write → export"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Workflow tipico:\n"
            "  1. paperhunter search -k 'prompt injection'   # Cerca e scarica paper\n"
            "  2. paperhunter ingest                          # Processa i PDF\n"
            "  3. paperhunter summarize                       # Genera L1 + L2\n"
            "  4. paperhunter index                           # Costruisci indici\n"
            "  5. paperhunter write                           # Genera il survey\n"
            "  6. paperhunter export --template acl           # Esporta in LaTeX\n"
        ),
    )

    # --- Opzioni globali ---
    parser.add_argument("--config", type=str, metavar="PATH",
                        help="Percorso al file config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Abilita output verbose (DEBUG)")
    parser.add_argument("--version", action="version", version="paperhunter 2.0.0")

    sub = parser.add_subparsers(dest="command", help="Comandi disponibili")

    # ── search ──
    p_search = sub.add_parser(
        "search", help="Cerca e scarica nuovi paper scientifici",
        description="Cerca paper su ArXiv, Semantic Scholar e altre fonti, filtra con LLM e scarica i PDF.",
    )
    p_search.add_argument("-k", "--keywords", type=str,
                          help="Keyword di ricerca (es. 'prompt injection')")
    p_search.add_argument("-c", "--criteria", type=str,
                          help="Criteri di filtro per LLM")
    p_search.add_argument("-n", "--max-results", type=int,
                          help="Max risultati per fonte (default: da config.yaml)")
    p_search.add_argument("--no-download", action="store_true",
                          help="Solo ricerca, nessun download")
    p_search.add_argument("--llm-provider", type=str, choices=["google", "anthropic"],
                          help="Provider LLM per filtro (default: google)")
    p_search.add_argument("-o", "--output-dir", type=str,
                          help="Directory output PDF")
    p_search.add_argument("--date-from", type=str, metavar="YYYY-MM",
                          help="Data minima pubblicazione (es. 2025-10)")
    p_search.add_argument("--date-to", type=str, metavar="YYYY-MM",
                          help="Data massima pubblicazione (es. 2026-03)")

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

    # ── summarize ──
    p_sum = sub.add_parser(
        "summarize", help="Genera/rigenera summary L1 e L2",
        description="Summary L1 (singolo paper) e L2 (meta per categoria). Senza flag, genera entrambi.",
    )
    p_sum.add_argument("--l1", action="store_true", help="Solo summary L1")
    p_sum.add_argument("--l2", action="store_true", help="Solo summary L2")
    p_sum.add_argument("--invalidate", action="store_true",
                       help="Check invalidazione L2 e rigenera se necessario")
    p_sum.add_argument("--category", type=str,
                       help="Limita a una categoria (es. 'Prompt Injection')")
    p_sum.add_argument("--force", action="store_true",
                       help="Forza rigenerazione anche se già esistenti")

    # ── index ──
    p_index = sub.add_parser(
        "index", help="Costruisci/aggiorna indici (BM25 + GraphRAG)",
        description="Costruisce BM25 e PropertyGraph. Senza flag, entrambi.",
    )
    p_index.add_argument("--bm25", action="store_true", help="Solo BM25")
    p_index.add_argument("--graph", action="store_true", help="Solo PropertyGraph")
    p_index.add_argument("--skip-neo4j", action="store_true",
                         help="Salta import Neo4j")
    p_index.add_argument("--clear-graph", action="store_true",
                         help="Pulisci Neo4j prima di importare")

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
