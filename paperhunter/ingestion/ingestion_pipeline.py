"""
Ingestion Pipeline — Orchestratore end-to-end per il processing incrementale dei PDF.

Pipeline completa che:
1. Scansiona la directory PDF e identifica nuovi/modificati (via hash registry)
2. Processa ogni PDF con GROBID (→ GROBIDOutput strutturato)
3. Arricchisce ogni paper con metadati Semantic Scholar
4. Genera chiavi e entry BibTeX
5. Crea chunk GROBID-aware
6. Indicizza i chunk nel vector store Qdrant
7. Aggiorna il registro hash

Supporta: resume from checkpoint, batch processing, progress bar Rich,
logging dettagliato, statistiche riassuntive.

Uso::

    from paperhunter.ingestion.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    report = pipeline.run()
    print(report)

    # Riesecuzione → 0 nuovi paper
    report2 = pipeline.run()
    assert report2.processed == 0
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.config import settings, ensure_output_dirs
from paperhunter.ingestion.bibtex_generator import enrich_paper_bibtex, save_master_bib
from paperhunter.ingestion.chunker import chunk_grobid_output, chunk_raw_text, get_chunk_stats
from paperhunter.ingestion.grobid_client import GrobidClient
from paperhunter.ingestion.pdf_hasher import (
    create_registry_entry,
    get_new_or_modified_pdfs,
    get_registry_stats,
    load_hash_registry,
    save_hash_registry,
    update_registry_entry,
)
from paperhunter.ingestion.semantic_scholar import SemanticScholarEnricher
from paperhunter.ingestion.figure_extractor import FigureExtractor
from paperhunter.ingestion.metadata_enricher import MetadataEnricher
from paperhunter.models import GROBIDOutput, Paper, PaperChunk
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_panel,
    print_success,
    print_table,
    print_warning,
    progress_bar,
)

logger = get_logger(__name__)


# =============================================================================
# Recency sorting helpers
# =============================================================================

# Matches a 4-digit year (2000–2099) in isolation (not part of a larger number)
_YEAR_RE = re.compile(r"(?<!\d)(20[0-2]\d)(?!\d)")


def _extract_year_from_filename(filename: str) -> int | None:
    """Prova ad estrarre un anno (2000-2029) dal nome del file PDF.

    Cerca sequenze tipo ``2024``, ``_2023_``, ecc.
    Se ci sono più match plausibili restituisce il **più recente**
    (euristica: più probabilmente è l'anno di pubblicazione).

    Args:
        filename: Nome del file (con o senza estensione).

    Returns:
        Anno come int, o None se non trovato.
    """
    matches = _YEAR_RE.findall(filename)
    if not matches:
        return None
    # Restituisci il più recente tra i match trovati
    return max(int(m) for m in matches)


def _get_pdf_sort_key(pdf_info: dict[str, Any]) -> tuple[int, int, float]:
    """Calcola una chiave di ordinamento per un PDF basata sulla recency.

    Strategia a 3 livelli (priorità decrescente):
    1. Anno estratto dal filename (se presente)
    2. Anno derivato dalla data di modifica del file
    3. Timestamp di modifica raw (per disambiguare a parità di anno)

    La tupla è costruita in modo che ``sorted(..., reverse=True)``
    metta i paper più recenti in cima.

    Args:
        pdf_info: Dizionario con le informazioni del PDF (da pdf_hasher).

    Returns:
        Tupla ``(year, month, mtime)`` per ordinamento decrescente.
    """
    filename = pdf_info.get("filename", "")
    abs_path = pdf_info.get("absolute_path")

    # 1. Anno dal filename
    year_from_name = _extract_year_from_filename(filename)

    # 2. Anno e mese dalla data di modifica del file
    mtime = pdf_info.get("mtime")
    if mtime is None and abs_path is not None:
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0
    mtime = mtime or 0.0

    dt = datetime.fromtimestamp(mtime) if mtime > 0 else datetime(2000, 1, 1)
    year_from_mtime = dt.year
    month_from_mtime = dt.month

    # L'anno dal filename vince, ma usiamo il mese da mtime come approssimazione
    final_year = year_from_name if year_from_name is not None else year_from_mtime
    final_month = month_from_mtime  # fallback, non estraibile dal filename

    return (final_year, final_month, mtime)


def sort_pdfs_by_recency(
    pdfs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ordina una lista di PDF per data di pubblicazione stimata (più recenti prima).

    L'ordinamento è **decrescente**: i paper più recenti vengono restituiti
    per primi, così che il processing della pipeline li elabori con priorità.

    Fonti di informazione temporale (in ordine di affidabilità):
    1. Anno estratto dal nome del file (regex ``20[0-2]\\d``)
    2. Data di modifica del file PDF (``os.path.getmtime``)

    Ogni PDF viene annotato con i campi:
    - ``sort_year``: anno usato per l'ordinamento
    - ``sort_source``: fonte dell'anno (``"filename"`` | ``"mtime"``)

    Args:
        pdfs: Lista di dict PDF (da ``get_new_or_modified_pdfs``).

    Returns:
        Lista ordinata (copia), non modifica l'originale.
    """
    if not pdfs:
        return []

    annotated = []
    for pdf_info in pdfs:
        key = _get_pdf_sort_key(pdf_info)
        year_from_name = _extract_year_from_filename(pdf_info.get("filename", ""))

        # Annotazione per tracking
        info_copy = dict(pdf_info)
        info_copy["sort_year"] = key[0]
        info_copy["sort_source"] = "filename" if year_from_name is not None else "mtime"
        annotated.append((key, info_copy))

    # Ordinamento decrescente (più recenti prima)
    annotated.sort(key=lambda x: x[0], reverse=True)

    sorted_pdfs = [item[1] for item in annotated]

    # Log statistiche
    years = [p["sort_year"] for p in sorted_pdfs]
    if years:
        from_filename = sum(1 for p in sorted_pdfs if p["sort_source"] == "filename")
        from_mtime = sum(1 for p in sorted_pdfs if p["sort_source"] == "mtime")
        logger.info(
            "Ordinamento per recency: %d PDF ordinati (range: %d-%d). "
            "Fonte anno: %d da filename, %d da mtime",
            len(sorted_pdfs),
            min(years),
            max(years),
            from_filename,
            from_mtime,
        )

    return sorted_pdfs


# =============================================================================
# Ingestion Report
# =============================================================================

@dataclass
class IngestionReport:
    """Report riassuntivo di un'esecuzione della pipeline di ingestion.

    Attributes:
        processed: Numero di PDF processati con successo.
        skipped: Numero di PDF saltati (già nel registro, non modificati).
        errors: Numero di PDF che hanno generato errori.
        total_on_disk: Numero totale di PDF trovati su disco.
        total_chunks: Numero totale di chunk generati e indicizzati.
        total_time_seconds: Tempo totale di esecuzione in secondi.
        papers: Lista di Paper processati con successo.
        error_details: Dettagli degli errori per ogni PDF fallito.
        grobid_stats: Statistiche GROBID (success, fallback, failed).
        s2_stats: Statistiche Semantic Scholar enrichment.
        resumed_from_checkpoint: Se True, l'esecuzione è stata ripresa da un checkpoint.
        checkpoint_path: Percorso del file checkpoint usato.
    """

    processed: int = 0
    skipped: int = 0
    errors: int = 0
    total_on_disk: int = 0
    total_chunks: int = 0
    total_time_seconds: float = 0.0
    papers: list[Paper] = field(default_factory=list)
    error_details: dict[str, str] = field(default_factory=dict)
    grobid_stats: dict[str, int] = field(default_factory=lambda: {
        "success": 0, "fallback": 0, "failed": 0,
    })
    s2_stats: dict[str, int] = field(default_factory=lambda: {
        "enriched": 0, "not_found": 0, "failed": 0,
    })
    resumed_from_checkpoint: bool = False
    checkpoint_path: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"IngestionReport(processed={self.processed}, skipped={self.skipped}, "
            f"errors={self.errors}, chunks={self.total_chunks}, "
            f"time={self.total_time_seconds:.1f}s)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializza il report in un dizionario (senza oggetti Paper)."""
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total_on_disk": self.total_on_disk,
            "total_chunks": self.total_chunks,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "grobid_stats": self.grobid_stats,
            "s2_stats": self.s2_stats,
            "error_details": self.error_details,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Checkpoint Manager (per resume from checkpoint)
# =============================================================================

class _CheckpointManager:
    """Gestisce il salvataggio e il caricamento dei checkpoint di ingestion.

    I checkpoint permettono di riprendere la pipeline dall'ultimo PDF
    processato con successo, evitando di rielaborare PDF già completati
    in un'esecuzione interrotta.

    Struttura checkpoint::

        {
            "started_at": "2026-02-20T10:00:00",
            "last_updated": "2026-02-20T10:15:00",
            "completed_pdfs": ["Hallucinations1/paper1.pdf", ...],
            "current_batch": 3,
            "total_pdfs_to_process": 118,
            "registry_snapshot": {...}
        }
    """

    def __init__(self, checkpoint_dir: str | Path | None = None):
        self.checkpoint_dir = Path(
            checkpoint_dir or settings.output.checkpoints_dir
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "ingestion_checkpoint.json"

    def has_checkpoint(self) -> bool:
        """Verifica se esiste un checkpoint pendente."""
        return self.checkpoint_file.exists()

    def load(self) -> dict[str, Any]:
        """Carica il checkpoint corrente.

        Returns:
            Dict con i dati del checkpoint, o dict vuoto se non esiste.
        """
        if not self.checkpoint_file.exists():
            return {}
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Checkpoint corrotto, ignoro: %s", e)
            return {}

    def save(
        self,
        completed_pdfs: list[str],
        total_to_process: int,
        registry: dict[str, Any],
    ) -> None:
        """Salva il checkpoint corrente.

        Args:
            completed_pdfs: Lista dei percorsi relativi dei PDF completati.
            total_to_process: Totale PDF da processare in questa esecuzione.
            registry: Snapshot del registro hash corrente.
        """
        checkpoint_data = {
            "started_at": getattr(self, "_started_at", datetime.now().isoformat()),
            "last_updated": datetime.now().isoformat(),
            "completed_pdfs": completed_pdfs,
            "total_pdfs_to_process": total_to_process,
        }
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            # Salva anche il registro hash ad ogni checkpoint
            save_hash_registry(registry)
        except OSError as e:
            logger.error("Errore salvataggio checkpoint: %s", e)

    def clear(self) -> None:
        """Rimuove il checkpoint (pipeline completata con successo)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.debug("Checkpoint rimosso: %s", self.checkpoint_file)


# =============================================================================
# Ingestion Pipeline
# =============================================================================

class IngestionPipeline:
    """Pipeline end-to-end per il processing incrementale dei PDF scientifici.

    Processa solo PDF nuovi o modificati rispetto al registro hash.
    Ogni PDF passa attraverso: GROBID → S2 enrichment → BibTeX → chunking → Qdrant.

    Args:
        pdf_base_dir: Directory base contenente i PDF categorizzati.
                      Default: da ``config.yaml`` → ``ingestion.pdf_base_dir``.
        batch_size: Numero di PDF per batch GROBID. Default: da config.
        skip_s2: Se True, salta l'enrichment Semantic Scholar.
        skip_qdrant: Se True, salta l'indicizzazione nel vector store.
        use_checkpoint: Se True, abilita resume from checkpoint.

    Esempio::

        pipeline = IngestionPipeline()
        report = pipeline.run()

        # Solo una categoria
        report = pipeline.run(pdf_base_dir="./scaricati/Hallucinations1")
    """

    def __init__(
        self,
        pdf_base_dir: str | Path | None = None,
        batch_size: int | None = None,
        skip_s2: bool = False,
        skip_qdrant: bool = False,
        use_checkpoint: bool = True,
        force: bool = False,
    ):
        self.pdf_base_dir = Path(pdf_base_dir or settings.ingestion.pdf_base_dir)
        self.batch_size = batch_size or settings.ingestion.batch_size
        self.skip_s2 = skip_s2
        self.skip_qdrant = skip_qdrant
        self.use_checkpoint = use_checkpoint
        self.force = force

        # Componenti della pipeline (lazy init)
        self._grobid: GrobidClient | None = None
        self._enricher: SemanticScholarEnricher | None = None
        self._store: Any | None = None  # QdrantStore (importato lazy)
        self._figure_extractor: FigureExtractor | None = None
        self._checkpoint = _CheckpointManager()
        self.metadata_enricher = MetadataEnricher()

    # ------------------------------------------------------------------
    # Lazy initialization dei componenti
    # ------------------------------------------------------------------

    def _get_grobid(self) -> GrobidClient:
        """Inizializza (lazy) il client GROBID e testa la connessione."""
        if self._grobid is None:
            self._grobid = GrobidClient()
            if not self._grobid.test_connection():
                logger.warning(
                    "GROBID non raggiungibile! I PDF saranno processati "
                    "con fallback PyMuPDF se abilitato."
                )
        return self._grobid

    def _get_enricher(self) -> SemanticScholarEnricher:
        """Inizializza (lazy) l'enricher Semantic Scholar."""
        if self._enricher is None:
            self._enricher = SemanticScholarEnricher()
        return self._enricher

    def _get_store(self):
        """Inizializza (lazy) il vector store Qdrant."""
        if self._store is None:
            from paperhunter.indexing.qdrant_store import QdrantStore
            self._store = QdrantStore()
        return self._store

    def _get_figure_extractor(self) -> FigureExtractor:
        """Inizializza (lazy) il figure extractor."""
        if self._figure_extractor is None:
            self._figure_extractor = FigureExtractor()
        return self._figure_extractor

    # ------------------------------------------------------------------
    # Pipeline principale
    # ------------------------------------------------------------------

    def run(self, pdf_base_dir: str | Path | None = None) -> IngestionReport:
        """Esegue la pipeline di ingestion completa.

        Workflow:
        1. Scan directory → identifica PDF nuovi/modificati
        2. Per ogni PDF → GROBID processing → GROBIDOutput
        3. Per ogni paper → Semantic Scholar enrichment
        4. Per ogni paper → BibTeX generation
        5. Per ogni paper → Chunking GROBID-aware → PaperChunk[]
        6. Indicizza chunk nel vector store Qdrant
        7. Aggiorna hash registry e salva master .bib

        Args:
            pdf_base_dir: Override della directory PDF (opzionale).

        Returns:
            IngestionReport con tutte le statistiche dell'esecuzione.
        """
        start_time = time.time()
        base_dir = Path(pdf_base_dir) if pdf_base_dir else self.pdf_base_dir
        report = IngestionReport()

        # Assicura che le directory di output esistano
        ensure_output_dirs()

        print_panel(
            "Ingestion Pipeline",
            f"Directory: {base_dir}\nBatch size: {self.batch_size}\n"
            f"S2 enrichment: {'ON' if not self.skip_s2 else 'OFF'}\n"
            f"Qdrant indexing: {'ON' if not self.skip_qdrant else 'OFF'}\n"
            f"Checkpoint: {'ON' if self.use_checkpoint else 'OFF'}",
            style="bold cyan",
        )

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Scan e identifica PDF nuovi/modificati
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("STEP 1: Scansione directory e rilevamento delta")
        logger.info("=" * 60)

        if self.force:
            logger.info("--force attivo: il registro hash verrà ignorato, tutti i PDF saranno processati")
            registry = {}
        else:
            registry = load_hash_registry()
        new_pdfs = get_new_or_modified_pdfs(pdf_dir=base_dir, registry=registry)
        report.total_on_disk = len(new_pdfs) + len(registry)
        report.skipped = len(registry) - sum(
            1 for p in new_pdfs if p.get("status") == "modified"
        )

        if not new_pdfs:
            elapsed = time.time() - start_time
            report.total_time_seconds = elapsed
            print_success(
                f"Nessun PDF nuovo o modificato trovato. "
                f"Registro: {len(registry)} PDF già processati."
            )
            return report

        logger.info("PDF da processare: %d (nuovi: %d, modificati: %d)",
            len(new_pdfs),
            sum(1 for p in new_pdfs if p.get("status") == "new"),
            sum(1 for p in new_pdfs if p.get("status") == "modified"),
        )

        # ═══════════════════════════════════════════════════════════════
        # STEP 1b: Resume from checkpoint (se disponibile)
        # ═══════════════════════════════════════════════════════════════
        already_completed: set[str] = set()
        if self.use_checkpoint and self._checkpoint.has_checkpoint():
            checkpoint = self._checkpoint.load()
            already_completed = set(checkpoint.get("completed_pdfs", []))
            if already_completed:
                before = len(new_pdfs)
                new_pdfs = [
                    p for p in new_pdfs
                    if p["relative_path"] not in already_completed
                ]
                skipped_by_cp = before - len(new_pdfs)
                if skipped_by_cp > 0:
                    report.resumed_from_checkpoint = True
                    report.checkpoint_path = str(self._checkpoint.checkpoint_file)
                    logger.info(
                        "Resume da checkpoint: %d PDF già completati, "
                        "%d rimanenti",
                        skipped_by_cp,
                        len(new_pdfs),
                    )

        if not new_pdfs:
            elapsed = time.time() - start_time
            report.total_time_seconds = elapsed
            print_success("Tutti i PDF del checkpoint sono già completati.")
            self._checkpoint.clear()
            return report

        # ═══════════════════════════════════════════════════════════════
        # STEP 1c: Ordinamento per recency (paper recenti prima)
        # ═══════════════════════════════════════════════════════════════
        if settings.recency_priority:
            logger.info("=" * 60)
            logger.info("STEP 1c: Ordinamento per recency (più recenti prima)")
            logger.info("=" * 60)
            new_pdfs = sort_pdfs_by_recency(new_pdfs)
        else:
            logger.info("=" * 60)
            logger.info("STEP 1c: Ordinamento per recency DISABILITATO (recency_priority=false)")
            logger.info("=" * 60)

        # Inizializza checkpoint per questa esecuzione
        self._checkpoint._started_at = datetime.now().isoformat()
        completed_pdfs_list: list[str] = list(already_completed)

        # ═══════════════════════════════════════════════════════════════
        # STEP 2-6: Processing in batch
        # ═══════════════════════════════════════════════════════════════
        all_bibtex_entries: list[str] = []
        total_pdfs = len(new_pdfs)

        # Dividi in batch
        batches = [
            new_pdfs[i:i + self.batch_size]
            for i in range(0, total_pdfs, self.batch_size)
        ]

        logger.info("Processing %d PDF in %d batch (batch_size=%d)",
                     total_pdfs, len(batches), self.batch_size)

        pipeline_aborted = False

        for batch_idx, batch in enumerate(batches, 1):
            logger.info("-" * 50)
            logger.info("BATCH %d/%d (%d PDF)", batch_idx, len(batches), len(batch))
            logger.info("-" * 50)

            # GC management: disabilita il GC automatico durante l'elaborazione
            # intensiva del batch per evitare pause non deterministiche.
            gc.disable()
            try:
                for pdf_info in progress_bar(
                    batch,
                    description=f"Batch {batch_idx}/{len(batches)}",
                    total=len(batch),
                ):
                    rel_path = pdf_info["relative_path"]
                    try:
                        result = self._process_single_pdf(
                            pdf_info=pdf_info,
                            registry=registry,
                            report=report,
                        )

                        if result is not None:
                            paper, chunks, figure_chunks = result
                            report.processed += 1
                            report.total_chunks += len(chunks) + len(figure_chunks)
                            report.papers.append(paper)

                            # Raccogli BibTeX
                            if paper.bibtex_entry:
                                all_bibtex_entries.append(paper.bibtex_entry)

                            del chunks, figure_chunks  # rilascia oggetti intermedi

                        del result  # rilascia esplicitamente l'oggetto result

                        # Segna come completato nel checkpoint
                        completed_pdfs_list.append(rel_path)

                    except RuntimeError as exc:
                        # Errore critico (modello non trovato, auth, ecc.)
                        # → ferma la pipeline, non ha senso continuare
                        report.errors += 1
                        report.error_details[rel_path] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        logger.error(
                            "ERRORE CRITICO — pipeline interrotta: %s\n%s",
                            exc,
                            traceback.format_exc(),
                        )
                        # Salva checkpoint parziale prima di uscire
                        if self.use_checkpoint:
                            self._checkpoint.save(
                                completed_pdfs=completed_pdfs_list,
                                total_to_process=total_pdfs,
                                registry=registry,
                            )
                        from paperhunter.utils.logging_setup import print_error
                        print_error(
                            f"Pipeline interrotta per errore critico: {exc}"
                        )
                        pipeline_aborted = True
                        break

                    except Exception as exc:
                        report.errors += 1
                        report.error_details[rel_path] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        logger.error(
                            "ERRORE fatale processing '%s': %s\n%s",
                            rel_path,
                            exc,
                            traceback.format_exc(),
                        )
            finally:
                gc.enable()
                gc.collect()

            # Se pipeline interrotta per errore critico, esci dal loop batch
            if pipeline_aborted:
                break

            # Checkpoint dopo ogni batch
            if self.use_checkpoint:
                self._checkpoint.save(
                    completed_pdfs=completed_pdfs_list,
                    total_to_process=total_pdfs,
                    registry=registry,
                )
                logger.info(
                    "Checkpoint salvato: %d/%d completati",
                    len(completed_pdfs_list),
                    total_pdfs + len(already_completed),
                )

        # ═══════════════════════════════════════════════════════════════
        # STEP 7: Finalizzazione
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("STEP 7: Finalizzazione")
        logger.info("=" * 60)

        # Salva il registro hash definitivo
        save_hash_registry(registry)

        # Salva master .bib cumulativo
        if all_bibtex_entries:
            # Carica anche entry BibTeX esistenti dal registro
            existing_entries = self._collect_existing_bibtex(registry)
            all_entries = existing_entries + all_bibtex_entries
            bib_path = save_master_bib(all_entries)
            logger.info("Master .bib salvato: %s (%d entry)", bib_path, len(all_entries))

        # Pulisci il checkpoint (pipeline completata con successo)
        if self.use_checkpoint:
            self._checkpoint.clear()

        # Report finale
        elapsed = time.time() - start_time
        report.total_time_seconds = elapsed

        self._print_report(report)

        return report

    # ------------------------------------------------------------------
    # Processing singolo PDF (Step 2-6 per un PDF)
    # ------------------------------------------------------------------

    def _process_single_pdf(
        self,
        pdf_info: dict[str, Any],
        registry: dict[str, Any],
        report: IngestionReport,
    ) -> tuple[Paper, list[PaperChunk], list[PaperChunk]] | None:
        """Processa un singolo PDF attraverso tutti gli step della pipeline.

        Args:
            pdf_info: Informazioni del PDF (da pdf_hasher.get_new_or_modified_pdfs).
            registry: Registro hash (aggiornato in-place).
            report: Report di ingestion (aggiornato in-place per le statistiche).

        Returns:
            Tupla (Paper, text_chunks, figure_chunks) se successo, None se fallimento.
        """
        rel_path = pdf_info["relative_path"]
        abs_path = pdf_info["absolute_path"]
        category = pdf_info["category"]

        logger.info("▶ Processing: %s [%s]", rel_path, category)

        # --- Step 2: GROBID Processing ---
        grobid_output = self._step_grobid(abs_path, rel_path, report)
        if grobid_output is None:
            # Aggiorna registro come fallito
            registry[rel_path] = create_registry_entry(
                pdf_info, grobid_status="failed"
            )
            return None

        # Determina status GROBID
        grobid_status = "success"
        if grobid_output.error_message:
            grobid_status = "fallback"

        # --- Step 3: Crea Paper e arricchisci con S2 ---
        paper = self._step_create_and_enrich_paper(
            pdf_info=pdf_info,
            grobid_output=grobid_output,
            report=report,
        )

        # --- Step 4: BibTeX generation ---
        paper = self._step_bibtex(paper)

        # --- Step 5: Chunking ---
        chunks = self._step_chunking(grobid_output, paper)

        if not chunks:
            logger.warning("Nessun chunk generato per '%s'", rel_path)

        # --- Step 5b: Figure extraction (se abilitato) ---
        figure_chunks = self._step_figure_extraction(abs_path, paper, grobid_output)

        # --- Step 6: Indicizzazione Qdrant ---
        if chunks and not self.skip_qdrant:
            self._step_qdrant_index(chunks, paper)

        # --- Step 6b: Indicizzazione figure in Qdrant ---
        if figure_chunks and not self.skip_qdrant:
            self._step_qdrant_index_figures(figure_chunks, paper)

        # --- Aggiorna registro hash ---
        registry[rel_path] = create_registry_entry(
            pdf_info,
            doi=paper.doi,
            bibtex_key=paper.bibtex_key,
            grobid_status=grobid_status,
        )

        logger.info(
            "✓ Completato: %s → bibtex_key=%s, %d chunk, %d figure, doi=%s",
            rel_path,
            paper.bibtex_key or "N/A",
            len(chunks),
            len(figure_chunks),
            paper.doi or "N/A",
        )

        return paper, chunks, figure_chunks

    # ------------------------------------------------------------------
    # Step individuali
    # ------------------------------------------------------------------

    def _step_grobid(
        self,
        abs_path: Path,
        rel_path: str,
        report: IngestionReport,
    ) -> GROBIDOutput | None:
        """Step 2: Processa il PDF con GROBID.

        Args:
            abs_path: Percorso assoluto del PDF.
            rel_path: Percorso relativo (per logging).
            report: Report di ingestion.

        Returns:
            GROBIDOutput se successo (anche con fallback), None se fallimento totale.
        """
        grobid = self._get_grobid()

        try:
            grobid_output = grobid.process_pdf(
                pdf_path=abs_path,
                paper_id=rel_path,
            )

            if grobid_output.success:
                if grobid_output.error_message:
                    # Fallback PyMuPDF
                    report.grobid_stats["fallback"] += 1
                    logger.info(
                        "GROBID fallback per '%s': %s",
                        rel_path,
                        grobid_output.error_message,
                    )
                else:
                    report.grobid_stats["success"] += 1

                return grobid_output
            else:
                report.grobid_stats["failed"] += 1
                logger.warning(
                    "GROBID completamente fallito per '%s': %s",
                    rel_path,
                    grobid_output.error_message,
                )
                return None

        except Exception as exc:
            report.grobid_stats["failed"] += 1
            logger.error("Eccezione GROBID per '%s': %s", rel_path, exc)
            return None

    def _step_create_and_enrich_paper(
        self,
        pdf_info: dict[str, Any],
        grobid_output: GROBIDOutput,
        report: IngestionReport,
    ) -> Paper:
        """Step 3: Crea oggetto Paper da GROBID + arricchisce con Semantic Scholar.

        Args:
            pdf_info: Informazioni del PDF.
            grobid_output: Output da GROBID.
            report: Report di ingestion.

        Returns:
            Paper arricchito (o base se S2 fallisce/è disabilitato).
        """
        enricher = self._get_enricher()

        # Crea Paper base dai dati GROBID
        paper = enricher.create_paper_from_pdf(
            pdf_info=pdf_info,
            grobid_title=grobid_output.title,
            grobid_authors=grobid_output.authors,
            grobid_abstract=grobid_output.abstract,
        )

        # Segna come processato da GROBID
        paper_dict = paper.model_dump()
        paper_dict["grobid_processed"] = True
        paper_dict["ingested_at"] = datetime.now()
        paper = Paper(**paper_dict)

        # Enrichment S2 (se abilitato)
        s2_raw: dict | None = None
        if not self.skip_s2:
            try:
                enriched = enricher.enrich_paper(paper, skip_if_enriched=False)
                s2_raw = getattr(enricher, "last_s2_raw", None)
                if enriched.semantic_scholar_id:
                    report.s2_stats["enriched"] += 1
                    paper = enriched
                else:
                    report.s2_stats["not_found"] += 1
            except Exception as exc:
                report.s2_stats["failed"] += 1
                logger.warning(
                    "S2 enrichment fallito per '%s': %s",
                    paper.title[:50],
                    exc,
                )
        else:
            logger.debug("S2 enrichment disabilitato, skip")

        # MetadataEnricher — integrazione GROBID + S2 + euristiche (Task 1.4)
        pdf_filename = Path(pdf_info.get("absolute_path", "")).name
        paper = self.metadata_enricher.enrich(
            paper=paper,
            grobid_output=grobid_output,
            s2_data=s2_raw,
            pdf_filename=pdf_filename,
        )
        logger.info(
            "MetadataEnricher: year=%s (src=%s), conf=%.2f",
            paper.year,
            paper.year_source,
            paper.metadata_confidence,
        )

        return paper

    def _step_bibtex(self, paper: Paper) -> Paper:
        """Step 4: Genera chiave e entry BibTeX.

        Args:
            paper: Paper da arricchire con BibTeX.

        Returns:
            Paper con bibtex_key e bibtex_entry.
        """
        try:
            paper = enrich_paper_bibtex(paper)
            logger.debug(
                "BibTeX generato: %s",
                paper.bibtex_key or "N/A",
            )
        except Exception as exc:
            logger.warning("Errore generazione BibTeX: %s", exc)
        return paper

    def _step_chunking(
        self,
        grobid_output: GROBIDOutput,
        paper: Paper,
    ) -> list[PaperChunk]:
        """Step 5: Crea chunk GROBID-aware.

        Args:
            grobid_output: Output strutturato da GROBID.
            paper: Paper con metadati completi.

        Returns:
            Lista di PaperChunk.
        """
        try:
            chunks = chunk_grobid_output(grobid_output, paper)
            logger.debug(
                "Chunking: %d chunk generati per '%s'",
                len(chunks),
                paper.bibtex_key or paper.title[:40],
            )
            return chunks
        except Exception as exc:
            logger.error(
                "Errore chunking per '%s': %s",
                paper.bibtex_key or paper.title[:40],
                exc,
            )
            return []

    def _step_qdrant_index(
        self,
        chunks: list[PaperChunk],
        paper: Paper,
    ) -> int:
        """Step 6: Indicizza i chunk nel vector store Qdrant.

        Se il paper è stato modificato (non nuovo), elimina prima
        i vecchi chunk per evitare duplicati.

        Args:
            chunks: Lista di PaperChunk da indicizzare.
            paper: Paper associato ai chunk.

        Returns:
            Numero di chunk indicizzati.
        """
        store = self._get_store()

        try:
            # Se il paper esisteva già, elimina i vecchi chunk
            if paper.bibtex_key and store.has_paper(paper.bibtex_key):
                deleted = store.delete_by_paper(paper.bibtex_key)
                if deleted:
                    logger.info(
                        "Rimossi %d vecchi chunk per '%s' (upsert)",
                        deleted,
                        paper.bibtex_key,
                    )

            # Indicizza i nuovi chunk
            n_indexed = store.add_chunks(
                chunks,
                collection_name=store.COLLECTION_CHUNKS,
                show_progress=False,
            )
            logger.debug(
                "Qdrant: %d chunk indicizzati per '%s'",
                n_indexed,
                paper.bibtex_key or paper.title[:40],
            )
            return n_indexed

        except Exception as exc:
            logger.error(
                "Errore indicizzazione Qdrant per '%s': %s",
                paper.bibtex_key or paper.title[:40],
                exc,
            )
            return 0

    def _step_figure_extraction(
        self,
        pdf_path: Path,
        paper: Paper,
        grobid_output: GROBIDOutput,
    ) -> list[PaperChunk]:
        """Step 5b: Estrae figure dal PDF e genera descrizioni multimodali.

        Usa Gemini Flash Vision per generare descrizioni semantiche
        delle figure, che vengono poi indicizzate come PaperChunk.

        Args:
            pdf_path: Percorso assoluto del PDF.
            paper: Paper con metadati completi.
            grobid_output: Output GROBID (per le caption).

        Returns:
            Lista di PaperChunk con descrizioni figure.
        """
        if not settings.figure_extraction.enabled:
            return []

        try:
            extractor = self._get_figure_extractor()
            figure_chunks = extractor.extract_and_describe(
                pdf_path=pdf_path,
                paper=paper,
                figure_captions=grobid_output.figure_captions,
            )
            if figure_chunks:
                logger.info(
                    "Figure: %d descrizioni generate per '%s'",
                    len(figure_chunks),
                    paper.bibtex_key or paper.title[:40],
                )

            # Indicizza image embeddings nella collection dedicata
            img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
            if img_chunks and img_embeddings:
                store = self._get_store()
                n_img = store.add_image_embeddings(img_chunks, img_embeddings)
                logger.info(
                    "Image embeddings: %d vettori indicizzati per '%s'",
                    n_img,
                    paper.bibtex_key or paper.title[:40],
                )

            return figure_chunks
        except RuntimeError as exc:
            # Errori critici (modello non trovato, auth) → propaga per fermare la pipeline
            logger.error(
                "Errore critico estrazione figure per '%s': %s",
                paper.bibtex_key or paper.title[:40],
                exc,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Errore estrazione figure per '%s': %s",
                paper.bibtex_key or paper.title[:40],
                exc,
            )
            return []

    def _step_qdrant_index_figures(
        self,
        figure_chunks: list[PaperChunk],
        paper: Paper,
    ) -> int:
        """Step 6b: Indicizza le descrizioni figure nel vector store Qdrant.

        Args:
            figure_chunks: Lista di PaperChunk con descrizioni figure.
            paper: Paper associato.

        Returns:
            Numero di chunk indicizzati.
        """
        store = self._get_store()

        try:
            n_indexed = store.add_chunks(
                figure_chunks,
                collection_name=store.COLLECTION_FIGURES,
                show_progress=False,
            )
            logger.debug(
                "Qdrant figure: %d descrizioni indicizzate per '%s'",
                n_indexed,
                paper.bibtex_key or paper.title[:40],
            )
            return n_indexed
        except Exception as exc:
            logger.error(
                "Errore indicizzazione figure Qdrant per '%s': %s",
                paper.bibtex_key or paper.title[:40],
                exc,
            )
            return 0

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _collect_existing_bibtex(self, registry: dict[str, Any]) -> list[str]:
        """Raccoglie entry BibTeX esistenti dai paper già processati.

        Utile per generare un master .bib cumulativo completo.

        Args:
            registry: Registro hash con i dati dei paper precedenti.

        Returns:
            Lista di entry BibTeX come stringhe.
        """
        entries: list[str] = []
        for _path, entry in registry.items():
            bib_key = entry.get("bibtex_key")
            if bib_key:
                # Le entry complete sono nei Paper, non nel registro.
                # Il registro ha solo la chiave. Le entry sono nel master .bib.
                pass
        # Carica dal master .bib esistente se presente
        master_bib_path = Path(settings.output.base_dir) / "references.bib"
        if master_bib_path.exists():
            try:
                from paperhunter.ingestion.bibtex_generator import load_master_bib
                existing = load_master_bib(master_bib_path)
                entries = list(existing.values())
                logger.debug("Caricate %d entry BibTeX esistenti", len(entries))
            except Exception as exc:
                logger.warning("Errore caricamento master .bib: %s", exc)
        return entries

    def _print_report(self, report: IngestionReport) -> None:
        """Stampa il report finale con tabelle Rich.

        Args:
            report: Report di ingestion completato.
        """
        # Tabella principale
        summary = {
            "PDF processati": report.processed,
            "PDF saltati (già presenti)": report.skipped,
            "PDF con errori": report.errors,
            "Totale PDF su disco": report.total_on_disk,
            "Chunk generati": report.total_chunks,
            "Tempo totale": f"{report.total_time_seconds:.1f}s",
        }
        print_table("Riepilogo Ingestion", summary)

        # Tabella GROBID
        if any(v > 0 for v in report.grobid_stats.values()):
            print_table("GROBID Stats", report.grobid_stats)

        # Tabella S2
        if any(v > 0 for v in report.s2_stats.values()):
            print_table("Semantic Scholar Stats", report.s2_stats)

        # Errori
        if report.error_details:
            print_warning(f"{len(report.error_details)} errori durante l'ingestion:")
            for path, error in list(report.error_details.items())[:10]:
                logger.error("  %s: %s", path, error)
            if len(report.error_details) > 10:
                logger.error(
                    "  ... e altri %d errori",
                    len(report.error_details) - 10,
                )

        # Checkpoint info
        if report.resumed_from_checkpoint:
            print_warning(f"Esecuzione ripresa da checkpoint: {report.checkpoint_path}")

        # Messaggio finale
        if report.errors == 0:
            print_success(
                f"Ingestion completata con successo: "
                f"{report.processed} PDF → {report.total_chunks} chunk"
            )
        else:
            print_warning(
                f"Ingestion completata con {report.errors} errori: "
                f"{report.processed} PDF OK, {report.total_chunks} chunk"
            )

    # ------------------------------------------------------------------
    # API di convenienza
    # ------------------------------------------------------------------

    def get_registry_stats(self) -> dict[str, Any]:
        """Restituisce statistiche sul registro hash corrente.

        Returns:
            Dict con statistiche (totale, per categoria, per status GROBID).
        """
        registry = load_hash_registry()
        return get_registry_stats(registry)

    def get_store_stats(self) -> dict[str, Any]:
        """Restituisce statistiche dal vector store Qdrant.

        Returns:
            Dict con statistiche (chunk totali, per categoria, ecc.).
        """
        if self.skip_qdrant:
            return {"message": "Qdrant indexing is disabled"}
        store = self._get_store()
        return store.get_stats()

    def reset(self, clear_registry: bool = False, clear_store: bool = False) -> None:
        """Reset della pipeline: rimuove checkpoint e opzionalmente registro/store.

        Args:
            clear_registry: Se True, rimuove il registro hash.
            clear_store: Se True, elimina tutte le collezioni Qdrant.
        """
        # Checkpoint
        self._checkpoint.clear()
        logger.info("Checkpoint rimosso")

        # Registro hash
        if clear_registry:
            registry_path = Path(settings.ingestion.hash_registry_path)
            if registry_path.exists():
                registry_path.unlink()
                logger.info("Registro hash rimosso: %s", registry_path)

        # Qdrant
        if clear_store and not self.skip_qdrant:
            store = self._get_store()
            for col in (store.COLLECTION_CHUNKS, store.COLLECTION_L1, store.COLLECTION_L2):
                store.delete_collection(col)
            logger.info("Collezioni Qdrant eliminate")
