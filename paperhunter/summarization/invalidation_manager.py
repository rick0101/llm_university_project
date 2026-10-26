"""
Invalidation Manager — Auto-invalidazione e rigenerazione L2 summary.

Quando nuovi paper vengono ingesti in una categoria, l'InvalidationManager
detecta quali categorie hanno L2 summary obsoleti e li rigenera
automaticamente includendo i nuovi L1 summary.

Funzionalità:
- Confronto paper_ids attuali nel vector store vs paper_ids nel L2 metadata
- Soglia di invalidazione configurabile (``summarization.l2_invalidation_threshold``)
- Marcatura invalidazione nel metadata su disco
- Rigenerazione L2 con versioning incrementale
- Storico dei L2 precedenti (archivio su disco)
- Logging dettagliato di ogni operazione

Uso::

    from paperhunter.summarization.invalidation_manager import InvalidationManager

    mgr = InvalidationManager()

    # Dopo aver ingesto nuovi paper, controlla e rigenera
    result = mgr.check_and_regenerate()

    # Oppure controlla solo quali categorie sono da rigenerare
    categories = mgr.check_invalidation()

    # Oppure forza l'invalidazione di una categoria
    mgr.invalidate_l2("Prompt Injection")

    # Rigenera una categoria specifica
    l2 = mgr.regenerate_l2("Prompt Injection")
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.models import L2Summary
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_success,
    print_table,
    print_warning,
    progress_bar,
)

logger = get_logger(__name__)


# ============================================================================
# Report dataclass
# ============================================================================

@dataclass
class InvalidationReport:
    """Report di un ciclo di check & regenerate.

    Attributes:
        checked_categories: Categorie controllate.
        invalidated_categories: Categorie invalidate.
        regenerated_categories: Categorie rigenerate con successo.
        failed_categories: Categorie la cui rigenerazione è fallita.
        skipped_categories: Categorie senza L2 precedente (nuove).
        new_papers_by_category: Mappa categoria → nuovi paper_ids.
        removed_papers_by_category: Mappa categoria → paper_ids rimossi.
        total_time_seconds: Tempo totale di esecuzione.
    """

    checked_categories: list[str] = field(default_factory=list)
    invalidated_categories: list[str] = field(default_factory=list)
    regenerated_categories: list[str] = field(default_factory=list)
    failed_categories: dict[str, str] = field(default_factory=dict)
    skipped_categories: list[str] = field(default_factory=list)
    new_papers_by_category: dict[str, list[str]] = field(default_factory=dict)
    removed_papers_by_category: dict[str, list[str]] = field(default_factory=dict)
    total_time_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"InvalidationReport(checked={len(self.checked_categories)}, "
            f"invalidated={len(self.invalidated_categories)}, "
            f"regenerated={len(self.regenerated_categories)}, "
            f"failed={len(self.failed_categories)}, "
            f"time={self.total_time_seconds:.1f}s)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializza il report in un dizionario."""
        return {
            "checked_categories": self.checked_categories,
            "invalidated_categories": self.invalidated_categories,
            "regenerated_categories": self.regenerated_categories,
            "failed_categories": self.failed_categories,
            "skipped_categories": self.skipped_categories,
            "new_papers_by_category": self.new_papers_by_category,
            "removed_papers_by_category": self.removed_papers_by_category,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# InvalidationManager
# ============================================================================

class InvalidationManager:
    """Gestore dell'auto-invalidazione e rigenerazione dei summary L2.

    Rileva automaticamente quando nuovi paper sono stati aggiunti a
    una categoria (confrontando i paper_ids nel vector store con quelli
    nel metadata L2 su disco) e rigenera i summary L2 obsoleti.

    Args:
        store: Istanza ``QdrantStore``. Se None, ne crea una nuova.
        threshold: Numero minimo di nuovi paper per triggerare la
            rigenerazione. Default da ``config.yaml`` →
            ``summarization.l2_invalidation_threshold``.
        auto_regenerate: Se True, ``check_and_regenerate()`` rigenera
            automaticamente i L2 invalidati. Se False, marca solo
            come invalidati.
        archive_old: Se True, archivia i L2 precedenti prima di
            rigenerare (conserva lo storico).
    """

    def __init__(
        self,
        *,
        store: Any = None,
        threshold: int | None = None,
        auto_regenerate: bool = True,
        archive_old: bool = True,
    ) -> None:
        self._store = store
        self._threshold = (
            threshold
            if threshold is not None
            else settings.summarization.l2_invalidation_threshold
        )
        self._auto_regenerate = auto_regenerate
        self._archive_old = archive_old

        # Directory per metadata e archivio
        self._l2_metadata_dir = Path(settings.output.base_dir) / "l2_metadata"
        self._l2_archive_dir = Path(settings.output.base_dir) / "l2_archive"

        logger.debug(
            f"InvalidationManager init: threshold={self._threshold}, "
            f"auto_regenerate={self._auto_regenerate}, "
            f"archive_old={self._archive_old}"
        )

    @property
    def store(self) -> Any:
        """Lazy-init del QdrantStore."""
        if self._store is None:
            from paperhunter.indexing.qdrant_store import QdrantStore
            self._store = QdrantStore()
        return self._store

    # ========================================================================
    # API principale
    # ========================================================================

    def check_and_regenerate(
        self,
        categories: list[str] | None = None,
        *,
        force_model: str | None = None,
        save_to_store: bool = True,
    ) -> InvalidationReport:
        """Controlla tutte le categorie e rigenera i L2 invalidati.

        Questo è il metodo principale da chiamare dopo l'ingestion
        di nuovi paper. Esegue il flusso completo:

        1. Scopre le categorie dal vector store (o usa quelle fornite)
        2. Per ogni categoria, confronta i paper_ids attuali con quelli
           nel L2 metadata
        3. Se il delta supera la soglia, invalida e (opzionalmente) rigenera

        Args:
            categories: Lista di categorie da controllare. Se None,
                controlla tutte le categorie nel vector store.
            force_model: Override tier modello per la rigenerazione.
            save_to_store: Se True, salva i nuovi L2 nel vector store.

        Returns:
            ``InvalidationReport`` con il dettaglio delle operazioni.
        """
        start_time = time.time()
        report = InvalidationReport()

        # --- Scoperta categorie ---
        if categories is None:
            categories = self._discover_categories()

        if not categories:
            logger.info("Nessuna categoria trovata per il check invalidazione.")
            report.total_time_seconds = time.time() - start_time
            return report

        report.checked_categories = list(categories)
        logger.info(
            f"Check invalidazione per {len(categories)} categorie: "
            f"{categories}"
        )

        # --- Check ogni categoria ---
        categories_to_regenerate: list[str] = []

        for category in categories:
            delta = self._compute_category_delta(category)

            new_ids = delta.get("new_paper_ids", [])
            removed_ids = delta.get("removed_paper_ids", [])
            has_l2 = delta.get("has_existing_l2", False)

            if new_ids:
                report.new_papers_by_category[category] = new_ids
            if removed_ids:
                report.removed_papers_by_category[category] = removed_ids

            # Decide se invalidare
            should_invalidate = len(new_ids) >= self._threshold or len(removed_ids) > 0

            if not has_l2:
                # Nessun L2 precedente → genera ex novo (sempre)
                if self._count_l1_for_category(category) > 0:
                    logger.info(
                        f"Categoria '{category}': nessun L2 esistente, "
                        f"da generare ex novo"
                    )
                    categories_to_regenerate.append(category)
                    report.skipped_categories.append(category)
                else:
                    logger.debug(
                        f"Categoria '{category}': nessun L2 e nessun L1, skip"
                    )
                    report.skipped_categories.append(category)
                continue

            if should_invalidate:
                reason_parts = []
                if new_ids:
                    reason_parts.append(f"{len(new_ids)} nuovi paper")
                if removed_ids:
                    reason_parts.append(f"{len(removed_ids)} paper rimossi")
                reason = ", ".join(reason_parts)

                logger.info(
                    f"Categoria '{category}': invalidazione triggerata — "
                    f"{reason} (soglia={self._threshold})"
                )
                self.invalidate_l2(category)
                report.invalidated_categories.append(category)
                categories_to_regenerate.append(category)
            else:
                logger.debug(
                    f"Categoria '{category}': L2 valido, "
                    f"nessun cambiamento significativo"
                )

        # --- Rigenerazione ---
        if categories_to_regenerate and self._auto_regenerate:
            logger.info(
                f"Rigenerazione L2 per {len(categories_to_regenerate)} "
                f"categorie: {categories_to_regenerate}"
            )

            for category in progress_bar(
                categories_to_regenerate,
                description="L2 Regeneration",
            ):
                try:
                    self.regenerate_l2(
                        category,
                        force_model=force_model,
                        save_to_store=save_to_store,
                    )
                    report.regenerated_categories.append(category)
                except Exception as e:
                    logger.error(
                        f"Errore rigenerazione L2 per '{category}': {e}"
                    )
                    report.failed_categories[category] = str(e)
        elif categories_to_regenerate:
            logger.info(
                f"{len(categories_to_regenerate)} categorie invalidate, "
                f"rigenerazione automatica disabilitata"
            )

        report.total_time_seconds = time.time() - start_time

        # --- Report finale ---
        self._print_report(report)

        return report

    def check_invalidation(
        self,
        newly_ingested: list[Any] | None = None,
        categories: list[str] | None = None,
    ) -> list[str]:
        """Ritorna le categorie che necessitano di rigenerazione L2.

        Metodo leggero: controlla solo senza rigenerare.

        Args:
            newly_ingested: Lista di Paper appena ingesti (opzionale).
                Se forniti, controlla solo le loro categorie.
            categories: Lista di categorie da controllare. Se None,
                controlla tutte.

        Returns:
            Lista di nomi categorie da rigenerare.
        """
        # Se ci sono paper appena ingesti, estrai le categorie coinvolte
        if newly_ingested is not None:
            cat_set: set[str] = set()
            for paper in newly_ingested:
                cat = getattr(paper, "category", None)
                if cat:
                    cat_set.add(cat)
            if categories is not None:
                cat_set.update(categories)
            categories = sorted(cat_set) if cat_set else None

        if categories is None:
            categories = self._discover_categories()

        to_regenerate: list[str] = []

        for category in categories:
            delta = self._compute_category_delta(category)
            new_ids = delta.get("new_paper_ids", [])
            removed_ids = delta.get("removed_paper_ids", [])
            has_l2 = delta.get("has_existing_l2", False)

            if not has_l2:
                # Nessun L2 → da generare se c'è almeno 1 L1
                if self._count_l1_for_category(category) > 0:
                    to_regenerate.append(category)
                continue

            if len(new_ids) >= self._threshold or len(removed_ids) > 0:
                to_regenerate.append(category)

        if to_regenerate:
            logger.info(
                f"Categorie da rigenerare: {to_regenerate}"
            )
        else:
            logger.info("Tutte le categorie sono aggiornate.")

        return to_regenerate

    def invalidate_l2(self, category: str) -> bool:
        """Marca il L2 di una categoria come invalidato.

        Aggiorna il metadata su disco impostando ``invalidated=True``
        e archiva opzionalmente il L2 precedente.

        Args:
            category: Nome della categoria da invalidare.

        Returns:
            True se l'invalidazione è andata a buon fine.
        """
        metadata = self._load_metadata(category)

        if metadata is None:
            logger.warning(
                f"Nessun metadata L2 trovato per '{category}', "
                f"niente da invalidare"
            )
            return False

        if metadata.get("invalidated", False):
            logger.debug(f"L2 per '{category}' già invalidato")
            return True

        # Archivia il vecchio L2 prima di invalidare
        if self._archive_old:
            self._archive_l2(category, metadata)

        # Marca come invalidato
        metadata["invalidated"] = True
        metadata["invalidated_at"] = datetime.now().isoformat()
        self._save_metadata(category, metadata)

        logger.info(
            f"L2 per '{category}' v{metadata.get('version', '?')} "
            f"marcato come invalidato"
        )
        return True

    def regenerate_l2(
        self,
        category: str,
        *,
        force_model: str | None = None,
        save_to_store: bool = True,
    ) -> L2Summary:
        """Raccoglie tutti gli L1 della categoria e rigenera il L2.

        Carica i L1 summary dal vector store, genera un nuovo L2
        con versione incrementale, e salva nel vector store e su disco.

        Args:
            category: Nome della categoria da rigenerare.
            force_model: Override tier modello (``'flash'`` o ``'pro'``).
            save_to_store: Se True, salva il nuovo L2 nel vector store.

        Returns:
            Nuovo ``L2Summary`` generato.

        Raises:
            ValueError: Se non ci sono L1 per la categoria.
            RuntimeError: Se la chiamata LLM fallisce.
        """
        from paperhunter.summarization.level2_summarizer import (
            generate_l2_summary,
        )

        logger.info(f"Rigenerazione L2 per '{category}'...")

        # Genera il nuovo L2 (la funzione gestisce versioning,
        # caricamento L1, e salvataggio nel store)
        l2 = generate_l2_summary(
            category,
            force_model=force_model,
            save_to_store=save_to_store,
            store=self.store,
        )

        print_success(
            f"L2 rigenerato per '{category}' → v{l2.version} "
            f"({l2.paper_count} paper, model={l2.model_used})"
        )

        return l2

    def get_category_status(
        self,
        categories: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Ritorna lo stato di ogni categoria (L1 count, L2 status, delta).

        Utile per diagnostica e per il CLI.

        Args:
            categories: Lista di categorie. Se None, tutte.

        Returns:
            Dict categoria → {l1_count, has_l2, l2_version, l2_invalidated,
            l2_paper_count, current_l1_count, new_papers, removed_papers}.
        """
        if categories is None:
            categories = self._discover_categories()

        status: dict[str, dict[str, Any]] = {}

        for category in categories:
            delta = self._compute_category_delta(category)
            metadata = self._load_metadata(category)

            cat_status: dict[str, Any] = {
                "l1_count_in_store": delta.get("current_l1_count", 0),
                "has_l2": delta.get("has_existing_l2", False),
                "l2_version": metadata.get("version") if metadata else None,
                "l2_invalidated": metadata.get("invalidated", False) if metadata else None,
                "l2_paper_count": metadata.get("paper_count", 0) if metadata else 0,
                "new_papers": len(delta.get("new_paper_ids", [])),
                "removed_papers": len(delta.get("removed_paper_ids", [])),
                "needs_regeneration": (
                    len(delta.get("new_paper_ids", [])) >= self._threshold
                    or len(delta.get("removed_paper_ids", [])) > 0
                    or not delta.get("has_existing_l2", False)
                ),
            }
            status[category] = cat_status

        return status

    # ========================================================================
    # Metodi interni: delta computation
    # ========================================================================

    def _compute_category_delta(self, category: str) -> dict[str, Any]:
        """Calcola il delta tra i paper_ids nel vector store e nel L2 metadata.

        Confronta:
        - ``current_l1_ids``: paper_ids attualmente nella collection
          ``l1_summaries`` per questa categoria
        - ``l2_paper_ids``: paper_ids registrati nel metadata L2 su disco

        Args:
            category: Nome della categoria.

        Returns:
            Dict con:
            - ``current_l1_count``: numero L1 nel store
            - ``current_l1_ids``: set di paper_ids dal store
            - ``l2_paper_ids``: set di paper_ids dal L2 metadata
            - ``new_paper_ids``: paper nel store ma non nel L2
            - ``removed_paper_ids``: paper nel L2 ma non nel store
            - ``has_existing_l2``: se esiste un L2 (anche invalidato)
        """
        # Carica paper_ids attuali dalla collection l1_summaries
        current_l1_ids = self._get_l1_paper_ids_for_category(category)

        # Carica paper_ids dal metadata L2 su disco
        metadata = self._load_metadata(category)
        has_l2 = metadata is not None
        l2_paper_ids = set(metadata.get("paper_ids", [])) if metadata else set()

        # Calcola delta
        new_ids = sorted(current_l1_ids - l2_paper_ids)
        removed_ids = sorted(l2_paper_ids - current_l1_ids)

        return {
            "current_l1_count": len(current_l1_ids),
            "current_l1_ids": current_l1_ids,
            "l2_paper_ids": l2_paper_ids,
            "new_paper_ids": new_ids,
            "removed_paper_ids": removed_ids,
            "has_existing_l2": has_l2,
        }

    def _get_l1_paper_ids_for_category(self, category: str) -> set[str]:
        """Recupera tutti i paper_ids dalla collection l1_summaries per una categoria.

        Args:
            category: Nome della categoria.

        Returns:
            Set di paper_ids.
        """
        try:
            collection = settings.qdrant.collection_l1_summaries
            from qdrant_client import models

            paper_ids: set[str] = set()
            offset = None

            while True:
                result, next_offset = self.store.client.scroll(
                    collection_name=collection,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="category",
                                match=models.MatchValue(value=category),
                            )
                        ]
                    ),
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in result:
                    payload = point.payload or {}
                    pid = payload.get("paper_id", "")
                    if pid and not pid.startswith("l2_"):
                        paper_ids.add(pid)

                if next_offset is None:
                    break
                offset = next_offset

            return paper_ids

        except Exception as e:
            logger.error(
                f"Errore lettura L1 paper_ids per '{category}': {e}"
            )
            return set()

    def _count_l1_for_category(self, category: str) -> int:
        """Conta gli L1 summary nel vector store per una categoria.

        Args:
            category: Nome della categoria.

        Returns:
            Numero di L1 summary.
        """
        return len(self._get_l1_paper_ids_for_category(category))

    def _discover_categories(self) -> list[str]:
        """Scopre le categorie uniche dalla collection l1_summaries.

        Returns:
            Lista ordinata di nomi di categorie.
        """
        try:
            collection = settings.qdrant.collection_l1_summaries
            stats = self.store.get_stats(collection_name=collection)
            categories = list(stats.get("categories", {}).keys())
            return sorted(categories)
        except Exception as e:
            logger.warning(f"Errore discovery categorie: {e}")
            return []

    # ========================================================================
    # Metodi interni: metadata I/O
    # ========================================================================

    def _load_metadata(self, category: str) -> dict | None:
        """Carica il metadata L2 da disco per una categoria.

        Args:
            category: Nome della categoria.

        Returns:
            Dict con i metadata, o None se non trovato.
        """
        try:
            safe_name = re.sub(r"[^a-z0-9_]", "_", category.lower())
            filepath = self._l2_metadata_dir / f"{safe_name}.json"

            if not filepath.exists():
                return None

            return json.loads(filepath.read_text())

        except Exception as e:
            logger.warning(
                f"Errore caricamento metadata L2 per '{category}': {e}"
            )
            return None

    def _save_metadata(self, category: str, metadata: dict) -> None:
        """Salva il metadata L2 su disco.

        Args:
            category: Nome della categoria.
            metadata: Dict da salvare.
        """
        try:
            self._l2_metadata_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^a-z0-9_]", "_", category.lower())
            filepath = self._l2_metadata_dir / f"{safe_name}.json"
            filepath.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False)
            )
            logger.debug(f"L2 metadata salvati: {filepath}")
        except Exception as e:
            logger.warning(f"Errore salvataggio metadata L2: {e}")

    # ========================================================================
    # Metodi interni: archivio storico
    # ========================================================================

    def _archive_l2(self, category: str, metadata: dict) -> bool:
        """Archivia il metadata L2 corrente prima della rigenerazione.

        Copia il file metadata in ``output/l2_archive/<category>/``
        con il nome ``v<N>_<timestamp>.json``.

        Args:
            category: Nome della categoria.
            metadata: Metadata L2 corrente.

        Returns:
            True se l'archiviazione è andata a buon fine.
        """
        try:
            safe_name = re.sub(r"[^a-z0-9_]", "_", category.lower())
            archive_dir = self._l2_archive_dir / safe_name
            archive_dir.mkdir(parents=True, exist_ok=True)

            version = metadata.get("version", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"v{version}_{timestamp}.json"

            archive_path = archive_dir / archive_name

            # Aggiungi info di archiviazione al metadata
            archive_data = {
                **metadata,
                "archived_at": datetime.now().isoformat(),
                "archive_reason": "invalidated_by_new_papers",
            }

            archive_path.write_text(
                json.dumps(archive_data, indent=2, ensure_ascii=False)
            )

            logger.debug(
                f"L2 archiviato per '{category}' v{version} → {archive_path}"
            )
            return True

        except Exception as e:
            logger.warning(
                f"Errore archiviazione L2 per '{category}': {e}"
            )
            return False

    def get_archive_history(
        self,
        category: str,
    ) -> list[dict[str, Any]]:
        """Recupera lo storico delle versioni L2 archiviate per una categoria.

        Args:
            category: Nome della categoria.

        Returns:
            Lista di metadata delle versioni archiviate (ordine cronologico).
        """
        safe_name = re.sub(r"[^a-z0-9_]", "_", category.lower())
        archive_dir = self._l2_archive_dir / safe_name

        history: list[dict[str, Any]] = []

        if not archive_dir.exists():
            return history

        for filepath in sorted(archive_dir.glob("*.json")):
            try:
                metadata = json.loads(filepath.read_text())
                metadata["archive_file"] = filepath.name
                history.append(metadata)
            except Exception as e:
                logger.warning(f"Errore lettura archivio {filepath}: {e}")

        return history

    # ========================================================================
    # Report
    # ========================================================================

    def _print_report(self, report: InvalidationReport) -> None:
        """Stampa un report riassuntivo dell'operazione di invalidazione.

        Args:
            report: InvalidationReport da stampare.
        """
        summary_data = {
            "Categorie controllate": len(report.checked_categories),
            "Invalidate": len(report.invalidated_categories),
            "Rigenerate con successo": len(report.regenerated_categories),
            "Rigenerazione fallita": len(report.failed_categories),
            "Nuove (senza L2)": len(report.skipped_categories),
            "Tempo totale": f"{report.total_time_seconds:.1f}s",
        }
        print_table("Invalidation Report", summary_data)

        # Dettaglio nuovi paper per categoria
        if report.new_papers_by_category:
            detail_rows = []
            for cat, ids in sorted(report.new_papers_by_category.items()):
                detail_rows.append({
                    "Categoria": cat,
                    "Nuovi paper": len(ids),
                    "IDs": ", ".join(ids[:5]) + ("..." if len(ids) > 5 else ""),
                })
            print_table("Nuovi paper per categoria", detail_rows)

        # Dettaglio paper rimossi
        if report.removed_papers_by_category:
            detail_rows = []
            for cat, ids in sorted(report.removed_papers_by_category.items()):
                detail_rows.append({
                    "Categoria": cat,
                    "Paper rimossi": len(ids),
                    "IDs": ", ".join(ids[:5]) + ("..." if len(ids) > 5 else ""),
                })
            print_table("Paper rimossi per categoria", detail_rows)

        # Errori
        if report.failed_categories:
            print_warning("Errori durante la rigenerazione:")
            for cat, err in report.failed_categories.items():
                logger.error(f"  {cat}: {err}")

        # Successo finale
        if report.regenerated_categories:
            print_success(
                f"Invalidazione completata: "
                f"{len(report.regenerated_categories)} L2 rigenerati"
            )
        elif not report.invalidated_categories:
            print_success("Tutti i L2 summary sono aggiornati — nessuna azione necessaria")
