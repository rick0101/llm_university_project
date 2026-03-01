"""
postprocessing/checkpoint_manager.py — Gestione centralizzata dei checkpoint

Fornisce un'API unificata per salvare, caricare, gestire e analizzare
checkpoint intermedi della pipeline di generazione survey.

Supporta due tipi di checkpoint:
  - **Pipeline (agentic):** snapshot dello stato SurveyState dopo ogni nodo
    LangGraph (plan, write, verify, assemble).
  - **Ingestion:** lista dei PDF processati per riprendere l'ingestion
    incrementale da dove è stata interrotta.

Funzionalità principali:
  - Salvataggio/caricamento checkpoint con metadata estesi
  - Elenco, ricerca e filtraggio checkpoint
  - Auto-cleanup con retention policy configurabile
  - Diff tra checkpoint per analizzare il progresso
  - Raccomandazioni di resume (quale checkpoint usare per riprendere)
  - Esportazione/importazione per backup

Usage:
    from paperhunter.postprocessing.checkpoint_manager import (
        CheckpointManager, CheckpointInfo, get_resume_recommendation,
    )

    mgr = CheckpointManager()

    # Salva un checkpoint pipeline
    info = mgr.save_pipeline_checkpoint(state, node_name="write")

    # Elenca tutti i checkpoint
    for cp in mgr.list_checkpoints():
        print(cp.node, cp.timestamp, cp.size_kb)

    # Raccomandazione di resume
    rec = mgr.get_resume_recommendation()
    print(rec)

    # Carica un checkpoint
    node, state = mgr.load_pipeline_checkpoint(rec["path"])

    # Cleanup
    mgr.clean(keep_latest_per_node=True, max_age_hours=72)
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_success,
    print_table,
    print_warning,
)

logger = get_logger(__name__)


# ============================================================================
# Checkpoint types and metadata
# ============================================================================

# Tipi di checkpoint supportati
CHECKPOINT_TYPE_PIPELINE = "pipeline"
CHECKPOINT_TYPE_INGESTION = "ingestion"

# Nomi dei nodi della pipeline agentica
PIPELINE_NODES = ("plan", "write", "verify", "assemble")

# Versione dello schema checkpoint (per future migrazioni)
CHECKPOINT_SCHEMA_VERSION = "1.0"


@dataclass
class CheckpointInfo:
    """Metadata di un singolo checkpoint.

    Attributes:
        path: Percorso assoluto al file checkpoint.
        filename: Nome del file.
        checkpoint_type: Tipo: 'pipeline' o 'ingestion'.
        node: Nome del nodo (per pipeline) o step (per ingestion).
        timestamp: Data/ora del salvataggio.
        timestamp_str: Stringa timestamp leggibile.
        size_kb: Dimensione in KB.
        is_latest: Se è il file _latest per il suo nodo.
        schema_version: Versione dello schema.
        metadata: Metadati extra dal checkpoint.
    """
    path: str
    filename: str
    checkpoint_type: str = CHECKPOINT_TYPE_PIPELINE
    node: str = "unknown"
    timestamp: Optional[datetime] = None
    timestamp_str: str = ""
    size_kb: float = 0.0
    is_latest: bool = False
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializza in dict JSON-safe."""
        return {
            "path": self.path,
            "filename": self.filename,
            "checkpoint_type": self.checkpoint_type,
            "node": self.node,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timestamp_str": self.timestamp_str,
            "size_kb": self.size_kb,
            "is_latest": self.is_latest,
            "schema_version": self.schema_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_file(cls, path: Path) -> CheckpointInfo:
        """Crea un CheckpointInfo leggendo il file checkpoint.

        Args:
            path: Percorso al file checkpoint JSON.

        Returns:
            CheckpointInfo con tutti i campi valorizzati.
        """
        info = cls(
            path=str(path),
            filename=path.name,
            size_kb=round(path.stat().st_size / 1024, 1),
            is_latest="_latest" in path.name,
        )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            info.node = "corrupted"
            return info

        # Determina il tipo di checkpoint
        if "state" in data and "node" in data:
            info.checkpoint_type = CHECKPOINT_TYPE_PIPELINE
            info.node = data.get("node", "unknown")
        elif "completed_pdfs" in data:
            info.checkpoint_type = CHECKPOINT_TYPE_INGESTION
            info.node = "ingestion"
        else:
            info.checkpoint_type = "unknown"

        # Timestamp
        ts_str = data.get("timestamp") or data.get("last_updated", "")
        info.timestamp_str = ts_str
        if ts_str:
            try:
                # Formato pipeline: "20260220_101500" o "20260220_101500_123456"
                if "_" in ts_str and len(ts_str) == 22:
                    info.timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M%S_%f")
                elif "_" in ts_str and len(ts_str) == 15:
                    info.timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                else:
                    info.timestamp = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                pass

        # Schema version
        info.schema_version = data.get("schema_version", "1.0")

        # Metadata extra
        if info.checkpoint_type == CHECKPOINT_TYPE_PIPELINE:
            state_data = data.get("state", {})
            info.metadata = {
                "current_phase": state_data.get("current_phase", ""),
                "iteration_count": state_data.get("iteration_count", 0),
                "outline_sections": len(state_data.get("outline", [])),
                "drafted_sections": len(state_data.get("sections", {})),
                "errors_count": len(state_data.get("errors", [])),
                "has_final_paper": bool(state_data.get("final_paper")),
            }
        elif info.checkpoint_type == CHECKPOINT_TYPE_INGESTION:
            info.metadata = {
                "completed_pdfs": len(data.get("completed_pdfs", [])),
                "total_to_process": data.get("total_pdfs_to_process", 0),
                "started_at": data.get("started_at", ""),
            }

        return info

    def __repr__(self) -> str:
        ts = self.timestamp_str or "?"
        return (
            f"CheckpointInfo(type={self.checkpoint_type}, node={self.node}, "
            f"ts={ts}, size={self.size_kb}kb)"
        )


# ============================================================================
# CheckpointManager
# ============================================================================

class CheckpointManager:
    """Gestore centralizzato dei checkpoint per pipeline e ingestion.

    Fornisce un'API unificata per salvare, caricare, elencare e gestire
    i checkpoint di tutte le fasi del sistema.

    Args:
        checkpoint_dir: Directory base per i checkpoint.
                        Default da ``config.yaml`` → ``output.checkpoints_dir``.

    Example::

        mgr = CheckpointManager()

        # Salva checkpoint pipeline
        info = mgr.save_pipeline_checkpoint(state, "write")

        # Elenca tutti
        cps = mgr.list_checkpoints()

        # Raccomanda resume
        rec = mgr.get_resume_recommendation()

        # Cleanup vecchi
        mgr.clean(max_age_hours=72)
    """

    def __init__(self, checkpoint_dir: str | Path | None = None):
        self.checkpoint_dir = Path(
            checkpoint_dir or settings.output.checkpoints_dir
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Pipeline checkpoints (agentic survey writing)
    # ------------------------------------------------------------------ #

    def save_pipeline_checkpoint(
        self,
        state: dict[str, Any],
        node_name: str,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> CheckpointInfo:
        """Salva un checkpoint della pipeline agentica.

        Salva sia un file timestamped sia un file ``_latest`` per il nodo.

        Args:
            state: Lo stato SurveyState (dict serializzabile).
            node_name: Nome del nodo completato (plan, write, verify, assemble).
            extra_metadata: Metadati aggiuntivi opzionali.

        Returns:
            CheckpointInfo del file checkpoint salvato.
        """
        # Importa state_to_dict solo quando serve (evita import circolare)
        from paperhunter.agents.state import state_to_dict

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"checkpoint_{node_name}_{timestamp}.json"
        filepath = self.checkpoint_dir / filename
        latest_filepath = self.checkpoint_dir / f"checkpoint_{node_name}_latest.json"

        # Serializza lo stato
        try:
            state_dict = state_to_dict(state)
        except Exception:
            # Fallback: tratta come dict diretto
            state_dict = dict(state) if state else {}

        data: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": CHECKPOINT_TYPE_PIPELINE,
            "node": node_name,
            "timestamp": timestamp,
            "state": state_dict,
        }
        if extra_metadata:
            data["extra_metadata"] = extra_metadata

        content = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        for fp in (filepath, latest_filepath):
            fp.write_text(content, encoding="utf-8")

        logger.info("Checkpoint pipeline salvato: %s", filename)
        return CheckpointInfo.from_file(filepath)

    def load_pipeline_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> tuple[str, dict[str, Any]]:
        """Carica un checkpoint pipeline e ritorna nodo + stato.

        Args:
            checkpoint_path: Path al file checkpoint JSON.

        Returns:
            Tupla (node_name, state_dict).

        Raises:
            FileNotFoundError: Se il file non esiste.
            ValueError: Se il file non è un checkpoint pipeline valido.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        if "state" not in data:
            raise ValueError(
                f"Il file {path.name} non è un checkpoint pipeline valido "
                f"(manca il campo 'state')."
            )

        node_name = data.get("node", "unknown")
        state_dict = data["state"]

        logger.info(
            "Checkpoint pipeline caricato: nodo='%s', timestamp=%s",
            node_name, data.get("timestamp", "?"),
        )
        return node_name, state_dict

    def load_pipeline_state(
        self,
        checkpoint_path: str | Path,
    ) -> tuple[str, Any]:
        """Carica un checkpoint pipeline e ricostruisce il SurveyState.

        Come ``load_pipeline_checkpoint`` ma restituisce un oggetto
        ``SurveyState`` (TypedDict) già deserializzato.

        Args:
            checkpoint_path: Path al file checkpoint JSON.

        Returns:
            Tupla (node_name, SurveyState).
        """
        from paperhunter.agents.state import state_from_dict

        node_name, state_dict = self.load_pipeline_checkpoint(checkpoint_path)
        state = state_from_dict(state_dict)
        return node_name, state

    # ------------------------------------------------------------------ #
    # Ingestion checkpoints
    # ------------------------------------------------------------------ #

    def save_ingestion_checkpoint(
        self,
        completed_pdfs: list[str],
        total_to_process: int,
        *,
        started_at: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> CheckpointInfo:
        """Salva un checkpoint di ingestion.

        Args:
            completed_pdfs: Lista dei percorsi relativi dei PDF completati.
            total_to_process: Totale PDF da processare.
            started_at: Timestamp di inizio (ISO format).
            extra_metadata: Metadati aggiuntivi opzionali.

        Returns:
            CheckpointInfo del checkpoint salvato.
        """
        filepath = self.checkpoint_dir / "ingestion_checkpoint.json"

        data: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": CHECKPOINT_TYPE_INGESTION,
            "started_at": started_at or datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_pdfs": completed_pdfs,
            "total_pdfs_to_process": total_to_process,
        }
        if extra_metadata:
            data["extra_metadata"] = extra_metadata

        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.debug(
            "Checkpoint ingestion salvato: %d/%d PDF completati",
            len(completed_pdfs), total_to_process,
        )
        return CheckpointInfo.from_file(filepath)

    def load_ingestion_checkpoint(self) -> dict[str, Any]:
        """Carica il checkpoint di ingestion corrente.

        Returns:
            Dict con i dati del checkpoint, o dict vuoto se non esiste.
        """
        filepath = self.checkpoint_dir / "ingestion_checkpoint.json"
        if not filepath.exists():
            return {}

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            logger.info(
                "Checkpoint ingestion caricato: %d/%d PDF completati",
                len(data.get("completed_pdfs", [])),
                data.get("total_pdfs_to_process", 0),
            )
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Checkpoint ingestion corrotto: %s", e)
            return {}

    def has_ingestion_checkpoint(self) -> bool:
        """Verifica se esiste un checkpoint di ingestion pendente."""
        filepath = self.checkpoint_dir / "ingestion_checkpoint.json"
        return filepath.exists()

    def clear_ingestion_checkpoint(self) -> bool:
        """Rimuove il checkpoint di ingestion (pipeline completata).

        Returns:
            True se il file esisteva ed è stato rimosso.
        """
        filepath = self.checkpoint_dir / "ingestion_checkpoint.json"
        if filepath.exists():
            filepath.unlink()
            logger.debug("Checkpoint ingestion rimosso.")
            return True
        return False

    # ------------------------------------------------------------------ #
    # Listing & search
    # ------------------------------------------------------------------ #

    def list_checkpoints(
        self,
        *,
        checkpoint_type: str | None = None,
        node: str | None = None,
        include_latest: bool = True,
        sort_by: str = "timestamp",
    ) -> list[CheckpointInfo]:
        """Elenca tutti i checkpoint disponibili.

        Args:
            checkpoint_type: Filtra per tipo ('pipeline', 'ingestion').
            node: Filtra per nodo (es. 'write', 'verify').
            include_latest: Se False, esclude i file *_latest.json.
            sort_by: Campo per ordinamento ('timestamp', 'size', 'node').

        Returns:
            Lista ordinata di CheckpointInfo.
        """
        if not self.checkpoint_dir.exists():
            return []

        results: list[CheckpointInfo] = []

        for f in self.checkpoint_dir.glob("*.json"):
            info = CheckpointInfo.from_file(f)

            # Filtri
            if not include_latest and info.is_latest:
                continue
            if checkpoint_type and info.checkpoint_type != checkpoint_type:
                continue
            if node and info.node != node:
                continue

            results.append(info)

        # Ordinamento
        if sort_by == "timestamp":
            results.sort(
                key=lambda i: i.timestamp or datetime.min,
                reverse=True,
            )
        elif sort_by == "size":
            results.sort(key=lambda i: i.size_kb, reverse=True)
        elif sort_by == "node":
            results.sort(key=lambda i: i.node)

        return results

    def get_latest_pipeline_checkpoint(
        self,
        node_name: str | None = None,
    ) -> CheckpointInfo | None:
        """Trova il checkpoint pipeline più recente.

        Args:
            node_name: Se specificato, cerca il latest di quel nodo.
                       Se None, cerca il più recente in assoluto.

        Returns:
            CheckpointInfo o None se non trovato.
        """
        if node_name:
            latest = self.checkpoint_dir / f"checkpoint_{node_name}_latest.json"
            if latest.exists():
                return CheckpointInfo.from_file(latest)
            return None

        # Trova il più recente tra tutti i pipeline checkpoint (no _latest)
        candidates = self.list_checkpoints(
            checkpoint_type=CHECKPOINT_TYPE_PIPELINE,
            include_latest=False,
            sort_by="timestamp",
        )
        return candidates[0] if candidates else None

    def find_checkpoints(
        self,
        *,
        after: datetime | None = None,
        before: datetime | None = None,
        min_size_kb: float = 0,
        max_size_kb: float = float("inf"),
    ) -> list[CheckpointInfo]:
        """Cerca checkpoint con criteri avanzati.

        Args:
            after: Solo checkpoint dopo questa data.
            before: Solo checkpoint prima di questa data.
            min_size_kb: Dimensione minima in KB.
            max_size_kb: Dimensione massima in KB.

        Returns:
            Lista filtrata di CheckpointInfo.
        """
        results: list[CheckpointInfo] = []

        for cp in self.list_checkpoints():
            if cp.size_kb < min_size_kb or cp.size_kb > max_size_kb:
                continue
            if after and cp.timestamp and cp.timestamp < after:
                continue
            if before and cp.timestamp and cp.timestamp > before:
                continue
            results.append(cp)

        return results

    # ------------------------------------------------------------------ #
    # Resume recommendation
    # ------------------------------------------------------------------ #

    def get_resume_recommendation(self) -> dict[str, Any]:
        """Analizza i checkpoint e raccomanda come riprendere la pipeline.

        Determina il miglior punto di ripresa basandosi su:
        1. Quale nodo è stato completato per ultimo
        2. Se ci sono sezioni che necessitano revisione
        3. Se l'ingestion ha un checkpoint pendente

        Returns:
            Dict con chiavi:
            - 'has_recommendation': bool
            - 'recommendation': stringa descrittiva
            - 'checkpoint_path': path al checkpoint consigliato (o None)
            - 'resume_node': nodo da cui riprendere (o None)
            - 'checkpoint_type': tipo di checkpoint
            - 'details': dict con informazioni aggiuntive
        """
        result: dict[str, Any] = {
            "has_recommendation": False,
            "recommendation": "Nessun checkpoint trovato. Esegui la pipeline da zero.",
            "checkpoint_path": None,
            "resume_node": None,
            "checkpoint_type": None,
            "details": {},
        }

        # 1. Controlla checkpoint ingestion
        ing_cp = self.has_ingestion_checkpoint()
        if ing_cp:
            ing_data = self.load_ingestion_checkpoint()
            completed = len(ing_data.get("completed_pdfs", []))
            total = ing_data.get("total_pdfs_to_process", 0)
            remaining = total - completed

            if remaining > 0:
                result.update({
                    "has_recommendation": True,
                    "recommendation": (
                        f"Ingestion interrotta: {completed}/{total} PDF completati. "
                        f"Riprendi con IngestionPipeline(use_checkpoint=True).run()"
                    ),
                    "checkpoint_path": str(
                        self.checkpoint_dir / "ingestion_checkpoint.json"
                    ),
                    "resume_node": "ingestion",
                    "checkpoint_type": CHECKPOINT_TYPE_INGESTION,
                    "details": {
                        "completed_pdfs": completed,
                        "total_pdfs": total,
                        "remaining_pdfs": remaining,
                        "progress_pct": round(completed / total * 100, 1) if total else 0,
                    },
                })
                return result

        # 2. Controlla checkpoint pipeline
        node_order = list(PIPELINE_NODES)
        # Cerca il checkpoint più recente per ogni nodo
        latest_per_node: dict[str, CheckpointInfo] = {}
        for node in node_order:
            cp = self.get_latest_pipeline_checkpoint(node)
            if cp:
                latest_per_node[node] = cp

        if not latest_per_node:
            return result

        # Trova il nodo più avanzato
        best_node: str | None = None
        best_cp: CheckpointInfo | None = None
        for node in reversed(node_order):
            if node in latest_per_node:
                best_node = node
                best_cp = latest_per_node[node]
                break

        if not best_node or not best_cp:
            return result

        # Determina il nodo successivo
        next_node_map = {
            "plan": "write",
            "write": "verify",
            "verify": "assemble",  # o write se needs revision (controlliamo sotto)
            "assemble": None,  # pipeline completata
        }
        next_node = next_node_map.get(best_node)

        # Se l'ultimo nodo è verify, controlla se serve revisione
        if best_node == "verify":
            try:
                _, state_dict = self.load_pipeline_checkpoint(best_cp.path)
                sections = state_dict.get("sections", {})
                needs_rev = [
                    k for k, v in sections.items()
                    if isinstance(v, dict)
                    and v.get("status") == "needs_revision"
                ]
                iter_count = state_dict.get("iteration_count", 0)
                max_iter = settings.agents.max_writer_iterations

                if needs_rev and iter_count < max_iter:
                    next_node = "write"
                    result["details"]["needs_revision"] = needs_rev
                    result["details"]["iteration_count"] = iter_count
                    result["details"]["max_iterations"] = max_iter
            except Exception as e:
                logger.debug("Errore analisi checkpoint verify: %s", e)

        if next_node is None:
            result.update({
                "has_recommendation": False,
                "recommendation": (
                    "Pipeline completata (ultimo nodo: assemble). "
                    "Nessuna ripresa necessaria."
                ),
                "checkpoint_path": best_cp.path,
                "resume_node": None,
                "checkpoint_type": CHECKPOINT_TYPE_PIPELINE,
                "details": {"last_completed_node": best_node},
            })
        else:
            # Merge con eventuali details già presenti (es. needs_revision)
            existing_details = result.get("details", {})
            merged_details = {
                "last_completed_node": best_node,
                "resume_from": next_node,
                **best_cp.metadata,
                **existing_details,
            }
            result.update({
                "has_recommendation": True,
                "recommendation": (
                    f"Pipeline interrotta dopo il nodo '{best_node}'. "
                    f"Riprendi dal nodo '{next_node}' con "
                    f"run_pipeline_from_checkpoint('{best_cp.path}')"
                ),
                "checkpoint_path": best_cp.path,
                "resume_node": next_node,
                "checkpoint_type": CHECKPOINT_TYPE_PIPELINE,
                "details": merged_details,
            })

        return result

    # ------------------------------------------------------------------ #
    # Diff between checkpoints
    # ------------------------------------------------------------------ #

    def diff_checkpoints(
        self,
        path_a: str | Path,
        path_b: str | Path,
    ) -> dict[str, Any]:
        """Confronta due checkpoint pipeline e genera un diff.

        Args:
            path_a: Path al primo checkpoint (più vecchio).
            path_b: Path al secondo checkpoint (più recente).

        Returns:
            Dict con le differenze rilevate tra i due stati.

        Raises:
            FileNotFoundError: Se uno dei file non esiste.
            ValueError: Se i file non sono checkpoint pipeline validi.
        """
        node_a, state_a = self.load_pipeline_checkpoint(path_a)
        node_b, state_b = self.load_pipeline_checkpoint(path_b)

        diff: dict[str, Any] = {
            "checkpoint_a": {
                "path": str(path_a),
                "node": node_a,
            },
            "checkpoint_b": {
                "path": str(path_b),
                "node": node_b,
            },
            "node_progression": f"{node_a} → {node_b}",
            "changes": {},
        }

        # Confronta campi chiave
        changes = diff["changes"]

        # Phase
        phase_a = state_a.get("current_phase", "")
        phase_b = state_b.get("current_phase", "")
        if phase_a != phase_b:
            changes["current_phase"] = {"from": phase_a, "to": phase_b}

        # Iteration count
        iter_a = state_a.get("iteration_count", 0)
        iter_b = state_b.get("iteration_count", 0)
        if iter_a != iter_b:
            changes["iteration_count"] = {"from": iter_a, "to": iter_b}

        # Outline
        outline_a = len(state_a.get("outline", []))
        outline_b = len(state_b.get("outline", []))
        if outline_a != outline_b:
            changes["outline_sections"] = {"from": outline_a, "to": outline_b}

        # Sections
        sections_a = set(state_a.get("sections", {}).keys())
        sections_b = set(state_b.get("sections", {}).keys())
        new_sections = sections_b - sections_a
        removed_sections = sections_a - sections_b
        if new_sections or removed_sections:
            changes["sections"] = {
                "added": sorted(new_sections),
                "removed": sorted(removed_sections),
                "total_a": len(sections_a),
                "total_b": len(sections_b),
            }

        # Section status changes
        common_sections = sections_a & sections_b
        status_changes: dict[str, dict[str, str]] = {}
        for key in common_sections:
            sec_a = state_a["sections"][key]
            sec_b = state_b["sections"][key]
            st_a = sec_a.get("status", "") if isinstance(sec_a, dict) else ""
            st_b = sec_b.get("status", "") if isinstance(sec_b, dict) else ""
            if st_a != st_b:
                status_changes[key] = {"from": st_a, "to": st_b}
        if status_changes:
            changes["section_status_changes"] = status_changes

        # Verifications
        verif_a = set(state_a.get("verification_results", {}).keys())
        verif_b = set(state_b.get("verification_results", {}).keys())
        new_verif = verif_b - verif_a
        if new_verif:
            changes["new_verifications"] = sorted(new_verif)

        # Provenance
        prov_a = len(state_a.get("provenance", []))
        prov_b = len(state_b.get("provenance", []))
        if prov_a != prov_b:
            changes["provenance_records"] = {"from": prov_a, "to": prov_b}

        # Errors
        errors_a = set(state_a.get("errors", []))
        errors_b = set(state_b.get("errors", []))
        new_errors = errors_b - errors_a
        if new_errors:
            changes["new_errors"] = sorted(new_errors)

        # Final paper
        has_paper_a = bool(state_a.get("final_paper"))
        has_paper_b = bool(state_b.get("final_paper"))
        if has_paper_a != has_paper_b:
            changes["final_paper"] = {"from": has_paper_a, "to": has_paper_b}

        # Bibliografia
        bib_a = len(state_a.get("bibliography", []))
        bib_b = len(state_b.get("bibliography", []))
        if bib_a != bib_b:
            changes["bibliography_keys"] = {"from": bib_a, "to": bib_b}

        diff["total_changes"] = len(changes)
        return diff

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def clean(
        self,
        *,
        keep_latest_per_node: bool = True,
        max_age_hours: float | None = None,
        max_checkpoints: int | None = None,
        dry_run: bool = False,
    ) -> int:
        """Rimuove checkpoint vecchi secondo la retention policy.

        Args:
            keep_latest_per_node: Se True, mantiene i file *_latest.json.
            max_age_hours: Rimuove checkpoint più vecchi di N ore.
            max_checkpoints: Mantiene solo gli ultimi N checkpoint (per tipo).
            dry_run: Se True, simula la pulizia senza cancellare niente.

        Returns:
            Numero di file rimossi (o che sarebbero stati rimossi in dry_run).
        """
        if not self.checkpoint_dir.exists():
            return 0

        all_files = sorted(
            self.checkpoint_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        to_remove: list[Path] = []
        now = datetime.now()

        # Prima passa: marca per rimozione in base ai criteri
        for f in all_files:
            # Non rimuovere mai i latest se richiesto
            if keep_latest_per_node and "_latest" in f.name:
                continue

            should_remove = False

            # Criterio età
            if max_age_hours is not None:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                age = now - mtime
                if age > timedelta(hours=max_age_hours):
                    should_remove = True

            if should_remove:
                to_remove.append(f)

        # Criterio max_checkpoints: per tipo, tieni solo gli ultimi N
        if max_checkpoints is not None:
            pipeline_cps = [
                f for f in all_files
                if f.name.startswith("checkpoint_")
                and "_latest" not in f.name
                and f not in to_remove
            ]
            if len(pipeline_cps) > max_checkpoints:
                excess = pipeline_cps[max_checkpoints:]
                for f in excess:
                    if f not in to_remove:
                        to_remove.append(f)

        # Rimuovi
        removed = 0
        for f in to_remove:
            if dry_run:
                logger.info("(dry-run) Rimuoverei: %s", f.name)
            else:
                try:
                    f.unlink()
                    logger.debug("Rimosso checkpoint: %s", f.name)
                except OSError as e:
                    logger.warning("Errore rimozione %s: %s", f.name, e)
                    continue
            removed += 1

        if removed:
            action = "Rimoverei" if dry_run else "Rimossi"
            logger.info("%s %d checkpoint.", action, removed)

        return removed

    def clear_all(self) -> int:
        """Rimuove TUTTI i checkpoint (sia pipeline che ingestion).

        Returns:
            Numero di file rimossi.
        """
        if not self.checkpoint_dir.exists():
            return 0

        removed = 0
        for f in self.checkpoint_dir.glob("*.json"):
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                logger.warning("Errore rimozione %s: %s", f.name, e)

        logger.info("Rimossi tutti i %d checkpoint.", removed)
        return removed

    # ------------------------------------------------------------------ #
    # Export / Import
    # ------------------------------------------------------------------ #

    def export_checkpoint(
        self,
        checkpoint_path: str | Path,
        export_dir: str | Path,
    ) -> Path:
        """Esporta un checkpoint in una directory di backup.

        Args:
            checkpoint_path: Path al checkpoint da esportare.
            export_dir: Directory di destinazione.

        Returns:
            Path del file esportato.

        Raises:
            FileNotFoundError: Se il checkpoint non esiste.
        """
        src = Path(checkpoint_path)
        if not src.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {src}")

        dst_dir = Path(export_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)

        logger.info("Checkpoint esportato: %s → %s", src.name, dst)
        return dst

    def import_checkpoint(
        self,
        external_path: str | Path,
    ) -> CheckpointInfo:
        """Importa un checkpoint da una directory esterna.

        Args:
            external_path: Path al file checkpoint esterno.

        Returns:
            CheckpointInfo del checkpoint importato.

        Raises:
            FileNotFoundError: Se il file non esiste.
            ValueError: Se il file non è un JSON valido.
        """
        src = Path(external_path)
        if not src.exists():
            raise FileNotFoundError(f"File non trovato: {src}")

        # Valida il JSON
        try:
            json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Il file non è un JSON valido: {e}") from e

        dst = self.checkpoint_dir / src.name
        shutil.copy2(src, dst)

        logger.info("Checkpoint importato: %s", dst.name)
        return CheckpointInfo.from_file(dst)

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict[str, Any]:
        """Genera statistiche aggregate sui checkpoint.

        Returns:
            Dict con statistiche (totale, per tipo, dimensione, ecc.).
        """
        all_cps = self.list_checkpoints(include_latest=True)

        if not all_cps:
            return {
                "total_checkpoints": 0,
                "total_size_kb": 0.0,
                "pipeline_checkpoints": 0,
                "ingestion_checkpoints": 0,
                "nodes": {},
                "latest_checkpoint": None,
                "oldest_checkpoint": None,
            }

        total_size = sum(cp.size_kb for cp in all_cps)
        pipeline_cps = [cp for cp in all_cps if cp.checkpoint_type == CHECKPOINT_TYPE_PIPELINE]
        ingestion_cps = [cp for cp in all_cps if cp.checkpoint_type == CHECKPOINT_TYPE_INGESTION]

        # Per nodo
        nodes: dict[str, int] = {}
        for cp in pipeline_cps:
            nodes[cp.node] = nodes.get(cp.node, 0) + 1

        # Più recente e più vecchio (con timestamp)
        with_ts = [cp for cp in all_cps if cp.timestamp]
        latest = max(with_ts, key=lambda c: c.timestamp) if with_ts else None  # type: ignore[arg-type]
        oldest = min(with_ts, key=lambda c: c.timestamp) if with_ts else None  # type: ignore[arg-type]

        return {
            "total_checkpoints": len(all_cps),
            "total_size_kb": round(total_size, 1),
            "total_size_mb": round(total_size / 1024, 2),
            "pipeline_checkpoints": len(pipeline_cps),
            "ingestion_checkpoints": len(ingestion_cps),
            "nodes": nodes,
            "latest_checkpoint": latest.to_dict() if latest else None,
            "oldest_checkpoint": oldest.to_dict() if oldest else None,
            "has_ingestion_pending": self.has_ingestion_checkpoint(),
        }

    def print_summary(self) -> None:
        """Stampa un riepilogo dei checkpoint con Rich.

        Genera 2 tabelle: statistiche generali e dettaglio checkpoint.
        """
        stats = self.get_stats()

        # Tabella generale
        general = {
            "Totale checkpoint": str(stats["total_checkpoints"]),
            "Dimensione totale": f"{stats.get('total_size_mb', 0):.2f} MB",
            "Pipeline": str(stats["pipeline_checkpoints"]),
            "Ingestion": str(stats["ingestion_checkpoints"]),
            "Ingestion pendente": "Sì" if stats.get("has_ingestion_pending") else "No",
        }

        # Nodi
        nodes = stats.get("nodes", {})
        if nodes:
            for node, count in sorted(nodes.items()):
                general[f"  Nodo '{node}'"] = str(count)

        print_table("Checkpoint Summary", general)

        # Dettaglio checkpoint
        all_cps = self.list_checkpoints(include_latest=False, sort_by="timestamp")
        if all_cps:
            rows = []
            for cp in all_cps[:20]:  # Max 20 righe
                rows.append({
                    "Tipo": cp.checkpoint_type[:8],
                    "Nodo": cp.node,
                    "Timestamp": cp.timestamp_str[:19] if cp.timestamp_str else "?",
                    "Dimensione": f"{cp.size_kb:.1f} KB",
                    "Latest": "✓" if cp.is_latest else "",
                })
            print_table("Dettaglio Checkpoint", rows)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"CheckpointManager(dir={self.checkpoint_dir}, "
            f"total={stats['total_checkpoints']}, "
            f"pipeline={stats['pipeline_checkpoints']}, "
            f"ingestion={stats['ingestion_checkpoints']})"
        )

    def __len__(self) -> int:
        return len(self.list_checkpoints())


# ============================================================================
# Convenience functions (standalone)
# ============================================================================

def get_checkpoint_manager(
    checkpoint_dir: str | Path | None = None,
) -> CheckpointManager:
    """Factory function per creare un CheckpointManager.

    Args:
        checkpoint_dir: Directory dei checkpoint (default da config).

    Returns:
        Istanza di CheckpointManager.
    """
    return CheckpointManager(checkpoint_dir=checkpoint_dir)


def get_resume_recommendation(
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Standalone: raccomanda come riprendere la pipeline.

    Args:
        checkpoint_dir: Directory dei checkpoint (default da config).

    Returns:
        Dict con raccomandazione (vedi CheckpointManager.get_resume_recommendation).
    """
    return CheckpointManager(checkpoint_dir=checkpoint_dir).get_resume_recommendation()


def list_all_checkpoints(
    checkpoint_dir: str | Path | None = None,
) -> list[CheckpointInfo]:
    """Standalone: elenca tutti i checkpoint.

    Args:
        checkpoint_dir: Directory dei checkpoint (default da config).

    Returns:
        Lista di CheckpointInfo.
    """
    return CheckpointManager(checkpoint_dir=checkpoint_dir).list_checkpoints()
