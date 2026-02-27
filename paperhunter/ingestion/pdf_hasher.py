"""
PDF Hasher — Logica hash SHA-256 per Upsert incrementale.

Calcola l'hash di ogni PDF, mantiene un registro persistente (JSON)
e identifica i file nuovi o modificati per il processing incrementale.

Uso:
    from paperhunter.ingestion.pdf_hasher import (
        compute_pdf_hash,
        load_hash_registry,
        save_hash_registry,
        get_new_or_modified_pdfs,
        scan_pdf_directory,
    )

    # Scansione completa con rilevamento delta
    registry = load_hash_registry()
    new_pdfs = get_new_or_modified_pdfs("./scaricati", registry)
    print(f"Nuovi/modificati: {len(new_pdfs)} PDF")

    # Dopo il processing, aggiornare il registro
    for pdf_info in new_pdfs:
        registry[pdf_info["relative_path"]] = {
            "hash": pdf_info["hash"],
            "category": pdf_info["category"],
            ...
        }
    save_hash_registry(registry)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.utils.logging_setup import get_logger, print_table, print_success

logger = get_logger(__name__)


# =============================================================================
# Hash computation
# =============================================================================

def compute_pdf_hash(pdf_path: str | Path, buffer_size: int = 65536) -> str:
    """Calcola l'hash SHA-256 di un file PDF.

    Legge il file a blocchi per supportare PDF di grandi dimensioni
    senza caricare tutto in memoria.

    Args:
        pdf_path: Percorso al file PDF.
        buffer_size: Dimensione dei blocchi di lettura in byte (default: 64KB).

    Returns:
        Hash SHA-256 come stringa esadecimale (64 caratteri).

    Raises:
        FileNotFoundError: Se il file non esiste.
        PermissionError: Se non si hanno i permessi per leggere il file.
        IsADirectoryError: Se il percorso punta a una directory.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File non trovato: {pdf_path}")
    if pdf_path.is_dir():
        raise IsADirectoryError(f"Il percorso è una directory, non un file: {pdf_path}")

    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


# =============================================================================
# Hash Registry — Persist e load
# =============================================================================

def _default_registry_path() -> Path:
    """Restituisce il percorso del registro hash dal config.yaml."""
    return Path(settings.ingestion.hash_registry_path)


def load_hash_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Carica il registro hash da file JSON.

    Il registro mappa ogni PDF (per percorso relativo) ai suoi metadati:
    {
        "Hallucinations1/paper_name.pdf": {
            "hash": "abc123...",
            "doi": "10.1234/...",
            "bibtex_key": "Author_2024_Title",
            "category": "LLM Hallucinations",
            "ingested_at": "2026-02-20T10:30:00",
            "grobid_status": "success"
        }
    }

    Args:
        path: Percorso al file JSON. Se None, usa il percorso da config.yaml.

    Returns:
        Dizionario del registro. Vuoto se il file non esiste.
    """
    registry_path = Path(path) if path else _default_registry_path()

    if not registry_path.exists():
        logger.debug("Registro hash non trovato in %s, creo nuovo registro vuoto", registry_path)
        return {}

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning("Registro hash corrotto (non è un dict), creo nuovo registro vuoto")
            return {}

        logger.info("Registro hash caricato: %d entry da %s", len(data), registry_path)
        return data

    except json.JSONDecodeError as e:
        logger.error("Errore parsing JSON del registro hash: %s", e)
        return {}
    except Exception as e:
        logger.error("Errore caricamento registro hash: %s", e)
        return {}


def save_hash_registry(
    registry: dict[str, dict[str, Any]],
    path: str | Path | None = None,
) -> Path:
    """Salva il registro hash su file JSON.

    Crea la directory padre se non esiste.

    Args:
        registry: Dizionario del registro da salvare.
        path: Percorso al file JSON. Se None, usa il percorso da config.yaml.

    Returns:
        Path del file salvato.

    Raises:
        TypeError: Se il registro contiene valori non serializzabili.
    """
    registry_path = Path(path) if path else _default_registry_path()

    # Crea la directory padre se necessario
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Registro hash salvato: %d entry in %s", len(registry), registry_path)
    return registry_path


# =============================================================================
# Scansione directory e rilevamento delta
# =============================================================================

def _get_category_from_path(pdf_path: Path, base_dir: Path) -> str:
    """Determina la categoria dal percorso relativo del PDF.

    Ogni sottocartella diretta di base_dir rappresenta una categoria.
    Usa il mapping definito in config.yaml se disponibile.

    Args:
        pdf_path: Percorso assoluto o relativo del PDF.
        base_dir: Directory base contenente le sotto-cartelle categoria.

    Returns:
        Nome della categoria (human-readable se nel mapping, altrimenti nome cartella).
    """
    try:
        relative = pdf_path.relative_to(base_dir)
        folder_name = relative.parts[0] if relative.parts else "uncategorized"
    except ValueError:
        folder_name = "uncategorized"

    # Usa il mapping da config.yaml se disponibile
    category_mapping = settings.ingestion.category_mapping
    return category_mapping.get(folder_name, folder_name)


def scan_pdf_directory(
    pdf_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Scansiona ricorsivamente una directory per trovare tutti i PDF.

    Ogni sottocartella diretta è considerata una categoria.

    Args:
        pdf_dir: Directory base da scansionare. Se None, usa config.yaml.

    Returns:
        Lista di dict con informazioni per ogni PDF trovato:
        [
            {
                "absolute_path": Path("/full/path/to/paper.pdf"),
                "relative_path": "Hallucinations1/paper_name.pdf",
                "filename": "paper_name.pdf",
                "category_folder": "Hallucinations1",
                "category": "LLM Hallucinations",
                "hash": "abc123...",
                "size_bytes": 1234567,
            }
        ]
    """
    base_dir = Path(pdf_dir) if pdf_dir else Path(settings.ingestion.pdf_base_dir)

    if not base_dir.exists():
        logger.warning("Directory PDF non trovata: %s", base_dir)
        return []

    if not base_dir.is_dir():
        logger.error("Il percorso non è una directory: %s", base_dir)
        return []

    pdf_files: list[dict[str, Any]] = []

    # Scansione ricorsiva di tutti i .pdf
    for pdf_path in sorted(base_dir.rglob("*.pdf")):
        if not pdf_path.is_file():
            continue

        try:
            relative = pdf_path.relative_to(base_dir)
            category_folder = relative.parts[0] if len(relative.parts) > 1 else "uncategorized"

            pdf_hash = compute_pdf_hash(pdf_path)

            pdf_info = {
                "absolute_path": pdf_path,
                "relative_path": str(relative),
                "filename": pdf_path.name,
                "category_folder": category_folder,
                "category": _get_category_from_path(pdf_path, base_dir),
                "hash": pdf_hash,
                "size_bytes": pdf_path.stat().st_size,
            }
            pdf_files.append(pdf_info)

        except Exception as e:
            logger.warning("Errore scansione PDF %s: %s", pdf_path.name, e)
            continue

    logger.info(
        "Scansione completata: %d PDF trovati in %d categorie da %s",
        len(pdf_files),
        len({p["category_folder"] for p in pdf_files}),
        base_dir,
    )

    return pdf_files


def get_new_or_modified_pdfs(
    pdf_dir: str | Path | None = None,
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Identifica i PDF nuovi o modificati rispetto al registro hash.

    Confronta gli hash SHA-256 attuali con quelli nel registro:
    - PDF nuovi: presenti su disco ma assenti nel registro.
    - PDF modificati: presenti nel registro ma con hash diverso.

    Args:
        pdf_dir: Directory base da scansionare. Se None, usa config.yaml.
        registry: Registro hash corrente. Se None, lo carica da file.

    Returns:
        Lista di dict (come scan_pdf_directory) con solo i PDF nuovi/modificati.
        Ogni dict ha un campo aggiuntivo "status": "new" | "modified".
    """
    if registry is None:
        registry = load_hash_registry()

    # Scansiona tutti i PDF su disco
    all_pdfs = scan_pdf_directory(pdf_dir)

    new_or_modified: list[dict[str, Any]] = []

    for pdf_info in all_pdfs:
        rel_path = pdf_info["relative_path"]
        current_hash = pdf_info["hash"]

        if rel_path not in registry:
            # PDF completamente nuovo
            pdf_info["status"] = "new"
            new_or_modified.append(pdf_info)
            logger.debug("Nuovo PDF: %s", rel_path)

        elif registry[rel_path].get("hash") != current_hash:
            # PDF esistente ma modificato
            pdf_info["status"] = "modified"
            pdf_info["old_hash"] = registry[rel_path].get("hash", "")
            new_or_modified.append(pdf_info)
            logger.debug("PDF modificato: %s", rel_path)

        # else: PDF non cambiato → skip

    # Conta anche i PDF rimossi dal disco (presenti nel registro ma non su disco)
    current_paths = {p["relative_path"] for p in all_pdfs}
    removed = [k for k in registry if k not in current_paths]
    if removed:
        logger.info("PDF rimossi dal disco (presenti nel registro): %d", len(removed))

    # Statistiche
    n_new = sum(1 for p in new_or_modified if p["status"] == "new")
    n_modified = sum(1 for p in new_or_modified if p["status"] == "modified")
    n_unchanged = len(all_pdfs) - len(new_or_modified)

    logger.info(
        "Delta rilevato: %d nuovi, %d modificati, %d invariati (totale su disco: %d)",
        n_new, n_modified, n_unchanged, len(all_pdfs),
    )

    return new_or_modified


def create_registry_entry(
    pdf_info: dict[str, Any],
    doi: Optional[str] = None,
    bibtex_key: Optional[str] = None,
    grobid_status: str = "pending",
) -> dict[str, Any]:
    """Crea una entry del registro hash per un PDF processato.

    Utility per costruire l'entry da inserire nel registro
    dopo il processing del PDF.

    Args:
        pdf_info: Informazioni del PDF (da scan_pdf_directory).
        doi: DOI del paper (se noto).
        bibtex_key: Chiave BibTeX generata.
        grobid_status: Stato GROBID ("pending", "success", "failed", "fallback").

    Returns:
        Dict pronto per l'inserimento nel registro.
    """
    return {
        "hash": pdf_info["hash"],
        "category": pdf_info["category"],
        "category_folder": pdf_info["category_folder"],
        "filename": pdf_info["filename"],
        "size_bytes": pdf_info["size_bytes"],
        "doi": doi,
        "bibtex_key": bibtex_key,
        "grobid_status": grobid_status,
        "ingested_at": datetime.now().isoformat(),
    }


def update_registry_entry(
    registry: dict[str, dict[str, Any]],
    relative_path: str,
    updates: dict[str, Any],
) -> None:
    """Aggiorna campi specifici di un'entry nel registro.

    Args:
        registry: Registro hash (modificato in-place).
        relative_path: Chiave del PDF nel registro.
        updates: Campi da aggiornare.
    """
    if relative_path not in registry:
        logger.warning("Entry non trovata nel registro: %s", relative_path)
        return

    registry[relative_path].update(updates)
    registry[relative_path]["updated_at"] = datetime.now().isoformat()


def get_registry_stats(registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Calcola statistiche dal registro hash.

    Args:
        registry: Registro hash.

    Returns:
        Dict con statistiche: totale, per categoria, per stato GROBID.
    """
    stats: dict[str, Any] = {
        "total_entries": len(registry),
        "by_category": {},
        "by_grobid_status": {},
        "total_size_bytes": 0,
    }

    for _path, entry in registry.items():
        # Per categoria
        cat = entry.get("category", "unknown")
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

        # Per status GROBID
        status = entry.get("grobid_status", "unknown")
        stats["by_grobid_status"][status] = stats["by_grobid_status"].get(status, 0) + 1

        # Size totale
        stats["total_size_bytes"] += entry.get("size_bytes", 0)

    stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)

    return stats
