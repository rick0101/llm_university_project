"""postprocessing/bibliography_generator.py — Generazione .bib finale filtrato.

Genera il file `.bib` finale contenente solo le entry BibTeX effettivamente
citate nel paper generato. Valida, filtra, ordina e produce report statistici
sulla bibliografia.

Si integra con:
- ``paperhunter.ingestion.bibtex_generator``: caricamento master .bib
- ``paperhunter.agents.assembler_agent``: raccolta chiavi usate dal survey
- ``paperhunter.postprocessing.provenance_tracker``: correlazione con provenance

Uso tipico::

    from paperhunter.postprocessing.bibliography_generator import (
        BibliographyGenerator,
        generate_final_bib,
        load_and_filter_bib,
    )

    # Uso completo con classe
    gen = BibliographyGenerator()
    gen.load_master_bib()  # oppure gen.load_master_bib("output/references.bib")
    report = gen.generate(used_keys={"Zou_2023_GCG", "Li_2023_Halueval"})
    gen.save_bib("output/versions/v1.0/references.bib")
    gen.save_report("output/versions/v1.0/bib_report.json")

    # Shortcut funzionale
    bib_path = generate_final_bib(
        used_keys={"Zou_2023_GCG", "Li_2023_Halueval"},
        master_bib_path="output/references.bib",
        output_path="output/final.bib",
    )

    # Da SurveyState
    from paperhunter.postprocessing.bibliography_generator import generate_bib_from_state
    bib_path, report = generate_bib_from_state(state)
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from paperhunter.config import settings
from paperhunter.ingestion.bibtex_generator import (
    load_master_bib,
    save_master_bib,
    validate_bibtex_entry,
)
from paperhunter.utils.logging_setup import (
    get_logger,
    print_panel,
    print_success,
    print_table,
    print_warning,
)

logger = get_logger(__name__)


# ============================================================================
# Utility interne
# ============================================================================

def _extract_bib_key(entry: str) -> str | None:
    """Estrae la chiave BibTeX da una entry.

    Args:
        entry: Stringa di una entry BibTeX (es. ``@article{Zou_2023_GCG, ...}``).

    Returns:
        La chiave BibTeX (es. ``Zou_2023_GCG``) o None se non trovata.
    """
    match = re.search(r"@\w+\{([^,]+),", entry)
    return match.group(1).strip() if match else None


def _extract_bib_type(entry: str) -> str | None:
    """Estrae il tipo di entry BibTeX (article, inproceedings, misc, ...).

    Args:
        entry: Stringa di una entry BibTeX.

    Returns:
        Il tipo (lowercase) o None.
    """
    match = re.search(r"@(\w+)\{", entry)
    return match.group(1).lower() if match else None


def _extract_field(entry: str, field_name: str) -> str | None:
    """Estrae il valore di un campo da una entry BibTeX.

    Supporta valori tra ``{...}`` e valori senza parentesi (es. ``month = jan``).

    Args:
        entry: Stringa dell'entry.
        field_name: Nome del campo (case-insensitive).

    Returns:
        Valore del campo (senza le parentesi graffe esterne) o None.
    """
    # Pattern: field = {value} (con possibili newline interni)
    pattern = rf"\b{field_name}\s*=\s*\{{((?:[^{{}}]|\{{[^{{}}]*\}})*)\}}"
    match = re.search(pattern, entry, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Pattern: field = value (senza parentesi, es. month = jan)
    pattern2 = rf"\b{field_name}\s*=\s*([^,\}}]+)"
    match2 = re.search(pattern2, entry, re.IGNORECASE)
    if match2:
        val = match2.group(1).strip().strip("{}")
        return val if val else None

    return None


def _extract_year_from_entry(entry: str) -> int | None:
    """Estrae l'anno da una entry BibTeX.

    Args:
        entry: Stringa dell'entry.

    Returns:
        Anno come intero, o None.
    """
    year_str = _extract_field(entry, "year")
    if year_str:
        digits = re.search(r"(\d{4})", year_str)
        if digits:
            return int(digits.group(1))
    return None


def _count_citations_in_text(text: str) -> Counter:
    """Conta le occorrenze di ogni chiave BibTeX citata in un testo.

    Cerca pattern ``[bibtex_key]`` nel testo.

    Args:
        text: Testo del survey (Markdown).

    Returns:
        Counter chiave → numero di occorrenze.
    """
    # Pattern: [Key_YYYY_Word] o simili
    matches = re.findall(r"\[([A-Za-z][A-Za-z0-9_]+(?:_\d{4})?[A-Za-z0-9_]*)\]", text)
    # Filtra false positive (Table_1, Figure_2, ecc.)
    filtered = [m for m in matches if not re.match(r"^(Table|Figure|Fig|Tab|Eq|Section|Sec)_", m)]
    return Counter(filtered)


def _reformat_entry(entry: str) -> str:
    """Riformatta un'entry BibTeX con indentazione consistente.

    Args:
        entry: Entry BibTeX grezza.

    Returns:
        Entry riformattata con indentazione a 2 spazi.
    """
    lines = entry.strip().split("\n")
    if not lines:
        return entry

    formatted: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Prima riga (@type{key,) e ultima riga (}) non indentate
        if stripped.startswith("@") or stripped == "}":
            formatted.append(stripped)
        else:
            formatted.append(f"  {stripped}")

    return "\n".join(formatted)


# ============================================================================
# BibliographyGenerator
# ============================================================================

class BibliographyGenerator:
    """Generatore della bibliografia finale per il survey.

    Carica il master .bib, filtra per le chiavi effettivamente citate,
    valida le entry, e genera report statistici.

    Attributes:
        master_bib: Dict chiave → entry dal master .bib.
        used_keys: Set di chiavi citate nel survey.
        filtered_entries: Dict chiave → entry filtrate (solo quelle usate).
        validation_results: Dict chiave → (is_valid, issues) per ogni entry.
    """

    def __init__(self) -> None:
        """Inizializza il generatore."""
        self._master_bib: dict[str, str] = {}
        self._used_keys: set[str] = set()
        self._filtered_entries: dict[str, str] = {}
        self._missing_keys: set[str] = set()
        self._invalid_entries: dict[str, list[str]] = {}
        self._validation_results: dict[str, tuple[bool, list[str]]] = {}
        self._generated_at: str | None = None
        self._citation_counts: Counter = Counter()

    # ------------------------------------------------------------------
    # Proprietà
    # ------------------------------------------------------------------

    @property
    def master_bib(self) -> dict[str, str]:
        """Dict chiave → entry dal master .bib."""
        return dict(self._master_bib)

    @property
    def used_keys(self) -> set[str]:
        """Set di chiavi BibTeX usate nel survey."""
        return set(self._used_keys)

    @property
    def filtered_entries(self) -> dict[str, str]:
        """Dict chiave → entry filtrate (solo quelle usate e trovate)."""
        return dict(self._filtered_entries)

    @property
    def missing_keys(self) -> set[str]:
        """Chiavi citate nel survey ma non presenti nel master .bib."""
        return set(self._missing_keys)

    @property
    def invalid_entries(self) -> dict[str, list[str]]:
        """Chiavi con entry non valide → lista di problemi."""
        return dict(self._invalid_entries)

    @property
    def citation_counts(self) -> Counter:
        """Conteggio occorrenze per chiave nel testo del survey."""
        return Counter(self._citation_counts)

    # ------------------------------------------------------------------
    # Caricamento
    # ------------------------------------------------------------------

    def load_master_bib(
        self,
        bib_path: str | Path | None = None,
    ) -> int:
        """Carica il master .bib dal file.

        Args:
            bib_path: Percorso del file .bib.
                      Default: ``output/references.bib`` da config.yaml.

        Returns:
            Numero di entry caricate.
        """
        if bib_path is None:
            bib_path = Path(settings.output.base_dir) / "references.bib"
        else:
            bib_path = Path(bib_path)

        self._master_bib = load_master_bib(bib_path)
        logger.info(
            f"Master .bib caricato: {len(self._master_bib)} entry da {bib_path}"
        )
        return len(self._master_bib)

    def load_master_bib_from_dict(self, entries: dict[str, str]) -> int:
        """Carica entry da un dict già in memoria.

        Args:
            entries: Dict chiave → entry BibTeX.

        Returns:
            Numero di entry caricate.
        """
        self._master_bib = dict(entries)
        logger.info(f"Master .bib caricato da dict: {len(self._master_bib)} entry")
        return len(self._master_bib)

    # ------------------------------------------------------------------
    # Generazione
    # ------------------------------------------------------------------

    def generate(
        self,
        used_keys: set[str] | list[str],
        *,
        validate: bool = True,
        reformat: bool = True,
        survey_text: str | None = None,
    ) -> dict[str, Any]:
        """Genera la bibliografia filtrata.

        Filtra il master .bib per le chiavi usate, valida le entry,
        e produce un report.

        Args:
            used_keys: Set o lista delle chiavi BibTeX citate nel survey.
            validate: Se True, valida ogni entry per campi obbligatori.
            reformat: Se True, riformatta le entry con indentazione consistente.
            survey_text: Testo del survey (opzionale). Se fornito, conta
                         le occorrenze di ogni citazione.

        Returns:
            Report dict con statistiche e dettagli della generazione.
        """
        start_time = time.time()
        self._used_keys = set(used_keys)
        self._filtered_entries = {}
        self._missing_keys = set()
        self._invalid_entries = {}
        self._validation_results = {}
        self._citation_counts = Counter()

        if not self._master_bib:
            logger.warning(
                "Master .bib vuoto. Chiamare load_master_bib() prima di generate()."
            )

        # Conta citazioni nel testo (se fornito)
        if survey_text:
            self._citation_counts = _count_citations_in_text(survey_text)
            logger.info(
                f"Contate {sum(self._citation_counts.values())} citazioni "
                f"per {len(self._citation_counts)} chiavi nel testo."
            )

        # Filtra e valida
        for key in sorted(self._used_keys):
            if key in self._master_bib:
                entry = self._master_bib[key]

                # Validazione
                if validate:
                    is_valid, issues = validate_bibtex_entry(entry)
                    self._validation_results[key] = (is_valid, issues)
                    if not is_valid:
                        self._invalid_entries[key] = issues
                        logger.warning(
                            f"Entry BibTeX non valida per '{key}': {issues}"
                        )

                # Riformattazione
                if reformat:
                    entry = _reformat_entry(entry)

                self._filtered_entries[key] = entry
            else:
                self._missing_keys.add(key)
                logger.warning(f"Citazione '{key}' non trovata nel master .bib")

        elapsed = time.time() - start_time
        self._generated_at = datetime.now().isoformat()

        report = self._build_report(elapsed)

        # Log riassuntivo
        logger.info(
            f"Bibliografia generata: {len(self._filtered_entries)} entry filtrate, "
            f"{len(self._missing_keys)} mancanti, "
            f"{len(self._invalid_entries)} non valide "
            f"({elapsed:.2f}s)"
        )

        if self._missing_keys:
            print_warning(
                f"{len(self._missing_keys)} citazioni non trovate nel master .bib: "
                f"{', '.join(sorted(self._missing_keys)[:10])}"
                f"{'...' if len(self._missing_keys) > 10 else ''}"
            )

        return report

    # ------------------------------------------------------------------
    # Salvataggio
    # ------------------------------------------------------------------

    def save_bib(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """Salva il file .bib filtrato.

        Args:
            output_path: Percorso del file .bib di output.
                         Default: ``output/references_final.bib``.

        Returns:
            Path del file salvato.

        Raises:
            RuntimeError: Se ``generate()`` non è stato chiamato.
        """
        if not self._filtered_entries and not self._used_keys:
            raise RuntimeError(
                "Nessuna entry da salvare. Chiamare generate() prima di save_bib()."
            )

        if output_path is None:
            output_path = (
                Path(settings.output.base_dir) / "references_final.bib"
            )
        else:
            output_path = Path(output_path)

        entries = list(self._filtered_entries.values())
        saved_path = save_master_bib(entries, output_path)

        logger.info(
            f"Bibliografia finale salvata: {saved_path} "
            f"({len(self._filtered_entries)} entry)"
        )
        print_success(
            f"Bibliografia salvata: {saved_path} ({len(self._filtered_entries)} entry)"
        )
        return saved_path

    def save_report(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """Salva il report JSON della bibliografia.

        Args:
            output_path: Percorso del file JSON.
                         Default: ``output/bib_report.json``.

        Returns:
            Path del file salvato.
        """
        if output_path is None:
            output_path = Path(settings.output.base_dir) / "bib_report.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self._build_report()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report bibliografia salvato: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Analisi e statistiche
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Statistiche sulla bibliografia generata.

        Returns:
            Dict con conteggi, distribuzione per tipo, per anno, ecc.
        """
        stats: dict[str, Any] = {
            "master_bib_total": len(self._master_bib),
            "used_keys_total": len(self._used_keys),
            "filtered_entries": len(self._filtered_entries),
            "missing_keys": len(self._missing_keys),
            "invalid_entries": len(self._invalid_entries),
            "valid_entries": len(self._filtered_entries) - len(self._invalid_entries),
            "coverage_rate": (
                len(self._filtered_entries) / len(self._used_keys)
                if self._used_keys
                else 0.0
            ),
        }

        # Distribuzione per tipo BibTeX
        type_dist: Counter = Counter()
        year_dist: Counter = Counter()
        for key, entry in self._filtered_entries.items():
            bib_type = _extract_bib_type(entry)
            if bib_type:
                type_dist[bib_type] += 1
            year = _extract_year_from_entry(entry)
            if year:
                year_dist[year] += 1

        stats["type_distribution"] = dict(type_dist.most_common())
        stats["year_distribution"] = dict(sorted(year_dist.items()))

        # Anno range
        if year_dist:
            stats["year_min"] = min(year_dist.keys())
            stats["year_max"] = max(year_dist.keys())
        else:
            stats["year_min"] = None
            stats["year_max"] = None

        # Top cited (se citation_counts disponibile)
        if self._citation_counts:
            stats["top_cited"] = dict(self._citation_counts.most_common(20))
            stats["total_citation_occurrences"] = sum(self._citation_counts.values())
            stats["avg_citations_per_key"] = (
                sum(self._citation_counts.values()) / len(self._citation_counts)
                if self._citation_counts
                else 0.0
            )
        else:
            stats["top_cited"] = {}
            stats["total_citation_occurrences"] = 0
            stats["avg_citations_per_key"] = 0.0

        return stats

    def get_missing_keys_report(self) -> list[dict[str, Any]]:
        """Report dettagliato delle chiavi mancanti.

        Per ogni chiave mancante, riporta:
        - La chiave stessa
        - Quante volte è citata nel testo (se disponibile)
        - Possibili match parziali nel master .bib (suggerimenti)

        Returns:
            Lista di dict con dettagli per ogni chiave mancante.
        """
        report: list[dict[str, Any]] = []

        for key in sorted(self._missing_keys):
            entry: dict[str, Any] = {
                "key": key,
                "citation_count": self._citation_counts.get(key, 0),
                "suggestions": [],
            }

            # Cerca match parziali nel master .bib
            key_lower = key.lower()
            for master_key in self._master_bib:
                master_lower = master_key.lower()
                # Match parziale: condividono almeno una parte significativa
                if (
                    key_lower in master_lower
                    or master_lower in key_lower
                    or _key_similarity(key_lower, master_lower) >= 0.5
                ):
                    entry["suggestions"].append(master_key)

            report.append(entry)

        return report

    def get_entry_details(self, bibtex_key: str) -> dict[str, Any] | None:
        """Dettagli di una singola entry BibTeX.

        Args:
            bibtex_key: Chiave BibTeX.

        Returns:
            Dict con tipo, anno, campi estratti, validazione. None se non trovata.
        """
        entry = self._filtered_entries.get(bibtex_key) or self._master_bib.get(
            bibtex_key
        )
        if not entry:
            return None

        details: dict[str, Any] = {
            "key": bibtex_key,
            "type": _extract_bib_type(entry),
            "title": _extract_field(entry, "title"),
            "author": _extract_field(entry, "author"),
            "year": _extract_year_from_entry(entry),
            "venue": (
                _extract_field(entry, "booktitle")
                or _extract_field(entry, "journal")
            ),
            "doi": _extract_field(entry, "doi"),
            "in_filtered": bibtex_key in self._filtered_entries,
            "in_master": bibtex_key in self._master_bib,
            "citation_count": self._citation_counts.get(bibtex_key, 0),
        }

        if bibtex_key in self._validation_results:
            is_valid, issues = self._validation_results[bibtex_key]
            details["valid"] = is_valid
            details["issues"] = issues
        else:
            details["valid"] = None
            details["issues"] = []

        return details

    # ------------------------------------------------------------------
    # Stampa console
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Stampa un riepilogo formattato in console con Rich."""
        stats = self.get_stats()

        # Tabella generale
        general_data = {
            "Master .bib entries": str(stats["master_bib_total"]),
            "Citazioni nel survey": str(stats["used_keys_total"]),
            "Entry trovate": str(stats["filtered_entries"]),
            "Entry mancanti": str(stats["missing_keys"]),
            "Entry non valide": str(stats["invalid_entries"]),
            "Copertura": f"{stats['coverage_rate']:.1%}",
        }
        print_table("📚 Bibliografia — Riepilogo", general_data)

        # Distribuzione per tipo
        if stats.get("type_distribution"):
            type_data = {
                k: str(v) for k, v in stats["type_distribution"].items()
            }
            print_table("📊 Distribuzione per tipo", type_data)

        # Distribuzione per anno
        if stats.get("year_distribution"):
            year_data = {
                str(k): str(v) for k, v in stats["year_distribution"].items()
            }
            print_table("📅 Distribuzione per anno", year_data)

        # Top cited
        if stats.get("top_cited"):
            top = {k: str(v) for k, v in list(stats["top_cited"].items())[:10]}
            print_table("🏆 Top 10 più citati", top)

        # Missing
        if self._missing_keys:
            print_warning(
                f"Chiavi mancanti: {', '.join(sorted(self._missing_keys)[:15])}"
                f"{'...' if len(self._missing_keys) > 15 else ''}"
            )

        # Invalid
        if self._invalid_entries:
            for key, issues in list(self._invalid_entries.items())[:5]:
                print_warning(f"Entry non valida '{key}': {'; '.join(issues)}")

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(self, elapsed: float = 0.0) -> dict[str, Any]:
        """Costruisce il report completo della generazione.

        Args:
            elapsed: Tempo impiegato in secondi.

        Returns:
            Report dict completo.
        """
        stats = self.get_stats()

        report: dict[str, Any] = {
            "generated_at": self._generated_at or datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "statistics": stats,
            "missing_keys": sorted(self._missing_keys),
            "invalid_entries": {
                k: v for k, v in sorted(self._invalid_entries.items())
            },
            "entries": [
                {
                    "key": key,
                    "type": _extract_bib_type(entry),
                    "year": _extract_year_from_entry(entry),
                    "title": _extract_field(entry, "title"),
                    "valid": self._validation_results.get(key, (True, []))[0],
                }
                for key, entry in sorted(self._filtered_entries.items())
            ],
        }

        # Aggiungi citation counts se disponibili
        if self._citation_counts:
            report["citation_counts"] = dict(self._citation_counts.most_common())

        return report

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BibliographyGenerator("
            f"master={len(self._master_bib)}, "
            f"used={len(self._used_keys)}, "
            f"filtered={len(self._filtered_entries)}, "
            f"missing={len(self._missing_keys)})"
        )

    def __len__(self) -> int:
        return len(self._filtered_entries)


# ============================================================================
# Utility helper
# ============================================================================

def _key_similarity(key_a: str, key_b: str) -> float:
    """Calcola la similarità Jaccard tra due chiavi BibTeX.

    Splitta le chiavi in token (su ``_``) e calcola la similarità.

    Args:
        key_a: Prima chiave (lowercase).
        key_b: Seconda chiave (lowercase).

    Returns:
        Score Jaccard 0.0-1.0.
    """
    tokens_a = set(key_a.split("_"))
    tokens_b = set(key_b.split("_"))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


# ============================================================================
# Convenience functions
# ============================================================================

def generate_final_bib(
    used_keys: set[str] | list[str],
    *,
    master_bib_path: str | Path | None = None,
    output_path: str | Path | None = None,
    validate: bool = True,
    reformat: bool = True,
    survey_text: str | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Genera il file .bib finale filtrando il master .bib.

    Convenience function che crea un ``BibliographyGenerator``, carica
    il master .bib, filtra, valida, e salva.

    Args:
        used_keys: Chiavi BibTeX citate nel survey.
        master_bib_path: Path del master .bib. Default da config.yaml.
        output_path: Path del .bib output. Default: ``output/references_final.bib``.
        validate: Se True, valida le entry.
        reformat: Se True, riformatta le entry.
        survey_text: Testo del survey per conteggio citazioni.

    Returns:
        Tupla (Path del .bib salvato o None, report dict).
    """
    gen = BibliographyGenerator()
    count = gen.load_master_bib(master_bib_path)

    if count == 0:
        logger.warning("Master .bib vuoto, nessun file generato.")
        return None, gen._build_report()

    report = gen.generate(
        used_keys=used_keys,
        validate=validate,
        reformat=reformat,
        survey_text=survey_text,
    )

    if len(gen) == 0:
        logger.warning("Nessuna entry trovata nel master .bib per le chiavi usate.")
        return None, report

    bib_path = gen.save_bib(output_path)
    return bib_path, report


def generate_bib_from_state(
    state: dict[str, Any],
    *,
    master_bib_path: str | Path | None = None,
    output_path: str | Path | None = None,
    validate: bool = True,
) -> tuple[Path | None, dict[str, Any]]:
    """Genera il .bib finale a partire dallo stato SurveyState.

    Raccoglie tutte le chiavi citazioni dalle sezioni nello stato,
    poi filtra il master .bib.

    Args:
        state: SurveyState dict.
        master_bib_path: Path master .bib. Default da config.yaml.
        output_path: Path .bib output.
        validate: Se True, valida entry.

    Returns:
        Tupla (Path .bib o None, report dict).
    """
    # Importa le funzioni per evitare import circolare
    from paperhunter.agents.state import collect_all_citation_keys

    used_keys = collect_all_citation_keys(state)
    logger.info(f"Chiavi citazione raccolte dallo stato: {len(used_keys)}")

    # Estrai il testo del survey per conteggio occorrenze
    survey_text = state.get("final_paper", "")

    return generate_final_bib(
        used_keys=used_keys,
        master_bib_path=master_bib_path,
        output_path=output_path,
        validate=validate,
        survey_text=survey_text or None,
    )


def load_and_filter_bib(
    master_bib_path: str | Path,
    used_keys: set[str] | list[str],
) -> dict[str, str]:
    """Carica un .bib e filtra per le chiavi specificate.

    Helper leggero, senza validazione o report.

    Args:
        master_bib_path: Path del file .bib sorgente.
        used_keys: Chiavi da tenere.

    Returns:
        Dict chiave → entry per le chiavi trovate.
    """
    master = load_master_bib(master_bib_path)
    keys_set = set(used_keys)
    return {k: v for k, v in master.items() if k in keys_set}


def merge_bib_files(
    *bib_paths: str | Path,
    output_path: str | Path | None = None,
) -> tuple[Path, int]:
    """Unisce più file .bib in uno, rimuovendo duplicati.

    Args:
        *bib_paths: Percorsi dei file .bib da unire.
        output_path: Percorso del file .bib output.
                     Default: ``output/references_merged.bib``.

    Returns:
        Tupla (Path del file salvato, numero di entry uniche).
    """
    merged: dict[str, str] = {}

    for bib_path in bib_paths:
        entries = load_master_bib(bib_path)
        for key, entry in entries.items():
            if key not in merged:
                merged[key] = entry
            else:
                logger.debug(f"Duplicato ignorato nel merge: {key}")

    if output_path is None:
        output_path = Path(settings.output.base_dir) / "references_merged.bib"

    entries_list = list(merged.values())
    saved_path = save_master_bib(entries_list, output_path)

    logger.info(
        f"Merge .bib completato: {len(merged)} entry uniche da "
        f"{len(bib_paths)} file → {saved_path}"
    )
    return saved_path, len(merged)
