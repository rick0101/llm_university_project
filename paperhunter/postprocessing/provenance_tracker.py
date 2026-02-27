"""postprocessing/provenance_tracker.py — Provenance tracking completo.

Genera un JSON di provenance che mappa ogni claim del survey alla sua
fonte originale nel corpus di paper, con stato di verifica.

Integra i dati prodotti dal Critic Agent (VerificationDetail) e li
organizza in un report strutturato per sezione, per citazione, e con
statistiche aggregate.

Uso tipico::

    from paperhunter.postprocessing.provenance_tracker import ProvenanceTracker

    tracker = ProvenanceTracker()

    # Aggiungere record singoli
    tracker.add_record(
        section="prompt_injection_attacks",
        claim="GCG generates adversarial suffixes that transfer across models.",
        citation_key="Zou_2023_GCG",
        chunk_id="abc123",
        source_text="We propose Greedy Coordinate Gradient...",
        verification_status="SUPPORTED",
        confidence=0.95,
    )

    # Oppure importare dal SurveyState della pipeline
    tracker.import_from_state(state)

    # Generare report e salvare
    report = tracker.generate_report()
    tracker.save_json("output/provenance/provenance.json")
    tracker.save_markdown("output/provenance/provenance_report.md")
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from paperhunter.config import settings
from paperhunter.models import ProvenanceRecord, VerificationStatus
from paperhunter.utils.logging_setup import (
    get_logger,
    print_panel,
    print_success,
    print_table,
    print_warning,
)

logger = get_logger(__name__)

# ============================================================================
# Schema version per il JSON di provenance
# ============================================================================

_PROVENANCE_SCHEMA_VERSION = "1.0"


# ============================================================================
# ProvenanceTracker
# ============================================================================

class ProvenanceTracker:
    """Tracker completo per la provenance delle citazioni nel survey.

    Raccoglie, organizza e genera report dettagliati sulla provenance
    di ogni claim → citazione → chunk sorgente nel paper generato.

    Attributes:
        records: Lista di tutti i ProvenanceRecord.
        version: Versione del survey associata (opzionale).
    """

    def __init__(self, version: str | None = None) -> None:
        """Inizializza il tracker.

        Args:
            version: Versione del survey (es. "1.0", "2.1").
        """
        self._records: list[ProvenanceRecord] = []
        self._version = version or "1.0"
        self._created_at = datetime.now()

    # ------------------------------------------------------------------
    # Proprietà
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[ProvenanceRecord]:
        """Lista di tutti i record di provenance."""
        return list(self._records)

    @property
    def record_count(self) -> int:
        """Numero totale di record."""
        return len(self._records)

    @property
    def version(self) -> str:
        """Versione del survey."""
        return self._version

    # ------------------------------------------------------------------
    # Aggiunta record
    # ------------------------------------------------------------------

    def add_record(
        self,
        section: str,
        claim: str,
        citation_key: str,
        chunk_id: str | None = None,
        source_text: str = "",
        verification_status: str | VerificationStatus = VerificationStatus.SOURCE_NOT_FOUND,
        confidence: float = 0.0,
        reasoning: str = "",
    ) -> ProvenanceRecord:
        """Aggiunge un singolo record di provenance.

        Args:
            section: Chiave/titolo della sezione del survey.
            claim: Testo del claim nel survey.
            citation_key: Chiave BibTeX citata (es. Zou_2023_GCG).
            chunk_id: ID del chunk sorgente nel vector store.
            source_text: Estratto del testo sorgente dal chunk.
            verification_status: Stato di verifica (stringa o enum).
            confidence: Confidenza della verifica (0.0-1.0).
            reasoning: Ragionamento sulla verifica.

        Returns:
            Il ProvenanceRecord creato.
        """
        # Normalizza lo status
        if isinstance(verification_status, str):
            verification_status = _normalize_status(verification_status)

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        record = ProvenanceRecord(
            section=section,
            claim_text=claim[:1000],  # Tronca claim troppo lunghi
            citation_key=citation_key,
            source_chunk_id=chunk_id,
            source_text_excerpt=source_text[:500] if source_text else "",
            verification_status=verification_status,
            confidence=confidence,
        )

        self._records.append(record)
        logger.debug(
            f"Provenance record aggiunto: [{citation_key}] in '{section}' "
            f"→ {verification_status.value}"
        )
        return record

    def add_records_batch(
        self,
        records: list[dict[str, Any]],
    ) -> int:
        """Aggiunge più record da una lista di dict.

        Ogni dict deve contenere le chiavi richieste da ``add_record()``.
        Le chiavi possono anche seguire il formato restituito dal Critic Agent.

        Args:
            records: Lista di dict con dati dei record.

        Returns:
            Numero di record aggiunti con successo.
        """
        added = 0
        for data in records:
            try:
                # Supporta sia il formato ProvenanceTracker che il formato Critic
                section = data.get("section") or data.get("section_key", "unknown")
                claim = data.get("claim") or data.get("claim_text", "")
                citation_key = data.get("citation_key", "")
                chunk_id = data.get("chunk_id") or data.get("source_chunk_id")
                source_text = (
                    data.get("source_text")
                    or data.get("source_text_excerpt", "")
                )
                status = (
                    data.get("verification_status", "SOURCE_NOT_FOUND")
                )
                confidence = data.get("confidence", 0.0)
                reasoning = data.get("reasoning", "")

                self.add_record(
                    section=section,
                    claim=claim,
                    citation_key=citation_key,
                    chunk_id=chunk_id,
                    source_text=source_text,
                    verification_status=status,
                    confidence=confidence,
                    reasoning=reasoning,
                )
                added += 1
            except Exception as e:
                logger.warning(f"Record provenance ignorato per errore: {e}")
                continue

        logger.info(f"Aggiunti {added}/{len(records)} record di provenance in batch.")
        return added

    def import_from_state(self, state: dict[str, Any]) -> int:
        """Importa record di provenance dallo stato della pipeline LangGraph.

        Lo stato ``SurveyState`` contiene una lista ``provenance`` di dict
        generati dal Critic Agent (``_build_provenance_records``).

        Args:
            state: Stato della pipeline (SurveyState TypedDict).

        Returns:
            Numero di record importati.
        """
        provenance_data = state.get("provenance", [])
        if not provenance_data:
            logger.info("Nessun record di provenance nello stato.")
            return 0

        logger.info(f"Importazione {len(provenance_data)} record dallo stato pipeline.")
        return self.add_records_batch(provenance_data)

    # ------------------------------------------------------------------
    # Generazione report
    # ------------------------------------------------------------------

    def generate_report(self) -> dict[str, Any]:
        """Genera il report di provenance completo.

        Returns:
            Dict strutturato con metadati, statistiche aggregate,
            statistiche per sezione, e tutti i record.
        """
        stats = self.get_stats()

        # Raggruppa record per sezione
        by_section = self._group_by_section()
        section_reports: list[dict[str, Any]] = []

        for section_key, section_records in sorted(by_section.items()):
            section_stats = _compute_section_stats(section_records)
            section_reports.append({
                "section": section_key,
                "statistics": section_stats,
                "records": [
                    _record_to_dict(r) for r in section_records
                ],
            })

        report = {
            "schema_version": _PROVENANCE_SCHEMA_VERSION,
            "survey_version": self._version,
            "generated_at": datetime.now().isoformat(),
            "created_at": self._created_at.isoformat(),
            "total_claims": stats["total_claims"],
            "verified_claims": stats["verified_claims"],
            "unverified_claims": stats["unverified_claims"],
            "coverage_rate": stats["coverage_rate"],
            "statistics": stats,
            "sections": section_reports,
        }

        return report

    def get_stats(self) -> dict[str, Any]:
        """Calcola statistiche aggregate su tutti i record.

        Returns:
            Dict con statistiche complete.
        """
        total = len(self._records)
        if total == 0:
            return {
                "total_claims": 0,
                "verified_claims": 0,
                "unverified_claims": 0,
                "supported": 0,
                "partially_supported": 0,
                "not_supported": 0,
                "source_not_found": 0,
                "coverage_rate": 0.0,
                "support_rate": 0.0,
                "avg_confidence": 0.0,
                "unique_citation_keys": 0,
                "unique_sections": 0,
                "sections": [],
                "top_cited_papers": [],
            }

        supported = sum(
            1 for r in self._records
            if r.verification_status == VerificationStatus.SUPPORTED
        )
        partially = sum(
            1 for r in self._records
            if r.verification_status == VerificationStatus.PARTIALLY_SUPPORTED
        )
        not_supported = sum(
            1 for r in self._records
            if r.verification_status == VerificationStatus.NOT_SUPPORTED
        )
        not_found = sum(
            1 for r in self._records
            if r.verification_status == VerificationStatus.SOURCE_NOT_FOUND
        )

        verified = supported + partially
        coverage = verified / total if total > 0 else 0.0
        support_rate = supported / total if total > 0 else 0.0
        avg_conf = (
            sum(r.confidence for r in self._records) / total
            if total > 0 else 0.0
        )

        # Sezioni uniche
        sections = sorted(set(r.section for r in self._records))

        # Citazioni uniche
        unique_keys = set(r.citation_key for r in self._records if r.citation_key)

        # Top cited papers (per numero di claim che li citano)
        citation_counts: dict[str, int] = defaultdict(int)
        for r in self._records:
            if r.citation_key:
                citation_counts[r.citation_key] += 1
        top_cited = sorted(
            citation_counts.items(), key=lambda x: x[1], reverse=True,
        )[:10]

        return {
            "total_claims": total,
            "verified_claims": verified,
            "unverified_claims": total - verified,
            "supported": supported,
            "partially_supported": partially,
            "not_supported": not_supported,
            "source_not_found": not_found,
            "coverage_rate": round(coverage, 4),
            "support_rate": round(support_rate, 4),
            "avg_confidence": round(avg_conf, 4),
            "unique_citation_keys": len(unique_keys),
            "unique_sections": len(sections),
            "sections": sections,
            "top_cited_papers": [
                {"citation_key": k, "claim_count": v}
                for k, v in top_cited
            ],
        }

    def get_section_stats(self, section: str) -> dict[str, Any]:
        """Calcola statistiche per una singola sezione.

        Args:
            section: Chiave della sezione.

        Returns:
            Dict con statistiche della sezione.
        """
        section_records = [r for r in self._records if r.section == section]
        return _compute_section_stats(section_records)

    # ------------------------------------------------------------------
    # Lookup e query
    # ------------------------------------------------------------------

    def get_records_by_section(self, section: str) -> list[ProvenanceRecord]:
        """Recupera tutti i record di una sezione specifica.

        Args:
            section: Chiave della sezione.

        Returns:
            Lista di ProvenanceRecord per la sezione.
        """
        return [r for r in self._records if r.section == section]

    def get_records_by_citation(self, citation_key: str) -> list[ProvenanceRecord]:
        """Recupera tutti i record per una citazione specifica.

        Args:
            citation_key: Chiave BibTeX.

        Returns:
            Lista di ProvenanceRecord per la citazione.
        """
        return [r for r in self._records if r.citation_key == citation_key]

    def get_records_by_status(
        self, status: str | VerificationStatus,
    ) -> list[ProvenanceRecord]:
        """Recupera record con uno specifico stato di verifica.

        Args:
            status: Stato come stringa o enum.

        Returns:
            Lista di ProvenanceRecord con lo stato specificato.
        """
        if isinstance(status, str):
            status = _normalize_status(status)
        return [r for r in self._records if r.verification_status == status]

    def get_unsupported_claims(self) -> list[ProvenanceRecord]:
        """Recupera tutti i claim non supportati o non trovati.

        Returns:
            Record con status NOT_SUPPORTED o SOURCE_NOT_FOUND.
        """
        bad_statuses = {
            VerificationStatus.NOT_SUPPORTED,
            VerificationStatus.SOURCE_NOT_FOUND,
        }
        return [r for r in self._records if r.verification_status in bad_statuses]

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    def save_json(self, output_path: str | Path | None = None) -> Path:
        """Salva il report di provenance completo in formato JSON.

        Args:
            output_path: Percorso del file di output. Se None, usa
                il percorso da config.yaml (output.provenance_dir).

        Returns:
            Path del file salvato.
        """
        if output_path is None:
            prov_dir = Path(settings.output.provenance_dir)
            prov_dir.mkdir(parents=True, exist_ok=True)
            output_path = prov_dir / "provenance.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            f"Report provenance salvato: {output_path} "
            f"({report['total_claims']} claim, "
            f"{report['verified_claims']} verificati)"
        )
        return output_path

    def save_markdown(self, output_path: str | Path | None = None) -> Path:
        """Salva un report provenance leggibile in formato Markdown.

        Args:
            output_path: Percorso del file di output. Se None, usa
                il provenance_dir da config + provenance_report.md.

        Returns:
            Path del file salvato.
        """
        if output_path is None:
            prov_dir = Path(settings.output.provenance_dir)
            prov_dir.mkdir(parents=True, exist_ok=True)
            output_path = prov_dir / "provenance_report.md"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        md = self._generate_markdown()

        output_path.write_text(md, encoding="utf-8")
        logger.info(f"Report Markdown provenance salvato: {output_path}")
        return output_path

    @classmethod
    def load_json(cls, json_path: str | Path) -> "ProvenanceTracker":
        """Carica un tracker da un file JSON precedentemente salvato.

        Args:
            json_path: Percorso del file JSON.

        Returns:
            ProvenanceTracker con i record caricati.

        Raises:
            FileNotFoundError: Se il file non esiste.
            ValueError: Se il formato JSON non è valido.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"File provenance non trovato: {json_path}")

        data = json.loads(json_path.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            raise ValueError(f"Formato JSON non valido: atteso dict, ricevuto {type(data).__name__}")

        tracker = cls(version=data.get("survey_version", "1.0"))

        # Carica i record dalle sezioni
        for section_data in data.get("sections", []):
            for record_data in section_data.get("records", []):
                try:
                    tracker.add_record(
                        section=record_data.get("section", section_data.get("section", "")),
                        claim=record_data.get("claim", record_data.get("claim_text", "")),
                        citation_key=record_data.get("citation_key", ""),
                        chunk_id=record_data.get("source_chunk_id"),
                        source_text=record_data.get("source_text_excerpt", ""),
                        verification_status=record_data.get(
                            "verification_status",
                            record_data.get("verification", "SOURCE_NOT_FOUND"),
                        ),
                        confidence=record_data.get("confidence", 0.0),
                        reasoning=record_data.get("reasoning", ""),
                    )
                except Exception as e:
                    logger.warning(f"Errore caricamento record: {e}")

        logger.info(
            f"Tracker caricato da {json_path}: "
            f"{tracker.record_count} record, versione {tracker.version}"
        )
        return tracker

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Stampa un riepilogo formattato in console con Rich."""
        stats = self.get_stats()

        # Tabella statistiche generali
        general = [
            {"Metrica": "Claim totali", "Valore": str(stats["total_claims"])},
            {"Metrica": "Verificati (SUPPORTED + PARTIALLY)", "Valore": str(stats["verified_claims"])},
            {"Metrica": "Non verificati", "Valore": str(stats["unverified_claims"])},
            {"Metrica": "Coverage rate", "Valore": f"{stats['coverage_rate']:.1%}"},
            {"Metrica": "Support rate", "Valore": f"{stats['support_rate']:.1%}"},
            {"Metrica": "Confidenza media", "Valore": f"{stats['avg_confidence']:.2f}"},
            {"Metrica": "Citazioni uniche", "Valore": str(stats["unique_citation_keys"])},
            {"Metrica": "Sezioni", "Valore": str(stats["unique_sections"])},
        ]
        print_table("Provenance — Riepilogo", general)

        # Breakdown per status
        breakdown = [
            {"Status": "SUPPORTED", "Count": str(stats["supported"])},
            {"Status": "PARTIALLY_SUPPORTED", "Count": str(stats["partially_supported"])},
            {"Status": "NOT_SUPPORTED", "Count": str(stats["not_supported"])},
            {"Status": "SOURCE_NOT_FOUND", "Count": str(stats["source_not_found"])},
        ]
        print_table("Breakdown per Status", breakdown)

        # Top cited papers
        if stats["top_cited_papers"]:
            top = [
                {"Paper": p["citation_key"], "Claims": str(p["claim_count"])}
                for p in stats["top_cited_papers"][:5]
            ]
            print_table("Top Cited Papers", top)

        # Per-section summary
        by_section = self._group_by_section()
        if by_section:
            section_rows = []
            for section_key in sorted(by_section):
                section_stats = _compute_section_stats(by_section[section_key])
                section_rows.append({
                    "Sezione": section_key[:30],
                    "Claims": str(section_stats["total"]),
                    "Supported": str(section_stats["supported"]),
                    "Coverage": f"{section_stats['coverage_rate']:.0%}",
                })
            print_table("Provenance per Sezione", section_rows)

    # ------------------------------------------------------------------
    # Gestione record
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Rimuove tutti i record."""
        count = len(self._records)
        self._records.clear()
        logger.info(f"Provenance tracker svuotato ({count} record rimossi).")

    def merge(self, other: "ProvenanceTracker") -> int:
        """Unisce i record di un altro tracker a questo.

        I duplicati (stesso section + citation_key + claim_text) vengono
        scartati. Il record con confidence più alta viene mantenuto.

        Args:
            other: Altro ProvenanceTracker da unire.

        Returns:
            Numero di nuovi record aggiunti.
        """
        existing_keys = set()
        for r in self._records:
            key = (r.section, r.citation_key, r.claim_text[:100])
            existing_keys.add(key)

        added = 0
        for r in other._records:
            key = (r.section, r.citation_key, r.claim_text[:100])
            if key not in existing_keys:
                self._records.append(r)
                existing_keys.add(key)
                added += 1
            else:
                # Se duplicato, mantieni quello con confidence più alta
                for i, existing in enumerate(self._records):
                    ex_key = (existing.section, existing.citation_key, existing.claim_text[:100])
                    if ex_key == key and r.confidence > existing.confidence:
                        self._records[i] = r
                        break

        logger.info(
            f"Merge: {added} nuovi record aggiunti da {other.record_count} totali."
        )
        return added

    # ------------------------------------------------------------------
    # Helpers interni
    # ------------------------------------------------------------------

    def _group_by_section(self) -> dict[str, list[ProvenanceRecord]]:
        """Raggruppa i record per sezione."""
        grouped: dict[str, list[ProvenanceRecord]] = defaultdict(list)
        for r in self._records:
            grouped[r.section].append(r)
        return dict(grouped)

    def _generate_markdown(self) -> str:
        """Genera il report in formato Markdown leggibile."""
        stats = self.get_stats()
        lines: list[str] = []

        lines.append(f"# Provenance Report — Survey v{self._version}")
        lines.append("")
        lines.append(f"**Generato:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Claim totali:** {stats['total_claims']}")
        lines.append(f"**Verificati:** {stats['verified_claims']}")
        lines.append(f"**Coverage rate:** {stats['coverage_rate']:.1%}")
        lines.append(f"**Support rate:** {stats['support_rate']:.1%}")
        lines.append(f"**Confidenza media:** {stats['avg_confidence']:.2f}")
        lines.append("")

        # Breakdown per status
        lines.append("## Breakdown per Status")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| SUPPORTED | {stats['supported']} |")
        lines.append(f"| PARTIALLY_SUPPORTED | {stats['partially_supported']} |")
        lines.append(f"| NOT_SUPPORTED | {stats['not_supported']} |")
        lines.append(f"| SOURCE_NOT_FOUND | {stats['source_not_found']} |")
        lines.append("")

        # Top cited papers
        if stats["top_cited_papers"]:
            lines.append("## Top Cited Papers")
            lines.append("")
            lines.append("| Paper | Claims |")
            lines.append("|-------|--------|")
            for p in stats["top_cited_papers"]:
                lines.append(f"| {p['citation_key']} | {p['claim_count']} |")
            lines.append("")

        # Dettaglio per sezione
        by_section = self._group_by_section()
        if by_section:
            lines.append("## Dettaglio per Sezione")
            lines.append("")

            for section_key in sorted(by_section):
                section_records = by_section[section_key]
                section_stats = _compute_section_stats(section_records)

                lines.append(f"### {section_key}")
                lines.append("")
                lines.append(
                    f"Claims: {section_stats['total']} | "
                    f"Supported: {section_stats['supported']} | "
                    f"Coverage: {section_stats['coverage_rate']:.0%}"
                )
                lines.append("")

                # Tabella record
                lines.append("| Citation | Status | Confidence | Claim (troncato) |")
                lines.append("|----------|--------|------------|------------------|")
                for r in section_records:
                    claim_short = r.claim_text[:60].replace("|", "\\|")
                    if len(r.claim_text) > 60:
                        claim_short += "..."
                    lines.append(
                        f"| {r.citation_key} | {r.verification_status.value} "
                        f"| {r.confidence:.2f} | {claim_short} |"
                    )
                lines.append("")

        # Problematiche
        unsupported = self.get_unsupported_claims()
        if unsupported:
            lines.append("## ⚠ Claim Problematici")
            lines.append("")
            for r in unsupported:
                lines.append(
                    f"- **[{r.citation_key}]** in _{r.section}_: "
                    f"{r.verification_status.value} "
                    f"(confidence: {r.confidence:.2f})"
                )
                if r.claim_text:
                    lines.append(f"  > {r.claim_text[:120]}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ProvenanceTracker(version={self._version!r}, "
            f"records={len(self._records)})"
        )

    def __len__(self) -> int:
        return len(self._records)


# ============================================================================
# Helper functions (module-level)
# ============================================================================

def _normalize_status(status_str: str) -> VerificationStatus:
    """Normalizza una stringa di status in VerificationStatus enum.

    Supporta varianti con/senza underscore, case-insensitive.

    Args:
        status_str: Stringa dello status.

    Returns:
        VerificationStatus corrispondente.
    """
    mapping = {
        "supported": VerificationStatus.SUPPORTED,
        "partially_supported": VerificationStatus.PARTIALLY_SUPPORTED,
        "partially supported": VerificationStatus.PARTIALLY_SUPPORTED,
        "partial": VerificationStatus.PARTIALLY_SUPPORTED,
        "not_supported": VerificationStatus.NOT_SUPPORTED,
        "not supported": VerificationStatus.NOT_SUPPORTED,
        "unsupported": VerificationStatus.NOT_SUPPORTED,
        "source_not_found": VerificationStatus.SOURCE_NOT_FOUND,
        "source not found": VerificationStatus.SOURCE_NOT_FOUND,
        "not_found": VerificationStatus.SOURCE_NOT_FOUND,
        "not found": VerificationStatus.SOURCE_NOT_FOUND,
        "missing": VerificationStatus.SOURCE_NOT_FOUND,
    }
    normalized = status_str.strip().lower()
    return mapping.get(normalized, VerificationStatus.SOURCE_NOT_FOUND)


def _record_to_dict(record: ProvenanceRecord) -> dict[str, Any]:
    """Converte un ProvenanceRecord in un dict serializzabile JSON.

    Args:
        record: Il record da convertire.

    Returns:
        Dict con tutti i campi del record.
    """
    return {
        "section": record.section,
        "claim": record.claim_text,
        "citation_key": record.citation_key,
        "source_chunk_id": record.source_chunk_id or "",
        "source_text_excerpt": record.source_text_excerpt,
        "verification": record.verification_status.value,
        "confidence": record.confidence,
    }


def _compute_section_stats(
    records: list[ProvenanceRecord],
) -> dict[str, Any]:
    """Calcola statistiche per un insieme di record (tipicamente una sezione).

    Args:
        records: Lista di ProvenanceRecord.

    Returns:
        Dict con statistiche della sezione.
    """
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "supported": 0,
            "partially_supported": 0,
            "not_supported": 0,
            "source_not_found": 0,
            "coverage_rate": 0.0,
            "support_rate": 0.0,
            "avg_confidence": 0.0,
            "unique_citations": 0,
        }

    supported = sum(
        1 for r in records
        if r.verification_status == VerificationStatus.SUPPORTED
    )
    partially = sum(
        1 for r in records
        if r.verification_status == VerificationStatus.PARTIALLY_SUPPORTED
    )
    not_supported = sum(
        1 for r in records
        if r.verification_status == VerificationStatus.NOT_SUPPORTED
    )
    not_found = sum(
        1 for r in records
        if r.verification_status == VerificationStatus.SOURCE_NOT_FOUND
    )
    verified = supported + partially
    unique_keys = set(r.citation_key for r in records if r.citation_key)
    avg_conf = sum(r.confidence for r in records) / total

    return {
        "total": total,
        "supported": supported,
        "partially_supported": partially,
        "not_supported": not_supported,
        "source_not_found": not_found,
        "coverage_rate": round(verified / total, 4) if total > 0 else 0.0,
        "support_rate": round(supported / total, 4) if total > 0 else 0.0,
        "avg_confidence": round(avg_conf, 4),
        "unique_citations": len(unique_keys),
    }


# ============================================================================
# Convenience functions
# ============================================================================

def create_provenance_from_state(
    state: dict[str, Any],
    version: str | None = None,
) -> ProvenanceTracker:
    """Crea un ProvenanceTracker dai dati della pipeline LangGraph.

    Convenience function che inizializza il tracker e importa
    tutti i record di provenance dallo stato della pipeline.

    Args:
        state: Stato della pipeline (SurveyState).
        version: Versione del survey (opzionale, fallback da state).

    Returns:
        ProvenanceTracker con tutti i record importati.
    """
    # Tenta di recuperare la versione dallo stato
    if version is None:
        version_info = state.get("version_info", {})
        if isinstance(version_info, dict):
            version = version_info.get("version_number", "1.0")
        else:
            version = "1.0"

    tracker = ProvenanceTracker(version=version)
    tracker.import_from_state(state)
    return tracker


def load_provenance(json_path: str | Path) -> ProvenanceTracker:
    """Shortcut per caricare un tracker da file JSON.

    Args:
        json_path: Percorso del file JSON.

    Returns:
        ProvenanceTracker caricato.
    """
    return ProvenanceTracker.load_json(json_path)
