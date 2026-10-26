"""
Versioning & Changelog — Gestione versioni successive del survey.

Centralizza tutta la logica di versioning: creazione nuove versioni,
generazione changelog automatico (rule-based e LLM), salvataggio/caricamento
metadati, confronto tra versioni, gestione directory di output.

Uso standalone::

    from paperhunter.postprocessing.versioning import VersionManager

    vm = VersionManager()
    v1 = vm.create_new_version(paper_count=120, categories=["Prompt Injection", ...])
    vm.save_version_metadata(v1)

    # Dopo aggiunta di nuovi paper...
    v2 = vm.create_new_version(paper_count=135, categories=[...])
    changelog = vm.generate_changelog(v1, v2)
    v2.changelog = changelog
    vm.save_version_metadata(v2)

    # Confronto
    diff = vm.diff_versions(v1, v2)
    vm.print_summary()

Uso con SurveyState::

    from paperhunter.postprocessing.versioning import (
        create_version_from_state, load_version_history,
    )

    version = create_version_from_state(state)
    history = load_version_history()
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.models import SurveyVersion
from paperhunter.utils.logging_setup import (
    get_logger,
    print_panel,
    print_success,
    print_table,
    print_warning,
)

logger = get_logger(__name__)


# ============================================================================
# Costanti
# ============================================================================

_VERSION_DIR_PATTERN = re.compile(r"Survey_v(\d+)\.(\d+)_(\w+)")
_VERSION_INFO_FILE = "version_info.json"
_CHANGELOG_FILE = "changelog.md"
_SURVEY_FILE = "survey.md"


# ============================================================================
# VersionManager
# ============================================================================


class VersionManager:
    """Gestisce il ciclo di vita delle versioni del survey.

    Responsabilità:
    - Determinare il prossimo numero di versione
    - Creare nuove versioni con metadati completi
    - Generare changelog tra versioni (rule-based + LLM opzionale)
    - Salvare/caricare metadati versione su disco
    - Elencare e confrontare versioni precedenti

    Args:
        versions_dir: Directory radice per le versioni.
            Default: ``config.yaml`` → ``output.versions_dir``.
    """

    def __init__(self, versions_dir: str | Path | None = None) -> None:
        self._versions_dir = Path(
            versions_dir or settings.output.versions_dir
        )
        self._versions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def versions_dir(self) -> Path:
        """Directory radice per tutte le versioni."""
        return self._versions_dir

    # ------------------------------------------------------------------
    # Lettura versioni esistenti
    # ------------------------------------------------------------------

    def get_current_version(self) -> SurveyVersion | None:
        """Ritorna la versione più recente trovata su disco.

        Scansiona ``versions_dir`` per directory con pattern
        ``Survey_vX.Y_MonYYYY/``, carica ``version_info.json``
        dalla più recente.

        Returns:
            SurveyVersion più recente, o None se nessuna versione trovata.
        """
        versions = self.list_versions()
        if not versions:
            return None
        # L'ultima è la più recente (list_versions ordina per major.minor)
        return versions[-1]

    def get_version(self, version_number: str) -> SurveyVersion | None:
        """Carica una versione specifica dato il suo numero (es. "1.2").

        Args:
            version_number: Stringa versione (es. "1.0", "2.3").

        Returns:
            SurveyVersion o None se non trovata.
        """
        for item in self._versions_dir.iterdir():
            if not item.is_dir():
                continue
            match = _VERSION_DIR_PATTERN.search(item.name)
            if match:
                vnum = f"{match.group(1)}.{match.group(2)}"
                if vnum == version_number:
                    return self._load_version_from_dir(item)
        return None

    def list_versions(self) -> list[SurveyVersion]:
        """Elenca tutte le versioni trovate su disco, ordinate per numero.

        Returns:
            Lista di SurveyVersion ordinate per (major, minor).
        """
        versions: list[tuple[int, int, SurveyVersion]] = []

        if not self._versions_dir.exists():
            return []

        for item in self._versions_dir.iterdir():
            if not item.is_dir():
                continue
            match = _VERSION_DIR_PATTERN.search(item.name)
            if not match:
                continue
            major = int(match.group(1))
            minor = int(match.group(2))

            version = self._load_version_from_dir(item)
            if version:
                versions.append((major, minor, version))

        versions.sort(key=lambda x: (x[0], x[1]))
        return [v for _, _, v in versions]

    def list_version_dirs(self) -> list[dict[str, Any]]:
        """Elenca le directory versione con info strutturali.

        Returns:
            Lista di dict con: version_number, dir_name, path,
            has_survey, has_changelog, has_provenance, size_mb.
        """
        result: list[dict[str, Any]] = []

        if not self._versions_dir.exists():
            return result

        for item in sorted(self._versions_dir.iterdir()):
            if not item.is_dir():
                continue
            match = _VERSION_DIR_PATTERN.search(item.name)
            if not match:
                continue

            vnum = f"{match.group(1)}.{match.group(2)}"

            # Calcola dimensione totale
            total_size = sum(
                f.stat().st_size for f in item.rglob("*") if f.is_file()
            )

            result.append({
                "version_number": vnum,
                "dir_name": item.name,
                "path": str(item),
                "has_survey": (item / _SURVEY_FILE).exists(),
                "has_changelog": (item / _CHANGELOG_FILE).exists(),
                "has_provenance": (item / "provenance.json").exists(),
                "has_version_info": (item / _VERSION_INFO_FILE).exists(),
                "size_mb": round(total_size / (1024 * 1024), 2),
            })

        return result

    # ------------------------------------------------------------------
    # Creazione nuova versione
    # ------------------------------------------------------------------

    def create_new_version(
        self,
        *,
        paper_count: int = 0,
        categories: list[str] | None = None,
        l1_count: int = 0,
        l2_count: int = 0,
        sections_count: int = 0,
        total_citations: int = 0,
        verified_citations: int = 0,
        auto_changelog: bool = True,
    ) -> SurveyVersion:
        """Crea una nuova versione con il prossimo numero disponibile.

        Determina automaticamente il version_number, genera changelog
        se esiste una versione precedente.

        Args:
            paper_count: Numero totale di paper nel corpus.
            categories: Categorie coperte.
            l1_count: Numero di L1 summary generati.
            l2_count: Numero di L2 summary generati.
            sections_count: Numero di sezioni nel survey.
            total_citations: Numero totale di citazioni.
            verified_citations: Citazioni verificate con provenance.
            auto_changelog: Se True, genera changelog automatico.

        Returns:
            Nuova SurveyVersion con tutti i metadati.
        """
        version_number = self._compute_next_version_number()

        version = SurveyVersion(
            version_number=version_number,
            timestamp=datetime.now(),
            paper_count=paper_count,
            categories=sorted(categories) if categories else [],
            l1_count=l1_count,
            l2_count=l2_count,
            sections_count=sections_count,
            total_citations=total_citations,
            verified_citations=verified_citations,
        )

        # Genera changelog se c'è una versione precedente
        if auto_changelog:
            prev = self.get_current_version()
            if prev is not None:
                changelog = self.generate_changelog(prev, version)
                version.changelog = changelog
                logger.info(
                    f"Changelog generato: {version_number} "
                    f"(dalla v{prev.version_number})"
                )

        logger.info(
            f"Nuova versione creata: v{version_number} "
            f"({paper_count} paper, {sections_count} sezioni, "
            f"{total_citations} citazioni)"
        )
        return version

    def create_major_version(
        self,
        *,
        paper_count: int = 0,
        categories: list[str] | None = None,
        **kwargs: Any,
    ) -> SurveyVersion:
        """Crea una nuova versione major (es. 2.0, 3.0).

        Incrementa il major version e resetta il minor a 0.

        Args:
            paper_count: Numero totale paper.
            categories: Categorie coperte.
            **kwargs: Passati a SurveyVersion.

        Returns:
            Nuova SurveyVersion con major incrementato.
        """
        current = self.get_current_version()
        if current:
            major, _ = self._parse_version_number(current.version_number)
            version_number = f"{major + 1}.0"
        else:
            version_number = "1.0"

        version = SurveyVersion(
            version_number=version_number,
            timestamp=datetime.now(),
            paper_count=paper_count,
            categories=sorted(categories) if categories else [],
            **kwargs,
        )

        # Changelog
        if current:
            changelog = self.generate_changelog(current, version)
            version.changelog = changelog

        logger.info(f"Major version creata: v{version_number}")
        return version

    # ------------------------------------------------------------------
    # Changelog
    # ------------------------------------------------------------------

    def generate_changelog(
        self,
        old_version: SurveyVersion,
        new_version: SurveyVersion,
        *,
        use_llm: bool = False,
    ) -> str:
        """Genera changelog tra due versioni.

        Di default genera un changelog rule-based. Se ``use_llm=True``,
        prova a usare Gemini Flash per un changelog più dettagliato,
        con fallback automatico al rule-based.

        Args:
            old_version: Versione precedente.
            new_version: Nuova versione.
            use_llm: Se usare LLM per il changelog.

        Returns:
            Stringa Markdown del changelog.
        """
        # Se è la prima versione, niente changelog dettagliato
        if old_version.paper_count == 0 and not old_version.categories:
            return "Initial release."

        if use_llm:
            try:
                return self._generate_changelog_llm(old_version, new_version)
            except Exception as e:
                logger.warning(
                    f"Changelog LLM fallito, fallback a rule-based: {e}"
                )

        return self._generate_changelog_rule_based(old_version, new_version)

    # ------------------------------------------------------------------
    # Salvataggio / Caricamento
    # ------------------------------------------------------------------

    def save_version_metadata(
        self,
        version: SurveyVersion,
        *,
        survey_text: str | None = None,
        provenance: list[dict] | None = None,
        bibliography_keys: list[str] | None = None,
    ) -> Path:
        """Salva i metadati della versione su disco.

        Crea la struttura directory:
            output/versions/Survey_vX.Y_MonYYYY/
                ├── version_info.json
                ├── survey.md           (se fornito)
                ├── provenance.json     (se fornito)
                ├── changelog.md        (se changelog non vuoto)
                └── citation_keys.json  (se forniti)

        Args:
            version: SurveyVersion da salvare.
            survey_text: Contenuto Markdown del survey (opzionale).
            provenance: Record di provenance (opzionale).
            bibliography_keys: Chiavi BibTeX usate (opzionale).

        Returns:
            Path della directory della versione.
        """
        version_dir = self.get_version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # 1. Version info JSON
        version_dict = self._serialize_version(version)
        info_path = version_dir / _VERSION_INFO_FILE
        info_path.write_text(
            json.dumps(version_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Version info salvata: {info_path}")

        # 2. Survey Markdown
        if survey_text:
            survey_path = version_dir / _SURVEY_FILE
            survey_path.write_text(survey_text, encoding="utf-8")
            logger.info(f"Survey salvato: {survey_path}")

        # 3. Provenance JSON
        if provenance:
            prov_path = version_dir / "provenance.json"
            prov_data = {
                "version": version.version_number,
                "generated_at": datetime.now().isoformat(),
                "total_records": len(provenance),
                "records": provenance,
            }
            prov_path.write_text(
                json.dumps(prov_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                f"Provenance salvata: {prov_path} ({len(provenance)} record)"
            )

        # 4. Changelog Markdown
        if version.changelog:
            cl_path = version_dir / _CHANGELOG_FILE
            cl_path.write_text(
                f"# Changelog — Version {version.version_number}\n\n"
                f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
                f"{version.changelog}\n",
                encoding="utf-8",
            )
            logger.info(f"Changelog salvato: {cl_path}")

        # 5. Citation keys JSON
        if bibliography_keys:
            keys_path = version_dir / "citation_keys.json"
            keys_path.write_text(
                json.dumps(sorted(bibliography_keys), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                f"Citation keys salvate: {keys_path} ({len(bibliography_keys)} chiavi)"
            )

        print_success(f"Versione v{version.version_number} salvata in: {version_dir}")
        return version_dir

    def load_version_metadata(self, version_number: str) -> SurveyVersion | None:
        """Carica i metadati di una versione specifica da disco.

        Alias di ``get_version(version_number)`` per coerenza API.

        Args:
            version_number: Stringa versione (es. "1.0").

        Returns:
            SurveyVersion o None se non trovata.
        """
        return self.get_version(version_number)

    # ------------------------------------------------------------------
    # Confronto versioni
    # ------------------------------------------------------------------

    def diff_versions(
        self,
        old_version: SurveyVersion,
        new_version: SurveyVersion,
    ) -> dict[str, Any]:
        """Calcola il diff strutturato tra due versioni.

        Args:
            old_version: Versione precedente.
            new_version: Versione più recente.

        Returns:
            Dict con: paper_diff, new_categories, removed_categories,
            sections_diff, citations_diff, verified_diff, time_delta.
        """
        old_cats = set(old_version.categories)
        new_cats = set(new_version.categories)

        old_ts = old_version.timestamp
        new_ts = new_version.timestamp

        if isinstance(old_ts, datetime) and isinstance(new_ts, datetime):
            time_delta_days = (new_ts - old_ts).days
        else:
            time_delta_days = None

        return {
            "old_version": old_version.version_number,
            "new_version": new_version.version_number,
            "paper_diff": new_version.paper_count - old_version.paper_count,
            "old_paper_count": old_version.paper_count,
            "new_paper_count": new_version.paper_count,
            "new_categories": sorted(new_cats - old_cats),
            "removed_categories": sorted(old_cats - new_cats),
            "common_categories": sorted(old_cats & new_cats),
            "sections_diff": new_version.sections_count - old_version.sections_count,
            "old_sections": old_version.sections_count,
            "new_sections": new_version.sections_count,
            "citations_diff": new_version.total_citations - old_version.total_citations,
            "old_citations": old_version.total_citations,
            "new_citations": new_version.total_citations,
            "verified_diff": (
                new_version.verified_citations - old_version.verified_citations
            ),
            "l1_diff": new_version.l1_count - old_version.l1_count,
            "l2_diff": new_version.l2_count - old_version.l2_count,
            "time_delta_days": time_delta_days,
        }

    # ------------------------------------------------------------------
    # Utility directory
    # ------------------------------------------------------------------

    def get_version_dir(self, version: SurveyVersion) -> Path:
        """Calcola la directory per una versione specifica.

        Formato: ``output/versions/Survey_vX.Y_MonYYYY/``

        Args:
            version: Metadati della versione.

        Returns:
            Path della directory.
        """
        ts = version.timestamp
        if isinstance(ts, datetime):
            month_year = ts.strftime("%b%Y")
        else:
            month_year = datetime.now().strftime("%b%Y")

        dir_name = f"Survey_v{version.version_number}_{month_year}"
        return self._versions_dir / dir_name

    def delete_version(self, version_number: str) -> bool:
        """Elimina una versione e la sua directory.

        Args:
            version_number: Versione da eliminare (es. "1.2").

        Returns:
            True se eliminata, False se non trovata.
        """
        for item in self._versions_dir.iterdir():
            if not item.is_dir():
                continue
            match = _VERSION_DIR_PATTERN.search(item.name)
            if match:
                vnum = f"{match.group(1)}.{match.group(2)}"
                if vnum == version_number:
                    shutil.rmtree(item)
                    logger.info(f"Versione v{version_number} eliminata: {item}")
                    return True
        logger.warning(f"Versione v{version_number} non trovata per eliminazione")
        return False

    def clean_old_versions(self, keep_last: int = 3) -> int:
        """Elimina le versioni più vecchie, mantenendo le ultime N.

        Args:
            keep_last: Numero di versioni recenti da mantenere.

        Returns:
            Numero di versioni eliminate.
        """
        if keep_last < 1:
            raise ValueError("keep_last deve essere >= 1")

        dirs_info = self.list_version_dirs()
        if len(dirs_info) <= keep_last:
            return 0

        to_remove = dirs_info[:-keep_last]
        removed = 0
        for info in to_remove:
            try:
                shutil.rmtree(info["path"])
                logger.info(f"Rimossa versione vecchia: {info['dir_name']}")
                removed += 1
            except Exception as e:
                logger.error(f"Errore rimozione {info['dir_name']}: {e}")

        return removed

    # ------------------------------------------------------------------
    # Statistiche
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Statistiche aggregate su tutte le versioni.

        Returns:
            Dict con: total_versions, latest_version, oldest_version,
            total_size_mb, versions_list.
        """
        dirs = self.list_version_dirs()
        versions = self.list_versions()

        total_size = sum(d["size_mb"] for d in dirs)

        return {
            "total_versions": len(versions),
            "versions_dir": str(self._versions_dir),
            "latest_version": (
                versions[-1].version_number if versions else None
            ),
            "oldest_version": (
                versions[0].version_number if versions else None
            ),
            "latest_paper_count": (
                versions[-1].paper_count if versions else 0
            ),
            "total_size_mb": round(total_size, 2),
            "versions_list": [v.version_number for v in versions],
        }

    def print_summary(self) -> None:
        """Stampa un riepilogo formattato in console con Rich."""
        stats = self.get_stats()

        # Tabella generale
        summary_data = {
            "Versioni totali": str(stats["total_versions"]),
            "Directory": stats["versions_dir"],
            "Ultima versione": stats["latest_version"] or "N/A",
            "Prima versione": stats["oldest_version"] or "N/A",
            "Paper (ultima)": str(stats["latest_paper_count"]),
            "Dimensione totale": f"{stats['total_size_mb']:.2f} MB",
        }
        print_table("Version Manager — Riepilogo", summary_data)

        # Tabella per versione
        dirs = self.list_version_dirs()
        if dirs:
            detail_data = [
                {
                    "Versione": d["version_number"],
                    "Directory": d["dir_name"],
                    "Survey": "✓" if d["has_survey"] else "✗",
                    "Changelog": "✓" if d["has_changelog"] else "✗",
                    "Provenance": "✓" if d["has_provenance"] else "✗",
                    "Size (MB)": f"{d['size_mb']:.2f}",
                }
                for d in dirs
            ]
            print_table("Versioni su disco", detail_data)

    # ------------------------------------------------------------------
    # Metodi interni
    # ------------------------------------------------------------------

    def _compute_next_version_number(self) -> str:
        """Determina il prossimo numero di versione (minor increment).

        Scansiona le directory esistenti e incrementa il minor.

        Returns:
            Stringa versione (es. "1.0", "1.1", "2.0").
        """
        if not self._versions_dir.exists():
            return "1.0"

        existing: list[tuple[int, int]] = []

        for item in self._versions_dir.iterdir():
            if not item.is_dir():
                continue
            match = _VERSION_DIR_PATTERN.search(item.name)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2))
                existing.append((major, minor))

        if not existing:
            return "1.0"

        latest = max(existing)
        return f"{latest[0]}.{latest[1] + 1}"

    def _parse_version_number(
        self, version_number: str
    ) -> tuple[int, int]:
        """Parsa una stringa versione in (major, minor).

        Args:
            version_number: Stringa tipo "1.2".

        Returns:
            Tupla (major, minor).
        """
        parts = version_number.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            major, minor = 1, 0
        return major, minor

    def _load_version_from_dir(self, version_dir: Path) -> SurveyVersion | None:
        """Carica SurveyVersion dal version_info.json in una directory.

        Args:
            version_dir: Path della directory versione.

        Returns:
            SurveyVersion o None se il file non esiste / è corrotto.
        """
        info_path = version_dir / _VERSION_INFO_FILE
        if not info_path.exists():
            # Prova a ricostruire da nome directory
            match = _VERSION_DIR_PATTERN.search(version_dir.name)
            if match:
                return SurveyVersion(
                    version_number=f"{match.group(1)}.{match.group(2)}",
                )
            return None

        try:
            data = json.loads(info_path.read_text(encoding="utf-8"))
            # Gestisci timestamp stringa → datetime
            if "timestamp" in data and isinstance(data["timestamp"], str):
                try:
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                except ValueError:
                    data["timestamp"] = datetime.now()
            return SurveyVersion(**data)
        except (json.JSONDecodeError, TypeError, Exception) as e:
            logger.warning(
                f"Errore caricamento version_info da {info_path}: {e}"
            )
            return None

    def _serialize_version(self, version: SurveyVersion) -> dict[str, Any]:
        """Serializza SurveyVersion in dict JSON-safe.

        Args:
            version: SurveyVersion da serializzare.

        Returns:
            Dict con timestamp come ISO string.
        """
        data = version.model_dump()
        if "timestamp" in data and isinstance(data["timestamp"], datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data

    def _generate_changelog_rule_based(
        self,
        old_version: SurveyVersion,
        new_version: SurveyVersion,
    ) -> str:
        """Genera changelog rule-based senza LLM.

        Confronta metriche numeriche e categorie per produrre
        un changelog in formato bullet list Markdown.

        Args:
            old_version: Versione precedente.
            new_version: Nuova versione.

        Returns:
            Stringa Markdown con bullet list delle modifiche.
        """
        parts: list[str] = []

        # Paper count
        paper_diff = new_version.paper_count - old_version.paper_count
        if paper_diff > 0:
            parts.append(
                f"- Added {paper_diff} new papers "
                f"(total: {new_version.paper_count})"
            )
        elif paper_diff < 0:
            parts.append(
                f"- Removed {abs(paper_diff)} papers "
                f"(total: {new_version.paper_count})"
            )

        # Nuove categorie
        new_cats = set(new_version.categories) - set(old_version.categories)
        if new_cats:
            parts.append(
                f"- New categories: {', '.join(sorted(new_cats))}"
            )

        # Categorie rimosse
        removed_cats = set(old_version.categories) - set(new_version.categories)
        if removed_cats:
            parts.append(
                f"- Removed categories: {', '.join(sorted(removed_cats))}"
            )

        # Sezioni
        section_diff = new_version.sections_count - old_version.sections_count
        if section_diff > 0:
            parts.append(f"- Added {section_diff} new sections")
        elif section_diff < 0:
            parts.append(f"- Reduced by {abs(section_diff)} sections")

        # Citazioni
        if new_version.total_citations != old_version.total_citations:
            parts.append(
                f"- Citations updated: "
                f"{old_version.total_citations} → {new_version.total_citations}"
            )

        # Citazioni verificate
        verified_diff = (
            new_version.verified_citations - old_version.verified_citations
        )
        if verified_diff > 0:
            parts.append(
                f"- {verified_diff} additional citations verified"
            )

        # L1 / L2 summary
        l1_diff = new_version.l1_count - old_version.l1_count
        if l1_diff > 0:
            parts.append(f"- {l1_diff} new L1 summaries generated")
        l2_diff = new_version.l2_count - old_version.l2_count
        if l2_diff > 0:
            parts.append(f"- {l2_diff} new L2 meta-summaries generated")

        if not parts:
            parts.append("- Minor updates and corrections")

        return "\n".join(parts)

    def _generate_changelog_llm(
        self,
        old_version: SurveyVersion,
        new_version: SurveyVersion,
    ) -> str:
        """Genera changelog con LLM (Gemini Flash).

        Args:
            old_version: Versione precedente.
            new_version: Nuova versione.

        Returns:
            Stringa Markdown del changelog generato dall'LLM.

        Raises:
            Exception: Se la chiamata LLM fallisce.
        """
        from paperhunter.utils.llm_factory import get_llm, llm_call_with_retry

        old_ts = old_version.timestamp
        old_date = (
            old_ts.strftime("%Y-%m-%d")
            if isinstance(old_ts, datetime)
            else str(old_ts)
        )

        new_ts = new_version.timestamp
        new_date = (
            new_ts.strftime("%Y-%m-%d")
            if isinstance(new_ts, datetime)
            else str(new_ts)
        )

        prompt = (
            "Generate a concise Markdown changelog for a scientific survey paper "
            "on LLM Security. Compare the two versions:\n\n"
            f"**Previous Version:** v{old_version.version_number} ({old_date})\n"
            f"- Papers: {old_version.paper_count}\n"
            f"- Categories: {', '.join(old_version.categories) or 'N/A'}\n"
            f"- Sections: {old_version.sections_count}\n"
            f"- Citations: {old_version.total_citations}\n\n"
            f"**New Version:** v{new_version.version_number} ({new_date})\n"
            f"- Papers: {new_version.paper_count}\n"
            f"- Categories: {', '.join(new_version.categories) or 'N/A'}\n"
            f"- Sections: {new_version.sections_count}\n"
            f"- Citations: {new_version.total_citations}\n\n"
            "Write a bullet list of changes. Focus on: new papers added, "
            "new categories, structural changes, citation improvements. "
            "Be specific and concise. Output ONLY the bullet list."
        )

        llm = get_llm("flash", timeout_seconds=60)
        result = llm_call_with_retry(
            llm, prompt, max_retries=2, base_delay=5.0, max_delay=30.0
        )
        return result.strip()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"VersionManager("
            f"versions={stats['total_versions']}, "
            f"latest={stats['latest_version'] or 'N/A'}, "
            f"dir='{self._versions_dir}')"
        )

    def __len__(self) -> int:
        return len(self.list_version_dirs())


# ============================================================================
# Convenience functions
# ============================================================================


def create_version_from_state(
    state: dict[str, Any],
    *,
    versions_dir: str | Path | None = None,
    auto_changelog: bool = True,
) -> SurveyVersion:
    """Crea una nuova versione dai dati nel SurveyState.

    Estrae paper_count, categories, sections_count, total_citations
    dallo stato della pipeline e crea una SurveyVersion.

    Args:
        state: SurveyState (TypedDict) dalla pipeline LangGraph.
        versions_dir: Override directory versioni.
        auto_changelog: Se generare changelog automatico.

    Returns:
        Nuova SurveyVersion.
    """
    from paperhunter.agents.state import (
        collect_all_citation_keys,
        get_l2_summaries_from_state,
        get_outline_from_state,
        get_sections_from_state,
        get_verifications_from_state,
    )

    vm = VersionManager(versions_dir=versions_dir)

    l2_summaries = get_l2_summaries_from_state(state)
    outline = get_outline_from_state(state)
    sections = get_sections_from_state(state)
    verifications = get_verifications_from_state(state)
    all_keys = collect_all_citation_keys(state)

    # Raccogli categorie
    categories: set[str] = set()
    for o in outline:
        categories.update(o.related_categories)
    for l2 in l2_summaries:
        if l2.category:
            categories.add(l2.category)

    paper_count = sum(l2.paper_count for l2 in l2_summaries)
    verified_count = sum(1 for vr in verifications.values() if vr.passed)

    version = vm.create_new_version(
        paper_count=paper_count,
        categories=list(categories),
        l1_count=paper_count,
        l2_count=len(l2_summaries),
        sections_count=len(sections),
        total_citations=len(all_keys),
        verified_citations=verified_count,
        auto_changelog=auto_changelog,
    )

    return version


def load_version_history(
    versions_dir: str | Path | None = None,
) -> list[SurveyVersion]:
    """Carica l'intera history delle versioni da disco.

    Args:
        versions_dir: Override directory versioni.

    Returns:
        Lista ordinata di SurveyVersion (dalla più vecchia alla più recente).
    """
    vm = VersionManager(versions_dir=versions_dir)
    return vm.list_versions()


def get_latest_version(
    versions_dir: str | Path | None = None,
) -> SurveyVersion | None:
    """Ritorna la versione più recente.

    Args:
        versions_dir: Override directory versioni.

    Returns:
        SurveyVersion più recente o None.
    """
    vm = VersionManager(versions_dir=versions_dir)
    return vm.get_current_version()
