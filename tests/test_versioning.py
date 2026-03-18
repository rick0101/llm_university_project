"""
Test suite per postprocessing/versioning.py — VersionManager e convenience functions.

Copertura:
- _parse_version_number: parsing versioni
- _compute_next_version_number: logica incremento
- _serialize_version: serializzazione JSON-safe
- _load_version_from_dir: caricamento da disco
- _generate_changelog_rule_based: changelog senza LLM
- create_new_version: creazione con auto-changelog
- create_major_version: creazione major
- save_version_metadata: salvataggio completo su disco
- get_version / get_current_version / list_versions: lettura
- list_version_dirs: elenco strutturale
- diff_versions: confronto tra versioni
- delete_version / clean_old_versions: cleanup
- get_stats / print_summary: statistiche
- __repr__ / __len__: rappresentazione
- create_version_from_state: integrazione con SurveyState
- load_version_history / get_latest_version: convenience

Uso: pytest test_versioning.py -v
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from paperhunter.models import SurveyVersion
from paperhunter.postprocessing.versioning import (
    VersionManager,
    create_version_from_state,
    get_latest_version,
    load_version_history,
    _VERSION_DIR_PATTERN,
    _VERSION_INFO_FILE,
    _CHANGELOG_FILE,
    _SURVEY_FILE,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_versions_dir(tmp_path: Path) -> Path:
    """Crea una directory temporanea per le versioni."""
    versions_dir = tmp_path / "output" / "versions"
    versions_dir.mkdir(parents=True)
    return versions_dir


@pytest.fixture
def vm(tmp_versions_dir: Path) -> VersionManager:
    """Crea un VersionManager con directory temporanea."""
    return VersionManager(versions_dir=tmp_versions_dir)


@pytest.fixture
def sample_version_v1() -> SurveyVersion:
    """Versione v1.0 di esempio."""
    return SurveyVersion(
        version_number="1.0",
        timestamp=datetime(2026, 1, 15, 10, 30, 0),
        paper_count=100,
        categories=["Prompt Injection", "LLM Hallucinations", "Data and Model Poisoning"],
        l1_count=100,
        l2_count=3,
        sections_count=8,
        total_citations=150,
        verified_citations=120,
        changelog="",
    )


@pytest.fixture
def sample_version_v2() -> SurveyVersion:
    """Versione v1.1 di esempio con più paper."""
    return SurveyVersion(
        version_number="1.1",
        timestamp=datetime(2026, 2, 20, 14, 0, 0),
        paper_count=135,
        categories=[
            "Prompt Injection",
            "LLM Hallucinations",
            "Data and Model Poisoning",
            "Adversarial Attacks",
        ],
        l1_count=135,
        l2_count=4,
        sections_count=10,
        total_citations=200,
        verified_citations=170,
        changelog="",
    )


def _create_version_dir_on_disk(
    versions_dir: Path,
    version: SurveyVersion,
    *,
    include_survey: bool = False,
    include_changelog: bool = False,
    include_provenance: bool = False,
) -> Path:
    """Helper: crea una directory versione su disco con version_info.json."""
    ts = version.timestamp
    if isinstance(ts, datetime):
        month_year = ts.strftime("%b%Y")
    else:
        month_year = "Jan2026"

    dir_name = f"Survey_v{version.version_number}_{month_year}"
    version_dir = versions_dir / dir_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # Salva version_info.json
    data = version.model_dump()
    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat()
    (version_dir / _VERSION_INFO_FILE).write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )

    if include_survey:
        (version_dir / _SURVEY_FILE).write_text(
            "# Test Survey\n\nContent here.", encoding="utf-8"
        )

    if include_changelog:
        (version_dir / _CHANGELOG_FILE).write_text(
            f"# Changelog\n\n{version.changelog or 'No changes.'}\n",
            encoding="utf-8",
        )

    if include_provenance:
        (version_dir / "provenance.json").write_text(
            json.dumps({"records": []}, indent=2), encoding="utf-8"
        )

    return version_dir


# ============================================================================
# Test: _parse_version_number
# ============================================================================


class TestParseVersionNumber:
    """Test per il parsing di stringhe versione."""

    def test_standard(self, vm: VersionManager) -> None:
        assert vm._parse_version_number("1.0") == (1, 0)

    def test_minor(self, vm: VersionManager) -> None:
        assert vm._parse_version_number("1.3") == (1, 3)

    def test_major(self, vm: VersionManager) -> None:
        assert vm._parse_version_number("2.0") == (2, 0)

    def test_high_numbers(self, vm: VersionManager) -> None:
        assert vm._parse_version_number("10.25") == (10, 25)

    def test_no_minor(self, vm: VersionManager) -> None:
        assert vm._parse_version_number("3") == (3, 0)

    def test_invalid(self, vm: VersionManager) -> None:
        assert vm._parse_version_number("abc") == (1, 0)

    def test_empty(self, vm: VersionManager) -> None:
        assert vm._parse_version_number("") == (1, 0)


# ============================================================================
# Test: _compute_next_version_number
# ============================================================================


class TestComputeNextVersion:
    """Test per il calcolo del prossimo numero versione."""

    def test_first_version(self, vm: VersionManager) -> None:
        assert vm._compute_next_version_number() == "1.0"

    def test_increment_minor(
        self, vm: VersionManager, tmp_versions_dir: Path
    ) -> None:
        (tmp_versions_dir / "Survey_v1.0_Jan2026").mkdir()
        assert vm._compute_next_version_number() == "1.1"

    def test_increment_from_multiple(
        self, vm: VersionManager, tmp_versions_dir: Path
    ) -> None:
        (tmp_versions_dir / "Survey_v1.0_Jan2026").mkdir()
        (tmp_versions_dir / "Survey_v1.1_Feb2026").mkdir()
        (tmp_versions_dir / "Survey_v1.2_Feb2026").mkdir()
        assert vm._compute_next_version_number() == "1.3"

    def test_ignores_non_matching_dirs(
        self, vm: VersionManager, tmp_versions_dir: Path
    ) -> None:
        (tmp_versions_dir / "random_folder").mkdir()
        (tmp_versions_dir / "notes.txt").write_text("test")
        assert vm._compute_next_version_number() == "1.0"

    def test_mixed_dirs(
        self, vm: VersionManager, tmp_versions_dir: Path
    ) -> None:
        (tmp_versions_dir / "Survey_v1.0_Jan2026").mkdir()
        (tmp_versions_dir / "random_folder").mkdir()
        assert vm._compute_next_version_number() == "1.1"

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        vm = VersionManager(versions_dir=tmp_path / "does_not_exist")
        # Il costruttore crea la directory, ma tentiamo con directory vuota
        assert vm._compute_next_version_number() == "1.0"


# ============================================================================
# Test: _serialize_version
# ============================================================================


class TestSerializeVersion:
    """Test per la serializzazione JSON-safe."""

    def test_datetime_converted(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        data = vm._serialize_version(sample_version_v1)
        assert isinstance(data["timestamp"], str)
        assert "2026-01-15" in data["timestamp"]

    def test_all_fields_present(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        data = vm._serialize_version(sample_version_v1)
        assert data["version_number"] == "1.0"
        assert data["paper_count"] == 100
        assert data["categories"] == [
            "Prompt Injection",
            "LLM Hallucinations",
            "Data and Model Poisoning",
        ]
        assert data["l1_count"] == 100
        assert data["l2_count"] == 3

    def test_json_serializable(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        data = vm._serialize_version(sample_version_v1)
        # Non deve sollevare eccezioni
        json_str = json.dumps(data)
        assert len(json_str) > 0


# ============================================================================
# Test: _load_version_from_dir
# ============================================================================


class TestLoadVersionFromDir:
    """Test per il caricamento da directory."""

    def test_load_valid(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        vdir = _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        loaded = vm._load_version_from_dir(vdir)
        assert loaded is not None
        assert loaded.version_number == "1.0"
        assert loaded.paper_count == 100

    def test_load_no_json(
        self, vm: VersionManager, tmp_versions_dir: Path
    ) -> None:
        # Directory con pattern ma senza version_info.json
        vdir = tmp_versions_dir / "Survey_v2.3_Mar2026"
        vdir.mkdir()
        loaded = vm._load_version_from_dir(vdir)
        assert loaded is not None
        assert loaded.version_number == "2.3"

    def test_load_corrupted_json(
        self, vm: VersionManager, tmp_versions_dir: Path
    ) -> None:
        vdir = tmp_versions_dir / "Survey_v1.0_Jan2026"
        vdir.mkdir()
        (vdir / _VERSION_INFO_FILE).write_text("not json{{{")
        loaded = vm._load_version_from_dir(vdir)
        assert loaded is None

    def test_load_no_pattern(
        self, vm: VersionManager, tmp_versions_dir: Path
    ) -> None:
        vdir = tmp_versions_dir / "random_dir"
        vdir.mkdir()
        loaded = vm._load_version_from_dir(vdir)
        assert loaded is None


# ============================================================================
# Test: _generate_changelog_rule_based
# ============================================================================


class TestGenerateChangelogRuleBased:
    """Test per il changelog rule-based."""

    def test_papers_added(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        cl = vm._generate_changelog_rule_based(sample_version_v1, sample_version_v2)
        assert "Added 35 new papers" in cl
        assert "total: 135" in cl

    def test_papers_removed(self, vm: VersionManager) -> None:
        old = SurveyVersion(version_number="1.0", paper_count=100, categories=[])
        new = SurveyVersion(version_number="1.1", paper_count=90, categories=[])
        cl = vm._generate_changelog_rule_based(old, new)
        assert "Removed 10 papers" in cl

    def test_new_categories(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        cl = vm._generate_changelog_rule_based(sample_version_v1, sample_version_v2)
        assert "New categories: Adversarial Attacks" in cl

    def test_removed_categories(self, vm: VersionManager) -> None:
        old = SurveyVersion(
            version_number="1.0",
            paper_count=100,
            categories=["A", "B", "C"],
        )
        new = SurveyVersion(
            version_number="1.1",
            paper_count=100,
            categories=["A", "B"],
        )
        cl = vm._generate_changelog_rule_based(old, new)
        assert "Removed categories: C" in cl

    def test_sections_added(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        cl = vm._generate_changelog_rule_based(sample_version_v1, sample_version_v2)
        assert "Added 2 new sections" in cl

    def test_sections_reduced(self, vm: VersionManager) -> None:
        old = SurveyVersion(
            version_number="1.0", paper_count=0, categories=[], sections_count=10
        )
        new = SurveyVersion(
            version_number="1.1", paper_count=0, categories=[], sections_count=8
        )
        cl = vm._generate_changelog_rule_based(old, new)
        assert "Reduced by 2 sections" in cl

    def test_citations_updated(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        cl = vm._generate_changelog_rule_based(sample_version_v1, sample_version_v2)
        assert "Citations updated: 150 → 200" in cl

    def test_verified_added(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        cl = vm._generate_changelog_rule_based(sample_version_v1, sample_version_v2)
        assert "50 additional citations verified" in cl

    def test_l1_added(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        cl = vm._generate_changelog_rule_based(sample_version_v1, sample_version_v2)
        assert "35 new L1 summaries" in cl

    def test_l2_added(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        cl = vm._generate_changelog_rule_based(sample_version_v1, sample_version_v2)
        assert "1 new L2 meta-summaries" in cl

    def test_no_changes(self, vm: VersionManager) -> None:
        v = SurveyVersion(version_number="1.0", paper_count=50, categories=["A"])
        cl = vm._generate_changelog_rule_based(v, v)
        assert "Minor updates" in cl

    def test_all_changes_combined(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        cl = vm._generate_changelog_rule_based(sample_version_v1, sample_version_v2)
        lines = cl.strip().split("\n")
        # Dovrebbe avere multiple bullet points
        assert len(lines) >= 5
        for line in lines:
            assert line.startswith("- ")


# ============================================================================
# Test: generate_changelog (dispatcher)
# ============================================================================


class TestGenerateChangelog:
    """Test per il dispatcher changelog."""

    def test_first_version_no_changelog(self, vm: VersionManager) -> None:
        old = SurveyVersion(version_number="0.0", paper_count=0, categories=[])
        new = SurveyVersion(version_number="1.0", paper_count=100, categories=["A"])
        cl = vm.generate_changelog(old, new)
        assert cl == "Initial release."

    def test_rule_based_default(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        cl = vm.generate_changelog(sample_version_v1, sample_version_v2)
        assert "Added 35 new papers" in cl

    def test_llm_fallback_to_rule_based(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        # use_llm=True ma senza API key configurata → fallback
        cl = vm.generate_changelog(
            sample_version_v1, sample_version_v2, use_llm=True
        )
        # Dovrebbe funzionare con fallback
        assert len(cl) > 0


# ============================================================================
# Test: create_new_version
# ============================================================================


class TestCreateNewVersion:
    """Test per la creazione di nuove versioni."""

    def test_first_version(self, vm: VersionManager) -> None:
        v = vm.create_new_version(
            paper_count=120,
            categories=["Prompt Injection", "LLM Hallucinations"],
        )
        assert v.version_number == "1.0"
        assert v.paper_count == 120
        assert len(v.categories) == 2
        assert isinstance(v.timestamp, datetime)

    def test_second_version_with_changelog(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        # Crea la v1.0 su disco
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)

        # Crea v1.1
        v = vm.create_new_version(
            paper_count=135,
            categories=[
                "Prompt Injection",
                "LLM Hallucinations",
                "Data and Model Poisoning",
                "Adversarial Attacks",
            ],
            sections_count=10,
            total_citations=200,
        )
        assert v.version_number == "1.1"
        assert v.paper_count == 135
        assert len(v.changelog) > 0
        assert "35 new papers" in v.changelog

    def test_no_auto_changelog(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        v = vm.create_new_version(
            paper_count=135,
            categories=["A"],
            auto_changelog=False,
        )
        assert v.changelog == ""

    def test_categories_sorted(self, vm: VersionManager) -> None:
        v = vm.create_new_version(
            paper_count=10,
            categories=["Zebra", "Alpha", "Middle"],
        )
        assert v.categories == ["Alpha", "Middle", "Zebra"]

    def test_summary_fields(self, vm: VersionManager) -> None:
        v = vm.create_new_version(
            paper_count=50,
            categories=["A"],
            l1_count=50,
            l2_count=1,
            sections_count=5,
            total_citations=80,
            verified_citations=70,
        )
        assert v.l1_count == 50
        assert v.l2_count == 1
        assert v.sections_count == 5
        assert v.total_citations == 80
        assert v.verified_citations == 70


# ============================================================================
# Test: create_major_version
# ============================================================================


class TestCreateMajorVersion:
    """Test per le major version."""

    def test_first_major(self, vm: VersionManager) -> None:
        v = vm.create_major_version(paper_count=100, categories=["A"])
        assert v.version_number == "1.0"

    def test_increment_major(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        v = vm.create_major_version(paper_count=200, categories=["A", "B"])
        assert v.version_number == "2.0"

    def test_major_with_changelog(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        v = vm.create_major_version(
            paper_count=200,
            categories=["A", "B"],
            sections_count=15,
        )
        assert len(v.changelog) > 0


# ============================================================================
# Test: save_version_metadata
# ============================================================================


class TestSaveVersionMetadata:
    """Test per il salvataggio su disco."""

    def test_save_basic(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        vdir = vm.save_version_metadata(sample_version_v1)
        assert vdir.exists()
        assert (vdir / _VERSION_INFO_FILE).exists()

        # Verifica contenuto
        data = json.loads((vdir / _VERSION_INFO_FILE).read_text())
        assert data["version_number"] == "1.0"
        assert data["paper_count"] == 100

    def test_save_with_survey(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        vdir = vm.save_version_metadata(
            sample_version_v1,
            survey_text="# My Survey\n\nContent here.",
        )
        assert (vdir / _SURVEY_FILE).exists()
        assert "My Survey" in (vdir / _SURVEY_FILE).read_text()

    def test_save_with_provenance(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        provenance = [
            {"section": "intro", "claim": "test claim", "citation_key": "K1"},
            {"section": "intro", "claim": "test claim 2", "citation_key": "K2"},
        ]
        vdir = vm.save_version_metadata(
            sample_version_v1, provenance=provenance
        )
        assert (vdir / "provenance.json").exists()
        prov_data = json.loads((vdir / "provenance.json").read_text())
        assert prov_data["total_records"] == 2

    def test_save_with_changelog(self, vm: VersionManager) -> None:
        v = SurveyVersion(
            version_number="1.1",
            paper_count=50,
            categories=["A"],
            changelog="- Added 10 papers\n- New category B",
        )
        vdir = vm.save_version_metadata(v)
        assert (vdir / _CHANGELOG_FILE).exists()
        content = (vdir / _CHANGELOG_FILE).read_text()
        assert "Added 10 papers" in content

    def test_save_with_bib_keys(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        vdir = vm.save_version_metadata(
            sample_version_v1,
            bibliography_keys=["Zou_2023_GCG", "Wang_2024_X", "Li_2025_Y"],
        )
        keys_path = vdir / "citation_keys.json"
        assert keys_path.exists()
        keys = json.loads(keys_path.read_text())
        assert len(keys) == 3
        assert keys[0] == "Li_2025_Y"  # Sorted

    def test_save_all_artifacts(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        sample_version_v1.changelog = "Some changes"
        vdir = vm.save_version_metadata(
            sample_version_v1,
            survey_text="# Survey",
            provenance=[{"claim": "x"}],
            bibliography_keys=["K1"],
        )
        assert (vdir / _VERSION_INFO_FILE).exists()
        assert (vdir / _SURVEY_FILE).exists()
        assert (vdir / "provenance.json").exists()
        assert (vdir / _CHANGELOG_FILE).exists()
        assert (vdir / "citation_keys.json").exists()

    def test_save_idempotent(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        """Salvare due volte la stessa versione sovrascrive senza errori."""
        vdir1 = vm.save_version_metadata(sample_version_v1, survey_text="v1")
        vdir2 = vm.save_version_metadata(sample_version_v1, survey_text="v2")
        assert vdir1 == vdir2
        assert (vdir2 / _SURVEY_FILE).read_text() == "v2"


# ============================================================================
# Test: get_version / get_current_version / list_versions
# ============================================================================


class TestReadVersions:
    """Test per la lettura di versioni da disco."""

    def test_get_current_empty(self, vm: VersionManager) -> None:
        assert vm.get_current_version() is None

    def test_get_current_single(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        current = vm.get_current_version()
        assert current is not None
        assert current.version_number == "1.0"

    def test_get_current_latest(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v2)
        current = vm.get_current_version()
        assert current is not None
        assert current.version_number == "1.1"

    def test_get_version_specific(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v2)
        v = vm.get_version("1.0")
        assert v is not None
        assert v.paper_count == 100

    def test_get_version_not_found(self, vm: VersionManager) -> None:
        assert vm.get_version("9.9") is None

    def test_list_versions_empty(self, vm: VersionManager) -> None:
        assert vm.list_versions() == []

    def test_list_versions_ordered(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        # Inserisci in ordine inverso
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v2)
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        versions = vm.list_versions()
        assert len(versions) == 2
        assert versions[0].version_number == "1.0"
        assert versions[1].version_number == "1.1"

    def test_list_version_dirs_info(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(
            tmp_versions_dir,
            sample_version_v1,
            include_survey=True,
            include_changelog=True,
            include_provenance=True,
        )
        dirs = vm.list_version_dirs()
        assert len(dirs) == 1
        info = dirs[0]
        assert info["version_number"] == "1.0"
        assert info["has_survey"] is True
        assert info["has_changelog"] is True
        assert info["has_provenance"] is True
        assert info["has_version_info"] is True
        assert info["size_mb"] >= 0

    def test_list_version_dirs_empty(self, vm: VersionManager) -> None:
        assert vm.list_version_dirs() == []


# ============================================================================
# Test: diff_versions
# ============================================================================


class TestDiffVersions:
    """Test per il confronto tra versioni."""

    def test_diff_basic(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        diff = vm.diff_versions(sample_version_v1, sample_version_v2)
        assert diff["old_version"] == "1.0"
        assert diff["new_version"] == "1.1"
        assert diff["paper_diff"] == 35
        assert diff["new_categories"] == ["Adversarial Attacks"]
        assert diff["removed_categories"] == []
        assert diff["sections_diff"] == 2
        assert diff["citations_diff"] == 50
        assert diff["verified_diff"] == 50
        assert diff["l1_diff"] == 35
        assert diff["l2_diff"] == 1

    def test_diff_time_delta(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        diff = vm.diff_versions(sample_version_v1, sample_version_v2)
        assert diff["time_delta_days"] == 36  # Jan 15 → Feb 20

    def test_diff_same_version(
        self,
        vm: VersionManager,
        sample_version_v1: SurveyVersion,
    ) -> None:
        diff = vm.diff_versions(sample_version_v1, sample_version_v1)
        assert diff["paper_diff"] == 0
        assert diff["sections_diff"] == 0
        assert diff["new_categories"] == []
        assert diff["removed_categories"] == []

    def test_diff_negative(self, vm: VersionManager) -> None:
        old = SurveyVersion(
            version_number="1.0", paper_count=100, categories=["A", "B"],
            sections_count=10, total_citations=200,
        )
        new = SurveyVersion(
            version_number="1.1", paper_count=80, categories=["A"],
            sections_count=8, total_citations=150,
        )
        diff = vm.diff_versions(old, new)
        assert diff["paper_diff"] == -20
        assert diff["removed_categories"] == ["B"]
        assert diff["sections_diff"] == -2
        assert diff["citations_diff"] == -50


# ============================================================================
# Test: delete_version / clean_old_versions
# ============================================================================


class TestDeleteAndClean:
    """Test per eliminazione e pulizia versioni."""

    def test_delete_existing(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        assert vm.delete_version("1.0") is True
        assert vm.list_versions() == []

    def test_delete_nonexistent(self, vm: VersionManager) -> None:
        assert vm.delete_version("9.9") is False

    def test_clean_old_versions(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
    ) -> None:
        # Crea 5 versioni
        for i in range(5):
            v = SurveyVersion(
                version_number=f"1.{i}",
                timestamp=datetime(2026, 1, i + 1),
                paper_count=100 + i * 10,
                categories=["A"],
            )
            _create_version_dir_on_disk(tmp_versions_dir, v)

        assert len(vm.list_version_dirs()) == 5

        # Mantieni solo le ultime 2
        removed = vm.clean_old_versions(keep_last=2)
        assert removed == 3
        remaining = vm.list_version_dirs()
        assert len(remaining) == 2
        assert remaining[0]["version_number"] == "1.3"
        assert remaining[1]["version_number"] == "1.4"

    def test_clean_fewer_than_keep(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        removed = vm.clean_old_versions(keep_last=5)
        assert removed == 0

    def test_clean_invalid_keep(self, vm: VersionManager) -> None:
        with pytest.raises(ValueError, match="keep_last"):
            vm.clean_old_versions(keep_last=0)


# ============================================================================
# Test: get_stats
# ============================================================================


class TestGetStats:
    """Test per le statistiche aggregate."""

    def test_stats_empty(self, vm: VersionManager) -> None:
        stats = vm.get_stats()
        assert stats["total_versions"] == 0
        assert stats["latest_version"] is None
        assert stats["oldest_version"] is None
        assert stats["total_size_mb"] == 0

    def test_stats_with_versions(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(
            tmp_versions_dir, sample_version_v1, include_survey=True
        )
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v2)
        stats = vm.get_stats()
        assert stats["total_versions"] == 2
        assert stats["latest_version"] == "1.1"
        assert stats["oldest_version"] == "1.0"
        assert stats["latest_paper_count"] == 135
        assert stats["total_size_mb"] >= 0
        assert "1.0" in stats["versions_list"]
        assert "1.1" in stats["versions_list"]


# ============================================================================
# Test: get_version_dir
# ============================================================================


class TestGetVersionDir:
    """Test per il calcolo della directory versione."""

    def test_dir_format(
        self, vm: VersionManager, sample_version_v1: SurveyVersion
    ) -> None:
        vdir = vm.get_version_dir(sample_version_v1)
        assert "Survey_v1.0_Jan2026" in str(vdir)

    def test_dir_format_feb(
        self, vm: VersionManager, sample_version_v2: SurveyVersion
    ) -> None:
        vdir = vm.get_version_dir(sample_version_v2)
        assert "Survey_v1.1_Feb2026" in str(vdir)


# ============================================================================
# Test: __repr__ / __len__
# ============================================================================


class TestReprLen:
    """Test per rappresentazione e lunghezza."""

    def test_repr_empty(self, vm: VersionManager) -> None:
        r = repr(vm)
        assert "VersionManager" in r
        assert "versions=0" in r
        assert "latest=N/A" in r

    def test_repr_with_versions(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        r = repr(vm)
        assert "versions=1" in r
        assert "latest=1.0" in r

    def test_len_empty(self, vm: VersionManager) -> None:
        assert len(vm) == 0

    def test_len_with_versions(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v2)
        assert len(vm) == 2


# ============================================================================
# Test: print_summary (smoke test)
# ============================================================================


class TestPrintSummary:
    """Smoke test per la stampa Rich."""

    def test_print_empty(self, vm: VersionManager) -> None:
        # Non deve crashare
        vm.print_summary()

    def test_print_with_data(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(
            tmp_versions_dir,
            sample_version_v1,
            include_survey=True,
            include_changelog=True,
        )
        # Non deve crashare
        vm.print_summary()


# ============================================================================
# Test: load_version_metadata (alias)
# ============================================================================


class TestLoadVersionMetadata:
    """Test per l'alias di get_version."""

    def test_load_existing(
        self,
        vm: VersionManager,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        loaded = vm.load_version_metadata("1.0")
        assert loaded is not None
        assert loaded.paper_count == 100

    def test_load_nonexistent(self, vm: VersionManager) -> None:
        assert vm.load_version_metadata("9.9") is None


# ============================================================================
# Test: Convenience functions
# ============================================================================


class TestConvenienceFunctions:
    """Test per create_version_from_state, load_version_history, get_latest_version."""

    def test_load_version_history_empty(self, tmp_versions_dir: Path) -> None:
        history = load_version_history(versions_dir=tmp_versions_dir)
        assert history == []

    def test_load_version_history(
        self,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v2)
        history = load_version_history(versions_dir=tmp_versions_dir)
        assert len(history) == 2
        assert history[0].version_number == "1.0"
        assert history[1].version_number == "1.1"

    def test_get_latest_version_empty(self, tmp_versions_dir: Path) -> None:
        assert get_latest_version(versions_dir=tmp_versions_dir) is None

    def test_get_latest_version(
        self,
        tmp_versions_dir: Path,
        sample_version_v1: SurveyVersion,
        sample_version_v2: SurveyVersion,
    ) -> None:
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v1)
        _create_version_dir_on_disk(tmp_versions_dir, sample_version_v2)
        latest = get_latest_version(versions_dir=tmp_versions_dir)
        assert latest is not None
        assert latest.version_number == "1.1"

    def test_create_version_from_state(
        self, tmp_versions_dir: Path
    ) -> None:
        """Test create_version_from_state con un mock state minimale."""
        from paperhunter.agents.state import create_initial_state
        from paperhunter.models import L2Summary

        l2_summaries = [
            L2Summary(
                category="Prompt Injection",
                paper_ids=["p1", "p2", "p3"],
                paper_count=3,
            ),
            L2Summary(
                category="LLM Hallucinations",
                paper_ids=["p4", "p5"],
                paper_count=2,
            ),
        ]

        state = create_initial_state(l2_summaries=l2_summaries)
        version = create_version_from_state(
            state, versions_dir=tmp_versions_dir
        )
        assert version.version_number == "1.0"
        assert version.paper_count == 5
        assert "LLM Hallucinations" in version.categories
        assert "Prompt Injection" in version.categories
        assert version.l2_count == 2

    def test_create_version_from_empty_state(
        self, tmp_versions_dir: Path
    ) -> None:
        from paperhunter.agents.state import create_initial_state

        state = create_initial_state()
        version = create_version_from_state(
            state, versions_dir=tmp_versions_dir
        )
        assert version.version_number == "1.0"
        assert version.paper_count == 0
        assert version.categories == []


# ============================================================================
# Test: Integrazione end-to-end
# ============================================================================


class TestEndToEnd:
    """Test di integrazione completa."""

    def test_full_workflow(self, tmp_versions_dir: Path) -> None:
        """Workflow completo: v1.0 → v1.1 → diff → clean."""
        vm = VersionManager(versions_dir=tmp_versions_dir)

        # Crea v1.0
        v1 = vm.create_new_version(
            paper_count=100,
            categories=["Prompt Injection", "LLM Hallucinations"],
            sections_count=8,
            total_citations=150,
            verified_citations=120,
        )
        assert v1.version_number == "1.0"
        assert v1.changelog == ""  # Prima versione, nessun changelog

        # Salva v1.0
        vdir1 = vm.save_version_metadata(
            v1,
            survey_text="# Survey v1.0\n\nPrima versione.",
            bibliography_keys=["Zou_2023_GCG", "Wang_2024_X"],
        )
        assert vdir1.exists()

        # Crea v1.1 (con auto-changelog)
        v2 = vm.create_new_version(
            paper_count=135,
            categories=[
                "Prompt Injection",
                "LLM Hallucinations",
                "Adversarial Attacks",
            ],
            sections_count=10,
            total_citations=200,
            verified_citations=170,
        )
        assert v2.version_number == "1.1"
        assert len(v2.changelog) > 0
        assert "35 new papers" in v2.changelog
        assert "Adversarial Attacks" in v2.changelog

        # Salva v1.1
        vdir2 = vm.save_version_metadata(
            v2,
            survey_text="# Survey v1.1\n\nSeconda versione.",
            provenance=[{"claim": "test", "key": "K1"}],
        )
        assert vdir2.exists()
        assert (vdir2 / _CHANGELOG_FILE).exists()

        # Verifica elenco
        versions = vm.list_versions()
        assert len(versions) == 2
        assert versions[0].version_number == "1.0"
        assert versions[1].version_number == "1.1"

        # Diff
        diff = vm.diff_versions(v1, v2)
        assert diff["paper_diff"] == 35
        assert diff["new_categories"] == ["Adversarial Attacks"]

        # Stats
        stats = vm.get_stats()
        assert stats["total_versions"] == 2
        assert stats["latest_version"] == "1.1"

        # Repr
        r = repr(vm)
        assert "versions=2" in r

    def test_version_round_trip(self, tmp_versions_dir: Path) -> None:
        """Salva e ricarica una versione, verifica integrità."""
        vm = VersionManager(versions_dir=tmp_versions_dir)

        original = SurveyVersion(
            version_number="3.5",
            timestamp=datetime(2026, 6, 15, 12, 0, 0),
            paper_count=400,
            categories=["A", "B", "C", "D"],
            changelog="- Major update",
            l1_count=400,
            l2_count=4,
            sections_count=12,
            total_citations=500,
            verified_citations=450,
        )

        vm.save_version_metadata(original, survey_text="# Test")
        loaded = vm.get_version("3.5")

        assert loaded is not None
        assert loaded.version_number == "3.5"
        assert loaded.paper_count == 400
        assert loaded.categories == ["A", "B", "C", "D"]
        assert loaded.changelog == "- Major update"
        assert loaded.l1_count == 400
        assert loaded.sections_count == 12
        assert loaded.total_citations == 500
        assert loaded.verified_citations == 450


# ============================================================================
# Test: Package imports
# ============================================================================


class TestPackageImports:
    """Verifica che gli import dal package funzionino."""

    def test_import_from_module(self) -> None:
        from paperhunter.postprocessing.versioning import (
            VersionManager,
            create_version_from_state,
            load_version_history,
            get_latest_version,
        )
        assert VersionManager is not None
        assert create_version_from_state is not None
        assert load_version_history is not None
        assert get_latest_version is not None

    def test_import_from_package(self) -> None:
        from paperhunter.postprocessing import (
            VersionManager,
            create_version_from_state,
            load_version_history,
            get_latest_version,
        )
        assert VersionManager is not None
