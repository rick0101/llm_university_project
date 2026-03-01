"""Test per postprocessing/bibliography_generator.py — Generazione .bib finale.

Tutti test offline, nessuna chiamata API necessaria.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from paperhunter.postprocessing.bibliography_generator import (
    BibliographyGenerator,
    _count_citations_in_text,
    _extract_bib_key,
    _extract_bib_type,
    _extract_field,
    _extract_year_from_entry,
    _key_similarity,
    _reformat_entry,
    generate_bib_from_state,
    generate_final_bib,
    load_and_filter_bib,
    merge_bib_files,
)


# ============================================================================
# Fixture: entry BibTeX di esempio
# ============================================================================

ENTRY_ZOU = """\
@inproceedings{Zou_2023_GCG,
  title = {Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author = {Andy Zou and Zifan Wang and Nicholas Carlini and Matt Fredrikson and J. Zico Kolter},
  year = {2023},
  booktitle = {NeurIPS}
}"""

ENTRY_LI = """\
@article{Li_2023_Halueval,
  title = {HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models},
  author = {Junyi Li and Xiaoxue Cheng and Wayne Xin Zhao and Jian-Yun Nie and Ji-Rong Wen},
  year = {2023},
  journal = {EMNLP}
}"""

ENTRY_GRESHAKE = """\
@inproceedings{Greshake_2023_LLM,
  title = {Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection},
  author = {Kai Greshake and Sahar Abdelnabi and Shailesh Mishra and Christoph Endres and Thorsten Holz and Mario Fritz},
  year = {2023},
  booktitle = {AISec@CCS},
  doi = {10.1145/3605764.3623985}
}"""

ENTRY_WAN = """\
@inproceedings{Wan_2023_Poisoning,
  title = {Poisoning Language Models During Instruction Tuning},
  author = {Alexander Wan and Eric Wallace and Sheng Shen and Dan Klein},
  year = {2023},
  booktitle = {ICML}
}"""

ENTRY_ZHANG = """\
@misc{Zhang_2024_AI,
  title = {Siren Song in the AI Ocean: A Survey on Hallucination in Large Language Models},
  author = {Yue Zhang and Yafu Li and Leyang Cui and Deng Cai and Lemao Liu and others},
  year = {2024}
}"""

ENTRY_INVALID_NO_AUTHOR = """\
@article{NoAuthor_2023_Bad,
  title = {This entry has no author},
  year = {2023},
  journal = {Unknown}
}"""

ENTRY_INVALID_NO_YEAR = """\
@article{NoYear_Bad,
  title = {This entry has no year},
  author = {John Doe}
}"""


def _make_master_bib() -> dict[str, str]:
    """Crea un master .bib di test con 5 entry."""
    return {
        "Zou_2023_GCG": ENTRY_ZOU,
        "Li_2023_Halueval": ENTRY_LI,
        "Greshake_2023_LLM": ENTRY_GRESHAKE,
        "Wan_2023_Poisoning": ENTRY_WAN,
        "Zhang_2024_AI": ENTRY_ZHANG,
    }


def _write_master_bib(path: Path) -> None:
    """Scrive un master .bib di test su file."""
    entries = _make_master_bib()
    header = (
        "% Master BibTeX file\n"
        f"% Total: {len(entries)}\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for entry in entries.values():
            f.write(entry + "\n\n")


# ============================================================================
# Test utility interne
# ============================================================================

class TestExtractBibKey:
    """Test _extract_bib_key."""

    def test_normal(self):
        assert _extract_bib_key(ENTRY_ZOU) == "Zou_2023_GCG"

    def test_article(self):
        assert _extract_bib_key(ENTRY_LI) == "Li_2023_Halueval"

    def test_misc(self):
        assert _extract_bib_key(ENTRY_ZHANG) == "Zhang_2024_AI"

    def test_empty(self):
        assert _extract_bib_key("") is None

    def test_no_match(self):
        assert _extract_bib_key("some random text") is None


class TestExtractBibType:
    """Test _extract_bib_type."""

    def test_inproceedings(self):
        assert _extract_bib_type(ENTRY_ZOU) == "inproceedings"

    def test_article(self):
        assert _extract_bib_type(ENTRY_LI) == "article"

    def test_misc(self):
        assert _extract_bib_type(ENTRY_ZHANG) == "misc"

    def test_empty(self):
        assert _extract_bib_type("") is None


class TestExtractField:
    """Test _extract_field."""

    def test_title(self):
        title = _extract_field(ENTRY_ZOU, "title")
        assert title is not None
        assert "Universal and Transferable" in title

    def test_author(self):
        author = _extract_field(ENTRY_ZOU, "author")
        assert author is not None
        assert "Andy Zou" in author

    def test_year(self):
        year = _extract_field(ENTRY_ZOU, "year")
        assert year == "2023"

    def test_booktitle(self):
        bt = _extract_field(ENTRY_ZOU, "booktitle")
        assert bt == "NeurIPS"

    def test_doi(self):
        doi = _extract_field(ENTRY_GRESHAKE, "doi")
        assert doi is not None
        assert "10.1145" in doi

    def test_missing_field(self):
        assert _extract_field(ENTRY_ZOU, "doi") is None

    def test_journal(self):
        journal = _extract_field(ENTRY_LI, "journal")
        assert journal == "EMNLP"

    def test_case_insensitive(self):
        """Il field name è case-insensitive nel pattern regex."""
        entry = "@article{test, Title = {My Title}, year = {2023}}"
        assert _extract_field(entry, "title") == "My Title"


class TestExtractYear:
    """Test _extract_year_from_entry."""

    def test_normal(self):
        assert _extract_year_from_entry(ENTRY_ZOU) == 2023

    def test_2024(self):
        assert _extract_year_from_entry(ENTRY_ZHANG) == 2024

    def test_no_year(self):
        assert _extract_year_from_entry(ENTRY_INVALID_NO_YEAR) is None

    def test_empty(self):
        assert _extract_year_from_entry("") is None


class TestCountCitations:
    """Test _count_citations_in_text."""

    def test_basic(self):
        text = (
            "GCG attacks [Zou_2023_GCG] are effective. "
            "HaluEval [Li_2023_Halueval] is a benchmark. "
            "As shown by [Zou_2023_GCG], this works."
        )
        counts = _count_citations_in_text(text)
        assert counts["Zou_2023_GCG"] == 2
        assert counts["Li_2023_Halueval"] == 1

    def test_no_citations(self):
        counts = _count_citations_in_text("No citations here.")
        assert len(counts) == 0

    def test_filter_table_figure(self):
        text = "See [Table_1] and [Figure_3] and also [Zou_2023_GCG]."
        counts = _count_citations_in_text(text)
        assert "Table_1" not in counts
        assert "Figure_3" not in counts
        assert "Zou_2023_GCG" in counts

    def test_empty(self):
        counts = _count_citations_in_text("")
        assert len(counts) == 0


class TestReformatEntry:
    """Test _reformat_entry."""

    def test_basic(self):
        messy = "@article{key,\ntitle = {Title},\nyear = {2023}\n}"
        result = _reformat_entry(messy)
        assert result.startswith("@article{key,")
        assert "  title = {Title}," in result
        assert result.endswith("}")

    def test_already_formatted(self):
        result = _reformat_entry(ENTRY_ZOU)
        assert "@inproceedings{Zou_2023_GCG," in result
        assert "}" in result

    def test_empty(self):
        assert _reformat_entry("") == ""


class TestKeySimilarity:
    """Test _key_similarity."""

    def test_identical(self):
        assert _key_similarity("zou_2023_gcg", "zou_2023_gcg") == 1.0

    def test_partial(self):
        # zou_2023_gcg vs zou_2023_universal: condividono "zou" e "2023"
        score = _key_similarity("zou_2023_gcg", "zou_2023_universal")
        assert score > 0.4  # 2/4 tokens in common

    def test_no_match(self):
        score = _key_similarity("zou_2023_gcg", "smith_2020_bert")
        assert score == 0.0

    def test_empty(self):
        assert _key_similarity("", "zou_2023_gcg") == 0.0


# ============================================================================
# Test BibliographyGenerator
# ============================================================================

class TestBibliographyGeneratorInit:
    """Test inizializzazione e proprietà."""

    def test_init(self):
        gen = BibliographyGenerator()
        assert len(gen.master_bib) == 0
        assert len(gen.used_keys) == 0
        assert len(gen.filtered_entries) == 0
        assert len(gen.missing_keys) == 0
        assert len(gen) == 0

    def test_repr(self):
        gen = BibliographyGenerator()
        r = repr(gen)
        assert "BibliographyGenerator" in r
        assert "master=0" in r


class TestBibliographyGeneratorLoad:
    """Test caricamento master .bib."""

    def test_load_from_dict(self):
        gen = BibliographyGenerator()
        count = gen.load_master_bib_from_dict(_make_master_bib())
        assert count == 5
        assert len(gen.master_bib) == 5

    def test_load_from_file(self, tmp_path):
        bib_path = tmp_path / "test.bib"
        _write_master_bib(bib_path)

        gen = BibliographyGenerator()
        count = gen.load_master_bib(bib_path)
        assert count == 5

    def test_load_nonexistent(self, tmp_path):
        gen = BibliographyGenerator()
        count = gen.load_master_bib(tmp_path / "nonexistent.bib")
        assert count == 0


class TestBibliographyGeneratorGenerate:
    """Test generazione bibliografia filtrata."""

    def test_basic_filter(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())

        report = gen.generate(used_keys={"Zou_2023_GCG", "Li_2023_Halueval"})

        assert len(gen.filtered_entries) == 2
        assert "Zou_2023_GCG" in gen.filtered_entries
        assert "Li_2023_Halueval" in gen.filtered_entries
        assert len(gen.missing_keys) == 0
        assert len(gen) == 2

    def test_missing_keys(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())

        report = gen.generate(
            used_keys={"Zou_2023_GCG", "Nonexistent_2025_Key"}
        )

        assert len(gen.filtered_entries) == 1
        assert len(gen.missing_keys) == 1
        assert "Nonexistent_2025_Key" in gen.missing_keys

    def test_all_missing(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())

        report = gen.generate(used_keys={"NoKey_A", "NoKey_B"})

        assert len(gen.filtered_entries) == 0
        assert len(gen.missing_keys) == 2

    def test_empty_used_keys(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())

        report = gen.generate(used_keys=set())

        assert len(gen.filtered_entries) == 0
        assert len(gen.missing_keys) == 0

    def test_filter_all(self):
        """Tutte le 5 entry usate."""
        gen = BibliographyGenerator()
        master = _make_master_bib()
        gen.load_master_bib_from_dict(master)

        report = gen.generate(used_keys=set(master.keys()))

        assert len(gen.filtered_entries) == 5
        assert len(gen.missing_keys) == 0

    def test_with_validation(self):
        gen = BibliographyGenerator()
        master = _make_master_bib()
        master["NoAuthor_2023_Bad"] = ENTRY_INVALID_NO_AUTHOR
        gen.load_master_bib_from_dict(master)

        report = gen.generate(
            used_keys={"Zou_2023_GCG", "NoAuthor_2023_Bad"},
            validate=True,
        )

        assert len(gen.filtered_entries) == 2  # Entrambe incluse
        assert len(gen.invalid_entries) == 1  # Ma una non valida
        assert "NoAuthor_2023_Bad" in gen.invalid_entries

    def test_without_validation(self):
        gen = BibliographyGenerator()
        master = _make_master_bib()
        master["NoAuthor_2023_Bad"] = ENTRY_INVALID_NO_AUTHOR
        gen.load_master_bib_from_dict(master)

        report = gen.generate(
            used_keys={"NoAuthor_2023_Bad"},
            validate=False,
        )

        assert len(gen.invalid_entries) == 0  # Validazione disabilitata

    def test_with_survey_text(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())

        text = (
            "GCG [Zou_2023_GCG] attacks [Zou_2023_GCG]. "
            "HaluEval [Li_2023_Halueval]."
        )
        report = gen.generate(
            used_keys={"Zou_2023_GCG", "Li_2023_Halueval"},
            survey_text=text,
        )

        assert gen.citation_counts["Zou_2023_GCG"] == 2
        assert gen.citation_counts["Li_2023_Halueval"] == 1

    def test_report_structure(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())

        report = gen.generate(
            used_keys={"Zou_2023_GCG", "Li_2023_Halueval", "Missing_2025"},
        )

        assert "generated_at" in report
        assert "statistics" in report
        assert "missing_keys" in report
        assert "entries" in report
        assert len(report["entries"]) == 2
        assert "Missing_2025" in report["missing_keys"]

    def test_list_as_used_keys(self):
        """Accetta anche lista (non solo set)."""
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())

        report = gen.generate(used_keys=["Zou_2023_GCG", "Li_2023_Halueval"])
        assert len(gen.filtered_entries) == 2

    def test_empty_master_bib(self):
        gen = BibliographyGenerator()
        report = gen.generate(used_keys={"Zou_2023_GCG"})
        assert len(gen.filtered_entries) == 0
        assert len(gen.missing_keys) == 1


class TestBibliographyGeneratorSave:
    """Test salvataggio .bib e report."""

    def test_save_bib(self, tmp_path):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys={"Zou_2023_GCG", "Li_2023_Halueval"})

        bib_path = gen.save_bib(tmp_path / "final.bib")

        assert bib_path.exists()
        content = bib_path.read_text(encoding="utf-8")
        assert "Zou_2023_GCG" in content
        assert "Li_2023_Halueval" in content
        # Entry non usate non presenti
        assert "Greshake_2023_LLM" not in content

    def test_save_bib_no_generate(self):
        gen = BibliographyGenerator()
        with pytest.raises(RuntimeError):
            gen.save_bib("/tmp/test.bib")

    def test_save_report(self, tmp_path):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys={"Zou_2023_GCG"})

        report_path = gen.save_report(tmp_path / "report.json")

        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "statistics" in data
        assert "entries" in data

    def test_save_bib_nested_dir(self, tmp_path):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys={"Zou_2023_GCG"})

        nested = tmp_path / "deep" / "nested" / "final.bib"
        bib_path = gen.save_bib(nested)
        assert bib_path.exists()


class TestBibliographyGeneratorStats:
    """Test statistiche."""

    def test_basic_stats(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys={"Zou_2023_GCG", "Li_2023_Halueval", "Missing_X"})

        stats = gen.get_stats()

        assert stats["master_bib_total"] == 5
        assert stats["used_keys_total"] == 3
        assert stats["filtered_entries"] == 2
        assert stats["missing_keys"] == 1
        assert stats["coverage_rate"] == pytest.approx(2 / 3)

    def test_type_distribution(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys=set(_make_master_bib().keys()))

        stats = gen.get_stats()
        td = stats["type_distribution"]

        assert "inproceedings" in td
        assert "article" in td
        assert "misc" in td

    def test_year_distribution(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys=set(_make_master_bib().keys()))

        stats = gen.get_stats()
        yd = stats["year_distribution"]

        assert 2023 in yd
        assert 2024 in yd
        assert stats["year_min"] == 2023
        assert stats["year_max"] == 2024

    def test_empty_stats(self):
        gen = BibliographyGenerator()
        stats = gen.get_stats()
        assert stats["master_bib_total"] == 0
        assert stats["coverage_rate"] == 0.0

    def test_stats_with_citations(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())

        text = "[Zou_2023_GCG] [Zou_2023_GCG] [Li_2023_Halueval]"
        gen.generate(
            used_keys={"Zou_2023_GCG", "Li_2023_Halueval"},
            survey_text=text,
        )

        stats = gen.get_stats()
        assert stats["total_citation_occurrences"] == 3
        assert stats["top_cited"]["Zou_2023_GCG"] == 2


class TestBibliographyGeneratorMissingReport:
    """Test report chiavi mancanti."""

    def test_missing_with_suggestions(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys={"Zou_2023_Universal"})

        report = gen.get_missing_keys_report()

        assert len(report) == 1
        entry = report[0]
        assert entry["key"] == "Zou_2023_Universal"
        # Deve suggerire Zou_2023_GCG (condividono "zou" e "2023")
        assert len(entry["suggestions"]) >= 1

    def test_missing_no_suggestions(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys={"CompletelyRandom_9999_XYZ"})

        report = gen.get_missing_keys_report()
        assert len(report) == 1
        assert len(report[0]["suggestions"]) == 0


class TestBibliographyGeneratorEntryDetails:
    """Test dettagli singola entry."""

    def test_existing(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys={"Zou_2023_GCG"}, validate=True)

        details = gen.get_entry_details("Zou_2023_GCG")

        assert details is not None
        assert details["key"] == "Zou_2023_GCG"
        assert details["type"] == "inproceedings"
        assert details["year"] == 2023
        assert "Universal" in (details["title"] or "")
        assert details["in_filtered"] is True
        assert details["in_master"] is True
        assert details["valid"] is True

    def test_not_in_filtered_but_in_master(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys={"Zou_2023_GCG"})  # Solo Zou nel filtro

        details = gen.get_entry_details("Li_2023_Halueval")
        assert details is not None
        assert details["in_filtered"] is False
        assert details["in_master"] is True

    def test_nonexistent(self):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(used_keys={"Zou_2023_GCG"})

        details = gen.get_entry_details("Nonexistent_Key")
        assert details is None


class TestBibliographyGeneratorPrint:
    """Test print_summary (smoke test)."""

    def test_print_with_data(self, capsys):
        gen = BibliographyGenerator()
        gen.load_master_bib_from_dict(_make_master_bib())
        gen.generate(
            used_keys={"Zou_2023_GCG", "Missing_Key"},
            survey_text="[Zou_2023_GCG]",
        )
        gen.print_summary()  # Non deve crashare

    def test_print_empty(self, capsys):
        gen = BibliographyGenerator()
        gen.print_summary()  # Non deve crashare con dati vuoti


# ============================================================================
# Test convenience functions
# ============================================================================

class TestGenerateFinalBib:
    """Test generate_final_bib."""

    def test_basic(self, tmp_path):
        bib_src = tmp_path / "master.bib"
        _write_master_bib(bib_src)

        output = tmp_path / "final.bib"
        bib_path, report = generate_final_bib(
            used_keys={"Zou_2023_GCG", "Li_2023_Halueval"},
            master_bib_path=bib_src,
            output_path=output,
        )

        assert bib_path is not None
        assert bib_path.exists()
        content = bib_path.read_text()
        assert "Zou_2023_GCG" in content
        assert "Li_2023_Halueval" in content
        assert "Greshake_2023_LLM" not in content

    def test_empty_master(self, tmp_path):
        empty_bib = tmp_path / "empty.bib"
        empty_bib.write_text("")

        bib_path, report = generate_final_bib(
            used_keys={"Zou_2023_GCG"},
            master_bib_path=empty_bib,
            output_path=tmp_path / "out.bib",
        )

        assert bib_path is None

    def test_no_match(self, tmp_path):
        bib_src = tmp_path / "master.bib"
        _write_master_bib(bib_src)

        bib_path, report = generate_final_bib(
            used_keys={"Nonexistent_Key"},
            master_bib_path=bib_src,
            output_path=tmp_path / "out.bib",
        )

        assert bib_path is None

    def test_with_survey_text(self, tmp_path):
        bib_src = tmp_path / "master.bib"
        _write_master_bib(bib_src)

        text = "[Zou_2023_GCG] [Zou_2023_GCG] [Li_2023_Halueval]"
        bib_path, report = generate_final_bib(
            used_keys={"Zou_2023_GCG", "Li_2023_Halueval"},
            master_bib_path=bib_src,
            output_path=tmp_path / "final.bib",
            survey_text=text,
        )

        assert bib_path is not None
        assert "citation_counts" in report
        assert report["citation_counts"]["Zou_2023_GCG"] == 2


class TestGenerateBibFromState:
    """Test generate_bib_from_state."""

    def test_basic(self, tmp_path):
        bib_src = tmp_path / "master.bib"
        _write_master_bib(bib_src)

        # Creiamo uno stato minimale
        from paperhunter.agents.state import (
            SectionDraft,
            SectionStatus,
            CitationRecord,
        )

        draft = SectionDraft(
            section_key="test_section",
            title="Test",
            text="GCG [Zou_2023_GCG] is effective.",
            citations=[
                CitationRecord(
                    bibtex_key="Zou_2023_GCG",
                    claim="GCG is effective",
                    chunk_id="chunk1",
                    retrieval_score=0.9,
                ),
            ],
            status=SectionStatus.VERIFIED,
        )

        state = {
            "sections": {"test_section": draft},
            "final_paper": "GCG [Zou_2023_GCG] is effective.",
            "l2_summaries": [],
            "outline": [],
            "verification_results": {},
            "bibliography": [],
            "provenance": [],
            "version_info": None,
            "current_phase": "COMPLETED",
            "current_section_key": None,
            "iteration_count": 0,
            "errors": [],
            "metadata": {},
        }

        bib_path, report = generate_bib_from_state(
            state,
            master_bib_path=bib_src,
            output_path=tmp_path / "from_state.bib",
        )

        assert bib_path is not None
        content = bib_path.read_text()
        assert "Zou_2023_GCG" in content


class TestLoadAndFilterBib:
    """Test load_and_filter_bib."""

    def test_basic(self, tmp_path):
        bib_src = tmp_path / "master.bib"
        _write_master_bib(bib_src)

        result = load_and_filter_bib(
            bib_src, {"Zou_2023_GCG", "Li_2023_Halueval"}
        )

        assert len(result) == 2
        assert "Zou_2023_GCG" in result
        assert "Li_2023_Halueval" in result

    def test_empty_keys(self, tmp_path):
        bib_src = tmp_path / "master.bib"
        _write_master_bib(bib_src)

        result = load_and_filter_bib(bib_src, set())
        assert len(result) == 0


class TestMergeBibFiles:
    """Test merge_bib_files."""

    def test_merge_two(self, tmp_path):
        # File 1: Zou + Li
        bib1 = tmp_path / "bib1.bib"
        with open(bib1, "w") as f:
            f.write(ENTRY_ZOU + "\n\n" + ENTRY_LI + "\n\n")

        # File 2: Greshake + Zou (duplicato)
        bib2 = tmp_path / "bib2.bib"
        with open(bib2, "w") as f:
            f.write(ENTRY_GRESHAKE + "\n\n" + ENTRY_ZOU + "\n\n")

        output = tmp_path / "merged.bib"
        saved_path, count = merge_bib_files(bib1, bib2, output_path=output)

        assert saved_path.exists()
        assert count == 3  # Zou + Li + Greshake (Zou dedup)
        content = saved_path.read_text()
        assert "Zou_2023_GCG" in content
        assert "Li_2023_Halueval" in content
        assert "Greshake_2023_LLM" in content

    def test_merge_empty(self, tmp_path):
        empty = tmp_path / "empty.bib"
        empty.write_text("")

        output = tmp_path / "merged.bib"
        saved_path, count = merge_bib_files(empty, output_path=output)
        assert count == 0


# ============================================================================
# Test importazione package
# ============================================================================

class TestPackageImports:
    """Verifica che gli import dal package funzionino."""

    def test_from_module(self):
        from paperhunter.postprocessing.bibliography_generator import (
            BibliographyGenerator,
            generate_final_bib,
            generate_bib_from_state,
            load_and_filter_bib,
            merge_bib_files,
        )
        assert BibliographyGenerator is not None

    def test_from_package(self):
        from paperhunter.postprocessing import (
            BibliographyGenerator,
            generate_final_bib,
            generate_bib_from_state,
            load_and_filter_bib,
            merge_bib_files,
        )
        assert BibliographyGenerator is not None
