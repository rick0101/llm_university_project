"""Test per ensure_complete_bibtex() e validate_bibtex_completeness() — Task 2.2.

Tutti i test usano oggetti Paper costruiti manualmente, senza mock S2.
"""

from __future__ import annotations

import pytest

from paperhunter.ingestion.bibtex_generator import (
    ensure_complete_bibtex,
    load_master_bib,
    validate_bibtex_completeness,
)
from paperhunter.models import Paper


# =============================================================================
# Helper
# =============================================================================

def _make_paper(**kwargs) -> Paper:
    defaults = {
        "title": "A Test Paper About Adversarial Attacks",
        "url": "https://example.com/paper",
        "authors": ["Alice Smith", "Bob Jones"],
        "year": 2023,
        "venue": "NeurIPS",
    }
    defaults.update(kwargs)
    return Paper(**defaults)


# =============================================================================
# Test 1: Paper senza year → entry con anno da bibtex_key o fallback "2000"
# =============================================================================

def test_ensure_bibtex_no_year_uses_fallback():
    """Paper senza year → il campo year nell'entry NON deve contenere 'XXXX' o essere vuoto."""
    paper = _make_paper(year=None, venue=None, arxiv_id=None)
    result = ensure_complete_bibtex(paper)

    assert result.bibtex_entry is not None
    # Verifica che il CAMPO year non sia placeholder
    import re
    year_match = re.search(r'year\s*=\s*\{([^}]*)\}', result.bibtex_entry)
    assert year_match is not None, "Nessun campo year nell'entry"
    year_val_str = year_match.group(1).strip()
    assert year_val_str != "XXXX", f"Anno è ancora placeholder 'XXXX'"
    assert year_val_str != "", "Anno è vuoto"
    assert year_val_str.isdigit(), f"Anno non numerico: {year_val_str}"
    assert int(year_val_str) > 0


# =============================================================================
# Test 2: Paper senza autori → entry con "Anonymous"
# =============================================================================

def test_ensure_bibtex_no_authors_uses_anonymous():
    """Paper con autori vuoti → bibtex_entry deve usare 'Anonymous'."""
    paper = _make_paper(authors=[])
    result = ensure_complete_bibtex(paper)

    assert result.bibtex_entry is not None
    assert "Anonymous" in result.bibtex_entry
    # Non deve contenere "Unknown" come autore
    import re
    author_match = re.search(r'author\s*=\s*\{([^}]*)\}', result.bibtex_entry)
    assert author_match is not None
    assert author_match.group(1).strip() != ""


# =============================================================================
# Test 3: Paper arXiv → tipo @misc, howpublished presente
# =============================================================================

def test_ensure_bibtex_arxiv_paper_uses_misc():
    """Paper arXiv → tipo @misc con howpublished = {arXiv preprint}."""
    paper = _make_paper(
        venue=None,
        journal=None,
        arxiv_id="2307.15043",
    )
    result = ensure_complete_bibtex(paper)

    assert result.bibtex_entry is not None
    assert "@misc" in result.bibtex_entry.lower()
    assert "arXiv preprint" in result.bibtex_entry or "arxiv" in result.bibtex_entry.lower()


# =============================================================================
# Test 4: Paper conference → tipo @inproceedings, booktitle presente
# =============================================================================

def test_ensure_bibtex_conference_paper():
    """Paper conference con venue → @inproceedings con booktitle."""
    paper = _make_paper(venue="NeurIPS 2023", arxiv_id=None)
    result = ensure_complete_bibtex(paper)

    assert result.bibtex_entry is not None
    assert "@inproceedings" in result.bibtex_entry.lower()
    assert "booktitle" in result.bibtex_entry.lower()


# =============================================================================
# Test 5: Paper journal → tipo @article, journal presente
# =============================================================================

def test_ensure_bibtex_journal_paper():
    """Paper journal → @article con journal."""
    paper = _make_paper(
        venue=None,
        journal="Journal of Machine Learning Research",
        arxiv_id=None,
    )
    result = ensure_complete_bibtex(paper)

    assert result.bibtex_entry is not None
    assert "@article" in result.bibtex_entry.lower()
    assert "journal" in result.bibtex_entry.lower()


# =============================================================================
# Test 6: validate_bibtex_completeness su entry completa → tutti "present"
# =============================================================================

def test_validate_complete_entry_all_present():
    """Entry BibTeX completa → tutti i campi critici sono 'present'."""
    entry = """\
@inproceedings{Zou_2023_GCG,
  title = {Universal and Transferable Adversarial Attacks},
  author = {Andy Zou and Zifan Wang},
  year = {2023},
  booktitle = {NeurIPS},
  url = {https://arxiv.org/abs/2307.15043}
}"""
    report = validate_bibtex_completeness(entry)

    assert report["title"] == "present"
    assert report["author"] == "present"
    assert report["year"] == "present"
    assert report["venue"] == "present"
    assert report["url"] == "present"


# =============================================================================
# Test 7: validate_bibtex_completeness su entry con placeholder anno
# =============================================================================

def test_validate_entry_with_year_placeholder():
    """Entry con year XXXX → year deve essere 'placeholder'."""
    entry = """\
@misc{Unknown_XXXX_Paper,
  title = {Some Paper},
  author = {Unknown},
  year = {XXXX}
}"""
    report = validate_bibtex_completeness(entry)
    assert report["year"] == "placeholder"
    assert report["author"] == "placeholder"


# =============================================================================
# Test 8: ensure_complete_bibtex idempotente su paper già completo
# =============================================================================

def test_ensure_bibtex_idempotent_on_complete_paper():
    """Su un paper già completo, la funzione non deve degradare il contenuto."""
    paper = _make_paper()
    result1 = ensure_complete_bibtex(paper)
    result2 = ensure_complete_bibtex(result1)

    # Tipo e chiave devono rimanere uguali
    import re
    type1 = re.match(r'@(\w+)\{', result1.bibtex_entry or "")
    type2 = re.match(r'@(\w+)\{', result2.bibtex_entry or "")
    assert type1 and type2
    assert type1.group(1).lower() == type2.group(1).lower()


# =============================================================================
# Test 9: bibtex_entry non può essere None dopo ensure_complete_bibtex
# =============================================================================

def test_ensure_bibtex_never_returns_none_entry():
    """bibtex_entry non può essere None per nessun paper, anche in assenza totale di metadati."""
    # Paper con il minimo assoluto
    paper = Paper(title="Minimal Paper", url="https://example.com", authors=[], year=None, venue=None)
    result = ensure_complete_bibtex(paper)

    assert result.bibtex_entry is not None
    assert len(result.bibtex_entry) > 10


def test_load_master_bib_preserves_at_signs_and_nested_braces(tmp_path):
    """An @ inside a field value must not truncate the surrounding entry."""
    first = """@inproceedings{Elchafei_2025_LLM,
  author = {Omar El Chafei and Jane Doe},
  title = {Hallucination Detectives for {LLM}-Generated Answers},
  booktitle = {SemEval@ACL 2025},
  year = {2025},
  url = {https://aclanthology.org/2025.semeval-1.84/}
}"""
    second = """@article{Other_2025_Paper,
  author = {Alice Smith},
  title = {A Complete Second Entry},
  journal = {CoRR},
  year = {2025}
}"""
    bib_path = tmp_path / "references.bib"
    bib_path.write_text(f"% header\n\n{first}\n\n{second}\n", encoding="utf-8")

    loaded = load_master_bib(bib_path)

    assert set(loaded) == {"Elchafei_2025_LLM", "Other_2025_Paper"}
    assert loaded["Elchafei_2025_LLM"] == first
    assert "booktitle = {SemEval@ACL 2025}" in loaded["Elchafei_2025_LLM"]
    assert loaded["Other_2025_Paper"] == second
