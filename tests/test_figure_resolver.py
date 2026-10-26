"""
test_figure_resolver.py — Test per paperhunter.postprocessing.figure_resolver.

Testa:
1. Tag figura non trovata → rimosso, aggiunto a unresolved_tags
2. Tag figura trovata ma path non esiste su disco → rimosso, aggiunto a unresolved_tags
3. Tag figura trovata con path valido → sostituito con markup Markdown
4. Testo misto (1 risolvibile + 2 non risolvibili) → 2 elementi in unresolved_tags
5. Output LaTeX per figura trovata con path valido
6. Tag malformati non vengono catturati dalla regex
7. Testo senza tag → restituito immutato, nessun unresolved
8. Resolver con store None → tutti i tag finiscono in unresolved_tags
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from paperhunter.postprocessing.figure_resolver import FigureResolver, _latex_escape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_store(records_by_collection: dict) -> MagicMock:
    """Crea uno store finto che restituisce records predefiniti per collection."""
    store = MagicMock()
    client = MagicMock()
    store.client = client

    def _scroll(collection_name, scroll_filter, limit, with_payload, with_vectors):
        records = records_by_collection.get(collection_name, [])
        return records, None

    client.scroll.side_effect = _scroll
    return store


def _make_record(record_id: str, payload: dict) -> MagicMock:
    rec = MagicMock()
    rec.id = record_id
    rec.payload = payload
    return rec


# ---------------------------------------------------------------------------
# Test 1: nessuna figura trovata → unresolved
# ---------------------------------------------------------------------------

def test_tag_not_found_goes_to_unresolved():
    store = _make_fake_store({})  # nessuna collection ha risultati
    resolver = FigureResolver(qdrant_store=store)
    text = "Introduzione.\n{{FIGURE:Smith_2023_Test:architecture_diagram}}\nFine."
    result, unresolved = resolver.resolve(text)
    assert "{{FIGURE:" not in result
    assert len(unresolved) == 1
    assert "Smith_2023_Test" in unresolved[0]


# ---------------------------------------------------------------------------
# Test 2: figura trovata ma path non esiste su disco → unresolved
# ---------------------------------------------------------------------------

def test_tag_found_but_missing_disk_file_goes_to_unresolved():
    payload = {
        "bibtex_key": "Jones_2022_Adv",
        "figure_type": "results_plot",
        "description": "Attack success rates",
        "image_path": "/nonexistent/path/fig.png",
        "caption": "Figure 1",
    }
    record = _make_record("rec-001", payload)
    store = _make_fake_store({"figure_descriptions": [record]})
    resolver = FigureResolver(qdrant_store=store)
    text = "Test.\n{{FIGURE:Jones_2022_Adv:results_plot}}\nEnd."
    result, unresolved = resolver.resolve(text)
    assert "{{FIGURE:" not in result
    assert len(unresolved) == 1
    assert "Jones_2022_Adv" in unresolved[0]


# ---------------------------------------------------------------------------
# Test 3: figura trovata con path valido → markup Markdown
# ---------------------------------------------------------------------------

def test_tag_resolved_to_markdown(tmp_path):
    img_file = tmp_path / "fig001.png"
    img_file.write_bytes(b"fake")
    payload = {
        "bibtex_key": "Zou_2023_GCG",
        "figure_type": "architecture_diagram",
        "description": "GCG attack pipeline",
        "image_path": str(img_file),
        "caption": "Architecture of GCG",
    }
    record = _make_record("rec-002", payload)
    store = _make_fake_store({"figure_descriptions": [record]})
    resolver = FigureResolver(qdrant_store=store)
    text = "The method.\n{{FIGURE:Zou_2023_GCG:architecture_diagram}}\nConclusion."
    result, unresolved = resolver.resolve(text)
    assert "{{FIGURE:" not in result
    assert f"!["  in result
    assert str(img_file) in result
    assert "[Zou_2023_GCG]" in result
    assert unresolved == []


# ---------------------------------------------------------------------------
# Test 4: testo misto — 1 risolvibile + 2 non risolvibili
# ---------------------------------------------------------------------------

def test_mixed_tags_two_unresolved(tmp_path):
    img_file = tmp_path / "arch.png"
    img_file.write_bytes(b"fake")

    found_payload = {
        "bibtex_key": "Alpha_2021",
        "figure_type": "any",
        "description": "Result plot",
        "image_path": str(img_file),
        "caption": "Results",
    }
    record = _make_record("rec-010", found_payload)

    # Solo figure_descriptions porta un risultato per Alpha_2021
    def _scroll(collection_name, scroll_filter, limit, with_payload, with_vectors):
        if collection_name == "figure_descriptions":
            # Isola il match per bibtex_key
            must = scroll_filter.must
            for cond in must:
                if getattr(cond, "key", None) == "bibtex_key":
                    if cond.match.value == "Alpha_2021":
                        return [record], None
        return [], None

    store = MagicMock()
    client = MagicMock()
    store.client = client
    client.scroll.side_effect = _scroll

    resolver = FigureResolver(qdrant_store=store)
    text = (
        "A.\n{{FIGURE:Alpha_2021:any}}\n"
        "B.\n{{FIGURE:Beta_2022:results_plot}}\n"
        "C.\n{{FIGURE:Gamma_2023:flowchart}}\n"
    )
    result, unresolved = resolver.resolve(text)
    assert "{{FIGURE:" not in result
    assert len(unresolved) == 2
    assert str(img_file) in result


# ---------------------------------------------------------------------------
# Test 5: output LaTeX
# ---------------------------------------------------------------------------

def test_tag_resolved_to_latex(tmp_path):
    img_file = tmp_path / "fig_latex.png"
    img_file.write_bytes(b"fake")
    payload = {
        "bibtex_key": "Li_2024_Def",
        "figure_type": "defense_pipeline",
        "description": "Defense architecture",
        "image_path": str(img_file),
        "caption": "Defense pipeline",
    }
    record = _make_record("rec-003", payload)
    store = _make_fake_store({"figure_descriptions": [record]})
    resolver = FigureResolver(qdrant_store=store)
    text = "Method.\n{{FIGURE:Li_2024_Def:defense_pipeline}}\nEnd."
    result, unresolved = resolver.resolve(text, output_format="latex")
    assert "{{FIGURE:" not in result
    assert r"\includegraphics" in result
    assert r"\cite{Li_2024_Def}" in result
    assert unresolved == []


# ---------------------------------------------------------------------------
# Test 6: tag malformati non vengono catturati
# ---------------------------------------------------------------------------

def test_malformed_tags_ignored():
    store = _make_fake_store({})
    resolver = FigureResolver(qdrant_store=store)
    text = "No {{FIGURE}} here, nor {{FIGURE:only_one_part}} or {FIGURE:x:y}."
    result, unresolved = resolver.resolve(text)
    assert result == text  # invariato
    assert unresolved == []


# ---------------------------------------------------------------------------
# Test 7: testo senza tag → invariato
# ---------------------------------------------------------------------------

def test_no_tags_returns_unchanged():
    store = _make_fake_store({})
    resolver = FigureResolver(qdrant_store=store)
    original = "This is a survey section without any figure tags."
    result, unresolved = resolver.resolve(original)
    assert result == original
    assert unresolved == []


# ---------------------------------------------------------------------------
# Test 8: store None → tutti i tag non risolti
# ---------------------------------------------------------------------------

def test_no_qdrant_store_all_unresolved():
    resolver = FigureResolver(qdrant_store=None)
    # Patcha QdrantStore per fare in modo che non sia importabile
    with patch("paperhunter.postprocessing.figure_resolver.FigureResolver._get_store", return_value=None):
        text = "A.\n{{FIGURE:X_2022_T1:any}}\nB.\n{{FIGURE:Y_2023_T2:flowchart}}\n"
        result, unresolved = resolver.resolve(text)
    assert "{{FIGURE:" not in result
    assert len(unresolved) == 2


# ---------------------------------------------------------------------------
# Test 9: _latex_escape
# ---------------------------------------------------------------------------

def test_latex_escape():
    assert _latex_escape("100% accuracy") == r"100\% accuracy"
    assert _latex_escape("A & B") == r"A \& B"
    assert _latex_escape("cost $10") == r"cost \$10"
    assert _latex_escape("a_b") == r"a\_b"
