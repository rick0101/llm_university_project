"""Test suite per postprocessing/latex_exporter.py.

Verifica tutte le funzionalità di conversione Markdown → LaTeX:
- Escape caratteri speciali
- Conversione heading, bold/italic, citazioni, link, tabelle, elenchi
- Code blocks, immagini, footnote, blockquote, regole orizzontali
- Template ACL / IEEEtran / plain
- Pipeline completa markdown_to_latex
- Export su file
- Export da SurveyState
- Statistiche
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from paperhunter.postprocessing.latex_exporter import (
    _clean_double_blank_lines,
    _convert_blockquotes,
    _convert_body,
    _convert_bold_italic,
    _convert_citations,
    _convert_code_blocks,
    _convert_footnotes,
    _convert_headings,
    _convert_horizontal_rules,
    _convert_images,
    _convert_lists,
    _convert_tables,
    _convert_urls,
    _escape_latex,
    _extract_abstract_from_md,
    _extract_title_from_md,
    _markdown_table_to_latex,
    _sanitize_latex,
    export_latex,
    export_latex_from_state,
    get_export_stats,
    markdown_to_latex,
    print_export_summary,
)


# ============================================================================
# Test _escape_latex
# ============================================================================

class TestEscapeLatex:
    def test_special_chars(self):
        assert _escape_latex("10% & 20$") == r"10\% \& 20\$"

    def test_hash_underscore(self):
        assert _escape_latex("item #1 name_test") == r"item \#1 name\_test"

    def test_braces(self):
        assert _escape_latex("set {a, b}") == r"set \{a, b\}"

    def test_tilde_caret(self):
        result = _escape_latex("x~y and a^b")
        assert r"\textasciitilde{}" in result
        assert r"\textasciicircum{}" in result

    def test_no_special_chars(self):
        plain = "This is normal text with no special characters."
        assert _escape_latex(plain) == plain

    def test_empty_string(self):
        assert _escape_latex("") == ""


# ============================================================================
# Test _convert_headings
# ============================================================================

class TestConvertHeadings:
    def test_section(self):
        assert _convert_headings("# Introduction") == r"\section{Introduction}"

    def test_subsection(self):
        assert _convert_headings("## Background") == r"\subsection{Background}"

    def test_subsubsection(self):
        assert _convert_headings("### Details") == r"\subsubsection{Details}"

    def test_paragraph(self):
        assert _convert_headings("#### Minor Point") == r"\paragraph{Minor Point}"

    def test_multiple_headings(self):
        md = "# A\n\nSome text.\n\n## B\n\nMore text."
        result = _convert_headings(md)
        assert r"\section{A}" in result
        assert r"\subsection{B}" in result

    def test_heading_with_spaces(self):
        result = _convert_headings("#   Title With Spaces  ")
        assert r"\section{Title With Spaces}" in result

    def test_no_headings(self):
        text = "Just plain text without headings."
        assert _convert_headings(text) == text


# ============================================================================
# Test _convert_citations
# ============================================================================

class TestConvertCitations:
    def test_single_citation(self):
        result = _convert_citations("As shown by [Zou_2023_GCG], this attack works.")
        assert r"\cite{Zou_2023_GCG}" in result

    def test_multiple_citations(self):
        result = _convert_citations("Previous work [Zou_2023_GCG, Li_2023_Halueval] shows...")
        assert r"\cite{Zou_2023_GCG, Li_2023_Halueval}" in result

    def test_ignore_markdown_link(self):
        text = "[click here](https://example.com)"
        result = _convert_citations(text)
        # Should NOT be converted to \cite
        assert r"\cite" not in result
        assert "[click here](https://example.com)" in result

    def test_ignore_figure_reference(self):
        result = _convert_citations("See [Figure 1] for details.")
        assert r"\cite" not in result
        assert "[Figure 1]" in result

    def test_ignore_table_reference(self):
        result = _convert_citations("Results in [Table 2].")
        assert r"\cite" not in result

    def test_ignore_section_reference(self):
        result = _convert_citations("As discussed in [Section 3].")
        assert r"\cite" not in result

    def test_ignore_numeric(self):
        result = _convert_citations("Reference [42] is important.")
        assert r"\cite" not in result

    def test_complex_key(self):
        result = _convert_citations("From [Smith-Jones_2024_LLM-Safety].")
        assert r"\cite{Smith-Jones_2024_LLM-Safety}" in result

    def test_no_brackets(self):
        text = "No citations here."
        assert _convert_citations(text) == text


# ============================================================================
# Test _convert_bold_italic
# ============================================================================

class TestConvertBoldItalic:
    def test_bold(self):
        assert r"\textbf{bold text}" in _convert_bold_italic("**bold text**")

    def test_italic(self):
        assert r"\textit{italic text}" in _convert_bold_italic("*italic text*")

    def test_bold_italic(self):
        result = _convert_bold_italic("***both***")
        assert r"\textbf" in result
        assert r"\textit" in result

    def test_inline_code(self):
        assert r"\texttt{code}" in _convert_bold_italic("`code`")

    def test_mixed(self):
        result = _convert_bold_italic("Here **bold** and *italic* and `code`.")
        assert r"\textbf{bold}" in result
        assert r"\textit{italic}" in result
        assert r"\texttt{code}" in result

    def test_no_formatting(self):
        text = "Plain text only."
        assert _convert_bold_italic(text) == text


# ============================================================================
# Test _convert_urls
# ============================================================================

class TestConvertUrls:
    def test_basic_link(self):
        result = _convert_urls("[GitHub](https://github.com)")
        assert r"\href{https://github.com}{GitHub}" in result

    def test_multiple_links(self):
        md = "[A](http://a.com) and [B](http://b.com)"
        result = _convert_urls(md)
        assert r"\href{http://a.com}{A}" in result
        assert r"\href{http://b.com}{B}" in result

    def test_no_links(self):
        text = "No links here."
        assert _convert_urls(text) == text


# ============================================================================
# Test _convert_lists
# ============================================================================

class TestConvertLists:
    def test_bullet_list(self):
        md = "- Item A\n- Item B\n- Item C"
        result = _convert_lists(md)
        assert r"\begin{itemize}" in result
        assert r"\item Item A" in result
        assert r"\item Item B" in result
        assert r"\item Item C" in result
        assert r"\end{itemize}" in result

    def test_numbered_list(self):
        md = "1. First\n2. Second\n3. Third"
        result = _convert_lists(md)
        assert r"\begin{enumerate}" in result
        assert r"\item First" in result
        assert r"\item Third" in result
        assert r"\end{enumerate}" in result

    def test_asterisk_bullet(self):
        md = "* Item X\n* Item Y"
        result = _convert_lists(md)
        assert r"\begin{itemize}" in result
        assert r"\item Item X" in result

    def test_list_ends_with_blank_line(self):
        md = "- A\n- B\n\nParagraph after."
        result = _convert_lists(md)
        assert r"\end{itemize}" in result
        assert "Paragraph after." in result

    def test_no_lists(self):
        text = "Normal paragraph text."
        assert _convert_lists(text) == text


# ============================================================================
# Test _convert_tables
# ============================================================================

class TestConvertTables:
    def test_basic_table(self):
        md = "| H1 | H2 |\n|---|---|\n| A | B |\n| C | D |"
        result = _convert_tables(md)
        assert r"\begin{table}" in result
        assert r"\begin{tabular}" in result
        assert r"\toprule" in result
        assert r"\midrule" in result
        assert r"\bottomrule" in result
        assert "H1 & H2" in result
        assert "A & B" in result
        assert "C & D" in result

    def test_three_column_table(self):
        md = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"
        result = _convert_tables(md)
        assert r"{lll}" in result
        assert "A & B & C" in result

    def test_no_table(self):
        text = "No table here."
        assert _convert_tables(text) == text

    def test_ampersand_inside_cell_is_escaped_not_treated_as_column(self):
        md = (
            "| Category | Benchmark | Metrics |\n"
            "|---|---|---|\n"
            "| Faithfulness & Citation Verification | CogniBench | Accuracy |"
        )

        result = _convert_tables(md)

        assert r"Faithfulness \& Citation Verification & CogniBench & Accuracy" in result
        assert r"\begin{tabular}{lll}" in result


class TestSanitizeLatex:
    def test_unicode_typography_and_symbols_are_compile_safe(self):
        text = 'τ‑bench × 2; ‘quoted’ and “double” with narrow\u202fspace'

        result = _sanitize_latex(text)

        assert result == r"$\tau$-bench $\times$ 2; 'quoted' and ``double'' with narrow space"
        assert all(ord(char) < 128 for char in result)


class TestMarkdownTableToLatex:
    def test_empty_table(self):
        # Solo header e separatore, nessun dato
        lines = ["| H1 | H2 |", "|---|---|"]
        result = _markdown_table_to_latex(lines)
        assert r"\begin{table}" in result
        assert "H1 & H2" in result

    def test_uneven_rows(self):
        lines = [
            "| H1 | H2 | H3 |",
            "|---|---|---|",
            "| A | B |",  # riga con meno colonne
        ]
        result = _markdown_table_to_latex(lines)
        assert r"\begin{table}" in result
        # La riga corta viene paddata


# ============================================================================
# Test _convert_code_blocks
# ============================================================================

class TestConvertCodeBlocks:
    def test_basic_code_block(self):
        md = "```python\nprint('hello')\n```"
        result = _convert_code_blocks(md)
        assert r"\begin{verbatim}" in result
        assert "print('hello')" in result
        assert r"\end{verbatim}" in result

    def test_code_block_no_language(self):
        md = "```\nsome code\n```"
        result = _convert_code_blocks(md)
        assert r"\begin{verbatim}" in result

    def test_no_code_blocks(self):
        text = "Plain text."
        assert _convert_code_blocks(text) == text


# ============================================================================
# Test _convert_images
# ============================================================================

class TestConvertImages:
    def test_basic_image(self):
        md = "![Architecture Diagram](images/arch.png)"
        result = _convert_images(md)
        assert r"\begin{figure}" in result
        assert r"\includegraphics" in result
        assert "images/arch.png" in result
        assert r"\caption{Architecture Diagram}" in result
        assert r"\label{fig:" in result

    def test_no_images(self):
        text = "No images."
        assert _convert_images(text) == text


# ============================================================================
# Test _convert_horizontal_rules
# ============================================================================

class TestConvertHorizontalRules:
    def test_basic_rule(self):
        result = _convert_horizontal_rules("---")
        assert r"\noindent\rule{\textwidth}{0.4pt}" in result

    def test_long_rule(self):
        result = _convert_horizontal_rules("-----")
        assert r"\rule" in result

    def test_no_rule(self):
        text = "Just text."
        assert _convert_horizontal_rules(text) == text


# ============================================================================
# Test _convert_footnotes
# ============================================================================

class TestConvertFootnotes:
    def test_basic_footnote(self):
        md = "Some text[^1].\n\n[^1]: This is the footnote."
        result = _convert_footnotes(md)
        assert r"\footnote{This is the footnote.}" in result
        # La definizione dovrebbe essere rimossa
        assert "[^1]:" not in result

    def test_no_footnotes(self):
        text = "No footnotes here."
        assert _convert_footnotes(text) == text


# ============================================================================
# Test _convert_blockquotes
# ============================================================================

class TestConvertBlockquotes:
    def test_basic_blockquote(self):
        md = "> This is a quote.\n> It has two lines."
        result = _convert_blockquotes(md)
        assert r"\begin{quote}" in result
        assert "This is a quote." in result
        assert r"\end{quote}" in result

    def test_no_blockquote(self):
        text = "No quotes."
        assert _convert_blockquotes(text) == text


# ============================================================================
# Test _clean_double_blank_lines
# ============================================================================

class TestCleanDoubleBlankLines:
    def test_trims_triple_blanks(self):
        text = "A\n\n\n\nB"
        result = _clean_double_blank_lines(text)
        assert result == "A\n\nB"

    def test_preserves_double(self):
        text = "A\n\nB"
        assert _clean_double_blank_lines(text) == text


# ============================================================================
# Test _extract_title_from_md
# ============================================================================

class TestExtractTitle:
    def test_from_heading(self):
        md = "# My Survey Title\n\n## Introduction"
        assert _extract_title_from_md(md) == "My Survey Title"

    def test_fallback_first_line(self):
        md = "This is a title without heading\n\nSome text."
        assert _extract_title_from_md(md) == "This is a title without heading"

    def test_empty_md(self):
        assert _extract_title_from_md("") == "Untitled Survey"

    def test_blank_lines_before_heading(self):
        md = "\n\n# Title Here\n\nBody."
        assert _extract_title_from_md(md) == "Title Here"


# ============================================================================
# Test _extract_abstract_from_md
# ============================================================================

class TestExtractAbstract:
    def test_extract_abstract(self):
        md = "# Title\n\n## Abstract\n\nThis is the abstract text.\n\n## Introduction\n\nIntro text."
        abstract = _extract_abstract_from_md(md)
        assert abstract is not None
        assert "This is the abstract text." in abstract

    def test_no_abstract(self):
        md = "# Title\n\n## Introduction\n\nIntro text."
        assert _extract_abstract_from_md(md) is None

    def test_abstract_case_insensitive(self):
        md = "# Title\n\n## ABSTRACT\n\nAbs text.\n\n## Intro"
        assert _extract_abstract_from_md(md) is not None


# ============================================================================
# Test _convert_body (pipeline completa)
# ============================================================================

class TestConvertBody:
    def test_mixed_content(self):
        md = (
            "# Introduction\n\n"
            "As shown by [Zou_2023_GCG], **adversarial attacks** pose risks.\n\n"
            "## Background\n\n"
            "- Item A\n"
            "- Item B\n\n"
            "| Method | Score |\n"
            "|--------|-------|\n"
            "| GCG    | 0.95  |\n"
        )
        result = _convert_body(md)
        assert r"\section{Introduction}" in result
        assert r"\cite{Zou_2023_GCG}" in result
        assert r"\textbf{adversarial attacks}" in result
        assert r"\subsection{Background}" in result
        assert r"\begin{itemize}" in result
        assert r"\begin{table}" in result

    def test_empty(self):
        assert _convert_body("") == ""


# ============================================================================
# Test markdown_to_latex (API principale)
# ============================================================================

class TestMarkdownToLatex:
    SAMPLE_MD = (
        "# A Survey on LLM Security\n\n"
        "## Abstract\n\n"
        "This survey reviews attacks and defenses.\n\n"
        "## Introduction\n\n"
        "Large language models [Zou_2023_GCG] have vulnerabilities.\n\n"
        "## Prompt Injection Attacks\n\n"
        "**Prompt injection** is a key threat [Greshake_2023_LLM].\n"
    )

    def test_acl_template(self):
        result = markdown_to_latex(self.SAMPLE_MD, template="acl")
        assert r"\documentclass" in result
        assert r"\usepackage{natbib}" in result
        assert r"\bibliography{" in result
        assert r"\cite{Zou_2023_GCG}" in result
        assert r"\begin{abstract}" in result
        assert "This survey reviews attacks and defenses." in result

    def test_ieeetran_template(self):
        result = markdown_to_latex(self.SAMPLE_MD, template="ieeetran")
        assert r"\documentclass[conference]{IEEEtran}" in result
        assert r"\usepackage{cite}" in result

    def test_plain_template(self):
        result = markdown_to_latex(self.SAMPLE_MD, template="plain")
        assert r"\documentclass[12pt,a4paper]{article}" in result

    def test_invalid_template(self):
        with pytest.raises(ValueError, match="non supportato"):
            markdown_to_latex(self.SAMPLE_MD, template="unknown")

    def test_custom_title(self):
        result = markdown_to_latex(self.SAMPLE_MD, title="Custom Title")
        assert r"\title{Custom Title}" in result

    def test_custom_author(self):
        result = markdown_to_latex(self.SAMPLE_MD, author="John Doe")
        assert r"\author{John Doe}" in result

    def test_custom_bib_file(self):
        result = markdown_to_latex(self.SAMPLE_MD, bib_file="my_refs.bib")
        assert r"\bibliography{my_refs}" in result

    def test_bib_file_without_extension(self):
        result = markdown_to_latex(self.SAMPLE_MD, bib_file="refs")
        assert r"\bibliography{refs}" in result

    def test_auto_title_extraction(self):
        result = markdown_to_latex(self.SAMPLE_MD)
        assert r"\title{A Survey on LLM Security}" in result

    def test_abstract_extraction(self):
        result = markdown_to_latex(self.SAMPLE_MD)
        assert r"\begin{abstract}" in result
        # L'abstract non deve apparire anche come \section{Abstract}
        # (dovrebbe essere rimosso dal body)

    def test_minimal_md(self):
        result = markdown_to_latex("Just text.", template="plain")
        assert r"\documentclass" in result
        assert "Just text." in result


# ============================================================================
# Test export_latex (file I/O)
# ============================================================================

class TestExportLatex:
    SAMPLE_MD = "# Test\n\nHello [Key_2023_Test]."

    def test_export_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "survey.tex"
            result = export_latex(self.SAMPLE_MD, output_path=out, template="plain")
            assert result == out
            assert out.exists()
            content = out.read_text(encoding="utf-8")
            assert r"\documentclass" in content
            assert r"\cite{Key_2023_Test}" in content

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "deep" / "nested" / "survey.tex"
            result = export_latex(self.SAMPLE_MD, output_path=out)
            assert result.exists()

    def test_copy_bib(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Crea un file .bib sorgente
            bib_src = Path(tmpdir) / "source.bib"
            bib_src.write_text("@article{test, title={Test}}", encoding="utf-8")

            out = Path(tmpdir) / "output" / "survey.tex"
            export_latex(
                self.SAMPLE_MD,
                output_path=out,
                bib_file="refs",
                copy_bib=True,
                bib_source_path=bib_src,
            )
            bib_dest = Path(tmpdir) / "output" / "refs.bib"
            assert bib_dest.exists()

    def test_copy_bib_missing_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "survey.tex"
            # Non dovrebbe crashare se il .bib sorgente non esiste
            export_latex(
                self.SAMPLE_MD,
                output_path=out,
                copy_bib=True,
                bib_source_path="/nonexistent/refs.bib",
            )
            assert out.exists()


# ============================================================================
# Test export_latex_from_state
# ============================================================================

class TestExportLatexFromState:
    def test_basic_export(self):
        state = {
            "final_paper": "# Survey\n\n## Abstract\n\nAbstract text.\n\n## Introduction\n\nHello [Key_2023_A].",
            "version_info": None,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_latex_from_state(state, output_dir=tmpdir)
            assert result is not None
            assert result.exists()
            content = result.read_text(encoding="utf-8")
            assert r"\cite{Key_2023_A}" in content

    def test_empty_paper(self):
        state = {"final_paper": "", "version_info": None}
        result = export_latex_from_state(state, output_dir="/tmp/test")
        assert result is None

    def test_no_paper_key(self):
        state = {"version_info": None}
        result = export_latex_from_state(state, output_dir="/tmp/test")
        assert result is None

    def test_with_version_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "final_paper": "# Title\n\nContent.",
                "version_info": {
                    "title": "Custom Title From Version",
                    "output_dir": tmpdir,
                },
            }
            result = export_latex_from_state(state)
            assert result is not None
            content = result.read_text(encoding="utf-8")
            assert "Custom Title From Version" in content


# ============================================================================
# Test get_export_stats
# ============================================================================

class TestGetExportStats:
    def test_stats(self):
        latex = (
            r"\documentclass{article}" + "\n"
            r"\begin{document}" + "\n"
            r"\section{Intro}" + "\n"
            r"\subsection{BG}" + "\n"
            r"\subsubsection{Detail}" + "\n"
            r"Text \cite{A} and \cite{B}." + "\n"
            r"\begin{table}[htbp]" + "\n"
            r"\end{table}" + "\n"
            r"\begin{figure}[htbp]" + "\n"
            r"\end{figure}" + "\n"
            r"\begin{abstract}" + "\n"
            r"Abstract." + "\n"
            r"\end{abstract}" + "\n"
            r"\bibliography{refs}" + "\n"
            r"\begin{itemize}" + "\n"
            r"\item A" + "\n"
            r"\end{itemize}" + "\n"
            r"\begin{enumerate}" + "\n"
            r"\item 1" + "\n"
            r"\end{enumerate}" + "\n"
            r"\end{document}"
        )
        stats = get_export_stats(latex)
        assert stats["sections"] == 1
        assert stats["subsections"] == 1
        assert stats["subsubsections"] == 1
        assert stats["citations"] == 2
        assert stats["tables"] == 1
        assert stats["figures"] == 1
        assert stats["has_abstract"] is True
        assert stats["has_bibliography"] is True
        assert stats["itemize_lists"] == 1
        assert stats["enumerate_lists"] == 1
        assert stats["total_chars"] > 0
        assert stats["total_lines"] > 0

    def test_empty(self):
        stats = get_export_stats("")
        assert stats["sections"] == 0
        assert stats["citations"] == 0
        assert stats["has_abstract"] is False


# ============================================================================
# Test print_export_summary (smoke)
# ============================================================================

class TestPrintExportSummary:
    def test_no_crash(self):
        latex = r"\section{A}" + "\n" + r"\cite{Key}" + "\n"
        print_export_summary(latex)  # Non deve crashare

    def test_with_path(self):
        print_export_summary(r"\section{A}", output_path=Path("/tmp/test.tex"))


# ============================================================================
# Test integrazione — pipeline completa su markdown realistico
# ============================================================================

class TestIntegration:
    """Test end-to-end con un mini survey Markdown realistico."""

    SURVEY_MD = """\
# A Survey on LLM Security: Attacks, Defenses, and Testing Methodologies

## Abstract

This survey provides a comprehensive review of security vulnerabilities
in large language models (LLMs), covering prompt injection [Greshake_2023_LLM],
adversarial attacks [Zou_2023_GCG], and data poisoning [Wan_2023_Poisoning].

## Introduction

Large language models have revolutionized NLP, but they introduce
significant security risks. **Prompt injection** attacks allow adversaries
to manipulate model behavior [Greshake_2023_LLM, Perez_2022_Ignore].
This survey covers:

- Prompt injection attacks and defenses
- Adversarial example generation
- Data and model poisoning
- Hallucination-related vulnerabilities

## Prompt Injection Attacks

### Direct Prompt Injection

Direct prompt injection involves crafting inputs that override system
instructions [Perez_2022_Ignore]. The *Ignore Previous Instructions* attack
demonstrated fundamental weaknesses in instruction-following models.

### Indirect Prompt Injection

Indirect attacks manipulate external data sources consumed by the model
[Greshake_2023_LLM]. These attacks are particularly dangerous in
retrieval-augmented generation (RAG) systems.

## Testing Methodologies

| Method | Metric | Dataset | Year |
|--------|--------|---------|------|
| GCG    | ASR    | AdvBench| 2023 |
| AutoDAN| ASR    | Custom  | 2023 |
| PAIR   | Judge  | JBB     | 2024 |

## Conclusion

LLM security remains an active research area. Future work should focus
on ***robust defenses*** that generalize across attack types.

---

![Pipeline Overview](figures/pipeline.png)

> Note: This survey is continuously updated as new research emerges.
"""

    def test_full_acl_pipeline(self):
        result = markdown_to_latex(
            self.SURVEY_MD,
            template="acl",
            author="Riccardo Test",
            bib_file="references",
        )
        # Struttura documento
        assert r"\documentclass" in result
        assert r"\begin{document}" in result
        assert r"\end{document}" in result
        assert r"\bibliography{references}" in result
        assert r"\maketitle" in result

        # Abstract estratto
        assert r"\begin{abstract}" in result

        # Citazioni convertite
        assert r"\cite{Greshake_2023_LLM}" in result
        assert r"\cite{Zou_2023_GCG}" in result
        assert r"\cite{Greshake_2023_LLM, Perez_2022_Ignore}" in result

        # Heading
        assert r"\section{Prompt Injection Attacks}" in result
        assert r"\subsection{Direct Prompt Injection}" in result

        # Formattazione
        assert r"\textbf{Prompt injection}" in result
        assert r"\textit{Ignore Previous Instructions}" in result

        # Elenco
        assert r"\begin{itemize}" in result
        assert r"\item Prompt injection attacks and defenses" in result

        # Tabella
        assert r"\begin{table}" in result
        assert "GCG & ASR & AdvBench & 2023" in result

        # Immagine
        assert r"\includegraphics" in result
        assert "figures/pipeline.png" in result

        # Blockquote
        assert r"\begin{quote}" in result

        # Regola orizzontale
        assert r"\rule" in result

    def test_full_ieeetran_pipeline(self):
        result = markdown_to_latex(self.SURVEY_MD, template="ieeetran")
        assert r"\documentclass[conference]{IEEEtran}" in result
        assert r"\cite{Zou_2023_GCG}" in result

    def test_stats_on_full_survey(self):
        latex = markdown_to_latex(self.SURVEY_MD, template="acl")
        stats = get_export_stats(latex)
        assert stats["sections"] >= 3
        assert stats["citations"] >= 3
        assert stats["tables"] >= 1
        assert stats["figures"] >= 1
        assert stats["has_abstract"] is True
        assert stats["has_bibliography"] is True

    def test_export_and_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "survey.tex"
            export_latex(self.SURVEY_MD, output_path=out, template="acl")
            assert out.exists()
            content = out.read_text(encoding="utf-8")
            assert len(content) > 500
            assert r"\cite{" in content


# ============================================================================
# Test import dal package
# ============================================================================

class TestPackageImports:
    def test_from_module(self):
        from paperhunter.postprocessing.latex_exporter import (
            markdown_to_latex,
            export_latex,
            export_latex_from_state,
            get_export_stats,
            print_export_summary,
        )
        assert callable(markdown_to_latex)
        assert callable(export_latex)
        assert callable(export_latex_from_state)
        assert callable(get_export_stats)
        assert callable(print_export_summary)

    def test_from_package(self):
        from paperhunter.postprocessing import (
            markdown_to_latex,
            export_latex,
            export_latex_from_state,
            get_export_stats,
            print_export_summary,
        )
        assert callable(markdown_to_latex)
