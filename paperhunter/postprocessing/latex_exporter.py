"""postprocessing/latex_exporter.py — Export LaTeX del survey generato.

Converte il paper Markdown prodotto dalla pipeline agentica in formato LaTeX
compilabile, con supporto per tre template:

- **acl**: ACL Anthology style (``acl_natbib``)
- **ieeetran**: IEEE Transactions style
- **plain**: template minimale ``article``

Gestisce la conversione di:
- Heading (``#`` → ``\\section{}``, ``##`` → ``\\subsection{}``, …)
- Formattazione (``**bold**`` → ``\\textbf{}``, ``*italic*`` → ``\\textit{}``, …)
- Citazioni (``[bibtex_key]`` → ``\\cite{bibtex_key}``)
- Tabelle Markdown → ``tabular`` LaTeX
- Figure placeholder
- Footnote, URL, elenchi puntati/numerati

Uso tipico::

    from paperhunter.postprocessing.latex_exporter import (
        markdown_to_latex,
        export_latex,
        export_latex_from_state,
    )

    # Conversione diretta
    latex_content = markdown_to_latex(md_text, template="acl", bib_file="refs.bib")

    # Export su file
    tex_path = export_latex(
        md_content=md_text,
        output_path="output/survey.tex",
        bib_file="references.bib",
        template="ieeetran",
    )

    # Da SurveyState
    tex_path = export_latex_from_state(state, output_dir="output/versions/v1.0/")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.config import settings
from paperhunter.utils.logging_setup import (
    get_logger,
    print_panel,
    print_success,
    print_table,
    print_warning,
)

logger = get_logger(__name__)


class CitationIntegrityError(RuntimeError):
    """L'export finale è stato bloccato da una violazione citazionale."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}


def _extract_markdown_citation_keys(text: str) -> set[str]:
    """Estrae chiavi citazionali, incluse liste ``[key1, key2]``."""
    keys: set[str] = set()
    false_prefixes = (
        "Table", "Figure", "Fig", "Section", "Appendix", "Equation",
        "Algorithm", "Step", "SUPPORTED", "NOT_SUPPORTED",
    )
    for match in re.finditer(r"\[([^\]]+)\]", text):
        if match.end() < len(text) and text[match.end()] == "(":
            continue
        for raw_key in re.split(r"[,;]", match.group(1)):
            key = raw_key.strip()
            if any(key.startswith(prefix) for prefix in false_prefixes):
                continue
            if (
                re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]+", key)
                and "_" in key
                and any(char.isdigit() for char in key)
            ):
                keys.add(key)
    return keys


def _find_dblp_manifest(bib_path: Path) -> Path | None:
    candidates = [
        bib_path.with_name("references_dblp_manifest.json"),
        Path(settings.output.base_dir) / "references_dblp_manifest.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def validate_citation_integrity(
    md_content: str,
    bib_path: str | Path | None,
    *,
    require_dblp: bool = True,
) -> dict[str, Any]:
    """Fail-closed gate: testo, BibTeX e manifest DBLP devono concordare."""
    cited_keys = _extract_markdown_citation_keys(md_content)
    internal_refs = sorted(set(re.findall(r"\[(?:l2_|cluster_)[^\]]+\]", md_content)))
    errors: list[str] = []
    if internal_refs:
        errors.append(f"placeholder interni presenti: {', '.join(internal_refs[:10])}")
    if not cited_keys:
        errors.append("nessuna citazione bibliografica rilevata nel survey")

    if bib_path is None or not Path(bib_path).exists():
        errors.append("file BibTeX finale assente")
        entries: dict[str, str] = {}
        duplicate_keys: set[str] = set()
        resolved_bib = None
    else:
        from paperhunter.ingestion.bibtex_generator import load_master_bib

        resolved_bib = Path(bib_path)
        raw_bib = resolved_bib.read_text(encoding="utf-8")
        raw_keys = re.findall(r"@\w+\{([^,]+),", raw_bib)
        duplicate_keys = {key for key in raw_keys if raw_keys.count(key) > 1}
        entries = load_master_bib(resolved_bib)

    if duplicate_keys:
        errors.append(f"chiavi BibTeX duplicate: {', '.join(sorted(duplicate_keys))}")
    missing = cited_keys - set(entries)
    if missing:
        errors.append(f"citazioni senza entry BibTeX: {', '.join(sorted(missing))}")

    manifest_path: Path | None = None
    manifest: dict[str, Any] = {}
    if require_dblp and resolved_bib is not None:
        manifest_path = _find_dblp_manifest(resolved_bib)
        if manifest_path is None:
            errors.append("manifest DBLP assente: re-eseguire l'ingestion strict")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"manifest DBLP illeggibile: {exc}")

        for key in sorted(cited_keys & set(entries)):
            record = manifest.get(key)
            if not isinstance(record, dict) or not record.get("verified"):
                errors.append(f"{key}: record DBLP non verificato nel manifest")
                continue
            dblp_key = str(record.get("dblp_key") or "")
            dblp_url = str(record.get("dblp_url") or "")
            if not dblp_key or "dblp.org/rec/" not in dblp_url:
                errors.append(f"{key}: identità/URL DBLP canonici mancanti")
            expected_hash = record.get("bibtex_sha256")
            actual_hash = hashlib.sha256(entries[key].strip().encode("utf-8")).hexdigest()
            if expected_hash != actual_hash:
                errors.append(f"{key}: entry BibTeX diversa da quella verificata su DBLP")

    report = {
        "cited_keys": sorted(cited_keys),
        "bibtex_keys": sorted(entries),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "errors": errors,
        "passed": not errors,
    }
    if errors:
        raise CitationIntegrityError(
            "Export LaTeX bloccato: " + "; ".join(errors), report=report,
        )
    return report


# ============================================================================
# Template LaTeX
# ============================================================================

_TEMPLATE_ACL = r"""\documentclass[11pt,a4paper]{{article}}
\usepackage[hyperref]{{ntheorem}}
\usepackage{{times}}
\usepackage{{latexsym}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{url}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{microtype}}
\usepackage{{natbib}}
\usepackage{{hyperref}}
\hypersetup{{hidelinks}}
\usepackage{{cleveref}}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

\setlength{{\emergencystretch}}{{3em}}

\begin{{document}}
\maketitle

{body}

\bibliographystyle{{abbrv}}
\bibliography{{{bibfile}}}

\end{{document}}
"""

_TEMPLATE_IEEETRAN = r"""\documentclass[conference]{{IEEEtran}}
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{textcomp}}
\usepackage{{xcolor}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{hyperref}}
\hypersetup{{hidelinks}}
\usepackage{{url}}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

\setlength{{\emergencystretch}}{{3em}}

\begin{{document}}
\maketitle

{body}

\bibliographystyle{{IEEEtran}}
\bibliography{{{bibfile}}}

\end{{document}}
"""

_TEMPLATE_PLAIN = r"""\documentclass[12pt,a4paper]{{article}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{url}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{hyperref}}
\hypersetup{{hidelinks}}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

\setlength{{\emergencystretch}}{{3em}}

\begin{{document}}
\maketitle

{body}

\bibliographystyle{{abbrv}}
\bibliography{{{bibfile}}}

\end{{document}}
"""

_TEMPLATES: dict[str, str] = {
    "acl": _TEMPLATE_ACL,
    "ieeetran": _TEMPLATE_IEEETRAN,
    "plain": _TEMPLATE_PLAIN,
}


# ============================================================================
# Caratteri speciali LaTeX
# ============================================================================

_LATEX_SPECIAL_CHARS: dict[str, str] = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Regex per caratteri speciali (escludi backslash che trattiamo separatamente)
_SPECIAL_CHARS_RE = re.compile(r"([&%$#_{}~^])")


# ============================================================================
# Sanitizzazione post-conversione del LaTeX generato da LLM
# ============================================================================

def _sanitize_latex(latex_content: str, bib_keys: set[str] | None = None) -> str:
    """Sanitizza il LaTeX generato, correggendo errori tipici da output LLM.

    Applica le seguenti correzioni:
    Non elimina mai citazioni o placeholder: il gate fail-closed deve
    segnalarli prima della scrittura del file finale.
    """
    text = latex_content

    # Normalize Unicode typography and symbols unsupported by the legacy Times
    # font used by the article templates. Keep replacements deterministic and
    # LaTeX-native so the generated source compiles without dropped glyphs.
    unicode_replacements = {
        "\u00d7": r"$\times$",
        "\u03c4": r"$\tau$",
        "\u2011": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": "``",
        "\u201d": "''",
        "\u202f": " ",
    }
    for source, replacement in unicode_replacements.items():
        text = text.replace(source, replacement)

    # Unicode dashes → LaTeX dashes
    text = text.replace("\u2014", "---")   # em-dash
    text = text.replace("\u2013", "--")     # en-dash

    # 3. Escape & dentro \textbf{} (tipico in tabelle generate dal LLM)
    def _fix_textbf_ampersand(m: re.Match) -> str:
        inner = re.sub(r"(?<!\\)&", r"\\&", m.group(1))
        return r"\textbf{" + inner + "}"

    text = re.sub(r"\\textbf\{([^}]*&[^}]*)\}", _fix_textbf_ampersand, text)

    # 3b. Escape bare % (commento LaTeX) dentro il testo generato dal LLM.
    #     Tocca solo righe che contengono testo visibile prima del %;
    #     non tocca % già escaped (\%) né righe che iniziano con %.
    def _escape_percent(line: str) -> str:
        if line.lstrip().startswith("%"):
            return line  # commento LaTeX legittimo
        return re.sub(r"(?<!\\)%", r"\\%", line)

    text = "\n".join(_escape_percent(l) for l in text.split("\n"))

    # 4. Subscript bare → math mode (es. ASR_k → ASR$_{k}$)
    #    Proteggi comandi LaTeX che usano legittimamente _
    _prot: dict[str, str] = {}
    _cnt = [0]

    def _protect(m: re.Match) -> str:
        key = f"\x00PROT{_cnt[0]}\x00"
        _cnt[0] += 1
        _prot[key] = m.group(0)
        return key

    # Proteggi comandi LaTeX che usano legittimamente _ nei loro argomenti
    text = re.sub(r"\\(?:cite|label|ref|url|cref|Cref|href)\{[^}]*\}", _protect, text)
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", _protect, text)
    text = re.sub(r"\\(?:caption|bibliography)\{[^}]*\}", _protect, text)
    text = re.sub(r"\$[^$]+\$", _protect, text)
    text = re.sub(r"(?<!\\)%.*$", _protect, text, flags=re.MULTILINE)
    # Fix bare subscripts
    text = re.sub(
        r"(?<![\\$])(\b[A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9]+)\b",
        lambda m: m.group(1) + "$_{" + m.group(2) + "}$",
        text,
    )
    for key, val in _prot.items():
        text = text.replace(key, val)

    return text


def _load_bib_keys(bib_path: Path | str | None) -> set[str] | None:
    """Carica le chiavi BibTeX da un file .bib. Ritorna None se non disponibile."""
    if bib_path is None:
        return None
    bib_path = Path(bib_path)
    if not bib_path.exists():
        return None
    try:
        content = bib_path.read_text(encoding="utf-8")
        return set(re.findall(r"@\w+\{([^,]+),", content))
    except Exception:
        return None


# ============================================================================
# Conversione Markdown → LaTeX (funzioni interne)
# ============================================================================

def _escape_latex(text: str) -> str:
    """Esegue l'escape dei caratteri speciali LaTeX nel testo piano.

    NON tocca i comandi LaTeX già presenti (es. ``\\cite{}``, ``\\textbf{}``).
    Lavora solo su testo semplice, non su blocchi di codice.
    """
    # Proteggi i backslash esistenti (non li raddoppiamo se già parte di comandi)
    # Non facciamo escape del backslash qui: lo gestiscono i converter
    result = _SPECIAL_CHARS_RE.sub(
        lambda m: _LATEX_SPECIAL_CHARS[m.group(1)], text
    )
    return result


def _convert_headings(text: str) -> str:
    """Converte heading Markdown in comandi LaTeX di sezionamento.

    - ``# Title``   → ``\\section{Title}``
    - ``## Title``  → ``\\subsection{Title}``
    - ``### Title`` → ``\\subsubsection{Title}``
    - ``#### Title`` → ``\\paragraph{Title}``
    """
    mapping = {
        1: "section",
        2: "subsection",
        3: "subsubsection",
        4: "paragraph",
    }

    def _heading_replacer(match: re.Match) -> str:
        level = len(match.group(1))
        title = match.group(2).strip()
        cmd = mapping.get(level, "paragraph")
        return f"\\{cmd}{{{title}}}"

    return re.sub(r"^(#{1,4})\s+(.+)$", _heading_replacer, text, flags=re.MULTILINE)


def _convert_citations(text: str) -> str:
    r"""Converte citazioni ``[bibtex_key]`` in ``\cite{bibtex_key}``.

    Gestisce citazioni multiple: ``[key1, key2]`` → ``\\cite{key1, key2}``
    e singole: ``[Key_2023_Word]`` → ``\\cite{Key_2023_Word}``.

    Ignora link Markdown ``[text](url)`` e referenze ``[Figure 1]``, ``[Table 2]``.
    """
    # Pattern per citazioni singole [Key_YYYY_Word] o multiple [key1, key2]
    # Deve matchare: lettere, numeri, underscore, trattini; opzionalmente separati da virgola
    # Non matchare: [text](url) — link Markdown
    # Non matchare: [Figure X], [Table X], [Section X], [Appendix X]

    def _cite_replacer(match: re.Match) -> str:
        inner = match.group(1).strip()

        # Ignora se seguito da (url) — link Markdown
        end_pos = match.end()
        rest = text[end_pos:end_pos + 1] if end_pos < len(text) else ""
        if rest == "(":
            return match.group(0)

        # Ignora referenze a figure/tabelle/sezioni
        ignore_prefixes = (
            "Figure", "Fig.", "Table", "Tab.", "Section", "Sec.",
            "Appendix", "App.", "Equation", "Eq.", "Algorithm", "Alg.",
            "Chapter", "Ch.", "Listing", "Theorem", "Lemma", "Definition",
        )
        stripped = inner.lstrip()
        for prefix in ignore_prefixes:
            if stripped.startswith(prefix):
                return match.group(0)

        # Ignora se contiene solo numeri (es. [1], [42])
        if inner.strip().isdigit():
            return match.group(0)

        # Ignora placement specifier LaTeX (es. [htbp], [h], [!ht])
        if re.match(r"^[!htbpH]+$", inner.strip()):
            return match.group(0)

        # Ignora se contiene spazi e non sembra una lista di chiavi BibTeX
        # Le chiavi BibTeX tipiche: Author_YYYY_Word, autore2024keyword
        keys = [k.strip() for k in re.split(r"[,;]", inner)]
        bibtex_pattern = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]+$")
        if all(bibtex_pattern.match(k) for k in keys if k):
            cite_keys = ", ".join(keys)
            return f"\\cite{{{cite_keys}}}"

        return match.group(0)

    return re.sub(r"\[([^\]]+)\]", _cite_replacer, text)


def _convert_bold_italic(text: str) -> str:
    r"""Converte formattazione inline Markdown in comandi LaTeX.

    - ``**bold**``   → ``\textbf{bold}``
    - ``*italic*``   → ``\textit{italic}``
    - ``***both***`` → ``\textbf{\textit{both}}``
    - `` `code` ``   → ``\texttt{code}``
    """
    # Bold + italic (***text***)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\\textbf{\\textit{\1}}", text)
    # Bold (**text**)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    # Italic (*text*) — non matchare ** o *non seguito da testo*
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", text)
    # Inline code (`code`)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    return text


def _convert_urls(text: str) -> str:
    r"""Converte link Markdown ``[text](url)`` in ``\href{url}{text}``."""
    return re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r"\\href{\2}{\1}",
        text,
    )


def _convert_lists(text: str) -> str:
    r"""Converte elenchi Markdown in ambienti LaTeX itemize/enumerate.

    Elenchi puntati (``- item``, ``* item``) → ``\begin{itemize}``
    Elenchi numerati (``1. item``) → ``\begin{enumerate}``
    """
    lines = text.split("\n")
    result: list[str] = []
    in_list: str | None = None  # "itemize" o "enumerate"

    for line in lines:
        stripped = line.strip()

        # Elenco puntato
        bullet_match = re.match(r"^[-*]\s+(.+)$", stripped)
        # Elenco numerato
        enum_match = re.match(r"^\d+\.\s+(.+)$", stripped)

        if bullet_match:
            if in_list != "itemize":
                if in_list:
                    result.append(f"\\end{{{in_list}}}")
                result.append("\\begin{itemize}")
                in_list = "itemize"
            result.append(f"  \\item {bullet_match.group(1)}")

        elif enum_match:
            if in_list != "enumerate":
                if in_list:
                    result.append(f"\\end{{{in_list}}}")
                result.append("\\begin{enumerate}")
                in_list = "enumerate"
            result.append(f"  \\item {enum_match.group(1)}")

        else:
            if in_list and stripped == "":
                # Riga vuota chiude la lista
                result.append(f"\\end{{{in_list}}}")
                in_list = None
            elif in_list and stripped:
                # Riga non vuota, non lista → chiudi lista
                result.append(f"\\end{{{in_list}}}")
                in_list = None
                result.append(line)
            else:
                result.append(line)

    # Chiudi lista rimasta aperta
    if in_list:
        result.append(f"\\end{{{in_list}}}")

    return "\n".join(result)


def _convert_tables(text: str) -> str:
    r"""Converte tabelle Markdown in ambiente ``tabular`` LaTeX.

    Input::

        | Header1 | Header2 | Header3 |
        |---------|---------|---------|
        | Cell1   | Cell2   | Cell3   |

    Output::

        \begin{table}[htbp]
        \centering
        \begin{tabular}{lll}
        \toprule
        Header1 & Header2 & Header3 \\
        \midrule
        Cell1 & Cell2 & Cell3 \\
        \bottomrule
        \end{tabular}
        \end{table}
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Rileva inizio tabella: riga con |
        if re.match(r"^\s*\|.+\|", line):
            table_lines: list[str] = []

            # Raccogli tutte le righe della tabella
            while i < len(lines) and re.match(r"^\s*\|.+\|", lines[i]):
                table_lines.append(lines[i])
                i += 1

            # Parse della tabella
            if len(table_lines) >= 2:
                result.append(_markdown_table_to_latex(table_lines))
            else:
                result.extend(table_lines)
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def _markdown_table_to_latex(table_lines: list[str]) -> str:
    """Converte righe di una tabella Markdown in LaTeX tabular."""

    def _parse_row(line: str) -> list[str]:
        """Estrae celle da una riga di tabella Markdown."""
        # Rimuovi | iniziale e finale, poi splitta per |
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    def _is_separator(line: str) -> bool:
        """Verifica se la riga è un separatore (es. |---|---|)."""
        return bool(re.match(r"^\s*\|[\s\-:]+(\|[\s\-:]+)*\|\s*$", line))

    def _escape_cell(cell: str) -> str:
        """Escape separators that are literal Markdown-cell content in LaTeX."""
        return re.sub(r"(?<!\\)&", r"\\&", cell)

    # Parse righe
    header_cells: list[str] = []
    data_rows: list[list[str]] = []

    for idx, line in enumerate(table_lines):
        if _is_separator(line):
            continue
        cells = [_escape_cell(cell) for cell in _parse_row(line)]
        if not header_cells:
            header_cells = cells
        else:
            data_rows.append(cells)

    if not header_cells:
        return "\n".join(table_lines)

    n_cols = len(header_cells)
    col_spec = rf"@{{}}*{{{n_cols}}}{{>{{\raggedright\arraybackslash}}X}}@{{}}"

    # Costruisci LaTeX. ``tabularx`` constrains the table to the text width and
    # wraps long benchmark descriptions instead of emitting off-page boxes.
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]

    for row in data_rows:
        # Pad row se necessario
        while len(row) < n_cols:
            row.append("")
        latex_lines.append(" & ".join(row[:n_cols]) + " \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{table}",
    ])

    return "\n".join(latex_lines)


def _convert_code_blocks(text: str) -> str:
    r"""Converte blocchi di codice fenced (````python ... ````) in ``verbatim``.

    Se il linguaggio è specificato, usa l'ambiente verbatim semplice.
    """
    def _code_block_replacer(match: re.Match) -> str:
        code = match.group(2)
        return f"\\begin{{verbatim}}\n{code}\n\\end{{verbatim}}"

    return re.sub(
        r"```(\w*)\n(.*?)```",
        _code_block_replacer,
        text,
        flags=re.DOTALL,
    )


def _convert_horizontal_rules(text: str) -> str:
    r"""Converte ``---`` (regole orizzontali) in ``\noindent\rule{\textwidth}{0.4pt}``."""
    return re.sub(
        r"^-{3,}\s*$",
        r"\\noindent\\rule{\\textwidth}{0.4pt}",
        text,
        flags=re.MULTILINE,
    )


def _convert_footnotes(text: str) -> str:
    r"""Converte footnote Markdown ``[^1]: text`` → ``\footnote{text}``.

    Nota: il supporto è semplificato. Converte referenze inline
    ``text[^1]`` → ``text\footnote{...}`` quando la definizione esiste.
    """
    # Raccogli definizioni
    footnote_defs: dict[str, str] = {}
    text_cleaned = text
    for match in re.finditer(r"^\[\^(\w+)\]:\s*(.+)$", text, re.MULTILINE):
        key = match.group(1)
        content = match.group(2).strip()
        footnote_defs[key] = content
        # Rimuovi la definizione dal testo
        text_cleaned = text_cleaned.replace(match.group(0), "")

    # Sostituisci referenze inline con footnote
    for key, content in footnote_defs.items():
        text_cleaned = text_cleaned.replace(f"[^{key}]", f"\\footnote{{{content}}}")

    return text_cleaned


def _convert_blockquotes(text: str) -> str:
    r"""Converte blockquote Markdown ``> text`` in ``\begin{quote}``."""
    lines = text.split("\n")
    result: list[str] = []
    in_quote = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("> "):
            if not in_quote:
                result.append("\\begin{quote}")
                in_quote = True
            # Rimuovi il > e aggiungi il testo
            result.append(stripped[2:])
        elif stripped == ">" and in_quote:
            result.append("")  # Riga vuota nel blockquote
        else:
            if in_quote:
                result.append("\\end{quote}")
                in_quote = False
            result.append(line)

    if in_quote:
        result.append("\\end{quote}")

    return "\n".join(result)


def _convert_images(text: str) -> str:
    r"""Converte immagini Markdown ``![alt](path)`` in ``\includegraphics``.

    Genera un ambiente ``figure`` con caption dall'alt text.
    """
    def _img_replacer(match: re.Match) -> str:
        alt = match.group(1)
        path = match.group(2)
        # Rimuovi estensione per label
        label = re.sub(r"[^a-zA-Z0-9]", "_", alt.lower())[:30]
        return (
            f"\\begin{{figure}}[htbp]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width=0.8\\textwidth]{{{path}}}\n"
            f"  \\caption{{{alt}}}\n"
            f"  \\label{{fig:{label}}}\n"
            f"\\end{{figure}}"
        )

    # Regex gestisce anche [] annidati nell'alt text (es. ![Figure from [key]](path))
    return re.sub(r"!\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)]+)\)", _img_replacer, text)


def _strip_title_and_shift_headings(md_content: str) -> str:
    """Rimuove il primo heading ``#`` (titolo) e promuove i livelli.

    In un survey Markdown il ``#`` è il titolo (già in ``\\title{}``),
    ``##`` sono le sezioni principali, ecc. Dopo la rimozione del titolo,
    i livelli vengono scalati: ``##`` → ``#``, ``###`` → ``##``, ``####`` → ``###``.
    """
    lines = md_content.split("\n")
    result: list[str] = []
    title_removed = False

    for line in lines:
        # Rimuovi il primo heading di livello 1
        if not title_removed and re.match(r"^#\s+", line) and not line.startswith("##"):
            title_removed = True
            continue  # skip la riga del titolo

        # Promuovi i livelli solo se abbiamo rimosso il titolo
        if title_removed:
            heading_match = re.match(r"^(#{2,5})\s+(.*)", line)
            if heading_match:
                hashes = heading_match.group(1)
                text = heading_match.group(2)
                new_level = len(hashes) - 1  # ## → #, ### → ##, ecc.
                line = "#" * new_level + " " + text

        result.append(line)

    return "\n".join(result)


def _clean_double_blank_lines(text: str) -> str:
    """Rimuove triple+ righe vuote consecutive, lasciandone al massimo due."""
    return re.sub(r"\n{3,}", "\n\n", text)


# ============================================================================
# Pipeline di conversione principale
# ============================================================================

def _convert_body(md_content: str) -> str:
    """Applica la pipeline completa di conversione Markdown → LaTeX sul body.

    L'ordine delle conversioni è importante:
    1. Code blocks (proteggi dal processing successivo)
    2. Immagini (prima dei link per non confondere ``![]()`` con ``[]()`` )
    3. URL / link
    4. Tabelle
    5. Citazioni ``[key]`` → ``\\cite{key}``
    6. Heading
    7. Formattazione inline (bold, italic, code)
    8. Elenchi
    9. Blockquote
    10. Footnote
    11. Regole orizzontali
    12. Pulizia finale
    """
    text = md_content

    # 1. Proteggi code blocks (li convertiamo e poi sono immuni)
    text = _convert_code_blocks(text)

    # 2. Immagini (prima dei link)
    text = _convert_images(text)

    # 3. URL e link Markdown
    text = _convert_urls(text)

    # 4. Tabelle
    text = _convert_tables(text)

    # 5. Citazioni [bibtex_key] → \cite{bibtex_key}
    text = _convert_citations(text)

    # 6. Heading
    text = _convert_headings(text)

    # 7. Formattazione inline
    text = _convert_bold_italic(text)

    # 8. Elenchi
    text = _convert_lists(text)

    # 9. Blockquote
    text = _convert_blockquotes(text)

    # 10. Footnote
    text = _convert_footnotes(text)

    # 11. Regole orizzontali
    text = _convert_horizontal_rules(text)

    # 12. Rimuovi sezioni duplicate (vecchi residui da checkpoint precedenti)
    text = _deduplicate_sections(text)

    # 13. Pulizia
    text = _clean_double_blank_lines(text)

    return text


def _deduplicate_sections(text: str) -> str:
    r"""Rimuove sezioni duplicate e heading ridondanti.

    Gestisce tre casi:
    1. ``\section{X}`` duplicata → tiene la versione con più contenuto
    2. ``\section{X}`` seguita da ``\subsection{X}`` (stesso titolo) → rimuove la subsection
    3. ``\subsection{X}`` duplicata consecutiva → rimuove la seconda
    """
    lines = text.split("\n")
    section_pat = re.compile(r"^\\(section|subsection|subsubsection)\{(.+)\}\s*$")

    # --- Pass 1: Remove \subsection{X} that immediately follows \section{X} with same title ---
    # Also handles \section{X} followed by \section{X} (same level consecutive dup)
    lines_to_remove: set[int] = set()
    i = 0
    while i < len(lines):
        m1 = section_pat.match(lines[i])
        if m1:
            # Look ahead past blank lines for next heading
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip() == "":
                    continue
                m2 = section_pat.match(lines[j])
                if m2:
                    t1 = re.sub(r"^\d+\.\s*", "", m1.group(2)).strip().lower()
                    t2 = re.sub(r"^\d+\.\s*", "", m2.group(2)).strip().lower()
                    if t1 == t2:
                        lvl1, lvl2 = m1.group(1), m2.group(1)
                        if lvl1 == "section" and lvl2 == "subsection":
                            # \section{X} + \subsection{X} → remove subsection
                            lines_to_remove.add(j)
                        elif lvl1 == lvl2:
                            # Same level consecutive dup → remove the first (empty) one
                            lines_to_remove.add(i)
                break  # stop at first non-blank line
        i += 1

    if lines_to_remove:
        lines = [line for idx, line in enumerate(lines) if idx not in lines_to_remove]

    # --- Pass 2: Deduplicate \section{} by title, keeping longest version ---
    section_spans: list[tuple[int, int, str, str]] = []  # (start, end, level, title)
    for i, line in enumerate(lines):
        m = section_pat.match(line)
        if m and m.group(1) == "section":
            section_spans.append((i, -1, m.group(1), m.group(2)))

    if not section_spans:
        return "\n".join(lines)

    for idx in range(len(section_spans)):
        if idx + 1 < len(section_spans):
            section_spans[idx] = (*section_spans[idx][:1], section_spans[idx + 1][0], *section_spans[idx][2:])
        else:
            section_spans[idx] = (*section_spans[idx][:1], len(lines), *section_spans[idx][2:])

    from collections import defaultdict
    title_groups: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    for span in section_spans:
        title_groups[span[3]].append(span)

    lines_to_remove2: set[int] = set()
    # Collect figure environments from removed spans to transplant into kept span
    figures_to_transplant: dict[str, list[str]] = {}  # title → list of figure blocks
    fig_env_re = re.compile(r"\\begin\{figure\}.*?\\end\{figure\}", re.DOTALL)

    for title, spans in title_groups.items():
        if len(spans) <= 1:
            continue
        best = max(spans, key=lambda s: sum(len(lines[i]) for i in range(s[0], s[1])))
        for span in spans:
            if span is not best:
                # Extract figure environments before removing
                removed_text = "\n".join(lines[span[0]:span[1]])
                figs_in_removed = fig_env_re.findall(removed_text)
                if figs_in_removed:
                    figures_to_transplant.setdefault(title, []).extend(figs_in_removed)
                for i in range(span[0], span[1]):
                    lines_to_remove2.add(i)

    if lines_to_remove2:
        lines = [line for i, line in enumerate(lines) if i not in lines_to_remove2]

    # Transplant rescued figures into the kept section (after the heading line)
    if figures_to_transplant:
        result_lines = []
        for line in lines:
            result_lines.append(line)
            m = section_pat.match(line)
            if m and m.group(1) == "section" and m.group(2) in figures_to_transplant:
                title_key = m.group(2)
                for fig_block in figures_to_transplant.pop(title_key):
                    result_lines.append("")
                    result_lines.append(fig_block)
                    result_lines.append("")
        lines = result_lines

    return "\n".join(lines)


def _copy_referenced_figures(latex_content: str, output_dir: Path) -> None:
    r"""Copia i file delle figure referenziate nel LaTeX nella directory di output.

    Cerca tutti i ``\includegraphics{...}`` nel LaTeX e, per ogni path,
    cerca il file sorgente in posizioni comuni e lo copia nella directory
    di output preservando la struttura relativa (es. ``generated_figures/fig.png``
    viene copiato in ``output_dir/generated_figures/fig.png``).
    """
    import shutil

    paths = re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", latex_content)
    if not paths:
        return

    copied = 0

    for rel_path in paths:
        # Destination preserves the relative path structure
        dst = output_dir / rel_path
        if dst.exists():
            continue

        filename = Path(rel_path).name
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Search for the source file in common locations
        for candidate in [
            Path("output") / rel_path,
            Path("output/figures") / filename,
            Path("output/generated_figures") / filename,
            Path(rel_path),
        ]:
            if candidate.exists() and candidate.is_file():
                shutil.copy2(candidate, dst)
                copied += 1
                break

    if copied:
        logger.info("Figure copiate in %s: %d file", output_dir, copied)


def _extract_title_from_md(md_content: str) -> str:
    """Estrae il titolo dal primo heading ``#`` del Markdown."""
    match = re.search(r"^#\s+(.+)$", md_content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Fallback: prima riga non vuota
    for line in md_content.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return "Untitled Survey"


def _extract_abstract_from_md(md_content: str) -> str | None:
    """Estrae l'abstract dal Markdown se presente come sezione."""
    match = re.search(
        r"^##?\s+Abstract\s*\n(.*?)(?=^##?\s+|\Z)",
        md_content,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if match:
        abstract = match.group(1).strip()
        # A horizontal separator often delimits the abstract from front matter;
        # keeping it inside the narrower abstract environment creates an
        # overfull rule. It is presentation markup, not abstract prose.
        abstract = re.sub(r"\n\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", abstract)
        return abstract.strip()
    return None


# ============================================================================
# API Pubblica
# ============================================================================

def markdown_to_latex(
    md_content: str,
    *,
    template: str = "acl",
    bib_file: str = "references",
    title: str | None = None,
    author: str = "",
    date: str | None = None,
) -> str:
    """Converte contenuto Markdown in un documento LaTeX completo.

    Parameters
    ----------
    md_content : str
        Il contenuto Markdown del survey.
    template : str
        Template LaTeX da usare: ``"acl"``, ``"ieeetran"``, ``"plain"``.
    bib_file : str
        Nome del file ``.bib`` (senza estensione) da includere nella bibliografia.
    title : str | None
        Titolo del documento. Se ``None``, viene estratto dal primo heading.
    author : str
        Nome dell'autore.
    date : str | None
        Data del documento. Se ``None``, usa la data odierna.

    Returns
    -------
    str
        Il documento LaTeX completo, pronto per la compilazione.
    """
    if template not in _TEMPLATES:
        raise ValueError(
            f"Template '{template}' non supportato. Usa uno di: {list(_TEMPLATES.keys())}"
        )

    # Estrai titolo se non fornito
    if title is None:
        title = _extract_title_from_md(md_content)

    # Data di default
    if date is None:
        date = datetime.now().strftime("%B %Y")

    # Rimuovi estensione .bib se presente
    if bib_file.endswith(".bib"):
        bib_file = bib_file[:-4]

    # Strip titolo e promuovi heading levels (## → #, ### → ##, ecc.)
    body_md = _strip_title_and_shift_headings(md_content)

    # Converti body
    body = _convert_body(body_md)

    # Gestione abstract per ACL/IEEE (lo estraggiamo e lo inseriamo come \begin{abstract})
    abstract = _extract_abstract_from_md(md_content)
    if abstract:
        abstract_latex = _convert_body(abstract)
        # Rimuovi la sezione abstract dal body (già estratta)
        # Potrebbe essere \section{Abstract} o \subsection{Abstract}
        body = re.sub(
            r"\\(?:sub)*section\{Abstract\}.*?(?=\\(?:sub)*section\{|\Z)",
            "",
            body,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Inserisci abstract come ambiente LaTeX all'inizio
        body = f"\\begin{{abstract}}\n{abstract_latex}\n\\end{{abstract}}\n\n{body}"

    # Applica template
    latex_doc = _TEMPLATES[template].format(
        title=title,
        author=author,
        date=date,
        bibfile=bib_file,
        body=body,
    )

    return latex_doc


def export_latex(
    md_content: str,
    output_path: str | Path | None = None,
    *,
    bib_file: str = "references",
    template: str | None = None,
    title: str | None = None,
    author: str = "",
    date: str | None = None,
    copy_bib: bool = False,
    bib_source_path: str | Path | None = None,
    strict_citations: bool = False,
) -> Path:
    """Converte Markdown in LaTeX e salva su file.

    Parameters
    ----------
    md_content : str
        Contenuto Markdown del survey.
    output_path : str | Path | None
        Percorso del file ``.tex`` di output. Se ``None``, usa
        ``output/survey.tex`` (da ``config.yaml``).
    bib_file : str
        Nome del file ``.bib`` per ``\\bibliography{}``.
    template : str | None
        Template LaTeX. Se ``None``, usa il valore da ``config.yaml``.
    title : str | None
        Titolo del documento.
    author : str
        Autore.
    date : str | None
        Data.
    copy_bib : bool
        Se ``True``, copia il file ``.bib`` nella stessa directory del ``.tex``.
    bib_source_path : str | Path | None
        Percorso del file ``.bib`` sorgente da copiare (se ``copy_bib=True``).
    strict_citations : bool
        Se True, blocca l'export quando testo, BibTeX e manifest DBLP non
        concordano. I flussi finali della pipeline lo impostano sempre.

    Returns
    -------
    Path
        Percorso del file ``.tex`` scritto.
    """
    import shutil

    start_time = time.monotonic()

    # Default da config
    if template is None:
        template = settings.output.latex_template
    if output_path is None:
        output_path = Path(settings.output.base_dir) / "survey.tex"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Export LaTeX: template=%s, output=%s", template, output_path)

    # Converti
    latex_content = markdown_to_latex(
        md_content,
        template=template,
        bib_file=bib_file,
        title=title,
        author=author,
        date=date,
    )

    # Determina il .bib definitivo per validazione citazioni.
    # Usa bib_source_path (se fornito), altrimenti cerca il master .bib
    # nella directory di output e nella directory base.
    bib_for_validation = bib_source_path
    if bib_for_validation is None:
        # Normalizza bib_file (può arrivare con o senza .bib)
        _bib_stem = Path(bib_file).stem if bib_file else "references"
        candidates = [
            output_path.parent / f"{_bib_stem}.bib",
            output_path.parent / "references_final.bib",
            output_path.parent / "references.bib",
            Path(settings.output.base_dir) / "references_final.bib",
            Path(settings.output.base_dir) / "references.bib",
        ]
        for candidate in candidates:
            if candidate.exists():
                bib_for_validation = candidate
                break

    bib_keys = _load_bib_keys(bib_for_validation)

    if strict_citations:
        try:
            integrity_report = validate_citation_integrity(
                md_content,
                bib_for_validation,
                require_dblp=settings.citability.require_dblp,
            )
        except CitationIntegrityError as exc:
            report_path = output_path.parent / "citation_integrity_report.json"
            report_path.write_text(
                json.dumps(exc.report, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            logger.error("%s Report: %s", exc, report_path)
            raise
        report_path = output_path.parent / "citation_integrity_report.json"
        report_path.write_text(
            json.dumps(integrity_report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info(
            "Citation integrity PASS: %d fonti DBLP verificate",
            len(integrity_report["cited_keys"]),
        )

    # Sanitizza solo aspetti tipografici; le citazioni non vengono mai rimosse.
    latex_content = _sanitize_latex(latex_content, bib_keys=bib_keys)
    if bib_keys is not None:
        logger.info("Sanitizzazione LaTeX completata (%d chiavi .bib caricate)", len(bib_keys))
    else:
        logger.info("Sanitizzazione LaTeX completata (nessun .bib per validazione citazioni)")

    # Copia i file delle figure referenziate nella directory di output
    _copy_referenced_figures(latex_content, output_path.parent)

    # Rimuovi figure environments che referenziano file inesistenti
    def _remove_missing_figures(content: str, base_dir: Path) -> str:
        def _check_fig(m: re.Match) -> str:
            fig_path = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", m.group(0))
            if fig_path:
                rel = fig_path.group(1)
                full = base_dir / rel
                if not full.exists():
                    resolved = (base_dir / rel).resolve()
                    if not resolved.exists():
                        logger.warning("Rimossa figura con file mancante: %s", rel)
                        return ""
            return m.group(0)
        return re.sub(
            r"\\begin\{figure\}.*?\\end\{figure\}",
            _check_fig,
            content,
            flags=re.DOTALL,
        )

    latex_content = _remove_missing_figures(latex_content, output_path.parent)

    # Scrivi file
    output_path.write_text(latex_content, encoding="utf-8")
    logger.info("File .tex scritto: %s (%d bytes)", output_path, len(latex_content))

    # Copia .bib se richiesto
    if copy_bib and bib_source_path:
        bib_source = Path(bib_source_path)
        if bib_source.exists():
            bib_dest = output_path.parent / f"{bib_file}.bib"
            if bib_dest.exists() and bib_source.resolve() == bib_dest.resolve():
                logger.debug("File .bib sorgente e destinazione coincidono, skip copia")
            else:
                shutil.copy2(bib_source, bib_dest)
                logger.info("File .bib copiato: %s → %s", bib_source, bib_dest)
        else:
            print_warning(f"File .bib sorgente non trovato: {bib_source}")

    elapsed = time.monotonic() - start_time
    print_success(f"Export LaTeX completato in {elapsed:.1f}s → {output_path}")

    return output_path


def export_latex_from_state(
    state: dict,
    output_dir: str | Path | None = None,
    *,
    template: str | None = None,
    bib_file: str = "references",
    author: str = "",
    copy_bib: bool = True,
    master_bib_path: str | Path | None = None,
    strict_citations: bool = False,
) -> Path | None:
    """Esporta il survey da un ``SurveyState`` completato.

    Estrae ``final_paper`` dallo stato, lo converte in LaTeX e lo salva
    nella directory di output.

    Parameters
    ----------
    state : dict
        Lo ``SurveyState`` contenente ``final_paper``.
    output_dir : str | Path | None
        Directory di output. Se ``None``, usa la directory della versione
        corrente dallo stato.
    template : str | None
        Template LaTeX.
    bib_file : str
        Nome file ``.bib``.
    author : str
        Autore.
    copy_bib : bool
        Se copiare il .bib nella directory di output.
    master_bib_path : str | Path | None
        Percorso del .bib sorgente.

    Returns
    -------
    Path | None
        Percorso del file .tex, o ``None`` se ``final_paper`` è vuoto.
    """
    final_paper = state.get("final_paper", "")
    if not final_paper or not final_paper.strip():
        print_warning("Nessun paper finale trovato nello stato. Export annullato.")
        return None

    # Determina output dir
    if output_dir is None:
        version_info = state.get("version_info")
        if version_info and isinstance(version_info, dict):
            ver_dir = version_info.get("output_dir")
            if ver_dir:
                output_dir = ver_dir
        if output_dir is None:
            output_dir = Path(settings.output.base_dir) / "latest"

    output_dir = Path(output_dir)
    output_path = output_dir / "survey.tex"

    # Titolo dalla versione o dal paper
    title = None
    version_info = state.get("version_info")
    if version_info and isinstance(version_info, dict):
        title = version_info.get("title")

    # Bib source — preferisci references_final.bib (generato da bibliography_generator)
    if master_bib_path is None:
        base = Path(settings.output.base_dir)
        candidates = [
            base / "references_final.bib",
            output_dir / f"{bib_file}.bib",
            base / "references.bib",
        ]
        for candidate in candidates:
            if candidate.exists():
                master_bib_path = candidate
                break

    return export_latex(
        md_content=final_paper,
        output_path=output_path,
        bib_file=bib_file,
        template=template,
        title=title,
        author=author,
        copy_bib=copy_bib,
        bib_source_path=master_bib_path,
        strict_citations=strict_citations,
    )


def get_export_stats(latex_content: str) -> dict[str, Any]:
    """Statistiche sul documento LaTeX generato.

    Returns
    -------
    dict
        Statistiche: sections, subsections, citations, tables, figures,
        total_chars, total_lines.
    """
    return {
        "total_chars": len(latex_content),
        "total_lines": latex_content.count("\n") + 1,
        "sections": len(re.findall(r"\\section\{", latex_content)),
        "subsections": len(re.findall(r"\\subsection\{", latex_content)),
        "subsubsections": len(re.findall(r"\\subsubsection\{", latex_content)),
        "citations": len(re.findall(r"\\cite\{", latex_content)),
        "unique_cite_keys": len(set(re.findall(r"\\cite\{([^}]+)\}", latex_content))),
        "tables": len(re.findall(r"\\begin\{table\}", latex_content)),
        "figures": len(re.findall(r"\\begin\{figure\}", latex_content)),
        "equations": len(re.findall(r"\\begin\{equation\}", latex_content)),
        "itemize_lists": len(re.findall(r"\\begin\{itemize\}", latex_content)),
        "enumerate_lists": len(re.findall(r"\\begin\{enumerate\}", latex_content)),
        "verbatim_blocks": len(re.findall(r"\\begin\{verbatim\}", latex_content)),
        "has_abstract": "\\begin{abstract}" in latex_content,
        "has_bibliography": "\\bibliography{" in latex_content,
    }


def print_export_summary(latex_content: str, output_path: Path | None = None) -> None:
    """Stampa un riepilogo dell'export LaTeX con Rich."""
    stats = get_export_stats(latex_content)

    summary = {
        "Caratteri totali": f"{stats['total_chars']:,}",
        "Righe totali": f"{stats['total_lines']:,}",
        "Sezioni": stats["sections"],
        "Sottosezioni": stats["subsections"],
        "Citazioni (\\cite)": stats["citations"],
        "Chiavi uniche": stats["unique_cite_keys"],
        "Tabelle": stats["tables"],
        "Figure": stats["figures"],
        "Liste puntate": stats["itemize_lists"],
        "Liste numerate": stats["enumerate_lists"],
        "Abstract": "✓" if stats["has_abstract"] else "✗",
        "Bibliografia": "✓" if stats["has_bibliography"] else "✗",
    }

    if output_path:
        summary["Output"] = str(output_path)

    print_table("📄 Export LaTeX — Riepilogo", summary)
