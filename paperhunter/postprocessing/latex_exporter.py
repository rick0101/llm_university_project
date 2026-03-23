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


# ============================================================================
# Template LaTeX
# ============================================================================

_TEMPLATE_ACL = r"""\documentclass[11pt,a4paper]{{article}}
\usepackage[hyperref]{{ntheorem}}
\usepackage{{times}}
\usepackage{{latexsym}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{url}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{microtype}}
\usepackage{{natbib}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

\begin{{document}}
\maketitle

{body}

\bibliographystyle{{plainnat}}
\bibliography{{{bibfile}}}

\end{{document}}
"""

_TEMPLATE_IEEETRAN = r"""\documentclass[conference]{{IEEEtran}}
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{textcomp}}
\usepackage{{xcolor}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{hyperref}}
\usepackage{{url}}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

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
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{url}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{hyperref}}
\usepackage{{natbib}}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

\begin{{document}}
\maketitle

{body}

\bibliographystyle{{plainnat}}
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
        keys = [k.strip() for k in inner.split(",")]
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

    # Parse righe
    header_cells: list[str] = []
    data_rows: list[list[str]] = []

    for idx, line in enumerate(table_lines):
        if _is_separator(line):
            continue
        cells = _parse_row(line)
        if not header_cells:
            header_cells = cells
        else:
            data_rows.append(cells)

    if not header_cells:
        return "\n".join(table_lines)

    n_cols = len(header_cells)
    col_spec = "l" * n_cols

    # Costruisci LaTeX
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
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
        "\\end{tabular}",
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

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img_replacer, text)


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

    # 12. Pulizia
    text = _clean_double_blank_lines(text)

    return text


# ============================================================================
# Estrazione metadati dal Markdown
# ============================================================================

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
        return match.group(1).strip()
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

    # Bib source
    if master_bib_path is None:
        # Cerca nella stessa directory un .bib
        possible_bib = output_dir / f"{bib_file}.bib"
        if possible_bib.exists():
            master_bib_path = possible_bib
        else:
            # Fallback al master .bib
            master_bib_path = Path(settings.output.base_dir) / "references.bib"

    return export_latex(
        md_content=final_paper,
        output_path=output_path,
        bib_file=bib_file,
        template=template,
        title=title,
        author=author,
        copy_bib=copy_bib,
        bib_source_path=master_bib_path,
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
