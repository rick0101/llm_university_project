"""FigureAgent — Integrator: inserisce riferimenti figure nel Markdown/LaTeX finale.

Dato il paper markdown e le figure renderizzate, inserisce i riferimenti
alle immagini nel punto appropriato di ogni sezione.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from paperhunter.agents.figure_agent.planner import FigurePlan, FigureSpec
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FigureInsertion(BaseModel):
    """Risultato dell'inserimento di una figura nel paper."""

    figure_id: str = Field(..., description="ID della figura inserita")
    section_key: str = Field(..., description="Sezione dove è stata inserita")
    position: str = Field("end", description="Posizione: 'end', 'after_first_paragraph'")
    markdown_ref: str = Field("", description="Riferimento Markdown generato")
    latex_ref: str = Field("", description="Riferimento LaTeX generato")
    inserted: bool = Field(False, description="Se l'inserimento è riuscito")


# ---------------------------------------------------------------------------
# Markdown / LaTeX generation
# ---------------------------------------------------------------------------

def _make_markdown_ref(
    spec: FigureSpec,
    figure_number: int,
    figures_dir: str = "figures",
) -> str:
    """Genera il blocco Markdown per una figura."""
    path = f"{figures_dir}/{spec.figure_id}.png"
    caption = spec.caption or f"Figure {figure_number}: {spec.title}"
    if not caption.startswith("Figure"):
        caption = f"Figure {figure_number}: {caption}"

    return (
        f"\n\n"
        f"![{caption}]({path})\n\n"
        f"*{caption}*\n"
    )


def _make_latex_ref(
    spec: FigureSpec,
    figure_number: int,
    figures_dir: str = "figures",
) -> str:
    """Genera il blocco LaTeX per una figura."""
    path = f"{figures_dir}/{spec.figure_id}.png"
    caption = spec.caption or spec.title
    label = spec.figure_id.replace("fig_", "fig:")

    return (
        f"\n\n\\begin{{figure}}[htbp]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width=0.85\\textwidth]{{{path}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"\\end{{figure}}\n"
    )


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _find_section_in_markdown(
    markdown: str,
    section_key: str,
) -> tuple[int, int] | None:
    """Trova inizio e fine di una sezione nel Markdown.

    Cerca heading ## o ### che contengano la section_key (dopo trasformazione).

    Returns:
        Tupla (start_pos, end_pos) o None se non trovata.
    """
    # Trasforma section_key in pattern per il titolo
    # Usa \w+ per estrarre solo parole alfanumeriche, così "Training-Time"
    # diventa ["training", "time"] e matcha correttamente con gli heading.
    key_words = re.findall(r"\w+", section_key.lower())

    # Cerca tutti gli heading
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    headings = list(heading_pattern.finditer(markdown))

    for i, match in enumerate(headings):
        heading_text = match.group(2).lower()
        heading_words = re.findall(r"\w+", heading_text)

        # Match se tutti i keyword della section_key sono nel titolo
        if all(kw in heading_words for kw in key_words):
            start = match.start()
            # Fine = inizio del prossimo heading allo stesso livello o superiore
            heading_level = len(match.group(1))
            end = len(markdown)
            for j in range(i + 1, len(headings)):
                next_level = len(headings[j].group(1))
                if next_level <= heading_level:
                    end = headings[j].start()
                    break
            return start, end

    return None


def _find_insertion_point(
    markdown: str,
    section_start: int,
    section_end: int,
    position: str = "end",
) -> int:
    """Trova il punto esatto dove inserire la figura nella sezione.

    Args:
        markdown: Testo completo del paper.
        section_start: Inizio della sezione.
        section_end: Fine della sezione.
        position: 'end' o 'after_first_paragraph'.

    Returns:
        Posizione di inserimento nel testo.
    """
    if position == "after_first_paragraph":
        # Trova la fine del primo paragrafo (doppio newline dopo l'heading)
        section_text = markdown[section_start:section_end]
        # Salta l'heading
        heading_end = section_text.find("\n")
        if heading_end != -1:
            # Trova il primo doppio newline dopo l'heading
            para_end = section_text.find("\n\n", heading_end + 1)
            if para_end != -1:
                return section_start + para_end

    # Default: fine della sezione (prima del prossimo heading)
    return section_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def integrate_figures(
    paper_markdown: str,
    plan: FigurePlan,
    rendered_figures: dict[str, tuple[bool, Path | None]],
    figures_dir: str = "figures",
    output_format: str = "markdown",
    position: str = "end",
) -> tuple[str, list[FigureInsertion]]:
    """Inserisce i riferimenti alle figure nel paper.

    Args:
        paper_markdown: Markdown del paper originale.
        plan: Piano delle figure.
        rendered_figures: Dict figure_id → (success, png_path) dal renderer.
        figures_dir: Percorso relativo per i riferimenti immagine.
        output_format: 'markdown' o 'latex'.
        position: 'end' (fine sezione) o 'after_first_paragraph'.

    Returns:
        Tupla (paper_con_figure, lista_inserimenti).
    """
    insertions: list[FigureInsertion] = []
    figure_number = 1

    # Ordina per priorità e poi per ordine sezione
    specs_to_insert = [
        s for s in plan.figures
        if s.figure_id in rendered_figures and rendered_figures[s.figure_id][0]
    ]
    specs_to_insert.sort(key=lambda s: s.priority)

    if not specs_to_insert:
        logger.info("No rendered figures to integrate")
        return paper_markdown, insertions

    # Inserisci dalla fine verso l'inizio per non invalidare le posizioni
    insert_ops: list[tuple[int, str, FigureInsertion]] = []

    for spec in specs_to_insert:
        section_range = _find_section_in_markdown(paper_markdown, spec.section_key)
        if section_range is None:
            logger.warning(
                "Section '%s' not found for figure %s",
                spec.section_key, spec.figure_id,
            )
            insertions.append(FigureInsertion(
                figure_id=spec.figure_id,
                section_key=spec.section_key,
                inserted=False,
            ))
            continue

        section_start, section_end = section_range

        if output_format == "latex":
            ref = _make_latex_ref(spec, figure_number, figures_dir)
        else:
            ref = _make_markdown_ref(spec, figure_number, figures_dir)

        ins_point = _find_insertion_point(
            paper_markdown, section_start, section_end, position,
        )

        insertion = FigureInsertion(
            figure_id=spec.figure_id,
            section_key=spec.section_key,
            position=position,
            markdown_ref=_make_markdown_ref(spec, figure_number, figures_dir),
            latex_ref=_make_latex_ref(spec, figure_number, figures_dir),
            inserted=True,
        )

        insert_ops.append((ins_point, ref, insertion))
        insertions.append(insertion)
        figure_number += 1

    # Inserisci dalla fine per mantenere validità posizioni
    insert_ops.sort(key=lambda x: x[0], reverse=True)
    result = paper_markdown
    for pos, ref, _ in insert_ops:
        result = result[:pos] + ref + result[pos:]

    succeeded = sum(1 for ins in insertions if ins.inserted)
    logger.info(
        "Integrated %d/%d figures into paper",
        succeeded, len(specs_to_insert),
    )

    return result, insertions


def copy_figures_to_output(
    rendered_figures: dict[str, tuple[bool, Path | None]],
    output_dir: str | Path,
    figures_subdir: str = "figures",
) -> dict[str, Path]:
    """Copia le figure renderizzate nella directory di output.

    Args:
        rendered_figures: Dict figure_id → (success, png_path).
        output_dir: Directory base di output.
        figures_subdir: Sottocartella per le figure.

    Returns:
        Dict figure_id → path di destinazione.
    """
    import shutil

    dest_dir = Path(output_dir) / figures_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Path] = {}
    for fig_id, (success, src_path) in rendered_figures.items():
        if success and src_path and src_path.exists():
            dest_path = dest_dir / src_path.name
            shutil.copy2(src_path, dest_path)
            copied[fig_id] = dest_path
            logger.debug("Copied %s → %s", src_path, dest_path)

    logger.info("Copied %d figures to %s", len(copied), dest_dir)
    return copied
