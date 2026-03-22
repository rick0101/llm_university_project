"""FigureAgent — Planner: analizza il paper e pianifica le figure necessarie.

Dato il Markdown del paper finale e le L2 summary, produce un piano di
figure ordinate per priorità con tipo, posizione, dati sorgente e descrizione.
"""

from __future__ import annotations

import json
import re
import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from paperhunter.config import settings
from paperhunter.utils.llm_factory import get_llm, llm_call_with_retry
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ChartType(str, Enum):
    """Tipi di chart supportati per code figures."""
    BAR = "bar"
    GROUPED_BAR = "grouped_bar"
    STACKED_BAR = "stacked_bar"
    TIMELINE = "timeline"
    HEATMAP = "heatmap"
    RADAR = "radar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    TAXONOMY = "taxonomy"
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture"
    VENN = "venn"
    TABLE = "table"
    OTHER = "other"


class FigureType(str, Enum):
    """Tipo di generazione della figura."""
    CODE = "code"    # Matplotlib/Seaborn → PNG
    IMAGE = "image"  # Prompt LLM/TikZ → concettuale


class FigureSpec(BaseModel):
    """Specifica di una singola figura da generare."""

    figure_id: str = Field(
        ...,
        description="ID univoco (es. 'fig_attack_timeline')"
    )
    figure_type: FigureType = Field(
        ...,
        description="Tipo: 'code' (chart via Matplotlib) o 'image' (diagramma concettuale)"
    )
    chart_type: ChartType = Field(
        ChartType.OTHER,
        description="Tipo specifico di chart (bar, timeline, heatmap, ecc.)"
    )
    title: str = Field(
        ...,
        description="Titolo della figura (es. 'Timeline of Adversarial Attacks on LLMs')"
    )
    caption: str = Field(
        "",
        description="Didascalia accademica per la figura"
    )
    description: str = Field(
        ...,
        description="Descrizione dettagliata di cosa la figura deve mostrare"
    )
    section_key: str = Field(
        ...,
        description="Chiave della sezione dove inserire la figura"
    )
    data_sources: list[str] = Field(
        default_factory=list,
        description="Fonti dati: nomi L2 summary, property graph, etc."
    )
    data_points: list[dict] = Field(
        default_factory=list,
        description="Dati estratti per la generazione (label, valori, ecc.)"
    )
    priority: int = Field(
        1,
        ge=1, le=5,
        description="Priorità 1 (alta) → 5 (bassa)"
    )
    rationale: str = Field(
        "",
        description="Motivazione per cui questa figura è utile nel survey"
    )

    @field_validator("figure_id")
    @classmethod
    def validate_figure_id(cls, v: str) -> str:
        if not re.match(r"^fig_[a-z0-9_]+$", v):
            raise ValueError(
                f"figure_id deve seguire il pattern 'fig_xxx_yyy': {v!r}"
            )
        return v


class FigurePlan(BaseModel):
    """Piano completo delle figure da generare per il survey."""

    figures: list[FigureSpec] = Field(
        default_factory=list,
        description="Lista ordinata di figure da generare"
    )
    total_figures: int = Field(0, description="Numero totale di figure pianificate")
    generated_at: str = Field("", description="Timestamp di generazione")
    model_used: str = Field("", description="Modello LLM usato per la pianificazione")

    def code_figures(self) -> list[FigureSpec]:
        return [f for f in self.figures if f.figure_type == FigureType.CODE]

    def image_figures(self) -> list[FigureSpec]:
        return [f for f in self.figures if f.figure_type == FigureType.IMAGE]

    def for_section(self, section_key: str) -> list[FigureSpec]:
        return [f for f in self.figures if f.section_key == section_key]


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_FIGURE_PLANNER_PROMPT = """\
You are an expert academic researcher specializing in data visualization \
for scientific survey papers on **LLM Security**.

## Task

Analyze the following survey paper and its category-level summaries, \
then produce a detailed plan of **scientific figures** that would enhance \
the paper's quality and readability.

## Paper Content (Markdown)

{paper_content}

## L2 Category Summaries

{l2_summaries}

## Figure Generation Constraints

- Minimum figures: {min_figures}
- Maximum figures: {max_figures}
- Each figure MUST serve a clear analytical purpose (not decorative)
- Prefer quantitative figures (charts, plots) over conceptual diagrams
- Every figure must reference data from the paper or L2 summaries

## Supported Figure Types

### Code Figures (type: "code") — Generated via Matplotlib/Seaborn:
- **bar / grouped_bar / stacked_bar**: Compare metrics across categories
- **timeline**: Show temporal evolution of research topics
- **heatmap**: Attack×Defense effectiveness matrix
- **radar**: Multi-dimensional method comparison
- **line**: Trends over time (publication count, ASR rates)
- **scatter**: Correlation between variables

### Image Figures (type: "image") — Conceptual diagrams:
- **taxonomy**: Hierarchical threat/defense classification
- **flowchart**: Attack pipeline or defense workflow
- **architecture**: System architecture diagrams

## Required Output Format

Return a JSON object with this exact structure:
```json
{{
  "figures": [
    {{
      "figure_id": "fig_attack_timeline",
      "figure_type": "code",
      "chart_type": "timeline",
      "title": "Timeline of Major Adversarial Attacks on LLMs (2022–2026)",
      "caption": "Figure N: Chronological overview of key adversarial attack ...",
      "description": "A horizontal timeline showing major attack papers ...",
      "section_key": "adversarial_attacks",
      "data_sources": ["l2_adversarial_attacks"],
      "data_points": [
        {{"label": "GCG (Zou et al., 2023)", "year": 2023, "category": "gradient-based"}},
        {{"label": "AutoDAN (Liu et al., 2024)", "year": 2024, "category": "genetic"}}
      ],
      "priority": 1,
      "rationale": "Provides temporal context for the evolution of attack strategies"
    }}
  ]
}}
```

IMPORTANT:
- `figure_id` must follow pattern `fig_<snake_case_name>`
- `data_points` should contain actual data extracted from the paper/summaries
- Order figures by priority (1 = most important)
- Every figure must be placed in an existing section via `section_key`
- Prefer data-driven code figures over conceptual image figures
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_figure_plan(raw_response: str) -> list[dict] | None:
    """Parse la risposta JSON del planner con fallback multi-livello."""

    text = raw_response.strip()

    # Livello 1: parse diretto
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "figures" in data:
            return data["figures"]
    except json.JSONDecodeError:
        pass

    # Livello 2: estrai primo JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "figures" in data:
                return data["figures"]
        except json.JSONDecodeError:
            pass

    # Livello 3: fix trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "figures" in data:
                return data["figures"]
        except json.JSONDecodeError:
            pass

    # Livello 4: estrai JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list) and len(arr) > 0:
                return arr
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse figure plan from LLM response")
    return None


def _validate_figure_specs(raw_figures: list[dict]) -> list[FigureSpec]:
    """Valida e converte le figure raw in FigureSpec, scartando quelle invalide."""
    specs: list[FigureSpec] = []
    for i, fig in enumerate(raw_figures):
        try:
            # Assicura figure_id se mancante
            if "figure_id" not in fig:
                fig["figure_id"] = f"fig_auto_{i:03d}"
            # Normalizza figure_id
            fid = fig["figure_id"]
            if not fid.startswith("fig_"):
                fig["figure_id"] = f"fig_{fid}"
            fig["figure_id"] = re.sub(r"[^a-z0-9_]", "_", fig["figure_id"].lower())

            spec = FigureSpec(**fig)
            specs.append(spec)
        except Exception as e:
            logger.warning("Skipping invalid figure spec #%d: %s", i, e)
    return specs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_figures(
    paper_markdown: str,
    l2_summaries: list[dict] | None = None,
    min_figures: int = 3,
    max_figures: int = 8,
    force_model: str | None = None,
) -> FigurePlan:
    """Analizza il paper e genera un piano di figure scientifiche.

    Args:
        paper_markdown: Contenuto Markdown del paper assemblato.
        l2_summaries: Lista di L2 summary dict (opzionale).
        min_figures: Minimo figure da pianificare.
        max_figures: Massimo figure da pianificare.
        force_model: Override modello LLM ('flash' o 'pro').

    Returns:
        FigurePlan con la lista di FigureSpec ordinate per priorità.
    """
    from rich.console import Console
    console = Console()

    logger.info(
        "Planning figures: paper=%d chars, l2=%d summaries, range=[%d, %d]",
        len(paper_markdown), len(l2_summaries or []), min_figures, max_figures,
    )

    # Prepara L2 text
    l2_text = ""
    if l2_summaries:
        for s in l2_summaries:
            cat = s.get("category", "unknown")
            synthesis = s.get("synthesis", "")
            taxonomy = s.get("taxonomy", "")
            gaps = s.get("open_gaps", "")
            l2_text += f"### {cat}\n{synthesis}\n"
            if taxonomy:
                l2_text += f"**Taxonomy:** {taxonomy}\n"
            if gaps:
                l2_text += f"**Open Gaps:** {gaps}\n"
            l2_text += "\n"

    # Tronca paper se troppo lungo per il contesto
    max_paper_chars = 60_000
    if len(paper_markdown) > max_paper_chars:
        paper_content = paper_markdown[:max_paper_chars] + "\n\n[... truncated ...]"
        logger.info("Paper truncated to %d chars for figure planning", max_paper_chars)
    else:
        paper_content = paper_markdown

    prompt = _FIGURE_PLANNER_PROMPT.format(
        paper_content=paper_content,
        l2_summaries=l2_text or "(No L2 summaries available)",
        min_figures=min_figures,
        max_figures=max_figures,
    )

    # Seleziona modello — pro di default per analisi complessa
    tier = force_model or settings.agents.planner_model
    llm = get_llm(tier, max_tokens=8192)

    logger.info("Calling LLM (%s) for figure planning...", tier)
    console.print(f"[bold cyan]🎨 Planning figures with {tier.upper()}...[/bold cyan]")

    t0 = time.time()
    max_retries = settings.summarization.pro_max_retries
    base_delay = settings.summarization.pro_base_delay

    raw_response = llm_call_with_retry(
        llm, prompt,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=120.0,
    )

    elapsed = time.time() - t0
    logger.info("Figure planning LLM call completed in %.1fs", elapsed)

    # Parse e valida
    raw_figures = _parse_figure_plan(raw_response)
    if raw_figures is None:
        logger.error("Failed to parse figure plan from LLM response")
        console.print("[bold red]❌ Failed to parse figure plan[/bold red]")
        return FigurePlan()

    specs = _validate_figure_specs(raw_figures)

    # Limita al range richiesto
    specs.sort(key=lambda f: f.priority)
    if len(specs) > max_figures:
        specs = specs[:max_figures]

    if len(specs) < min_figures:
        logger.warning(
            "Only %d valid figures planned (min=%d)", len(specs), min_figures
        )

    from datetime import datetime, timezone
    plan = FigurePlan(
        figures=specs,
        total_figures=len(specs),
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_used=tier,
    )

    console.print(
        f"[bold green]✅ Figure plan: {plan.total_figures} figures "
        f"({len(plan.code_figures())} code, {len(plan.image_figures())} image)[/bold green]"
    )

    for spec in specs:
        icon = "📊" if spec.figure_type == FigureType.CODE else "🖼️"
        console.print(
            f"  {icon} [{spec.priority}] {spec.figure_id}: {spec.title}"
        )

    return plan


def save_figure_plan(plan: FigurePlan, output_dir: str | Path) -> Path:
    """Salva il piano figure su file JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "figure_plan.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, indent=2, ensure_ascii=False, default=str)
    logger.info("Figure plan saved to %s", path)
    return path


def load_figure_plan(path: str | Path) -> FigurePlan:
    """Carica un piano figure da file JSON."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FigurePlan(**data)
