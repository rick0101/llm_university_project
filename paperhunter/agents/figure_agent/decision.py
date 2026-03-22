"""FigureAgent — Decision: decide se una figura è code (chart) o image (concettuale).

Per ogni FigureSpec nel piano, classifica automaticamente in base a regole
deterministiche + fallback LLM per casi ambigui.
"""

from __future__ import annotations

from paperhunter.agents.figure_agent.planner import (
    ChartType,
    FigureSpec,
    FigureType,
)
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Chart type → figure type mapping (deterministic rules)
# ---------------------------------------------------------------------------

_CODE_CHART_TYPES: set[ChartType] = {
    ChartType.BAR,
    ChartType.GROUPED_BAR,
    ChartType.STACKED_BAR,
    ChartType.TIMELINE,
    ChartType.HEATMAP,
    ChartType.RADAR,
    ChartType.LINE,
    ChartType.SCATTER,
    ChartType.PIE,
    ChartType.VENN,
    ChartType.TABLE,
}

_IMAGE_CHART_TYPES: set[ChartType] = {
    ChartType.TAXONOMY,
    ChartType.FLOWCHART,
    ChartType.ARCHITECTURE,
}

# Keywords nei campi description/title che suggeriscono un tipo
_CODE_KEYWORDS = {
    "chart", "plot", "graph", "bar", "line", "scatter", "heatmap",
    "histogram", "distribution", "comparison", "metric", "percentage",
    "count", "trend", "statistic", "rate", "score", "benchmark",
    "correlation", "quantitative", "number", "value", "data",
}

_IMAGE_KEYWORDS = {
    "taxonomy", "hierarchy", "tree", "flowchart", "workflow", "pipeline",
    "architecture", "diagram", "classification", "framework", "overview",
    "conceptual", "schematic", "process", "structure", "relationship",
}


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _has_quantitative_data(spec: FigureSpec) -> bool:
    """Controlla se la spec ha dati quantitativi nei data_points."""
    if not spec.data_points:
        return False
    for dp in spec.data_points:
        if isinstance(dp, dict):
            for v in dp.values():
                if isinstance(v, (int, float)):
                    return True
    return False


def _keyword_score(text: str, keywords: set[str]) -> int:
    """Conta quante keyword sono presenti nel testo."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def decide_figure_type(spec: FigureSpec) -> FigureType:
    """Decide il tipo di generazione per una FigureSpec.

    Logica a 3 livelli:
    1. Chart type esplicito → mapping deterministico
    2. Analisi keyword in title + description
    3. Presenza dati quantitativi → code; altrimenti → image

    Args:
        spec: La specifica della figura da classificare.

    Returns:
        FigureType.CODE o FigureType.IMAGE.
    """
    # Livello 1: chart type esplicito (non OTHER)
    if spec.chart_type != ChartType.OTHER:
        if spec.chart_type in _CODE_CHART_TYPES:
            logger.debug(
                "Figure %s: chart_type %s → CODE (rule-based)",
                spec.figure_id, spec.chart_type.value,
            )
            return FigureType.CODE
        if spec.chart_type in _IMAGE_CHART_TYPES:
            logger.debug(
                "Figure %s: chart_type %s → IMAGE (rule-based)",
                spec.figure_id, spec.chart_type.value,
            )
            return FigureType.IMAGE

    # Livello 2: keyword analysis su title + description
    combined = f"{spec.title} {spec.description}"
    code_score = _keyword_score(combined, _CODE_KEYWORDS)
    image_score = _keyword_score(combined, _IMAGE_KEYWORDS)

    if code_score > image_score:
        logger.debug(
            "Figure %s: keywords code=%d > image=%d → CODE",
            spec.figure_id, code_score, image_score,
        )
        return FigureType.CODE
    if image_score > code_score:
        logger.debug(
            "Figure %s: keywords code=%d < image=%d → IMAGE",
            spec.figure_id, code_score, image_score,
        )
        return FigureType.IMAGE

    # Livello 3: dati quantitativi
    if _has_quantitative_data(spec):
        logger.debug("Figure %s: has quantitative data → CODE", spec.figure_id)
        return FigureType.CODE

    # Default: code (più comuni in survey accademici)
    logger.debug("Figure %s: no strong signal → CODE (default)", spec.figure_id)
    return FigureType.CODE


def classify_all(specs: list[FigureSpec]) -> list[FigureSpec]:
    """Classifica il tipo per tutte le figure e aggiorna figure_type.

    Non modifica le spec con figure_type già settato correttamente dal planner;
    ricalcola solo quelle dove il tipo potrebbe essere incoerente col chart_type.

    Args:
        specs: Lista di FigureSpec da classificare.

    Returns:
        Le stesse spec con figure_type aggiornato.
    """
    updated = 0
    for spec in specs:
        decided = decide_figure_type(spec)
        if spec.figure_type != decided:
            logger.info(
                "Figure %s: reclassified %s → %s",
                spec.figure_id, spec.figure_type.value, decided.value,
            )
            spec.figure_type = decided
            updated += 1

    if updated > 0:
        logger.info("Reclassified %d/%d figures", updated, len(specs))
    else:
        logger.info("All %d figures already correctly classified", len(specs))

    return specs
