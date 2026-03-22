"""FigureAgent — Critic: valuta qualità, leggibilità e stile accademico delle figure.

Analizza le figure generate verificando: leggibilità label/assi, correttezza
dati vs fonti, stile coerente, e restituisce APPROVED o NEEDS_REVISION.
"""

from __future__ import annotations

import json
import re
import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from paperhunter.agents.figure_agent.planner import FigureSpec
from paperhunter.config import settings
from paperhunter.utils.llm_factory import get_llm, llm_call_with_retry
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FigureVerdict(str, Enum):
    """Verdetto del critic sulla figura."""
    APPROVED = "APPROVED"
    NEEDS_REVISION = "NEEDS_REVISION"


class FigureCritique(BaseModel):
    """Risultato della valutazione di una singola figura."""

    figure_id: str = Field(..., description="ID della figura valutata")
    verdict: FigureVerdict = Field(..., description="APPROVED o NEEDS_REVISION")
    overall_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Score complessivo 0.0-1.0"
    )
    readability_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Leggibilità: label, assi, font size"
    )
    accuracy_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Accuratezza dati rispetto alle fonti"
    )
    style_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Aderenza stile accademico"
    )
    feedback: str = Field("", description="Feedback dettagliato per revisione")
    issues: list[str] = Field(
        default_factory=list,
        description="Lista di problemi specifici trovati"
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_CRITIC_PROMPT = """\
You are an expert reviewer of scientific figures for academic survey papers. \
Evaluate the following figure based on its specification and generated code.

## Figure Specification

- **Title:** {title}
- **Chart Type:** {chart_type}
- **Description:** {description}
- **Caption:** {caption}

## Generated Code

```python
{code}
```

## Evaluation Criteria

Rate each criterion from 0.0 to 1.0:

### 1. Readability (readability_score)
- Labels are clear and not overlapping
- Font sizes are appropriate (10-14pt for academic papers)
- Colors have sufficient contrast
- Axes have proper labels and units
- Legend is clear (if applicable)

### 2. Data Accuracy (accuracy_score)
- Data points match the specification
- Scale is appropriate and not misleading
- No truncated axes that distort proportions
- Proper number formatting

### 3. Academic Style (style_score)
- Professional color scheme (not garish)
- Clean, minimalist design
- DPI >= 300 for print quality
- Proper title placement
- Grid lines are subtle (if present)
- Appropriate figure size

## Required Output Format

Return a JSON object:
```json
{{
  "verdict": "APPROVED",
  "overall_score": 0.85,
  "readability_score": 0.9,
  "accuracy_score": 0.8,
  "style_score": 0.85,
  "feedback": "Overall good figure. Minor suggestions: ...",
  "issues": ["Label overlap on x-axis", "Consider adding units to y-axis"]
}}
```

Rules:
- "APPROVED" if overall_score >= 0.7 AND no critical issues
- "NEEDS_REVISION" if overall_score < 0.7 OR has critical issues
- overall_score = weighted average: readability×0.35 + accuracy×0.35 + style×0.30
- Be specific in feedback — what exactly should change
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_critique(raw_response: str) -> dict | None:
    """Parse la risposta JSON del critic con fallback."""
    text = raw_response.strip()

    for attempt_text in [text, re.sub(r",\s*([}\]])", r"\1", text)]:
        match = re.search(r"\{.*\}", attempt_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if "verdict" in data or "overall_score" in data:
                    return data
            except json.JSONDecodeError:
                continue

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def critique_figure(
    spec: FigureSpec,
    code: str,
    force_model: str | None = None,
) -> FigureCritique:
    """Valuta la qualità di una figura generata.

    Args:
        spec: La specifica della figura.
        code: Il codice Python generato.
        force_model: Override modello LLM.

    Returns:
        FigureCritique con score, verdict e feedback.
    """
    prompt = _CRITIC_PROMPT.format(
        title=spec.title,
        chart_type=spec.chart_type.value,
        description=spec.description,
        caption=spec.caption or spec.title,
        code=code,
    )

    tier = force_model or "flash"
    llm = get_llm(tier, max_tokens=2048)

    logger.info("Critiquing figure %s with %s...", spec.figure_id, tier)
    t0 = time.time()

    raw_response = llm_call_with_retry(
        llm, prompt,
        max_retries=3,
        base_delay=2.0,
        max_delay=30.0,
    )

    elapsed = time.time() - t0
    logger.info("Figure critique for %s completed in %.1fs", spec.figure_id, elapsed)

    data = _parse_critique(raw_response)
    if data is None:
        logger.warning("Failed to parse critique for %s, returning default NEEDS_REVISION", spec.figure_id)
        return FigureCritique(
            figure_id=spec.figure_id,
            verdict=FigureVerdict.NEEDS_REVISION,
            feedback="Failed to parse LLM critique response.",
            issues=["Critique parsing failed"],
        )

    # Calcola verdict se non fornito
    overall = data.get("overall_score", 0.0)
    if "verdict" not in data:
        data["verdict"] = "APPROVED" if overall >= 0.7 else "NEEDS_REVISION"

    return FigureCritique(
        figure_id=spec.figure_id,
        verdict=FigureVerdict(data.get("verdict", "NEEDS_REVISION")),
        overall_score=min(1.0, max(0.0, data.get("overall_score", 0.0))),
        readability_score=min(1.0, max(0.0, data.get("readability_score", 0.0))),
        accuracy_score=min(1.0, max(0.0, data.get("accuracy_score", 0.0))),
        style_score=min(1.0, max(0.0, data.get("style_score", 0.0))),
        feedback=data.get("feedback", ""),
        issues=data.get("issues", []),
    )


def critique_code_only(
    spec: FigureSpec,
    code: str,
    threshold: float = 0.7,
) -> FigureCritique:
    """Valutazione rapida basata solo sull'analisi statica del codice (no LLM).

    Controlla pattern comuni di qualità senza chiamare l'LLM.
    Utile come pre-filtro rapido.

    Args:
        spec: La specifica della figura.
        code: Il codice Python generato.
        threshold: Soglia per APPROVED.

    Returns:
        FigureCritique con score basato su euristiche.
    """
    issues: list[str] = []
    scores = {"readability": 1.0, "accuracy": 1.0, "style": 1.0}

    # Check DPI
    if "dpi=300" not in code and "dpi=600" not in code:
        if "dpi=" in code:
            issues.append("DPI should be at least 300 for academic figures")
            scores["style"] -= 0.2
        else:
            issues.append("No DPI specified — default may be too low")
            scores["style"] -= 0.1

    # Check figure size
    if "figsize" not in code:
        issues.append("No explicit figsize — may produce awkward proportions")
        scores["readability"] -= 0.1

    # Check labels
    if "set_xlabel" not in code and "xlabel" not in code:
        issues.append("Missing x-axis label")
        scores["readability"] -= 0.15
    if "set_ylabel" not in code and "ylabel" not in code:
        issues.append("Missing y-axis label")
        scores["readability"] -= 0.15

    # Check title
    if "set_title" not in code and "suptitle" not in code and "title(" not in code:
        issues.append("Missing figure title")
        scores["readability"] -= 0.15

    # Check tight_layout or bbox_inches
    if "tight_layout" not in code and "bbox_inches" not in code:
        issues.append("No tight_layout or bbox_inches — labels may be cut off")
        scores["style"] -= 0.1

    # Check for plt.show() (should NOT be present in headless mode)
    if "plt.show()" in code:
        issues.append("plt.show() should be removed for headless rendering")
        scores["style"] -= 0.1

    # Check backend
    if 'Agg' not in code and 'agg' not in code:
        issues.append("Consider using matplotlib.use('Agg') for headless rendering")

    # Compute overall
    readability = max(0.0, scores["readability"])
    accuracy = max(0.0, scores["accuracy"])
    style = max(0.0, scores["style"])
    overall = readability * 0.35 + accuracy * 0.35 + style * 0.30

    verdict = FigureVerdict.APPROVED if overall >= threshold and not any(
        "Missing" in i for i in issues
    ) else FigureVerdict.NEEDS_REVISION

    return FigureCritique(
        figure_id=spec.figure_id,
        verdict=verdict,
        overall_score=round(overall, 2),
        readability_score=round(readability, 2),
        accuracy_score=round(accuracy, 2),
        style_score=round(style, 2),
        feedback="; ".join(issues) if issues else "Code passes all static checks.",
        issues=issues,
    )
