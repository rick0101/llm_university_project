"""FigureAgent — Generazione automatica di figure scientifiche per il survey.

Sub-package che analizza il paper generato e produce figure accademiche:
- Code figures (bar chart, timeline, heatmap → Matplotlib/Seaborn)
- Image figures (tassonomie, flowchart → prompt LLM)
"""

from paperhunter.agents.figure_agent.planner import (
    FigureSpec,
    FigurePlan,
    plan_figures,
)

__all__ = [
    "FigureSpec",
    "FigurePlan",
    "plan_figures",
]
