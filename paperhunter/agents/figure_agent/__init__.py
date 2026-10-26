"""FigureAgent — Generazione automatica di figure scientifiche per il survey.

Sub-package che analizza il paper generato e produce figure accademiche:
- Code figures (bar chart, timeline, heatmap → Matplotlib/Seaborn)
- Image figures (tassonomie, flowchart → prompt LLM)
"""

from paperhunter.agents.figure_agent.planner import (
    ChartType,
    FigurePlan,
    FigureSpec,
    FigureType,
    plan_figures,
    save_figure_plan,
    load_figure_plan,
)
from paperhunter.agents.figure_agent.decision import (
    decide_figure_type,
    classify_all,
)
from paperhunter.agents.figure_agent.critic import (
    FigureCritique,
    FigureVerdict,
    critique_figure,
    critique_code_only,
)
from paperhunter.agents.figure_agent.renderer import (
    render_figure,
    render_all_code_figures,
)
from paperhunter.agents.figure_agent.integrator import (
    FigureInsertion,
    integrate_figures,
    copy_figures_to_output,
)

__all__ = [
    # planner
    "ChartType", "FigurePlan", "FigureSpec", "FigureType",
    "plan_figures", "save_figure_plan", "load_figure_plan",
    # decision
    "decide_figure_type", "classify_all",
    # critic
    "FigureCritique", "FigureVerdict", "critique_figure", "critique_code_only",
    # renderer
    "render_figure", "render_all_code_figures",
    # integrator
    "FigureInsertion", "integrate_figures", "copy_figures_to_output",
]
