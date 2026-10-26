"""FigureAgent — CodeGen: genera codice Matplotlib/Seaborn per le figure.

Dato un FigureSpec, genera uno script Python self-contained che produce
un'immagine PNG accademica (DPI 300, stile paper-ready).
"""

from __future__ import annotations

import json
import re
import time

from paperhunter.agents.figure_agent.planner import ChartType, FigureSpec, FigureType
from paperhunter.config import settings
from paperhunter.utils.llm_factory import get_llm, llm_call_with_retry
from paperhunter.utils.logging_setup import get_logger
from paperhunter.utils.prompt_loader import get_prompt

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_CODEGEN_PROMPT = get_prompt("FIGURE_CODEGEN_PROMPT")

_ERROR_FEEDBACK_TEMPLATE = get_prompt("FIGURE_ERROR_FEEDBACK")


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def _extract_python_code(raw_response: str) -> str | None:
    """Estrai codice Python dal response LLM."""
    # Cerca blocco ```python ... ```
    match = re.search(r"```python\s*\n(.*?)```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Cerca blocco ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", raw_response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "import" in code and ("plt" in code or "matplotlib" in code):
            return code

    # Se il response intero sembra codice Python
    if "import matplotlib" in raw_response or "import seaborn" in raw_response:
        lines = raw_response.strip().split("\n")
        code_lines = [l for l in lines if not l.startswith("#") or l.startswith("#!/")]
        if any("plt.savefig" in l for l in code_lines):
            return raw_response.strip()

    return None


def _validate_code(code: str, output_path: str) -> str:
    """Valida e correggi il codice generato."""
    lines = code.split("\n")

    # Rimuovi plt.show() — causa hang in headless mode
    lines = [l for l in lines if "plt.show()" not in l]

    # Assicura backend Agg in testa (prima di qualsiasi import matplotlib)
    has_agg = any("matplotlib.use" in l for l in lines)
    if not has_agg:
        agg_lines = ["import matplotlib", "matplotlib.use('Agg')"]
        # Inserisci prima del primo import matplotlib/pyplot/seaborn
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(stripped.startswith(f"import {m}") or stripped.startswith(f"from {m}")
                   for m in ("matplotlib", "seaborn", "mpl_toolkits")):
                insert_idx = i
                break
        # Se il primo import è già 'import matplotlib', serve solo use('Agg')
        if lines[insert_idx].strip() == "import matplotlib":
            lines.insert(insert_idx + 1, "matplotlib.use('Agg')")
        else:
            for j, agg_line in enumerate(agg_lines):
                lines.insert(insert_idx + j, agg_line)

    # Assicura OUTPUT_PATH
    if "OUTPUT_PATH" not in code:
        lines.insert(0, f'OUTPUT_PATH = "{output_path}"')
    else:
        # Aggiorna il path se diverso
        for i, line in enumerate(lines):
            if line.strip().startswith("OUTPUT_PATH"):
                lines[i] = f'OUTPUT_PATH = "{output_path}"'
                break

    # Assicura plt.savefig con OUTPUT_PATH
    has_savefig = any("plt.savefig" in l for l in lines)
    if not has_savefig:
        lines.append("plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')")
        lines.append("plt.close()")

    # Assicura plt.close()
    has_close = any("plt.close()" in l for l in lines)
    if not has_close:
        lines.append("plt.close()")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_code(
    spec: FigureSpec,
    output_path: str,
    error_feedback: str = "",
    force_model: str | None = None,
) -> str | None:
    """Genera codice Python per una code figure.

    Args:
        spec: La specifica della figura.
        output_path: Path dove salvare il PNG risultante.
        error_feedback: Messaggio di errore dall'iterazione precedente (per retry).
        force_model: Override modello LLM.

    Returns:
        Codice Python self-contained, oppure None se la generazione fallisce.
    """
    if spec.figure_type != FigureType.CODE:
        logger.warning("generate_code called on non-code figure: %s", spec.figure_id)
        return None

    data_points_json = json.dumps(spec.data_points, indent=2, ensure_ascii=False)
    if not spec.data_points:
        data_points_json = "(No pre-extracted data points — infer appropriate data from the description)"

    feedback = ""
    if error_feedback:
        feedback = _ERROR_FEEDBACK_TEMPLATE.format(error=error_feedback)

    prompt = _CODEGEN_PROMPT.format(
        title=spec.title,
        chart_type=spec.chart_type.value,
        description=spec.description,
        caption=spec.caption or spec.title,
        data_points_json=data_points_json,
        output_path=output_path,
        error_feedback=feedback,
    )

    tier = force_model or "flash"
    llm = get_llm(tier)

    logger.info("Generating code for figure %s with %s...", spec.figure_id, tier)
    t0 = time.time()

    raw_response = llm_call_with_retry(
        llm, prompt,
        max_retries=3,
        base_delay=2.0,
        max_delay=30.0,
    )

    elapsed = time.time() - t0
    logger.info("Code generation for %s completed in %.1fs", spec.figure_id, elapsed)

    code = _extract_python_code(raw_response)
    if code is None:
        logger.error("Failed to extract Python code for %s", spec.figure_id)
        return None

    code = _validate_code(code, output_path)
    return code
