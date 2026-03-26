"""FigureAgent — Renderer: esegue codice Python in subprocess e cattura PNG.

Esegue lo script generato da codegen in un subprocess isolato con timeout,
cattura errori per il retry loop, e verifica che il PNG sia valido.

Miglioramenti:
- Critic quality gate (analisi statica post-render)
- Validazione PNG con PIL (contenuto, non solo size)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from paperhunter.agents.figure_agent.codegen import generate_code
from paperhunter.agents.figure_agent.planner import FigureSpec, FigureType
from paperhunter.config import settings
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

_EXECUTION_TIMEOUT = 60  # secondi


def _execute_code(code: str, timeout: int = _EXECUTION_TIMEOUT) -> tuple[bool, str]:
    """Esegue codice Python in un subprocess isolato.

    Args:
        code: Codice Python da eseguire.
        timeout: Timeout in secondi.

    Returns:
        Tupla (success: bool, error_or_output: str).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd()),
        )

        if result.returncode == 0:
            return True, result.stdout.strip()

        error = result.stderr.strip()
        # Estrai solo l'errore significativo (ultime righe)
        error_lines = error.split("\n")
        if len(error_lines) > 10:
            error = "\n".join(error_lines[-10:])
        return False, error

    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout}s"
    except Exception as e:
        return False, f"Execution failed: {e}"


def _verify_png(path: Path) -> tuple[bool, str]:
    """Verifica che il file PNG esista, abbia dimensione ragionevole e sia valido.

    Returns:
        Tupla (valid: bool, reason: str).
    """
    if not path.exists():
        return False, "PNG file does not exist"

    size = path.stat().st_size
    if size < 1000:
        return False, f"PNG too small ({size} bytes) — likely corrupted or empty"
    if size > 50_000_000:
        return False, f"PNG too large ({size} bytes) — likely runaway rendering"

    # Validazione contenuto con PIL
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        # verify() chiude il file — riapri per check blank
        with Image.open(path) as img:
            extrema = img.convert("RGB").getextrema()
            # Se tutti i canali hanno min==max, l'immagine è uniforme (blank)
            if all(lo == hi for lo, hi in extrema):
                return False, "PNG is a blank/uniform image — no visible content"
    except ImportError:
        logger.debug("PIL not available — skipping image content validation")
    except Exception as e:
        return False, f"PNG validation failed: {e}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Static quality gate
# ---------------------------------------------------------------------------

def _static_quality_check(spec: FigureSpec, code: str) -> tuple[bool, str]:
    """Quality gate rapido basato su analisi statica del codice (no LLM).

    Returns:
        Tupla (passed: bool, feedback: str con problemi trovati).
    """
    from paperhunter.agents.figure_agent.critic import critique_code_only
    critique = critique_code_only(spec, code)

    if critique.verdict.value == "APPROVED":
        return True, ""

    feedback = "; ".join(critique.issues) if critique.issues else critique.feedback
    logger.info(
        "Figure %s: static critique → NEEDS_REVISION (score=%.2f): %s",
        spec.figure_id, critique.overall_score, feedback,
    )
    return False, feedback


# ---------------------------------------------------------------------------
# Vision quality gate (LLM-based image analysis)
# ---------------------------------------------------------------------------

_VISION_PROMPT = """\
You are reviewing a rendered scientific figure for an academic survey paper.

**Figure specification:**
- Title: {title}
- Chart type: {chart_type}
- Expected content: {description}

**Instructions:**
Look at the rendered image and check for these issues:
1. **Text readability**: Are labels, titles, legends readable? Any overlapping text?
2. **Data correctness**: Does the chart look reasonable for the described data?
3. **Visual artifacts**: Any clipping, missing elements, blank areas, or distortion?
4. **Layout**: Is the figure well-proportioned? Any elements cut off?
5. **Professional quality**: Does it look suitable for an academic publication?

Return ONLY a JSON object (no markdown fences):
{{"pass": true, "issues": []}}
or
{{"pass": false, "issues": ["issue 1", "issue 2"]}}

Be strict but fair. Minor cosmetic issues are acceptable. Focus on problems that \
would make the figure unsuitable for publication."""


def _vision_quality_check(
    spec: FigureSpec, png_path: Path,
) -> tuple[bool, str]:
    """Controlla la qualità visiva della figura usando Gemini Vision.

    Returns:
        Tupla (passed: bool, feedback: str con problemi trovati).
    """
    import json as _json

    from google import genai
    from google.genai import types

    prompt = _VISION_PROMPT.format(
        title=spec.title,
        chart_type=spec.chart_type.value,
        description=spec.description,
    )

    png_bytes = png_path.read_bytes()
    image_part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")

    try:
        from paperhunter.utils.llm_factory import _get_api_key

        client = genai.Client(api_key=_get_api_key())
        response = client.models.generate_content(
            model=settings.gemini.flash_model,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )

        if not response or not response.text:
            logger.warning("Vision check returned empty response")
            return True, ""

        raw = response.text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = _json.loads(raw)
        passed = data.get("pass", True)
        issues = data.get("issues", [])

        if not passed:
            feedback = "; ".join(issues) if issues else "Vision check failed"
            logger.info(
                "Figure %s: vision check → NEEDS_REVISION: %s",
                spec.figure_id, feedback,
            )
            return False, feedback

        logger.info("Figure %s: vision check → APPROVED", spec.figure_id)
        return True, ""

    except Exception as exc:
        logger.warning("Vision check error for %s: %s", spec.figure_id, exc)
        return True, ""  # non bloccare se il vision check fallisce

def render_figure(
    spec: FigureSpec,
    output_dir: str | Path,
    max_iterations: int = 5,
    force_model: str | None = None,
) -> tuple[bool, Path | None, str | None]:
    """Genera e renderizza una code figure con retry automatico.

    Flow: codegen → execute → verify PNG → static critic → vision check → retry.

    Args:
        spec: La specifica della figura da generare.
        output_dir: Directory dove salvare il PNG.
        max_iterations: Massimo iterazioni codegen → execute (default 3).
        force_model: Override modello LLM per codegen.

    Returns:
        Tupla (success, png_path, generated_code).
    """
    if spec.figure_type != FigureType.CODE:
        logger.warning("render_figure called on non-code figure: %s", spec.figure_id)
        return False, None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.figure_id}.png"

    error_feedback = ""
    last_code = None

    for iteration in range(1, max_iterations + 1):
        logger.info(
            "Figure %s: iteration %d/%d",
            spec.figure_id, iteration, max_iterations,
        )

        # Genera codice
        code = generate_code(
            spec=spec,
            output_path=str(output_path),
            error_feedback=error_feedback,
            force_model=force_model,
        )

        if code is None:
            logger.error("Figure %s: codegen failed at iteration %d", spec.figure_id, iteration)
            error_feedback = "Code generation returned None — the LLM did not produce valid Python code."
            continue

        last_code = code

        # Salva lo script per debug
        script_path = output_dir / f"{spec.figure_id}_code.py"
        script_path.write_text(code, encoding="utf-8")

        # Esegui
        success, output = _execute_code(code)

        if not success:
            logger.warning(
                "Figure %s: execution failed at iteration %d: %s",
                spec.figure_id, iteration, output[:200],
            )
            error_feedback = output
            continue

        # Verifica PNG (contenuto + size + PIL validation)
        png_valid, png_reason = _verify_png(output_path)
        if not png_valid:
            logger.warning("Figure %s: PNG invalid — %s", spec.figure_id, png_reason)
            error_feedback = f"The script executed but the PNG is invalid: {png_reason}"
            continue

        # Quality gate: analisi statica del codice
        quality_ok, quality_feedback = _static_quality_check(spec, code)
        if not quality_ok and iteration < max_iterations:
            error_feedback = f"Code quality issues: {quality_feedback}"
            logger.info("Figure %s: quality gate failed, retrying...", spec.figure_id)
            continue

        # Vision quality gate: LLM analizza il PNG renderizzato
        vision_ok, vision_feedback = _vision_quality_check(spec, output_path)
        if not vision_ok and iteration < max_iterations:
            error_feedback = f"Visual issues found in rendered figure: {vision_feedback}"
            logger.info("Figure %s: vision check failed, retrying...", spec.figure_id)
            continue

        # Successo (o ultima iterazione con quality issues non bloccanti)
        size_kb = output_path.stat().st_size / 1024
        logger.info(
            "Figure %s: rendered successfully (%.1f KB) at iteration %d",
            spec.figure_id, size_kb, iteration,
        )
        return True, output_path, code

    logger.error(
        "Figure %s: failed after %d iterations",
        spec.figure_id, max_iterations,
    )
    return False, None, last_code


def render_all_code_figures(
    specs: list[FigureSpec],
    output_dir: str | Path,
    max_iterations: int = 5,
    force_model: str | None = None,
) -> dict[str, tuple[bool, Path | None]]:
    """Renderizza tutte le code figures in un piano.

    Args:
        specs: Lista di FigureSpec (filtra automaticamente solo CODE).
        output_dir: Directory output per i PNG.
        max_iterations: Massimo iterazioni per figura.
        force_model: Override modello LLM.

    Returns:
        Dict figure_id → (success, png_path).
    """
    from rich.console import Console
    console = Console()

    code_specs = [s for s in specs if s.figure_type == FigureType.CODE]
    if not code_specs:
        logger.info("No code figures to render")
        return {}

    results: dict[str, tuple[bool, Path | None]] = {}
    total = len(code_specs)

    console.print(f"[bold cyan]🎨 Rendering {total} code figures...[/bold cyan]")

    for i, spec in enumerate(code_specs, 1):
        console.print(f"  [{i}/{total}] {spec.figure_id}: {spec.title}")
        t0 = time.time()

        success, png_path, _ = render_figure(
            spec=spec,
            output_dir=output_dir,
            max_iterations=max_iterations,
            force_model=force_model,
        )

        elapsed = time.time() - t0
        results[spec.figure_id] = (success, png_path)

        if success:
            console.print(f"    [green]✅ OK ({elapsed:.1f}s)[/green]")
        else:
            console.print(f"    [red]❌ FAILED ({elapsed:.1f}s)[/red]")

    succeeded = sum(1 for s, _ in results.values() if s)
    console.print(
        f"[bold]Rendering complete: {succeeded}/{total} figures successful[/bold]"
    )

    return results
