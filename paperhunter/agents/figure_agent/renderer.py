"""FigureAgent — Renderer: esegue codice Python in subprocess e cattura PNG.

Esegue lo script generato da codegen in un subprocess isolato con timeout,
cattura errori per il retry loop, e verifica che il PNG sia valido.
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


def _verify_png(path: Path) -> bool:
    """Verifica che il file PNG esista e abbia dimensione ragionevole."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < 1000:  # < 1KB → probabilmente corrotto
        logger.warning("PNG too small (%d bytes): %s", size, path)
        return False
    if size > 50_000_000:  # > 50MB → qualcosa non va
        logger.warning("PNG too large (%d bytes): %s", size, path)
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_figure(
    spec: FigureSpec,
    output_dir: str | Path,
    max_iterations: int = 3,
    force_model: str | None = None,
) -> tuple[bool, Path | None, str | None]:
    """Genera e renderizza una code figure con retry automatico.

    Flow: codegen → execute → verify PNG → retry on failure.

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

        if success and _verify_png(output_path):
            size_kb = output_path.stat().st_size / 1024
            logger.info(
                "Figure %s: rendered successfully (%.1f KB) at iteration %d",
                spec.figure_id, size_kb, iteration,
            )
            return True, output_path, code

        # Errore → prepara feedback per retry
        if not success:
            logger.warning(
                "Figure %s: execution failed at iteration %d: %s",
                spec.figure_id, iteration, output[:200],
            )
            error_feedback = output
        elif not _verify_png(output_path):
            error_feedback = "The script executed without errors but did not produce a valid PNG file at the expected path."
            logger.warning("Figure %s: PNG not found/invalid after execution", spec.figure_id)

    logger.error(
        "Figure %s: failed after %d iterations",
        spec.figure_id, max_iterations,
    )
    return False, None, last_code


def render_all_code_figures(
    specs: list[FigureSpec],
    output_dir: str | Path,
    max_iterations: int = 3,
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
