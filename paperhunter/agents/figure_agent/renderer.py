"""FigureAgent — Renderer: genera figure tramite Gemini Image Generation.

Usa gemini-3-pro-image-preview per generare immagini scientifiche direttamente,
con vision quality check (Gemini Flash) e loop di refinement iterativo.

Flow: generate image → vision check → refine image (se necessario) → retry.
"""

from __future__ import annotations

import json as _json
import time
from pathlib import Path

from paperhunter.agents.figure_agent.planner import FigureSpec, FigureType
from paperhunter.config import settings
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_MODEL = "gemini-3-pro-image-preview"

_GENERATION_PROMPT = """\
Generate a high-quality scientific figure for an academic survey paper.

**Figure specification:**
- Title: {title}
- Chart type: {chart_type}
- Description: {description}
- Caption: {caption}

**Data points (if available):**
{data_points_json}

**Style requirements:**
- Professional academic style suitable for a peer-reviewed publication
- Clean, minimalist design with clear labels and legends
- Use a professional color palette (blues, teals, grays — no neon/bright colors)
- All text must be clearly readable (no overlapping labels)
- Include axis labels with units where applicable
- Title should be concise and placed above the figure
- High resolution, sharp lines and text
- White or light background
- For charts: include grid lines, proper spacing between elements
- For diagrams/flowcharts: use clean boxes with rounded corners, clear arrows, \
hierarchical layout with no overlapping elements

Generate ONLY the image, no text response."""

_REFINEMENT_PROMPT = """\
Modify this scientific figure to fix the following issues:
{issues}

Keep the same data, structure, and overall design. Only fix the listed problems.
The figure must remain suitable for an academic publication.

Generate ONLY the corrected image, no text response."""

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


# ---------------------------------------------------------------------------
# PNG verification
# ---------------------------------------------------------------------------

def _verify_png(path: Path) -> tuple[bool, str]:
    """Verifica che il file PNG esista, abbia dimensione ragionevole e sia valido."""
    if not path.exists():
        return False, "PNG file does not exist"

    size = path.stat().st_size
    if size < 1000:
        return False, f"PNG too small ({size} bytes) — likely corrupted or empty"
    if size > 50_000_000:
        return False, f"PNG too large ({size} bytes) — likely runaway rendering"

    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            extrema = img.convert("RGB").getextrema()
            if all(lo == hi for lo, hi in extrema):
                return False, "PNG is a blank/uniform image — no visible content"
    except ImportError:
        logger.debug("PIL not available — skipping image content validation")
    except Exception as e:
        return False, f"PNG validation failed: {e}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Image generation via Gemini
# ---------------------------------------------------------------------------

def _generate_image(
    spec: FigureSpec, output_path: Path,
) -> tuple[bool, str]:
    """Genera un'immagine con gemini-3-pro-image-preview.

    Returns:
        (success, error_message)
    """
    import json

    from google import genai
    from google.genai import types

    from paperhunter.utils.llm_factory import _get_api_key

    data_points_json = json.dumps(spec.data_points, indent=2, ensure_ascii=False)
    if not spec.data_points:
        data_points_json = "(No pre-extracted data — infer appropriate data from the description)"

    prompt = _GENERATION_PROMPT.format(
        title=spec.title,
        chart_type=spec.chart_type.value,
        description=spec.description,
        caption=spec.caption or spec.title,
        data_points_json=data_points_json,
    )

    try:
        client = genai.Client(api_key=_get_api_key())
        response = client.models.generate_content(
            model=_IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        if not response or not response.candidates:
            return False, "Empty response from image generation model"

        # Extract image from response parts
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(part.inline_data.data)
                return True, ""

        return False, "No image data in model response"

    except Exception as exc:
        return False, f"Image generation failed: {exc}"


def _refine_image(
    spec: FigureSpec, png_path: Path, issues: str,
) -> tuple[bool, str]:
    """Invia l'immagine + feedback a gemini-3-pro-image-preview per il refinement.

    Returns:
        (success, error_message)
    """
    from google import genai
    from google.genai import types

    from paperhunter.utils.llm_factory import _get_api_key

    prompt = _REFINEMENT_PROMPT.format(issues=issues)
    image_bytes = png_path.read_bytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    try:
        client = genai.Client(api_key=_get_api_key())
        response = client.models.generate_content(
            model=_IMAGE_MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        if not response or not response.candidates:
            return False, "Empty response from refinement model"

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                png_path.write_bytes(part.inline_data.data)
                return True, ""

        return False, "No image data in refinement response"

    except Exception as exc:
        return False, f"Image refinement failed: {exc}"


# ---------------------------------------------------------------------------
# Vision quality gate
# ---------------------------------------------------------------------------

def _vision_quality_check(
    spec: FigureSpec, png_path: Path,
) -> tuple[bool, str]:
    """Controlla la qualità visiva della figura usando Gemini Vision (Flash)."""
    from google import genai
    from google.genai import types

    from paperhunter.utils.llm_factory import _get_api_key

    prompt = _VISION_PROMPT.format(
        title=spec.title,
        chart_type=spec.chart_type.value,
        description=spec.description,
    )

    png_bytes = png_path.read_bytes()
    image_part = types.Part.from_bytes(data=png_bytes, mime_type="image/png")

    try:
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
        return True, ""


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def render_figure(
    spec: FigureSpec,
    output_dir: str | Path,
    max_iterations: int = 5,
    force_model: str | None = None,
) -> tuple[bool, Path | None, str | None]:
    """Genera una figura con Gemini Image Generation e quality check iterativo.

    Flow: generate image → verify PNG → vision check → refine (se necessario).

    Args:
        spec: La specifica della figura da generare.
        output_dir: Directory dove salvare il PNG.
        max_iterations: Massimo iterazioni generate/refine (default 5).
        force_model: Non usato (kept for API compatibility).

    Returns:
        Tupla (success, png_path, None).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.figure_id}.png"

    last_feedback = ""

    for iteration in range(1, max_iterations + 1):
        logger.info(
            "Figure %s: iteration %d/%d",
            spec.figure_id, iteration, max_iterations,
        )

        if iteration == 1:
            # First iteration: generate from scratch
            logger.info("Generating image for %s with %s...", spec.figure_id, _IMAGE_MODEL)
            success, error = _generate_image(spec, output_path)
        else:
            # Subsequent iterations: refine existing image with feedback
            logger.info(
                "Refining image for %s with %s (feedback: %s)...",
                spec.figure_id, _IMAGE_MODEL, last_feedback[:100],
            )
            success, error = _refine_image(spec, output_path, last_feedback)

        if not success:
            logger.warning(
                "Figure %s: %s failed at iteration %d: %s",
                spec.figure_id,
                "generation" if iteration == 1 else "refinement",
                iteration, error,
            )
            last_feedback = error
            # If refinement fails, try fresh generation on next iteration
            if iteration > 1:
                logger.info("Refinement failed, will retry fresh generation")
                success2, error2 = _generate_image(spec, output_path)
                if not success2:
                    continue
            else:
                continue

        # Verify PNG
        png_valid, png_reason = _verify_png(output_path)
        if not png_valid:
            logger.warning("Figure %s: PNG invalid — %s", spec.figure_id, png_reason)
            last_feedback = f"The generated image is invalid: {png_reason}"
            continue

        # Vision quality check
        vision_ok, vision_feedback = _vision_quality_check(spec, output_path)
        if not vision_ok and iteration < max_iterations:
            last_feedback = vision_feedback
            logger.info("Figure %s: vision check failed, refining...", spec.figure_id)
            continue

        # Success (or last iteration — accept whatever we have)
        size_kb = output_path.stat().st_size / 1024
        logger.info(
            "Figure %s: generated successfully (%.1f KB) at iteration %d",
            spec.figure_id, size_kb, iteration,
        )
        return True, output_path, None

    logger.error(
        "Figure %s: failed after %d iterations",
        spec.figure_id, max_iterations,
    )
    return False, None, None


def render_all_figures(
    specs: list[FigureSpec],
    output_dir: str | Path,
    max_iterations: int = 5,
    force_model: str | None = None,
) -> dict[str, tuple[bool, Path | None]]:
    """Renderizza tutte le figure del piano tramite Gemini Image Generation.

    Args:
        specs: Lista di FigureSpec.
        output_dir: Directory output per i PNG.
        max_iterations: Massimo iterazioni per figura.
        force_model: Non usato (kept for API compatibility).

    Returns:
        Dict figure_id → (success, png_path).
    """
    from rich.console import Console
    console = Console()

    if not specs:
        logger.info("No figures to render")
        return {}

    results: dict[str, tuple[bool, Path | None]] = {}
    total = len(specs)

    console.print(f"[bold cyan]🎨 Generating {total} figures with {_IMAGE_MODEL}...[/bold cyan]")

    for i, spec in enumerate(specs, 1):
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
        f"[bold]Generation complete: {succeeded}/{total} figures successful[/bold]"
    )

    return results


# Backward compatibility alias
render_all_code_figures = render_all_figures
