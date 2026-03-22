"""Generate textual overlays from stored lessons for LLM prompt injection.

The overlay is a short Markdown block inserted into the system prompt of
writer / planner / critic agents so the LLM avoids repeating past mistakes.

Skills (persistent rules) are loaded from Markdown files and prepended
before the temporal lessons, forming the complete injection overlay.
"""

from __future__ import annotations

from paperhunter.evolution.lesson_store import (
    LessonCategory,
    LessonSeverity,
    LessonStore,
    ScoredLesson,
)
from paperhunter.evolution.skill_loader import load_skills
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Severity → emoji mapping
# ---------------------------------------------------------------------------

_SEVERITY_ICON: dict[LessonSeverity, str] = {
    LessonSeverity.ERROR: "❌",
    LessonSeverity.WARNING: "⚠️",
    LessonSeverity.INFO: "ℹ️",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_overlay(
    store: LessonStore,
    *,
    stage: str | None = None,
    node: str | None = None,
    section_key: str | None = None,
    category: LessonCategory | None = None,
    max_lessons: int = 5,
    max_chars: int = 2000,
) -> str:
    """Build a Markdown overlay block from the most relevant lessons.

    Parameters
    ----------
    store:
        The :class:`LessonStore` to query.
    stage:
        Filter by pipeline stage (e.g. ``"write"``, ``"summarize"``).
    node:
        Filter by LangGraph node (e.g. ``"verify"``).
    section_key:
        Filter by survey section key (e.g. ``"prompt_injection"``).
    category:
        Filter by lesson category.
    max_lessons:
        Maximum number of lessons to include (default 5).
    max_chars:
        Hard cap on total overlay length in characters (default 2000,
        roughly ~500 tokens).

    Returns
    -------
    str
        A Markdown block ready for prompt injection, or ``""`` if no
        relevant lessons exist.
    """
    scored = store.query(
        stage=stage,
        node=node,
        section_key=section_key,
        category=category,
        top_k=max_lessons,
    )

    if not scored:
        return ""

    lines = _format_lessons(scored, max_chars)

    if not lines:
        return ""

    header = "## Lessons from Prior Runs\n"
    body = "\n".join(lines)
    footer = "\nUse these lessons to avoid repeating past mistakes."
    overlay = f"{header}\n{body}\n{footer}"

    logger.debug(
        "Generated overlay: %d lesson(s), %d chars (stage=%s, section=%s)",
        len(lines), len(overlay), stage, section_key,
    )
    return overlay


def generate_overlay_for_writer(
    store: LessonStore,
    section_key: str,
    *,
    max_lessons: int = 5,
    max_chars: int = 2000,
) -> str:
    """Shortcut: overlay tailored for the writer agent on a specific section.

    Queries lessons from both the exact section and general writing lessons,
    prioritising section-specific ones.
    """
    # Section-specific lessons first
    section_lessons = store.query(
        stage="write",
        section_key=section_key,
        top_k=max_lessons,
    )

    # Fill remaining slots with general writing lessons
    remaining = max_lessons - len(section_lessons)
    general_lessons: list[ScoredLesson] = []
    if remaining > 0:
        general_lessons = store.query(
            stage="write",
            category=LessonCategory.WRITING,
            top_k=remaining + len(section_lessons),
        )
        # Deduplicate (same description)
        seen = {sl.lesson.description for sl in section_lessons}
        general_lessons = [
            sl for sl in general_lessons
            if sl.lesson.description not in seen
        ][:remaining]

    combined = section_lessons + general_lessons

    if not combined:
        return ""

    lines = _format_lessons(combined, max_chars)
    if not lines:
        return ""

    header = f"## Lessons from Prior Runs (section: {section_key})\n"
    body = "\n".join(lines)
    footer = "\nUse these lessons to avoid repeating past mistakes."
    return f"{header}\n{body}\n{footer}"


def generate_overlay_for_planner(
    store: LessonStore,
    *,
    max_lessons: int = 5,
    max_chars: int = 2000,
) -> str:
    """Shortcut: overlay tailored for the planner agent.

    Includes lessons from the ``plan`` stage and high-severity writing
    lessons that might inform outline structure.
    """
    plan_lessons = store.query(stage="plan", top_k=max_lessons)

    remaining = max_lessons - len(plan_lessons)
    write_lessons: list[ScoredLesson] = []
    if remaining > 0:
        write_lessons = store.query(
            stage="write",
            top_k=remaining + len(plan_lessons),
        )
        seen = {sl.lesson.description for sl in plan_lessons}
        write_lessons = [
            sl for sl in write_lessons
            if sl.lesson.description not in seen
        ][:remaining]

    combined = plan_lessons + write_lessons

    if not combined:
        return ""

    lines = _format_lessons(combined, max_chars)
    if not lines:
        return ""

    header = "## Lessons from Prior Runs (planning)\n"
    body = "\n".join(lines)
    footer = "\nUse these lessons to improve the outline structure."
    return f"{header}\n{body}\n{footer}"


def generate_full_overlay(
    store: LessonStore,
    *,
    section_key: str | None = None,
    agent_type: str = "writer",
    skills_dir: str | None = None,
    skills_enabled: bool = True,
    max_skill_chars: int = 4000,
    max_lessons: int = 5,
    max_lesson_chars: int = 2000,
) -> str:
    """Genera overlay completo: [Skills] + [Lessons].

    Ordine nel prompt: Skills (persistenti) → Lessons (temporali con decay).

    Args:
        store: LessonStore per le lezioni.
        section_key: Sezione specifica (per writer).
        agent_type: "writer" o "planner" — determina quale overlay lezioni usare.
        skills_dir: Directory skill .md (None = skip skills).
        skills_enabled: Se False, skip skills.
        max_skill_chars: Limite caratteri per skills.
        max_lessons: Massimo lezioni.
        max_lesson_chars: Limite caratteri per lezioni.

    Returns:
        Overlay combinato (skills + lessons), oppure stringa vuota.
    """
    parts: list[str] = []

    # --- Skills (persistenti, sempre prima) ---
    if skills_enabled and skills_dir:
        skills_text = load_skills(skills_dir, max_total_chars=max_skill_chars)
        if skills_text:
            parts.append(skills_text)

    # --- Lessons (temporali con decay) ---
    if agent_type == "planner":
        lessons_text = generate_overlay_for_planner(
            store, max_lessons=max_lessons, max_chars=max_lesson_chars,
        )
    else:
        lessons_text = generate_overlay_for_writer(
            store, section_key=section_key,
            max_lessons=max_lessons, max_chars=max_lesson_chars,
        )

    if lessons_text:
        parts.append(lessons_text)

    if not parts:
        return ""

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internal formatting
# ---------------------------------------------------------------------------

def _format_lessons(
    scored_lessons: list[ScoredLesson],
    max_chars: int,
) -> list[str]:
    """Format scored lessons into numbered Markdown lines, respecting *max_chars*."""
    lines: list[str] = []
    total_chars = 0

    for idx, sl in enumerate(scored_lessons, 1):
        lesson = sl.lesson
        icon = _SEVERITY_ICON.get(lesson.severity, "•")
        cat = lesson.category.value if isinstance(lesson.category, LessonCategory) else lesson.category

        # Truncate description if very long
        desc = lesson.description
        if len(desc) > 300:
            desc = desc[:297] + "..."

        line = f"{idx}. {icon} [{cat}] {desc}"

        # Append bibtex keys hint if present
        if lesson.bibtex_keys_involved:
            keys_str = ", ".join(lesson.bibtex_keys_involved[:5])
            line += f"\n   Keys: {keys_str}"

        if total_chars + len(line) > max_chars:
            break

        lines.append(line)
        total_chars += len(line)

    return lines
