"""Evolution — Skill Loader: carica skill persistenti da file Markdown.

Le skill sono regole/pattern che il pipeline deve SEMPRE seguire, a
differenza delle lezioni che hanno scadenza temporale. Sono definite come
file Markdown in `data/skills/` e iniettate nel system prompt.
"""

from __future__ import annotations

from pathlib import Path

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


def load_skills(
    skills_dir: str | Path,
    max_total_chars: int = 4000,
) -> str:
    """Carica tutti i file .md dalla directory skill e li concatena.

    Args:
        skills_dir: Directory contenente file `.md` skill.
        max_total_chars: Lunghezza massima totale dell'output.

    Returns:
        Blocco di testo formattato con tutte le skill, oppure stringa vuota
        se la directory non esiste o è vuota.
    """
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        logger.debug("Skills directory not found: %s", skills_dir)
        return ""

    md_files = sorted(skills_dir.glob("*.md"))
    if not md_files:
        logger.debug("No skill files found in %s", skills_dir)
        return ""

    parts: list[str] = []
    total_chars = 0

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8").strip()
            if not content:
                continue

            # Controlla limite
            if total_chars + len(content) + 50 > max_total_chars:
                remaining = max_total_chars - total_chars - 50
                if remaining > 100:
                    content = content[:remaining] + "\n[... truncated]"
                else:
                    logger.info(
                        "Skill '%s' skipped (max_total_chars reached)", md_file.name,
                    )
                    break

            parts.append(f"### {md_file.stem.replace('_', ' ').title()}\n\n{content}")
            total_chars += len(content) + 50
            logger.debug("Loaded skill: %s (%d chars)", md_file.name, len(content))

        except Exception as e:
            logger.warning("Failed to load skill %s: %s", md_file.name, e)

    if not parts:
        return ""

    header = "## Persistent Skills (Always Apply)\n\n"
    body = "\n\n---\n\n".join(parts)
    footer = "\n\nApply these rules consistently throughout the writing process."

    result = f"{header}{body}{footer}"
    logger.info("Loaded %d skill(s) (%d chars) from %s", len(parts), len(result), skills_dir)
    return result


def list_skills(skills_dir: str | Path) -> list[dict]:
    """Elenca le skill disponibili con metadata.

    Returns:
        Lista di dict con name, path, size_chars per ogni skill file.
    """
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        return []

    result = []
    for md_file in sorted(skills_dir.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            result.append({
                "name": md_file.stem,
                "path": str(md_file),
                "size_chars": len(content),
            })
        except Exception:
            pass

    return result
