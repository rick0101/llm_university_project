"""Carica i prompt template dal file centralizzato ``prompts/writer_pipeline.md``.

Uso::

    from paperhunter.utils.prompt_loader import get_prompt

    prompt = get_prompt("WRITER_PROMPT").format(title=title, ...)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Il file si trova in <project_root>/prompts/writer_pipeline.md
_PROMPTS_FILE = Path(__file__).resolve().parent.parent.parent / "prompts" / "writer_pipeline.md"


@lru_cache(maxsize=1)
def _load_prompts() -> dict[str, str]:
    """Parsa il file prompts, ritornando ``{KEY: template_text}``."""
    if not _PROMPTS_FILE.exists():
        logger.warning("File prompts non trovato: %s", _PROMPTS_FILE)
        return {}

    content = _PROMPTS_FILE.read_text(encoding="utf-8")
    prompts: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in content.split("\n"):
        # Top-level heading = prompt delimiter (# KEY but NOT ## subheading)
        if line.startswith("# ") and not line.startswith("## "):
            if current_key is not None:
                prompts[current_key] = "\n".join(current_lines).strip()
            current_key = line[2:].strip()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    # Ultimo prompt
    if current_key is not None:
        prompts[current_key] = "\n".join(current_lines).strip()

    logger.debug("Caricati %d prompt da %s", len(prompts), _PROMPTS_FILE)
    return prompts


def get_prompt(key: str) -> str:
    """Ritorna il template di un prompt per chiave.

    Il testo ritornato contiene placeholder ``{...}`` da riempire con ``.format()``.

    Raises
    ------
    KeyError
        Se la chiave non esiste nel file prompts.
    """
    prompts = _load_prompts()
    if key not in prompts:
        available = ", ".join(sorted(prompts.keys()))
        raise KeyError(f"Prompt '{key}' non trovato. Disponibili: {available}")
    return prompts[key]


def reload_prompts() -> None:
    """Svuota la cache e ricarica i prompt da disco."""
    _load_prompts.cache_clear()


def list_prompts() -> list[str]:
    """Ritorna la lista delle chiavi prompt disponibili."""
    return sorted(_load_prompts().keys())
