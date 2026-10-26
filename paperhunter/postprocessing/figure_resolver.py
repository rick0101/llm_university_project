"""FigureResolver — risoluzione dei tag {{FIGURE:key:type}} nel testo finale.

Il writer agentico può inserire placeholder del tipo::

    {{FIGURE:Zou_2023_GCG:architecture_diagram}}

Questo modulo scansiona il testo finale e sostituisce ogni tag con il markup
appropriato (Markdown o LaTeX), interrogando Qdrant per recuperare i metadati
della figura. Se la figura non è trovata o il file immagine non esiste su disco,
il tag viene rimosso e aggiunto alla lista ``unresolved_tags``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Literal

logger = logging.getLogger(__name__)

# Regex per tag figura: {{FIGURE:bibtex_key:figure_type}}
_FIGURE_TAG_RE = re.compile(r"\{\{FIGURE:([^:}\s]+):([^}\s]+)\}\}")


class FigureResolver:
    """Risolve i tag ``{{FIGURE:key:type}}`` nel testo del survey.

    Interroga Qdrant per recuperare metadati della figura (descrizione,
    percorso immagine, caption) e sostituisce ogni tag con il markup
    appropriato. I tag non risolvibili vengono rimossi e restituiti
    nella lista ``unresolved_tags``.

    Args:
        qdrant_store: Istanza opzionale di ``QdrantStore``. Se ``None``,
            viene inizializzata automaticamente.
    """

    def __init__(self, qdrant_store: object | None = None) -> None:
        self._store = qdrant_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        text: str,
        output_format: Literal["markdown", "latex"] = "markdown",
    ) -> tuple[str, list[str]]:
        """Risolve tutti i tag ``{{FIGURE:key:type}}`` nel testo.

        Args:
            text: Testo del survey con eventuali tag figura.
            output_format: Formato di output — ``"markdown"`` o ``"latex"``.

        Returns:
            Tupla ``(resolved_text, unresolved_tags)`` dove:
            - ``resolved_text`` è il testo con i tag sostituiti dal markup.
            - ``unresolved_tags`` è la lista dei tag originali non risolti
              (figura non trovata o file immagine mancante su disco).
        """
        unresolved: list[str] = []
        resolved_count = 0

        def _replace(match: re.Match) -> str:
            nonlocal resolved_count
            bibtex_key = match.group(1)
            figure_type = match.group(2)
            original_tag = match.group(0)

            figure = self._lookup_figure(bibtex_key, figure_type)
            if figure is None:
                logger.debug("FigureResolver: figura non trovata per tag %s", original_tag)
                unresolved.append(original_tag)
                return ""

            image_path: str = figure.get("image_path", "")
            if image_path and not os.path.isfile(image_path):
                logger.debug(
                    "FigureResolver: image_path non su disco '%s' per tag %s",
                    image_path, original_tag,
                )
                unresolved.append(original_tag)
                return ""

            resolved_count += 1
            return self._render_figure(figure, bibtex_key, output_format)

        result = _FIGURE_TAG_RE.sub(_replace, text)

        logger.info(
            "FigureResolver: %d tag risolti, %d non risolti",
            resolved_count, len(unresolved),
        )
        return result, unresolved

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lookup_figure(self, bibtex_key: str, figure_type: str) -> dict | None:
        """Interroga Qdrant per trovare la figura (prima figure_descriptions,
        poi image_embeddings). Restituisce un dict con i metadati o None."""
        store = self._get_store()
        if store is None:
            return None

        for collection in ("figure_descriptions", "image_embeddings"):
            result = _query_qdrant(store, collection, bibtex_key, figure_type)
            if result is not None:
                return result
        return None

    def _get_store(self) -> object | None:
        """Restituisce l'istanza di QdrantStore, inizializzandola se necessario."""
        if self._store is None:
            try:
                from paperhunter.indexing.qdrant_store import QdrantStore
                self._store = QdrantStore()
            except Exception as exc:
                logger.warning("FigureResolver: QdrantStore non disponibile: %s", exc)
                return None
        return self._store

    @staticmethod
    def _render_figure(
        figure: dict,
        bibtex_key: str,
        output_format: Literal["markdown", "latex"],
    ) -> str:
        """Genera il markup per una figura trovata."""
        caption: str = figure.get("caption") or figure.get("description", "") or ""
        description: str = figure.get("description", "")
        image_path: str = figure.get("image_path", "")

        if not caption:
            caption = f"Figure from {bibtex_key}"

        if output_format == "latex":
            # Escape del path per LaTeX (le underscore vanno mantenute nell'argomento)
            latex_path = image_path.replace("\\", "/")
            latex_caption = _latex_escape(caption)
            return (
                f"\n\\begin{{figure}}[h]\n"
                f"\\centering\n"
                f"\\includegraphics[width=0.8\\textwidth]{{{latex_path}}}\n"
                f"\\caption{{{latex_caption} \\cite{{{bibtex_key}}}}}\n"
                f"\\end{{figure}}\n"
            )
        else:
            # Markdown
            md_caption = caption.replace("[", "\\[").replace("]", "\\]")
            lines = [f"\n![{md_caption}]({image_path})"]
            if description and description != caption:
                lines.append(f"*Figura: {description}* [{bibtex_key}]")
            else:
                lines.append(f"[{bibtex_key}]")
            lines.append("")
            return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helper (avoids duplication with retrieval_tools._query_figure_collection)
# ---------------------------------------------------------------------------

def _query_qdrant(
    store: object,
    collection: str,
    bibtex_key: str,
    figure_type: str,
) -> dict | None:
    """Interroga una collection Qdrant per trovare una figura di un paper.

    Filtra per ``bibtex_key`` nel payload e, se figure_type != "any",
    anche per ``figure_type``. Restituisce un dict con i metadati oppure None.
    """
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must: list = [
            FieldCondition(key="bibtex_key", match=MatchValue(value=bibtex_key))
        ]
        if figure_type != "any":
            must.append(
                FieldCondition(key="figure_type", match=MatchValue(value=figure_type))
            )

        qdrant_filter = Filter(must=must)
        client = getattr(store, "client", None) or getattr(store, "_client", None)
        if client is None:
            return None

        records, _ = client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            return None

        payload = records[0].payload or {}
        return {
            "figure_id": str(records[0].id),
            "paper_key": payload.get("bibtex_key", bibtex_key),
            "figure_type": payload.get("figure_type", figure_type),
            "description": payload.get("description", payload.get("text", "")),
            "image_path": payload.get("image_path", ""),
            "caption": payload.get("caption", payload.get("figure_caption", "")),
        }

    except Exception as exc:
        logger.debug("_query_qdrant(%s, %s): %s", collection, bibtex_key, exc)
        return None


def _latex_escape(text: str) -> str:
    """Escape minimo per testo da inserire in una caption LaTeX."""
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )
