"""
PDF Page Embedder — Genera embedding dai PDF per chunk GROBID.

Approccio dual: GROBID estrae testo strutturato + metadati (rigore scientifico),
Gemini Embedding 2 Preview genera embedding direttamente dalle pagine PDF
(cattura layout, formule, tabelle, figure).

Il testo GROBID resta nel payload Qdrant per il LLM; i vettori derivano
dalla comprensione visiva del PDF → retrieval semantico migliore.

Flusso:
    1. PyMuPDF estrae testo per pagina (per matching)
    2. MultimodalEmbedder genera embedding per ogni pagina PDF
    3. Ogni chunk GROBID viene mappato alla pagina migliore via text overlap
    4. Il chunk riceve l'embedding della sua pagina PDF primaria
"""

from __future__ import annotations

import re
from pathlib import Path

from paperhunter.models import PaperChunk
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _extract_page_texts(pdf_path: str | Path) -> list[str]:
    """Estrae il testo grezzo da ogni pagina del PDF.

    Args:
        pdf_path: Percorso al file PDF.

    Returns:
        Lista di stringhe (una per pagina, in ordine).
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    texts = []
    for page in doc:
        texts.append(page.get_text("text") or "")
    doc.close()
    return texts


def _normalize_for_matching(text: str) -> str:
    """Normalizza testo per il matching chunk→pagina."""
    return re.sub(r"\s+", " ", text.lower().strip())


def map_chunks_to_pages(
    pdf_path: str | Path,
    chunks: list[PaperChunk],
) -> list[int]:
    """Mappa ogni chunk alla pagina PDF con massimo overlap testuale.

    Usa una finestra di testo dal chunk (primi 200 char normalizzati)
    e cerca la pagina con la migliore corrispondenza.

    Args:
        pdf_path: Percorso al file PDF.
        chunks: Lista di PaperChunk (testo GROBID).

    Returns:
        Lista di indici pagina (0-based), uno per chunk.
        Se nessuna pagina matcha, assegna pagina 0.
    """
    page_texts = _extract_page_texts(pdf_path)
    normalized_pages = [_normalize_for_matching(t) for t in page_texts]
    n_pages = len(normalized_pages)

    if n_pages == 0:
        return [0] * len(chunks)

    mappings: list[int] = []

    for chunk in chunks:
        # Usa i primi 200 char del chunk per il matching
        chunk_snippet = _normalize_for_matching(chunk.text[:200])
        if not chunk_snippet:
            mappings.append(0)
            continue

        # Cerca la pagina con il miglior overlap
        best_page = 0
        best_score = 0.0

        # Tokenizza lo snippet in parole
        snippet_words = set(chunk_snippet.split())
        if not snippet_words:
            mappings.append(0)
            continue

        for page_idx, page_text in enumerate(normalized_pages):
            page_words = set(page_text.split())
            if not page_words:
                continue
            overlap = len(snippet_words & page_words)
            score = overlap / len(snippet_words)
            if score > best_score:
                best_score = score
                best_page = page_idx

        mappings.append(best_page)

    return mappings


def generate_pdf_embeddings_for_chunks(
    pdf_path: str | Path,
    chunks: list[PaperChunk],
    show_progress: bool = False,
) -> list[list[float]]:
    """Genera embedding PDF-based per una lista di chunk GROBID.

    Flusso:
        1. Genera embedding per ogni pagina PDF (multimodale)
        2. Mappa ogni chunk alla sua pagina primaria
        3. Restituisce l'embedding della pagina per ogni chunk

    Args:
        pdf_path: Percorso al file PDF.
        chunks: Lista di PaperChunk con testo GROBID.
        show_progress: Se True, mostra progress bar.

    Returns:
        Lista di vettori 3072-dim, uno per chunk (allineati per indice).
    """
    from paperhunter.utils.multimodal_embedder import get_multimodal_embedder

    pdf_path = Path(pdf_path)
    embedder = get_multimodal_embedder()

    # Step 1: Genera embedding per ogni pagina PDF
    page_embeddings = embedder.embed_pdf_pages(
        pdf_path, show_progress=show_progress
    )

    if not page_embeddings:
        logger.warning(
            f"Nessun embedding pagina per {pdf_path.name}, "
            f"fallback a embedding testuali"
        )
        return []

    # Step 2: Mappa chunk → pagine
    chunk_page_map = map_chunks_to_pages(pdf_path, chunks)

    # Step 3: Assegna embedding pagina a ogni chunk
    chunk_embeddings: list[list[float]] = []
    n_pages = len(page_embeddings)

    for chunk_idx, page_idx in enumerate(chunk_page_map):
        # Clamp page index to valid range
        safe_idx = min(page_idx, n_pages - 1)
        chunk_embeddings.append(page_embeddings[safe_idx])

    logger.info(
        f"PDF embeddings: {len(chunk_embeddings)} chunk mappati su "
        f"{n_pages} pagine per {pdf_path.name}"
    )
    return chunk_embeddings
