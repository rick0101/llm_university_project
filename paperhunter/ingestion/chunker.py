"""
Chunker GROBID-aware per la pipeline di ingestion.

Crea PaperChunk dal GROBIDOutput sfruttando la struttura a sezioni
estratta da GROBID. Se GROBID non ha estratto sezioni, fallback
a splitting per dimensione con rilevamento heading (stile legacy).

Ogni chunk ha metadata ricchi (paper_id, bibtex_key, section_name,
category, year, month, chunk_type) per il retrieval ibrido.

Uso:
    from paperhunter.ingestion.chunker import chunk_grobid_output, chunk_paper

    # Da GROBIDOutput (preferito)
    chunks = chunk_grobid_output(grobid_output, paper)

    # Da testo grezzo (fallback)
    chunks = chunk_raw_text(text, paper)
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from paperhunter.config import settings
from paperhunter.models import (
    ChunkType,
    GROBIDOutput,
    GROBIDSection,
    Paper,
    PaperChunk,
)
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ============================================================================
# Parametri globali (caricati da config.yaml → sezione 'ingestion')
# ============================================================================

def _get_chunk_size() -> int:
    """Chunk size in caratteri da config.yaml."""
    return settings.ingestion.chunk_size


def _get_chunk_overlap() -> int:
    """Overlap tra chunk consecutivi da config.yaml."""
    return settings.ingestion.chunk_overlap


# Dimensione minima sotto la quale un chunk viene scartato
_MIN_CHUNK_SIZE = 80


# ============================================================================
# API Pubblica
# ============================================================================

def chunk_grobid_output(
    grobid_output: GROBIDOutput,
    paper: Paper,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[PaperChunk]:
    """Crea chunk dal GROBIDOutput di un paper, sfruttando la struttura a sezioni.

    Strategia:
    1. Se GROBID ha estratto sezioni → ogni sezione diventa uno o più chunk.
    2. Se una sezione è più corta di chunk_size → chunk singolo.
    3. Se una sezione eccede chunk_size → split con overlap a confini di frase.
    4. Abstract e figure/table captions diventano chunk separati.
    5. Fallback: se nessuna sezione, usa full_text con rilevamento heading.

    Args:
        grobid_output: Output strutturato da GROBID.
        paper: Modello Paper con metadati (bibtex_key, category, year, ecc.).
        chunk_size: Override chunk size (default da config.yaml).
        chunk_overlap: Override overlap (default da config.yaml).

    Returns:
        Lista di PaperChunk con metadati completi per il vector store.
    """
    cs = chunk_size or _get_chunk_size()
    co = chunk_overlap or _get_chunk_overlap()

    bibtex_key = paper.bibtex_key or paper.title[:30]  # fallback per ID
    paper_id = paper.semantic_scholar_id or paper.arxiv_id or bibtex_key

    chunks: list[PaperChunk] = []
    global_index = 0  # indice globale progressivo per il paper

    # --- 1. Abstract come chunk separato ---
    abstract_text = _clean_text(grobid_output.abstract or paper.abstract or "")
    if abstract_text and len(abstract_text) >= _MIN_CHUNK_SIZE:
        chunk = _make_chunk(
            text=abstract_text,
            bibtex_key=bibtex_key,
            section_name="Abstract",
            chunk_index=global_index,
            paper=paper,
            paper_id=paper_id,
        )
        chunks.append(chunk)
        global_index += 1

    # --- 2. Sezioni GROBID ---
    if grobid_output.sections:
        for section in grobid_output.sections:
            section_text = _clean_text(section.text)
            if not section_text or len(section_text) < _MIN_CHUNK_SIZE:
                continue

            section_name = section.title.strip() if section.title else f"Section_{global_index}"

            if len(section_text) <= cs:
                # Sezione corta → chunk singolo
                chunk = _make_chunk(
                    text=section_text,
                    bibtex_key=bibtex_key,
                    section_name=section_name,
                    chunk_index=global_index,
                    paper=paper,
                    paper_id=paper_id,
                )
                chunks.append(chunk)
                global_index += 1
            else:
                # Sezione lunga → split con overlap
                fragments = _split_by_size(section_text, cs, co)
                for frag in fragments:
                    if len(frag) < _MIN_CHUNK_SIZE:
                        continue
                    chunk = _make_chunk(
                        text=frag,
                        bibtex_key=bibtex_key,
                        section_name=section_name,
                        chunk_index=global_index,
                        paper=paper,
                        paper_id=paper_id,
                    )
                    chunks.append(chunk)
                    global_index += 1
    else:
        # --- Fallback: nessuna sezione GROBID → usa full_text ---
        full_text = _clean_text(grobid_output.full_text)
        if full_text and len(full_text) >= _MIN_CHUNK_SIZE:
            logger.warning(
                f"Nessuna sezione GROBID per '{paper.title[:60]}': "
                f"fallback a splitting per dimensione dal full_text"
            )
            fallback_chunks = _chunk_from_raw_text(
                full_text, bibtex_key, paper, paper_id, cs, co, start_index=global_index
            )
            chunks.extend(fallback_chunks)
            global_index += len(fallback_chunks)

    # --- 3. Figure captions come chunk separato (se sufficienti) ---
    if grobid_output.figure_captions:
        captions_text = "\n\n".join(
            f"Figure: {cap}" for cap in grobid_output.figure_captions if cap.strip()
        )
        captions_text = _clean_text(captions_text)
        if captions_text and len(captions_text) >= _MIN_CHUNK_SIZE:
            chunk = _make_chunk(
                text=captions_text,
                bibtex_key=bibtex_key,
                section_name="Figure_Captions",
                chunk_index=global_index,
                paper=paper,
                paper_id=paper_id,
            )
            chunks.append(chunk)
            global_index += 1

    # --- 4. Table captions come chunk separato (se sufficienti) ---
    if grobid_output.table_captions:
        tables_text = "\n\n".join(
            f"Table: {cap}" for cap in grobid_output.table_captions if cap.strip()
        )
        tables_text = _clean_text(tables_text)
        if tables_text and len(tables_text) >= _MIN_CHUNK_SIZE:
            chunk = _make_chunk(
                text=tables_text,
                bibtex_key=bibtex_key,
                section_name="Table_Captions",
                chunk_index=global_index,
                paper=paper,
                paper_id=paper_id,
            )
            chunks.append(chunk)
            global_index += 1

    logger.info(
        f"Chunked '{paper.title[:50]}...' → {len(chunks)} chunk "
        f"(bibtex_key={bibtex_key}, sections={len(grobid_output.sections)})"
    )
    return chunks


def chunk_raw_text(
    text: str,
    paper: Paper,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[PaperChunk]:
    """Crea chunk da testo grezzo (senza GROBID).

    Usa rilevamento heading basato su regex per segmentare.
    Utile come fallback quando GROBID non è disponibile.

    Args:
        text: Testo completo del paper.
        paper: Modello Paper con metadati.
        chunk_size: Override chunk size.
        chunk_overlap: Override overlap.

    Returns:
        Lista di PaperChunk.
    """
    cs = chunk_size or _get_chunk_size()
    co = chunk_overlap or _get_chunk_overlap()

    bibtex_key = paper.bibtex_key or paper.title[:30]
    paper_id = paper.semantic_scholar_id or paper.arxiv_id or bibtex_key

    cleaned = _clean_text(text)
    if not cleaned or len(cleaned) < _MIN_CHUNK_SIZE:
        return []

    return _chunk_from_raw_text(cleaned, bibtex_key, paper, paper_id, cs, co, start_index=0)


def get_chunk_stats(chunks: list[PaperChunk]) -> dict:
    """Restituisce statistiche sui chunk generati.

    Returns:
        Dict con: total, avg_length, min_length, max_length,
        sections (set di section_name), categories.
    """
    if not chunks:
        return {"total": 0}

    lengths = [len(c.text) for c in chunks]
    sections = set(c.section_name for c in chunks if c.section_name)
    categories = set(c.category for c in chunks if c.category)

    return {
        "total": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": sum(lengths),
        "sections": sorted(sections),
        "section_count": len(sections),
        "categories": sorted(categories),
    }


# ============================================================================
# Funzioni interne
# ============================================================================

def _make_chunk(
    text: str,
    bibtex_key: str,
    section_name: str,
    chunk_index: int,
    paper: Paper,
    paper_id: str,
    chunk_type: ChunkType = ChunkType.ORIGINAL,
) -> PaperChunk:
    """Crea un PaperChunk con ID deterministico e metadata completi.

    Chunk ID: SHA-256 di "bibtex_key::section_name::chunk_index" (hex, 12 char).
    """
    raw_id = f"{bibtex_key}::{section_name}::{chunk_index}"
    chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

    return PaperChunk(
        chunk_id=chunk_id,
        text=text,
        chunk_type=chunk_type,
        paper_id=paper_id,
        bibtex_key=bibtex_key,
        section_name=section_name,
        category=paper.category or "",
        year=paper.year,
        month=paper.month,
        chunk_index=chunk_index,
    )


def _clean_text(text: str) -> str:
    """Normalizza whitespace e rimuove rumore dal testo."""
    if not text:
        return ""
    # Collassa newline multipli (>2) in doppio newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Rimuove numeri di pagina isolati (es. "\n 23 \n")
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    # Normalizza spazi/tab multipli in spazio singolo
    text = re.sub(r"[ \t]+", " ", text)
    # Rimuove whitespace iniziale/finale
    return text.strip()


def _split_by_size(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split testo in frammenti di dimensione fissa con overlap a confini di frase.

    Tenta di rompere ai confini ". ", ".\n", "\n\n", "\n", " "
    (in ordine di preferenza) per non spezzare frasi a metà.

    Args:
        text: Testo da splittare.
        chunk_size: Dimensione target in caratteri.
        chunk_overlap: Overlap in caratteri.

    Returns:
        Lista di frammenti di testo.
    """
    if len(text) <= chunk_size:
        return [text]

    fragments: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Cerco il confine di frase migliore nella seconda metà del chunk
            best_break = -1
            for boundary in [". ", ".\n", "\n\n", "\n", "; ", ", ", " "]:
                bp = text.rfind(boundary, start + chunk_size // 2, end)
                if bp != -1:
                    best_break = bp + len(boundary)
                    break

            if best_break > start:
                end = best_break

        fragment = text[start:end].strip()
        if fragment:
            fragments.append(fragment)

        # Avanza con overlap
        next_start = end - chunk_overlap
        if next_start <= start:
            # Evita loop infiniti: avanza almeno di chunk_size // 2
            next_start = start + max(chunk_size // 2, 1)
        start = next_start

    return fragments


# Regex per rilevamento heading in testo grezzo (fallback senza GROBID)
_SECTION_HEADING_RE = re.compile(
    r"^(?:\d{1,2}\.?\s+)?"  # Numerazione opzionale: "1. ", "2 ", "3.1 "
    r"(?:Abstract|Introduction|Related\s+Work|Background|Preliminaries|"
    r"Problem\s+(?:Statement|Formulation|Definition)|"
    r"Threat\s+Model|System\s+(?:Model|Design|Overview)|"
    r"Methodology|Method(?:s)?|Approach|"
    r"(?:Our\s+)?(?:Proposed\s+)?(?:Framework|Architecture|System)|"
    r"Experiment(?:s|al)?(?:\s+(?:Setup|Results|Design))?|"
    r"Evaluation|Results?\s*(?:and\s+(?:Discussion|Analysis))?|"
    r"Discussion|Analysis|"
    r"Ablation\s+Stud(?:y|ies)|Case\s+Stud(?:y|ies)|"
    r"Conclusion(?:s)?(?:\s+and\s+Future\s+Work)?|"
    r"Limitations|Future\s+Work|"
    r"Acknowledge?ments?|"
    r"(?:Appendix|Supplementary)\s*(?:[A-Z])?|"
    r"References|Bibliography)"
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _detect_sections_from_text(text: str) -> list[tuple[str, str]]:
    """Rileva sezioni da testo grezzo usando regex sui heading.

    Returns:
        Lista di tuple (section_name, section_text).
    """
    matches = list(_SECTION_HEADING_RE.finditer(text))

    if not matches:
        return [("Full_Text", text)]

    sections: list[tuple[str, str]] = []

    # Testo prima del primo heading
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("Preamble", preamble))

    # Ogni heading → sezione fino al prossimo heading
    for i, match in enumerate(matches):
        heading = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((heading, section_text))

    return sections


def _chunk_from_raw_text(
    text: str,
    bibtex_key: str,
    paper: Paper,
    paper_id: str,
    chunk_size: int,
    chunk_overlap: int,
    start_index: int = 0,
) -> list[PaperChunk]:
    """Crea chunk da testo grezzo con rilevamento heading (fallback).

    1. Tenta di segmentare per heading con regex.
    2. Per ogni segmento, applica _split_by_size se necessario.

    Returns:
        Lista di PaperChunk.
    """
    sections = _detect_sections_from_text(text)
    chunks: list[PaperChunk] = []
    idx = start_index

    for section_name, section_text in sections:
        if len(section_text) < _MIN_CHUNK_SIZE:
            continue

        if len(section_text) <= chunk_size:
            chunk = _make_chunk(
                text=section_text,
                bibtex_key=bibtex_key,
                section_name=section_name,
                chunk_index=idx,
                paper=paper,
                paper_id=paper_id,
            )
            chunks.append(chunk)
            idx += 1
        else:
            fragments = _split_by_size(section_text, chunk_size, chunk_overlap)
            for frag in fragments:
                if len(frag) < _MIN_CHUNK_SIZE:
                    continue
                chunk = _make_chunk(
                    text=frag,
                    bibtex_key=bibtex_key,
                    section_name=section_name,
                    chunk_index=idx,
                    paper=paper,
                    paper_id=paper_id,
                )
                chunks.append(chunk)
                idx += 1

    return chunks
