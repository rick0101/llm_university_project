"""
Paper Chunker - Splits papers into semantically meaningful chunks.

Preserves paper metadata on every chunk so that sources can always
be traced back to the original PDF, title, and authors.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field

from .pdf_processor import ProcessedPaper

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with full source metadata."""

    text: str
    chunk_id: str
    metadata: dict = field(default_factory=dict)
    # metadata keys: paper_title, authors, filename, year,
    #                chunk_index, source_path, total_pages


class PaperChunker:
    """
    Splits processed papers into overlapping text chunks.

    Strategy:
    1. Try to split by section headings first (Introduction, Methods, …).
    2. Within each section, split by character count with overlap.
    3. Prefer breaking at sentence boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """
        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
            min_chunk_size: Minimum characters for a valid chunk.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    # ── Public API ───────────────────────────────────────────────────────

    def chunk_paper(self, paper: ProcessedPaper) -> list[Chunk]:
        """Split a single paper into chunks with metadata."""
        text = self._clean_text(paper.full_text)

        if len(text) < self.min_chunk_size:
            return []

        sections = self._split_into_sections(text)

        chunks: list[Chunk] = []
        chunk_index = 0

        for section_text in sections:
            for fragment in self._split_by_size(section_text):
                if len(fragment) < self.min_chunk_size:
                    continue

                chunk_id = self._generate_chunk_id(
                    paper.metadata.filename, chunk_index
                )

                meta = {
                    "paper_title": paper.metadata.title,
                    "authors": (
                        ", ".join(paper.metadata.authors)
                        if paper.metadata.authors
                        else "Unknown"
                    ),
                    "filename": paper.metadata.filename,
                    "year": paper.metadata.year or 0,
                    "chunk_index": chunk_index,
                    "source_path": paper.metadata.source_path,
                    "total_pages": paper.num_pages,
                }

                chunks.append(Chunk(text=fragment, chunk_id=chunk_id, metadata=meta))
                chunk_index += 1

        logger.debug(
            f"Created {len(chunks)} chunks from '{paper.metadata.title}'"
        )
        return chunks

    def chunk_papers(self, papers: list[ProcessedPaper]) -> list[Chunk]:
        """Chunk multiple papers at once."""
        all_chunks: list[Chunk] = []
        for paper in papers:
            all_chunks.extend(self.chunk_paper(paper))
        logger.info(
            f"Total chunks created: {len(all_chunks)} from {len(papers)} papers"
        )
        return all_chunks

    # ── Private helpers ──────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and remove noise."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n\d+\n", "\n", text)          # isolated page numbers
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    _SECTION_RE = re.compile(
        r"\n(?=(?:\d+\.?\s+)?(?:Abstract|Introduction|Related\s+Work|Background|"
        r"Methodology|Method|Methods|Approach|Experiments?|Experimental|Results|"
        r"Evaluation|Discussion|Conclusion|Limitations|Future\s+Work|"
        r"Acknowledgment|References)s?\b)",
        re.IGNORECASE,
    )

    def _split_into_sections(self, text: str) -> list[str]:
        """Split text at common academic section headings."""
        sections = self._SECTION_RE.split(text)
        sections = [s.strip() for s in sections if s.strip()]
        return sections if sections else [text]

    def _split_by_size(self, text: str) -> list[str]:
        """Split text into fixed-size chunks with overlap at sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                # Try to break at a sentence boundary
                for boundary in [". ", ".\n", "\n\n", "\n", " "]:
                    bp = text.rfind(boundary, start + self.chunk_size // 2, end)
                    if bp != -1:
                        end = bp + len(boundary)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    @staticmethod
    def _generate_chunk_id(filename: str, chunk_index: int) -> str:
        """Generate a deterministic unique chunk ID."""
        raw = f"{filename}::{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()
