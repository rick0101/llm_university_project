"""
PDF Processor - Extracts text and metadata from scientific paper PDFs.

Uses PyMuPDF (fitz) for fast, reliable text extraction.
Handles metadata extraction including title, authors, year, and abstract.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Metadata extracted from a PDF paper."""

    title: str
    authors: list[str] = field(default_factory=list)
    filename: str = ""
    year: Optional[int] = None
    abstract: Optional[str] = None
    source_path: str = ""


@dataclass
class ProcessedPaper:
    """A fully processed paper with extracted text and metadata."""

    metadata: PaperMetadata
    full_text: str
    pages: list[str] = field(default_factory=list)
    num_pages: int = 0


class PDFProcessor:
    """Extracts text and metadata from PDF files."""

    def process_pdf(self, pdf_path: Path) -> Optional[ProcessedPaper]:
        """
        Process a single PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ProcessedPaper with text and metadata, or None if processing fails.
        """
        try:
            doc = fitz.open(str(pdf_path))

            # Extract text from each page
            pages: list[str] = []
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    pages.append(text)

            if not pages:
                logger.warning(f"No text extracted from {pdf_path.name}")
                doc.close()
                return None

            full_text = "\n\n".join(pages)

            # Extract metadata
            meta = doc.metadata or {}
            title = meta.get("title", "")
            if not title or len(title.strip()) < 5:
                title = self._extract_title_from_text(pages[0])
            if not title:
                title = pdf_path.stem.replace("_", " ").replace("-", " ").title()

            authors = self._parse_authors(meta.get("author", ""))
            year = self._extract_year(meta, full_text)
            abstract = self._extract_abstract(full_text)

            doc.close()

            metadata = PaperMetadata(
                title=title.strip(),
                authors=authors,
                filename=pdf_path.name,
                year=year,
                abstract=abstract,
                source_path=str(pdf_path.resolve()),
            )

            return ProcessedPaper(
                metadata=metadata,
                full_text=full_text,
                pages=pages,
                num_pages=len(pages),
            )

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return None

    def process_directory(
        self, dir_path: Path, recursive: bool = True
    ) -> list[ProcessedPaper]:
        """
        Process all PDFs in a directory.

        Args:
            dir_path: Path to the directory containing PDFs.
            recursive: Whether to search subdirectories.

        Returns:
            List of successfully processed papers.
        """
        papers: list[ProcessedPaper] = []
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(dir_path.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {dir_path}")
            return papers

        logger.info(f"Found {len(pdf_files)} PDF files in {dir_path}")

        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"  [{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            paper = self.process_pdf(pdf_path)
            if paper and paper.full_text.strip():
                papers.append(paper)
            else:
                logger.warning(f"  Skipped {pdf_path.name}: no usable text extracted")

        logger.info(f"Successfully processed {len(papers)}/{len(pdf_files)} papers")
        return papers

    # ── Private helpers ──────────────────────────────────────────────────

    def _extract_title_from_text(self, first_page: str) -> str:
        """Try to extract the title from the first page text."""
        lines = first_page.strip().split("\n")
        candidates: list[str] = []
        for line in lines[:10]:
            line = line.strip()
            # Skip very short lines, arXiv headers, dates, etc.
            if len(line) < 10:
                continue
            if re.match(r'^(arXiv|http|www\.|doi:|preprint|\d{4})', line, re.IGNORECASE):
                continue
            if re.match(r'^\d+$', line):
                continue
            candidates.append(line)
            if len(candidates) >= 2:
                break

        if candidates:
            # Title is often the longest of the first few meaningful lines
            return max(candidates, key=len)
        return ""

    def _parse_authors(self, author_str: str) -> list[str]:
        """Parse author string into a list of names."""
        if not author_str or not author_str.strip():
            return []
        # Handle various separators
        for sep in [";", " and ", ", and "]:
            if sep in author_str:
                return [a.strip() for a in author_str.split(sep) if a.strip()]
        # Try comma separation (but careful with "Last, First" format)
        parts = [a.strip() for a in author_str.split(",")]
        if len(parts) > 1 and all(len(p.split()) <= 2 for p in parts):
            return parts
        return [author_str.strip()]

    def _extract_year(self, metadata: dict, text: str) -> Optional[int]:
        """Extract publication year from metadata or text."""
        # Try PDF metadata dates first
        for key in ["creationDate", "modDate"]:
            date_str = metadata.get(key, "")
            if date_str:
                match = re.search(r"(20\d{2})", date_str)
                if match:
                    return int(match.group(1))

        # Try text header area (first 2000 chars)
        match = re.search(r"(20[12]\d)", text[:2000])
        if match:
            return int(match.group(1))
        return None

    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract the abstract from the paper text."""
        patterns = [
            r"(?i)abstract\s*[:\-—]?\s*\n?(.*?)(?:\n\s*\n|\n(?:\d+\.?\s*)?[Ii]ntroduction)",
            r"(?i)abstract\s*\n(.*?)(?:\n\s*\n)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Reasonable length for an abstract
                if 50 < len(abstract) < 3000:
                    return abstract
        return None
