"""
Paper Writer - High-level pipeline for generating a full scientific paper.

Orchestrates PDF ingestion, vector store population, section-by-section
generation, and bibliography assembly.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .chunker import PaperChunker
from .citation_manager import CitationManager
from .pdf_processor import PDFProcessor
from .rag_engine import RAGEngine
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class PaperWriter:
    """
    End-to-end pipeline: PDFs → Vector DB → Generated Paper with citations.

    Typical usage::

        writer = PaperWriter(llm_model="gemini-3-flash-preview", heavy_llm_model="gemini-3-pro-preview")
        writer.ingest_papers(Path("./my_pdfs"))
        paper = writer.generate_paper(
            title="A Survey on LLM Security",
            outline={
                "Introduction": "Overview of …",
                "Threat Model": "Describe …",
                "Conclusion": "Summarise …",
            },
            output_dir=Path("./output"),
        )
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "papers",
        llm_provider: str = "google",
        llm_model: str = "gemini-3-flash-preview",
        heavy_llm_model: str = "gemini-3-pro-preview",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.3,
    ):
        self.pdf_processor = PDFProcessor()
        self.chunker = PaperChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
        self.citation_manager = CitationManager()
        self.rag_engine = RAGEngine(
            vector_store=self.vector_store,
            citation_manager=self.citation_manager,
            llm_provider=llm_provider,
            llm_model=llm_model,
            heavy_llm_model=heavy_llm_model,
            temperature=temperature,
        )

    # ── Ingestion ────────────────────────────────────────────────────────

    def ingest_papers(
        self, pdf_dir: Path, recursive: bool = True
    ) -> dict:
        """
        Extract text from all PDFs in *pdf_dir*, chunk them,
        and add them to the vector store.

        Returns:
            Summary dict with processing statistics.
        """
        logger.info(f"=== Ingestion start: {pdf_dir} ===")

        papers = self.pdf_processor.process_directory(pdf_dir, recursive=recursive)
        if not papers:
            logger.warning("No papers were processed.")
            return {"papers_processed": 0, "chunks_created": 0}

        chunks = self.chunker.chunk_papers(papers)
        added = self.vector_store.add_chunks(chunks)
        stats = self.vector_store.get_collection_stats()

        summary = {
            "papers_processed": len(papers),
            "chunks_created": len(chunks),
            "chunks_added_to_store": added,
            "total_chunks_in_store": stats["total_chunks"],
            "total_papers_in_store": stats["total_papers"],
        }

        logger.info(f"=== Ingestion complete: {json.dumps(summary, indent=2)} ===")
        return summary

    # ── Paper generation ─────────────────────────────────────────────────

    def generate_paper(
        self,
        title: str,
        outline: dict[str, str],
        n_results_per_section: int = 20,
        output_dir: Optional[Path] = None,
        language: str = "english",
        generate_abstract: bool = True,
    ) -> str:
        """
        Generate a complete paper from a section outline.

        Args:
            title: Paper title.
            outline: Mapping of section_title → description / instructions.
            n_results_per_section: Chunks retrieved per section.
            output_dir: If given, save .md and _meta.json there.
            language: Output language (e.g. "english", "italian").
            generate_abstract: Auto-generate an abstract at the end.

        Returns:
            The full paper as a Markdown string.
        """
        logger.info(f"{'=' * 60}")
        logger.info(f"Generating paper: {title}")
        logger.info(f"Sections: {list(outline.keys())}")
        logger.info(f"{'=' * 60}")

        # Fresh citation set for every new paper
        self.citation_manager.reset()

        parts: list[str] = [f"# {title}\n"]
        parts.append(
            f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"
        )

        sections_summary_parts: list[str] = []

        for i, (sec_title, sec_desc) in enumerate(outline.items(), 1):
            logger.info(f"[{i}/{len(outline)}] Generating: {sec_title}")

            result = self.rag_engine.generate_section(
                section_title=sec_title,
                section_description=sec_desc,
                n_results=n_results_per_section,
                language=language,
            )

            parts.append(f"\n## {sec_title}\n")
            parts.append(result["content"])

            sections_summary_parts.append(
                f"- {sec_title}: {sec_desc}"
            )

            n_cit = len(result.get("citations_used", []))
            logger.info(f"  → {n_cit} citations used")

        # Optional auto-abstract
        if generate_abstract:
            logger.info("Generating abstract …")
            abstract = self.rag_engine.generate_abstract(
                paper_title=title,
                sections_summary="\n".join(sections_summary_parts),
                language=language,
            )
            # Insert abstract right after the title
            parts.insert(2, f"\n## Abstract\n\n{abstract}\n")

        # Bibliography
        bibliography = self.citation_manager.generate_bibliography()
        parts.append(f"\n\n{bibliography}")

        full_paper = "\n\n".join(parts)

        # Persist to disk
        if output_dir:
            self._save_paper(full_paper, title, outline, output_dir)

        total_cit = self.citation_manager.get_citation_count()
        logger.info(f"Paper generation complete — {total_cit} unique citations")
        return full_paper

    # ── Interactive query ────────────────────────────────────────────────

    def query(self, question: str, n_results: int = 15) -> dict:
        """Ask a free-form question against the knowledge base."""
        return self.rag_engine.query_with_sources(
            question=question, n_results=n_results
        )

    def get_stats(self) -> dict:
        """Return vector store statistics."""
        return self.vector_store.get_collection_stats()

    # ── Private helpers ──────────────────────────────────────────────────

    def _save_paper(
        self,
        full_paper: str,
        title: str,
        outline: dict[str, str],
        output_dir: Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in title
        ).replace(" ", "_")[:50]

        md_path = output_dir / f"paper_{safe}_{ts}.md"
        md_path.write_text(full_paper, encoding="utf-8")
        logger.info(f"Paper saved → {md_path}")

        meta_path = output_dir / f"paper_{safe}_{ts}_meta.json"
        meta = {
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "total_citations": self.citation_manager.get_citation_count(),
            "citations": [
                {
                    "key": c.key,
                    "title": c.paper_title,
                    "authors": c.authors,
                    "year": c.year,
                    "filename": c.filename,
                }
                for c in self.citation_manager.get_all_citations()
            ],
            "outline": outline,
            "collection_stats": self.vector_store.get_collection_stats(),
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Metadata saved → {meta_path}")
