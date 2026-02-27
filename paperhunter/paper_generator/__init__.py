"""
PaperGenerator - RAG-based scientific paper generation from PDF sources.

This subpackage provides:
- PDF text extraction with metadata
- Smart chunking preserving paper structure
- ChromaDB vector store for embedding storage and retrieval
- RAG engine for source-grounded text generation with citations
- Citation manager for tracking and formatting references
- PaperWriter: high-level pipeline orchestrating the full workflow
"""

from .pdf_processor import PDFProcessor, PaperMetadata, ProcessedPaper
from .chunker import PaperChunker, Chunk
from .vector_store import VectorStore
from .citation_manager import CitationManager, Citation
from .rag_engine import RAGEngine
from .paper_writer import PaperWriter

__all__ = [
    "PDFProcessor",
    "PaperMetadata",
    "ProcessedPaper",
    "PaperChunker",
    "Chunk",
    "VectorStore",
    "CitationManager",
    "Citation",
    "RAGEngine",
    "PaperWriter",
]
