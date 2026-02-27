"""
RAG Engine - Retrieval-Augmented Generation with citations.

Queries the vector store for relevant chunks and uses an LLM to generate
academic text grounded exclusively in the retrieved sources.
"""

import logging
import os
import re
from typing import Optional

from .citation_manager import CitationManager
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieves relevant chunks and generates source-grounded text."""

    def __init__(
        self,
        vector_store: VectorStore,
        citation_manager: CitationManager,
        llm_provider: str = "google",
        llm_model: str = "gemini-3-flash-preview",
        heavy_llm_model: str = "gemini-3-pro-preview",
        temperature: float = 0.3,
    ):
        self.vector_store = vector_store
        self.citation_manager = citation_manager
        self.llm_provider = llm_provider
        self.llm_model = llm_model                # fast model for queries
        self.heavy_llm_model = heavy_llm_model    # powerful model for paper writing
        self.temperature = temperature
        self._llm = None        # lazy: fast model
        self._heavy_llm = None  # lazy: heavy model

    # ── LLM initialisation ───────────────────────────────────────────────

    def _create_llm(self, model_name: str):
        """Create an LLM instance for the given model name."""
        if self.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            )
        elif self.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model_name,
                temperature=self.temperature,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _ensure_llm(self) -> None:
        """Initialise the fast LLM on first use (lazy)."""
        if self._llm is None:
            logger.info(f"Initialising fast LLM: {self.llm_model}")
            self._llm = self._create_llm(self.llm_model)

    def _ensure_heavy_llm(self) -> None:
        """Initialise the heavy/powerful LLM on first use (lazy)."""
        if self._heavy_llm is None:
            logger.info(f"Initialising heavy LLM: {self.heavy_llm_model}")
            self._heavy_llm = self._create_llm(self.heavy_llm_model)

    # ── Public API ───────────────────────────────────────────────────────

    def query_with_sources(
        self,
        question: str,
        n_results: int = 15,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Answer a question using only the knowledge base.

        Returns:
            dict with keys: answer, sources, citations_used, chunks_retrieved
        """
        chunks = self.vector_store.query(query_text=question, n_results=n_results)

        if not chunks:
            return {
                "answer": "No relevant sources found in the knowledge base.",
                "sources": [],
                "citations_used": [],
                "chunks_retrieved": 0,
            }

        source_context = self.citation_manager.format_source_context(chunks)

        prompt = (
            f"{system_prompt or self._default_system_prompt()}\n\n"
            f"## Source Materials\n\n{source_context}\n\n"
            f"## Question / Task\n{question}\n\n"
            f"## Your Response (cite with [N]):"
        )

        self._ensure_llm()
        response = self._llm.invoke(prompt)
        answer: str = self._extract_text(response.content)

        used_keys = self.citation_manager.extract_used_keys(answer)

        sources = self._collect_sources(chunks, used_keys)

        return {
            "answer": answer,
            "sources": sources,
            "citations_used": sorted(used_keys),
            "chunks_retrieved": len(chunks),
        }

    def generate_section(
        self,
        section_title: str,
        section_description: str,
        n_results: int = 20,
        language: str = "english",
    ) -> dict:
        """
        Generate one section of the final paper.

        Uses multiple queries to maximise source coverage:
        one broad query for the section title, one for the detailed description.

        Returns:
            dict with keys: content, sources, citations_used
        """
        # Multi-query retrieval for better coverage
        chunks_1 = self.vector_store.query(
            query_text=section_title, n_results=n_results // 2
        )
        chunks_2 = self.vector_store.query(
            query_text=f"{section_title}: {section_description}",
            n_results=n_results,
        )

        # Deduplicate by chunk_id while preserving order
        seen: set[str] = set()
        chunks: list[dict] = []
        for c in chunks_1 + chunks_2:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                chunks.append(c)

        if not chunks:
            return {
                "content": f"_No relevant sources found for this section._",
                "sources": [],
                "citations_used": [],
            }

        source_context = self.citation_manager.format_source_context(chunks)

        prompt = (
            "You are a scientific paper writer creating a section for a peer-reviewed article.\n\n"
            "STRICT RULES:\n"
            "1. Write ONLY based on the provided source materials. NEVER invent information.\n"
            "2. Cite every factual claim using [N] notation where N is the source number.\n"
            "3. Be precise, detailed, and academic in tone.\n"
            "4. Synthesise information across multiple sources when they cover the same topic.\n"
            "5. If sources disagree, present both viewpoints and cite each.\n"
            f"6. Write in {language}.\n"
            "7. Do NOT include the section title — write the body content only.\n"
            "8. Aim for comprehensive coverage — use as many relevant sources as possible.\n\n"
            f"## Source Materials\n\n{source_context}\n\n"
            f"## Section to Write\n"
            f"Title: {section_title}\n"
            f"Description: {section_description}\n\n"
            f"## Content:"
        )

        self._ensure_heavy_llm()
        response = self._heavy_llm.invoke(prompt)
        content: str = self._extract_text(response.content)

        used_keys = self.citation_manager.extract_used_keys(content)
        sources = self._collect_sources(chunks, used_keys)

        return {
            "content": content,
            "sources": sources,
            "citations_used": sorted(used_keys),
        }

    def generate_abstract(
        self,
        paper_title: str,
        sections_summary: str,
        language: str = "english",
    ) -> str:
        """Generate an abstract summarising the entire paper."""
        prompt = (
            "Write a concise academic abstract (150-300 words) for the following paper.\n"
            f"Language: {language}\n\n"
            f"Paper title: {paper_title}\n\n"
            f"Summary of sections:\n{sections_summary}\n\n"
            "Abstract:"
        )
        self._ensure_heavy_llm()
        response = self._heavy_llm.invoke(prompt)
        return self._extract_text(response.content)

    # ── Private helpers ──────────────────────────────────────────────────

    def _collect_sources(
        self, chunks: list[dict], used_keys: set[int]
    ) -> list[dict]:
        """Build a list of source metadata for the keys actually used."""
        sources: list[dict] = []
        seen_titles: set[str] = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            key = self.citation_manager.get_or_create_citation(meta)
            title = meta.get("paper_title", "")
            if key in used_keys and title not in seen_titles:
                seen_titles.add(title)
                sources.append(
                    {
                        "citation_key": key,
                        "paper_title": title,
                        "authors": meta.get("authors", ""),
                        "year": meta.get("year", 0),
                        "filename": meta.get("filename", ""),
                    }
                )
        return sources

    @staticmethod
    def _extract_text(content) -> str:
        """Normalise LLM response content to a plain string.

        Some LangChain / google-genai versions return a list of content
        parts instead of a single string.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a scientific research assistant. Answer EXCLUSIVELY "
            "based on the provided source materials.\n\n"
            "RULES:\n"
            "1. ONLY use information from the sources. Never invent facts.\n"
            "2. Cite every claim using [N] notation (N = source number).\n"
            "3. If sources conflict, present both perspectives with citations.\n"
            "4. If sources are insufficient, say so explicitly.\n"
            "5. Be precise and academic in tone."
        )
