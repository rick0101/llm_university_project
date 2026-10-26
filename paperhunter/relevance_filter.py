"""
Intelligent Relevance Filter for PaperHunter.

Uses LangChain with LLM to analyze paper abstracts against user-defined criteria.
"""

import asyncio
import json
import logging
import re
from typing import Optional

from .config import settings
from .models import FilteredPaper, Paper, RelevanceResult
from .utils.llm_factory import get_pro_llm, llm_call_with_retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelevanceFilter:
    """
    LLM-powered relevance filter for scientific papers.
    
    Analyzes paper abstracts against user-defined criteria using
    LangChain and outputs structured relevance decisions.
    """
    
    SYSTEM_PROMPT = """You are an expert academic research assistant specializing in AI and LLM research.
Your task is to determine if a scientific paper is relevant to a user's specific research criteria.

You must analyze the paper's title and abstract carefully, then make a precise judgment about whether 
it matches the user's criteria. Be strict in your evaluation - only mark papers as relevant if they 
directly address the specified topics.

Consider:
1. Does the paper's main focus align with the criteria?
2. Are the key concepts mentioned in the abstract?
3. Would this paper be useful for someone researching the specified topic?

Provide your analysis in the specified JSON format."""

    USER_PROMPT = """## Paper Information

**Title:** {title}

**Abstract:** {abstract}

---

## User's Research Criteria

{criteria}

---

## Task

Analyze whether this paper is relevant to the user's research criteria.

{format_instructions}"""

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        temperature: float = None
    ):
        """
        Initialize the relevance filter.
        
        Args:
            provider: LLM provider ("google" or "deepseek")
            model: Model name to use
            temperature: Temperature for LLM responses
        """
        self.provider = provider or settings.llm_provider
        default_model = (
            settings.gemini.pro_model
            if self.provider == "google"
            else settings.deepseek.pro_model
        )
        self.model = model or default_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        
        self.llm = self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM based on configured provider."""
        if self.provider not in {"google", "deepseek"}:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        return get_pro_llm(
            provider=self.provider,
            model_name=self.model,
            temperature=self.temperature,
        )

    @staticmethod
    def _parse_result(raw: str) -> RelevanceResult:
        """Parse robusto del solo oggetto JSON prodotto dal modello."""
        text = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Risposta relevance priva di oggetto JSON")
        data = json.loads(text[start:end + 1])
        return RelevanceResult.model_validate(data)
    
    async def analyze_paper(
        self,
        paper: Paper,
        criteria: str
    ) -> FilteredPaper:
        """
        Analyze a single paper's relevance to the given criteria.
        
        Args:
            paper: The paper to analyze
            criteria: User's specific research criteria
            
        Returns:
            FilteredPaper with relevance analysis
        """
        try:
            format_instructions = (
                'Return ONLY JSON with this exact schema: '
                '{"is_relevant": true|false, "reasoning": "...", '
                '"confidence": 0.0-1.0}.'
            )
            prompt = (
                f"{self.SYSTEM_PROMPT}\n\n"
                + self.USER_PROMPT.format(
                    title=paper.title,
                    abstract=paper.abstract or "No abstract available.",
                    criteria=criteria,
                    format_instructions=format_instructions,
                )
            )
            raw = await asyncio.to_thread(llm_call_with_retry, self.llm, prompt)
            result = self._parse_result(raw)
            
            logger.info(
                f"Analyzed: '{paper.title[:50]}...' -> "
                f"Relevant: {result.is_relevant}"
            )
            
            return FilteredPaper(paper=paper, relevance=result)
            
        except Exception as e:
            logger.error(f"Failed to analyze paper '{paper.title}': {e}")
            # Return non-relevant on error with error message
            return FilteredPaper(
                paper=paper,
                relevance=RelevanceResult(
                    is_relevant=False,
                    reasoning=f"Analysis failed: {str(e)}",
                    confidence=0.0
                )
            )
    
    async def filter_papers(
        self,
        papers: list[Paper],
        criteria: str,
        batch_size: int = 1,  # Sequential by default to avoid rate limits
        delay_between_requests: float = 1.5  # Delay between each API call
    ) -> list[FilteredPaper]:
        """
        Filter a list of papers based on relevance criteria.
        
        Processes papers sequentially by default to respect rate limits.
        
        Args:
            papers: List of papers to filter
            criteria: User's specific research criteria
            batch_size: Number of papers to process in parallel (1 = sequential)
            delay_between_requests: Seconds to wait between requests
            
        Returns:
            List of FilteredPaper objects for all papers
        """
        logger.info(f"Filtering {len(papers)} papers with criteria: {criteria[:50]}...")
        
        all_results = []
        
        for i, paper in enumerate(papers):
            logger.info(f"Analyzing paper {i + 1}/{len(papers)}: {paper.title[:50]}...")
            
            result = await self.analyze_paper(paper, criteria)
            all_results.append(result)
            
            # Rate limiting between requests
            if i < len(papers) - 1:
                await asyncio.sleep(delay_between_requests)
        
        relevant_count = sum(1 for r in all_results if r.relevance.is_relevant)
        logger.info(f"Filtering complete: {relevant_count}/{len(papers)} papers relevant")
        
        return all_results
    
    def get_relevant_papers(self, filtered: list[FilteredPaper]) -> list[Paper]:
        """
        Extract only the relevant papers from filtered results.
        
        Args:
            filtered: List of FilteredPaper objects
            
        Returns:
            List of Paper objects that were marked as relevant
        """
        return [f.paper for f in filtered if f.relevance.is_relevant]


async def main():
    """Example usage of RelevanceFilter."""
    # Create sample papers
    papers = [
        Paper(
            title="Prompt Injection Attacks on Large Language Models",
            abstract="This paper explores various prompt injection attacks that can manipulate LLMs to bypass safety guardrails. We analyze attack vectors and propose defensive mechanisms.",
            url="https://arxiv.org/abs/2023.12345",
            source="arxiv"
        ),
        Paper(
            title="Efficient Training of Large Language Models",
            abstract="We present a new method for training large language models more efficiently using distributed computing and gradient checkpointing.",
            url="https://arxiv.org/abs/2023.12346",
            source="arxiv"
        ),
        Paper(
            title="Red Teaming Language Models for Jailbreak Detection",
            abstract="This work presents a comprehensive framework for red teaming LLMs. We demonstrate successful jailbreak attacks and analyze model vulnerabilities.",
            url="https://arxiv.org/abs/2023.12347",
            source="arxiv"
        ),
    ]
    
    criteria = """
    I am strictly interested in papers about:
    - Prompt Injection attacks
    - Jailbreaking LLMs
    - Red Teaming AI systems
    
    I am NOT interested in:
    - General LLM training methods
    - GDPR or data privacy
    - Reinforcement learning from human feedback
    """
    
    filter = RelevanceFilter()
    results = await filter.filter_papers(papers, criteria)
    
    print("\n=== Filtering Results ===\n")
    for filtered in results:
        status = "✓ RELEVANT" if filtered.relevance.is_relevant else "✗ NOT RELEVANT"
        print(f"{status}: {filtered.paper.title}")
        print(f"  Reasoning: {filtered.relevance.reasoning}\n")


if __name__ == "__main__":
    asyncio.run(main())
