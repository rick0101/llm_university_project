"""
Multi-Source Paper Aggregator for PaperHunter.

This module implements asynchronous searching across multiple sources:
- ArXiv (using official arxiv library)
- Semantic Scholar (using their API)
- Hugging Face Papers (web scraping)
"""

import asyncio
import logging
import random
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional

import arxiv
import httpx
from bs4 import BeautifulSoup

from .config import settings
from .models import Paper, SearchResult, SourceType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 2.0):
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be multiplied exponentially)
        
    Returns:
        Result of the function
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    raise last_exception


def _parse_date_bound(date_str: str | None) -> datetime | None:
    """Parse a YYYY-MM or YYYY-MM-DD string into a UTC-aware datetime.

    Returns ``None`` when *date_str* is ``None`` or empty.
    The returned datetime is always timezone-aware (UTC) so it can
    be safely compared with ArXiv's offset-aware published dates.
    """
    if not date_str:
        return None
    try:
        if len(date_str) == 7:  # YYYY-MM
            dt = datetime.strptime(date_str, "%Y-%m")
        else:
            dt = datetime.fromisoformat(date_str)
        # Ensure the result is timezone-aware (UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        logger.warning(f"Invalid date bound '{date_str}' — ignoring")
        return None


def _paper_in_date_range(
    paper: Paper,
    date_from: datetime | None,
    date_to: datetime | None,
) -> bool:
    """Return True if *paper.published_date* falls within [date_from, date_to].

    Papers without a ``published_date`` are kept (conservative approach).
    ``date_to`` is treated as *inclusive* of the entire month
    (i.e. ``2026-03`` means "up to end of March 2026").
    """
    if paper.published_date is None:
        return True  # keep papers without a date
    pd = paper.published_date
    # Normalize to UTC-aware so comparisons never fail
    if pd.tzinfo is None:
        pd = pd.replace(tzinfo=timezone.utc)
    if date_from and pd < date_from:
        return False
    if date_to:
        # inclusive: if date_to is 2026-03-01 00:00, we want papers up to end of March
        import calendar
        last_day = calendar.monthrange(date_to.year, date_to.month)[1]
        end = date_to.replace(day=last_day, hour=23, minute=59, second=59)
        if pd > end:
            return False
    return True


def _split_or_query(query: str) -> list[str]:
    """Split a query string on top-level ``OR`` operators.

    Preserves quoted phrases.  A query without ``OR`` returns a
    single-element list with the original query unchanged.

    Examples::

        >>> _split_or_query('"jailbreak prompt" LLM OR "adversarial prompt"')
        ['"jailbreak prompt" LLM', '"adversarial prompt"']
        >>> _split_or_query('simple query')
        ['simple query']
    """
    parts = re.split(r'\s+OR\s+', query)
    return [p.strip() for p in parts if p.strip()]


class ArxivStrategy:
    """Search strategy for ArXiv papers."""
    
    def __init__(self, max_results: int = 20, categories: list[str] = None):
        self.max_results = max_results
        self.categories = categories or ["cs.CL", "cs.AI"]
    
    async def search(self, query: str, date_from: str | None = None, date_to: str | None = None) -> list[Paper]:
        """
        Search ArXiv for papers matching the query.
        
        Args:
            query: Search keywords
            date_from: Optional start date (YYYY-MM or YYYY-MM-DD)
            date_to: Optional end date (YYYY-MM or YYYY-MM-DD)
            
        Returns:
            List of Paper objects from ArXiv
        """
        try:
            # Build category filter
            cat_filter = " OR ".join([f"cat:{cat}" for cat in self.categories])
            full_query = f"({query}) AND ({cat_filter})"
            
            logger.info(f"Searching ArXiv with query: {full_query}")
            
            # ArXiv library is synchronous, run in executor
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._sync_search,
                full_query
            )
            
            papers = []
            for result in results:
                # Extract GitHub URL from abstract or comments
                github_url = self._extract_github_url(
                    result.summary + " " + (result.comment or "")
                )
                
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    source=SourceType.ARXIV,
                    published_date=result.published,
                    arxiv_id=result.get_short_id(),
                    categories=list(result.categories),
                    github_url=github_url
                )
                papers.append(paper)
            
            # --- Date filtering ---
            df = _parse_date_bound(date_from)
            dt = _parse_date_bound(date_to)
            if df or dt:
                before = len(papers)
                papers = [p for p in papers if _paper_in_date_range(p, df, dt)]
                logger.info(f"ArXiv date filter: {before} → {len(papers)} papers")

            logger.info(f"Found {len(papers)} papers from ArXiv")
            return papers
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            raise
    
    def _sync_search(self, query: str) -> list:
        """Synchronous ArXiv search with conservative settings."""
        # Use conservative settings to avoid rate limiting
        client = arxiv.Client(
            page_size=10,  # Small page size to reduce load
            delay_seconds=3.0,  # Delay between requests
            num_retries=3
        )
        search = arxiv.Search(
            query=query,
            max_results=min(self.max_results, 20),  # Cap at 20 to avoid 429
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        return list(client.results(search))
    
    def _extract_github_url(self, text: str) -> Optional[str]:
        """Extract GitHub URL from text."""
        pattern = r'https?://github\.com/[\w\-\.]+/[\w\-\.]+'
        match = re.search(pattern, text)
        return match.group(0) if match else None


class SemanticScholarStrategy:
    """Search strategy for Semantic Scholar papers."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, max_results: int = 20):
        self.max_results = max_results
        self.api_key = settings.semantic_scholar_api_key
    
    async def search(self, query: str, date_from: str | None = None, date_to: str | None = None) -> list[Paper]:
        """
        Search Semantic Scholar for papers matching the query.
        
        Args:
            query: Search keywords
            date_from: Optional start date (YYYY-MM or YYYY-MM-DD)
            date_to: Optional end date (YYYY-MM or YYYY-MM-DD)
            
        Returns:
            List of Paper objects from Semantic Scholar
        """
        # Skip if no API key - Semantic Scholar heavily rate limits without one
        if not self.api_key:
            logger.warning("Semantic Scholar: No API key configured. Skipping to avoid rate limits. "
                          "Get a free key at https://www.semanticscholar.org/product/api")
            return []

        # --- Handle OR queries (S2 API doesn't support boolean OR) ---
        sub_queries = _split_or_query(query)
        if len(sub_queries) > 1:
            logger.info(f"Semantic Scholar: OR query → {len(sub_queries)} sub-queries")
            all_papers: list[Paper] = []
            seen_titles: set[str] = set()
            for i, sq in enumerate(sub_queries):
                try:
                    papers = await self.search(sq, date_from, date_to)
                    for p in papers:
                        norm = unicodedata.normalize("NFKD", p.title).lower().strip()
                        if norm not in seen_titles:
                            seen_titles.add(norm)
                            all_papers.append(p)
                except Exception as e:
                    logger.warning(f"S2 sub-query '{sq}' failed: {e}")
                if i < len(sub_queries) - 1:
                    await asyncio.sleep(1.0)
            logger.info(f"Found {len(all_papers)} unique papers from Semantic Scholar ({len(sub_queries)} sub-queries)")
            return all_papers

        try:
            logger.info(f"Searching Semantic Scholar with query: {query}")
            
            headers = {
                "User-Agent": "PaperHunter/1.0 (Academic Research Tool)",
                "x-api-key": self.api_key
            }
            
            params = {
                "query": query,
                "limit": min(self.max_results, 10),  # Conservative limit
                "fields": "title,authors,abstract,url,externalIds,publicationDate,citationCount,openAccessPdf"
            }

            # Semantic Scholar supports publicationDateOrYear filter (YYYY-MM-DD:YYYY-MM-DD)
            if date_from or date_to:
                s2_from = date_from + "-01" if date_from and len(date_from) == 7 else (date_from or "")
                s2_to_raw = date_to or ""
                if s2_to_raw and len(s2_to_raw) == 7:
                    import calendar
                    y, m = int(s2_to_raw[:4]), int(s2_to_raw[5:7])
                    last_day = calendar.monthrange(y, m)[1]
                    s2_to_raw = f"{s2_to_raw}-{last_day:02d}"
                params["publicationDateOrYear"] = f"{s2_from}:{s2_to_raw}"
            
            async def make_request():
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(
                        f"{self.BASE_URL}/paper/search",
                        params=params,
                        headers=headers
                    )
                    if response.status_code == 429:
                        raise httpx.HTTPStatusError(
                            "Rate limited by Semantic Scholar",
                            request=response.request,
                            response=response
                        )
                    response.raise_for_status()
                    return response.json()
            
            data = await retry_with_backoff(make_request, max_retries=3, base_delay=5.0)
            
            papers = []
            for item in data.get("data", []):
                # Extract ArXiv ID if available
                external_ids = item.get("externalIds", {}) or {}
                arxiv_id = external_ids.get("ArXiv")
                doi = external_ids.get("DOI")
                
                # Get PDF URL
                pdf_url = None
                open_access = item.get("openAccessPdf")
                if open_access:
                    pdf_url = open_access.get("url")
                
                # Parse publication date
                pub_date = None
                if item.get("publicationDate"):
                    try:
                        pub_date = datetime.fromisoformat(item["publicationDate"])
                    except ValueError:
                        pass
                
                paper = Paper(
                    title=item.get("title", ""),
                    authors=[
                        author.get("name", "") 
                        for author in item.get("authors", [])
                    ],
                    abstract=item.get("abstract") or "",
                    url=item.get("url", ""),
                    pdf_url=pdf_url,
                    source=SourceType.SEMANTIC_SCHOLAR,
                    published_date=pub_date,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    citations=item.get("citationCount")
                )
                papers.append(paper)
            
            # Rate limiting
            await asyncio.sleep(settings.request_delay)
            
            # --- Date filtering (post-filter for safety) ---
            df = _parse_date_bound(date_from)
            dt = _parse_date_bound(date_to)
            if df or dt:
                before = len(papers)
                papers = [p for p in papers if _paper_in_date_range(p, df, dt)]
                logger.info(f"S2 date filter: {before} → {len(papers)} papers")

            logger.info(f"Found {len(papers)} papers from Semantic Scholar")
            return papers
            
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            raise


class HuggingFaceStrategy:
    """Search strategy for Hugging Face papers with search functionality."""
    
    BASE_URL = "https://huggingface.co/papers"
    
    def __init__(self, max_results: int = 20):
        self.max_results = max_results
    
    async def search(self, query: str, date_from: str | None = None, date_to: str | None = None) -> list[Paper]:
        """
        Search Hugging Face papers using search query.
        
        Args:
            query: Search keywords
            date_from: Optional start date (YYYY-MM or YYYY-MM-DD) — used for post-filtering
            date_to: Optional end date (YYYY-MM or YYYY-MM-DD) — used for post-filtering
            
        Returns:
            List of Paper objects from Hugging Face
        """
        # --- Handle OR queries (HF search doesn't support boolean OR) ---
        sub_queries = _split_or_query(query)
        if len(sub_queries) > 1:
            logger.info(f"HuggingFace: OR query → {len(sub_queries)} sub-queries")
            all_papers: list[Paper] = []
            seen_titles: set[str] = set()
            for i, sq in enumerate(sub_queries):
                try:
                    papers = await self.search(sq, date_from, date_to)
                    for p in papers:
                        norm = unicodedata.normalize("NFKD", p.title).lower().strip()
                        if norm not in seen_titles:
                            seen_titles.add(norm)
                            all_papers.append(p)
                except Exception as e:
                    logger.warning(f"HF sub-query '{sq}' failed: {e}")
                if i < len(sub_queries) - 1:
                    await asyncio.sleep(1.0)
            logger.info(f"Found {len(all_papers)} unique papers from HuggingFace ({len(sub_queries)} sub-queries)")
            return all_papers

        try:
            # Use search URL with query parameter
            search_url = f"{self.BASE_URL}?q={query.replace(' ', '+')}"
            logger.info(f"Searching Hugging Face papers: {search_url}")
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(search_url, headers=headers)
                response.raise_for_status()
                html = response.text
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Find all paper links and deduplicate by arxiv_id
            paper_links = soup.find_all("a", href=re.compile(r"^/papers/\d+\.\d+"))
            
            # Group by arxiv_id and pick best title
            papers_by_id = {}
            for link in paper_links:
                href = link.get("href", "")
                # Skip community/comment links
                if "#" in href:
                    continue
                    
                arxiv_match = re.search(r"/papers/([\d\.]+)", href)
                if not arxiv_match:
                    continue
                    
                arxiv_id = arxiv_match.group(1)
                text = link.get_text(strip=True)
                
                # Skip short texts (likely not titles)
                if len(text) < 15:
                    continue
                
                # Skip texts that look like abstracts (too long or start with lowercase)
                if len(text) > 200 or (text and text[0].islower()):
                    continue
                
                # Keep the best title (prefer longer, title-case text)
                if arxiv_id not in papers_by_id:
                    papers_by_id[arxiv_id] = text
                elif len(text) > len(papers_by_id[arxiv_id]) and len(text) < 200:
                    papers_by_id[arxiv_id] = text
            
            # Create Paper objects
            papers = []
            for arxiv_id, title in list(papers_by_id.items())[:self.max_results]:
                paper = Paper(
                    title=title,
                    authors=[],
                    abstract="",
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    source=SourceType.HUGGINGFACE,
                    arxiv_id=arxiv_id,
                    github_url=None
                )
                papers.append(paper)
            
            # Rate limiting
            await asyncio.sleep(settings.request_delay)
            
            # HuggingFace papers come from ArXiv → try to infer date from ArXiv ID
            for p in papers:
                if p.arxiv_id and not p.published_date:
                    m = re.match(r'(\d{2})(\d{2})', p.arxiv_id)
                    if m:
                        yr = 2000 + int(m.group(1))
                        mo = int(m.group(2))
                        if 1 <= mo <= 12:
                            p.published_date = datetime(yr, mo, 1)

            # --- Date filtering ---
            df = _parse_date_bound(date_from)
            dt = _parse_date_bound(date_to)
            if df or dt:
                before = len(papers)
                papers = [p for p in papers if _paper_in_date_range(p, df, dt)]
                logger.info(f"HuggingFace date filter: {before} → {len(papers)} papers")

            logger.info(f"Found {len(papers)} papers from Hugging Face")
            return papers
            
        except Exception as e:
            logger.error(f"Hugging Face search failed: {e}")
            raise


class PaperAggregator:
    """
    Aggregates paper search results from multiple sources asynchronously.
    
    Uses asyncio.gather() to run searches in parallel and deduplicates
    results based on normalized titles.
    """
    
    def __init__(
        self,
        max_results_per_source: int = 20,
        categories: list[str] = None
    ):
        """
        Initialize the aggregator.
        
        Args:
            max_results_per_source: Maximum number of results per source
            categories: ArXiv categories to search (default: cs.CL, cs.AI)
        """
        self.max_results = max_results_per_source
        self.categories = categories or settings.default_categories
        
        # Initialize strategies
        self.arxiv = ArxivStrategy(max_results_per_source, self.categories)
        self.semantic_scholar = SemanticScholarStrategy(max_results_per_source)
        self.huggingface = HuggingFaceStrategy(max_results_per_source)
    
    async def search(
        self,
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> SearchResult:
        """
        Search all sources in parallel and aggregate results.
        
        Args:
            query: Search keywords
            date_from: Optional lower bound (YYYY-MM or YYYY-MM-DD)
            date_to: Optional upper bound (YYYY-MM or YYYY-MM-DD)
            
        Returns:
            SearchResult with deduplicated papers
        """
        logger.info(f"Starting aggregated search for: {query}")
        sub_queries = _split_or_query(query)
        if len(sub_queries) > 1:
            logger.info(f"OR query detected: {len(sub_queries)} sub-queries → {sub_queries}")
        if date_from or date_to:
            logger.info(f"Date range filter: {date_from or '*'} → {date_to or '*'}")
        
        # Run all searches in parallel with exception handling
        results = await asyncio.gather(
            self.arxiv.search(query, date_from=date_from, date_to=date_to),
            self.semantic_scholar.search(query, date_from=date_from, date_to=date_to),
            self.huggingface.search(query, date_from=date_from, date_to=date_to),
            return_exceptions=True
        )
        
        all_papers = []
        sources_queried = []
        errors = []
        
        source_names = ["ArXiv", "Semantic Scholar", "Hugging Face"]
        
        for i, result in enumerate(results):
            source_name = source_names[i]
            
            if isinstance(result, Exception):
                error_msg = f"{source_name}: {str(result)}"
                errors.append(error_msg)
                logger.error(error_msg)
            else:
                all_papers.extend(result)
                sources_queried.append(source_name)
        
        # Deduplicate papers
        deduplicated = self._deduplicate_papers(all_papers)
        
        logger.info(
            f"Search complete: {len(deduplicated)} unique papers "
            f"from {len(sources_queried)} sources"
        )
        
        return SearchResult(
            papers=deduplicated,
            total_found=len(deduplicated),
            sources_queried=sources_queried,
            errors=errors
        )
    
    def _deduplicate_papers(self, papers: list[Paper]) -> list[Paper]:
        """
        Deduplicate papers based on normalized titles.
        
        Prioritizes entries with ArXiv PDF links.
        
        Args:
            papers: List of papers to deduplicate
            
        Returns:
            Deduplicated list of papers
        """
        seen = {}
        
        for paper in papers:
            normalized_title = self._normalize_title(paper.title)
            
            if normalized_title not in seen:
                seen[normalized_title] = paper
            else:
                existing = seen[normalized_title]
                # Prefer ArXiv source for reliable PDF links
                if paper.source == SourceType.ARXIV and existing.source != SourceType.ARXIV:
                    seen[normalized_title] = paper
                # Merge GitHub URL if found
                elif paper.github_url and not existing.github_url:
                    existing.github_url = paper.github_url
                # Merge abstract if missing
                elif paper.abstract and not existing.abstract:
                    existing.abstract = paper.abstract
        
        return list(seen.values())
    
    def _normalize_title(self, title: str) -> str:
        """
        Normalize a paper title for deduplication.
        
        - Convert to lowercase
        - Remove punctuation
        - Normalize unicode characters
        
        Args:
            title: Original title
            
        Returns:
            Normalized title string
        """
        # Normalize unicode
        title = unicodedata.normalize("NFKD", title)
        # Lowercase
        title = title.lower()
        # Remove punctuation
        title = re.sub(r'[^\w\s]', '', title)
        # Collapse whitespace
        title = ' '.join(title.split())
        return title


async def main():
    """Example usage of PaperAggregator."""
    aggregator = PaperAggregator(max_results_per_source=10)
    result = await aggregator.search("LLM security prompt injection")
    
    print(f"\nFound {result.total_found} papers")
    print(f"Sources: {', '.join(result.sources_queried)}")
    
    if result.errors:
        print(f"Errors: {', '.join(result.errors)}")
    
    for paper in result.papers[:5]:
        print(f"\n- {paper.title}")
        print(f"  Source: {paper.source}")
        print(f"  PDF: {paper.pdf_url}")
        if paper.github_url:
            print(f"  GitHub: {paper.github_url}")


if __name__ == "__main__":
    asyncio.run(main())
