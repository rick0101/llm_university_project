"""
PaperHunter - Main Controller

Orchestrates the paper hunting pipeline:
1. Collect broad search keywords from user
2. Collect filtering criteria from user
3. Aggregate papers from multiple sources
4. Filter papers using LLM
5. Download relevant PDFs
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import ensure_download_dir, settings
from .download_manager import Downloader
from .models import DownloadResult, FilteredPaper, Paper, SearchQuery, SearchResult
from .relevance_filter import RelevanceFilter
from .report_generator import generate_report
from .search_engine import PaperAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperHunter:
    """
    Main orchestrator for the paper hunting pipeline.
    
    Coordinates searching, filtering, and downloading of scientific papers
    based on user-defined keywords and criteria.
    """
    
    def __init__(
        self,
        max_results_per_source: int = None,
        categories: list[str] = None,
        llm_provider: str = None,
        llm_model: str = None,
        download_dir: Path = None
    ):
        """
        Initialize PaperHunter.
        
        Args:
            max_results_per_source: Max papers to fetch per source
            categories: ArXiv categories to search
            llm_provider: LLM provider for filtering ("google" or "anthropic")
            llm_model: Specific LLM model to use
            download_dir: Directory for downloaded PDFs
        """
        self.max_results = max_results_per_source or settings.max_results_per_source
        self.categories = categories or settings.default_categories
        self.download_dir = download_dir or ensure_download_dir()
        
        # Initialize components
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        self.aggregator = PaperAggregator(
            max_results_per_source=self.max_results,
            categories=self.categories
        )
        self.filter = RelevanceFilter(
            provider=llm_provider,
            model=llm_model
        )
        self.downloader = Downloader(download_dir=self.download_dir)
    
    async def hunt(
        self,
        keywords: str,
        criteria: str,
        auto_download: bool = True,
        interactive: bool = False
    ) -> dict:
        """
        Execute the full paper hunting pipeline.
        
        Args:
            keywords: Broad search keywords
            criteria: Specific filtering criteria for LLM
            auto_download: Whether to automatically download relevant papers
            interactive: Whether to prompt for confirmation before downloading
            
        Returns:
            Dictionary with results summary
        """
        logger.info("=" * 60)
        logger.info("PaperHunter - Starting Hunt")
        logger.info("=" * 60)
        
        # Step 1: Search and aggregate papers
        logger.info(f"\n📚 Step 1: Searching for papers with keywords: '{keywords}'")
        search_result = await self.aggregator.search(keywords)
        
        logger.info(f"   Found {search_result.total_found} papers from {len(search_result.sources_queried)} sources")
        if search_result.errors:
            logger.warning(f"   Errors: {', '.join(search_result.errors)}")
        
        if not search_result.papers:
            logger.warning("No papers found. Try different keywords.")
            return {
                "status": "no_results",
                "papers_found": 0,
                "papers_relevant": 0,
                "papers_downloaded": 0
            }
        
        # Step 2: Filter papers using LLM
        logger.info(f"\n🔍 Step 2: Filtering papers with criteria...")
        logger.info(f"   Criteria: {criteria[:100]}...")
        
        filtered_results = await self.filter.filter_papers(
            search_result.papers,
            criteria
        )
        
        relevant_papers = self.filter.get_relevant_papers(filtered_results)
        logger.info(f"   Found {len(relevant_papers)} relevant papers")
        
        if not relevant_papers:
            logger.warning("No relevant papers found. Try adjusting your criteria.")
            return {
                "status": "no_relevant",
                "papers_found": search_result.total_found,
                "papers_relevant": 0,
                "papers_downloaded": 0
            }
        
        # Display relevant papers
        self._display_relevant_papers(filtered_results)
        
        # Step 3: Download relevant papers
        download_results = []
        if auto_download:
            if interactive:
                confirm = input("\n📥 Download these papers? (y/n): ").strip().lower()
                if confirm != 'y':
                    logger.info("Download cancelled by user.")
                    return {
                        "status": "cancelled",
                        "papers_found": search_result.total_found,
                        "papers_relevant": len(relevant_papers),
                        "papers_downloaded": 0,
                        "relevant_papers": relevant_papers
                    }
            
            logger.info(f"\n📥 Step 3: Downloading {len(relevant_papers)} papers...")
            download_results = await self.downloader.batch_download(relevant_papers)
            
            successful_downloads = sum(1 for r in download_results if r.success)
            logger.info(f"   Downloaded {successful_downloads}/{len(relevant_papers)} papers")
            
            # Show download results
            self._display_download_results(relevant_papers, download_results)
        
        # Step 4: Generate report
        logger.info(f"\n📝 Step 4: Generating session report...")
        report_path = generate_report(
            keywords=keywords,
            criteria=criteria,
            search_result=search_result,
            filtered_results=filtered_results,
            download_results=download_results,
            relevant_papers=relevant_papers,
            output_dir=self.download_dir,
            max_results_per_source=self.max_results,
            categories=self.categories,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
        )
        logger.info(f"   Report saved to: {report_path}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Hunt Complete!")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "papers_found": search_result.total_found,
            "papers_relevant": len(relevant_papers),
            "papers_downloaded": sum(1 for r in download_results if r.success),
            "search_result": search_result,
            "filtered_results": filtered_results,
            "download_results": download_results,
            "relevant_papers": relevant_papers,
            "report_path": str(report_path)
        }
    
    def _display_relevant_papers(self, filtered_results: list[FilteredPaper]) -> None:
        """Display relevant papers in a formatted way."""
        print("\n" + "=" * 60)
        print("📄 RELEVANT PAPERS")
        print("=" * 60)
        
        for i, filtered in enumerate(filtered_results, 1):
            if filtered.relevance.is_relevant:
                paper = filtered.paper
                print(f"\n{i}. {paper.title}")
                print(f"   Source: {paper.source}")
                if paper.authors:
                    authors = ", ".join(paper.authors[:3])
                    if len(paper.authors) > 3:
                        authors += " et al."
                    print(f"   Authors: {authors}")
                if paper.github_url:
                    print(f"   🔗 GitHub: {paper.github_url}")
                print(f"   📎 URL: {paper.url}")
                print(f"   💭 Reasoning: {filtered.relevance.reasoning[:150]}...")
    
    def _display_download_results(
        self,
        papers: list[Paper],
        results: list[DownloadResult]
    ) -> None:
        """Display download results."""
        print("\n" + "-" * 40)
        print("Download Results:")
        print("-" * 40)
        
        for paper, result in zip(papers, results):
            status = "✅" if result.success else "❌"
            print(f"{status} {paper.title[:50]}...")
            if result.success:
                print(f"   📂 {result.file_path}")
            else:
                print(f"   ⚠️ {result.error_message}")


def get_user_input() -> tuple[str, str]:
    """
    Interactive mode: Get search parameters from user.
    
    Returns:
        Tuple of (keywords, criteria)
    """
    print("\n" + "=" * 60)
    print("🔎 PaperHunter - Autonomous Scientific Paper Finder")
    print("=" * 60)
    
    print("\n📌 Step 1: Enter your BROAD SEARCH KEYWORDS")
    print("   (These will be used to query ArXiv, Semantic Scholar, etc.)")
    print("   Example: 'LLM security', 'prompt injection', 'AI safety'")
    keywords = input("\n   Keywords: ").strip()
    
    if not keywords:
        keywords = "LLM security"
        print(f"   Using default: '{keywords}'")
    
    print("\n📌 Step 2: Enter your FILTERING CRITERIA")
    print("   (The LLM will use this to judge relevance)")
    print("   Be specific about what you want AND what you don't want.")
    print("   Example: 'Papers about prompt injection and jailbreaking. Ignore GDPR papers.'")
    
    criteria = input("\n   Criteria: ").strip()
    
    if not criteria:
        criteria = "I want papers about LLM security, prompt injection, or jailbreaking attacks."
        print(f"   Using default criteria.")
    
    return keywords, criteria


async def main():
    """Main entry point for PaperHunter."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PaperHunter - Autonomous Scientific Paper Finder"
    )
    parser.add_argument(
        "-k", "--keywords",
        type=str,
        help="Broad search keywords"
    )
    parser.add_argument(
        "-c", "--criteria",
        type=str,
        help="Filtering criteria for LLM (optional - if omitted, uses keywords as generic criteria)"
    )
    parser.add_argument(
        "-n", "--max-results",
        type=int,
        default=40,
        help="Maximum results per source (default: 40)"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download papers, only search and filter"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode with confirmation prompts"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="Output directory for downloaded PDFs"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["google", "anthropic"],
        help="LLM provider (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Get keywords and criteria
    if args.keywords:
        keywords = args.keywords
        if args.criteria:
            criteria = args.criteria
        else:
            # Auto-generate generic criteria from keywords
            criteria = f"I want papers related to: {keywords}. Include all relevant research on this topic."
            logger.info(f"Using auto-generated criteria: {criteria}")
    else:
        keywords, criteria = get_user_input()
    
    # Initialize PaperHunter
    hunter = PaperHunter(
        max_results_per_source=args.max_results,
        llm_provider=args.llm_provider,
        download_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    # Run the hunt
    result = await hunter.hunt(
        keywords=keywords,
        criteria=criteria,
        auto_download=not args.no_download,
        interactive=args.interactive
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   Status: {result['status']}")
    print(f"   Papers Found: {result['papers_found']}")
    print(f"   Papers Relevant: {result['papers_relevant']}")
    print(f"   Papers Downloaded: {result['papers_downloaded']}")
    
    return result


def run():
    """Wrapper to run async main."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    run()
