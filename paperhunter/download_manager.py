"""
Hybrid Download Manager for PaperHunter.

Implements a two-tier download strategy:
1. Fast: Direct HTTP download using httpx
2. Robust: Playwright browser automation for protected content
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

import httpx

from .config import ensure_download_dir, settings
from .models import DownloadResult, Paper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Downloader:
    """
    Hybrid PDF downloader with fallback to browser automation.
    
    Attempts direct download first, then falls back to Playwright
    for protected or JavaScript-rendered content.
    """
    
    # Common PDF download button patterns
    PDF_BUTTON_PATTERNS = [
        "Download PDF",
        "PDF",
        "Download",
        "Get PDF",
        "Full Text PDF",
        "View PDF",
    ]
    
    # User agent for direct downloads
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    
    def __init__(
        self,
        download_dir: Path = None,
        timeout: int = None,
        headless: bool = None
    ):
        """
        Initialize the downloader.
        
        Args:
            download_dir: Directory to save downloaded PDFs
            timeout: Download timeout in seconds
            headless: Whether to run Playwright in headless mode
        """
        self.download_dir = download_dir or ensure_download_dir()
        self.timeout = timeout or settings.download_timeout
        self.headless = headless if headless is not None else settings.headless
    
    def _sanitize_filename(self, title: str) -> str:
        """
        Sanitize a paper title to create a valid filename.
        
        Args:
            title: Paper title
            
        Returns:
            Sanitized filename (without extension)
        """
        # Remove/replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', title)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Limit length
        filename = filename[:100]
        # Remove leading/trailing underscores
        filename = filename.strip('_')
        return filename or "paper"
    
    async def download_paper(
        self,
        paper: Paper,
        custom_filename: str = None
    ) -> DownloadResult:
        """
        Download a paper PDF using hybrid strategy.
        
        Args:
            paper: Paper object with URL and title
            custom_filename: Optional custom filename (without extension)
            
        Returns:
            DownloadResult indicating success/failure
        """
        title = custom_filename or self._sanitize_filename(paper.title)
        
        logger.info(f"Attempting to download: {paper.title[:50]}...")
        
        # Build list of URLs to try, prioritizing ArXiv
        urls_to_try = []
        
        # If we have an ArXiv ID, try ArXiv PDF first (most reliable)
        if paper.arxiv_id:
            arxiv_pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
            urls_to_try.append(arxiv_pdf_url)
        
        # Then try the explicit PDF URL if available
        if paper.pdf_url and paper.pdf_url not in urls_to_try:
            urls_to_try.append(paper.pdf_url)
        
        # Finally try the paper URL
        if paper.url and paper.url not in urls_to_try:
            urls_to_try.append(paper.url)
        
        # Attempt direct download on each URL
        for url in urls_to_try:
            result = await self._direct_download(url, title)
            if result.success:
                return result
        
        logger.info(f"Direct downloads failed, trying Playwright...")
        
        # Attempt Playwright as last resort
        primary_url = urls_to_try[0] if urls_to_try else paper.url
        result = await self._playwright_download(primary_url, paper.url, title)
        return result
    
    async def _direct_download(self, url: str, filename: str) -> DownloadResult:
        """
        Attempt direct HTTP download of PDF.
        
        Args:
            url: URL to download from
            filename: Base filename (without extension)
            
        Returns:
            DownloadResult
        """
        try:
            headers = {
                "User-Agent": self.USER_AGENT,
                "Accept": "application/pdf,*/*",
            }
            
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True
            ) as client:
                response = await client.get(url, headers=headers)
                
                # Check for success
                if response.status_code != 200:
                    return DownloadResult(
                        success=False,
                        method="direct",
                        error_message=f"HTTP {response.status_code}"
                    )
                
                # Verify content type
                content_type = response.headers.get("content-type", "")
                if "application/pdf" not in content_type.lower():
                    # Check if content starts with PDF magic bytes
                    if not response.content[:4] == b'%PDF':
                        return DownloadResult(
                            success=False,
                            method="direct",
                            error_message=f"Not a PDF: {content_type}"
                        )
                
                # Save the file
                file_path = self.download_dir / f"{filename}.pdf"
                file_path.write_bytes(response.content)
                
                logger.info(f"Successfully downloaded: {file_path}")
                
                return DownloadResult(
                    success=True,
                    file_path=str(file_path),
                    method="direct"
                )
                
        except httpx.TimeoutException:
            return DownloadResult(
                success=False,
                method="direct",
                error_message="Download timed out"
            )
        except Exception as e:
            return DownloadResult(
                success=False,
                method="direct",
                error_message=str(e)
            )
    
    async def _playwright_download(
        self,
        pdf_url: str,
        page_url: str,
        filename: str
    ) -> DownloadResult:
        """
        Use Playwright browser automation to download PDF.
        
        Args:
            pdf_url: Direct PDF URL to try first
            page_url: Paper page URL to find download link
            filename: Base filename (without extension)
            
        Returns:
            DownloadResult
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return DownloadResult(
                success=False,
                method="playwright",
                error_message="Playwright not installed. Run: pip install playwright && playwright install"
            )
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context(
                    user_agent=self.USER_AGENT,
                    accept_downloads=True
                )
                page = await context.new_page()
                
                # Set download path
                file_path = self.download_dir / f"{filename}.pdf"
                
                # First try the PDF URL directly
                if pdf_url:
                    try:
                        async with page.expect_download(timeout=self.timeout * 1000) as download_info:
                            await page.goto(pdf_url)
                        download = await download_info.value
                        await download.save_as(file_path)
                        await browser.close()
                        
                        logger.info(f"Successfully downloaded via Playwright: {file_path}")
                        return DownloadResult(
                            success=True,
                            file_path=str(file_path),
                            method="playwright"
                        )
                    except Exception:
                        pass  # Try finding download button
                
                # Navigate to paper page and find download button
                await page.goto(page_url, wait_until="networkidle")
                await asyncio.sleep(settings.playwright_delay)
                
                # Try to find PDF download link
                pdf_link = await self._find_pdf_link(page)
                
                if pdf_link:
                    try:
                        async with page.expect_download(timeout=self.timeout * 1000) as download_info:
                            await pdf_link.click()
                        download = await download_info.value
                        await download.save_as(file_path)
                        await browser.close()
                        
                        logger.info(f"Successfully downloaded via Playwright: {file_path}")
                        return DownloadResult(
                            success=True,
                            file_path=str(file_path),
                            method="playwright"
                        )
                    except Exception as e:
                        await browser.close()
                        return DownloadResult(
                            success=False,
                            method="playwright",
                            error_message=f"Click download failed: {e}"
                        )
                
                await browser.close()
                return DownloadResult(
                    success=False,
                    method="playwright",
                    error_message="Could not find PDF download link"
                )
                
        except Exception as e:
            return DownloadResult(
                success=False,
                method="playwright",
                error_message=str(e)
            )
    
    async def _find_pdf_link(self, page):
        """
        Find PDF download link on a page using heuristics.
        
        Args:
            page: Playwright page object
            
        Returns:
            Locator for PDF link or None
        """
        # Try finding links ending in .pdf
        pdf_links = page.locator('a[href$=".pdf"]')
        count = await pdf_links.count()
        if count > 0:
            return pdf_links.first
        
        # Try finding buttons/links with PDF-related text
        for pattern in self.PDF_BUTTON_PATTERNS:
            # Case-insensitive text match
            selector = f'a:has-text("{pattern}"), button:has-text("{pattern}")'
            elements = page.locator(selector)
            count = await elements.count()
            if count > 0:
                return elements.first
        
        # Try ArXiv-specific selector
        arxiv_pdf = page.locator('a.abs-button[href*="pdf"]')
        count = await arxiv_pdf.count()
        if count > 0:
            return arxiv_pdf.first
        
        return None
    
    async def batch_download(
        self,
        papers: list[Paper],
        max_concurrent: int = 3,
        delay_between: float = 2.0
    ) -> list[DownloadResult]:
        """
        Download multiple papers with rate limiting.
        
        Args:
            papers: List of papers to download
            max_concurrent: Maximum concurrent downloads
            delay_between: Delay between starting downloads
            
        Returns:
            List of DownloadResult objects
        """
        logger.info(f"Starting batch download of {len(papers)} papers...")
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(paper: Paper) -> DownloadResult:
            async with semaphore:
                result = await self.download_paper(paper)
                await asyncio.sleep(delay_between)
                return result
        
        tasks = [download_with_semaphore(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(DownloadResult(
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        successful = sum(1 for r in final_results if r.success)
        logger.info(f"Batch download complete: {successful}/{len(papers)} successful")
        
        return final_results


async def main():
    """Example usage of Downloader."""
    downloader = Downloader()
    
    # Test paper
    paper = Paper(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "et al."],
        abstract="The dominant sequence transduction models...",
        url="https://arxiv.org/abs/1706.03762",
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        source="arxiv"
    )
    
    result = await downloader.download_paper(paper)
    
    if result.success:
        print(f"Downloaded: {result.file_path}")
        print(f"Method: {result.method}")
    else:
        print(f"Failed: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
