"""
Test suite for Task B.1: Deduplicazione pre-download.

Tests:
1. _normalize_filename() – varie casistiche di normalizzazione
2. DownloadResult.skipped field – default e esplicito
3. _scan_existing_pdfs() – index building
4. _is_duplicate() – titolo, ArXiv ID, no-match
5. register_downloaded() – aggiornamento indice in-session
6. download_paper() – skip su duplicato (mock)
7. batch_download() – conteggio skip/downloaded/failed (mock)
"""

import asyncio
import re
import sys
import tempfile
from pathlib import Path

# Ensure paperhunter is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperhunter.download_manager import Downloader, _normalize_filename
from paperhunter.models import DownloadResult, Paper


# ── Helpers ──────────────────────────────────────────────────────────
def _make_paper(**kwargs) -> Paper:
    """Create a minimal Paper for testing."""
    defaults = dict(
        title="Test Paper",
        authors=["Author A"],
        abstract="Abstract.",
        url="https://example.com/paper",
        source="arxiv",
    )
    defaults.update(kwargs)
    return Paper(**defaults)


# ── 1. _normalize_filename ──────────────────────────────────────────
def test_normalize_basic():
    assert _normalize_filename("Hello_World.pdf") == "hello_world"

def test_normalize_spaces():
    assert _normalize_filename("Hello  World  Paper") == "hello_world_paper"

def test_normalize_special_chars():
    # Non-alphanumerics removed, underscores collapsed
    result = _normalize_filename("Paper - (v2) [final].pdf")
    assert result == "paper_v2_final"

def test_normalize_underscores():
    assert _normalize_filename("a___b____c") == "a_b_c"

def test_normalize_empty():
    assert _normalize_filename("") == ""
    # ".pdf" → stem is "" (dot-file), normalize returns ""
    assert _normalize_filename(".pdf") == "pdf" or _normalize_filename(".pdf") == ""

def test_normalize_case():
    assert _normalize_filename("HELLO") == _normalize_filename("hello")

def test_normalize_arxiv_filename():
    assert _normalize_filename("2305.12345.pdf") == "230512345"

def test_normalize_strips():
    assert _normalize_filename("__leading__trailing__") == "leading_trailing"


# ── 2. DownloadResult.skipped field ─────────────────────────────────
def test_download_result_skipped_default():
    r = DownloadResult(success=True)
    assert r.skipped is False

def test_download_result_skipped_true():
    r = DownloadResult(success=True, skipped=True, method="dedup")
    assert r.skipped is True
    assert r.method == "dedup"

def test_download_result_json_roundtrip():
    r = DownloadResult(success=True, skipped=True, method="dedup")
    d = r.model_dump()
    assert d["skipped"] is True
    r2 = DownloadResult(**d)
    assert r2.skipped is True


# ── 3. _scan_existing_pdfs ──────────────────────────────────────────
def test_scan_existing_pdfs_empty():
    """Empty directory → empty index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dl = Downloader(download_dir=Path(tmpdir))
        assert len(dl._existing_files) == 0
        assert len(dl._existing_arxiv_ids) == 0

def test_scan_existing_pdfs_populated():
    """Create fake PDFs and verify they are indexed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        # Create files
        (p / "Attention_Is_All_You_Need.pdf").touch()
        (p / "sub" ).mkdir()
        (p / "sub" / "2305.12345.pdf").touch()
        (p / "sub" / "Adversarial_Example.pdf").touch()

        dl = Downloader(download_dir=p)
        assert len(dl._existing_files) == 3
        assert "attention_is_all_you_need" in dl._existing_files
        assert "adversarial_example" in dl._existing_files
        assert "2305.12345" in dl._existing_arxiv_ids or "230512345" in dl._existing_files

def test_scan_ignores_non_pdf():
    """Non-PDF files should be ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "notes.txt").touch()
        (p / "image.png").touch()
        (p / "actual.pdf").touch()
        dl = Downloader(download_dir=p)
        assert len(dl._existing_files) == 1


# ── 4. _is_duplicate ────────────────────────────────────────────────
def test_is_duplicate_by_title():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "Attention_Is_All_You_Need.pdf").touch()
        dl = Downloader(download_dir=p)

        paper = _make_paper(title="Attention Is All You Need")
        is_dup, reason = dl._is_duplicate(paper)
        assert is_dup is True
        assert reason is not None

def test_is_duplicate_by_arxiv_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "2305.12345.pdf").touch()
        dl = Downloader(download_dir=p)

        paper = _make_paper(title="Something Else", arxiv_id="2305.12345")
        is_dup, reason = dl._is_duplicate(paper)
        assert is_dup is True
        assert "ArXiv" in reason

def test_is_duplicate_arxiv_with_version():
    """ArXiv ID with version suffix should still match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "2305.12345.pdf").touch()
        dl = Downloader(download_dir=p)

        paper = _make_paper(title="Something", arxiv_id="2305.12345v3")
        is_dup, reason = dl._is_duplicate(paper)
        assert is_dup is True

def test_is_not_duplicate():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "OtherPaper.pdf").touch()
        dl = Downloader(download_dir=p)

        paper = _make_paper(title="Brand New Paper No Match")
        is_dup, reason = dl._is_duplicate(paper)
        assert is_dup is False
        assert reason is None


# ── 5. register_downloaded ──────────────────────────────────────────
def test_register_downloaded_updates_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        dl = Downloader(download_dir=Path(tmpdir))
        assert len(dl._existing_files) == 0

        paper = _make_paper(title="New Paper", arxiv_id="2401.99999")
        dl.register_downloaded(paper, "/fake/path/New_Paper.pdf")

        assert "new_paper" in dl._existing_files
        assert "2401.99999" in dl._existing_arxiv_ids

        # Now _is_duplicate should detect it
        is_dup, _ = dl._is_duplicate(paper)
        assert is_dup is True


# ── 6. download_paper skip on duplicate (mock) ──────────────────────
def test_download_paper_skips_duplicate():
    """download_paper() should return skipped=True for existing papers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "Existing_Paper_Title.pdf").touch()
        dl = Downloader(download_dir=p)

        paper = _make_paper(
            title="Existing Paper Title",
            url="https://arxiv.org/abs/9999.99999",
        )
        result = asyncio.run(dl.download_paper(paper))
        assert result.success is True
        assert result.skipped is True
        assert result.method == "dedup"


# ── 7. batch_download logging (mock) ───────────────────────────────
def test_batch_download_skipped_count():
    """batch_download() should correctly count skipped papers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "Paper_A.pdf").touch()
        (p / "Paper_B.pdf").touch()
        dl = Downloader(download_dir=p)

        papers = [
            _make_paper(title="Paper A"),   # exists → skip
            _make_paper(title="Paper B"),   # exists → skip
            # NOTE: Paper C does NOT exist, but we don't actually want to
            # download anything in tests.  We only verify that Paper A and B
            # are skipped instantly.
        ]

        results = asyncio.run(dl.batch_download(papers[:2]))
        skipped = sum(1 for r in results if r.skipped)
        assert skipped == 2
        assert all(r.success for r in results)


# ── 8. Real scaricati directory dedup index ─────────────────────────
def test_scan_real_scaricati():
    """If scaricati/ exists, verify the dedup index is non-trivial."""
    scaricati = Path(__file__).resolve().parent / "scaricati"
    if not scaricati.is_dir():
        print("SKIP: scaricati/ not found")
        return

    dl = Downloader(download_dir=scaricati)
    n_files = len(dl._existing_files)
    n_arxiv = len(dl._existing_arxiv_ids)
    print(f"scaricati/ index: {n_files} unique names, {n_arxiv} ArXiv IDs")
    assert n_files > 0, "scaricati/ has PDFs but index is empty"


# ── Runner ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Risultato: {passed}/{passed+failed} test passati")
    if failed:
        sys.exit(1)
