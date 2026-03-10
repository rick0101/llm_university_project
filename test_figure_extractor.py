"""
Test suite per il modulo FigureExtractor.

Testa estrazione immagini da PDF, descrizione multimodale con Gemini Vision,
pipeline end-to-end di generazione chunk per figure, e integrazione
nella pipeline di ingestion.

Tutti i test usano mock — non richiedono API keys o PDF reali.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

from paperhunter.models import ChunkType, Paper, PaperChunk


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_settings():
    """Mock delle settings con configurazione figure_extraction."""
    mock = MagicMock()
    mock.figure_extraction.enabled = True
    mock.figure_extraction.min_width = 150
    mock.figure_extraction.min_height = 150
    mock.figure_extraction.max_figures_per_paper = 10
    mock.gemini.flash_model = "gemini-3-flash-preview"
    mock.google_api_key = "fake-key"
    return mock


@pytest.fixture
def sample_paper():
    """Paper di esempio per i test."""
    return Paper(
        title="Adversarial Attacks on Large Language Models",
        authors=["Alice Smith", "Bob Jones"],
        abstract="We study adversarial attacks...",
        url="https://arxiv.org/abs/2024.12345",
        bibtex_key="Smith_2024_Adversarial",
        category="Adversarial Attacks",
        year=2024,
        month=6,
        arxiv_id="2024.12345",
    )


@pytest.fixture
def sample_figure_captions():
    """Caption GROBID di esempio."""
    return [
        "Figure 1: Attack success rate across different models.",
        "Figure 2: Overview of the proposed defense architecture.",
        "Figure 3: Comparison of adversarial robustness scores.",
    ]


def _make_fake_png_bytes(width=200, height=200) -> bytes:
    """Genera un PNG valido per i test usando PIL."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================================
# Test: extract_images_from_pdf
# ============================================================================

class TestExtractImagesFromPdf:
    """Test per l'estrazione delle immagini da PDF con PyMuPDF."""

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_extract_images_filters_small_images(self, mock_settings_mod):
        """Immagini troppo piccole vengono filtrate."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 150
        mock_settings_mod.figure_extraction.min_height = 150
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=150, min_height=150)

        # Mock del PDF con immagini di diverse dimensioni
        mock_page = MagicMock()
        mock_page.get_images.return_value = [
            (1, 0, 100, 50, 8, "DeviceRGB", "", "Im1", "FlateDecode"),  # Troppo piccola
            (2, 0, 300, 200, 8, "DeviceRGB", "", "Im2", "FlateDecode"),  # Abbastanza grande
        ]

        # Mock dei Pixmap
        small_pix = MagicMock()
        small_pix.width = 100
        small_pix.height = 50
        small_pix.n = 3
        small_pix.tobytes.return_value = b"small_png"

        big_pix = MagicMock()
        big_pix.width = 300
        big_pix.height = 200
        big_pix.n = 3
        big_pix.tobytes.return_value = _make_fake_png_bytes(300, 200)

        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)

        with patch("paperhunter.ingestion.figure_extractor.fitz") as mock_fitz, \
             patch("pathlib.Path.exists", return_value=True):
            mock_fitz.open.return_value = mock_doc

            # Pixmap per xref 1 (piccola) e xref 2 (grande)
            def pixmap_side_effect(doc, xref):
                if xref == 1:
                    return small_pix
                return big_pix

            mock_fitz.Pixmap.side_effect = pixmap_side_effect

            images = extractor.extract_images_from_pdf("/fake/paper.pdf")

        # Solo l'immagine grande passa il filtro
        assert len(images) == 1
        assert images[0]["width"] == 300
        assert images[0]["height"] == 200

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_extract_images_max_limit(self, mock_settings_mod):
        """Rispetta il limite massimo di figure per paper."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 2

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50, max_figures=2)

        # 5 immagini nel PDF, ma max_figures=2
        images_info = [
            (i, 0, 300, 200, 8, "DeviceRGB", "", f"Im{i}", "FlateDecode")
            for i in range(5)
        ]

        mock_page = MagicMock()
        mock_page.get_images.return_value = images_info

        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)

        pix = MagicMock()
        pix.width = 300
        pix.height = 200
        pix.n = 3

        # Ogni immagine ha bytes diversi per evitare deduplica
        call_count = [0]
        def tobytes_side_effect(fmt):
            call_count[0] += 1
            return _make_fake_png_bytes(300, 200) + bytes([call_count[0]])
        pix.tobytes.side_effect = tobytes_side_effect

        with patch("paperhunter.ingestion.figure_extractor.fitz") as mock_fitz, \
             patch("pathlib.Path.exists", return_value=True):
            mock_fitz.open.return_value = mock_doc
            mock_fitz.Pixmap.return_value = pix

            images = extractor.extract_images_from_pdf("/fake/paper.pdf")

        assert len(images) <= 2

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_extract_images_deduplicates(self, mock_settings_mod):
        """Immagini identiche (stesso hash) vengono deduplicate."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        # 3 immagini identiche
        images_info = [
            (i, 0, 300, 200, 8, "DeviceRGB", "", f"Im{i}", "FlateDecode")
            for i in range(3)
        ]

        mock_page = MagicMock()
        mock_page.get_images.return_value = images_info

        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)

        pix = MagicMock()
        pix.width = 300
        pix.height = 200
        pix.n = 3
        # Stessi bytes → stesso hash → deduplico
        pix.tobytes.return_value = _make_fake_png_bytes(300, 200)

        with patch("paperhunter.ingestion.figure_extractor.fitz") as mock_fitz, \
             patch("pathlib.Path.exists", return_value=True):
            mock_fitz.open.return_value = mock_doc
            mock_fitz.Pixmap.return_value = pix

            images = extractor.extract_images_from_pdf("/fake/paper.pdf")

        assert len(images) == 1  # Solo 1 perché sono tutte identiche

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_extract_images_file_not_found(self, mock_settings_mod):
        """FileNotFoundError se il PDF non esiste."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 150
        mock_settings_mod.figure_extraction.min_height = 150
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract_images_from_pdf("/non/existent/paper.pdf")


# ============================================================================
# Test: describe_figure
# ============================================================================

class TestDescribeFigure:
    """Test per la descrizione multimodale delle figure."""

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_describe_figure_returns_description(self, mock_settings_mod):
        """Genera una descrizione strutturata per una figura."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.vision_model = "gemini-2.5-flash"
        mock_settings_mod.figure_extraction.min_width = 150
        mock_settings_mod.figure_extraction.min_height = 150
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.gemini.flash_model = "gemini-3-flash-preview"
        mock_settings_mod.google_api_key = "fake-key"

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor()

        mock_response = MagicMock()
        mock_response.text = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: The plot shows attack success rate.\n\n"
            "**Key Findings**: - ASR reaches 97% on GPT-4.\n\n"
            "**Relevance**: Demonstrates vulnerability of frontier models."
        )

        with patch("paperhunter.ingestion.figure_extractor.genai") as mock_genai, \
             patch("paperhunter.ingestion.figure_extractor.get_gemini_limiter") as mock_limiter, \
             patch("paperhunter.ingestion.figure_extractor.types") as mock_types:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            mock_limiter.return_value.wait.return_value = None

            desc = extractor.describe_figure(
                image_bytes=_make_fake_png_bytes(),
                caption="Attack success rate comparison",
            )

        assert "results_plot" in desc
        assert "attack success rate" in desc.lower()
        assert len(desc) > 50

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_describe_figure_handles_api_error(self, mock_settings_mod):
        """Ritorna stringa vuota in caso di errore API."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.vision_model = "gemini-2.5-flash"
        mock_settings_mod.figure_extraction.min_width = 150
        mock_settings_mod.figure_extraction.min_height = 150
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.gemini.flash_model = "gemini-3-flash-preview"
        mock_settings_mod.google_api_key = "fake-key"

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor()

        with patch("paperhunter.ingestion.figure_extractor.genai") as mock_genai, \
             patch("paperhunter.ingestion.figure_extractor.get_gemini_limiter") as mock_limiter, \
             patch("paperhunter.ingestion.figure_extractor.types") as mock_types:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.side_effect = Exception("API error")
            mock_limiter.return_value.wait.return_value = None

            desc = extractor.describe_figure(image_bytes=_make_fake_png_bytes())

        assert desc == ""

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_describe_figure_without_caption(self, mock_settings_mod):
        """Funziona anche senza caption GROBID."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.vision_model = "gemini-2.5-flash"
        mock_settings_mod.figure_extraction.min_width = 150
        mock_settings_mod.figure_extraction.min_height = 150
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.gemini.flash_model = "gemini-3-flash-preview"
        mock_settings_mod.google_api_key = "fake-key"

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor()

        mock_response = MagicMock()
        mock_response.text = "**Figure Type**: architecture_diagram\n\n**Content Description**: A flowchart showing the attack pipeline."

        with patch("paperhunter.ingestion.figure_extractor.genai") as mock_genai, \
             patch("paperhunter.ingestion.figure_extractor.get_gemini_limiter") as mock_limiter, \
             patch("paperhunter.ingestion.figure_extractor.types") as mock_types:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            mock_limiter.return_value.wait.return_value = None

            desc = extractor.describe_figure(
                image_bytes=_make_fake_png_bytes(),
                caption=None,  # No caption
            )

        assert "architecture_diagram" in desc


# ============================================================================
# Test: extract_and_describe (pipeline end-to-end)
# ============================================================================

class TestExtractAndDescribe:
    """Test per la pipeline completa estrazione + descrizione."""

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_full_pipeline_generates_chunks(self, mock_settings_mod, sample_paper, sample_figure_captions, tmp_path):
        """La pipeline completa genera PaperChunk con descrizioni e image_path."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.gemini.flash_model = "gemini-3-flash-preview"
        mock_settings_mod.google_api_key = "fake-key"

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        # Mock estrazione immagini
        fake_images = [
            {
                "image_bytes": _make_fake_png_bytes(300, 200) + bytes([i]),
                "page_number": i + 1,
                "width": 300,
                "height": 200,
                "image_index": i,
                "mime_type": "image/png",
                "image_hash": f"hash_{i}",
            }
            for i in range(2)
        ]

        mock_description = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: Shows attack success rate across models.\n\n"
            "**Key Findings**: - GCG achieves 97% ASR on GPT-4.\n\n"
            "**Relevance**: Critical for understanding model vulnerabilities."
        )

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=mock_description):

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
                figure_captions=sample_figure_captions,
            )

        assert len(chunks) == 2
        for chunk in chunks:
            assert isinstance(chunk, PaperChunk)
            assert chunk.section_name == "Figure_Description"
            assert chunk.bibtex_key == "Smith_2024_Adversarial"
            assert chunk.category == "Adversarial Attacks"
            assert "results_plot" in chunk.text
            assert chunk.chunk_type == ChunkType.ORIGINAL
            # Verifica che image_path sia impostato
            assert chunk.image_path is not None
            assert chunk.image_path.startswith("figures/")
            assert chunk.image_path.endswith(".png")

        # Verifica che i file immagine siano stati salvati su disco
        figures_dir = tmp_path / "figures"
        assert figures_dir.exists()
        saved_files = list(figures_dir.glob("*.png"))
        assert len(saved_files) == 2

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_pipeline_skips_short_descriptions(self, mock_settings_mod, sample_paper, tmp_path):
        """Descrizioni troppo corte vengono saltate."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.gemini.flash_model = "gemini-3-flash-preview"
        mock_settings_mod.google_api_key = "fake-key"

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        fake_images = [
            {
                "image_bytes": _make_fake_png_bytes(),
                "page_number": 1,
                "width": 300,
                "height": 200,
                "image_index": 0,
                "mime_type": "image/png",
                "image_hash": "hash_0",
            }
        ]

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value="Too short"):

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        assert len(chunks) == 0  # Descrizione troppo corta, saltata

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_pipeline_no_figures_returns_empty(self, mock_settings_mod, sample_paper):
        """Nessuna figura nel PDF → lista vuota."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 150
        mock_settings_mod.figure_extraction.min_height = 150
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor()

        with patch.object(extractor, "extract_images_from_pdf", return_value=[]):
            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        assert chunks == []

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_pipeline_includes_caption_in_text(self, mock_settings_mod, sample_paper, sample_figure_captions, tmp_path):
        """La caption GROBID viene inclusa nel testo del chunk."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.gemini.flash_model = "gemini-3-flash-preview"
        mock_settings_mod.google_api_key = "fake-key"

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        fake_images = [
            {
                "image_bytes": _make_fake_png_bytes(),
                "page_number": 3,
                "width": 300,
                "height": 200,
                "image_index": 0,
                "mime_type": "image/png",
                "image_hash": "hash_0",
            }
        ]

        description = "**Figure Type**: results_plot\n\n**Content Description**: A bar chart showing attack rates across five LLMs."

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=description):

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
                figure_captions=sample_figure_captions,
            )

        assert len(chunks) == 1
        text = chunks[0].text
        assert "Figure 1" in text  # [Figure 1, page 3]
        assert "page 3" in text
        assert "Attack success rate" in text  # Caption content
        assert "bar chart" in text  # Description content


# ============================================================================
# Test: _build_enriched_text
# ============================================================================

class TestBuildEnrichedText:
    """Test per la costruzione del testo arricchito."""

    def test_with_caption(self):
        from paperhunter.ingestion.figure_extractor import FigureExtractor

        text = FigureExtractor._build_enriched_text(
            description="A bar chart showing attack rates.",
            caption="Attack success rates comparison",
            page_number=5,
            figure_index=2,
        )

        assert "[Figure 2, page 5]" in text
        assert "Caption: Attack success rates comparison" in text
        assert "Description: A bar chart showing attack rates." in text

    def test_without_caption(self):
        from paperhunter.ingestion.figure_extractor import FigureExtractor

        text = FigureExtractor._build_enriched_text(
            description="A flowchart of the defense pipeline.",
            caption=None,
            page_number=1,
            figure_index=1,
        )

        assert "[Figure 1, page 1]" in text
        assert "Caption" not in text
        assert "Description: A flowchart" in text


# ============================================================================
# Test: integrazione con pipeline di ingestion
# ============================================================================

class TestIngestionPipelineIntegration:
    """Test per l'integrazione nella pipeline di ingestion."""

    @patch("paperhunter.ingestion.ingestion_pipeline.settings")
    def test_figure_extraction_disabled_returns_empty(self, mock_settings_mod, sample_paper):
        """Se figure_extraction.enabled=False, restituisce lista vuota."""
        mock_settings_mod.figure_extraction.enabled = False

        from paperhunter.ingestion.ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline._figure_extractor = None

        from paperhunter.models import GROBIDOutput

        grobid_output = GROBIDOutput(
            title="Test", abstract="Test abstract", figure_captions=["Fig 1"]
        )

        result = pipeline._step_figure_extraction(
            pdf_path=Path("/fake/path.pdf"),
            paper=sample_paper,
            grobid_output=grobid_output,
        )

        assert result == []

    @patch("paperhunter.ingestion.ingestion_pipeline.settings")
    def test_figure_extraction_handles_errors_gracefully(self, mock_settings_mod, sample_paper):
        """Errori nell'estrazione figure non bloccano la pipeline."""
        mock_settings_mod.figure_extraction.enabled = True

        from paperhunter.ingestion.ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline.__new__(IngestionPipeline)

        mock_extractor = MagicMock()
        mock_extractor.extract_and_describe.side_effect = RuntimeError("Image extraction failed")
        pipeline._figure_extractor = mock_extractor

        from paperhunter.models import GROBIDOutput

        grobid_output = GROBIDOutput(title="Test", abstract="Test abstract")

        result = pipeline._step_figure_extraction(
            pdf_path=Path("/fake/path.pdf"),
            paper=sample_paper,
            grobid_output=grobid_output,
        )

        assert result == []  # Errore gestito gracefully, pipeline non bloccata
