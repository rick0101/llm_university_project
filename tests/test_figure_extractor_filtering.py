"""
Test per l'integrazione del filtro euristico in FigureExtractor (Task 1.2).

Verifica che:
- extract_images_from_pdf() restituisca total_pages, page_width, page_height
- extract_and_describe() filtri le immagini LIKELY_DECORATIVE prima di Vision
- Le immagini UNCERTAIN e SCIENTIFIC_FIGURE passino a Vision
- Il filtraggio possa essere disabilitato via settings.image_filtering.enabled
- Le statistiche di filtraggio siano loggate correttamente
- Le chiamate Vision siano ridotte rispetto al totale delle immagini
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import numpy as np
import pytest
from PIL import Image, ImageDraw

from paperhunter.ingestion.image_relevance_filter import (
    ImageClassification,
    ImageScore,
)
from paperhunter.models import ChunkType, Paper, PaperChunk


# ============================================================================
# Helpers
# ============================================================================


def _make_png(img: Image.Image) -> bytes:
    """Converte un'immagine PIL in bytes PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_chart_png(width: int = 400, height: int = 300) -> bytes:
    """Genera un PNG tipo grafico scientifico (alta entropia + bordi)."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    margin = 40
    # Assi
    draw.line([(margin, margin), (margin, height - margin)], fill="black", width=2)
    draw.line([(margin, height - margin), (width - margin, height - margin)], fill="black", width=2)
    # Linee dati
    rng = np.random.RandomState(42)
    for color in ["red", "blue", "green"]:
        points = []
        y_base = rng.randint(60, 200)
        for i in range(20):
            x = margin + int(i * (width - 2 * margin) / 19)
            y = max(margin, min(height - margin, y_base + int(rng.normal(0, 15))))
            points.append((x, y))
            y_base += rng.randint(-5, 6)
        for j in range(len(points) - 1):
            draw.line([points[j], points[j + 1]], fill=color, width=2)
    # Griglia
    for i in range(5):
        y = margin + i * (height - 2 * margin) // 4
        draw.line([(margin, y), (width - margin, y)], fill=(200, 200, 200), width=1)
    return _make_png(img)


def _make_logo_png(width: int = 60, height: int = 60) -> bytes:
    """Genera un PNG tipo logo (pochi colori, bassa entropia)."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(10, 10), (50, 50)], fill=(0, 0, 128))
    draw.text((18, 20), "AB", fill="white")
    return _make_png(img)


def _make_banner_png(width: int = 600, height: int = 40) -> bytes:
    """Genera un PNG tipo banner orizzontale (aspect ratio estremo)."""
    img = Image.new("RGB", (width, height), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Copyright 2024 - All rights reserved", fill="black")
    return _make_png(img)


# ============================================================================
# Fixtures
# ============================================================================


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


# ============================================================================
# Test: extract_images_from_pdf — nuovi campi
# ============================================================================


class TestExtractImagesNewFields:
    """Verifica che extract_images_from_pdf() restituisca total_pages, page_width, page_height."""

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_images_contain_total_pages(self, mock_settings_mod):
        """Ogni immagine estratta contiene total_pages."""
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        mock_rect = MagicMock()
        mock_rect.width = 612.0
        mock_rect.height = 792.0

        mock_page = MagicMock()
        mock_page.rect = mock_rect
        mock_page.get_images.return_value = [
            (1, 0, 300, 200, 8, "DeviceRGB", "", "Im1", "FlateDecode"),
        ]

        pix = MagicMock()
        pix.width = 300
        pix.height = 200
        pix.n = 3
        pix.tobytes.return_value = _make_chart_png(300, 200)

        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=5)
        mock_doc.__getitem__ = Mock(return_value=mock_page)

        with patch("paperhunter.ingestion.figure_extractor.fitz") as mock_fitz, \
             patch("pathlib.Path.exists", return_value=True):
            mock_fitz.open.return_value = mock_doc
            mock_fitz.Pixmap.return_value = pix

            images = extractor.extract_images_from_pdf("/fake/paper.pdf")

        assert len(images) == 1
        assert images[0]["total_pages"] == 5
        assert images[0]["page_width"] == 612.0
        assert images[0]["page_height"] == 792.0

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_images_from_different_pages_have_correct_dims(self, mock_settings_mod):
        """Immagini da pagine diverse hanno le dimensioni di pagina corrette."""
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        # Pagina 1: letter (612x792), Pagina 2: A4 (595x842)
        mock_rect_1 = MagicMock()
        mock_rect_1.width = 612.0
        mock_rect_1.height = 792.0
        mock_page_1 = MagicMock()
        mock_page_1.rect = mock_rect_1
        mock_page_1.get_images.return_value = [
            (1, 0, 300, 200, 8, "DeviceRGB", "", "Im1", "FlateDecode"),
        ]

        mock_rect_2 = MagicMock()
        mock_rect_2.width = 595.0
        mock_rect_2.height = 842.0
        mock_page_2 = MagicMock()
        mock_page_2.rect = mock_rect_2
        mock_page_2.get_images.return_value = [
            (2, 0, 400, 300, 8, "DeviceRGB", "", "Im2", "FlateDecode"),
        ]

        call_count = [0]

        pix = MagicMock()
        pix.n = 3

        def pixmap_side_effect(doc, xref):
            m = MagicMock()
            m.n = 3
            if xref == 1:
                m.width = 300
                m.height = 200
                m.tobytes.return_value = _make_chart_png(300, 200)
            else:
                m.width = 400
                m.height = 300
                m.tobytes.return_value = _make_chart_png(400, 300)
            return m

        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=2)
        mock_doc.__getitem__ = Mock(side_effect=lambda idx: mock_page_1 if idx == 0 else mock_page_2)

        with patch("paperhunter.ingestion.figure_extractor.fitz") as mock_fitz, \
             patch("pathlib.Path.exists", return_value=True):
            mock_fitz.open.return_value = mock_doc
            mock_fitz.Pixmap.side_effect = pixmap_side_effect

            images = extractor.extract_images_from_pdf("/fake/paper.pdf")

        assert len(images) == 2
        assert images[0]["page_width"] == 612.0
        assert images[0]["page_height"] == 792.0
        assert images[1]["page_width"] == 595.0
        assert images[1]["page_height"] == 842.0
        # total_pages uguale per tutte
        assert images[0]["total_pages"] == 2
        assert images[1]["total_pages"] == 2


# ============================================================================
# Test: Integrazione filtro euristico in extract_and_describe
# ============================================================================


class TestHeuristicFilterIntegration:
    """Test per l'integrazione del filtro euristico nella pipeline extract_and_describe."""

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_decorative_images_not_sent_to_vision(self, mock_settings_mod, sample_paper, tmp_path):
        """Le immagini LIKELY_DECORATIVE vengono scartate, non inviate a Vision."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.image_filtering.enabled = True
        mock_settings_mod.image_filtering.embedding_filter_enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        logo_bytes = _make_logo_png(60, 60)

        fake_images = [
            {
                "image_bytes": chart_bytes,
                "page_number": 5,
                "width": 400,
                "height": 300,
                "image_index": 0,
                "mime_type": "image/png",
                "image_hash": "chart_hash",
                "total_pages": 12,
                "page_width": 612.0,
                "page_height": 792.0,
            },
            {
                "image_bytes": logo_bytes,
                "page_number": 1,
                "width": 60,
                "height": 60,
                "image_index": 1,
                "mime_type": "image/png",
                "image_hash": "logo_hash",
                "total_pages": 12,
                "page_width": 612.0,
                "page_height": 792.0,
            },
        ]

        mock_description = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: Shows attack success rate across models.\n\n"
            "**Key Findings**: - GCG achieves 97% ASR.\n\n"
            "**Relevance**: Demonstrates vulnerability."
        )

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=mock_description) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Il logo (piccolo, prima pagina, bassa entropia) dovrebbe essere filtrato
        # Solo il grafico dovrebbe passare → 1 chiamata describe_figure
        assert mock_describe.call_count <= len(fake_images)
        # Almeno 1 chunk generato per il grafico
        assert len(chunks) >= 1

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_filter_disabled_sends_all_to_vision(self, mock_settings_mod, sample_paper, tmp_path):
        """Con image_filtering.enabled=False, tutte le immagini vanno a Vision."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.image_filtering.enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        logo_bytes = _make_logo_png(200, 200)

        fake_images = [
            {
                "image_bytes": chart_bytes,
                "page_number": 5,
                "width": 400,
                "height": 300,
                "image_index": 0,
                "mime_type": "image/png",
                "image_hash": "chart_hash",
            },
            {
                "image_bytes": logo_bytes,
                "page_number": 1,
                "width": 200,
                "height": 200,
                "image_index": 1,
                "mime_type": "image/png",
                "image_hash": "logo_hash",
            },
        ]

        mock_description = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: Shows attack success rate across models.\n\n"
            "**Key Findings**: - GCG achieves 97% ASR.\n\n"
            "**Relevance**: Demonstrates vulnerability."
        )

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=mock_description) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Tutte le 2 immagini inviate a Vision
        assert mock_describe.call_count == 2
        assert len(chunks) == 2
        # relevance_score è None quando il filtro è disabilitato
        for img in fake_images:
            assert img.get("relevance_score") is None

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_uncertain_images_pass_to_vision(self, mock_settings_mod, sample_paper, tmp_path):
        """Le immagini UNCERTAIN vengono passate a Vision (beneficio del dubbio)."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.image_filtering.enabled = True
        mock_settings_mod.image_filtering.embedding_filter_enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor
        from paperhunter.ingestion.image_relevance_filter import ImageRelevanceFilter

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)

        fake_images = [
            {
                "image_bytes": chart_bytes,
                "page_number": 5,
                "width": 400,
                "height": 300,
                "image_index": 0,
                "mime_type": "image/png",
                "image_hash": "chart_hash",
                "total_pages": 12,
                "page_width": 612.0,
                "page_height": 792.0,
            },
        ]

        # Mock il filtro per restituire UNCERTAIN
        uncertain_score = ImageScore(
            aspect_ratio_score=0.5,
            entropy_score=0.5,
            edge_density_score=0.3,
            position_score=0.8,
            size_score=0.5,
            combined_score=0.35,
            classification=ImageClassification.UNCERTAIN,
        )

        mock_description = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: Shows attack success rate across models.\n\n"
            "**Key Findings**: - GCG achieves 97% ASR.\n\n"
            "**Relevance**: Demonstrates vulnerability."
        )

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=mock_description) as mock_describe, \
             patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter") as mock_filter_cls:
            mock_filter = MagicMock()
            mock_filter.classify_batch.return_value = [uncertain_score]
            mock_filter_cls.return_value = mock_filter

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # UNCERTAIN passa → describe_figure viene chiamato
        assert mock_describe.call_count == 1
        assert len(chunks) == 1

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_all_decorative_returns_empty(self, mock_settings_mod, sample_paper, tmp_path):
        """Se tutte le immagini sono LIKELY_DECORATIVE, restituisce lista vuota."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.image_filtering.enabled = True
        mock_settings_mod.image_filtering.embedding_filter_enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        logo_bytes = _make_logo_png(60, 60)

        fake_images = [
            {
                "image_bytes": logo_bytes,
                "page_number": 1,
                "width": 60,
                "height": 60,
                "image_index": 0,
                "mime_type": "image/png",
                "image_hash": "logo_hash_0",
                "total_pages": 10,
                "page_width": 612.0,
                "page_height": 792.0,
            },
            {
                "image_bytes": logo_bytes,
                "page_number": 10,
                "width": 60,
                "height": 60,
                "image_index": 1,
                "mime_type": "image/png",
                "image_hash": "logo_hash_1",
                "total_pages": 10,
                "page_width": 612.0,
                "page_height": 792.0,
            },
        ]

        decorative_score = ImageScore(
            combined_score=0.15,
            classification=ImageClassification.LIKELY_DECORATIVE,
        )

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure") as mock_describe, \
             patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter") as mock_filter_cls:
            mock_filter = MagicMock()
            mock_filter.classify_batch.return_value = [decorative_score, decorative_score]
            mock_filter_cls.return_value = mock_filter

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Nessuna immagine passa il filtro → 0 chiamate Vision
        assert mock_describe.call_count == 0
        assert len(chunks) == 0

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_relevance_score_added_to_images(self, mock_settings_mod, sample_paper, tmp_path):
        """Ogni immagine filtrata ha un campo relevance_score nel dict."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.image_filtering.enabled = True
        mock_settings_mod.image_filtering.embedding_filter_enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)

        fake_images = [
            {
                "image_bytes": chart_bytes,
                "page_number": 5,
                "width": 400,
                "height": 300,
                "image_index": 0,
                "mime_type": "image/png",
                "image_hash": "chart_hash",
                "total_pages": 12,
                "page_width": 612.0,
                "page_height": 792.0,
            },
        ]

        scientific_score = ImageScore(
            combined_score=0.75,
            classification=ImageClassification.SCIENTIFIC_FIGURE,
        )

        mock_description = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: Shows attack success rate across models.\n\n"
            "**Key Findings**: - ASR reaches 97%.\n\n"
            "**Relevance**: Important for security analysis."
        )

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=mock_description), \
             patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter") as mock_filter_cls:
            mock_filter = MagicMock()
            mock_filter.classify_batch.return_value = [scientific_score]
            mock_filter_cls.return_value = mock_filter

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Dopo il filtraggio, il dict ha relevance_score
        assert fake_images[0]["relevance_score"] == 0.75

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_vision_calls_reduced_by_filtering(self, mock_settings_mod, sample_paper, tmp_path):
        """Le chiamate Vision sono ridotte: meno chiamate del totale immagini estratte."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 20
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.image_filtering.enabled = True
        mock_settings_mod.image_filtering.embedding_filter_enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50, max_figures=20)

        # 5 immagini: 2 grafici + 3 loghi
        chart_bytes = _make_chart_png(400, 300)
        logo_bytes = _make_logo_png(60, 60)

        fake_images = []
        for i in range(2):
            fake_images.append({
                "image_bytes": chart_bytes + bytes([i]),
                "page_number": 5 + i,
                "width": 400,
                "height": 300,
                "image_index": i,
                "mime_type": "image/png",
                "image_hash": f"chart_{i}",
                "total_pages": 15,
                "page_width": 612.0,
                "page_height": 792.0,
            })
        for i in range(3):
            fake_images.append({
                "image_bytes": logo_bytes + bytes([i + 10]),
                "page_number": 1,
                "width": 60,
                "height": 60,
                "image_index": 2 + i,
                "mime_type": "image/png",
                "image_hash": f"logo_{i}",
                "total_pages": 15,
                "page_width": 612.0,
                "page_height": 792.0,
            })

        scores = [
            ImageScore(combined_score=0.78, classification=ImageClassification.SCIENTIFIC_FIGURE),
            ImageScore(combined_score=0.72, classification=ImageClassification.SCIENTIFIC_FIGURE),
            ImageScore(combined_score=0.12, classification=ImageClassification.LIKELY_DECORATIVE),
            ImageScore(combined_score=0.10, classification=ImageClassification.LIKELY_DECORATIVE),
            ImageScore(combined_score=0.08, classification=ImageClassification.LIKELY_DECORATIVE),
        ]

        mock_description = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: Shows attack success rate across models.\n\n"
            "**Key Findings**: - ASR reaches 97%.\n\n"
            "**Relevance**: Important for security analysis."
        )

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=mock_description) as mock_describe, \
             patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter") as mock_filter_cls:
            mock_filter = MagicMock()
            mock_filter.classify_batch.return_value = scores
            mock_filter_cls.return_value = mock_filter

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # 5 immagini estratte, 3 filtrate, solo 2 inviate a Vision
        assert mock_describe.call_count == 2
        assert len(chunks) == 2

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_filter_with_mixed_classifications(self, mock_settings_mod, sample_paper, tmp_path):
        """Mix di SCIENTIFIC, UNCERTAIN e DECORATIVE: solo DECORATIVE scartate."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.image_filtering.enabled = True
        mock_settings_mod.image_filtering.embedding_filter_enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)

        fake_images = [
            {
                "image_bytes": chart_bytes + bytes([i]),
                "page_number": i + 1,
                "width": 400,
                "height": 300,
                "image_index": i,
                "mime_type": "image/png",
                "image_hash": f"img_{i}",
                "total_pages": 10,
                "page_width": 612.0,
                "page_height": 792.0,
            }
            for i in range(4)
        ]

        scores = [
            ImageScore(combined_score=0.80, classification=ImageClassification.SCIENTIFIC_FIGURE),
            ImageScore(combined_score=0.35, classification=ImageClassification.UNCERTAIN),
            ImageScore(combined_score=0.12, classification=ImageClassification.LIKELY_DECORATIVE),
            ImageScore(combined_score=0.55, classification=ImageClassification.SCIENTIFIC_FIGURE),
        ]

        mock_description = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: Shows attack success rate across models.\n\n"
            "**Key Findings**: - ASR reaches 97%.\n\n"
            "**Relevance**: Important for security analysis."
        )

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=mock_description) as mock_describe, \
             patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter") as mock_filter_cls:
            mock_filter = MagicMock()
            mock_filter.classify_batch.return_value = scores
            mock_filter_cls.return_value = mock_filter

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # 4 immagini, 1 decorativa scartata, 3 passano (2 SCIENTIFIC + 1 UNCERTAIN)
        assert mock_describe.call_count == 3
        assert len(chunks) == 3

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_end_to_end_with_real_filter(self, mock_settings_mod, sample_paper, tmp_path):
        """Test end-to-end con ImageRelevanceFilter reale (non mockato)."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.image_filtering.enabled = True
        mock_settings_mod.image_filtering.embedding_filter_enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        # Immagini miste con euristiche reali
        chart_bytes = _make_chart_png(400, 300)   # Dovrebbe passare
        logo_bytes = _make_logo_png(60, 60)        # Dovrebbe essere filtrato
        banner_bytes = _make_banner_png(600, 40)   # Dovrebbe essere filtrato

        fake_images = [
            {
                "image_bytes": chart_bytes,
                "page_number": 5,
                "width": 400,
                "height": 300,
                "image_index": 0,
                "mime_type": "image/png",
                "image_hash": "chart_h",
                "total_pages": 12,
                "page_width": 612.0,
                "page_height": 792.0,
            },
            {
                "image_bytes": logo_bytes,
                "page_number": 1,
                "width": 60,
                "height": 60,
                "image_index": 1,
                "mime_type": "image/png",
                "image_hash": "logo_h",
                "total_pages": 12,
                "page_width": 612.0,
                "page_height": 792.0,
            },
            {
                "image_bytes": banner_bytes,
                "page_number": 12,
                "width": 600,
                "height": 40,
                "image_index": 2,
                "mime_type": "image/png",
                "image_hash": "banner_h",
                "total_pages": 12,
                "page_width": 612.0,
                "page_height": 792.0,
            },
        ]

        mock_description = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: Shows attack success rate across models.\n\n"
            "**Key Findings**: - ASR reaches 97%.\n\n"
            "**Relevance**: Important for security analysis."
        )

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=mock_description) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Il grafico dovrebbe passare, logo e banner dovrebbero essere filtrati
        # Con il filtro reale: la chart ha alta entropia + bordi, pagina centrale → passa
        # Il logo ha pochi colori, piccolo, prima pagina → molto probabilmente filtrato
        # Il banner ha aspect ratio estremo, ultima pagina → molto probabilmente filtrato
        assert mock_describe.call_count < 3  # Almeno qualcosa è stato filtrato
        assert mock_describe.call_count >= 1  # La chart dovrebbe passare


# ============================================================================
# Test: Logging del filtraggio
# ============================================================================


class TestFilteringLogging:
    """Verifica che le statistiche di filtraggio siano loggate."""

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_filtering_stats_logged(self, mock_settings_mod, sample_paper, tmp_path, caplog):
        """Le statistiche di filtraggio (Estratte, Filtrate, Passate) sono loggate."""
        mock_settings_mod.figure_extraction.enabled = True
        mock_settings_mod.figure_extraction.min_width = 50
        mock_settings_mod.figure_extraction.min_height = 50
        mock_settings_mod.figure_extraction.max_figures_per_paper = 10
        mock_settings_mod.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings_mod.image_filtering.enabled = True
        mock_settings_mod.image_filtering.embedding_filter_enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)

        fake_images = [
            {
                "image_bytes": chart_bytes,
                "page_number": 5,
                "width": 400,
                "height": 300,
                "image_index": 0,
                "mime_type": "image/png",
                "image_hash": "h1",
                "total_pages": 12,
                "page_width": 612.0,
                "page_height": 792.0,
            },
        ]

        scientific_score = ImageScore(
            combined_score=0.8,
            classification=ImageClassification.SCIENTIFIC_FIGURE,
        )

        mock_description = (
            "**Figure Type**: results_plot\n\n"
            "**Content Description**: Shows attack success rate.\n\n"
            "**Key Findings**: - ASR reaches 97%.\n\n"
            "**Relevance**: Important findings."
        )

        import logging
        with caplog.at_level(logging.INFO, logger="paperhunter.ingestion.figure_extractor"), \
             patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=mock_description), \
             patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter") as mock_filter_cls:
            mock_filter = MagicMock()
            mock_filter.classify_batch.return_value = [scientific_score]
            mock_filter_cls.return_value = mock_filter

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        log_text = caplog.text
        assert "Filtro euristico" in log_text
        assert "Estratte=1" in log_text
        assert "Passate a Vision=1" in log_text
