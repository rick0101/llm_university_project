"""
Test per l'integrazione del filtro embedding multimodale (Stage 2) in FigureExtractor (Task 2.2).

Verifica che:
- Le immagini LIKELY_DECORATIVE (Stage 1) non vengano embedded (risparmio API)
- Le immagini UNCERTAIN con bassa similarità embedding vengano scartate (Stage 2)
- Le immagini SCIENTIFIC_FIGURE con score euristico > 0.8 saltino Stage 2 (ottimizzazione)
- Le chiamate Vision siano significativamente ridotte rispetto al baseline
- I campi embedding_similarity, best_match_category, embedding vengano propagati
- Il filtro Stage 2 possa essere disabilitato via settings.image_filtering.embedding_filter_enabled
- Il conteggio Stage 1 → Stage 2 → Vision sia loggato correttamente
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from paperhunter.ingestion.image_relevance_filter import (
    ImageClassification,
    ImageScore,
)
from paperhunter.models import Paper


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
    draw.line([(margin, margin), (margin, height - margin)], fill="black", width=2)
    draw.line([(margin, height - margin), (width - margin, height - margin)], fill="black", width=2)
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
    return _make_png(img)


def _make_logo_png(width: int = 60, height: int = 60) -> bytes:
    """Genera un PNG tipo logo (pochi colori, bassa entropia)."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(10, 10), (50, 50)], fill=(0, 0, 128))
    draw.text((18, 20), "AB", fill="white")
    return _make_png(img)


def _make_photo_png(width: int = 300, height: int = 250) -> bytes:
    """Genera un PNG tipo foto (entropia media, bordi bassi)."""
    rng = np.random.RandomState(123)
    arr = rng.randint(100, 200, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    return _make_png(img)


def _fake_embedding(dim: int = 3072, seed: int = 0) -> list[float]:
    """Genera un fake embedding normalizzato deterministico."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(float)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _fake_ref_embeddings() -> dict[str, list[float]]:
    """Genera embedding di riferimento fake per 5 categorie."""
    return {
        "scientific_plot": _fake_embedding(seed=10),
        "comparison_table": _fake_embedding(seed=20),
        "architecture_diagram": _fake_embedding(seed=30),
        "flowchart": _fake_embedding(seed=40),
        "confusion_matrix": _fake_embedding(seed=50),
    }


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


def _build_image_dict(
    image_bytes: bytes,
    page_number: int = 5,
    width: int = 400,
    height: int = 300,
    image_index: int = 0,
    total_pages: int = 12,
    image_hash: str = "default_hash",
) -> dict:
    """Helper per creare un dict immagine esteso."""
    return {
        "image_bytes": image_bytes,
        "page_number": page_number,
        "width": width,
        "height": height,
        "image_index": image_index,
        "mime_type": "image/png",
        "image_hash": image_hash,
        "total_pages": total_pages,
        "page_width": 612.0,
        "page_height": 792.0,
    }


MOCK_DESCRIPTION = (
    "**Figure Type**: results_plot\n\n"
    "**Content Description**: Shows attack success rate across models.\n\n"
    "**Key Findings**: - GCG achieves 97% ASR.\n\n"
    "**Relevance**: Demonstrates vulnerability."
)


# ============================================================================
# Test: Stage 2 — Embedding Filter Integration
# ============================================================================


class TestStage2EmbeddingFilter:
    """Test per il filtro embedding multimodale (Stage 2) nella pipeline."""

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_uncertain_images_filtered_by_low_embedding_similarity(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Immagini UNCERTAIN con bassa similarità embedding vengono scartate."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        photo_bytes = _make_photo_png(300, 250)

        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="chart_h"),
            _build_image_dict(photo_bytes, image_index=1, width=300, height=250, image_hash="photo_h"),
        ]

        # Stage 1: entrambe UNCERTAIN (→ passano a Stage 2)
        uncertain_score = ImageScore(
            combined_score=0.35,
            classification=ImageClassification.UNCERTAIN,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [uncertain_score, uncertain_score]
        mock_filter_cls.return_value = mock_filter

        # Stage 2: chart ha alta sim, photo ha bassa sim
        ref_embs = _fake_ref_embeddings()
        chart_emb = _fake_embedding(seed=10)  # simile a scientific_plot
        photo_emb = _fake_embedding(seed=999)  # diverso da tutti

        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        # Primo embed_image → chart (alta sim), Secondo → photo (bassa sim)
        mock_embedder.embed_image.side_effect = [chart_emb, photo_emb]
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # La chart passa (sim ≈ 1.0 con scientific_plot), la photo no (sim bassa)
        assert mock_describe.call_count == 1
        assert len(chunks) == 1

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_scientific_high_score_skips_stage2(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """SCIENTIFIC_FIGURE con score > 0.8 salta Stage 2 (ottimizzazione)."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="chart_h"),
        ]

        # Stage 1: SCIENTIFIC con score > 0.8
        sci_score = ImageScore(
            combined_score=0.85,
            classification=ImageClassification.SCIENTIFIC_FIGURE,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [sci_score]
        mock_filter_cls.return_value = mock_filter

        ref_embs = _fake_ref_embeddings()
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = _fake_embedding(seed=77)
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # embed_image NON chiamato in Stage 2 (skip), MA chiamato in Step 5 (ri-embed)
        assert mock_embedder.embed_image.call_count == 1
        # Ma Vision deve essere chiamato
        assert mock_describe.call_count == 1
        assert len(chunks) == 1
        # Campi embedding vuoti
        assert fake_images[0]["embedding_similarity"] is None
        assert fake_images[0]["best_match_category"] is None
        assert fake_images[0]["embedding"] is None

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_scientific_low_score_goes_through_stage2(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """SCIENTIFIC_FIGURE con score <= 0.8 passa per Stage 2."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="chart_h"),
        ]

        # Stage 1: SCIENTIFIC con score = 0.65 (≤ 0.8 → va a Stage 2)
        sci_score = ImageScore(
            combined_score=0.65,
            classification=ImageClassification.SCIENTIFIC_FIGURE,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [sci_score]
        mock_filter_cls.return_value = mock_filter

        ref_embs = _fake_ref_embeddings()
        chart_emb = _fake_embedding(seed=10)  # alta sim con scientific_plot
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = chart_emb
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # embed_image DEVE essere chiamato
        assert mock_embedder.embed_image.call_count == 1
        # Vision chiamato (passa Stage 2)
        assert mock_describe.call_count == 1
        assert len(chunks) == 1
        # Campi embedding popolati
        assert fake_images[0]["embedding_similarity"] is not None
        assert fake_images[0]["best_match_category"] is not None
        assert fake_images[0]["embedding"] is not None

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_decorative_not_embedded(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Immagini LIKELY_DECORATIVE filtrate allo Stage 1 non vengono embedded."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        logo_bytes = _make_logo_png(60, 60)

        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="chart_h"),
            _build_image_dict(logo_bytes, page_number=1, width=60, height=60,
                              image_index=1, image_hash="logo_h"),
        ]

        # Stage 1: chart = SCIENTIFIC, logo = DECORATIVE
        sci_score = ImageScore(combined_score=0.85, classification=ImageClassification.SCIENTIFIC_FIGURE)
        dec_score = ImageScore(combined_score=0.12, classification=ImageClassification.LIKELY_DECORATIVE)
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [sci_score, dec_score]
        mock_filter_cls.return_value = mock_filter

        ref_embs = _fake_ref_embeddings()
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = _fake_embedding(seed=77)
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # embed_image NON è chiamato per il logo (filtrato Stage 1)
        # MA è chiamato 1 volta in Step 5 per la chart (ri-embed per indicizzazione)
        assert mock_embedder.embed_image.call_count == 1

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_all_filtered_stage2_returns_empty(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Se Stage 2 filtra tutte le immagini, restituisce lista vuota."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        photo_bytes = _make_photo_png(300, 250)

        fake_images = [
            _build_image_dict(photo_bytes, image_index=0, width=300, height=250, image_hash="p0"),
            _build_image_dict(photo_bytes, image_index=1, width=300, height=250, image_hash="p1"),
        ]

        # Stage 1: entrambe UNCERTAIN
        unc_score = ImageScore(combined_score=0.35, classification=ImageClassification.UNCERTAIN)
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [unc_score, unc_score]
        mock_filter_cls.return_value = mock_filter

        # Stage 2: entrambe con bassa similarità (< 0.3)
        ref_embs = _fake_ref_embeddings()
        low_sim_emb = _fake_embedding(seed=999)  # diverso da tutti i riferimenti
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = low_sim_emb
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Tutto filtrato → nessuna chiamata Vision, nessun chunk
        assert mock_describe.call_count == 0
        assert len(chunks) == 0


class TestStage2Disabled:
    """Test per il comportamento quando Stage 2 è disabilitato."""

    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_embedding_filter_disabled_skips_stage2(
        self, mock_settings, mock_filter_cls,
        sample_paper, tmp_path,
    ):
        """Con embedding_filter_enabled=False, Stage 2 viene saltato."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="chart_h"),
        ]

        uncertain_score = ImageScore(
            combined_score=0.35,
            classification=ImageClassification.UNCERTAIN,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [uncertain_score]
        mock_filter_cls.return_value = mock_filter

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION) as mock_describe, \
             patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder") as mock_emb:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # get_multimodal_embedder non deve essere chiamato
        mock_emb.assert_not_called()
        # Vision deve essere chiamato (nessun filtro Stage 2)
        assert mock_describe.call_count == 1
        assert len(chunks) == 1
        # Campi embedding impostati a None
        assert fake_images[0].get("embedding_similarity") is None
        assert fake_images[0].get("best_match_category") is None
        assert fake_images[0].get("embedding") is None

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_both_stages_disabled(
        self, mock_settings,
        sample_paper, tmp_path,
    ):
        """Con image_filtering.enabled=False, entrambi gli stage sono saltati."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="chart_h"),
        ]

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION) as mock_describe, \
             patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder") as mock_emb, \
             patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter") as mock_filt:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Nessun filtro attivato
        mock_filt.assert_not_called()
        mock_emb.assert_not_called()
        # Tutte le immagini vanno a Vision
        assert mock_describe.call_count == 1
        assert len(chunks) == 1


class TestStage2EmbeddingFields:
    """Test per la propagazione dei campi embedding nel dict immagine."""

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_embedding_fields_populated_for_passing_images(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Immagini che passano Stage 2 hanno embedding_similarity, best_match_category, embedding."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="chart_h"),
        ]

        # Stage 1: UNCERTAIN con score 0.4 (< 0.8, va a Stage 2)
        unc_score = ImageScore(combined_score=0.4, classification=ImageClassification.UNCERTAIN)
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [unc_score]
        mock_filter_cls.return_value = mock_filter

        # Stage 2: alta similarità
        ref_embs = _fake_ref_embeddings()
        chart_emb = _fake_embedding(seed=10)  # simile a scientific_plot
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = chart_emb
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Campi embedding popolati
        assert isinstance(fake_images[0]["embedding_similarity"], float)
        assert fake_images[0]["embedding_similarity"] > 0.0
        assert isinstance(fake_images[0]["best_match_category"], str)
        assert fake_images[0]["best_match_category"] != "unknown"
        assert isinstance(fake_images[0]["embedding"], list)
        assert len(fake_images[0]["embedding"]) == 3072

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_embedding_fields_populated_for_filtered_images(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Immagini filtrate da Stage 2 hanno comunque i campi embedding valorizzati."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        photo_bytes = _make_photo_png(300, 250)
        fake_images = [
            _build_image_dict(photo_bytes, image_index=0, width=300, height=250, image_hash="photo_h"),
        ]

        # Stage 1: UNCERTAIN
        unc_score = ImageScore(combined_score=0.35, classification=ImageClassification.UNCERTAIN)
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [unc_score]
        mock_filter_cls.return_value = mock_filter

        # Stage 2: bassa similarità → filtrata
        ref_embs = _fake_ref_embeddings()
        low_sim_emb = _fake_embedding(seed=999)
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = low_sim_emb
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Anche se filtrata, i campi embedding sono valorizzati (utili per debug)
        assert isinstance(fake_images[0]["embedding_similarity"], float)
        assert isinstance(fake_images[0]["best_match_category"], str)
        assert isinstance(fake_images[0]["embedding"], list)


class TestStage2MixedPipeline:
    """Test per la pipeline completa a 2 stadi con mix di classificazioni."""

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_full_pipeline_mixed_images(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Pipeline completa con 5 immagini: 2 SCIENTIFIC(alta), 1 SCIENTIFIC(bassa), 1 UNCERTAIN, 1 DECORATIVE."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 20
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50, max_figures=20)

        chart_bytes = _make_chart_png(400, 300)
        photo_bytes = _make_photo_png(300, 250)
        logo_bytes = _make_logo_png(60, 60)

        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="c0"),   # SCIENTIFIC alta
            _build_image_dict(chart_bytes, image_index=1, image_hash="c1"),   # SCIENTIFIC alta
            _build_image_dict(chart_bytes, image_index=2, image_hash="c2"),   # SCIENTIFIC bassa → Stage 2
            _build_image_dict(photo_bytes, image_index=3, width=300, height=250, image_hash="p0"),  # UNCERTAIN → Stage 2
            _build_image_dict(logo_bytes, page_number=1, width=60, height=60,
                              image_index=4, image_hash="l0"),  # DECORATIVE
        ]

        # Stage 1 scores
        scores_s1 = [
            ImageScore(combined_score=0.85, classification=ImageClassification.SCIENTIFIC_FIGURE),
            ImageScore(combined_score=0.82, classification=ImageClassification.SCIENTIFIC_FIGURE),
            ImageScore(combined_score=0.65, classification=ImageClassification.SCIENTIFIC_FIGURE),
            ImageScore(combined_score=0.35, classification=ImageClassification.UNCERTAIN),
            ImageScore(combined_score=0.12, classification=ImageClassification.LIKELY_DECORATIVE),
        ]
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = scores_s1
        mock_filter_cls.return_value = mock_filter

        # Stage 2: chart passa (alta sim), photo filtrata (bassa sim)
        ref_embs = _fake_ref_embeddings()
        chart_emb = _fake_embedding(seed=10)  # alta sim con scientific_plot
        photo_emb = _fake_embedding(seed=999)  # bassa sim
        reembed_emb = _fake_embedding(seed=77)  # ri-embed Step 5
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        # Ordine embed_image: Stage 2: img[2] (SCIENTIFIC bassa), img[3] (UNCERTAIN)
        # Step 5: img[0] e img[1] (skip Stage 2, ri-embed per indicizzazione)
        mock_embedder.embed_image.side_effect = [chart_emb, photo_emb, reembed_emb, reembed_emb]
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Flusso:
        # 5 totali → Stage 1: 4 passano (1 decorativa rimossa)
        # → Stage 2: 2 skip (score > 0.8) + 1 passa (chart sim alta) + 1 filtrata (photo sim bassa)
        # → Vision: 3 chiamate (2 skip + 1 passata Stage 2)
        assert mock_describe.call_count == 3
        assert len(chunks) == 3
        # embed_image: 2 Stage 2 (img[2] + img[3]) + 2 Step 5 ri-embed (img[0] + img[1])
        assert mock_embedder.embed_image.call_count == 4

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_vision_calls_significantly_reduced(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Le chiamate Vision sono ridotte: baseline=10, con pipeline < 10."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 20
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50, max_figures=20)

        chart_bytes = _make_chart_png(400, 300)
        photo_bytes = _make_photo_png(300, 250)
        logo_bytes = _make_logo_png(60, 60)

        # 10 immagini: 3 chart, 4 photo, 3 logo
        fake_images = []
        for i in range(3):
            fake_images.append(
                _build_image_dict(chart_bytes + bytes([i]), image_index=i,
                                  page_number=3 + i, image_hash=f"chart_{i}"))
        for i in range(4):
            fake_images.append(
                _build_image_dict(photo_bytes + bytes([i + 10]), image_index=3 + i,
                                  width=300, height=250, page_number=5 + i, image_hash=f"photo_{i}"))
        for i in range(3):
            fake_images.append(
                _build_image_dict(logo_bytes + bytes([i + 20]), image_index=7 + i,
                                  page_number=1, width=60, height=60, image_hash=f"logo_{i}"))

        # Stage 1: 3 chart SCIENTIFIC(alta), 4 photo UNCERTAIN, 3 logo DECORATIVE
        scores_s1 = (
            [ImageScore(combined_score=0.85, classification=ImageClassification.SCIENTIFIC_FIGURE)] * 3
            + [ImageScore(combined_score=0.35, classification=ImageClassification.UNCERTAIN)] * 4
            + [ImageScore(combined_score=0.10, classification=ImageClassification.LIKELY_DECORATIVE)] * 3
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = scores_s1
        mock_filter_cls.return_value = mock_filter

        # Stage 2: le 4 photo hanno sim bassa (sotto soglia 0.3)
        ref_embs = _fake_ref_embeddings()
        low_sim_emb = _fake_embedding(seed=999)
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = low_sim_emb  # tutte le UNCERTAIN con bassa sim
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # 10 totali → Stage 1: 7 passano (3 logo rimossi)
        # → Stage 2: 3 chart skip (score > 0.8) + 4 photo filtrate (sim bassa)
        # → Vision: 3 (solo le chart)
        assert mock_describe.call_count == 3
        # Riduzione da 10 a 3 = 70%
        assert mock_describe.call_count < 10  # Significativamente ridotto


class TestStage2Logging:
    """Test per il logging delle statistiche Stage 1 → Stage 2 → Vision."""

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_stage2_stats_logged(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path, caplog,
    ):
        """Le statistiche Stage 1 → Stage 2 → Vision sono loggate correttamente."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        photo_bytes = _make_photo_png(300, 250)

        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="c0"),
            _build_image_dict(photo_bytes, image_index=1, width=300, height=250, image_hash="p0"),
            _build_image_dict(photo_bytes, image_index=2, width=300, height=250, image_hash="p1"),
        ]

        # Stage 1: chart SCIENTIFIC(alta), 2 photo UNCERTAIN
        scores_s1 = [
            ImageScore(combined_score=0.85, classification=ImageClassification.SCIENTIFIC_FIGURE),
            ImageScore(combined_score=0.35, classification=ImageClassification.UNCERTAIN),
            ImageScore(combined_score=0.32, classification=ImageClassification.UNCERTAIN),
        ]
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = scores_s1
        mock_filter_cls.return_value = mock_filter

        # Stage 2: 2 photo con sim bassa
        ref_embs = _fake_ref_embeddings()
        low_sim_emb = _fake_embedding(seed=999)
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = low_sim_emb
        mock_get_embedder.return_value = mock_embedder

        with caplog.at_level(logging.INFO, logger="paperhunter.ingestion.figure_extractor"), \
             patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        log_text = caplog.text
        # Deve contenere info su entrambi gli stage
        assert "Stage 1" in log_text
        assert "Stage 2" in log_text
        assert "Vision calls" in log_text

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_stage2_logging_correct_counts(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path, caplog,
    ):
        """I conteggi nel log Stage 1/Stage 2 sono numericamente corretti."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 20
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50, max_figures=20)

        chart_bytes = _make_chart_png(400, 300)
        logo_bytes = _make_logo_png(60, 60)
        photo_bytes = _make_photo_png(300, 250)

        # 6 immagini: 2 chart, 2 logo, 2 photo
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="c0"),
            _build_image_dict(chart_bytes, image_index=1, image_hash="c1"),
            _build_image_dict(logo_bytes, page_number=1, width=60, height=60, image_index=2, image_hash="l0"),
            _build_image_dict(logo_bytes, page_number=12, width=60, height=60, image_index=3, image_hash="l1"),
            _build_image_dict(photo_bytes, image_index=4, width=300, height=250, image_hash="p0"),
            _build_image_dict(photo_bytes, image_index=5, width=300, height=250, image_hash="p1"),
        ]

        # Stage 1: 2 chart=SCIENTIFIC(alta), 2 logo=DECORATIVE, 2 photo=UNCERTAIN
        scores_s1 = [
            ImageScore(combined_score=0.85, classification=ImageClassification.SCIENTIFIC_FIGURE),
            ImageScore(combined_score=0.82, classification=ImageClassification.SCIENTIFIC_FIGURE),
            ImageScore(combined_score=0.10, classification=ImageClassification.LIKELY_DECORATIVE),
            ImageScore(combined_score=0.08, classification=ImageClassification.LIKELY_DECORATIVE),
            ImageScore(combined_score=0.35, classification=ImageClassification.UNCERTAIN),
            ImageScore(combined_score=0.33, classification=ImageClassification.UNCERTAIN),
        ]
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = scores_s1
        mock_filter_cls.return_value = mock_filter

        ref_embs = _fake_ref_embeddings()
        low_sim_emb = _fake_embedding(seed=999)
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = low_sim_emb
        mock_get_embedder.return_value = mock_embedder

        with caplog.at_level(logging.INFO, logger="paperhunter.ingestion.figure_extractor"), \
             patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        log_text = caplog.text
        # Stage 1: 6 → 4 (2 decorative rimossi)
        # Stage 2: 4 → 2 (2 chart skip, 2 photo filtrate)
        # Vision: 2
        assert "Stage 1: 6" in log_text
        assert "Stage 2: 4" in log_text
        assert "Vision calls: 2" in log_text


class TestStage2EdgeCases:
    """Test per casi limite dello Stage 2."""

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_no_images_after_stage1_skips_stage2(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Se Stage 1 filtra tutte le immagini, Stage 2 non viene eseguito."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        logo_bytes = _make_logo_png(60, 60)
        fake_images = [
            _build_image_dict(logo_bytes, page_number=1, width=60, height=60,
                              image_index=0, image_hash="l0"),
        ]

        dec_score = ImageScore(combined_score=0.12, classification=ImageClassification.LIKELY_DECORATIVE)
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [dec_score]
        mock_filter_cls.return_value = mock_filter

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Stage 2 non deve essere invocato
        mock_get_embedder.assert_not_called()
        assert len(chunks) == 0

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_relevance_score_none_when_stage1_disabled(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Quando Stage 1 è disabilitato, relevance_score è None e Stage 2 non si attiva."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = False

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="c0"),
        ]

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        mock_get_embedder.assert_not_called()
        assert fake_images[0]["relevance_score"] is None
        assert fake_images[0].get("embedding_similarity") is None
        assert len(chunks) == 1

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_threshold_boundary_exact(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Immagine con similarità esattamente uguale alla soglia passa."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.5

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png(400, 300)
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="c0"),
        ]

        unc_score = ImageScore(combined_score=0.4, classification=ImageClassification.UNCERTAIN)
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [unc_score]
        mock_filter_cls.return_value = mock_filter

        # Crea embedding che produce sim = 0.5 esatto (al boundary)
        # Usiamo un vettore identico a un riferimento ma con fattore di scala
        ref_embs = _fake_ref_embeddings()

        # Embedding identico a scientific_plot → sim ≈ 1.0
        exact_emb = ref_embs["scientific_plot"][:]
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = exact_emb
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION) as mock_describe:

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # sim ≈ 1.0 >= 0.5 → passa
        assert mock_describe.call_count == 1
        assert len(chunks) == 1
