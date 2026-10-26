"""
Test per ImageRelevanceFilter — Classificatore euristico immagini.

Genera immagini sintetiche e verifica che le classificazioni corrispondano
alle aspettative: grafici → SCIENTIFIC_FIGURE, loghi → LIKELY_DECORATIVE,
foto → UNCERTAIN.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from paperhunter.config import ImageFilteringConfig
from paperhunter.ingestion.image_relevance_filter import (
    ImageClassification,
    ImageRelevanceFilter,
    ImageScore,
)

pytestmark = pytest.mark.skip(reason="Visual and Gemini features are disabled in the DeepSeek-only profile.")


# ============================================================================
# Helper: generatori di immagini sintetiche
# ============================================================================


def _make_png(img: Image.Image) -> bytes:
    """Converte un'immagine PIL in bytes PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_chart_image(width: int = 400, height: int = 300) -> bytes:
    """Genera un'immagine sintetica tipo 'grafico scientifico'.

    - Assi X/Y con tick marks
    - Linee di dati con colori diversi
    - Griglia di sfondo
    - Alta entropia colore e alta densità bordi
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Margini
    margin = 40

    # Asse Y
    draw.line([(margin, margin), (margin, height - margin)], fill="black", width=2)
    # Asse X
    draw.line([(margin, height - margin), (width - margin, height - margin)], fill="black", width=2)

    # Tick marks e griglia
    for i in range(5):
        y = margin + i * (height - 2 * margin) // 4
        draw.line([(margin - 5, y), (margin, y)], fill="black", width=1)
        draw.line([(margin, y), (width - margin, y)], fill=(200, 200, 200), width=1)

    for i in range(6):
        x = margin + i * (width - 2 * margin) // 5
        draw.line([(x, height - margin), (x, height - margin + 5)], fill="black", width=1)
        draw.line([(x, margin), (x, height - margin)], fill=(200, 200, 200), width=1)

    # Linee dati
    rng = np.random.RandomState(42)
    colors = ["red", "blue", "green", "orange"]
    for color in colors:
        points = []
        y_base = rng.randint(60, 200)
        for i in range(20):
            x = margin + int(i * (width - 2 * margin) / 19)
            y = y_base + int(rng.normal(0, 15))
            y = max(margin, min(height - margin, y))
            points.append((x, y))
            y_base += rng.randint(-5, 6)
        for j in range(len(points) - 1):
            draw.line([points[j], points[j + 1]], fill=color, width=2)

    # Etichette assi (testo semplice)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((width // 2 - 20, height - 15), "Epoch", fill="black", font=font)
    draw.text((5, margin - 15), "Loss", fill="black", font=font)

    return _make_png(img)


def make_logo_image(width: int = 80, height: int = 80) -> bytes:
    """Genera un'immagine sintetica tipo 'logo'.

    - Piccola, pochi colori, bassa entropia
    - Forma semplice (cerchio + testo)
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Cerchio blu
    draw.ellipse([(10, 10), (width - 10, height - 10)], fill="navy", outline="navy")
    # Testo bianco
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((width // 2 - 10, height // 2 - 5), "AB", fill="white", font=font)

    return _make_png(img)


def make_photo_image(width: int = 300, height: int = 250) -> bytes:
    """Genera un'immagine sintetica tipo 'foto decorativa'.

    - Area prevalentemente uniforme con rumore leggero (entropia moderata)
    - Bordi morbidi (bassa densità bordi)
    - Simula una foto sfocata (es. sfondo, panorama)
    """
    rng = np.random.RandomState(123)
    # Base quasi uniforme (cielo azzurro) con rumore leggero
    arr = np.full((height, width, 3), [135, 180, 220], dtype=np.uint8)
    noise = rng.normal(0, 8, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Sfocatura pesante per eliminare strutture
    img = Image.fromarray(arr)
    from PIL import ImageFilter
    img = img.filter(ImageFilter.GaussianBlur(radius=4))
    return _make_png(img)


def make_table_image(width: int = 500, height: int = 350) -> bytes:
    """Genera un'immagine sintetica tipo 'tabella'.

    - Griglia rettangolare con bordi netti
    - Testo nelle celle
    - Alta densità bordi, buona entropia
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    n_rows, n_cols = 6, 4
    cell_w = (width - 40) // n_cols
    cell_h = (height - 40) // n_rows
    x0, y0 = 20, 20

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for r in range(n_rows + 1):
        y = y0 + r * cell_h
        draw.line([(x0, y), (x0 + n_cols * cell_w, y)], fill="black", width=2)
    for c in range(n_cols + 1):
        x = x0 + c * cell_w
        draw.line([(x, y0), (x, y0 + n_rows * cell_h)], fill="black", width=2)

    # Header row
    headers = ["Method", "Acc%", "F1", "Time"]
    for c, hdr in enumerate(headers):
        x = x0 + c * cell_w + 5
        draw.text((x, y0 + 3), hdr, fill="black", font=font)

    # Dati fittizi
    rng = np.random.RandomState(99)
    for r in range(1, n_rows):
        draw.text((x0 + 5, y0 + r * cell_h + 3), f"M{r}", fill="black", font=font)
        for c in range(1, n_cols):
            val = f"{rng.uniform(70, 99):.1f}"
            draw.text((x0 + c * cell_w + 5, y0 + r * cell_h + 3), val, fill="black", font=font)

    return _make_png(img)


def make_banner_image(width: int = 600, height: int = 50) -> bytes:
    """Genera un banner orizzontale (aspect ratio estremo)."""
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.text((10, 15), "Copyright 2025 Publisher", fill="white")
    return _make_png(img)


# ============================================================================
# Costanti test
# ============================================================================

# Parametri pagina standard Letter (~612 × 792 pt)
PAGE_W, PAGE_H = 612.0, 792.0
TOTAL_PAGES = 12


# ============================================================================
# Test singole euristiche
# ============================================================================


class TestHeuristics:
    """Test dei singoli metodi privati di scoring."""

    @pytest.fixture
    def filt(self):
        return ImageRelevanceFilter()

    # --- aspect ratio ---

    def test_aspect_ratio_ideal(self, filt):
        """Ratio 4:3 (1.33) → dentro range ideale → score = 1.0."""
        score = filt._score_aspect_ratio(400, 300)
        assert score == 1.0

    def test_aspect_ratio_extreme_wide(self, filt):
        """Ratio 12:1 → fuori range configurato → score = 0.0."""
        score = filt._score_aspect_ratio(600, 50)
        assert score == 0.0

    def test_aspect_ratio_square(self, filt):
        """Ratio 1:1 → dentro range ideale → score = 1.0."""
        score = filt._score_aspect_ratio(200, 200)
        assert score == 1.0

    def test_aspect_ratio_extreme_tall(self, filt):
        """Ratio 1:5 (0.2) → fuori range configurato → score = 0.0."""
        score = filt._score_aspect_ratio(50, 250)
        assert score == 0.0

    # --- color entropy ---

    def test_entropy_uniform_image(self, filt):
        """Immagine completamente bianca → entropia bassissima → score basso."""
        img = Image.new("RGB", (100, 100), "white")
        score = filt._score_color_entropy(img)
        assert score < 0.3

    def test_entropy_colorful_image(self, filt):
        """Immagine con molti colori → entropia alta → score alto."""
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        score = filt._score_color_entropy(img)
        assert score > 0.6

    # --- edge density ---

    def test_edge_density_grid(self, filt):
        """Griglia con bordi netti → alta densità → score alto."""
        img = Image.new("L", (200, 200), 255)
        draw = ImageDraw.Draw(img)
        for i in range(0, 200, 20):
            draw.line([(i, 0), (i, 200)], fill=0, width=2)
            draw.line([(0, i), (200, i)], fill=0, width=2)
        img_rgb = img.convert("RGB")
        score = filt._score_edge_density(img_rgb)
        assert score > 0.3

    def test_edge_density_smooth(self, filt):
        """Immagine liscia (sfumatura) → bassa densità → score basso."""
        arr = np.zeros((200, 200), dtype=np.uint8)
        for x in range(200):
            arr[:, x] = int(x / 200 * 255)
        img = Image.fromarray(arr).convert("RGB")
        score = filt._score_edge_density(img)
        assert score < 0.5

    # --- page position ---

    def test_position_first_page(self, filt):
        """Prima pagina → penalizzata."""
        score = filt._score_page_position(1, 12)
        assert score == 0.3

    def test_position_last_page(self, filt):
        """Ultima pagina → penalizzata."""
        score = filt._score_page_position(12, 12)
        assert score == 0.3

    def test_position_middle_page(self, filt):
        """Pagina centrale → score pieno."""
        score = filt._score_page_position(6, 12)
        assert score == 1.0

    # --- relative size ---

    def test_size_tiny(self, filt):
        """Immagine minuscola (2% pagina) → score basso."""
        page_area = PAGE_W * PAGE_H
        img_area = page_area * 0.02
        side = int(math.sqrt(img_area))
        score = filt._score_relative_size(side, side, PAGE_W, PAGE_H)
        assert score < 0.3

    def test_size_ideal(self, filt):
        """Immagine ~30% pagina → score pieno."""
        page_area = PAGE_W * PAGE_H
        img_area = page_area * 0.3
        side = int(math.sqrt(img_area))
        score = filt._score_relative_size(side, side, PAGE_W, PAGE_H)
        assert score == 1.0

    def test_size_fullpage(self, filt):
        """Immagine ~95% pagina → score basso (background)."""
        score = filt._score_relative_size(int(PAGE_W * 0.97), int(PAGE_H * 0.97), PAGE_W, PAGE_H)
        assert score < 0.3


# ============================================================================
# Test classificazione complessiva
# ============================================================================


class TestClassification:
    """Verifica che le immagini sintetiche vengano classificate come atteso."""

    @pytest.fixture
    def filt(self):
        return ImageRelevanceFilter()

    def test_chart_is_scientific(self, filt):
        """Un grafico con assi, linee e colori → SCIENTIFIC_FIGURE."""
        chart = make_chart_image()
        score = filt.classify(chart, page_number=5, total_pages=12,
                              page_width=PAGE_W, page_height=PAGE_H)
        assert score.classification == ImageClassification.SCIENTIFIC_FIGURE
        assert score.combined_score >= 0.5

    def test_table_is_scientific(self, filt):
        """Una tabella con griglia e dati → SCIENTIFIC_FIGURE."""
        table = make_table_image()
        score = filt.classify(table, page_number=7, total_pages=12,
                              page_width=PAGE_W, page_height=PAGE_H)
        assert score.classification == ImageClassification.SCIENTIFIC_FIGURE
        assert score.combined_score >= 0.5

    def test_logo_is_decorative(self, filt):
        """Un logo piccolo con pochi colori → LIKELY_DECORATIVE o UNCERTAIN.

        Il logo ha aspect ratio 1:1 (score alto), ma è piccolo e con
        entropia bassa, quindi il punteggio combinato è sotto la soglia.
        """
        logo = make_logo_image()
        score = filt.classify(logo, page_number=1, total_pages=12,
                              page_width=PAGE_W, page_height=PAGE_H)
        assert score.classification in (
            ImageClassification.LIKELY_DECORATIVE,
            ImageClassification.UNCERTAIN,
        )
        assert score.combined_score < 0.5

    def test_banner_is_decorative(self, filt):
        """Un banner orizzontale con aspect ratio estremo → LIKELY_DECORATIVE."""
        banner = make_banner_image()
        score = filt.classify(banner, page_number=12, total_pages=12,
                              page_width=PAGE_W, page_height=PAGE_H)
        # Banner ha aspect ratio 12:1 → fuori range → score basso
        assert score.classification in (
            ImageClassification.LIKELY_DECORATIVE,
            ImageClassification.UNCERTAIN,
        )
        assert score.combined_score < 0.5

    def test_photo_is_uncertain_or_less(self, filt):
        """Una foto sfocata e uniforme su prima pagina → UNCERTAIN o LIKELY_DECORATIVE.

        Nota: una foto con dimensioni normali su pagina centrale ha 3/5 score
        perfetti (aspect ratio, position, size), quindi il filtro euristico da
        solo non la può distinguere — serve lo Stage 2 (embedding).
        Su prima pagina il position penalty aiuta la classificazione.
        """
        photo = make_photo_image()
        score = filt.classify(photo, page_number=1, total_pages=12,
                              page_width=PAGE_W, page_height=PAGE_H)
        # Foto su prima pagina: position penalizzata + entropy bassa + bordi bassi
        assert score.classification in (
            ImageClassification.UNCERTAIN,
            ImageClassification.LIKELY_DECORATIVE,
        )
        assert score.combined_score < 0.5


# ============================================================================
# Test classify_batch
# ============================================================================


class TestBatch:
    """Verifica il batch processing e le statistiche."""

    def test_batch_processing(self):
        filt = ImageRelevanceFilter()
        images = [
            {
                "image_bytes": make_chart_image(),
                "page_number": 5,
                "total_pages": 12,
                "page_width": PAGE_W,
                "page_height": PAGE_H,
            },
            {
                "image_bytes": make_logo_image(),
                "page_number": 1,
                "total_pages": 12,
                "page_width": PAGE_W,
                "page_height": PAGE_H,
            },
            {
                "image_bytes": make_photo_image(),
                "page_number": 6,
                "total_pages": 12,
                "page_width": PAGE_W,
                "page_height": PAGE_H,
            },
        ]

        scores = filt.classify_batch(images)
        assert len(scores) == 3

        # Il grafico dovrebbe essere scientifico
        assert scores[0].classification == ImageClassification.SCIENTIFIC_FIGURE
        # Il logo dovrebbe essere decorativo o incerto (basso score)
        assert scores[1].classification in (
            ImageClassification.LIKELY_DECORATIVE,
            ImageClassification.UNCERTAIN,
        )


# ============================================================================
# Test edge cases
# ============================================================================


class TestEdgeCases:
    """Test di robustezza per casi limite."""

    def test_invalid_image_bytes(self):
        filt = ImageRelevanceFilter()
        score = filt.classify(
            b"not an image",
            page_number=1, total_pages=1,
            page_width=612, page_height=792,
        )
        assert score.classification == ImageClassification.UNCERTAIN

    def test_single_page_document(self):
        filt = ImageRelevanceFilter()
        chart = make_chart_image()
        score = filt.classify(chart, page_number=1, total_pages=1,
                              page_width=PAGE_W, page_height=PAGE_H)
        # Single page: ancora classificabile
        assert isinstance(score.classification, ImageClassification)

    def test_custom_config(self):
        """Config personalizzata con soglia alta: più restrittivo."""
        config = ImageFilteringConfig(
            enabled=True,
            relevance_threshold=0.8,  # Soglia alta
            min_aspect_ratio=0.5,
            max_aspect_ratio=3.0,
            min_color_entropy=4.0,
            min_edge_density=0.1,
        )
        filt = ImageRelevanceFilter(config=config)
        chart = make_chart_image()
        score = filt.classify(chart, page_number=5, total_pages=12,
                              page_width=PAGE_W, page_height=PAGE_H)
        # Con soglia alta, anche un buon grafico potrebbe essere UNCERTAIN
        assert isinstance(score.classification, ImageClassification)

    def test_zero_page_dimensions(self):
        """Pagina con dimensioni zero: non deve crashare."""
        filt = ImageRelevanceFilter()
        chart = make_chart_image()
        score = filt.classify(chart, page_number=1, total_pages=1,
                              page_width=0, page_height=0)
        assert isinstance(score.classification, ImageClassification)

    def test_greyscale_image(self):
        """Immagine greyscale: deve funzionare correttamente."""
        filt = ImageRelevanceFilter()
        img = Image.new("L", (300, 200), 128)
        draw = ImageDraw.Draw(img)
        for i in range(0, 300, 30):
            draw.line([(i, 0), (i, 200)], fill=0, width=2)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        score = filt.classify(buf.getvalue(), page_number=5, total_pages=12,
                              page_width=PAGE_W, page_height=PAGE_H)
        assert isinstance(score.classification, ImageClassification)

    def test_all_scores_in_range(self):
        """Tutti i punteggi devono essere in [0, 1]."""
        filt = ImageRelevanceFilter()
        for factory in [make_chart_image, make_logo_image, make_photo_image,
                        make_table_image, make_banner_image]:
            score = filt.classify(factory(), page_number=5, total_pages=12,
                                  page_width=PAGE_W, page_height=PAGE_H)
            assert 0.0 <= score.aspect_ratio_score <= 1.0
            assert 0.0 <= score.entropy_score <= 1.0
            assert 0.0 <= score.edge_density_score <= 1.0
            assert 0.0 <= score.position_score <= 1.0
            assert 0.0 <= score.size_score <= 1.0
            assert 0.0 <= score.combined_score <= 1.0
