"""
Image Relevance Filter — Classificazione euristica delle immagini estratte dai PDF.

Classifica le immagini come SCIENTIFIC_FIGURE, LIKELY_DECORATIVE o UNCERTAIN
usando solo euristiche locali (PIL + NumPy), senza chiamate API.

Pipeline a punteggio combinato basata su:
- Aspect ratio (penalizza barre e banner)
- Entropia colore Shannon (bassa → loghi monocromo)
- Densità bordi (alta → grafici/tabelle)
- Posizione nel documento (prima/ultima pagina → logo/copyright)
- Dimensione relativa alla pagina (troppo piccola o troppo grande)

Uso::

    from paperhunter.ingestion.image_relevance_filter import ImageRelevanceFilter

    filt = ImageRelevanceFilter()
    score = filt.classify(
        image_bytes=png_bytes,
        page_number=3,
        total_pages=12,
        page_width=612.0,
        page_height=792.0,
    )
    print(score.classification, score.combined_score)
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from paperhunter.config import settings, ImageFilteringConfig
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ============================================================================
# Enum e Dataclass
# ============================================================================


class ImageClassification(str, Enum):
    """Classificazione euristica di un'immagine estratta da PDF."""
    SCIENTIFIC_FIGURE = "scientific_figure"
    LIKELY_DECORATIVE = "likely_decorative"
    UNCERTAIN = "uncertain"


@dataclass
class ImageScore:
    """Punteggi euristici e classificazione finale di un'immagine."""
    aspect_ratio_score: float = 0.0
    entropy_score: float = 0.0
    edge_density_score: float = 0.0
    position_score: float = 0.0
    size_score: float = 0.0
    combined_score: float = 0.0
    classification: ImageClassification = ImageClassification.UNCERTAIN


# ============================================================================
# Pesi per la media pesata
# ============================================================================

_DEFAULT_WEIGHTS = {
    "aspect_ratio": 0.15,
    "entropy": 0.25,
    "edge_density": 0.30,
    "position": 0.15,
    "size": 0.15,
}


# ============================================================================
# Classe principale
# ============================================================================


class ImageRelevanceFilter:
    """Classificatore euristico per immagini estratte da PDF scientifici.

    Analizza caratteristiche visive locali (senza chiamate API) per decidere
    se un'immagine è probabilmente una figura scientifica, decorativa, o incerta.

    Args:
        config: Configurazione filtraggio. Se ``None``, usa ``settings.image_filtering``.
        weights: Pesi per la media pesata dei punteggi. Se ``None``, usa i default.
    """

    def __init__(
        self,
        config: ImageFilteringConfig | None = None,
        weights: dict[str, float] | None = None,
    ):
        self._config = config or settings.image_filtering
        self._weights = weights or _DEFAULT_WEIGHTS.copy()
        # Normalizza pesi a somma 1
        total_w = sum(self._weights.values())
        if total_w > 0:
            self._weights = {k: v / total_w for k, v in self._weights.items()}

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def classify(
        self,
        image_bytes: bytes,
        page_number: int,
        total_pages: int,
        page_width: float,
        page_height: float,
    ) -> ImageScore:
        """Classifica una singola immagine.

        Args:
            image_bytes: Contenuto dell'immagine (PNG/JPEG).
            page_number: Numero pagina di origine (1-based).
            total_pages: Numero totale di pagine del PDF.
            page_width: Larghezza pagina in punti PDF (es. 612).
            page_height: Altezza pagina in punti PDF (es. 792).

        Returns:
            :class:`ImageScore` con punteggi individuali, combinato e classificazione.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.warning("Impossibile aprire immagine per classificazione: %s", e)
            return ImageScore(classification=ImageClassification.UNCERTAIN)

        img_width, img_height = img.size

        aspect = self._score_aspect_ratio(img_width, img_height)
        entropy = self._score_color_entropy(img)
        edge = self._score_edge_density(img)
        position = self._score_page_position(page_number, total_pages)
        size = self._score_relative_size(img_width, img_height, page_width, page_height)

        combined = (
            self._weights["aspect_ratio"] * aspect
            + self._weights["entropy"] * entropy
            + self._weights["edge_density"] * edge
            + self._weights["position"] * position
            + self._weights["size"] * size
        )

        threshold = self._config.relevance_threshold
        if combined >= threshold:
            classification = ImageClassification.SCIENTIFIC_FIGURE
        elif combined < threshold / 2:
            classification = ImageClassification.LIKELY_DECORATIVE
        else:
            classification = ImageClassification.UNCERTAIN

        return ImageScore(
            aspect_ratio_score=round(aspect, 4),
            entropy_score=round(entropy, 4),
            edge_density_score=round(edge, 4),
            position_score=round(position, 4),
            size_score=round(size, 4),
            combined_score=round(combined, 4),
            classification=classification,
        )

    def classify_batch(
        self,
        images: list[dict[str, Any]],
    ) -> list[ImageScore]:
        """Classifica un batch di immagini con logging riassuntivo.

        Args:
            images: Lista di dict, ognuno con almeno le chiavi:
                ``image_bytes``, ``page_number``, ``total_pages``,
                ``page_width``, ``page_height``.

        Returns:
            Lista di :class:`ImageScore`, uno per ogni immagine.
        """
        scores: list[ImageScore] = []
        for img_info in images:
            score = self.classify(
                image_bytes=img_info["image_bytes"],
                page_number=img_info["page_number"],
                total_pages=img_info["total_pages"],
                page_width=img_info["page_width"],
                page_height=img_info["page_height"],
            )
            scores.append(score)

        # --- Logging riassuntivo ---
        if scores:
            n_scientific = sum(
                1 for s in scores if s.classification == ImageClassification.SCIENTIFIC_FIGURE
            )
            n_decorative = sum(
                1 for s in scores if s.classification == ImageClassification.LIKELY_DECORATIVE
            )
            n_uncertain = sum(
                1 for s in scores if s.classification == ImageClassification.UNCERTAIN
            )
            avg_score = sum(s.combined_score for s in scores) / len(scores)

            logger.info(
                "Filtro euristico batch: %d immagini → "
                "SCIENTIFIC_FIGURE=%d, LIKELY_DECORATIVE=%d, UNCERTAIN=%d  "
                "(score medio: %.3f)",
                len(scores), n_scientific, n_decorative, n_uncertain, avg_score,
            )

            # Log tabella Rich se disponibile
            try:
                from rich.console import Console
                from rich.table import Table

                table = Table(title="Image Filtering — Stage 1 (Heuristic)")
                table.add_column("Classification", style="bold")
                table.add_column("Count", justify="right")
                table.add_column("Avg Score", justify="right")

                for label, cls_enum in [
                    ("Scientific Figure", ImageClassification.SCIENTIFIC_FIGURE),
                    ("Likely Decorative", ImageClassification.LIKELY_DECORATIVE),
                    ("Uncertain", ImageClassification.UNCERTAIN),
                ]:
                    subset = [s for s in scores if s.classification == cls_enum]
                    count = len(subset)
                    avg = sum(s.combined_score for s in subset) / count if count else 0.0
                    table.add_row(label, str(count), f"{avg:.3f}")

                Console(stderr=True).print(table)
            except ImportError:
                pass  # Rich non obbligatorio

        return scores

    # ------------------------------------------------------------------
    # Euristiche private
    # ------------------------------------------------------------------

    def _score_aspect_ratio(self, width: int, height: int) -> float:
        """Punteggio [0, 1] per aspect ratio.

        Grafici e tabelle hanno tipicamente ratio 0.5–2.5.
        Penalizza ratio estremi (banner orizzontali, barre verticali).
        """
        if height == 0:
            return 0.0
        ratio = width / height

        min_r = self._config.min_aspect_ratio
        max_r = self._config.max_aspect_ratio

        # Fuori range configurato → score 0
        if ratio < min_r or ratio > max_r:
            return 0.0

        # Range ideale: 0.5–2.5 → score pieno
        ideal_min, ideal_max = 0.5, 2.5
        if ideal_min <= ratio <= ideal_max:
            return 1.0

        # Zona di transizione: score lineare tra bordo e range ideale
        if ratio < ideal_min:
            return (ratio - min_r) / (ideal_min - min_r) if ideal_min > min_r else 0.0
        else:  # ratio > ideal_max
            return (max_r - ratio) / (max_r - ideal_max) if max_r > ideal_max else 0.0

    def _score_color_entropy(self, img: Image.Image) -> float:
        """Punteggio [0, 1] per entropia colore Shannon.

        Immagini con poca variazione cromatica (loghi monocromo, icone)
        hanno entropia bassa. Grafici e figure complesse hanno entropia alta.
        Scala: entropia massima teorica per 8-bit = 8.0.
        """
        # Converti a RGB se necessario
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        histogram = img.histogram()
        total_pixels = sum(histogram)
        if total_pixels == 0:
            return 0.0

        # Calcolo entropia Shannon
        entropy = 0.0
        for count in histogram:
            if count > 0:
                p = count / total_pixels
                entropy -= p * math.log2(p)

        # Normalizza: l'entropia massima per un istogramma RGB 256×3=768 bin
        # è log2(768) ≈ 9.58, ma realisticamente 0–8 per la maggior parte.
        # Usiamo la soglia configurata come punto di cut-off.
        max_entropy = 8.0  # Massima pratica per 8-bit per canale
        normalized = min(entropy / max_entropy, 1.0)

        # Penalizza sotto la soglia minima configurata
        min_threshold = self._config.min_color_entropy / max_entropy
        if normalized < min_threshold:
            # Score proporzionale: 0 se entropia=0, min_threshold se alla soglia
            return normalized / min_threshold * 0.3 if min_threshold > 0 else 0.0

        return normalized

    def _score_edge_density(self, img: Image.Image) -> float:
        """Punteggio [0, 1] per densità di bordi.

        Grafici, tabelle e diagrammi hanno alta densità di bordi.
        Foto e loghi lisci hanno densità bassa.
        Usa il filtro FIND_EDGES di PIL.
        """
        # Converti a greyscale
        grey = img.convert("L")

        # Applica filtro bordi
        edges = grey.filter(ImageFilter.FIND_EDGES)

        # Conta pixel bordo (> soglia)
        edge_arr = np.array(edges, dtype=np.float32)
        total_pixels = edge_arr.size
        if total_pixels == 0:
            return 0.0

        # Pixel con valore > 30 sono considerati bordi significativi
        edge_pixels = np.count_nonzero(edge_arr > 30)
        density = edge_pixels / total_pixels

        # Penalizza sotto la soglia minima configurata
        min_density = self._config.min_edge_density
        if density < min_density:
            return density / min_density * 0.3 if min_density > 0 else 0.0

        # Normalizza: densità > 0.4 è rara, clampiamo a 1.0
        return min(density / 0.4, 1.0)

    def _score_page_position(self, page_number: int, total_pages: int) -> float:
        """Punteggio [0, 1] per posizione nel documento.

        Prima pagina → probabilmente logo/header → penalizzata.
        Ultima pagina → probabilmente copyright/footer → penalizzata.
        Pagine centrali → score pieno.
        """
        if total_pages <= 0:
            return 0.5

        # Pagina 1: penalizzazione se configurata
        if page_number == 1 and self._config.skip_first_page:
            return 0.3

        # Ultima pagina: penalizzazione se configurata
        if page_number == total_pages and self._config.skip_last_page:
            return 0.3

        # Pagina 2 e penultima: leggera penalizzazione
        if page_number == 2 or page_number == total_pages - 1:
            return 0.8

        # Pagine centrali: score pieno
        return 1.0

    def _score_relative_size(
        self,
        img_width: int,
        img_height: int,
        page_width: float,
        page_height: float,
    ) -> float:
        """Punteggio [0, 1] per dimensione relativa alla pagina.

        Troppo piccola (< 5% pagina) → icona, ornamento.
        Troppo grande (> 90% pagina) → background full-page.
        Range ideale: 10–70% dell'area della pagina.
        """
        page_area = page_width * page_height
        if page_area <= 0:
            return 0.5

        img_area = img_width * img_height
        ratio = img_area / page_area

        # Troppo piccola
        if ratio < 0.05:
            return ratio / 0.05 * 0.2  # 0.0–0.2

        # Troppo grande
        if ratio > 0.90:
            return max(0.0, (1.0 - ratio) / 0.10 * 0.3)  # 0.3–0.0

        # Range ideale: 10–70%
        if 0.10 <= ratio <= 0.70:
            return 1.0

        # Transizioni
        if ratio < 0.10:
            return 0.2 + (ratio - 0.05) / 0.05 * 0.8  # 0.2–1.0
        else:  # 0.70 < ratio <= 0.90
            return 1.0 - (ratio - 0.70) / 0.20 * 0.7  # 1.0–0.3
