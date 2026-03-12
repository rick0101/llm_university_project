"""
Figure Extractor — Estrazione e descrizione intelligente delle figure dai PDF.

Estrae le immagini dai PDF scientifici usando PyMuPDF, poi genera
descrizioni strutturate tramite Gemini Flash (multimodale) per ogni figura.
Le descrizioni vengono convertite in embedding testuali e indicizzate
in Qdrant per il retrieval.

Workflow:
    PDF → PyMuPDF (estrazione immagini) → Gemini Flash Vision (descrizione)
        → Testo strutturato → Embedding testuale → Qdrant (figure_descriptions)

L'approccio "intelligente" sfrutta la capacità multimodale di Gemini:
- Identifica il tipo di figura (grafico, diagramma, tabella, architettura, ecc.)
- Descrive il contenuto semantico (non solo OCR)
- Estrae dati quantitativi da grafici e tabelle
- Collega la figura al contesto del paper (tramite caption GROBID)

Uso::

    from paperhunter.ingestion.figure_extractor import FigureExtractor

    extractor = FigureExtractor()
    descriptions = extractor.extract_and_describe(
        pdf_path="paper.pdf",
        paper=paper_obj,
        figure_captions=grobid_output.figure_captions,
    )
"""

from __future__ import annotations

import base64
import hashlib
import io
import time
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
from google import genai
from google.genai import types

from paperhunter.config import settings
from paperhunter.ingestion.image_relevance_filter import (
    ImageClassification,
    ImageRelevanceFilter,
)
from paperhunter.models import ChunkType, PaperChunk
from paperhunter.utils.logging_setup import get_logger
from paperhunter.utils.rate_limiter import get_gemini_limiter

logger = get_logger(__name__)

# Dimensioni minime per considerare un'immagine significativa (pixel)
_MIN_IMAGE_WIDTH = 150
_MIN_IMAGE_HEIGHT = 150
_MIN_IMAGE_AREA = 40_000  # Esclude icone e loghi piccoli

# Formati immagine supportati da Gemini Vision
_SUPPORTED_FORMATS = {"png", "jpeg", "jpg", "webp"}

# Prompt per la descrizione multimodale delle figure
_FIGURE_DESCRIPTION_PROMPT = """\
You are analyzing a figure extracted from a scientific paper about LLM security.

{caption_context}

Analyze this figure and provide a structured description in the following format:

**Figure Type**: (one of: results_plot, architecture_diagram, flowchart, comparison_table, \
confusion_matrix, attack_diagram, defense_pipeline, data_distribution, timeline, other)

**Content Description**: A detailed 2-4 sentence description of what the figure shows. \
Focus on the semantic meaning, not visual appearance. If it's a plot, describe the axes, \
trends, and key data points. If it's a diagram, describe the components and their relationships.

**Key Findings**: 1-3 bullet points of quantitative or qualitative insights visible in the figure. \
Include specific numbers, percentages, or trends if visible.

**Relevance**: One sentence explaining how this figure relates to the paper's contribution \
in the context of LLM security research.

Be precise and factual. Only describe what you can actually see in the figure.
"""


class FigureExtractor:
    """Estrattore e descrittore intelligente di figure dai PDF scientifici.

    Combina PyMuPDF per l'estrazione delle immagini con Gemini Flash Vision
    per la generazione di descrizioni semantiche strutturate.

    Args:
        min_width: Larghezza minima per considerare un'immagine (pixel).
        min_height: Altezza minima per considerare un'immagine (pixel).
        max_figures: Numero massimo di figure da processare per paper.
    """

    def __init__(
        self,
        min_width: int | None = None,
        min_height: int | None = None,
        max_figures: int | None = None,
    ):
        cfg = settings.figure_extraction
        self.min_width = min_width or cfg.min_width
        self.min_height = min_height or cfg.min_height
        self.max_figures = max_figures or cfg.max_figures_per_paper
        self._llm = None

    def _get_llm(self):
        """Ottieni l'istanza LLM per la descrizione delle figure (lazy init)."""
        if self._llm is None:
            from paperhunter.utils.llm_factory import get_flash_llm
            self._llm = get_flash_llm(temperature=0.1)
        return self._llm

    # ------------------------------------------------------------------
    # Estrazione immagini da PDF
    # ------------------------------------------------------------------

    def extract_images_from_pdf(
        self,
        pdf_path: str | Path,
    ) -> list[dict[str, Any]]:
        """Estrae le immagini significative da un PDF usando PyMuPDF.

        Filtra immagini troppo piccole (icone, loghi, watermark) e
        restituisce solo figure scientifiche rilevanti.

        Args:
            pdf_path: Percorso al file PDF.

        Returns:
            Lista di dict con chiavi:
            - ``image_bytes``: bytes dell'immagine (PNG)
            - ``page_number``: pagina di origine (1-based)
            - ``width``, ``height``: dimensioni in pixel
            - ``image_index``: indice progressivo nel PDF
            - ``mime_type``: tipo MIME dell'immagine
            - ``total_pages``: numero totale di pagine nel PDF
            - ``page_width``: larghezza pagina in punti PDF
            - ``page_height``: altezza pagina in punti PDF
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trovato: {pdf_path}")

        images: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()  # Deduplica immagini identiche

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            logger.warning("Impossibile aprire PDF '%s' per estrazione figure: %s", pdf_path.name, e)
            return []

        try:
            n_pages = len(doc)
            for page_idx in range(n_pages):
                page = doc[page_idx]
                page_rect = page.rect
                p_width = page_rect.width
                p_height = page_rect.height
                image_list = page.get_images(full=True)

                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]

                    try:
                        pix = fitz.Pixmap(doc, xref)
                    except Exception:
                        continue

                    # Converti CMYK → RGB se necessario
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    width = pix.width
                    height = pix.height
                    area = width * height

                    # Filtra immagini troppo piccole
                    if width < self.min_width or height < self.min_height:
                        continue
                    if area < _MIN_IMAGE_AREA:
                        continue

                    # Converti a PNG bytes
                    try:
                        png_bytes = pix.tobytes("png")
                    except Exception:
                        continue

                    # Deduplica per hash
                    img_hash = hashlib.sha256(png_bytes).hexdigest()[:16]
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    images.append({
                        "image_bytes": png_bytes,
                        "page_number": page_idx + 1,
                        "width": width,
                        "height": height,
                        "image_index": len(images),
                        "mime_type": "image/png",
                        "image_hash": img_hash,
                        "total_pages": n_pages,
                        "page_width": p_width,
                        "page_height": p_height,
                    })

                    # Limite massimo raggiunto
                    if len(images) >= self.max_figures:
                        logger.info(
                            "Raggiunto limite massimo figure (%d) per '%s'",
                            self.max_figures, pdf_path.name,
                        )
                        return images
        finally:
            doc.close()

        logger.info(
            "Estratte %d figure da '%s' (%d pagine)",
            len(images), pdf_path.name, n_pages,
        )
        return images

    # ------------------------------------------------------------------
    # Descrizione multimodale con Gemini Vision
    # ------------------------------------------------------------------

    # Dimensione massima per invio a Gemini Vision (pixel per lato)
    _MAX_VISION_DIMENSION = 512

    def _resize_if_needed(self, image_bytes: bytes, mime_type: str) -> bytes:
        """Ridimensiona l'immagine se supera _MAX_VISION_DIMENSION."""
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        if max(w, h) <= self._MAX_VISION_DIMENSION:
            return image_bytes

        ratio = self._MAX_VISION_DIMENSION / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        fmt = "PNG" if "png" in mime_type else "JPEG"
        img.save(buf, format=fmt)
        logger.debug("Immagine ridimensionata %dx%d → %dx%d", w, h, new_w, new_h)
        return buf.getvalue()

    def describe_figure(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        caption: str | None = None,
    ) -> str:
        """Genera una descrizione strutturata di una figura usando Gemini Vision.

        Args:
            image_bytes: Bytes dell'immagine (PNG/JPEG).
            mime_type: Tipo MIME dell'immagine.
            caption: Caption GROBID opzionale per contesto aggiuntivo.

        Returns:
            Descrizione testuale strutturata della figura.
        """
        caption_context = ""
        if caption:
            caption_context = f"The paper's caption for this figure is: \"{caption}\"\n\n"

        prompt = _FIGURE_DESCRIPTION_PROMPT.format(caption_context=caption_context)

        # Ridimensiona immagini grandi per evitare timeout API
        resized_bytes = self._resize_if_needed(image_bytes, mime_type)

        # Costruisci il contenuto multimodale
        image_part = types.Part.from_bytes(
            data=resized_bytes,
            mime_type=mime_type,
        )

        try:
            limiter = get_gemini_limiter()
            limiter.wait()

            client = genai.Client(api_key=settings.google_api_key)
            response = client.models.generate_content(
                model=settings.figure_extraction.vision_model,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("Gemini Vision ha restituito risposta vuota")
                return ""

        except Exception as e:
            logger.warning("Errore Gemini Vision per descrizione figura: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Pipeline completa: estrazione + descrizione + chunking
    # ------------------------------------------------------------------

    def extract_and_describe(
        self,
        pdf_path: str | Path,
        paper: Any,
        figure_captions: list[str] | None = None,
    ) -> list[PaperChunk]:
        """Pipeline completa: estrae figure dal PDF e genera chunk descrittivi.

        Per ogni figura:
        1. Estrae l'immagine con PyMuPDF
        2. Salva l'immagine su disco in figures_dir
        3. Associa la caption GROBID più vicina (per posizione)
        4. Genera una descrizione con Gemini Vision
        5. Crea un PaperChunk con la descrizione e il path dell'immagine

        Args:
            pdf_path: Percorso al file PDF.
            paper: Oggetto Paper con metadati.
            figure_captions: Caption GROBID estratte (opzionale).

        Returns:
            Lista di PaperChunk con le descrizioni delle figure.
        """
        pdf_path = Path(pdf_path)

        # Step 1: Estrai immagini
        images = self.extract_images_from_pdf(pdf_path)
        if not images:
            logger.debug("Nessuna figura significativa in '%s'", pdf_path.name)
            return []

        total_extracted = len(images)

        # Step 1.5: Filtro euristico (Stage 1) — se abilitato
        if settings.image_filtering.enabled:
            relevance_filter = ImageRelevanceFilter()
            scores = relevance_filter.classify_batch(images)

            filtered_images: list[dict[str, Any]] = []
            for img_info, score in zip(images, scores):
                img_info["relevance_score"] = score.combined_score
                if score.classification == ImageClassification.LIKELY_DECORATIVE:
                    # Calcola il motivo principale del filtraggio per logging
                    reasons = []
                    if score.edge_density_score < 0.3:
                        reasons.append("bassa densità bordi")
                    if score.entropy_score < 0.3:
                        reasons.append("bassa entropia")
                    if score.position_score < 0.5:
                        reasons.append("prima/ultima pagina")
                    if score.aspect_ratio_score < 0.3:
                        reasons.append("aspect ratio estremo")
                    if score.size_score < 0.3:
                        reasons.append("dimensione inadeguata")
                    reason_str = ", ".join(reasons) if reasons else "score combinato basso"
                    logger.debug(
                        "Filtrata immagine %d (score %.2f): %s",
                        img_info["image_index"], score.combined_score, reason_str,
                    )
                else:
                    filtered_images.append(img_info)

            n_filtered = total_extracted - len(filtered_images)
            logger.info(
                "Filtro euristico '%s': Estratte=%d, Filtrate(decorative)=%d, "
                "Passate a Vision=%d",
                pdf_path.name, total_extracted, n_filtered, len(filtered_images),
            )
            images = filtered_images
        else:
            # Se filtro disabilitato, aggiungi relevance_score None
            for img_info in images:
                img_info["relevance_score"] = None

        if not images:
            logger.debug(
                "Tutte le immagini di '%s' filtrate come decorative", pdf_path.name,
            )
            return []

        # Prepara captions (per matching posizionale)
        captions = figure_captions or []
        chunks: list[PaperChunk] = []
        bibtex_key = paper.bibtex_key or paper.title[:30]
        paper_id = paper.semantic_scholar_id or paper.arxiv_id or bibtex_key

        # Crea directory per le immagini
        figures_dir = Path(settings.figure_extraction.figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Descrizione di %d figure per '%s'...",
            len(images), pdf_path.name,
        )

        for img_info in images:
            idx = img_info["image_index"]

            # Associa caption per indice (euristica: stessa posizione)
            caption = captions[idx] if idx < len(captions) else None

            # Salva immagine su disco
            # Sanitize bibtex_key per nome file sicuro
            safe_key = "".join(c if c.isalnum() or c in "_-" else "_" for c in bibtex_key)
            img_filename = f"{safe_key}_fig_{idx + 1:03d}.png"
            img_path = figures_dir / img_filename
            try:
                img_path.write_bytes(img_info["image_bytes"])
            except Exception as e:
                logger.warning("Impossibile salvare immagine %s: %s", img_filename, e)
                continue

            # Path relativo per LaTeX (da output/ verso output/figures/)
            image_rel_path = f"figures/{img_filename}"

            # Step 2: Genera descrizione multimodale
            description = self.describe_figure(
                image_bytes=img_info["image_bytes"],
                mime_type=img_info["mime_type"],
                caption=caption,
            )

            if not description or len(description) < 50:
                logger.debug(
                    "Descrizione troppo corta per figura %d di '%s', skip",
                    idx, pdf_path.name,
                )
                continue

            # Step 3: Costruisci testo arricchito per il chunk
            enriched_text = self._build_enriched_text(
                description=description,
                caption=caption,
                page_number=img_info["page_number"],
                figure_index=idx + 1,
            )

            # Step 4: Genera PaperChunk
            chunk_id = hashlib.sha256(
                f"{bibtex_key}::figure::{idx}".encode()
            ).hexdigest()[:16]

            chunk = PaperChunk(
                chunk_id=chunk_id,
                text=enriched_text,
                chunk_type=ChunkType.ORIGINAL,
                paper_id=paper_id,
                bibtex_key=bibtex_key,
                section_name="Figure_Description",
                category=paper.category or "",
                year=paper.year,
                month=paper.month,
                chunk_index=idx,
                image_path=image_rel_path,
            )
            chunks.append(chunk)

        logger.info(
            "Generate %d descrizioni figure per '%s' [%s]",
            len(chunks), pdf_path.name, bibtex_key,
        )
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_enriched_text(
        description: str,
        caption: str | None,
        page_number: int,
        figure_index: int,
    ) -> str:
        """Costruisce il testo arricchito per un chunk di descrizione figura.

        Combina la descrizione Gemini Vision con la caption GROBID
        e metadata posizionali in un formato strutturato.

        Args:
            description: Descrizione generata da Gemini Vision.
            caption: Caption GROBID opzionale.
            page_number: Numero di pagina (1-based).
            figure_index: Indice della figura (1-based).

        Returns:
            Testo strutturato per l'indicizzazione.
        """
        parts = [f"[Figure {figure_index}, page {page_number}]"]

        if caption:
            parts.append(f"Caption: {caption}")

        parts.append(f"Description: {description}")

        return "\n\n".join(parts)
