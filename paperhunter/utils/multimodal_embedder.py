"""
Multimodal Embedder — Wrapper Gemini Embedding 2 Preview per immagini.

Genera embedding multimodali (immagini, testo, coppie) usando il SDK
``google-genai`` diretto, bypassando LlamaIndex che supporta solo input
testuali.

Funzionalità principali:
- Embedding di singole immagini (PNG, JPEG, WebP)
- Embedding di testo (per embedding di riferimento)
- Batch processing con progress bar Rich
- Embedding di riferimento pre-calcolati per categorie scientifiche
- Classificazione rilevanza immagini tramite similarità coseno

Uso::

    from paperhunter.utils.multimodal_embedder import (
        MultimodalEmbedder,
        get_multimodal_embedder,
    )

    embedder = get_multimodal_embedder()
    vec = embedder.embed_image(image_bytes, mime_type="image/png")
    sim, cat = embedder.classify_image_relevance(image_bytes, "image/png")

Nota: Gemini Embedding 2 Preview produce vettori 3072-dimensionali nello
stesso spazio per testo e immagini, abilitando retrieval cross-modale.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from paperhunter.utils.logging_setup import get_logger
from paperhunter.utils.rate_limiter import get_gemini_limiter

logger = get_logger(__name__)

# Dimensione vettore per Gemini Embedding 2 Preview
_EMBEDDING_DIM = 3072

# Categorie di riferimento per figure scientifiche
_REFERENCE_DESCRIPTIONS: dict[str, str] = {
    "scientific_plot": (
        "scientific data plot with labeled axes showing experimental results"
    ),
    "comparison_table": (
        "comparison table with rows and columns of numerical data"
    ),
    "architecture_diagram": (
        "system architecture diagram showing components and connections"
    ),
    "flowchart": (
        "flowchart or pipeline diagram showing process steps and decisions"
    ),
    "confusion_matrix": (
        "confusion matrix or heatmap showing classification results"
    ),
}

# Path cache per embedding di riferimento
_REFERENCE_CACHE_PATH = Path("cache/reference_embeddings.json")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcola la similarità coseno tra due vettori.

    Args:
        a: Primo vettore.
        b: Secondo vettore.

    Returns:
        Similarità coseno in [-1, 1].
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class MultimodalEmbedder:
    """Wrapper per Gemini Embedding 2 Preview in modalità multimodale.

    Genera embedding da immagini e testo usando il SDK ``google-genai``
    diretto (non LlamaIndex, che supporta solo input testuali).

    Args:
        model: Nome del modello di embedding. Se None, usa
            ``settings.gemini.embedding_model``.
    """

    def __init__(self, model: str | None = None):
        from paperhunter.config import settings

        self._model = model or settings.gemini.embedding_model
        self._client = self._create_client()
        self._limiter = get_gemini_limiter()
        self._reference_embeddings: dict[str, list[float]] | None = None

        logger.debug(f"MultimodalEmbedder inizializzato con modello: {self._model}")

    def _create_client(self) -> genai.Client:
        """Crea il client google-genai con API key e http options."""
        from paperhunter.utils.llm_factory import _get_api_key, _get_http_options

        return genai.Client(
            api_key=_get_api_key(),
            http_options=_get_http_options(),
        )

    # ------------------------------------------------------------------
    # Embedding singoli
    # ------------------------------------------------------------------

    def embed_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> list[float]:
        """Genera embedding multimodale da un'immagine.

        Args:
            image_bytes: Bytes dell'immagine.
            mime_type: Tipo MIME (``image/png``, ``image/jpeg``, ``image/webp``).

        Returns:
            Vettore 3072-dimensionale.
        """
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        self._limiter.wait()
        result = self._client.models.embed_content(
            model=self._model,
            contents=[image_part],
        )
        return list(result.embeddings[0].values)

    def embed_text(self, text: str) -> list[float]:
        """Genera embedding testuale tramite lo stesso modello multimodale.

        Usa lo stesso spazio vettoriale delle immagini, abilitando
        il confronto cross-modale (testo ↔ immagine).

        Args:
            text: Testo da embeddare.

        Returns:
            Vettore 3072-dimensionale.
        """
        self._limiter.wait()
        result = self._client.models.embed_content(
            model=self._model,
            contents=[text],
        )
        return list(result.embeddings[0].values)

    def embed_pdf_page(
        self,
        pdf_bytes: bytes,
    ) -> list[float]:
        """Genera embedding da una singola pagina PDF.

        Passa il PDF direttamente al modello multimodale, che ne
        comprende testo, layout, formule, tabelle e figure.

        Args:
            pdf_bytes: Bytes di un PDF (tipicamente una singola pagina).

        Returns:
            Vettore 3072-dimensionale.
        """
        pdf_part = types.Part.from_bytes(
            data=pdf_bytes, mime_type="application/pdf"
        )
        self._limiter.wait()
        result = self._client.models.embed_content(
            model=self._model,
            contents=[pdf_part],
        )
        return list(result.embeddings[0].values)

    def embed_pdf_pages(
        self,
        pdf_path: str | Path,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Genera embedding per ogni pagina di un PDF.

        Splitta il PDF in pagine singole e genera un embedding
        multimodale per ciascuna, catturando testo + layout + figure.

        Args:
            pdf_path: Percorso al file PDF.
            show_progress: Se True, mostra progress bar Rich.

        Returns:
            Lista di vettori 3072-dim (uno per pagina, in ordine).
        """
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)
        embeddings: list[list[float]] = []

        iterator = range(n_pages)
        if show_progress and n_pages > 1:
            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]PDF page embeddings ({n_pages}p)...",
                    total=n_pages,
                )
                for page_num in range(n_pages):
                    page_pdf = fitz.open()
                    page_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    emb = self.embed_pdf_page(page_pdf.tobytes())
                    embeddings.append(emb)
                    page_pdf.close()
                    progress.update(task, advance=1)
        else:
            for page_num in range(n_pages):
                page_pdf = fitz.open()
                page_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                emb = self.embed_pdf_page(page_pdf.tobytes())
                embeddings.append(emb)
                page_pdf.close()

        doc.close()
        logger.info(
            f"PDF page embeddings: {len(embeddings)} pagine da {Path(pdf_path).name}"
        )
        return embeddings

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def embed_images_batch(
        self,
        images: list[dict],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Embedding batch di immagini con progress bar Rich.

        Args:
            images: Lista di dict con chiavi ``image_bytes`` e ``mime_type``.
            show_progress: Se True, mostra progress bar Rich.

        Returns:
            Lista di vettori 3072-dimensionali (uno per immagine).
        """
        if not images:
            return []

        embeddings: list[list[float]] = []

        if show_progress:
            from rich.progress import Progress

            with Progress() as progress:
                task = progress.add_task(
                    "[cyan]Embedding immagini...",
                    total=len(images),
                )
                for img_info in images:
                    emb = self.embed_image(
                        image_bytes=img_info["image_bytes"],
                        mime_type=img_info.get("mime_type", "image/png"),
                    )
                    embeddings.append(emb)
                    progress.update(task, advance=1)
        else:
            for img_info in images:
                emb = self.embed_image(
                    image_bytes=img_info["image_bytes"],
                    mime_type=img_info.get("mime_type", "image/png"),
                )
                embeddings.append(emb)

        logger.info(
            f"Embedding batch completato: {len(embeddings)} immagini processate"
        )
        return embeddings

    # ------------------------------------------------------------------
    # Embedding di riferimento per classificazione
    # ------------------------------------------------------------------

    def compute_reference_embeddings(self) -> dict[str, list[float]]:
        """Calcola e cacha embedding di riferimento per categorie scientifiche.

        Le categorie definite sono:
        - ``scientific_plot``: Grafico con assi e risultati sperimentali
        - ``comparison_table``: Tabella comparativa con dati numerici
        - ``architecture_diagram``: Diagramma architetturale
        - ``flowchart``: Diagramma di flusso o pipeline
        - ``confusion_matrix``: Matrice di confusione o heatmap

        Gli embedding vengono cachati su disco in
        ``cache/reference_embeddings.json`` per evitare ricalcolo.

        Returns:
            Dict categoria → vettore 3072-dimensionale.
        """
        # Controlla cache in memoria
        if self._reference_embeddings is not None:
            return self._reference_embeddings

        # Controlla cache su disco
        if _REFERENCE_CACHE_PATH.exists():
            try:
                data = json.loads(_REFERENCE_CACHE_PATH.read_text(encoding="utf-8"))
                # Verifica che tutte le categorie siano presenti
                if all(cat in data for cat in _REFERENCE_DESCRIPTIONS):
                    self._reference_embeddings = data
                    logger.info(
                        f"Caricati {len(data)} embedding di riferimento da cache"
                    )
                    return self._reference_embeddings
            except (json.JSONDecodeError, KeyError):
                logger.warning(
                    "Cache embedding di riferimento corrotta, ricalcolo..."
                )

        # Calcola embedding per ogni categoria
        logger.info(
            f"Calcolo embedding di riferimento per {len(_REFERENCE_DESCRIPTIONS)} "
            "categorie scientifiche..."
        )
        ref_embeddings: dict[str, list[float]] = {}
        for category, description in _REFERENCE_DESCRIPTIONS.items():
            emb = self.embed_text(description)
            ref_embeddings[category] = emb
            logger.debug(f"  Embedding calcolato per '{category}' (dim={len(emb)})")

        # Salva cache su disco
        _REFERENCE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REFERENCE_CACHE_PATH.write_text(
            json.dumps(ref_embeddings), encoding="utf-8"
        )
        logger.info(
            f"Embedding di riferimento salvati in cache: {_REFERENCE_CACHE_PATH}"
        )

        self._reference_embeddings = ref_embeddings
        return self._reference_embeddings

    # ------------------------------------------------------------------
    # Classificazione rilevanza immagine
    # ------------------------------------------------------------------

    def classify_image_relevance(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        reference_embeddings: dict[str, list[float]] | None = None,
    ) -> tuple[float, str]:
        """Classifica la rilevanza di un'immagine via similarità embedding.

        Embeds l'immagine e calcola la similarità coseno con gli embedding
        di riferimento per categorie di figure scientifiche. Se la similarità
        massima è sotto la soglia configurata, l'immagine è considerata
        non rilevante.

        Args:
            image_bytes: Bytes dell'immagine.
            mime_type: Tipo MIME dell'immagine.
            reference_embeddings: Embedding di riferimento pre-calcolati.
                Se None, usa ``compute_reference_embeddings()``.

        Returns:
            Tupla ``(max_similarity, best_matching_category)``. Se nessun
            riferimento è disponibile, restituisce ``(0.0, "unknown")``.
        """
        if reference_embeddings is None:
            reference_embeddings = self.compute_reference_embeddings()

        if not reference_embeddings:
            return 0.0, "unknown"

        # Embed dell'immagine
        image_embedding = self.embed_image(image_bytes, mime_type)

        # Calcola similarità con tutti i riferimenti
        best_similarity = -1.0
        best_category = "unknown"

        for category, ref_emb in reference_embeddings.items():
            sim = _cosine_similarity(image_embedding, ref_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_category = category

        logger.debug(
            f"Classificazione immagine: sim={best_similarity:.4f}, "
            f"categoria='{best_category}'"
        )
        return best_similarity, best_category


# ---------------------------------------------------------------------------
# Singleton lazy
# ---------------------------------------------------------------------------

_instance: MultimodalEmbedder | None = None


def get_multimodal_embedder() -> MultimodalEmbedder:
    """Ottieni l'istanza singleton del MultimodalEmbedder.

    Lazy initialization: l'istanza viene creata al primo accesso.

    Returns:
        Istanza condivisa di MultimodalEmbedder.
    """
    global _instance
    if _instance is None:
        _instance = MultimodalEmbedder()
    return _instance
