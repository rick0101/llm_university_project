"""
Test per Task 2.3 — Indicizzazione diretta immagini come vettori multimodali in Qdrant.

Verifica che:
- IMAGE_EMBEDDING sia presente nell'enum ChunkType
- La collection image_embeddings venga creata in QdrantStore
- add_image_embeddings() indicizzi correttamente nella collection dedicata
- query_image_embeddings() cerchi nella collection image_embeddings
- get_stats() includa la collection image_embeddings
- FigureExtractor.get_pending_image_embeddings() restituisca chunk e embedding
- I chunk IMAGE_EMBEDDING riusino gli embedding pre-calcolati dallo Stage 2
- Le immagini che saltano Stage 2 (score > 0.8) vengano ri-embedded
- get_pending_image_embeddings() svuoti i dati pendenti dopo il consumo
- L'ingestion pipeline indicizzi i chunk IMAGE_EMBEDDING nella collection dedicata
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from paperhunter.ingestion.image_relevance_filter import (
    ImageClassification,
    ImageScore,
)
from paperhunter.models import ChunkType, Paper, PaperChunk

pytestmark = pytest.mark.skip(reason="Visual and Gemini features are disabled in the DeepSeek-only profile.")


# ============================================================================
# Helpers
# ============================================================================


def _make_png(img: Image.Image) -> bytes:
    """Converte un'immagine PIL in bytes PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_chart_png(width: int = 400, height: int = 300) -> bytes:
    """Genera un PNG tipo grafico scientifico."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.line([(40, 40), (40, height - 40)], fill="black", width=2)
    draw.line([(40, height - 40), (width - 40, height - 40)], fill="black", width=2)
    rng = np.random.RandomState(42)
    for color in ["red", "blue"]:
        points = []
        y_base = rng.randint(60, 200)
        for i in range(15):
            x = 40 + int(i * (width - 80) / 14)
            y = max(40, min(height - 40, y_base + int(rng.normal(0, 10))))
            points.append((x, y))
            y_base += rng.randint(-3, 4)
        for j in range(len(points) - 1):
            draw.line([points[j], points[j + 1]], fill=color, width=2)
    return _make_png(img)


def _make_photo_png(width: int = 300, height: int = 250) -> bytes:
    """Genera un PNG tipo foto."""
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


@pytest.fixture
def qdrant_temp_dir():
    """Directory temporanea per Qdrant."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ============================================================================
# Test: ChunkType.IMAGE_EMBEDDING
# ============================================================================


class TestChunkTypeImageEmbedding:
    """Test per il nuovo valore IMAGE_EMBEDDING nell'enum ChunkType."""

    def test_image_embedding_exists(self):
        """ChunkType.IMAGE_EMBEDDING esiste."""
        assert hasattr(ChunkType, "IMAGE_EMBEDDING")
        assert ChunkType.IMAGE_EMBEDDING.value == "image_embedding"

    def test_image_embedding_in_chunk(self):
        """PaperChunk può usare chunk_type=IMAGE_EMBEDDING."""
        chunk = PaperChunk(
            chunk_id="test_img_emb_001",
            text="Figure 1 from TestPaper",
            chunk_type=ChunkType.IMAGE_EMBEDDING,
            paper_id="test_paper",
            bibtex_key="Test_2024",
            section_name="Image_Embedding",
            category="Adversarial Attacks",
            year=2024,
            image_path="figures/test_fig_001.png",
        )
        assert chunk.chunk_type == ChunkType.IMAGE_EMBEDDING
        assert chunk.image_path == "figures/test_fig_001.png"

    def test_image_embedding_metadata_dict(self):
        """to_metadata_dict() serializza correttamente IMAGE_EMBEDDING."""
        chunk = PaperChunk(
            chunk_id="test_img_emb_002",
            text="Figure 2",
            chunk_type=ChunkType.IMAGE_EMBEDDING,
        )
        meta = chunk.to_metadata_dict()
        assert meta["chunk_type"] == "image_embedding"


# ============================================================================
# Test: QdrantStore collection image_embeddings
# ============================================================================


class TestQdrantStoreImageEmbeddings:
    """Test per la collection image_embeddings in QdrantStore."""

    @patch("paperhunter.indexing.qdrant_store.QdrantStore._get_embedding_model")
    def test_collection_created(self, mock_emb, qdrant_temp_dir):
        """La collection image_embeddings viene creata all'init."""
        from paperhunter.indexing.qdrant_store import QdrantStore

        store = QdrantStore(persist_dir=qdrant_temp_dir)
        assert store.client.collection_exists(store.COLLECTION_IMAGE_EMBEDDINGS)
        store.close()

    @patch("paperhunter.indexing.qdrant_store.QdrantStore._get_embedding_model")
    def test_collection_name_from_config(self, mock_emb, qdrant_temp_dir):
        """Il nome della collection proviene dalla config."""
        from paperhunter.config import settings
        from paperhunter.indexing.qdrant_store import QdrantStore

        store = QdrantStore(persist_dir=qdrant_temp_dir)
        assert store.COLLECTION_IMAGE_EMBEDDINGS == settings.qdrant.collection_image_embeddings
        info = store.client.get_collection(store.COLLECTION_IMAGE_EMBEDDINGS)
        assert info.config.params.vectors.size == settings.qdrant.image_embedding_dimension
        store.close()

    @patch("paperhunter.indexing.qdrant_store.QdrantStore._get_embedding_model")
    def test_add_image_embeddings(self, mock_emb, qdrant_temp_dir):
        """add_image_embeddings() indicizza nella collection dedicata."""
        from paperhunter.indexing.qdrant_store import QdrantStore

        store = QdrantStore(persist_dir=qdrant_temp_dir)

        chunks = [
            PaperChunk(
                chunk_id=f"img_emb_{i}",
                text=f"Figure {i + 1} description",
                chunk_type=ChunkType.IMAGE_EMBEDDING,
                paper_id="paper_123",
                bibtex_key="Test_2024",
                category="Adversarial Attacks",
                year=2024,
                image_path=f"figures/test_fig_{i + 1:03d}.png",
            )
            for i in range(3)
        ]
        embeddings = [_fake_embedding(seed=i) for i in range(3)]

        n = store.add_image_embeddings(chunks, embeddings)
        assert n == 3

        # Verifica che siano nella collection giusta
        count = store.get_collection_count(store.COLLECTION_IMAGE_EMBEDDINGS)
        assert count == 3

        # Non nella collection figure_descriptions
        count_figs = store.get_collection_count(store.COLLECTION_FIGURES)
        assert count_figs == 0

        store.close()

    @patch("paperhunter.indexing.qdrant_store.QdrantStore._get_embedding_model")
    def test_add_image_embeddings_empty(self, mock_emb, qdrant_temp_dir):
        """add_image_embeddings() con lista vuota restituisce 0."""
        from paperhunter.indexing.qdrant_store import QdrantStore

        store = QdrantStore(persist_dir=qdrant_temp_dir)
        n = store.add_image_embeddings([], [])
        assert n == 0
        store.close()

    def test_query_image_embeddings(self, qdrant_temp_dir):
        """query_image_embeddings() cerca nella collection image_embeddings."""
        from paperhunter.indexing.qdrant_store import QdrantStore

        mock_model = MagicMock()
        mock_model.get_query_embedding.return_value = _fake_embedding(seed=42)
        mock_model.embed_batch_size = 10

        store = QdrantStore(persist_dir=qdrant_temp_dir, embedding_model=mock_model)

        # Indicizza
        chunks = [
            PaperChunk(
                chunk_id="img_q_1",
                text="Attack success rate comparison plot",
                chunk_type=ChunkType.IMAGE_EMBEDDING,
                paper_id="paper_abc",
                bibtex_key="Alice_2024",
                category="Adversarial Attacks",
                year=2024,
                image_path="figures/alice_fig_001.png",
            ),
        ]
        emb = [_fake_embedding(seed=42)]
        store.add_image_embeddings(chunks, emb)

        with patch("paperhunter.utils.multimodal_embedder.get_multimodal_embedder") as mock_get_embedder:
            mock_get_embedder.return_value.embed_text.return_value = _fake_embedding(seed=42)
            results = store.query_image_embeddings(
                query_text="attack success rate",
                n_results=5,
            )
        assert len(results) >= 1
        assert results[0]["text"] == "Attack success rate comparison plot"
        assert results[0]["metadata"]["image_path"] == "figures/alice_fig_001.png"
        assert results[0]["metadata"]["chunk_type"] == "image_embedding"

        store.close()

    @patch("paperhunter.indexing.qdrant_store.QdrantStore._get_embedding_model")
    def test_query_image_embeddings_with_filter(self, mock_emb, qdrant_temp_dir):
        """query_image_embeddings() supporta filtri metadata."""
        from paperhunter.indexing.qdrant_store import QdrantStore

        store = QdrantStore(persist_dir=qdrant_temp_dir)

        chunks = [
            PaperChunk(
                chunk_id="filt_1",
                text="Figure 1",
                chunk_type=ChunkType.IMAGE_EMBEDDING,
                category="Prompt Injection",
                year=2024,
            ),
            PaperChunk(
                chunk_id="filt_2",
                text="Figure 2",
                chunk_type=ChunkType.IMAGE_EMBEDDING,
                category="Adversarial Attacks",
                year=2023,
            ),
        ]
        embeddings = [_fake_embedding(seed=100), _fake_embedding(seed=200)]
        store.add_image_embeddings(chunks, embeddings)

        mock_model = MagicMock()
        mock_model.get_query_embedding.return_value = _fake_embedding(seed=100)
        store._embedding_model = mock_model
        store._embedding_model_initialized = True

        with patch("paperhunter.utils.multimodal_embedder.get_multimodal_embedder") as mock_get_embedder:
            mock_get_embedder.return_value.embed_text.return_value = _fake_embedding(seed=100)
            results = store.query_image_embeddings(
                query_text="test",
                n_results=10,
                where_filter={"category": "Prompt Injection"},
            )
        assert all(
            r["metadata"]["category"] == "Prompt Injection" for r in results
        )

        store.close()

    @patch("paperhunter.indexing.qdrant_store.QdrantStore._get_embedding_model")
    def test_get_stats_includes_image_embeddings(self, mock_emb, qdrant_temp_dir):
        """get_stats() include la collection image_embeddings."""
        from paperhunter.indexing.qdrant_store import QdrantStore

        store = QdrantStore(persist_dir=qdrant_temp_dir)

        # Indicizza un punto
        chunks = [
            PaperChunk(
                chunk_id="stats_1",
                text="Figure stats",
                chunk_type=ChunkType.IMAGE_EMBEDDING,
                category="Adversarial Attacks",
                year=2024,
            ),
        ]
        store.add_image_embeddings(chunks, [_fake_embedding(seed=1)])

        stats = store.get_stats()
        assert store.COLLECTION_IMAGE_EMBEDDINGS in stats
        img_stats = stats[store.COLLECTION_IMAGE_EMBEDDINGS]
        assert img_stats["total_points"] == 1
        assert img_stats["exists"] is True

        store.close()

    def test_same_paper_id_in_both_collections(self, qdrant_temp_dir):
        """Lo stesso paper produce entry sia in figure_descriptions che image_embeddings."""
        from paperhunter.indexing.qdrant_store import QdrantStore

        mock_model = MagicMock()
        mock_model.get_text_embedding_batch.return_value = [_fake_embedding(dim=1024, seed=10)]
        mock_model.embed_batch_size = 10

        store = QdrantStore(persist_dir=qdrant_temp_dir, embedding_model=mock_model)

        # Chunk testuale (figure_descriptions)
        text_chunk = PaperChunk(
            chunk_id="fig_desc_1",
            text="Description of Figure 1",
            chunk_type=ChunkType.ORIGINAL,
            paper_id="paper_xyz",
            bibtex_key="Author_2024",
        )
        store.add_chunks([text_chunk], collection_name=store.COLLECTION_FIGURES)

        # Chunk image embedding
        img_chunk = PaperChunk(
            chunk_id="img_emb_1",
            text="Figure 1 from Author_2024",
            chunk_type=ChunkType.IMAGE_EMBEDDING,
            paper_id="paper_xyz",
            bibtex_key="Author_2024",
        )
        store.add_image_embeddings([img_chunk], [_fake_embedding(seed=20)])

        # Verifica entrambe le collection
        assert store.get_collection_count(store.COLLECTION_FIGURES) == 1
        assert store.get_collection_count(store.COLLECTION_IMAGE_EMBEDDINGS) == 1

        # Verifica paper_id in entrambe
        assert store.has_paper_id("paper_xyz", store.COLLECTION_FIGURES)
        assert store.has_paper_id("paper_xyz", store.COLLECTION_IMAGE_EMBEDDINGS)

        store.close()


# ============================================================================
# Test: FigureExtractor.get_pending_image_embeddings()
# ============================================================================


class TestGetPendingImageEmbeddings:
    """Test per la preparazione e il consumo dei chunk IMAGE_EMBEDDING."""

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_image_chunks_prepared_with_precomputed_embeddings(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """I chunk IMAGE_EMBEDDING riusano gli embedding pre-calcolati dallo Stage 2."""
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

        # Stage 1: UNCERTAIN → passa a Stage 2
        uncertain_score = ImageScore(
            combined_score=0.45,
            classification=ImageClassification.UNCERTAIN,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [uncertain_score]
        mock_filter_cls.return_value = mock_filter

        # Stage 2: alta similarità → passa
        ref_embs = _fake_ref_embeddings()
        precomputed_emb = _fake_embedding(seed=10)  # uguale a scientific_plot
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = precomputed_emb
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Verifica che i chunk testuali siano stati creati
        assert len(chunks) == 1

        # Verifica chunk IMAGE_EMBEDDING pendenti
        img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
        assert len(img_chunks) == 1
        assert len(img_embeddings) == 1
        assert img_chunks[0].chunk_type == ChunkType.IMAGE_EMBEDDING
        assert img_chunks[0].section_name == "Image_Embedding"
        assert img_chunks[0].bibtex_key == "Smith_2024_Adversarial"
        assert img_chunks[0].image_path == "figures/Smith_2024_Adversarial_fig_001.png"
        # L'embedding è quello pre-calcolato (stesso seed=10)
        assert img_embeddings[0] == precomputed_emb
        # embed_image è stato chiamato una sola volta (Stage 2), non due
        assert mock_embedder.embed_image.call_count == 1

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_high_score_images_reembedded(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Immagini con score > 0.8 (Stage 2 skippato) vengono ri-embedded."""
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

        # Stage 1: SCIENTIFIC con score > 0.8 → salta Stage 2
        sci_score = ImageScore(
            combined_score=0.85,
            classification=ImageClassification.SCIENTIFIC_FIGURE,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [sci_score]
        mock_filter_cls.return_value = mock_filter

        # Stage 2: l'embedder viene usato per reference +  Step 5 (ri-embed)
        ref_embs = _fake_ref_embeddings()
        new_emb = _fake_embedding(seed=77)
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = new_emb
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
        assert len(img_chunks) == 1
        assert len(img_embeddings) == 1
        # L'embedding è quello calcolato in Step 5 (non pre-calcolato)
        assert img_embeddings[0] == new_emb
        # embed_image chiamato una volta (Step 5: ri-embed per immagine skip Stage 2)
        assert mock_embedder.embed_image.call_count == 1

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_pending_cleared_after_get(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """get_pending_image_embeddings() svuota i dati dopo il consumo."""
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

        uncertain_score = ImageScore(
            combined_score=0.45,
            classification=ImageClassification.UNCERTAIN,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [uncertain_score]
        mock_filter_cls.return_value = mock_filter

        ref_embs = _fake_ref_embeddings()
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = _fake_embedding(seed=10)
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        # Prima chiamata: dati presenti
        c1, e1 = extractor.get_pending_image_embeddings()
        assert len(c1) == 1
        assert len(e1) == 1

        # Seconda chiamata: svuotato
        c2, e2 = extractor.get_pending_image_embeddings()
        assert len(c2) == 0
        assert len(e2) == 0

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_no_pending_when_embedding_disabled(
        self, mock_settings, sample_paper, tmp_path,
    ):
        """Nessun chunk IMAGE_EMBEDDING se embedding_filter_enabled=False."""
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

        with patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter") as mock_filt_cls:
            score = ImageScore(
                combined_score=0.6,
                classification=ImageClassification.SCIENTIFIC_FIGURE,
            )
            mock_filt_cls.return_value.classify_batch.return_value = [score]

            with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
                 patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

                extractor.extract_and_describe(
                    pdf_path="/fake/paper.pdf",
                    paper=sample_paper,
                )

        img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
        assert len(img_chunks) == 0
        assert len(img_embeddings) == 0

    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_no_pending_when_filtering_disabled(
        self, mock_settings, sample_paper, tmp_path,
    ):
        """Nessun chunk IMAGE_EMBEDDING se image_filtering.enabled=False."""
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
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
        assert len(img_chunks) == 0
        assert len(img_embeddings) == 0

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_multiple_images_mixed(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Mix di immagini: pre-computed e ri-embedded producono chunk corretti."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart1 = _make_chart_png(400, 300)
        chart2 = _make_chart_png(350, 250)
        fake_images = [
            _build_image_dict(chart1, image_index=0, image_hash="c1"),
            _build_image_dict(chart2, image_index=1, image_hash="c2"),
        ]

        # Stage 1: immagine 0 → SCIENTIFIC (score 0.85, skip Stage 2)
        #           immagine 1 → UNCERTAIN (score 0.45, va a Stage 2)
        scores = [
            ImageScore(
                combined_score=0.85,
                classification=ImageClassification.SCIENTIFIC_FIGURE,
            ),
            ImageScore(
                combined_score=0.45,
                classification=ImageClassification.UNCERTAIN,
            ),
        ]
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = scores
        mock_filter_cls.return_value = mock_filter

        # Stage 2: per immagine 1 (UNCERTAIN), alta sim → passa
        ref_embs = _fake_ref_embeddings()
        emb_uncertain = _fake_embedding(seed=10)  # uguale a scientific_plot
        emb_recomputed = _fake_embedding(seed=99)  # ri-embed per immagine 0

        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        # Stage 2 chiama embed_image per immagine 1 (UNCERTAIN)
        # Step 5 chiama embed_image per immagine 0 (skip Stage 2, embedding=None)
        mock_embedder.embed_image.side_effect = [emb_uncertain, emb_recomputed]
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        assert len(chunks) == 2  # Due descrizioni testuali

        img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
        assert len(img_chunks) == 2
        assert len(img_embeddings) == 2

        # Chunk 0: ri-embedded (era skip Stage 2, embedding=None)
        assert img_chunks[0].chunk_index == 0
        assert img_chunks[0].chunk_type == ChunkType.IMAGE_EMBEDDING
        # Chunk 1: pre-computed (aveva embedding da Stage 2)
        assert img_chunks[1].chunk_index == 1

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_chunk_text_uses_caption(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Il testo del chunk IMAGE_EMBEDDING usa la caption se disponibile."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png()
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="ch"),
        ]

        uncertain_score = ImageScore(
            combined_score=0.45,
            classification=ImageClassification.UNCERTAIN,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [uncertain_score]
        mock_filter_cls.return_value = mock_filter

        ref_embs = _fake_ref_embeddings()
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = _fake_embedding(seed=10)
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
                figure_captions=["Attack success rates across different LLMs"],
            )

        img_chunks, _ = extractor.get_pending_image_embeddings()
        assert img_chunks[0].text == "Attack success rates across different LLMs"

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_chunk_text_fallback_no_caption(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Senza caption, il testo del chunk è un fallback con bibtex_key."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png()
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="ch"),
        ]

        uncertain_score = ImageScore(
            combined_score=0.45,
            classification=ImageClassification.UNCERTAIN,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [uncertain_score]
        mock_filter_cls.return_value = mock_filter

        ref_embs = _fake_ref_embeddings()
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = _fake_embedding(seed=10)
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
                figure_captions=[],
            )

        img_chunks, _ = extractor.get_pending_image_embeddings()
        assert "Smith_2024_Adversarial" in img_chunks[0].text

    def test_get_pending_before_extract(self):
        """get_pending_image_embeddings() su estractor vergine restituisce vuoto."""
        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor()
        img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
        assert img_chunks == []
        assert img_embeddings == []


# ============================================================================
# Test: Chunk IMAGE_EMBEDDING — metadata
# ============================================================================


class TestImageEmbeddingChunkMetadata:
    """Test che i chunk IMAGE_EMBEDDING abbiano metadata corretti."""

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_metadata_fields(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """I chunk IMAGE_EMBEDDING hanno tutti i metadata necessari."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png()
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="ch"),
        ]

        uncertain_score = ImageScore(
            combined_score=0.45,
            classification=ImageClassification.UNCERTAIN,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [uncertain_score]
        mock_filter_cls.return_value = mock_filter

        ref_embs = _fake_ref_embeddings()
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = _fake_embedding(seed=10)
        mock_get_embedder.return_value = mock_embedder

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):

            extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
        chunk = img_chunks[0]

        assert chunk.chunk_type == ChunkType.IMAGE_EMBEDDING
        assert chunk.paper_id == "2024.12345"  # arxiv_id
        assert chunk.bibtex_key == "Smith_2024_Adversarial"
        assert chunk.section_name == "Image_Embedding"
        assert chunk.category == "Adversarial Attacks"
        assert chunk.year == 2024
        assert chunk.month == 6
        assert chunk.chunk_index == 0
        assert chunk.image_path == "figures/Smith_2024_Adversarial_fig_001.png"
        assert chunk.chunk_id is not None and len(chunk.chunk_id) == 16

        # Embedding è 3072-dim
        assert len(img_embeddings[0]) == 3072

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_chunk_id_deterministic(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """chunk_id per IMAGE_EMBEDDING è deterministico (bibtex_key + index)."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        chart_bytes = _make_chart_png()
        fake_images = [
            _build_image_dict(chart_bytes, image_index=0, image_hash="ch"),
        ]

        score = ImageScore(
            combined_score=0.45,
            classification=ImageClassification.UNCERTAIN,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [score]
        mock_filter_cls.return_value = mock_filter

        ref_embs = _fake_ref_embeddings()
        mock_embedder = MagicMock()
        mock_embedder.compute_reference_embeddings.return_value = ref_embs
        mock_embedder.embed_image.return_value = _fake_embedding(seed=10)
        mock_get_embedder.return_value = mock_embedder

        # Esegui due volte e verifica che chunk_id sia uguale
        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):
            extractor.extract_and_describe("/fake/paper.pdf", sample_paper)
        c1, _ = extractor.get_pending_image_embeddings()

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images), \
             patch.object(extractor, "describe_figure", return_value=MOCK_DESCRIPTION):
            extractor.extract_and_describe("/fake/paper.pdf", sample_paper)
        c2, _ = extractor.get_pending_image_embeddings()

        assert c1[0].chunk_id == c2[0].chunk_id


# ============================================================================
# Test: No empty chunks when all images filtered
# ============================================================================


class TestEdgeCases:
    """Test per casi limite."""

    @patch("paperhunter.ingestion.figure_extractor.get_multimodal_embedder")
    @patch("paperhunter.ingestion.figure_extractor.ImageRelevanceFilter")
    @patch("paperhunter.ingestion.figure_extractor.settings")
    def test_all_filtered_no_image_chunks(
        self, mock_settings, mock_filter_cls, mock_get_embedder,
        sample_paper, tmp_path,
    ):
        """Se tutte le immagini sono filtrate, nessun chunk IMAGE_EMBEDDING."""
        mock_settings.figure_extraction.min_width = 50
        mock_settings.figure_extraction.min_height = 50
        mock_settings.figure_extraction.max_figures_per_paper = 10
        mock_settings.figure_extraction.figures_dir = str(tmp_path / "figures")
        mock_settings.image_filtering.enabled = True
        mock_settings.image_filtering.embedding_filter_enabled = True
        mock_settings.image_filtering.embedding_similarity_threshold = 0.3

        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor(min_width=50, min_height=50)

        fake_images = [
            _build_image_dict(_make_photo_png(), image_index=0, image_hash="ph"),
        ]

        # Stage 1: LIKELY_DECORATIVE → tutte filtrate
        deco_score = ImageScore(
            combined_score=0.15,
            classification=ImageClassification.LIKELY_DECORATIVE,
        )
        mock_filter = MagicMock()
        mock_filter.classify_batch.return_value = [deco_score]
        mock_filter_cls.return_value = mock_filter

        with patch.object(extractor, "extract_images_from_pdf", return_value=fake_images):
            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        assert chunks == []
        img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
        assert img_chunks == []
        assert img_embeddings == []

    def test_no_images_extracted(self, sample_paper, tmp_path):
        """Se nessuna immagine è estratta, nessun chunk pendente."""
        from paperhunter.ingestion.figure_extractor import FigureExtractor

        extractor = FigureExtractor()

        with patch.object(extractor, "extract_images_from_pdf", return_value=[]):
            chunks = extractor.extract_and_describe(
                pdf_path="/fake/paper.pdf",
                paper=sample_paper,
            )

        assert chunks == []
        img_chunks, img_embeddings = extractor.get_pending_image_embeddings()
        assert img_chunks == []
        assert img_embeddings == []
