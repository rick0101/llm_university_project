"""
Test per MultimodalEmbedder — Wrapper Gemini Embedding 2 Preview multimodale.

Verifica:
- Embedding di immagini sintetiche (grafico matplotlib, rettangolo logo)
- Embedding di testo
- Batch processing
- Cache embedding di riferimento su disco
- Classificazione rilevanza per similarità coseno
- Singleton lazy get_multimodal_embedder()
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image, ImageDraw


# ============================================================================
# Helper: generatori di immagini sintetiche
# ============================================================================


def _make_png(img: Image.Image) -> bytes:
    """Converte un'immagine PIL in bytes PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_chart_image(width: int = 400, height: int = 300) -> bytes:
    """Genera un'immagine sintetica tipo 'grafico scientifico' (assi, linee, griglia)."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    margin = 40

    # Assi
    draw.line([(margin, margin), (margin, height - margin)], fill="black", width=2)
    draw.line([(margin, height - margin), (width - margin, height - margin)], fill="black", width=2)

    # Griglia
    for i in range(5):
        y = margin + i * (height - 2 * margin) // 4
        draw.line([(margin, y), (width - margin, y)], fill="lightgray", width=1)
        x = margin + i * (width - 2 * margin) // 4
        draw.line([(x, margin), (x, height - margin)], fill="lightgray", width=1)

    # Linea dati (rossa)
    rng = np.random.RandomState(42)
    points = []
    for i in range(20):
        px = margin + i * (width - 2 * margin) // 19
        py = height - margin - int(rng.uniform(20, height - 2 * margin))
        points.append((px, py))
    draw.line(points, fill="red", width=2)

    # Linea dati (blu)
    points2 = []
    for i in range(20):
        px = margin + i * (width - 2 * margin) // 19
        py = height - margin - int(rng.uniform(10, height - 2 * margin - 20))
        points2.append((px, py))
    draw.line(points2, fill="blue", width=2)

    return _make_png(img)


def make_logo_image(width: int = 60, height: int = 60) -> bytes:
    """Genera un'immagine tipo 'logo' (piccola, pochi colori)."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 50, 50], fill="darkblue")
    draw.text((15, 20), "AB", fill="white")
    return _make_png(img)


def make_photo_image(width: int = 400, height: int = 300) -> bytes:
    """Genera un'immagine sintetica tipo 'foto' (rumore colorato, pochi bordi netti)."""
    rng = np.random.RandomState(123)
    arr = rng.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Sfumatura morbida (no bordi netti)
    from PIL import ImageFilter
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return _make_png(img)


# ============================================================================
# Helper: generazione embedding finti (deterministici per test)
# ============================================================================

def _make_fake_embedding(seed: int = 0, dim: int = 3072) -> list[float]:
    """Genera un embedding finto deterministico e normalizzato."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(float)
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _make_similar_embedding(base: list[float], noise_scale: float = 0.1, seed: int = 99) -> list[float]:
    """Genera un embedding simile a `base` con rumore controllato."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(base)) * noise_scale
    vec = [b + n for b, n in zip(base, noise)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _make_orthogonal_embedding(base: list[float], seed: int = 77) -> list[float]:
    """Genera un embedding quasi-ortogonale rispetto a `base`."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(len(base))
    # Rimuovi la componente nella direzione di base
    dot = sum(v * b for v, b in zip(vec, base))
    vec = [v - dot * b for v, b in zip(vec, base)]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return _make_fake_embedding(seed + 1)
    return [x / norm for x in vec]


# ============================================================================
# Fixture: mock settings
# ============================================================================

@pytest.fixture
def mock_settings():
    """Mock minimo delle settings per evitare di caricare config.yaml."""
    with patch("paperhunter.utils.multimodal_embedder.get_gemini_limiter") as mock_limiter:
        mock_limiter_instance = MagicMock()
        mock_limiter_instance.wait.return_value = 0.0
        mock_limiter.return_value = mock_limiter_instance
        yield mock_limiter_instance


@pytest.fixture
def mock_embedder(mock_settings):
    """Crea un MultimodalEmbedder con client mockato."""
    with patch("paperhunter.utils.multimodal_embedder.MultimodalEmbedder._create_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        with patch("paperhunter.config.settings") as mock_cfg:
            mock_cfg.gemini.embedding_model = "gemini-embedding-2-preview"

            from paperhunter.utils.multimodal_embedder import MultimodalEmbedder
            embedder = MultimodalEmbedder(model="gemini-embedding-2-preview")
            embedder._client = mock_client

            yield embedder, mock_client


# ============================================================================
# Test: embed_image
# ============================================================================


class TestEmbedImage:
    """Test per embedding di singole immagini."""

    def test_embed_image_returns_3072_vector(self, mock_embedder):
        """embed_image deve restituire un vettore 3072-dim."""
        embedder, mock_client = mock_embedder
        fake_emb = _make_fake_embedding(seed=1)

        mock_result = MagicMock()
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = fake_emb
        mock_result.embeddings = [mock_emb_obj]
        mock_client.models.embed_content.return_value = mock_result

        chart_bytes = make_chart_image()
        result = embedder.embed_image(chart_bytes, mime_type="image/png")

        assert len(result) == 3072
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
        mock_client.models.embed_content.assert_called_once()

    def test_embed_image_calls_rate_limiter(self, mock_embedder):
        """embed_image deve chiamare il rate limiter prima dell'API call."""
        embedder, mock_client = mock_embedder
        fake_emb = _make_fake_embedding(seed=2)

        mock_result = MagicMock()
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = fake_emb
        mock_result.embeddings = [mock_emb_obj]
        mock_client.models.embed_content.return_value = mock_result

        embedder.embed_image(make_chart_image())
        embedder._limiter.wait.assert_called()

    def test_embed_image_passes_correct_mime_type(self, mock_embedder):
        """embed_image deve passare il mime_type corretto all'SDK."""
        embedder, mock_client = mock_embedder
        fake_emb = _make_fake_embedding(seed=3)

        mock_result = MagicMock()
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = fake_emb
        mock_result.embeddings = [mock_emb_obj]
        mock_client.models.embed_content.return_value = mock_result

        embedder.embed_image(make_logo_image(), mime_type="image/jpeg")

        call_args = mock_client.models.embed_content.call_args
        assert call_args.kwargs["model"] == "gemini-embedding-2-preview"


# ============================================================================
# Test: embed_text
# ============================================================================


class TestEmbedText:
    """Test per embedding testuali."""

    def test_embed_text_returns_3072_vector(self, mock_embedder):
        """embed_text deve restituire un vettore 3072-dim."""
        embedder, mock_client = mock_embedder
        fake_emb = _make_fake_embedding(seed=10)

        mock_result = MagicMock()
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = fake_emb
        mock_result.embeddings = [mock_emb_obj]
        mock_client.models.embed_content.return_value = mock_result

        result = embedder.embed_text("scientific data plot with labeled axes")

        assert len(result) == 3072
        assert isinstance(result, list)
        mock_client.models.embed_content.assert_called_once()

    def test_embed_text_passes_string_content(self, mock_embedder):
        """embed_text deve passare il testo come contenuto."""
        embedder, mock_client = mock_embedder
        fake_emb = _make_fake_embedding(seed=11)

        mock_result = MagicMock()
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = fake_emb
        mock_result.embeddings = [mock_emb_obj]
        mock_client.models.embed_content.return_value = mock_result

        embedder.embed_text("test text")

        call_args = mock_client.models.embed_content.call_args
        assert call_args.kwargs["contents"] == ["test text"]


# ============================================================================
# Test: embed_images_batch
# ============================================================================


class TestBatchEmbedding:
    """Test per batch processing di immagini."""

    def test_batch_empty_list(self, mock_embedder):
        """embed_images_batch su lista vuota deve restituire lista vuota."""
        embedder, _ = mock_embedder
        result = embedder.embed_images_batch([])
        assert result == []

    def test_batch_returns_correct_count(self, mock_embedder):
        """embed_images_batch deve restituire un embedding per ogni immagine."""
        embedder, mock_client = mock_embedder

        call_count = [0]
        def side_effect(**kwargs):
            seed = 20 + call_count[0]
            call_count[0] += 1
            mock_result = MagicMock()
            mock_emb_obj = MagicMock()
            mock_emb_obj.values = _make_fake_embedding(seed=seed)
            mock_result.embeddings = [mock_emb_obj]
            return mock_result

        mock_client.models.embed_content.side_effect = side_effect

        images = [
            {"image_bytes": make_chart_image(), "mime_type": "image/png"},
            {"image_bytes": make_logo_image(), "mime_type": "image/png"},
            {"image_bytes": make_photo_image(), "mime_type": "image/png"},
        ]

        result = embedder.embed_images_batch(images, show_progress=False)

        assert len(result) == 3
        assert all(len(v) == 3072 for v in result)
        assert mock_client.models.embed_content.call_count == 3

    def test_batch_with_progress(self, mock_embedder):
        """embed_images_batch con show_progress=True non deve crashare."""
        embedder, mock_client = mock_embedder

        call_count = [0]
        def side_effect(**kwargs):
            seed = 30 + call_count[0]
            call_count[0] += 1
            mock_result = MagicMock()
            mock_emb_obj = MagicMock()
            mock_emb_obj.values = _make_fake_embedding(seed=seed)
            mock_result.embeddings = [mock_emb_obj]
            return mock_result

        mock_client.models.embed_content.side_effect = side_effect

        images = [
            {"image_bytes": make_chart_image()},
            {"image_bytes": make_logo_image()},
        ]

        result = embedder.embed_images_batch(images, show_progress=True)
        assert len(result) == 2

    def test_batch_default_mime_type(self, mock_embedder):
        """Se mime_type non è specificato nel dict, deve usare image/png."""
        embedder, mock_client = mock_embedder

        mock_result = MagicMock()
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = _make_fake_embedding(seed=40)
        mock_result.embeddings = [mock_emb_obj]
        mock_client.models.embed_content.return_value = mock_result

        images = [{"image_bytes": make_chart_image()}]  # No mime_type
        result = embedder.embed_images_batch(images, show_progress=False)
        assert len(result) == 1


# ============================================================================
# Test: compute_reference_embeddings
# ============================================================================


class TestReferenceEmbeddings:
    """Test per calcolo e cache degli embedding di riferimento."""

    def test_compute_reference_embeddings_returns_all_categories(self, mock_embedder, tmp_path):
        """compute_reference_embeddings deve restituire tutte le 5 categorie."""
        embedder, mock_client = mock_embedder

        call_count = [0]
        def side_effect(**kwargs):
            seed = 50 + call_count[0]
            call_count[0] += 1
            mock_result = MagicMock()
            mock_emb_obj = MagicMock()
            mock_emb_obj.values = _make_fake_embedding(seed=seed)
            mock_result.embeddings = [mock_emb_obj]
            return mock_result

        mock_client.models.embed_content.side_effect = side_effect

        cache_path = tmp_path / "reference_embeddings.json"
        with patch(
            "paperhunter.utils.multimodal_embedder._REFERENCE_CACHE_PATH",
            cache_path,
        ):
            embedder._reference_embeddings = None  # Reset cache memoria
            result = embedder.compute_reference_embeddings()

        expected_categories = {
            "scientific_plot", "comparison_table",
            "architecture_diagram", "flowchart", "confusion_matrix"
        }
        assert set(result.keys()) == expected_categories
        assert all(len(v) == 3072 for v in result.values())
        assert mock_client.models.embed_content.call_count == 5

    def test_reference_embeddings_cached_in_memory(self, mock_embedder, tmp_path):
        """Dopo il primo calcolo, i reference embeddings devono essere in cache."""
        embedder, mock_client = mock_embedder

        call_count = [0]
        def side_effect(**kwargs):
            seed = 60 + call_count[0]
            call_count[0] += 1
            mock_result = MagicMock()
            mock_emb_obj = MagicMock()
            mock_emb_obj.values = _make_fake_embedding(seed=seed)
            mock_result.embeddings = [mock_emb_obj]
            return mock_result

        mock_client.models.embed_content.side_effect = side_effect

        cache_path = tmp_path / "reference_embeddings.json"
        with patch(
            "paperhunter.utils.multimodal_embedder._REFERENCE_CACHE_PATH",
            cache_path,
        ):
            embedder._reference_embeddings = None  # Reset cache memoria
            result1 = embedder.compute_reference_embeddings()
            result2 = embedder.compute_reference_embeddings()

        # Seconda chiamata non deve fare API calls aggiuntive
        assert mock_client.models.embed_content.call_count == 5  # Solo la prima
        assert result1 is result2

    def test_reference_embeddings_cache_to_disk(self, mock_embedder, tmp_path):
        """compute_reference_embeddings deve salvare cache su disco."""
        embedder, mock_client = mock_embedder

        cache_path = tmp_path / "reference_embeddings.json"
        call_count = [0]
        def side_effect(**kwargs):
            seed = 70 + call_count[0]
            call_count[0] += 1
            mock_result = MagicMock()
            mock_emb_obj = MagicMock()
            mock_emb_obj.values = _make_fake_embedding(seed=seed)
            mock_result.embeddings = [mock_emb_obj]
            return mock_result

        mock_client.models.embed_content.side_effect = side_effect

        # Patch del path della cache
        with patch(
            "paperhunter.utils.multimodal_embedder._REFERENCE_CACHE_PATH",
            cache_path,
        ):
            embedder._reference_embeddings = None  # Reset cache memoria
            result = embedder.compute_reference_embeddings()

        assert cache_path.exists()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert set(data.keys()) == set(result.keys())
        assert all(len(v) == 3072 for v in data.values())

    def test_reference_embeddings_load_from_disk(self, mock_embedder, tmp_path):
        """Se cache su disco esiste, compute_reference_embeddings non deve chiamare API."""
        embedder, mock_client = mock_embedder

        cache_path = tmp_path / "reference_embeddings.json"
        # Crea cache finta su disco
        from paperhunter.utils.multimodal_embedder import _REFERENCE_DESCRIPTIONS
        fake_cache = {
            cat: _make_fake_embedding(seed=80 + i)
            for i, cat in enumerate(_REFERENCE_DESCRIPTIONS.keys())
        }
        cache_path.write_text(json.dumps(fake_cache), encoding="utf-8")

        with patch(
            "paperhunter.utils.multimodal_embedder._REFERENCE_CACHE_PATH",
            cache_path,
        ):
            embedder._reference_embeddings = None  # Reset cache memoria
            result = embedder.compute_reference_embeddings()

        # Nessuna API call: tutto da cache disco
        mock_client.models.embed_content.assert_not_called()
        assert set(result.keys()) == set(fake_cache.keys())

    def test_corrupted_cache_triggers_recalculation(self, mock_embedder, tmp_path):
        """Cache disco corrotta deve triggare ricalcolo."""
        embedder, mock_client = mock_embedder

        cache_path = tmp_path / "reference_embeddings.json"
        cache_path.write_text("invalid json{{{", encoding="utf-8")

        call_count = [0]
        def side_effect(**kwargs):
            seed = 90 + call_count[0]
            call_count[0] += 1
            mock_result = MagicMock()
            mock_emb_obj = MagicMock()
            mock_emb_obj.values = _make_fake_embedding(seed=seed)
            mock_result.embeddings = [mock_emb_obj]
            return mock_result

        mock_client.models.embed_content.side_effect = side_effect

        with patch(
            "paperhunter.utils.multimodal_embedder._REFERENCE_CACHE_PATH",
            cache_path,
        ):
            embedder._reference_embeddings = None
            result = embedder.compute_reference_embeddings()

        # Deve aver ricalcolato via API
        assert mock_client.models.embed_content.call_count == 5
        assert len(result) == 5


# ============================================================================
# Test: classify_image_relevance
# ============================================================================


class TestClassifyImageRelevance:
    """Test per classificazione rilevanza immagini via embedding."""

    def test_chart_more_similar_to_scientific_plot(self, mock_embedder):
        """Un grafico deve avere similarità più alta con 'scientific_plot' che un logo."""
        embedder, mock_client = mock_embedder

        # Embedding di riferimento: una per scientific_plot, una per altri
        ref_scientific = _make_fake_embedding(seed=100)
        ref_table = _make_fake_embedding(seed=101)
        ref_arch = _make_fake_embedding(seed=102)
        ref_flow = _make_fake_embedding(seed=103)
        ref_matrix = _make_fake_embedding(seed=104)

        reference_embeddings = {
            "scientific_plot": ref_scientific,
            "comparison_table": ref_table,
            "architecture_diagram": ref_arch,
            "flowchart": ref_flow,
            "confusion_matrix": ref_matrix,
        }

        # Grafico → simile a scientific_plot
        chart_emb = _make_similar_embedding(ref_scientific, noise_scale=0.1, seed=200)
        # Logo → ortogonale a tutto
        logo_emb = _make_orthogonal_embedding(ref_scientific, seed=201)

        embed_call_count = [0]
        def side_effect(**kwargs):
            # Prima chiamata: chart, seconda: logo
            if embed_call_count[0] == 0:
                emb = chart_emb
            else:
                emb = logo_emb
            embed_call_count[0] += 1
            mock_result = MagicMock()
            mock_emb_obj = MagicMock()
            mock_emb_obj.values = emb
            mock_result.embeddings = [mock_emb_obj]
            return mock_result

        mock_client.models.embed_content.side_effect = side_effect

        sim_chart, cat_chart = embedder.classify_image_relevance(
            make_chart_image(), "image/png", reference_embeddings
        )
        sim_logo, cat_logo = embedder.classify_image_relevance(
            make_logo_image(), "image/png", reference_embeddings
        )

        assert sim_chart > sim_logo, (
            f"Chart sim ({sim_chart:.4f}) should be higher than logo sim ({sim_logo:.4f})"
        )
        assert cat_chart == "scientific_plot"

    def test_classify_returns_tuple(self, mock_embedder):
        """classify_image_relevance deve restituire (float, str)."""
        embedder, mock_client = mock_embedder

        fake_emb = _make_fake_embedding(seed=110)
        mock_result = MagicMock()
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = fake_emb
        mock_result.embeddings = [mock_emb_obj]
        mock_client.models.embed_content.return_value = mock_result

        ref_embeddings = {
            "scientific_plot": _make_fake_embedding(seed=111),
            "comparison_table": _make_fake_embedding(seed=112),
        }

        sim, cat = embedder.classify_image_relevance(
            make_chart_image(), "image/png", ref_embeddings
        )

        assert isinstance(sim, float)
        assert isinstance(cat, str)
        assert cat in ref_embeddings

    def test_classify_empty_references(self, mock_embedder):
        """Con reference_embeddings vuoti, deve restituire (0.0, 'unknown')."""
        embedder, _ = mock_embedder

        sim, cat = embedder.classify_image_relevance(
            make_chart_image(), "image/png", reference_embeddings={}
        )

        assert sim == 0.0
        assert cat == "unknown"

    def test_classify_uses_compute_reference_if_none(self, mock_embedder, tmp_path):
        """Se reference_embeddings è None, deve chiamare compute_reference_embeddings."""
        embedder, mock_client = mock_embedder

        call_count = [0]
        def side_effect(**kwargs):
            seed = 120 + call_count[0]
            call_count[0] += 1
            mock_result = MagicMock()
            mock_emb_obj = MagicMock()
            mock_emb_obj.values = _make_fake_embedding(seed=seed)
            mock_result.embeddings = [mock_emb_obj]
            return mock_result

        mock_client.models.embed_content.side_effect = side_effect

        cache_path = tmp_path / "reference_embeddings.json"
        with patch(
            "paperhunter.utils.multimodal_embedder._REFERENCE_CACHE_PATH",
            cache_path,
        ):
            embedder._reference_embeddings = None  # Reset cache memoria
            sim, cat = embedder.classify_image_relevance(
                make_chart_image(), "image/png", reference_embeddings=None
            )

        # 5 calls per reference + 1 per l'immagine = 6
        assert mock_client.models.embed_content.call_count == 6
        assert isinstance(sim, float)
        assert cat in {
            "scientific_plot", "comparison_table",
            "architecture_diagram", "flowchart", "confusion_matrix",
        }

    def test_similarity_in_valid_range(self, mock_embedder):
        """La similarità coseno deve essere in [-1, 1]."""
        embedder, mock_client = mock_embedder

        fake_emb = _make_fake_embedding(seed=130)
        mock_result = MagicMock()
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = fake_emb
        mock_result.embeddings = [mock_emb_obj]
        mock_client.models.embed_content.return_value = mock_result

        ref_embeddings = {
            "scientific_plot": _make_fake_embedding(seed=131),
        }

        sim, _ = embedder.classify_image_relevance(
            make_chart_image(), "image/png", ref_embeddings
        )

        assert -1.0 <= sim <= 1.0


# ============================================================================
# Test: cosine_similarity helper
# ============================================================================


class TestCosineSimilarity:
    """Test per la funzione helper _cosine_similarity."""

    def test_identical_vectors(self):
        """Vettori identici devono avere similarità 1.0."""
        from paperhunter.utils.multimodal_embedder import _cosine_similarity
        vec = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        """Vettori ortogonali devono avere similarità 0.0."""
        from paperhunter.utils.multimodal_embedder import _cosine_similarity
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self):
        """Vettori opposti devono avere similarità -1.0."""
        from paperhunter.utils.multimodal_embedder import _cosine_similarity
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-9

    def test_zero_vector(self):
        """Vettore zero deve restituire 0.0 (no divisione per zero)."""
        from paperhunter.utils.multimodal_embedder import _cosine_similarity
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_normalized_vectors_high_dim(self):
        """Vettori normalizzati ad alta dimensionalità devono funzionare."""
        from paperhunter.utils.multimodal_embedder import _cosine_similarity
        a = _make_fake_embedding(seed=200, dim=3072)
        b = _make_similar_embedding(a, noise_scale=0.01, seed=201)
        sim = _cosine_similarity(a, b)
        # Embedding simili → alta similarità (in 3072-dim, noise_scale=0.01 produce ~0.87)
        assert sim > 0.8


# ============================================================================
# Test: singleton get_multimodal_embedder
# ============================================================================


class TestSingleton:
    """Test per la funzione singleton get_multimodal_embedder."""

    def test_singleton_returns_same_instance(self):
        """get_multimodal_embedder deve restituire la stessa istanza."""
        import paperhunter.utils.multimodal_embedder as mod

        # Reset singleton
        mod._instance = None

        with patch.object(mod.MultimodalEmbedder, "_create_client", return_value=MagicMock()):
            with patch("paperhunter.utils.multimodal_embedder.get_gemini_limiter") as mock_lim:
                mock_lim.return_value = MagicMock()
                with patch("paperhunter.config.settings") as mock_cfg:
                    mock_cfg.gemini.embedding_model = "gemini-embedding-2-preview"

                    inst1 = mod.get_multimodal_embedder()
                    inst2 = mod.get_multimodal_embedder()

                    assert inst1 is inst2

        # Cleanup
        mod._instance = None

    def test_singleton_is_multimodal_embedder(self):
        """get_multimodal_embedder deve restituire un'istanza di MultimodalEmbedder."""
        import paperhunter.utils.multimodal_embedder as mod

        mod._instance = None

        with patch.object(mod.MultimodalEmbedder, "_create_client", return_value=MagicMock()):
            with patch("paperhunter.utils.multimodal_embedder.get_gemini_limiter") as mock_lim:
                mock_lim.return_value = MagicMock()
                with patch("paperhunter.config.settings") as mock_cfg:
                    mock_cfg.gemini.embedding_model = "gemini-embedding-2-preview"

                    inst = mod.get_multimodal_embedder()
                    assert isinstance(inst, mod.MultimodalEmbedder)

        mod._instance = None


# ============================================================================
# Test: edge cases
# ============================================================================


class TestEdgeCases:
    """Test di robustezza."""

    def test_embed_image_jpeg_mime(self, mock_embedder):
        """Deve funzionare con mime_type image/jpeg."""
        embedder, mock_client = mock_embedder
        fake_emb = _make_fake_embedding(seed=300)

        mock_result = MagicMock()
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = fake_emb
        mock_result.embeddings = [mock_emb_obj]
        mock_client.models.embed_content.return_value = mock_result

        # Crea JPEG
        img = Image.new("RGB", (100, 100), "red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        result = embedder.embed_image(jpeg_bytes, mime_type="image/jpeg")
        assert len(result) == 3072

    def test_custom_model_name(self, mock_settings):
        """MultimodalEmbedder deve accettare un modello personalizzato."""
        with patch(
            "paperhunter.utils.multimodal_embedder.MultimodalEmbedder._create_client",
            return_value=MagicMock(),
        ):
            with patch("paperhunter.config.settings") as mock_cfg:
                mock_cfg.gemini.embedding_model = "default-model"

                from paperhunter.utils.multimodal_embedder import MultimodalEmbedder
                embedder = MultimodalEmbedder(model="custom-model-v2")
                assert embedder._model == "custom-model-v2"

    def test_default_model_from_settings(self, mock_settings):
        """Se model non è specificato, deve usare settings.gemini.embedding_model."""
        with patch(
            "paperhunter.utils.multimodal_embedder.MultimodalEmbedder._create_client",
            return_value=MagicMock(),
        ):
            with patch("paperhunter.config.settings") as mock_cfg:
                mock_cfg.gemini.embedding_model = "gemini-embedding-2-preview"

                from paperhunter.utils.multimodal_embedder import MultimodalEmbedder
                embedder = MultimodalEmbedder()
                assert embedder._model == "gemini-embedding-2-preview"

    def test_partial_cache_triggers_recalculation(self, mock_embedder, tmp_path):
        """Cache con categorie mancanti deve triggare ricalcolo."""
        embedder, mock_client = mock_embedder

        cache_path = tmp_path / "reference_embeddings.json"
        # Cache con solo 2 categorie su 5
        partial_cache = {
            "scientific_plot": _make_fake_embedding(seed=310),
            "flowchart": _make_fake_embedding(seed=311),
        }
        cache_path.write_text(json.dumps(partial_cache), encoding="utf-8")

        call_count = [0]
        def side_effect(**kwargs):
            seed = 320 + call_count[0]
            call_count[0] += 1
            mock_result = MagicMock()
            mock_emb_obj = MagicMock()
            mock_emb_obj.values = _make_fake_embedding(seed=seed)
            mock_result.embeddings = [mock_emb_obj]
            return mock_result

        mock_client.models.embed_content.side_effect = side_effect

        with patch(
            "paperhunter.utils.multimodal_embedder._REFERENCE_CACHE_PATH",
            cache_path,
        ):
            embedder._reference_embeddings = None
            result = embedder.compute_reference_embeddings()

        # Deve aver ricalcolato via API (5 chiamate)
        assert mock_client.models.embed_content.call_count == 5
        assert len(result) == 5
