"""Regressioni per il profilo PaperHunter DeepSeek-only."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


def test_runtime_profile_uses_v4_flash_for_every_pipeline_stage_with_high_reasoning():
    """Il profilo economico non deve poter instradare una fase verso V4 Pro."""
    from paperhunter.config import settings
    from paperhunter.utils.llm_factory import get_flash_llm, get_pro_llm

    assert settings.deepseek.flash_model == "deepseek-v4-flash"
    assert settings.deepseek.pro_model == "deepseek-v4-flash"
    assert settings.deepseek.thinking_enabled is True
    assert settings.deepseek.reasoning_effort == "high"
    assert {
        settings.agents.planner_model,
        settings.agents.writer_model,
        settings.agents.critic_model,
        settings.agents.assembler_model,
    } == {"flash"}
    assert settings.summarization.use_pro_for_l1 is False
    assert settings.summarization.use_pro_for_l2 is False

    flash = get_flash_llm()
    legacy_pro_tier = get_pro_llm()
    assert flash.model == "deepseek-v4-flash"
    assert legacy_pro_tier.model == "deepseek-v4-flash"
    assert flash.thinking_enabled is True
    assert legacy_pro_tier.thinking_enabled is True
    assert flash.reasoning_effort == "high"
    assert legacy_pro_tier.reasoning_effort == "high"


def test_runtime_profile_disables_google_and_all_image_features():
    """Il profilo versionato non deve abilitare provider Google o percorsi immagini."""
    from paperhunter.config import settings

    assert settings.llm_provider == "deepseek"
    assert settings.embedding.provider == "huggingface"
    assert settings.visual_features_enabled is False
    assert settings.figure_extraction.enabled is False
    assert settings.image_filtering.enabled is False
    assert settings.image_filtering.embedding_filter_enabled is False
    assert settings.figure_agent.enabled is False


@pytest.mark.parametrize(
    ("factory_name", "module_name", "class_name"),
    [
        ("get_flash_llm", "llama_index.llms.google_genai", "GoogleGenAI"),
        ("get_pro_llm", "llama_index.llms.google_genai", "GoogleGenAI"),
        ("get_embedding_model", "llama_index.embeddings.google_genai", "GoogleGenAIEmbedding"),
    ],
)
def test_google_factories_fail_before_constructing_a_client(
    monkeypatch,
    factory_name: str,
    module_name: str,
    class_name: str,
):
    """Un override Google deve fallire localmente, prima di creare un client/API call."""
    from paperhunter import utils
    from paperhunter.utils import llm_factory

    class UnexpectedGoogleClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Google client construction must be unreachable")

    fake_module = types.ModuleType(module_name)
    setattr(fake_module, class_name, UnexpectedGoogleClient)
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    factory = getattr(llm_factory, factory_name)
    with pytest.raises(RuntimeError, match="DeepSeek-only"):
        factory(provider="google")


def test_multimodal_embedding_is_disabled_before_client_initialization(monkeypatch):
    """La policy deve bloccare gli embedding immagine prima di inizializzare Gemini."""
    from paperhunter.utils import multimodal_embedder

    class UnexpectedMultimodalEmbedder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Multimodal Gemini initialization must be unreachable")

    monkeypatch.setattr(multimodal_embedder, "MultimodalEmbedder", UnexpectedMultimodalEmbedder)
    multimodal_embedder._instance = None

    with pytest.raises(RuntimeError, match="DeepSeek-only"):
        multimodal_embedder.get_multimodal_embedder()


def test_figure_extractor_does_not_open_pdf_when_visual_processing_is_disabled():
    """Disabilitare Vision significa non aprire né analizzare immagini dei PDF."""
    from paperhunter.ingestion.figure_extractor import FigureExtractor

    assert FigureExtractor().extract_images_from_pdf("missing.pdf") == []


def test_figures_cli_is_disabled_before_importing_figure_agent():
    """Il comando figure deve fermarsi subito e non avviare rendering/generazione."""
    from paperhunter.main import cmd_figures

    result = cmd_figures(SimpleNamespace(input_file=None))

    assert result == 2


def test_qdrant_does_not_create_an_image_collection_when_visual_features_are_disabled(tmp_path):
    """La collection immagini inattiva non deve essere creata in nuovi store."""
    from paperhunter.indexing.qdrant_store import QdrantStore

    store = QdrantStore(persist_dir=tmp_path)
    try:
        assert store.client.collection_exists(store.COLLECTION_IMAGE_EMBEDDINGS) is False
    finally:
        store.close()


def test_direct_visual_components_fail_closed_without_creating_output(tmp_path):
    """Anche gli entry point interni non devono avviare servizi Google o creare PNG."""
    from paperhunter.agents.figure_agent.renderer import render_figure
    from paperhunter.utils.multimodal_embedder import MultimodalEmbedder

    with pytest.raises(RuntimeError, match="DeepSeek-only"):
        MultimodalEmbedder()

    spec = SimpleNamespace(figure_id="blocked", title="blocked")
    success, image_path, error = render_figure(spec, tmp_path / "figures")

    assert success is False
    assert image_path is None
    assert "DeepSeek-only" in (error or "")
    assert not (tmp_path / "figures").exists()
