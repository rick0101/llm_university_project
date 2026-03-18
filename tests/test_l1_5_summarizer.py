"""
Test per paperhunter.summarization.level1_5_summarizer — Meta-summary L1.5 per cluster RAPTOR.

Verifica:
- Generazione di un singolo L1.5 summary per un cluster mock
- Batch processing di più cluster
- Parsing JSON dalla risposta LLM
- Validazione qualità del summary
- Formattazione L1 per il prompt
- Calcolo temporal_span e paper_ids
- Salvataggio e caricamento da Qdrant
- Ricostruzione L1_5Summary dal payload Qdrant
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from paperhunter.models import L1Summary, L1_5Summary


# ===========================================================================
# Helpers
# ===========================================================================

def _make_l1_summary(
    paper_id: str,
    category: str = "Prompt Injection",
    year: int = 2024,
    bibtex_key: str | None = None,
) -> L1Summary:
    """Crea un L1Summary mock minimale per i test."""
    return L1Summary(
        paper_id=paper_id,
        bibtex_key=bibtex_key or paper_id,
        category=category,
        year=year,
        key_contributions=[
            f"Contribution 1 of {paper_id}",
            f"Contribution 2 of {paper_id}",
        ],
        methodology=f"Methodology description for {paper_id}: uses transformer-based approach.",
        testing_methodology=f"Testing for {paper_id}: benchmarks on AdvBench, ASR metric, GPT-4 model.",
        temporal_context=f"Proposed in {year}, building on prior work.",
        results_summary=f"Results for {paper_id}: 85% ASR on AdvBench.",
        limitations=f"Limitations of {paper_id}: limited to English.",
    )


def _make_cluster(
    n_papers: int = 15,
    category: str = "Prompt Injection",
    base_year: int = 2022,
) -> list[L1Summary]:
    """Crea un cluster mock di N L1 Summary."""
    summaries = []
    for i in range(n_papers):
        year = base_year + (i % 4)  # 2022-2025
        summaries.append(
            _make_l1_summary(
                paper_id=f"Author_{year}_Paper{i}",
                category=category,
                year=year,
                bibtex_key=f"Author_{year}_Paper{i}",
            )
        )
    return summaries


def _make_mock_llm_response() -> str:
    """Crea una risposta JSON mock come la produrrebbe il LLM."""
    data = {
        "common_theme": (
            "This cluster focuses on prompt injection attacks against LLMs, "
            "exploring various techniques to bypass safety guardrails. "
            "Papers [Author_2023_Paper1] and [Author_2024_Paper5] propose "
            "automated methods for generating adversarial prompts."
        ),
        "shared_methodologies": (
            "Several papers employ gradient-based optimization for generating adversarial "
            "suffixes [Author_2023_Paper1], while others use genetic algorithms "
            "[Author_2024_Paper5]. The approaches diverge on the threat model: "
            "[Author_2022_Paper0] assumes white-box access while [Author_2024_Paper9] "
            "operates in a black-box setting."
        ),
        "key_findings": (
            "Across the cluster, Attack Success Rate (ASR) ranges from 60% to 95% "
            "depending on the target model. [Author_2023_Paper1] achieves 85% ASR on "
            "GPT-4 using GCG suffixes. [Author_2024_Paper5] reports 92% ASR with "
            "evolutionary search. Notably, [Author_2025_Paper3] shows that defenses "
            "reduce ASR to below 20% when perplexity filtering is applied. "
            "There is convergence on the finding that larger models are more susceptible "
            "to suffix-based attacks [Author_2023_Paper1], [Author_2024_Paper9]."
        ),
        "testing_overview": (
            "Common benchmarks: AdvBench (used by 10/15 papers), HarmBench "
            "[Author_2024_Paper5], JailbreakBench [Author_2025_Paper3]. "
            "Metrics: ASR, Perplexity, BLEU. Models tested: GPT-4, LLaMA-2, Claude-3."
        ),
    }
    return json.dumps(data)


# ===========================================================================
# Test: modello L1_5Summary
# ===========================================================================

class TestL1_5SummaryModel:
    """Test per il modello Pydantic L1_5Summary."""

    def test_create_minimal(self):
        """Crea un L1_5Summary con campi minimi required."""
        s = L1_5Summary(cluster_id=0, category="test")
        assert s.cluster_id == 0
        assert s.category == "test"
        assert s.paper_ids == []
        assert s.paper_count == 0

    def test_create_full(self):
        """Crea un L1_5Summary con tutti i campi."""
        s = L1_5Summary(
            cluster_id=3,
            category="Prompt Injection",
            paper_ids=["paper_a", "paper_b"],
            paper_count=2,
            common_theme="Jailbreak attacks",
            shared_methodologies="Gradient-based optimization",
            key_findings="High ASR on GPT-4",
            testing_overview="AdvBench benchmark, ASR metric",
            temporal_span="2023-2025",
            generated_at="2026-03-05T10:00:00",
            model_used="pro",
        )
        assert s.cluster_id == 3
        assert s.paper_count == 2
        assert len(s.paper_ids) == 2
        assert s.temporal_span == "2023-2025"
        assert s.model_used == "pro"

    def test_serialization_json(self):
        """Verifica serializzazione/deserializzazione JSON."""
        s = L1_5Summary(
            cluster_id=1,
            category="test",
            paper_ids=["a", "b"],
            paper_count=2,
            common_theme="Theme",
            generated_at="2026-03-05T10:00:00",
        )
        json_str = s.model_dump_json()
        restored = L1_5Summary.model_validate_json(json_str)
        assert restored.cluster_id == s.cluster_id
        assert restored.category == s.category
        assert restored.paper_ids == s.paper_ids


# ===========================================================================
# Test: formatting helpers
# ===========================================================================

class TestFormattingHelpers:
    """Test per le funzioni helper di formattazione."""

    def test_format_l1_for_cluster_prompt(self):
        """Verifica che le L1 vengano formattate correttamente."""
        from paperhunter.summarization.level1_5_summarizer import (
            _format_l1_for_cluster_prompt,
        )

        cluster = _make_cluster(3, base_year=2023)
        text = _format_l1_for_cluster_prompt(cluster)

        # Deve contenere l'header "Paper" per ogni summary
        assert "Paper 1:" in text
        assert "Paper 2:" in text
        assert "Paper 3:" in text

        # Deve contenere bibtex_key tra parentesi quadre
        assert "[Author_2023_Paper0]" in text

        # Deve contenere i campi principali
        assert "Key Contributions:" in text
        assert "Methodology:" in text
        assert "Testing:" in text

    def test_format_l1_sorted_by_year_descending(self):
        """Verifica che le L1 siano ordinate per anno decrescente."""
        from paperhunter.summarization.level1_5_summarizer import (
            _format_l1_for_cluster_prompt,
        )

        summaries = [
            _make_l1_summary("old_paper", year=2020),
            _make_l1_summary("new_paper", year=2025),
            _make_l1_summary("mid_paper", year=2023),
        ]
        text = _format_l1_for_cluster_prompt(summaries)

        # Il paper più recente deve apparire per primo
        pos_new = text.index("new_paper")
        pos_mid = text.index("mid_paper")
        pos_old = text.index("old_paper")
        assert pos_new < pos_mid < pos_old

    def test_compute_temporal_span(self):
        """Verifica il calcolo del temporal span."""
        from paperhunter.summarization.level1_5_summarizer import _compute_temporal_span

        cluster = _make_cluster(10, base_year=2022)
        span = _compute_temporal_span(cluster)
        assert span == "2022-2025"  # 2022 + (0..3)

    def test_compute_temporal_span_single_year(self):
        """Temporal span con tutti i paper dello stesso anno."""
        from paperhunter.summarization.level1_5_summarizer import _compute_temporal_span

        summaries = [_make_l1_summary(f"p{i}", year=2024) for i in range(5)]
        assert _compute_temporal_span(summaries) == "2024"

    def test_compute_temporal_span_no_years(self):
        """Temporal span con paper senza anno."""
        from paperhunter.summarization.level1_5_summarizer import _compute_temporal_span

        summaries = [L1Summary(paper_id=f"p{i}") for i in range(3)]
        assert _compute_temporal_span(summaries) == "N/A"

    def test_extract_paper_ids(self):
        """Verifica estrazione paper_ids dal cluster."""
        from paperhunter.summarization.level1_5_summarizer import _extract_paper_ids

        cluster = _make_cluster(5)
        ids = _extract_paper_ids(cluster)
        assert len(ids) == 5
        assert all(isinstance(pid, str) for pid in ids)

    def test_extract_max_year(self):
        """Verifica estrazione anno massimo dal temporal_span."""
        from paperhunter.summarization.level1_5_summarizer import _extract_max_year

        assert _extract_max_year("2022-2025") == 2025
        assert _extract_max_year("2024") == 2024
        assert _extract_max_year("N/A") is None
        assert _extract_max_year("") is None


# ===========================================================================
# Test: JSON parsing
# ===========================================================================

class TestJsonParsing:
    """Test per il parsing JSON della risposta LLM."""

    def test_parse_clean_json(self):
        """JSON pulito senza wrapper."""
        from paperhunter.summarization.level1_5_summarizer import (
            _parse_l1_5_json_response,
        )

        response = _make_mock_llm_response()
        parsed = _parse_l1_5_json_response(response, cluster_id=0)
        assert "common_theme" in parsed
        assert "shared_methodologies" in parsed
        assert "key_findings" in parsed
        assert "testing_overview" in parsed

    def test_parse_json_with_code_fence(self):
        """JSON dentro code fence markdown."""
        from paperhunter.summarization.level1_5_summarizer import (
            _parse_l1_5_json_response,
        )

        response = f"```json\n{_make_mock_llm_response()}\n```"
        parsed = _parse_l1_5_json_response(response, cluster_id=0)
        assert "common_theme" in parsed

    def test_parse_json_with_surrounding_text(self):
        """JSON con testo attorno."""
        from paperhunter.summarization.level1_5_summarizer import (
            _parse_l1_5_json_response,
        )

        response = f"Here is the summary:\n\n{_make_mock_llm_response()}\n\nDone."
        parsed = _parse_l1_5_json_response(response, cluster_id=0)
        assert "common_theme" in parsed

    def test_parse_invalid_json_raises(self):
        """JSON non valido solleva ValueError."""
        from paperhunter.summarization.level1_5_summarizer import (
            _parse_l1_5_json_response,
        )

        with pytest.raises(ValueError, match="Impossibile estrarre JSON"):
            _parse_l1_5_json_response("This is not JSON at all.", cluster_id=99)


# ===========================================================================
# Test: validazione qualità
# ===========================================================================

class TestValidation:
    """Test per la validazione qualità dei summary L1.5."""

    def test_validate_good_summary(self):
        """Un summary ben formato non deve generare warning."""
        from paperhunter.summarization.level1_5_summarizer import (
            _validate_l1_5_summary,
        )

        s = L1_5Summary(
            cluster_id=0,
            category="test",
            common_theme="This cluster focuses on prompt injection [Author_2024] techniques.",
            shared_methodologies="Papers use gradient optimization [Author_2024] and search [Author_2023].",
            key_findings="ASR of 85% achieved by [Author_2024], while [Author_2023] reports 70%.",
            testing_overview="AdvBench [Author_2024], HarmBench [Author_2023].",
        )
        # No exception expected
        _validate_l1_5_summary(s, cluster_id=0)

    def test_validate_empty_fields_warns(self):
        """Campi vuoti devono generare warning (ma non eccezioni)."""
        from paperhunter.summarization.level1_5_summarizer import (
            _validate_l1_5_summary,
        )

        s = L1_5Summary(
            cluster_id=1,
            category="test",
            common_theme="",
            shared_methodologies="",
            key_findings="",
            testing_overview="",
        )
        # No exception, just warnings logged
        _validate_l1_5_summary(s, cluster_id=1)


# ===========================================================================
# Test: generate_cluster_summary (con LLM mock)
# ===========================================================================

class TestGenerateClusterSummary:
    """Test per la generazione di un singolo L1.5 summary con LLM mock."""

    @patch("paperhunter.summarization.level1_5_summarizer.llm_call_with_retry")
    @patch("paperhunter.summarization.level1_5_summarizer.get_pro_llm")
    @patch("paperhunter.summarization.level1_5_summarizer.get_gemini_limiter")
    def test_generate_single_cluster(self, mock_limiter, mock_get_pro, mock_llm_call):
        """Genera un L1.5 per un cluster di 15 paper mock."""
        from paperhunter.summarization.level1_5_summarizer import (
            generate_cluster_summary,
        )

        # Setup mocks
        mock_limiter_instance = MagicMock()
        mock_limiter.return_value = mock_limiter_instance
        mock_get_pro.return_value = MagicMock()
        mock_llm_call.return_value = _make_mock_llm_response()

        cluster = _make_cluster(15, category="Prompt Injection", base_year=2022)

        result = generate_cluster_summary(
            cluster_id=0,
            summaries=cluster,
            category="Prompt Injection",
        )

        # Verifica tipo e campi
        assert isinstance(result, L1_5Summary)
        assert result.cluster_id == 0
        assert result.category == "Prompt Injection"
        assert result.paper_count == 15
        assert len(result.paper_ids) == 15
        assert result.temporal_span == "2022-2025"

        # Verifica che i campi di contenuto siano non vuoti
        assert len(result.common_theme) > 0
        assert len(result.shared_methodologies) > 0
        assert len(result.key_findings) > 0
        assert len(result.testing_overview) > 0

        # Verifica che generated_at sia un timestamp ISO valido
        assert result.generated_at != ""
        datetime.fromisoformat(result.generated_at)

        # Verifica che il LLM sia stato chiamato
        assert mock_llm_call.called

    @patch("paperhunter.summarization.level1_5_summarizer.llm_call_with_retry")
    @patch("paperhunter.summarization.level1_5_summarizer.get_pro_llm")
    @patch("paperhunter.summarization.level1_5_summarizer.get_gemini_limiter")
    def test_generate_cluster_with_save(self, mock_limiter, mock_get_pro, mock_llm_call):
        """Genera un L1.5 con salvataggio su store mock."""
        from paperhunter.summarization.level1_5_summarizer import (
            generate_cluster_summary,
        )

        mock_limiter.return_value = MagicMock()
        mock_get_pro.return_value = MagicMock()
        mock_llm_call.return_value = _make_mock_llm_response()

        mock_store = MagicMock()
        mock_store.add_chunks.return_value = 1

        cluster = _make_cluster(5, category="Test")

        result = generate_cluster_summary(
            cluster_id=1,
            summaries=cluster,
            category="Test",
            save_to_store=True,
            store=mock_store,
        )

        assert isinstance(result, L1_5Summary)
        # Verifica che add_chunks sia stato chiamato
        assert mock_store.add_chunks.called

    def test_generate_empty_cluster_raises(self):
        """Cluster vuoto deve sollevare ValueError."""
        from paperhunter.summarization.level1_5_summarizer import (
            generate_cluster_summary,
        )

        with pytest.raises(ValueError, match="lista di L1 Summary vuota"):
            generate_cluster_summary(
                cluster_id=0,
                summaries=[],
                category="test",
            )


# ===========================================================================
# Test: generate_all_cluster_summaries (batch)
# ===========================================================================

class TestGenerateAllClusterSummaries:
    """Test per il batch processing di tutti i cluster."""

    @patch("paperhunter.summarization.level1_5_summarizer.llm_call_with_retry")
    @patch("paperhunter.summarization.level1_5_summarizer.get_pro_llm")
    @patch("paperhunter.summarization.level1_5_summarizer.get_gemini_limiter")
    def test_batch_generates_all(self, mock_limiter, mock_get_pro, mock_llm_call):
        """Batch processing di 3 cluster produce 3 L1.5 summary."""
        from paperhunter.summarization.level1_5_summarizer import (
            generate_all_cluster_summaries,
        )

        mock_limiter.return_value = MagicMock()
        mock_get_pro.return_value = MagicMock()
        mock_llm_call.return_value = _make_mock_llm_response()

        clusters = {
            0: _make_cluster(10, category="Cat_A"),
            1: _make_cluster(8, category="Cat_A"),
            2: _make_cluster(12, category="Cat_A"),
        }

        results = generate_all_cluster_summaries(
            clusters=clusters,
            category="Cat_A",
            save_to_store=False,
        )

        assert len(results) == 3
        assert all(isinstance(r, L1_5Summary) for r in results)
        assert {r.cluster_id for r in results} == {0, 1, 2}

    def test_batch_empty_clusters(self):
        """Batch vuoto restituisce lista vuota."""
        from paperhunter.summarization.level1_5_summarizer import (
            generate_all_cluster_summaries,
        )

        results = generate_all_cluster_summaries(
            clusters={},
            category="test",
            save_to_store=False,
        )
        assert results == []


# ===========================================================================
# Test: chunk ID e testo per Qdrant
# ===========================================================================

class TestQdrantHelpers:
    """Test per gli helper di salvataggio Qdrant."""

    def test_make_l1_5_chunk_id_deterministic(self):
        """Lo stesso cluster_id e categoria producono lo stesso chunk_id."""
        from paperhunter.summarization.level1_5_summarizer import _make_l1_5_chunk_id

        id1 = _make_l1_5_chunk_id(0, "Prompt Injection")
        id2 = _make_l1_5_chunk_id(0, "Prompt Injection")
        assert id1 == id2

    def test_make_l1_5_chunk_id_unique(self):
        """Cluster diversi producono chunk_id diversi."""
        from paperhunter.summarization.level1_5_summarizer import _make_l1_5_chunk_id

        id1 = _make_l1_5_chunk_id(0, "Prompt Injection")
        id2 = _make_l1_5_chunk_id(1, "Prompt Injection")
        id3 = _make_l1_5_chunk_id(0, "Adversarial Attacks")
        assert id1 != id2
        assert id1 != id3

    def test_l1_5_summary_to_text(self):
        """Verifica che il testo generato contenga tutti i campi."""
        from paperhunter.summarization.level1_5_summarizer import _l1_5_summary_to_text

        s = L1_5Summary(
            cluster_id=2,
            category="Prompt Injection",
            paper_ids=["paper_a", "paper_b"],
            paper_count=2,
            common_theme="Jailbreak attacks on LLMs",
            shared_methodologies="Gradient-based optimization",
            key_findings="High ASR on GPT-4",
            testing_overview="AdvBench benchmark",
            temporal_span="2023-2025",
            generated_at="2026-03-05T10:00:00",
        )

        text = _l1_5_summary_to_text(s)

        assert "Cluster 2" in text
        assert "Prompt Injection" in text
        assert "2 " in text  # paper count
        assert "2023-2025" in text
        assert "paper_a" in text
        assert "Jailbreak attacks on LLMs" in text
        assert "Gradient-based optimization" in text
        assert "High ASR on GPT-4" in text
        assert "AdvBench benchmark" in text

    def test_reconstruct_l1_5_from_payload(self):
        """Ricostruzione L1_5Summary da payload e testo Qdrant."""
        from paperhunter.summarization.level1_5_summarizer import (
            _l1_5_summary_to_text,
            _reconstruct_l1_5_from_payload,
        )

        original = L1_5Summary(
            cluster_id=5,
            category="Adversarial Attacks",
            paper_ids=["paper_x", "paper_y", "paper_z"],
            paper_count=3,
            common_theme="Adversarial perturbations",
            shared_methodologies="PGD and FGSM variants",
            key_findings="Robust defense via adversarial training",
            testing_overview="CIFAR-10, ImageNet benchmarks",
            temporal_span="2021-2024",
            generated_at="2026-03-05T10:00:00",
        )

        text = _l1_5_summary_to_text(original)
        payload = {
            "paper_id": "cluster_5",
            "category": "Adversarial Attacks",
            "year": 2024,
            "text": text,
        }

        reconstructed = _reconstruct_l1_5_from_payload(payload, text)

        assert reconstructed is not None
        assert reconstructed.cluster_id == 5
        assert reconstructed.category == "Adversarial Attacks"
        assert reconstructed.paper_count == 3
        assert reconstructed.temporal_span == "2021-2024"
        assert "paper_x" in reconstructed.paper_ids

    def test_reconstruct_invalid_payload_returns_none(self):
        """Payload non valido restituisce None."""
        from paperhunter.summarization.level1_5_summarizer import (
            _reconstruct_l1_5_from_payload,
        )

        payload = {"paper_id": "not_a_cluster", "category": "test"}
        result = _reconstruct_l1_5_from_payload(payload, "some text")
        assert result is None


# ===========================================================================
# Test: integrazione con config
# ===========================================================================

class TestConfigIntegration:
    """Test per l'integrazione con il sistema di configurazione."""

    def test_l1_5_summaries_collection_in_config(self):
        """Verifica che la collection l1_5_summaries sia definita in config."""
        from paperhunter.config import settings

        assert hasattr(settings.qdrant, "collection_l1_5_summaries")
        assert settings.qdrant.collection_l1_5_summaries == "l1_5_summaries"

    def test_l1_5_summary_model_fields(self):
        """Verifica che tutti i campi richiesti dal task siano presenti."""
        fields = L1_5Summary.model_fields
        expected = [
            "cluster_id",
            "category",
            "paper_ids",
            "paper_count",
            "common_theme",
            "shared_methodologies",
            "key_findings",
            "testing_overview",
            "temporal_span",
            "generated_at",
        ]
        for f in expected:
            assert f in fields, f"Campo '{f}' mancante da L1_5Summary"
