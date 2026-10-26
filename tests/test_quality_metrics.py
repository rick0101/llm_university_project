"""Test per paperhunter/utils/quality_metrics.py.

Verifica che:
1. estimate_bpb_from_token_count() restituisca un float positivo su testo valido.
2. estimate_bpb_from_token_count() funzioni senza GPU.
3. compute_survey_quality_report() restituisca un dizionario con tutte le chiavi attese.
4. Testo ad alta qualità (articolo scientifico) abbia BPB diverso da testo casuale.
5. compute_bpb() restituisca un float positivo (con mock del modello locale).
6. Le funzioni sollevino ValueError per testo vuoto.
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from paperhunter.utils.quality_metrics import (
    compute_bpb,
    compute_survey_quality_report,
    estimate_bpb_from_token_count,
)


# ============================================================================
# Fixtures
# ============================================================================

SCIENTIFIC_TEXT = (
    "Large language models (LLMs) have demonstrated remarkable capabilities "
    "in natural language understanding and generation. However, these models "
    "are vulnerable to various adversarial attacks, including prompt injection, "
    "data poisoning, and model extraction. This survey provides a comprehensive "
    "analysis of security threats targeting LLMs and evaluates existing defense "
    "mechanisms. We categorize attacks into four main classes: adversarial code "
    "generation, data and model poisoning, hallucination exploitation, and "
    "prompt injection. For each category, we analyze the threat model, review "
    "state-of-the-art attack techniques, and discuss mitigation strategies."
)

RANDOM_TEXT = (
    "xkq7 zPmN 3$! fRt9w bL@2 vYjH 8&#x dKs5 qW1! nMp4 gT6* cXz3 "
    "jR8@ hFb2 wL7# yNk9 tQp5 mVx1 sD4& aEj6 uBn3 oGw8 iCr2 lKf7 "
    "pAh5 xTm9 bUq1 vRd4 nYs6 cJk3 wEg8 tLb2 hNp7 mXf5 aQj9 uDn1 "
    "oVr4 sGk6 iCw3 lBm8 xAh2 pTf7 bUq5 yRd9 nEs1 cJk4 wLg6 tNb3"
)


# ============================================================================
# Test estimate_bpb_from_token_count
# ============================================================================


class TestEstimateBpbFromTokenCount:
    """Test per la variante leggera basata su tiktoken."""

    def test_returns_positive_float(self):
        """BPB stimato deve essere un float positivo."""
        result = estimate_bpb_from_token_count(SCIENTIFIC_TEXT)
        assert isinstance(result, float)
        assert result > 0

    def test_works_without_gpu(self):
        """La funzione non richiede torch né GPU."""
        # Se arriviamo qui senza ImportError, il test passa
        result = estimate_bpb_from_token_count("Hello world, this is a test.")
        assert result > 0

    def test_empty_text_raises_value_error(self):
        """Testo vuoto deve sollevare ValueError."""
        with pytest.raises(ValueError, match="vuoto"):
            estimate_bpb_from_token_count("")

    def test_scientific_vs_random_text(self):
        """Testo casuale dovrebbe avere BPB stimato più alto di testo scientifico."""
        bpb_scientific = estimate_bpb_from_token_count(SCIENTIFIC_TEXT)
        bpb_random = estimate_bpb_from_token_count(RANDOM_TEXT)
        # Testo casuale/rumoroso ha più token per byte → BPB più alto
        assert bpb_random > bpb_scientific

    def test_custom_encoding(self):
        """Deve funzionare con encoding diversi da quello di default."""
        result = estimate_bpb_from_token_count(
            SCIENTIFIC_TEXT, encoding_name="cl100k_base"
        )
        assert isinstance(result, float)
        assert result > 0

    def test_unicode_text(self):
        """Deve gestire correttamente testo con caratteri Unicode multi-byte."""
        unicode_text = "Les modèles de langage présentent des vulnérabilités. 日本語テスト。"
        result = estimate_bpb_from_token_count(unicode_text)
        assert isinstance(result, float)
        assert result > 0

    def test_long_text(self):
        """Testo molto lungo deve funzionare senza errori."""
        long_text = SCIENTIFIC_TEXT * 100
        result = estimate_bpb_from_token_count(long_text)
        assert isinstance(result, float)
        assert result > 0


# ============================================================================
# Test compute_survey_quality_report
# ============================================================================


class TestComputeSurveyQualityReport:
    """Test per il report di qualità del survey."""

    def test_returns_dict_with_all_keys(self):
        """Il report deve contenere tutte le chiavi documentate."""
        report = compute_survey_quality_report(SCIENTIFIC_TEXT)
        expected_keys = {"n_tokens", "n_bytes", "tokens_per_byte", "estimated_bpb", "encoding"}
        assert set(report.keys()) == expected_keys

    def test_n_tokens_positive(self):
        """Il numero di token deve essere positivo."""
        report = compute_survey_quality_report(SCIENTIFIC_TEXT)
        assert report["n_tokens"] > 0

    def test_n_bytes_positive(self):
        """Il numero di byte deve essere positivo."""
        report = compute_survey_quality_report(SCIENTIFIC_TEXT)
        assert report["n_bytes"] > 0

    def test_tokens_per_byte_range(self):
        """Il rapporto token/byte per testo inglese dovrebbe essere tra 0.1 e 1.0."""
        report = compute_survey_quality_report(SCIENTIFIC_TEXT)
        assert 0.1 <= report["tokens_per_byte"] <= 1.0

    def test_estimated_bpb_matches_standalone(self):
        """Il BPB nel report deve corrispondere alla funzione standalone."""
        report = compute_survey_quality_report(SCIENTIFIC_TEXT)
        standalone_bpb = estimate_bpb_from_token_count(SCIENTIFIC_TEXT)
        assert abs(report["estimated_bpb"] - round(standalone_bpb, 4)) < 0.001

    def test_encoding_field(self):
        """Il campo encoding deve riflettere l'encoding usato."""
        report = compute_survey_quality_report(SCIENTIFIC_TEXT)
        assert report["encoding"] == "cl100k_base"

    def test_empty_text_raises_value_error(self):
        """Testo vuoto deve sollevare ValueError."""
        with pytest.raises(ValueError, match="vuoto"):
            compute_survey_quality_report("")


# ============================================================================
# Test compute_bpb (con mock del modello locale)
# ============================================================================


class TestComputeBpb:
    """Test per la variante completa con modello locale (mocked)."""

    def test_empty_text_raises_value_error(self):
        """Testo vuoto deve sollevare ValueError."""
        with pytest.raises(ValueError, match="vuoto"):
            compute_bpb("", MagicMock(), MagicMock())

    def test_returns_positive_float_with_mock(self):
        """compute_bpb deve restituire un float positivo con modello mocked."""
        torch = pytest.importorskip("torch")

        mock_tokenizer = MagicMock()
        # Simula tokenizer: restituisce input_ids con 10 token
        mock_input_ids = torch.randint(0, 1000, (1, 10))
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        # Simula output del modello: loss = 2.5 (valore realistico)
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.5)
        mock_model.return_value = mock_output

        text = "This is a test text for BPB computation."
        result = compute_bpb(text, mock_model, mock_tokenizer)

        assert isinstance(result, float)
        assert result > 0

    def test_model_moved_to_device(self):
        """Il modello deve essere spostato sul dispositivo corretto."""
        torch = pytest.importorskip("torch")

        mock_tokenizer = MagicMock()
        mock_input_ids = torch.randint(0, 1000, (1, 5))
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        mock_output = MagicMock()
        mock_output.loss = torch.tensor(3.0)
        mock_model.return_value = mock_output

        compute_bpb("Test text.", mock_model, mock_tokenizer)

        # model.to() deve essere stato chiamato con un device
        mock_model.to.assert_called_once()

    def test_bpb_formula_correctness(self):
        """Verifica la correttezza della formula BPB = loss / ln(2) / (bytes/tokens)."""
        torch = pytest.importorskip("torch")

        text = "Hello world"
        n_bytes = len(text.encode("utf-8"))
        n_tokens = 3
        loss_value = 2.0

        mock_tokenizer = MagicMock()
        mock_input_ids = torch.randint(0, 1000, (1, n_tokens))
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        mock_output = MagicMock()
        mock_output.loss = torch.tensor(loss_value)
        mock_model.return_value = mock_output

        result = compute_bpb(text, mock_model, mock_tokenizer)

        # Calcolo atteso: loss / ln(2) / (n_bytes / n_tokens)
        mean_bytes_per_token = n_bytes / n_tokens
        expected = loss_value / math.log(2) / mean_bytes_per_token

        assert abs(result - expected) < 1e-6

    def test_import_error_without_torch(self):
        """Se torch non è installato, deve sollevare ImportError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="torch"):
                compute_bpb("test text", MagicMock(), MagicMock())
