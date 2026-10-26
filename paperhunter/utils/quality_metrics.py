"""Metriche di qualità per il testo generato dal survey.

Implementa la metrica BPB (Bits Per Byte) / perplexity normalizzata
per byte, utile per valutare oggettivamente la qualità del testo generato
e confrontare output di modelli diversi.

BPB è agnostica al vocabolario del tokenizer: normalizzando per byte
anziché per token, si ottiene una metrica direttamente comparabile tra
modelli con vocabolari diversi (es. Gemini Flash vs Pro, GPT-4 vs Claude).

Uso::

    # --- Variante leggera (solo tiktoken, nessun modello locale) ---
    from paperhunter.utils.quality_metrics import (
        estimate_bpb_from_token_count,
        compute_survey_quality_report,
    )

    bpb_est = estimate_bpb_from_token_count("Testo del survey generato...")
    report = compute_survey_quality_report("Testo del survey generato...")

    # --- Variante completa (richiede torch + transformers + modello locale) ---
    from paperhunter.utils.quality_metrics import compute_bpb
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    bpb = compute_bpb("Testo di esempio...", model, tokenizer)
"""

from __future__ import annotations

import math
from typing import Any

import tiktoken

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Encoding di default per la stima token/byte con tiktoken.
# cl100k_base è usato da GPT-4 / GPT-3.5-turbo ed è un buon proxy
# per stimare il rapporto token/byte su testo inglese accademico.
_DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"

# BPB teorico per testo inglese ben scritto: ~0.8–1.2 BPB.
# Testo casuale/non coerente: > 2.0 BPB.
# Fonte: Karpathy, autoresearch/prepare.py


# ---------------------------------------------------------------------------
# Funzioni pubbliche
# ---------------------------------------------------------------------------


def compute_bpb(text: str, model: Any, tokenizer: Any) -> float:
    """Calcola bits-per-byte per il testo generato usando un modello locale.

    BPB = cross_entropy_loss / ln(2) / mean_bytes_per_token

    Valori più bassi indicano testo di qualità migliore (più predicibile
    dal language model).

    Args:
        text: Testo da valutare (es. sezione del survey generato).
        model: Modello HuggingFace ``AutoModelForCausalLM`` già caricato.
        tokenizer: Tokenizer HuggingFace ``AutoTokenizer`` corrispondente.

    Returns:
        BPB come float positivo.

    Raises:
        ValueError: Se il testo è vuoto.
        ImportError: Se ``torch`` non è installato.

    Note:
        Richiede ``torch`` e ``transformers`` installati. Per ambienti
        senza GPU o per stime veloci, usare :func:`estimate_bpb_from_token_count`.
    """
    if not text:
        raise ValueError("Il testo non può essere vuoto.")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch è necessario per compute_bpb(). "
            "Installa con: pip install torch"
        ) from exc

    from paperhunter.utils.device_utils import get_device

    device = get_device()
    model = model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_bytes = len(text.encode("utf-8"))
    n_tokens = inputs["input_ids"].shape[1]

    if n_tokens == 0:
        raise ValueError("Il testo non produce token con il tokenizer fornito.")

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    mean_bytes_per_token = n_bytes / n_tokens
    bpb = loss.item() / math.log(2) / mean_bytes_per_token

    logger.debug(
        "compute_bpb: tokens=%d, bytes=%d, bytes/token=%.2f, loss=%.4f, BPB=%.4f",
        n_tokens, n_bytes, mean_bytes_per_token, loss.item(), bpb,
    )
    return bpb


def estimate_bpb_from_token_count(
    text: str,
    encoding_name: str = _DEFAULT_TIKTOKEN_ENCODING,
) -> float:
    """Stima il BPB usando il rapporto token/byte (senza modello locale).

    Questa è una proxy leggera: non calcola la vera cross-entropy, ma
    stima il BPB dal rapporto token/byte. La formula è:

        estimated_BPB ≈ (n_tokens / n_bytes) * 8

    dove 8 = bits per byte. Il rapporto token/byte è un indicatore della
    "comprimibilità" del testo: testo ripetitivo/predicibile ha meno
    token per byte; testo casuale/rumoroso ha più token per byte.

    Args:
        text: Testo da valutare.
        encoding_name: Nome dell'encoding tiktoken (default: ``cl100k_base``).

    Returns:
        BPB stimato come float positivo.

    Raises:
        ValueError: Se il testo è vuoto.
    """
    if not text:
        raise ValueError("Il testo non può essere vuoto.")

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    n_tokens = len(tokens)
    n_bytes = len(text.encode("utf-8"))

    if n_bytes == 0:
        raise ValueError("Il testo produce 0 byte UTF-8.")

    # Rapporto token/byte * 8 bits — proxy per entropia per byte.
    estimated_bpb = (n_tokens / n_bytes) * 8

    logger.debug(
        "estimate_bpb: tokens=%d, bytes=%d, tokens/byte=%.4f, estimated_BPB=%.4f",
        n_tokens, n_bytes, n_tokens / n_bytes, estimated_bpb,
    )
    return estimated_bpb


def compute_survey_quality_report(
    survey_text: str,
    encoding_name: str = _DEFAULT_TIKTOKEN_ENCODING,
) -> dict:
    """Genera un report di qualità per il testo del survey.

    Calcola metriche aggregate utili per la sezione di valutazione
    del progetto: BPB stimato, conteggio token, conteggio byte
    e rapporto token/byte.

    Args:
        survey_text: Testo completo del survey generato.
        encoding_name: Nome dell'encoding tiktoken (default: ``cl100k_base``).

    Returns:
        Dizionario con chiavi:
        - ``n_tokens``: Numero di token (tiktoken).
        - ``n_bytes``: Numero di byte UTF-8.
        - ``tokens_per_byte``: Rapporto token/byte.
        - ``estimated_bpb``: BPB stimato (proxy leggera).
        - ``encoding``: Nome dell'encoding usato.

    Raises:
        ValueError: Se il testo è vuoto.
    """
    if not survey_text:
        raise ValueError("Il testo del survey non può essere vuoto.")

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(survey_text)
    n_tokens = len(tokens)
    n_bytes = len(survey_text.encode("utf-8"))
    tokens_per_byte = n_tokens / n_bytes if n_bytes > 0 else 0.0
    estimated_bpb = estimate_bpb_from_token_count(survey_text, encoding_name)

    report = {
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "tokens_per_byte": round(tokens_per_byte, 6),
        "estimated_bpb": round(estimated_bpb, 4),
        "encoding": encoding_name,
    }

    logger.info(
        "Survey Quality Report: %d tokens, %d bytes, %.4f tokens/byte, BPB≈%.4f",
        n_tokens, n_bytes, tokens_per_byte, estimated_bpb,
    )
    return report
