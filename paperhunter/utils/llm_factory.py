"""
Factory centralizzata per istanze LLM e Embedding (LlamaIndex + Gemini).

Usa i plugin basati sul nuovo SDK ``google-genai`` (REST, non gRPC):
- ``llama-index-llms-google-genai``       → GoogleGenAI
- ``llama-index-embeddings-google-genai`` → GoogleGenAIEmbedding

Gestisce due tier:
- **Flash** (gemini-3-flash-preview): task semplici, classificazione,
  reranking, estrazione entità, verifica citazioni.
- **Pro** (gemini-3-pro-preview): task complessi, summarization,
  academic writing, planning.

Uso::

    from paperhunter.utils.llm_factory import get_flash_llm, get_pro_llm, get_embedding_model

    flash = get_flash_llm()
    pro   = get_pro_llm()
    embed = get_embedding_model()

    response = flash.complete("Riassumi in una frase:")
    vector   = embed.get_text_embedding("prompt injection attack")
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pro Availability Tracker
# ---------------------------------------------------------------------------

class ProAvailabilityTracker:
    """Tiene traccia dei fallimenti recenti di Gemini Pro.

    Se Pro ha fallito negli ultimi ``cooldown_seconds``, consiglia
    di saltare direttamente a Flash per evitare attese inutili.

    Thread-safe, singleton.
    """

    _instance: "ProAvailabilityTracker | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ProAvailabilityTracker":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._last_failure_time = 0.0
                inst._consecutive_failures = 0
                inst._cooldown_seconds = 300.0  # 5 minuti
                cls._instance = inst
            return cls._instance

    def record_failure(self) -> None:
        """Registra un fallimento di Pro."""
        with self._lock:
            self._last_failure_time = time.monotonic()
            self._consecutive_failures += 1
            logger.warning(
                f"Pro failure #{self._consecutive_failures} registrato. "
                f"Cooldown attivo per {self._cooldown_seconds:.0f}s."
            )

    def record_success(self) -> None:
        """Registra un successo di Pro → reset cooldown."""
        with self._lock:
            if self._consecutive_failures > 0:
                logger.info(
                    f"Pro tornato disponibile dopo {self._consecutive_failures} fallimenti."
                )
            self._consecutive_failures = 0
            self._last_failure_time = 0.0

    def is_pro_available(self) -> bool:
        """Controlla se Pro è probabilmente disponibile.

        Returns:
            True se non ci sono fallimenti recenti nel periodo di cooldown.
        """
        with self._lock:
            if self._consecutive_failures == 0:
                return True
            elapsed = time.monotonic() - self._last_failure_time
            # Cooldown dinamico: più fallimenti → cooldown più lungo
            effective_cooldown = self._cooldown_seconds * min(self._consecutive_failures, 5)
            if elapsed >= effective_cooldown:
                logger.info(
                    f"Pro cooldown scaduto ({elapsed:.0f}s >= {effective_cooldown:.0f}s). "
                    f"Riprovo Pro."
                )
                return True
            remaining = effective_cooldown - elapsed
            logger.info(
                f"Pro in cooldown: {remaining:.0f}s rimanenti "
                f"({self._consecutive_failures} fallimenti consecutivi). "
                f"Usando Flash."
            )
            return False

    def reset(self) -> None:
        """Reset completo del tracker."""
        with self._lock:
            self._last_failure_time = 0.0
            self._consecutive_failures = 0


def get_pro_tracker() -> ProAvailabilityTracker:
    """Ottieni il singleton ProAvailabilityTracker."""
    return ProAvailabilityTracker()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Ottieni la Google API key dalla configurazione.

    Returns:
        API key come stringa.

    Raises:
        ValueError: Se la API key non è configurata.
    """
    from paperhunter.config import settings

    key = settings.google_api_key
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY non configurata. "
            "Impostala nel file .env o come variabile d'ambiente."
        )
    return key


def _get_http_options(timeout_seconds: Optional[float] = None):
    """Crea HttpOptions con timeout adeguato.

    Il timeout di default è letto da config.yaml (gemini.timeout, in secondi)
    e convertito in millisecondi per l'SDK google-genai.
    Se ``timeout_seconds`` è fornito, sovrascrive il valore da config.

    IMPORTANTE: il retry interno dell'SDK google-genai è disabilitato
    (``attempts=1``) perché gestiamo noi i retry in ``llm_call_with_retry``.
    Senza questo, l'SDK fa 5 tentativi interni prima di propagare l'errore,
    e il nostro retry esterno ne fa altri 6, moltiplicando le chiamate
    (fino a 30 HTTP request per una singola operazione logica).

    Args:
        timeout_seconds: Override del timeout in secondi. Se None, usa
            il valore da config.yaml (gemini.timeout).
    """
    from google.genai import types
    from paperhunter.config import settings

    effective_timeout = timeout_seconds if timeout_seconds is not None else settings.gemini.timeout
    timeout_ms = effective_timeout * 1000  # secondi → millisecondi
    return types.HttpOptions(
        timeout=timeout_ms,
        retry_options=types.HttpRetryOptions(
            attempts=1,  # Nessun retry interno SDK — gestiamo noi
        ),
    )


# ---------------------------------------------------------------------------
# LLM factory (google-genai REST SDK)
# ---------------------------------------------------------------------------

def get_flash_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
):
    """Crea un'istanza LLM Gemini Flash (task semplici).

    Modello veloce e economico per: classificazione, reranking,
    estrazione entità, verifica citazioni.

    Args:
        temperature: Override della temperatura (default da config.yaml).
        max_tokens: Override max output tokens (default da config.yaml).
        timeout_seconds: Override del timeout HTTP in secondi. Se None,
            usa il valore globale da config.yaml (gemini.timeout).
            Per task Flash brevi (reranking, classificazione) si consiglia
            un valore più basso (es. 90s) rispetto al default Pro (600s).

    Returns:
        Istanza ``GoogleGenAI`` configurata.
    """
    from llama_index.llms.google_genai import GoogleGenAI
    from paperhunter.config import settings

    cfg = settings.gemini
    temp = temperature if temperature is not None else cfg.temperature_flash
    tokens = max_tokens if max_tokens is not None else cfg.max_output_tokens_flash
    effective_timeout = timeout_seconds if timeout_seconds is not None else cfg.timeout

    llm = GoogleGenAI(
        model=cfg.flash_model,
        api_key=_get_api_key(),
        temperature=temp,
        max_tokens=tokens,
        http_options=_get_http_options(timeout_seconds=timeout_seconds),
        max_retries=0,  # Disabilita retry interni LlamaIndex — gestiamo noi in llm_call_with_retry
    )

    logger.debug(f"LLM Flash creato: {cfg.flash_model} (temp={temp}, timeout={effective_timeout}s)")
    return llm


def get_pro_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    """Crea un'istanza LLM Gemini Pro (task complessi).

    Modello potente per: summarization, academic writing, planning,
    analisi approfondite.

    Args:
        temperature: Override della temperatura (default da config.yaml).
        max_tokens: Override max output tokens (default da config.yaml).

    Returns:
        Istanza ``GoogleGenAI`` configurata.
    """
    from llama_index.llms.google_genai import GoogleGenAI
    from paperhunter.config import settings

    cfg = settings.gemini
    temp = temperature if temperature is not None else cfg.temperature_pro
    tokens = max_tokens if max_tokens is not None else cfg.max_output_tokens_pro

    llm = GoogleGenAI(
        model=cfg.pro_model,
        api_key=_get_api_key(),
        temperature=temp,
        max_tokens=tokens,
        http_options=_get_http_options(),
        max_retries=0,  # Disabilita retry interni LlamaIndex — gestiamo noi in llm_call_with_retry
    )

    logger.debug(f"LLM Pro creato: {cfg.pro_model} (temp={temp}, timeout={cfg.timeout}s)")
    return llm


def get_llm(tier: str = "flash", **kwargs):
    """Ottieni un LLM per tier ('flash' o 'pro').

    Args:
        tier: ``'flash'`` per task semplici, ``'pro'`` per task complessi.
        **kwargs: Argomenti passati a ``get_flash_llm()`` / ``get_pro_llm()``.

    Returns:
        Istanza LLM configurata.

    Raises:
        ValueError: Se tier non è ``'flash'`` o ``'pro'``.
    """
    if tier == "flash":
        return get_flash_llm(**kwargs)
    elif tier == "pro":
        return get_pro_llm(**kwargs)
    else:
        raise ValueError(f"Tier LLM non valido: {tier!r}. Usa 'flash' o 'pro'.")


# ---------------------------------------------------------------------------
# Embedding factory (google-genai REST SDK)
# ---------------------------------------------------------------------------

def get_embedding_model(
    model_name: Optional[str] = None,
    embed_batch_size: int = 10,
):
    """Crea un'istanza del modello di embedding Gemini.

    Usa ``gemini-embedding-001`` (3072 dimensioni) per default.

    Args:
        model_name: Override del nome modello (default da config.yaml).
        embed_batch_size: Dimensione batch per embedding multipli.

    Returns:
        Istanza ``GoogleGenAIEmbedding`` configurata.
    """
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from paperhunter.config import settings

    name = model_name or settings.gemini.embedding_model

    embed_model = GoogleGenAIEmbedding(
        model_name=name,
        api_key=_get_api_key(),
        embed_batch_size=embed_batch_size,
        http_options=_get_http_options(),
    )

    logger.debug(f"Embedding model creato: {name} (batch_size={embed_batch_size})")
    return embed_model


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

# Cooldown minimo tra chiamate LLM successive (secondi).
# Evita di bombardare l'API con richieste ravvicinate che
# provocano risposte 503 UNAVAILABLE a cascata.
_MIN_CALL_INTERVAL = 5.0  # secondi
_last_llm_call_time = 0.0
_call_time_lock = threading.Lock()


def _enforce_call_cooldown() -> None:
    """Garantisce un intervallo minimo tra chiamate LLM successive.

    Se l'ultima chiamata è avvenuta meno di ``_MIN_CALL_INTERVAL``
    secondi fa, dorme per il tempo rimanente. Questo evita che
    chiamate rapide successive provochino 503 UNAVAILABLE.
    """
    global _last_llm_call_time
    with _call_time_lock:
        now = time.monotonic()
        elapsed = now - _last_llm_call_time
        if elapsed < _MIN_CALL_INTERVAL and _last_llm_call_time > 0:
            wait = _MIN_CALL_INTERVAL - elapsed
            logger.debug(f"Cooldown pre-chiamata: attesa {wait:.1f}s")
            time.sleep(wait)
        _last_llm_call_time = time.monotonic()


def llm_call_with_retry(
    llm,
    prompt: str,
    max_retries: Optional[int] = None,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
) -> str:
    """Esegui una chiamata LLM con retry e backoff esponenziale cappato.

    Gestisce errori di rate limiting e transient errors.
    Il delay tra retry cresce esponenzialmente ma è cappato a ``max_delay``.

    Prima di ogni tentativo viene applicato un cooldown minimo di
    ``_MIN_CALL_INTERVAL`` secondi dall'ultima chiamata, per evitare
    di bombardare l'API con richieste ravvicinate.

    Il retry interno dell'SDK google-genai è disabilitato (attempts=1
    in HttpOptions), quindi questa è l'UNICA fonte di retry.

    Args:
        llm: Istanza LLM (Flash o Pro).
        prompt: Il prompt da inviare.
        max_retries: Numero massimo di tentativi (default da config.yaml).
        base_delay: Delay base in secondi (raddoppia ad ogni retry).
        max_delay: Delay massimo in secondi (cap al backoff esponenziale).

    Returns:
        Testo della risposta.

    Raises:
        Exception: Se tutti i tentativi falliscono.
    """
    from paperhunter.config import settings

    retries = max_retries if max_retries is not None else settings.gemini.max_retries

    last_error = None
    for attempt in range(retries + 1):
        try:
            # Cooldown pre-chiamata: evita 503 da richieste troppo ravvicinate
            _enforce_call_cooldown()
            response = llm.complete(prompt)
            return str(response)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Rate limiting o errori transient → retry con backoff
            transient_keywords = (
                "rate", "429", "quota", "resource_exhausted",
                "503", "500", "504", "overloaded", "temporarily",
                "deadline", "unavailable", "max_tokens",
            )
            if any(kw in error_str for kw in transient_keywords):
                delay = min(base_delay * (2 ** attempt), max_delay)
                remaining = retries - attempt
                logger.warning(
                    f"LLM call fallita (tentativo {attempt + 1}/{retries + 1}): {e}. "
                    f"Retry tra {delay:.0f}s... ({remaining} tentativi rimanenti)"
                )
                time.sleep(delay)
            else:
                # Errore non-transient: non riprovare
                logger.error(f"LLM call fallita con errore non recuperabile: {e}")
                raise

    logger.error(f"LLM call fallita dopo {retries + 1} tentativi: {last_error}")
    raise last_error
