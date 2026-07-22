"""
Rate limiter per API calls (Gemini, Semantic Scholar, GROBID).

Implementa un token-bucket sincrono e un semaforo per GROBID.
Supporta sia uso sync che async.

Uso::

    from paperhunter.utils.rate_limiter import (
        get_gemini_limiter, get_s2_limiter, get_grobid_semaphore
    )

    gemini = get_gemini_limiter()
    gemini.wait()  # Blocca se necessario per rispettare RPM

    s2 = get_s2_limiter()
    s2.wait()

    grobid_sem = get_grobid_semaphore()
    with grobid_sem:
        # max N richieste GROBID concorrenti
        ...
"""

from __future__ import annotations

import asyncio
import threading
import time
from functools import lru_cache
from typing import Optional

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token Bucket (sync)
# ---------------------------------------------------------------------------

class TokenBucketLimiter:
    """Rate limiter basato su token-bucket con supporto RPM.

    Thread-safe. Ogni chiamata a ``wait()`` consuma un token;
    se il bucket è vuoto, dorme fino a quando un token è disponibile.

    Args:
        rpm: Richieste al minuto consentite.
        name: Nome del servizio (per logging).
        daily_limit: Limite giornaliero opzionale.
    """

    def __init__(self, rpm: int, name: str = "default", daily_limit: Optional[int] = None):
        self.rpm = rpm
        self.name = name
        self.daily_limit = daily_limit

        # Token bucket
        self._interval = 60.0 / rpm  # secondi tra un token e l'altro
        self._max_tokens = rpm  # burst max = 1 minuto di token
        self._tokens = float(rpm)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

        # Contatori giornalieri
        self._daily_count = 0
        self._daily_reset_time = time.monotonic()

        logger.debug(f"Rate limiter '{name}' creato: {rpm} RPM, interval={self._interval:.2f}s")

    def _refill(self) -> None:
        """Riempi il bucket in base al tempo trascorso."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed / self._interval
        self._tokens = min(self._max_tokens, self._tokens + new_tokens)
        self._last_refill = now

        # Reset giornaliero ogni 24h
        if now - self._daily_reset_time >= 86400:
            self._daily_count = 0
            self._daily_reset_time = now

    def wait(self) -> float:
        """Attendi fino a quando un token è disponibile.

        Returns:
            Tempo di attesa effettivo in secondi (0.0 se nessuna attesa).

        Raises:
            RuntimeError: Se il limite giornaliero è esaurito.
        """
        with self._lock:
            # Check limite giornaliero
            if self.daily_limit and self._daily_count >= self.daily_limit:
                raise RuntimeError(
                    f"Rate limiter '{self.name}': limite giornaliero esaurito "
                    f"({self._daily_count}/{self.daily_limit})"
                )

            self._refill()

            wait_time = 0.0
            if self._tokens < 1.0:
                # Calcola tempo di attesa necessario
                deficit = 1.0 - self._tokens
                wait_time = deficit * self._interval
                logger.debug(
                    f"Rate limiter '{self.name}': attesa {wait_time:.2f}s "
                    f"(tokens={self._tokens:.2f})"
                )

        # Dormi fuori dal lock per non bloccare altri thread
        if wait_time > 0:
            time.sleep(wait_time)

        with self._lock:
            self._refill()
            self._tokens -= 1.0
            self._daily_count += 1

        return wait_time

    async def async_wait(self) -> float:
        """Versione asincrona di ``wait()``.

        Returns:
            Tempo di attesa effettivo in secondi.
        """
        with self._lock:
            if self.daily_limit and self._daily_count >= self.daily_limit:
                raise RuntimeError(
                    f"Rate limiter '{self.name}': limite giornaliero esaurito "
                    f"({self._daily_count}/{self.daily_limit})"
                )

            self._refill()

            wait_time = 0.0
            if self._tokens < 1.0:
                deficit = 1.0 - self._tokens
                wait_time = deficit * self._interval
                logger.debug(
                    f"Rate limiter '{self.name}': async attesa {wait_time:.2f}s"
                )

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        with self._lock:
            self._refill()
            self._tokens -= 1.0
            self._daily_count += 1

        return wait_time

    @property
    def stats(self) -> dict:
        """Statistiche correnti del limiter."""
        with self._lock:
            self._refill()
            return {
                "name": self.name,
                "rpm": self.rpm,
                "tokens_available": round(self._tokens, 2),
                "daily_count": self._daily_count,
                "daily_limit": self.daily_limit,
            }

    def reset(self) -> None:
        """Reset completo del limiter."""
        with self._lock:
            self._tokens = float(self._max_tokens)
            self._last_refill = time.monotonic()
            self._daily_count = 0
            self._daily_reset_time = time.monotonic()


class NoOpLimiter:
    """Limiter compatibile che non introduce attese o contatori fittizi."""

    name = "disabled"

    def wait(self) -> float:
        return 0.0

    async def async_wait(self) -> float:
        return 0.0

    @property
    def stats(self) -> dict:
        return {"name": self.name, "disabled": True}

    def reset(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Semaforo per GROBID (concorrenza limitata)
# ---------------------------------------------------------------------------

class GrobidSemaphore:
    """Semaforo per limitare le richieste GROBID concorrenti.

    Può essere usato come context manager::

        sem = GrobidSemaphore(max_concurrent=5)
        with sem:
            # richiesta GROBID
            ...
    """

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._active = 0
        self._lock = threading.Lock()

    def __enter__(self):
        self._semaphore.acquire()
        with self._lock:
            self._active += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            self._active -= 1
        self._semaphore.release()
        return False

    async def __aenter__(self):
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self.max_concurrent)
        await self._async_semaphore.acquire()
        with self._lock:
            self._active += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            self._active -= 1
        self._async_semaphore.release()
        return False

    @property
    def active_count(self) -> int:
        """Numero di richieste attualmente attive."""
        with self._lock:
            return self._active


# ---------------------------------------------------------------------------
# Singleton getters (configurati da config.yaml)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_gemini_limiter() -> TokenBucketLimiter:
    """Ottieni il rate limiter globale per Gemini API.

    Il daily_limit si applica solo se è configurata SOLO la key FREE.
    Se è presente anche la key PAID, il limite viene disabilitato
    perché la PAID non ha restrizioni giornaliere.
    """
    from paperhunter.config import settings

    cfg = settings.rate_limiting

    # Se esiste la key PAID, nessun daily limit
    effective_daily_limit = cfg.gemini_daily_limit
    if settings.google_api_key:
        effective_daily_limit = None

    return TokenBucketLimiter(
        rpm=cfg.gemini_rpm,
        name="gemini",
        daily_limit=effective_daily_limit,
    )


@lru_cache(maxsize=2)
def get_llm_limiter(provider: str | None = None) -> TokenBucketLimiter | NoOpLimiter:
    """Limiter del provider testuale selezionato.

    DeepSeek gestisce i 429 tramite il retry autonomo del client; applicare il
    bucket Gemini ne falserebbe sia la velocità sia il limite giornaliero.
    """
    from paperhunter.config import settings

    selected = provider or settings.llm_provider
    if selected == "google":
        return get_gemini_limiter()
    if selected == "deepseek":
        return NoOpLimiter()
    raise ValueError(f"Provider LLM non valido: {selected!r}")


@lru_cache(maxsize=2)
def get_embedding_limiter(provider: str | None = None) -> TokenBucketLimiter | NoOpLimiter:
    """Limiter coerente con il backend embedding configurato."""
    from paperhunter.config import settings

    selected = provider or settings.embedding.provider
    if selected == "google":
        return get_gemini_limiter()
    if selected == "huggingface":
        return NoOpLimiter()
    raise ValueError(f"Provider embedding non valido: {selected!r}")


@lru_cache(maxsize=1)
def get_s2_limiter() -> TokenBucketLimiter:
    """Ottieni il rate limiter globale per Semantic Scholar API.

    Configurato da ``config.yaml → rate_limiting.semantic_scholar_rpm``.
    """
    from paperhunter.config import settings

    cfg = settings.rate_limiting
    return TokenBucketLimiter(
        rpm=cfg.semantic_scholar_rpm,
        name="semantic_scholar",
    )


@lru_cache(maxsize=1)
def get_grobid_semaphore() -> GrobidSemaphore:
    """Ottieni il semaforo globale per GROBID.

    Configurato da ``config.yaml → rate_limiting.grobid_concurrent``.
    """
    from paperhunter.config import settings

    cfg = settings.rate_limiting
    return GrobidSemaphore(max_concurrent=cfg.grobid_concurrent)
