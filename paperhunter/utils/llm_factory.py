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

import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
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

class ApiKeyManager:
    """Gestisce la rotazione delle API key Google (FREE → PAID).

    Usa la chiave gratuita (GOOGLE_API_KEY_FREE) come primaria.
    Quando si raggiunge il rate limit, ruota alla chiave a pagamento
    (GOOGLE_API_KEY). Thread-safe, singleton.
    """

    _instance: "ApiKeyManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ApiKeyManager":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._keys: list[tuple[str, str]] = []
                inst._current_idx = 0
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    def _init_keys(self) -> None:
        if self._initialized:
            return
        from paperhunter.config import settings
        keys: list[tuple[str, str]] = []
        if settings.google_api_key_free:
            keys.append(("FREE", settings.google_api_key_free))
        if settings.google_api_key:
            keys.append(("PAID", settings.google_api_key))
        self._keys = keys
        self._initialized = True

    @property
    def current_key(self) -> str:
        """Restituisce la API key attualmente attiva."""
        self._init_keys()
        if not self._keys:
            raise ValueError(
                "Nessuna GOOGLE_API_KEY configurata. "
                "Imposta GOOGLE_API_KEY_FREE e/o GOOGLE_API_KEY nel file .env."
            )
        return self._keys[self._current_idx][1]

    @property
    def current_label(self) -> str:
        """Restituisce l'etichetta della key attiva (FREE/PAID)."""
        self._init_keys()
        if not self._keys:
            return "NONE"
        return self._keys[self._current_idx][0]

    def rotate(self) -> bool:
        """Ruota alla prossima API key disponibile.

        Returns:
            True se la rotazione è avvenuta, False se non ci sono altre chiavi.
        """
        with self._lock:
            self._init_keys()
            next_idx = self._current_idx + 1
            if next_idx < len(self._keys):
                old_label = self._keys[self._current_idx][0]
                self._current_idx = next_idx
                new_label = self._keys[self._current_idx][0]
                logger.warning(
                    f"🔑 API key rotata: {old_label} → {new_label}"
                )
                return True
            return False

    def reset(self) -> None:
        """Torna alla prima key (FREE) per il prossimo ciclo."""
        with self._lock:
            if self._current_idx != 0:
                old_label = self._keys[self._current_idx][0] if self._keys else "?"
                self._current_idx = 0
                new_label = self._keys[0][0] if self._keys else "?"
                logger.info(f"🔑 API key reset: {old_label} → {new_label}")

    @property
    def has_fallback(self) -> bool:
        """True se esiste un'altra key dopo quella corrente."""
        self._init_keys()
        return self._current_idx + 1 < len(self._keys)


def _get_key_manager() -> ApiKeyManager:
    """Ottieni il singleton ApiKeyManager."""
    return ApiKeyManager()


def _get_api_key() -> str:
    """Ottieni la Google API key attiva (FREE → PAID fallback).

    Returns:
        API key come stringa.

    Raises:
        ValueError: Se nessuna API key è configurata.
    """
    return _get_key_manager().current_key


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
    timeout_seconds: Optional[float] = None,
):
    """Crea un'istanza LLM Gemini Pro (task complessi).

    Modello potente per: summarization, academic writing, planning,
    analisi approfondite.

    Args:
        temperature: Override della temperatura (default da config.yaml).
        max_tokens: Override max output tokens (default da config.yaml).
        timeout_seconds: Override del timeout HTTP in secondi. Se None,
            usa il valore globale da config.yaml (gemini.timeout).

    Returns:
        Istanza ``GoogleGenAI`` configurata.
    """
    from llama_index.llms.google_genai import GoogleGenAI
    from paperhunter.config import settings

    cfg = settings.gemini
    temp = temperature if temperature is not None else cfg.temperature_pro
    tokens = max_tokens if max_tokens is not None else cfg.max_output_tokens_pro
    effective_timeout = timeout_seconds if timeout_seconds is not None else cfg.timeout

    llm = GoogleGenAI(
        model=cfg.pro_model,
        api_key=_get_api_key(),
        temperature=temp,
        max_tokens=tokens,
        http_options=_get_http_options(timeout_seconds=timeout_seconds),
        max_retries=0,  # Disabilita retry interni LlamaIndex — gestiamo noi in llm_call_with_retry
    )

    logger.debug(f"LLM Pro creato: {cfg.pro_model} (temp={temp}, timeout={effective_timeout}s)")
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
# OpenRouter fallback factory (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _get_openrouter_api_key() -> str:
    """Ottieni la OpenRouter API key dalla configurazione.

    Returns:
        API key come stringa.

    Raises:
        ValueError: Se la API key non è configurata.
    """
    from paperhunter.config import settings

    key = settings.openrouter_api_key
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY non configurata. "
            "Impostala nel file .env o come variabile d'ambiente."
        )
    return key


def _is_openrouter_available() -> bool:
    """Verifica se OpenRouter è configurato e abilitato.

    Returns:
        True se OpenRouter è abilitato e la API key è presente.
    """
    from paperhunter.config import settings

    if not settings.openrouter.enabled:
        return False
    return bool(settings.openrouter_api_key)


def _create_openrouter_client():
    """Crea un client OpenAI raw puntato a OpenRouter.

    Usa il SDK ``openai`` diretto (NON il wrapper LlamaIndex) per evitare
    la validazione dei nomi modello che blocca modelli non-OpenAI come
    ``google/gemini-3.1-pro-preview``.

    Returns:
        Istanza ``openai.OpenAI`` configurata per OpenRouter.
    """
    from openai import OpenAI as OpenAIClient
    from paperhunter.config import settings

    cfg = settings.openrouter
    return OpenAIClient(
        base_url=cfg.api_base,
        api_key=_get_openrouter_api_key(),
    )


def _get_openrouter_model_config(google_llm) -> Optional[dict]:
    """Determina modello e parametri OpenRouter equivalenti al modello Google.

    Args:
        google_llm: Istanza GoogleGenAI originale.

    Returns:
        Dict con chiavi ``model``, ``temperature``, ``max_tokens``
        o None se OpenRouter non è disponibile.
    """
    if not _is_openrouter_available():
        return None

    from paperhunter.config import settings
    cfg = settings.openrouter

    model_name = getattr(google_llm, "model", "")
    temperature = getattr(google_llm, "temperature", None)
    max_tokens = getattr(google_llm, "max_tokens", None)

    if "pro" in model_name.lower():
        return {
            "model": cfg.pro_model,
            "temperature": temperature if temperature is not None else cfg.temperature_pro,
            "max_tokens": max_tokens if max_tokens is not None else cfg.max_output_tokens_pro,
        }
    else:
        return {
            "model": cfg.flash_model,
            "temperature": temperature if temperature is not None else cfg.temperature_flash,
            "max_tokens": max_tokens if max_tokens is not None else cfg.max_output_tokens_flash,
        }


# ---------------------------------------------------------------------------
# Embedding factory (google-genai REST SDK)
# ---------------------------------------------------------------------------

def get_embedding_model(
    model_name: Optional[str] = None,
    embed_batch_size: int = 100,
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
# Continuation helper — gestione output lunghi troncati a MAX_TOKENS
# ---------------------------------------------------------------------------

# Directory di default per i file temporanei delle parti parziali.
_TEMP_OUTPUT_DIR: Path | None = None


def _get_temp_output_dir() -> Path:
    """Restituisce la directory per i file temporanei LLM.

    Usa ``output/temp_llm/`` nella root del progetto, creando la
    directory se non esiste.
    """
    global _TEMP_OUTPUT_DIR
    if _TEMP_OUTPUT_DIR is None:
        from paperhunter.config import settings
        base = Path(settings.output.base_dir)
        _TEMP_OUTPUT_DIR = base / "temp_llm"
    _TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _TEMP_OUTPUT_DIR


def _raw_complete_google(llm, prompt: str) -> tuple[str, str]:
    """Chiama il modello Google GenAI via SDK diretto, senza raise su MAX_TOKENS.

    Bypassa il check ``_error_if_finished_early`` di LlamaIndex per
    recuperare il testo parziale anche quando ``finish_reason`` è
    MAX_TOKENS.

    Args:
        llm: Istanza GoogleGenAI di LlamaIndex (ha ``_client`` e ``_generation_config``).
        prompt: Il prompt da inviare.

    Returns:
        Tupla ``(text, finish_reason_name)`` dove ``finish_reason_name``
        è una stringa come ``"STOP"``, ``"MAX_TOKENS"``, ecc.
    """
    from google.genai import types as genai_types

    client = llm._client
    config = genai_types.GenerateContentConfig(**llm._generation_config)

    response = client.models.generate_content(
        model=llm.model,
        contents=prompt,
        config=config,
    )

    if not response.candidates:
        return "", "NO_CANDIDATES"

    candidate = response.candidates[0]
    text = ""
    if candidate.content and candidate.content.parts:
        text = "".join(
            part.text for part in candidate.content.parts if hasattr(part, "text") and part.text
        )

    finish_reason = "UNKNOWN"
    if candidate.finish_reason:
        finish_reason = candidate.finish_reason.name

    return text, finish_reason


def _save_partial_output(text: str, session_dir: Path, part_num: int) -> Path:
    """Salva un output parziale in un file temporaneo.

    Args:
        text: Testo parziale da salvare.
        session_dir: Directory della sessione corrente.
        part_num: Numero della parte (0-indexed).

    Returns:
        Path del file creato.
    """
    file_path = session_dir / f"part_{part_num:02d}.txt"
    file_path.write_text(text, encoding="utf-8")
    logger.info(f"Output parziale salvato: {file_path} ({len(text)} chars)")
    return file_path


def _complete_with_continuation(
    llm,
    prompt: str,
    max_continuations: int = 5,
) -> str:
    """Completa un prompt gestendo il troncamento MAX_TOKENS con continuazioni.

    Quando la risposta viene troncata a MAX_TOKENS, salva il testo parziale
    in un file temporaneo e chiede al modello di continuare da dove si è
    fermato. Ripete fino a completamento o ``max_continuations`` raggiunte.

    Ogni parte viene salvata in ``output/temp_llm/<timestamp>/``.

    Args:
        llm: Istanza GoogleGenAI di LlamaIndex.
        prompt: Il prompt originale.
        max_continuations: Numero massimo di chiamate di continuazione.

    Returns:
        Testo completo (concatenazione di tutte le parti).

    Raises:
        RuntimeError: Se la prima chiamata non produce nessun testo.
    """
    # Crea directory per questa sessione di continuazione
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = llm.model.split("/")[-1] if "/" in llm.model else llm.model
    session_dir = _get_temp_output_dir() / f"{timestamp}_{model_short}"
    session_dir.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []

    for i in range(max_continuations + 1):
        _enforce_call_cooldown()

        if i == 0:
            # Prima chiamata: prompt originale
            current_prompt = prompt
        else:
            # Continuazione: includi le ultime righe per contesto
            accumulated = "".join(parts)
            # Prendi le ultime ~800 chars come contesto di overlap
            tail = accumulated[-800:] if len(accumulated) > 800 else accumulated
            current_prompt = (
                f"{prompt}\n\n"
                f"--- CONTINUAZIONE ---\n"
                f"La tua risposta precedente è stata troncata al limite di token. "
                f"Ecco la parte finale di ciò che hai generato finora:\n\n"
                f"...{tail}\n\n"
                f"CONTINUA ESATTAMENTE da dove ti sei fermato. "
                f"NON ripetere contenuto già generato. "
                f"Riprendi dalla riga/parola successiva al punto di troncamento."
            )

        # Retry transient errors within each continuation step
        _continuation_max_retries = 4
        for _retry in range(_continuation_max_retries + 1):
            try:
                text, finish_reason = _raw_complete_google(llm, current_prompt)
                break
            except Exception as _e:
                _err_str = str(_e).lower()
                _transient = (
                    "503", "500", "504", "429", "unavailable",
                    "overloaded", "temporarily", "timeout", "rate",
                )
                if any(kw in _err_str for kw in _transient) and _retry < _continuation_max_retries:
                    _delay = min(5.0 * (2 ** _retry), 120.0)
                    logger.warning(
                        f"Continuation part {i} retry {_retry + 1}/{_continuation_max_retries}: "
                        f"{_e}. Attendo {_delay:.0f}s..."
                    )
                    time.sleep(_delay)
                else:
                    raise

        if not text and i == 0:
            raise RuntimeError(
                f"Nessun testo nella risposta (finish_reason={finish_reason})"
            )

        if text:
            _save_partial_output(text, session_dir, i)
            parts.append(text)

        if finish_reason == "STOP":
            logger.info(
                f"Generazione completata in {i + 1} parti "
                f"({sum(len(p) for p in parts)} chars totali)"
            )
            break
        elif finish_reason == "MAX_TOKENS":
            logger.warning(
                f"Output troncato (MAX_TOKENS) — parte {i + 1}/{max_continuations + 1}, "
                f"{len(text)} chars. Continuo generazione..."
            )
        else:
            logger.warning(
                f"Finish reason inatteso: {finish_reason} — interrompo continuazione"
            )
            break

    # Salva output combinato
    combined = "".join(parts)
    combined_file = session_dir / "combined_output.txt"
    combined_file.write_text(combined, encoding="utf-8")
    logger.info(
        f"Output completo ({len(parts)} parti) salvato: {combined_file} "
        f"({len(combined)} chars)"
    )

    return combined


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
    openrouter_fallback: bool = True,
) -> str:
    """Esegui una chiamata LLM con retry e gestione errori configurabile.

    Il comportamento dipende dalla strategia configurata in
    ``gemini.retry_strategy`` (config.yaml):

    **Strategia "autonomous":**
    Solo Google API, nessuna interazione utente. In caso di errori
    transient, il sistema ritenta all'infinito con delay incrementali
    (backoff esponenziale cappato a ``max_delay``). Quando il modello
    torna disponibile, il delay si resetta. Ideale per esecuzioni
    non presidiate.

    **Strategia "interactive":**
    Dopo ``max_retries`` errori con Google, chiede all'utente se vuole
    usare OpenRouter come fallback. Se OpenRouter fallisce, torna a
    Google e ripete il ciclo. Ideale quando si è davanti al PC e si
    vuole velocità.

    Args:
        llm: Istanza LLM (Flash o Pro).
        prompt: Il prompt da inviare.
        max_retries: Numero massimo di tentativi per ciclo Google
            (default da config.yaml). Usato solo in strategia "interactive";
            in "autonomous" i retry sono illimitati.
        base_delay: Delay base in secondi (raddoppia ad ogni retry).
        max_delay: Delay massimo in secondi (cap al backoff esponenziale).
        openrouter_fallback: Se True (default), in strategia "interactive"
            propone OpenRouter dopo fallimento Google.

    Returns:
        Testo della risposta.

    Raises:
        Exception: Se la chiamata definitivamente non riesce.
    """
    from paperhunter.config import settings

    strategy = settings.gemini.retry_strategy  # "autonomous" | "interactive"

    if strategy == "autonomous":
        return _llm_call_autonomous(llm, prompt, base_delay, max_delay)
    else:
        retries = max_retries if max_retries is not None else settings.gemini.max_retries
        return _llm_call_interactive(
            llm, prompt, retries, base_delay, max_delay, openrouter_fallback,
        )


def _recreate_llm_with_new_key(llm):
    """Ricrea un'istanza LLM con la API key attualmente attiva nel KeyManager.

    Copia modello, temperatura, max_tokens dall'istanza originale.

    Args:
        llm: Istanza GoogleGenAI da clonare con nuova key.

    Returns:
        Nuova istanza GoogleGenAI con la key aggiornata.
    """
    from llama_index.llms.google_genai import GoogleGenAI

    new_key = _get_api_key()
    model = getattr(llm, "model", None)
    temperature = getattr(llm, "temperature", None)
    max_tokens = getattr(llm, "max_tokens", None)

    new_llm = GoogleGenAI(
        model=model,
        api_key=new_key,
        temperature=temperature,
        max_tokens=max_tokens,
        http_options=_get_http_options(),
        max_retries=0,
    )
    logger.info(
        f"LLM ricreato con key {_get_key_manager().current_label}: "
        f"{model} (temp={temperature})"
    )
    return new_llm


def _llm_call_autonomous(
    llm,
    prompt: str,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
) -> str:
    """Strategia autonoma: Google API con rotazione chiavi FREE → PAID.

    Prima usa GOOGLE_API_KEY_FREE. Quando raggiunge il rate limit,
    ruota automaticamente a GOOGLE_API_KEY. Il delay cresce
    esponenzialmente e si resetta appena il modello risponde.

    Non richiede nessuna interazione utente.

    Args:
        llm: Istanza LLM (Flash o Pro).
        prompt: Il prompt da inviare.
        base_delay: Delay base in secondi.
        max_delay: Delay massimo in secondi.

    Returns:
        Testo della risposta.
    """
    consecutive_errors = 0
    key_mgr = _get_key_manager()

    while True:
        try:
            _enforce_call_cooldown()
            response = llm.complete(prompt)
            if consecutive_errors > 0:
                logger.info(
                    f"Google API ({key_mgr.current_label}) tornata disponibile "
                    f"dopo {consecutive_errors} errori consecutivi. Delay resettato."
                )
            return str(response)
        except Exception as e:
            error_str = str(e).lower()

            # --- MAX_TOKENS: output troncato → continuazione ---
            if "max_tokens" in error_str:
                logger.warning(
                    "Output troncato (MAX_TOKENS). "
                    "Passo a modalità continuazione con file temporanei..."
                )
                try:
                    return _complete_with_continuation(llm, prompt)
                except Exception as cont_err:
                    logger.error(f"Continuazione fallita: {cont_err}")
                    consecutive_errors += 1
                    delay = min(base_delay * (2 ** consecutive_errors), max_delay)
                    logger.warning(
                        f"[autonomous] Errore #{consecutive_errors}. "
                        f"Attesa {delay:.0f}s prima di riprovare..."
                    )
                    time.sleep(delay)
                    continue

            # Errori quota/rate → prova rotazione chiavi
            quota_keywords = (
                "rate", "429", "quota", "resource_exhausted",
            )
            transient_keywords = (
                "503", "500", "504", "overloaded", "temporarily",
                "deadline", "unavailable",
            )
            is_quota = any(kw in error_str for kw in quota_keywords)
            is_transient = any(kw in error_str for kw in transient_keywords)

            if is_quota and key_mgr.has_fallback:
                logger.warning(
                    f"[autonomous] Quota/rate limit raggiunto con key "
                    f"{key_mgr.current_label}: {e}"
                )
                key_mgr.rotate()
                llm = _recreate_llm_with_new_key(llm)
                consecutive_errors = 0
                continue

            if is_quota or is_transient:
                consecutive_errors += 1
                delay = min(base_delay * (2 ** consecutive_errors), max_delay)
                logger.warning(
                    f"[autonomous] Google API ({key_mgr.current_label}) "
                    f"errore #{consecutive_errors}: {e}. "
                    f"Attesa {delay:.0f}s prima di riprovare..."
                )
                time.sleep(delay)
            else:
                # Errore non-transient → non recuperabile
                logger.error(f"[autonomous] Errore non recuperabile: {e}")
                raise


def _llm_call_interactive(
    llm,
    prompt: str,
    retries: int,
    base_delay: float,
    max_delay: float,
    openrouter_fallback: bool,
) -> str:
    """Strategia interattiva: Google API + fallback OpenRouter su conferma utente.

    Dopo ``retries`` errori Google, chiede all'utente se vuole usare
    OpenRouter. Se anche OpenRouter fallisce, torna a Google e ripete.

    Args:
        llm: Istanza LLM (Flash o Pro).
        prompt: Il prompt da inviare.
        retries: Numero massimo di tentativi per ciclo Google.
        base_delay: Delay base in secondi.
        max_delay: Delay massimo in secondi.
        openrouter_fallback: Se True, propone OpenRouter.

    Returns:
        Testo della risposta.
    """
    google_cycle = 0
    last_error = None

    while True:
        google_cycle += 1

        # --- Fase 1: Tentativi con Google API ---
        if google_cycle > 1:
            logger.info(
                f"Ritorno a Google API (ciclo {google_cycle}) dopo fallback OpenRouter."
            )

        for attempt in range(retries + 1):
            try:
                _enforce_call_cooldown()
                response = llm.complete(prompt)
                return str(response)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # --- MAX_TOKENS → continuazione ---
                if "max_tokens" in error_str:
                    logger.warning(
                        f"Output troncato (MAX_TOKENS) al tentativo {attempt + 1}. "
                        f"Passo a modalità continuazione con file temporanei..."
                    )
                    try:
                        return _complete_with_continuation(llm, prompt)
                    except Exception as cont_err:
                        logger.error(f"Continuazione fallita: {cont_err}")
                        last_error = cont_err
                        break

                quota_keywords = (
                    "rate", "429", "quota", "resource_exhausted",
                )
                transient_keywords = (
                    "503", "500", "504", "overloaded", "temporarily",
                    "deadline", "unavailable",
                )
                is_quota = any(kw in error_str for kw in quota_keywords)
                is_transient = any(kw in error_str for kw in transient_keywords)

                if is_quota and _get_key_manager().has_fallback:
                    logger.warning(
                        f"Quota raggiunta con key {_get_key_manager().current_label}. "
                        f"Rotazione chiave..."
                    )
                    _get_key_manager().rotate()
                    llm = _recreate_llm_with_new_key(llm)
                    break  # Ricomincia il ciclo for con nuova key

                if is_quota or is_transient:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    remaining = retries - attempt
                    logger.warning(
                        f"LLM call fallita (tentativo {attempt + 1}/{retries + 1}): {e}. "
                        f"Retry tra {delay:.0f}s... ({remaining} tentativi rimanenti)"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM call fallita con errore non recuperabile: {e}")
                    last_error = e
                    break
        else:
            # for loop exhausted without break → all retries consumed
            pass

        # Se abbiamo fatto break per key rotation, continua il while loop
        if _get_key_manager().current_label != "FREE" or not _get_key_manager().has_fallback:
            # Abbiamo già ruotato o non c'è più fallback → continua normalmente
            pass

        # --- Fase 2: Google esaurita, proponi fallback OpenRouter ---
        if not (openrouter_fallback and _is_openrouter_available()):
            logger.error(
                f"LLM call fallita dopo {retries + 1} tentativi "
                f"(ciclo {google_cycle}, senza fallback OpenRouter): {last_error}"
            )
            raise last_error

        logger.warning(
            f"Google API esaurita dopo {retries + 1} tentativi (ciclo {google_cycle}). "
            f"OpenRouter fallback disponibile."
        )
        try:
            answer = input(
                "\n⚠  Errore Google API. Vuoi usare OpenRouter come fallback? [S/n] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer not in ("", "s", "si", "sì", "y", "yes"):
            logger.info("Utente ha rifiutato fallback OpenRouter.")
            logger.error(
                f"LLM call fallita dopo {retries + 1} tentativi "
                f"(ciclo {google_cycle}): {last_error}"
            )
            raise last_error

        # --- Fase 3: Tenta OpenRouter ---
        logger.info("Utente ha confermato fallback OpenRouter.")
        try:
            result = _openrouter_fallback_call(llm, prompt)
            return result
        except Exception as or_error:
            logger.warning(
                f"Anche OpenRouter fallback fallito (ciclo {google_cycle}): {or_error}. "
                f"Riprovo con Google API..."
            )
            last_error = or_error


def _openrouter_fallback_call(
    original_llm,
    prompt: str,
) -> str:
    """Esegui una chiamata LLM via OpenRouter come fallback.

    Crea un LLM OpenRouter equivalente al modello Google originale
    e tenta la chiamata con retry dedicati.

    Args:
        original_llm: Istanza LLM Google originale (per determinare tier).
        prompt: Il prompt da inviare.

    Returns:
        Testo della risposta da OpenRouter.

    Raises:
        Exception: Se anche OpenRouter fallisce.
    """
    from paperhunter.config import settings

    or_config = _get_openrouter_model_config(original_llm)
    if or_config is None:
        raise RuntimeError(
            "OpenRouter fallback non disponibile: API key mancante o configurazione disabilitata."
        )

    client = _create_openrouter_client()
    or_retries = settings.openrouter.max_retries

    # --- Prima chiamata con retry per errori transient ---
    text, finish_reason = _openrouter_single_call(
        client, or_config, prompt, or_retries,
    )

    # --- Continuazione se troncato ---
    if finish_reason == "length":
        max_continuations = 5
        parts = [text]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = _get_temp_output_dir() / f"{timestamp}_openrouter"
        session_dir.mkdir(parents=True, exist_ok=True)
        _save_partial_output(text, session_dir, 0)

        for i in range(1, max_continuations + 1):
            accumulated = "".join(parts)
            tail = accumulated[-800:] if len(accumulated) > 800 else accumulated
            cont_prompt = (
                f"{prompt}\n\n"
                f"--- CONTINUAZIONE ---\n"
                f"La tua risposta precedente è stata troncata al limite di token. "
                f"Ecco la parte finale di ciò che hai generato finora:\n\n"
                f"...{tail}\n\n"
                f"CONTINUA ESATTAMENTE da dove ti sei fermato. "
                f"NON ripetere contenuto già generato. "
                f"Riprendi dalla riga/parola successiva al punto di troncamento."
            )
            try:
                cont_text, cont_reason = _openrouter_single_call(
                    client, or_config, cont_prompt, or_retries,
                )
            except Exception as cont_err:
                logger.warning(
                    f"OpenRouter continuazione parte {i} fallita: {cont_err}. "
                    f"Restituisco testo parziale ({len(accumulated)} chars)."
                )
                break

            if cont_text:
                _save_partial_output(cont_text, session_dir, i)
                parts.append(cont_text)

            if cont_reason != "length":
                logger.info(
                    f"OpenRouter generazione completata in {i + 1} parti "
                    f"({sum(len(p) for p in parts)} chars totali)"
                )
                break
            else:
                logger.warning(
                    f"OpenRouter output troncato — parte {i + 1}/{max_continuations + 1}, "
                    f"{len(cont_text)} chars. Continuo..."
                )

        text = "".join(parts)
        combined_file = session_dir / "combined_output.txt"
        combined_file.write_text(text, encoding="utf-8")
        logger.info(f"OpenRouter output combinato salvato: {combined_file} ({len(text)} chars)")

    return text


def _openrouter_single_call(
    client,
    or_config: dict,
    prompt: str,
    max_retries: int,
) -> tuple[str, str]:
    """Singola chiamata OpenRouter con retry per errori transient.

    Args:
        client: Istanza ``openai.OpenAI`` configurata per OpenRouter.
        or_config: Dict con ``model``, ``temperature``, ``max_tokens``.
        prompt: Il prompt da inviare.
        max_retries: Numero massimo di retry.

    Returns:
        Tupla ``(text, finish_reason)`` dove ``finish_reason`` è
        ``"stop"`` o ``"length"`` (troncato).

    Raises:
        Exception: Se tutti i retry falliscono.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            _enforce_call_cooldown()
            response = client.chat.completions.create(
                model=or_config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=or_config["temperature"],
                max_tokens=or_config["max_tokens"],
            )
            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"
            logger.info(
                f"OpenRouter call riuscita con {or_config['model']} "
                f"(tentativo {attempt + 1}/{max_retries + 1}, "
                f"finish_reason={finish_reason}, {len(text)} chars)"
            )
            return text, finish_reason
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            transient_keywords = (
                "rate", "429", "quota", "503", "500", "504",
                "overloaded", "temporarily", "timeout", "unavailable",
            )
            if any(kw in error_str for kw in transient_keywords):
                delay = min(2.0 * (2 ** attempt), 60.0)
                logger.warning(
                    f"OpenRouter fallback tentativo {attempt + 1}/{max_retries + 1} "
                    f"fallito: {e}. Retry tra {delay:.0f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"OpenRouter fallback errore non recuperabile: {e}")
                break

    logger.error(
        f"Anche OpenRouter fallback fallito dopo {max_retries + 1} tentativi: {last_error}"
    )
    raise last_error
