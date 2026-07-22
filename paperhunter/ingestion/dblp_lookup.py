"""
DBLP Lookup — Verifica e recupero BibTeX ufficiali da dblp.org.

Cerca ogni paper su DBLP per titolo/autore/DOI e, se trovato come
pubblicazione in conferenza o journal, scarica il .bib ufficiale.
Preferisce pubblicazioni formali (conferenze, journal) rispetto a
preprint informali. Se non trovato, restituisce None e il chiamante
può usare il fallback locale.

Uso::

    from paperhunter.ingestion.dblp_lookup import DblpLookup

    dblp = DblpLookup()
    result = dblp.lookup_bibtex(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        year=2017,
    )
    if result:
        print(result.bibtex)   # Entry BibTeX ufficiale
        print(result.dblp_key) # es. "conf/nips/VaswaniSPUJGKP17"
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import httpx

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

_DBLP_SEARCH_URL = "https://dblp.org/search/publ/api"
_DBLP_BIB_URL = "https://dblp.org/rec/{key}.bib"

_REQUEST_TIMEOUT = 20.0
_MIN_REQUEST_INTERVAL = 2.0  # secondi tra richieste (rispetta fair use DBLP)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 3.0  # secondi di backoff iniziale (raddoppia ad ogni retry)
_RATE_LIMIT_COOLDOWN = 30.0  # secondi di attesa dopo un 429

# Soglie di similarità titolo
_TITLE_SIM_THRESHOLD_WITH_AUTHOR = 0.82
_TITLE_SIM_THRESHOLD_TITLE_ONLY = 0.90

# Tipi DBLP preferiti (in ordine di preferenza)
_PREFERRED_TYPES = [
    "Conference and Workshop Papers",
    "Journal Articles",
]

# Campi DBLP da rimuovere dal BibTeX scaricato
_DBLP_STRIP_FIELDS = {"timestamp", "biburl", "bibsource"}


# ---------------------------------------------------------------------------
# Risultato lookup
# ---------------------------------------------------------------------------

@dataclass
class DblpResult:
    """Risultato di un lookup DBLP riuscito."""
    dblp_key: str
    bibtex: str
    dblp_type: str
    title: str
    similarity: float
    authors: list[str] | None = None
    year: int | None = None
    venue: str = ""
    doi: str | None = None
    record_url: str = ""


# ---------------------------------------------------------------------------
# Normalizzazione titolo
# ---------------------------------------------------------------------------

_NOISE_SUFFIX_RE = re.compile(
    r"\s*(Content Warning|Disclaimer|Note|Abstract|Keywords)\s*[:.].*$",
    re.IGNORECASE,
)


def _strip_title_noise(title: str) -> str:
    """Rimuovi metadati appesi al titolo (artefatti di parsing BibTeX)."""
    return _NOISE_SUFFIX_RE.sub("", title).strip() if title else ""


def _normalize_title(title: str) -> str:
    """Normalizza un titolo per il confronto di similarità.

    Rimuove accenti, punteggiatura, spazi extra, e converte in lowercase.
    Rimuove anche metadati appesi al titolo (Content Warning, ecc.)."""
    if not title:
        return ""
    text = _strip_title_noise(title)
    # Rimuovi accenti
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase
    text = text.lower()
    # Rimuovi punteggiatura (tranne trattini interni alle parole)
    text = re.sub(r"[^\w\s-]", " ", text)
    # Collassa spazi
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fix_broken_hyphens(title: str) -> str:
    """Ricomponi parole spezzate da trattini di fine riga.

    Individua pattern come 'STRAT-EGY' dove una parola è stata spezzata
    a fine riga e le ricompone. Preserva parole composte legittime come
    'SELF-EXPLORATION' o 'PRE-TRAINING' usando un whitelist di prefissi comuni.

    Usata SOLO come fallback nella ricerca DBLP, mai nella normalizzazione
    standard per evitare regressioni.
    """
    _COMMON_PREFIXES = {
        "pre", "post", "non", "multi", "co", "anti", "semi", "re", "un",
        "bi", "tri", "sub", "mid", "neo", "pan", "self", "cross", "over",
        "under", "out", "well", "fine", "long", "short", "high", "low",
        "meta", "super", "ultra", "inter", "intra", "auto", "mini", "full",
        "half", "near", "far", "end",
        # Abbreviazioni tecniche
        "llm", "vlm", "nlp", "gpt", "ai", "ml", "dl", "rl", "cv", "nlg",
        "api", "gui", "rag", "ocr", "pdf", "llms", "vlms",
        # Parole brevi comuni nelle parole composte
        "to", "of", "in", "on", "no", "an", "at", "as", "or", "by",
        "is", "it", "up", "do", "go", "so", "if", "be", "we", "my",
        "all", "and", "for", "the", "but", "not", "yet", "nor", "off",
        "own", "out", "new", "old", "big", "top", "end", "run", "day",
        "red", "one", "two", "way", "key", "set", "use", "get",
        "box", "hot", "net", "web", "war", "law", "age", "art",
        "free", "open", "code", "data", "text", "base", "like",
        "turn", "step", "time", "task", "test", "type", "term",
        "shot", "tune", "loop", "aware", "based", "level", "scale",
        "black", "white", "chain", "guided",
    }

    def _merge_if_broken(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        # Non mergiare se il lato sinistro è un prefisso comune
        if left.lower() in _COMMON_PREFIXES:
            return match.group(0)
        # Non mergiare se il lato destro è una parola/abbreviazione comune
        if right.lower() in _COMMON_PREFIXES:
            return match.group(0)
        # Merge se un lato è molto corto (≤3 caratteri)
        if len(left) <= 3 or len(right) <= 3:
            return left + right
        return match.group(0)
    return re.sub(r"(\w+)-(\w+)", _merge_if_broken, title)


def _title_similarity(title_a: str, title_b: str) -> float:
    """Calcola la similarità tra due titoli normalizzati."""
    norm_a = _normalize_title(title_a)
    norm_b = _normalize_title(title_b)
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def _extract_surname(author: str) -> str:
    """Estrae cognome dal nome autore."""
    if not author or not author.strip():
        return ""
    name = author.strip()
    if "," in name:
        return name.split(",")[0].strip()
    parts = name.split()
    return parts[-1] if parts else name


def _extract_hit_authors(hit: dict) -> list[str]:
    """Normalizza la forma variabile ``authors.author`` dell'API DBLP."""
    authors_block = hit.get("authors", {})
    if not isinstance(authors_block, dict):
        return []
    raw = authors_block.get("author", [])
    if isinstance(raw, (str, dict)):
        raw = [raw]
    authors: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = str(item.get("text") or item.get("name") or "")
        else:
            name = ""
        name = html.unescape(re.sub(r"\s+\d{4}$", "", name)).strip()
        if name:
            authors.append(name)
    return authors


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = value.strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.rstrip("./") or None


def _extract_hit_doi(hit: dict) -> str | None:
    raw = hit.get("ee", [])
    if isinstance(raw, str):
        raw = [raw]
    for value in raw if isinstance(raw, list) else []:
        normalized = _normalize_doi(str(value))
        if normalized and normalized.startswith("10."):
            return normalized
    return None


def _authors_match(target: list[str], candidate: list[str]) -> bool:
    """True se almeno un cognome non banale coincide."""
    target_surnames = {
        _normalize_title(_extract_surname(a))
        for a in target
        if _extract_surname(a)
    }
    candidate_surnames = {
        _normalize_title(_extract_surname(a))
        for a in candidate
        if _extract_surname(a)
    }
    return bool(target_surnames & candidate_surnames)


def _cache_key(title: str, year: int | None, first_author: str) -> str:
    """Genera chiave di cache stabile da titolo+anno+autore."""
    parts = [
        _normalize_title(title),
        str(year or ""),
        _normalize_title(first_author),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _strip_dblp_fields(bibtex: str) -> str:
    """Rimuove campi specifici DBLP (timestamp, biburl, bibsource) dal BibTeX."""
    for field in _DBLP_STRIP_FIELDS:
        # Rimuove righe come:   timestamp = {...},
        bibtex = re.sub(
            rf"^\s*{field}\s*=\s*\{{[^}}]*\}},?\s*\n?",
            "",
            bibtex,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    # Pulisci virgole dangling prima della chiusura }
    bibtex = re.sub(r",\s*\}", "\n}", bibtex)
    return bibtex


# ---------------------------------------------------------------------------
# DblpLookup
# ---------------------------------------------------------------------------

class DblpLookup:
    """Client per verificare e recuperare BibTeX ufficiali da DBLP.

    Implementa cache su disco e rate limiting.

    Args:
        cache_dir: Directory per la cache. Default: ``cache/dblp``.
        enabled: Se False, tutti i lookup restituiscono None (disabilitato).
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._last_request_time = 0.0

        if cache_dir is None:
            cache_dir = Path("cache") / "dblp"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "dblp_cache.json"
        self._cache: dict[str, dict] = self._load_cache()

        self._client = httpx.Client(
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "PaperHunter/2.0 (academic survey tool)"},
        )

        self._stats = {"hits": 0, "misses": 0, "errors": 0, "cache_hits": 0}

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, dict]:
        """Carica la cache da disco."""
        if self._cache_file.exists():
            try:
                data = json.loads(self._cache_file.read_text(encoding="utf-8"))
                logger.debug("Cache DBLP caricata: %d entry", len(data))
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Errore lettura cache DBLP: %s", exc)
        return {}

    def _save_cache(self) -> None:
        """Salva la cache su disco."""
        try:
            self._cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Errore scrittura cache DBLP: %s", exc)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _wait_rate_limit(self) -> None:
        """Attende il tempo necessario per rispettare il rate limit DBLP."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            sleep_time = _MIN_REQUEST_INTERVAL - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Esegue una richiesta HTTP con retry automatico su errori di connessione.

        Gestisce disconnessioni del server DBLP e 429 Too Many Requests.
        """
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            self._wait_rate_limit()
            try:
                resp = self._client.get(url, **kwargs) if method == "GET" else self._client.request(method, url, **kwargs)
                if resp.status_code == 429:
                    logger.debug("DBLP 429 rate limited (attempt %d/%d), waiting %ds", attempt + 1, _MAX_RETRIES, _RATE_LIMIT_COOLDOWN)
                    self._recreate_client()
                    time.sleep(_RATE_LIMIT_COOLDOWN)
                    continue
                return resp
            except httpx.RemoteProtocolError as exc:
                last_exc = exc
                logger.debug("DBLP connection reset (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, exc)
                self._recreate_client()
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))
            except httpx.RequestError as exc:
                last_exc = exc
                logger.debug("DBLP request error (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, exc)
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))
        if last_exc:
            raise last_exc
        raise httpx.RequestError("DBLP max retries exceeded")

    def _recreate_client(self) -> None:
        """Ricrea il client HTTP (utile dopo disconnessioni)."""
        try:
            self._client.close()
        except Exception:
            pass
        self._client = httpx.Client(
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "PaperHunter/2.0 (academic survey tool)"},
        )

    def _search_dblp(self, query: str, max_hits: int = 5) -> list[dict]:
        """Esegue una ricerca sull'API DBLP.

        Args:
            query: Stringa di ricerca.
            max_hits: Numero massimo di risultati.

        Returns:
            Lista di hit (dict con info del paper).
        """
        try:
            resp = self._request_with_retry(
                "GET",
                _DBLP_SEARCH_URL,
                params={"q": query, "format": "json", "h": str(max_hits)},
            )
            resp.raise_for_status()
            data = resp.json()

            hits_data = data.get("result", {}).get("hits", {})
            hit_list = hits_data.get("hit", [])

            # DBLP restituisce un singolo dict se c'è un solo risultato
            if isinstance(hit_list, dict):
                hit_list = [hit_list]

            return [h.get("info", {}) for h in hit_list if "info" in h]

        except httpx.HTTPStatusError as exc:
            logger.warning("DBLP search HTTP error %d: %s", exc.response.status_code, query)
            self._stats["errors"] += 1
            return []
        except (httpx.RequestError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("DBLP search error: %s — query: %s", exc, query)
            self._stats["errors"] += 1
            return []

    def _fetch_bibtex(self, dblp_key: str) -> str | None:
        """Scarica il BibTeX ufficiale di un record DBLP.

        Args:
            dblp_key: Chiave DBLP (es. ``conf/nips/VaswaniSPUJGKP17``).

        Returns:
            Stringa BibTeX o None se errore.
        """
        url = _DBLP_BIB_URL.format(key=dblp_key)
        try:
            resp = self._request_with_retry("GET", url)
            resp.raise_for_status()
            bibtex = resp.text.strip()
            if bibtex and bibtex.startswith("@"):
                return bibtex
            logger.warning("DBLP BibTeX vuoto o invalido per key=%s", dblp_key)
            return None
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("DBLP fetch bibtex error per key=%s: %s", dblp_key, exc)
            self._stats["errors"] += 1
            return None

    # ------------------------------------------------------------------
    # Selezione miglior match
    # ------------------------------------------------------------------

    def _select_best_hit(
        self,
        hits: list[dict],
        target_title: str,
        target_year: int | None,
        threshold: float,
        target_authors: list[str] | None = None,
        require_author_match: bool = False,
        require_exact_year: bool = False,
    ) -> tuple[dict | None, float]:
        """Seleziona il miglior hit DBLP per il paper cercato.

        Preferisce pubblicazioni in conferenza/journal. Tra i match con
        tipo preferito, sceglie quello con maggiore similarità di titolo.
        Se nessun tipo preferito, accetta anche preprint con soglia più alta.

        Args:
            hits: Lista di hit DBLP (info dicts).
            target_title: Titolo del paper cercato.
            target_year: Anno atteso (±1 di tolleranza).
            threshold: Soglia minima di similarità.

        Returns:
            Tupla (best_hit, similarity). best_hit è None se nessun match.
        """
        candidates: list[tuple[dict, float, int]] = []

        for hit in hits:
            hit_title = html.unescape(str(hit.get("title", "")))
            # DBLP a volte aggiunge un punto finale
            hit_title = hit_title.rstrip(".")
            sim = _title_similarity(target_title, hit_title)

            if sim < threshold:
                continue

            hit_authors = _extract_hit_authors(hit)
            if (
                require_author_match
                and target_authors
                and not _authors_match(target_authors, hit_authors)
            ):
                logger.debug(
                    "DBLP candidate scartato per autori: %s — %s",
                    hit_title[:60],
                    hit_authors[:3],
                )
                continue

            # Verifica anno: esatta in strict mode, altrimenti tolleranza ±1.
            hit_year_str = hit.get("year", "")
            if require_exact_year and target_year and not hit_year_str:
                continue
            if target_year and hit_year_str:
                try:
                    hit_year = int(hit_year_str)
                    if require_exact_year and hit_year != target_year:
                        continue
                    if not require_exact_year and abs(hit_year - target_year) > 1:
                        continue
                except ValueError:
                    continue

            # Priorità tipo: conferenza > journal > altro
            hit_type = hit.get("type", "")
            if hit_type in _PREFERRED_TYPES:
                type_priority = _PREFERRED_TYPES.index(hit_type)
            else:
                type_priority = len(_PREFERRED_TYPES)  # bassa priorità

            candidates.append((hit, sim, type_priority))

        if not candidates:
            return None, 0.0

        # Ordina: prima per tipo preferito (asc), poi per similarità (desc)
        candidates.sort(key=lambda x: (x[2], -x[1]))
        best = candidates[0]
        return best[0], best[1]

    # ------------------------------------------------------------------
    # Lookup pubblico
    # ------------------------------------------------------------------

    def lookup_bibtex(
        self,
        title: str,
        authors: list[str] | None = None,
        year: int | None = None,
        doi: str | None = None,
    ) -> DblpResult | None:
        """Cerca un paper su DBLP e restituisce il BibTeX ufficiale.

        Strategia di ricerca a cascata:
        1. Se DOI presente, cerca per DOI
        2. Cerca per titolo + primo autore
        3. Fallback: cerca solo per titolo (soglia più alta)

        Args:
            title: Titolo del paper.
            authors: Lista autori (opzionale).
            year: Anno di pubblicazione (opzionale, tolleranza ±1).
            doi: DOI del paper (opzionale).

        Returns:
            DblpResult con BibTeX ufficiale, o None se non trovato.
        """
        if not self._enabled or not title or not title.strip():
            return None

        try:
            from paperhunter.config import settings

            citability = settings.citability
            min_similarity = citability.minimum_title_similarity
            require_author = citability.require_author_match
            require_exact_year = citability.require_exact_year
        except Exception:
            min_similarity = _TITLE_SIM_THRESHOLD_TITLE_ONLY
            require_author = True
            require_exact_year = True

        normalized_authors = authors or []

        first_author = ""
        if authors:
            first_author = _extract_surname(authors[0])

        # Controlla cache
        ckey = _cache_key(title, year, first_author)
        if ckey in self._cache:
            cached = self._cache[ckey]
            if cached.get("bibtex") and cached.get("verification_version") == 2:
                cached_authors = cached.get("authors") or []
                cached_year = cached.get("year")
                cache_valid = cached.get("similarity", 0.0) >= min_similarity
                if require_author and normalized_authors:
                    cache_valid = cache_valid and _authors_match(
                        normalized_authors, cached_authors,
                    )
                if require_exact_year and year is not None:
                    cache_valid = cache_valid and cached_year is not None and year == cached_year
                if not cache_valid:
                    logger.warning("DBLP cache strict non valida: %s", title[:60])
                else:
                    self._stats["cache_hits"] += 1
                    logger.debug("DBLP cache verificata: %s", title[:60])
                    return DblpResult(
                        dblp_key=cached["dblp_key"],
                        bibtex=cached["bibtex"],
                        dblp_type=cached.get("dblp_type", ""),
                        title=cached.get("title", title),
                        similarity=cached.get("similarity", 1.0),
                        authors=cached_authors,
                        year=cached_year,
                        venue=cached.get("venue", ""),
                        doi=cached.get("doi"),
                        record_url=cached.get("record_url", ""),
                    )

        # --- Strategia 1: DOI ---
        if doi:
            result = self._try_doi_lookup(
                doi, title, year, normalized_authors, min_similarity,
                require_author, require_exact_year,
            )
            if result:
                self._cache_result(ckey, result)
                return result

        # --- Strategia 2: titolo + primo autore ---
        if first_author:
            result = self._try_title_author_lookup(
                title, first_author, year, normalized_authors, min_similarity,
                require_author, require_exact_year,
            )
            if result:
                self._cache_result(ckey, result)
                return result

        # --- Strategia 3: solo titolo (soglia più restrittiva) ---
        result = self._try_title_only_lookup(
            title, year, normalized_authors, min_similarity,
            require_author, require_exact_year,
        )
        if result:
            self._cache_result(ckey, result)
            return result

        # --- Strategia 4: titolo de-iphenato (artefatti OCR) ---
        dehyphenated = _fix_broken_hyphens(title)
        if dehyphenated != title:
            result = self._try_title_only_lookup(
                dehyphenated, year, normalized_authors, min_similarity,
                require_author, require_exact_year,
            )
            if result:
                self._cache_result(ckey, result)
                return result

        # I miss non vengono memorizzati: un errore di rete DBLP non deve
        # diventare un falso negativo persistente nei run successivi.
        self._stats["misses"] += 1
        logger.debug("DBLP: non trovato — %s", title[:80])
        return None

    def _try_doi_lookup(
        self, doi: str, title: str, year: int | None,
        authors: list[str], min_similarity: float,
        require_author: bool, require_exact_year: bool,
    ) -> DblpResult | None:
        """Cerca su DBLP per DOI."""
        query = doi.strip()
        hits = self._search_dblp(query, max_hits=3)
        if not hits:
            return None
        best, sim = self._select_best_hit(
            hits, title, year, threshold=min_similarity,
            target_authors=authors,
            require_author_match=require_author,
            require_exact_year=require_exact_year,
        )
        if best:
            return self._build_result(best, sim)
        return None

    def _try_title_author_lookup(
        self, title: str, first_author_surname: str, year: int | None,
        authors: list[str], min_similarity: float,
        require_author: bool, require_exact_year: bool,
    ) -> DblpResult | None:
        """Cerca su DBLP per titolo + primo autore.

        Prova prima con 10 parole, poi con 6 se non trova nulla
        (DBLP usa AND logic, quindi troppe parole possono causare miss).
        """
        clean_title = re.sub(r"[^\w\s]", " ", _strip_title_noise(title))
        all_words = clean_title.split()

        for n_words in (10, 6):
            words = all_words[:n_words]
            query = " ".join(words) + " " + first_author_surname
            hits = self._search_dblp(query, max_hits=7)
            if hits:
                best, sim = self._select_best_hit(
                    hits, title, year,
                    threshold=max(_TITLE_SIM_THRESHOLD_WITH_AUTHOR, min_similarity),
                    target_authors=authors,
                    require_author_match=require_author,
                    require_exact_year=require_exact_year,
                )
                if best:
                    return self._build_result(best, sim)
        return None

    def _try_title_only_lookup(
        self, title: str, year: int | None,
        authors: list[str], min_similarity: float,
        require_author: bool, require_exact_year: bool,
    ) -> DblpResult | None:
        """Cerca su DBLP solo per titolo (soglia più alta).

        Prova con 12 parole, poi 7 se non trova nulla.
        """
        clean_title = re.sub(r"[^\w\s]", " ", _strip_title_noise(title))
        all_words = clean_title.split()

        for n_words in (12, 7):
            words = all_words[:n_words]
            query = " ".join(words)
            hits = self._search_dblp(query, max_hits=5)
            if hits:
                best, sim = self._select_best_hit(
                    hits, title, year,
                    threshold=max(_TITLE_SIM_THRESHOLD_TITLE_ONLY, min_similarity),
                    target_authors=authors,
                    require_author_match=require_author,
                    require_exact_year=require_exact_year,
                )
                if best:
                    return self._build_result(best, sim)
        return None

    def _build_result(self, hit: dict, similarity: float) -> DblpResult | None:
        """Costruisce un DblpResult scaricando il BibTeX dal record DBLP."""
        dblp_key = hit.get("key", "")
        if not dblp_key:
            return None

        bibtex = self._fetch_bibtex(dblp_key)
        if not bibtex:
            return None

        # Rimuovi campi specifici DBLP
        bibtex = _strip_dblp_fields(bibtex)

        self._stats["hits"] += 1
        hit_title = html.unescape(str(hit.get("title", ""))).rstrip(".")
        dblp_type = hit.get("type", "")
        hit_authors = _extract_hit_authors(hit)
        try:
            hit_year = int(hit.get("year")) if hit.get("year") else None
        except (TypeError, ValueError):
            hit_year = None
        record_url = str(hit.get("url") or f"https://dblp.org/rec/{dblp_key}")
        logger.info(
            "DBLP match [%s] sim=%.2f tipo=%s — %s",
            dblp_key, similarity, dblp_type, hit_title[:60],
        )
        return DblpResult(
            dblp_key=dblp_key,
            bibtex=bibtex,
            dblp_type=dblp_type,
            title=hit_title,
            similarity=similarity,
            authors=hit_authors,
            year=hit_year,
            venue=html.unescape(str(hit.get("venue") or "")),
            doi=_extract_hit_doi(hit),
            record_url=record_url,
        )

    def _cache_result(self, ckey: str, result: DblpResult) -> None:
        """Salva un risultato positivo in cache."""
        self._cache[ckey] = {
            "verification_version": 2,
            "dblp_key": result.dblp_key,
            "bibtex": result.bibtex,
            "dblp_type": result.dblp_type,
            "title": result.title,
            "similarity": result.similarity,
            "authors": result.authors or [],
            "year": result.year,
            "venue": result.venue,
            "doi": result.doi,
            "record_url": result.record_url,
        }
        self._save_cache()

    # ------------------------------------------------------------------
    # Statistiche
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        """Statistiche di utilizzo del lookup."""
        return dict(self._stats)

    def close(self) -> None:
        """Chiude il client HTTP."""
        self._client.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Singleton globale
# ---------------------------------------------------------------------------

_global_lookup: DblpLookup | None = None


def get_dblp_lookup(enabled: bool = True) -> DblpLookup:
    """Restituisce il singleton globale di DblpLookup.

    Args:
        enabled: Se False, il lookup è disabilitato.

    Returns:
        Istanza condivisa di DblpLookup.
    """
    global _global_lookup
    if _global_lookup is None:
        _global_lookup = DblpLookup(enabled=enabled)
    return _global_lookup
