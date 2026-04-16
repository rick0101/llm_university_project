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

_REQUEST_TIMEOUT = 15.0
_MIN_REQUEST_INTERVAL = 1.1  # secondi tra richieste (rispetta fair use DBLP)

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


# ---------------------------------------------------------------------------
# Normalizzazione titolo
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """Normalizza un titolo per il confronto di similarità.

    Rimuove accenti, punteggiatura, spazi extra, e converte in lowercase.
    """
    if not title:
        return ""
    # Rimuovi accenti
    nfkd = unicodedata.normalize("NFKD", title)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase
    text = text.lower()
    # Rimuovi punteggiatura (tranne trattini interni alle parole)
    text = re.sub(r"[^\w\s-]", " ", text)
    # Collassa spazi
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

    def _search_dblp(self, query: str, max_hits: int = 5) -> list[dict]:
        """Esegue una ricerca sull'API DBLP.

        Args:
            query: Stringa di ricerca.
            max_hits: Numero massimo di risultati.

        Returns:
            Lista di hit (dict con info del paper).
        """
        self._wait_rate_limit()
        try:
            resp = self._client.get(
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
        self._wait_rate_limit()
        url = _DBLP_BIB_URL.format(key=dblp_key)
        try:
            resp = self._client.get(url)
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
            hit_title = hit.get("title", "")
            # DBLP a volte aggiunge un punto finale
            hit_title = hit_title.rstrip(".")
            sim = _title_similarity(target_title, hit_title)

            if sim < threshold:
                continue

            # Verifica anno (tolleranza ±1)
            hit_year_str = hit.get("year", "")
            if target_year and hit_year_str:
                try:
                    hit_year = int(hit_year_str)
                    if abs(hit_year - target_year) > 1:
                        continue
                except ValueError:
                    pass

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

        first_author = ""
        if authors:
            first_author = _extract_surname(authors[0])

        # Controlla cache
        ckey = _cache_key(title, year, first_author)
        if ckey in self._cache:
            cached = self._cache[ckey]
            if cached.get("bibtex"):
                self._stats["cache_hits"] += 1
                logger.debug("DBLP cache hit: %s", title[:60])
                return DblpResult(
                    dblp_key=cached["dblp_key"],
                    bibtex=cached["bibtex"],
                    dblp_type=cached.get("dblp_type", ""),
                    title=cached.get("title", title),
                    similarity=cached.get("similarity", 1.0),
                )
            elif cached.get("not_found"):
                self._stats["cache_hits"] += 1
                return None

        # --- Strategia 1: DOI ---
        if doi:
            result = self._try_doi_lookup(doi, title, year)
            if result:
                self._cache_result(ckey, result)
                return result

        # --- Strategia 2: titolo + primo autore ---
        if first_author:
            result = self._try_title_author_lookup(title, first_author, year)
            if result:
                self._cache_result(ckey, result)
                return result

        # --- Strategia 3: solo titolo (soglia più restrittiva) ---
        result = self._try_title_only_lookup(title, year)
        if result:
            self._cache_result(ckey, result)
            return result

        # Non trovato: cache il miss
        self._cache[ckey] = {"not_found": True, "title": title}
        self._save_cache()
        self._stats["misses"] += 1
        logger.debug("DBLP: non trovato — %s", title[:80])
        return None

    def _try_doi_lookup(
        self, doi: str, title: str, year: int | None,
    ) -> DblpResult | None:
        """Cerca su DBLP per DOI."""
        query = doi.strip()
        hits = self._search_dblp(query, max_hits=3)
        if not hits:
            return None
        best, sim = self._select_best_hit(
            hits, title, year, threshold=0.70,
        )
        if best:
            return self._build_result(best, sim)
        return None

    def _try_title_author_lookup(
        self, title: str, first_author_surname: str, year: int | None,
    ) -> DblpResult | None:
        """Cerca su DBLP per titolo + primo autore."""
        # Usa le prime parole significative del titolo + autore
        clean_title = re.sub(r"[^\w\s]", " ", title)
        words = clean_title.split()[:10]
        query = " ".join(words) + " " + first_author_surname
        hits = self._search_dblp(query, max_hits=7)
        if not hits:
            return None
        best, sim = self._select_best_hit(
            hits, title, year, threshold=_TITLE_SIM_THRESHOLD_WITH_AUTHOR,
        )
        if best:
            return self._build_result(best, sim)
        return None

    def _try_title_only_lookup(
        self, title: str, year: int | None,
    ) -> DblpResult | None:
        """Cerca su DBLP solo per titolo (soglia più alta)."""
        clean_title = re.sub(r"[^\w\s]", " ", title)
        words = clean_title.split()[:12]
        query = " ".join(words)
        hits = self._search_dblp(query, max_hits=5)
        if not hits:
            return None
        best, sim = self._select_best_hit(
            hits, title, year, threshold=_TITLE_SIM_THRESHOLD_TITLE_ONLY,
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
        hit_title = hit.get("title", "").rstrip(".")
        dblp_type = hit.get("type", "")
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
        )

    def _cache_result(self, ckey: str, result: DblpResult) -> None:
        """Salva un risultato positivo in cache."""
        self._cache[ckey] = {
            "dblp_key": result.dblp_key,
            "bibtex": result.bibtex,
            "dblp_type": result.dblp_type,
            "title": result.title,
            "similarity": result.similarity,
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
