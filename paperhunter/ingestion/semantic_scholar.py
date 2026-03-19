"""
Semantic Scholar Enrichment — Arricchimento metadati da S2 Academic Graph API.

Cerca ogni paper tramite titolo, recupera metadati estesi (venue, journal,
citation count, fields of study, BibTeX, DOI) e calcola citation velocity.
Integra cache locale JSON per evitare ri-query.

Uso:
    from paperhunter.ingestion.semantic_scholar import SemanticScholarEnricher

    enricher = SemanticScholarEnricher()

    # Arricchisci un singolo paper
    paper = enricher.enrich_paper(paper)
    print(paper.venue, paper.doi, paper.citations)

    # Cerca un paper per titolo
    result = enricher.search_paper("Universal and Transferable Adversarial Attacks")
    print(result)

    # Batch enrichment
    papers = enricher.enrich_batch(papers_list)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from paperhunter.config import settings
from paperhunter.models import Paper, SourceType
from paperhunter.utils.logging_setup import get_logger
from paperhunter.utils.rate_limiter import get_s2_limiter

logger = get_logger(__name__)


# =============================================================================
# Cache locale per risultati S2
# =============================================================================

class S2Cache:
    """Cache locale su disco per i risultati Semantic Scholar.

    Organizzata per paper_id (S2 ID). Evita ri-query per paper già arricchiti.
    Struttura: {cache_dir}/{s2_paper_id}.json
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir or settings.semantic_scholar.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict] = {}

    def _key_to_path(self, key: str) -> Path:
        """Converte una chiave in percorso file (sanitizzando caratteri speciali)."""
        safe_key = re.sub(r'[^\w\-.]', '_', key)
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> dict | None:
        """Recupera un risultato dalla cache.

        Args:
            key: Chiave di lookup (S2 paper ID o titolo normalizzato).

        Returns:
            Dict con i dati S2, o None se non in cache.
        """
        # Prima controlla cache in memoria
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Poi controlla su disco
        path = self._key_to_path(key)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._memory_cache[key] = data
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Cache read error per %s: %s", key, e)
                return None
        return None

    def put(self, key: str, data: dict) -> None:
        """Salva un risultato nella cache.

        Args:
            key: Chiave di lookup.
            data: Dati S2 da salvare.
        """
        self._memory_cache[key] = data
        path = self._key_to_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except OSError as e:
            logger.debug("Cache write error per %s: %s", key, e)

    def has(self, key: str) -> bool:
        """Verifica se una chiave è in cache."""
        if key in self._memory_cache:
            return True
        return self._key_to_path(key).exists()

    def stats(self) -> dict[str, int]:
        """Statistiche della cache."""
        cached_files = list(self.cache_dir.glob("*.json"))
        return {
            "files_on_disk": len(cached_files),
            "entries_in_memory": len(self._memory_cache),
        }


# =============================================================================
# Fuzzy title matching
# =============================================================================

def _normalize_title(title: str) -> str:
    """Normalizza un titolo per il confronto fuzzy.

    Rimuove punteggiatura, spazi multipli, e converte a lowercase.
    """
    title = title.lower().strip()
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title


def _title_similarity(title_a: str, title_b: str) -> float:
    """Calcola la similarità tra due titoli (Jaccard su token).

    Returns:
        Score tra 0.0 e 1.0.
    """
    tokens_a = set(_normalize_title(title_a).split())
    tokens_b = set(_normalize_title(title_b).split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# =============================================================================
# Semantic Scholar Enricher
# =============================================================================

class SemanticScholarEnricher:
    """Client per arricchire paper con metadati da Semantic Scholar API.

    Funzionalità:
    - Ricerca per titolo con fuzzy matching
    - Recupero dettagli completi (venue, journal, citations, BibTeX, ecc.)
    - Calcolo citation velocity (citazioni/anno)
    - Cache locale per evitare ri-query
    - Rate limiting integrato (100 RPM)

    Uso:
        enricher = SemanticScholarEnricher()
        enriched_paper = enricher.enrich_paper(paper)
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        """Inizializza l'enricher.

        Args:
            api_key: API key S2 (opzionale, da .env se non specificata).
            cache_dir: Directory per la cache locale.
        """
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.base_url = settings.semantic_scholar.base_url
        self.timeout = settings.semantic_scholar.timeout
        self.max_retries = settings.semantic_scholar.max_retries
        self.fields = settings.semantic_scholar.fields
        self.search_fields = settings.semantic_scholar.search_fields
        self._limiter = get_s2_limiter()
        self._cache = S2Cache(cache_dir)

        # Headers HTTP
        self._headers: dict[str, str] = {
            "User-Agent": "PaperHunter/2.0 (Academic Research Tool; LLM Security Survey)",
        }
        if self.api_key:
            self._headers["x-api-key"] = self.api_key
            logger.debug("S2 Enricher inizializzato con API key")
        else:
            logger.info(
                "S2 Enricher senza API key — rate limit più restrittivo. "
                "Ottieni una chiave su https://www.semanticscholar.org/product/api"
            )

    # -------------------------------------------------------------------------
    # Metodi HTTP con retry e rate limiting
    # -------------------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        retries_left: int | None = None,
    ) -> dict | None:
        """Esegue una richiesta HTTP con rate limiting e retry.

        Args:
            method: "GET" o "POST".
            url: URL completo.
            params: Query parameters.
            retries_left: Tentativi rimanenti.

        Returns:
            JSON response come dict, o None se tutti i retry falliscono.
        """
        if retries_left is None:
            retries_left = self.max_retries

        for attempt in range(retries_left + 1):
            # Rate limiting
            self._limiter.wait()

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(
                        method=method,
                        url=url,
                        params=params,
                        headers=self._headers,
                    )

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 404:
                    logger.debug("S2 404: risorsa non trovata per %s", url)
                    return None

                if response.status_code == 429:
                    # Rate limit hit — attesa esponenziale
                    wait = min(2 ** (attempt + 1) * 5, 120)
                    logger.warning(
                        "S2 rate limit (429). Attesa %ds (tentativo %d/%d)",
                        wait, attempt + 1, retries_left + 1,
                    )
                    time.sleep(wait)
                    continue

                if response.status_code >= 500:
                    # Server error — retry
                    wait = 2 ** attempt * 2
                    logger.warning(
                        "S2 server error %d. Retry in %ds (tentativo %d/%d)",
                        response.status_code, wait, attempt + 1, retries_left + 1,
                    )
                    time.sleep(wait)
                    continue

                # Altri errori (400, 403, ecc.)
                logger.error(
                    "S2 errore HTTP %d: %s",
                    response.status_code,
                    response.text[:200],
                )
                return None

            except httpx.TimeoutException:
                wait = 2 ** attempt * 2
                logger.warning(
                    "S2 timeout. Retry in %ds (tentativo %d/%d)",
                    wait, attempt + 1, retries_left + 1,
                )
                time.sleep(wait)
                continue

            except httpx.ConnectError as e:
                logger.error("S2 connection error: %s", e)
                return None

            except Exception as e:
                logger.error("S2 errore imprevisto: %s", e)
                return None

        logger.error("S2 tutti i retry esauriti per %s", url)
        return None

    # -------------------------------------------------------------------------
    # Ricerca paper per titolo
    # -------------------------------------------------------------------------

    def search_paper(
        self,
        title: str,
        authors: list[str] | None = None,
        limit: int = 5,
    ) -> dict | None:
        """Cerca un paper su S2 per titolo e restituisce il match migliore.

        Usa fuzzy matching per gestire differenze nel titolo (punteggiatura,
        abbreviazioni, ecc.).

        Args:
            title: Titolo del paper da cercare.
            authors: Lista di autori (opzionale, per disambiguazione).
            limit: Numero massimo di risultati dalla search API.

        Returns:
            Dict con i dati S2 del paper trovato, o None se non trovato.
        """
        # Controlla cache per titolo normalizzato
        cache_key = f"search_{_normalize_title(title)[:80]}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("S2 search cache hit: %s", title[:60])
            return cached

        # Pulisci titolo per la query
        query_title = title.strip()
        if len(query_title) < 5:
            logger.warning("Titolo troppo corto per la ricerca S2: '%s'", title)
            return None

        data = self._request(
            "GET",
            f"{self.base_url}/paper/search",
            params={
                "query": query_title,
                "limit": limit,
                "fields": self.search_fields,
            },
        )

        if not data or not data.get("data"):
            logger.debug("S2 search: nessun risultato per '%s'", title[:60])
            return None

        # Trova il match migliore con fuzzy matching
        best_match = None
        best_score = 0.0

        for candidate in data["data"]:
            candidate_title = candidate.get("title", "")
            score = _title_similarity(title, candidate_title)

            # Bonus se gli autori corrispondono
            if authors and candidate.get("authors"):
                candidate_authors = [
                    a.get("name", "").lower() for a in candidate["authors"]
                ]
                # Controlla se almeno il primo autore corrisponde
                first_author_lower = authors[0].lower() if authors else ""
                if any(first_author_lower in ca for ca in candidate_authors):
                    score += 0.15

            if score > best_score:
                best_score = score
                best_match = candidate

        # Soglia minima di similarità
        if best_score < 0.5:
            logger.debug(
                "S2 search: nessun match sufficientemente simile per '%s' "
                "(best score: %.2f)",
                title[:60], best_score,
            )
            return None

        logger.debug(
            "S2 search match: '%s' → '%s' (score: %.2f)",
            title[:50],
            best_match.get("title", "")[:50],
            best_score,
        )

        # Salva in cache
        self._cache.put(cache_key, best_match)
        return best_match

    # -------------------------------------------------------------------------
    # Dettagli completi di un paper
    # -------------------------------------------------------------------------

    def get_paper_details(self, paper_id: str) -> dict | None:
        """Recupera i dettagli completi di un paper da S2.

        Args:
            paper_id: S2 paper ID (es. "649def34f8be52c8b66281af98ae884c09aef38b")
                       oppure "DOI:10.xxx" o "ArXiv:2306.xxxxx".

        Returns:
            Dict con tutti i campi configurati in config.yaml, o None.
        """
        # Controlla cache
        cache_key = f"details_{paper_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("S2 details cache hit: %s", paper_id[:30])
            return cached

        data = self._request(
            "GET",
            f"{self.base_url}/paper/{paper_id}",
            params={"fields": self.fields},
        )

        if data:
            self._cache.put(cache_key, data)

        return data

    # -------------------------------------------------------------------------
    # Citation velocity
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_citation_velocity(
        citation_count: int | None,
        year: int | None,
    ) -> float | None:
        """Calcola le citazioni per anno (citation velocity).

        Args:
            citation_count: Numero totale di citazioni.
            year: Anno di pubblicazione.

        Returns:
            Citazioni/anno (float), o None se i dati sono insufficienti.
        """
        if citation_count is None or year is None:
            return None

        current_year = datetime.now().year
        years_since = current_year - year
        if years_since <= 0:
            # Paper dell'anno corrente — velocity = citations stesse
            return float(citation_count)

        return round(citation_count / years_since, 2)

    # -------------------------------------------------------------------------
    # Parsing robusto della data di pubblicazione (Task 1.2)
    # -------------------------------------------------------------------------

    _ARXIV_YEAR_RE = re.compile(r"^(\d{4})\.")
    _DOI_YEAR_RE = re.compile(r"/(\d{4})[./]")

    @staticmethod
    def _parse_publication_date(
        raw: str | None,
        year_fallback: int | None = None,
    ) -> tuple[int | None, int | None]:
        """Parsa la data di pubblicazione S2 in (year, month).

        Gestisce formati: "YYYY-MM-DD", "YYYY-MM", "YYYY", None.
        Non lancia mai eccezioni: ritorna (None, None) in caso di errore.

        Args:
            raw: Stringa `publicationDate` da S2 (es. "2023-09-15", "2023", None).
            year_fallback: Campo `year` intero da S2 da usare se raw è None.

        Returns:
            Tupla (year, month) — ciascuno può essere None.
        """
        if raw is None:
            # Fallback al campo year intero S2
            if year_fallback is not None:
                try:
                    return int(year_fallback), None
                except (ValueError, TypeError):
                    pass
            return None, None

        raw = raw.strip()
        # Formato YYYY-MM-DD
        if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
            try:
                dt = datetime.fromisoformat(raw)
                return dt.year, dt.month
            except (ValueError, TypeError):
                pass

        # Formato YYYY-MM
        if len(raw) == 7 and raw[4] == "-":
            try:
                parts = raw.split("-")
                year = int(parts[0])
                month = int(parts[1])
                if 1990 <= year <= 2035 and 1 <= month <= 12:
                    return year, month
            except (ValueError, TypeError, IndexError):
                pass

        # Formato YYYY (solo anno)
        if len(raw) == 4:
            try:
                year = int(raw)
                if 1990 <= year <= 2035:
                    return year, None
            except (ValueError, TypeError):
                pass

        # Fallback: prova a estrarre l'anno con regex
        match = re.search(r"\b(20[0-2]\d)\b", raw)
        if match:
            try:
                return int(match.group(1)), None
            except (ValueError, TypeError):
                pass

        # Ultima spiaggia: fallback intero
        if year_fallback is not None:
            try:
                return int(year_fallback), None
            except (ValueError, TypeError):
                pass

        return None, None

    # -------------------------------------------------------------------------
    # Enrichment singolo paper
    # -------------------------------------------------------------------------

    def enrich_paper(
        self,
        paper: Paper,
        skip_if_enriched: bool = True,
    ) -> Paper:
        """Arricchisce un paper con metadati da Semantic Scholar.

        Workflow:
        1. Cerca il paper per titolo (+ primo autore per disambiguazione)
        2. Se trovato, recupera dettagli completi
        3. Popola i campi estesi del Paper (venue, journal, doi, citations, ecc.)
        4. Calcola citation velocity
        5. Estrae BibTeX se disponibile

        Args:
            paper: Paper da arricchire.
            skip_if_enriched: Se True, salta paper che hanno già semantic_scholar_id.

        Returns:
            Paper arricchito (nuovo oggetto, l'originale non viene modificato).
        """
        # Skip se già arricchito
        if skip_if_enriched and paper.semantic_scholar_id:
            logger.debug("Paper già arricchito, skip: %s", paper.title[:50])
            return paper

        # Step 1: Cerca per titolo
        search_result = self.search_paper(
            title=paper.title,
            authors=paper.authors if paper.authors else None,
        )

        if not search_result:
            logger.info("S2: paper non trovato — '%s'", paper.title[:60])
            return paper

        # Ottieni S2 paper ID
        s2_id = search_result.get("paperId")
        if not s2_id:
            logger.debug("S2: nessun paperId nel risultato search")
            return paper

        # Step 2: Dettagli completi
        details = self.get_paper_details(s2_id)
        if not details:
            # Fallback: usa i dati dalla search
            details = search_result

        # Step 3: Popola i campi
        update_data = self._extract_enrichment_data(details)

        # Step 4: Citation velocity
        citations = update_data.get("citations") or paper.citations
        year = update_data.get("year") or paper.year
        velocity = self.compute_citation_velocity(citations, year)
        if velocity is not None:
            update_data["citation_velocity"] = velocity

        # Crea nuovo paper con dati arricchiti (non sovrascrivere valori già presenti)
        paper_dict = paper.model_dump()
        for key, value in update_data.items():
            if value is not None and (paper_dict.get(key) is None or paper_dict.get(key) == "" or paper_dict.get(key) == []):
                paper_dict[key] = value

        enriched = Paper(**paper_dict)

        logger.info(
            "S2 enrichment OK: '%s' → S2_ID=%s, venue=%s, citations=%s, doi=%s",
            paper.title[:50],
            s2_id[:20],
            enriched.venue or "N/A",
            enriched.citations,
            enriched.doi or "N/A",
        )

        return enriched

    def _extract_enrichment_data(self, details: dict) -> dict[str, Any]:
        """Estrae i dati di arricchimento dal response S2.

        Args:
            details: Response JSON da S2 API.

        Returns:
            Dict con i campi da aggiornare nel Paper.
        """
        data: dict[str, Any] = {}

        # ID e base
        data["semantic_scholar_id"] = details.get("paperId")
        data["year"] = details.get("year")
        data["venue"] = details.get("venue") or None
        data["citations"] = details.get("citationCount")
        data["influential_citation_count"] = details.get("influentialCitationCount")

        # Journal
        journal_info = details.get("journal")
        if journal_info and isinstance(journal_info, dict):
            data["journal"] = journal_info.get("name")
        elif isinstance(journal_info, str):
            data["journal"] = journal_info

        # Fields of study
        fos = details.get("fieldsOfStudy")
        if fos and isinstance(fos, list):
            data["fields_of_study"] = fos

        # External IDs → DOI, ArXiv ID
        ext_ids = details.get("externalIds") or {}
        if ext_ids.get("DOI"):
            data["doi"] = ext_ids["DOI"]
        if ext_ids.get("ArXiv"):
            data["arxiv_id"] = ext_ids["ArXiv"]

        # Published date — parsing robusto (Task 1.2)
        pub_date_str = details.get("publicationDate")
        year_s2 = details.get("year")
        parsed_year, parsed_month = self._parse_publication_date(pub_date_str, year_s2)

        if parsed_year is not None:
            data["year"] = parsed_year
        elif year_s2 is not None:
            # year_s2 già estratto sopra, manteniamo compatibilità
            data["year"] = year_s2

        if parsed_month is not None:
            data["month"] = parsed_month

        if parsed_year is not None:
            data["published_date"] = datetime(parsed_year, parsed_month or 1, 1)

        # Fallback anno da ArXiv ID ("YYMM.NNNNN" o "YYYY.NNNNN")
        if data.get("year") is None and ext_ids.get("ArXiv"):
            arxiv_id = ext_ids["ArXiv"]
            m = self._ARXIV_YEAR_RE.match(arxiv_id)
            if m:
                candidate = int(m.group(1))
                # Se sembra un anno pieno (20xx), usalo direttamente
                if 1990 <= candidate <= 2035:
                    data["year"] = candidate
                else:
                    # Vecchio formato arXiv: "YYMM.NNNNN"
                    yy = int(arxiv_id[:2])
                    if yy >= 90:
                        data["year"] = 1900 + yy
                    else:
                        data["year"] = 2000 + yy

        # Fallback anno da DOI
        if data.get("year") is None and data.get("doi"):
            doi_match = self._DOI_YEAR_RE.search(data["doi"])
            if doi_match:
                y = int(doi_match.group(1))
                if 1990 <= y <= 2035:
                    data["year"] = y

        # Authors (lista di nomi)
        authors_list = details.get("authors")
        if authors_list and isinstance(authors_list, list):
            names = []
            for a in authors_list:
                if isinstance(a, dict) and a.get("name"):
                    names.append(a["name"])
            if names:
                data["authors"] = names

        # Abstract (solo se migliore di quello esistente)
        abstract = details.get("abstract")
        if abstract and len(abstract) > 50:
            data["abstract"] = abstract

        # URL
        url = details.get("url")
        if url:
            data["url"] = url

        # BibTeX dal campo citationStyles
        citation_styles = details.get("citationStyles")
        if citation_styles and isinstance(citation_styles, dict):
            bibtex = citation_styles.get("bibtex")
            if bibtex:
                data["bibtex_entry"] = bibtex

        return data

    # -------------------------------------------------------------------------
    # Batch enrichment
    # -------------------------------------------------------------------------

    def enrich_batch(
        self,
        papers: list[Paper],
        skip_if_enriched: bool = True,
    ) -> list[Paper]:
        """Arricchisce una lista di paper in batch.

        Args:
            papers: Lista di paper da arricchire.
            skip_if_enriched: Salta paper già arricchiti.

        Returns:
            Lista di paper arricchiti (stesso ordine).
        """
        results: list[Paper] = []
        enriched_count = 0
        skipped_count = 0
        failed_count = 0

        for i, paper in enumerate(papers, 1):
            try:
                enriched = self.enrich_paper(paper, skip_if_enriched=skip_if_enriched)

                if enriched.semantic_scholar_id and not paper.semantic_scholar_id:
                    enriched_count += 1
                elif skip_if_enriched and paper.semantic_scholar_id:
                    skipped_count += 1
                else:
                    failed_count += 1

                results.append(enriched)

            except Exception as e:
                logger.error(
                    "Errore enrichment paper %d/%d '%s': %s",
                    i, len(papers), paper.title[:40], e,
                )
                results.append(paper)  # Mantieni originale in caso di errore
                failed_count += 1

        logger.info(
            "Batch enrichment completato: %d arricchiti, %d skippati, %d falliti (su %d)",
            enriched_count, skipped_count, failed_count, len(papers),
        )

        return results

    # -------------------------------------------------------------------------
    # Utility: Crea Paper da PDF info + GROBID output
    # -------------------------------------------------------------------------

    def create_paper_from_pdf(
        self,
        pdf_info: dict[str, Any],
        grobid_title: str = "",
        grobid_authors: list[str] | None = None,
        grobid_abstract: str = "",
    ) -> Paper:
        """Crea un oggetto Paper da informazioni PDF + dati GROBID.

        Utility per il workflow di ingestion: crea un Paper base
        che può poi essere arricchito con enrich_paper().

        Args:
            pdf_info: Informazioni dal pdf_hasher (absolute_path, category, ecc.)
            grobid_title: Titolo estratto da GROBID.
            grobid_authors: Autori estratti da GROBID.
            grobid_abstract: Abstract estratto da GROBID.

        Returns:
            Paper con informazioni base, pronto per l'enrichment.
        """
        title = grobid_title or pdf_info.get("filename", "Unknown")
        # Pulisci il titolo dal filename se necessario
        if title == pdf_info.get("filename", ""):
            title = title.replace(".pdf", "").replace("_", " ")

        return Paper(
            title=title,
            authors=grobid_authors or [],
            abstract=grobid_abstract,
            url="",  # Sarà popolato da S2 enrichment
            source=SourceType.SEMANTIC_SCHOLAR,
            pdf_path=str(pdf_info.get("absolute_path", "")),
            pdf_hash=pdf_info.get("hash", ""),
            category=pdf_info.get("category", ""),
        )

    def get_cache_stats(self) -> dict[str, int]:
        """Restituisce statistiche della cache S2."""
        return self._cache.stats()
