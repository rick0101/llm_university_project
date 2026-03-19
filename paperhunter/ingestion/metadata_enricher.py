"""
MetadataEnricher — Pipeline a cascata per l'arricchimento robusto dei metadati.

Coordina GROBID output + risposta Semantic Scholar + euristiche sul filename
per produrre un Paper completamente arricchito con campo `metadata_confidence`.

Strategia a cascata:
  Anno    : S2 publicationDate → S2 year → GROBID extracted_year → regex abstract
            → anno da filename → anno mediano citazioni → None
  Venue   : S2 venue → S2 journal → GROBID publisher/note → normalizzazione alias
  Abstract: S2 abstract → GROBID abstract → primo paragrafo full_text

Uso:
    enricher = MetadataEnricher()
    paper = enricher.enrich(
        paper=paper,
        grobid_output=grobid_result,
        s2_data=s2_raw_dict,
        pdf_filename="Zou_2023_GCG.pdf",
    )
    print(paper.year, paper.year_source, paper.metadata_confidence)
"""

from __future__ import annotations

import re
from typing import Optional

from paperhunter.models import GROBIDOutput, Paper
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# =============================================================================
# Costanti
# =============================================================================

_YEAR_RE = re.compile(r"\b(20[0-2]\d)\b")
_FILENAME_YEAR_RE = re.compile(r"(?:^|[_\-\s])?(20[0-2]\d)(?:[_\-\s]|$)")

# Alias di normalizzazione venue: chiave in minuscolo → nome canonico
_VENUE_ALIASES: dict[str, str] = {
    "arxiv": "arXiv",
    "arxiv preprint": "arXiv",
    "neurips": "NeurIPS",
    "nips": "NeurIPS",
    "neural information processing systems": "NeurIPS",
    "icml": "ICML",
    "international conference on machine learning": "ICML",
    "iclr": "ICLR",
    "international conference on learning representations": "ICLR",
    "acl": "ACL",
    "association for computational linguistics": "ACL",
    "emnlp": "EMNLP",
    "naacl": "NAACL",
    "coling": "COLING",
    "cvpr": "CVPR",
    "iccv": "ICCV",
    "eccv": "ECCV",
    "aaai": "AAAI",
    "ijcai": "IJCAI",
    "ieee s&p": "IEEE S&P",
    "ieee security & privacy": "IEEE S&P",
    "usenix security": "USENIX Security",
    "ccs": "ACM CCS",
    "acm ccs": "ACM CCS",
    "ndss": "NDSS",
    "jmlr": "JMLR",
    "tacl": "TACL",
}


# =============================================================================
# MetadataEnricher
# =============================================================================

class MetadataEnricher:
    """Orchestratore per l'arricchimento metadati multi-source di un Paper.

    Integra output GROBID, risposta S2 grezza ed euristiche sul filename
    per massimizzare la qualità dei metadati estratti.
    Il campo `metadata_confidence` indica l'affidabilità complessiva (0.0-1.0).
    """

    def enrich(
        self,
        paper: Paper,
        grobid_output: Optional[GROBIDOutput] = None,
        s2_data: Optional[dict] = None,
        pdf_filename: str = "",
    ) -> Paper:
        """Arricchisce il Paper con metadati provenienti da tutte le fonti.

        Args:
            paper: Paper da arricchire. Non viene modificato in-place.
            grobid_output: Output strutturato di GROBID (può essere None).
            s2_data: Risposta raw dict di Semantic Scholar (può essere None).
            pdf_filename: Nome del file PDF (usato per estrarre anno dal filename).

        Returns:
            Nuovo Paper con metadati migliorati e `metadata_confidence` calcolato.
        """
        paper_dict = paper.model_dump()
        confidence = 0.0

        # ---- Anno ----------------------------------------------------------------
        year, year_src, year_conf = self._resolve_year(
            paper_dict.get("year"),
            s2_data,
            grobid_output,
            pdf_filename,
        )
        if year is not None:
            paper_dict["year"] = year
            paper_dict["year_source"] = year_src
            confidence += year_conf

        # ---- Venue ---------------------------------------------------------------
        venue, venue_conf = self._resolve_venue(
            paper_dict.get("venue"),
            paper_dict.get("journal"),
            s2_data,
            grobid_output,
        )
        if venue:
            paper_dict["venue"] = venue
            confidence += venue_conf

        # ---- Abstract ------------------------------------------------------------
        abstract, abstract_conf = self._resolve_abstract(
            paper_dict.get("abstract", ""),
            s2_data,
            grobid_output,
        )
        if abstract:
            paper_dict["abstract"] = abstract
            confidence += abstract_conf

        # ---- published_date (aggiorna se year migliorato) ------------------------
        if year is not None and paper_dict.get("published_date") is None:
            from datetime import datetime
            paper_dict["published_date"] = datetime(year, paper_dict.get("month") or 1, 1)

        # ---- metadata_confidence (cap a 1.0) ------------------------------------
        paper_dict["metadata_confidence"] = round(min(1.0, confidence), 3)

        enriched = Paper(**paper_dict)

        logger.info(
            "MetadataEnricher: year=%s (src=%s), venue=%s, conf=%.2f — '%s'",
            enriched.year,
            enriched.year_source,
            enriched.venue or "N/A",
            enriched.metadata_confidence,
            enriched.title[:60],
        )

        return enriched

    # -------------------------------------------------------------------------
    # Risoluzione anno
    # -------------------------------------------------------------------------

    def _resolve_year(
        self,
        existing_year: Optional[int],
        s2_data: Optional[dict],
        grobid_output: Optional[GROBIDOutput],
        pdf_filename: str,
    ) -> tuple[Optional[int], str, float]:
        """Cascata multi-source per l'anno.

        Returns: (year, source_label, confidence_contribution)
        """
        # 1. S2 publicationDate (già parsato in _extract_enrichment_data,
        #    ma qui usiamo il raw per coerenza con la cascata)
        if s2_data:
            pub_date = s2_data.get("publicationDate")
            if pub_date:
                m = _YEAR_RE.search(str(pub_date))
                if m:
                    y = int(m.group(1))
                    if 1990 <= y <= 2035:
                        return y, "s2_publication_date", 0.15

            # 2. S2 year intero
            s2_year = s2_data.get("year")
            if s2_year:
                try:
                    y = int(s2_year)
                    if 1990 <= y <= 2035:
                        return y, "s2_year", 0.12
                except (ValueError, TypeError):
                    pass

        # 3. GROBID extracted_year (da Task 1.1)
        if grobid_output and grobid_output.extracted_year:
            y = grobid_output.extracted_year
            if 1990 <= y <= 2035:
                src = grobid_output.year_extraction_source or "grobid"
                return y, f"grobid_{src}", 0.08

        # 4. Regex sull'abstract GROBID
        if grobid_output and grobid_output.abstract:
            m = _YEAR_RE.search(grobid_output.abstract)
            if m:
                y = int(m.group(1))
                if 1990 <= y <= 2035:
                    return y, "grobid_abstract_regex", 0.05

        # 5. Anno dal filename
        if pdf_filename:
            m = _FILENAME_YEAR_RE.search(pdf_filename)
            if m:
                y = int(m.group(1))
                if 1990 <= y <= 2035:
                    return y, "filename", 0.04

        # 6. Anno mediano citazioni (già estratto da GROBID come citation_median)
        if (
            grobid_output
            and grobid_output.extracted_year
            and grobid_output.year_extraction_source == "citation_median"
        ):
            return grobid_output.extracted_year, "citation_median", 0.03

        # 7. Anno già sul paper (passato dall'esterno)
        if existing_year and 1990 <= existing_year <= 2035:
            return existing_year, "paper_existing", 0.05

        return None, "not_found", 0.0

    # -------------------------------------------------------------------------
    # Risoluzione venue
    # -------------------------------------------------------------------------

    def _resolve_venue(
        self,
        existing_venue: Optional[str],
        existing_journal: Optional[str],
        s2_data: Optional[dict],
        grobid_output: Optional[GROBIDOutput],
    ) -> tuple[Optional[str], float]:
        """Cascata multi-source per la venue.

        Returns: (venue_normalizzata, confidence_contribution)
        """
        candidates: list[tuple[str, float]] = []  # (venue_raw, conf)

        # 1. S2 venue
        if s2_data:
            s2_venue = s2_data.get("venue") or ""
            if s2_venue.strip():
                candidates.append((s2_venue.strip(), 0.15))

            # 2. S2 journal
            journal_info = s2_data.get("journal")
            if journal_info:
                if isinstance(journal_info, dict):
                    j_name = journal_info.get("name", "")
                else:
                    j_name = str(journal_info)
                if j_name.strip():
                    candidates.append((j_name.strip(), 0.12))

        # 3. GROBID — publisher o note di tipo report
        if grobid_output:
            # Cerca nei riferimenti il venue/journal del paper stesso (tipicamente il primo)
            for ref in (grobid_output.references or [])[:1]:
                if ref.venue:
                    candidates.append((ref.venue, 0.06))

        # 4. Valori già sul paper
        if existing_venue:
            candidates.append((existing_venue, 0.08))
        if existing_journal:
            candidates.append((existing_journal, 0.06))

        # Prendi il primo con valore non vuoto, normalizzalo
        for raw_venue, conf in candidates:
            normalized = self._normalize_venue(raw_venue)
            if normalized:
                return normalized, conf

        return None, 0.0

    @staticmethod
    def _normalize_venue(venue: str) -> str:
        """Normalizza il nome della venue usando l'alias dictionary."""
        key = venue.lower().strip()
        # Cerca corrispondenza esatta
        if key in _VENUE_ALIASES:
            return _VENUE_ALIASES[key]
        # Cerca corrispondenza parziale (substring)
        for alias_key, canonical in _VENUE_ALIASES.items():
            if alias_key in key:
                return canonical
        # Nessun alias trovato — restituisce il valore originale capitalizzato
        return venue.strip()

    # -------------------------------------------------------------------------
    # Risoluzione abstract
    # -------------------------------------------------------------------------

    def _resolve_abstract(
        self,
        existing_abstract: str,
        s2_data: Optional[dict],
        grobid_output: Optional[GROBIDOutput],
    ) -> tuple[str, float]:
        """Cascata multi-source per l'abstract.

        Returns: (abstract, confidence_contribution)
        """
        # 1. S2 abstract (tipicamente più pulito)
        if s2_data:
            s2_abstract = s2_data.get("abstract") or ""
            if len(s2_abstract) > 50:
                return s2_abstract, 0.15

        # 2. GROBID abstract
        if grobid_output and grobid_output.abstract and len(grobid_output.abstract) > 50:
            return grobid_output.abstract, 0.08

        # 3. Primo paragrafo del full_text (heuristic)
        if grobid_output and grobid_output.full_text:
            paragraphs = [
                p.strip()
                for p in grobid_output.full_text.split("\n\n")
                if len(p.strip()) > 100
            ]
            if paragraphs:
                return paragraphs[0][:2000], 0.03

        # 4. Mantieni quello già presente
        if existing_abstract and len(existing_abstract) > 50:
            return existing_abstract, 0.04

        return "", 0.0
