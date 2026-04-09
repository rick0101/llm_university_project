"""
Client GROBID per estrazione testo strutturato da PDF scientifici.

Invia i PDF al servizio GROBID Docker (TEI-XML) ed estrae:
- Titolo, autori, abstract
- Sezioni con titoli e livelli heading
- Riferimenti bibliografici
- Didascalie figure e tabelle
- Testo completo come fallback

Fallback a PyMuPDF se GROBID non è raggiungibile o fallisce su un singolo PDF.

Setup GROBID:
    docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.2-full

Uso:
    from paperhunter.ingestion.grobid_client import GrobidClient

    client = GrobidClient()
    if client.test_connection():
        result = client.process_pdf("/path/to/paper.pdf")
        for section in result.sections:
            print(f"[{section.level}] {section.title}: {section.text[:100]}...")
"""

from __future__ import annotations

import re
import statistics
import time
from pathlib import Path
from typing import Optional

import requests
from lxml import etree

from paperhunter.config import settings
from paperhunter.models import GROBIDOutput, GROBIDReference, GROBIDSection
from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Namespace TEI per parsing XML
TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}


class GrobidClient:
    """Client per il servizio GROBID (PDF → TEI-XML → strutturato).

    Attributes:
        base_url: URL base del servizio GROBID (es. http://localhost:8070).
        timeout: Timeout in secondi per singola richiesta.
        consolidate_header: Se consolidare l'header con citazioni.
        consolidate_citations: Se consolidare le citazioni.
        include_raw_citations: Se includere citazioni raw nel TEI-XML.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        consolidate_header: Optional[bool] = None,
        consolidate_citations: Optional[bool] = None,
        include_raw_citations: Optional[bool] = None,
    ) -> None:
        """Inizializza il client GROBID.

        Parametri non specificati vengono letti da config.yaml (sezione grobid).

        Args:
            base_url: URL base GROBID. Default da config.
            timeout: Timeout per richiesta in secondi. Default da config.
            consolidate_header: Consolidamento header. Default da config.
            consolidate_citations: Consolidamento citazioni. Default da config.
            include_raw_citations: Include citazioni raw. Default da config.
        """
        cfg = settings.grobid
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else cfg.timeout
        self.consolidate_header = (
            consolidate_header if consolidate_header is not None else cfg.consolidate_header
        )
        self.consolidate_citations = (
            consolidate_citations if consolidate_citations is not None else cfg.consolidate_citations
        )
        self.include_raw_citations = (
            include_raw_citations if include_raw_citations is not None else cfg.include_raw_citations
        )

    # =========================================================================
    # Connessione
    # =========================================================================

    def test_connection(self) -> bool:
        """Testa la connessione al servizio GROBID.

        Returns:
            True se GROBID è raggiungibile e risponde, False altrimenti.
        """
        url = f"{self.base_url}/api/isalive"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                logger.info("GROBID raggiungibile su %s", self.base_url)
                return True
            else:
                logger.warning(
                    "GROBID ha risposto con status %d su %s",
                    resp.status_code,
                    self.base_url,
                )
                return False
        except requests.ConnectionError:
            logger.warning("GROBID non raggiungibile su %s (ConnectionError)", self.base_url)
            return False
        except requests.Timeout:
            logger.warning("GROBID timeout su %s", self.base_url)
            return False
        except Exception as exc:
            logger.warning("Errore imprevisto nel test GROBID: %s", exc)
            return False

    # =========================================================================
    # Processing PDF
    # =========================================================================

    def process_pdf(self, pdf_path: str | Path, paper_id: Optional[str] = None) -> GROBIDOutput:
        """Processa un singolo PDF con GROBID e restituisce output strutturato.

        Se GROBID fallisce e `ingestion.fallback_to_pymupdf` è abilitato in config,
        usa PyMuPDF come fallback per estrarre almeno il testo grezzo.

        Args:
            pdf_path: Percorso al file PDF.
            paper_id: ID opzionale del paper (per metadata).

        Returns:
            GROBIDOutput con le sezioni estratte.

        Raises:
            FileNotFoundError: Se il file PDF non esiste.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trovato: {pdf_path}")

        start_time = time.time()
        logger.info("Processing PDF con GROBID: %s", pdf_path.name)

        try:
            tei_xml = self._call_grobid(pdf_path)
            result = self._parse_tei_xml(tei_xml, paper_id=paper_id)
            result.processing_time = time.time() - start_time
            result.success = True

            n_sections = len(result.sections)
            n_refs = len(result.references)
            logger.info(
                "GROBID OK: %s → %d sezioni, %d riferimenti (%.1fs)",
                pdf_path.name,
                n_sections,
                n_refs,
                result.processing_time,
            )
            return result

        except Exception as exc:
            elapsed = time.time() - start_time
            logger.warning(
                "GROBID fallito per %s dopo %.1fs: %s",
                pdf_path.name,
                elapsed,
                exc,
            )

            # Fallback a PyMuPDF
            if settings.ingestion.fallback_to_pymupdf:
                logger.info("Tentativo fallback con PyMuPDF per %s", pdf_path.name)
                return self._pymupdf_fallback(pdf_path, paper_id=paper_id, error_msg=str(exc))
            else:
                return GROBIDOutput(
                    paper_id=paper_id,
                    success=False,
                    error_message=f"GROBID error: {exc}",
                    processing_time=elapsed,
                )

    # =========================================================================
    # Chiamata API GROBID
    # =========================================================================

    def _call_grobid(self, pdf_path: Path) -> str:
        """Invia il PDF a GROBID e restituisce il TEI-XML raw.

        Args:
            pdf_path: Percorso al file PDF.

        Returns:
            Stringa TEI-XML.

        Raises:
            requests.HTTPError: Se GROBID risponde con status di errore.
            requests.Timeout: Se la richiesta va in timeout.
            requests.ConnectionError: Se GROBID non è raggiungibile.
        """
        url = f"{self.base_url}/api/processFulltextDocument"

        # Parametri GROBID
        data: dict[str, str | int] = {}
        if self.consolidate_header:
            data["consolidateHeader"] = "1"
        if self.consolidate_citations:
            data["consolidateCitations"] = "1"
        if self.include_raw_citations:
            data["includeRawCitations"] = "1"
        # teiCoordinates per figure/tabelle (utile per future elaborazioni)
        data["teiCoordinates"] = "figure"

        with open(pdf_path, "rb") as f:
            files = {"input": (pdf_path.name, f, "application/pdf")}
            resp = requests.post(url, files=files, data=data, timeout=self.timeout)

        resp.raise_for_status()
        return resp.text

    # =========================================================================
    # Parsing TEI-XML
    # =========================================================================

    def _parse_tei_xml(self, tei_xml: str, paper_id: Optional[str] = None) -> GROBIDOutput:
        """Parsa il TEI-XML di GROBID ed estrae i dati strutturati.

        Args:
            tei_xml: Stringa TEI-XML da GROBID.
            paper_id: ID opzionale del paper.

        Returns:
            GROBIDOutput strutturato.
        """
        root = etree.fromstring(tei_xml.encode("utf-8"))

        title = self._extract_title(root)
        authors = self._extract_authors(root)
        abstract = self._extract_abstract(root)
        sections = self._extract_sections(root)
        references = self._extract_references(root)
        figure_captions = self._extract_figure_captions(root)
        table_captions = self._extract_table_captions(root)
        full_text = self._extract_full_text(root)
        extracted_year, year_extraction_source = self._extract_year_from_tei(root)

        return GROBIDOutput(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            references=references,
            figure_captions=figure_captions,
            table_captions=table_captions,
            full_text=full_text,
            extracted_year=extracted_year,
            year_extraction_source=year_extraction_source,
        )

    # =========================================================================
    # Estrazione anno multi-source (Task 1.1)
    # =========================================================================

    _YEAR_RE = re.compile(r"\b(20[0-2]\d)\b")
    _DOI_YEAR_RE = re.compile(r"/(\d{4})[./]")

    def _extract_year_from_tei(
        self, tei_root: etree._Element
    ) -> tuple[int | None, str]:
        """Estrae l'anno di pubblicazione dal TEI-XML con strategia multi-source.

        Ordine di priorità:
          1. <date type="published">, <date type="submission">, <date> nell'header
          2. <idno type="DOI"> — anno dal DOI
          3. <note> o <title level="j"> con pattern anno (regex)
          4. Affiliazioni degli autori (date presenti)
          5. Anno mediano delle references nella listBibl (upper bound stima)

        Returns:
            Tupla (year, source) dove source è una stringa descrittiva.
            Restituisce (None, "not_found") se nessuna fonte è disponibile.
        """
        # 1) Date tags nell'header GROBID
        try:
            for date_type in ("published", "submission", None):
                xpath = (
                    f".//tei:teiHeader//tei:date[@type='{date_type}']"
                    if date_type
                    else ".//tei:teiHeader//tei:date"
                )
                for date_el in tei_root.findall(xpath, NS):
                    when = date_el.get("when", "")
                    if when and len(when) >= 4:
                        try:
                            year = int(when[:4])
                            if 1990 <= year <= 2035:
                                src = f"tei_header_date_{date_type or 'generic'}"
                                logger.debug("Anno da TEI header date: %d (%s)", year, src)
                                return year, src
                        except ValueError:
                            pass
        except Exception as exc:
            logger.debug("Errore estrazione anno da TEI header date: %s", exc)

        # 2) <idno type="DOI"> nell'header
        try:
            for idno_el in tei_root.findall(".//tei:teiHeader//tei:idno[@type='DOI']", NS):
                doi_text = idno_el.text or ""
                match = self._DOI_YEAR_RE.search(doi_text)
                if match:
                    year = int(match.group(1))
                    if 1990 <= year <= 2035:
                        logger.debug("Anno da DOI '%s': %d", doi_text[:40], year)
                        return year, "doi_year"
        except Exception as exc:
            logger.debug("Errore estrazione anno da DOI: %s", exc)

        # 3) Regex negli elementi <note> e <title> del header
        try:
            candidates: list[str] = []
            for tag_path in (
                ".//tei:teiHeader//tei:note",
                ".//tei:teiHeader//tei:title[@level='j']",
                ".//tei:teiHeader//tei:title[@level='m']",
            ):
                for el in tei_root.findall(tag_path, NS):
                    text = self._get_all_text(el).strip()
                    if text:
                        candidates.append(text)
            for text in candidates:
                match = self._YEAR_RE.search(text)
                if match:
                    year = int(match.group(1))
                    if 1990 <= year <= 2035:
                        logger.debug("Anno da regex in note/title: %d", year)
                        return year, "text_regex_header"
        except Exception as exc:
            logger.debug("Errore estrazione anno da note/title: %s", exc)

        # 4) Date nelle affiliazioni degli autori
        try:
            for date_el in tei_root.findall(".//tei:teiHeader//tei:author//tei:date", NS):
                when = date_el.get("when", "")
                if when and len(when) >= 4:
                    try:
                        year = int(when[:4])
                        if 1990 <= year <= 2035:
                            logger.debug("Anno da affiliazione autore: %d", year)
                            return year, "author_affiliation_date"
                    except ValueError:
                        pass
                # Prova anche il testo del tag
                text = date_el.text or ""
                match = self._YEAR_RE.search(text)
                if match:
                    year = int(match.group(1))
                    if 1990 <= year <= 2035:
                        logger.debug("Anno da testo affiliazione: %d", year)
                        return year, "author_affiliation_date"
        except Exception as exc:
            logger.debug("Errore estrazione anno da affiliazioni: %s", exc)

        # 5) Anno mediano delle references (stima upper bound)
        try:
            ref_years: list[int] = []
            list_bibl = tei_root.find(".//tei:text/tei:back//tei:listBibl", NS)
            if list_bibl is not None:
                for bibl in list_bibl.findall("tei:biblStruct", NS):
                    date_el = bibl.find(".//tei:monogr/tei:imprint/tei:date[@type='published']", NS)
                    if date_el is not None:
                        when = date_el.get("when", "")
                        if when and len(when) >= 4:
                            try:
                                y = int(when[:4])
                                if 1990 <= y <= 2035:
                                    ref_years.append(y)
                            except ValueError:
                                pass
            if len(ref_years) >= 3:
                median_year = int(statistics.median(ref_years))
                logger.debug(
                    "Anno stimato da mediana %d references: %d",
                    len(ref_years),
                    median_year,
                )
                return median_year, "citation_median"
        except Exception as exc:
            logger.debug("Errore estrazione anno da references: %s", exc)

        return None, "not_found"

    def _extract_title(self, root: etree._Element) -> str:
        """Estrae il titolo dal TEI header."""
        title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title[@type='main']", NS)
        if title_el is None:
            # Fallback: qualsiasi <title> nel titleStmt
            title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", NS)
        if title_el is not None:
            return self._get_text(title_el).strip()
        return ""

    # Pattern per nomi autori invalidi (modelli AI, affiliazioni, token corti)
    _INVALID_AUTHOR_PATTERNS = [
        re.compile(r'^(Qwen|Gemini|GPT|Claude|Llama|Mistral)', re.IGNORECASE),
        re.compile(r'^(Efficient|Ving|Binyuan)$', re.IGNORECASE),
        re.compile(r'(Labs?|Inc|Corp|University|Institute)\b', re.IGNORECASE),
        re.compile(r'^(Microsoft|Google|Meta|Apple|Amazon|OpenAI|Anthropic|DeepMind|Nvidia|Intel|IBM|Huawei|Alibaba|Baidu|Tencent)$', re.IGNORECASE),
        re.compile(r'^\w{1,2}$'),
    ]

    def _extract_authors(self, root: etree._Element) -> list[str]:
        """Estrae la lista degli autori dal TEI header.

        Filtra elementi <orgName>/<affiliation> e nomi invalidi
        (modelli AI, affiliazioni, token corti).
        """
        authors = []
        # sourceDesc → biblStruct → analytic → author
        for author_el in root.findall(
            ".//tei:teiHeader//tei:sourceDesc//tei:author", NS
        ):
            persname = author_el.find("tei:persName", NS)
            if persname is None:
                # Ignora <author> senza <persName> (es. solo <orgName>/<affiliation>)
                continue
            forename_el = persname.find("tei:forename", NS)
            surname_el = persname.find("tei:surname", NS)
            forename = forename_el.text.strip() if forename_el is not None and forename_el.text else ""
            surname = surname_el.text.strip() if surname_el is not None and surname_el.text else ""
            name = f"{forename} {surname}".strip()
            if not name:
                continue
            # Filtra nomi invalidi
            if any(p.search(name) for p in self._INVALID_AUTHOR_PATTERNS):
                logger.debug("GROBID autore filtrato come invalido: '%s'", name)
                continue
            authors.append(name)
        return authors

    def _extract_abstract(self, root: etree._Element) -> str:
        """Estrae l'abstract dal TEI header."""
        abstract_el = root.find(".//tei:teiHeader//tei:profileDesc/tei:abstract", NS)
        if abstract_el is not None:
            return self._get_all_text(abstract_el).strip()
        return ""

    def _extract_sections(self, root: etree._Element) -> list[GROBIDSection]:
        """Estrae le sezioni dal body del TEI-XML.

        Ogni <div> con un <head> diventa una sezione. Il livello viene
        determinato dall'attributo @n o dalla profondità di nesting.
        """
        sections: list[GROBIDSection] = []
        body = root.find(".//tei:text/tei:body", NS)
        if body is None:
            return sections

        self._walk_divs(body, sections, current_level=1)
        return sections

    def _walk_divs(
        self,
        element: etree._Element,
        sections: list[GROBIDSection],
        current_level: int,
    ) -> None:
        """Percorre ricorsivamente i <div> per estrarre sezioni.

        Args:
            element: Elemento XML corrente.
            sections: Lista sezioni da popolare (in-place).
            current_level: Livello heading corrente.
        """
        for div in element.findall("tei:div", NS):
            head_el = div.find("tei:head", NS)

            # Determina il livello heading
            level = current_level
            if head_el is not None:
                # GROBID a volte mette @n="1", @n="1.1", ecc.
                n_attr = head_el.get("n", "")
                if n_attr:
                    # Conta il numero di punti per determinare il livello
                    dot_count = n_attr.count(".")
                    level = min(dot_count + 1, 4)
                title = self._get_text(head_el).strip()
            else:
                title = ""

            # Raccogli il testo dei <p> diretti (non quelli nei sotto-div)
            paragraphs = []
            for child in div:
                tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ""
                if tag == "p":
                    p_text = self._get_all_text(child).strip()
                    if p_text:
                        paragraphs.append(p_text)

            text = "\n\n".join(paragraphs)

            if title or text:
                sections.append(GROBIDSection(title=title, text=text, level=level))

            # Ricorsione per sotto-sezioni
            self._walk_divs(div, sections, current_level=min(level + 1, 4))

    def _extract_references(self, root: etree._Element) -> list[GROBIDReference]:
        """Estrae i riferimenti bibliografici dalla listBibl."""
        references: list[GROBIDReference] = []
        list_bibl = root.find(".//tei:text/tei:back/tei:div/tei:listBibl", NS)
        if list_bibl is None:
            # Prova path alternativo
            list_bibl = root.find(".//tei:text/tei:back//tei:listBibl", NS)
        if list_bibl is None:
            return references

        for bibl_struct in list_bibl.findall("tei:biblStruct", NS):
            ref = self._parse_bibl_struct(bibl_struct)
            if ref is not None:
                references.append(ref)

        return references

    def _parse_bibl_struct(self, bibl: etree._Element) -> Optional[GROBIDReference]:
        """Parsa un singolo <biblStruct> in un GROBIDReference."""
        ref_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id", None)

        # Titolo (da analytic o monogr)
        title_el = bibl.find(".//tei:analytic/tei:title[@type='main']", NS)
        if title_el is None:
            title_el = bibl.find(".//tei:monogr/tei:title", NS)
        title = self._get_text(title_el).strip() if title_el is not None else None

        # Autori
        authors = []
        for author_el in bibl.findall(".//tei:analytic/tei:author", NS):
            persname = author_el.find("tei:persName", NS)
            if persname is not None:
                forename_el = persname.find("tei:forename", NS)
                surname_el = persname.find("tei:surname", NS)
                forename = (
                    forename_el.text.strip() if forename_el is not None and forename_el.text else ""
                )
                surname = (
                    surname_el.text.strip() if surname_el is not None and surname_el.text else ""
                )
                name = f"{forename} {surname}".strip()
                if name:
                    authors.append(name)

        # Anno
        date_el = bibl.find(".//tei:monogr/tei:imprint/tei:date[@type='published']", NS)
        year = None
        if date_el is not None:
            when = date_el.get("when", "")
            if when and len(when) >= 4:
                try:
                    year = int(when[:4])
                except ValueError:
                    pass

        # Venue / Journal
        venue_el = bibl.find(".//tei:monogr/tei:title[@level='j']", NS)
        if venue_el is None:
            venue_el = bibl.find(".//tei:monogr/tei:title[@level='m']", NS)
        venue = self._get_text(venue_el).strip() if venue_el is not None else None

        # DOI
        doi_el = bibl.find(".//tei:idno[@type='DOI']", NS)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        # Raw text (note con tipo raw_reference)
        raw_el = bibl.find(".//tei:note[@type='raw_reference']", NS)
        raw_text = raw_el.text.strip() if raw_el is not None and raw_el.text else None

        # Se non c'è nemmeno il titolo, proviamo con il raw_text
        if not title and not raw_text:
            return None

        return GROBIDReference(
            id=ref_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            raw_text=raw_text,
        )

    def _extract_figure_captions(self, root: etree._Element) -> list[str]:
        """Estrae le didascalie delle figure."""
        captions = []
        for fig in root.findall(".//tei:figure[@type='figure']", NS):
            # Cerca anche figure senza type per compatibility
            figdesc = fig.find("tei:figDesc", NS)
            if figdesc is not None:
                text = self._get_all_text(figdesc).strip()
                if text:
                    # Aggiungi il label se presente
                    label_el = fig.find("tei:label", NS)
                    label = label_el.text.strip() if label_el is not None and label_el.text else ""
                    head_el = fig.find("tei:head", NS)
                    head = head_el.text.strip() if head_el is not None and head_el.text else ""
                    prefix = f"Figure {label}" if label else "Figure"
                    if head:
                        prefix = f"{prefix}: {head}"
                    captions.append(f"{prefix}. {text}")

        # Cerca anche figure senza type (GROBID a volte le omette)
        for fig in root.findall(".//tei:figure", NS):
            fig_type = fig.get("type", "")
            if fig_type in ("", "figure"):
                continue  # Già processate sopra o generiche
            # Non è una tabella
            if fig_type != "table":
                figdesc = fig.find("tei:figDesc", NS)
                if figdesc is not None:
                    text = self._get_all_text(figdesc).strip()
                    if text and text not in [c.split(". ", 1)[-1] if ". " in c else c for c in captions]:
                        captions.append(text)

        return captions

    def _extract_table_captions(self, root: etree._Element) -> list[str]:
        """Estrae le didascalie delle tabelle."""
        captions = []
        for fig in root.findall(".//tei:figure[@type='table']", NS):
            figdesc = fig.find("tei:figDesc", NS)
            head_el = fig.find("tei:head", NS)
            label_el = fig.find("tei:label", NS)

            text = ""
            if figdesc is not None:
                text = self._get_all_text(figdesc).strip()

            label = label_el.text.strip() if label_el is not None and label_el.text else ""
            head = head_el.text.strip() if head_el is not None and head_el.text else ""

            prefix = f"Table {label}" if label else "Table"
            if head:
                prefix = f"{prefix}: {head}"

            if text:
                captions.append(f"{prefix}. {text}")
            elif head:
                captions.append(prefix)

        return captions

    def _extract_full_text(self, root: etree._Element) -> str:
        """Estrae il testo completo di tutto il body come fallback."""
        body = root.find(".//tei:text/tei:body", NS)
        if body is not None:
            return self._get_all_text(body).strip()
        return ""

    # =========================================================================
    # Helper per estrazione testo da elementi XML
    # =========================================================================

    @staticmethod
    def _get_text(el: etree._Element) -> str:
        """Estrae il testo diretto di un elemento (senza figli)."""
        return el.text or "" if el is not None else ""

    @staticmethod
    def _get_all_text(el: etree._Element) -> str:
        """Estrae tutto il testo di un elemento, inclusi sotto-elementi.

        Concatena ricorsivamente .text e .tail di tutti i nodi.
        """
        if el is None:
            return ""
        # etree.tostring con method="text" è il modo più efficiente
        raw = etree.tostring(el, method="text", encoding="unicode")
        return raw if raw else ""

    # =========================================================================
    # Fallback PyMuPDF
    # =========================================================================

    def _pymupdf_fallback(
        self,
        pdf_path: Path,
        paper_id: Optional[str] = None,
        error_msg: str = "",
    ) -> GROBIDOutput:
        """Estrae testo da PDF usando PyMuPDF come fallback.

        Estrae il testo pagina per pagina e prova a rilevare la struttura
        minima (titolo dalla prima pagina, sezioni dai heading).

        Args:
            pdf_path: Percorso al file PDF.
            paper_id: ID opzionale del paper.
            error_msg: Messaggio di errore originale da GROBID.

        Returns:
            GROBIDOutput con full_text e sezioni basilari.
        """
        try:
            import pymupdf

            start_time = time.time()
            doc = pymupdf.open(str(pdf_path))

            all_text = []
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    all_text.append(page_text)
            doc.close()

            full_text = "\n\n".join(all_text)
            elapsed = time.time() - start_time

            # Tentativo basilare di estrarre titolo (prima riga non vuota)
            title = ""
            for line in full_text.split("\n"):
                stripped = line.strip()
                if stripped and len(stripped) > 5:
                    title = stripped
                    break

            # Tentativo di estrarre sezioni tramite pattern heading
            sections = self._detect_sections_from_text(full_text)

            logger.info(
                "PyMuPDF fallback OK: %s → %d sezioni, %d caratteri (%.1fs)",
                pdf_path.name,
                len(sections),
                len(full_text),
                elapsed,
            )

            return GROBIDOutput(
                paper_id=paper_id,
                title=title,
                sections=sections,
                full_text=full_text,
                processing_time=elapsed,
                success=True,
                error_message=f"GROBID failed ({error_msg}), used PyMuPDF fallback",
            )

        except Exception as fallback_exc:
            logger.error("Anche PyMuPDF fallback fallito per %s: %s", pdf_path.name, fallback_exc)
            return GROBIDOutput(
                paper_id=paper_id,
                success=False,
                error_message=f"GROBID error: {error_msg}; PyMuPDF fallback error: {fallback_exc}",
            )

    @staticmethod
    def _detect_sections_from_text(text: str) -> list[GROBIDSection]:
        """Rileva sezioni dal testo grezzo usando pattern di heading comuni.

        Cerca pattern come:
        - "1. Introduction"
        - "2 Related Work"
        - "Abstract"
        - "METHODOLOGY"
        - "3.1 Sub-section"

        Args:
            text: Testo grezzo completo.

        Returns:
            Lista di sezioni rilevate.
        """
        import re

        sections: list[GROBIDSection] = []
        # Pattern per heading: numero opzionale + titolo su una riga breve
        heading_pattern = re.compile(
            r"^(?:"
            r"(\d+(?:\.\d+)*)[\s.\-]+(.+)"  # "1. Introduction" / "2.1 Methods"
            r"|"
            r"(Abstract|Introduction|Related Work|Background|"
            r"Method(?:ology|s)?|Approach|Experiment(?:s|al)?(?:\s+(?:Setup|Results))?"
            r"|Results?(?:\s+and\s+Discussion)?|Discussion|Conclusion(?:s)?"
            r"|Future\s+Work|Acknowledgment(?:s)?|References|Appendix(?:\s+\w)?"
            r"|Limitations|Evaluation|Analysis|Implementation)"
            r")\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        # Trova tutte le posizioni di heading
        headings: list[tuple[int, str, int]] = []  # (pos, title, level)
        for match in heading_pattern.finditer(text):
            num_part = match.group(1)
            title_part = match.group(2)
            keyword = match.group(3)

            if num_part and title_part:
                title = f"{num_part} {title_part.strip()}"
                level = min(num_part.count(".") + 1, 4)
            elif keyword:
                title = keyword.strip()
                level = 1
            else:
                continue

            # Ignora falsi positivi (titolo troppo lungo = probabilmente non un heading)
            if len(title) > 80:
                continue

            headings.append((match.start(), title, level))

        # Costruisci le sezioni dal testo tra heading consecutivi
        for i, (pos, title, level) in enumerate(headings):
            # Inizio del testo: dopo il newline successivo al heading
            text_start = text.find("\n", pos)
            if text_start == -1:
                continue
            text_start += 1

            # Fine del testo: inizio del prossimo heading o fine del testo
            if i + 1 < len(headings):
                text_end = headings[i + 1][0]
            else:
                text_end = len(text)

            section_text = text[text_start:text_end].strip()
            if section_text:
                sections.append(GROBIDSection(title=title, text=section_text, level=level))

        return sections
