"""
DBLP SPARQL — Client per l'endpoint SPARQL pubblico di DBLP.

Da settembre 2026 ``dblp.org/search/publ/api``, ``dblp.org/rec/<key>.bib`` e i
mirror (uni-trier, dagstuhl) sono protetti dall'anti-bot Anubis e restituiscono
una pagina HTML di challenge invece di JSON/BibTeX. L'endpoint SPARQL ufficiale
(``https://sparql.dblp.org/sparql``, motore QLever) non è protetto, è gratuito
e contiene gli stessi dati: questo modulo lo usa per

- trovare record per DOI (lookup esatto su IRI, istantaneo);
- trovare record per titolo (regex con prefisso ``^``, risolta sull'indice);
- leggere i metadati di un record nello stesso formato degli hit della
  vecchia search API (così ``DblpLookup._select_best_hit`` resta invariato);
- ricostruire un BibTeX in stile DBLP a partire dai triple RDF.

Uso::

    from paperhunter.ingestion.dblp_sparql import DblpSparqlClient

    client = DblpSparqlClient()
    keys = client.find_by_doi("10.48550/arXiv.2404.05530")
    hit = client.get_record(keys[0])
    print(client.get_bibtex(keys[0]))
"""

from __future__ import annotations

import re
import time
import unicodedata

import httpx

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

SPARQL_ENDPOINT = "https://sparql.dblp.org/sparql"
_REC_PREFIX = "https://dblp.org/rec/"
_SCHEMA = "https://dblp.org/rdf/schema#"
_PREFIXES = "PREFIX dblp: <https://dblp.org/rdf/schema#>\n"

_REQUEST_TIMEOUT = 60.0
_MIN_REQUEST_INTERVAL = 0.3  # secondi tra query (fair use)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0
_RATE_LIMIT_COOLDOWN = 30.0

# rdf:type DBLP → campo ``type`` della vecchia search API
_TYPE_LABELS = {
    "Inproceedings": "Conference and Workshop Papers",
    "Article": "Journal Articles",
    "Informal": "Informal and Other Publications",
    "Book": "Books and Theses",
    "Incollection": "Parts in Books or Collections",
    "Editorship": "Editorship",
    "Reference": "Reference Works",
    "Data": "Data and Artifacts",
}

# Diacritici Unicode combinanti → comando di accento LaTeX
_LATEX_ACCENTS = {
    "̀": "`", "́": "'", "̂": "^", "̃": "~", "̄": "=",
    "̆": "u", "̇": ".", "̈": '"', "̊": "r", "̋": "H",
    "̌": "v", "̧": "c", "̨": "k",
}
_LATEX_SPECIAL = {"ß": r"{\ss}", "ø": r"{\o}", "Ø": r"{\O}", "æ": r"{\ae}", "Æ": r"{\AE}",
                  "ł": r"{\l}", "Ł": r"{\L}", "ı": r"{\i}", "œ": r"{\oe}", "å": r"{\aa}"}


class DblpSparqlError(RuntimeError):
    """Errore restituito dall'endpoint SPARQL (timeout di query, sintassi, HTTP)."""


# ---------------------------------------------------------------------------
# Helper puri
# ---------------------------------------------------------------------------

def _rec_iri(key: str) -> str:
    return key if key.startswith("http") else _REC_PREFIX + key


def _rec_key(iri: str) -> str:
    return iri.replace(_REC_PREFIX, "")


def _local(iri: str) -> str:
    return iri.rsplit("#", 1)[-1] if "#" in iri else iri


def _strip_homonym(name: str) -> str:
    """DBLP disambigua gli omonimi con un suffisso numerico ("Tobias Lorenz 0002")."""
    return re.sub(r"\s+\d{4}$", "", name).strip()


def clean_doi(doi: str) -> str:
    """Normalizza un DOI dal .bib: niente prefisso URL né escape LaTeX (``\\_``)."""
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip())
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.replace("\\", "").rstrip("./")


def latex_escape(text: str) -> str:
    """Converte testo Unicode in LaTeX sicuro per BibTeX (accenti, &, %, _, #, $)."""
    text = re.sub(r"([&%_#$])", r"\\\1", text)
    out: list[str] = []
    for char in unicodedata.normalize("NFD", text):
        if char in _LATEX_ACCENTS and out:
            base = out.pop()
            if base == "i":
                base = r"\i"
            out.append("{\\" + _LATEX_ACCENTS[char] + "{" + base + "}}")
        else:
            out.append(char)
    return "".join(_LATEX_SPECIAL.get(c, c) for c in unicodedata.normalize("NFC", "".join(out)))


def protect_title(title: str) -> str:
    """Protegge con graffe acronimi e CamelCase (convenzione DBLP), es. ``{RLHF}``.

    Gli stili BibTeX come ``plain`` portano il titolo in minuscolo tranne la
    prima lettera e le parti tra graffe.
    """
    def _protect(match: re.Match) -> str:
        word = match.group(0)
        upper_inside = any(c.isupper() for c in word[1:])
        single_cap = len(word) == 1 and word.isupper() and match.start() > 0
        return "{" + word + "}" if upper_inside or single_cap else word

    return re.sub(r"[A-Za-z][\w']*", _protect, title)


def title_prefix_regex(title: str, n_words: int) -> str:
    """Regex ``^<prime n parole>`` da inserire in un literal SPARQL.

    QLever risolve ``REGEX(?t, "^...")`` sull'indice ordinato dei literal (match
    case-insensitive in pratica), quindi è istantanea anche su 7M+ titoli. I
    metacaratteri vengono escapati con doppio backslash (escape del literal);
    ``"`` diventa ``.`` per non chiudere il literal.
    """
    prefix = " ".join(title.split()[:n_words]).replace('"', ".")
    return "^" + re.sub(r"([\\^$|?*+()\[\]{}])", r"\\\\\1", prefix)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class DblpSparqlClient:
    """Client minimale con rate limiting e retry per ``sparql.dblp.org``."""

    def __init__(
        self,
        endpoint: str = SPARQL_ENDPOINT,
        user_agent: str = "PaperHunter/2.0 (academic survey tool)",
    ) -> None:
        self._endpoint = endpoint
        self._last_request_time = 0.0
        self._client = httpx.Client(
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": user_agent, "Accept": "application/sparql-results+json"},
        )

    # ------------------------------------------------------------------
    # Trasporto
    # ------------------------------------------------------------------

    def query(self, sparql: str) -> list[dict[str, str]]:
        """Esegue una SELECT e restituisce le righe come dict variabile → valore."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            elapsed = time.time() - self._last_request_time
            if elapsed < _MIN_REQUEST_INTERVAL:
                time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
            self._last_request_time = time.time()
            try:
                resp = self._client.post(self._endpoint, data={"query": _PREFIXES + sparql})
                if resp.status_code == 429:
                    logger.debug("DBLP SPARQL 429, attendo %ds", _RATE_LIMIT_COOLDOWN)
                    time.sleep(float(resp.headers.get("Retry-After", _RATE_LIMIT_COOLDOWN)))
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "exception" in data:
                    # Errore di query (timeout QLever, sintassi): inutile riprovare.
                    raise DblpSparqlError(str(data["exception"]))
                return [
                    {var: cell["value"] for var, cell in row.items()}
                    for row in data["results"]["bindings"]
                ]
            except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
                last_exc = exc
                logger.debug("DBLP SPARQL errore (tentativo %d/%d): %s",
                             attempt + 1, _MAX_RETRIES, exc)
                time.sleep(_RETRY_BACKOFF * (2 ** attempt))
        raise DblpSparqlError(f"DBLP SPARQL non raggiungibile: {last_exc}")

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Ricerca
    # ------------------------------------------------------------------

    def find_by_doi(self, doi: str) -> list[str]:
        """Chiavi DBLP dei record con questo DOI.

        DBLP memorizza il DOI come IRI con il case del publisher (es.
        ``10.48550/ARXIV.2404.05530``), quindi si provano più varianti.
        """
        doi = clean_doi(doi)
        if not doi or re.search(r'[\s<>"{}|^`\\]', doi):
            return []
        for variant in dict.fromkeys([doi, doi.upper(), doi.lower()]):
            rows = self.query(f"SELECT ?p WHERE {{ ?p dblp:doi <https://doi.org/{variant}> }}")
            if rows:
                return [_rec_key(r["p"]) for r in rows]
        return []

    def find_by_arxiv(self, arxiv_id: str) -> list[str]:
        """Chiavi DBLP del record CoRR con questo identificativo arXiv."""
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        return self.find_by_doi(f"10.48550/ARXIV.{arxiv_id}") if arxiv_id else []

    def find_by_title(self, title: str, n_words: int = 6, limit: int = 400) -> list[tuple[str, str]]:
        """Record il cui titolo inizia con le prime ``n_words`` parole di ``title``.

        Restituisce coppie (chiave DBLP, titolo DBLP); il confronto fine sul
        titolo completo resta al chiamante.
        """
        if not title.strip():
            return []
        regex = title_prefix_regex(title, n_words)
        rows = self.query(
            f'SELECT ?p ?t WHERE {{ ?p dblp:title ?t . FILTER(REGEX(?t, "{regex}")) }} '
            f"LIMIT {int(limit)}"
        )
        return [(_rec_key(r["p"]), r["t"]) for r in rows]

    # ------------------------------------------------------------------
    # Metadati record
    # ------------------------------------------------------------------

    def _properties(self, iri: str) -> dict[str, list[str]]:
        rows = self.query(f"SELECT ?pr ?o WHERE {{ <{iri}> ?pr ?o }}")
        props: dict[str, list[str]] = {}
        for row in rows:
            props.setdefault(_local(row["pr"]), []).append(row["o"])
        return props

    def _signatures(self, iri: str) -> tuple[list[str], list[str]]:
        """(autori, editor) ordinati per ``signatureOrdinal``."""
        rows = self.query(f"""SELECT ?kind ?ord ?name WHERE {{
            <{iri}> dblp:hasSignature ?s .
            ?s a ?kind ; dblp:signatureOrdinal ?ord ; dblp:signatureDblpName ?name .
            FILTER(?kind IN (dblp:AuthorSignature, dblp:EditorSignature))
        }}""")
        rows.sort(key=lambda r: int(r["ord"]))
        authors = [_strip_homonym(r["name"]) for r in rows if r["kind"].endswith("AuthorSignature")]
        editors = [_strip_homonym(r["name"]) for r in rows if r["kind"].endswith("EditorSignature")]
        return authors, editors

    def get_record(self, key: str) -> dict | None:
        """Metadati di un record nel formato hit della vecchia search API.

        Campi: ``key``, ``title``, ``year``, ``type``, ``venue``, ``authors``
        (``{"author": [{"text": ...}]}``), ``ee``, ``doi``, ``url``.
        """
        iri = _rec_iri(key)
        props = self._properties(iri)
        if "title" not in props:
            return None
        authors, _ = self._signatures(iri)
        rdf_types = [_local(t) for t in props.get("type", []) if _local(t) != "Publication"]
        doi = props.get("doi", [""])[0].replace("https://doi.org/", "")
        return {
            "key": _rec_key(iri),
            "title": props["title"][0],
            "year": props.get("yearOfPublication", [""])[0],
            "type": next((_TYPE_LABELS[t] for t in rdf_types if t in _TYPE_LABELS), ""),
            "venue": props.get("publishedIn", [""])[0],
            "authors": {"author": [{"text": a} for a in authors]},
            "ee": props.get("primaryDocumentPage", []) + props.get("documentPage", []),
            "doi": doi or None,
            "url": iri,
        }

    def get_bibtex(self, key: str, bibtex_key: str | None = None) -> str | None:
        """Ricostruisce il BibTeX in stile DBLP di un record (senza timestamp/biburl).

        ``dblp.org/rec/<key>.bib`` è dietro anti-bot, quindi i campi vengono
        composti dai triple RDF: per gli Inproceedings booktitle, editor,
        series e publisher sono letti dal record dei proceedings.
        """
        iri = _rec_iri(key)
        props = self._properties(iri)
        if "title" not in props:
            return None
        authors, _ = self._signatures(iri)
        first = lambda p, d="": (p or [d])[0]  # noqa: E731
        rdf_types = {_local(t) for t in props.get("type", [])}
        bibtype = _local(first(props.get("bibtexType"), "Misc")).lower()
        doi = first(props.get("doi")).replace("https://doi.org/", "")

        fields: list[tuple[str, str]] = []
        if authors:
            fields.append(("author", " and\n                  ".join(latex_escape(a) for a in authors)))

        venue_fields: list[tuple[str, str]] = []
        editors: list[str] = []
        if bibtype == "inproceedings" and props.get("publishedAsPartOf"):
            book = self._properties(first(props["publishedAsPartOf"]))
            _, editors = self._signatures(first(props["publishedAsPartOf"]))
            # alcuni titoli di proceedings DBLP finiscono con "." (es. NeurIPS):
            # lo stile BibTeX aggiunge già la punteggiatura dopo il booktitle
            venue_fields.append(("booktitle", latex_escape(protect_title(
                first(book.get("title"), first(props.get("publishedIn"))).rstrip(".")))))
            if book.get("publishedInSeries"):
                venue_fields.append(("series", latex_escape(first(book["publishedInSeries"]))))
            if book.get("publishedInSeriesVolume"):
                venue_fields.append(("volume", first(book["publishedInSeriesVolume"])))
            pages = first(props.get("pagination"))
            if pages:
                venue_fields.append(("pages", pages.replace("-", "--")))
            if book.get("publishedBy"):
                venue_fields.append(("publisher", latex_escape(protect_title(first(book["publishedBy"])))))
        elif bibtype == "inproceedings":
            venue_fields.append(("booktitle", latex_escape(protect_title(first(props.get("publishedIn"))))))
        else:
            journal = first(props.get("publishedInJournal"), first(props.get("publishedIn")))
            if journal:
                venue_fields.append(("journal", latex_escape(journal)))
            for name, prop in (("volume", "publishedInJournalVolume"),
                               ("number", "publishedInJournalVolumeIssue")):
                if props.get(prop):
                    venue_fields.append((name, first(props[prop])))
            pages = first(props.get("pagination"))
            if pages:
                venue_fields.append(("pages", pages.replace("-", "--")))

        if editors:
            fields.append(("editor", " and\n                  ".join(latex_escape(e) for e in editors)))
        fields.append(("title", latex_escape(protect_title(first(props["title"]).rstrip(".")))))
        fields.extend(venue_fields)
        fields.append(("year", first(props.get("yearOfPublication"))))
        url = first(props.get("primaryDocumentPage")) or first(props.get("documentPage"))
        if url:
            fields.append(("url", latex_escape(url) if "_" in url else url))
        if doi:
            fields.append(("doi", latex_escape(doi)))
        arxiv = re.match(r"10\.48550/arxiv\.(.+)$", doi, re.IGNORECASE)
        if "Informal" in rdf_types and arxiv:
            fields.append(("eprinttype", "arXiv"))
            fields.append(("eprint", arxiv.group(1)))

        bib_key = bibtex_key or ("DBLP:" + _rec_key(iri))
        body = ",\n".join(f"  {name:<12} = {{{value}}}" for name, value in fields if value)
        return f"@{bibtype}{{{bib_key},\n{body}\n}}"
