"""
BibTeX Generator — Generazione automatica di entry BibTeX per ogni paper.

Genera chiavi BibTeX nel formato `PrimoAutore_YYYY_Keyword` e entry
complete @article/@inproceedings. Se Semantic Scholar fornisce già
un BibTeX nel campo `citationStyles.bibtex`, lo usa come base e
sostituisce solo la chiave con quella custom.

Uso:
    from paperhunter.ingestion.bibtex_generator import (
        generate_bibtex_key,
        generate_bibtex_entry,
        save_master_bib,
        enrich_paper_bibtex,
    )

    # Genera solo la chiave
    key = generate_bibtex_key("Universal and Transferable Adversarial Attacks", 2023, ["Andy Zou"])
    # → "Zou_2023_Universal"

    # Genera entry completa per un Paper
    entry = generate_bibtex_entry(paper)

    # Salva file .bib cumulativo
    save_master_bib([entry1, entry2, ...], "output/references.bib")

    # Helper: arricchisce paper con bibtex_key e bibtex_entry
    paper = enrich_paper_bibtex(paper)
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

from paperhunter.models import Paper
from paperhunter.utils.logging_setup import get_logger
from paperhunter.ingestion.dblp_lookup import DblpLookup, get_dblp_lookup

logger = get_logger(__name__)


# =============================================================================
# Costanti e stop-words
# =============================================================================

# Stop-words comuni da escludere per la keyword nel bibtex_key
_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with",
    "by", "from", "at", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "it", "its", "as", "but", "not", "no", "how", "what",
    "which", "who", "when", "where", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "may", "might", "has", "have", "had",
    "via", "vs", "through", "towards", "toward", "into", "between", "about",
    "their", "our", "your", "we", "they", "us", "you", "more", "most",
})

# Acronimi noti nel campo LLM security (vengono preservati come keyword)
_KNOWN_ACRONYMS = frozenset({
    "LLM", "LLMs", "GPT", "BERT", "NLP", "AI", "ML", "DL", "NLI",
    "GCG", "RAG", "RLHF", "DPO", "PPO", "SFT", "ICL", "CoT",
    "PII", "OWASP", "CVE", "SQL", "XSS", "CSRF", "RCE",
    "GAN", "VAE", "LoRA", "QLoRA", "PEFT",
})

# Mapping venue → tipo BibTeX
_CONFERENCE_KEYWORDS = frozenset({
    "proceedings", "conference", "workshop", "symposium", "meeting",
    "neurips", "icml", "iclr", "acl", "emnlp", "naacl", "aaai",
    "cvpr", "iccv", "eccv", "sigir", "www", "kdd", "ccs", "usenix",
    "ieee s&p", "ndss",
})


# =============================================================================
# Validazione autori
# =============================================================================

_INVALID_AUTHOR_PATTERNS = [
    r'^(Qwen|Gemini|GPT|Claude|Llama|Mistral)',   # Nomi di modelli AI
    r'^(Efficient|Ving|Binyuan)$',                  # Token isolati sospetti
    r'(Labs?|Inc|Corp|University|Institute)\b',     # Affiliazioni
    r'^(Microsoft|Google|Meta|Apple|Amazon|OpenAI|Anthropic|DeepMind|Nvidia|Intel|IBM|Huawei|Alibaba|Baidu|Tencent)$',  # Aziende tech
    r'^\w{1,2}$',                                    # Nomi troppo corti (1-2 char)
]

_MAX_AUTHORS = 15


def _is_valid_author_name(name: str) -> bool:
    """Filtra nomi autori palesemente non validi."""
    if not name or len(name.strip()) < 3:
        return False
    for pattern in _INVALID_AUTHOR_PATTERNS:
        if re.search(pattern, name.strip(), re.IGNORECASE):
            return False
    return True


# =============================================================================
# Utility interne
# =============================================================================

def _sanitize_for_bibtex(text: str) -> str:
    """Rimuove o escape-a caratteri problematici per BibTeX.

    Args:
        text: Testo da sanitizzare.

    Returns:
        Testo safety per BibTeX.
    """
    if not text:
        return ""
    # Sostituisci & con \&
    text = text.replace("&", r"\&")
    # Rimuovi caratteri di controllo
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text.strip()


def _strip_accents(text: str) -> str:
    """Rimuove accenti da testo Unicode per generare chiavi ASCII-safe.

    Args:
        text: Testo potenzialmente accentato.

    Returns:
        Testo senza accenti.
    """
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def _extract_surname(author_name: str) -> str:
    """Estrae il cognome dall'autore (assume formato "Nome Cognome" o "Cognome, Nome").

    Args:
        author_name: Nome completo dell'autore.

    Returns:
        Cognome pulito, ASCII-safe, capitalizzato.
    """
    if not author_name or not author_name.strip():
        return "Unknown"

    name = author_name.strip()

    # Formato "Cognome, Nome"
    if "," in name:
        surname = name.split(",")[0].strip()
    else:
        # Formato "Nome Cognome" → ultimo token
        parts = name.split()
        surname = parts[-1] if parts else name

    # Pulisci e normalizza
    surname = _strip_accents(surname)
    surname = re.sub(r'[^a-zA-Z]', '', surname)

    if not surname:
        return "Unknown"

    return surname.capitalize()


def _extract_keyword_from_title(title: str) -> str:
    """Estrae una keyword significativa dal titolo.

    Logica:
    1. Se il titolo contiene un acronimo noto (GCG, RAG, ecc.), usa quello.
    2. Altrimenti, prendi la prima parola significativa (non stop-word).
    3. Fallback: prima parola del titolo.

    Args:
        title: Titolo del paper.

    Returns:
        Keyword breve (max ~20 char), ASCII-safe, CamelCase.
    """
    if not title:
        return "Paper"

    # Step 1: cerca acronimi noti nel titolo
    title_upper_words = set(re.findall(r'\b[A-Z][A-Za-z0-9]*\b', title))
    for acronym in _KNOWN_ACRONYMS:
        if acronym in title_upper_words:
            return acronym

    # Cerca anche fra parentesi, e.g. "... (HaluEval) ..."
    in_parens = re.findall(r'\(([A-Z][A-Za-z0-9]+)\)', title)
    for token in in_parens:
        if len(token) >= 3 and token.upper() not in _STOP_WORDS:
            return token

    # Step 2: prima parola significativa (non stop-word, non troppo corta)
    words = re.findall(r'[A-Za-z]+', title)
    for word in words:
        if word.lower() not in _STOP_WORDS and len(word) >= 3:
            return _strip_accents(word).capitalize()

    # Step 3: fallback — prima parola qualsiasi
    if words:
        return _strip_accents(words[0]).capitalize()

    return "Paper"


def _is_conference_venue(venue: str) -> bool:
    """Determina se il venue è una conferenza (→ @inproceedings) o journal (→ @article).

    Args:
        venue: Nome del venue/journal.

    Returns:
        True se sembra una conferenza, False se journal o sconosciuto.
    """
    if not venue:
        return False
    venue_lower = venue.lower()
    return any(kw in venue_lower for kw in _CONFERENCE_KEYWORDS)


def _format_authors_bibtex(authors: list[str]) -> str:
    """Formatta la lista di autori per BibTeX (separati da " and ").

    Filtra nomi invalidi (modelli AI, affiliazioni, token corti) e
    tronca a _MAX_AUTHORS se la lista è sospettamente lunga.

    Args:
        authors: Lista di nomi autori.

    Returns:
        Stringa formattata per il campo `author` BibTeX.
    """
    if not authors:
        return "Unknown"
        
    flat_authors = []
    for a in authors:
        parts = re.split(r'\s+and\s+|,', a, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip()
            if p:
                flat_authors.append(p)

    # Filtra autori invalidi
    valid = [a for a in flat_authors if _is_valid_author_name(a)]
    if not valid:
        logger.warning("Tutti gli autori filtrati come invalidi: %s", authors[:5])
        return "Unknown"

    # Tronca se lista sospettamente lunga
    if len(valid) > _MAX_AUTHORS:
        logger.warning(
            "Lista autori troncata da %d a %d (possibile corruzione)",
            len(valid), _MAX_AUTHORS,
        )
        valid = valid[:_MAX_AUTHORS]

    return " and ".join(_sanitize_for_bibtex(a) for a in valid)


# =============================================================================
# Funzioni pubbliche
# =============================================================================

def generate_bibtex_key(
    title: str,
    year: int | None = None,
    authors: list[str] | None = None,
) -> str:
    """Genera una chiave BibTeX nel formato `PrimoAutore_YYYY_Keyword`.

    Esempi:
        - "Universal and Transferable Adversarial Attacks on Aligned LMs", 2023, ["Andy Zou"]
          → "Zou_2023_Universal"
        - "HaluEval: A Large-Scale Hallucination Evaluation Benchmark", 2023, ["Junyi Li"]
          → "Li_2023_HaluEval"
        - "Siren's Song in the AI Ocean: A Survey on Hallucination in LLMs", 2024
          → "Unknown_2024_Siren"

    Args:
        title: Titolo del paper.
        year: Anno di pubblicazione (default: "XXXX" se mancante).
        authors: Lista di autori (per estrarre il cognome del primo autore).

    Returns:
        Chiave BibTeX unica e leggibile (es. "Zou_2023_GCG").
    """
    # Cognome primo autore
    if authors and authors[0]:
        surname = _extract_surname(authors[0])
    else:
        surname = "Unknown"

    # Anno
    year_str = str(year) if year else "XXXX"

    # Keyword dal titolo
    keyword = _extract_keyword_from_title(title)

    # Componi la chiave
    key = f"{surname}_{year_str}_{keyword}"

    # Assicurati sia ASCII-safe e senza spazi
    key = re.sub(r'[^a-zA-Z0-9_]', '', key)

    return key


def generate_bibtex_entry(paper: Paper) -> str:
    """Genera un'entry BibTeX completa per un paper.

    Tenta prima di recuperare l'entry BibTeX ufficiale da DBLP
    (preferendo pubblicazioni in conferenza/journal). Se non trovata
    su DBLP, genera l'entry da zero con i metadati disponibili.

    Args:
        paper: Paper con metadati (title, authors, year, venue, ecc.).

    Returns:
        Entry BibTeX completa come stringa. Stringa vuota se i dati
        sono insufficienti (mancano title e year).
    """
    # Validazione minima: serve almeno il titolo
    if not paper.title or not paper.title.strip():
        logger.warning("Impossibile generare BibTeX: titolo mancante")
        return ""

    # Genera la chiave custom
    key = paper.bibtex_key or generate_bibtex_key(
        title=paper.title,
        year=paper.year,
        authors=paper.authors,
    )

    # --- Tenta lookup DBLP ---
    dblp_entry = _try_dblp_lookup(paper, key)
    if dblp_entry is not None:
        return dblp_entry

    # --- Fallback: costruisci da zero ---
    return _build_bibtex_from_scratch(paper, key)


def _try_dblp_lookup(paper: Paper, key: str) -> str | None:
    """Tenta di recuperare il BibTeX ufficiale da DBLP.

    Se trovato, sostituisce la chiave DBLP con la chiave custom
    e marca il paper come proveniente da DBLP.

    Returns:
        Entry BibTeX con chiave custom, o None se non trovato.
    """
    try:
        dblp = get_dblp_lookup()
        result = dblp.lookup_bibtex(
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            doi=paper.doi,
        )
        if result is None:
            return None

        # Sostituisci la chiave DBLP con la nostra chiave custom
        entry = _replace_bibtex_key(result.bibtex, key)

        logger.info(
            "BibTeX DBLP ufficiale per '%s' [%s] (sim=%.2f, tipo=%s)",
            paper.title[:50], result.dblp_key, result.similarity, result.dblp_type,
        )
        return entry

    except Exception as exc:
        logger.warning("Errore DBLP lookup, fallback a generazione locale: %s", exc)
        return None


def _replace_bibtex_key(bibtex_str: str, new_key: str) -> str:
    """Sostituisce la chiave in un'entry BibTeX esistente.

    Gestisce sia il formato @type{old_key, che @type{old_key,\n

    Args:
        bibtex_str: Entry BibTeX originale (da S2).
        new_key: Nuova chiave da inserire.

    Returns:
        Entry BibTeX con la chiave sostituita.
    """
    # Pattern: @tipo{chiave_vecchia,
    pattern = r'(@\w+)\s*\{[^,]+,'
    replacement = rf'\1{{{new_key},'
    result = re.sub(pattern, replacement, bibtex_str, count=1)

    if result == bibtex_str:
        # Se la regex non ha matchato, prova a forzare
        logger.debug("Fallback per sostituzione chiave BibTeX")
        # Trova la prima { e la prima ,
        brace_idx = bibtex_str.find('{')
        comma_idx = bibtex_str.find(',', brace_idx)
        if brace_idx >= 0 and comma_idx > brace_idx:
            result = bibtex_str[:brace_idx + 1] + new_key + bibtex_str[comma_idx:]

    return result


def _build_bibtex_from_scratch(paper: Paper, key: str) -> str:
    """Costruisce un'entry BibTeX da zero usando i metadati del Paper.

    Args:
        paper: Paper con i metadati disponibili.
        key: Chiave BibTeX generata.

    Returns:
        Entry BibTeX formattata.
    """
    # Determina il tipo: @inproceedings se conferenza, altrimenti @article
    venue = paper.venue or ""
    journal = paper.journal or ""
    is_conference = _is_conference_venue(venue) or _is_conference_venue(journal)

    # FASE 3: Se ha arxiv_id ma non DOI → è un preprint, non pubblicato in conferenza
    if paper.arxiv_id and not paper.doi:
        entry_type = "misc"
        is_conference = False
        if venue or journal:
            logger.debug(
                "Paper con arxiv_id senza DOI: forzato @misc (venue ignorata: '%s')",
                venue or journal,
            )
    elif is_conference:
        entry_type = "inproceedings"
    elif journal:
        entry_type = "article"
    elif paper.arxiv_id:
        entry_type = "article"  # ArXiv preprint come article
    else:
        entry_type = "misc"

    # Costruisci i campi
    fields: list[tuple[str, str]] = []

    # Title (obbligatorio)
    fields.append(("title", f"{{{_sanitize_for_bibtex(paper.title)}}}"))

    # Author (obbligatorio)
    authors_str = _format_authors_bibtex(paper.authors) if paper.authors else "Unknown"
    fields.append(("author", f"{{{authors_str}}}"))

    # Year (obbligatorio)
    if paper.year:
        fields.append(("year", f"{{{paper.year}}}"))
    else:
        fields.append(("year", "{XXXX}"))

    # Venue/Journal/Booktitle
    if entry_type == "inproceedings":
        if venue:
            fields.append(("booktitle", f"{{{_sanitize_for_bibtex(venue)}}}"))
        elif journal:
            fields.append(("booktitle", f"{{{_sanitize_for_bibtex(journal)}}}"))
    elif entry_type == "article":
        if journal:
            fields.append(("journal", f"{{{_sanitize_for_bibtex(journal)}}}"))
        elif venue:
            fields.append(("journal", f"{{{_sanitize_for_bibtex(venue)}}}"))
        elif paper.arxiv_id:
            fields.append(("journal", "{arXiv preprint}"))
    elif entry_type == "misc" and paper.arxiv_id:
        fields.append(("howpublished", "{arXiv preprint}"))

    # Month (opzionale)
    if paper.month:
        month_names = {
            1: "jan", 2: "feb", 3: "mar", 4: "apr",
            5: "may", 6: "jun", 7: "jul", 8: "aug",
            9: "sep", 10: "oct", 11: "nov", 12: "dec",
        }
        fields.append(("month", month_names.get(paper.month, str(paper.month))))

    # DOI
    if paper.doi:
        fields.append(("doi", f"{{{paper.doi}}}"))

    # ArXiv ID (come eprint) — FASE 4: fallback da URL + validazione formato
    arxiv_id = paper.arxiv_id
    if not arxiv_id and paper.url:
        m = re.search(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', paper.url)
        if m:
            arxiv_id = m.group(1)
            logger.debug("arxiv_id estratto da URL: %s", arxiv_id)
    if arxiv_id:
        # Validazione formato: deve essere YYMM.NNNNN(vN)?
        if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id):
            fields.append(("eprint", f"{{{arxiv_id}}}"))
            fields.append(("archiveprefix", "{arXiv}"))
        else:
            logger.warning("arxiv_id formato invalido, ignorato: %s", arxiv_id)

    # URL
    if paper.url and paper.url.strip():
        fields.append(("url", f"{{{paper.url}}}"))

    # Abstract (opzionale, troncato se lungo)
    if paper.abstract and len(paper.abstract) > 50:
        abstract_truncated = paper.abstract[:500]
        if len(paper.abstract) > 500:
            abstract_truncated += "..."
        fields.append(("abstract", f"{{{_sanitize_for_bibtex(abstract_truncated)}}}"))

    # Formatta l'entry
    lines = [f"@{entry_type}{{{key},"]
    for i, (field_name, field_value) in enumerate(fields):
        comma = "," if i < len(fields) - 1 else ""
        lines.append(f"  {field_name} = {field_value}{comma}")
    lines.append("}")

    return "\n".join(lines)


def enrich_paper_bibtex(paper: Paper) -> Paper:
    """Arricchisce un Paper con bibtex_key e bibtex_entry generati.

    Tenta prima un lookup su DBLP per ottenere l'entry BibTeX ufficiale.
    Se trovata su DBLP, marca il paper con bibtex_source="dblp" così
    che ensure_complete_bibtex() non sovrascriva l'entry verificata.

    Args:
        paper: Paper da arricchire.

    Returns:
        Paper con bibtex_key e bibtex_entry popolati.
    """
    paper_dict = paper.model_dump()

    # Genera la chiave se mancante
    if not paper.bibtex_key:
        paper_dict["bibtex_key"] = generate_bibtex_key(
            title=paper.title,
            year=paper.year,
            authors=paper.authors,
        )

    key = paper_dict["bibtex_key"]

    # --- Tenta DBLP direttamente per tracciare la provenienza ---
    dblp_entry = _try_dblp_lookup(Paper(**paper_dict), key)
    if dblp_entry is not None:
        paper_dict["bibtex_entry"] = dblp_entry
        paper_dict["bibtex_source"] = "dblp"
        # v3.6 — Citability: DBLP-verified ⇒ citable di default
        paper_dict["dblp_verified"] = True
        paper_dict["is_citable"] = True
        paper_dict["citation_tier"] = "dblp"
    else:
        # Fallback: genera da zero
        paper_dict["bibtex_entry"] = _build_bibtex_from_scratch(
            Paper(**paper_dict), key,
        )
        paper_dict["bibtex_source"] = "local"
        paper_dict["dblp_verified"] = False

        # v3.6 — Tier "s2_high_conf" opt-in: paper non DBLP ma con molte citazioni S2
        try:
            from paperhunter.config import settings as _settings
            cit_cfg = _settings.citability
        except Exception:
            cit_cfg = None

        cit_count = paper_dict.get("citation_count") or 0
        if (
            cit_cfg is not None
            and cit_cfg.allow_s2_high_conf
            and isinstance(cit_count, int)
            and cit_count >= cit_cfg.s2_min_citation_count
        ):
            paper_dict["is_citable"] = True
            paper_dict["citation_tier"] = "s2_high_conf"
        else:
            paper_dict["is_citable"] = False
            # arXiv preprint vs uncited context
            arxiv_id = paper_dict.get("arxiv_id")
            paper_dict["citation_tier"] = "preprint" if arxiv_id else "uncited_context"

    result = Paper(**paper_dict)
    return ensure_complete_bibtex(result)


def validate_bibtex_completeness(entry: str) -> dict[str, str]:
    """Valida la completezza di un'entry BibTeX per i campi critici.

    Restituisce un dizionario campo → stato per i campi obbligatori.
    Lo stato può essere: "present", "missing", "placeholder".

    Placeholder noti: "Unknown", "XXXX", "", "arXiv Preprint" quando
    inaspettato, ecc.

    Args:
        entry: Stringa BibTeX completa.

    Returns:
        Dict con chiavi: "title", "author", "year", "venue", "url".
        Ogni valore è "present", "missing", o "placeholder".
    """
    _PLACEHOLDER_VALUES = frozenset({"unknown", "xxxx", "", "n/a", "none"})

    result: dict[str, str] = {}

    def _check_field(pattern: str) -> str:
        m = re.search(pattern, entry, re.IGNORECASE | re.DOTALL)
        if not m:
            return "missing"
        val = m.group(1).strip("{} \t\n")
        if val.lower() in _PLACEHOLDER_VALUES:
            return "placeholder"
        if not val:
            return "missing"
        return "present"

    result["title"] = _check_field(r'title\s*=\s*(\{[^}]*\}|"[^"]*")')
    result["author"] = _check_field(r'author\s*=\s*(\{[^}]*\}|"[^"]*")')
    result["year"] = _check_field(r'year\s*=\s*(\{[^}]*\}|"[^"]*"|\d{4})')

    # venue = booktitle oppure journal (in @misc: howpublished)
    if re.search(r'booktitle\s*=', entry, re.IGNORECASE):
        result["venue"] = _check_field(r'booktitle\s*=\s*(\{[^}]*\}|"[^"]*")')
    elif re.search(r'journal\s*=', entry, re.IGNORECASE):
        result["venue"] = _check_field(r'journal\s*=\s*(\{[^}]*\}|"[^"]*")')
    elif re.search(r'howpublished\s*=', entry, re.IGNORECASE):
        result["venue"] = _check_field(r'howpublished\s*=\s*(\{[^}]*\}|"[^"]*")')
    else:
        result["venue"] = "missing"

    result["url"] = _check_field(r'url\s*=\s*(\{[^}]*\}|"[^"]*")')

    return result


def ensure_complete_bibtex(paper: Paper) -> Paper:
    """Garantisce che il Paper abbia un'entry BibTeX valida e completa.

    Se l'entry proviene da DBLP (bibtex_source="dblp"), la considera
    già verificata e non la sovrascrive con @misc. Altrimenti applica
    le euristiche standard per completare campi mancanti.

    Il campo `bibtex_entry` non sarà mai None dopo questa funzione.

    Args:
        paper: Paper da completare.

    Returns:
        Paper con bibtex_entry garantita, valida e completa.
    """
    paper_dict = paper.model_dump()

    # Assicura che la chiave esista
    if not paper_dict.get("bibtex_key"):
        paper_dict["bibtex_key"] = generate_bibtex_key(
            title=paper.title,
            year=paper.year,
            authors=paper.authors,
        )

    # Genera entry base se mancante
    if not paper_dict.get("bibtex_entry"):
        temp = Paper(**paper_dict)
        paper_dict["bibtex_entry"] = generate_bibtex_entry(temp)

    entry = paper_dict["bibtex_entry"] or ""
    key = paper_dict["bibtex_key"]
    is_dblp = paper_dict.get("bibtex_source") == "dblp"

    # --- Correggi il tipo se è un preprint arXiv ---
    # NON sovrascrivere entry verificate da DBLP: se DBLP dice
    # @inproceedings, il paper è effettivamente pubblicato in conferenza
    # anche se ha un arxiv_id.
    if not is_dblp:
        venue_str = (paper.venue or paper.journal or "").lower()
        is_arxiv = (
            "arxiv" in venue_str
            or bool(paper.arxiv_id)
            or "preprint" in venue_str
        )
        if is_arxiv and not re.match(r"@misc\{", entry, re.IGNORECASE):
            entry = _build_misc_bibtex(paper, key)
        elif not entry.strip():
            entry = _build_misc_bibtex(paper, key)

        # --- Completa campi critici con placeholder mancanti ---
        entry = _patch_bibtex_field(entry, "title", paper.title or "Untitled")
        authors_str = _format_authors_bibtex(paper.authors) if paper.authors else "Anonymous"
        entry = _patch_bibtex_field(entry, "author", authors_str)

        # Year: prova anche dal bibtex_key (es. "Zou_2023_GCG" → 2023)
        year_val = paper.year
        if year_val is None and key:
            m = re.search(r'_(\d{4})_', key)
            if m:
                year_val = int(m.group(1))
        entry = _patch_bibtex_field(entry, "year", str(year_val) if year_val else "2000")
    elif not entry.strip():
        # Entry DBLP vuota (non dovrebbe succedere): fallback
        entry = _build_misc_bibtex(paper, key)

    paper_dict["bibtex_entry"] = entry
    return Paper(**paper_dict)


def _build_misc_bibtex(paper: Paper, key: str) -> str:
    """Costruisce un'entry @misc per preprint o paper senza venue."""
    fields: list[tuple[str, str]] = [
        ("title", f"{{{_sanitize_for_bibtex(paper.title)}}}"),
        ("author", f"{{{_format_authors_bibtex(paper.authors) if paper.authors else 'Anonymous'}}}"),
        ("year", f"{{{paper.year or 2000}}}"),
        ("howpublished", f"{{arXiv preprint}}"),
    ]
    if paper.arxiv_id:
        fields.append(("eprint", f"{{{paper.arxiv_id}}}"))
        fields.append(("archiveprefix", "{arXiv}"))
    if paper.url:
        fields.append(("url", f"{{{paper.url}}}"))

    lines = [f"@misc{{{key},"]
    for i, (fname, fval) in enumerate(fields):
        comma = "," if i < len(fields) - 1 else ""
        lines.append(f"  {fname} = {fval}{comma}")
    lines.append("}")
    return "\n".join(lines)


def _patch_bibtex_field(entry: str, field_name: str, value: str) -> str:
    """Aggiunge o sostituisce un campo nell'entry BibTeX se mancante o placeholder.

    Non sovrascrive valori validi già presenti.

    Args:
        entry: Entry BibTeX da aggiornare.
        field_name: Nome del campo (es. "year").
        value: Valore da inserire se il campo è assente o placeholder.

    Returns:
        Entry aggiornata.
    """
    _PLACEHOLDER_VALUES = frozenset({"unknown", "xxxx", "", "n/a", "none", "anonymous"})

    # Controlla se il campo esiste già con un valore valido
    pattern = rf'{field_name}\s*=\s*(\{{[^}}]*\}}|"[^"]*"|\d{{4}})'
    m = re.search(pattern, entry, re.IGNORECASE)
    if m:
        existing = m.group(1).strip("{} \t\n\"")
        if existing.lower() not in _PLACEHOLDER_VALUES and existing:
            return entry  # già presente e valido
        # Sostituisci il placeholder
        new_field = f'{field_name} = {{{_sanitize_for_bibtex(value)}}}'
        return re.sub(pattern, new_field, entry, count=1, flags=re.IGNORECASE)

    # Campo assente: inserisci prima della chiusura "}"
    new_field = f'  {field_name} = {{{_sanitize_for_bibtex(value)}}}'
    # Trova l'ultima } che chiude l'entry
    last_brace = entry.rfind("}")
    if last_brace >= 0:
        return entry[:last_brace] + f",\n{new_field}\n" + entry[last_brace:]
    return entry + f"\n{new_field}"


def _warn_suspicious_entries(entries: list[str]) -> None:
    """Loga warning per entry BibTeX con possibili problemi di qualità.

    Controlla:
    - Numero autori > _MAX_AUTHORS
    - Nomi autori che matchano pattern sospetti
    - @inproceedings senza DOI
    - arxiv_id presente ma eprint mancante
    - Abstract assente o troppo corto (< 50 char)
    - Anno fuori range ragionevole
    """
    for entry in entries:
        key_match = re.search(r'@(\w+)\{([^,]+),', entry)
        if not key_match:
            continue
        entry_type = key_match.group(1).lower()
        bib_key = key_match.group(2).strip()

        # Autori: numero eccessivo
        author_match = re.search(r'author\s*=\s*\{([^}]*)\}', entry, re.IGNORECASE)
        if author_match:
            author_str = author_match.group(1)
            author_count = len(author_str.split(" and "))
            if author_count > _MAX_AUTHORS:
                logger.warning("[QA] %s: %d autori (possibile corruzione)", bib_key, author_count)
            # Nomi sospetti
            for name in author_str.split(" and "):
                name = name.strip()
                if not _is_valid_author_name(name):
                    logger.warning("[QA] %s: autore sospetto '%s'", bib_key, name)

        # @inproceedings senza DOI
        if entry_type == "inproceedings" and not re.search(r'doi\s*=', entry, re.IGNORECASE):
            logger.warning("[QA] %s: @inproceedings senza DOI", bib_key)

        # arxiv_id/URL arXiv ma eprint mancante
        has_arxiv_url = bool(re.search(r'arxiv\.org', entry, re.IGNORECASE))
        has_eprint = bool(re.search(r'eprint\s*=', entry, re.IGNORECASE))
        if has_arxiv_url and not has_eprint:
            logger.warning("[QA] %s: URL arXiv presente ma eprint mancante", bib_key)

        # Abstract assente o troppo corto
        abs_match = re.search(r'abstract\s*=\s*\{([^}]*)\}', entry, re.IGNORECASE)
        if not abs_match or len(abs_match.group(1).strip()) < 50:
            logger.warning("[QA] %s: abstract assente o troppo corto", bib_key)

        # Anno fuori range
        year_match = re.search(r'year\s*=\s*\{?(\d{4})\}?', entry, re.IGNORECASE)
        if year_match:
            year_val = int(year_match.group(1))
            if year_val < 1990 or year_val > 2035:
                logger.warning("[QA] %s: anno fuori range (%d)", bib_key, year_val)


def save_master_bib(
    entries: list[str],
    output_path: str | Path | None = None,
    papers: list[Paper] | None = None,
) -> Path:
    """Salva tutte le entry BibTeX in un file .bib cumulativo.

    Rimuove entry duplicate (per chiave), ordina alfabeticamente per chiave,
    e aggiunge un header informativo.

    v3.6 — Two-Tier Corpus:
      Se ``papers`` è fornito, le entry vengono filtrate mantenendo solo
      i paper con ``is_citable=True``. Le entry escluse possono essere
      salvate in un file parallelo ``references_uncited.bib`` per audit
      (controllato da ``settings.citability.write_uncited_audit_bib``).

    Args:
        entries: Lista di entry BibTeX (stringhe). Se ``papers`` è fornito
                 questo argomento può essere lasciato vuoto: le entry verranno
                 estratte dai Paper.
        output_path: Percorso del file .bib di output.
                     Default: `output/references.bib`.
        papers: Opzionale, lista di Paper per applicare il filtro citability.

    Returns:
        Path del file .bib salvato (citable).
    """
    from paperhunter.config import settings

    if output_path is None:
        output_path = Path(settings.output.base_dir) / "references.bib"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- v3.6: split citable vs uncited usando i Paper, se forniti ---
    citable_entries: list[str] = []
    uncited_entries: list[str] = []
    if papers is not None:
        for p in papers:
            entry = (p.bibtex_entry or "").strip()
            if not entry:
                continue
            if getattr(p, "is_citable", False):
                citable_entries.append(entry)
            else:
                uncited_entries.append(entry)
        # Concatena anche eventuali stringhe "extra" passate via `entries`
        # (tutte considerate citable per backward-compat se nessun Paper match).
        if entries:
            known_keys = {
                m.group(1).strip()
                for e in citable_entries + uncited_entries
                if (m := re.search(r'@\w+\{([^,]+),', e))
            }
            for e in entries:
                if not e or not e.strip():
                    continue
                m = re.search(r'@\w+\{([^,]+),', e)
                if m and m.group(1).strip() in known_keys:
                    continue
                citable_entries.append(e.strip())
        valid_entries = citable_entries
    else:
        # Backward-compat: filtra solo entry vuote
        valid_entries = [e.strip() for e in entries if e and e.strip()]

    # Deduplica per chiave BibTeX
    def _dedupe(lst: list[str]) -> list[str]:
        seen: dict[str, str] = {}
        for entry in lst:
            key_match = re.search(r'@\w+\{([^,]+),', entry)
            if key_match:
                bib_key = key_match.group(1).strip()
                if bib_key not in seen:
                    seen[bib_key] = entry
                else:
                    logger.debug("BibTeX duplicato, skip: %s", bib_key)
            else:
                logger.warning("Entry BibTeX senza chiave valida, skip: %s", entry[:60])
        return sorted(seen.values(), key=lambda e: e.lower())

    sorted_entries = _dedupe(valid_entries)

    # FASE 6: Warning hook — segnala entry sospette prima della scrittura
    _warn_suspicious_entries(sorted_entries)

    # Scrivi file principale (citable)
    header = (
        "% ==========================================================================\n"
        "% LLM Security Survey — Riferimenti Bibliografici (CITABLE)\n"
        "% Generato automaticamente da PaperHunter v3.6\n"
        f"% Totale entry citable: {len(sorted_entries)}\n"
        "% Solo paper verificati su DBLP (o S2 high-conf se abilitato).\n"
        "% ==========================================================================\n"
        "\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for entry in sorted_entries:
            f.write(entry)
            f.write("\n\n")

    logger.info("Master .bib salvato: %s (%d entry citable)", output_path, len(sorted_entries))

    # --- Audit file: paper non citable usati solo come contesto ---
    try:
        cit_cfg = settings.citability
    except Exception:
        cit_cfg = None
    if uncited_entries and cit_cfg is not None and cit_cfg.write_uncited_audit_bib:
        audit_path = output_path.with_name("references_uncited.bib")
        sorted_uncited = _dedupe(uncited_entries)
        audit_header = (
            "% ==========================================================================\n"
            "% LLM Security Survey — Riferimenti UNCITED (audit only)\n"
            "% Paper ingeriti per il contesto ma NON citabili nel survey.\n"
            f"% Totale entry: {len(sorted_uncited)}\n"
            "% ==========================================================================\n"
            "\n"
        )
        with open(audit_path, "w", encoding="utf-8") as f:
            f.write(audit_header)
            for entry in sorted_uncited:
                f.write(entry)
                f.write("\n\n")
        logger.info(
            "Audit .bib salvato: %s (%d entry uncited)",
            audit_path,
            len(sorted_uncited),
        )

    return output_path


def load_master_bib(bib_path: str | Path) -> dict[str, str]:
    """Carica un file .bib e restituisce un dict chiave → entry.

    Args:
        bib_path: Percorso del file .bib.

    Returns:
        Dict dove le chiavi sono le bibtex_key e i valori le entry complete.
    """
    bib_path = Path(bib_path)
    if not bib_path.exists():
        logger.warning("File .bib non trovato: %s", bib_path)
        return {}

    content = bib_path.read_text(encoding="utf-8")

    # Split sulle entry (ogni @tipo{...} è un'entry)
    # Regex: trova ogni blocco @type{key, ... }
    entries: dict[str, str] = {}
    pattern = r'(@\w+\{[^@]+\})'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        key_match = re.search(r'@\w+\{([^,]+),', match)
        if key_match:
            key = key_match.group(1).strip()
            entries[key] = match.strip()

    logger.debug("Caricate %d entry da %s", len(entries), bib_path)
    return entries


def validate_bibtex_entry(entry: str) -> tuple[bool, list[str]]:
    """Valida un'entry BibTeX verificando i campi minimi obbligatori.

    Campi obbligatori: title, author, year.

    Args:
        entry: Entry BibTeX come stringa.

    Returns:
        Tuple (is_valid, list_of_issues).
    """
    issues: list[str] = []

    if not entry or not entry.strip():
        return False, ["Entry vuota"]

    # Verifica struttura base @tipo{chiave,
    if not re.search(r'@\w+\{[^,]+,', entry):
        issues.append("Struttura @type{key, mancante")

    # Campi obbligatori
    required_fields = ["title", "author", "year"]
    for field in required_fields:
        # pattern per: field = {value} o field = value
        pattern = rf'\b{field}\s*='
        if not re.search(pattern, entry, re.IGNORECASE):
            issues.append(f"Campo obbligatorio mancante: {field}")

    return len(issues) == 0, issues
