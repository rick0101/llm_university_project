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

    Args:
        authors: Lista di nomi autori.

    Returns:
        Stringa formattata per il campo `author` BibTeX.
    """
    if not authors:
        return "Unknown"
    return " and ".join(_sanitize_for_bibtex(a) for a in authors)


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

    Se il paper ha già `bibtex_entry` proveniente da Semantic Scholar,
    lo usa come base ma sostituisce la chiave con quella custom generata
    da `generate_bibtex_key()`.

    Se non esiste un BibTeX S2, ne genera uno da zero con i metadati
    disponibili nel Paper.

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

    # Se S2 ha fornito un BibTeX, sostituisci solo la chiave
    if paper.bibtex_entry and paper.bibtex_entry.strip():
        return _replace_bibtex_key(paper.bibtex_entry, key)

    # Altrimenti: genera da zero
    return _build_bibtex_from_scratch(paper, key)


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

    if is_conference:
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

    # ArXiv ID (come eprint)
    if paper.arxiv_id:
        fields.append(("eprint", f"{{{paper.arxiv_id}}}"))
        fields.append(("archiveprefix", "{arXiv}"))

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

    Non sovrascrive se il paper ha già una bibtex_key valida
    E una bibtex_entry con la chiave corretta.

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

    # Genera/aggiorna l'entry BibTeX
    # Prima aggiorna il paper con la nuova key (serve a generate_bibtex_entry)
    temp_paper = Paper(**paper_dict)
    paper_dict["bibtex_entry"] = generate_bibtex_entry(temp_paper)

    return Paper(**paper_dict)


def save_master_bib(
    entries: list[str],
    output_path: str | Path | None = None,
) -> Path:
    """Salva tutte le entry BibTeX in un file .bib cumulativo.

    Rimuove entry duplicate (per chiave), ordina alfabeticamente per chiave,
    e aggiunge un header informativo.

    Args:
        entries: Lista di entry BibTeX (stringhe).
        output_path: Percorso del file .bib di output.
                     Default: `output/references.bib`.

    Returns:
        Path del file .bib salvato.
    """
    from paperhunter.config import settings

    if output_path is None:
        output_path = Path(settings.output.base_dir) / "references.bib"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filtra entry vuote
    valid_entries = [e.strip() for e in entries if e and e.strip()]

    # Deduplica per chiave BibTeX
    seen_keys: dict[str, str] = {}
    for entry in valid_entries:
        key_match = re.search(r'@\w+\{([^,]+),', entry)
        if key_match:
            bib_key = key_match.group(1).strip()
            if bib_key not in seen_keys:
                seen_keys[bib_key] = entry
            else:
                logger.debug("BibTeX duplicato, skip: %s", bib_key)
        else:
            logger.warning("Entry BibTeX senza chiave valida, skip: %s", entry[:60])

    # Ordina per chiave
    sorted_entries = sorted(seen_keys.values(), key=lambda e: e.lower())

    # Scrivi file
    header = (
        "% ==========================================================================\n"
        "% LLM Security Survey — Riferimenti Bibliografici\n"
        "% Generato automaticamente da PaperHunter v2.0\n"
        f"% Totale entry: {len(sorted_entries)}\n"
        "% ==========================================================================\n"
        "\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for entry in sorted_entries:
            f.write(entry)
            f.write("\n\n")

    logger.info("Master .bib salvato: %s (%d entry)", output_path, len(sorted_entries))
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
