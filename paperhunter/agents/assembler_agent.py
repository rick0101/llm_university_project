"""
Assembler Agent — Assembla il survey finale con abstract, intro, conclusion, bibliografia.

Agent 4 nella pipeline agentica LangGraph. Step:
1. Ordina le sezioni secondo l'outline
2. Genera Abstract (Gemini Pro, da riassunto di tutte le sezioni)
3. Genera/Raffina Introduction (contesto + motivazione + overview sezioni)
4. Genera Conclusion (sintesi + future work)
5. Assembla tutte le sezioni in un unico documento Markdown
6. Genera bibliografia .bib dalle citazioni effettivamente usate
7. Versioning: salva con timestamp e versione, genera changelog se esiste versione precedente

Uso standalone::

    from paperhunter.agents.assembler_agent import assemble_paper, generate_abstract
    from paperhunter.agents.state import create_initial_state

    state = create_initial_state(...)
    # ... dopo Writer e Critic ...
    result = assemble_paper(state)

Uso come nodo LangGraph::

    from langgraph.graph import StateGraph
    from paperhunter.agents.state import SurveyState
    from paperhunter.agents.assembler_agent import assemble

    graph = StateGraph(SurveyState)
    graph.add_node("assemble", assemble)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from paperhunter.agents.state import (
    PipelinePhase,
    SectionDraft,
    SectionOutline,
    SectionStatus,
    SurveyState,
    collect_all_citation_keys,
    get_outline_from_state,
    get_provenance_from_state,
    get_sections_from_state,
    get_verifications_from_state,
    get_version_from_state,
)
from paperhunter.config import settings
from paperhunter.ingestion.bibtex_generator import load_master_bib, save_master_bib
from paperhunter.models import SurveyVersion
from paperhunter.utils.llm_factory import (
    get_llm,
    llm_call_with_retry,
)
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_panel,
    print_success,
    print_table,
    print_warning,
)
from paperhunter.utils.prompt_loader import get_prompt

logger = get_logger(__name__)


# ============================================================================
# Prompt templates
# ============================================================================

_ABSTRACT_PROMPT_TEMPLATE = get_prompt("ABSTRACT_PROMPT")

_INTRODUCTION_PROMPT_TEMPLATE = get_prompt("INTRODUCTION_PROMPT")

_CONCLUSION_PROMPT_TEMPLATE = get_prompt("CONCLUSION_PROMPT")

_CHANGELOG_PROMPT_TEMPLATE = get_prompt("CHANGELOG_PROMPT")


# ============================================================================
# Core assembly functions
# ============================================================================

def generate_abstract(
    state: SurveyState,
    *,
    force_model: str | None = None,
) -> SectionDraft:
    """Genera l'abstract del survey basandosi su tutte le sezioni scritte.

    Args:
        state: Stato corrente con sezioni scritte.
        force_model: Override modello LLM (default: assembler_model da config).

    Returns:
        SectionDraft per l'abstract.
    """
    logger.info("Generazione abstract...")

    sections = get_sections_from_state(state)
    outline = get_outline_from_state(state)

    # Prepara overview delle sezioni
    sections_overview = _format_sections_overview(sections, outline)
    all_keys = collect_all_citation_keys(state)

    # Categorie coperte (dalle sezioni body)
    categories = set()
    for o in outline:
        if o.section_type == "body":
            categories.update(o.related_categories)
    categories_str = ", ".join(sorted(categories)) if categories else "N/A"

    prompt = _ABSTRACT_PROMPT_TEMPLATE.format(
        sections_overview=sections_overview,
        total_sections=len(sections),
        total_citations=len(all_keys),
        categories=categories_str,
    )

    model_tier = force_model or settings.agents.assembler_model
    llm = get_llm(
        model_tier,
        max_tokens=4096,
        thinking_enabled=False,
    )

    t0 = time.time()
    try:
        raw = llm_call_with_retry(
            llm, prompt,
            max_retries=2,
            base_delay=5.0,
            max_delay=30.0,
        )
        text = raw.strip()
    except Exception as e:
        err = f"Errore generazione abstract: {e}"
        logger.error(err, exc_info=True)
        print_warning(err)
        raise RuntimeError(err) from e

    elapsed = time.time() - t0
    word_count = len(text.split())
    logger.info(f"Abstract generato: {word_count} parole in {elapsed:.1f}s")

    return SectionDraft(
        section_key="abstract",
        title="Abstract",
        text=text,
        citations=[],  # Abstract tipicamente non ha citazioni
        word_count=word_count,
        status=SectionStatus.FINALIZED,
        iteration=1,
        drafted_at=datetime.now(),
    )


def generate_introduction(
    state: SurveyState,
    *,
    force_model: str | None = None,
) -> SectionDraft:
    """Genera o raffina l'introduction del survey.

    Se un'introduction già esiste (dal Writer), la raffina con
    contesto dalle L2 summaries e dall'outline completo.
    Se non esiste, la genera da zero.

    Args:
        state: Stato corrente.
        force_model: Override modello LLM.

    Returns:
        SectionDraft per l'introduction.
    """
    logger.info("Generazione/Raffinamento introduction...")

    sections = get_sections_from_state(state)
    outline = get_outline_from_state(state)
    from paperhunter.agents.state import get_l2_summaries_from_state
    l2_summaries = get_l2_summaries_from_state(state)

    # Outline overview
    outline_overview = "\n".join(
        f"  {o.order}. **{o.title}** ({o.section_type}): {o.description[:120]}"
        for o in sorted(outline, key=lambda x: x.order)
    )

    # L2 overview compatto
    l2_overview = ""
    if l2_summaries:
        parts = []
        for l2 in l2_summaries:
            parts.append(
                f"### {l2.category} ({l2.paper_count} papers)\n"
                f"- Trends: {l2.temporal_trends[:200]}...\n"
                f"- Gaps: {l2.open_gaps[:200]}...\n"
                f"- Testing: {l2.testing_overview[:200]}...\n"
            )
        l2_overview = "\n".join(parts)
    else:
        l2_overview = "(No L2 summaries available)"

    # Categorie e time span
    categories = set()
    for o in outline:
        categories.update(o.related_categories)
    categories_str = ", ".join(sorted(categories)) if categories else "N/A"

    version_info = get_version_from_state(state)
    paper_count = version_info.paper_count if version_info.paper_count > 0 else "N/A"

    # Check per intro esistente
    existing_intro_text = ""
    if "introduction" in sections:
        existing_intro_text = (
            f"\n## Existing introduction (refine and improve):\n"
            f"{sections['introduction'].text[:3000]}\n"
        )

    prompt = _INTRODUCTION_PROMPT_TEMPLATE.format(
        outline_overview=outline_overview,
        l2_overview=l2_overview,
        paper_count=paper_count,
        categories=categories_str,
        time_span="2019-2026",  # TODO: calcolare dinamicamente dai metadata
        existing_intro=existing_intro_text,
    )

    model_tier = force_model or settings.agents.assembler_model
    llm = get_llm(model_tier)

    t0 = time.time()
    try:
        raw = llm_call_with_retry(
            llm, prompt,
            max_retries=6,
            base_delay=15.0,
            max_delay=120.0,
        )
        text = raw.strip()
    except Exception as e:
        err = f"Errore generazione introduction: {e}"
        logger.error(err, exc_info=True)
        # Se c'è un'intro esistente, rimane quella
        if "introduction" in sections:
            logger.info("Mantenuta introduction esistente dal Writer.")
            return sections["introduction"]
        text = (
            "Large Language Models (LLMs) have emerged as transformative tools "
            "across numerous domains. However, their widespread adoption has "
            "raised significant security concerns. "
            "[Introduction generation failed — placeholder]"
        )
        print_warning(err)

    elapsed = time.time() - t0

    # Estrai citazioni dal testo generato
    citations = _extract_citations_from_text(text)
    word_count = len(text.split())
    logger.info(
        f"Introduction generata: {word_count} parole, "
        f"{len(citations)} citazioni in {elapsed:.1f}s"
    )

    return SectionDraft(
        section_key="introduction",
        title="Introduction",
        text=text,
        citations=citations,
        word_count=word_count,
        status=SectionStatus.FINALIZED,
        iteration=1,
        drafted_at=datetime.now(),
    )


def generate_conclusion(
    state: SurveyState,
    *,
    force_model: str | None = None,
) -> SectionDraft:
    """Genera la conclusion e future work del survey.

    Args:
        state: Stato corrente con sezioni verificate.
        force_model: Override modello LLM.

    Returns:
        SectionDraft per la conclusion.
    """
    logger.info("Generazione conclusion...")

    sections = get_sections_from_state(state)
    outline = get_outline_from_state(state)
    from paperhunter.agents.state import get_l2_summaries_from_state
    l2_summaries = get_l2_summaries_from_state(state)

    # Sommario delle sezioni scritte
    sections_summary = _format_sections_summary(sections, outline)

    # Open gaps dai L2
    open_gaps = ""
    if l2_summaries:
        parts = []
        for l2 in l2_summaries:
            if l2.open_gaps:
                parts.append(f"**{l2.category}:** {l2.open_gaps[:300]}")
        open_gaps = "\n".join(parts) if parts else "No open gaps identified."
    else:
        open_gaps = "No L2 summaries available for gap analysis."

    # Key findings (estratti dalle sezioni)
    key_findings = _extract_key_findings(sections)

    prompt = _CONCLUSION_PROMPT_TEMPLATE.format(
        sections_summary=sections_summary,
        open_gaps=open_gaps,
        key_findings=key_findings,
    )

    model_tier = force_model or settings.agents.assembler_model
    llm = get_llm(model_tier)

    t0 = time.time()
    try:
        raw = llm_call_with_retry(
            llm, prompt,
            max_retries=6,
            base_delay=15.0,
            max_delay=120.0,
        )
        text = raw.strip()
    except Exception as e:
        err = f"Errore generazione conclusion: {e}"
        logger.error(err, exc_info=True)
        # Se esiste già una conclusion dal Writer, mantenerla
        if "conclusion" in sections:
            logger.info("Mantenuta conclusion esistente dal Writer.")
            return sections["conclusion"]
        text = (
            "This survey has provided a comprehensive overview of security "
            "challenges in Large Language Models. "
            "[Conclusion generation failed — placeholder]"
        )
        print_warning(err)

    elapsed = time.time() - t0

    citations = _extract_citations_from_text(text)
    word_count = len(text.split())
    logger.info(
        f"Conclusion generata: {word_count} parole, "
        f"{len(citations)} citazioni in {elapsed:.1f}s"
    )

    return SectionDraft(
        section_key="conclusion",
        title="Conclusion and Future Work",
        text=text,
        citations=citations,
        word_count=word_count,
        status=SectionStatus.FINALIZED,
        iteration=1,
        drafted_at=datetime.now(),
    )


def assemble_document(
    state: SurveyState,
    *,
    abstract_draft: SectionDraft | None = None,
    intro_draft: SectionDraft | None = None,
    conclusion_draft: SectionDraft | None = None,
) -> str:
    """Assembla tutte le sezioni in un unico documento Markdown.

    Ordina le sezioni secondo l'outline e aggiunge:
    - Titolo del survey
    - Abstract
    - Table of Contents (opzionale)
    - Sezioni body in ordine
    - References placeholder

    Args:
        state: Stato corrente con tutte le sezioni.
        abstract_draft: SectionDraft per l'abstract (se generato separatamente).
        intro_draft: SectionDraft per l'introduction (se generato separatamente).
        conclusion_draft: SectionDraft per la conclusion (se generato separatamente).

    Returns:
        Documento Markdown completo come stringa.
    """
    logger.info("Assemblaggio documento finale...")

    sections = get_sections_from_state(state)
    outline = get_outline_from_state(state)
    version_info = get_version_from_state(state)

    # Override con sezioni generate dall'assembler
    if abstract_draft:
        sections["abstract"] = abstract_draft
    if intro_draft:
        sections["introduction"] = intro_draft
    if conclusion_draft:
        sections["conclusion"] = conclusion_draft

    # Ordina le sezioni secondo l'outline
    sorted_outline = sorted(outline, key=lambda x: x.order)

    # Rinumera i titoli body in base all'ordine fisico (skippa abstract).
    # Evita inconsistenze tipo "8. Cross-Cutting" che appare prima di "7. Testing".
    sorted_outline = _renumber_outline_titles(sorted_outline)

    # Costruisci il documento
    parts: list[str] = []

    # --- Titolo ---
    parts.append("# A Comprehensive Survey on Security Challenges in Large Language Models\n")
    parts.append(
        f"*Version {version_info.version_number} — "
        f"{version_info.timestamp.strftime('%B %Y') if isinstance(version_info.timestamp, datetime) else 'N/A'}*\n"
    )
    parts.append("---\n")

    # --- Abstract ---
    if "abstract" in sections:
        parts.append("## Abstract\n")
        parts.append(sections["abstract"].text.strip())
        parts.append("\n")
        parts.append("---\n")

    # --- Table of Contents ---
    toc_parts = ["## Table of Contents\n"]
    section_number = 0
    for o in sorted_outline:
        if o.section_type == "abstract":
            continue
        section_number += 1
        indent = "  " * (0 if not o.parent_key else 1)
        toc_parts.append(f"{indent}{section_number}. [{o.title}](#{_slugify(o.title)})")
    toc_parts.append("")
    parts.append("\n".join(toc_parts))
    parts.append("---\n")

    # --- Body sections ---
    for o in sorted_outline:
        if o.section_type == "abstract":
            continue

        key = o.key

        if key in sections:
            draft = sections[key]
            heading_level = "##" if not o.parent_key else "###"
            parts.append(f"\n{heading_level} {o.title}\n")
            cleaned = _clean_section_text(draft.text, o.title, base_level=heading_level)
            parts.append(cleaned)
            parts.append("")
        else:
            # Sezione mancante: placeholder
            parts.append(f"\n## {o.title}\n")
            parts.append(f"*[Section not yet written — {o.description[:100]}]*\n")
            logger.warning(f"Sezione mancante nell'assemblaggio: '{key}'")

    # --- References ---
    parts.append("\n---\n")
    parts.append("## References\n")
    parts.append(
        "*See the accompanying `references.bib` file for the complete bibliography.*\n"
    )

    document = "\n".join(parts)

    # --- Risolvi i tag figura {{FIGURE:key:type}} ---
    try:
        from paperhunter.postprocessing.figure_resolver import FigureResolver
        resolver = FigureResolver()
        document, unresolved = resolver.resolve(document, output_format="markdown")
        if unresolved:
            logger.warning(
                "assemble_document: %d tag figura non risolti: %s",
                len(unresolved), unresolved,
            )
    except Exception as _exc:
        logger.warning("assemble_document: FigureResolver non disponibile: %s", _exc)

    # Statistiche
    total_words = len(document.split())
    logger.info(f"Documento assemblato: {total_words} parole totali")

    return document


def generate_bibliography(
    state: SurveyState,
    *,
    output_path: str | Path | None = None,
) -> tuple[list[str], Path | None]:
    """Genera il file .bib con solo le citazioni effettivamente usate.

    Carica il master .bib, filtra per le chiavi usate nelle sezioni,
    e salva un .bib ridotto.

    Args:
        state: Stato corrente con sezioni scritte.
        output_path: Percorso output .bib. Default: output/versions/vX.Y/references.bib.

    Returns:
        Tuple di (lista chiavi usate, Path del file .bib salvato o None).
    """
    logger.info("Generazione bibliografia...")

    used_keys = collect_all_citation_keys(state)
    logger.info(f"Citazioni uniche usate: {len(used_keys)}")

    if not used_keys:
        logger.warning("Nessuna citazione trovata nelle sezioni.")
        return [], None

    # Carica il master .bib
    master_bib_path = Path(settings.output.base_dir) / "references.bib"
    master_bib = load_master_bib(master_bib_path)

    if not master_bib:
        message = f"Master .bib vuoto o non trovato: {master_bib_path}"
        if settings.citability.final_latex_fail_on_error is True:
            raise RuntimeError(message)
        logger.warning(message)
        return list(used_keys), None

    # Filtra per chiavi usate
    filtered_entries: list[str] = []
    found_keys: set[str] = set()
    missing_keys: set[str] = set()

    for key in sorted(used_keys):
        if key in master_bib:
            filtered_entries.append(master_bib[key])
            found_keys.add(key)
        else:
            missing_keys.add(key)
            logger.warning(f"Citazione usata ma non nel master .bib: {key}")

    if missing_keys:
        message = (
            f"{len(missing_keys)} citazioni non trovate nel master .bib: "
            f"{', '.join(sorted(missing_keys))}"
        )
        if settings.citability.final_latex_fail_on_error is True:
            raise RuntimeError(message)
        print_warning(
            f"{len(missing_keys)} citazioni non trovate nel master .bib: "
            f"{', '.join(sorted(missing_keys)[:10])}{'...' if len(missing_keys) > 10 else ''}"
        )

    # Salva .bib filtrato
    if output_path is None:
        version_info = get_version_from_state(state)
        version_dir = _get_version_dir(version_info)
        output_path = version_dir / "references.bib"

    bib_path = save_master_bib(filtered_entries, output_path)
    logger.info(
        f"Bibliografia salvata: {bib_path} "
        f"({len(found_keys)} entry, {len(missing_keys)} mancanti)"
    )

    return list(sorted(found_keys)), bib_path


def create_version(
    state: SurveyState,
    *,
    sections_count: int = 0,
    total_citations: int = 0,
    verified_citations: int = 0,
) -> SurveyVersion:
    """Crea o aggiorna il SurveyVersion con le statistiche finali.

    Args:
        state: Stato corrente.
        sections_count: Numero di sezioni nel survey.
        total_citations: Numero totale di citazioni.
        verified_citations: Citazioni verificate con provenance.

    Returns:
        SurveyVersion con tutti i metadati aggiornati.
    """
    existing_version = get_version_from_state(state)
    outline = get_outline_from_state(state)
    from paperhunter.agents.state import get_l2_summaries_from_state
    l2_summaries = get_l2_summaries_from_state(state)

    categories = set()
    for o in outline:
        categories.update(o.related_categories)
    for l2 in l2_summaries:
        if l2.category:
            categories.add(l2.category)

    # Calcola paper count dai L2
    paper_count = sum(l2.paper_count for l2 in l2_summaries)

    # Determina numero versione
    version_number = _get_next_version_number()

    now = datetime.now()

    version = SurveyVersion(
        version_number=version_number,
        timestamp=now,
        paper_count=paper_count,
        categories=sorted(categories),
        l1_count=paper_count,  # Approssimazione: un L1 per paper
        l2_count=len(l2_summaries),
        sections_count=sections_count,
        total_citations=total_citations,
        verified_citations=verified_citations,
    )

    # Genera changelog se esiste versione precedente
    changelog = _generate_changelog(existing_version, version)
    if changelog:
        version.changelog = changelog

    return version


def save_version(
    document: str,
    version: SurveyVersion,
    bibliography_keys: list[str] | None = None,
    provenance: list[dict] | None = None,
) -> Path:
    """Salva il survey con versioning nel filesystem.

    Crea la struttura:
        output/versions/Survey_vX.Y_MonYYYY/
            ├── survey.md
            ├── references.bib  (se generato)
            ├── version_info.json
            ├── provenance.json (se disponibile)
            └── changelog.md    (se disponibile)

    Args:
        document: Documento Markdown completo.
        version: Metadati della versione.
        bibliography_keys: Chiavi BibTeX usate.
        provenance: Record di provenance.

    Returns:
        Path della directory della versione.
    """
    version_dir = _get_version_dir(version)
    version_dir.mkdir(parents=True, exist_ok=True)

    # 1. Salva il survey
    survey_path = version_dir / "survey.md"
    survey_path.write_text(document, encoding="utf-8")
    logger.info(f"Survey salvato: {survey_path}")

    # 2. Salva version info
    version_info_path = version_dir / "version_info.json"
    version_dict = version.model_dump()
    if "timestamp" in version_dict and isinstance(version_dict["timestamp"], datetime):
        version_dict["timestamp"] = version_dict["timestamp"].isoformat()
    version_info_path.write_text(
        json.dumps(version_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Version info salvata: {version_info_path}")

    # 3. Salva provenance
    if provenance:
        prov_path = version_dir / "provenance.json"
        prov_data = {
            "version": version.version_number,
            "generated_at": datetime.now().isoformat(),
            "total_records": len(provenance),
            "records": provenance,
        }
        prov_path.write_text(
            json.dumps(prov_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Provenance salvata: {prov_path} ({len(provenance)} record)")

    # 4. Salva changelog
    if version.changelog:
        changelog_path = version_dir / "changelog.md"
        changelog_path.write_text(
            f"# Changelog — Version {version.version_number}\n\n"
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
            f"{version.changelog}\n",
            encoding="utf-8",
        )
        logger.info(f"Changelog salvato: {changelog_path}")

    # 5. Salva lista citazioni usate
    if bibliography_keys:
        keys_path = version_dir / "citation_keys.json"
        keys_path.write_text(
            json.dumps(sorted(bibliography_keys), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return version_dir


# ============================================================================
# Main assembler function
# ============================================================================

def assemble_paper(
    state: SurveyState,
    *,
    force_model: str | None = None,
    generate_abstract_flag: bool = True,
    generate_intro_flag: bool = True,
    generate_conclusion_flag: bool = True,
    save_output: bool = True,
) -> dict[str, Any]:
    """Assembla il paper completo: abstract, intro, conclusion, bibliografia, versioning.

    Funzione orchestratrice che chiama tutti gli step dell'assemblaggio.

    Args:
        state: Stato corrente con sezioni scritte e verificate.
        force_model: Override modello LLM per generazione abstract/intro/conclusion.
        generate_abstract_flag: Se generare l'abstract.
        generate_intro_flag: Se generare/raffinare l'introduction.
        generate_conclusion_flag: Se generare la conclusion.
        save_output: Se salvare l'output su disco con versioning.

    Returns:
        Dict parziale per aggiornamento stato LangGraph con:
        - sections: sezioni aggiornate (abstract, intro, conclusion)
        - bibliography: chiavi .bib usate
        - final_paper: documento Markdown assemblato
        - version_info: metadati versione
        - current_phase: ASSEMBLING
        - provenance: record di provenance (se presenti nello stato)
        - errors: eventuali errori
        - metadata: metadati aggiuntivi
    """
    logger.info("=" * 60)
    logger.info("ASSEMBLER AGENT — Assemblaggio finale del survey")
    logger.info("=" * 60)

    t0 = time.time()
    new_sections: dict[str, dict] = {}
    errors: list[str] = []

    # --- Step 1: Genera Abstract ---
    abstract_draft = None
    if generate_abstract_flag:
        try:
            abstract_draft = generate_abstract(state, force_model=force_model)
            new_sections["abstract"] = abstract_draft.to_dict()
            print_success(f"Abstract generato: {abstract_draft.word_count} parole")
        except Exception as e:
            err = f"Errore generazione abstract: {e}"
            logger.error(err, exc_info=True)
            errors.append(err)

    if generate_abstract_flag and abstract_draft is None:
        # Do not write/version a paper with a synthetic placeholder or no
        # abstract. A later resume can retry this bounded editorial call.
        return {
            "current_phase": PipelinePhase.FAILED.value,
            "errors": errors or ["Abstract generation failed."],
            "final_paper": "",
            "sections": {},
            "metadata": {"assembler_time": round(time.time() - t0, 1)},
        }

    # --- Step 2: Genera/Raffina Introduction ---
    intro_draft = None
    if generate_intro_flag:
        try:
            intro_draft = generate_introduction(state, force_model=force_model)
            new_sections["introduction"] = intro_draft.to_dict()
            print_success(f"Introduction generata: {intro_draft.word_count} parole")
        except Exception as e:
            err = f"Errore generazione introduction: {e}"
            logger.error(err, exc_info=True)
            errors.append(err)

    # --- Step 3: Genera Conclusion ---
    conclusion_draft = None
    if generate_conclusion_flag:
        try:
            conclusion_draft = generate_conclusion(state, force_model=force_model)
            new_sections["conclusion"] = conclusion_draft.to_dict()
            print_success(f"Conclusion generata: {conclusion_draft.word_count} parole")
        except Exception as e:
            err = f"Errore generazione conclusion: {e}"
            logger.error(err, exc_info=True)
            errors.append(err)

    # --- Aggiorna stato con le nuove sezioni ---
    merged_state: SurveyState = {  # type: ignore[typeddict-item]
        **state,
        "sections": {**state.get("sections", {}), **new_sections},
    }

    # --- Step 4: Marca tutte le sezioni come FINALIZED ---
    finalized_sections = _finalize_all_sections(merged_state)
    new_sections.update(finalized_sections)
    merged_state["sections"] = {**merged_state.get("sections", {}), **finalized_sections}

    # --- Step 5: Assembla documento ---
    document = assemble_document(
        merged_state,
        abstract_draft=abstract_draft,
        intro_draft=intro_draft,
        conclusion_draft=conclusion_draft,
    )

    # --- Step 6: Genera bibliografia ---
    bib_keys, bib_path = generate_bibliography(merged_state)

    # --- Step 7: Versioning ---
    all_sections = get_sections_from_state(merged_state)
    all_citations = collect_all_citation_keys(merged_state)
    verifications = get_verifications_from_state(merged_state)
    verified_count = sum(
        1 for vr in verifications.values() if vr.passed
    )

    version = create_version(
        merged_state,
        sections_count=len(all_sections),
        total_citations=len(all_citations),
        verified_citations=verified_count,
    )

    # Provenance dal stato
    provenance = state.get("provenance", [])

    # --- Step 8: Salva su disco ---
    version_dir = None
    if save_output:
        try:
            version_dir = save_version(
                document,
                version,
                bibliography_keys=bib_keys,
                provenance=provenance if provenance else None,
            )
            print_success(f"Survey salvato in: {version_dir}")
        except Exception as e:
            err = f"Errore salvataggio su disco: {e}"
            logger.error(err, exc_info=True)
            errors.append(err)

    elapsed = time.time() - t0

    # --- Report finale ---
    _print_assembly_report(
        all_sections, version, bib_keys, provenance, elapsed, version_dir
    )

    # Serializza version_info
    version_dict = version.model_dump()
    if "timestamp" in version_dict and isinstance(version_dict["timestamp"], datetime):
        version_dict["timestamp"] = version_dict["timestamp"].isoformat()

    return {
        "sections": new_sections,
        "bibliography": bib_keys,
        "final_paper": document,
        "version_info": version_dict,
        "current_phase": PipelinePhase.ASSEMBLING.value,
        "provenance": [],  # Non aggiunge nuovi record, li preserva
        "errors": errors if errors else [],
        "metadata": {
            **state.get("metadata", {}),
            "assembler_time": round(elapsed, 1),
            "assembler_model": force_model or settings.agents.assembler_model,
            "version_dir": str(version_dir) if version_dir else None,
            "bib_path": str(bib_path) if bib_path else None,
            "total_words": len(document.split()),
        },
    }


# ============================================================================
# LangGraph node function
# ============================================================================


def _validate_assembly_readiness(state: SurveyState) -> tuple[bool, list[str]]:
    """Fail closed unless every planned body section passed the Critic."""
    outline = get_outline_from_state(state)
    sections = get_sections_from_state(state)
    verifications = get_verifications_from_state(state)
    errors: list[str] = []

    required = [item for item in outline if item.section_type != "abstract"]
    missing = [item.key for item in required if item.key not in sections]
    if missing:
        errors.append(f"Missing planned sections: {', '.join(missing)}")

    unverified = [
        item.key
        for item in required
        if item.key in sections
        and sections[item.key].status not in (SectionStatus.VERIFIED, SectionStatus.FINALIZED)
    ]
    if unverified:
        errors.append(f"Sections not verified: {', '.join(unverified)}")

    without_pass = [
        item.key
        for item in required
        if item.key in sections
        and (
            item.key not in verifications
            or not verifications[item.key].passed
        )
    ]
    if without_pass:
        errors.append(
            "Sections without a passing citation verification result: "
            + ", ".join(without_pass)
        )
    return not errors, errors


def assemble(state: SurveyState) -> dict[str, Any]:
    """Nodo LangGraph: Assembler Agent.

    Assembla il paper finale: genera abstract/intro/conclusion,
    compone il documento, crea la bibliografia, salva con versioning.

    Args:
        state: Stato corrente della pipeline LangGraph.

    Returns:
        Dict parziale con ``sections``, ``bibliography``, ``final_paper``,
        ``version_info``, ``current_phase``, ``errors``, ``metadata``.
    """
    logger.info("=" * 60)
    logger.info("ASSEMBLER AGENT — Nodo LangGraph")
    logger.info("=" * 60)

    # Verifica prerequisiti
    sections = get_sections_from_state(state)
    if not sections:
        err = "Nessuna sezione disponibile per l'assemblaggio."
        logger.error(err)
        print_error(err)
        return {
            "current_phase": PipelinePhase.FAILED.value,
            "errors": [err],
        }

    ready, readiness_errors = _validate_assembly_readiness(state)
    if not ready:
        err = "Assemblaggio bloccato: " + " | ".join(readiness_errors)
        logger.error(err)
        print_error(err)
        return {
            "current_phase": PipelinePhase.FAILED.value,
            "errors": [err],
        }

    # Introduction and conclusion are already written and citation-verified by
    # Writer/Critic. Regenerating them here would introduce unverified claims.
    result = assemble_paper(
        state,
        generate_intro_flag=False,
        generate_conclusion_flag=False,
    )

    # Report sommario
    total_words = result.get("metadata", {}).get("total_words", 0)
    bib_count = len(result.get("bibliography", []))
    logger.info(
        f"Assembler completato: {total_words} parole, "
        f"{bib_count} citazioni in bibliografia"
    )

    if result.get("final_paper"):
        print_success(
            f"Survey assemblato con successo! "
            f"{total_words} parole, {bib_count} riferimenti."
        )
    else:
        print_error("Assemblaggio fallito — nessun documento generato.")

    return result


# ============================================================================
# Internal helpers
# ============================================================================

def _format_sections_overview(
    sections: dict[str, SectionDraft],
    outline: list[SectionOutline],
) -> str:
    """Formatta un overview compatto delle sezioni per il prompt abstract."""
    sorted_outline = sorted(outline, key=lambda x: x.order)
    parts: list[str] = []

    for o in sorted_outline:
        if o.section_type == "abstract":
            continue
        key = o.key
        if key in sections:
            draft = sections[key]
            # Ground the abstract in more than one opening sentence per
            # already-verified section.
            preview = draft.text[:600].replace("\n", " ").strip()
            n_citations = len(draft.citations)
            parts.append(
                f"**{o.title}** ({draft.word_count} words, {n_citations} citations): "
                f"{preview}..."
            )
        else:
            parts.append(f"**{o.title}**: [not yet written]")

    return "\n".join(parts)


def _format_sections_summary(
    sections: dict[str, SectionDraft],
    outline: list[SectionOutline],
) -> str:
    """Formatta un sommario delle sezioni per il prompt conclusion."""
    sorted_outline = sorted(outline, key=lambda x: x.order)
    parts: list[str] = []

    for o in sorted_outline:
        if o.section_type in ("abstract", "conclusion"):
            continue
        key = o.key
        if key in sections:
            draft = sections[key]
            # Primi 400 caratteri per dare contesto sufficiente
            preview = draft.text[:400].replace("\n", " ").strip()
            parts.append(f"### {o.title}\n{preview}...\n")

    return "\n".join(parts)


def _extract_key_findings(
    sections: dict[str, SectionDraft],
) -> str:
    """Estrae i finding principali dalle sezioni body per la conclusion."""
    findings: list[str] = []

    for key, draft in sections.items():
        if draft.section_key in ("abstract", "introduction", "conclusion"):
            continue
        # Cerca patterns di risultati nel testo
        # Prendi le prime 2-3 frasi significative
        text = draft.text.strip()
        if len(text) < 50:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Prendi le prime 3 frasi come rappresentazione
        key_sentences = sentences[:3]
        findings.append(
            f"**{draft.title}**: {' '.join(key_sentences)[:300]}"
        )

    return "\n".join(findings) if findings else "No key findings extracted."


def _extract_citations_from_text(text: str) -> list:
    """Estrae citazioni [bibtex_key] dal testo generato.

    Args:
        text: Testo con citazioni nel formato [bibtex_key].

    Returns:
        Lista di CitationRecord.
    """
    from paperhunter.agents.state import CitationRecord

    matches: list[str] = []
    for group in re.findall(r"\[([^\]]+)\]", text):
        for raw_key in re.split(r"[,;]", group):
            key = raw_key.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", key):
                matches.append(key)

    seen: set[str] = set()
    records: list[CitationRecord] = []
    for key in matches:
        if key not in seen:
            seen.add(key)
            records.append(CitationRecord(
                bibtex_key=key,
                claim="",
                chunk_id=None,
                retrieval_score=0.0,
            ))
    return records


def _finalize_all_sections(state: SurveyState) -> dict[str, dict]:
    """Marca come FINALIZED soltanto sezioni già VERIFIED.

    Returns:
        Dict di sezioni aggiornate (solo quelle modificate).
    """
    sections = get_sections_from_state(state)
    updated: dict[str, dict] = {}

    for key, draft in sections.items():
        if draft.status == SectionStatus.VERIFIED:
            draft.status = SectionStatus.FINALIZED
            updated[key] = draft.to_dict()

    if updated:
        logger.info(f"Marcate {len(updated)} sezioni come FINALIZED")

    return updated


def _get_version_dir(version: SurveyVersion) -> Path:
    """Calcola la directory per una versione specifica.

    Formato: output/versions/Survey_vX.Y_MonYYYY/

    Args:
        version: Metadati della versione.

    Returns:
        Path della directory.
    """
    versions_dir = Path(settings.output.versions_dir)
    ts = version.timestamp
    if isinstance(ts, datetime):
        month_year = ts.strftime("%b%Y")
    else:
        month_year = datetime.now().strftime("%b%Y")

    dir_name = f"Survey_v{version.version_number}_{month_year}"
    return versions_dir / dir_name


def _get_next_version_number() -> str:
    """Determina il numero di versione successivo.

    Scansiona la directory versions alla ricerca di versioni precedenti
    e incrementa il minor version.

    Returns:
        Stringa versione (es. "1.0", "1.1", "2.0").
    """
    versions_dir = Path(settings.output.versions_dir)
    if not versions_dir.exists():
        return "1.0"

    existing_versions: list[tuple[int, int]] = []
    pattern = re.compile(r'Survey_v(\d+)\.(\d+)_')

    for item in versions_dir.iterdir():
        if item.is_dir():
            match = pattern.search(item.name)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2))
                existing_versions.append((major, minor))

    if not existing_versions:
        return "1.0"

    # Incrementa minor version
    latest = max(existing_versions)
    return f"{latest[0]}.{latest[1] + 1}"


def _generate_changelog(
    old_version: SurveyVersion,
    new_version: SurveyVersion,
) -> str:
    """Genera changelog tra due versioni.

    Se la vecchia versione ha paper_count=0, è la prima versione
    e non genera changelog.

    Args:
        old_version: Versione precedente.
        new_version: Nuova versione.

    Returns:
        Stringa Markdown del changelog, o stringa vuota se prima versione.
    """
    # Se è la prima versione, niente changelog
    if old_version.paper_count == 0 and not old_version.categories:
        logger.info("Prima versione del survey — nessun changelog da generare.")
        return ""

    # Prova a generare changelog con LLM
    try:
        old_ts = old_version.timestamp
        old_date = old_ts.strftime("%Y-%m-%d") if isinstance(old_ts, datetime) else str(old_ts)

        new_ts = new_version.timestamp
        new_date = new_ts.strftime("%Y-%m-%d") if isinstance(new_ts, datetime) else str(new_ts)

        prompt = _CHANGELOG_PROMPT_TEMPLATE.format(
            old_version=old_version.version_number,
            old_date=old_date,
            old_paper_count=old_version.paper_count,
            old_categories=", ".join(old_version.categories) if old_version.categories else "N/A",
            old_sections=old_version.sections_count,
            new_version=new_version.version_number,
            new_date=new_date,
            new_paper_count=new_version.paper_count,
            new_categories=", ".join(new_version.categories) if new_version.categories else "N/A",
            new_sections=new_version.sections_count,
        )

        llm = get_llm(
            "flash", max_tokens=2048, timeout_seconds=60, thinking_enabled=False,
        )
        changelog = llm_call_with_retry(
            llm, prompt,
            max_retries=2,
            base_delay=5.0,
            max_delay=30.0,
        )
        return changelog.strip()
    except Exception as e:
        logger.warning(f"Impossibile generare changelog con LLM: {e}")
        # Fallback: changelog manuale
        return _manual_changelog(old_version, new_version)


def _manual_changelog(
    old_version: SurveyVersion,
    new_version: SurveyVersion,
) -> str:
    """Genera un changelog minimal senza LLM."""
    parts: list[str] = []

    paper_diff = new_version.paper_count - old_version.paper_count
    if paper_diff > 0:
        parts.append(f"- Added {paper_diff} new papers (total: {new_version.paper_count})")
    elif paper_diff < 0:
        parts.append(f"- Removed {abs(paper_diff)} papers (total: {new_version.paper_count})")

    new_cats = set(new_version.categories) - set(old_version.categories)
    if new_cats:
        parts.append(f"- New categories: {', '.join(sorted(new_cats))}")

    section_diff = new_version.sections_count - old_version.sections_count
    if section_diff > 0:
        parts.append(f"- Added {section_diff} new sections")
    elif section_diff < 0:
        parts.append(f"- Reduced by {abs(section_diff)} sections")

    if new_version.total_citations != old_version.total_citations:
        parts.append(
            f"- Citations updated: {old_version.total_citations} → {new_version.total_citations}"
        )

    if not parts:
        parts.append("- Minor updates and corrections")

    return "\n".join(parts)


def _slugify(text: str) -> str:
    """Converte un titolo in slug per anchor link Markdown."""
    slug = text.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = slug.strip('-')
    return slug


def _renumber_outline_titles(outline: list) -> list:
    """Rinumera i titoli body in base all'ordine fisico.

    Esempio: se l'outline ha "8. Cross-Cutting" prima di "7. Testing",
    li rinomina rispettivamente in "7. Cross-Cutting" e "8. Testing".
    Lascia inalterati: abstract, introduction (sempre n.1), conclusion.
    Le sotto-sezioni (parent_key set) ereditano il prefisso del padre.
    """
    NUMBER_RE = re.compile(r"^(\d+)(\.\d+)?\.?\s+(.+)$")

    # Mappa parent.key → nuovo numero "N"
    parent_new_num: dict[str, int] = {}
    n = 0
    skip_keys = {"abstract"}

    new_outline = []
    for o in outline:
        if o.section_type in ("abstract",) or o.key in skip_keys:
            new_outline.append(o)
            continue

        title = o.title
        # Solo titoli che iniziano con un numero esplicito
        m = NUMBER_RE.match(title)

        if not o.parent_key:
            n += 1
            if m:
                rest = m.group(3)
                new_title = f"{n}. {rest}"
            else:
                new_title = title
            parent_new_num[o.key] = n
            try:
                o.title = new_title
            except Exception:
                pass
            new_outline.append(o)
        else:
            # Sub-section: rinumera con prefisso padre.M
            parent_n = parent_new_num.get(o.parent_key)
            if parent_n is not None and m and m.group(2):
                # m.group(2) è ".X"
                rest = m.group(3)
                new_title = f"{parent_n}{m.group(2)} {rest}"
                try:
                    o.title = new_title
                except Exception:
                    pass
            new_outline.append(o)

    return new_outline


def _clean_section_text(text: str, expected_title: str, *, base_level: str = "##") -> str:
    """Pulisce il testo di una sezione body prodotta dal Writer.

    Rimuove il titolo iniziale duplicato (writer ripete il proprio H1/H2/bold)
    e normalizza i livelli heading interni:
    - ``## sub`` o ``# sub`` (interni) → demoted di un livello
    - ``**N.X Title**`` su riga propria → ``### N.X Title``
    - ``**Title**`` su riga propria che assomiglia a un heading → ``### Title``

    Args:
        text: Testo Markdown grezzo del draft.
        expected_title: Titolo che l'assembler ha già scritto come heading.
        base_level: ``"##"`` se la sezione è top-level, ``"###"`` se sub-section.
    """
    if not text:
        return ""

    text = text.strip()

    # --- 1. Rimuovi heading duplicato all'inizio ---
    # Estrai una "firma" del titolo atteso (lowercased, senza numerazione)
    expected_norm = re.sub(r"^\d+(\.\d+)*\.?\s*", "", expected_title).strip().lower()

    lines = text.split("\n")
    # Skippa fino a 3 righe vuote/heading duplicate
    drop_until = 0
    for idx in range(min(6, len(lines))):
        line = lines[idx].strip()
        if not line:
            drop_until = idx + 1
            continue
        m = re.match(r"^(#{1,3}\s+|\*\*)(.+?)(\*\*)?\s*$", line)
        if m:
            cand = re.sub(r"^\d+(\.\d+)*\.?\s*", "", m.group(2)).strip().lower().rstrip("*")
            if cand and (cand == expected_norm or
                         (len(expected_norm) >= 8 and cand.startswith(expected_norm[:20])) or
                         (len(cand) >= 8 and expected_norm.startswith(cand[:20]))):
                drop_until = idx + 1
                continue
        break
    if drop_until:
        lines = lines[drop_until:]

    # --- 2. Demote heading interni ---
    # Se base è "##", i ## interni diventano ###, gli # diventano ###
    # Se base è "###", i ### interni diventano ####, gli ## diventano ####
    target_sub = "###" if base_level == "##" else "####"
    target_subsub = "####" if base_level == "##" else "#####"

    out_lines: list[str] = []
    for line in lines:
        m = re.match(r"^(#{1,3})\s+(.+)$", line)
        if m:
            n_hash = len(m.group(1))
            title = m.group(2).strip()
            if n_hash == 1 or n_hash == 2:
                line = f"{target_sub} {title}"
            elif n_hash == 3:
                line = f"{target_subsub} {title}"
        else:
            # **N.X Title** su riga propria → heading
            mb = re.match(r"^\*\*(\d+(?:\.\d+)+\s+[^*]+?)\*\*\s*$", line)
            if mb:
                line = f"{target_sub} {mb.group(1).strip()}"
            else:
                # **Title** su riga propria (Title plausibile, niente punto finale)
                mb2 = re.match(r"^\*\*([A-Z][A-Za-z0-9][^*\n]{2,80})\*\*\s*$", line)
                if mb2 and not mb2.group(1).rstrip().endswith((".", ":", ",")):
                    line = f"{target_sub} {mb2.group(1).strip()}"
        out_lines.append(line)

    return "\n".join(out_lines).strip()


def _print_assembly_report(
    sections: dict[str, SectionDraft],
    version: SurveyVersion,
    bib_keys: list[str],
    provenance: list[dict],
    elapsed: float,
    version_dir: Path | None,
) -> None:
    """Stampa il report finale dell'assemblaggio con tabelle Rich."""

    # Sommario generale
    summary_data = {
        "Versione": version.version_number,
        "Timestamp": version.timestamp.strftime("%Y-%m-%d %H:%M") if isinstance(version.timestamp, datetime) else str(version.timestamp),
        "Sezioni": str(len(sections)),
        "Parole totali": str(sum(d.word_count for d in sections.values())),
        "Citazioni uniche": str(len(bib_keys)),
        "Provenance records": str(len(provenance)) if provenance else "0",
        "Tempo": f"{elapsed:.1f}s",
        "Output": str(version_dir) if version_dir else "Non salvato",
    }
    print_table("Assembler — Report Finale", summary_data)

    # Dettaglio per sezione
    section_data = [
        {
            "Sezione": d.title[:30],
            "Parole": str(d.word_count),
            "Citazioni": str(len(d.citations)),
            "Status": d.status.value if isinstance(d.status, SectionStatus) else d.status,
        }
        for d in sorted(sections.values(), key=lambda x: x.section_key)
    ]
    if section_data:
        print_table("Sezioni assemblate", section_data)

    if version.changelog:
        print_panel("Changelog", version.changelog)


# ============================================================================
# Convenience functions
# ============================================================================

def get_assembly_stats(state: SurveyState) -> dict[str, Any]:
    """Calcola statistiche sull'assemblaggio del survey.

    Args:
        state: Stato corrente.

    Returns:
        Dict con statistiche aggregate.
    """
    sections = get_sections_from_state(state)
    verifications = get_verifications_from_state(state)
    all_keys = collect_all_citation_keys(state)
    version = get_version_from_state(state)

    total_words = sum(d.word_count for d in sections.values())
    finalized_count = sum(
        1 for d in sections.values()
        if d.status == SectionStatus.FINALIZED
    )
    verified_count = sum(
        1 for vr in verifications.values()
        if vr.passed
    )

    return {
        "total_sections": len(sections),
        "finalized_sections": finalized_count,
        "total_words": total_words,
        "unique_citations": len(all_keys),
        "verified_sections": verified_count,
        "version": version.version_number,
        "paper_count": version.paper_count,
        "categories": version.categories,
        "has_final_paper": bool(state.get("final_paper")),
        "has_provenance": bool(state.get("provenance")),
    }
