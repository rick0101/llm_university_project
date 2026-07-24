"""
Planner Agent — Genera l'outline completo del survey paper.

Agent 1 nella pipeline agentica LangGraph. Riceve i summary L2
(uno per categoria) e produce un outline strutturato con sezioni
ordinate logicamente e cronologicamente, pronto per il Writer.

Supporta:
- Generazione outline completa via Gemini Pro
- Override outline da file JSON o da utente
- Caricamento automatico L2 dal QdrantStore se non presenti nello stato
- Sezione "Testing Methodologies" dedicata (requisito tesi)

Uso standalone::

    from paperhunter.agents.planner_agent import plan, generate_outline
    from paperhunter.agents.state import create_initial_state

    state = create_initial_state(l2_summaries=[...])
    new_state = plan(state)
    outline = new_state["outline"]

Uso come nodo LangGraph::

    from langgraph.graph import StateGraph
    from paperhunter.agents.state import SurveyState
    from paperhunter.agents.planner_agent import plan

    graph = StateGraph(SurveyState)
    graph.add_node("plan", plan)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from paperhunter.agents.state import (
    PipelinePhase,
    SectionOutline,
    SurveyState,
    get_l2_summaries_from_state,
    get_outline_from_state,
)
from paperhunter.config import settings
from paperhunter.evolution.lesson_store import LessonStore
from paperhunter.evolution.prompt_overlay import generate_full_overlay
from paperhunter.models import L2Summary
from paperhunter.utils.llm_factory import (
    get_llm,
    get_pro_tracker,
    llm_call_with_retry,
)
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_success,
    print_table,
    print_warning,
)
from paperhunter.utils.prompt_loader import get_prompt

logger = get_logger(__name__)


# ============================================================================
# Prompt template
# ============================================================================

_PLANNER_PROMPT_TEMPLATE = get_prompt("PLANNER_PROMPT")


# ============================================================================
# Helpers — L2 text formatting
# ============================================================================

def _format_l2_for_prompt(l2_summaries: list[L2Summary]) -> str:
    """Formatta tutti i L2 summary per il prompt del Planner.

    Crea un testo compatto con le informazioni chiave di ogni L2.
    """
    parts: list[str] = []

    for i, l2 in enumerate(l2_summaries, start=1):
        section = [
            f"### Category {i}: {l2.category} ({l2.paper_count} papers)",
        ]

        if l2.temporal_trends:
            section.append(f"\n**Temporal Trends:**\n{l2.temporal_trends[:2000]}")
        if l2.taxonomy:
            section.append(f"\n**Taxonomy:**\n{l2.taxonomy[:1500]}")
        if l2.testing_overview:
            section.append(f"\n**Testing Overview:**\n{l2.testing_overview[:1500]}")
        if l2.open_gaps:
            section.append(f"\n**Open Gaps:**\n{l2.open_gaps[:1000]}")
        if l2.synthesis:
            section.append(f"\n**Synthesis:**\n{l2.synthesis[:1500]}")

        parts.append("\n".join(section))

    return "\n\n---\n\n".join(parts)


def _compute_target_words(n_categories: int) -> int:
    """Calcola il target di parole totale in base alle categorie.

    Circa 2000 parole per categoria di corpo + 2000 per sezioni fisse.
    """
    body_words = n_categories * 2000
    fixed_words = 2000  # intro + background + testing + defenses + discussion + conclusion
    return body_words + fixed_words


# ============================================================================
# JSON parsing
# ============================================================================

def _parse_outline_json(raw_response: str) -> list[dict] | None:
    """Parse la risposta LLM come JSON array di sezioni.

    Applica diversi livelli di fallback per gestire
    risposte non perfettamente formattate.
    """
    if not raw_response or not raw_response.strip():
        return None

    text = raw_response.strip()

    # Rimuovi code fences markdown
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Tentativo 1: parse JSON diretto
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "sections" in data:
            return data["sections"]
        elif isinstance(data, dict) and "outline" in data:
            return data["outline"]
        return None
    except json.JSONDecodeError:
        pass

    # Tentativo 2: estrai il primo array JSON
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Tentativo 3: fix trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    logger.error(f"Impossibile parsare l'outline JSON. Primi 500 chars: {text[:500]}")
    return None


# ============================================================================
# Outline validation & normalization
# ============================================================================

_REQUIRED_SECTION_TYPES = {"abstract", "introduction", "body", "testing", "discussion", "conclusion"}

_MANDATORY_SECTION_KEYS = {
    "abstract": ("abstract", "Abstract"),
    "introduction": ("introduction", "Introduction"),
    "testing": ("testing_methodologies", "Testing Methodologies and Evaluation"),
    "conclusion": ("conclusion", "Conclusion"),
}


def _validate_and_normalize_outline(
    raw_sections: list[dict],
    categories: list[str],
) -> list[SectionOutline]:
    """Valida e normalizza le sezioni dell'outline.

    - Verifica che ogni sezione abbia i campi obbligatori
    - Assegna ordine sequenziale se mancante
    - Aggiunge sezioni mancanti obbligatorie (abstract, intro, testing, conclusion)
    - Normalizza section_type

    Returns:
        Lista ordinata di SectionOutline.
    """
    outlines: list[SectionOutline] = []
    seen_keys: set[str] = set()
    seen_types: set[str] = set()

    for i, raw in enumerate(raw_sections):
        if not isinstance(raw, dict):
            logger.warning(f"Sezione {i} non è un dict, saltata: {type(raw)}")
            continue

        # Campi obbligatori
        key = raw.get("key", "").strip().lower().replace(" ", "_").replace("-", "_")
        if not key:
            key = f"section_{i}"
        title = raw.get("title", "").strip()
        if not title:
            logger.warning(f"Sezione con key={key!r} senza titolo, assegno key come titolo")
            title = key.replace("_", " ").title()

        # Deduplicazione key
        if key in seen_keys:
            key = f"{key}_{i}"
        seen_keys.add(key)

        # section_type normalization
        section_type = raw.get("section_type", "body").strip().lower()
        if section_type not in _REQUIRED_SECTION_TYPES:
            section_type = "body"
        seen_types.add(section_type)

        # Costruisci SectionOutline
        try:
            outline = SectionOutline(
                key=key,
                title=title,
                description=raw.get("description", ""),
                target_length_words=_clamp(
                    raw.get("target_length_words", 1500), 100, 5000
                ),
                key_topics=raw.get("key_topics", []) if isinstance(raw.get("key_topics"), list) else [],
                related_categories=raw.get("related_categories", []) if isinstance(raw.get("related_categories"), list) else [],
                order=raw.get("order", i),
                section_type=section_type,
                parent_key=raw.get("parent_key"),
                subsections=raw.get("subsections", []) if isinstance(raw.get("subsections"), list) else [],
            )
            outlines.append(outline)
        except Exception as e:
            logger.warning(f"Errore nella costruzione SectionOutline per key={key!r}: {e}")
            continue

    # --- Aggiungi sezioni mancanti obbligatorie ---
    for stype, (default_key, default_title) in _MANDATORY_SECTION_KEYS.items():
        if stype not in seen_types:
            logger.info(f"Aggiunta sezione mancante: {default_title} (type={stype})")
            outlines.append(SectionOutline(
                key=default_key,
                title=default_title,
                description=f"Auto-generated {stype} section.",
                target_length_words=500 if stype in ("abstract", "conclusion") else 1500,
                key_topics=[],
                related_categories=categories,
                order=9999,  # Placeholder — will be reordered below
                section_type=stype,
            ))

    # --- Riordina in modo logico ---
    type_order = {
        "abstract": 0,
        "introduction": 1,
        "body": 2,
        "testing": 3,
        "discussion": 4,
        "conclusion": 5,
    }

    # Prima ordina per section_type, poi per ordine esplicito
    outlines.sort(key=lambda o: (type_order.get(o.section_type, 2), o.order))

    # Riassegna ordine sequenziale
    for i, outline in enumerate(outlines):
        outline.order = i

    return outlines


def _clamp(value: int | float, min_val: int, max_val: int) -> int:
    """Clampa un valore tra min e max."""
    try:
        return max(min_val, min(int(value), max_val))
    except (TypeError, ValueError):
        return min_val


# ============================================================================
# Outline from user / JSON file
# ============================================================================

def load_outline_from_json(json_path: str | Path) -> list[SectionOutline]:
    """Carica un outline da file JSON.

    Il file può contenere:
    - Un array di sezioni nel formato SectionOutline
    - Un dict con chiave "sections" contenente l'array
    - Il formato semplice di sample_outline.json:
      ``{"title": "...", "sections": {"Section Title": "Description", ...}}``

    Args:
        json_path: Percorso al file JSON.

    Returns:
        Lista ordinata di SectionOutline.

    Raises:
        FileNotFoundError: Se il file non esiste.
        ValueError: Se il formato non è riconosciuto.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"File outline non trovato: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Formato 1: array diretto di SectionOutline
    if isinstance(data, list):
        return _validate_and_normalize_outline(data, [])

    if not isinstance(data, dict):
        raise ValueError(f"Formato outline non riconosciuto: {type(data)}")

    # Formato 2: dict con "sections" come array
    if "sections" in data and isinstance(data["sections"], list):
        return _validate_and_normalize_outline(data["sections"], [])

    # Formato 3: sample_outline.json format
    # {"title": "...", "sections": {"Title": "Description", ...}}
    if "sections" in data and isinstance(data["sections"], dict):
        sections: list[dict] = []
        for i, (title, description) in enumerate(data["sections"].items()):
            key = _title_to_key(title)

            # Inferisci section_type dal titolo
            section_type = _infer_section_type(title)

            sections.append({
                "key": key,
                "title": title,
                "description": description,
                "target_length_words": 500 if section_type in ("abstract", "conclusion") else 1500,
                "key_topics": [],
                "related_categories": [],
                "order": i,
                "section_type": section_type,
            })
        return _validate_and_normalize_outline(sections, [])

    # Formato 4: dict con "outline" come array
    if "outline" in data and isinstance(data["outline"], list):
        return _validate_and_normalize_outline(data["outline"], [])

    raise ValueError(
        f"Formato outline non riconosciuto. "
        f"Atteso: array, {{sections: [...]}} o {{sections: {{...}}}}"
    )


def _title_to_key(title: str) -> str:
    """Converte un titolo in una chiave snake_case."""
    key = title.lower().strip()
    key = re.sub(r"[^a-z0-9\s]", "", key)
    key = re.sub(r"\s+", "_", key)
    return key[:60]  # Limita lunghezza


def _infer_section_type(title: str) -> str:
    """Inferisce il section_type dal titolo."""
    t = title.lower()
    if "abstract" in t:
        return "abstract"
    if "introduction" in t:
        return "introduction"
    if "conclusion" in t:
        return "conclusion"
    if "discussion" in t or "open challenge" in t or "future" in t:
        return "discussion"
    if "testing" in t or "evaluation" in t or "benchmark" in t:
        return "testing"
    return "body"


# ============================================================================
# L2 loading from store
# ============================================================================

def _load_l2_from_store() -> list[L2Summary]:
    """Carica tutti i L2 summary non-invalidati dal QdrantStore.

    Returns:
        Lista di L2Summary.
    """
    try:
        from paperhunter.indexing.qdrant_store import QdrantStore

        store = QdrantStore()
        collection = settings.qdrant.collection_l2_summaries

        # Scroll all L2 summaries
        all_points = []
        offset = None
        try:
            while True:
                points, next_offset = store.client.scroll(
                    collection_name=collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                all_points.extend(points)
                if next_offset is None or not points:
                    break
                offset = next_offset
        finally:
            store.close()

        l2s: list[L2Summary] = []
        for point in all_points:
            payload = point.payload or {}
            meta = payload.get("metadata", {})
            text = payload.get("text", "")

            # Ricostruisci L2Summary dai campi del payload
            category = meta.get("category", payload.get("category", "Unknown"))

            # Se il summary è invalidato, skip
            if meta.get("invalidated", False):
                continue

            l2 = L2Summary(
                category=category,
                paper_ids=meta.get("paper_ids", []),
                paper_count=meta.get("paper_count", 0),
                version=meta.get("version", 1),
                synthesis=text,  # Il testo primario è la synthesis
            )

            # Proviamo a caricare i metadati estesi da disco
            try:
                from paperhunter.summarization.level2_summarizer import load_all_l2_metadata
                all_meta = load_all_l2_metadata()
                if category in all_meta:
                    disk_meta = all_meta[category]
                    for field in ("temporal_trends", "taxonomy", "gaps_filled",
                                  "open_gaps", "methodological_convergences",
                                  "testing_overview", "synthesis"):
                        if field in disk_meta and disk_meta[field]:
                            setattr(l2, field, disk_meta[field])
                    if "paper_ids" in disk_meta:
                        l2.paper_ids = disk_meta["paper_ids"]
                    if "paper_count" in disk_meta:
                        l2.paper_count = disk_meta["paper_count"]
            except Exception as e:
                logger.debug(f"Impossibile caricare metadati L2 estesi per {category}: {e}")

            l2s.append(l2)

        logger.info(f"Caricati {len(l2s)} L2 summary dal QdrantStore")
        return l2s

    except Exception as e:
        logger.error(f"Errore caricamento L2 dal QdrantStore: {e}")
        return []


# ============================================================================
# Core function — generate outline via LLM
# ============================================================================

def generate_outline(
    l2_summaries: list[L2Summary],
    *,
    force_model: str | None = None,
    min_sections: int = 8,
    target_sections: int = 12,
    total_words: int | None = None,
) -> list[SectionOutline]:
    """Genera l'outline del survey tramite Gemini Pro.

    Args:
        l2_summaries: Lista dei L2 summary (uno per categoria).
        force_model: Override del modello (``'flash'`` o ``'pro'``).
            Default: da ``config.yaml`` → ``agents.planner_model``.
        min_sections: Numero minimo di sezioni nell'outline.
        target_sections: Numero target di sezioni.
        total_words: Target parole totali. Se None, calcolato
            automaticamente dalle categorie.

    Returns:
        Lista ordinata di SectionOutline.

    Raises:
        ValueError: Se non ci sono L2 summaries.
        RuntimeError: Se la generazione fallisce dopo tutti i retry.
    """
    if not l2_summaries:
        raise ValueError(
            "Nessun L2 summary fornito. Impossibile generare l'outline. "
            "Esegui prima la generazione L2."
        )

    n_categories = len(l2_summaries)
    total_papers = sum(l2.paper_count for l2 in l2_summaries)
    categories = [l2.category for l2 in l2_summaries]

    if total_words is None:
        total_words = _compute_target_words(n_categories)

    # Ordina L2 per nome categoria (ordine logico/alfabetico)
    l2_sorted = sorted(l2_summaries, key=lambda l: l.category)

    # Formatta L2 per il prompt
    l2_text = _format_l2_for_prompt(l2_sorted)

    # Calcola placeholder numeri sezione
    # Le sezioni body occupano posizioni 4..N, poi testing, defenses, discussion, conclusion
    n_plus_1 = n_categories + 4  # Testing
    n_plus_2 = n_categories + 5  # Defenses
    n_plus_3 = n_categories + 6  # Discussion
    n_plus_4 = n_categories + 7  # Conclusion

    prompt = _PLANNER_PROMPT_TEMPLATE.format(
        n_categories=n_categories,
        total_papers=total_papers,
        l2_texts=l2_text,
        n_plus_1=n_plus_1,
        n_plus_2=n_plus_2,
        n_plus_3=n_plus_3,
        n_plus_4=n_plus_4,
        min_sections=min_sections,
        target_sections=target_sections,
        total_words=total_words,
    )

    # --- Self-evolution: inject lessons overlay for planner ---
    if settings.evolution.enabled or settings.skills.enabled:
        try:
            evo_store = LessonStore(
                path=settings.evolution.lessons_path,
                half_life_days=settings.evolution.half_life_days,
                expiry_days=settings.evolution.expiry_days,
            )
            overlay = generate_full_overlay(
                evo_store,
                agent_type="planner",
                skills_dir=settings.skills.skills_dir if settings.skills.enabled else None,
                skills_enabled=settings.skills.enabled,
                max_skill_chars=settings.skills.max_total_chars,
                max_lessons=settings.evolution.max_overlay_lessons,
                max_lesson_chars=settings.evolution.max_overlay_chars,
            )
            if overlay:
                prompt = f"{prompt}\n\n{overlay}"
                logger.info("[evolution] Injected %d-char overlay for planner", len(overlay))
        except Exception as e:
            logger.warning("[evolution] Failed to load overlay: %s", e)

    # Selezione modello
    tier = force_model or settings.agents.planner_model
    logger.info(
        f"Generazione outline con {tier.upper()} — "
        f"{n_categories} categorie, {total_papers} paper, "
        f"target {target_sections} sezioni, ~{total_words} parole"
    )

    t0 = time.time()

    # Retry pazienti (come per L1/L2)
    max_retries = settings.summarization.pro_max_retries
    base_delay = settings.summarization.pro_base_delay
    max_delay = 120.0

    # Con DeepSeek il budget di output include il reasoning. Il vecchio limite
    # fisso di 16k poteva esaurirsi prima che il modello producesse l'outline.
    planner_max_tokens = (
        settings.deepseek.max_output_tokens_pro
        if settings.llm_provider == "deepseek"
        else 16384
    )
    llm = get_llm(tier, max_tokens=planner_max_tokens)

    raw_response = llm_call_with_retry(
        llm,
        prompt,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    elapsed = time.time() - t0
    logger.info(f"Risposta LLM ricevuta in {elapsed:.1f}s ({len(raw_response)} chars)")

    # Parse JSON
    raw_sections = _parse_outline_json(raw_response)
    if raw_sections is None:
        raise RuntimeError(
            "Impossibile parsare l'outline dalla risposta LLM. "
            f"Primi 500 chars: {raw_response[:500]}"
        )

    logger.info(f"Parsate {len(raw_sections)} sezioni dall'outline LLM")

    # Valida e normalizza
    outline = _validate_and_normalize_outline(raw_sections, categories)

    # Log riepilogo
    _log_outline_summary(outline)

    return outline


def _log_outline_summary(outline: list[SectionOutline]) -> None:
    """Logga un riepilogo dell'outline generato."""
    total_words = sum(o.target_length_words for o in outline)
    types_count: dict[str, int] = {}
    for o in outline:
        types_count[o.section_type] = types_count.get(o.section_type, 0) + 1

    data = [
        {"#": str(o.order), "Key": o.key, "Title": o.title,
         "Type": o.section_type, "Words": str(o.target_length_words),
         "Topics": str(len(o.key_topics)),
         "Categories": ", ".join(o.related_categories[:3]) or "-"}
        for o in outline
    ]
    print_table("Survey Outline", data)

    logger.info(
        f"Outline: {len(outline)} sezioni, ~{total_words} parole target, "
        f"tipi: {types_count}"
    )


# ============================================================================
# LangGraph node function
# ============================================================================

def plan(state: SurveyState) -> dict[str, Any]:
    """Nodo LangGraph: Planner Agent.

    Genera o carica l'outline del survey partendo dai L2 summaries
    presenti nello stato. Se l'outline è già presente nello stato
    (es. override utente), lo mantiene.

    Args:
        state: Stato corrente della pipeline LangGraph.

    Returns:
        Dict parziale con i campi aggiornati: ``outline``,
        ``current_phase``, ``errors``, ``metadata``.
    """
    logger.info("=" * 60)
    logger.info("PLANNER AGENT — Inizio generazione outline")
    logger.info("=" * 60)

    errors: list[str] = []

    # --- Controlla se c'è già un outline (override utente) ---
    existing_outline = get_outline_from_state(state)
    if existing_outline:
        logger.info(
            f"Outline già presente nello stato ({len(existing_outline)} sezioni). "
            f"Uso l'outline esistente (override utente)."
        )
        print_success(f"Outline esistente con {len(existing_outline)} sezioni — skip generazione.")
        _log_outline_summary(existing_outline)
        return {
            "outline": [o.to_dict() for o in existing_outline],
            "current_phase": PipelinePhase.PLANNING.value,
            "metadata": {
                **state.get("metadata", {}),
                "planner_source": "user_override",
                "planner_sections": len(existing_outline),
            },
        }

    # --- Carica L2 summaries ---
    l2_summaries = get_l2_summaries_from_state(state)

    if not l2_summaries:
        logger.info("Nessun L2 nello stato. Tentativo caricamento dal QdrantStore...")
        l2_summaries = _load_l2_from_store()

    if not l2_summaries:
        err = (
            "Impossibile generare l'outline: nessun L2 summary disponibile. "
            "Eseguire prima la summarization L1 + L2."
        )
        logger.error(err)
        print_error(err)
        return {
            "outline": [],
            "current_phase": PipelinePhase.FAILED.value,
            "errors": [err],
        }

    logger.info(
        f"L2 disponibili: {len(l2_summaries)} categorie — "
        + ", ".join(f"{l.category} ({l.paper_count}p)" for l in l2_summaries)
    )

    # --- Controlla se c'è un outline_file nel metadata ---
    outline_file = state.get("metadata", {}).get("outline_file")
    if outline_file:
        try:
            logger.info(f"Caricamento outline da file: {outline_file}")
            outline = load_outline_from_json(outline_file)
            print_success(f"Outline caricato da {outline_file}: {len(outline)} sezioni.")
            _log_outline_summary(outline)
            return {
                "outline": [o.to_dict() for o in outline],
                "current_phase": PipelinePhase.PLANNING.value,
                "metadata": {
                    **state.get("metadata", {}),
                    "planner_source": f"file:{outline_file}",
                    "planner_sections": len(outline),
                },
            }
        except Exception as e:
            logger.warning(f"Errore caricamento outline da file: {e}. Procedo con generazione LLM.")
            errors.append(f"Fallback da file outline ({outline_file}): {e}")

    # --- Generazione outline via LLM ---
    try:
        outline = generate_outline(l2_summaries)

        categories = [l.category for l in l2_summaries]
        total_papers = sum(l.paper_count for l in l2_summaries)

        print_success(
            f"Outline generato con successo: {len(outline)} sezioni, "
            f"~{sum(o.target_length_words for o in outline)} parole target."
        )

        return {
            "outline": [o.to_dict() for o in outline],
            "current_phase": PipelinePhase.PLANNING.value,
            "errors": errors if errors else [],
            "metadata": {
                **state.get("metadata", {}),
                "planner_source": "llm_generated",
                "planner_model": settings.agents.planner_model,
                "planner_sections": len(outline),
                "planner_categories": categories,
                "planner_total_papers": total_papers,
            },
        }

    except Exception as e:
        err = f"Errore generazione outline: {e}"
        logger.error(err, exc_info=True)
        print_error(err)
        errors.append(err)

        return {
            "outline": [],
            "current_phase": PipelinePhase.FAILED.value,
            "errors": errors,
        }


# ============================================================================
# Convenience functions
# ============================================================================

def plan_from_l2_summaries(
    l2_summaries: list[L2Summary],
    *,
    outline_file: str | Path | None = None,
    force_model: str | None = None,
) -> list[SectionOutline]:
    """Genera l'outline partendo direttamente da una lista di L2Summary.

    Funzione di convenienza che non richiede il SurveyState.

    Args:
        l2_summaries: Lista dei L2 summary (uno per categoria).
        outline_file: Se fornito, carica l'outline da file (override).
        force_model: Override modello LLM.

    Returns:
        Lista ordinata di SectionOutline.
    """
    if outline_file:
        return load_outline_from_json(outline_file)
    return generate_outline(l2_summaries, force_model=force_model)


def get_outline_stats(outline: list[SectionOutline]) -> dict[str, Any]:
    """Calcola statistiche sull'outline.

    Args:
        outline: Lista di SectionOutline.

    Returns:
        Dict con statistiche aggregate.
    """
    if not outline:
        return {"total_sections": 0, "total_target_words": 0}

    total_words = sum(o.target_length_words for o in outline)
    types_count: dict[str, int] = {}
    all_categories: set[str] = set()
    all_topics: set[str] = set()

    for o in outline:
        types_count[o.section_type] = types_count.get(o.section_type, 0) + 1
        all_categories.update(o.related_categories)
        all_topics.update(o.key_topics)

    body_sections = [o for o in outline if o.section_type == "body"]
    avg_body_words = (
        sum(o.target_length_words for o in body_sections) / len(body_sections)
        if body_sections else 0
    )

    return {
        "total_sections": len(outline),
        "total_target_words": total_words,
        "section_types": types_count,
        "body_sections": len(body_sections),
        "avg_body_words": round(avg_body_words),
        "categories_covered": sorted(all_categories),
        "total_topics": len(all_topics),
        "has_abstract": any(o.section_type == "abstract" for o in outline),
        "has_introduction": any(o.section_type == "introduction" for o in outline),
        "has_testing": any(o.section_type == "testing" for o in outline),
        "has_discussion": any(o.section_type == "discussion" for o in outline),
        "has_conclusion": any(o.section_type == "conclusion" for o in outline),
    }
