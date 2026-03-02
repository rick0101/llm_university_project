"""
agents/pipeline.py — LangGraph Workflow Orchestration

Definisce il grafo LangGraph che orchestra tutti gli agent della
pipeline di scrittura survey:

    START → plan → write → verify → should_revise?
        ├─ YES → write (loop write↔verify, max N iterazioni)
        └─ NO  → assemble → END

Funzionalità principali:
- `build_pipeline()` → compila il StateGraph in un grafo eseguibile
- `run_pipeline()` → esegue l'intera pipeline da zero o da checkpoint
- `run_pipeline_from_checkpoint()` → riprende da un checkpoint salvato
- Checkpoint JSON dopo ogni nodo per resume-from-failure
- Progress bar Rich per tracking avanzamento
- Error handling: se un agente fallisce, logga e continua (dove possibile)
- Callback opzionale per progress tracking

Usage:
    from paperhunter.agents.pipeline import build_pipeline, run_pipeline

    # Esecuzione completa
    final_state = run_pipeline(l2_summaries=[...])

    # Con outline pre-definito
    final_state = run_pipeline(
        l2_summaries=[...],
        outline=my_outline,
    )

    # Resume da checkpoint
    final_state = run_pipeline_from_checkpoint("output/.checkpoints/checkpoint_verify.json")
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from langgraph.graph import END, START, StateGraph

from paperhunter.agents.assembler_agent import assemble
from paperhunter.agents.critic_agent import verify
from paperhunter.agents.planner_agent import plan
from paperhunter.agents.state import (
    PipelinePhase,
    SectionOutline,
    SectionStatus,
    SurveyState,
    create_initial_state,
    get_outline_from_state,
    get_pipeline_summary,
    get_sections_from_state,
    get_verifications_from_state,
    state_from_dict,
    state_to_dict,
)
from paperhunter.agents.writer_agent import write
from paperhunter.config import settings
from paperhunter.models import L2Summary, SurveyVersion
from paperhunter.utils.logging_setup import (
    get_logger,
    print_error,
    print_panel,
    print_success,
    print_table,
    print_warning,
)
from paperhunter.utils.token_counter import BudgetReport

logger = get_logger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Nomi dei nodi nel grafo
NODE_PLAN = "plan"
NODE_WRITE = "write"
NODE_VERIFY = "verify"
NODE_ASSEMBLE = "assemble"

# Chiave per l'edge condizionale
EDGE_REVISE = "revise"
EDGE_DONE = "done"


# ============================================================================
# Checkpoint management
# ============================================================================

def _get_checkpoint_dir() -> Path:
    """Ritorna la directory per i checkpoint, creandola se necessario."""
    cp_dir = Path(settings.output.checkpoints_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)
    return cp_dir


def save_checkpoint(
    state: SurveyState,
    node_name: str,
    checkpoint_dir: Path | None = None,
) -> Path:
    """Salva un checkpoint JSON dello stato dopo un nodo.

    Args:
        state: Stato corrente della pipeline.
        node_name: Nome del nodo appena completato (es. 'plan', 'write').
        checkpoint_dir: Directory dove salvare (default da config).

    Returns:
        Path del file checkpoint salvato.
    """
    cp_dir = checkpoint_dir or _get_checkpoint_dir()
    cp_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"checkpoint_{node_name}_{timestamp}.json"
    filepath = cp_dir / filename

    # Salva anche un 'latest' per ogni nodo per facile resume
    latest_filepath = cp_dir / f"checkpoint_{node_name}_latest.json"

    data = {
        "node": node_name,
        "timestamp": timestamp,
        "state": state_to_dict(state),
    }

    for fp in (filepath, latest_filepath):
        fp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    logger.info(f"Checkpoint salvato: {filepath.name}")
    return filepath


def load_checkpoint(checkpoint_path: str | Path) -> tuple[str, SurveyState]:
    """Carica un checkpoint JSON e ritorna nodo + stato.

    Args:
        checkpoint_path: Path al file checkpoint JSON.

    Returns:
        Tupla (node_name, SurveyState) dove node_name è l'ultimo
        nodo completato con successo.

    Raises:
        FileNotFoundError: Se il file non esiste.
        json.JSONDecodeError: Se il file non è JSON valido.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    node_name = data.get("node", "unknown")
    state_dict = data.get("state", {})
    state = state_from_dict(state_dict)

    logger.info(
        f"Checkpoint caricato: nodo='{node_name}', "
        f"timestamp={data.get('timestamp', '?')}"
    )
    return node_name, state


def get_latest_checkpoint(node_name: str | None = None) -> Path | None:
    """Cerca l'ultimo checkpoint disponibile.

    Args:
        node_name: Se specificato, cerca il latest di quel nodo.
                   Se None, cerca il checkpoint più recente in assoluto.

    Returns:
        Path del checkpoint più recente, o None se non ne esistono.
    """
    cp_dir = _get_checkpoint_dir()
    if not cp_dir.exists():
        return None

    if node_name:
        latest = cp_dir / f"checkpoint_{node_name}_latest.json"
        return latest if latest.exists() else None

    # Trova il più recente tra tutti i checkpoint (esclusi i _latest)
    checkpoints = sorted(
        [f for f in cp_dir.glob("checkpoint_*.json") if "_latest" not in f.name],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def clean_checkpoints(keep_latest: bool = True) -> int:
    """Rimuove vecchi checkpoint.

    Args:
        keep_latest: Se True, mantiene i file *_latest.json.

    Returns:
        Numero di file rimossi.
    """
    cp_dir = _get_checkpoint_dir()
    if not cp_dir.exists():
        return 0

    removed = 0
    for f in cp_dir.glob("checkpoint_*.json"):
        if keep_latest and "_latest" in f.name:
            continue
        f.unlink()
        removed += 1

    if removed:
        logger.info(f"Rimossi {removed} checkpoint vecchi.")
    return removed


# ============================================================================
# Budget summary
# ============================================================================

def print_budget_summary(
    n_categories: int = 4,
    n_sections_estimate: int = 8,
) -> BudgetReport:
    """Calcola e stampa un sommario di budget prima di avviare la pipeline.

    Stima i token necessari per ciascuna fase (L1, L2, Writer, Critic)
    rispetto al budget token massimo configurato. Se il budget stimato
    supera il limite, avvisa e suggerisce azioni correttive.

    Args:
        n_categories: Numero di categorie di paper (default: 4).
        n_sections_estimate: Stima del numero di sezioni nell'outline
            finale. Se un outline concreto è disponibile, passare
            ``len(outline)``.

    Returns:
        L'istanza BudgetReport con tutte le stime calcolate.
    """
    logger.info(
        f"Calcolo budget stimato: {n_categories} categorie, "
        f"~{n_sections_estimate} sezioni"
    )
    report = BudgetReport.from_config(
        n_categories=n_categories,
        n_sections_estimate=n_sections_estimate,
    )
    report.print_report()

    if not report.within_budget:
        logger.warning(
            f"Budget stimato FUORI LIMITE — "
            f"{len(report.warnings)} warning(s). "
            f"La pipeline proseguirà ma potrebbe generare output degradato."
        )
    else:
        logger.info("Budget stimato OK — tutte le fasi entro il limite.")

    return report


# ============================================================================
# Conditional edge: should we revise (write↔verify loop)?
# ============================================================================

def _should_revise(state: SurveyState) -> str:
    """Conditional edge: determina se servono ulteriori iterazioni write↔verify.

    Controlla:
    1. Se ci sono sezioni con status NEEDS_REVISION
    2. Se il conteggio iterazioni è sotto il massimo consentito

    Returns:
        "revise" se bisogna tornare al Writer, "done" altrimenti.
    """
    max_iterations = settings.agents.max_writer_iterations
    iteration_count = state.get("iteration_count", 0)  # type: ignore[attr-defined]

    # Verifica se ci sono sezioni che necessitano revisione
    sections = get_sections_from_state(state)
    needs_revision = [
        k for k, v in sections.items()
        if v.status == SectionStatus.NEEDS_REVISION
    ]

    if not needs_revision:
        logger.info("Tutte le sezioni verificate — nessuna revisione necessaria.")
        return EDGE_DONE

    if iteration_count >= max_iterations:
        logger.warning(
            f"Raggiunto il limite di {max_iterations} iterazioni. "
            f"{len(needs_revision)} sezioni rimangono non verificate: {needs_revision}"
        )
        print_warning(
            f"Max iterazioni raggiunte ({max_iterations}). "
            f"{len(needs_revision)} sezioni non verificate."
        )
        return EDGE_DONE

    logger.info(
        f"Iterazione {iteration_count + 1}/{max_iterations}: "
        f"{len(needs_revision)} sezioni da riscrivere: {needs_revision}"
    )
    return EDGE_REVISE


# ============================================================================
# Wrapper nodes con checkpoint e progress tracking
# ============================================================================

def _make_node_with_checkpoint(
    node_fn: Callable[[SurveyState], dict[str, Any]],
    node_name: str,
    save_checkpoints: bool = True,
    on_progress: Callable[[str, str], None] | None = None,
) -> Callable[[SurveyState], dict[str, Any]]:
    """Crea un wrapper per un nodo che salva checkpoint e tracka progress.

    Args:
        node_fn: La funzione nodo originale (plan/write/verify/assemble).
        node_name: Nome identificativo del nodo.
        save_checkpoints: Se True, salva checkpoint dopo ogni nodo.
        on_progress: Callback opzionale (node_name, status).

    Returns:
        Funzione wrapper con la stessa signature.
    """

    def wrapped(state: SurveyState) -> dict[str, Any]:
        start_time = time.time()

        if on_progress:
            on_progress(node_name, "started")

        logger.info(f"\n{'='*70}")
        logger.info(f"  PIPELINE NODE: {node_name.upper()}")
        logger.info(f"{'='*70}")

        try:
            result = node_fn(state)

            # Incrementa il contatore iterazioni per il loop write↔verify
            if node_name == NODE_VERIFY:
                current_iter = state.get("iteration_count", 0)  # type: ignore[attr-defined]
                result["iteration_count"] = current_iter + 1

            elapsed = time.time() - start_time
            logger.info(
                f"Nodo '{node_name}' completato in {elapsed:.1f}s"
            )

            # Salva checkpoint
            if save_checkpoints:
                # Merge del risultato nello stato per il checkpoint
                merged_state = {**state, **result}  # type: ignore[misc]
                try:
                    save_checkpoint(merged_state, node_name)  # type: ignore[arg-type]
                except Exception as e:
                    logger.warning(f"Errore salvataggio checkpoint '{node_name}': {e}")

            if on_progress:
                on_progress(node_name, "completed")

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            err_msg = f"Errore nel nodo '{node_name}' dopo {elapsed:.1f}s: {e}"
            logger.error(err_msg, exc_info=True)
            print_error(err_msg)

            if on_progress:
                on_progress(node_name, "failed")

            # Per plan e assemble, il fallimento è critico
            if node_name in (NODE_PLAN, NODE_ASSEMBLE):
                return {
                    "current_phase": PipelinePhase.FAILED.value,
                    "errors": [err_msg],
                }

            # Per write/verify, logga l'errore ma permetti continuazione
            return {
                "errors": [err_msg],
            }

    wrapped.__name__ = f"{node_name}_node"
    wrapped.__doc__ = f"Wrapper checkpoint per nodo '{node_name}'."
    return wrapped


# ============================================================================
# Build LangGraph pipeline
# ============================================================================

def build_pipeline(
    save_checkpoints: bool = True,
    on_progress: Callable[[str, str], None] | None = None,
) -> Any:
    """Costruisce e compila il grafo LangGraph per la pipeline survey.

    Grafo:
        START → plan → write → verify → should_revise?
            ├─ "revise" → write (loop)
            └─ "done"   → assemble → END

    Args:
        save_checkpoints: Se True, salva checkpoint dopo ogni nodo.
        on_progress: Callback (node_name, status) per progress tracking.

    Returns:
        Il grafo LangGraph compilato, pronto per `.invoke()`.
    """
    logger.info("Costruzione grafo LangGraph...")

    # Crea nodi con wrapper checkpoint
    plan_node = _make_node_with_checkpoint(
        plan, NODE_PLAN,
        save_checkpoints=save_checkpoints,
        on_progress=on_progress,
    )
    write_node = _make_node_with_checkpoint(
        write, NODE_WRITE,
        save_checkpoints=save_checkpoints,
        on_progress=on_progress,
    )
    verify_node = _make_node_with_checkpoint(
        verify, NODE_VERIFY,
        save_checkpoints=save_checkpoints,
        on_progress=on_progress,
    )
    assemble_node = _make_node_with_checkpoint(
        assemble, NODE_ASSEMBLE,
        save_checkpoints=save_checkpoints,
        on_progress=on_progress,
    )

    # Definisci il grafo
    graph = StateGraph(SurveyState)

    # Aggiungi nodi
    graph.add_node(NODE_PLAN, plan_node)
    graph.add_node(NODE_WRITE, write_node)
    graph.add_node(NODE_VERIFY, verify_node)
    graph.add_node(NODE_ASSEMBLE, assemble_node)

    # Definisci edges
    graph.add_edge(START, NODE_PLAN)
    graph.add_edge(NODE_PLAN, NODE_WRITE)
    graph.add_edge(NODE_WRITE, NODE_VERIFY)

    # Conditional edge: verify → write (revise) o verify → assemble (done)
    graph.add_conditional_edges(
        NODE_VERIFY,
        _should_revise,
        {
            EDGE_REVISE: NODE_WRITE,
            EDGE_DONE: NODE_ASSEMBLE,
        },
    )

    graph.add_edge(NODE_ASSEMBLE, END)

    # Compila il grafo
    compiled = graph.compile()
    logger.info("Grafo LangGraph compilato con successo.")

    return compiled


# ============================================================================
# Run pipeline
# ============================================================================

def run_pipeline(
    l2_summaries: list[L2Summary] | None = None,
    outline: list[SectionOutline] | None = None,
    outline_file: str | None = None,
    version_info: SurveyVersion | None = None,
    metadata: dict | None = None,
    save_checkpoints: bool = True,
    on_progress: Callable[[str, str], None] | None = None,
) -> SurveyState:
    """Esegue l'intera pipeline di generazione survey.

    Args:
        l2_summaries: L2 summaries da usare come materiale sorgente.
        outline: Outline pre-definito (opzionale — il Planner lo genera se assente).
        outline_file: Path a un file JSON outline (alternativo a `outline`).
        version_info: Metadata versioning (opzionale — auto-creato se assente).
        metadata: Dict metadata aggiuntivo.
        save_checkpoints: Se True, salva checkpoint dopo ogni nodo.
        on_progress: Callback (node_name, status) per progress tracking.

    Returns:
        Lo stato finale della pipeline con il paper assemblato.
    """
    start_time = time.time()

    print_panel(
        "LLM Security Survey Generator",
        "Pipeline LangGraph multi-agent\n"
        "Plan → Write → Verify ↔ Revise → Assemble",
    )

    # --- Budget summary prima di avviare la pipeline ---
    n_sections_est = len(outline) if outline else 8
    budget_report = print_budget_summary(
        n_categories=4,
        n_sections_estimate=n_sections_est,
    )

    # Prepara metadata
    run_metadata = metadata or {}
    run_metadata["budget_report"] = budget_report.to_dict()
    if outline_file:
        run_metadata["outline_file"] = outline_file
    run_metadata["pipeline_start"] = datetime.now().isoformat()
    run_metadata["max_iterations"] = settings.agents.max_writer_iterations

    # Crea stato iniziale
    initial_state = create_initial_state(
        l2_summaries=l2_summaries,
        outline=outline,
        version_info=version_info,
        metadata=run_metadata,
    )

    logger.info(
        f"Pipeline avviata: "
        f"{len(l2_summaries or [])} L2 summaries, "
        f"{'outline fornito' if outline else 'outline da generare'}, "
        f"max {settings.agents.max_writer_iterations} iterazioni write↔verify"
    )

    # Costruisci e esegui il grafo
    compiled_graph = build_pipeline(
        save_checkpoints=save_checkpoints,
        on_progress=on_progress,
    )

    try:
        final_state = compiled_graph.invoke(initial_state)
    except Exception as e:
        err_msg = f"Pipeline fallita con errore: {e}"
        logger.error(err_msg, exc_info=True)
        print_error(err_msg)
        # Ritorna lo stato iniziale con l'errore
        initial_state["errors"] = [err_msg]  # type: ignore[typeddict-item]
        initial_state["current_phase"] = PipelinePhase.FAILED.value  # type: ignore[typeddict-item]
        return initial_state

    elapsed = time.time() - start_time

    # Report finale
    _print_pipeline_report(final_state, elapsed)

    return final_state


def run_pipeline_from_checkpoint(
    checkpoint_path: str | Path,
    save_checkpoints: bool = True,
    on_progress: Callable[[str, str], None] | None = None,
) -> SurveyState:
    """Riprende la pipeline da un checkpoint salvato.

    Determina l'ultimo nodo completato e riprende dal nodo successivo.
    Il flusso del grafo (compreso il loop write↔verify) viene
    rispettato: la pipeline riparte dal nodo corretto nel grafo
    LangGraph.

    Args:
        checkpoint_path: Path al file checkpoint JSON.
        save_checkpoints: Se True, salva checkpoint dopo ogni nodo.
        on_progress: Callback (node_name, status) per progress tracking.

    Returns:
        Lo stato finale della pipeline.
    """
    start_time = time.time()

    logger.info(f"Resume pipeline da checkpoint: {checkpoint_path}")
    node_name, state = load_checkpoint(checkpoint_path)

    print_panel(
        "Resume Pipeline",
        f"Ultimo nodo completato: {node_name}\n"
        f"Ripresa dall'ultimo stato salvato",
    )

    # Budget summary anche al resume
    print_budget_summary()

    # Determina il nodo da cui ripartire
    node_sequence = [NODE_PLAN, NODE_WRITE, NODE_VERIFY, NODE_ASSEMBLE]
    try:
        completed_idx = node_sequence.index(node_name)
    except ValueError:
        logger.warning(
            f"Nodo checkpoint '{node_name}' non riconosciuto. "
            f"Ripartenza completa."
        )
        completed_idx = -1

    # Se l'ultimo nodo completato è 'verify', controlla se servono revisioni
    if node_name == NODE_VERIFY:
        decision = _should_revise(state)
        if decision == EDGE_REVISE:
            logger.info("Checkpoint da verify → servono revisioni → riparto da write")
            next_node = NODE_WRITE
        else:
            logger.info("Checkpoint da verify → tutte le sezioni OK → procedo ad assemble")
            next_node = NODE_ASSEMBLE
    elif completed_idx < len(node_sequence) - 1:
        next_node = node_sequence[completed_idx + 1]
    else:
        logger.info("Pipeline già completata (ultimo nodo = assemble).")
        return state

    logger.info(f"Ripresa dal nodo: {next_node}")

    # Costruisci il grafo completo
    compiled_graph = build_pipeline(
        save_checkpoints=save_checkpoints,
        on_progress=on_progress,
    )

    # Per il resume, sovrascriviamo lo stato iniziale con il checkpoint
    # e invochiamo partendo dal nodo appropriato.
    # LangGraph non supporta nativamente "start from node X", quindi
    # eseguiamo manualmente i nodi rimanenti in sequenza, rispettando
    # il loop write↔verify.

    final_state = _resume_from_node(
        state=state,
        start_node=next_node,
        save_checkpoints=save_checkpoints,
        on_progress=on_progress,
    )

    elapsed = time.time() - start_time
    _print_pipeline_report(final_state, elapsed)

    return final_state


def _resume_from_node(
    state: SurveyState,
    start_node: str,
    save_checkpoints: bool = True,
    on_progress: Callable[[str, str], None] | None = None,
) -> SurveyState:
    """Esegue manualmente i nodi rimanenti da un punto di ripresa.

    Rispetta il loop write↔verify e il flusso condizionale.

    Args:
        state: Stato da cui ripartire.
        start_node: Primo nodo da eseguire.
        save_checkpoints: Se True, salva checkpoint dopo ogni nodo.
        on_progress: Callback.

    Returns:
        Stato finale dopo l'esecuzione di tutti i nodi rimanenti.
    """
    max_iterations = settings.agents.max_writer_iterations

    # Crea i nodi con checkpoint
    nodes = {
        NODE_PLAN: _make_node_with_checkpoint(
            plan, NODE_PLAN, save_checkpoints, on_progress
        ),
        NODE_WRITE: _make_node_with_checkpoint(
            write, NODE_WRITE, save_checkpoints, on_progress
        ),
        NODE_VERIFY: _make_node_with_checkpoint(
            verify, NODE_VERIFY, save_checkpoints, on_progress
        ),
        NODE_ASSEMBLE: _make_node_with_checkpoint(
            assemble, NODE_ASSEMBLE, save_checkpoints, on_progress
        ),
    }

    current_state = state
    node_sequence = [NODE_PLAN, NODE_WRITE, NODE_VERIFY, NODE_ASSEMBLE]

    try:
        start_idx = node_sequence.index(start_node)
    except ValueError:
        start_idx = 0

    # Esegui i nodi in sequenza a partire da start_node
    i = start_idx
    while i < len(node_sequence):
        node_name = node_sequence[i]

        # Esegui il nodo
        try:
            result = nodes[node_name](current_state)
            # Merge del risultato nello stato corrente
            current_state = {**current_state, **result}  # type: ignore[misc]
        except Exception as e:
            err_msg = f"Errore critico nel nodo '{node_name}': {e}"
            logger.error(err_msg, exc_info=True)
            current_state["errors"] = current_state.get("errors", []) + [err_msg]  # type: ignore[attr-defined]
            current_state["current_phase"] = PipelinePhase.FAILED.value  # type: ignore[typeddict-item]
            break

        # Controlla fase fallita
        if current_state.get("current_phase") == PipelinePhase.FAILED.value:  # type: ignore[attr-defined]
            logger.error(f"Pipeline fallita al nodo '{node_name}'.")
            break

        # Dopo verify: controlla se serve il loop write↔verify
        if node_name == NODE_VERIFY:
            decision = _should_revise(current_state)  # type: ignore[arg-type]
            if decision == EDGE_REVISE:
                # Torna a write (i rimane su verify, ma sovrascriviamo)
                i = node_sequence.index(NODE_WRITE)
                continue

        i += 1

    return current_state  # type: ignore[return-value]


# ============================================================================
# Pipeline report
# ============================================================================

def _print_pipeline_report(state: SurveyState, elapsed: float) -> None:
    """Stampa un report finale della pipeline con statistiche.

    Args:
        state: Stato finale della pipeline.
        elapsed: Tempo totale in secondi.
    """
    summary = get_pipeline_summary(state)

    phase = summary.get("current_phase", "unknown")
    is_success = phase in (
        PipelinePhase.COMPLETED.value,
        PipelinePhase.ASSEMBLING.value,
    )
    has_paper = summary.get("has_final_paper", False)

    # Statistiche di base in tabella
    stats_data = {
        "Stato finale": phase,
        "Sezioni nell'outline": str(summary.get("outline_sections", 0)),
        "Sezioni scritte": str(summary.get("drafted_sections", 0)),
        "Citazioni uniche": str(summary.get("total_unique_citations", 0)),
        "Record provenance": str(summary.get("provenance_records", 0)),
        "Paper generato": "Sì" if has_paper else "No",
        "Iterazioni totali": str(summary.get("iteration_count", 0)),
        "Tempo totale": f"{elapsed:.1f}s ({elapsed / 60:.1f} min)",
    }
    print_table("Pipeline Report", stats_data)

    # Dettaglio verifica per sezione
    verif = summary.get("verification_summary", {})
    if verif:
        verif_data = []
        for key, v in verif.items():
            verif_data.append({
                "Sezione": key[:30],
                "Citazioni": str(v.get("total", 0)),
                "Supportate": str(v.get("supported", 0)),
                "Pass Rate": f"{v.get('pass_rate', 0):.1%}",
                "Passata": "✓" if v.get("passed") else "✗",
                "Iterazioni": str(v.get("iteration", 0)),
            })
        print_table("Dettaglio Verifiche", verif_data)

    # Errori
    errors = summary.get("errors", [])
    if errors:
        print_warning(f"{len(errors)} errori durante la pipeline:")
        for err in errors[:10]:  # Max 10 errori
            logger.warning(f"  • {err}")
        if len(errors) > 10:
            logger.warning(f"  ... e altri {len(errors) - 10} errori.")

    # Status sezioni
    statuses = summary.get("section_statuses", {})
    if statuses:
        status_counts: dict[str, int] = {}
        for s in statuses.values():
            status_counts[s] = status_counts.get(s, 0) + 1
        status_str = ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))
        logger.info(f"Status sezioni: {status_str}")

    # Messaggio finale
    if has_paper:
        print_success(
            f"Pipeline completata con successo in {elapsed:.1f}s! "
            f"Paper generato con {summary.get('total_unique_citations', 0)} citazioni."
        )
    elif is_success:
        print_success(f"Pipeline terminata in {elapsed:.1f}s (fase: {phase}).")
    else:
        print_error(f"Pipeline terminata con errori in {elapsed:.1f}s (fase: {phase}).")


# ============================================================================
# Quick-access helpers
# ============================================================================

def get_pipeline_graph_description() -> str:
    """Ritorna una descrizione testuale del grafo della pipeline.

    Utile per documentazione e debugging.
    """
    return (
        "LangGraph Survey Pipeline:\n"
        "  START → plan → write → verify → should_revise?\n"
        "    ├─ YES (needs revision) → write (loop, max N iter)\n"
        "    └─ NO  (all verified)   → assemble → END\n"
        f"\n"
        f"  Max write↔verify iterations: {settings.agents.max_writer_iterations}\n"
        f"  Verification threshold: {settings.agents.verification_threshold:.0%}\n"
        f"  Models: planner={settings.agents.planner_model}, "
        f"writer={settings.agents.writer_model}, "
        f"critic={settings.agents.critic_model}, "
        f"assembler={settings.agents.assembler_model}"
    )


def list_checkpoints() -> list[dict[str, Any]]:
    """Elenca tutti i checkpoint disponibili.

    Returns:
        Lista di dict con info su ogni checkpoint (path, node, timestamp, size).
    """
    cp_dir = _get_checkpoint_dir()
    if not cp_dir.exists():
        return []

    result = []
    for f in sorted(cp_dir.glob("checkpoint_*.json"), key=lambda f: f.stat().st_mtime):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result.append({
                "path": str(f),
                "filename": f.name,
                "node": data.get("node", "unknown"),
                "timestamp": data.get("timestamp", "unknown"),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "is_latest": "_latest" in f.name,
            })
        except Exception:
            result.append({
                "path": str(f),
                "filename": f.name,
                "node": "error",
                "timestamp": "error",
                "size_kb": round(f.stat().st_size / 1024, 1),
                "is_latest": "_latest" in f.name,
            })

    return result
