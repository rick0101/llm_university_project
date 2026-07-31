"""
Test per agents/pipeline.py — LangGraph Workflow Orchestration.

Test offline (mock dei singoli agent) che verificano:
1. build_pipeline — costruzione e compilazione del grafo
2. _should_revise — logica condizionale write↔verify loop
3. save_checkpoint / load_checkpoint — round-trip JSON
4. get_latest_checkpoint — ricerca checkpoint più recente
5. clean_checkpoints — pulizia file vecchi
6. list_checkpoints — elenco checkpoint disponibili
7. _make_node_with_checkpoint — wrapper nodi con checkpoint
8. run_pipeline — esecuzione completa con mock agent
9. run_pipeline_from_checkpoint — resume da checkpoint
10. _resume_from_node — esecuzione manuale nodi rimanenti
11. _print_pipeline_report — report finale
12. get_pipeline_graph_description — descrizione testuale

Esecuzione:
    python test_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Aggiungi il progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Helper: crea stato mock
# ============================================================================

def _create_mock_state(
    include_outline: bool = False,
    include_sections: bool = False,
    include_verifications: bool = False,
    needs_revision: bool = False,
    all_verified: bool = False,
    include_l2: bool = True,
) -> dict:
    """Crea un SurveyState di test con dati finti."""
    from paperhunter.agents.state import (
        CitationRecord,
        SectionDraft,
        SectionOutline,
        SectionStatus,
        VerificationResult,
        VerificationDetail,
        PipelinePhase,
        create_initial_state,
    )
    from paperhunter.models import L2Summary, SurveyVersion, VerificationStatus

    # L2 summaries
    l2_summaries = None
    if include_l2:
        l2_summaries = [
            L2Summary(
                category="Prompt Injection",
                summary_text="Overview of prompt injection attacks on LLMs.",
                paper_count=10,
                version=1,
            ),
            L2Summary(
                category="Adversarial Attacks",
                summary_text="Survey of adversarial attacks against LLMs.",
                paper_count=8,
                version=1,
            ),
        ]

    # Outline
    outline = None
    if include_outline:
        outline = [
            SectionOutline(
                key="introduction", title="Introduction", order=0,
                section_type="introduction", description="Intro",
                target_length_words=500,
            ),
            SectionOutline(
                key="prompt_injection", title="Prompt Injection", order=1,
                section_type="body", description="Overview PI",
                target_length_words=1000,
                related_categories=["Prompt Injection"],
            ),
            SectionOutline(
                key="adversarial_attacks", title="Adversarial Attacks", order=2,
                section_type="body", description="Overview AA",
                target_length_words=1000,
                related_categories=["Adversarial Attacks"],
            ),
            SectionOutline(
                key="conclusion", title="Conclusion", order=3,
                section_type="conclusion", description="Wrap-up",
                target_length_words=400,
            ),
        ]

    state = create_initial_state(l2_summaries=l2_summaries, outline=outline)

    # Sections  
    if include_sections:
        sections = {}
        for key in ["introduction", "prompt_injection", "adversarial_attacks", "conclusion"]:
            status = SectionStatus.DRAFTED
            critic_feedback = ""
            if needs_revision and key == "prompt_injection":
                status = SectionStatus.NEEDS_REVISION
                critic_feedback = "Citazione Zou_2023_GCG non supportata dal chunk."
            if all_verified:
                status = SectionStatus.VERIFIED

            sections[key] = SectionDraft(
                section_key=key,
                title=key.replace("_", " ").title(),
                text=f"This is the text for section {key}. " * 20,
                citations=[
                    CitationRecord(bibtex_key="Zou_2023_GCG", claim="Test claim"),
                    CitationRecord(bibtex_key="Li_2023_Halueval", claim="Another claim"),
                ],
                word_count=200,
                status=status,
                iteration=1,
                critic_feedback=critic_feedback,
            ).to_dict()
        state["sections"] = sections

    if include_verifications:
        verifications = {}
        for key in ["introduction", "prompt_injection", "adversarial_attacks", "conclusion"]:
            passed = True
            if needs_revision and key == "prompt_injection":
                passed = False
            vr = VerificationResult(
                section_key=key,
                total_citations=2,
                supported=2 if passed else 0,
                partially_supported=0,
                not_supported=0 if passed else 2,
                source_not_found=0,
                pass_rate=1.0 if passed else 0.0,
                passed=passed,
                iteration=1,
            ).to_dict()
            verifications[key] = vr
        state["verification_results"] = verifications

    return state


# ============================================================================
# Test 1: build_pipeline
# ============================================================================

def test_build_pipeline():
    """Verifica che build_pipeline compila il grafo senza errori."""
    from paperhunter.agents.pipeline import build_pipeline

    graph = build_pipeline(save_checkpoints=False)
    assert graph is not None
    print("[PASS] test_build_pipeline — grafo compilato con successo")


# ============================================================================
# Test 2: _should_revise — logica condizionale
# ============================================================================

def test_should_revise_no_revisions_needed():
    """Se tutte le sezioni sono verified → DONE."""
    from paperhunter.agents.pipeline import _should_revise, EDGE_DONE

    state = _create_mock_state(
        include_outline=True, include_sections=True,
        all_verified=True,
    )
    state["iteration_count"] = 1

    result = _should_revise(state)
    assert result == EDGE_DONE, f"Expected DONE, got {result}"
    print("[PASS] test_should_revise_no_revisions — tutte verificate → DONE")


def test_should_revise_needs_revision():
    """Se ci sono sezioni NEEDS_REVISION e iter < max → REVISE."""
    from paperhunter.agents.pipeline import _should_revise, EDGE_REVISE

    state = _create_mock_state(
        include_outline=True, include_sections=True,
        needs_revision=True,
    )
    state["iteration_count"] = 1

    result = _should_revise(state)
    assert result == EDGE_REVISE, f"Expected REVISE, got {result}"
    print("[PASS] test_should_revise_needs_revision — sezione da riscrivere → REVISE")


def test_should_revise_max_iterations():
    """A section stops only after its own writer-attempt budget is exhausted."""
    from paperhunter.agents.pipeline import _should_revise, EDGE_DONE
    from paperhunter.config import settings

    state = _create_mock_state(
        include_outline=True, include_sections=True,
        needs_revision=True,
    )
    state["iteration_count"] = settings.agents.max_writer_iterations
    for section in state["sections"].values():
        section["iteration"] = settings.agents.max_writer_iterations

    result = _should_revise(state)
    assert result == EDGE_DONE, f"Expected DONE (max iter), got {result}"
    print("[PASS] test_should_revise_max_iterations — max iter raggiunto → DONE")


def test_should_revise_allows_one_extra_coherence_repair():
    from paperhunter.agents.pipeline import EDGE_REVISE, _should_revise
    from paperhunter.config import settings

    state = _create_mock_state(
        include_outline=True,
        include_sections=True,
        needs_revision=True,
    )
    state["iteration_count"] = settings.agents.max_writer_iterations
    for section in state["sections"].values():
        section["iteration"] = settings.agents.max_writer_iterations + 1
    target = state["sections"]["prompt_injection"]
    target["iteration"] = settings.agents.max_writer_iterations
    target["critic_feedback"] = (
        "[COHERENCE PASS — issues to fix]\nRemove duplicated prose."
    )

    assert _should_revise(state) == EDGE_REVISE


def test_should_revise_uses_section_attempts_not_global_verify_cycles():
    """A resumable section must not lose retries to unrelated global cycles."""
    from paperhunter.agents.pipeline import _should_revise, EDGE_REVISE
    from paperhunter.config import settings

    state = _create_mock_state(
        include_outline=True,
        include_sections=True,
        needs_revision=True,
    )
    state["iteration_count"] = settings.agents.max_writer_iterations
    state["sections"]["prompt_injection"]["iteration"] = 2

    assert _should_revise(state) == EDGE_REVISE


def test_should_revise_no_sections():
    """Un outline senza sezioni scritte deve ritornare al Writer."""
    from paperhunter.agents.pipeline import _should_revise, EDGE_REVISE

    state = _create_mock_state(include_outline=True, include_sections=False)
    state["iteration_count"] = 0

    result = _should_revise(state)
    assert result == EDGE_REVISE, f"Expected REVISE (missing sections), got {result}"
    print("[PASS] test_should_revise_no_sections — sezioni mancanti → REVISE")


# ============================================================================
# Test 3: save_checkpoint / load_checkpoint — round-trip
# ============================================================================

def test_checkpoint_round_trip():
    """Salva e ricarica un checkpoint, verifica integrità."""
    from paperhunter.agents.pipeline import save_checkpoint, load_checkpoint

    state = _create_mock_state(include_outline=True, include_sections=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = Path(tmpdir)
        filepath = save_checkpoint(state, "write", checkpoint_dir=cp_dir)

        assert filepath.exists(), "File checkpoint non creato"
        assert filepath.stat().st_size > 0, "File checkpoint vuoto"

        # Verifica che anche il _latest esiste
        latest = cp_dir / "checkpoint_write_latest.json"
        assert latest.exists(), "File _latest non creato"

        # Ricarica
        node_name, loaded_state = load_checkpoint(filepath)
        assert node_name == "write", f"Nodo atteso 'write', ottenuto '{node_name}'"
        assert loaded_state.get("current_phase") == state.get("current_phase")

        # Verifica che le sezioni siano preservate
        loaded_sections = loaded_state.get("sections", {})
        original_sections = state.get("sections", {})
        assert set(loaded_sections.keys()) == set(original_sections.keys()), \
            "Chiavi sezioni diverse dopo round-trip"

    print("[PASS] test_checkpoint_round_trip — salvataggio/caricamento OK")


# ============================================================================
# Test 4: get_latest_checkpoint
# ============================================================================

def test_get_latest_checkpoint():
    """Verifica ricerca checkpoint più recente."""
    from paperhunter.agents.pipeline import (
        save_checkpoint, get_latest_checkpoint,
    )

    state = _create_mock_state(include_outline=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = Path(tmpdir)

        # Mock settings per usare la tmpdir
        with patch("paperhunter.agents.pipeline.settings") as mock_settings:
            mock_settings.output.checkpoints_dir = str(cp_dir)
            mock_settings.agents.max_writer_iterations = 3
            mock_settings.agents.verification_threshold = 0.80
            mock_settings.agents.planner_model = "pro"
            mock_settings.agents.writer_model = "pro"
            mock_settings.agents.critic_model = "flash"
            mock_settings.agents.assembler_model = "pro"

            # Salva due checkpoint
            save_checkpoint(state, "plan", checkpoint_dir=cp_dir)
            save_checkpoint(state, "write", checkpoint_dir=cp_dir)

            # Cerca latest per nodo specifico
            latest_plan = get_latest_checkpoint("plan")
            assert latest_plan is not None, "Nessun latest per 'plan'"
            assert "plan" in latest_plan.name

            latest_write = get_latest_checkpoint("write")
            assert latest_write is not None, "Nessun latest per 'write'"
            assert "write" in latest_write.name

            # Cerca latest generico (il più recente)
            latest_any = get_latest_checkpoint()
            assert latest_any is not None, "Nessun latest generico"

    print("[PASS] test_get_latest_checkpoint — ricerca checkpoint OK")


# ============================================================================
# Test 5: clean_checkpoints
# ============================================================================

def test_clean_checkpoints():
    """Verifica pulizia checkpoint vecchi."""
    from paperhunter.agents.pipeline import (
        save_checkpoint, clean_checkpoints,
    )

    state = _create_mock_state()

    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = Path(tmpdir)

        with patch("paperhunter.agents.pipeline.settings") as mock_settings:
            mock_settings.output.checkpoints_dir = str(cp_dir)
            mock_settings.agents.max_writer_iterations = 3

            # Salva 3 checkpoint
            save_checkpoint(state, "plan", checkpoint_dir=cp_dir)
            save_checkpoint(state, "write", checkpoint_dir=cp_dir)
            save_checkpoint(state, "verify", checkpoint_dir=cp_dir)

            # File totali: 3 timestamped + 3 _latest = 6
            all_files = list(cp_dir.glob("checkpoint_*.json"))
            assert len(all_files) == 6, f"Attesi 6 file, trovati {len(all_files)}"

            # Pulisci mantenendo _latest
            removed = clean_checkpoints(keep_latest=True)
            assert removed == 3, f"Attesi 3 rimossi, rimossi {removed}"

            remaining = list(cp_dir.glob("checkpoint_*.json"))
            assert len(remaining) == 3, f"Attesi 3 rimanenti, trovati {len(remaining)}"
            for f in remaining:
                assert "_latest" in f.name, f"File non-latest rimasto: {f.name}"

    print("[PASS] test_clean_checkpoints — pulizia OK")


# ============================================================================
# Test 6: list_checkpoints
# ============================================================================

def test_list_checkpoints():
    """Verifica elenco checkpoint."""
    from paperhunter.agents.pipeline import (
        save_checkpoint, list_checkpoints,
    )

    state = _create_mock_state()

    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = Path(tmpdir)

        with patch("paperhunter.agents.pipeline.settings") as mock_settings:
            mock_settings.output.checkpoints_dir = str(cp_dir)
            mock_settings.agents.max_writer_iterations = 3

            save_checkpoint(state, "plan", checkpoint_dir=cp_dir)
            save_checkpoint(state, "write", checkpoint_dir=cp_dir)

            checkpoints = list_checkpoints()
            assert len(checkpoints) == 4, f"Attesi 4 checkpoint, trovati {len(checkpoints)}"

            # Verifica struttura
            for cp in checkpoints:
                assert "path" in cp
                assert "node" in cp
                assert "timestamp" in cp
                assert "size_kb" in cp
                assert "is_latest" in cp

            # Verifica che ci siano sia latest che timestamped
            latest_count = sum(1 for cp in checkpoints if cp["is_latest"])
            assert latest_count == 2, f"Attesi 2 latest, trovati {latest_count}"

    print("[PASS] test_list_checkpoints — elenco OK")


# ============================================================================
# Test 7: _make_node_with_checkpoint — wrapper
# ============================================================================

def test_make_node_with_checkpoint():
    """Verifica che il wrapper esegue il nodo e salva checkpoint."""
    from paperhunter.agents.pipeline import _make_node_with_checkpoint

    # Mock node function
    def mock_node(state):
        return {"current_phase": "testing", "metadata": {"mock": True}}

    state = _create_mock_state()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("paperhunter.agents.pipeline.settings") as mock_settings:
            mock_settings.output.checkpoints_dir = tmpdir
            mock_settings.agents.max_writer_iterations = 3

            # Senza checkpoint
            wrapped = _make_node_with_checkpoint(
                mock_node, "test_node", save_checkpoints=False
            )
            result = wrapped(state)
            assert result["current_phase"] == "testing"
            assert result["metadata"]["mock"] is True

            # Con checkpoint
            wrapped_cp = _make_node_with_checkpoint(
                mock_node, "test_node", save_checkpoints=True
            )
            result2 = wrapped_cp(state)
            assert result2["current_phase"] == "testing"

            # Verifica file checkpoint creato
            cp_files = list(Path(tmpdir).glob("checkpoint_test_node_*.json"))
            assert len(cp_files) >= 1, "Nessun checkpoint creato dal wrapper"

    print("[PASS] test_make_node_with_checkpoint — wrapper OK")


def test_checkpoint_wrapper_preserves_sections_omitted_from_partial_update():
    """Serialized checkpoints must apply the same reducers as LangGraph."""
    from paperhunter.agents.pipeline import _make_node_with_checkpoint, load_checkpoint

    state = _create_mock_state(include_outline=True, include_sections=True)
    updated = dict(state["sections"]["prompt_injection"])
    updated["text"] = "revised prompt-injection section"

    def partial_write(_state):
        return {
            "sections": {"prompt_injection": updated},
            "current_phase": "writing",
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("paperhunter.agents.pipeline.settings") as mock_settings:
            mock_settings.output.checkpoints_dir = tmpdir
            mock_settings.agents.max_writer_iterations = 3
            wrapped = _make_node_with_checkpoint(
                partial_write,
                "write",
                save_checkpoints=True,
            )
            wrapped(state)

        latest = Path(tmpdir) / "checkpoint_write_latest.json"
        _, checkpoint_state = load_checkpoint(latest)

    assert set(checkpoint_state["sections"]) == set(state["sections"])
    assert (
        checkpoint_state["sections"]["prompt_injection"]["text"]
        == "revised prompt-injection section"
    )


def test_make_node_error_handling():
    """Verifica che errori nel nodo vengono gestiti correttamente."""
    from paperhunter.agents.pipeline import _make_node_with_checkpoint

    def failing_node(state):
        raise ValueError("Test error in node")

    state = _create_mock_state()

    # Per nodi critici (plan), deve ritornare FAILED
    wrapped_plan = _make_node_with_checkpoint(
        failing_node, "plan", save_checkpoints=False
    )
    result = wrapped_plan(state)
    assert result.get("current_phase") == "failed", \
        f"Expected 'failed' for plan node, got {result.get('current_phase')}"
    assert any("Test error" in e for e in result.get("errors", []))

    # Per nodi non-critici (write), deve ritornare errore ma non FAILED
    wrapped_write = _make_node_with_checkpoint(
        failing_node, "write", save_checkpoints=False
    )
    result2 = wrapped_write(state)
    assert "current_phase" not in result2, \
        f"Write node should not set current_phase on error"
    assert any("Test error" in e for e in result2.get("errors", []))

    print("[PASS] test_make_node_error_handling — gestione errori OK")


# ============================================================================
# Test 9: _make_node_with_checkpoint — progress callback
# ============================================================================

def test_node_progress_callback():
    """Verifica che il callback di progress viene chiamato."""
    from paperhunter.agents.pipeline import _make_node_with_checkpoint

    progress_events = []

    def on_progress(node_name, status):
        progress_events.append((node_name, status))

    def mock_node(state):
        return {"current_phase": "testing"}

    wrapped = _make_node_with_checkpoint(
        mock_node, "test_node",
        save_checkpoints=False,
        on_progress=on_progress,
    )
    wrapped(_create_mock_state())

    assert len(progress_events) == 2, f"Attesi 2 eventi, ottenuti {len(progress_events)}"
    assert progress_events[0] == ("test_node", "started")
    assert progress_events[1] == ("test_node", "completed")

    print("[PASS] test_node_progress_callback — callback OK")


# ============================================================================
# Test 10: _should_revise con iteration_count incrementato dal wrapper
# ============================================================================

def test_verify_node_increments_iteration():
    """Verifica che il wrapper del nodo verify incrementa iteration_count."""
    from paperhunter.agents.pipeline import _make_node_with_checkpoint

    def mock_verify(state):
        return {"current_phase": "verifying"}

    state = _create_mock_state(include_outline=True, include_sections=True)
    state["iteration_count"] = 0

    wrapped = _make_node_with_checkpoint(
        mock_verify, "verify", save_checkpoints=False
    )
    result = wrapped(state)
    assert result.get("iteration_count") == 1, \
        f"Expected iteration_count=1, got {result.get('iteration_count')}"

    print("[PASS] test_verify_node_increments_iteration — incremento OK")


# ============================================================================
# Test 11: run_pipeline con mock completo
# ============================================================================

def test_run_pipeline_mock():
    """Esecuzione completa della pipeline con mock di tutti gli agent."""
    from paperhunter.agents.pipeline import run_pipeline, NODE_PLAN, NODE_WRITE, NODE_VERIFY, NODE_ASSEMBLE
    from paperhunter.agents.state import PipelinePhase, SectionStatus
    from paperhunter.models import L2Summary

    l2 = [
        L2Summary(
            category="Prompt Injection",
            summary_text="Test summary PI.",
            paper_count=5,
            version=1,
        ),
    ]

    # Outline che il Planner "genera"
    mock_outline = [
        {
            "key": "intro", "title": "Introduction", "order": 0,
            "section_type": "introduction", "description": "Intro",
            "target_length_words": 500, "key_topics": [],
            "related_categories": [], "parent_key": None,
            "subsections": [],
        },
        {
            "key": "body", "title": "Body", "order": 1,
            "section_type": "body", "description": "Body",
            "target_length_words": 1000, "key_topics": ["PI"],
            "related_categories": ["Prompt Injection"], "parent_key": None,
            "subsections": [],
        },
    ]

    # Sezioni che il Writer "scrive"
    mock_sections = {
        "intro": {
            "section_key": "intro", "title": "Introduction",
            "text": "This is the intro. " * 30,
            "citations": [{"bibtex_key": "Test_2024_A", "claim": "claim", "chunk_id": None, "retrieval_score": 0.9}],
            "word_count": 150, "status": "drafted", "iteration": 1,
            "rag_queries": [], "retrieved_chunk_ids": [],
            "writer_notes": "", "critic_feedback": "", "drafted_at": None,
        },
        "body": {
            "section_key": "body", "title": "Body",
            "text": "This is the body section. " * 50,
            "citations": [{"bibtex_key": "Test_2024_B", "claim": "claim2", "chunk_id": None, "retrieval_score": 0.8}],
            "word_count": 250, "status": "drafted", "iteration": 1,
            "rag_queries": [], "retrieved_chunk_ids": [],
            "writer_notes": "", "critic_feedback": "", "drafted_at": None,
        },
    }

    # Verifica che il Critic "approva" tutto
    mock_verifications = {}
    mock_verified_sections = {}
    for key in mock_sections:
        mock_verifications[key] = {
            "section_key": key, "total_citations": 1, "supported": 1,
            "partially_supported": 0, "not_supported": 0, "source_not_found": 0,
            "details": [], "pass_rate": 1.0, "passed": True, "iteration": 1,
            "verified_at": None,
        }
        sec = dict(mock_sections[key])
        sec["status"] = "verified"
        mock_verified_sections[key] = sec

    # Mock dei singoli agent node functions
    def mock_plan(state):
        return {
            "outline": mock_outline,
            "current_phase": PipelinePhase.PLANNING.value,
            "errors": [],
            "metadata": {"planner_source": "mock"},
        }

    def mock_write(state):
        return {
            "sections": mock_sections,
            "current_phase": PipelinePhase.WRITING.value,
            "errors": [],
            "metadata": {"writer_sections_written": 2},
        }

    def mock_verify(state):
        return {
            "verification_results": mock_verifications,
            "sections": mock_verified_sections,
            "provenance": [],
            "current_phase": PipelinePhase.VERIFYING.value,
            "errors": [],
            "metadata": {},
        }

    def mock_assemble(state):
        return {
            "final_paper": "# Survey\n\nThis is the final paper.",
            "bibliography": ["Test_2024_A", "Test_2024_B"],
            "current_phase": PipelinePhase.ASSEMBLING.value,
            "version_info": {"version_number": "1.0"},
            "errors": [],
            "metadata": {"total_words": 400},
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("paperhunter.agents.pipeline.plan", mock_plan), \
             patch("paperhunter.agents.pipeline.write", mock_write), \
             patch("paperhunter.agents.pipeline.verify", mock_verify), \
             patch("paperhunter.agents.pipeline.assemble", mock_assemble), \
             patch("paperhunter.agents.pipeline.settings") as mock_settings:

            mock_settings.output.checkpoints_dir = tmpdir
            mock_settings.agents.max_writer_iterations = 3
            mock_settings.agents.verification_threshold = 0.80
            mock_settings.agents.planner_model = "pro"
            mock_settings.agents.writer_model = "pro"
            mock_settings.agents.critic_model = "flash"
            mock_settings.agents.assembler_model = "pro"

            final = run_pipeline(
                l2_summaries=l2,
                save_checkpoints=True,
            )

            # Verifica risultati
            assert final.get("final_paper"), "Nessun paper finale generato"
            assert "Survey" in final["final_paper"]
            assert len(final.get("bibliography", [])) == 2
            assert final.get("current_phase") in (
                PipelinePhase.ASSEMBLING.value,
                PipelinePhase.COMPLETED.value,
            )

            # Verifica checkpoint salvati
            cp_files = list(Path(tmpdir).glob("checkpoint_*.json"))
            assert len(cp_files) >= 4, \
                f"Attesi almeno 4 checkpoint (plan+write+verify+assemble), trovati {len(cp_files)}"

    print("[PASS] test_run_pipeline_mock — pipeline completa con mock OK")


# ============================================================================
# Test 12: run_pipeline con write↔verify loop
# ============================================================================

def test_run_pipeline_with_revision_loop():
    """Pipeline con loop: critic boccia una sezione, writer la riscrive."""
    from paperhunter.agents.pipeline import run_pipeline
    from paperhunter.agents.state import PipelinePhase
    from paperhunter.models import L2Summary

    l2 = [L2Summary(category="Test", summary_text="Test.", paper_count=1, version=1)]

    call_count = {"write": 0, "verify": 0}

    mock_outline = [
        {"key": "s1", "title": "Section 1", "order": 0, "section_type": "body",
         "description": "S1", "target_length_words": 500, "key_topics": [],
         "related_categories": [], "parent_key": None, "subsections": []},
    ]

    def mock_plan(state):
        return {
            "outline": mock_outline,
            "current_phase": PipelinePhase.PLANNING.value,
            "errors": [],
            "metadata": {},
        }

    def mock_write(state):
        call_count["write"] += 1
        return {
            "sections": {
                "s1": {
                    "section_key": "s1", "title": "Section 1",
                    "text": f"Draft iteration {call_count['write']}. " * 30,
                    "citations": [{"bibtex_key": "A_2024_X", "claim": "c", "chunk_id": None, "retrieval_score": 0.9}],
                    "word_count": 150, "status": "drafted",
                    "iteration": call_count["write"],
                    "rag_queries": [], "retrieved_chunk_ids": [],
                    "writer_notes": "", "critic_feedback": "", "drafted_at": None,
                },
            },
            "current_phase": PipelinePhase.WRITING.value,
            "errors": [],
            "metadata": {},
        }

    def mock_verify(state):
        call_count["verify"] += 1
        # Prima volta: boccia. Seconda volta: approva.
        if call_count["verify"] == 1:
            return {
                "verification_results": {
                    "s1": {
                        "section_key": "s1", "total_citations": 1,
                        "supported": 0, "partially_supported": 0,
                        "not_supported": 1, "source_not_found": 0,
                        "details": [], "pass_rate": 0.0, "passed": False,
                        "iteration": 1, "verified_at": None,
                    },
                },
                "sections": {
                    "s1": {
                        "section_key": "s1", "title": "Section 1",
                        "text": "Draft 1. " * 30,
                        "citations": [{"bibtex_key": "A_2024_X", "claim": "c", "chunk_id": None, "retrieval_score": 0.9}],
                        "word_count": 150, "status": "needs_revision",
                        "iteration": 1,
                        "rag_queries": [], "retrieved_chunk_ids": [],
                        "writer_notes": "", "critic_feedback": "Citation not supported",
                        "drafted_at": None,
                    },
                },
                "provenance": [],
                "current_phase": PipelinePhase.VERIFYING.value,
                "errors": [],
                "metadata": {},
            }
        else:
            return {
                "verification_results": {
                    "s1": {
                        "section_key": "s1", "total_citations": 1,
                        "supported": 1, "partially_supported": 0,
                        "not_supported": 0, "source_not_found": 0,
                        "details": [], "pass_rate": 1.0, "passed": True,
                        "iteration": 2, "verified_at": None,
                    },
                },
                "sections": {
                    "s1": {
                        "section_key": "s1", "title": "Section 1",
                        "text": "Draft 2 revised. " * 30,
                        "citations": [{"bibtex_key": "A_2024_X", "claim": "c", "chunk_id": None, "retrieval_score": 0.9}],
                        "word_count": 150, "status": "verified",
                        "iteration": 2,
                        "rag_queries": [], "retrieved_chunk_ids": [],
                        "writer_notes": "", "critic_feedback": "",
                        "drafted_at": None,
                    },
                },
                "provenance": [],
                "current_phase": PipelinePhase.VERIFYING.value,
                "errors": [],
                "metadata": {},
            }

    def mock_assemble(state):
        return {
            "final_paper": "# Final\n\nRevised paper.",
            "bibliography": ["A_2024_X"],
            "current_phase": PipelinePhase.ASSEMBLING.value,
            "version_info": {"version_number": "1.0"},
            "errors": [],
            "metadata": {"total_words": 150},
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("paperhunter.agents.pipeline.plan", mock_plan), \
             patch("paperhunter.agents.pipeline.write", mock_write), \
             patch("paperhunter.agents.pipeline.verify", mock_verify), \
             patch("paperhunter.agents.pipeline.assemble", mock_assemble), \
             patch("paperhunter.agents.pipeline.settings") as mock_settings:

            mock_settings.output.checkpoints_dir = tmpdir
            mock_settings.agents.max_writer_iterations = 3
            mock_settings.agents.verification_threshold = 0.80
            mock_settings.agents.planner_model = "pro"
            mock_settings.agents.writer_model = "pro"
            mock_settings.agents.critic_model = "flash"
            mock_settings.agents.assembler_model = "pro"

            final = run_pipeline(l2_summaries=l2, save_checkpoints=False)

            # Writer chiamato 2 volte (prima bozza + revisione)
            assert call_count["write"] == 2, \
                f"Writer atteso 2 volte, chiamato {call_count['write']}"
            # Verify chiamato 2 volte
            assert call_count["verify"] == 2, \
                f"Verify atteso 2 volte, chiamato {call_count['verify']}"

            assert final.get("final_paper"), "Nessun paper finale"

    print("[PASS] test_run_pipeline_with_revision_loop — loop write↔verify OK")


# ============================================================================
# Test 13: _resume_from_node
# ============================================================================

def test_resume_from_node():
    """Verifica resume manuale da un nodo specifico."""
    from paperhunter.agents.pipeline import _resume_from_node
    from paperhunter.agents.state import PipelinePhase

    call_log = []

    def mock_write(state):
        call_log.append("write")
        return {
            "sections": {"s1": {"section_key": "s1", "status": "drafted", "title": "S1",
                                "text": "t", "citations": [], "word_count": 1,
                                "iteration": 1, "rag_queries": [], "retrieved_chunk_ids": [],
                                "writer_notes": "", "critic_feedback": "", "drafted_at": None}},
            "current_phase": PipelinePhase.WRITING.value,
            "errors": [],
            "metadata": {},
        }

    def mock_verify(state):
        call_log.append("verify")
        return {
            "verification_results": {
                "s1": {"section_key": "s1", "total_citations": 0, "supported": 0,
                       "partially_supported": 0, "not_supported": 0, "source_not_found": 0,
                       "details": [], "pass_rate": 1.0, "passed": True, "iteration": 1,
                       "verified_at": None}
            },
            "sections": {"s1": {"section_key": "s1", "status": "verified", "title": "S1",
                                "text": "t", "citations": [], "word_count": 1,
                                "iteration": 1, "rag_queries": [], "retrieved_chunk_ids": [],
                                "writer_notes": "", "critic_feedback": "", "drafted_at": None}},
            "provenance": [],
            "current_phase": PipelinePhase.VERIFYING.value,
            "errors": [],
            "metadata": {},
        }

    def mock_assemble(state):
        call_log.append("assemble")
        return {
            "final_paper": "# Paper",
            "bibliography": [],
            "current_phase": PipelinePhase.ASSEMBLING.value,
            "version_info": {},
            "errors": [],
            "metadata": {"total_words": 1},
        }

    state = _create_mock_state(include_outline=True)
    # This unit test exercises resume ordering, not missing-section recovery.
    state["outline"] = [{
        "key": "s1", "title": "S1", "order": 1, "section_type": "body",
        "description": "Only section", "key_topics": [],
        "target_length_words": 500, "related_categories": [], "parent_key": None,
    }]

    with patch("paperhunter.agents.pipeline.plan") as _, \
         patch("paperhunter.agents.pipeline.write", mock_write), \
         patch("paperhunter.agents.pipeline.verify", mock_verify), \
         patch("paperhunter.agents.pipeline.assemble", mock_assemble), \
         patch("paperhunter.agents.pipeline.settings") as mock_settings:

        mock_settings.output.checkpoints_dir = "/tmp/test_resume"
        mock_settings.agents.max_writer_iterations = 3
        mock_settings.agents.verification_threshold = 0.80
        mock_settings.agents.planner_model = "pro"
        mock_settings.agents.writer_model = "pro"
        mock_settings.agents.critic_model = "flash"
        mock_settings.agents.assembler_model = "pro"

        # Resume da 'write' → deve eseguire write, verify, assemble (no plan)
        final = _resume_from_node(state, "write", save_checkpoints=False)

        assert call_log == ["write", "verify", "assemble"], \
            f"Sequenza attesa [write, verify, assemble], ottenuta {call_log}"
        assert final.get("final_paper") == "# Paper"

    print("[PASS] test_resume_from_node — resume corretto")


def test_resume_manual_merge_preserves_untouched_verified_sections():
    """Manual resume must honor SurveyState reducers between partial nodes."""
    from paperhunter.agents.pipeline import _resume_from_node
    from paperhunter.agents.state import PipelinePhase, SectionStatus

    state = _create_mock_state(
        include_outline=True,
        include_sections=True,
        all_verified=True,
    )
    revised = dict(state["sections"]["prompt_injection"])
    revised["status"] = SectionStatus.NEEDS_REVISION.value
    revised["critic_feedback"] = "Repair evidence."
    state["sections"]["prompt_injection"] = revised
    seen_by_assembler: set[str] = set()

    def mock_write(_state):
        update = dict(revised)
        update["status"] = SectionStatus.DRAFTED.value
        return {
            "sections": {"prompt_injection": update},
            "current_phase": PipelinePhase.WRITING.value,
            "errors": [],
        }

    def mock_verify(_state):
        update = dict(revised)
        update["status"] = SectionStatus.VERIFIED.value
        return {
            "sections": {"prompt_injection": update},
            "verification_results": {},
            "current_phase": PipelinePhase.VERIFYING.value,
            "errors": [],
        }

    def mock_assemble(assembled_state):
        seen_by_assembler.update(assembled_state["sections"])
        return {
            "final_paper": "# Paper",
            "current_phase": PipelinePhase.ASSEMBLING.value,
        }

    with patch("paperhunter.agents.pipeline.write", mock_write), \
         patch("paperhunter.agents.pipeline.verify", mock_verify), \
         patch("paperhunter.agents.pipeline.assemble", mock_assemble), \
         patch("paperhunter.agents.pipeline.settings") as mock_settings:
        mock_settings.agents.max_writer_iterations = 1
        final = _resume_from_node(state, "write", save_checkpoints=False)

    assert set(final["sections"]) == set(state["sections"])
    assert seen_by_assembler == set(state["sections"])


# ============================================================================
# Test 14: run_pipeline_from_checkpoint
# ============================================================================

def test_run_pipeline_from_checkpoint():
    """Verifica resume completo da checkpoint file."""
    from paperhunter.agents.pipeline import (
        save_checkpoint, run_pipeline_from_checkpoint,
    )
    from paperhunter.agents.state import PipelinePhase

    # Crea stato come se il planner avesse completato
    state = _create_mock_state(include_outline=True, include_l2=True)
    state["current_phase"] = PipelinePhase.PLANNING.value

    def mock_write(state):
        return {
            "sections": {"s1": {"section_key": "s1", "status": "drafted", "title": "S1",
                                "text": "text " * 50, "citations": [], "word_count": 50,
                                "iteration": 1, "rag_queries": [], "retrieved_chunk_ids": [],
                                "writer_notes": "", "critic_feedback": "", "drafted_at": None}},
            "current_phase": PipelinePhase.WRITING.value,
            "errors": [],
            "metadata": {},
        }

    def mock_verify(state):
        return {
            "verification_results": {
                "s1": {"section_key": "s1", "total_citations": 0, "supported": 0,
                       "partially_supported": 0, "not_supported": 0, "source_not_found": 0,
                       "details": [], "pass_rate": 1.0, "passed": True, "iteration": 1,
                       "verified_at": None}
            },
            "sections": {"s1": {"section_key": "s1", "status": "verified", "title": "S1",
                                "text": "text " * 50, "citations": [], "word_count": 50,
                                "iteration": 1, "rag_queries": [], "retrieved_chunk_ids": [],
                                "writer_notes": "", "critic_feedback": "", "drafted_at": None}},
            "provenance": [],
            "current_phase": PipelinePhase.VERIFYING.value,
            "errors": [],
            "metadata": {},
        }

    def mock_assemble(state):
        return {
            "final_paper": "# Resumed Paper",
            "bibliography": [],
            "current_phase": PipelinePhase.ASSEMBLING.value,
            "version_info": {},
            "errors": [],
            "metadata": {"total_words": 50},
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        cp_dir = Path(tmpdir)

        # Salva checkpoint "after plan"
        cp_path = save_checkpoint(state, "plan", checkpoint_dir=cp_dir)

        with patch("paperhunter.agents.pipeline.plan") as mock_plan_not_called, \
             patch("paperhunter.agents.pipeline.write", mock_write), \
             patch("paperhunter.agents.pipeline.verify", mock_verify), \
             patch("paperhunter.agents.pipeline.assemble", mock_assemble), \
             patch("paperhunter.agents.pipeline.settings") as mock_settings:

            mock_settings.output.checkpoints_dir = str(cp_dir)
            mock_settings.agents.max_writer_iterations = 3
            mock_settings.agents.verification_threshold = 0.80
            mock_settings.agents.planner_model = "pro"
            mock_settings.agents.writer_model = "pro"
            mock_settings.agents.critic_model = "flash"
            mock_settings.agents.assembler_model = "pro"

            final = run_pipeline_from_checkpoint(
                cp_path,
                save_checkpoints=False,
            )

            assert final.get("final_paper") == "# Resumed Paper"
            # Plan NON deve essere stato chiamato (resume da dopo plan)
            mock_plan_not_called.assert_not_called()

    print("[PASS] test_run_pipeline_from_checkpoint — resume da file OK")


# ============================================================================
# Test 15: get_pipeline_graph_description
# ============================================================================

def test_get_pipeline_graph_description():
    """Verifica descrizione testuale del grafo."""
    from paperhunter.agents.pipeline import get_pipeline_graph_description

    desc = get_pipeline_graph_description()
    assert "START" in desc
    assert "plan" in desc
    assert "write" in desc
    assert "verify" in desc
    assert "assemble" in desc
    assert "END" in desc
    print(f"\n{desc}\n")
    print("[PASS] test_get_pipeline_graph_description — descrizione OK")


# ============================================================================
# Test 16: _print_pipeline_report (no assertion, just coverage)
# ============================================================================

def test_print_pipeline_report():
    """Verifica che il report non crasha con vari stati."""
    from paperhunter.agents.pipeline import _print_pipeline_report

    # Stato con paper generato
    state = _create_mock_state(
        include_outline=True, include_sections=True,
        include_verifications=True, all_verified=True,
    )
    state["final_paper"] = "# Survey Paper\n\nContent here."
    state["bibliography"] = ["Zou_2023_GCG", "Li_2023_Halueval"]
    _print_pipeline_report(state, 120.5)

    # Stato fallito
    failed_state = _create_mock_state()
    failed_state["current_phase"] = "failed"
    failed_state["errors"] = ["Error 1", "Error 2"]
    _print_pipeline_report(failed_state, 5.0)

    # Stato vuoto
    empty_state = _create_mock_state(include_l2=False)
    _print_pipeline_report(empty_state, 0.1)

    print("[PASS] test_print_pipeline_report — report senza crash")


# ============================================================================
# Test 17: Pipeline con errore nel plan (nodo critico)
# ============================================================================

def test_pipeline_plan_failure():
    """Se il Planner fallisce, la pipeline deve gestire l'errore."""
    from paperhunter.agents.pipeline import run_pipeline
    from paperhunter.agents.state import PipelinePhase
    from paperhunter.models import L2Summary

    l2 = [L2Summary(category="Test", summary_text="T.", paper_count=1, version=1)]

    def mock_plan_fail(state):
        raise RuntimeError("LLM API unavailable")

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("paperhunter.agents.pipeline.plan", mock_plan_fail), \
             patch("paperhunter.agents.pipeline.settings") as mock_settings:

            mock_settings.output.checkpoints_dir = tmpdir
            mock_settings.agents.max_writer_iterations = 3
            mock_settings.agents.verification_threshold = 0.80
            mock_settings.agents.planner_model = "pro"
            mock_settings.agents.writer_model = "pro"
            mock_settings.agents.critic_model = "flash"
            mock_settings.agents.assembler_model = "pro"

            final = run_pipeline(l2_summaries=l2, save_checkpoints=False)

            # Deve avere errori ma non crashare
            errors = final.get("errors", [])
            assert len(errors) > 0, "Nessun errore registrato"
            assert final.get("current_phase") == PipelinePhase.FAILED.value

    print("[PASS] test_pipeline_plan_failure — fallimento plan gestito")


# ============================================================================
# Test 18: Checkpoint file non trovato
# ============================================================================

def test_load_checkpoint_not_found():
    """Verifica errore su checkpoint inesistente."""
    from paperhunter.agents.pipeline import load_checkpoint

    try:
        load_checkpoint("/tmp/nonexistent_checkpoint_12345.json")
        assert False, "Dovrebbe aver lanciato FileNotFoundError"
    except FileNotFoundError:
        pass

    print("[PASS] test_load_checkpoint_not_found — errore gestito")


# ============================================================================
# Main
# ============================================================================

def main():
    """Esegue tutti i test in sequenza."""
    tests = [
        ("1. build_pipeline", test_build_pipeline),
        ("2. should_revise — no revision", test_should_revise_no_revisions_needed),
        ("3. should_revise — needs revision", test_should_revise_needs_revision),
        ("4. should_revise — max iterations", test_should_revise_max_iterations),
        ("5. should_revise — no sections", test_should_revise_no_sections),
        ("6. checkpoint round-trip", test_checkpoint_round_trip),
        ("7. get_latest_checkpoint", test_get_latest_checkpoint),
        ("8. clean_checkpoints", test_clean_checkpoints),
        ("9. list_checkpoints", test_list_checkpoints),
        ("10. make_node_with_checkpoint", test_make_node_with_checkpoint),
        ("11. node error handling", test_make_node_error_handling),
        ("12. node progress callback", test_node_progress_callback),
        ("13. verify increments iteration", test_verify_node_increments_iteration),
        ("14. run_pipeline mock", test_run_pipeline_mock),
        ("15. run_pipeline revision loop", test_run_pipeline_with_revision_loop),
        ("16. _resume_from_node", test_resume_from_node),
        ("17. run_pipeline_from_checkpoint", test_run_pipeline_from_checkpoint),
        ("18. get_pipeline_graph_description", test_get_pipeline_graph_description),
        ("19. _print_pipeline_report", test_print_pipeline_report),
        ("20. pipeline plan failure", test_pipeline_plan_failure),
        ("21. load_checkpoint not found", test_load_checkpoint_not_found),
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("  TEST: agents/pipeline.py — LangGraph Workflow Orchestration")
    print("=" * 70)
    print()

    for name, test_fn in tests:
        try:
            print(f"--- Test {name} ---")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 70)
    print(f"  RISULTATI: {passed}/{passed + failed} passati, {failed} falliti")
    print("=" * 70)

    if errors:
        print("\nTest falliti:")
        for name, err in errors:
            print(f"  ✗ {name}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
