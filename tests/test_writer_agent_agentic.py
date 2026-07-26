"""
Test suite per la modalità agentic (ReAct) del Writer Agent.

Copre:
1. _write_section_agentic con mock ReActAgent
2. _build_section_result helper
3. _run_react_agent async
4. Routing in write_section: modalità static vs agentic
5. Fallback da agentic a static quando il ReAct fallisce
6. Integrazione con _REACT_SYSTEM_PROMPT template
7. Logging modalità nel nodo write()
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from paperhunter.agents.state import (
    CitationRecord,
    PipelinePhase,
    SectionDraft,
    SectionOutline,
    SectionStatus,
    SurveyState,
    create_initial_state,
)
from paperhunter.config import settings

# ============================================================================
# Fixtures
# ============================================================================

MOCK_SECTION_TEXT = """\
## Prompt Injection Attacks

Prompt injection attacks represent a critical vulnerability in large language \
models (LLMs). Early work by [Perez_2022_PromptInject] demonstrated that \
carefully crafted inputs can override system-level instructions in conversational \
AI systems. Building on this, [Greshake_2023_Indirect] introduced the concept \
of indirect prompt injection, where malicious instructions are embedded in \
external data sources retrieved by the model.

Recent advances include the GCG attack [Zou_2023_GCG], which generates universal \
adversarial suffixes that transfer across different LLM families. In parallel, \
[Liu_2024_AutoDAN] proposed an automatic jailbreak method using hierarchical \
genetic algorithms, achieving high attack success rates against GPT-4 and \
Claude-2.

Defense mechanisms have also evolved. [Jain_2023_Defense] proposed perplexity-based \
filtering to detect adversarial prompts, while [Alon_2023_Sandwich] introduced \
the sandwich defense that wraps user input between system instruction reminders.
"""

MOCK_OUTLINE = SectionOutline(
    key="prompt_injection_attacks",
    title="Prompt Injection Attacks",
    description="Survey of prompt injection techniques and defenses for LLMs",
    target_length_words=1500,
    key_topics=[
        "Direct prompt injection",
        "Indirect prompt injection",
        "Adversarial suffixes",
        "Defense mechanisms",
    ],
    related_categories=["Prompt Injection", "Adversarial Code"],
    order=2,
    section_type="body",
)

MOCK_TESTING_OUTLINE = SectionOutline(
    key="testing_methodologies",
    title="Testing Methodologies",
    description="Comprehensive overview of testing approaches for LLM security",
    target_length_words=2000,
    key_topics=["Benchmarks", "Metrics", "Evaluation"],
    related_categories=["Prompt Injection"],
    order=5,
    section_type="testing",
)


def _make_state_with_outline(outline_list=None):
    """Crea uno stato di test con un outline."""
    if outline_list is None:
        outline_list = [MOCK_OUTLINE]

    state = create_initial_state()
    state["outline"] = [o.to_dict() for o in outline_list]
    return state


def _make_mock_agent_output(text: str):
    """Crea un mock AgentOutput con una risposta testuale."""
    mock_response = MagicMock()
    mock_response.content = text
    mock_output = MagicMock()
    mock_output.response = mock_response
    return mock_output


# ============================================================================
# TEST 1: _build_section_result helper
# ============================================================================


class TestBuildSectionResult:
    """Test per _build_section_result."""

    def test_basic_build(self):
        """Verifica costruzione base del risultato."""
        from paperhunter.agents.writer_agent import _build_section_result

        parsed = {
            "text": MOCK_SECTION_TEXT,
            "citations": [
                {"bibtex_key": "Perez_2022_PromptInject", "claim": "early work"},
                {"bibtex_key": "Zou_2023_GCG", "claim": "GCG attack"},
            ],
            "notes": "Section completed.",
        }

        result = _build_section_result(
            parsed, MOCK_OUTLINE, "prompt_injection_attacks",
            iteration=1, elapsed=5.0, errors=[],
            mode="static",
        )

        assert "sections" in result
        assert "prompt_injection_attacks" in result["sections"]
        assert result["current_section_key"] == "prompt_injection_attacks"
        assert result["errors"] == []

        draft_data = result["sections"]["prompt_injection_attacks"]
        assert draft_data["status"] == "drafted"
        assert draft_data["iteration"] == 1
        assert draft_data["word_count"] > 0

    def test_agentic_mode_notes_prefix(self):
        """Verifica che il prefisso [agentic] venga aggiunto alle note."""
        from paperhunter.agents.writer_agent import _build_section_result

        parsed = {
            "text": "Some text with [Zou_2023_GCG] citation.",
            "citations": [{"bibtex_key": "Zou_2023_GCG", "claim": "test"}],
            "notes": "ReAct agent completed.",
        }

        result = _build_section_result(
            parsed, MOCK_OUTLINE, "prompt_injection_attacks",
            iteration=1, elapsed=3.0, errors=[],
            mode="agentic",
        )

        draft_data = result["sections"]["prompt_injection_attacks"]
        assert draft_data["writer_notes"].startswith("[agentic]")

    def test_untracked_citations_auto_detected(self):
        """Verifica rilevamento automatico citazioni non tracciate."""
        from paperhunter.agents.writer_agent import _build_section_result

        parsed = {
            "text": "Found in [Perez_2022_PromptInject] and [Zou_2023_GCG].",
            "citations": [
                {"bibtex_key": "Perez_2022_PromptInject", "claim": "test"},
            ],
            "notes": "",
        }

        result = _build_section_result(
            parsed, MOCK_OUTLINE, "test_section",
            iteration=1, elapsed=1.0, errors=[],
        )

        draft_data = result["sections"]["test_section"]
        keys = {c["bibtex_key"] for c in draft_data["citations"]}
        assert "Zou_2023_GCG" in keys
        assert "Perez_2022_PromptInject" in keys

    def test_errors_propagated(self):
        """Verifica che gli errori vengano propagati."""
        from paperhunter.agents.writer_agent import _build_section_result

        parsed = {"text": "Short text.", "citations": [], "notes": ""}
        result = _build_section_result(
            parsed, MOCK_OUTLINE, "test_section",
            iteration=1, elapsed=1.0,
            errors=["some warning"],
        )
        assert result["errors"] == ["some warning"]

    def test_queries_and_chunk_ids_stored(self):
        """Verifica che queries e chunk_ids vengano salvati nel draft."""
        from paperhunter.agents.writer_agent import _build_section_result

        parsed = {"text": "Some text.", "citations": [], "notes": ""}
        result = _build_section_result(
            parsed, MOCK_OUTLINE, "test_section",
            iteration=1, elapsed=1.0, errors=[],
            queries=["query1", "query2"],
            chunk_ids=["chunk_a", "chunk_b"],
        )
        draft_data = result["sections"]["test_section"]
        assert draft_data["rag_queries"] == ["query1", "query2"]
        assert draft_data["retrieved_chunk_ids"] == ["chunk_a", "chunk_b"]


# ============================================================================
# TEST 2: _write_section_agentic con mock
# ============================================================================


class TestWriteSectionAgentic:
    """Test per _write_section_agentic con mock ReActAgent."""

    @patch("llama_index.core.agent.ReActAgent")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent.asyncio.run")
    @patch("paperhunter.agents.tools.retrieval_tools.get_all_retrieval_tools")
    def test_success(self, mock_tools, mock_asyncio_run, mock_get_llm, mock_react):
        """Verifica flusso completo con ReAct Agent mock."""
        from paperhunter.agents.writer_agent import _write_section_agentic

        mock_tools.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_get_llm.return_value = MagicMock()

        mock_output = _make_mock_agent_output(MOCK_SECTION_TEXT)
        mock_asyncio_run.return_value = mock_output

        result = _write_section_agentic(MOCK_OUTLINE, force_model="pro")

        assert result is not None
        assert "text" in result
        assert result["text"] == MOCK_SECTION_TEXT.strip()
        assert len(result["citations"]) > 0
        assert "[agentic]" in result["notes"]

    @patch("llama_index.core.agent.ReActAgent")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent.asyncio.run")
    @patch("paperhunter.agents.tools.retrieval_tools.get_all_retrieval_tools")
    def test_returns_none_on_exception(self, mock_tools, mock_asyncio_run, mock_get_llm, mock_react):
        """Verifica che ritorni None se il ReAct agent lancia eccezione."""
        from paperhunter.agents.writer_agent import _write_section_agentic

        mock_tools.return_value = [MagicMock()]
        mock_get_llm.return_value = MagicMock()
        mock_asyncio_run.side_effect = Exception("LLM timeout")

        result = _write_section_agentic(MOCK_OUTLINE)
        assert result is None

    @patch("llama_index.core.agent.ReActAgent")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent.asyncio.run")
    @patch("paperhunter.agents.tools.retrieval_tools.get_all_retrieval_tools")
    def test_returns_none_on_short_response(self, mock_tools, mock_asyncio_run, mock_get_llm, mock_react):
        """Verifica che ritorni None per risposte troppo corte."""
        from paperhunter.agents.writer_agent import _write_section_agentic

        mock_tools.return_value = [MagicMock()]
        mock_get_llm.return_value = MagicMock()

        mock_output = _make_mock_agent_output("Too short")
        mock_asyncio_run.return_value = mock_output

        result = _write_section_agentic(MOCK_OUTLINE)
        assert result is None

    @patch("llama_index.core.agent.ReActAgent")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent.asyncio.run")
    @patch("paperhunter.agents.tools.retrieval_tools.get_all_retrieval_tools")
    def test_returns_none_when_agent_returns_none(self, mock_tools, mock_asyncio_run, mock_get_llm, mock_react):
        """Verifica che ritorni None se l'agente restituisce None."""
        from paperhunter.agents.writer_agent import _write_section_agentic

        mock_tools.return_value = [MagicMock()]
        mock_get_llm.return_value = MagicMock()
        mock_asyncio_run.return_value = None

        result = _write_section_agentic(MOCK_OUTLINE)
        assert result is None

    @patch("llama_index.core.agent.ReActAgent")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent.asyncio.run")
    @patch("paperhunter.agents.tools.retrieval_tools.get_all_retrieval_tools")
    def test_critic_feedback_included(self, mock_tools, mock_asyncio_run, mock_get_llm, mock_react):
        """Verifica che il critic feedback venga incluso nel system prompt."""
        from paperhunter.agents.writer_agent import _write_section_agentic

        mock_tools.return_value = [MagicMock()]
        mock_get_llm.return_value = MagicMock()

        mock_output = _make_mock_agent_output(MOCK_SECTION_TEXT)
        mock_asyncio_run.return_value = mock_output

        result = _write_section_agentic(
            MOCK_OUTLINE,
            critic_feedback="Need more citations for GCG attack claims",
        )
        assert result is not None

    @patch("llama_index.core.agent.ReActAgent")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent.asyncio.run")
    @patch("paperhunter.agents.tools.retrieval_tools.get_all_retrieval_tools")
    def test_json_response_parsed(self, mock_tools, mock_asyncio_run, mock_get_llm, mock_react):
        """Verifica che risposte JSON vengano parsate correttamente."""
        from paperhunter.agents.writer_agent import _write_section_agentic

        mock_tools.return_value = [MagicMock()]
        mock_get_llm.return_value = MagicMock()

        json_response = json.dumps({
            "text": "A section about [Zou_2023_GCG] adversarial attacks.",
            "citations": [
                {"bibtex_key": "Zou_2023_GCG", "claim": "adversarial attacks"}
            ],
            "notes": "Completed via JSON output.",
        })
        mock_output = _make_mock_agent_output(json_response)
        mock_asyncio_run.return_value = mock_output

        result = _write_section_agentic(MOCK_OUTLINE)
        assert result is not None
        assert "Zou_2023_GCG" in result["text"]
        assert any(c["bibtex_key"] == "Zou_2023_GCG" for c in result["citations"])


# ============================================================================
# TEST 3: _run_react_agent async
# ============================================================================


class TestRunReactAgent:
    """Test per _run_react_agent."""

    def test_success(self):
        """Verifica che _run_react_agent restituisca il risultato correttamente."""
        from paperhunter.agents.writer_agent import _run_react_agent

        mock_output = _make_mock_agent_output("test output")
        mock_handler = AsyncMock(return_value=mock_output)

        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_handler()

        result = asyncio.run(_run_react_agent(mock_agent, "test", max_iterations=3))
        assert result is not None

    def test_returns_none_on_error(self):
        """Verifica che ritorni None su errore."""
        from paperhunter.agents.writer_agent import _run_react_agent

        mock_agent = MagicMock()
        mock_agent.run.side_effect = Exception("agent error")

        result = asyncio.run(_run_react_agent(mock_agent, "test"))
        assert result is None


# ============================================================================
# TEST 4: Routing in write_section (static vs agentic)
# ============================================================================


class TestWriteSectionRouting:
    """Test per il routing tra modalità static e agentic in write_section."""

    @patch("paperhunter.agents.writer_agent.llm_call_with_retry")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent._generate_rag_queries")
    @patch("paperhunter.agents.writer_agent._retrieve_context_for_section")
    def test_static_mode_uses_rag_pipeline(
        self, mock_retrieve, mock_gen_queries, mock_get_llm, mock_llm_call
    ):
        """In modalità static, usa il pipeline RAG tradizionale."""
        from paperhunter.agents.writer_agent import write_section

        mock_gen_queries.return_value = ["query1", "query2"]
        mock_retrieve.return_value = (
            [{"chunk_id": "c1", "text": "Some text", "score": 0.9,
              "metadata": {"bibtex_key": "Test_2024_Key", "category": "Test",
                           "year": 2024, "section_name": "intro",
                           "chunk_type": "original", "is_citable": True}}],
            ["c1"],
        )
        mock_get_llm.return_value = MagicMock()
        mock_llm_call.return_value = json.dumps({
            "text": "A section about LLM security with [Test_2024_Key].",
            "citations": [{"bibtex_key": "Test_2024_Key", "claim": "test"}],
            "notes": "",
        })

        state = _make_state_with_outline()
        original_mode = settings.agents.writer_mode
        try:
            settings.agents.writer_mode = "static"
            with patch.object(settings.agents, "min_citations_per_section", 1):
                result = write_section(state, "prompt_injection_attacks")
        finally:
            settings.agents.writer_mode = original_mode

        assert "sections" in result
        assert "prompt_injection_attacks" in result["sections"]
        # In static mode, RAG queries should have been called
        mock_gen_queries.assert_called_once()
        mock_retrieve.assert_called_once()

    @patch("paperhunter.agents.writer_agent._write_section_agentic")
    def test_agentic_mode_calls_react(self, mock_agentic):
        """In modalità agentic, chiama _write_section_agentic."""
        from paperhunter.agents.writer_agent import write_section

        mock_agentic.return_value = {
            "text": MOCK_SECTION_TEXT,
            "citations": [
                {"bibtex_key": "Zou_2023_GCG", "claim": "test"},
            ],
            "notes": "[agentic] done",
        }

        state = _make_state_with_outline()
        original_mode = settings.agents.writer_mode
        try:
            settings.agents.writer_mode = "agentic"
            result = write_section(state, "prompt_injection_attacks")
        finally:
            settings.agents.writer_mode = original_mode

        assert "sections" in result
        mock_agentic.assert_called_once()

    @patch("paperhunter.agents.writer_agent.llm_call_with_retry")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent._generate_rag_queries")
    @patch("paperhunter.agents.writer_agent._retrieve_context_for_section")
    @patch("paperhunter.agents.writer_agent._write_section_agentic")
    def test_agentic_fallback_to_static(
        self, mock_agentic, mock_retrieve, mock_gen_queries,
        mock_get_llm, mock_llm_call,
    ):
        """Se agentic fallisce (returns None), fallback a static."""
        from paperhunter.agents.writer_agent import write_section

        # Agentic returns None → fallback
        mock_agentic.return_value = None

        # Static path setup
        mock_gen_queries.return_value = ["query1"]
        mock_retrieve.return_value = (
            [{"chunk_id": "c1", "text": "Info", "score": 0.8,
              "metadata": {"bibtex_key": "Key_2024_Test", "category": "Test",
                           "year": 2024, "section_name": "intro",
                           "chunk_type": "original", "is_citable": True}}],
            ["c1"],
        )
        mock_get_llm.return_value = MagicMock()
        mock_llm_call.return_value = json.dumps({
            "text": "Fallback text with [Key_2024_Test].",
            "citations": [{"bibtex_key": "Key_2024_Test", "claim": "test"}],
            "notes": "Fallback.",
        })

        state = _make_state_with_outline()
        original_mode = settings.agents.writer_mode
        try:
            settings.agents.writer_mode = "agentic"
            with patch.object(settings.agents, "min_citations_per_section", 1):
                result = write_section(state, "prompt_injection_attacks")
        finally:
            settings.agents.writer_mode = original_mode

        assert "sections" in result
        # Both agentic and static were called
        mock_agentic.assert_called_once()
        mock_gen_queries.assert_called_once()

    def test_section_not_found(self):
        """Ritorna errore se la sezione non è nell'outline."""
        from paperhunter.agents.writer_agent import write_section

        state = _make_state_with_outline()
        result = write_section(state, "nonexistent_section")
        assert "errors" in result
        assert any("non trovata" in e for e in result["errors"])

    def test_already_verified_section_skipped(self):
        """Sezioni già verificate vengono saltate."""
        from paperhunter.agents.writer_agent import write_section

        state = _make_state_with_outline()
        state["sections"] = {
            "prompt_injection_attacks": SectionDraft(
                section_key="prompt_injection_attacks",
                title="Prompt Injection Attacks",
                text="Already written",
                status=SectionStatus.VERIFIED,
                iteration=1,
            ).to_dict(),
        }
        result = write_section(state, "prompt_injection_attacks")
        assert result == {}


# ============================================================================
# TEST 5: REACT_SYSTEM_PROMPT template
# ============================================================================


class TestReactSystemPrompt:
    """Test per il template del system prompt ReAct."""

    def test_prompt_formatting(self):
        """Verifica che il prompt template si formatti correttamente."""
        from paperhunter.agents.writer_agent import _REACT_SYSTEM_PROMPT

        formatted = _REACT_SYSTEM_PROMPT.format(
            title="Test Section",
            description="A test section description",
            key_topics="Topic A, Topic B",
            related_categories="Category 1, Category 2",
            target_words=1500,
            special_instructions="",
            critic_feedback_section="",
        )

        assert "Test Section" in formatted
        assert "Topic A, Topic B" in formatted
        assert "Category 1, Category 2" in formatted
        assert "1500" in formatted
        assert "[bibtex_key]" in formatted
        assert "Search First" in formatted

    def test_prompt_with_critic_feedback(self):
        """Verifica inclusione del critic feedback nel prompt."""
        from paperhunter.agents.writer_agent import _REACT_SYSTEM_PROMPT

        critic_section = (
            "## CRITIC FEEDBACK\n\n"
            "Need more citations for defense mechanisms.\n"
        )
        formatted = _REACT_SYSTEM_PROMPT.format(
            title="Test",
            description="desc",
            key_topics="A",
            related_categories="B",
            target_words=500,
            special_instructions="",
            critic_feedback_section=critic_section,
        )
        assert "CRITIC FEEDBACK" in formatted
        assert "defense mechanisms" in formatted

    def test_prompt_with_special_instructions(self):
        """Verifica inclusione di istruzioni speciali (es. testing)."""
        from paperhunter.agents.writer_agent import (
            _REACT_SYSTEM_PROMPT,
            _get_special_instructions,
        )

        special = _get_special_instructions("testing")
        formatted = _REACT_SYSTEM_PROMPT.format(
            title="Testing Methodologies",
            description="Testing overview",
            key_topics="Benchmarks",
            related_categories="all",
            target_words=2000,
            special_instructions=special,
            critic_feedback_section="",
        )
        assert "Comprehensive Table" in formatted


# ============================================================================
# TEST 6: Config writer_mode validation
# ============================================================================


class TestWriterModeConfig:
    """Test per la validazione del writer_mode in config."""

    def test_valid_modes(self):
        """Verifica che 'static' e 'agentic' siano accettati."""
        from paperhunter.config import AgentsConfig

        cfg = AgentsConfig(writer_mode="static")
        assert cfg.writer_mode == "static"

        cfg = AgentsConfig(writer_mode="agentic")
        assert cfg.writer_mode == "agentic"

    def test_invalid_mode_rejected(self):
        """Verifica che modalità invalide vengano rifiutate."""
        from pydantic import ValidationError
        from paperhunter.config import AgentsConfig

        with pytest.raises(ValidationError):
            AgentsConfig(writer_mode="invalid_mode")


# ============================================================================
# TEST 7: write_all_sections rispetta la modalità
# ============================================================================


class TestWriteAllSectionsAgentic:
    """Test per write_all_sections con routing modale."""

    @patch("paperhunter.agents.writer_agent._load_section_checkpoints", return_value={})
    @patch("paperhunter.agents.writer_agent.write_section")
    def test_write_all_delegates_to_write_section(self, mock_write_section, mock_checkpoints):
        """write_all_sections delega a write_section che gestisce il routing."""
        from paperhunter.agents.writer_agent import write_all_sections

        mock_write_section.return_value = {
            "sections": {
                "prompt_injection_attacks": {
                    "section_key": "prompt_injection_attacks",
                    "title": "Prompt Injection Attacks",
                    "text": "Text",
                    "citations": [],
                    "word_count": 100,
                    "status": "drafted",
                    "iteration": 1,
                    "rag_queries": [],
                    "retrieved_chunk_ids": [],
                    "writer_notes": "",
                    "critic_feedback": "",
                    "drafted_at": datetime.now().isoformat(),
                },
            },
            "errors": [],
        }

        state = _make_state_with_outline()
        result = write_all_sections(state)

        assert "sections" in result
        mock_write_section.assert_called_once()


# ============================================================================
# TEST 8: Integrazione end-to-end agentic (mock completo)
# ============================================================================


class TestAgenticE2E:
    """Test end-to-end del flusso agentic con tutti i mock."""

    @patch("llama_index.core.agent.ReActAgent")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent.asyncio.run")
    @patch("paperhunter.agents.tools.retrieval_tools.get_all_retrieval_tools")
    def test_full_agentic_flow(self, mock_tools, mock_asyncio_run, mock_get_llm, mock_react):
        """Flusso completo: config agentic → ReAct → SectionDraft."""
        from paperhunter.agents.writer_agent import write_section

        # Setup mocks
        mock_tools.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_get_llm.return_value = MagicMock()

        section_text = (
            "Prompt injection attacks have been extensively studied. "
            "Initial work [Perez_2022_PromptInject] showed that LLMs are "
            "vulnerable to instruction-following attacks. The GCG attack "
            "[Zou_2023_GCG] demonstrated transferability across models. "
            "Defense mechanisms include perplexity filtering [Jain_2023_Defense] "
            "and sandwich defense [Alon_2023_Sandwich]. "
            "These approaches represent significant progress in LLM security."
        )
        mock_output = _make_mock_agent_output(section_text)
        mock_asyncio_run.return_value = mock_output

        state = _make_state_with_outline()
        original_mode = settings.agents.writer_mode
        try:
            settings.agents.writer_mode = "agentic"
            with patch(
                "paperhunter.agents.writer_agent._load_global_citable_keys",
                return_value={
                    "Perez_2022_PromptInject",
                    "Zou_2023_GCG",
                    "Jain_2023_Defense",
                    "Alon_2023_Sandwich",
                },
            ):
                result = write_section(state, "prompt_injection_attacks")
        finally:
            settings.agents.writer_mode = original_mode

        assert "sections" in result
        draft_data = result["sections"]["prompt_injection_attacks"]
        assert draft_data["status"] == "drafted"
        assert draft_data["word_count"] > 0
        assert "[agentic]" in draft_data["writer_notes"]

        # Verifica citazioni estratte
        keys = {c["bibtex_key"] for c in draft_data["citations"]}
        assert "Perez_2022_PromptInject" in keys
        assert "Zou_2023_GCG" in keys

    @patch("llama_index.core.agent.ReActAgent")
    @patch("paperhunter.agents.writer_agent.get_llm")
    @patch("paperhunter.agents.writer_agent.asyncio.run")
    @patch("paperhunter.agents.tools.retrieval_tools.get_all_retrieval_tools")
    @patch("paperhunter.agents.writer_agent.llm_call_with_retry")
    @patch("paperhunter.agents.writer_agent._generate_rag_queries")
    @patch("paperhunter.agents.writer_agent._retrieve_context_for_section")
    def test_agentic_failure_then_static_success(
        self, mock_retrieve, mock_gen_queries,
        mock_llm_call, mock_tools, mock_asyncio_run, mock_get_llm, mock_react,
    ):
        """Se agentic fallisce, il static path produce il risultato."""
        from paperhunter.agents.writer_agent import write_section

        # Agentic path fallisce
        mock_tools.return_value = [MagicMock()]
        mock_get_llm.return_value = MagicMock()
        mock_asyncio_run.side_effect = Exception("Agent crashed")

        # Static path funziona
        mock_gen_queries.return_value = ["q1"]
        mock_retrieve.return_value = (
            [{"chunk_id": "c1", "text": "content", "score": 0.7,
              "metadata": {"bibtex_key": "Fallback_2024_Test", "category": "Test",
                           "year": 2024, "section_name": "body",
                           "chunk_type": "original", "is_citable": True}}],
            ["c1"],
        )
        mock_llm_call.return_value = json.dumps({
            "text": "Fallback section with [Fallback_2024_Test].",
            "citations": [{"bibtex_key": "Fallback_2024_Test", "claim": "test"}],
            "notes": "Static fallback.",
        })

        state = _make_state_with_outline()
        original_mode = settings.agents.writer_mode
        try:
            settings.agents.writer_mode = "agentic"
            with patch.object(settings.agents, "min_citations_per_section", 1):
                result = write_section(state, "prompt_injection_attacks")
        finally:
            settings.agents.writer_mode = original_mode

        assert "sections" in result
        draft_data = result["sections"]["prompt_injection_attacks"]
        # Il risultato è dal path statico, non ha il prefisso [agentic]
        assert draft_data["status"] == "drafted"


# ============================================================================
# TEST 9: _REACT_MAX_ITERATIONS costante
# ============================================================================


class TestReactConstants:
    """Test per le costanti ReAct."""

    def test_max_iterations_defined(self):
        from paperhunter.agents.writer_agent import _REACT_MAX_ITERATIONS
        assert isinstance(_REACT_MAX_ITERATIONS, int)
        assert _REACT_MAX_ITERATIONS >= 3
        assert _REACT_MAX_ITERATIONS <= 15


# ============================================================================
# TEST 10: write() LangGraph node logga modalità
# ============================================================================


class TestWriteNodeLogging:
    """Test per il logging della modalità nel nodo write()."""

    @patch("paperhunter.agents.writer_agent.write_all_sections")
    @patch("paperhunter.agents.writer_agent.get_outline_from_state")
    def test_write_node_logs_mode(self, mock_outline, mock_write_all, caplog):
        """Verifica che il nodo write() logga la modalità."""
        from paperhunter.agents.writer_agent import write
        import logging

        mock_outline.return_value = [MOCK_OUTLINE]
        mock_write_all.return_value = {
            "sections": {},
            "errors": [],
            "metadata": {},
        }

        state = _make_state_with_outline()
        with caplog.at_level(logging.INFO):
            write(state)

        # Verifica che il log contenga la modalità
        log_text = caplog.text
        assert "Modalità:" in log_text


# ============================================================================
# Esecuzione diretta
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
