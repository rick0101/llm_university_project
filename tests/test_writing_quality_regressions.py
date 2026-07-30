"""Regression tests for the DeepSeek text-only writing quality pipeline.

These tests are intentionally offline.  They cover the failure modes observed in
production: reasoning consuming the whole output budget, evidence-starved writer
contexts, citation occurrences being silently deduplicated, and assembly of an
incomplete or unverified paper.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from paperhunter.agents.state import (
    SectionDraft,
    SectionOutline,
    SectionStatus,
    create_initial_state,
)


def _completion(content: str | None, finish_reason: str, reasoning: str = ""):
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=190),
    )
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _chunk(key: str, chunk_id: str, score: float, *, citable: bool = True, kind: str = "l1"):
    return {
        "chunk_id": chunk_id,
        "text": f"Evidence from {key}",
        "score": score,
        "metadata": {
            "bibtex_key": key,
            "is_citable": citable,
            "chunk_type": kind,
            "summary_level": "L1" if kind == "l1" else "L2",
        },
    }


def test_deepseek_recovers_once_without_thinking_when_reasoning_exhausts_output():
    """A length+empty response gets one bounded non-thinking recovery, not blind retries."""
    from paperhunter.utils.llm_factory import DeepSeekLLM

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion("", "length", reasoning="x" * 500),
        _completion("Recovered visible answer", "stop"),
    ]

    with patch("openai.OpenAI", return_value=client) as openai_cls:
        llm = DeepSeekLLM(
            api_key="test-key",
            model="deepseek-v4-pro",
            max_tokens=4096,
            thinking_enabled=True,
            reasoning_effort="high",
        )
        response = llm.complete("Write a grounded section")

    assert response.text == "Recovered visible answer"
    assert client.chat.completions.create.call_count == 2
    first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
    recovery_kwargs = client.chat.completions.create.call_args_list[1].kwargs
    assert first_kwargs["extra_body"]["thinking"]["type"] == "enabled"
    assert recovery_kwargs["extra_body"]["thinking"]["type"] == "disabled"
    # Transport retries are owned by llm_call_with_retry; do not multiply them in the SDK.
    assert openai_cls.call_args.kwargs["max_retries"] == 0


def test_deepseek_empty_recovery_is_bounded_to_two_paid_calls():
    """An empty recovery is terminal; it must not start a retry cascade."""
    from paperhunter.utils.llm_factory import DeepSeekLLM

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion("", "length", reasoning="x" * 500),
        _completion("", "length", reasoning=""),
    ]
    with patch("openai.OpenAI", return_value=client):
        llm = DeepSeekLLM(api_key="test", model="deepseek-v4-pro", max_tokens=4096)
        with pytest.raises(RuntimeError, match="vuota anche dopo"):
            llm.complete("write")

    assert client.chat.completions.create.call_count == 2


def test_deepseek_provider_retry_cap_overrides_large_caller_request():
    """Legacy stages asking for six retries are capped by DeepSeek config."""
    from paperhunter.config import settings
    from paperhunter.utils.llm_factory import llm_call_with_retry

    llm = MagicMock()
    llm.provider = "deepseek"
    llm.complete.side_effect = RuntimeError("503 service unavailable")
    with (
        patch("paperhunter.utils.llm_factory.time.sleep"),
        patch.object(settings.deepseek, "retry_strategy", "autonomous"),
    ):
        with pytest.raises(RuntimeError, match="dopo 3 tentativi"):
            llm_call_with_retry(
                llm,
                "prompt",
                max_retries=99,
                base_delay=0.01,
                max_delay=0.01,
            )

    assert llm.complete.call_count == 3


def test_deepseek_never_returns_a_still_truncated_partial_section():
    """Bounded continuation must fail closed rather than export a fragment."""
    from paperhunter.utils.llm_factory import DeepSeekLLM

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion("partial", "length"),
        _completion(" more", "length"),
    ]
    with patch("openai.OpenAI", return_value=client):
        llm = DeepSeekLLM(
            api_key="test",
            model="deepseek-v4-pro",
            max_tokens=4096,
            max_continuation_attempts=1,
        )
        with pytest.raises(RuntimeError, match="ancora troncato"):
            llm.complete("write")

    assert client.chat.completions.create.call_count == 2


def test_writer_context_diversification_prefers_distinct_citable_papers():
    """One prolific paper cannot crowd all other evidence out of the writer context."""
    from paperhunter.agents.writer_agent import _diversify_chunks_by_paper

    chunks = [_chunk("A_2024_One", f"a{i}", 1.0 - i * 0.01) for i in range(6)] + [
        _chunk("B_2024_Two", "b1", 0.80),
        _chunk("C_2024_Three", "c1", 0.79),
        _chunk("D_2024_Four", "d1", 0.78),
        _chunk("L2_background", "l2", 0.99, citable=False, kind="l2"),
    ]

    selected = _diversify_chunks_by_paper(chunks, limit=6, max_per_paper=2)
    selected_keys = [c["metadata"]["bibtex_key"] for c in selected]

    assert {"A_2024_One", "B_2024_Two", "C_2024_Three", "D_2024_Four"}.issubset(selected_keys)
    assert selected_keys.count("A_2024_One") <= 2
    assert selected_keys[0] != "L2_background"


def test_writer_whitelist_contains_only_evidence_that_fit_the_prompt_budget():
    """A key without its evidence text must never be authorized for citation."""
    from paperhunter.agents.writer_agent import _format_context_for_prompt

    chunks = [
        {
            "chunk_id": f"c{i}",
            "text": f"Evidence for paper {i}. " + ("x" * 1800),
            "metadata": {
                "bibtex_key": f"Paper_{i}",
                "chunk_type": "L1_SUMMARY",
                "is_citable": True,
            },
        }
        for i in range(8)
    ]
    formatted = _format_context_for_prompt(chunks, max_chars=3000)

    included_keys = {
        key for key in (f"Paper_{i}" for i in range(8)) if f"Paper: [{key}]" in formatted
    }
    whitelist_text = formatted.split("=== END WHITELIST ===", 1)[0]
    for key in (f"Paper_{i}" for i in range(8)):
        assert (key in whitelist_text) is (key in included_keys)
    assert len(formatted) <= 3000


def test_writer_strips_disallowed_citations_from_text_and_records():
    from paperhunter.agents.state import SectionOutline
    from paperhunter.agents.writer_agent import _build_section_result
    from paperhunter.config import settings

    outline = SectionOutline.model_validate(
        {
            "key": "section",
            "title": "Section",
            "section_type": "body",
        }
    )
    parsed = {
        "text": "Supported [Allowed_2024_Source]. Unsupported [Blocked_2023_Source].",
        "citations": [
            {"bibtex_key": "Allowed_2024_Source", "claim": "supported"},
            {"bibtex_key": "Blocked_2023_Source", "claim": "unsupported"},
        ],
        "notes": "",
    }
    with patch.object(settings.agents, "citation_whitelist_strict", True):
        result = _build_section_result(
            parsed,
            outline,
            "section",
            1,
            0.1,
            [],
            citable_keys={"Allowed_2024_Source"},
        )
    draft = result["sections"]["section"]
    assert "Blocked_2023_Source" not in draft["text"]
    assert [item["bibtex_key"] for item in draft["citations"]] == [
        "Allowed_2024_Source",
    ]


def test_writer_canonicalizes_unique_hashless_citations_from_llm_output():
    """A model may omit the manifest hash, but only an unambiguous key is recoverable."""
    from paperhunter.agents.state import SectionOutline
    from paperhunter.agents.writer_agent import _build_section_result
    from paperhunter.config import settings

    outline = SectionOutline.model_validate(
        {"key": "section", "title": "Section", "section_type": "body"}
    )
    full_key = "Abbas_2025_Small_ea340ecf"
    parsed = {
        "text": "Supported claim [Abbas_2025_Small].",
        "citations": [
            {"bibtex_key": "Abbas_2025_Small", "claim": "supported claim"},
        ],
        "notes": "",
    }
    errors: list[str] = []
    with patch.object(settings.agents, "citation_whitelist_strict", True):
        result = _build_section_result(
            parsed,
            outline,
            "section",
            1,
            0.1,
            errors,
            citable_keys={full_key},
        )

    draft = result["sections"]["section"]
    assert draft["text"] == f"Supported claim [{full_key}]."
    assert [item["bibtex_key"] for item in draft["citations"]] == [full_key]
    assert errors == []


def test_writer_rejects_ambiguous_hashless_citation():
    """Prefix recovery remains fail-closed when more than one manifest key matches."""
    from paperhunter.agents.state import SectionOutline
    from paperhunter.agents.writer_agent import _build_section_result
    from paperhunter.config import settings

    outline = SectionOutline.model_validate(
        {"key": "section", "title": "Section", "section_type": "body"}
    )
    parsed = {
        "text": "Ambiguous claim [Lee_2025_Guard].",
        "citations": [{"bibtex_key": "Lee_2025_Guard", "claim": "ambiguous"}],
        "notes": "",
    }
    errors: list[str] = []
    with patch.object(settings.agents, "citation_whitelist_strict", True):
        result = _build_section_result(
            parsed,
            outline,
            "section",
            1,
            0.1,
            errors,
            citable_keys={"Lee_2025_Guard_aaaaaaaa", "Lee_2025_Guard_bbbbbbbb"},
        )

    draft = result["sections"]["section"]
    assert "Lee_2025_Guard" not in draft["text"]
    assert draft["citations"] == []
    assert errors


def test_semicolon_citation_groups_are_supported_end_to_end():
    """DeepSeek often emits ``[key1; key2]``; every citation stage must parse it."""
    from paperhunter.agents.assembler_agent import (
        _extract_citations_from_text as extract_assembler_citations,
    )
    from paperhunter.agents.critic_agent import (
        _extract_citation_occurrences,
        _extract_citations_from_text as extract_critic_citations,
        _extract_claims_for_citation,
    )
    from paperhunter.agents.state import SectionOutline
    from paperhunter.agents.writer_agent import _build_section_result
    from paperhunter.config import settings
    from paperhunter.postprocessing.bibliography_generator import (
        _count_citations_in_text,
    )
    from paperhunter.postprocessing.latex_exporter import (
        _convert_citations,
        _extract_markdown_citation_keys,
    )

    first = "Abbas_2025_Small_ea340ecf"
    second = "Zhu_2025_LLM_dbd8c508"
    raw_text = f"Joint evidence [{first}; {second}]."

    assert extract_critic_citations(raw_text) == [first, second]
    assert len(_extract_claims_for_citation(raw_text, second)) == 1
    assert [item["bibtex_key"] for item in _extract_citation_occurrences(raw_text)] == [
        first,
        second,
    ]
    assert [item.bibtex_key for item in extract_assembler_citations(raw_text)] == [
        first,
        second,
    ]
    assert _count_citations_in_text(raw_text) == {first: 1, second: 1}
    assert _extract_markdown_citation_keys(raw_text) == {first, second}
    assert _convert_citations(raw_text) == rf"Joint evidence \cite{{{first}, {second}}}."

    outline = SectionOutline.model_validate(
        {"key": "section", "title": "Section", "section_type": "body"}
    )
    parsed = {
        "text": "Joint evidence [Abbas_2025_Small; Zhu_2025_LLM].",
        "citations": [
            {"bibtex_key": "Abbas_2025_Small", "claim": "joint"},
            {"bibtex_key": "Zhu_2025_LLM", "claim": "joint"},
        ],
        "notes": "",
    }
    with patch.object(settings.agents, "citation_whitelist_strict", True):
        result = _build_section_result(
            parsed,
            outline,
            "section",
            1,
            0.1,
            [],
            citable_keys={first, second},
        )
    draft = result["sections"]["section"]
    assert draft["text"] == f"Joint evidence [{first}, {second}]."
    assert [item["bibtex_key"] for item in draft["citations"]] == [first, second]


def test_revision_floor_recounts_legacy_draft_when_word_count_is_zero():
    from paperhunter.agents.writer_agent import _minimum_acceptable_revision_words

    existing = SectionDraft.model_validate(
        {
            "section_key": "body",
            "title": "Body",
            "text": "word " * 2000,
            "word_count": 0,
        }
    )
    outline = SectionOutline.model_validate(
        {
            "key": "body",
            "title": "Body",
            "section_type": "body",
            "target_length_words": 1500,
        }
    )
    assert _minimum_acceptable_revision_words(existing, outline) == 2000


def test_revision_floor_allows_critic_directed_pruning_to_target_floor():
    from paperhunter.agents.writer_agent import _minimum_acceptable_revision_words

    existing = SectionDraft.model_validate(
        {
            "section_key": "body",
            "title": "Body",
            "text": "word " * 1750,
            "word_count": 1750,
            "critic_feedback": "Remove unsupported claims and citations.",
            "status": SectionStatus.NEEDS_REVISION,
        }
    )
    outline = SectionOutline.model_validate(
        {
            "key": "body",
            "title": "Body",
            "section_type": "body",
            "target_length_words": 1500,
        }
    )
    assert _minimum_acceptable_revision_words(existing, outline) == 1200


def test_writer_prompt_requires_atomic_strict_citations():
    from paperhunter.utils.prompt_loader import get_prompt

    prompt = get_prompt("WRITER_PROMPT")
    assert "one atomic factual claim" in prompt
    assert "each source independently supports" in prompt


def test_evidence_repair_instructions_set_numeric_length_and_citation_budget():
    from paperhunter.agents.writer_agent import _build_critic_repair_instructions

    outline = SectionOutline.model_validate(
        {
            "key": "body",
            "title": "Body",
            "section_type": "body",
            "target_length_words": 1500,
        }
    )
    instructions = _build_critic_repair_instructions(
        outline,
        "Remove Unsupported_2025_Key.",
    )
    assert "at least 1200 words" in instructions
    assert "5-8 distinct citation keys" in instructions
    assert "at most 12 citation occurrences" in instructions
    assert "Remove Unsupported_2025_Key." in instructions


def test_shorter_revision_cannot_replace_long_legacy_draft():
    import json

    from paperhunter.agents.writer_agent import write_section
    from paperhunter.config import settings

    outline = SectionOutline.model_validate(
        {
            "key": "body",
            "title": "Body",
            "section_type": "body",
            "target_length_words": 1500,
        }
    )
    state = create_initial_state(outline=[outline])
    state["sections"] = {
        "body": SectionDraft.model_validate(
            {
                "section_key": "body",
                "title": "Body",
                "text": "original " * 2000,
                "word_count": 0,
                "status": SectionStatus.NEEDS_REVISION,
                "iteration": 1,
            }
        ).to_dict(),
    }
    chunks = [_chunk(f"Paper_{i}", f"c{i}", 1.0) for i in range(5)]
    with (
        patch.object(settings.agents, "writer_mode", "static"),
        patch.object(settings.agents, "min_citations_per_section", 5),
        patch(
            "paperhunter.agents.writer_agent._generate_rag_queries",
            return_value=["query"],
        ),
        patch(
            "paperhunter.agents.writer_agent._retrieve_context_for_section",
            return_value=(chunks, [item["chunk_id"] for item in chunks]),
        ),
        patch("paperhunter.agents.writer_agent.get_llm", return_value=MagicMock()),
        patch(
            "paperhunter.agents.writer_agent.llm_call_with_retry",
            return_value=json.dumps(
                {
                    "text": "replacement " * 1300,
                    "citations": [],
                    "notes": "revision",
                }
            ),
        ),
    ):
        result = write_section(
            state,
            "body",
            retriever=MagicMock(),
            use_reranker=False,
        )

    assert result["sections"] == {}
    assert state["sections"]["body"]["text"] == "original " * 2000
    assert any("rejected" in error for error in result["errors"])


@pytest.mark.parametrize("chunks", [[], [_chunk(f"Paper_{i}", f"c{i}", 1.0) for i in range(4)]])
def test_writer_evidence_failures_return_uniform_state_without_llm(chunks):
    from paperhunter.agents.writer_agent import write_section
    from paperhunter.config import settings

    outline = SectionOutline.model_validate(
        {
            "key": "body",
            "title": "Body",
            "section_type": "body",
            "target_length_words": 1000,
        }
    )
    state = create_initial_state(outline=[outline])
    with (
        patch.object(settings.agents, "writer_mode", "static"),
        patch.object(settings.agents, "min_citations_per_section", 5),
        patch(
            "paperhunter.agents.writer_agent._generate_rag_queries",
            return_value=["query"],
        ),
        patch(
            "paperhunter.agents.writer_agent._retrieve_context_for_section",
            return_value=(chunks, [item["chunk_id"] for item in chunks]),
        ),
        patch("paperhunter.agents.writer_agent.get_llm") as get_llm,
    ):
        result = write_section(
            state,
            "body",
            retriever=MagicMock(),
            use_reranker=False,
        )

    assert result["sections"] == {}
    assert result["provenance"] == []
    assert result["errors"]
    get_llm.assert_not_called()


def test_critic_extracts_every_citation_occurrence_not_only_unique_keys():
    """Reusing one paper for two claims must produce two independent checks."""
    from paperhunter.agents.critic_agent import _extract_citation_occurrences

    text = (
        "The method reduces attack success rate [Smith_2024_Defense]. "
        "The same paper evaluates three model families [Smith_2024_Defense]."
    )
    occurrences = _extract_citation_occurrences(text)

    assert len(occurrences) == 2
    assert [item["bibtex_key"] for item in occurrences] == [
        "Smith_2024_Defense",
        "Smith_2024_Defense",
    ]
    assert "reduces attack success" in occurrences[0]["claim"]
    assert "three model families" in occurrences[1]["claim"]


def test_strict_length_gate_uses_outline_target_not_only_global_minimum():
    """A 600-word draft cannot pass a 1,500-word outline target."""
    from paperhunter.agents.critic_agent import _check_section_length

    outline = SectionOutline(
        key="body",
        title="Body",
        section_type="body",
        target_length_words=1500,
    )
    draft = SectionDraft(
        section_key="body",
        title="Body",
        text="word " * 600,
        word_count=600,
        status=SectionStatus.DRAFTED,
    )

    passed, message = _check_section_length(draft, outline, strict=True)
    assert not passed
    assert "1200" in message  # 80% of the outline target


def test_pipeline_retries_missing_required_sections_before_assembly():
    """A failed Writer section is retried instead of being mistaken for completion."""
    from paperhunter.agents.pipeline import EDGE_REVISE, _should_revise

    outline = [
        SectionOutline(key="abstract", title="Abstract", section_type="abstract", order=0),
        SectionOutline(key="body", title="Body", section_type="body", order=1),
    ]
    state = create_initial_state(outline=outline)
    state["iteration_count"] = 0

    assert _should_revise(state) == EDGE_REVISE


def test_assembly_readiness_is_fail_closed_for_missing_or_unverified_sections():
    """The Assembler may only consume every required, Critic-verified section."""
    from paperhunter.agents.assembler_agent import _validate_assembly_readiness

    outline = [
        SectionOutline(key="abstract", title="Abstract", section_type="abstract", order=0),
        SectionOutline(key="intro", title="Introduction", section_type="introduction", order=1),
        SectionOutline(key="body", title="Body", section_type="body", order=2),
    ]
    state = create_initial_state(outline=outline)
    state["sections"] = {
        "intro": SectionDraft(
            section_key="intro",
            title="Introduction",
            text="Draft only",
            status=SectionStatus.DRAFTED,
        ).to_dict(),
    }

    ready, errors = _validate_assembly_readiness(state)
    assert not ready
    assert any("body" in error and "missing" in error.lower() for error in errors)
    assert any("intro" in error and "verified" in error.lower() for error in errors)


def test_finalizer_never_promotes_unverified_drafts():
    from paperhunter.agents.assembler_agent import _finalize_all_sections

    state = create_initial_state()
    state["sections"] = {
        "good": SectionDraft(
            section_key="good", title="Good", text="Verified", status=SectionStatus.VERIFIED
        ).to_dict(),
        "bad": SectionDraft(
            section_key="bad", title="Bad", text="Draft", status=SectionStatus.DRAFTED
        ).to_dict(),
    }

    updated = _finalize_all_sections(state)
    assert updated["good"]["status"] == SectionStatus.FINALIZED.value
    assert "bad" not in updated


def test_production_writer_configuration_is_single_call_and_full_context():
    """Static writing avoids expensive ReAct loops; fixed budget is per section."""
    from paperhunter.agents.writer_agent import _compute_section_budget
    from paperhunter.config import settings

    assert settings.agents.writer_mode == "static"
    assert settings.agents.writer_budget_mode == "fixed"
    assert _compute_section_budget(n_total_sections=11) == settings.agents.writer_max_context_chars
    assert settings.deepseek.reasoning_effort != "max"


def test_citable_provenance_reaches_l1_5_l2_and_planner_context():
    """Planner must know which categories can support five strict citations."""
    from paperhunter.agents.planner_agent import _format_l2_for_prompt
    from paperhunter.models import L1_5Summary, L1Summary, L2Summary
    from paperhunter.summarization.level1_5_summarizer import (
        _extract_citable_paper_ids,
    )

    l1_inputs = [
        L1Summary.model_validate(
            {
                "paper_id": "p1",
                "bibtex_key": "Citable_2024_One",
                "is_citable": True,
            }
        ),
        L1Summary.model_validate(
            {
                "paper_id": "p2",
                "bibtex_key": "Background_2023_Two",
                "is_citable": False,
            }
        ),
        L1Summary.model_validate(
            {
                "paper_id": "p3",
                "bibtex_key": "Citable_2025_Three",
                "is_citable": True,
            }
        ),
        L1Summary.model_validate(
            {
                "paper_id": "local-id-only",
                "is_citable": True,
            }
        ),
    ]
    citable = _extract_citable_paper_ids(l1_inputs)
    assert citable == ["Citable_2024_One", "Citable_2025_Three"]

    cluster = L1_5Summary.model_validate(
        {
            "cluster_id": 1,
            "category": "Defenses",
            "paper_ids": [item.bibtex_key or item.paper_id for item in l1_inputs],
            "paper_count": 4,
            "citable_paper_ids": citable,
        }
    )
    l2 = L2Summary.model_validate(
        {
            "category": cluster.category,
            "paper_ids": cluster.paper_ids,
            "paper_count": cluster.paper_count,
            "citable_paper_ids": cluster.citable_paper_ids,
        }
    )
    planner_context = _format_l2_for_prompt([l2])
    assert "2 DBLP-citable" in planner_context
    assert "Citable_2024_One" in planner_context
    assert "Background_2023_Two" not in planner_context


def test_planner_keeps_sections_with_structured_subsections():
    """DeepSeek may return subsection objects; parent body sections must survive."""
    from paperhunter.agents.planner_agent import _validate_and_normalize_outline

    raw = [
        {
            "key": "adversarial_attacks",
            "title": "Adversarial Attacks",
            "section_type": "body",
            "subsections": [
                {"key": "white_box", "title": "White-box Attacks", "subsections": []},
                {"key": "black_box", "title": "Black-box Attacks", "subsections": []},
            ],
        }
    ]
    outline = _validate_and_normalize_outline(raw, ["Adversarial Attacks"])
    body = next(section for section in outline if section.key == "adversarial_attacks")
    assert body.subsections == ["White-box Attacks", "Black-box Attacks"]


def test_agentic_retrieval_output_marks_citable_and_background_sources():
    from paperhunter.agents.tools.retrieval_tools import (
        _format_bm25_results,
        _format_qdrant_results,
    )

    results = [
        _chunk("Allowed_2024_Source", "a", 0.9, citable=True),
        _chunk("Context_2023_Only", "b", 0.8, citable=False),
    ]
    for formatter in (_format_qdrant_results, _format_bm25_results):
        output = formatter(results)
        assert "[Allowed_2024_Source] [CITABLE]" in output
        assert "[Context_2023_Only] [BACKGROUND]" in output


def test_critic_rejects_background_sources_and_malformed_confidence():
    from paperhunter.agents.critic_agent import (
        _normalize_verification,
        _source_chunks_are_citable,
    )

    background = [_chunk("Context_2023_Only", "b", 0.8, citable=False)]
    citable = [_chunk("Allowed_2024_Source", "a", 0.9, citable=True)]
    assert not _source_chunks_are_citable(background)
    assert _source_chunks_are_citable(citable)
    normalized = _normalize_verification(
        {
            "status": "SOURCE_NOT_FOUND",
            "confidence": "not-a-number",
        }
    )
    assert normalized["confidence"] == 0.0


def test_critic_missing_source_fails_without_spending_verifier_call():
    from paperhunter.agents.critic_agent import verify_section
    from paperhunter.agents.state import CitationRecord
    from paperhunter.config import settings

    outline = SectionOutline.model_validate(
        {
            "key": "body",
            "title": "Body",
            "section_type": "body",
            "target_length_words": 100,
        }
    )
    state = create_initial_state(outline=[outline])
    state["sections"] = {
        "body": SectionDraft.model_validate(
            {
                "section_key": "body",
                "title": "Body",
                "text": "A factual claim [Missing_2024_Source].",
                "citations": [
                    CitationRecord.model_validate(
                        {
                            "bibtex_key": "Missing_2024_Source",
                        }
                    )
                ],
                "status": SectionStatus.DRAFTED,
                "iteration": 1,
            }
        ).to_dict(),
    }
    store = MagicMock()
    store.get_paper_chunks.return_value = []
    llm = MagicMock()
    with patch.object(settings.agents, "critic_two_pass", False):
        result = verify_section(
            state,
            "body",
            store=store,
            llm=llm,
            generate_feedback=False,
        )

    llm.complete.assert_not_called()
    detail = result["verification_results"]["body"]["details"][0]
    assert detail["status"] == "SOURCE_NOT_FOUND"


def test_atomic_claim_selection_requires_meaningful_overlap():
    from paperhunter.agents.critic_agent import _select_best_atomic_claim

    claim = "The defense reduces attack success rate across ten language models."
    candidates = [
        "The benchmark evaluates models using a public dataset.",
        "The defense reduces attack success rate across ten language models.",
    ]
    assert _select_best_atomic_claim(claim, candidates) == candidates[1]
    assert _select_best_atomic_claim(claim, [candidates[0]]) is None
