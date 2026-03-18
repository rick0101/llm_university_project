"""
Test suite for paperhunter.agents.writer_agent

Verifica:
1. Query generation (LLM + fallback)
2. Response parsing (JSON, code fences, trailing commas, plain text)
3. Citation extraction from text
4. Context formatting for prompt
5. Special instructions per section type
6. write_section con mock retriever/LLM
7. write_all_sections batch
8. get_section_stats
9. LangGraph write node integration
10. Edge cases (no context, no outline, NEEDS_REVISION rewrite)
11. (LIVE) write_section con Gemini reale (opzionale)
"""

import json
import re
import time
from datetime import datetime
from unittest.mock import MagicMock, patch


if __name__ == '__main__':
    # ============================================================================
    # TEST 1: Query JSON parsing
    # ============================================================================
    print("=" * 70)
    print("TEST 1: Parsing query JSON")
    print("=" * 70)
    
    from paperhunter.agents.writer_agent import _parse_query_json
    
    # Clean JSON
    result = _parse_query_json('["query 1", "query 2", "query 3"]')
    assert result == ["query 1", "query 2", "query 3"]
    print("  ✓ Clean JSON array parsed")
    
    # With code fences
    result = _parse_query_json('```json\n["a", "b"]\n```')
    assert result == ["a", "b"]
    print("  ✓ Code fences removed")
    
    # Embedded in text
    result = _parse_query_json('Here are the queries:\n["q1", "q2"]\nHope this helps!')
    assert result == ["q1", "q2"]
    print("  ✓ Array extracted from surrounding text")
    
    # Empty
    assert _parse_query_json("") is None
    assert _parse_query_json(None) is None
    print("  ✓ Empty/None → None")
    
    # Non-JSON text
    assert _parse_query_json("Just some text without JSON") is None
    print("  ✓ Non-JSON text → None")
    
    print("TEST 1 PASSED ✓\n")
    
    # ============================================================================
    # TEST 2: Writer response parsing
    # ============================================================================
    print("=" * 70)
    print("TEST 2: Writer response parsing")
    print("=" * 70)
    
    from paperhunter.agents.writer_agent import _parse_writer_response
    
    # Clean JSON
    good_response = json.dumps({
        "text": "Prompt injection attacks represent a major threat [Zou_2023_GCG].",
        "citations": [
            {"bibtex_key": "Zou_2023_GCG", "claim": "major threat", "chunk_id": "c1"}
        ],
        "notes": "All good"
    })
    parsed = _parse_writer_response(good_response, "test_section")
    assert parsed is not None
    assert "Zou_2023_GCG" in parsed["text"]
    assert len(parsed["citations"]) == 1
    assert parsed["citations"][0]["bibtex_key"] == "Zou_2023_GCG"
    print("  ✓ Clean JSON response parsed")
    
    # With code fences
    fenced = f"```json\n{good_response}\n```"
    parsed2 = _parse_writer_response(fenced, "test_section")
    assert parsed2 is not None
    assert len(parsed2["citations"]) == 1
    print("  ✓ Code-fenced JSON parsed")
    
    # With trailing commas
    trailing = '{"text": "Test [Author_2023_Paper].", "citations": [{"bibtex_key": "Author_2023_Paper", "claim": "test",},], "notes": "",}'
    parsed3 = _parse_writer_response(trailing, "test_section")
    assert parsed3 is not None
    assert len(parsed3["citations"]) == 1
    print("  ✓ Trailing commas handled")
    
    # Plain text fallback (not JSON)
    plain = "This is a section about prompt injection [Zou_2023_GCG] and defenses [Lee_2025_Shield]."
    parsed4 = _parse_writer_response(plain, "test_section")
    assert parsed4 is not None
    assert "Zou_2023_GCG" in parsed4["text"]
    # Check if regex extracted the citations
    citations_keys = {c["bibtex_key"] for c in parsed4["citations"]}
    assert "Zou_2023_GCG" in citations_keys
    assert "Lee_2025_Shield" in citations_keys
    print(f"  ✓ Plain text fallback: extracted {len(parsed4['citations'])} citations via regex")
    
    # Empty
    assert _parse_writer_response("", "test") is None
    assert _parse_writer_response(None, "test") is None
    print("  ✓ Empty/None → None")
    
    print("TEST 2 PASSED ✓\n")
    
    # ============================================================================
    # TEST 3: Citation extraction from text
    # ============================================================================
    print("=" * 70)
    print("TEST 3: Citation extraction from text")
    print("=" * 70)
    
    from paperhunter.agents.writer_agent import _extract_citations_from_text
    
    text = (
        "Adversarial attacks [Zou_2023_GCG] can transfer across models [Zou_2023_GCG]. "
        "Defense mechanisms [Li_2024_SmoothLLM] have been proposed. "
        "Earlier work on jailbreaking [Wei_2023_Jailbreak] showed promise. "
        "Some regular text [not_a_citation] and [also_not_one]."
    )
    citations = _extract_citations_from_text(text)
    keys = {c["bibtex_key"] for c in citations}
    assert "Zou_2023_GCG" in keys
    assert "Li_2024_SmoothLLM" in keys
    assert "Wei_2023_Jailbreak" in keys
    # Verify deduplication
    assert len([c for c in citations if c["bibtex_key"] == "Zou_2023_GCG"]) == 1
    print(f"  ✓ Extracted {len(citations)} unique citations: {keys}")
    
    # Edge: no citations
    assert _extract_citations_from_text("No citations here.") == []
    print("  ✓ No citations → empty list")
    
    print("TEST 3 PASSED ✓\n")
    
    # ============================================================================
    # TEST 4: Context formatting
    # ============================================================================
    print("=" * 70)
    print("TEST 4: Context formatting for prompt")
    print("=" * 70)
    
    from paperhunter.agents.writer_agent import _format_context_for_prompt
    
    chunks = [
        {
            "chunk_id": "chunk_001",
            "text": "GCG uses gradient-based optimization to find adversarial suffixes.",
            "score": 0.95,
            "metadata": {
                "bibtex_key": "Zou_2023_GCG",
                "category": "Adversarial Attacks",
                "year": 2023,
                "section_name": "Introduction",
                "chunk_type": "original",
            },
        },
        {
            "chunk_id": "chunk_002",
            "text": "SmoothLLM applies random perturbations to detect adversarial inputs.",
            "score": 0.88,
            "metadata": {
                "bibtex_key": "Li_2024_SmoothLLM",
                "category": "Adversarial Attacks",
                "year": 2024,
                "section_name": "Methodology",
                "chunk_type": "original",
            },
        },
    ]
    
    formatted = _format_context_for_prompt(chunks)
    assert "chunk_001" in formatted
    assert "Zou_2023_GCG" in formatted
    assert "Adversarial Attacks" in formatted
    assert "2023" in formatted
    assert "GCG uses gradient" in formatted
    assert "chunk_002" in formatted
    assert "Li_2024_SmoothLLM" in formatted
    print(f"  ✓ Formatted {len(chunks)} chunks ({len(formatted)} chars)")
    
    # Empty
    empty_fmt = _format_context_for_prompt([])
    assert "No relevant context" in empty_fmt
    print("  ✓ Empty chunks → placeholder text")
    
    # Max chars truncation
    long_chunks = [
        {"chunk_id": f"c_{i}", "text": "x" * 1000, "score": 0.5, "metadata": {}}
        for i in range(100)
    ]
    truncated = _format_context_for_prompt(long_chunks, max_chars=5000)
    assert "truncated" in truncated.lower() or len(truncated) <= 6000
    print(f"  ✓ Max chars truncation: {len(truncated)} chars")
    
    print("TEST 4 PASSED ✓\n")
    
    # ============================================================================
    # TEST 5: Special instructions per section type
    # ============================================================================
    print("=" * 70)
    print("TEST 5: Special instructions per section type")
    print("=" * 70)
    
    from paperhunter.agents.writer_agent import _get_special_instructions
    
    testing_inst = _get_special_instructions("testing")
    assert "TESTING METHODOLOGIES" in testing_inst
    assert "table" in testing_inst.lower() or "Table" in testing_inst
    print("  ✓ Testing section: table instructions present")
    
    intro_inst = _get_special_instructions("introduction")
    assert "introduction" in intro_inst.lower() or "Introduction" in intro_inst
    print("  ✓ Introduction section: special instructions present")
    
    discussion_inst = _get_special_instructions("discussion")
    assert "discussion" in discussion_inst.lower() or "Discussion" in discussion_inst
    print("  ✓ Discussion section: special instructions present")
    
    body_inst = _get_special_instructions("body")
    assert body_inst == ""
    print("  ✓ Body section: no special instructions")
    
    print("TEST 5 PASSED ✓\n")
    
    # ============================================================================
    # TEST 6: Fallback query generation
    # ============================================================================
    print("=" * 70)
    print("TEST 6: Fallback query generation")
    print("=" * 70)
    
    from paperhunter.agents.writer_agent import _build_fallback_queries
    from paperhunter.agents.state import SectionOutline
    
    outline = SectionOutline(
        key="prompt_injection_attacks",
        title="Prompt Injection Attacks",
        description="Describe direct and indirect prompt injection attacks. Cover jailbreaking techniques.",
        key_topics=["goal hijacking", "payload splitting", "jailbreaking"],
        related_categories=["Prompt Injection"],
        order=2,
        section_type="body",
    )
    
    queries = _build_fallback_queries(outline, n_queries=5)
    assert len(queries) == 5
    assert any("Prompt Injection" in q for q in queries)
    assert any("goal hijacking" in q or "payload splitting" in q or "jailbreaking" in q for q in queries)
    print(f"  ✓ Generated {len(queries)} fallback queries:")
    for q in queries:
        print(f"    - {q[:70]}")
    
    # Minimal outline
    minimal = SectionOutline(
        key="test", title="Test Section", description="", key_topics=[], 
        related_categories=[], order=0, section_type="body",
    )
    min_queries = _build_fallback_queries(minimal, n_queries=3)
    assert len(min_queries) == 3
    print(f"  ✓ Minimal outline → {len(min_queries)} fallback queries")
    
    print("TEST 6 PASSED ✓\n")
    
    # ============================================================================
    # TEST 7: write_section with mock retriever and LLM
    # ============================================================================
    print("=" * 70)
    print("TEST 7: write_section with mock retriever and LLM")
    print("=" * 70)
    
    from paperhunter.agents.state import create_initial_state, SectionDraft, SectionStatus
    from paperhunter.agents.writer_agent import write_section
    
    # Setup mock retriever
    mock_retriever = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.chunk_id = "chunk_001"
    mock_chunk.text = "GCG generates adversarial suffixes via gradient optimization [Zou et al., 2023]."
    mock_chunk.score = 0.95
    mock_chunk.metadata = {
        "bibtex_key": "Zou_2023_GCG",
        "category": "Adversarial Attacks",
        "year": 2023,
        "section_name": "Introduction",
        "chunk_type": "original",
    }
    mock_chunk.to_dict.return_value = {
        "chunk_id": "chunk_001",
        "text": mock_chunk.text,
        "score": 0.95,
        "metadata": mock_chunk.metadata,
    }
    
    mock_chunk2 = MagicMock()
    mock_chunk2.chunk_id = "chunk_002"
    mock_chunk2.text = "SmoothLLM applies perturbations to detect adversarial inputs [Li et al., 2024]."
    mock_chunk2.score = 0.88
    mock_chunk2.metadata = {
        "bibtex_key": "Li_2024_Defense",
        "category": "Adversarial Attacks",
        "year": 2024,
        "section_name": "Methodology",
        "chunk_type": "original",
    }
    mock_chunk2.to_dict.return_value = {
        "chunk_id": "chunk_002",
        "text": mock_chunk2.text,
        "score": 0.88,
        "metadata": mock_chunk2.metadata,
    }
    
    mock_retriever.retrieve_multi_collection.return_value = [mock_chunk, mock_chunk2]
    
    # Setup state with outline
    test_outline = SectionOutline(
        key="adversarial_attacks",
        title="Adversarial Attacks on LLMs",
        description="Discuss adversarial examples and defenses.",
        target_length_words=1000,
        key_topics=["GCG", "adversarial suffixes", "transferability"],
        related_categories=["Adversarial Attacks"],
        order=3,
        section_type="body",
    )
    
    state = create_initial_state(outline=[test_outline])
    
    # Mock LLM response
    mock_llm_response = json.dumps({
        "text": (
            "Adversarial attacks on large language models have emerged as a critical "
            "security concern. Zou et al. introduced the Greedy Coordinate Gradient (GCG) "
            "attack, which generates adversarial suffixes via gradient-based optimization [Zou_2023_GCG]. "
            "This approach demonstrated remarkable transferability across different LLM families. "
            "In response, Li et al. proposed SmoothLLM, a defense mechanism that applies "
            "random perturbations to detect adversarial inputs [Li_2024_Defense]."
        ),
        "citations": [
            {"bibtex_key": "Zou_2023_GCG", "claim": "GCG attack using gradient optimization", "chunk_id": "chunk_001"},
            {"bibtex_key": "Li_2024_Defense", "claim": "SmoothLLM defense mechanism", "chunk_id": "chunk_002"},
        ],
        "notes": "Good coverage of attack and defense."
    })
    
    # Patch both the query generation LLM and the writer LLM
    with patch("paperhunter.agents.writer_agent._generate_rag_queries") as mock_gen_queries, \
         patch("paperhunter.agents.writer_agent.get_llm") as mock_get_llm, \
         patch("paperhunter.agents.writer_agent.llm_call_with_retry") as mock_llm_call:
    
        mock_gen_queries.return_value = [
            "GCG adversarial attack LLMs",
            "adversarial suffix transferability",
            "defense against adversarial attacks LLM",
        ]
    
        mock_llm_instance = MagicMock()
        mock_get_llm.return_value = mock_llm_instance
        mock_llm_call.return_value = mock_llm_response
    
        result = write_section(
            state,
            "adversarial_attacks",
            retriever=mock_retriever,
            reranker=None,
            use_reranker=False,
        )
    
    assert "sections" in result
    assert "adversarial_attacks" in result["sections"]
    draft_data = result["sections"]["adversarial_attacks"]
    assert isinstance(draft_data, dict)
    assert draft_data["status"] == SectionStatus.DRAFTED.value
    assert draft_data["iteration"] == 1
    assert len(draft_data["citations"]) >= 2
    assert draft_data["word_count"] > 0
    print(f"  ✓ write_section: {draft_data['word_count']} words, {len(draft_data['citations'])} citations")
    print(f"    Status: {draft_data['status']}, Iteration: {draft_data['iteration']}")
    print(f"    Citation keys: {[c['bibtex_key'] for c in draft_data['citations']]}")
    
    # Verify section not found
    result_bad = write_section(state, "nonexistent_section", retriever=mock_retriever, use_reranker=False)
    assert "errors" in result_bad
    assert any("non trovata" in e for e in result_bad["errors"])
    print("  ✓ Missing section key → error")
    
    print("TEST 7 PASSED ✓\n")
    
    # ============================================================================
    # TEST 8: get_section_stats
    # ============================================================================
    print("=" * 70)
    print("TEST 8: get_section_stats")
    print("=" * 70)
    
    from paperhunter.agents.writer_agent import get_section_stats
    from paperhunter.agents.state import CitationRecord
    
    sections = {
        "intro": SectionDraft(
            section_key="intro",
            title="Introduction",
            text="This is the introduction. " * 50,
            citations=[
                CitationRecord(bibtex_key="Zou_2023_GCG", claim="test"),
                CitationRecord(bibtex_key="Li_2023_Halueval", claim="test"),
            ],
            status=SectionStatus.DRAFTED,
            iteration=1,
        ),
        "attacks": SectionDraft(
            section_key="attacks",
            title="Attacks",
            text="Adversarial attacks are dangerous. " * 100,
            citations=[
                CitationRecord(bibtex_key="Zou_2023_GCG", claim="test"),
                CitationRecord(bibtex_key="Wei_2023_Jailbreak", claim="test"),
                CitationRecord(bibtex_key="Greshake_2023_LLM", claim="test"),
            ],
            status=SectionStatus.VERIFIED,
            iteration=2,
        ),
    }
    
    # Compute word counts
    for d in sections.values():
        d.compute_word_count()
    
    stats = get_section_stats(sections)
    assert stats["total_sections"] == 2
    assert stats["total_words"] > 0
    assert stats["total_citations"] == 5
    assert stats["unique_citation_keys"] == 4  # Zou, Li, Wei, Greshake
    assert "drafted" in stats["statuses"]
    assert "verified" in stats["statuses"]
    assert len(stats["per_section"]) == 2
    print(f"  ✓ Stats: {stats['total_sections']} sections, {stats['total_words']} words, "
          f"{stats['total_citations']} citations ({stats['unique_citation_keys']} unique)")
    print(f"    Statuses: {stats['statuses']}")
    print(f"    Avg words: {stats['avg_words_per_section']}")
    
    # Empty
    empty_stats = get_section_stats({})
    assert empty_stats["total_sections"] == 0
    print("  ✓ Empty sections → zeroed stats")
    
    print("TEST 8 PASSED ✓\n")
    
    # ============================================================================
    # TEST 9: write_all_sections with mock
    # ============================================================================
    print("=" * 70)
    print("TEST 9: write_all_sections with mock")
    print("=" * 70)
    
    from paperhunter.agents.writer_agent import write_all_sections
    
    outline_list = [
        SectionOutline(
            key="abstract", title="Abstract",
            description="Survey abstract.", target_length_words=300,
            order=0, section_type="abstract",
        ),
        SectionOutline(
            key="introduction", title="Introduction",
            description="Intro to LLM security.", target_length_words=1000,
            key_topics=["LLM security", "survey motivation"],
            related_categories=["all"],
            order=1, section_type="introduction",
        ),
        SectionOutline(
            key="prompt_injection", title="Prompt Injection",
            description="Direct and indirect injection.", target_length_words=1500,
            key_topics=["direct injection", "indirect injection"],
            related_categories=["Prompt Injection"],
            order=2, section_type="body",
        ),
        SectionOutline(
            key="conclusion", title="Conclusion",
            description="Summary.", target_length_words=500,
            order=3, section_type="conclusion",
        ),
    ]
    
    state_batch = create_initial_state(outline=outline_list)
    
    # Mock everything
    section_counter = {"count": 0}
    
    def mock_write_section_side_effect(state, section_key, **kwargs):
        section_counter["count"] += 1
        return {
            "sections": {
                section_key: SectionDraft(
                    section_key=section_key,
                    title=f"Mock {section_key}",
                    text=f"Content for {section_key}. " * 100,
                    citations=[CitationRecord(bibtex_key="Mock_2024_Paper", claim="mock")],
                    word_count=300,
                    status=SectionStatus.DRAFTED,
                    iteration=1,
                ).to_dict(),
            },
            "errors": [],
        }
    
    with patch("paperhunter.agents.writer_agent.write_section", side_effect=mock_write_section_side_effect):
        batch_result = write_all_sections(state_batch, skip_types={"abstract"})
    
    assert "sections" in batch_result
    # Should have written intro, prompt_injection, conclusion (3 sections, skip abstract)
    section_keys = set(batch_result["sections"].keys())
    assert "introduction" in section_keys
    assert "prompt_injection" in section_keys
    assert "conclusion" in section_keys
    assert "abstract" not in section_keys
    print(f"  ✓ write_all_sections: {len(section_keys)} sections written (abstract skipped)")
    print(f"    Keys: {section_keys}")
    
    print("TEST 9 PASSED ✓\n")
    
    # ============================================================================
    # TEST 10: LangGraph integration
    # ============================================================================
    print("=" * 70)
    print("TEST 10: LangGraph integration")
    print("=" * 70)
    
    from langgraph.graph import StateGraph
    from paperhunter.agents.state import SurveyState, PipelinePhase
    from paperhunter.agents.writer_agent import write
    
    # Build graph with planner → writer
    def mock_planner(state: SurveyState) -> dict:
        return {
            "outline": [
                SectionOutline(
                    key="intro", title="Introduction",
                    description="LLM security intro.",
                    target_length_words=500,
                    order=0, section_type="introduction",
                    key_topics=["LLM", "security"],
                ).to_dict(),
            ],
            "current_phase": PipelinePhase.PLANNING.value,
        }
    
    # Mock the writer to avoid real LLM calls
    def mock_writer(state: SurveyState) -> dict:
        outline = state.get("outline", [])
        sections = {}
        for o in outline:
            key = o["key"] if isinstance(o, dict) else o.key
            sections[key] = SectionDraft(
                section_key=key,
                title=o.get("title", key) if isinstance(o, dict) else o.title,
                text=f"Mock content for {key}. " * 50,
                citations=[CitationRecord(bibtex_key="Test_2024_Mock", claim="test")],
                word_count=250,
                status=SectionStatus.DRAFTED,
                iteration=1,
            ).to_dict()
        return {
            "sections": sections,
            "current_phase": PipelinePhase.WRITING.value,
        }
    
    graph = StateGraph(SurveyState)
    graph.add_node("planner", mock_planner)
    graph.add_node("writer", mock_writer)
    graph.add_edge("planner", "writer")
    graph.set_entry_point("planner")
    graph.set_finish_point("writer")
    
    compiled = graph.compile()
    result = compiled.invoke(create_initial_state())
    
    assert result["current_phase"] == PipelinePhase.WRITING.value
    assert "intro" in result["sections"]
    draft = result["sections"]["intro"]
    assert draft["word_count"] == 250
    assert len(draft["citations"]) == 1
    print(f"  ✓ LangGraph planner→writer: section 'intro' written")
    print(f"    Words: {draft['word_count']}, Citations: {len(draft['citations'])}")
    
    print("TEST 10 PASSED ✓\n")
    
    # ============================================================================
    # TEST 11: NEEDS_REVISION detection
    # ============================================================================
    print("=" * 70)
    print("TEST 11: NEEDS_REVISION handling")
    print("=" * 70)
    
    # Create state with a section that needs revision
    revision_outline = SectionOutline(
        key="attacks", title="Attacks",
        description="Attack methods.", target_length_words=1000,
        key_topics=["adversarial"], related_categories=["Adversarial Attacks"],
        order=1, section_type="body",
    )
    revision_state = create_initial_state(outline=[revision_outline])
    
    # Add existing section that needs revision
    revision_state["sections"] = {
        "attacks": SectionDraft(
            section_key="attacks",
            title="Attacks",
            text="Bad content.",
            citations=[],
            word_count=2,
            status=SectionStatus.NEEDS_REVISION,
            iteration=1,
            critic_feedback="Citations are missing. Add specific references for all claims.",
        ).to_dict(),
    }
    
    # Test that write_section increments iteration
    with patch("paperhunter.agents.writer_agent._generate_rag_queries") as mock_q, \
         patch("paperhunter.agents.writer_agent.get_llm") as mock_get, \
         patch("paperhunter.agents.writer_agent.llm_call_with_retry") as mock_call:
    
        mock_q.return_value = ["adversarial attacks LLM"]
        mock_get.return_value = MagicMock()
        mock_call.return_value = json.dumps({
            "text": "Improved content about attacks [Zou_2023_GCG].",
            "citations": [{"bibtex_key": "Zou_2023_GCG", "claim": "attacks", "chunk_id": "c1"}],
            "notes": "Improved based on feedback",
        })
    
        result_rev = write_section(
            revision_state,
            "attacks",
            retriever=mock_retriever,
            use_reranker=False,
        )
    
    assert "sections" in result_rev
    assert "attacks" in result_rev["sections"]
    draft_rev = result_rev["sections"]["attacks"]
    assert draft_rev["iteration"] == 2  # Incremented from 1
    assert len(draft_rev["citations"]) >= 1
    print(f"  ✓ NEEDS_REVISION: iteration incremented to {draft_rev['iteration']}")
    print(f"    Citations after revision: {len(draft_rev['citations'])}")
    
    # Test max iterations
    max_iter_draft = SectionDraft(
        section_key="attacks", title="Attacks", text="...",
        iteration=3,  # At max (config default = 3)
        status=SectionStatus.NEEDS_REVISION,
    ).to_dict()
    revision_state["sections"] = {"attacks": max_iter_draft}
    
    result_max = write_section(
        revision_state, "attacks",
        retriever=mock_retriever, use_reranker=False,
    )
    # Should skip because max iterations reached
    assert "sections" not in result_max or "attacks" not in result_max.get("sections", {})
    print("  ✓ Max iterations reached → section skipped")
    
    print("TEST 11 PASSED ✓\n")
    
    # ============================================================================
    # TEST 12: Skip verified/finalized sections
    # ============================================================================
    print("=" * 70)
    print("TEST 12: Skip verified/finalized sections")
    print("=" * 70)
    
    verified_state = create_initial_state(outline=[test_outline])
    verified_state["sections"] = {
        "adversarial_attacks": SectionDraft(
            section_key="adversarial_attacks",
            title="Adversarial Attacks",
            text="Already verified content.",
            status=SectionStatus.VERIFIED,
            iteration=2,
        ).to_dict(),
    }
    
    result_skip = write_section(
        verified_state, "adversarial_attacks",
        retriever=mock_retriever, use_reranker=False,
    )
    assert result_skip == {}
    print("  ✓ Verified section → skipped (empty result)")
    
    print("TEST 12 PASSED ✓\n")
    
    # ============================================================================
    # TEST 13: Imports from __init__.py
    # ============================================================================
    print("=" * 70)
    print("TEST 13: Imports from agents package")
    print("=" * 70)
    
    from paperhunter.agents import (
        write,
        write_section,
        write_all_sections,
        write_sections_from_outline,
        get_section_stats,
    )
    assert callable(write)
    assert callable(write_section)
    assert callable(write_all_sections)
    assert callable(write_sections_from_outline)
    assert callable(get_section_stats)
    print("  ✓ All writer_agent functions importable from paperhunter.agents")
    
    print("TEST 13 PASSED ✓\n")
    
    # ============================================================================
    # Summary
    # ============================================================================
    offline_tests = 13
    print("=" * 70)
    print(f"Test offline: {offline_tests}/{offline_tests} passed")
    print("=" * 70)
    
    # ============================================================================
    # TEST 14 (LIVE): write_section con Gemini reale
    # ============================================================================
    print(f"\n{'=' * 70}")
    print("TEST 14 (LIVE): write_section con Gemini reale")
    print("=" * 70)
    
    try:
        from paperhunter.utils.llm_factory import get_llm
    
        # Quick check that LLM import works
        test_llm = get_llm("flash", max_tokens=2048)
    
        # Test query generation with real LLM
        from paperhunter.agents.writer_agent import _generate_rag_queries
    
        live_outline = SectionOutline(
            key="prompt_injection_defense",
            title="Defenses Against Prompt Injection",
            description=(
                "Survey of defense mechanisms against prompt injection attacks including "
                "input sanitization, output filtering, instruction hierarchy, and "
                "sandwich defense techniques."
            ),
            target_length_words=1500,
            key_topics=["input sanitization", "instruction hierarchy", "sandwich defense", "guardrails"],
            related_categories=["Prompt Injection"],
            order=4,
            section_type="body",
        )
    
        t0 = time.time()
        live_queries = _generate_rag_queries(live_outline, n_queries=5)
        elapsed = time.time() - t0
        print(f"  Query generation: {len(live_queries)} queries in {elapsed:.1f}s")
        for q in live_queries:
            print(f"    - {q[:80]}")
        assert len(live_queries) >= 3
        print("  ✓ Live query generation OK")
    
    except Exception as e:
        print(f"  ✗ Test 14 (LIVE) FAILED: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"RIEPILOGO: {offline_tests}/{offline_tests} test offline passati")
    print("=" * 70)
