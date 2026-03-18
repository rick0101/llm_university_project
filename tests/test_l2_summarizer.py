#!/usr/bin/env python3
"""
Test script per Level 2 Summarizer.

Testa la generazione di meta-summary L2 per una categoria,
usando L1 summary sintetici (senza bisogno di chiamate API reali
per caricare PDF).

Uso:
    python test_l2_summarizer.py
"""

import sys
import time
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

console = Console()


def create_mock_l1_summaries():
    """Crea L1 summary sintetici per testare il L2 summarizer.

    Simula 5 paper nella categoria "Prompt Injection" con dati realistici.
    """
    from paperhunter.models import L1Summary

    summaries = [
        L1Summary(
            paper_id="Greshake_2023_LLM",
            bibtex_key="Greshake_2023_LLM",
            category="Prompt Injection",
            year=2023,
            key_contributions=[
                "First systematic study of indirect prompt injection attacks on LLM-integrated applications",
                "Demonstrated that adversarial instructions can be embedded in external data sources to manipulate LLM behavior",
                "Showed attacks on real-world applications including Bing Chat and code completion tools",
            ],
            methodology="The authors systematically explored indirect prompt injection by embedding adversarial prompts in web pages, emails, and documents that LLMs process as part of their context. They tested both manual and automated injection techniques.",
            testing_methodology="Tested on ChatGPT (GPT-3.5, GPT-4), Bing Chat, and GitHub Copilot. Evaluation included manual assessment of attack success on 50 web pages with embedded prompts. Metrics: Attack Success Rate (ASR), qualitative analysis of LLM responses. No standardized benchmark used.",
            temporal_context="Published in 2023, this was among the first papers to formalize indirect prompt injection as a security threat, building on earlier observations about direct prompt injection in 2022.",
            results_summary="Successfully demonstrated indirect prompt injection on Bing Chat with >80% ASR for simple instructions. Code completion attacks achieved ~60% success rate on Copilot.",
            limitations="Limited to qualitative evaluation; no formal benchmark. Manual testing only. Did not propose robust defenses.",
            generated_at=datetime.now(),
            model_used="mock",
        ),
        L1Summary(
            paper_id="Liu_2024_Formalizing",
            bibtex_key="Liu_2024_Formalizing",
            category="Prompt Injection",
            year=2024,
            key_contributions=[
                "Proposed a formal taxonomy of prompt injection attacks with 7 categories",
                "Developed HOUYI, an automated black-box prompt injection framework",
                "Evaluated 36 real-world LLM-integrated applications and found vulnerabilities in 31",
            ],
            methodology="HOUYI uses a three-phase approach: (1) context inference via probing queries, (2) separator identification through fuzzing, (3) payload generation with predefined templates. The framework automates the entire attack pipeline.",
            testing_methodology="Evaluated on 36 real-world LLM applications from the web. Metrics: Attack Success Rate (ASR), number of vulnerable apps. Comparison baselines: manual prompt injection, naive concatenation attacks. No open-source benchmark; custom evaluation setup.",
            temporal_context="Published in 2024, building on Greshake et al. (2023). Extended the threat model from academic demos to real-world deployed applications.",
            results_summary="31/36 (86.1%) real-world applications were vulnerable. HOUYI achieved 71.3% ASR vs 29.8% for naive attacks.",
            limitations="Evaluation on proprietary apps limits reproducibility. No defense evaluation. Framework requires manual template crafting.",
            generated_at=datetime.now(),
            model_used="mock",
        ),
        L1Summary(
            paper_id="Yi_2023_Benchmarking",
            bibtex_key="Yi_2023_Benchmarking",
            category="Prompt Injection",
            year=2023,
            key_contributions=[
                "Created the first comprehensive benchmark for prompt injection attacks",
                "Evaluated both direct and indirect injection across 10 LLMs",
                "Proposed a severity scoring system for prompt injection vulnerabilities",
            ],
            methodology="The authors built a benchmark of 500 prompt injection test cases across 5 attack categories, tested on open-source and commercial LLMs using automated evaluation pipelines.",
            testing_methodology="Benchmark: 500 test cases across direct injection, indirect injection, jailbreaking, goal hijacking, and prompt leaking. Models tested: GPT-3.5, GPT-4, Claude, LLaMA-2-7B/13B/70B, Vicuna-7B/13B, Mistral-7B, Falcon-40B. Metrics: ASR, Defense Success Rate (DSR), severity score (1-5). Open-source: benchmark released at github.com/example/pibench.",
            temporal_context="Published in late 2023, this paper consolidated earlier findings by Greshake et al. (2023) into a systematic evaluation framework.",
            results_summary="GPT-4 showed 23% ASR (most robust), while LLaMA-2-7B showed 67% ASR (most vulnerable). Commercial LLMs were 30-40% more robust than open-source counterparts.",
            limitations="Static benchmark may not capture evolving attack techniques. Limited to English prompts only.",
            generated_at=datetime.now(),
            model_used="mock",
        ),
        L1Summary(
            paper_id="Zhan_2025_LLM",
            bibtex_key="Zhan_2025_LLM",
            category="Prompt Injection",
            year=2025,
            key_contributions=[
                "Conducted first comprehensive evaluation of 8 IPI defenses using adaptive attacks",
                "Showed all existing defenses can be bypassed with >50% ASR",
                "Adapted GCG and AutoDAN attack techniques for indirect prompt injection",
                "Demonstrated defenses provide false sense of security",
            ],
            methodology="Evaluated 8 representative defense mechanisms across detection-based, input-level, and output-level categories using Multi-objective GCG, Two-stage GCG, and adaptive AutoDAN attacks.",
            testing_methodology="Benchmarks: InjecAgent (1,054 cases, subset of 100: 50 direct harm, 50 data stealing), AgentDojo (16 cases across workspace, Slack, travel, banking). Metrics: ASR, ASR-defense, ASR-adaptive. Models: Vicuna-7B, GPT-3.5-turbo, GPT-4o-mini.",
            temporal_context="Published in 2025 at NAACL. Builds on GCG (Zou et al., 2023) and AutoDAN foundations to evaluate whether defenses proposed in 2023-2024 actually work.",
            results_summary="All 8 defenses bypassed with ASR-adaptive >50%. Adversarial Finetuning initially reduced ASR from 56% to 12%, but adaptive attack raised it back to 52%.",
            limitations="Focused on first-step agent actions. Limited to 3 LLMs. Computational cost of adaptive attacks not fully analyzed.",
            generated_at=datetime.now(),
            model_used="mock",
        ),
        L1Summary(
            paper_id="Chen_2024_StruQ",
            bibtex_key="Chen_2024_StruQ",
            category="Prompt Injection",
            year=2024,
            key_contributions=[
                "Proposed StruQ, a structured query approach that separates prompts from data",
                "Introduced secure front-end that tags user inputs vs system prompts",
                "Achieved strong defense while maintaining model utility",
            ],
            methodology="StruQ uses special delimiter tokens trained into the model to distinguish system prompts from user data. The model is instruction-tuned to only follow instructions in the system prompt section.",
            testing_methodology="Evaluated on Alpaca Eval and MT-Bench for utility. ASR measured against 5 attack types: naive, escape, ignore, fake completion, combined. Models: LLaMA-2-7B fine-tuned. Comparison: no defense, instruction defense, sandwich defense, known-answer detection.",
            temporal_context="Published in 2024, proposes a proactive architectural defense rather than reactive detection, addressing limitations of post-hoc defenses demonstrated by Greshake et al. (2023).",
            results_summary="Reduced ASR from 56.2% to 0.6% against combined attacks while maintaining 95.3% of original model utility on Alpaca Eval.",
            limitations="Requires model retraining/fine-tuning. Only tested on LLaMA-2-7B. May not generalize to API-only models.",
            generated_at=datetime.now(),
            model_used="mock",
        ),
    ]

    return summaries


def main():
    console.print(Panel("Test L2 Summarizer", style="bold blue"))
    start_total = time.time()

    # ====================================================================
    # Step 1: Crea L1 summary mock
    # ====================================================================
    console.print(Panel("Step 1: Creating mock L1 summaries", style="bold cyan"))

    l1_summaries = create_mock_l1_summaries()
    console.print(f"✓ Created {len(l1_summaries)} mock L1 summaries for 'Prompt Injection'")

    for s in l1_summaries:
        console.print(
            f"  - [{s.bibtex_key}] ({s.year}) — "
            f"{len(s.key_contributions)} contributions, "
            f"testing_methodology={len(s.testing_methodology)} chars"
        )

    # ====================================================================
    # Step 2: Genera L2 summary
    # ====================================================================
    console.print(Panel("Step 2: Generating L2 summary", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import generate_l2_summary

    try:
        l2 = generate_l2_summary(
            "Prompt Injection",
            l1_summaries,
            save_to_store=False,  # Non salva nel Qdrant per questo test
        )
        console.print("✓ L2 summary generated successfully!")
    except Exception as e:
        console.print(f"✗ L2 generation failed: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ====================================================================
    # Step 3: Validazione e output
    # ====================================================================
    console.print(Panel("Step 3: Validation & Output", style="bold cyan"))

    console.print(f"\n=== L2 Summary: {l2.category} (v{l2.version}) ===\n")

    # Metadata
    console.print(f"  Category: {l2.category}")
    console.print(f"  Paper count: {l2.paper_count}")
    console.print(f"  Paper IDs: {l2.paper_ids}")
    console.print(f"  Version: {l2.version}")
    console.print(f"  Model: {l2.model_used}")
    console.print(f"  Generated at: {l2.generated_at}")
    console.print()

    # Campi L2
    fields = [
        ("Temporal Trends", l2.temporal_trends),
        ("Taxonomy", l2.taxonomy),
        ("Gaps Filled", l2.gaps_filled),
        ("Open Gaps", l2.open_gaps),
        ("Methodological Convergences", l2.methodological_convergences),
        ("Testing Overview", l2.testing_overview),
        ("Synthesis", l2.synthesis),
    ]

    all_ok = True
    for field_name, field_value in fields:
        length = len(field_value) if field_value else 0
        status = "[OK]" if length >= 100 else "[WARN]"
        if length < 100:
            all_ok = False
        console.print(f"  {status} {field_name} ({length} chars):")
        # Stampa le prime 300 chars
        preview = (field_value or "")[:300]
        if len(field_value or "") > 300:
            preview += "..."
        console.print(f"    {preview}")
        console.print()

    # ====================================================================
    # Step 4: Test helper functions
    # ====================================================================
    console.print(Panel("Step 4: Testing helper functions", style="bold cyan"))

    from paperhunter.summarization.level2_summarizer import (
        _format_l1_summaries_for_prompt,
        _l2_summary_to_text,
        _make_l2_chunk_id,
        _validate_l2_summary,
        _parse_l2_json_response,
    )

    # Test formatting
    formatted = _format_l1_summaries_for_prompt(l1_summaries)
    console.print(f"  [OK] _format_l1_summaries_for_prompt: {len(formatted)} chars for {len(l1_summaries)} papers")

    # Test text conversion
    l2_text = _l2_summary_to_text(l2)
    console.print(f"  [OK] _l2_summary_to_text: {len(l2_text)} chars")

    # Test chunk ID
    chunk_id = _make_l2_chunk_id("Prompt Injection", 1)
    console.print(f"  [OK] _make_l2_chunk_id: {chunk_id}")

    # Test validation
    _validate_l2_summary(l2)
    console.print(f"  [OK] _validate_l2_summary: no exceptions")

    # Test JSON parsing with sample
    sample_json = json.dumps({
        "temporal_trends": "Test temporal trends",
        "taxonomy": "Test taxonomy",
        "gaps_filled": "Test gaps",
        "open_gaps": "Test open gaps",
        "methodological_convergences": "Test convergences",
        "testing_overview": "Test testing",
        "synthesis": "Test synthesis",
    })
    parsed = _parse_l2_json_response(f"```json\n{sample_json}\n```", "test")
    assert len(parsed) == 7, f"Expected 7 fields, got {len(parsed)}"
    console.print(f"  [OK] _parse_l2_json_response: 7/7 fields parsed from code-fenced JSON")

    # ====================================================================
    # Final Report
    # ====================================================================
    console.print(Panel("Final Report", style="bold green"))

    elapsed = time.time() - start_total

    # Count total chars across all L2 fields
    total_chars = sum(len(getattr(l2, f, "") or "") for f in [
        "temporal_trends", "taxonomy", "gaps_filled", "open_gaps",
        "methodological_convergences", "testing_overview", "synthesis",
    ])

    from paperhunter.utils.logging_setup import print_table
    print_table("L2 Summary Test Results", {
        "Category": l2.category,
        "Paper count": l2.paper_count,
        "Version": l2.version,
        "Model": l2.model_used,
        "Total L2 chars": total_chars,
        "All fields >= 100 chars": "Yes" if all_ok else "No",
        "Tempo totale": f"{elapsed:.1f}s",
    })

    if all_ok:
        console.print("✓ All tests passed!", style="bold green")
    else:
        console.print("✗ Some fields are too short (see warnings above).", style="bold yellow")


if __name__ == "__main__":
    import json
    main()
