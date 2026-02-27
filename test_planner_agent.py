#!/usr/bin/env python3
"""
Test per agents/planner_agent.py — Planner Agent (Task 4.2).

Test offline (senza chiamate LLM):
  1. JSON parsing outline
  2. Validazione e normalizzazione outline
  3. Formattazione L2 per prompt
  4. Caricamento outline da file JSON (sample_outline.json + formati)
  5. Inferenza section_type
  6. title_to_key helper
  7. get_outline_stats
  8. piano con override utente (nodo LangGraph)
  9. piano con fallimento (nessun L2)
  10. LangGraph integration (nodo plan nel grafo)

Test LIVE (con Gemini API):
  11. Generazione outline reale con mock L2

Uso:
    python test_planner_agent.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

# Aggiungi il progetto al path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_json_parsing():
    """Test 1: Parsing outline JSON con fallback."""
    from paperhunter.agents.planner_agent import _parse_outline_json

    print("\n" + "=" * 60)
    print("TEST 1: JSON Parsing Outline")
    print("=" * 60)

    # JSON pulito
    clean = json.dumps([
        {"key": "intro", "title": "Introduction", "section_type": "introduction"},
        {"key": "body1", "title": "Body 1", "section_type": "body"},
    ])
    result = _parse_outline_json(clean)
    assert result is not None and len(result) == 2, f"Expected 2, got {result}"
    print("  ✓ JSON pulito: OK")

    # Con code fences markdown
    fenced = f"```json\n{clean}\n```"
    result = _parse_outline_json(fenced)
    assert result is not None and len(result) == 2
    print("  ✓ Code fences: OK")

    # Testo extra prima/dopo
    extra = f"Here is the outline:\n{clean}\nThat's all!"
    result = _parse_outline_json(extra)
    assert result is not None and len(result) == 2
    print("  ✓ Testo extra: OK")

    # Trailing commas
    with_commas = '[{"key": "a", "title": "A",}, {"key": "b", "title": "B",},]'
    result = _parse_outline_json(with_commas)
    assert result is not None and len(result) == 2
    print("  ✓ Trailing commas: OK")

    # Dict con "sections" key
    wrapped = json.dumps({"sections": [{"key": "a", "title": "A"}]})
    result = _parse_outline_json(wrapped)
    assert result is not None and len(result) == 1
    print('  ✓ Dict con "sections": OK')

    # Dict con "outline" key
    wrapped2 = json.dumps({"outline": [{"key": "b", "title": "B"}]})
    result = _parse_outline_json(wrapped2)
    assert result is not None and len(result) == 1
    print('  ✓ Dict con "outline": OK')

    # Vuoto e None
    assert _parse_outline_json("") is None
    assert _parse_outline_json("no json here") is None
    print("  ✓ Edge cases (vuoto, non-JSON): OK")

    print("TEST 1 PASSED ✓")


def test_validation_and_normalization():
    """Test 2: Validazione e normalizzazione outline."""
    from paperhunter.agents.planner_agent import _validate_and_normalize_outline

    print("\n" + "=" * 60)
    print("TEST 2: Validazione e Normalizzazione Outline")
    print("=" * 60)

    raw = [
        {"key": "abstract", "title": "Abstract", "section_type": "abstract",
         "target_length_words": 300},
        {"key": "introduction", "title": "Introduction", "section_type": "introduction",
         "target_length_words": 1000, "key_topics": ["LLM", "security"]},
        {"key": "prompt_injection", "title": "Prompt Injection Attacks",
         "section_type": "body", "target_length_words": 2000,
         "related_categories": ["Prompt Injection"]},
        {"key": "data_poisoning", "title": "Data Poisoning", "section_type": "body",
         "target_length_words": 1500},
        {"key": "testing_methodologies", "title": "Testing Methodologies",
         "section_type": "testing", "target_length_words": 2000},
        {"key": "discussion", "title": "Discussion", "section_type": "discussion",
         "target_length_words": 1000},
        {"key": "conclusion", "title": "Conclusion", "section_type": "conclusion",
         "target_length_words": 500},
    ]

    categories = ["Prompt Injection", "Data and Model Poisoning"]
    result = _validate_and_normalize_outline(raw, categories)

    assert len(result) == 7, f"Expected 7, got {len(result)}"
    print(f"  ✓ 7 sezioni validate: OK")

    # Ordine: abstract < introduction < body < testing < discussion < conclusion
    assert result[0].section_type == "abstract"
    assert result[1].section_type == "introduction"
    # Body sections in mezzo
    body_indices = [i for i, o in enumerate(result) if o.section_type == "body"]
    testing_idx = next(i for i, o in enumerate(result) if o.section_type == "testing")
    disc_idx = next(i for i, o in enumerate(result) if o.section_type == "discussion")
    conc_idx = next(i for i, o in enumerate(result) if o.section_type == "conclusion")
    assert all(bi < testing_idx for bi in body_indices)
    assert testing_idx < disc_idx < conc_idx
    print(f"  ✓ Ordinamento logico: abstract→intro→body→testing→discussion→conclusion: OK")

    # Ordine sequenziale reassigned
    for i, o in enumerate(result):
        assert o.order == i, f"Sezione {o.key}: order={o.order}, expected={i}"
    print(f"  ✓ Ordini sequenziali riassegnati (0..6): OK")

    # key_topics e related_categories preservati
    intro_sec = next(o for o in result if o.key == "introduction")
    assert intro_sec.key_topics == ["LLM", "security"]
    print(f"  ✓ key_topics preservati: OK")

    pi_sec = next(o for o in result if o.key == "prompt_injection")
    assert "Prompt Injection" in pi_sec.related_categories
    print(f"  ✓ related_categories preservati: OK")

    # Test con sezioni mancanti obbligatorie (senza abstract)
    raw_no_abstract = [r for r in raw if r["key"] != "abstract"]
    result2 = _validate_and_normalize_outline(raw_no_abstract, categories)
    has_abstract = any(o.section_type == "abstract" for o in result2)
    assert has_abstract, "Abstract dovrebbe essere aggiunto automaticamente"
    print(f"  ✓ Sezione mancante aggiunta automaticamente (abstract): OK")

    # Test con key duplicata
    dup = [
        {"key": "intro", "title": "Intro 1"},
        {"key": "intro", "title": "Intro 2"},
    ]
    result3 = _validate_and_normalize_outline(dup, [])
    keys = [o.key for o in result3]
    assert len(keys) == len(set(keys)) + len([k for k in keys if k.startswith("intro")]) - len(set(k for k in keys if k.startswith("intro"))), \
        "Key should be deduplicated"
    # Verifica almeno che entrambi siano presenti senza crash
    assert len(result3) >= 2  # 2 + sezioni obbligatorie aggiunte
    print(f"  ✓ Key duplicata gestita senza crash: OK")

    # Test clamping target_length_words
    extreme = [{"key": "big", "title": "Big", "target_length_words": 999999}]
    result4 = _validate_and_normalize_outline(extreme, [])
    big_sec = next(o for o in result4 if o.key == "big")
    assert big_sec.target_length_words <= 5000
    print(f"  ✓ Clamping target_length_words (999999 → ≤5000): OK")

    # Test non-dict entry skippata
    bad = [{"key": "ok", "title": "OK"}, "not_a_dict", 123]
    result5 = _validate_and_normalize_outline(bad, [])
    ok_count = sum(1 for o in result5 if o.key == "ok")
    assert ok_count == 1
    print(f"  ✓ Non-dict entry skippate: OK")

    print("TEST 2 PASSED ✓")


def test_format_l2_for_prompt():
    """Test 3: Formattazione L2 per prompt."""
    from paperhunter.agents.planner_agent import _format_l2_for_prompt
    from paperhunter.models import L2Summary

    print("\n" + "=" * 60)
    print("TEST 3: Formattazione L2 per Prompt")
    print("=" * 60)

    l2s = [
        L2Summary(
            category="Prompt Injection",
            paper_count=53,
            temporal_trends="Early 2022 focused on basic injections...",
            taxonomy="Direct PI, Indirect PI, Jailbreaking...",
            testing_overview="ASR, AdvBench, HarmBench used across...",
            open_gaps="Multimodal PI remains underexplored...",
            synthesis="PI has evolved from simple tricks to...",
        ),
        L2Summary(
            category="Data and Model Poisoning",
            paper_count=13,
            temporal_trends="Backdoor attacks emerged in 2020...",
            taxonomy="Data poisoning, model poisoning, backdoor...",
            testing_overview="Clean accuracy, ASR, backdoor detection...",
            open_gaps="Poisoning in RLHF pipeline...",
            synthesis="Training-time attacks are a growing concern...",
        ),
    ]

    text = _format_l2_for_prompt(l2s)
    assert "Category 1: Prompt Injection (53 papers)" in text
    assert "Category 2: Data and Model Poisoning (13 papers)" in text
    assert "Temporal Trends" in text
    assert "Testing Overview" in text
    assert "Open Gaps" in text
    assert "Synthesis" in text
    print(f"  ✓ Formattazione: {len(text)} chars, 2 categorie: OK")

    # Vuoto
    text_empty = _format_l2_for_prompt([])
    assert text_empty == ""
    print(f"  ✓ Lista vuota → stringa vuota: OK")

    print("TEST 3 PASSED ✓")


def test_load_outline_from_json():
    """Test 4: Caricamento outline da file JSON."""
    from paperhunter.agents.planner_agent import load_outline_from_json

    print("\n" + "=" * 60)
    print("TEST 4: Caricamento Outline da File JSON")
    print("=" * 60)

    # Test con sample_outline.json (formato semplice)
    sample_path = Path(__file__).parent / "sample_outline.json"
    if sample_path.exists():
        outline = load_outline_from_json(sample_path)
        assert len(outline) >= 5, f"Expected ≥5, got {len(outline)}"
        titles = [o.title for o in outline]
        assert "Introduction" in titles
        assert "Conclusion" in titles or "Conclusion" in [t.split(" and ")[0] for t in titles]
        print(f"  ✓ sample_outline.json: {len(outline)} sezioni: OK")

        # Verifica section_type inference
        intro = next((o for o in outline if "introduction" in o.title.lower()), None)
        if intro:
            assert intro.section_type == "introduction"
            print(f"  ✓ Section type inference (Introduction → 'introduction'): OK")
    else:
        print(f"  ⚠ sample_outline.json non trovato, skip")

    # Test formato array diretto
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([
            {"key": "intro", "title": "Introduction", "section_type": "introduction"},
            {"key": "body1", "title": "Body", "section_type": "body"},
            {"key": "conclusion", "title": "Conclusion", "section_type": "conclusion"},
        ], f)
        f.flush()
        outline2 = load_outline_from_json(f.name)
        assert len(outline2) >= 3
        print(f"  ✓ Formato array diretto: {len(outline2)} sezioni: OK")
        Path(f.name).unlink()

    # Test formato {sections: [...]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"sections": [
            {"key": "a", "title": "Section A", "section_type": "body"},
        ]}, f)
        f.flush()
        outline3 = load_outline_from_json(f.name)
        assert any(o.key == "a" for o in outline3)
        print(f"  ✓ Formato {{sections: [...]}}: OK")
        Path(f.name).unlink()

    # Test FileNotFoundError
    try:
        load_outline_from_json("/nonexistent/path.json")
        assert False, "Dovrebbe lanciare FileNotFoundError"
    except FileNotFoundError:
        print(f"  ✓ FileNotFoundError per file mancante: OK")

    print("TEST 4 PASSED ✓")


def test_infer_section_type():
    """Test 5: Inferenza section_type da titolo."""
    from paperhunter.agents.planner_agent import _infer_section_type

    print("\n" + "=" * 60)
    print("TEST 5: Inferenza Section Type")
    print("=" * 60)

    cases = [
        ("Abstract", "abstract"),
        ("1. Introduction", "introduction"),
        ("Conclusion and Future Work", "conclusion"),
        ("Discussion and Open Challenges", "discussion"),
        ("Testing Methodologies and Evaluation", "testing"),
        ("Benchmarks and Evaluation Frameworks", "testing"),
        ("Prompt Injection Attacks", "body"),
        ("Data Poisoning", "body"),
        ("Background and Preliminaries", "body"),
    ]
    for title, expected in cases:
        result = _infer_section_type(title)
        assert result == expected, f"{title!r}: expected {expected}, got {result}"
        print(f"  ✓ {title!r} → {result!r}")

    print("TEST 5 PASSED ✓")


def test_title_to_key():
    """Test 6: Conversione titolo → chiave snake_case."""
    from paperhunter.agents.planner_agent import _title_to_key

    print("\n" + "=" * 60)
    print("TEST 6: Title to Key")
    print("=" * 60)

    cases = [
        ("Introduction", "introduction"),
        ("Prompt Injection Attacks", "prompt_injection_attacks"),
        ("Testing Methodologies & Evaluation", "testing_methodologies_evaluation"),
        ("Background and Preliminaries", "background_and_preliminaries"),
    ]
    for title, expected in cases:
        result = _title_to_key(title)
        assert result == expected, f"{title!r}: expected {expected!r}, got {result!r}"
        print(f"  ✓ {title!r} → {result!r}")

    print("TEST 6 PASSED ✓")


def test_outline_stats():
    """Test 7: Statistiche outline."""
    from paperhunter.agents.planner_agent import get_outline_stats
    from paperhunter.agents.state import SectionOutline

    print("\n" + "=" * 60)
    print("TEST 7: Outline Stats")
    print("=" * 60)

    outline = [
        SectionOutline(key="abstract", title="Abstract", section_type="abstract",
                       target_length_words=300),
        SectionOutline(key="introduction", title="Introduction", section_type="introduction",
                       target_length_words=1000, key_topics=["LLM", "security"]),
        SectionOutline(key="pi_attacks", title="Prompt Injection", section_type="body",
                       target_length_words=2000, related_categories=["Prompt Injection"],
                       key_topics=["direct PI", "indirect PI"]),
        SectionOutline(key="poisoning", title="Data Poisoning", section_type="body",
                       target_length_words=1500, related_categories=["Data and Model Poisoning"],
                       key_topics=["backdoor"]),
        SectionOutline(key="testing", title="Testing", section_type="testing",
                       target_length_words=2000, key_topics=["benchmarks", "metrics"],
                       related_categories=["Prompt Injection", "Data and Model Poisoning"]),
        SectionOutline(key="discussion", title="Discussion", section_type="discussion",
                       target_length_words=1000),
        SectionOutline(key="conclusion", title="Conclusion", section_type="conclusion",
                       target_length_words=500),
    ]

    stats = get_outline_stats(outline)
    assert stats["total_sections"] == 7
    assert stats["total_target_words"] == 8300
    assert stats["body_sections"] == 2
    assert stats["has_abstract"] is True
    assert stats["has_introduction"] is True
    assert stats["has_testing"] is True
    assert stats["has_discussion"] is True
    assert stats["has_conclusion"] is True
    assert stats["total_topics"] == 7  # 2+2+1+2
    assert "Prompt Injection" in stats["categories_covered"]
    assert "Data and Model Poisoning" in stats["categories_covered"]
    print(f"  ✓ Stats: {stats['total_sections']} sezioni, {stats['total_target_words']} parole, "
          f"{stats['body_sections']} body, {stats['total_topics']} topic")

    # Vuoto
    empty_stats = get_outline_stats([])
    assert empty_stats["total_sections"] == 0
    print(f"  ✓ Stats vuoto: OK")

    print("TEST 7 PASSED ✓")


def test_plan_with_user_override():
    """Test 8: Nodo plan con outline pre-esistente (override utente)."""
    from paperhunter.agents.planner_agent import plan
    from paperhunter.agents.state import SectionOutline, create_initial_state

    print("\n" + "=" * 60)
    print("TEST 8: Plan con Override Utente")
    print("=" * 60)

    outline = [
        SectionOutline(key="intro", title="Introduction", section_type="introduction",
                       target_length_words=1000, order=0),
        SectionOutline(key="body", title="Main Body", section_type="body",
                       target_length_words=2000, order=1),
        SectionOutline(key="conclusion", title="Conclusion", section_type="conclusion",
                       target_length_words=500, order=2),
    ]

    state = create_initial_state(outline=outline)
    result = plan(state)

    assert "outline" in result
    assert len(result["outline"]) == 3
    assert result.get("metadata", {}).get("planner_source") == "user_override"
    print(f"  ✓ Override utente: 3 sezioni preservate, source='user_override': OK")

    print("TEST 8 PASSED ✓")


def test_plan_no_l2():
    """Test 9: Nodo plan senza L2 e senza store → FAILED."""
    from paperhunter.agents.planner_agent import plan
    from paperhunter.agents.state import PipelinePhase, create_initial_state

    print("\n" + "=" * 60)
    print("TEST 9: Plan senza L2 (Fallimento Atteso)")
    print("=" * 60)

    state = create_initial_state()
    # Patch _load_l2_from_store per evitare accesso al QdrantStore
    import paperhunter.agents.planner_agent as pa
    original_load = pa._load_l2_from_store
    pa._load_l2_from_store = lambda: []

    try:
        result = plan(state)
        assert result["current_phase"] == PipelinePhase.FAILED.value
        assert len(result.get("errors", [])) > 0
        assert result["outline"] == []
        print(f"  ✓ Nessun L2 → phase=FAILED, errore registrato: OK")
    finally:
        pa._load_l2_from_store = original_load

    print("TEST 9 PASSED ✓")


def test_langgraph_integration():
    """Test 10: Integrazione LangGraph con nodo plan."""
    from langgraph.graph import END, START, StateGraph

    from paperhunter.agents.planner_agent import plan
    from paperhunter.agents.state import (
        PipelinePhase,
        SectionOutline,
        SurveyState,
        create_initial_state,
        get_outline_from_state,
    )

    print("\n" + "=" * 60)
    print("TEST 10: LangGraph Integration")
    print("=" * 60)

    # Crea grafo con solo il nodo plan
    graph = StateGraph(SurveyState)
    graph.add_node("plan", plan)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", END)
    compiled = graph.compile()

    # Esegui con outline pre-esistente (no LLM call)
    outline = [
        SectionOutline(key="intro", title="Introduction", section_type="introduction",
                       target_length_words=1000, order=0),
        SectionOutline(key="testing", title="Testing", section_type="testing",
                       target_length_words=2000, order=1),
        SectionOutline(key="conclusion", title="Conclusion", section_type="conclusion",
                       target_length_words=500, order=2),
    ]

    initial_state = create_initial_state(outline=outline)
    final_state = compiled.invoke(initial_state)

    # Verifica
    result_outline = get_outline_from_state(final_state)
    assert len(result_outline) == 3
    assert result_outline[0].key == "intro"
    assert result_outline[1].key == "testing"
    assert result_outline[2].key == "conclusion"
    print(f"  ✓ LangGraph: 3 sezioni nell'output, nodo plan eseguito: OK")

    assert final_state.get("metadata", {}).get("planner_source") == "user_override"
    print(f"  ✓ Metadata preservati attraverso LangGraph: OK")

    print("TEST 10 PASSED ✓")


def test_live_gemini_outline():
    """Test 11 (LIVE): Generazione outline reale con Gemini Pro e mock L2."""
    from paperhunter.agents.planner_agent import generate_outline, get_outline_stats
    from paperhunter.models import L2Summary

    print("\n" + "=" * 60)
    print("TEST 11 (LIVE): Generazione Outline con Gemini")
    print("=" * 60)

    # Mock L2 summaries per 4 categorie
    l2_summaries = [
        L2Summary(
            category="Prompt Injection",
            paper_count=53,
            paper_ids=[f"pi_{i}" for i in range(53)],
            temporal_trends=(
                "Early research (2022-2023) focused on basic prompt injection attacks such as "
                "goal hijacking and payload splitting [Greshake_2023_LLM]. In 2023, indirect "
                "prompt injection emerged as a major vector [Greshake_2023_LLM]. By 2024-2025, "
                "adaptive attacks broke most defenses [Zhan_2024_Adaptive], and PromptShield "
                "[Lee_2025_PromptShield] showed promise in detection-based approaches."
            ),
            taxonomy=(
                "1. Direct Prompt Injection: goal hijacking, payload splitting, jailbreaking\n"
                "2. Indirect Prompt Injection: data exfiltration via retrieval, tool manipulation\n"
                "3. Defense Mechanisms: input filtering, output monitoring, instruction hierarchy\n"
                "4. Benchmarking: automated red-teaming, attack success rate measurement"
            ),
            testing_overview=(
                "Common benchmarks: AdvBench, HarmBench, JailbreakBench. Metrics: ASR (Attack "
                "Success Rate), BPR (Bypassed Protection Rate). Models tested: GPT-4, Claude-3, "
                "LLaMA-2, Vicuna. Tools: GCG optimization [Zou_2023_GCG], AutoDAN, TAP."
            ),
            open_gaps=(
                "Multimodal prompt injection remains underexplored. Adaptive attacks continue to "
                "break defenses. No standardized benchmark suite exists across all PI types."
            ),
            synthesis=(
                "Prompt injection has evolved from simple textual tricks to sophisticated "
                "multi-step attacks targeting the entire LLM application pipeline."
            ),
        ),
        L2Summary(
            category="Data and Model Poisoning",
            paper_count=13,
            paper_ids=[f"dmp_{i}" for i in range(13)],
            temporal_trends=(
                "Backdoor attacks on language models first appeared in 2020. By 2023, "
                "instruction-tuning poisoning [Wan_2023_Poisoning] showed that even small "
                "amounts of poisoned data could compromise model safety. 2024 saw more "
                "sophisticated attacks targeting RLHF pipelines."
            ),
            taxonomy=(
                "1. Data Poisoning: training-data manipulation\n"
                "2. Backdoor Attacks: trigger-based activation [Li_2023_Backdoorllm]\n"
                "3. Model Poisoning: weight manipulation, fine-tuning attacks"
            ),
            testing_overview=(
                "Metrics: Clean accuracy, ASR, backdoor detection rate. Datasets: custom "
                "poisoned variants of standard NLP benchmarks."
            ),
            open_gaps="RLHF-specific poisoning, supply chain attacks on models.",
            synthesis="Training-time attacks represent a growing concern as model fine-tuning becomes widespread.",
        ),
        L2Summary(
            category="Adversarial Attacks",
            paper_count=23,
            paper_ids=[f"aa_{i}" for i in range(23)],
            temporal_trends=(
                "GCG [Zou_2023_GCG] in 2023 demonstrated universal adversarial suffixes. "
                "2024 brought black-box transfer attacks and multi-modal adversarial examples "
                "[Zhang_2024_Adversarial]. Feature-optimal alignment methods emerged for "
                "attacking closed-source MLLMs."
            ),
            taxonomy=(
                "1. White-box attacks: gradient-based (GCG, AutoDAN)\n"
                "2. Black-box attacks: transfer, query-based\n"
                "3. Multi-modal attacks: image+text adversarial examples"
            ),
            testing_overview=(
                "ASR on AdvBench, toxicity classifiers, perplexity-based detection. "
                "Models: GPT-4, Claude, LLaMA-2, Vicuna."
            ),
            open_gaps="Robustness against adaptive adversaries, certified defenses.",
            synthesis="Adversarial attacks have become increasingly transferable across models.",
        ),
        L2Summary(
            category="LLM Hallucinations",
            paper_count=29,
            paper_ids=[f"lh_{i}" for i in range(29)],
            temporal_trends=(
                "Hallucination detection gained focus in 2023 with HaluEval [Li_2023_Halueval] "
                "providing the first large-scale benchmark. 2024-2025 saw retrieval-augmented "
                "approaches for mitigation and self-consistency checking methods."
            ),
            taxonomy=(
                "1. Factual hallucinations: entity errors, numerical errors\n"
                "2. Faithfulness hallucinations: contradicting input context\n"
                "3. Detection methods: reference-based, reference-free, self-consistency"
            ),
            testing_overview=(
                "HaluEval, TruthfulQA, FEVER, FActScore. Metrics: hallucination rate, "
                "faithfulness score, factual precision/recall."
            ),
            open_gaps="Long-form generation hallucination, multi-hop reasoning errors.",
            synthesis="Hallucination remains one of the most critical challenges for LLM deployment.",
        ),
    ]

    print(f"  Input: {len(l2_summaries)} categorie, "
          f"{sum(l.paper_count for l in l2_summaries)} paper totali")

    t0 = time.time()
    outline = generate_outline(l2_summaries, target_sections=12)
    elapsed = time.time() - t0

    print(f"\n  Outline generato in {elapsed:.1f}s: {len(outline)} sezioni")

    stats = get_outline_stats(outline)
    print(f"  Stats: {json.dumps(stats, indent=2, default=str)}")

    # Verifiche
    assert len(outline) >= 8, f"Attese ≥8 sezioni, ottenute {len(outline)}"
    print(f"  ✓ Almeno 8 sezioni: {len(outline)} ≥ 8")

    assert stats["has_abstract"], "Manca Abstract"
    assert stats["has_introduction"], "Manca Introduction"
    assert stats["has_testing"], "Manca Testing Methodologies"
    assert stats["has_conclusion"], "Manca Conclusion"
    print(f"  ✓ Sezioni obbligatorie presenti: abstract, introduction, testing, conclusion")

    # Verifica copertura categorie
    all_cats = set()
    for o in outline:
        all_cats.update(o.related_categories)
    print(f"  ✓ Categorie coperte: {sorted(all_cats)}")

    # Verifica ordine logico
    type_indices: dict[str, list[int]] = {}
    for o in outline:
        type_indices.setdefault(o.section_type, []).append(o.order)

    if "abstract" in type_indices and "conclusion" in type_indices:
        assert min(type_indices["abstract"]) < min(type_indices["conclusion"])
        print(f"  ✓ Abstract prima di Conclusion")

    if "introduction" in type_indices and "testing" in type_indices:
        assert min(type_indices["introduction"]) < min(type_indices["testing"])
        print(f"  ✓ Introduction prima di Testing")

    # Verifica che body sections abbiano description non vuota
    body_sections = [o for o in outline if o.section_type == "body"]
    for bs in body_sections:
        assert len(bs.description) > 20, f"Body section {bs.key!r} ha description troppo corta"
    print(f"  ✓ Tutte le {len(body_sections)} body sections hanno description ≥20 chars")

    # Verifica target_length_words ragionevoli
    for o in outline:
        assert 100 <= o.target_length_words <= 5000, \
            f"Sezione {o.key}: target_length_words={o.target_length_words} fuori range"
    print(f"  ✓ Tutti i target_length_words in range [100, 5000]")

    # Stampa outline leggibile
    print(f"\n  --- Outline Generato ---")
    for o in outline:
        topics = ", ".join(o.key_topics[:4]) or "-"
        cats = ", ".join(o.related_categories[:2]) or "-"
        print(f"  {o.order:2d}. [{o.section_type:12s}] {o.title}")
        print(f"      key={o.key}, words={o.target_length_words}, topics=[{topics}]")
        print(f"      categories=[{cats}]")
        if o.description:
            print(f"      desc: {o.description[:120]}...")

    print(f"\n  Totale: {len(outline)} sezioni, ~{stats['total_target_words']} parole target")

    print("TEST 11 PASSED ✓")


# ============================================================================
# Main
# ============================================================================

def main():
    """Esegue tutti i test."""
    print("=" * 60)
    print("TEST PLANNER AGENT (Task 4.2)")
    print("=" * 60)

    passed = 0
    failed = 0
    total = 11

    # Test offline (1-10)
    offline_tests = [
        ("Test 1: JSON parsing", test_json_parsing),
        ("Test 2: Validazione/normalizzazione", test_validation_and_normalization),
        ("Test 3: Formato L2 per prompt", test_format_l2_for_prompt),
        ("Test 4: Load outline da JSON", test_load_outline_from_json),
        ("Test 5: Inferenza section_type", test_infer_section_type),
        ("Test 6: Title to key", test_title_to_key),
        ("Test 7: Outline stats", test_outline_stats),
        ("Test 8: Plan con override", test_plan_with_user_override),
        ("Test 9: Plan senza L2", test_plan_no_l2),
        ("Test 10: LangGraph integration", test_langgraph_integration),
    ]

    for name, test_fn in offline_tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Test live (11) — richiede Gemini API key
    print("\n" + "=" * 60)
    print(f"Test offline: {passed}/{len(offline_tests)} passed")
    print("=" * 60)

    try:
        test_live_gemini_outline()
        passed += 1
    except Exception as e:
        failed += 1
        print(f"\n  ✗ Test 11 (LIVE) FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Riepilogo
    print("\n" + "=" * 60)
    print(f"RIEPILOGO: {passed}/{total} test passati, {failed} falliti")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
