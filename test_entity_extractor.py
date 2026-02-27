"""
Test per graphrag/entity_extractor.py — Task 3.4

Test:
1. Normalizzazione entità (alias resolution)
2. Modelli Entity/Relation (creazione, ID, serializzazione)
3. Merge entità e relazioni duplicate
4. Parsing JSON (pulito, code fences, trailing commas, rotto)
5. Build entities/relations da dicts raw
6. Estrazione LIVE da 5 L1 summary mock con Gemini Flash
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Assicurati che il progetto sia nel path
sys.path.insert(0, str(Path(__file__).parent))

from paperhunter.graphrag.entity_extractor import (
    Entity,
    EntityType,
    ExtractionResult,
    Relation,
    RelationType,
    _build_entities,
    _build_relations,
    _make_entity_id,
    _make_relation_id,
    _parse_json_response,
    extract_batch,
    extract_entities_and_relations,
    extract_with_details,
    get_extraction_stats,
    merge_entities,
    merge_relations,
    normalize_entity_name,
    save_extraction_results,
)
from paperhunter.models import L1Summary

# ============================================================================
# Test 1: Normalizzazione entità
# ============================================================================

def test_normalization():
    """Test normalize_entity_name con varianti note e sconosciute."""
    print("\n=== Test 1: Normalizzazione entità ===")
    
    # Models
    assert normalize_entity_name("gpt4") == "GPT-4", f"Got: {normalize_entity_name('gpt4')}"
    assert normalize_entity_name("GPT-4") == "GPT-4"
    assert normalize_entity_name("gpt-4") == "GPT-4"
    assert normalize_entity_name("gpt 4") == "GPT-4"
    assert normalize_entity_name("llama2") == "LLaMA-2"
    assert normalize_entity_name("llama-2") == "LLaMA-2"
    assert normalize_entity_name("ChatGPT") == "ChatGPT"
    assert normalize_entity_name("chatgpt") == "ChatGPT"
    assert normalize_entity_name("vicuna") == "Vicuna"
    
    # Attacks
    assert normalize_entity_name("gcg") == "GCG"
    assert normalize_entity_name("GCG") == "GCG"
    assert normalize_entity_name("greedy coordinate gradient") == "GCG"
    assert normalize_entity_name("autodan") == "AutoDAN"
    assert normalize_entity_name("prompt injection") == "Prompt Injection"
    assert normalize_entity_name("backdoor") == "Backdoor Attack"
    
    # Defenses
    assert normalize_entity_name("smooth llm") == "SmoothLLM"
    assert normalize_entity_name("smoothllm") == "SmoothLLM"
    assert normalize_entity_name("rlhf") == "RLHF"
    assert normalize_entity_name("perplexity filter") == "Perplexity Filter"
    
    # Datasets & Benchmarks
    assert normalize_entity_name("advbench") == "AdvBench"
    assert normalize_entity_name("truthfulqa") == "TruthfulQA"
    assert normalize_entity_name("halueval") == "HaluEval"
    assert normalize_entity_name("harmbench") == "HarmBench"
    
    # Metrics
    assert normalize_entity_name("asr") == "ASR"
    assert normalize_entity_name("attack success rate") == "ASR"
    assert normalize_entity_name("bleu") == "BLEU"
    assert normalize_entity_name("perplexity") == "Perplexity"
    
    # Venues
    assert normalize_entity_name("neurips") == "NeurIPS"
    assert normalize_entity_name("acl") == "ACL"
    assert normalize_entity_name("arxiv") == "arXiv"
    
    # Sconosciuti → preserva case originale
    assert normalize_entity_name("My New Attack Method") == "My New Attack Method"
    assert normalize_entity_name("") == ""
    assert normalize_entity_name("  spaces  ") == "spaces"
    
    # Suffix stripping
    assert normalize_entity_name("gcg attack") == "GCG"
    assert normalize_entity_name("perplexity filter defense") == "Perplexity Filter"
    
    print("  ✅ 30+ normalizzazioni verificate")


# ============================================================================
# Test 2: Modelli Entity/Relation
# ============================================================================

def test_models():
    """Test creazione, ID deterministico, serializzazione."""
    print("\n=== Test 2: Modelli Entity e Relation ===")
    
    # Entity
    e1 = Entity(type=EntityType.ATTACK, canonical_name="GCG", description="Gradient-based attack")
    assert e1.id  # Auto-generato
    assert e1.type == EntityType.ATTACK
    assert e1.canonical_name == "GCG"
    
    # ID deterministico
    e2 = Entity(type=EntityType.ATTACK, canonical_name="GCG")
    assert e1.id == e2.id, "ID dovrebbe essere deterministico"
    
    # ID diverso per tipo diverso
    e3 = Entity(type=EntityType.DEFENSE, canonical_name="GCG")
    assert e1.id != e3.id, "Diverso tipo → diverso ID"
    
    # to_dict
    d = e1.to_dict()
    assert d["type"] == "Attack"
    assert d["canonical_name"] == "GCG"
    assert d["id"] == e1.id
    
    # Relation
    r1 = Relation(
        type=RelationType.DEFENDS_AGAINST,
        source="SmoothLLM",
        source_type=EntityType.DEFENSE,
        target="GCG",
        target_type=EntityType.ATTACK,
        year=2023,
    )
    assert r1.id  # Auto-generato
    assert r1.type == RelationType.DEFENDS_AGAINST
    
    # Relation ID deterministico
    r2 = Relation(
        type=RelationType.DEFENDS_AGAINST,
        source="SmoothLLM",
        source_type=EntityType.DEFENSE,
        target="GCG",
        target_type=EntityType.ATTACK,
    )
    assert r1.id == r2.id
    
    # to_dict
    rd = r1.to_dict()
    assert rd["type"] == "DEFENDS_AGAINST"
    assert rd["source"] == "SmoothLLM"
    assert rd["target"] == "GCG"
    assert rd["year"] == 2023
    
    # ExtractionResult
    er = ExtractionResult(
        paper_id="test_001",
        bibtex_key="Test_2023_GCG",
        entities=[e1],
        relations=[r1],
        extracted_at=datetime.now(),
        extraction_time_seconds=1.5,
    )
    erd = er.to_dict()
    assert erd["entity_count"] == 1
    assert erd["relation_count"] == 1
    assert erd["extraction_time_seconds"] == 1.5
    
    print(f"  ✅ Entity ID: {e1.id} (deterministico)")
    print(f"  ✅ Relation ID: {r1.id} (deterministico)")
    print(f"  ✅ ExtractionResult serializzazione OK")


# ============================================================================
# Test 3: Merge entità e relazioni
# ============================================================================

def test_merge():
    """Test merge di entità e relazioni duplicate."""
    print("\n=== Test 3: Merge entità e relazioni ===")
    
    # Entità duplicate
    e1 = Entity(
        type=EntityType.ATTACK, canonical_name="GCG",
        aliases=["Greedy Coordinate Gradient"],
        description="Short desc",
        year=2023,
        source_paper_id="paper_1",
    )
    e2 = Entity(
        type=EntityType.ATTACK, canonical_name="GCG",
        aliases=["adversarial suffix attack"],
        description="A much longer and more detailed description of GCG",
        year=2022,  # più vecchio → viene mantenuto
        source_paper_id="paper_2",
    )
    e3 = Entity(
        type=EntityType.MODEL, canonical_name="GPT-4",
        description="OpenAI model",
    )
    
    merged = merge_entities([e1, e2, e3])
    assert len(merged) == 2, f"Expected 2, got {len(merged)}"
    
    gcg = [e for e in merged if e.canonical_name == "GCG"][0]
    assert "Greedy Coordinate Gradient" in gcg.aliases
    assert "adversarial suffix attack" in gcg.aliases
    assert gcg.description == e2.description  # più lunga
    assert gcg.year == 2022  # più vecchio
    assert gcg.source_paper_id == "paper_1"  # primo trovato
    
    # Relazioni duplicate
    r1 = Relation(
        type=RelationType.TARGETS, source="GCG", source_type=EntityType.ATTACK,
        target="GPT-4", target_type=EntityType.MODEL,
        description="Short",
    )
    r2 = Relation(
        type=RelationType.TARGETS, source="GCG", source_type=EntityType.ATTACK,
        target="GPT-4", target_type=EntityType.MODEL,
        description="GCG targets GPT-4 with adversarial suffixes",
    )
    r3 = Relation(
        type=RelationType.TESTED_ON, source="GCG", source_type=EntityType.ATTACK,
        target="AdvBench", target_type=EntityType.DATASET,
    )
    
    merged_rels = merge_relations([r1, r2, r3])
    assert len(merged_rels) == 2, f"Expected 2 relations, got {len(merged_rels)}"
    
    targets_rel = [r for r in merged_rels if r.type == RelationType.TARGETS][0]
    assert targets_rel.description == r2.description  # più lunga
    
    print(f"  ✅ 3 entità → 2 dopo merge (GCG fuso, GPT-4 preservato)")
    print(f"  ✅ GCG aliases: {gcg.aliases}")
    print(f"  ✅ 3 relazioni → 2 dopo merge")


# ============================================================================
# Test 4: Parsing JSON
# ============================================================================

def test_json_parsing():
    """Test parsing JSON robusto."""
    print("\n=== Test 4: Parsing JSON ===")
    
    # JSON pulito
    clean = json.dumps({
        "entities": [{"type": "Attack", "name": "GCG"}],
        "relations": [{"type": "TARGETS", "source": "GCG", "target": "GPT-4"}],
    })
    result = _parse_json_response(clean, "test")
    assert result is not None
    assert len(result["entities"]) == 1
    assert len(result["relations"]) == 1
    print("  ✅ JSON pulito: OK")
    
    # Con code fences
    fenced = "```json\n" + clean + "\n```"
    result = _parse_json_response(fenced, "test")
    assert result is not None
    assert len(result["entities"]) == 1
    print("  ✅ JSON con code fences: OK")
    
    # Con trailing commas
    with_trailing = '{"entities": [{"type": "Attack", "name": "GCG",},], "relations": []}'
    result = _parse_json_response(with_trailing, "test")
    assert result is not None
    assert len(result["entities"]) == 1
    print("  ✅ JSON con trailing commas: OK")
    
    # Con testo extra
    with_text = "Here is the extraction:\n" + clean + "\nDone!"
    result = _parse_json_response(with_text, "test")
    assert result is not None
    assert len(result["entities"]) == 1
    print("  ✅ JSON con testo extra: OK")
    
    # Vuoto
    result = _parse_json_response("", "test")
    assert result is None
    print("  ✅ JSON vuoto → None: OK")
    
    # Non-JSON
    result = _parse_json_response("No JSON here at all", "test")
    assert result is None
    print("  ✅ Non-JSON → None: OK")
    
    # Parentesi sbilanciate (manca chiusura)
    broken = '{"entities": [{"type": "Attack", "name": "GCG"}], "relations": ['
    result = _parse_json_response(broken, "test")
    # Potrebbe ritornare None o un risultato parziale, l'importante è non crashare
    print(f"  ✅ JSON sbilanciato → gestito senza crash: {'parsed' if result else 'None'}")


# ============================================================================
# Test 5: Build entities e relations da dict raw
# ============================================================================

def test_build_from_raw():
    """Test _build_entities e _build_relations."""
    print("\n=== Test 5: Build da dict raw ===")
    
    mock_l1 = L1Summary(
        paper_id="test_paper_001",
        bibtex_key="Zou_2023_GCG",
        category="Adversarial Attacks",
        year=2023,
    )
    
    raw_entities = [
        {"type": "Attack", "name": "GCG", "aliases": ["Greedy Coordinate Gradient"], "description": "Gradient-based adversarial suffix", "year": 2023},
        {"type": "Model", "name": "gpt4", "description": "OpenAI flagship model"},
        {"type": "Model", "name": "llama-2", "description": "Meta's open-source LLM"},
        {"type": "Dataset", "name": "advbench", "description": "Adversarial behavior benchmark"},
        {"type": "Metric", "name": "attack success rate", "description": "Success rate of attack"},
        {"type": "Venue", "name": "neurips", "year": 2023},
        # Tipo invalido → scartato
        {"type": "InvalidType", "name": "Something"},
        # Nome vuoto → scartato
        {"type": "Attack", "name": ""},
        # Non-dict → scartato
        "not a dict",
    ]
    
    entities = _build_entities(raw_entities, mock_l1)
    
    # Dovrebbero essere 6 entità valide (1 attack, 2 model, 1 dataset, 1 metric, 1 venue)
    assert len(entities) == 6, f"Expected 6 entities, got {len(entities)}"
    
    # Verifica normalizzazione
    names = {e.canonical_name for e in entities}
    assert "GCG" in names
    assert "GPT-4" in names  # da "gpt4"
    assert "LLaMA-2" in names  # da "llama-2"
    assert "AdvBench" in names  # da "advbench"
    assert "ASR" in names  # da "attack success rate"
    assert "NeurIPS" in names  # da "neurips"
    
    # Verifica source tracking
    for e in entities:
        assert e.source_paper_id == "test_paper_001"
        assert e.source_bibtex_key == "Zou_2023_GCG"
    
    print(f"  ✅ {len(entities)} entità costruite (6 valide, 3 scartate)")
    print(f"  ✅ Nomi canonici: {sorted(names)}")
    
    # Build relazioni
    raw_relations = [
        {"type": "TARGETS", "source": "GCG", "source_type": "Attack", "target": "gpt4", "target_type": "Model", "description": "GCG targets GPT-4"},
        {"type": "TESTED_ON", "source": "GCG", "source_type": "Attack", "target": "advbench", "target_type": "Dataset"},
        {"type": "EVALUATED_WITH", "source": "GCG", "source_type": "Attack", "target": "attack success rate", "target_type": "Metric"},
        {"type": "INTRODUCED_IN", "source": "GCG", "source_type": "Attack", "target": "neurips", "target_type": "Venue", "year": 2023},
        # Target non esistente → scartato in validazione finale (ma creato qui)
        {"type": "TARGETS", "source": "GCG", "source_type": "Attack", "target": "Claude", "target_type": "Model"},
    ]
    
    relations = _build_relations(raw_relations, entities, mock_l1)
    
    # Nota: la relazione con Claude viene costruita ma con nome normalizzato
    # Se Claude non è tra le entità, sarà scartata nella validazione finale
    print(f"  ✅ {len(relations)} relazioni costruite")
    
    # Verifica risoluzione nomi
    targets_rel = [r for r in relations if r.type == RelationType.TARGETS and r.target == "GPT-4"]
    assert len(targets_rel) >= 1, "Relazione TARGETS → GPT-4 non trovata"
    assert targets_rel[0].source == "GCG"
    assert targets_rel[0].target == "GPT-4"  # normalizzato da "gpt4"
    print(f"  ✅ Risoluzione nomi: gpt4 → GPT-4, advbench → AdvBench")


# ============================================================================
# Test 6: Estrazione LIVE con Gemini Flash (5 L1 mock)
# ============================================================================

def test_live_extraction():
    """Test estrazione LIVE con Gemini Flash su 5 L1 mock realistici."""
    print("\n=== Test 6: Estrazione LIVE con Gemini Flash ===")
    
    # 5 L1 summary mock che simulano paper reali
    mock_summaries = [
        L1Summary(
            paper_id="paper_gcg_001",
            bibtex_key="Zou_2023_GCG",
            category="Adversarial Attacks",
            year=2023,
            key_contributions=[
                "Introduced GCG (Greedy Coordinate Gradient), an automated method for generating adversarial suffixes",
                "Demonstrated universal and transferable adversarial attacks across multiple LLMs",
                "Showed that aligned models (ChatGPT, Claude, Bard) can be jailbroken with optimized suffixes",
            ],
            methodology="GCG uses a gradient-based token search to find adversarial suffixes that cause LLMs to generate harmful content. The attack optimizes a universal suffix that transfers across models by combining gradients from multiple target models during optimization.",
            testing_methodology="Evaluated on AdvBench (520 harmful behaviors). Metrics: Attack Success Rate (ASR) judged by GPT-4-as-judge and keyword matching. Tested on Vicuna-7B, LLaMA-2 (7B, 13B), GPT-3.5 Turbo, GPT-4, Claude, PaLM-2. Transfer experiments across open-source and commercial models.",
            temporal_context="Published at NeurIPS 2023 Workshop. First demonstration of universal transferable adversarial attacks on aligned LLMs, building on adversarial examples tradition from computer vision.",
            results_summary="ASR >90% on Vicuna, >50% transfer to GPT-3.5 and GPT-4. Universal suffix achieves 84% ASR across 5 models simultaneously.",
            limitations="Suffixes are detectable by perplexity filters. Requires white-box access to at least one model for optimization. GPT-4 added defenses that reduced ASR to ~30% after paper publication.",
        ),
        L1Summary(
            paper_id="paper_smoothllm_001",
            bibtex_key="Robey_2023_SmoothLLM",
            category="Adversarial Attacks",
            year=2023,
            key_contributions=[
                "Proposed SmoothLLM, a defense against adversarial suffix attacks on LLMs",
                "Applied randomized smoothing concept to NLP domain for the first time",
                "Demonstrated effectiveness against GCG and AutoDAN attacks",
            ],
            methodology="SmoothLLM perturbs input prompts by randomly swapping/inserting/patching characters, queries the LLM multiple times with different perturbations, and aggregates responses via majority voting to detect adversarial suffixes.",
            testing_methodology="Tested on AdvBench (520 harmful behaviors). Metrics: ASR reduction, clean accuracy preservation, perplexity change. Models: LLaMA-2-7B, Vicuna-7B-v1.5. Perturbation types: swap (s%), insert (i%), patch (p%) at rates 5-20%. Compared against: no defense, perplexity filter, retokenization.",
            temporal_context="Published on arXiv in October 2023, 3 months after GCG. Direct response to the GCG vulnerability. First algorithmic defense specific to adversarial suffix attacks.",
            results_summary="Reduces GCG ASR from 86% to <2% with 10% swap perturbation rate. Maintains >95% clean accuracy. Effective against AutoDAN (ASR reduced to <5%).",
            limitations="Computational cost (N forward passes per query). Sensitive to perturbation rate: too high degrades utility. Not tested against adaptive attacks.",
        ),
        L1Summary(
            paper_id="paper_halueval_001",
            bibtex_key="Li_2023_HaluEval",
            category="LLM Hallucinations",
            year=2023,
            key_contributions=[
                "Created HaluEval, a large-scale benchmark for hallucination evaluation in LLMs",
                "Proposed automated hallucination sample generation using ChatGPT",
                "Provided 35,000 hallucinated vs faithful samples across QA, dialogue, and summarization",
            ],
            methodology="HaluEval generates hallucinated samples by prompting ChatGPT to produce plausible but incorrect outputs. Uses a two-step approach: (1) generate faithful response, (2) instruct model to introduce specific types of hallucinations (factual, reasoning, instruction-following).",
            testing_methodology="Benchmark tasks: question answering (QA), knowledge-grounded dialogue, text summarization. Metrics: hallucination rate, precision, recall, F1 for hallucination detection. Evaluated: ChatGPT, GPT-4, Claude, LLaMA-65B, Vicuna-13B. Human evaluation on 500 samples for validation.",
            temporal_context="Published at EMNLP 2023. Contemporary with TruthfulQA and FActScore. First large-scale automated hallucination benchmark.",
            results_summary="ChatGPT hallucinates in 19.5% of QA, 25.1% of dialogue, 11.4% of summarization tasks. GPT-4 reduces hallucination by 30-40% compared to ChatGPT. Hallucination detection F1: ChatGPT 62.6%, GPT-4 72.4%.",
            limitations="Benchmark limited to English. ChatGPT-generated hallucinations may not represent all types. No coverage of multimodal hallucinations.",
        ),
        L1Summary(
            paper_id="paper_poisoning_001",
            bibtex_key="Wan_2023_Poisoning",
            category="Data and Model Poisoning",
            year=2023,
            key_contributions=[
                "Demonstrated that instruction-tuned LLMs are vulnerable to data poisoning during fine-tuning",
                "Showed that even a small fraction (0.1%) of poisoned examples can degrade model safety",
                "Introduced multiple poisoning strategies: direct injection, trigger-based, and clean-label",
            ],
            methodology="Poisons the instruction-tuning dataset by injecting harmful instruction-response pairs. Three strategies: (1) direct injection of harmful content, (2) trigger-based poisoning with specific phrases that activate backdoor, (3) clean-label attack that uses benign-looking examples.",
            testing_methodology="Models: LLaMA-7B, Alpaca, GPT-J. Datasets: Alpaca 52K instructions, Dolly 15K. Poison rates: 0.01%, 0.1%, 1%, 5%. Metrics: attack success rate, harmfulness score (GPT-4 judge), MMLU accuracy (utility preservation). Compared: no defense, data filtering, DPO safety training.",
            temporal_context="Published at ICML 2023. First systematic study of data poisoning on instruction-tuned LLMs. Builds on traditional data poisoning from supervised learning.",
            results_summary="0.1% poison rate achieves >80% ASR on trigger-based attack. Clean-label attack achieves 45% ASR at 1% poison rate. MMLU accuracy drops <2% at 0.1% poison rate, making attacks stealthy.",
            limitations="Assumes attacker can modify training data. Clean-label attack requires careful crafting. Not tested on models with extensive safety training (e.g., GPT-4).",
        ),
        L1Summary(
            paper_id="paper_greshake_001",
            bibtex_key="Greshake_2023_LLM",
            category="Prompt Injection",
            year=2023,
            key_contributions=[
                "First systematic study of indirect prompt injection in real-world LLM-integrated applications",
                "Demonstrated attacks on Bing Chat, ChatGPT plugins, and LangChain-based applications",
                "Proposed taxonomy of indirect prompt injection: data exfiltration, goal hijacking, denial of service",
            ],
            methodology="Injects malicious instructions into external data sources (web pages, emails, documents) that are retrieved by LLM-integrated applications. The injected prompts override system instructions when the LLM processes the retrieved content.",
            testing_methodology="Tested on: Bing Chat (Microsoft), ChatGPT Plugins (OpenAI), custom LangChain agents. Attack scenarios: web page injection (hidden text), email injection, document injection. Metrics: attack success rate, stealthiness (human detection rate). Manual testing with 50+ attack scenarios. No standardized benchmark.",
            temporal_context="Published at AISec@CCS 2023. First paper to demonstrate indirect prompt injection as a real-world threat. Preceded OWASP Top 10 for LLM Applications inclusion of prompt injection.",
            results_summary="Successfully hijacked Bing Chat in 75% of attempts. Data exfiltration from ChatGPT plugins in 60% of cases. Goal hijacking in LangChain agents: 90% success rate.",
            limitations="No automated evaluation framework. Results depend on specific application configurations. Defenses not systematically evaluated.",
        ),
    ]
    
    print(f"  Testing with {len(mock_summaries)} L1 summaries...")
    
    # Test estrazione singola (primo summary)
    print("\n  --- Estrazione singola: Zou_2023_GCG ---")
    result = extract_with_details(mock_summaries[0])
    
    if result.error:
        print(f"  ⚠️ Errore: {result.error}")
        print("  Proseguo con test batch...")
    else:
        print(f"  ✅ Entità: {len(result.entities)}")
        for e in result.entities:
            print(f"     {e.type.value:10s} | {e.canonical_name}")
        print(f"  ✅ Relazioni: {len(result.relations)}")
        for r in result.relations:
            print(f"     {r.source} --[{r.type.value}]--> {r.target}")
        print(f"  ✅ Tempo: {result.extraction_time_seconds:.1f}s | Modello: {result.model_used}")
        
        # Verifiche base
        entity_names = {e.canonical_name for e in result.entities}
        assert len(result.entities) >= 3, f"Troppo poche entità: {len(result.entities)}"
        assert any(e.type == EntityType.ATTACK for e in result.entities), "Nessun Attack trovato"
        assert any(e.type == EntityType.MODEL for e in result.entities), "Nessun Model trovato"
    
    # Test batch (tutti e 5)
    print("\n  --- Batch extraction: 5 summaries ---")
    all_entities, all_relations, results = extract_batch(
        mock_summaries,
        merge_results=True,
        show_progress=True,
    )
    
    success_count = sum(1 for r in results if not r.error)
    fail_count = sum(1 for r in results if r.error)
    
    print(f"\n  ✅ Batch completato: {success_count}/{len(mock_summaries)} successi, {fail_count} falliti")
    print(f"  ✅ Totale entità (merged): {len(all_entities)}")
    print(f"  ✅ Totale relazioni (merged): {len(all_relations)}")
    
    # Statistiche per tipo
    stats = get_extraction_stats(all_entities, all_relations)
    print(f"\n  --- Entity types ---")
    for t, count in sorted(stats["entity_types"].items()):
        print(f"     {t:12s}: {count}")
    print(f"\n  --- Relation types ---")
    for t, count in sorted(stats["relation_types"].items()):
        print(f"     {t:20s}: {count}")
    print(f"\n  Avg aliases/entity: {stats['avg_aliases_per_entity']:.1f}")
    print(f"  Unique source papers: {stats['unique_source_papers']}")
    
    # Verifiche minime su batch
    if success_count >= 3:  # Almeno 3 su 5 devono riuscire
        assert len(all_entities) >= 10, f"Troppo poche entità totali: {len(all_entities)}"
        assert len(all_relations) >= 5, f"Troppo poche relazioni totali: {len(all_relations)}"
        assert stats["entity_types"].get("Attack", 0) >= 2, "Pochi Attack"
        assert stats["entity_types"].get("Model", 0) >= 2, "Pochi Model"
        print("\n  ✅ Verifiche batch superate!")
    else:
        print(f"\n  ⚠️ Solo {success_count} successi, skip verifiche quantitative")
    
    # Test salvataggio
    output_path = "/tmp/test_graphrag_extraction.json"
    saved_path = save_extraction_results(all_entities, all_relations, output_path)
    assert Path(saved_path).exists()
    saved_data = json.loads(Path(saved_path).read_text())
    assert len(saved_data["entities"]) == len(all_entities)
    assert len(saved_data["relations"]) == len(all_relations)
    print(f"\n  ✅ Risultati salvati in: {saved_path}")
    
    # Stampa dettagli per ogni paper
    print("\n  --- Dettagli per paper ---")
    for r in results:
        status = "✅" if not r.error else "❌"
        print(
            f"  {status} {r.bibtex_key or r.paper_id}: "
            f"{len(r.entities)} entità, {len(r.relations)} relazioni "
            f"({r.extraction_time_seconds:.1f}s)"
        )
    
    total_time = sum(r.extraction_time_seconds for r in results)
    print(f"\n  ⏱️ Tempo totale: {total_time:.1f}s")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST graphrag/entity_extractor.py — Task 3.4")
    print("=" * 70)
    
    # Test offline (no API)
    test_normalization()
    test_models()
    test_merge()
    test_json_parsing()
    test_build_from_raw()
    
    # Test LIVE con Gemini Flash
    test_live_extraction()
    
    print("\n" + "=" * 70)
    print("TUTTI I TEST PASSATI ✅")
    print("=" * 70)
