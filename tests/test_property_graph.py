#!/usr/bin/env python3
"""
Test per graphrag/property_graph.py — PropertyGraph Builder.

Test:
1. Data classes (GraphQueryResult, BuildReport)
2. Keyword extraction
3. EntityNode mapping (Entity → LlamaIndex EntityNode)
4. Relation mapping (Relation → LlamaIndex Relation)
5. Empty graph init, save/load cycle
6. Build from entities (pre-extracted)
7. Graph navigation (get_stats, remove_paper)
8. Query: direct traversal (no LLM)
9. Query: entity neighborhood
10. Temporal chain
11. Update incrementale
12. Convenience functions
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------------------
# Mock L1 summaries + entities + relations per test
# -------------------------------------------------------------------

def _make_test_entities():
    """Crea un set di entità mock per test."""
    from paperhunter.graphrag.entity_extractor import Entity, EntityType

    return [
        Entity(
            type=EntityType.ATTACK,
            canonical_name="GCG",
            aliases=["Greedy Coordinate Gradient"],
            description="Universal adversarial suffix attack against LLMs",
            year=2023,
            source_paper_id="paper_gcg",
            source_bibtex_key="Zou_2023_GCG",
            properties={"technique": "gradient-based"},
        ),
        Entity(
            type=EntityType.ATTACK,
            canonical_name="AutoDAN",
            aliases=["Automatic DAN"],
            description="Automated jailbreak prompt generation",
            year=2023,
            source_paper_id="paper_autodan",
            source_bibtex_key="Liu_2023_AutoDAN",
        ),
        Entity(
            type=EntityType.DEFENSE,
            canonical_name="SmoothLLM",
            aliases=["Smooth LLM"],
            description="Randomized smoothing defense against adversarial prompts",
            year=2023,
            source_paper_id="paper_smoothllm",
            source_bibtex_key="Robey_2023_SmoothLLM",
        ),
        Entity(
            type=EntityType.MODEL,
            canonical_name="GPT-4",
            aliases=["gpt4", "gpt-4-turbo"],
            description="OpenAI large language model",
            year=2023,
            source_paper_id="paper_gcg",
            source_bibtex_key="Zou_2023_GCG",
        ),
        Entity(
            type=EntityType.MODEL,
            canonical_name="LLaMA-2",
            aliases=["Llama 2", "llama2"],
            description="Meta open-source LLM",
            year=2023,
            source_paper_id="paper_gcg",
            source_bibtex_key="Zou_2023_GCG",
        ),
        Entity(
            type=EntityType.DATASET,
            canonical_name="AdvBench",
            aliases=["Adversarial Benchmark"],
            description="Adversarial behavior benchmark for LLMs",
            year=2023,
            source_paper_id="paper_gcg",
            source_bibtex_key="Zou_2023_GCG",
        ),
        Entity(
            type=EntityType.METRIC,
            canonical_name="ASR",
            aliases=["Attack Success Rate"],
            description="Fraction of successful attacks",
            year=None,
            source_paper_id="paper_gcg",
            source_bibtex_key="Zou_2023_GCG",
        ),
        Entity(
            type=EntityType.VENUE,
            canonical_name="NeurIPS",
            aliases=["NeurIPS 2023"],
            description="Conference on Neural Information Processing Systems",
            year=None,
            source_paper_id="paper_gcg",
            source_bibtex_key="Zou_2023_GCG",
        ),
        Entity(
            type=EntityType.ATTACK,
            canonical_name="PromptShield",
            aliases=[],
            description="Advanced prompt injection defense framework",
            year=2025,
            source_paper_id="paper_ps",
            source_bibtex_key="Lee_2025_PromptShield",
        ),
        Entity(
            type=EntityType.DEFENSE,
            canonical_name="Perplexity Filter",
            aliases=["PPL Filter"],
            description="Detects adversarial inputs based on perplexity score",
            year=2022,
            source_paper_id="paper_ppl",
            source_bibtex_key="Alon_2022_PPL",
        ),
    ]


def _make_test_relations():
    """Crea un set di relazioni mock per test."""
    from paperhunter.graphrag.entity_extractor import (
        EntityType,
        Relation,
        RelationType,
    )

    return [
        Relation(
            type=RelationType.TARGETS,
            source="GCG", source_type=EntityType.ATTACK,
            target="GPT-4", target_type=EntityType.MODEL,
            description="GCG generates adversarial suffixes targeting GPT-4",
            year=2023,
            source_paper_id="paper_gcg",
            source_bibtex_key="Zou_2023_GCG",
        ),
        Relation(
            type=RelationType.TARGETS,
            source="GCG", source_type=EntityType.ATTACK,
            target="LLaMA-2", target_type=EntityType.MODEL,
            description="GCG targets LLaMA-2",
            year=2023,
        ),
        Relation(
            type=RelationType.TESTED_ON,
            source="GCG", source_type=EntityType.ATTACK,
            target="AdvBench", target_type=EntityType.DATASET,
            description="GCG evaluated on AdvBench",
            year=2023,
        ),
        Relation(
            type=RelationType.EVALUATED_WITH,
            source="GCG", source_type=EntityType.ATTACK,
            target="ASR", target_type=EntityType.METRIC,
            description="GCG measured by ASR",
        ),
        Relation(
            type=RelationType.DEFENDS_AGAINST,
            source="SmoothLLM", source_type=EntityType.DEFENSE,
            target="GCG", target_type=EntityType.ATTACK,
            description="SmoothLLM defends against GCG by input randomization",
            year=2023,
        ),
        Relation(
            type=RelationType.INTRODUCED_IN,
            source="GCG", source_type=EntityType.ATTACK,
            target="NeurIPS", target_type=EntityType.VENUE,
            description="GCG was introduced at NeurIPS 2023",
            year=2023,
        ),
        Relation(
            type=RelationType.EXTENDS,
            source="AutoDAN", source_type=EntityType.ATTACK,
            target="GCG", target_type=EntityType.ATTACK,
            description="AutoDAN extends GCG with genetic algorithm optimization",
            year=2023,
        ),
        Relation(
            type=RelationType.UPDATES,
            source="PromptShield", source_type=EntityType.ATTACK,
            target="Perplexity Filter", target_type=EntityType.DEFENSE,
            description="PromptShield is an updated version of detection-based defenses",
            year=2025,
        ),
    ]


def _make_test_l1_summaries():
    """Crea L1 summaries mock per test."""
    from paperhunter.models import L1Summary

    return [
        L1Summary(
            paper_id="paper_gcg",
            bibtex_key="Zou_2023_GCG",
            category="Adversarial Attacks",
            year=2023,
            key_contributions=[
                "Universal adversarial suffixes for jailbreaking LLMs",
                "Transferable attacks across models",
            ],
            methodology="Greedy Coordinate Gradient optimization over token embeddings",
            testing_methodology="Evaluated on AdvBench with ASR and human evaluation on GPT-4, LLaMA-2, Vicuna",
            temporal_context="First gradient-based universal attack on aligned LLMs (2023)",
            results_summary="99.9% ASR on Vicuna, 84% on GPT-4 with transfer",
            limitations="Requires white-box access for suffix generation",
            generated_at=datetime.now(),
            model_used="gemini-3-pro-preview",
        ),
        L1Summary(
            paper_id="paper_smoothllm",
            bibtex_key="Robey_2023_SmoothLLM",
            category="Adversarial Attacks",
            year=2023,
            key_contributions=[
                "First randomized smoothing defense for LLMs",
                "Character-level perturbations defeat adversarial suffixes",
            ],
            methodology="Random character perturbations on input + majority voting on outputs",
            testing_methodology="Tested against GCG, AutoDAN on LLaMA-2 using ASR metric",
            temporal_context="Response to GCG attack, same year (2023)",
            results_summary="Reduces GCG ASR from 99% to <2% on LLaMA-2",
            limitations="Increases inference latency by N times; may break benign queries",
            generated_at=datetime.now(),
            model_used="gemini-3-pro-preview",
        ),
    ]


# ===================================================================
# Tests
# ===================================================================

def test_dataclasses():
    """Test 1: Data classes GraphQueryResult e BuildReport."""
    from paperhunter.graphrag.property_graph import GraphQueryResult, BuildReport

    # GraphQueryResult
    result = GraphQueryResult(
        context_text="Test context",
        entities_found=[{"name": "GCG"}],
        relations_found=[{"source": "GCG", "type": "TARGETS", "target": "GPT-4"}],
        query="test query",
        retrieval_method="test",
        elapsed_seconds=1.23,
    )
    d = result.to_dict()
    assert d["context_text"] == "Test context"
    assert d["query"] == "test query"
    assert d["elapsed_seconds"] == 1.23
    assert d["paths_count"] == 0

    # BuildReport
    report = BuildReport(
        entities_added=10,
        relations_added=5,
        elapsed_seconds=2.5,
    )
    d = report.to_dict()
    assert d["entities_added"] == 10
    assert d["relations_added"] == 5
    print("✓ Test 1: Data classes OK")


def test_keyword_extraction():
    """Test 2: Simple keyword extraction."""
    from paperhunter.graphrag.property_graph import _extract_keywords_simple

    kws = _extract_keywords_simple("evolution of prompt injection defenses")
    assert "prompt" in kws
    assert "injection" in kws
    assert "defenses" in kws
    assert "of" not in kws  # stopword

    kws2 = _extract_keywords_simple("GCG adversarial attack on GPT-4")
    assert "gcg" in kws2
    assert "adversarial" in kws2
    assert "gpt-4" in kws2

    kws3 = _extract_keywords_simple("")
    assert kws3 == []

    print("✓ Test 2: Keyword extraction OK")


def test_entity_node_mapping():
    """Test 3: Entity → LlamaIndex EntityNode mapping."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    builder = PropertyGraphBuilder(persist_dir=tempfile.mkdtemp(), auto_load=False)
    entities = _make_test_entities()

    for entity in entities:
        li_node = builder._entity_to_li_node(entity)
        assert li_node.name == entity.canonical_name
        assert li_node.label == entity.type.value
        assert li_node.properties["entity_type"] == entity.type.value
        assert li_node.properties["aliases"] == entity.aliases
        assert li_node.properties["description"] == entity.description
        if entity.year:
            assert li_node.properties["year"] == entity.year

    print(f"✓ Test 3: Entity mapping OK ({len(entities)} entities)")


def test_relation_mapping():
    """Test 4: Relation → LlamaIndex Relation mapping."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder
    from paperhunter.graphrag.entity_extractor import _make_entity_id

    builder = PropertyGraphBuilder(persist_dir=tempfile.mkdtemp(), auto_load=False)
    entities = _make_test_entities()
    relations = _make_test_relations()

    # Build entity_id_map
    entity_id_map = {}
    for e in entities:
        li_node = builder._entity_to_li_node(e)
        entity_id_map[e.id] = li_node.id

    converted = 0
    for rel in relations:
        li_rel = builder._relation_to_li_relation(rel, entity_id_map)
        assert li_rel is not None
        assert li_rel.label == rel.type.value
        assert li_rel.properties["description"] == (rel.description or "")
        converted += 1

    print(f"✓ Test 4: Relation mapping OK ({converted} relations)")


def test_empty_graph_and_persistence():
    """Test 5: Init vuoto, save, load."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    tmp_dir = tempfile.mkdtemp()
    try:
        # Create empty graph
        builder = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=False)
        assert builder.entity_count == 0
        assert builder.relation_count == 0
        assert builder.paper_ids == []

        # Save
        path = builder.save()
        assert Path(path).exists()
        assert (Path(tmp_dir) / "graph_store.json").exists()
        assert (Path(tmp_dir) / "graph_metadata.json").exists()

        # Load
        builder2 = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=True)
        assert builder2.entity_count == 0

        print("✓ Test 5: Empty graph + persistence OK")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_from_entities():
    """Test 6: Build graph from pre-extracted entities + relations."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    tmp_dir = tempfile.mkdtemp()
    try:
        builder = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=False)
        entities = _make_test_entities()
        relations = _make_test_relations()

        report = builder.build_from_entities(
            entities, relations,
            paper_ids=["paper_gcg", "paper_autodan", "paper_smoothllm"],
            show_progress=False,
        )

        assert report.entities_added == len(entities), f"Expected {len(entities)}, got {report.entities_added}"
        assert report.relations_added == len(relations), f"Expected {len(relations)}, got {report.relations_added}"
        assert report.elapsed_seconds > 0

        # Verify stats
        assert builder.entity_count == len(entities)
        assert builder.relation_count == len(relations)
        assert set(builder.paper_ids) == {"paper_gcg", "paper_autodan", "paper_smoothllm"}

        # Verify persistence
        builder2 = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=True)
        assert builder2.entity_count == len(entities)
        assert builder2.relation_count == len(relations)

        print(f"✓ Test 6: Build from entities OK ({report.entities_added}E + {report.relations_added}R)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_get_stats_and_remove():
    """Test 7: get_stats and remove_paper."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    tmp_dir = tempfile.mkdtemp()
    try:
        builder = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=False)
        entities = _make_test_entities()
        relations = _make_test_relations()
        builder.build_from_entities(
            entities, relations,
            paper_ids=["paper_gcg", "paper_autodan", "paper_smoothllm", "paper_ps", "paper_ppl"],
            show_progress=False,
        )

        # Stats
        stats = builder.get_stats()
        assert stats["entity_count"] == len(entities)
        assert stats["relation_count"] == len(relations)
        assert stats["paper_count"] == 5
        assert "entities_by_type" in stats
        assert "Attack" in stats["entities_by_type"]
        assert stats["entities_by_type"]["Attack"] == 3  # GCG, AutoDAN, PromptShield
        print(f"  Stats by type: {stats['entities_by_type']}")
        print(f"  Relations by type: {stats.get('relations_by_type', {})}")

        # Remove paper
        initial_count = builder.entity_count
        removed = builder.remove_paper("paper_ps")
        assert removed >= 1, f"Expected at least 1 removed, got {removed}"
        assert "paper_ps" not in builder.paper_ids
        assert builder.entity_count < initial_count

        print(f"✓ Test 7: get_stats + remove_paper OK (removed {removed} nodes for paper_ps)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_direct_traversal_query():
    """Test 8: Query via direct traversal (no LLM needed)."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    tmp_dir = tempfile.mkdtemp()
    try:
        builder = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=False)
        entities = _make_test_entities()
        relations = _make_test_relations()
        builder.build_from_entities(entities, relations, show_progress=False)

        # Query diretto (usa il fallback senza LLM)
        result = builder._query_via_direct_traversal(
            "GCG adversarial attack defense"
        )

        assert result.retrieval_method == "direct_traversal"
        assert len(result.entities_found) > 0, "Expected at least one entity found"
        # GCG should be found
        entity_names = [e["name"] for e in result.entities_found]
        assert "GCG" in entity_names, f"GCG not found in {entity_names}"
        assert len(result.context_text) > 0, "Expected non-empty context"

        print(f"✓ Test 8: Direct traversal OK ({len(result.entities_found)} entities, "
              f"{len(result.relations_found)} relations, {len(result.context_text)} chars)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_entity_neighborhood():
    """Test 9: Query entity neighborhood."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    tmp_dir = tempfile.mkdtemp()
    try:
        builder = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=False)
        entities = _make_test_entities()
        relations = _make_test_relations()
        builder.build_from_entities(entities, relations, show_progress=False)

        result = builder.query_entity_neighborhood("GCG", depth=1)
        assert len(result.entities_found) >= 1, "Expected at least GCG itself"
        # GCG has 6+ relations: TARGETS GPT-4, TARGETS LLaMA-2, TESTED_ON AdvBench,
        # EVALUATED_WITH ASR, (incoming) SmoothLLM DEFENDS_AGAINST, INTRODUCED_IN NeurIPS, AutoDAN EXTENDS
        print(f"  GCG neighborhood: {len(result.entities_found)} entities, {len(result.relations_found)} relations")

        # Non-existent entity
        result2 = builder.query_entity_neighborhood("NonExistent", depth=1)
        assert len(result2.entities_found) == 0

        print(f"✓ Test 9: Entity neighborhood OK")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_temporal_chain():
    """Test 10: Temporal chain retrieval."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    tmp_dir = tempfile.mkdtemp()
    try:
        builder = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=False)
        entities = _make_test_entities()
        relations = _make_test_relations()
        builder.build_from_entities(entities, relations, show_progress=False)

        result = builder.get_temporal_chain("Attack")
        assert result.retrieval_method == "temporal_chain"
        # Should find EXTENDS (AutoDAN→GCG), INTRODUCED_IN (GCG→NeurIPS), UPDATES (PromptShield→Perplexity Filter)
        # Attack type entities in these relations: GCG, AutoDAN, PromptShield
        entity_names = [e["name"] for e in result.entities_found]
        print(f"  Temporal chain entities: {entity_names}")
        print(f"  Temporal chain relations: {len(result.relations_found)}")
        assert len(result.context_text) > 0

        # Defense chain
        result2 = builder.get_temporal_chain("Defense")
        print(f"  Defense chain entities: {[e['name'] for e in result2.entities_found]}")

        print(f"✓ Test 10: Temporal chain OK")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_update_incremental():
    """Test 11: Aggiornamento incrementale."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    tmp_dir = tempfile.mkdtemp()
    try:
        from paperhunter.graphrag.entity_extractor import Entity, EntityType, Relation as PHRelation, RelationType

        builder = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=False)
        entities = _make_test_entities()[:5]  # Solo primi 5
        relations = _make_test_relations()[:3]  # Solo primi 3
        builder.build_from_entities(
            entities, relations,
            paper_ids=["paper_gcg", "paper_autodan"],
            show_progress=False,
        )
        initial_e = builder.entity_count
        initial_r = builder.relation_count

        # Aggiornamento con nuove entità
        new_entities = [
            Entity(
                type=EntityType.ATTACK,
                canonical_name="PAIR",
                description="Prompt Automatic Iterative Refinement",
                year=2024,
                source_paper_id="paper_pair",
                source_bibtex_key="Chao_2024_PAIR",
            ),
        ]
        new_relations = [
            PHRelation(
                type=RelationType.EXTENDS,
                source="PAIR", source_type=EntityType.ATTACK,
                target="GCG", target_type=EntityType.ATTACK,
                description="PAIR extends GCG with iterative refinement",
                year=2024,
            ),
        ]

        builder.build_from_entities(
            new_entities, new_relations,
            paper_ids=["paper_pair"],
            show_progress=False,
        )

        assert builder.entity_count == initial_e + 1
        assert builder.relation_count == initial_r + 1
        assert "paper_pair" in builder.paper_ids

        print(f"✓ Test 11: Incremental update OK ({initial_e}→{builder.entity_count}E, "
              f"{initial_r}→{builder.relation_count}R)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_convenience_functions():
    """Test 12: Standalone convenience functions."""
    from paperhunter.graphrag.property_graph import (
        GraphQueryResult,
        BuildReport,
        _extract_keywords_simple,
    )

    # Extract keywords
    kws = _extract_keywords_simple("prompt injection defense mechanisms 2024")
    assert len(kws) > 0
    assert "prompt" in kws
    assert "2024" in kws

    # GraphQueryResult serialization
    r = GraphQueryResult()
    d = r.to_dict()
    assert "context_text" in d
    assert "elapsed_seconds" in d

    # BuildReport serialization
    b = BuildReport()
    d = b.to_dict()
    assert "entities_added" in d

    print("✓ Test 12: Convenience functions OK")


def test_repr():
    """Test 13: __repr__."""
    from paperhunter.graphrag.property_graph import PropertyGraphBuilder

    tmp_dir = tempfile.mkdtemp()
    try:
        builder = PropertyGraphBuilder(persist_dir=tmp_dir, auto_load=False)
        r = repr(builder)
        assert "PropertyGraphBuilder" in r
        assert "entities=0" in r
        assert "relations=0" in r
        print(f"✓ Test 13: repr OK → {r}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Test graphrag/property_graph.py — PropertyGraph Builder")
    print("=" * 70)
    print()

    tests = [
        test_dataclasses,
        test_keyword_extraction,
        test_entity_node_mapping,
        test_relation_mapping,
        test_empty_graph_and_persistence,
        test_build_from_entities,
        test_get_stats_and_remove,
        test_direct_traversal_query,
        test_entity_neighborhood,
        test_temporal_chain,
        test_update_incremental,
        test_convenience_functions,
        test_repr,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 70)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)
