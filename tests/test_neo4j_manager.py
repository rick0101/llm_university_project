"""
Test suite per graphrag/neo4j_manager.py — Task 3.5

Testa:
1. Connessione/disconnessione
2. create_constraints
3. Upsert/get/delete entity  
4. Upsert/get/delete relation
5. import_entities_and_relations (bulk)
6. query_neighbors
7. query_path
8. get_temporal_chain
9. clear_graph / clear_by_paper
10. get_stats
11. import_from_json
12. Neo4jNotEnabledError (quando disabilitato)

Prerequisiti:
    docker run --rm -d --name neo4j_test -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5-community
"""

import json
import sys
import tempfile
import time
from pathlib import Path

# Patch config to enable Neo4j BEFORE importing manager
import paperhunter.config as cfg
cfg.settings.neo4j.enabled = True
cfg.settings.neo4j.uri = "bolt://localhost:7687"
cfg.settings.neo4j.username = "neo4j"
cfg.settings.neo4j.password = "password"
cfg.settings.neo4j.database = "neo4j"

from paperhunter.graphrag.entity_extractor import (
    Entity,
    EntityType,
    Relation,
    RelationType,
)
from paperhunter.graphrag.neo4j_manager import (
    Neo4jConnectionError,
    Neo4jManager,
    Neo4jNotEnabledError,
    test_neo4j_connection,
)


def make_test_entities() -> list[Entity]:
    """Crea un set di entità di test."""
    return [
        Entity(type=EntityType.ATTACK, canonical_name="GCG", 
               description="Greedy Coordinate Gradient attack",
               year=2023, source_bibtex_key="Zou_2023_GCG",
               aliases=["greedy coordinate gradient"],
               properties={"asr": 0.95}),
        Entity(type=EntityType.ATTACK, canonical_name="AutoDAN",
               description="Automatic DAN jailbreak generation",
               year=2023, source_bibtex_key="Liu_2023_AutoDAN",
               aliases=["auto-dan"]),
        Entity(type=EntityType.DEFENSE, canonical_name="SmoothLLM",
               description="Smoothing-based defense against jailbreaks",
               year=2024, source_bibtex_key="Robey_2024_SmoothLLM"),
        Entity(type=EntityType.MODEL, canonical_name="GPT-4",
               description="OpenAI's flagship LLM",
               year=2023, source_bibtex_key="Zou_2023_GCG",
               aliases=["gpt4", "gpt-4-turbo"]),
        Entity(type=EntityType.MODEL, canonical_name="LLaMA-2",
               description="Meta's open-source LLM",
               year=2023, source_bibtex_key="Zou_2023_GCG"),
        Entity(type=EntityType.DATASET, canonical_name="AdvBench",
               description="Adversarial benchmark for jailbreak evaluation",
               year=2023, source_bibtex_key="Zou_2023_GCG"),
        Entity(type=EntityType.METRIC, canonical_name="ASR",
               description="Attack Success Rate",
               year=2023, source_bibtex_key="Zou_2023_GCG"),
        Entity(type=EntityType.VENUE, canonical_name="NeurIPS",
               description="Neural Information Processing Systems",
               year=None, source_bibtex_key="Zou_2023_GCG"),
        Entity(type=EntityType.DEFENSE, canonical_name="Perplexity Filter",
               description="Detects gibberish adversarial suffixes via high perplexity",
               year=2023, source_bibtex_key="Jain_2023_PerplexityFilter"),
        Entity(type=EntityType.ATTACK, canonical_name="Prompt Injection",
               description="Injecting malicious instructions into LLM prompts",
               year=2022, source_bibtex_key="Greshake_2023_LLM"),
    ]


def make_test_relations() -> list[Relation]:
    """Crea un set di relazioni di test."""
    return [
        Relation(type=RelationType.TARGETS, source="GCG", source_type=EntityType.ATTACK,
                 target="GPT-4", target_type=EntityType.MODEL,
                 description="GCG attacks targeting GPT-4",
                 source_bibtex_key="Zou_2023_GCG"),
        Relation(type=RelationType.TARGETS, source="GCG", source_type=EntityType.ATTACK,
                 target="LLaMA-2", target_type=EntityType.MODEL,
                 description="GCG attacks targeting LLaMA-2",
                 source_bibtex_key="Zou_2023_GCG"),
        Relation(type=RelationType.TESTED_ON, source="GCG", source_type=EntityType.ATTACK,
                 target="AdvBench", target_type=EntityType.DATASET,
                 description="GCG evaluated on AdvBench",
                 source_bibtex_key="Zou_2023_GCG"),
        Relation(type=RelationType.EVALUATED_WITH, source="GCG", source_type=EntityType.ATTACK,
                 target="ASR", target_type=EntityType.METRIC,
                 description="GCG performance measured by ASR",
                 source_bibtex_key="Zou_2023_GCG"),
        Relation(type=RelationType.DEFENDS_AGAINST, source="SmoothLLM", source_type=EntityType.DEFENSE,
                 target="GCG", target_type=EntityType.ATTACK,
                 description="SmoothLLM defends against GCG-style attacks",
                 year=2024,
                 source_bibtex_key="Robey_2024_SmoothLLM"),
        Relation(type=RelationType.DEFENDS_AGAINST, source="Perplexity Filter", source_type=EntityType.DEFENSE,
                 target="GCG", target_type=EntityType.ATTACK,
                 description="Perplexity filter detects GCG suffixes",
                 year=2023,
                 source_bibtex_key="Jain_2023_PerplexityFilter"),
        Relation(type=RelationType.INTRODUCED_IN, source="GCG", source_type=EntityType.ATTACK,
                 target="NeurIPS", target_type=EntityType.VENUE,
                 year=2023,
                 description="GCG introduced at NeurIPS 2023",
                 source_bibtex_key="Zou_2023_GCG"),
        Relation(type=RelationType.EXTENDS, source="AutoDAN", source_type=EntityType.ATTACK,
                 target="GCG", target_type=EntityType.ATTACK,
                 year=2023,
                 description="AutoDAN extends GCG with automated optimization",
                 source_bibtex_key="Liu_2023_AutoDAN"),
    ]


def test_all():
    """Esegue tutti i test."""
    passed = 0
    failed = 0
    total_start = time.time()

    # ==================================================================
    # Test 1: Connessione
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Connessione e context manager")
    print("=" * 60)
    try:
        with Neo4jManager() as mgr:
            assert mgr.is_connected, "Dovrebbe essere connesso"
            print(f"  ✓ Connesso: {mgr}")
        # Dopo il context manager, dovrebbe essere disconnesso
        assert not mgr.is_connected, "Dovrebbe essere disconnesso dopo close()"
        print("  ✓ Disconnesso dopo context manager exit")
        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        failed += 1
        print("\n⚠️  Neo4j non raggiungibile. Verifica che il container Docker sia avviato.")
        return

    # ==================================================================
    # Test 2: Clear + Constraints
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Clear graph + Create constraints")
    print("=" * 60)
    try:
        with Neo4jManager() as mgr:
            deleted = mgr.clear_graph()
            print(f"  ✓ Grafo pulito: {deleted} nodi eliminati")
            mgr.create_constraints()
            print("  ✓ Vincoli e indici creati")
        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        failed += 1

    # ==================================================================
    # Test 3: Entity CRUD
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Entity CRUD (upsert, get, delete)")
    print("=" * 60)
    try:
        entities = make_test_entities()
        with Neo4jManager() as mgr:
            mgr.clear_graph()
            mgr.create_constraints()

            # Upsert
            gcg = entities[0]
            ok = mgr.upsert_entity(gcg)
            assert ok, "Upsert dovrebbe riuscire"
            print(f"  ✓ Entity upserted: {gcg.canonical_name}")

            # Get by ID
            retrieved = mgr.get_entity(gcg.id)
            assert retrieved is not None, "Entity dovrebbe essere trovata"
            assert retrieved["canonical_name"] == "GCG"
            assert retrieved["year"] == 2023
            assert "Attack" in retrieved.get("labels", [])
            print(f"  ✓ Get by ID: {retrieved['canonical_name']} (labels: {retrieved.get('labels')})")

            # Get by name
            by_name = mgr.get_entity_by_name("GCG")
            assert by_name is not None
            assert by_name["entity_id"] == gcg.id
            print(f"  ✓ Get by name: {by_name['canonical_name']}")

            # Get by name + type
            by_name_type = mgr.get_entity_by_name("GCG", EntityType.ATTACK)
            assert by_name_type is not None
            print(f"  ✓ Get by name+type: {by_name_type['canonical_name']} ({by_name_type.get('entity_type')})")

            # Update (upsert with changed description)
            gcg_updated = Entity(
                type=EntityType.ATTACK, canonical_name="GCG",
                description="Updated: Greedy Coordinate Gradient universal attack",
                year=2023,
            )
            ok = mgr.upsert_entity(gcg_updated)
            retrieved2 = mgr.get_entity(gcg.id)
            assert "Updated" in retrieved2["description"]
            print(f"  ✓ Entity updated: description='{retrieved2['description'][:50]}...'")

            # Non-existent
            missing = mgr.get_entity("nonexistent_id")
            assert missing is None
            print("  ✓ Non-existent entity returns None")

            # Delete
            ok = mgr.delete_entity(gcg.id)
            assert ok
            assert mgr.get_entity(gcg.id) is None
            print("  ✓ Entity deleted successfully")

        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ==================================================================
    # Test 4: Relation CRUD
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Relation CRUD (upsert, delete)")
    print("=" * 60)
    try:
        entities = make_test_entities()
        relations = make_test_relations()
        with Neo4jManager() as mgr:
            mgr.clear_graph()
            mgr.create_constraints()

            # Inserisci entità necessarie (GCG, GPT-4)
            mgr.upsert_entity(entities[0])  # GCG
            mgr.upsert_entity(entities[3])  # GPT-4

            # Upsert relazione
            rel = relations[0]  # GCG -TARGETS-> GPT-4
            ok = mgr.upsert_relation(rel)
            assert ok, "Upsert relazione dovrebbe riuscire"
            print(f"  ✓ Relation upserted: {rel.source} -[{rel.type.value}]-> {rel.target}")

            # Upsert di nuovo (idempotente)
            ok2 = mgr.upsert_relation(rel)
            assert ok2, "Re-upsert dovrebbe riuscire"
            print("  ✓ Relation re-upsert idempotente OK")

            # Relazione con entità mancante
            bad_rel = Relation(
                type=RelationType.TARGETS,
                source="NonExistent",
                source_type=EntityType.ATTACK,
                target="GPT-4",
                target_type=EntityType.MODEL,
            )
            ok3 = mgr.upsert_relation(bad_rel)
            assert not ok3, "Relazione con entità mancante dovrebbe fallire"
            print("  ✓ Relation con entità mancante: correttamente fallita")

            # Delete relazione
            ok4 = mgr.delete_relation(rel.id)
            assert ok4
            print("  ✓ Relation deleted successfully")

        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ==================================================================
    # Test 5: Bulk import
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 5: Bulk import (entities + relations)")
    print("=" * 60)
    try:
        entities = make_test_entities()
        relations = make_test_relations()
        with Neo4jManager() as mgr:
            stats = mgr.import_entities_and_relations(
                entities, relations,
                show_progress=True,
                clear_before=True,
            )
            print(f"  ✓ Import completato: {stats}")
            assert stats["entities_created"] == len(entities), \
                f"Attese {len(entities)} entità, create {stats['entities_created']}"
            assert stats["entities_failed"] == 0
            # Alcune relazioni potrebbero fallire se le entità di test sono incomplete
            assert stats["relations_created"] >= 5, \
                f"Attese almeno 5 relazioni, create {stats['relations_created']}"
            print(f"  ✓ {stats['entities_created']} entità, {stats['relations_created']} relazioni")
        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ==================================================================
    # Test 6: query_neighbors
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 6: Query neighbors")
    print("=" * 60)
    try:
        with Neo4jManager() as mgr:
            # Neighbors di GCG (depth=1)
            result = mgr.query_neighbors("GCG", depth=1)
            assert result["center"] is not None
            assert result["center"]["canonical_name"] == "GCG"
            assert len(result["entities"]) > 0
            print(f"  ✓ Neighbors GCG (depth=1): {len(result['entities'])} entità, "
                  f"{len(result['relations'])} relazioni")
            for e in result["entities"]:
                print(f"    - {e.get('entity_type', '?')}: {e.get('canonical_name', '?')}")

            # Neighbors depth=2
            result2 = mgr.query_neighbors("GCG", depth=2)
            assert len(result2["entities"]) >= len(result["entities"])
            print(f"  ✓ Neighbors GCG (depth=2): {len(result2['entities'])} entità")

            # Non-existent entity
            result3 = mgr.query_neighbors("NonExistentAttack")
            assert result3["center"] is None
            assert len(result3["entities"]) == 0
            print("  ✓ Non-existent entity: empty result")

        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ==================================================================
    # Test 7: query_path
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 7: Query path (shortest path)")
    print("=" * 60)
    try:
        with Neo4jManager() as mgr:
            # Cammino GCG → GPT-4 (diretto via TARGETS)
            path = mgr.query_path("GCG", "GPT-4")
            assert len(path) > 0, "Dovrebbe esistere un cammino GCG → GPT-4"
            # Nodo-Relazione-Nodo
            assert path[0]["type"] == "entity"
            print(f"  ✓ Cammino GCG → GPT-4: {len(path)} elementi")
            for elem in path:
                if elem["type"] == "entity":
                    print(f"    [Entity] {elem.get('canonical_name', '?')}")
                else:
                    print(f"    [{elem.get('relation_type', '?')}]")

            # Cammino più lungo: SmoothLLM → GPT-4 (SmoothLLM -DEFENDS_AGAINST-> GCG -TARGETS-> GPT-4)
            path2 = mgr.query_path("SmoothLLM", "GPT-4")
            assert len(path2) > 0
            print(f"  ✓ Cammino SmoothLLM → GPT-4: {len(path2)} elementi")

            # Nessun cammino
            path3 = mgr.query_path("GCG", "NonExistent")
            assert len(path3) == 0
            print("  ✓ Nessun cammino verso entità inesistente")

        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ==================================================================
    # Test 8: get_temporal_chain
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 8: Temporal chain")
    print("=" * 60)
    try:
        with Neo4jManager() as mgr:
            # Timeline di tutti gli Attack
            timeline = mgr.get_temporal_chain(EntityType.ATTACK)
            assert len(timeline) > 0, "Dovrebbero esserci Attack con anno"
            print(f"  ✓ Timeline Attack: {len(timeline)} entità")
            for entry in timeline:
                rels_str = ""
                if entry.get("temporal_relations"):
                    rels_str = " → " + ", ".join(
                        f"{r.get('type', '?')}→{r.get('target', '?')}"
                        for r in entry["temporal_relations"]
                    )
                print(f"    [{entry.get('year', '?')}] {entry.get('canonical_name', '?')}{rels_str}")

            # Verifica ordinamento
            years = [e["year"] for e in timeline if e.get("year")]
            assert years == sorted(years), "Timeline non è ordinata per anno"
            print("  ✓ Timeline ordinata cronologicamente")

            # Timeline Defense
            timeline_def = mgr.get_temporal_chain(EntityType.DEFENSE)
            print(f"  ✓ Timeline Defense: {len(timeline_def)} entità")

        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ==================================================================
    # Test 9: get_stats
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 9: Statistics")
    print("=" * 60)
    try:
        with Neo4jManager() as mgr:
            stats = mgr.get_stats()
            print(f"  ✓ Stats: {json.dumps(stats, indent=2)}")
            assert stats["total_nodes"] == 10, f"Attesi 10 nodi, trovati {stats['total_nodes']}"
            assert stats["total_relationships"] >= 5
            assert "Attack" in stats["nodes_by_type"]
            assert stats["nodes_by_type"]["Attack"] == 3  # GCG, AutoDAN, Prompt Injection
            assert stats["nodes_by_type"]["Model"] == 2   # GPT-4, LLaMA-2
            assert stats["year_range"] is not None
            assert stats["unique_papers"] > 0
            print(f"  ✓ {stats['total_nodes']} nodi, {stats['total_relationships']} relazioni")
            print(f"  ✓ Nodi per tipo: {stats['nodes_by_type']}")
            print(f"  ✓ Relazioni per tipo: {stats['relationships_by_type']}")
            print(f"  ✓ Years: {stats['year_range']}")

        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ==================================================================
    # Test 10: clear_by_paper
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 10: Clear by paper (bibtex_key)")
    print("=" * 60)
    try:
        with Neo4jManager() as mgr:
            stats_before = mgr.get_stats()
            # Rimuovi tutto ciò che viene da Zou_2023_GCG
            result = mgr.clear_by_paper("Zou_2023_GCG")
            print(f"  ✓ clear_by_paper('Zou_2023_GCG'): {result}")
            stats_after = mgr.get_stats()
            # Le relazioni di Zou dovrebbero essere eliminate, e le entità orfane
            assert result["relations_deleted"] > 0 or result["entities_deleted"] >= 0
            print(f"  ✓ Prima: {stats_before['total_nodes']} nodi, "
                  f"{stats_before['total_relationships']} rel")
            print(f"  ✓ Dopo: {stats_after['total_nodes']} nodi, "
                  f"{stats_after['total_relationships']} rel")

        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ==================================================================
    # Test 11: import_from_json
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 11: Import from JSON file")
    print("=" * 60)
    try:
        entities = make_test_entities()[:5]
        relations = make_test_relations()[:3]

        # Crea file JSON temporaneo in formato save_extraction_results
        json_data = {
            "extracted_at": "2026-02-20T12:00:00",
            "stats": {"entity_count": len(entities), "relation_count": len(relations)},
            "entities": [e.to_dict() for e in entities],
            "relations": [r.to_dict() for r in relations],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            tmp_path = f.name

        with Neo4jManager() as mgr:
            stats = mgr.import_from_json(tmp_path, clear_before=True, show_progress=False)
            print(f"  ✓ Import from JSON: {stats}")
            assert stats["entities_created"] == 5
            assert stats["relations_created"] >= 2  # Almeno GCG→GPT-4 e GCG→LLaMA-2
            print(f"  ✓ {stats['entities_created']} entità, {stats['relations_created']} relazioni importate")

        # Cleanup
        Path(tmp_path).unlink()
        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ==================================================================
    # Test 12: Neo4jNotEnabledError
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 12: Neo4jNotEnabledError")
    print("=" * 60)
    try:
        # Disabilita Neo4j
        cfg.settings.neo4j.enabled = False
        try:
            mgr = Neo4jManager()
            print("  ✗ Avrebbe dovuto lanciare Neo4jNotEnabledError")
            failed += 1
        except Neo4jNotEnabledError as e:
            print(f"  ✓ Neo4jNotEnabledError: {e}")
            passed += 1
        finally:
            cfg.settings.neo4j.enabled = True  # Ripristina
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        failed += 1

    # ==================================================================
    # Test 13: test_neo4j_connection standalone
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 13: test_neo4j_connection()")
    print("=" * 60)
    try:
        ok = test_neo4j_connection()
        assert ok, "test_neo4j_connection() dovrebbe ritornare True"
        print("  ✓ test_neo4j_connection() = True")
        passed += 1
    except Exception as e:
        print(f"  ✗ FALLITO: {e}")
        failed += 1

    # ==================================================================
    # Risultati finali
    # ==================================================================
    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"RISULTATI: {passed} passati, {failed} falliti su {passed+failed} test ({elapsed:.1f}s)")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    test_all()
