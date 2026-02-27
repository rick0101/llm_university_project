"""
GraphRAG — Knowledge Graph per LLM Security Survey.

Moduli:
- ``entity_extractor``: Estrazione entità e relazioni dai summary L1
- ``neo4j_manager``: Connessione e gestione Neo4j Community Edition
- ``property_graph``: LlamaIndex PropertyGraph builder
"""

from paperhunter.graphrag.entity_extractor import (
    Entity,
    EntityType,
    ExtractionResult,
    Relation,
    RelationType,
    extract_batch,
    extract_entities_and_relations,
    extract_with_details,
    get_extraction_stats,
    merge_entities,
    merge_relations,
    normalize_entity_name,
    save_extraction_results,
)
from paperhunter.graphrag.neo4j_manager import (
    Neo4jConnectionError,
    Neo4jManager,
    Neo4jNotEnabledError,
    test_neo4j_connection,
)
from paperhunter.graphrag.property_graph import (
    BuildReport,
    GraphQueryResult,
    PropertyGraphBuilder,
    build_property_graph,
    query_property_graph,
)

__all__ = [
    # Entity extractor
    "Entity",
    "EntityType",
    "ExtractionResult",
    "Relation",
    "RelationType",
    "extract_entities_and_relations",
    "extract_with_details",
    "extract_batch",
    "merge_entities",
    "merge_relations",
    "normalize_entity_name",
    "get_extraction_stats",
    "save_extraction_results",
    # Neo4j manager
    "Neo4jManager",
    "Neo4jNotEnabledError",
    "Neo4jConnectionError",
    "test_neo4j_connection",
    # PropertyGraph builder
    "PropertyGraphBuilder",
    "GraphQueryResult",
    "BuildReport",
    "build_property_graph",
    "query_property_graph",
]
