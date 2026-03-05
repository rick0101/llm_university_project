"""
paperhunter.agents.tools — Tool invocabili dal Writer ReAct Agent.

Espone i retrieval tool (vector, BM25, graph community) come
``FunctionTool`` di LlamaIndex, pronti per essere registrati
in un ``ReActAgent`` o ``AgentRunner``.

Exports principali:
    search_vector_database: Ricerca vettoriale su Qdrant
    search_keyword_index: Ricerca BM25 keyword-based
    search_graph_communities: Ricerca nei community summary del knowledge graph
    get_all_retrieval_tools: Helper per ottenere tutti i tool in una lista

Uso::

    from paperhunter.agents.tools import get_all_retrieval_tools

    tools = get_all_retrieval_tools()
    # → [FunctionTool(search_vector_database), FunctionTool(search_keyword_index),
    #    FunctionTool(search_graph_communities)]
"""

from paperhunter.agents.tools.retrieval_tools import (
    get_all_retrieval_tools,
    search_graph_communities,
    search_keyword_index,
    search_vector_database,
)

__all__ = [
    "search_vector_database",
    "search_keyword_index",
    "search_graph_communities",
    "get_all_retrieval_tools",
]
