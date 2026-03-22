"""Bridge modules for connecting PaperHunter to external research tools."""

from paperhunter.bridge.researchclaw_adapter import (
    BridgeReport,
    ExportResult,
    ImportReport,
    KnowledgeCard,
    ResearchClawAdapter,
)

__all__ = [
    "BridgeReport",
    "ExportResult",
    "ImportReport",
    "KnowledgeCard",
    "ResearchClawAdapter",
]
