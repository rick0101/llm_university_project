"""postprocessing — Modulo di post-processing per il survey generato.

Contiene:
- ProvenanceTracker: tracking completo claim → fonte con verifica
- (futuro) BibliographyGenerator: generazione .bib finale
- (futuro) LaTeXExporter: export .tex
- (futuro) VersionManager: versioning e changelog
- (futuro) CheckpointManager: resume from checkpoint
"""

from paperhunter.postprocessing.provenance_tracker import (
    ProvenanceTracker,
    create_provenance_from_state,
    load_provenance,
)

__all__ = [
    "ProvenanceTracker",
    "create_provenance_from_state",
    "load_provenance",
]
