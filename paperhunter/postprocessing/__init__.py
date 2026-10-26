"""postprocessing — Modulo di post-processing per il survey generato.

Contiene:
- ProvenanceTracker: tracking completo claim → fonte con verifica
- BibliographyGenerator: generazione .bib finale filtrato
- LaTeXExporter: export Markdown → .tex (ACL / IEEEtran / plain)
- VersionManager: versioning e changelog automatico
- CheckpointManager: gestione centralizzata checkpoint per resume from failure
"""

from paperhunter.postprocessing.provenance_tracker import (
    ProvenanceTracker,
    create_provenance_from_state,
    load_provenance,
)

from paperhunter.postprocessing.bibliography_generator import (
    BibliographyGenerator,
    generate_final_bib,
    generate_bib_from_state,
    load_and_filter_bib,
    merge_bib_files,
)

from paperhunter.postprocessing.latex_exporter import (
    markdown_to_latex,
    export_latex,
    export_latex_from_state,
    get_export_stats,
    print_export_summary,
)

from paperhunter.postprocessing.versioning import (
    VersionManager,
    create_version_from_state,
    load_version_history,
    get_latest_version,
)

from paperhunter.postprocessing.checkpoint_manager import (
    CheckpointManager,
    CheckpointInfo,
    get_checkpoint_manager,
    get_resume_recommendation,
    list_all_checkpoints,
)

__all__ = [
    # Provenance
    "ProvenanceTracker",
    "create_provenance_from_state",
    "load_provenance",
    # Bibliography
    "BibliographyGenerator",
    "generate_final_bib",
    "generate_bib_from_state",
    "load_and_filter_bib",
    "merge_bib_files",
    # LaTeX Export
    "markdown_to_latex",
    "export_latex",
    "export_latex_from_state",
    "get_export_stats",
    "print_export_summary",
    # Versioning
    "VersionManager",
    "create_version_from_state",
    "load_version_history",
    "get_latest_version",
    # Checkpoint Manager
    "CheckpointManager",
    "CheckpointInfo",
    "get_checkpoint_manager",
    "get_resume_recommendation",
    "list_all_checkpoints",
]
