"""Test launcher: esegue la pipeline writer in modalità ``summaries-only``.

Lancia ``Planner → Writer → Critic → Assembler`` ma il writer (e i suoi
tool di retrieval) attingono SOLO alle collezioni di riassunti
``l1_summaries`` + ``l1_5_summaries`` + ``l2_summaries`` invece che ai
chunk full-text dei paper. Le citazioni ``[bibtex_key]`` vengono
preservate perché sia gli L1 (per-paper) sia il testo degli L1.5/L2
contengono i riferimenti originali.

L'output viene scritto in ``output_solo_riassunti/`` per non sovrascrivere
il survey baseline. Riusa l'outline ``sample_outline.json`` per garantire
che le due esecuzioni partano dalla stessa struttura.

Uso::

    python run_writer_summaries_only.py [--outline path/to/outline.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outline",
        default="sample_outline.json",
        help="Outline JSON da riusare (default: sample_outline.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_solo_riassunti",
        help="Directory di output dedicata (default: output_solo_riassunti).",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Salta export LaTeX/BibTeX/provenance/version.",
    )
    args = parser.parse_args()

    # ── 1. Attiva la modalità summaries-only PRIMA di importare paperhunter ──
    os.environ["PAPERHUNTER_WRITER_SUMMARIES_ONLY"] = "1"

    # ── 2. Reindirizza tutte le directory di output ──
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    from paperhunter.config import settings
    settings.output.base_dir = str(out_root)
    settings.output.versions_dir = str(out_root / "versions")
    settings.output.checkpoints_dir = str(out_root / ".checkpoints")
    settings.output.provenance_dir = str(out_root / "provenance")
    for sub in ("versions", ".checkpoints", "provenance"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    print(f"[summaries-only] output → {out_root}")
    print(f"[summaries-only] writer_mode = {settings.agents.writer_mode}")

    # ── 3. Esegui la pipeline ──
    from paperhunter.agents import (
        run_pipeline, get_pipeline_summary,
    )

    outline_path = Path(args.outline)
    if not outline_path.exists():
        print(f"ERRORE: outline non trovato: {outline_path}", file=sys.stderr)
        return 1

    metadata = {"outline_file": str(outline_path), "experiment": "summaries_only"}

    t0 = time.time()
    state = run_pipeline(
        outline_file=str(outline_path),
        metadata=metadata,
        save_checkpoints=True,
    )
    elapsed = time.time() - t0
    print(f"[summaries-only] pipeline completata in {elapsed:.0f}s")

    summary = get_pipeline_summary(state)
    print("[summaries-only] summary:", json.dumps(summary, indent=2, default=str))

    # ── 4. Auto-export ──
    if not args.no_export:
        try:
            bib_path = None
            if settings.output.generate_bibtex:
                from paperhunter.postprocessing import generate_bib_from_state
                bib_path, bib_report = generate_bib_from_state(state)
                if bib_path:
                    print(
                        f"[summaries-only] BibTeX: {bib_path} "
                        f"({bib_report.get('total_filtered', 0)} entry)"
                    )

            if settings.output.generate_latex:
                from paperhunter.postprocessing import export_latex_from_state
                tex_path = export_latex_from_state(
                    state,
                    template=settings.output.latex_template,
                    master_bib_path=bib_path,
                )
                if tex_path:
                    print(f"[summaries-only] LaTeX: {tex_path}")

            if settings.output.generate_provenance_json:
                from paperhunter.postprocessing import create_provenance_from_state
                tracker = create_provenance_from_state(state)
                prov_path = tracker.save_json()
                print(f"[summaries-only] Provenance: {prov_path}")

            from paperhunter.postprocessing import create_version_from_state
            version = create_version_from_state(state)
            print(f"[summaries-only] Versione: {version.version_number}")
        except Exception as e:
            print(f"[summaries-only] Errore export (non bloccante): {e}")

    return 0 if summary.get("phase") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
