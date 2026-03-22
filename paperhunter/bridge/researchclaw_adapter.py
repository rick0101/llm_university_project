"""researchclaw_adapter — Bridge PaperHunter output → AutoResearchClaw.

Exports PaperHunter survey data (property graph, L2 summaries, bibliography)
into the stage directory format expected by ResearchClaw, allowing the
experimental pipeline to resume from stage 9 (EXPERIMENT_DESIGN).

Mapping:
  Property graph entities → stage-06/cards/*.md    (knowledge cards)
  L2 syntheses + taxonomy → stage-07/synthesis.md  (synthesis document)
  L2 open_gaps            → stage-08/hypotheses.md (research hypotheses)
  references.bib          → run_dir/references.bib (bibliography)
  checkpoint.json         → run_dir/checkpoint.json (resume point)
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ── Pydantic models ──────────────────────────────────────────────────────────


class KnowledgeCard(BaseModel):
    """Single knowledge card in ResearchClaw format."""

    card_id: str = Field(..., description="Unique card identifier")
    title: str = Field(..., description="Entity or paper title")
    cite_key: str = Field("", description="BibTeX citation key")
    entity_type: str = Field("", description="Entity type (Attack, Defense, …)")
    description: str = Field("", description="Entity description")
    aliases: list[str] = Field(default_factory=list, description="Known aliases")
    source_paper: str = Field("", description="Source paper identifier")
    year: int | None = Field(None, description="Publication year")

    def to_markdown(self) -> str:
        """Render the card as Markdown in ResearchClaw format."""
        sections = [f"# {self.title}"]

        if self.cite_key:
            sections.append(f"\n## Cite_key\n{self.cite_key}")

        if self.entity_type:
            sections.append(f"\n## Problem\nEntity type: {self.entity_type}")

        if self.description:
            sections.append(f"\n## Method\n{self.description}")

        if self.aliases:
            sections.append(f"\n## Findings\nAliases: {', '.join(self.aliases)}")

        if self.source_paper:
            sections.append(f"\n## Citation\n{self.source_paper}")

        if self.year:
            sections.append(f"\n## Data\nYear: {self.year}")

        return "\n".join(sections)


class ExportResult(BaseModel):
    """Result of a single bridge export operation."""

    stage: str = Field(..., description="Stage name (e.g. 'stage-06')")
    path: Path = Field(..., description="Output path")
    files_written: int = Field(0, description="Number of files written")
    success: bool = Field(True, description="Whether export succeeded")
    message: str = Field("", description="Status message")


class BridgeReport(BaseModel):
    """Summary of the full bridge export."""

    run_id: str = Field(..., description="ResearchClaw run identifier")
    run_dir: Path = Field(..., description="ResearchClaw run directory")
    exports: list[ExportResult] = Field(default_factory=list)
    checkpoint_path: Path | None = Field(None, description="Path to checkpoint.json")


# ── Adapter ──────────────────────────────────────────────────────────────────


class ResearchClawAdapter:
    """Bridge PaperHunter output → AutoResearchClaw run directory.

    Converts PaperHunter's property graph, L2 summaries, and bibliography
    into ResearchClaw's stage directory format, placing a checkpoint at
    stage 8 (HYPOTHESIS_GEN) so ``researchclaw run --resume`` begins at
    stage 9 (EXPERIMENT_DESIGN).
    """

    STAGE_KNOWLEDGE = 6
    STAGE_SYNTHESIS = 7
    STAGE_HYPOTHESIS = 8

    def __init__(
        self,
        paperhunter_root: Path,
        researchclaw_run_dir: Path,
        run_id: str | None = None,
    ):
        self.ph_root = Path(paperhunter_root)
        self.rc_dir = Path(researchclaw_run_dir)
        self.run_id = run_id or f"ph-bridge-{uuid.uuid4().hex[:8]}"
        self._ensure_dirs()

    # ── directory setup ──────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        """Create ResearchClaw stage directories."""
        for stage_num in (
            self.STAGE_KNOWLEDGE,
            self.STAGE_SYNTHESIS,
            self.STAGE_HYPOTHESIS,
        ):
            (self.rc_dir / f"stage-{stage_num:02d}").mkdir(
                parents=True, exist_ok=True
            )
        (self.rc_dir / "stage-06" / "cards").mkdir(parents=True, exist_ok=True)

    # ── data loaders ─────────────────────────────────────────────────────

    def _load_property_graph(self) -> dict[str, Any]:
        """Load PaperHunter's property graph from ``graph_store.json``."""
        graph_path = self.ph_root / "data" / "property_graph" / "graph_store.json"
        if not graph_path.exists():
            logger.warning("Property graph not found: %s", graph_path)
            return {"nodes": {}, "relations": {}}

        with open(graph_path, encoding="utf-8") as f:
            data = json.load(f)

        logger.info(
            "Loaded property graph: %d nodes, %d relations",
            len(data.get("nodes", {})),
            len(data.get("relations", {})),
        )
        return data

    def _load_l2_summaries(self) -> list[dict[str, Any]]:
        """Load L2 summaries — Qdrant first, disk metadata fallback."""
        try:
            return self._load_l2_from_qdrant()
        except Exception as exc:
            logger.warning("Cannot load L2 from Qdrant: %s — falling back to disk.", exc)
            return self._load_l2_from_disk()

    def _load_l2_from_qdrant(self) -> list[dict[str, Any]]:
        """Scroll all non-invalidated L2 summaries from Qdrant."""
        from paperhunter.config import settings
        from paperhunter.indexing.qdrant_store import QdrantStore

        store = QdrantStore()
        collection = settings.qdrant.collection_l2_summaries

        all_points: list[Any] = []
        offset = None
        while True:
            points, next_offset = store.client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_offset is None or not points:
                break
            offset = next_offset

        l2s: list[dict[str, Any]] = []
        for point in all_points:
            payload = point.payload or {}
            meta = payload.get("metadata", {})
            if meta.get("invalidated", False):
                continue

            category = meta.get("category", payload.get("category", "Unknown"))
            l2: dict[str, Any] = {
                "category": category,
                "paper_ids": meta.get("paper_ids", []),
                "paper_count": meta.get("paper_count", 0),
                "version": meta.get("version", 1),
                "synthesis": payload.get("text", ""),
                "taxonomy": "",
                "open_gaps": "",
                "temporal_trends": "",
                "gaps_filled": "",
                "methodological_convergences": "",
                "testing_overview": "",
            }
            self._enrich_l2_from_disk(l2, category)
            l2s.append(l2)

        logger.info("Loaded %d L2 summaries from Qdrant", len(l2s))
        return l2s

    def _enrich_l2_from_disk(self, l2: dict[str, Any], category: str) -> None:
        """Enrich an L2 dict with extended fields from disk metadata."""
        try:
            from paperhunter.summarization.level2_summarizer import (
                load_all_l2_metadata,
            )

            all_meta = load_all_l2_metadata()
            if category not in all_meta:
                return

            disk = all_meta[category]
            for field in (
                "temporal_trends",
                "taxonomy",
                "gaps_filled",
                "open_gaps",
                "methodological_convergences",
                "testing_overview",
                "synthesis",
            ):
                if field in disk and disk[field]:
                    l2[field] = disk[field]
        except Exception as exc:
            logger.debug("Cannot enrich L2 '%s' from disk: %s", category, exc)

    def _load_l2_from_disk(self) -> list[dict[str, Any]]:
        """Load minimal L2 metadata from disk JSON files."""
        meta_dir = self.ph_root / "output" / "l2_metadata"
        if not meta_dir.exists():
            logger.warning("L2 metadata dir not found: %s", meta_dir)
            return []

        l2s: list[dict[str, Any]] = []
        for fp in sorted(meta_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                l2s.append(
                    {
                        "category": data.get("category", fp.stem),
                        "paper_ids": data.get("paper_ids", []),
                        "paper_count": data.get("paper_count", 0),
                        "version": data.get("version", 1),
                        "synthesis": data.get("synthesis", ""),
                        "taxonomy": data.get("taxonomy", ""),
                        "open_gaps": data.get("open_gaps", ""),
                        "temporal_trends": data.get("temporal_trends", ""),
                        "gaps_filled": data.get("gaps_filled", ""),
                        "methodological_convergences": data.get(
                            "methodological_convergences", ""
                        ),
                        "testing_overview": data.get("testing_overview", ""),
                    }
                )
            except Exception as exc:
                logger.warning("Cannot load L2 metadata from %s: %s", fp, exc)

        logger.info("Loaded %d L2 summaries from disk metadata", len(l2s))
        return l2s

    def _load_bibtex(self) -> str:
        """Load BibTeX from ``output/references.bib``."""
        for bib_name in ("references_final.bib", "references.bib"):
            bib_path = self.ph_root / "output" / bib_name
            if bib_path.exists():
                content = bib_path.read_text(encoding="utf-8")
                logger.info("Loaded BibTeX from %s (%d chars)", bib_path, len(content))
                return content

        logger.warning("No BibTeX file found in output/")
        return ""

    # ── export methods ───────────────────────────────────────────────────

    def export_knowledge_cards(self) -> ExportResult:
        """Property graph entities → ``stage-06/cards/*.md``."""
        cards_dir = self.rc_dir / "stage-06" / "cards"
        graph = self._load_property_graph()
        nodes = graph.get("nodes", {})

        if not nodes:
            return ExportResult(
                stage="stage-06",
                path=cards_dir,
                success=False,
                message="No property graph nodes found",
            )

        count = 0
        for node_name, node_data in nodes.items():
            props = node_data.get("properties", {})

            card = KnowledgeCard(
                card_id=f"card-{count + 1}",
                title=node_name,
                cite_key=props.get(
                    "source_bibtex_key", props.get("source_paper_id", "")
                ),
                entity_type=props.get("entity_type", node_data.get("label", "")),
                description=props.get("description", ""),
                aliases=props.get("aliases", []),
                source_paper=props.get("source_paper_id", ""),
                year=props.get("year"),
            )

            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", node_name)[:80]
            card_path = cards_dir / f"{card.card_id}_{safe_name}.md"
            card_path.write_text(card.to_markdown(), encoding="utf-8")
            count += 1

        logger.info("Exported %d knowledge cards to %s", count, cards_dir)
        return ExportResult(
            stage="stage-06",
            path=cards_dir,
            files_written=count,
            message=f"Exported {count} knowledge cards from property graph",
        )

    def export_synthesis(self) -> ExportResult:
        """L2 syntheses + taxonomy → ``stage-07/synthesis.md``."""
        output_path = self.rc_dir / "stage-07" / "synthesis.md"
        l2s = self._load_l2_summaries()

        if not l2s:
            return ExportResult(
                stage="stage-07",
                path=output_path,
                success=False,
                message="No L2 summaries available",
            )

        sections = [
            "# Research Synthesis\n",
            (
                f"*Exported from PaperHunter — "
                f"{datetime.now(timezone.utc).isoformat()}*\n"
            ),
        ]

        for l2 in l2s:
            category = l2.get("category", "Unknown")
            sections.append(f"\n## {category}\n")

            for field, heading in (
                ("synthesis", "Synthesis"),
                ("taxonomy", "Taxonomy"),
                ("temporal_trends", "Temporal Trends"),
                ("methodological_convergences", "Methodological Convergences"),
                ("gaps_filled", "Gaps Filled"),
            ):
                value = l2.get(field, "")
                if value:
                    sections.append(f"### {heading}\n{value}\n")

            paper_ids = l2.get("paper_ids", [])
            if paper_ids:
                refs = ", ".join(f"[{pid}]" for pid in paper_ids)
                sections.append(f"### References\n{refs}\n")

        output_path.write_text("\n".join(sections), encoding="utf-8")
        logger.info(
            "Exported synthesis (%d categories) to %s", len(l2s), output_path
        )

        return ExportResult(
            stage="stage-07",
            path=output_path,
            files_written=1,
            message=f"Combined {len(l2s)} L2 syntheses",
        )

    def export_hypotheses(self) -> ExportResult:
        """L2 open_gaps + testing_overview → ``stage-08/hypotheses.md``."""
        output_path = self.rc_dir / "stage-08" / "hypotheses.md"
        l2s = self._load_l2_summaries()

        if not l2s:
            return ExportResult(
                stage="stage-08",
                path=output_path,
                success=False,
                message="No L2 summaries available",
            )

        sections = [
            "# Research Hypotheses\n",
            (
                f"*Generated from PaperHunter open gaps — "
                f"{datetime.now(timezone.utc).isoformat()}*\n"
            ),
        ]

        hyp_count = 0
        for l2 in l2s:
            category = l2.get("category", "Unknown")
            open_gaps = l2.get("open_gaps", "")
            testing = l2.get("testing_overview", "")

            if not open_gaps and not testing:
                continue

            sections.append(f"\n## {category}\n")

            if open_gaps:
                sections.append(f"### Open Research Gaps\n{open_gaps}\n")
                hyp_count += 1

            if testing:
                sections.append(f"### Testing Opportunities\n{testing}\n")

        if hyp_count == 0:
            sections.append(
                "\n*No open gaps found in L2 summaries. "
                "Run `paperhunter summarize` with Qdrant to populate.*\n"
            )

        output_path.write_text("\n".join(sections), encoding="utf-8")
        logger.info(
            "Exported hypotheses (%d gap sections) to %s", hyp_count, output_path
        )

        return ExportResult(
            stage="stage-08",
            path=output_path,
            files_written=1,
            message=f"Generated hypotheses from {hyp_count} category gaps",
        )

    def export_references(self) -> ExportResult:
        """Copy BibTeX to ResearchClaw run directory."""
        bibtex = self._load_bibtex()
        output_path = self.rc_dir / "references.bib"

        if not bibtex:
            return ExportResult(
                stage="references",
                path=output_path,
                success=False,
                message="No BibTeX file found",
            )

        output_path.write_text(bibtex, encoding="utf-8")
        logger.info("Exported references to %s", output_path)

        return ExportResult(
            stage="references",
            path=output_path,
            files_written=1,
            message=f"Copied BibTeX ({len(bibtex)} chars)",
        )

    def create_checkpoint(self) -> ExportResult:
        """Create ``checkpoint.json`` at stage 8 (HYPOTHESIS_GEN)."""
        ckpt = {
            "last_completed_stage": self.STAGE_HYPOTHESIS,
            "last_completed_name": "HYPOTHESIS_GEN",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        output_path = self.rc_dir / "checkpoint.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Created checkpoint at stage 8: %s", output_path)

        return ExportResult(
            stage="checkpoint",
            path=output_path,
            files_written=1,
            message=f"Checkpoint at stage 8 (HYPOTHESIS_GEN), run_id={self.run_id}",
        )

    # ── orchestrator ─────────────────────────────────────────────────────

    def export_all(self) -> BridgeReport:
        """Run all exports and return a comprehensive report."""
        report = BridgeReport(
            run_id=self.run_id,
            run_dir=self.rc_dir,
        )

        logger.info("Starting full bridge export (run_id=%s)", self.run_id)
        logger.info("  PaperHunter root: %s", self.ph_root)
        logger.info("  ResearchClaw dir:  %s", self.rc_dir)

        for export_fn in (
            self.export_knowledge_cards,
            self.export_synthesis,
            self.export_hypotheses,
            self.export_references,
            self.create_checkpoint,
        ):
            try:
                result = export_fn()
                report.exports.append(result)
            except Exception as exc:
                logger.error("Export failed (%s): %s", export_fn.__name__, exc)
                report.exports.append(
                    ExportResult(
                        stage=export_fn.__name__,
                        path=self.rc_dir,
                        success=False,
                        message=str(exc),
                    )
                )

        report.checkpoint_path = self.rc_dir / "checkpoint.json"

        ok = sum(1 for e in report.exports if e.success)
        total = len(report.exports)
        logger.info("Bridge export complete: %d/%d stages succeeded", ok, total)

        return report
