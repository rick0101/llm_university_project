"""
Search Checkpoint Manager for PaperHunter.

Tracks search results, LLM filtering, and download progress to allow
resuming interrupted search sessions.  Generates a human-readable .md
alongside a machine-readable .json checkpoint after every paper update.

Checkpoint identity is a deterministic hash of ALL search parameters
(keywords, criteria, max_results, output_dir, date_from, date_to).
Different parameters → different checkpoint.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paper status values ─────────────────────────────────────────────────
STATUS_PENDING = "pending"
STATUS_RELEVANT = "relevant"
STATUS_NOT_RELEVANT = "not_relevant"
STATUS_FILTER_ERROR = "filter_error"
STATUS_DOWNLOADED = "downloaded"
STATUS_DOWNLOAD_FAILED = "download_failed"
STATUS_DEDUP_SKIPPED = "dedup_skipped"

# ── Pipeline phases ─────────────────────────────────────────────────────
PHASE_SEARCH_DONE = "search_complete"
PHASE_FILTERING = "filtering"
PHASE_FILTER_DONE = "filter_complete"
PHASE_DOWNLOADING = "downloading"
PHASE_COMPLETE = "complete"


class SearchCheckpointManager:
    """Manages JSON + Markdown checkpoints for the ``paperhunter search`` flow.

    Lifecycle
    ---------
    1. ``create_from_search``  – after multi-source search, saves all found papers.
    2. ``update_paper``        – after each LLM filter / download step.
    3. ``set_phase``           – marks phase transitions.
    4. ``find_checkpoint``     – on re-run, detects an identical search.
    """

    def __init__(self, checkpoint_dir: Path | str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir or "cache/search_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════
    # Identity
    # ════════════════════════════════════════════════════════════════════

    @staticmethod
    def compute_search_id(
        keywords: str,
        criteria: str,
        max_results: int,
        output_dir: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        """Deterministic 16-char hex hash of all search parameters."""
        params = {
            "keywords": keywords.strip(),
            "criteria": criteria.strip(),
            "max_results": max_results,
            "output_dir": str(Path(output_dir).resolve()),
            "date_from": date_from or "",
            "date_to": date_to or "",
        }
        canonical = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ════════════════════════════════════════════════════════════════════
    # Lookup / persistence
    # ════════════════════════════════════════════════════════════════════

    def find_checkpoint(self, search_id: str) -> dict | None:
        """Load an existing checkpoint by *search_id*, or ``None``."""
        json_path = self.checkpoint_dir / f"{search_id}.json"
        if not json_path.exists():
            return None
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Checkpoint corrotto ({json_path}): {e}")
            return None

    def save(self, data: dict) -> Path:
        """Persist checkpoint (JSON + .md) and return the .md path."""
        sid = data["search_id"]
        data["updated_at"] = datetime.now().isoformat()

        json_path = self.checkpoint_dir / f"{sid}.json"
        json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        md_path = self.checkpoint_dir / f"{sid}.md"
        md_path.write_text(self._render_md(data), encoding="utf-8")
        return md_path

    # ════════════════════════════════════════════════════════════════════
    # Create / mutate
    # ════════════════════════════════════════════════════════════════════

    def create_from_search(self, search_id, params, papers, search_meta):
        """Create a new checkpoint right after the search phase.

        Parameters
        ----------
        search_id : str
            Deterministic hash of search params.
        params : dict
            ``{keywords, criteria, max_results, output_dir, date_from, date_to}``.
        papers : list[Paper]
            All papers returned by the aggregator search.
        search_meta : dict
            ``{total_found, sources_queried, errors, source_counts}``.
        """
        data = {
            "search_id": search_id,
            "params": params,
            "search_meta": search_meta,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "phase": PHASE_SEARCH_DONE,
            "papers": [],
        }
        for i, p in enumerate(papers):
            src = p.source
            if hasattr(src, "value"):
                src = src.value
            data["papers"].append({
                "index": i,
                "title": p.title,
                "authors": p.authors[:5],
                "abstract": p.abstract or "",
                "url": p.url,
                "pdf_url": p.pdf_url,
                "arxiv_id": p.arxiv_id,
                "doi": p.doi,
                "source": str(src),
                "published_date": p.published_date.isoformat() if p.published_date else None,
                "github_url": p.github_url,
                "categories": p.categories[:5] if p.categories else [],
                "citations": p.citations,
                # Tracking fields
                "status": STATUS_PENDING,
                "relevance_reasoning": None,
                "relevance_confidence": None,
                "download_path": None,
                "download_error": None,
            })
        self.save(data)
        return data

    def update_paper(self, data, index, status, **extra):
        """Update a single paper's status (+ optional fields) and persist."""
        data["papers"][index]["status"] = status
        for k, v in extra.items():
            data["papers"][index][k] = v
        self.save(data)

    def set_phase(self, data, phase):
        """Update the pipeline phase and persist."""
        data["phase"] = phase
        self.save(data)

    # ════════════════════════════════════════════════════════════════════
    # Queries
    # ════════════════════════════════════════════════════════════════════

    def get_pending_filter(self, data):
        """Indices of papers not yet analysed by the LLM filter."""
        return [p["index"] for p in data["papers"] if p["status"] == STATUS_PENDING]

    def get_pending_download(self, data):
        """Indices of relevant papers not yet downloaded."""
        return [p["index"] for p in data["papers"] if p["status"] == STATUS_RELEVANT]

    # ════════════════════════════════════════════════════════════════════
    # Reconstruction helpers
    # ════════════════════════════════════════════════════════════════════

    def reconstruct_paper(self, paper_data):
        """Rebuild a ``Paper`` Pydantic object from checkpoint dict."""
        from .models import Paper

        pub_date = None
        if paper_data.get("published_date"):
            try:
                pub_date = datetime.fromisoformat(paper_data["published_date"])
            except (ValueError, TypeError):
                pass

        source = paper_data.get("source", "arxiv")
        if source not in ("arxiv", "semantic_scholar", "huggingface"):
            source = "arxiv"

        return Paper(
            title=paper_data["title"],
            authors=paper_data.get("authors", []),
            abstract=paper_data.get("abstract", ""),
            url=paper_data.get("url", ""),
            pdf_url=paper_data.get("pdf_url"),
            source=source,
            published_date=pub_date,
            arxiv_id=paper_data.get("arxiv_id"),
            doi=paper_data.get("doi"),
            github_url=paper_data.get("github_url"),
            categories=paper_data.get("categories", []),
            citations=paper_data.get("citations"),
        )

    def reconstruct_for_report(self, data):
        """Rebuild the objects expected by ``generate_report`` from checkpoint.

        Returns
        -------
        (SearchResult, list[FilteredPaper], list[DownloadResult], list[Paper])
        """
        from .models import (
            DownloadResult,
            FilteredPaper,
            Paper,
            RelevanceResult,
            SearchResult,
        )

        all_papers = []
        filtered_results = []
        download_results = []
        relevant_papers = []

        for pd in data["papers"]:
            paper = self.reconstruct_paper(pd)
            all_papers.append(paper)

            is_relevant = pd["status"] in (
                STATUS_RELEVANT,
                STATUS_DOWNLOADED,
                STATUS_DOWNLOAD_FAILED,
                STATUS_DEDUP_SKIPPED,
            )

            filtered_results.append(FilteredPaper(
                paper=paper,
                relevance=RelevanceResult(
                    is_relevant=is_relevant,
                    reasoning=pd.get("relevance_reasoning") or "N/A",
                    confidence=pd.get("relevance_confidence") or 0.0,
                ),
            ))

            if is_relevant:
                relevant_papers.append(paper)
                if pd["status"] == STATUS_DOWNLOADED:
                    download_results.append(DownloadResult(
                        success=True, file_path=pd.get("download_path"), method="direct",
                    ))
                elif pd["status"] == STATUS_DEDUP_SKIPPED:
                    download_results.append(DownloadResult(
                        success=True, skipped=True, method="dedup",
                    ))
                elif pd["status"] == STATUS_DOWNLOAD_FAILED:
                    download_results.append(DownloadResult(
                        success=False, error_message=pd.get("download_error"),
                    ))

        meta = data.get("search_meta", {})
        search_result = SearchResult(
            papers=all_papers,
            total_found=meta.get("total_found", len(all_papers)),
            sources_queried=meta.get("sources_queried", []),
            errors=meta.get("errors", []),
        )

        return search_result, filtered_results, download_results, relevant_papers

    def get_md_path(self, search_id):
        return self.checkpoint_dir / f"{search_id}.md"

    # ════════════════════════════════════════════════════════════════════
    # Markdown rendering
    # ════════════════════════════════════════════════════════════════════

    def _render_md(self, data):
        p = data["params"]
        papers = data["papers"]
        phase = data["phase"]

        phase_label = {
            PHASE_SEARCH_DONE: "🔍 Ricerca completata — in attesa filtro LLM",
            PHASE_FILTERING: "⏳ Filtro LLM in corso…",
            PHASE_FILTER_DONE: "✅ Filtro completato — in attesa download",
            PHASE_DOWNLOADING: "⬇️  Download in corso…",
            PHASE_COMPLETE: "✅ Completato",
        }.get(phase, phase)

        # Counts by status
        counts: dict[str, int] = {}
        for pp in papers:
            s = pp["status"]
            counts[s] = counts.get(s, 0) + 1

        lines = [
            "# 🔍 Search Checkpoint",
            "",
            f"> **Checkpoint ID:** `{data['search_id']}`  ",
            f"> **Creato:** {data['created_at'][:19]}  ",
            f"> **Aggiornato:** {data['updated_at'][:19]}  ",
            f"> **Fase:** {phase_label}",
            "",
            "## Parametri di Ricerca",
            "",
            "| Parametro | Valore |",
            "|-----------|--------|",
            f"| Keywords | `{p['keywords']}` |",
            f"| Criteria | {p['criteria'][:120]} |",
            f"| Max Results | {p['max_results']} |",
            f"| Output Dir | `{p['output_dir']}` |",
            f"| Date From | {p.get('date_from') or '—'} |",
            f"| Date To | {p.get('date_to') or '—'} |",
            "",
            "## Riepilogo",
            "",
            f"- **Totale:** {len(papers)} paper trovati",
        ]

        status_labels = [
            (STATUS_PENDING, "In attesa"),
            (STATUS_RELEVANT, "Rilevanti (da scaricare)"),
            (STATUS_NOT_RELEVANT, "Non rilevanti"),
            (STATUS_FILTER_ERROR, "Errore filtro"),
            (STATUS_DOWNLOADED, "Scaricati"),
            (STATUS_DOWNLOAD_FAILED, "Download falliti"),
            (STATUS_DEDUP_SKIPPED, "Già esistenti (dedup)"),
        ]
        for status, label in status_labels:
            if counts.get(status):
                lines.append(f"- **{label}:** {counts[status]}")

        lines.append("")
        lines.append(f"## Papers ({len(papers)})")
        lines.append("")

        # Checkbox markers
        marker_map = {
            STATUS_PENDING: "[ ]",
            STATUS_RELEVANT: "[ ]",
            STATUS_NOT_RELEVANT: "[x]",
            STATUS_FILTER_ERROR: "[x]",
            STATUS_DOWNLOADED: "[✓]",
            STATUS_DOWNLOAD_FAILED: "[!]",
            STATUS_DEDUP_SKIPPED: "[~]",
        }
        tag_map = {
            STATUS_PENDING: "",
            STATUS_RELEVANT: " — 🟡 Rilevante",
            STATUS_NOT_RELEVANT: " — ❌ Non rilevante",
            STATUS_FILTER_ERROR: " — ⚠️ Errore filtro",
            STATUS_DOWNLOADED: " — ✅ Scaricato",
            STATUS_DOWNLOAD_FAILED: " — ⚠️ Download fallito",
            STATUS_DEDUP_SKIPPED: " — 📎 Già esistente",
        }

        for pp in papers:
            idx = pp["index"] + 1
            marker = marker_map.get(pp["status"], "[ ]")
            title = pp["title"][:90]
            src = pp.get("source", "?")
            arxiv_part = f" `{pp['arxiv_id']}`" if pp.get("arxiv_id") else ""
            url = pp.get("url", "")
            tag = tag_map.get(pp["status"], "")
            link_part = f" — [link]({url})" if url else ""
            lines.append(
                f"- {marker} **{idx}.** {title} — *{src}*{arxiv_part}{link_part}{tag}"
            )

        lines.extend([
            "",
            "### Legenda",
            "",
            "- `[ ]` Non ancora processato",
            "- `[x]` Non rilevante (filtro LLM)",
            "- `[✓]` Scaricato con successo",
            "- `[!]` Download fallito",
            "- `[~]` Già esistente (dedup)",
            "",
        ])

        return "\n".join(lines)
