"""
PaperWriter CLI — Generate scientific papers from PDF sources using RAG.

Commands
--------
  ingest        Process PDFs and load them into the vector database.
  stats         Show knowledge-base statistics.
  query         Ask a free-form question to the knowledge base.
  write         Generate a paper from a JSON outline file.
  interactive   Guided step-by-step workflow.

Examples
--------
  # 1. Ingest PDFs
  python -m paperhunter.generate_paper ingest ./my_pdfs

  # 2. Check what's in the DB
  python -m paperhunter.generate_paper stats

  # 3. Ask a question
  python -m paperhunter.generate_paper query "What are the main prompt injection attacks?"

  # 4. Generate a full paper from an outline
  python -m paperhunter.generate_paper write --title "Survey on LLM Security" --outline outline.json --output ./output

  # 5. Interactive mode (guided)
  python -m paperhunter.generate_paper interactive
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .paper_generator.paper_writer import PaperWriter

# Load .env before anything else
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("paperhunter.cli")


# ── Shared builder ───────────────────────────────────────────────────────

def _build_writer(args: argparse.Namespace) -> PaperWriter:
    """Construct a PaperWriter from CLI arguments."""
    return PaperWriter(
        persist_dir=args.db_dir,
        collection_name=args.collection,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        heavy_llm_model=args.heavy_llm_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        temperature=args.temperature,
    )


# ── Sub-commands ─────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        logger.error(f"Directory not found: {pdf_dir}")
        sys.exit(1)

    writer = _build_writer(args)
    result = writer.ingest_papers(pdf_dir, recursive=not args.no_recursive)

    print("\n✔  Ingestion complete")
    for k, v in result.items():
        print(f"   {k}: {v}")


def cmd_stats(args: argparse.Namespace) -> None:
    writer = _build_writer(args)
    stats = writer.get_stats()

    print(f"\n📊  Knowledge-base statistics")
    print(f"   Total chunks : {stats['total_chunks']}")
    print(f"   Total papers : {stats['total_papers']}")
    print(f"   Total files  : {stats['total_files']}")
    if stats.get("paper_titles"):
        print("\n   Papers indexed:")
        for t in stats["paper_titles"]:
            print(f"     • {t}")


def cmd_query(args: argparse.Namespace) -> None:
    writer = _build_writer(args)
    result = writer.query(args.question, n_results=args.n_results)

    print(f"\n{'=' * 60}")
    print(result["answer"])
    print(f"\n{'=' * 60}")
    print(f"Chunks retrieved: {result['chunks_retrieved']}")
    if result["sources"]:
        print("Sources used:")
        for s in result["sources"]:
            print(f"  [{s['citation_key']}] {s['paper_title']}  ({s['authors']})")


def cmd_write(args: argparse.Namespace) -> None:
    outline_path = Path(args.outline)
    if not outline_path.exists():
        logger.error(f"Outline file not found: {outline_path}")
        sys.exit(1)

    with open(outline_path, encoding="utf-8") as f:
        outline = json.load(f)

    # Outline can be either a flat dict or {"title": "...", "sections": {...}}
    if "sections" in outline:
        title = outline.get("title", args.title or "Untitled Paper")
        sections = outline["sections"]
    else:
        title = args.title or "Untitled Paper"
        sections = outline

    output_dir = Path(args.output) if args.output else Path("./generated_papers")

    writer = _build_writer(args)
    paper = writer.generate_paper(
        title=title,
        outline=sections,
        n_results_per_section=args.n_results,
        output_dir=output_dir,
        language=getattr(args, "write_language", None) or args.language,
        generate_abstract=not args.no_abstract,
    )

    print(f"\n✔  Paper generated ({len(paper)} chars)")
    print(f"   Saved to: {output_dir}")


def cmd_interactive(args: argparse.Namespace) -> None:
    """Guided interactive workflow."""
    print("=" * 60)
    print("  PaperWriter — Interactive Mode")
    print("=" * 60)

    writer = _build_writer(args)

    # ── Step 1: Ingest ───────────────────────────────────────────────
    stats = writer.get_stats()
    print(f"\nCurrent DB has {stats['total_chunks']} chunks from {stats['total_papers']} papers.")

    ingest = input("\nDo you want to ingest new PDFs? [y/N]: ").strip().lower()
    if ingest == "y":
        pdf_dir = input("PDF directory path: ").strip()
        if pdf_dir and Path(pdf_dir).is_dir():
            result = writer.ingest_papers(Path(pdf_dir))
            print(f"  Ingested {result['papers_processed']} papers "
                  f"({result['chunks_added_to_store']} new chunks)")
        else:
            print("  Invalid directory, skipping ingestion.")

    # ── Step 2: Query mode or paper generation ───────────────────────
    mode = input("\nMode? [Q]uery / [W]rite paper / [B]oth: ").strip().upper() or "B"

    if mode in ("Q", "B"):
        print("\n--- Query Mode (type 'quit' to stop) ---")
        while True:
            q = input("\nQuestion: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            result = writer.query(q)
            print(f"\n{result['answer']}")
            if result["sources"]:
                print("\nSources:")
                for s in result["sources"]:
                    print(f"  [{s['citation_key']}] {s['paper_title']}")

    if mode in ("W", "B"):
        print("\n--- Paper Generation ---")
        title = input("Paper title: ").strip() or "Untitled Paper"
        language = input(f"Language [{args.language}]: ").strip() or args.language

        print("\nDefine your outline. Enter sections one at a time.")
        print("Format: section_title | description")
        print("Type 'done' when finished.\n")

        outline: dict[str, str] = {}
        while True:
            line = input("Section: ").strip()
            if line.lower() in ("done", ""):
                if outline:
                    break
                print("  Please add at least one section.")
                continue
            if "|" in line:
                sec_title, sec_desc = line.split("|", 1)
            else:
                sec_title = line
                sec_desc = input("  Description: ").strip()
            outline[sec_title.strip()] = sec_desc.strip()
            print(f"  ✔ Added: {sec_title.strip()}")

        output_dir = (
            input("Output directory [./generated_papers]: ").strip()
            or "./generated_papers"
        )

        paper = writer.generate_paper(
            title=title,
            outline=outline,
            output_dir=Path(output_dir),
            language=language,
        )

        print(f"\n✔  Paper generated! ({len(paper)} chars)")
        print(f"   Check {output_dir} for the output files.")


# ── Argument parser ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m paperhunter.generate_paper",
        description="PaperWriter — RAG-based scientific paper generation.",
    )

    # Global options
    parser.add_argument(
        "--db-dir", default="./vector_db",
        help="Vector DB persistence directory (default: ./vector_db)",
    )
    parser.add_argument(
        "--collection", default="papers",
        help="Collection name (default: papers)",
    )
    parser.add_argument(
        "--llm-provider", default="google",
        choices=["google", "anthropic"],
        help="LLM provider (default: google)",
    )
    parser.add_argument(
        "--llm-model", default="gemini-3-flash-preview",
        help="Fast LLM for queries (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--heavy-llm-model", default="gemini-3-pro-preview",
        help="Powerful LLM for paper writing (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="Chunk overlap in characters (default: 200)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM temperature (default: 0.3)",
    )
    parser.add_argument(
        "--language", default="english",
        help="Output language (default: english)",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest PDFs into the vector DB")
    p_ingest.add_argument("pdf_dir", help="Path to directory with PDF files")
    p_ingest.add_argument(
        "--no-recursive", action="store_true",
        help="Don't search subdirectories",
    )

    # stats
    sub.add_parser("stats", help="Show knowledge-base statistics")

    # query
    p_query = sub.add_parser("query", help="Ask a question to the KB")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--n-results", type=int, default=15)

    # write
    p_write = sub.add_parser("write", help="Generate a paper from a JSON outline")
    p_write.add_argument("--title", help="Paper title (can also be in the JSON)")
    p_write.add_argument(
        "--outline", required=True,
        help="Path to a JSON file with the outline",
    )
    p_write.add_argument("--output", help="Output directory")
    p_write.add_argument("--n-results", type=int, default=20)
    p_write.add_argument("--no-abstract", action="store_true")
    p_write.add_argument(
        "--language", dest="write_language",
        help="Output language (overrides global --language)",
    )

    # interactive
    sub.add_parser("interactive", help="Guided interactive mode")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "ingest": cmd_ingest,
        "stats": cmd_stats,
        "query": cmd_query,
        "write": cmd_write,
        "interactive": cmd_interactive,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()
