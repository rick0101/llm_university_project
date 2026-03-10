# PaperHunter v3.0

**End-to-end LLM Security Survey Generator — from PDF ingestion to Nature/ACL-style paper**

PaperHunter is a modular, agentic system that autonomously generates a scientific survey paper on LLM security. Starting from 1.000+ categorized PDF papers, it runs a complete pipeline: structured extraction (GROBID), metadata enrichment (Semantic Scholar), RAPTOR hierarchical summarization with semantic clustering, community-driven GraphRAG, agentic multi-step RAG retrieval, multi-agent writing (LangGraph ReAct), and LaTeX export — all with verifiable citations and provenance tracking.

**v3.0 highlights:** Scalability to 1.000+ papers/category without context rot via RAPTOR clustering (L1 → L1.5 → L2), Graph Community detection and summarization (Louvain/Leiden via Neo4j GDS), and Agentic RAG writer with ReAct loop and on-demand retrieval tools.

**v2.1 highlights:** Budget-aware scalability (~100 papers/category within 150k token budget), download deduplication, and configurable recency priority across the entire pipeline.

---

## Table of Contents

- [PaperHunter v3.0](#paperhunter-v30)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Key Features (v3.0)](#key-features-v30)
    - [RAPTOR Hierarchical Clustering (1.000+ papers)](#raptor-hierarchical-clustering-1000-papers)
    - [Community-Driven GraphRAG](#community-driven-graphrag)
    - [Agentic Multi-Step Writer (ReAct)](#agentic-multi-step-writer-react)
  - [Key Features (v2.1)](#key-features-v21)
    - [Budget-Aware Scalability (~100 papers/category)](#budget-aware-scalability-100-paperscategory)
    - [Download Deduplication](#download-deduplication)
    - [Recency Priority](#recency-priority)
  - [Tech Stack](#tech-stack)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [API Keys](#api-keys)
    - [Docker Services](#docker-services)
      - [GROBID (required for structured PDF extraction)](#grobid-required-for-structured-pdf-extraction)
      - [Neo4j (optional — for GraphRAG + Community Detection)](#neo4j-optional--for-graphrag--community-detection)
  - [Quick Start](#quick-start)
  - [CLI Reference](#cli-reference)
    - [paperhunter search](#paperhunter-search)
    - [paperhunter ingest](#paperhunter-ingest)
    - [paperhunter summarize](#paperhunter-summarize)
    - [paperhunter index](#paperhunter-index)
    - [paperhunter write](#paperhunter-write)
    - [paperhunter query](#paperhunter-query)
    - [paperhunter stats](#paperhunter-stats)
    - [paperhunter export](#paperhunter-export)
  - [Pipeline Workflow](#pipeline-workflow)
    - [Phase 1 — Incremental Ingestion](#phase-1--incremental-ingestion)
    - [Phase 2 — Hierarchical Summarization (RAPTOR)](#phase-2--hierarchical-summarization-raptor)
    - [Phase 3 — Hybrid Indexing + GraphRAG + Community Detection](#phase-3--hybrid-indexing--graphrag--community-detection)
    - [Phase 4 — Agentic Writing (LangGraph + ReAct)](#phase-4--agentic-writing-langgraph--react)
    - [Phase 5 — Post-processing \& Output](#phase-5--post-processing--output)
  - [API Call Estimates (per phase)](#api-call-estimates-per-phase)
    - [`paperhunter summarize` — ~144–152 API calls](#paperhunter-summarize--144152-api-calls)
    - [`paperhunter index` — ~129–151 API calls](#paperhunter-index--129151-api-calls)
    - [`paperhunter write` — ~210–335 API calls](#paperhunter-write--210335-api-calls)
    - [Summary](#summary)
  - [Configuration](#configuration)
    - [Key Configuration Sections](#key-configuration-sections)
    - [Category Mapping](#category-mapping)
  - [Module API Reference](#module-api-reference)
    - [Ingestion (`paperhunter.ingestion`)](#ingestion-paperhunteringestion)
    - [Summarization (`paperhunter.summarization`)](#summarization-paperhuntersummarization)
    - [Indexing (`paperhunter.indexing`)](#indexing-paperhunterindexing)
    - [GraphRAG (`paperhunter.graphrag`)](#graphrag-paperhuntergraphrag)
    - [Agents (`paperhunter.agents`)](#agents-paperhunteragents)
    - [Post-processing (`paperhunter.postprocessing`)](#post-processing-paperhunterpostprocessing)
    - [Utilities (`paperhunter.utils`)](#utilities-paperhunterutils)
  - [Output Examples](#output-examples)
  - [Testing](#testing)
  - [Troubleshooting](#troubleshooting)
    - [GROBID not responding](#grobid-not-responding)
    - [Gemini API rate limits / timeouts](#gemini-api-rate-limits--timeouts)
    - [Qdrant issues](#qdrant-issues)
    - [Neo4j connection refused](#neo4j-connection-refused)
    - [Pipeline interrupted mid-run](#pipeline-interrupted-mid-run)
    - [Missing citations in output](#missing-citations-in-output)
    - [`pip install -e .` fails](#pip-install--e--fails)
  - [Project Structure After Setup](#project-structure-after-setup)
  - [License](#license)

---

## Architecture

```
scaricati/                            ← Categorized PDFs (~400 papers)
    ├── Prompt_injections1/
    ├── Hallucinations1/
    ├── Data_and_Model_Poisoning1/
    ├── Adversarial_Code1/
    └── ... (new categories)

config.yaml                           ← Centralized YAML configuration

paperhunter/
├── config.py                         ← YAML + pydantic-settings loader
├── models.py                         ← Extended Pydantic v2 models (+ L1_5Summary)
├── search_engine.py                  ← Multi-source search (ArXiv, S2, HF)
├── download_manager.py               ← Hybrid PDF downloader
├── relevance_filter.py               ← LLM relevance filter
├── main.py                           ← CLI with 8 subcommands
├── generate_paper.py                 ← Legacy CLI (v1 compatibility)
│
├── ingestion/                        ← PDF Processing Pipeline
│   ├── grobid_client.py              ← GROBID Docker client (TEI-XML → structured)
│   ├── pdf_hasher.py                 ← SHA-256 hash for incremental upsert
│   ├── semantic_scholar.py           ← S2 API enrichment (venue, BibTeX, citations)
│   ├── bibtex_generator.py           ← Automatic BibTeX generation
│   ├── chunker.py                    ← GROBID-aware section chunking
│   └── ingestion_pipeline.py         ← End-to-end ingestion orchestrator
│
├── summarization/                    ← Hierarchical Summarization (RAPTOR)
│   ├── clustering_utils.py           ← [v3.0] Semantic clustering (Agglomerative)
│   ├── level1_summarizer.py          ← Per-paper summary (with testing_methodology)
│   ├── level1_5_summarizer.py        ← [v3.0] Cluster meta-summary (L1.5 RAPTOR)
│   ├── level2_summarizer.py          ← Category meta-summary (L1.5-aware + legacy fallback)
│   └── invalidation_manager.py       ← Auto-invalidation of L2 on new papers
│
├── indexing/                         ← Hybrid Retrieval
│   ├── qdrant_store.py               ← Qdrant vector store (+ l1_5, graph_communities)
│   ├── bm25_index.py                 ← BM25 keyword index
│   ├── hybrid_retriever.py           ← Vector + BM25 + RRF fusion
│   └── reranker.py                   ← Gemini Flash cross-encoder reranking
│
├── graphrag/                         ← Knowledge Graph + Community Detection
│   ├── entity_extractor.py           ← Entity/relation extraction from L1
│   ├── neo4j_manager.py              ← Neo4j connector (+ GDS graph projection)
│   ├── property_graph.py             ← LlamaIndex PropertyGraph builder
│   ├── community_detector.py         ← [v3.0] Louvain/Leiden community detection
│   └── community_summarizer.py       ← [v3.0] Narrative summaries per community
│
├── agents/                           ← Multi-Agent Pipeline (LangGraph + ReAct)
│   ├── state.py                      ← Shared state (+ scratchpad, retrieved_facts)
│   ├── planner_agent.py              ← Agent 1: Outline generation from L2
│   ├── writer_agent.py               ← Agent 2: Static RAG or ReAct agentic writer
│   ├── critic_agent.py               ← Agent 3: Citation provenance verification
│   ├── assembler_agent.py            ← Agent 4: Assembly + abstract/intro/conclusion
│   ├── pipeline.py                   ← LangGraph workflow orchestration
│   └── tools/                        ← [v3.0] Retrieval tools for ReAct agent
│       ├── __init__.py
│       └── retrieval_tools.py        ← Vector, BM25, graph community search
│
├── postprocessing/                   ← Output Generation
│   ├── provenance_tracker.py         ← JSON provenance for every claim
│   ├── bibliography_generator.py     ← Filtered .bib from cited keys
│   ├── latex_exporter.py             ← Markdown → LaTeX (ACL/IEEEtran/plain)
│   ├── versioning.py                 ← Survey versioning + changelog
│   └── checkpoint_manager.py         ← Resume from failure
│
└── utils/                            ← Shared Utilities
    ├── logging_setup.py              ← Rich logging + progress bars
    ├── llm_factory.py                ← LlamaIndex LLM factory (Flash/Pro)
    ├── rate_limiter.py               ← Token bucket rate limiting
    └── token_counter.py              ← Budget estimation & token counting
```

---

## Key Features (v3.0)

### RAPTOR Hierarchical Clustering (1.000+ papers)

The system scales to 1.000+ papers per category without context rot by introducing a RAPTOR-inspired hierarchical summarization layer between L1 (per-paper) and L2 (per-category) summaries.

**How it works:**

```
L1 Summaries (1.000+) → Semantic Clustering → L1.5 Cluster Summaries (~50) → L2 Meta-Summary
```

1. **Semantic Clustering** (`clustering_utils.py`): Groups L1 summaries using AgglomerativeClustering on their Qdrant embeddings (cosine similarity). Clusters exceeding `max_cluster_size` are recursively split.
2. **L1.5 Summarization** (`level1_5_summarizer.py`): For each cluster, generates a meta-summary capturing the common theme, shared methodologies, key findings, and temporal span — with all `[bibtex_key]` citations preserved.
3. **L2 Auto-Routing** (`level2_summarizer.py`): Automatically detects L1.5 summaries and uses them instead of raw L1s. Falls back to legacy recency-tier compression if L1.5 is unavailable.

**Key config parameters** (`config.yaml`):

```yaml
clustering:
  max_cluster_size: 20            # Max papers per L1.5 cluster
  similarity_threshold: 0.75      # Cosine similarity threshold
```

**Usage:**

```python
from paperhunter.summarization.clustering_utils import cluster_summaries, fetch_l1_embeddings_from_qdrant
from paperhunter.summarization.level1_5_summarizer import generate_all_cluster_summaries

embeddings = fetch_l1_embeddings_from_qdrant(l1_summaries)
clusters = cluster_summaries(l1_summaries, embeddings)  # → {0: [...], 1: [...], ...}
l1_5 = generate_all_cluster_summaries(clusters, category="Prompt Injection")
```

### Community-Driven GraphRAG

The knowledge graph now supports automatic community detection and narrative summarization, enabling macro-theme discovery across entities.

**How it works:**

1. **Community Detection** (`community_detector.py`): Runs Louvain or Leiden algorithm via Neo4j GDS plugin. Tags each entity node with a `community_id`. Filters out communities below a configurable minimum size.
2. **Community Summarization** (`community_summarizer.py`): Transforms each community's sub-graph (entities + relations) into a 200-500 word narrative summary via Gemini Pro. Summaries are stored in Qdrant (`graph_communities` collection) for retrieval.

**Prerequisites:** Neo4j with GDS plugin:

```bash
docker run --rm -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS='["graph-data-science"]' \
    neo4j:5-community
```

**Usage:**

```python
from paperhunter.graphrag.community_detector import CommunityDetector
from paperhunter.graphrag.community_summarizer import CommunitySummarizer
from paperhunter.graphrag.neo4j_manager import Neo4jManager
from paperhunter.indexing.qdrant_store import QdrantStore

with Neo4jManager() as mgr:
    detector = CommunityDetector(mgr)
    communities = detector.detect_communities(algorithm="louvain", min_size=3)

    store = QdrantStore()
    summarizer = CommunitySummarizer(detector, store)
    results = summarizer.summarize_all_communities(communities)
    summarizer.save_to_qdrant(results)
```

### Agentic Multi-Step Writer (ReAct)

The writer agent can now operate in two modes, controlled by `agents.writer_mode` in `config.yaml`:

| Mode | Behavior |
|------|----------|
| `"static"` (default) | Legacy: pre-computed RAG context injected into a single prompt. Works well for ≤100 papers. |
| `"agentic"` | ReAct loop: the agent autonomously searches for information on-demand via retrieval tools. Scales to 1.000+ papers. |

**Retrieval tools available to the agentic writer:**

- `search_vector_database(query, category?, year_min?)` — Qdrant semantic search with filters
- `search_keyword_index(query, top_k?)` — BM25 keyword-based search
- `search_graph_communities(topic, top_k?)` — Community summary search

The agent follows a `Thought → Action → Observation → ... → Final Answer` loop (max 7 iterations). If the ReAct agent fails, it automatically falls back to `"static"` mode.

**Config:**

```yaml
agents:
  writer_mode: "agentic"          # "static" or "agentic"
```

**State tracking** (`state.py`):
- `current_scratchpad`: Agent notes during ReAct iterations (reset per section)
- `retrieved_facts`: Full log of tool calls per section (query, tool, results, timestamp)
- `agentic_stats`: Aggregate stats (total tool calls, iterations, time per section)

---

## Key Features (v2.1)

### Budget-Aware Scalability (~100 papers/category)

The system supports up to ~400 papers (N papers × 4 categories) while keeping total LLM context ≤ 150k tokens to prevent context rot and hallucinations. Budget control is automatic and centralized.

**How it works:**

| Phase | Budget Mechanism | Config Key |
|-------|-----------------|------------|
| **L1 Summarization** | Adaptive compression tiers based on N papers: *standard* (≤30), *concise* (31–70), *compressed* (>70) | `summarization.l1_target_chars_auto`, `summarization.l1_min_chars` |
| **L2 Summarization** | Recency-based tier assignment (full/reduced/compressed) + progressive downgrade if over budget | `summarization.max_context_tokens`, `summarization.l2_recency_tiers` |
| **Writer Agent** | Per-section budget: `total_budget / N_sections` (proportional) or fixed | `agents.writer_max_context_chars`, `agents.writer_budget_mode` |
| **Pipeline Start** | Budget summary printed before execution with warnings if over budget | Automatic |

**Budget report example:**

```python
from paperhunter.utils.token_counter import BudgetReport

report = BudgetReport.from_config(n_categories=4, n_sections_estimate=8)
report.print_report()  # Rich-formatted table with per-phase token estimates
```

**Key config parameters** (`config.yaml`):

```yaml
ingestion:
  papers_per_category: 100          # Target papers per category

summarization:
  max_context_tokens: 150000        # Global token budget (≤ model context window)
  l1_target_chars_auto: true        # Auto-compute L1 target size from N papers
  l1_min_chars: 2000                # Floor: L1 never shorter than this
  l2_recency_tiers:
    full: 2                         # Papers ≤2 years old: full L1 text
    reduced: 4                      # Papers 3-4 years old: ~70% detail
    compressed: 999                 # Papers >4 years old: key_contributions only

agents:
  writer_max_context_chars: 60000   # Cap per section
  writer_budget_mode: "proportional" # "fixed" | "proportional"
```

### Download Deduplication

Before downloading any PDF, the system checks if a paper with the same title or ArXiv ID already exists in `scaricati/`. This avoids redundant downloads and bandwidth waste.

- **Filename matching**: normalized (lowercase, no special chars) comparison against all existing PDFs
- **ArXiv ID matching**: extracts `YYMM.NNNNN` patterns from filenames, strips version suffixes (`v1`, `v2`)
- **In-session tracking**: newly downloaded papers are immediately registered, preventing duplicates within the same batch
- **Result**: `DownloadResult(success=True, skipped=True, method="dedup")` — no HTTP request made

No configuration needed — deduplication is always active.

### Recency Priority

When enabled (default), the system gives higher priority to more recent papers across the entire pipeline:

| Component | Recency Behavior |
|-----------|------------------|
| **Ingestion** | PDFs sorted by estimated publication year (descending) before processing — recent papers processed first |
| **L2 Summarization** | Recent papers (≤2yr) get full detail; older papers progressively compressed |
| **Hybrid Retrieval** | Post-RRF boost: `score × (1 + boost × recency_factor)` — recent papers rank higher at equal relevance |

**Global toggle** (`config.yaml`):

```yaml
recency_priority: true   # Master switch — set to false to treat all papers equally

retrieval:
  recency_boost: 0.1     # Max +10% score for most recent paper (0.0 = disabled)
```

When `recency_priority: false`, all recency logic is bypassed: L2 uses uniform detail, retrieval has no temporal boost, ingestion processes in filesystem order.

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| LLM | **Gemini 3 Flash/Pro** (via `google-genai` SDK) | Flash for simple tasks, Pro for writing/planning |
| Embeddings | **Gemini Embedding 001** (3072-dim) | Via `llama-index-embeddings-google-genai` |
| RAG Framework | **LlamaIndex Core** | Orchestration, PropertyGraph, ReAct Agent |
| Vector DB | **Qdrant** (local, no server) | 5 collections: chunks, L1, L2, L1.5, communities |
| Keyword Search | **BM25** (`rank-bm25`) | Hybrid retrieval with RRF fusion |
| Clustering | **scikit-learn** + **NumPy** | AgglomerativeClustering for RAPTOR L1.5 |
| PDF Extraction | **GROBID 0.8.2** (Docker) | TEI-XML → structured sections, refs, captions |
| Metadata | **Semantic Scholar API** (free) | BibTeX, venue, citation count, DOI |
| Multi-Agent | **LangGraph** + **ReAct** | Planner → Writer (static/agentic) → Critic → Assembler |
| Knowledge Graph | **Neo4j Community + GDS** | PropertyGraph + Louvain/Leiden community detection |
| Output | **LaTeX** (ACL/IEEEtran) | + .bib, provenance JSON, versioning |

---

## Setup

### Prerequisites

- **Python 3.10+** (tested up to 3.14)
- **Docker** (for GROBID; optional but strongly recommended)
- **Google Gemini API key** (free tier with Gemini Advanced subscription)

### Installation

```bash
# Clone the repository
git clone https://github.com/rick0101/llm_university_project.git
cd llm_university_project

# If the repo includes large files via Git LFS:
# sudo dnf install git-lfs   # Fedora
# sudo apt install git-lfs   # Ubuntu/Debian
# brew install git-lfs        # macOS
git lfs install && git lfs pull

# Create virtual environment
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

# Install dependencies + package
pip install --upgrade pip
pip install -e .

# Install Playwright browsers (for paper download)
playwright install chromium

# (Optional) Install GraphRAG dependencies
pip install -e ".[graphrag]"

# (Optional) Install dev dependencies
pip install -e ".[dev]"
```

### API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required — Gemini API key (LLM + embeddings)
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional — higher rate limits for Semantic Scholar
SEMANTIC_SCHOLAR_API_KEY=your_s2_key_here
```

### Docker Services

#### GROBID (required for structured PDF extraction)

```bash
# Start GROBID (first pull takes ~2 min)
docker run --rm -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.2-full

# Wait ~30 seconds, then verify
curl -s http://localhost:8070/api/isalive
# Should return: true
```

> If GROBID is unavailable, the system automatically falls back to PyMuPDF (`config.yaml` → `ingestion.fallback_to_pymupdf: true`).

#### Neo4j (optional — for GraphRAG + Community Detection)

```bash
# Without GDS (basic GraphRAG only)
docker run --rm -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:5-community

# With GDS plugin (required for community detection — v3.0)
docker run --rm -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS='["graph-data-science"]' \
    neo4j:2026.02.2-community
```

Then set `neo4j.enabled: true` in `config.yaml`.

---

## Quick Start

The typical end-to-end workflow is:

```bash
# 1. (Optional) Search and download new papers
paperhunter search -k '"jailbreak prompt" LLM OR "LLM jailbreaking" OR "adversarial prompt" LLM OR "LLM injection attack"' -c "Papers about prompt injection or jailbreaking attacks" -o ./scaricati/Prompt_injections1 -n 20 --date-from 2025-01 --date-to 2026-03

paperhunter search -k '"hallucination detection" OR "factuality evaluation" OR "grounding" OR "hallucination benchmark"' -c "Papers about detecting or evaluating hallucinations in large language models" -o ./scaricati/Hallucinations1 -n 20 --date-from 2025-01 --date-to 2026-03

paperhunter search -k '"instruction tuning poisoning" OR "RLHF poisoning" OR "alignment compromise" OR "sleeper agent"' -c "Papers about Data poisoning LLM instruction tuning, RLHF, or sleeper agent attacks" -o ./scaricati/Data_and_Model_Poisoning1 -n 20 --date-from 2025-01 --date-to 2026-03

paperhunter search -k '"semantic-preserving adversarial code" OR "functionality-preserving transformations" OR "obfuscation-based adversarial attacks" OR "dead code insertion evasion"' -c "Papers focusing on adversarial attacks that maintain code functionality while deceiving the model" -o ./scaricati/Adversarial_Code1 -n 10 --date-from 2025-01 --date-to 2026-03

paperhunter search -k '" adversarial code" OR "Adeversarial code large language models"' -c "Papers about adversarial code on LLMs" -o ./scaricati/Adversarial_Code1 -n 10 --date-from 2024-01 --date-to 2026-03

Paramentro diltro per evitare survey e review, che non sono l'obiettivo di questa raccolta:
"-c "Only original research papers reporting novel experiments, methods, or empirical results. EXCLUDE: survey papers, literature reviews, systematization-of-knowledge (SoK), position papers, and AI-generated summaries. The paper must present its own contributions (new attack, defense, benchmark, or evaluation), not merely summarize existing work.""

```

Then run the full pipeline:

```bash

# 2. Ingest PDFs → GROBID extraction + S2 enrichment + Qdrant indexing + gemini-embedding-001
paperhunter ingest

# 3. Generate hierarchical summaries (L1 per paper, L2 per category)
paperhunter summarize

# 4. Build search indices (BM25 + PropertyGraph)
paperhunter index

# 5. Run the multi-agent writing pipeline
paperhunter write

# 6. Export to LaTeX
paperhunter export --template acl
```

Check system status at any point:

```bash
paperhunter stats
```

Query the knowledge base interactively:

```bash
paperhunter query "What defenses exist against prompt injection attacks?"
```

---

## CLI Reference

Global options available for all subcommands:

| Option | Description |
|--------|-------------|
| `--config PATH` | Custom config.yaml path |
| `-v, --verbose` | Enable DEBUG logging |
| `--version` | Show version |

### paperhunter search

Search and download scientific papers from ArXiv, Semantic Scholar, and HuggingFace Papers.

```bash
paperhunter search -k "LLM security" -c "Focus on prompt injection and jailbreaking"
paperhunter search -k "adversarial attacks" -n 30 -o ./papers --no-download
```

| Option | Description |
|--------|-------------|
| `-k, --keywords` | Search keywords |
| `-c, --criteria` | LLM filtering criteria |
| `-n, --max-results` | Max results per source |
| `--no-download` | Search only, skip PDF download |
| `--llm-provider` | `google` or `anthropic` |
| `-o, --output-dir` | PDF output directory |

### paperhunter ingest

Run the incremental ingestion pipeline: GROBID extraction → S2 enrichment → BibTeX → chunking → Qdrant.

```bash
paperhunter ingest
paperhunter ingest --pdf-dir ./my_papers --batch-size 5 --skip-s2
```

| Option | Description |
|--------|-------------|
| `--pdf-dir` | PDF root directory (default: from config.yaml) |
| `--batch-size` | PDFs per batch (default: 10) |
| `--skip-s2` | Skip Semantic Scholar enrichment |
| `--skip-qdrant` | Skip Qdrant indexing |
| `--no-checkpoint` | Disable checkpoint for resume |

**Incremental behavior:** Only new or modified PDFs are processed (SHA-256 hash comparison). Re-running produces 0 changes if nothing changed.

### paperhunter summarize

Generate/regenerate L1 (per-paper) and L2 (per-category meta) summaries.

```bash
paperhunter summarize            # Generate both L1 and L2
paperhunter summarize --l1       # Only L1
paperhunter summarize --l2       # Only L2
paperhunter summarize --l2 --category "Prompt Injection" --force
paperhunter summarize --invalidate  # Check and regenerate stale L2s
```

| Option | Description |
|--------|-------------|
| `--l1` | Only generate L1 summaries |
| `--l2` | Only generate L2 summaries |
| `--invalidate` | Check L2 invalidation and regenerate |
| `--category` | Limit to a specific category |
| `--force` | Force regeneration even if existing |

### paperhunter index

Build/update BM25 keyword index and PropertyGraph.

```bash
paperhunter index               # Build both
paperhunter index --bm25        # Only BM25
paperhunter index --graph       # Only PropertyGraph
```

| Option | Description |
|--------|-------------|
| `--bm25` | Only build BM25 index |
| `--graph` | Only build PropertyGraph |
| `--skip-neo4j` | Skip Neo4j import |
| `--clear-graph` | Clear Neo4j before importing |

### paperhunter write

Run the full multi-agent writing pipeline (Planner → Writer → Critic → Assembler).

```bash
paperhunter write
paperhunter write --outline sample_outline.json
paperhunter write --resume                      # Resume from last checkpoint
paperhunter write --resume --checkpoint-file output/.checkpoints/verify_20260301.json
```

| Option | Description |
|--------|-------------|
| `--outline PATH` | JSON outline file (otherwise auto-generated by Planner) |
| `--resume` | Resume from checkpoint |
| `--checkpoint-file PATH` | Specific checkpoint to resume from |
| `--no-checkpoint` | Disable checkpointing |
| `--no-export` | Skip auto LaTeX/BibTeX export |
| `--template` | LaTeX template: `acl`, `ieeetran`, `plain` |

The pipeline automatically exports LaTeX, BibTeX, provenance JSON, and creates a versioned output directory after completion.

### paperhunter query

Query the knowledge base with hybrid retrieval (vector + BM25 + reranking).

```bash
paperhunter query "How does GCG attack work?"
paperhunter query --category "Prompt Injection" --year-min 2024
paperhunter query   # Interactive mode
```

| Option | Description |
|--------|-------------|
| `question` | Query text (omit for interactive mode) |
| `-n, --n-results` | Number of results (default: 10) |
| `--category` | Filter by category |
| `--year-min` | Minimum year |
| `--year-max` | Maximum year |
| `--no-rerank` | Disable Gemini reranking |

### paperhunter stats

Show system-wide statistics: ingestion registry, Qdrant collections, BM25 index, PropertyGraph, versions, checkpoints.

```bash
paperhunter stats
```

### paperhunter export

Export the generated survey to LaTeX and/or filtered BibTeX.

```bash
paperhunter export                        # Export latest version
paperhunter export --template ieeetran    # Use IEEEtran template
paperhunter export --input output/latest/survey.md -o ./export
paperhunter export --bib-only             # Only generate filtered .bib
```

| Option | Description |
|--------|-------------|
| `--input PATH` | Markdown source file (default: latest version) |
| `-o, --output DIR` | Output directory |
| `--template` | `acl`, `ieeetran`, `plain` |
| `--title` | Override title |
| `--author` | Override author |
| `--bib-file` | BibTeX filename (default: `references.bib`) |
| `--latex` | Only generate LaTeX |
| `--bib-only` | Only generate filtered BibTeX |

---

## Pipeline Workflow

### Phase 1 — Incremental Ingestion

```
PDF files → GROBID (TEI-XML) → S2 Enrichment → BibTeX → GROBID-aware Chunking → Qdrant + BM25
```

1. **Scan & Delta Detection**: SHA-256 hash registry identifies new/modified PDFs
2. **Recency Sort** *(v2.1)*: PDFs sorted by estimated publication year (descending) — recent papers processed first, so interrupted runs still have the most relevant data
3. **GROBID Extraction**: Structured text (sections, references, figure captions) via Docker API
4. **S2 Enrichment**: Venue, DOI, citation count, BibTeX from Semantic Scholar API
5. **BibTeX Generation**: Automatic keys in `Author_YYYY_Keyword` format
6. **Chunking**: GROBID section-aware splitting (~1500 chars, 200 overlap, per-section boundaries)
7. **Indexing**: Qdrant (3072-dim Gemini embeddings) + hash registry update

Key features:
- Re-running processes only delta changes (0 operations if nothing changed)
- Download deduplication: papers already in `scaricati/` are never re-downloaded

### Phase 2 — Hierarchical Summarization (RAPTOR)

```
Chunks → L1 Summary (per paper) → [Clustering → L1.5 (per cluster)] → L2 Meta-Summary → Auto-invalidation
```

- **L1 Summaries** (Gemini Pro): Structured JSON with `key_contributions`, `methodology`, `testing_methodology` (critical field), `temporal_context`, `results_summary`, `limitations`
  - *Budget-aware (v2.1)*: Compression tier adapts to N papers/category — standard (≤30), concise (31–70), compressed (>70) — keeping total L1 text within token budget
- **Semantic Clustering** *(v3.0)*: AgglomerativeClustering groups L1 summaries by cosine similarity of their Qdrant embeddings. Oversized clusters are recursively split. Produces ~50 clusters from 1.000+ papers.
- **L1.5 Cluster Summaries** *(v3.0)* (Gemini Pro): Per-cluster meta-summary with `common_theme`, `shared_methodologies`, `key_findings`, `testing_overview`, `temporal_span`. Stored in Qdrant `l1_5_summaries` collection.
- **L2 Summaries** (Gemini Pro): 4-6k token meta-summaries per category with `temporal_trends`, `taxonomy`, `gaps_filled`, `open_gaps`, `methodological_convergences`, `testing_overview`, `synthesis`
  - *RAPTOR path (v3.0)*: Auto-detects L1.5 summaries and uses them as input (cross-cluster pattern detection). Falls back to legacy path if L1.5 unavailable.
  - *Legacy path (v2.1)*: Recency-tiered formatting — recent papers at full detail, older papers progressively compressed. Multi-phase downgrade if over budget (reduced→compressed→truncation)
- **Auto-invalidation**: When new papers are ingested, affected L2 summaries are automatically regenerated (configurable threshold)

### Phase 3 — Hybrid Indexing + GraphRAG + Community Detection

```
Chunks + L1 + L2 → [Vector Search (Qdrant) + BM25] → RRF Fusion → Gemini Reranking
                  → PropertyGraph (entities & relations) → Graph traversal
                  → Community Detection (Louvain/Leiden) → Community Summaries → Qdrant
```

- **Hybrid Retrieval**: Vector search (Qdrant, cosine similarity) + BM25 keyword match → Reciprocal Rank Fusion
- **Recency Boost** *(v2.1)*: Post-RRF temporal scoring — `score × (1 + boost × recency_factor)` — configurable via `retrieval.recency_boost`
- **Reranking**: Gemini Flash as cross-encoder scores candidates 0.0–1.0
- **Entity Extraction**: Attacks, Defenses, Models, Datasets, Benchmarks, Metrics with temporal relations (UPDATES, INTRODUCED_IN, EXTENDS)
- **PropertyGraph**: LlamaIndex `SimplePropertyGraphStore` with optional Neo4j sync
- **Community Detection** *(v3.0)*: Louvain/Leiden algorithm via Neo4j GDS identifies macro-theme communities. Each community's sub-graph is summarized into a 200-500 word narrative stored in Qdrant (`graph_communities` collection), searchable by the agentic writer.

### Phase 4 — Agentic Writing (LangGraph + ReAct)

```
START → Planner → Writer (static or agentic) → Critic → [loop if needed] → Assembler → END
```

1. **Planner Agent** (Pro): Generates structured outline from L2 summaries — sections, topics, target word counts
2. **Writer Agent** (Pro) — dual mode *(v3.0)*:
   - **Static mode** (`writer_mode: "static"`): Legacy behavior — generates 3-5 RAG queries → hybrid retrieval → optional reranking → writes with pre-computed context. Budget-aware *(v2.1)*: per-section context budget computed from `writer_budget_mode`
   - **Agentic mode** (`writer_mode: "agentic"`): ReAct loop — the agent autonomously searches vector DB, BM25 index, and graph communities on-demand via `FunctionTool`. Max 7 iterations per section. Falls back to static on failure. State tracks all tool calls in `retrieved_facts` and `agentic_stats`.
3. **Critic Agent** (Flash): Verifies every citation against source chunks — SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED / SOURCE_NOT_FOUND. If <80% pass rate → flags section for rewrite (max 3 iterations)
4. **Assembler Agent** (Pro): Orders sections, generates abstract/introduction/conclusion, assembles final Markdown + bibliography

Checkpoint saving after every node enables resume from failure.

### Phase 5 — Post-processing & Output

- **Provenance Tracking**: JSON mapping every claim to source chunk + verification status
- **Bibliography**: Filtered `.bib` with only cited entries (from master .bib)
- **LaTeX Export**: Markdown → `.tex` with ACL, IEEEtran, or plain template
- **Versioning**: `Survey_vX.Y_MonYYYY/` directories with automatic changelog between versions
- **Checkpoint Manager**: Save/load pipeline state for resume from any failure point

---

## API Call Estimates (per phase)

Estimated number of API calls for a typical run with **118 papers**, **4 categories**, **~11 sections**, **agentic writer mode**. Values are approximate and depend on content complexity, number of citations per section, and Critic pass rate.

### `paperhunter summarize` — ~144–152 API calls

| Sub-phase | API Type | Calls | Notes |
|-----------|----------|-------|-------|
| L1 (per-paper) | Gemini **Pro** | **118** | 1 LLM call per paper |
| L1.5 (RAPTOR clusters) | Gemini **Pro** | **~8–16** | 1 per cluster (~2–4 clusters/category) |
| L2 (meta-summary) | Gemini **Pro** | **4** | 1 per category |
| L1 → Qdrant | **Embedding** | **~12** | 118 embeddings batched (batch_size=10) |
| L1.5 → Qdrant | **Embedding** | **~1–2** | ~8–16 embeddings batched |
| L2 → Qdrant | **Embedding** | **~1** | 4 embeddings batched |
| **Total** | | **~144–152** | |

### `paperhunter index` — ~129–151 API calls

| Sub-phase | API Type | Calls | Notes |
|-----------|----------|-------|-------|
| BM25 build | — | **0** | Local tokenization, no API |
| Entity extraction | Gemini **Flash** | **118** | 1 per L1 summary |
| Neo4j import | — | **0** | Cypher queries only |
| Community detection | — | **0** | Louvain/Leiden algorithm in Neo4j GDS |
| Community summaries | Gemini **Pro** | **~10–30** | 1 per community (min_size=3) |
| Community → Qdrant | **Embedding** | **~1–3** | Batched |
| **Total** | | **~129–151** | If Neo4j disabled: ~118 total |

### `paperhunter write` — ~210–335 API calls

| Sub-phase | API Type | Calls (est.) | Notes |
|-----------|----------|--------------|-------|
| **Planner** | Gemini **Pro** | **1** | Generate outline |
| **Writer (agentic)** ×11 sec. | Gemini **Pro** | **~88–132** | ~8–12 ReAct iterations × 11 sections |
| Writer tool calls (vector search) | **Embedding** | **~22–44** | ~2–4 `_embed_query()` per section × 11 |
| **Critic** (batch verification) | Gemini **Flash** | **~22–55** | ~15 citations/sec ÷ batch_size=5 → ~3–5 calls/sec × 11 |
| Critic feedback | Gemini **Flash** | **~3–5** | Only for sections failing <80% threshold |
| **Re-write** (~4–5 sections) | Gemini **Pro** | **~32–60** | Sections not passing → agentic rewrite |
| Re-write embeddings | **Embedding** | **~8–20** | Tool calls for rewrite |
| Re-verify | Gemini **Flash** | **~9–25** | Critic round 2 |
| **Assembler** (abstract) | Gemini **Pro** | **1** | |
| **Assembler** (introduction) | Gemini **Pro** | **1** | |
| **Assembler** (conclusion) | Gemini **Pro** | **1** | |
| **Assembler** (changelog) | Gemini **Flash** | **1** | |
| **Total** | | **~210–335** | |

### Summary

| Phase | LLM calls | Embedding calls | **Total API** |
|-------|-----------|-----------------|---------------|
| **Summarize** | ~130–138 | ~14 | **~144–152** |
| **Index** | ~128–148 | ~1–3 | **~129–151** |
| **Write** | ~175–280 | ~35–55 | **~210–335** |
| **TOTAL** | **~433–566** | **~50–72** | **~483–638** |

> **Note:** The **write** phase is the most expensive due to the ReAct loop (each iteration = 1 Pro call) and the Writer↔Critic cycle. The **summarize** phase is the most predictable (~1:1 with paper count). The **index** phase depends heavily on how many communities are found in the graph.

---

## Configuration

All settings are in `config.yaml`. API keys go in `.env` (never committed).

### Key Configuration Sections

| Section | Description |
|---------|-------------|
| `recency_priority` | Global toggle for recency weighting across all modules |
| `gemini` | Model names, temperature, timeout, max tokens |
| `grobid` | GROBID Docker URL, timeout, batch size |
| `neo4j` | Neo4j connection (optional, `enabled: false` by default) |
| `clustering` | *(v3.0)* RAPTOR clustering: `max_cluster_size`, `similarity_threshold` |
| `qdrant` | Vector DB local path, collections (+ `l1_5_summaries`, `graph_communities`), embedding dimension |
| `semantic_scholar` | S2 API settings, rate limits, cache directory |
| `ingestion` | PDF directory, chunk size/overlap, category mapping, `papers_per_category` |
| `summarization` | L1/L2 token limits, invalidation, `max_context_tokens`, `l1_target_chars_auto`, `l2_recency_tiers` |
| `retrieval` | Vector/BM25 weights, top-k, rerank batch size, `recency_boost` |
| `agents` | Writer iterations, verification threshold, model per agent, `writer_max_context_chars`, `writer_budget_mode`, `writer_mode` |
| `output` | Output directories, LaTeX template, export flags |
| `rate_limiting` | Gemini RPM/daily, S2 RPM, GROBID concurrency |
| `logging` | Level, Rich enabled, log file path |

### Category Mapping

Map folder names to human-readable categories in `config.yaml`:

```yaml
ingestion:
  category_mapping:
    Prompt_injections1: "Prompt Injection"
    Hallucinations1: "LLM Hallucinations"
    Data_and_Model_Poisoning1: "Data and Model Poisoning"
    Adversarial_Code1: "Adversarial Attacks"
    My_New_Category: "My New Topic"       # Add new categories here
```

To add papers to an existing category, simply copy PDFs into the corresponding subfolder under `scaricati/`. The ingestion pipeline will detect them automatically.

---

## Module API Reference

### Ingestion (`paperhunter.ingestion`)

```python
from paperhunter.ingestion import IngestionPipeline, IngestionReport

# Run full ingestion
pipeline = IngestionPipeline(pdf_base_dir="./scaricati", batch_size=10)
report: IngestionReport = pipeline.run()
print(report.to_dict())  # processed, skipped, errors, total_chunks, ...
```

Individual components:

```python
from paperhunter.ingestion.grobid_client import GrobidClient
from paperhunter.ingestion.pdf_hasher import compute_pdf_hash, get_new_or_modified_pdfs
from paperhunter.ingestion.semantic_scholar import SemanticScholarEnricher
from paperhunter.ingestion.bibtex_generator import generate_bibtex_key, generate_bibtex_entry
from paperhunter.ingestion.chunker import chunk_grobid_output, chunk_raw_text
```

### Summarization (`paperhunter.summarization`)

```python
from paperhunter.summarization import (
    generate_l1_summary, generate_l1_batch,
    generate_l2_summary, generate_l2_for_all_categories,
    load_l1_summaries_for_category,
    InvalidationManager,
    # v3.0 — RAPTOR clustering + L1.5
    generate_cluster_summary, generate_all_cluster_summaries,
    load_l1_5_summaries_for_category, count_l1_5_for_category,
)

# Generate L1 for a single paper
summary = generate_l1_summary(paper, chunks, save_to_store=True)

# v3.0 — RAPTOR: Cluster L1s and generate L1.5 summaries
from paperhunter.summarization.clustering_utils import (
    cluster_summaries, fetch_l1_embeddings_from_qdrant,
)

embeddings = fetch_l1_embeddings_from_qdrant(l1_summaries)
clusters = cluster_summaries(l1_summaries, embeddings)
l1_5 = generate_all_cluster_summaries(clusters, category="Prompt Injection")

# Generate L2 (auto-detects L1.5 if available, otherwise falls back to L1)
l2_summaries = generate_l2_for_all_categories(save_to_store=True)

# Check and regenerate invalidated L2s
mgr = InvalidationManager(threshold=1)
report = mgr.check_and_regenerate()
```

### Indexing (`paperhunter.indexing`)

```python
from paperhunter.indexing import (
    QdrantStore, BM25Index, HybridRetriever, GeminiReranker,
    RetrievedChunk, reciprocal_rank_fusion,
)

# Vector search
store = QdrantStore()
results = store.query("prompt injection defense", n_results=10,
                      where_filter={"category": "Prompt Injection"})

# Hybrid retrieval
retriever = HybridRetriever()
chunks = retriever.retrieve("GCG attack mechanism",
                            category_filter="Adversarial Attacks",
                            year_range=(2023, 2026))

# Multi-collection search (paper_chunks + l1_summaries + l2_summaries)
chunks = retriever.retrieve_multi_collection("defense overview")

# Reranking
reranker = GeminiReranker()
reranked = reranker.rerank("defense mechanisms", chunks, top_k=5)
```

### GraphRAG (`paperhunter.graphrag`)

```python
from paperhunter.graphrag import (
    extract_entities_and_relations, extract_batch,
    PropertyGraphBuilder, Neo4jManager,
    Entity, Relation, EntityType, RelationType,
    # v3.0 — Community Detection + Summarization
    CommunityDetector, CommunitySummarizer,
    detect_graph_communities, summarize_graph_communities,
    load_community_summaries, count_community_summaries,
)

# Extract entities from L1 summaries
entities, relations = extract_entities_and_relations(l1_summary)

# Build PropertyGraph
builder = PropertyGraphBuilder()
report = builder.build_from_summaries(l1_summaries)
builder.save()

# Query the graph
result = builder.query_graph("evolution of prompt injection defenses")
timeline = builder.get_temporal_chain(EntityType.ATTACK)

# Neo4j (optional)
with Neo4jManager() as mgr:
    mgr.import_entities_and_relations(entities, relations)
    neighbors = mgr.query_neighbors("GCG", depth=2)

# v3.0 — Community Detection + Summarization (requires Neo4j + GDS)
with Neo4jManager() as mgr:
    detector = CommunityDetector(mgr)
    communities = detector.detect_communities(algorithm="louvain", min_size=3)
    stats = detector.get_community_stats()

    store = QdrantStore()
    summarizer = CommunitySummarizer(detector, store)
    results = summarizer.summarize_all_communities(communities)
    summarizer.save_to_qdrant(results)

# Or one-shot:
results = summarize_graph_communities(algorithm="louvain", min_size=3)
loaded = load_community_summaries()
```

### Agents (`paperhunter.agents`)

```python
from paperhunter.agents import (
    run_pipeline, run_pipeline_from_checkpoint,
    plan, write, verify, assemble,
    create_initial_state, get_pipeline_summary,
    reset_scratchpad,  # v3.0
    SurveyState, SectionOutline, SectionDraft,
)

# Run the full pipeline
state = run_pipeline(outline_file="my_outline.json", save_checkpoints=True)
summary = get_pipeline_summary(state)

# Resume from checkpoint
state = run_pipeline_from_checkpoint("output/.checkpoints/verify_latest.json")

# Use individual agents
state = create_initial_state(l2_summaries=my_l2s)
state = plan(state)   # Generate outline
state = write(state)  # Write all sections (static or agentic based on config)
state = verify(state) # Verify citations
state = assemble(state) # Assemble final paper

# v3.0 — Retrieval Tools (for standalone use or ReAct agent)
from paperhunter.agents.tools import (
    search_vector_database, search_keyword_index,
    search_graph_communities, get_all_retrieval_tools,
)

# Standalone tool calls
result = search_vector_database("prompt injection attacks", category="Prompt Injection")
result = search_keyword_index("GCG adversarial suffix")
result = search_graph_communities("adversarial robustness testing")

# Get all tools as LlamaIndex FunctionTool (for ReActAgent)
tools = get_all_retrieval_tools()
```

### Post-processing (`paperhunter.postprocessing`)

```python
from paperhunter.postprocessing import (
    ProvenanceTracker, create_provenance_from_state,
    BibliographyGenerator, generate_final_bib, generate_bib_from_state,
    markdown_to_latex, export_latex, export_latex_from_state,
    VersionManager, create_version_from_state,
    CheckpointManager, get_resume_recommendation,
)

# Provenance tracking
tracker = create_provenance_from_state(state)
tracker.save_json("output/provenance.json")
tracker.save_markdown("output/provenance.md")

# Bibliography
bib_path, report = generate_final_bib(used_keys={"Zou_2023_GCG", "Li_2023_HaluEval"})

# LaTeX export
tex = markdown_to_latex(md_content, template="acl", bib_file="references.bib")
export_latex(md_content, output_path="output/survey.tex", template="acl", copy_bib=True)

# Versioning
vm = VersionManager()
version = vm.create_new_version(paper_count=20, categories=["PI", "DMP", "LLM_H", "AA"])
vm.save_version_metadata(version, survey_text=md_content)
changelog = vm.generate_changelog(old_version, new_version)

# Checkpoint resume
recommendation = get_resume_recommendation()
```

### Utilities (`paperhunter.utils`)

```python
from paperhunter.utils.llm_factory import (
    get_flash_llm, get_pro_llm, get_embedding_model, llm_call_with_retry,
)
from paperhunter.utils.rate_limiter import get_gemini_limiter, get_s2_limiter
from paperhunter.utils.logging_setup import (
    setup_logging, get_logger, progress_bar, print_table,
)

# Token counting & budget management (v2.1)
from paperhunter.utils.token_counter import (
    estimate_tokens,                 # ~4 chars/token approximation
    compute_l1_target_chars,         # Target chars per L1 based on N papers
    compute_l2_budget,               # Available chars for L2 prompt
    compute_writer_budget,           # Per-section budget (proportional/fixed)
    get_l1_compression_tier,         # "standard" / "concise" / "compressed"
    chars_budget_for_n_papers,       # Total chars budget for N papers
    BudgetReport,                    # Full budget summary with Rich output
)

# Budget report before pipeline run
report = BudgetReport.from_config(n_categories=4, n_sections_estimate=8)
report.print_report()   # Rich table: phase → tokens/call → % budget → notes
print(report.within_budget)  # True if all phases fit in max_context_tokens
```

---

## Output Examples

After running the full pipeline, the output directory contains:

```
output/
├── references.bib                    # Master BibTeX (all papers)
├── references_final.bib              # Filtered BibTeX (only cited)
├── l2_metadata/                      # L2 metadata JSON per category
│   ├── prompt_injection.json
│   ├── adversarial_attacks.json
│   └── ...
├── versions/
│   ├── Survey_v1.0_Feb2026/
│   │   ├── survey.md                 # Final survey in Markdown
│   │   ├── survey.tex                # LaTeX version
│   │   ├── references.bib            # Filtered bibliography
│   │   ├── version_info.json         # Version metadata
│   │   ├── provenance.json           # Claim → source mapping
│   │   ├── changelog.md              # Diff from previous version
│   │   └── citation_keys.json        # List of cited keys
│   └── Survey_v1.1_Mar2026/
│       └── ...                       # Auto-generated changelog
├── provenance/                       # Provenance reports
└── .checkpoints/                     # Pipeline checkpoints
    ├── plan_20260301_120000.json
    ├── write_20260301_121500.json
    └── ...
```

---

## Testing

The project includes 35 test files with ~1000+ tests covering all modules. Tests use mocks and do not require live APIs or Docker services.

```bash
# Run all tests
python -m pytest test_*.py -v

# Run a specific test module
python -m pytest test_bm25_index.py -v
python -m pytest test_pipeline.py -v
python -m pytest test_assembler_agent.py -v

# Run tests matching a pattern
python -m pytest test_*.py -k "test_rerank"
```

Test modules:

| Module | Test File | Tests |
|--------|-----------|-------|
| BM25 Index | `test_bm25_index.py` | 15 |
| Hybrid Retriever | `test_hybrid_retriever.py` | 17 |
| Reranker | `test_reranker.py` | 12 |
| Entity Extractor | `test_entity_extractor.py` | 12 |
| Neo4j Manager | `test_neo4j_manager.py` | 13 |
| PropertyGraph | `test_property_graph.py` | 13 |
| LangGraph State | `test_state.py` | 11 |
| Planner Agent | `test_planner_agent.py` | 10 |
| Writer Agent | `test_writer_agent.py` | 13 |
| Critic Agent | `test_critic_agent.py` | 19 |
| Assembler Agent | `test_assembler_agent.py` | 22 |
| Pipeline | `test_pipeline.py` | 21 |
| Provenance Tracker | `test_provenance_tracker.py` | 50 |
| Bib Generator | `test_bibliography_generator.py` | 75 |
| LaTeX Exporter | `test_latex_exporter.py` | 93 |
| Versioning | `test_versioning.py` | 90 |
| Checkpoint Manager | `test_checkpoint_manager.py` | 69 |
| L1 Summarizer | `test_l1_summarizer.py` | — |
| L2 Summarizer | `test_l2_summarizer.py` | — |
| E2E Integration | `test_e2e_integration.py` | — |
| Budget-Aware L1 | `test_l1_budget_aware.py` | 6 |
| Budget-Aware L2 | `test_l2_budget_aware.py` | 9 |
| Budget-Aware Writer | `test_l1_budget_aware_writer.py` | 8 |
| Budget Summary | `test_budget_summary.py` | 10 |
| Download Dedup | `test_dedup_download.py` | 22 |
| Recency Sort | `test_recency_sort.py` | 48 |
| Clustering Utils | `test_clustering_utils.py` | 16 |
| L1.5 Summarizer | `test_l1_5_summarizer.py` | 28 |
| L2 with L1.5 | `test_l2_with_l1_5.py` | 10 |
| Community Detector | `test_community_detector.py` | 41 |
| Community Summarizer | `test_community_summarizer.py` | 63 |
| Retrieval Tools | `test_retrieval_tools.py` | 34 |
| Writer Agent Agentic | `test_writer_agent_agentic.py` | 28 |

---

## Troubleshooting

### GROBID not responding

```bash
# Check if GROBID is running
curl -s http://localhost:8070/api/isalive

# Restart GROBID
docker stop grobid
docker run --rm -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.2-full
# Wait ~30 seconds before using
```

If GROBID is unavailable, set `ingestion.fallback_to_pymupdf: true` in `config.yaml` (enabled by default).

### Gemini API rate limits / timeouts

- The system uses patient retry (6 retries with exponential backoff up to 120s) for Gemini Pro
- Timeout is set to 600s by default for thinking models
- Rate limiter: 15 RPM (free tier). Configurable in `config.yaml` → `rate_limiting.gemini_rpm`
- If you consistently hit 429 errors, reduce batch sizes or add delay between operations

### Qdrant issues

Qdrant runs locally (no server needed). If the database gets corrupted:

```bash
# Reset Qdrant (will require re-ingestion)
rm -rf qdrant_db/
paperhunter ingest
```

### Neo4j connection refused

```bash
# Verify Neo4j is running
docker ps | grep neo4j

# Check config.yaml
# neo4j.enabled: true
# neo4j.uri: "bolt://localhost:7687"
# neo4j.password: "password"
```

### Pipeline interrupted mid-run

The pipeline saves checkpoints after every agent node. Resume with:

```bash
paperhunter write --resume
```

To list available checkpoints:

```python
from paperhunter.postprocessing import list_all_checkpoints
for cp in list_all_checkpoints():
    print(cp)
```

### Missing citations in output

Some citations may be missing from the final `.bib` if:
- The LLM hallucinated a reference not in the corpus
- The BibTeX key format doesn't match (check `Author_YYYY_Keyword` pattern)

Use the provenance report to identify which claims lack sources:

```python
from paperhunter.postprocessing import load_provenance
tracker = load_provenance("output/provenance/provenance.json")
for claim in tracker.get_unsupported_claims():
    print(claim)
```

### `pip install -e .` fails

Ensure you have the latest pip and setuptools:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

---

## Project Structure After Setup

```
vscode_agent_paper/
├── .env                          ← Create from .env.example
├── venv/                         ← Python virtual environment
├── config.yaml                   ← Centralized configuration
├── scaricati/                    ← Categorized PDFs (included in repo)
│   ├── Prompt_injections1/       #    (~53 PDFs)
│   ├── Adversarial_Code1/        #    (~23 PDFs)
│   ├── Data_and_Model_Poisoning1/ #   (~13 PDFs)
│   └── Hallucinations1/          #    (~29 PDFs)
├── qdrant_db/                    ← Created by ingestion (vector store)
├── data/
│   ├── hash_registry.json        ← Created by ingestion (PDF hashes)
│   ├── bm25_index.pkl            ← Created by index (BM25)
│   └── property_graph/           ← Created by index (graph store)
├── output/
│   ├── references.bib            ← Master BibTeX
│   ├── references_final.bib      ← Filtered BibTeX
│   ├── versions/                 ← Versioned survey outputs
│   ├── provenance/               ← Provenance reports
│   ├── l2_metadata/              ← L2 summary metadata
│   └── .checkpoints/             ← Pipeline checkpoints
├── cache/semantic_scholar/       ← S2 API response cache
├── logs/                         ← Log files
├── paperhunter/                  ← Source code
├── test_*.py                     ← Test files (35 modules)
├── pyproject.toml                ← Project metadata + dependencies
├── requirements.txt              ← Pinned dependencies
├── TODO.md                       ← Development roadmap (v2.0)
├── TODO3.md                      ← Scalability improvements roadmap (v2.1)
└── TODO4_Scalability.md          ← 1.000+ paper scalability roadmap (v3.0)
```

---

## License

MIT License
