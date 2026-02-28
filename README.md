# 🔎 PaperHunter

**Autonomous Scientific Paper Search, Filter, and Download System**

PaperHunter is a robust, modular Python system that autonomously searches, filters, and downloads scientific papers focusing on LLMs and AI research. It uses a **Funnel Strategy** to efficiently find relevant papers:

1. **Broad Search** - Aggregates results from multiple sources asynchronously
2. **Smart Filtering** - Uses an LLM to judge relevance based on abstracts
3. **Hybrid Downloading** - Prioritizes direct API downloads, falls back to browser automation

## 🏗️ Architecture

```
paperhunter/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── models.py            # Pydantic data models
├── search_engine.py     # Multi-source paper aggregator
├── relevance_filter.py  # LLM-powered relevance filtering
├── download_manager.py  # Hybrid PDF downloader
└── main.py              # Main controller and CLI
```

## ✨ Features

- **Multi-Source Search**: ArXiv, Semantic Scholar, Hugging Face Papers
- **Async Processing**: Fast parallel searching with `asyncio`
- **Smart Deduplication**: Normalizes titles and merges duplicate entries
- **LLM Filtering**: Uses Gemini or Claude to analyze paper relevance
- **Hybrid Downloads**: Direct HTTP → Playwright fallback
- **GitHub Detection**: Finds associated code repositories
- **Rate Limiting**: Respects API limits to avoid bans

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project
cd vscode_agent_paper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required API keys:
- **Google Gemini** OR **Anthropic Claude** API key for LLM filtering
- **Semantic Scholar** API key (optional, but recommended for higher rate limits)

### Usage

#### Interactive Mode

```bash
python -m paperhunter.main
```

#### Command Line

```bash
# Basic search
python -m paperhunter.main -k "LLM security" -c "Papers about prompt injection and jailbreaking"

# With options
python -m paperhunter.main \
    -k "large language model attacks" \
    -c "Focus on red teaming and adversarial attacks. Ignore training methods." \
    -n 30 \
    -o ./papers \
    --llm-provider google
```

#### Command Line Options

| Option | Description |
|--------|-------------|
| `-k, --keywords` | Broad search keywords |
| `-c, --criteria` | Filtering criteria for LLM |
| `-n, --max-results` | Max results per source (default: 20) |
| `-o, --output-dir` | Output directory for PDFs |
| `--no-download` | Only search and filter, don't download |
| `-i, --interactive` | Prompt for confirmation before downloading |
| `--llm-provider` | LLM provider: `google` or `anthropic` |

### Python API

```python
import asyncio
from paperhunter.main import PaperHunter

async def main():
    hunter = PaperHunter(
        max_results_per_source=20,
        llm_provider="google"
    )
    
    result = await hunter.hunt(
        keywords="LLM security prompt injection",
        criteria="""
        I want papers about:
        - Prompt injection attacks
        - Jailbreaking LLMs
        - Red teaming AI systems
        
        Ignore papers about:
        - General training methods
        - GDPR compliance
        """,
        auto_download=True
    )
    
    print(f"Found {result['papers_relevant']} relevant papers")
    print(f"Downloaded {result['papers_downloaded']} PDFs")

asyncio.run(main())
```

## 🔧 Components

### Search Engine (`search_engine.py`)

```python
from paperhunter.search_engine import PaperAggregator

aggregator = PaperAggregator(max_results_per_source=20)
result = await aggregator.search("transformer architecture")

for paper in result.papers:
    print(f"{paper.title} - {paper.source}")
    if paper.github_url:
        print(f"  Code: {paper.github_url}")
```

### Relevance Filter (`relevance_filter.py`)

```python
from paperhunter.relevance_filter import RelevanceFilter

filter = RelevanceFilter(provider="google")
filtered = await filter.filter_papers(
    papers=papers,
    criteria="Must focus on security vulnerabilities"
)

relevant = filter.get_relevant_papers(filtered)
```

### Download Manager (`download_manager.py`)

```python
from paperhunter.download_manager import Downloader

downloader = Downloader(download_dir="./papers")
result = await downloader.download_paper(paper)

if result.success:
    print(f"Saved to: {result.file_path}")
```

## 📊 Data Models

All data is validated using Pydantic models:

- `Paper` - Scientific paper metadata
- `RelevanceResult` - LLM relevance judgment
- `FilteredPaper` - Paper with relevance analysis
- `DownloadResult` - Download attempt result
- `SearchQuery` - Search configuration
- `SearchResult` - Aggregated search results

## ⚠️ Notes

- **Rate Limits**: The system includes delays between requests. Semantic Scholar especially has strict limits without an API key.
- **Playwright**: Required for downloading from protected sources. Run `playwright install chromium` after installation.
- **LLM Costs**: Each paper's abstract is analyzed by the LLM, which may incur API costs.

---

## �️ Setup completo per chi clona la repository

La repository include già i **PDF** di ricerca (`scaricati/`) e il **database vettoriale FAISS** (`vector_db/`). I file `.faiss` (>100 MB) sono gestiti tramite **Git LFS**. Segui questi passaggi per configurare l'ambiente.

### 1. Clonare la repository

```bash
# Assicurati di avere Git LFS installato (necessario per i file .faiss >100 MB)
# Fedora: sudo dnf install git-lfs
# Ubuntu/Debian: sudo apt install git-lfs
# macOS: brew install git-lfs
git lfs install

git clone https://github.com/rick0101/llm_university_project.git
cd llm_university_project
```

> **Nota:** Git LFS scarica automaticamente i file `.faiss` durante il clone. Se il clone è lento, puoi usare `GIT_LFS_SKIP_SMUDGE=1 git clone ...` e poi `git lfs pull` in un secondo momento.

### 2. Creare l'ambiente virtuale (venv)

```bash
# Richiede Python 3.10+
python3 -m venv venv
source venv/bin/activate    # Linux / macOS
# oppure: venv\Scripts\activate   # Windows

# Installare le dipendenze
pip install --upgrade pip
pip install -r requirements.txt

# Installare il pacchetto in modalità sviluppo
pip install -e ".[dev]"

# Installare i browser per Playwright (necessario per il download da fonti protette)
playwright install chromium
```

### 3. Configurare le API keys

```bash
cp .env.example .env
nano .env   # oppure usa il tuo editor preferito
```

Compilare almeno:
- `GOOGLE_API_KEY` — chiave API Gemini (obbligatoria per embedding e LLM)
- `SEMANTIC_SCHOLAR_API_KEY` — opzionale, ma consigliata per rate limit più alti

### 4. Avviare GROBID (parsing PDF strutturato)

GROBID è usato per estrarre testo strutturato (sezioni, citazioni, metadata) dai PDF. Funziona come container Docker:

```bash
# Scarica e avvia GROBID (prima volta ci vuole un po')
docker run --rm -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.2-full

# Verifica che sia pronto (attendi ~30 secondi dopo l'avvio)
curl -s http://localhost:8070/api/isalive
# Deve rispondere: true
```

> **Nota:** Se GROBID non è disponibile, il sistema usa automaticamente PyMuPDF come fallback (configurato in `config.yaml` → `ingestion.fallback_to_pymupdf: true`).

### 5. Ingestion dei PDF e creazione dei database

> **Tutte le directory necessarie** (`qdrant_db/`, `data/`, `output/`, `logs/`, `cache/`) **vengono create automaticamente** dal sistema al primo avvio. Non è necessario crearle manualmente.

I PDF di ricerca sono già inclusi nel repository, organizzati in sottocartelle tematiche dentro `scaricati/`. Il mapping tra nome cartella e categoria è definito in `config.yaml` → `ingestion.category_mapping`:

| Cartella | Categoria |
|----------|-----------|
| `Prompt_injections1/` | Prompt Injection |
| `Adversarial_Code1/` | Adversarial Attacks |
| `Data_and_Model_Poisoning1/` | Data and Model Poisoning |
| `Hallucinations1/` | LLM Hallucinations |

Per aggiungere nuove categorie, basta creare una sottocartella e aggiungerla al `category_mapping` in `config.yaml`. Per aggiungere nuovi PDF a una categoria esistente, copiarli nella sottocartella corrispondente: la pipeline di ingestion rileverà automaticamente i file nuovi o modificati (delta update basato su hash SHA-256).

#### 5a. Eseguire la pipeline di ingestion (popola Qdrant + BM25)

```bash
# Esegue il parsing dei PDF, chunking, embedding e indicizzazione
# Crea automaticamente: qdrant_db/, data/hash_registry.json, data/bm25_index.pkl
python -m paperhunter.generate_paper ingest ./scaricati/
```

Questo comando:
1. Crea tutte le directory necessarie (`qdrant_db/`, `data/`, `output/`, `cache/`, ecc.)
2. Scansiona le sottocartelle di `scaricati/` e rileva i PDF nuovi/modificati
3. Estrae il testo strutturato con GROBID (o PyMuPDF come fallback)
4. Arricchisce i metadati con Semantic Scholar API (venue, citazioni, BibTeX)
5. Genera BibTeX automatico per ogni paper
6. Divide il testo in chunk di ~1500 caratteri (GROBID-aware, per sezione)
7. Genera gli embedding con Gemini (`gemini-embedding-001`, 3072 dim)
8. Salva i chunk in Qdrant (`qdrant_db/`) e l'indice BM25 (`data/bm25_index.pkl`)
9. Aggiorna il registro hash (`data/hash_registry.json`) e il master `.bib`

> **Riesecuzione incrementale:** una seconda esecuzione processa solo i PDF nuovi o modificati (0 se nulla è cambiato).

#### 5b. Verificare lo stato del database

```bash
# Statistiche su cosa è stato indicizzato
python -m paperhunter.generate_paper stats
```

### 6. Eseguire i test

I test usano mock e non richiedono database reali o API key:

```bash
# Eseguire tutti i test
python -m pytest test_*.py -v

# Esegui un singolo test
python test_bm25_index.py
python test_hybrid_retriever.py
python test_pipeline.py
```

### 7. Neo4j (opzionale — GraphRAG)

Se vuoi usare la componente GraphRAG:

```bash
# Avvia Neo4j Community Edition
docker run --rm -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:5-community

# Installa le dipendenze aggiuntive
pip install -e ".[graphrag]"

# Abilita Neo4j in config.yaml
# neo4j.enabled: true
```

### Riepilogo struttura dopo il setup

```
vscode_agent_paper/
├── .env                      # ← DA CREARE (cp .env.example .env)
├── venv/                     # ← DA CREARE (python -m venv venv)
├── scaricati/                # ✅ GIÀ NEL REPO (PDF di ricerca)
│   ├── Prompt_injections1/   #    (63 PDF)
│   ├── Adversarial_Code1/    #    (29 PDF)
│   ├── Data_and_Model_Poisoning1/ # (17 PDF)
│   └── Hallucinations1/      #    (35 PDF)
├── qdrant_db/                # ← CREATO AUTOMATICAMENTE dall'ingestion pipeline
├── data/
│   ├── hash_registry.json    # ← CREATO AUTOMATICAMENTE dall'ingestion pipeline
│   └── bm25_index.pkl        # ← CREATO AUTOMATICAMENTE dall'ingestion pipeline
├── output/                   # ← CREATO AUTOMATICAMENTE (ensure_output_dirs)
│   ├── versions/             #    Versioni del survey generato
│   ├── provenance/           #    Report provenance JSON/Markdown
│   └── .checkpoints/         #    Checkpoint per resume from crash
├── logs/                     # ← CREATO AUTOMATICAMENTE
├── cache/semantic_scholar/   # ✅ GIÀ NEL REPO (cache risposte S2 API)
├── vector_db/                # ✅ GIÀ NEL REPO (FAISS legacy v1, via Git LFS)
├── config.yaml               # Configurazione centralizzata
├── .env.example              # Template per le API keys
├── requirements.txt          # Dipendenze Python
└── pyproject.toml            # Metadata progetto
```

> **Nota:** I file `vector_db/papers_store.faiss` (gestito via Git LFS) e `vector_db/papers_store_meta.json` appartengono alla prima versione (PaperWriter v1 con FAISS) e sono inclusi nel repository. Il sistema attuale (v2) usa **Qdrant** come vector store.

---

## �📝 PaperWriter — RAG-based Paper Generation

PaperWriter is the second component of the project: given a collection of downloaded PDFs, it builds a **vector knowledge base** and generates a complete scientific paper with **automatic citations**.

### Architecture

```
paperhunter/paper_generator/
├── pdf_processor.py      # PDF text + metadata extraction (PyMuPDF)
├── chunker.py            # Section-aware text chunking
├── vector_store.py       # FAISS index + Gemini text-embedding-004
├── citation_manager.py   # Citation tracking [1], [2], … + bibliography
├── rag_engine.py         # Retrieval-Augmented Generation with sources
└── paper_writer.py       # End-to-end pipeline orchestrator

paperhunter/
└── generate_paper.py     # CLI entry point (5 commands)
```

### Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| PDF parsing | PyMuPDF | Fast, robust |
| Embeddings | Gemini `text-embedding-004` | Free tier (1500 RPM) |
| Vector DB | FAISS (local) | Persistent on disk, zero cost |
| LLM | Gemini Flash (default) | ~$0.05-0.15 for a full paper |

### Quick Start

```bash
# 1. Ingest PDFs into the knowledge base
python -m paperhunter.generate_paper ingest ./scaricati/

# 2. Check what's indexed
python -m paperhunter.generate_paper stats

# 3. Ask a question to the knowledge base
python -m paperhunter.generate_paper query "What are the main prompt injection techniques?"

# 4. Generate a full paper from a JSON outline
python -m paperhunter.generate_paper write \
  --title "Survey on LLM Security" \
  --outline sample_outline.json \
  --output ./output \
  --language english

# 5. Interactive guided mode
python -m paperhunter.generate_paper interactive
```

### Outline Format

Create a JSON file (see `sample_outline.json`):

```json
{
  "title": "A Survey on LLM Security Threats",
  "sections": {
    "Introduction": "Overview of LLMs and emerging security concerns.",
    "Prompt Injection": "Direct/indirect prompt injection, jailbreaking, defenses.",
    "Conclusion": "Summary of findings and future directions."
  }
}
```

### Global CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db-dir` | `./vector_db` | Vector DB persistence directory |
| `--llm-provider` | `google` | `google` or `anthropic` |
| `--llm-model` | `gemini-3-flash-preview` | LLM model name |
| `--chunk-size` | `1000` | Chunk size in characters |
| `--language` | `english` | Output language |
| `--temperature` | `0.3` | LLM temperature |

### Python API

```python
from paperhunter.paper_generator import PaperWriter

writer = PaperWriter(llm_model="gemini-3-flash-preview")

# Ingest PDFs
writer.ingest_papers(Path("./my_pdfs"))

# Generate paper
paper = writer.generate_paper(
    title="Survey on LLM Security",
    outline={
        "Introduction": "Overview of LLMs and security threats.",
        "Attacks": "Prompt injection, data poisoning, adversarial attacks.",
        "Conclusion": "Summary and future work.",
    },
    output_dir=Path("./output"),
    language="english",
)

# Or just ask a question
result = writer.query("What defenses exist against prompt injection?")
print(result["answer"])
```

## 📝 License

MIT License
