# AGENTS.md — PaperHunter

> **Guida sintetica per agenti AI che lavorano su questa repository.**
> Volutamente compatta: dà il contesto essenziale e lascia spazio all'esplorazione autonoma.
> Per dettagli approfonditi → `README.md` (90KB, leggere a sezioni con `view_range`).

---

## 1. Cos'è il progetto

**PaperHunter** (v3.6) è una pipeline modulare e agentica che genera autonomamente un **survey paper scientifico** (dominio: sicurezza degli LLM) partendo da centinaia/migliaia di PDF.

Output finale: survey in Markdown + LaTeX (ACL/IEEEtran) con citazioni `[bibtex_key]` verificabili, bibliografia, figure, provenance report.

Linguaggio: **Python ≥3.10**. Package distribuibile (`pyproject.toml`), entry point CLI: `paperhunter`.

---

## 2. Pipeline a 4 fasi (CLI subcommands)

```
paperhunter ingest     → PDF → GROBID → metadata (S2/DBLP) → BibTeX → chunk → Qdrant
paperhunter summarize  → L1 (per paper) → L1.5 (cluster RAPTOR) → L2 (per categoria)
paperhunter index      → BM25 + KnowledgeGraph (Neo4j) + Community Detection (Louvain/Leiden)
paperhunter write      → Planner → Writer ↔ Critic (loop) → Assembler → survey.md
```

Subcommands ausiliari: `search`, `query`, `stats`, `export`, `cleanup`.

**Compressione gerarchica RAPTOR** (per evitare context rot):
chunk (~75M tok) → L1 (~2M) → L1.5 (~100k) → L2 (~24k).

---

## 3. Struttura della repository

```
paperhunter/                  # Package principale
├── main.py                   # CLI entry point (argparse, lazy imports)
├── config.py                 # Pydantic settings (legge config.yaml + .env)
├── models.py                 # Modelli Pydantic condivisi
├── ingestion/                # Fase 1: GROBID, S2, DBLP, BibTeX, chunker, figure extractor
├── summarization/            # Fase 2: L1, L1.5 (clustering), L2, taxonomy, invalidation
├── indexing/                 # BM25, Qdrant store, hybrid retriever, reranker, venue ranker
├── graphrag/                 # Neo4j PropertyGraph, entity extraction, community detect/summarize
├── agents/                   # Fase 4 LangGraph: planner, writer, critic, assembler
│   ├── pipeline.py           # Orchestrazione LangGraph
│   ├── state.py              # Stato condiviso degli agenti
│   ├── tools/retrieval_tools.py  # Tool agentic (vector/keyword/graph)
│   └── figure_agent/         # Legacy visual feature, hard-disabled (DeepSeek-only)
├── postprocessing/           # bibliography, figure_resolver, latex_exporter, versioning, checkpoint, provenance
├── evolution/                # Self-evolution: lesson_store, lesson_extractor, prompt_overlay, skill_loader
├── bridge/                   # Adapter verso AutoResearchClaw
└── utils/                    # logging, helpers

tests/                        # ~50 test pytest (unit + e2e)
prompts/                      # Prompt template per LLM/agenti
docs/                         # Documentazione aggiuntiva
experiments/                  # ollama_A100_local (eval jailbreak + grafici + adaptive_suite), real_test_001
tools/                        # Script utility
config.yaml                   # Configurazione centrale (modelli, GROBID, Neo4j, Qdrant, ecc.)
config.researchclaw.yaml      # Config per il bridge AutoResearchClaw
.env                          # API keys (DeepSeek, S2) — MAI committare
docker-compose.yml            # Servizi: GROBID, Neo4j
qdrant_db/                    # Storage locale Qdrant (vettori) — non in git
scaricati/                    # PDF scaricati per ingestion
output/                       # Survey generati
cache/, logs/                 # Cache e log runtime
```

File ausiliari informativi (NON sono il codice principale, sono note storiche / piani):
`FIX_PLAN_CITATIONS.md`, `SCALABILITY_FIXES.md`, `README_old.md`, `README_dots.ocr.md`, `prezzo_stima.txt`.

Script standalone alla root: `run_writer_summaries_only.py`, `fix_hashes.py`, `fix_survey_md.py`, `verify_dblp.py`, `debug_mocha.py`, `test_bibtex.py`, `test_grobid.py` (test ad-hoc, non pytest).

---

## 4. Stack tecnologico chiave

- **LLM testuale**: solo API nativa DeepSeek (`deepseek-v4-pro` per tutti i tier,
  thinking + reasoning `max`). Google AI Studio e OpenRouter sono disabilitati.
- **Embedding**: BGE-M3 locale (1024d); gli embedding Gemini e multimodali sono
  hard-disabled.
- **RAG framework**: LlamaIndex (≥0.14).
- **Multi-agent**: LangGraph (Planner/Writer/Critic/Assembler ReAct).
- **Vector DB**: Qdrant (locale, persist su disco).
- **Knowledge Graph**: Neo4j (opzionale, GDS per community detection).
- **PDF parsing**: GROBID (Docker `lfoppiano/grobid:0.8.2-full` su :8070), backup PyMuPDF.
- **Hybrid retrieval**: BM25 (`rank-bm25`) + vettori, Reciprocal Rank Fusion,
  venue ranking, reranker del provider selezionato.
- **Clustering RAPTOR**: scikit-learn.
- **Metadati**: Semantic Scholar API + DBLP lookup.

---

## 5. Configurazione & comandi essenziali

- Settings: `config.yaml` (modelli, timeout, batch size, recency, paths) → letto da `paperhunter/config.py` (Pydantic).
- Secrets: `.env` (vedi `.env.example`).
- Provider obbligatorio: `LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`. Google/Gemini
  non sono supportati; Vision, embedding multimodali e FigureAgent sono hard-disabled.
- **Setup servizi**: `docker-compose up -d` (GROBID + Neo4j).
- **Install**: `pip install -e .` (oppure `pip install -r requirements.txt`).
- **Test**: `pytest` (config in `pyproject.toml`, `asyncio_mode=auto`, `testpaths=["tests"]`).
- **Lint/format**: `ruff` + `black` (line-length 100), `mypy` (configurati in `pyproject.toml`).
- **Esecuzione remota obbligatoria**: qualsiasi subcommand `paperhunter` e i servizi Docker di questa pipeline devono essere eseguiti esclusivamente sul server personale `ricca-f@192.168.1.63`, nella copia sincronizzata `~/llm_university_project`; mai su questo PC. Prima di una run, sincronizzare anche gli artefatti runtime necessari (`scaricati/`, `qdrant_db/`, `cache/`, `output/`, `logs/`) oltre ai file versionati. Sequenza standard: `docker compose up -d`, `paperhunter ingest --force`, `paperhunter summarize --force`, `paperhunter index`, `paperhunter write --force`. Verificare l'exit status e i log di ogni fase; applicare solo correzioni operative non sostanziali senza alterare il progetto.

---

## 6. Convenzioni & pattern ricorrenti

- **Lazy imports** in `main.py`: ogni subcommand importa i propri moduli solo quando invocato (la pipeline è pesante).
- **Citazioni** nel formato `[bibtex_key]`. Solo record verificati uno per uno
  su DBLP sono citabili: match strict titolo/autore/anno, BibTeX ufficiale e
  manifest SHA-256. I fallback locali restano in `references_uncited.bib` solo
  come contesto.
- **Export finale fail-closed**: il LaTeX non viene scritto se una chiave è
  assente/duplicata/non verificata o se il BibTeX diverge dal manifest DBLP;
  il dettaglio è in `citation_integrity_report.json`.
- **Critic strict**: 100% dei claim citati deve risultare `SUPPORTED`, con
  almeno 5 fonti distinte per sezione; nessuna auto-accettazione al limite.
- **Auto-invalidazione** dei summary quando i chunk a monte cambiano (`summarization/invalidation_manager.py`).
- **Checkpoint + resume** in fase write (`postprocessing/checkpoint_manager.py`, `--resume`).
- **Provenance tracking** per ogni claim nel survey (`postprocessing/provenance_tracker.py`).
- **Self-evolution**: `evolution/` apprende dai fallimenti del Critic e produce overlay di prompt + skill.
- **Recency priority** globale (config) → paper recenti pesati di più in ingestion/retrieval/L2.
- 7 collezioni Qdrant BGE-M3 distinte (suffisso `_bge_m3`); la collection
  multimodale legacy non viene creata o interrogata nel profilo DeepSeek-only.
- Italiano nei commenti/docstring; codice e identificatori in inglese.
- In `experiments/ollama_A100_local`, i grafici per singolo run usano
  `generate_charts.py`; i confronti tra modelli usano
  `generate_cross_model_charts.py` e per default prendono l'ultimo
  `eval_*.json` completo per modello.
- In `experiments/ollama_A100_local/adaptive_suite`, gli stessi script sono
  disponibili come `adaptive_suite/generate_charts.py` e
  `adaptive_suite/generate_cross_model_charts.py`: leggono da
  `adaptive_suite/results/` e scrivono in `adaptive_suite/charts/`.
- In `experiments/ollama_A100_local`, le chiamate Ollama condividono
  `ollama_health.py` per health/restart safe e impostano `num_ctx` tramite
  `config.CONTEXT_WINDOW` / `OLLAMA_EVAL_NUM_CTX` (default 32768) per evitare il
  default 131k di DeepSeek.
- `experiments/ollama_A100_local/adaptive_suite` e' una suite compatta
  data-mined: legge i modelli da `../models.txt`, salva output locali in
  `adaptive_suite/results|traces|logs`, e usa `expand_guide.md` per generare
  future varianti da risultati precedenti.
  - **v2** (2026-05-01): 10 attacchi × 40 prompt × 5 difese = **2000 test/modello**.
    Sostituiti Adaptive-Crescendo e Adaptive-Stacked-MS-Refusal con
    Adaptive-CognitiveOverload (specialista poi-1, 60% no-def) e
    Adaptive-ObfuscationFlip (FlipAttack+RS, coverage obfuscation).
    Completati tutti e 40 i prompt (erano troncati a 31 per bug SyntaxError).
    Metadata risultati usa `suite: "adaptive_suite_v2"` per evitare contaminazione
    dei checkpoint precedenti.

---

## 7. Esplorazione consigliata per nuove sessioni

Quando ti serve contesto su un'area specifica, leggi prima questi punti d'ingresso:

| Area                 | File da aprire per primo                                              |
|----------------------|------------------------------------------------------------------------|
| CLI / overview       | `paperhunter/main.py` (docstring + parser)                             |
| Configurazione       | `config.yaml` + `paperhunter/config.py`                                |
| Modelli dati         | `paperhunter/models.py`                                                |
| Pipeline ingestion   | `paperhunter/ingestion/ingestion_pipeline.py`                          |
| Verifica DBLP/BibTeX | `paperhunter/ingestion/dblp_lookup.py` + `bibtex_generator.py`          |
| Summarization        | `paperhunter/summarization/level{1,1_5,2}_summarizer.py`               |
| Retrieval            | `paperhunter/indexing/hybrid_retriever.py`                             |
| Agenti / writing     | `paperhunter/agents/pipeline.py` + `state.py`                          |
| Feature visuale (disabilitata) | `paperhunter/agents/figure_agent/planner.py`                    |
| Export finale/gate   | `paperhunter/postprocessing/latex_exporter.py`                         |
| Test di un modulo X  | `tests/test_X.py` (naming 1:1 col modulo)                              |

Per il README completo usa `view` con `view_range` (è ~90KB, supera il limite di 50KB di lettura singola).

---

## 8. ⚠️ Regola di manutenzione di questo file

> **Ogni volta che il progetto viene modificato (nuovi moduli, rinomine, nuove fasi della pipeline, cambi di stack, nuovi comandi CLI, nuove convenzioni, refactor strutturali, ecc.), questo file `AGENTS.md` DEVE essere aggiornato di conseguenza nella stessa sessione/PR.**
>
> L'aggiornamento è **obbligatorio** e prevede di:
> - **aggiungere** sezioni/voci per le parti nuove,
> - **modificare** quelle che sono cambiate,
> - **eliminare** quelle ormai obsolete o rimosse.
>
> Mantieni il file **sintetico** (questa è una mappa, non la documentazione completa). Se una sezione cresce troppo, sposta i dettagli nel `README.md` o in `docs/` e lascia qui solo il puntatore. L'obiettivo è preservare il contesto essenziale per gli agenti senza saturare la finestra di contesto, lasciando spazio all'esplorazione autonoma del codice.
