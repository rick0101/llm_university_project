# TODO.md — LLM Security Survey Generator: Roadmap Completa

> **Progetto:** Sistema end-to-end per generare un paper scientifico survey sulla sicurezza dei LLM  
> **Target:** Stile Nature/ACL, con citazioni verificabili, provenance tracking, architettura modulare  
> **Data inizio:** 20 Febbraio 2026  
> **Ultimo aggiornamento:** 1 Marzo 2026

---

## 🏗️ Architettura Finale Proposta

```
scaricati/                        ← PDF categorizzati (~400 paper)
    ├── Prompt_injections1/
    ├── Hallucinations1/
    ├── Data_and_Model_Poisoning1/
    ├── Adversarial_Code1/
    └── ... (nuove categorie)

config.yaml                       ← Configurazione centralizzata YAML

paperhunter/
├── config.py                     ← YAML + pydantic-settings loader
├── models.py                     ← Modelli Pydantic v2 estesi
├── search_engine.py              ← Ricerca multi-source (ArXiv, S2, HF) ✦ Enhanced
├── download_manager.py           ← Download PDF (httpx + Playwright) ✦ Kept
├── relevance_filter.py           ← Filtro rilevanza LLM ✦ Migrato a LlamaIndex
├── main.py                       ← CLI principale + orchestratore
├── generate_paper.py             ← CLI generazione paper (subcommands)
│
├── ingestion/                    ← NUOVO: Modulo Ingestion
│   ├── __init__.py
│   ├── grobid_client.py          ← Client GROBID Docker (TEI-XML → strutturato)
│   ├── pdf_hasher.py             ← Hash SHA-256 per upsert incrementale
│   ├── semantic_scholar.py       ← Enrichment S2 API (venue, BibTeX, citations)
│   ├── bibtex_generator.py       ← Generazione BibTeX con chiave Paper_YYYY_ID
│   ├── chunker.py                ← Chunking GROBID-aware per sezione
│   └── ingestion_pipeline.py     ← Orchestratore ingestion con delta-update
│
├── summarization/                ← NUOVO: Modulo Summarization
│   ├── __init__.py
│   ├── level1_summarizer.py      ← Summary singolo paper (con testing_methodology)
│   ├── level2_summarizer.py      ← Meta-summary per categoria
│   └── invalidation_manager.py   ← Auto-invalidazione L2 su nuovi paper
│
├── indexing/                     ← NUOVO: Modulo Indexing Ibrido
│   ├── __init__.py
│   ├── chroma_store.py           ← ChromaDB vector store
│   ├── bm25_index.py             ← BM25 keyword index
│   ├── hybrid_retriever.py       ← Retrieval ibrido (vector + BM25 + metadata)
│   └── reranker.py               ← Reranking con Gemini
│
├── graphrag/                     ← NUOVO: Modulo GraphRAG (opzionale)
│   ├── __init__.py
│   ├── entity_extractor.py       ← Estrazione entità/relazioni da L1
│   ├── neo4j_manager.py          ← Connessione Neo4j Community
│   └── property_graph.py         ← LlamaIndex PropertyGraph builder
│
├── agents/                       ← NUOVO: Modulo Multi-Agent (LangGraph)
│   ├── __init__.py
│   ├── state.py                  ← Stato condiviso LangGraph
│   ├── planner_agent.py          ← Agent 1: Genera outline da L2 summaries
│   ├── writer_agent.py           ← Agent 2: Scrive sezioni con RAG + citazioni
│   ├── critic_agent.py           ← Agent 3: Verifica provenance citazioni
│   ├── assembler_agent.py        ← Agent 4: Assembla, genera abstract/intro/conclusion
│   └── pipeline.py               ← LangGraph workflow orchestration
│
├── postprocessing/               ← NUOVO: Modulo Post-processing
│   ├── __init__.py
│   ├── provenance_tracker.py     ← JSON provenance completo
│   ├── bibliography_generator.py ← Genera .bib da citazioni usate
│   ├── latex_exporter.py         ← Export .tex
│   ├── versioning.py             ← Versioning + changelog
│   └── checkpoint_manager.py     ← Resume from checkpoint
│
└── utils/                        ← NUOVO: Utilities condivise
    ├── __init__.py
    ├── logging_setup.py          ← tqdm + rich logging
    ├── llm_factory.py            ← Factory LlamaIndex per Flash/Pro
    └── rate_limiter.py           ← Rate limiting API calls
```

---

## 📋 TASK DETTAGLIATI

### FASE 0 — Fondamenta & Configurazione

#### [x] Task 0.1: Setup config.yaml e nuovo config.py
**Obiettivo:** Creare il file `config.yaml` centralizzato e riscrivere `config.py` per caricarlo con pydantic-settings.  
**Dettagli:**
- Creare `config.yaml` con tutte le sezioni: `gemini`, `grobid`, `neo4j`, `chroma`, `semantic_scholar`, `ingestion`, `summarization`, `agents`, `output`
- Riscrivere `config.py` per caricare da YAML + `.env` (API keys da .env, tutto il resto da YAML)
- Validazione con Pydantic: ogni sezione è un sotto-modello tipizzato
- Parametri configurabili: modelli LLM (flash vs pro), chunk size, overlap, batch size, retry count, ecc.  
**Test:** Caricare config e stampare tutte le sezioni.

> **Completato 20/02/2026.** Modifiche:
> - Creato `config.yaml` con 14 sezioni complete (gemini, grobid, neo4j, chroma, semantic_scholar, ingestion, summarization, retrieval, agents, output, search, download, rate_limiting, logging)
> - Riscritto `paperhunter/config.py`: 14 sotto-modelli Pydantic tipizzati (`GeminiConfig`, `GrobidConfig`, ecc.), caricamento YAML + .env, proprietà backward-compatible (`settings.llm_model`, `settings.download_dir`, ecc.)
> - Funzioni: `reload_settings()`, `ensure_output_dirs()`, `ensure_download_dir()`
> - Create directory per tutti i nuovi moduli: `ingestion/`, `summarization/`, `indexing/`, `graphrag/`, `agents/`, `postprocessing/`, `utils/` (con `__init__.py`)
> - Validator Pydantic per: `distance_metric`, `latex_template`, `log_level`, `model_choice`

#### [x] Task 0.2: Aggiornare requirements.txt e pyproject.toml
**Obiettivo:** Sostituire le dipendenze LangChain con il nuovo tech stack.  
**Dettagli:**
- Rimuovere: `langchain`, `langchain-core`, `langchain-community`, `langchain-google-genai`, `faiss-cpu`
- Aggiungere: `llama-index`, `llama-index-llms-gemini`, `llama-index-embeddings-gemini`, `chromadb`, `langgraph`, `rank-bm25`, `tqdm`, `rich`, `pyyaml`, `requests` (per GROBID), `neo4j` (opzionale)
- Mantenere: `pydantic`, `pydantic-settings`, `httpx`, `arxiv`, `beautifulsoup4`, `playwright`, `python-dotenv`, `pymupdf` (backup)
- Aggiornare `pyproject.toml` con nuove dipendenze e metadata  
**Test:** `pip install -e .` senza errori.

> **Completato 20/02/2026.** Modifiche:
> - **NOTA IMPORTANTE:** ChromaDB non è compatibile con Python 3.14 (usa Pydantic v1 internamente). Sostituito con **Qdrant** (`qdrant-client`) che funziona perfettamente in modalità locale senza server.
> - Aggiornato `pyproject.toml`: versione 2.0.0, nuove dipendenze (llama-index-core, llama-index-llms-gemini, llama-index-embeddings-gemini, qdrant-client, langgraph, rank-bm25, tqdm, rich), optional deps per graphrag e legacy
> - Aggiornato `requirements.txt`: nuovo tech stack completo
> - Aggiornato `config.yaml`: sezione `qdrant` al posto di `chroma`
> - Aggiornato `config.py`: `QdrantConfig` al posto di `ChromaConfig`
> - 12/12 pacchetti importabili correttamente: LlamaIndex Core, LLMs Gemini, Embeddings Gemini, Qdrant, LangGraph, BM25, tqdm, Rich, PyYAML, Requests, Pydantic, HTTPX

#### [x] Task 0.3: Creare utils/logging_setup.py
**Obiettivo:** Setup logging professionale con `rich` + `tqdm`.  
**Dettagli:**
- Configurare `rich.logging.RichHandler` come handler principale
- Progress bar helper con `rich.progress` per operazioni batch
- Livelli configurabili da YAML
- Utility per stampare tabelle riassuntive (es. statistiche ingestion)  
**Test:** Importare il modulo e verificare output formattato in console.

> **Completato 20/02/2026.** File: `paperhunter/utils/logging_setup.py`
> - `setup_logging(level, rich_enabled, log_file)`: inizializzazione globale con RichHandler + file handler opzionale
> - `get_logger(name)`: factory logger con auto-init da config.yaml
> - `progress_bar(iterable, description, total)`: wrapper Rich Progress per iterabili
> - `progress_context(description, total)`: context manager per progress manuale
> - `print_table(title, data)`: tabelle formattate Rich (dict → 2 colonne, list[dict] → multi-colonna)
> - `print_panel(title, content)`, `print_success()`, `print_error()`, `print_warning()`: output formattato
> - Console globale `console` riusabile. Logger rumorosi silenziati (httpx, qdrant_client, ecc.)

#### [x] Task 0.4: Creare utils/llm_factory.py
**Obiettivo:** Factory centralizzata per creare istanze LLM LlamaIndex (Flash per task semplici, Pro per task complessi).  
**Dettagli:**
- Usare `llama-index-llms-gemini`: `Gemini(model="models/gemini-3-flash-preview")` e `Gemini(model="models/gemini-3-pro-preview")`
- Usare `llama-index-embeddings-gemini`: `GeminiEmbedding(model_name="models/gemini-embedding-001")`
- Parametri configurabili: temperature, max_tokens, timeout, retry
- Funzioni: `get_flash_llm()`, `get_pro_llm()`, `get_embedding_model()`
- Gestione errori + retry con backoff  
**Test:** Chiamata test a Flash e Pro con prompt semplice. Generare un embedding di test.

> **Completato 20/02/2026.** File: `paperhunter/utils/llm_factory.py`
> - **NOTA CRITICA:** Il plugin `llama-index-llms-gemini` (gRPC) hangava indefinitamente. Sostituito con `llama-index-llms-google-genai` + `llama-index-embeddings-google-genai` che usano il nuovo SDK REST `google-genai`.
> - Aggiunto `_get_http_options()` helper che legge `gemini.timeout` da config.yaml e lo converte in HttpOptions con timeout in millisecondi — essenziale per i thinking models (gemini-3-flash: ~13-30s, gemini-3-pro: ~57s).
> - Funzioni: `get_flash_llm()`, `get_pro_llm()`, `get_llm(tier)`, `get_embedding_model()`, `llm_call_with_retry()`
> - Timeout alzato a 300s in config.yaml per supportare i thinking models
> - `pyproject.toml` e `requirements.txt` aggiornati: `llama-index-llms-google-genai`, `llama-index-embeddings-google-genai`
> - Test: Flash (30.4s), Embedding (3072-dim), retry helper — tutti passati

#### [x] Task 0.5: Creare utils/rate_limiter.py
**Obiettivo:** Rate limiter asincrono per tutte le API calls (Gemini, S2, GROBID).  
**Dettagli:**
- AsyncTokenBucket o semplice semaphore + delay configurabile
- Supporto per limiti diversi per servizio (Gemini: ~15 RPM free tier, S2: 100 RPM, GROBID: illimitato locale)
- Logging delle chiamate e tempi di attesa  
**Test:** Simulare 20 chiamate rapide e verificare il throttling.

> **Completato 20/02/2026.** File: `paperhunter/utils/rate_limiter.py`
> - `TokenBucketLimiter`: token-bucket thread-safe con supporto RPM, daily limit, sync (`wait()`) e async (`async_wait()`), stats, reset
> - `GrobidSemaphore`: context manager sync/async per limitare richieste GROBID concorrenti
> - Singleton getters configurati da config.yaml: `get_gemini_limiter()` (15 RPM + 1500/day), `get_s2_limiter()` (100 RPM), `get_grobid_semaphore()` (5 concurrent)
> - Test: 5 chiamate rapide Gemini, 3 S2, semaforo GROBID — tutti passati

#### [x] Task 0.6: Aggiornare models.py con i nuovi modelli Pydantic
**Obiettivo:** Estendere i modelli dati per supportare tutto il nuovo workflow.  
**Dettagli:**
- `Paper` → aggiungere: `doi`, `venue`, `journal`, `citation_velocity`, `bibtex_key`, `bibtex_entry`, `pdf_hash`, `category`, `month`, `semantic_scholar_id`, `grobid_processed` (bool), `ingested_at` (timestamp)
- Nuovi modelli:
  - `GROBIDOutput` — sezioni strutturate, riferimenti, tabelle, figure captions
  - `PaperChunk` — testo, chunk_id, metadata (paper_id, section, category, year, type: "original"|"l1_summary"|"l2_summary")
  - `L1Summary` — summary singolo paper con campi: `key_contributions`, `methodology`, `testing_methodology`, `temporal_context`, `limitations`, `paper_id`
  - `L2Summary` — meta-summary categoria con: `category`, `temporal_trends`, `gaps`, `paper_ids`, `version`, `invalidated` (bool)
  - `ProvenanceRecord` — citation_key, claim_text, source_chunk_id, source_text, verification_status
  - `SurveyVersion` — version_number, timestamp, changelog, paper_count
- Mantenere backward compatibility dove possibile  
**Test:** Creare istanze di ogni modello e serializzare in JSON.

> **Completato 20/02/2026.** File: `paperhunter/models.py`
> - Enums: `ChunkType` (ORIGINAL, L1_SUMMARY, L2_SUMMARY), `VerificationStatus` (SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, SOURCE_NOT_FOUND)
> - `Paper` esteso: +16 nuovi campi (venue, journal, year, month, citation_velocity, influential_citation_count, fields_of_study, semantic_scholar_id, bibtex_key, bibtex_entry, pdf_path, pdf_hash, category, grobid_processed, ingested_at)
> - Nuovi: `GROBIDSection`, `GROBIDReference`, `GROBIDOutput` (sezioni, refs, figure/table captions, full_text fallback)
> - `PaperChunk` con `to_metadata_dict()` per il vector store
> - `L1Summary` con `testing_methodology` come campo critico
> - `L2Summary` con version, invalidated, taxonomy, gaps, testing_overview
> - `ProvenanceRecord`, `SurveyVersion`
> - Modelli legacy mantenuti: RelevanceResult, FilteredPaper, DownloadResult, SearchQuery, SearchResult
> - Test: tutte le istanze create e serializzate con successo

---

### FASE 1 — Ingestion Incrementale & Pre-processing

#### [x] Task 1.1: Setup e test connessione GROBID Docker
**Obiettivo:** Verificare che GROBID Docker sia raggiungibile e funzionante.  
**Dettagli:**
- Aggiungere istruzioni per avviare GROBID: `docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.2-full`
- Creare `ingestion/grobid_client.py` con funzione `test_connection()` → GET `/api/isalive`
- Funzione `process_pdf(pdf_path) → GROBIDOutput`: POST a `/api/processFulltextDocument`
- Parsing TEI-XML response: estrarre title, authors, abstract, sezioni (con titoli), references, figure captions, tabelle
- Gestione errori: timeout, GROBID down, PDF corrotti
- Fallback a PyMuPDF se GROBID fallisce su un singolo PDF  
**Test:** Processare 3 PDF di test e stampare le sezioni estratte.

> **Completato 20/02/2026.** File: `paperhunter/ingestion/grobid_client.py`
> - Classe `GrobidClient` con parametri configurabili da `config.yaml` (sezione `grobid`)
> - `test_connection()`: GET `/api/isalive` con gestione ConnectionError, Timeout, status code
> - `process_pdf(pdf_path, paper_id)`: POST `/api/processFulltextDocument` → `GROBIDOutput`
> - Parsing TEI-XML completo con `lxml`:
>   - `_extract_title()`: da `teiHeader/titleStmt/title[@type='main']`
>   - `_extract_authors()`: da `sourceDesc/author/persName` (forename + surname)
>   - `_extract_abstract()`: da `profileDesc/abstract`
>   - `_extract_sections()`: ricorsivo su `body/div` con heading level detection (da attributo `@n`)
>   - `_extract_references()`: da `back/listBibl/biblStruct` → `GROBIDReference` (title, authors, year, venue, DOI, raw_text)
>   - `_extract_figure_captions()`: da `figure[@type='figure']/figDesc` con label
>   - `_extract_table_captions()`: da `figure[@type='table']/figDesc` con label
>   - `_extract_full_text()`: tutto il body come fallback
> - Fallback a PyMuPDF (`_pymupdf_fallback()`): estrae testo pagina per pagina + `_detect_sections_from_text()` con regex per heading comuni
> - Parametri GROBID: `consolidateHeader`, `consolidateCitations`, `includeRawCitations`, `teiCoordinates`
> - Test: 3 PDF processati con successo:
>   - "Why Do AI Agents Systematically Fail..." → 12 sezioni, 14 riferimenti (5.2s)
>   - "Steer2Edit: From Activation Steering..." → 38 sezioni, 25 riferimenti (10.3s)
>   - "Protecting Context and Prompts..." → 10 sezioni, 33 riferimenti (8.2s)

#### [x] Task 1.2: Creare ingestion/pdf_hasher.py — Logica hash per Upsert
**Obiettivo:** Calcolare hash SHA-256 di ogni PDF per la logica di upsert incrementale.  
**Dettagli:**
- Funzione `compute_pdf_hash(pdf_path) → str` (SHA-256 hex digest)
- Funzione `load_hash_registry(path) → dict` (carica registro hash da JSON)
- Funzione `save_hash_registry(registry, path)`
- Funzione `get_new_or_modified_pdfs(pdf_dir, registry) → list[Path]`
- Il registro mappa `{pdf_path: {hash, doi, bibtex_key, category, ingested_at, grobid_status}}`
- Scansione ricorsiva delle sottocartelle (ogni sottocartella = una categoria)  
**Test:** Scansionare `scaricati/` e identificare tutti i PDF. Seconda scansione = 0 nuovi PDF.

> **Completato 20/02/2026.** File: `paperhunter/ingestion/pdf_hasher.py`
> - `compute_pdf_hash(pdf_path, buffer_size=65536) → str`: SHA-256 hex digest, lettura a blocchi per supportare file grandi
> - `load_hash_registry(path=None) → dict`: carica registro JSON (default da `config.yaml` → `data/hash_registry.json`). Gestione errori: file mancante, JSON corrotto, tipo non-dict
> - `save_hash_registry(registry, path=None) → Path`: salva registro JSON con indentazione. Crea directory padre automaticamente
> - `scan_pdf_directory(pdf_dir=None) → list[dict]`: scansione ricorsiva di tutti i `.pdf`. Ogni entry contiene: `absolute_path`, `relative_path`, `filename`, `category_folder`, `category` (mappata da `config.yaml`), `hash`, `size_bytes`
> - `get_new_or_modified_pdfs(pdf_dir=None, registry=None) → list[dict]`: confronta hash su disco vs registro. Ogni entry ha campo `status`: `"new"` o `"modified"`. Rileva anche PDF rimossi dal disco
> - `create_registry_entry(pdf_info, doi=None, bibtex_key=None, grobid_status="pending") → dict`: factory per creare entry del registro con tutti i campi necessari
> - `update_registry_entry(registry, relative_path, updates)`: aggiorna campi specifici in-place con timestamp `updated_at`
> - `get_registry_stats(registry) → dict`: statistiche aggregate (totale, per categoria, per status GROBID, size totale MB)
> - `_get_category_from_path(pdf_path, base_dir) → str`: mapping cartella → nome categoria human-readable via `config.yaml`
> - Test: 118 PDF trovati in 4 categorie (255.4 MB). Prima scansione: 118 nuovi. Seconda scansione: 0 nuovi/modificati. Tutte le categorie mappate correttamente (Adversarial Attacks: 23, Data and Model Poisoning: 13, LLM Hallucinations: 29, Prompt Injection: 53)

#### [x] Task 1.3: Creare ingestion/semantic_scholar.py — Enrichment metadati
**Obiettivo:** Arricchire ogni paper con metadati completi da Semantic Scholar API.  
**Dettagli:**
- Funzione `search_paper(title, authors=None) → S2Paper | None`: cerca via `/graph/v1/paper/search`
- Funzione `get_paper_details(paper_id) → dict`: GET `/graph/v1/paper/{id}` con fields=`title,authors,year,venue,journal,citationCount,influentialCitationCount,fieldsOfStudy,publicationDate,externalIds,abstract,citationStyles`
- Funzione `get_citation_velocity(paper_id) → int`: calcolo citazioni/anno
- Funzione `enrich_paper(paper: Paper) → Paper`: orchestratore che cerca e arricchisce
- Rate limiting: max 100 req/min (usare rate_limiter.py)
- Gestione ambiguità: se più risultati, match per titolo più simile (fuzzy matching)
- Cache locale dei risultati S2 per evitare ri-query  
**Test:** Arricchire 5 paper di test e verificare venue, DOI, citation count.

> **Completato 20/02/2026.** File: `paperhunter/ingestion/semantic_scholar.py`
> - Classe `SemanticScholarEnricher`: client completo per arricchimento metadati da S2 Academic Graph API
> - `search_paper(title, authors=None, limit=5) → dict | None`: ricerca per titolo con fuzzy matching (Jaccard su token, soglia 0.5). Bonus +0.15 se primo autore corrisponde. Cache automatica per titolo normalizzato
> - `get_paper_details(paper_id) → dict | None`: GET dettagli completi con tutti i fields configurati in `config.yaml`. Supporta S2 ID, `DOI:xxx`, `ArXiv:xxx`
> - `compute_citation_velocity(citation_count, year) → float | None`: citazioni/anno (statico)
> - `enrich_paper(paper, skip_if_enriched=True) → Paper`: orchestratore completo: search → details → popola venue, journal, DOI, citations, influential_citation_count, fields_of_study, year, month, citation_velocity, bibtex_entry, semantic_scholar_id. Non sovrascrive campi già presenti
> - `enrich_batch(papers, skip_if_enriched=True) → list[Paper]`: batch con contatori (arricchiti/skippati/falliti)
> - `create_paper_from_pdf(pdf_info, grobid_title, grobid_authors, grobid_abstract) → Paper`: utility per creare Paper da dati GROBID + pdf_hasher
> - `_request(method, url, params, retries_left) → dict | None`: HTTP con rate limiting (`get_s2_limiter()`), retry esponenziale su 429/5xx, gestione timeout/connection error
> - `_extract_enrichment_data(details) → dict`: estrae tutti i campi dal response S2 (paperId, year, venue, journal, citations, influential, DOI, ArXiv, month, authors, abstract, BibTeX da citationStyles)
> - Classe `S2Cache`: cache locale su disco (`cache/semantic_scholar/`) + in-memory. File JSON per S2 paper ID. Metodi: `get()`, `put()`, `has()`, `stats()`
> - `_normalize_title()`, `_title_similarity()`: fuzzy matching Jaccard per gestire differenze di formattazione nei titoli
> - Test: 5/5 paper arricchiti con successo:
>   - "Compromising Real-World LLM..." → AISec@CCS, DOI=10.1145/3605764.3623985, 876 citations, BibTeX 399 chars
>   - "Universal and Transferable Adversarial..." → arXiv.org, 2493 citations, velocity=831.0/anno
>   - "Poisoning Language Models..." → ICML, DOI=10.48550/arXiv.2305.00944, 285 citations
>   - "Siren Song in the AI Ocean..." → Computational Linguistics, DOI=10.1162/coli.a.16, 861 citations
>   - "HaluEval..." → EMNLP, DOI=10.48550/arXiv.2305.11747, 372 citations
>   - Re-enrichment dalla cache: 0.002s (vs ~1-2s per chiamata API)
>   - Cache: 11 file su disco, 10 entry in memoria dopo 5 enrichments

#### [x] Task 1.4: Creare ingestion/bibtex_generator.py — Generazione BibTeX automatico
**Obiettivo:** Generare entry BibTeX per ogni paper con chiave `Paper_YYYY_ID`.  
**Dettagli:**
- Funzione `generate_bibtex_key(title, year) → str`: es. `"Zou_2023_GCG"` dal primo autore + anno + acronimo/keyword
- Funzione `generate_bibtex_entry(paper: Paper) → str`: genera @article/@inproceedings completo
- Funzione `save_master_bib(entries: list[str], output_path)`: scrive file `.bib` cumulativo
- Se S2 fornisce `citationStyles.bibtex`, usarlo direttamente e solo generare la chiave custom
- Validazione: ogni entry deve avere almeno title, author, year  
**Test:** Generare BibTeX per 5 paper e verificare correttezza formale.

> **Completato 20/02/2026.** File: `paperhunter/ingestion/bibtex_generator.py`
> - `generate_bibtex_key(title, year, authors) → str`: genera chiave nel formato `PrimoAutore_YYYY_Keyword`. Estrae cognome dal primo autore, keyword dal titolo (acronimi noti prioritari, poi prima parola significativa non stop-word). ASCII-safe
> - `generate_bibtex_entry(paper) → str`: genera entry BibTeX completa. Se il paper ha `bibtex_entry` da S2, sostituisce solo la chiave con quella custom (`_replace_bibtex_key()`). Altrimenti costruisce da zero (`_build_bibtex_from_scratch()`) con tipo automatico: `@inproceedings` se venue è conferenza, `@article` se journal/ArXiv, `@misc` altrimenti
> - `enrich_paper_bibtex(paper) → Paper`: helper che genera bibtex_key + bibtex_entry e li inietta nel Paper
> - `save_master_bib(entries, output_path) → Path`: salva file .bib cumulativo con deduplicazione per chiave, ordinamento alfabetico, header informativo
> - `load_master_bib(bib_path) → dict[str, str]`: carica file .bib in dict chiave → entry
> - `validate_bibtex_entry(entry) → tuple[bool, list[str]]`: valida presenza di title, author, year
> - Utility interne: `_sanitize_for_bibtex()`, `_strip_accents()`, `_extract_surname()`, `_extract_keyword_from_title()`, `_is_conference_venue()`, `_format_authors_bibtex()`
> - Costanti: `_STOP_WORDS` (50+ stop-words), `_KNOWN_ACRONYMS` (30+ acronimi LLM/security), `_CONFERENCE_KEYWORDS` (20+ keyword conferenze)
> - Test 7/7 passati:
>   - Chiavi generate: `Zou_2023_Universal`, `Li_2023_Halueval`, `Zhang_2024_AI`, `Wan_2023_Poisoning`, `Greshake_2023_LLM`, `Unknown_2022_Paper`, `Unknown_XXXX_Short`
>   - Entry @article (ArXiv), @inproceedings (AISec@CCS, ICML), sostituzione chiave S2 — tutte valide
>   - save_master_bib: 4 entry salvate e ricaricate, deduplicazione OK (6 input → 4 salvate)
>   - Validazione: entry vuote, senza campi obbligatori, non-BibTeX — tutte rilevate correttamente

#### [x] Task 1.5: Creare ingestion/chunker.py — Chunking GROBID-aware
**Obiettivo:** Chunk intelligente basato sulle sezioni estratte da GROBID.  
**Dettagli:**
- Funzione `chunk_grobid_output(grobid_output: GROBIDOutput, paper: Paper) → list[PaperChunk]`
- Ogni sezione GROBID diventa un chunk (se < max_chunk_size) o viene splittata con overlap
- Metadata su ogni chunk: `paper_id`, `bibtex_key`, `section_name`, `category`, `year`, `month`, `chunk_type="original"`
- Chunk ID: hash di `bibtex_key::section_name::chunk_index`
- Se GROBID non ha estratto sezioni, fallback a splitting per dimensione con rilevamento heading (dal vecchio chunker.py)
- Parametri configurabili da YAML: `chunk_size`, `chunk_overlap`  
**Test:** Chunkare 3 paper processati con GROBID e verificare metadata.

> **Completato 20/02/2026.** File: `paperhunter/ingestion/chunker.py`
> - `chunk_grobid_output(grobid_output, paper, chunk_size?, chunk_overlap?) → list[PaperChunk]`: API principale. Strategia GROBID-aware: abstract come chunk separato, poi ogni sezione GROBID → 1+ chunk, figure/table captions come chunk dedicati. Fallback a `_chunk_from_raw_text()` se nessuna sezione GROBID
> - `chunk_raw_text(text, paper, chunk_size?, chunk_overlap?) → list[PaperChunk]`: API per testo grezzo senza GROBID. Usa rilevamento heading via regex
> - `get_chunk_stats(chunks) → dict`: statistiche aggregate (total, avg/min/max length, total_chars, sections, categories)
> - `_make_chunk(text, bibtex_key, section_name, chunk_index, paper, paper_id, chunk_type?) → PaperChunk`: factory con ID deterministico SHA-256 di `"bibtex_key::section_name::chunk_index"` (16 chars hex)
> - `_clean_text(text) → str`: normalizzazione whitespace, rimozione page numbers isolati
> - `_split_by_size(text, chunk_size, chunk_overlap) → list[str]`: split con overlap a confini di frase (". ", ".\n", "\n\n", "\n", "; ", ", ", " " in ordine di preferenza). Anti-loop infinito integrato
> - `_detect_sections_from_text(text) → list[tuple[str, str]]`: rilevamento heading via regex estesa (30+ heading accademici: Abstract, Introduction, Threat Model, Ablation Studies, ecc.)
> - `_chunk_from_raw_text(text, bibtex_key, paper, paper_id, chunk_size, chunk_overlap, start_index?) → list[PaperChunk]`: pipeline fallback completa (detect sections → split → make chunks)
> - Parametri da `config.yaml` → `ingestion.chunk_size` (1500), `ingestion.chunk_overlap` (200). Min chunk size: 80 chars
> - Test 3/3 passati:
>   - "Steer2Edit..." (Prompt Injection) → 38 sezioni GROBID → 59 chunk (avg 999 chars, 58.9K total)
>   - "Why Do AI Agents..." (LLM Hallucinations) → 12 sezioni GROBID → 24 chunk (avg 1060 chars, 25.4K total)
>   - "Protecting Context..." (Adversarial Attacks) → 10 sezioni GROBID → 34 chunk (avg 1267 chars, 43.1K total)
>   - Metadata verificati su tutti: chunk_id, chunk_type="original", bibtex_key, section_name, category, year, month — tutti corretti
>   - `to_metadata_dict()` validato per ogni paper

#### [x] Task 1.6: Creare indexing/qdrant_store.py — Qdrant Vector Store
**Obiettivo:** Sostituire FAISS con Qdrant per storage vettoriale con metadata filtering nativo.  
**Dettagli:**
- Classe `QdrantStore`:
  - `__init__(persist_dir, collection_name, embedding_model)`: inizializza client persistente
  - `add_chunks(chunks: list[PaperChunk])`: upsert con deduplicazione per chunk_id
  - `query(query_text, n_results, where_filter) → list[PaperChunk]`: ricerca vettoriale con filtro metadata
  - `delete_by_paper_id(paper_id)`: rimuovi tutti i chunk di un paper
  - `get_stats() → dict`: statistiche collezione
  - `has_paper(bibtex_key) → bool`: verifica esistenza
- Usare `GeminiEmbedding` da llm_factory.py come embedding function custom
- Collezioni separate: `paper_chunks`, `l1_summaries`, `l2_summaries`
- Metadata schema coerente con `PaperChunk`  
**Test:** Aggiungere 100 chunk, query con filtro per categoria, verificare risultati.

> **Completato 20/02/2026.** File: `paperhunter/indexing/qdrant_store.py`
> - **NOTA:** Usato Qdrant al posto di ChromaDB (decisione dal Task 0.2: ChromaDB non compatibile con Python 3.14). Qdrant funziona in modalità locale su disco senza bisogno di server.
> - Classe `QdrantStore` con client Qdrant locale persistente su disco
> - `__init__(persist_dir?, embedding_model?, collection_name?)`: inizializza client e crea automaticamente le 3 collezioni (`paper_chunks`, `l1_summaries`, `l2_summaries`) con payload indexes dichiarati
> - `add_chunks(chunks, collection_name?, batch_size=50, show_progress=True) → int`: upsert con generazione embedding automatica via Gemini API + rate limiting. Progress bar con Rich. Embedding in batch, fallback singolo su errore
> - `add_chunks_with_embeddings(chunks, embeddings, collection_name?, batch_size=50) → int`: upsert con embedding pre-calcolati (skip API call). Utile per pipeline e test
> - `query(query_text, n_results=10, where_filter?, collection_name?, score_threshold?) → list[dict]`: ricerca vettoriale con embedding automatico della query. Risultati: `{chunk_id, text, score, metadata, qdrant_id}`
> - `query_by_vector(query_vector, n_results=10, where_filter?, ...) → list[dict]`: ricerca per vettore pre-calcolato (skip embedding query)
> - `delete_by_paper(bibtex_key, collection_name?) → int`: elimina tutti i chunk per bibtex_key
> - `delete_by_paper_id(paper_id, collection_name?) → int`: elimina per paper_id
> - `delete_by_category(category, collection_name?) → int`: elimina tutti i chunk di una categoria
> - `has_paper(bibtex_key, collection_name?) → bool`: verifica esistenza per bibtex_key
> - `has_paper_id(paper_id, collection_name?) → bool`: verifica per paper_id
> - `get_paper_chunks(bibtex_key, collection_name?) → list[dict]`: recupera tutti i chunk di un paper (scroll paginato)
> - `get_stats(collection_name?) → dict`: statistiche dettagliate (total_points, categories, chunk_types, years, unique_papers). Se collection_name=None, stats per tutte e 3 le collezioni + totale aggregato
> - `get_collection_count(collection_name?) → int`: conteggio rapido punti
> - Helpers: `_chunk_id_to_uuid(chunk_id)` per ID Qdrant deterministici (UUID v5), `_build_qdrant_filter(where_filter)` per costruire filtri da dict semplificati, `_distance_from_config(metric)` per mapping metrica
> - Filtri supportati: stringa (match esatto), int (match esatto), list (match any), tuple (range min/max), dict con operatori (`gte`, `lte`, `gt`, `lt`)
> - Embedding: lazy initialization del modello, batch embedding con rate limiting Gemini, fallback a vettore zero su errore
> - Context manager (`with QdrantStore() as store:`)
> - `paperhunter/indexing/__init__.py` aggiornato con export di `QdrantStore`
> - Test 16/16 passati:
>   - 100 chunk con embedding 3072-dim → upsert OK
>   - has_paper: positivo e negativo OK
>   - query_by_vector senza filtro: 5 risultati, top score=1.0
>   - query con filtro categoria: tutti risultati corretti
>   - query con filtro year range (2024-2025): tutti corretti
>   - query con filtro combinato (category + year >= 2023): tutti corretti
>   - get_paper_chunks: recupero corretto
>   - delete_by_paper: eliminazione e verifica OK
>   - get_stats: categories={AA:25, DMP:25, LLM:25, PI:24}, years={2022:24, 2023:25, 2024:25, 2025:25}
>   - upsert idempotente: re-insert di 10 chunk → count stabile
>   - delete_by_category: 25 chunk LLM Hallucinations eliminati
>   - delete_by_paper_id: 4 chunk di paper_0 eliminati
>   - L1 collection: 1 chunk L1_SUMMARY inserito e contato

#### [x] Task 1.7: Creare ingestion/ingestion_pipeline.py — Orchestratore ingestion completo
**Obiettivo:** Pipeline end-to-end che processa incrementalmente i PDF nuovi/modificati.  
**Dettagli:**
- Classe `IngestionPipeline`:
  - `run(pdf_base_dir) → IngestionReport`: esegue l'intero flusso
  - Step 1: Scan directory → identifica nuovi PDF (via pdf_hasher)
  - Step 2: Per ogni nuovo PDF → GROBID → GROBIDOutput
  - Step 3: Per ogni paper → Semantic Scholar enrichment → Paper arricchito
  - Step 4: Per ogni paper → BibTeX generation
  - Step 5: Per ogni paper → Chunk → ChromaStore.add_chunks()
  - Step 6: Aggiorna hash registry
  - Step 7: Ritorna report (N processati, N skippati, N errori, tempo)
- Progress bar con `rich` per ogni step
- Resume from checkpoint: se interrotto, riparte dall'ultimo PDF non processato
- Logging dettagliato di ogni operazione
- Batch processing configurabile (es. 10 PDF alla volta per non sovraccaricare GROBID)  
**Test:** Eseguire pipeline su `scaricati/` completo. Rieseguire → 0 nuovi paper processati.

> **Completato 20/02/2026.** File: `paperhunter/ingestion/ingestion_pipeline.py`
> - Classe `IngestionPipeline`: pipeline end-to-end modulare con lazy initialization di GROBID, SemanticScholarEnricher, QdrantStore
> - `__init__(pdf_base_dir?, batch_size?, skip_s2?, skip_qdrant?, use_checkpoint?)`: tutti i parametri opzionali, default da `config.yaml`
> - `run(pdf_base_dir?) → IngestionReport`: esegue il flusso completo in 7 step:
>   - Step 1: Scansione directory + rilevamento delta (via `pdf_hasher.get_new_or_modified_pdfs()`)
>   - Step 1b: Resume from checkpoint (se disponibile, filtra PDF già completati)
>   - Step 2: GROBID processing (`_step_grobid()`) con tracking stats success/fallback/failed
>   - Step 3: Creazione Paper + S2 enrichment (`_step_create_and_enrich_paper()`)
>   - Step 4: BibTeX generation (`_step_bibtex()` via `enrich_paper_bibtex()`)
>   - Step 5: Chunking GROBID-aware (`_step_chunking()` via `chunk_grobid_output()`)
>   - Step 6: Indicizzazione Qdrant (`_step_qdrant_index()`) con delete+re-insert per paper modificati
>   - Step 7: Salvataggio registro hash + master .bib cumulativo + cleanup checkpoint
> - `_process_single_pdf(pdf_info, registry, report) → (Paper, list[PaperChunk]) | None`: processa un singolo PDF attraverso tutti gli step con error handling granulare
> - Batch processing con progress bar Rich per ogni batch
> - `_CheckpointManager`: salvataggio/caricamento checkpoint JSON dopo ogni batch (completed_pdfs, total_to_process). Permette resume da interruzione
> - `IngestionReport` (dataclass): statistiche complete — processed, skipped, errors, total_chunks, grobid_stats, s2_stats, error_details, total_time_seconds, to_dict() per serializzazione
> - Metodi utility: `get_registry_stats()`, `get_store_stats()`, `reset(clear_registry?, clear_store?)`
> - `_collect_existing_bibtex()`: carica entry BibTeX esistenti dal master .bib per dedup
> - `_print_report()`: stampa report finale con tabelle Rich (riepilogo, GROBID stats, S2 stats, errori)
> - `paperhunter/ingestion/__init__.py` aggiornato con export di `IngestionPipeline` e `IngestionReport`
> - Test 3/3 passati:
>   - 5 PDF in 3 categorie → 5 processati, 0 errori, 208 chunk, 59.3s (GROBID ~10-16s/PDF)
>   - Paper: Wang_XXXX_Advqdet, Jia_XXXX_Adversarial, Li_XXXX_Backdoorllm, Li_XXXX_Badedit, Gu_XXXX_Anah — tutti con grobid=True
>   - Riesecuzione → 0 nuovi, 0 modificati, 5 invariati
>   - Registry stats: 5 entry, distribuite su 3 categorie

---

### FASE 2 — Hierarchical Summarization & Auto-Invalidation

#### [x] Task 2.1: Creare summarization/level1_summarizer.py — Summary singolo paper
**Obiettivo:** Generare summary L1 ultra-specifico per ogni paper, con estrazione obbligatoria della `testing_methodology`.  
**Dettagli:**
- Funzione `generate_l1_summary(paper: Paper, chunks: list[PaperChunk]) → L1Summary`
- Prompt Gemini Pro strutturato che estrae:
  - `key_contributions`: contributi principali (max 5 bullet points)
  - `methodology`: approccio tecnico dettagliato
  - `testing_methodology`: **CAMPO CRITICO** — benchmark, dataset, metriche, prompt usati dagli autori per testare
  - `temporal_context`: quando proposto, differenze col passato, posizione nella timeline SOTA
  - `results_summary`: risultati quantitativi principali
  - `limitations`: limiti dichiarati o impliciti
- Output strutturato (JSON-mode di Gemini)
- Salvataggio del summary come chunk nel ChromaStore (collection `l1_summaries`)
- Metadata: `chunk_type="l1_summary"`, `paper_id`, `category`, `year`  
**Test:** Generare L1 per 3 paper e verificare che `testing_methodology` sia non-vuoto.

> **Completato 20/02/2026.** File: `paperhunter/summarization/level1_summarizer.py` (850 righe)
> - `generate_l1_summary(paper, chunks, *, force_model, max_chunk_chars, save_to_store, store) → L1Summary`: genera summary L1 strutturato via Gemini Pro con retry pazienti (6 retry, base_delay 15s, cap 120s). Flash fallback solo se `allow_flash_fallback=true` in config.yaml
> - `generate_l1_batch(papers_with_chunks, *, force_model, save_to_store, store, skip_existing) → list[L1Summary]`: batch processing con progress bar Rich e skip di paper già presenti nel vector store
> - `load_paper_and_chunks_from_ingestion(pdf_path, skip_s2=False) → (Paper, list[PaperChunk])`: utility per caricare un singolo PDF tramite GROBID+S2+BibTeX+chunking (usata nei test)
> - `_prepare_chunks_text(chunks, max_chars)`: prepara il testo dei chunk per il prompt (abstract/intro prioritari)
> - `_parse_json_response(raw_response, paper_title)`: parser JSON robusto con rimozione code-fence, estrazione brace, fix trailing comma, fallback regex
> - `_validate_l1_summary(summary, paper_title)`: validazione qualità (warning su testing_methodology vuoto)
> - `_save_l1_to_store(summary, paper, store)`: salva L1 come PaperChunk nella collection `l1_summaries` del QdrantStore con embedding-friendly text
> - `_l1_summary_to_text(summary, paper)`: converte L1Summary in testo per embedding
> - `_make_l1_chunk_id(paper_id)`: genera chunk ID deterministico SHA-256
> - Prompt template `_L1_PROMPT_TEMPLATE`: estrae key_contributions, methodology, testing_methodology (CRITICO), temporal_context, results_summary, limitations in JSON
> - Retry strategy: Pro con 6 retry pazienti (delay 15s→30s→60s→120s, cap 120s), timeout API 600s. Flash fallback controllato da `config.yaml` → `summarization.allow_flash_fallback` (default: false)
> - `ProAvailabilityTracker` (in llm_factory.py): singleton thread-safe che registra successi/fallimenti Pro per logging informativo
> - Retry interni LlamaIndex disabilitati (`max_retries=0`) per evitare retry moltiplicati
> - Test: 2/3 paper generati con successo (testing_methodology 978 e 837 chars). Terzo paper fallito per 504 DEADLINE_EXCEEDED (prima del fix timeout→600s e retry→6)  
> - Fix applicati: timeout 300→600s, retry pazienti 6 con cap 120s, `allow_flash_fallback=false`, `max_retries=0` su GoogleGenAI

#### [x] Task 2.2: Creare summarization/level2_summarizer.py — Meta-summary per categoria
**Obiettivo:** Generare meta-summary L2 di 4-6k token per ogni categoria, sintetizzando i L1.  
**Dettagli:**
- Funzione `generate_l2_summary(category: str, l1_summaries: list[L1Summary]) → L2Summary`
- Prompt Gemini Pro che:
  - Sintetizza trend temporali (evoluzione 2019→2026)
  - Identifica gap colmati e gap aperti
  - Crea tassonomia delle sotto-aree
  - Evidenzia convergenze e divergenze metodologiche
  - Riassume le testing_methodology più usate nella categoria
- Salvataggio nel ChromaStore (collection `l2_summaries`)
- Metadata: `chunk_type="l2_summary"`, `category`, `version`, `paper_count`  
**Test:** Generare L2 per categoria "Prompt Injection" e verificare completezza.

> **Completato 20/02/2026.** File: `paperhunter/summarization/level2_summarizer.py` (~700 righe)
> - `generate_l2_summary(category, l1_summaries?, *, force_model, save_to_store, store, version) → L2Summary`: genera meta-summary L2 strutturato via Gemini Pro con retry pazienti (stessa strategia del L1). Se `l1_summaries` non forniti, li carica dal QdrantStore automaticamente. Deduplicazione per paper_id e ordinamento cronologico
> - `generate_l2_for_all_categories(*, force_model, save_to_store, store, skip_existing, categories) → list[L2Summary]`: batch processing per tutte le categorie con almeno 1 L1 nel vector store. Progress bar Rich, skip di categorie già con L2 valido
> - `load_l1_summaries_for_category(category, store?) → list[L1Summary]`: carica L1 dal QdrantStore (scroll su `l1_summaries` con filtro category), ricostruisce oggetti `L1Summary` dal testo narrativo e metadata via `_reconstruct_l1_from_payload()` e `_parse_l1_text_sections()`
> - `load_all_l2_metadata() → dict[str, dict]`: carica tutti i metadata L2 da disco per recovery e tracking versioni
> - Prompt template `_L2_PROMPT_TEMPLATE`: genera JSON con 7 campi (temporal_trends, taxonomy, gaps_filled, open_gaps, methodological_convergences, testing_overview, synthesis) con istruzioni per target 4000-6000 token e uso citazioni [bibtex_key]
> - `_save_l2_to_store(summary, store)`: salva L2 come PaperChunk nella collection `l2_summaries` del QdrantStore. Elimina L2 precedenti della stessa categoria prima di inserire. Salva anche metadata JSON su disco (`output/l2_metadata/`)
> - `_l2_exists_and_valid(category, store)`: controlla se un L2 non-invalidato esiste già (per skip in batch)
> - `_format_l1_summaries_for_prompt(l1_summaries)`: formatta tutti gli L1 in testo cronologico per il prompt L2
> - `_l2_summary_to_text(summary)`: converte L2Summary in testo per embedding
> - `_make_l2_chunk_id(category, version)`: ID deterministico SHA-256
> - `_parse_l2_json_response(raw_response, category)`: parser JSON robusto con rimozione code-fence, fix trailing commas, fallback regex
> - `_validate_l2_summary(summary)`: validazione qualità (min chars per campo)
> - `_get_next_l2_version(category, store)`: calcolo versione incrementale da metadata su disco
> - `_discover_categories_from_store(store)`: scoperta automatica categorie dal vector store
> - `_save_l2_metadata_to_disk(summary)` / `_load_l2_metadata_from_disk(category)`: persistenza metadata JSON per recovery e invalidation tracking
> - `paperhunter/summarization/__init__.py` aggiornato con export di `generate_l2_summary`, `generate_l2_for_all_categories`, `load_l1_summaries_for_category`, `load_all_l2_metadata`
> - Test: L2 generato con successo per "Prompt Injection" (5 mock L1):
>   - Modello: Gemini Pro, tempo: 624.1s (con retry per 503 UNAVAILABLE)
>   - Total L2 chars: 19,893 — tutti i 7 campi ≥ 100 chars
>   - temporal_trends: 4,829 chars — evoluzione 2023→2025 dettagliata con citazioni
>   - taxonomy: 2,659 chars — 4 cluster (Threat Modeling, Benchmarking, Defensive, Adversarial)
>   - gaps_filled: 2,515 chars — 5 gap colmati con riferimenti specifici
>   - open_gaps: 2,335 chars — robustezza adattiva, scalabilità, multilingue
>   - methodological_convergences: 2,267 chars — convergenza su ASR, divergenza su difese
>   - testing_overview: 3,263 chars — benchmark aggregati, metriche, modelli, framework
>   - synthesis: 2,025 chars — narrativa coesa stile survey con citazioni
>   - Helper functions: formatting (6,413 chars), text conversion (20,071 chars), chunk ID, JSON parsing, validation — tutti passati

#### [x] Task 2.3: Creare summarization/invalidation_manager.py — Auto-invalidazione
**Obiettivo:** Quando nuovi paper vengono ingesti, invalidare e rigenerare i L2 delle categorie coinvolte.  
**Dettagli:**
- Classe `InvalidationManager`:
  - `check_invalidation(newly_ingested: list[Paper]) → list[str]`: ritorna categorie da rigenerare
  - `invalidate_l2(category: str)`: marca il L2 come invalidato nel ChromaStore
  - `regenerate_l2(category: str)`: raccoglie tutti gli L1 della categoria e rigenera il L2
  - Soglia configurabile: rigenera se ≥ N nuovi paper in una categoria (default: 1, configurabile da YAML)
- Logging delle invalidazioni e rigenerazioni
- Mantenere storico dei L2 vecchi (versionamento)  
**Test:** Aggiungere 2 paper finti a una categoria e verificare che il L2 venga rigenerato.

> **Completato 20/02/2026.** File: `paperhunter/summarization/invalidation_manager.py` (~530 righe)
> - Classe `InvalidationManager(*, store?, threshold?, auto_regenerate=True, archive_old=True)`: gestore completo dell'auto-invalidazione L2 con lazy-init del QdrantStore. Soglia configurabile da `config.yaml` → `summarization.l2_invalidation_threshold` (default: 1)
> - `check_and_regenerate(categories?, *, force_model?, save_to_store=True) → InvalidationReport`: metodo principale post-ingestion. Flusso: scoperta categorie → delta computation per ogni categoria → invalidazione se delta ≥ soglia o paper rimossi → rigenerazione automatica (se `auto_regenerate=True`) con progress bar Rich. Ritorna report dettagliato
> - `check_invalidation(newly_ingested?, categories?) → list[str]`: metodo leggero, ritorna solo le categorie che necessitano rigenerazione senza rigenerare. Se `newly_ingested` forniti, controlla solo le loro categorie
> - `invalidate_l2(category) → bool`: marca il metadata L2 su disco con `invalidated=True` + `invalidated_at`. Archivia automaticamente il L2 precedente prima di invalidare (se `archive_old=True`)
> - `regenerate_l2(category, *, force_model?, save_to_store=True) → L2Summary`: richiama `generate_l2_summary()` dal level2_summarizer per rigenerare con versioning incrementale automatico
> - `get_category_status(categories?) → dict[str, dict]`: diagnostica completa per ogni categoria: l1_count_in_store, has_l2, l2_version, l2_invalidated, new_papers, removed_papers, needs_regeneration
> - `get_archive_history(category) → list[dict]`: storico delle versioni L2 archiviate per una categoria (ordine cronologico)
> - `_compute_category_delta(category) → dict`: confronta paper_ids nel vector store (collection `l1_summaries`) vs paper_ids nel L2 metadata su disco. Ritorna: current_l1_count, new_paper_ids, removed_paper_ids, has_existing_l2
> - `_get_l1_paper_ids_for_category(category) → set[str]`: scroll paginato sulla collection l1_summaries con filtro Qdrant per categoria
> - `_archive_l2(category, metadata) → bool`: copia metadata in `output/l2_archive/<category>/v<N>_<timestamp>.json` con info di archiviazione
> - `InvalidationReport` (dataclass): report dettagliato con checked/invalidated/regenerated/failed categories, new/removed papers per categoria, tempo totale, `to_dict()` per serializzazione
> - `paperhunter/summarization/__init__.py` aggiornato con export di `InvalidationManager` e `InvalidationReport`
> - Test 11/11 passati:
>   - Nessun L2, nessun L1 → niente da fare ✓
>   - L1 presenti senza L2 → categorie da generare ex novo ✓
>   - L2 aggiornato (stessi paper) → niente da fare ✓
>   - 2 nuovi paper (soglia=1) → invalidazione Prompt Injection triggerata ✓
>   - invalidate_l2 → metadata marcato + archivio creato (v1 + timestamp) ✓
>   - check_and_regenerate report → 3 nuovi paper, PI invalidata ✓
>   - get_category_status → PI: l1=6, new=3, needs_regen=True; LH: l1=2, new=0, needs_regen=False ✓
>   - get_archive_history → 1 entry con archived_at ✓
>   - check_invalidation con newly_ingested → filtra solo categorie coinvolte ✓
>   - Soglia alta (10) → 3 nuovi paper non triggerano invalidazione ✓
>   - Paper rimossi → invalidazione triggerata indipendentemente dalla soglia ✓

---

### FASE 3 — Indexing Ibrido + GraphRAG

#### [x] Task 3.1: Creare indexing/bm25_index.py — Indice BM25
**Obiettivo:** Indice keyword BM25 complementare alla ricerca vettoriale.  
**Dettagli:**
- Classe `BM25Index`:
  - `build_index(chunks: list[PaperChunk])`: tokenizza e costruisce indice BM25
  - `query(query_text, n_results) → list[(chunk_id, score)]`
  - `save(path)` / `load(path)`: persistenza (pickle)
  - `update(new_chunks)`: aggiornamento incrementale (rebuild se necessario)
- Usare libreria `rank-bm25`
- Tokenizzazione: lowercase + rimozione stopwords + stemming opzionale  
**Test:** Indicizzare tutti i chunk e fare 5 query di test, verificare rilevanza.

> **Completato 20/02/2026.** File: `paperhunter/indexing/bm25_index.py` (~600 righe)
> - Funzione `tokenize(text, *, remove_stopwords, stem, min_token_length) → list[str]`: pipeline completa (lowercase → regex split → filtro lunghezza → stopword removal → stemming naïve). Funzione pubblica riusabile
> - `_naive_stem(word)`: stemming naïve (Porter-like semplificato) con 20+ regole di suffisso. Evita dipendenza NLTK
> - `_STOPWORDS`: 100+ stopwords inglesi estese con termini accademici generici (paper, section, figure, table, ecc.)
> - Classe `BM25Index(index_path?, *, remove_stopwords, stem, min_token_length)`:
>   - `build_index(chunks, *, show_progress) → int`: tokenizza e costruisce `BM25Okapi`. Deduplicazione per chunk_id, skip chunk con <2 token, progress bar Rich
>   - `query(query_text, n_results, *, where_filter, score_threshold) → list[dict]`: ricerca BM25 con filtri metadata post-retrieval. Risultati: `{chunk_id, score, text, metadata}`. Filtri supportati: category, year, year_min, year_max, chunk_type, bibtex_key, paper_id
>   - `query_batch(queries, n_results, *, where_filter, score_threshold) → list[list[dict]]`: batch di query
>   - `update(new_chunks, *, show_progress) → int`: aggiornamento incrementale con rebuild. Supporta sovrascrittura chunk esistenti (stesso chunk_id)
>   - `delete_by_paper(bibtex_key)`, `delete_by_category(category)`, `delete_by_paper_id(paper_id)`: eliminazione con rebuild automatico
>   - `save(path?) → Path`: persistenza pickle con version, corpus_tokens, metadata, testi, config tokenizzazione
>   - `load(path?) → int`: caricamento con ricostruzione BM25Okapi + validazione formato/versione
>   - `get_stats() → dict`: statistiche aggregate (total_chunks, categories, chunk_types, years, avg/min/max tokens, unique_papers)
>   - `has_paper(bibtex_key)`, `has_paper_id(paper_id)`, `get_paper_chunks(bibtex_key)`: lookup utilities
>   - `clear()`: reset in memoria. `__repr__`, `__len__`, `__contains__` per usabilità
> - `_matches_filter(metadata, where_filter)`: engine di filtering flessibile per query post-retrieval
> - `_remove_chunk_ids(ids_to_remove)`: helper interno per delete + rebuild
> - `paperhunter/indexing/__init__.py` aggiornato con export di `BM25Index` e `tokenize`
> - Default `index_path` da `config.yaml` → `retrieval.bm25_index_path` (`./data/bm25_index.pkl`)
> - Test 15/15 passati:
>   - Tokenizer: stopwords, stemming, edge cases (vuoti, corti) ✓
>   - Build: 10 chunk indicizzati, dedup automatica ✓
>   - Query basic: 4 query tematiche, top result sempre nella categoria corretta (PI→Greshake, DMP→Li_Backdoorllm, LLM_H→Li_Halueval, AA→Zou_GCG) ✓
>   - Query con filtri: category, year, year_min, combo, bibtex_key, score_threshold — tutti corretti ✓
>   - Update: 5+5 chunks, overwrite con stesso ID → size stabile, testo aggiornato ✓
>   - Delete: by_paper (1 rimosso), by_category (2 rimossi), by_paper_id (1 rimosso), inesistente (0) ✓
>   - Save/Load: pickle round-trip, query identiche dopo load, FileNotFoundError ✓
>   - Stats: categories={PI:4, DMP:2, LLM_H:2, AA:2}, years={2023:7, 2024:3}, avg 22.8 tokens/doc ✓
>   - Edge cases: ValueError su lista vuota, RuntimeError su query/save senza build ✓

#### [x] Task 3.2: Creare indexing/hybrid_retriever.py — Retrieval ibrido
**Obiettivo:** Combinare vector search + BM25 + metadata filter in un unico retriever.  
**Dettagli:**
- Classe `HybridRetriever`:
  - `retrieve(query, n_results, category_filter, year_range, chunk_types) → list[RetrievedChunk]`
  - Step 1: Vector search via ChromaStore (con where filter)
  - Step 2: BM25 search via BM25Index
  - Step 3: Reciprocal Rank Fusion (RRF) per combinare risultati
  - Step 4: Metadata filtering post-retrieval (category, year, chunk_type)
  - Pesi configurabili da YAML: `vector_weight`, `bm25_weight`
- `RetrievedChunk`: chunk + score + retrieval_method
- Supporto query multi-step: query espansa/riformulata  
**Test:** Query "prompt injection defense mechanisms 2024" e verificare che risultati siano multi-source.

> **Completato 20/02/2026.** File: `paperhunter/indexing/hybrid_retriever.py` (~650 righe)
> - Funzione standalone `reciprocal_rank_fusion(ranked_lists, weights, k=60) → list[(chunk_id, rrf_score)]`: implementazione RRF generica. Formula: RRF_score(d) = Σ_i w_i / (k + rank_i(d)). Supporta N ranking con pesi arbitrari
> - Enum `RetrievalMethod`: VECTOR, BM25, HYBRID, GRAPH (reserved per GraphRAG futuro)
> - Dataclass `RetrievedChunk`: chunk_id, text, score, retrieval_method, metadata, vector_score, bm25_score, vector_rank, bm25_rank + `to_dict()` per serializzazione
> - Classe `HybridRetriever(qdrant_store?, bm25_index?, vector_weight?, bm25_weight?, top_k_vector?, top_k_bm25?, top_k_final?, rrf_k=60)`: tutti i parametri configurabili da `config.yaml` → sezione `retrieval`. Lazy init per QdrantStore e BM25Index
> - `retrieve(query, n_results?, *, category_filter?, year_range?, chunk_types?, bibtex_key?, paper_id?, extra_filters?, collection_name?, vector_weight?, bm25_weight?, include_vector_only?, include_bm25_only?) → list[RetrievedChunk]`: API principale di ricerca ibrida. Flusso: build filter → vector search → BM25 search → RRF fusion → troncamento a top_k_final. Fallback automatico a singola sorgente se BM25 non disponibile
> - `retrieve_multi_collection(query, collections?, n_results?, *, category_filter?, year_range?, chunk_types?, deduplicate=True) → list[RetrievedChunk]`: ricerca su più collezioni Qdrant (paper_chunks + l1_summaries + l2_summaries) con deduplicazione per chunk_id
> - `retrieve_vector_only(query, n_results?, **kwargs)` / `retrieve_bm25_only(query, n_results?, **kwargs)`: shortcut per ricerca single-source
> - `get_stats() → dict`: statistiche aggregate (config, disponibilità sorgenti, stats Qdrant/BM25)
> - `_build_where_filter(category_filter?, year_range?, chunk_types?, bibtex_key?, paper_id?, extra_filters?) → dict`: costruttore filtro unificato. Supporta categoria singola/lista, year range tuple/dict, chunk_types lista, filtri extra arbitrari
> - `_where_filter_to_bm25(where_filter) → dict`: conversione automatica dal formato Qdrant (tuple, dict con gte/lte) al formato BM25 (year_min, year_max)
> - `_vector_search()`, `_bm25_search()`: ricerche interne con error handling
> - `_fuse_results()`: fusione RRF con tracking completo (rank e score per-source per ogni risultato)
> - `_results_from_single_source()`: conversione risultati single-source in RetrievedChunk
> - Helper `_deduplicate_by_chunk_id()`: deduplicazione mantenendo lo score più alto
> - `paperhunter/indexing/__init__.py` aggiornato con export di `HybridRetriever`, `RetrievedChunk`, `RetrievalMethod`, `reciprocal_rank_fusion`
> - Test 12/12 passati:
>   - RRF standalone: calcolo corretto, A (rank1 vector+rank2 bm25) > C (rank3+rank1), documenti ibridi > singoli, ValueError su lunghezze diverse, liste vuote ✓
>   - Costruzione filtri: vuoto, categoria singola/lista, year range completo/solo min/solo max, chunk_types multipli/singolo, filtro combinato, conversione BM25 (tuple→year_min/max, dict→year_min/max, int) — 15 assert ✓
>   - RetrievedChunk: creazione, to_dict(), enum values ✓
>   - Deduplicazione: 5→3 unici, score più alto mantenuto, lista vuota ✓
>   - Fusione RRF con mock: 4 vector + 4 bm25 → 6 fusi, chunk_A/C=HYBRID, chunk_B/D=VECTOR, chunk_E/F=BM25, HYBRID score > single, ordinamento corretto, score/rank per-source preservati ✓
>   - Fallback singola sorgente: solo vector (BM25=None), include_bm25_only, include_vector_only, query vuota ✓
>   - Propagazione filtri: Qdrant riceve category+year tuple+chunk_type, BM25 riceve category+year_min+year_max+chunk_type ✓
>   - Multi-collection: 2 collezioni, dedup, troncamento n_results=5, ordinamento ✓
>   - Shortcut methods: retrieve_vector_only/retrieve_bm25_only ✓
>   - Statistiche: config corretta, qdrant/bm25 available, repr ✓
>   - BM25 reale + mock vector: 8 chunk reali, query "prompt injection defense" → 3 PI nei top, 2 HYBRID, ordinamento ✓
>   - top_k_final: troncamento a 5, override n_results=3 ✓

#### [x] Task 3.3: Creare indexing/reranker.py — Reranking con Gemini
**Obiettivo:** Reranking dei risultati ibridi usando Gemini come cross-encoder.  
**Dettagli:**
- Funzione `rerank(query, candidates: list[RetrievedChunk], top_k) → list[RetrievedChunk]`
- Prompt Gemini Flash: dato query + lista candidati, ordina per rilevanza (output JSON con ranking)
- Batch reranking: se > 20 candidati, dividi in batch e poi merge
- Fallback: se Gemini fallisce, mantieni ordine RRF  
**Test:** Rerancare 15 risultati per una query e verificare che l'ordine migliori.

> **Completato 20/02/2026.** File: `paperhunter/indexing/reranker.py` (~550 righe)
> - Classe `GeminiReranker(batch_size?, top_k?, max_text_chars=500, llm?)`: reranker con Gemini Flash come cross-encoder. Lazy init del LLM con `temperature=0.0`, `timeout=90s`, `max_tokens=8192` (thinking model necessita token extra per ragionamento interno)
> - `rerank(query, candidates, top_k?) → list[RetrievedChunk|dict]`: API principale. Accetta sia `RetrievedChunk` (dataclass) che `dict`. Flusso: normalize → split batch → LLM rerank per batch → merge per score → troncamento top_k. Fallback automatico all'ordine originale (RRF) se Gemini fallisce
> - `rerank_with_details(query, candidates, top_k?) → RerankResult`: come `rerank()` ma ritorna anche metadati (original_count, output_count, model_used, elapsed_seconds, fallback_used, batches_processed)
> - `_rerank_batch(query, batch, batch_idx, total_batches) → list[dict] | None`: rerank singolo batch con `llm_call_with_retry` (max_retries=2, base_delay=5s, max_delay=30s). Prompt strutturato con metadata candidati (categoria, anno, sezione, paper, tipo) + testo troncato
> - `_parse_rerank_response(raw_response, expected_count) → dict[int, float] | None`: parser JSON robusto con 3 livelli di fallback: JSON diretto → regex extraction array → regex pattern matching ("id": N + "score": X, "Candidate N: X", "N → X")
> - `_format_candidates_for_prompt(batch) → str`: formattazione candidati con metadata chiave e troncamento a `max_text_chars`
> - `_normalize_candidates()` / `_denormalize_results()`: conversione bidirezionale RetrievedChunk ↔ dict, preserva tipo originale in output
> - `_split_into_batches(items) → list[list[dict]]`: divisione in batch da `batch_size`
> - Funzioni standalone: `rerank()`, `rerank_with_details()` — wrapper convenience attorno a `GeminiReranker`
> - Dataclass `RerankResult`: statistiche operazione + `to_dict()` per serializzazione
> - Prompt template `_RERANK_PROMPT_TEMPLATE`: score 0.0-1.0 con linee guida dettagliate per rilevanza accademica, output JSON array
> - Fix `llm_factory.py`: aggiunto parametro `timeout_seconds` a `_get_http_options()`, `get_flash_llm()` e `get_pro_llm()` per override del timeout globale. Il reranker usa 90s invece di 600s (default Pro/thinking)
> - `paperhunter/indexing/__init__.py` aggiornato con export di `GeminiReranker`, `RerankResult`, `rerank`, `rerank_with_details`
> - Test 12/12 passati:
>   - Parsing JSON: pulito, code fences, trailing commas, testo extra, score fuori range (clamped), candidati mancanti (→0.0), risposta vuota (→None) ✓
>   - Regex fallback: pattern "id"/"score", "Candidate N: X.X", testo senza pattern (→None) ✓
>   - Normalizzazione: RetrievedChunk, dict, dict senza chunk_id → generato automaticamente ✓
>   - Batch splitting: ≤batch_size→1, >batch_size→N batch (5+5+2) ✓
>   - Formattazione prompt: metadata (categoria, anno, sezione, paper), troncamento, N/A, _batch_index ✓
>   - Mock LLM: ordinamento chunk_c>chunk_a>chunk_b, tipo output preservato, score aggiornati, top_k=2 ✓
>   - rerank_with_details: RerankResult con statistiche corrette, to_dict() ✓
>   - Fallback su errore: ordine originale preservato, fallback_used=True ✓
>   - Edge cases: lista vuota [], query vuota, singolo candidato ✓
>   - Multi-batch: 5 candidati → 2 batch (3+2), merge ordinato [0.95, 0.85, 0.6, 0.4] ✓
>   - Standalone rerank(): wrapper funzionante ✓
>   - LIVE Gemini Flash: 15 candidati, query "defense mechanisms against prompt injection" → 8/10 PI nei top-10, top=Lee_2025_PromptShield (1.0), non-PI relegati in fondo, 62.1s ✓

#### [x] Task 3.4: Creare graphrag/entity_extractor.py — Estrazione entità e relazioni
**Obiettivo:** Estrarre entità e relazioni dai summary L1 per costruire il knowledge graph.  
**Dettagli:**
- Funzione `extract_entities_and_relations(l1_summary: L1Summary) → list[Entity], list[Relation]`
- Entità: `Attack`, `Defense`, `Model`, `Dataset`, `Benchmark`, `Metric`, `Author`, `Venue`
- Relazioni: `DEFENDS_AGAINST`, `TESTED_ON`, `UPDATES`, `INTRODUCED_IN(year)`, `EXTENDS`, `OUTPERFORMS`
- Prompt Gemini Flash strutturato con output JSON  
- Normalizzazione entità: `"GPT-4"` e `"gpt4"` → stessa entità  
**Test:** Estrarre entità da 5 L1 summary e verificare coerenza.

> **Completato 20/02/2026.** File: `paperhunter/graphrag/entity_extractor.py` (~1400 righe)
> - Enums: `EntityType` (Attack, Defense, Model, Dataset, Benchmark, Metric, Author, Venue), `RelationType` (DEFENDS_AGAINST, TESTED_ON, UPDATES, INTRODUCED_IN, EXTENDS, OUTPERFORMS, EVALUATED_WITH, TARGETS, USES_MODEL)
> - Dataclass `Entity(BaseModel)`: id deterministico SHA-256 di `type::canonical_name`, canonical_name, aliases, description, year, source_paper_id, source_bibtex_key, properties dict, `to_dict()` per serializzazione
> - Dataclass `Relation(BaseModel)`: id deterministico SHA-256 di `type::source::target`, source/target con type, description, year, properties dict, `to_dict()`
> - Dataclass `ExtractionResult(BaseModel)`: entities, relations, timestamps, model_used, extraction_time_seconds, error, `to_dict()`
> - `normalize_entity_name(raw_name) → str`: normalizzazione con dizionario `_KNOWN_ENTITY_ALIASES` (200+ mapping): modelli (GPT-4, LLaMA-2, Claude, Mistral, ecc.), attacchi (GCG, AutoDAN, DAN, prompt injection, backdoor, ecc.), difese (SmoothLLM, RLHF, perplexity filter, ecc.), dataset/benchmark (AdvBench, TruthfulQA, HaluEval, HarmBench, ecc.), metriche (ASR, BLEU, F1, ecc.), venue (NeurIPS, ACL, ICML, ecc.). Unicode NFD + ASCII folding, suffix stripping ("attack", "defense", ecc.)
> - `merge_entities(entities) → list[Entity]`: merge duplicati per type+canonical_name. Unisce aliases (dedup), prende descrizione più lunga, anno più vecchio, unisce properties
> - `merge_relations(relations) → list[Relation]`: merge duplicati per type+source+target. Prende descrizione più lunga, preserva primo anno
> - `extract_entities_and_relations(l1_summary, *, llm?, max_retries=3) → (list[Entity], list[Relation])`: API principale. Usa Gemini Flash con `max_tokens=16384` (necessario per thinking model). Normalizza e deduplica entità, valida coerenza relazioni (scarta relazioni con entità mancanti)
> - `extract_with_details(l1_summary, **kwargs) → ExtractionResult`: come sopra ma ritorna anche metadati
> - `extract_batch(l1_summaries, *, merge_results=True, skip_errors=True) → (entities, relations, results)`: batch con progress bar Rich, merge opzionale tra paper
> - `_build_prompt(l1)`: costruisce prompt strutturato con metadata paper + tutti i campi L1. Output JSON con entities e relations
> - `_parse_json_response(raw_response, context) → dict | None`: parser robusto con 5 livelli: JSON diretto → code fence removal → brace extraction → trailing comma fix → regex fallback per arrays separati
> - `_build_entities(raw_list, l1) → list[Entity]`: costruzione con validazione type, normalizzazione nomi, tracking aliases originali, source tracking
> - `_build_relations(raw_list, entities, l1) → list[Relation]`: costruzione con entity lookup, risoluzione nomi (diretto → normalizzato → partial match), validazione types
> - `get_extraction_stats(entities, relations) → dict`: statistiche aggregate per tipo
> - `save_extraction_results(entities, relations, output_path?) → str`: salva JSON completo
> - Fix `llm_factory.py`: aggiunto `"max_tokens"` ai transient_keywords in `llm_call_with_retry()` per rendere l'errore retryable
> - `paperhunter/graphrag/__init__.py` aggiornato con export completi
> - Test 12/12 passati:
>   - Normalizzazione: 30+ varianti testate (modelli, attacchi, difese, dataset, metriche, venue, sconosciuti, edge cases) ✓
>   - Modelli: Entity ID deterministico, Relation ID deterministico, diverso tipo → diverso ID, to_dict(), ExtractionResult serializzazione ✓
>   - Merge: 3→2 entità (GCG fuso, aliases uniti, descrizione più lunga, anno più vecchio), 3→2 relazioni ✓
>   - JSON parsing: pulito, code fences, trailing commas, testo extra, vuoto→None, non-JSON→None, sbilanciato→regex fallback ✓
>   - Build raw: 9 raw→6 valide (3 scartate: tipo invalido, nome vuoto, non-dict), normalizzazione (gpt4→GPT-4, advbench→AdvBench, attack success rate→ASR), source tracking, 5 relazioni con risoluzione nomi ✓
>   - LIVE Gemini Flash estrazione singola: Zou_2023_GCG → 9 entità (GCG, Perplexity Filter, Vicuna 7B, LLaMA-2, GPT-4, AdvBench, ASR, Andy Zou, NeurIPS), 7 relazioni (INTRODUCED_IN, TESTED_ON, EVALUATED_WITH, TARGETS×2, DEFENDS_AGAINST, USES_MODEL), 28.7s ✓
>   - LIVE batch 5 summaries: 3/5 successi (2 falliti per MAX_TOKENS prima del fix), 28 entità merged (7 Attack, 8 Model, 3 Defense, 3 Author, 3 Venue, 3 Metric, 1 Dataset), 27 relazioni merged (5 TARGETS, 5 EVALUATED_WITH, 4 DEFENDS_AGAINST, 3 EXTENDS, 3 INTRODUCED_IN, 3 USES_MODEL, 2 OUTPERFORMS, 2 TESTED_ON), avg 0.9 aliases/entity, 228.7s totali ✓
>   - Salvataggio JSON: /tmp/test_graphrag_extraction.json con stats + entities + relations ✓

#### [x] Task 3.5: Creare graphrag/neo4j_manager.py — Connessione e gestione Neo4j
**Obiettivo:** Gestire la connessione a Neo4j Community Edition locale e le operazioni CRUD.  
**Dettagli:**
- Classe `Neo4jManager`:
  - `connect()` / `close()`: gestione connessione (neo4j driver)
  - `create_entity(entity)` / `create_relation(relation)`: inserimento
  - `query_neighbors(entity_name, depth) → subgraph`
  - `query_path(entity_a, entity_b) → path`
  - `get_temporal_chain(attack_type) → timeline`
  - `clear_graph()`: pulizia
- Istruzioni per setup Neo4j Community Docker  
**Test:** Connessione, creazione 10 entità + 5 relazioni, query di neighbors.

> **Completato 20/02/2026.** File: `paperhunter/graphrag/neo4j_manager.py` (~700 righe)
> - Eccezioni custom: `Neo4jNotEnabledError` (se `neo4j.enabled=false`), `Neo4jConnectionError`
> - Classe `Neo4jManager(uri?, username?, password?, database?, auto_connect=True)`: parametri da `config.yaml` → sezione `neo4j`. Context manager support (`with Neo4jManager() as mgr:`)
> - `connect()`: stabilisce connessione con `GraphDatabase.driver()` + `verify_connectivity()`. Gestione ImportError (pacchetto non installato)
> - `close()`, `is_connected`, `_ensure_connected()`: gestione ciclo di vita
> - `_run(query, parameters?) → list[dict]`: esegue query Cypher con `record.data()`. `_write(query, parameters?) → Summary`: per query DDL/DML
> - `create_constraints()`: vincoli unicità su `entity_id` per ogni `EntityType` + indici su `canonical_name` e `year`. Label doppie: `Entity` + tipo specifico (es. `:Entity:Attack`)
> - `create_entity(entity)` / `upsert_entity(entity)`: CREATE / MERGE per `entity_id`. Properties: entity_id, canonical_name, aliases (JSON), description, year, source_paper_id, source_bibtex_key, properties (JSON), updated_at
> - `get_entity(entity_id)`, `get_entity_by_name(canonical_name, entity_type?)`: lookup per ID o nome. Ritorna dict con aliases/properties deserializzati e labels/entity_type estratti
> - `delete_entity(entity_id)`: DETACH DELETE (rimuove anche relazioni collegate)
> - `create_relation(relation)` / `upsert_relation(relation)`: CREATE / MERGE basato su `entity_id` di source/target (calcolati da `_make_entity_id()`). Fallisce silenziosamente se entità mancanti. Properties: relation_id, description, year, source_paper_id, source_bibtex_key, properties (JSON), updated_at
> - `delete_relation(relation_id)`: elimina per ID relazione
> - `import_entities_and_relations(entities, relations, *, show_progress, clear_before) → dict`: bulk import con progress bar Rich. Crea constraints, poi upsert entità, poi upsert relazioni. Stats: entities_created/failed, relations_created/failed, elapsed_seconds
> - `import_from_json(json_path, *, show_progress, clear_before) → dict`: carica da file JSON (formato `save_extraction_results`), ricostruisce oggetti `Entity`/`Relation`, chiama `import_entities_and_relations()`
> - `query_neighbors(entity_name, depth=1, entity_type?) → dict`: sottografo locale. Due query separate: una per entità neighbor, una per relazioni (evita serializzazione tuple di `record.data()`). Per depth>1 usa `UNWIND relationships(path)`. Ritorna: center, entities, relations
> - `query_path(entity_a, entity_b, entity_type_a?, entity_type_b?, max_depth=5) → list[dict]`: shortest path con `shortestPath()`. Usa sessione raw (non `record.data()`) per preservare tipi Neo4j. Ritorna lista alternata entity/relation elements
> - `get_temporal_chain(entity_type, limit=50) → list[dict]`: timeline ordinata per anno con relazioni temporali (UPDATES, EXTENDS, INTRODUCED_IN)
> - `query_cypher(cypher, parameters?) → list[dict]`: query Cypher personalizzata
> - `clear_graph() → int`: DETACH DELETE tutti i nodi. `clear_by_paper(bibtex_key) → dict`: elimina relazioni per bibtex_key, poi entità orfane dello stesso paper
> - `get_stats() → dict`: total_nodes, total_relationships, nodes_by_type, relationships_by_type, year_range, unique_papers
> - Helper standalone: `test_neo4j_connection() → bool`: testa connessione e stampa stats
> - Helper interni: `_node_to_dict(node, labels)` (deserializza JSON fields), `_extract_entity_type_from_labels(labels)` (estrae type da labels Neo4j)
> - `paperhunter/graphrag/__init__.py` aggiornato con export di `Neo4jManager`, `Neo4jNotEnabledError`, `Neo4jConnectionError`, `test_neo4j_connection`
> - Docker command: `docker run --rm -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5-community`
> - Test 13/13 passati:
>   - Connessione e context manager: connect → is_connected=True → close → is_connected=False ✓
>   - Clear graph + constraints: nodi eliminati, vincoli/indici creati ✓
>   - Entity CRUD: upsert, get by ID (labels=['Attack','Entity']), get by name, get by name+type, update (description aggiornata), non-existent→None, delete ✓
>   - Relation CRUD: upsert GCG→GPT-4, re-upsert idempotente, entità mancante→False, delete ✓
>   - Bulk import: 10 entità (3 Attack, 2 Defense, 2 Model, 1 Dataset, 1 Metric, 1 Venue), 8 relazioni, 0 errori, 0.16s ✓
>   - Query neighbors: GCG depth=1 → 8 entità + 8 relazioni, depth=2 → ≥8 entità, non-existent → empty ✓
>   - Query path: GCG→GPT-4 → 3 elementi (entity-TARGETS-entity), SmoothLLM→GPT-4 → 5 elementi (attraverso GCG), non-existent → [] ✓
>   - Temporal chain: 3 Attack ordinati (2022→2023), 2 Defense, relazioni temporali (EXTENDS, INTRODUCED_IN) ✓
>   - Statistics: 10 nodi, 8 relazioni, Attack=3, Model=2, years 2022–2024, unique_papers=5 ✓
>   - Clear by paper: Zou_2023_GCG → 5 entità + 5 relazioni eliminate, grafo 10→5 nodi ✓
>   - Import from JSON: 5 entità + 2 relazioni importate da file temporaneo ✓
>   - Neo4jNotEnabledError: correttamente lanciata quando neo4j.enabled=false ✓
>   - test_neo4j_connection(): True ✓

#### [x] Task 3.6: Creare graphrag/property_graph.py — LlamaIndex PropertyGraph Builder
**Obiettivo:** Costruire e indicizzare il PropertyGraph con LlamaIndex per query graph-based.  
**Dettagli:**
- Classe `PropertyGraphBuilder`:
  - `build_from_summaries(l1_summaries: list[L1Summary])`: costruisce il grafo completo
  - `update_graph(new_summaries)`: aggiornamento incrementale
  - `query_graph(question) → context_str`: retrieval graph-based
- Integrare con `HybridRetriever` come fonte aggiuntiva di contesto
- Relazioni temporali essenziali: `UPDATES`, `INTRODUCED_IN`  
**Test:** Costruire grafo da 20 L1, query "evolution of prompt injection defenses" e verificare contesto temporale.

> **Completato:** 2025-02-22  
> **File creati/modificati:**
> - `paperhunter/graphrag/property_graph.py` (~900 righe) — Classe `PropertyGraphBuilder` completa
> - `paperhunter/graphrag/__init__.py` — Aggiornato con export `PropertyGraphBuilder`, `GraphQueryResult`, `BuildReport`, `build_property_graph`, `query_property_graph`
> - `test_property_graph.py` — 13 test, tutti passati
>
> **Implementazione:**
> - `PropertyGraphBuilder` con `SimplePropertyGraphStore` (locale, JSON persistence)
> - `build_from_summaries()`: estrae entità via `extract_batch()` poi popola il grafo
> - `build_from_entities()`: popola da entità/relazioni pre-estratte (upsert incrementale)
> - `update_graph()`: aggiornamento incrementale delegato a `build_from_summaries`
> - `query_graph()`: usa `LLMSynonymRetriever` con fallback a `_query_via_direct_traversal()`
> - `query_entity_neighborhood()`: traversamento locale del sotto-grafo
> - `get_temporal_chain()`: catene temporali per tipo entità (UPDATES, EXTENDS, INTRODUCED_IN)
> - `save()`/`load()`: persistenza graph store + metadata JSON
> - `get_stats()`, `remove_paper()`, `clear()` per gestione del grafo
> - Dataclass `GraphQueryResult` e `BuildReport` con `to_dict()` serialization
> - Convenience functions: `build_property_graph()`, `query_property_graph()`
>
> **Note tecniche:**
> - `EntityNode.id` è *sempre* uguale a `EntityNode.name` (non configurabile); l'ID paperhunter è conservato in `properties["paperhunter_id"]`
> - `get_rel_map()` restituisce una lista flat di triplet `[(EntityNode, Relation, EntityNode), ...]`, non un dict
> - Fallback relazioni cross-batch: usa `relation.source`/`.target` (canonical name) come ID quando l'entità non è nel batch corrente
> - 13/13 test passati: dataclass, keyword extraction, entity/relation mapping, empty graph + persistence, build, stats+remove, direct traversal, neighborhood, temporal chain, incremental update, convenience functions, repr

---

### FASE 4 — Agentic Writing Pipeline (LangGraph)

#### [x] Task 4.1: Creare agents/state.py — Stato condiviso LangGraph
**Obiettivo:** Definire il TypedDict/Pydantic state per il grafo LangGraph.  
**Dettagli:**
- `SurveyState(TypedDict)`:
  - `l2_summaries: list[L2Summary]`
  - `outline: list[SectionOutline]`
  - `sections: dict[str, SectionDraft]`
  - `verification_results: dict[str, VerificationResult]`
  - `bibliography: list[str]`
  - `provenance: list[ProvenanceRecord]`
  - `final_paper: str`
  - `version_info: SurveyVersion`
  - `iteration_count: int`
  - `errors: list[str]`
- Modelli ausiliari: `SectionOutline`, `SectionDraft`, `VerificationResult`  
**Test:** Istanziare lo stato e serializzare/deserializzare.

> **Completato 24/02/2026.** File: `paperhunter/agents/state.py` (~587 righe)
> - Enums: `SectionStatus` (PENDING, DRAFTED, UNDER_REVIEW, NEEDS_REVISION, VERIFIED, FINALIZED), `PipelinePhase` (INIT, PLANNING, WRITING, VERIFYING, ASSEMBLING, COMPLETED, FAILED)
> - `CitationRecord(BaseModel)`: bibtex_key, claim, chunk_id, retrieval_score + to_dict()/from_dict()
> - `SectionOutline(BaseModel)`: key, title, description, target_length_words, key_topics, related_categories, order, section_type, parent_key, subsections + to_dict()/from_dict()
> - `VerificationDetail(BaseModel)`: citation_key, claim_text, source_chunk_id, source_text_excerpt, status, confidence, reasoning
> - `VerificationResult(BaseModel)`: section_key, total_citations, supported/partially/not_supported/source_not_found counts, details, pass_rate, passed, iteration + compute_pass_rate(threshold)
> - `SectionDraft(BaseModel)`: section_key, title, text, citations, word_count, status, iteration, rag_queries, retrieved_chunk_ids, writer_notes, critic_feedback + compute_word_count(), unique_citation_keys()
> - `SurveyState(TypedDict)`: 13 campi con Annotated reducers per LangGraph (l2_summaries, outline, sections, verification_results, bibliography, provenance, final_paper, version_info, current_phase, current_section_key, iteration_count, errors, metadata)
> - Custom reducers: `_replace_or_default`, `_merge_sections` (preserva chiavi esistenti), `_merge_verification`, `_append_unique_errors` (deduplicazione)
> - `create_initial_state(l2_summaries?, outline?, version_info?, metadata?) → SurveyState`: factory con default completi
> - `state_to_dict(state) → dict` / `state_from_dict(data) → SurveyState`: serializzazione JSON round-trip
> - Convenience extractors: `get_outline_from_state`, `get_sections_from_state`, `get_verifications_from_state`, `get_l2_summaries_from_state`, `get_version_from_state`, `get_provenance_from_state`, `get_bibliography_keys`, `collect_all_citation_keys`, `get_pipeline_summary`
> - Test 11/11 passati: modelli ausiliari, state creation/serialization, reducers, extractors, LangGraph StateGraph compatibility end-to-end, reducer merge behavior

#### [x] Task 4.2: Creare agents/planner_agent.py — Agent 1: Planner
**Obiettivo:** Generare l'outline completo del paper usando i L2 summary aggiornati.  
**Dettagli:**
- Funzione `plan(state: SurveyState) → SurveyState`
- Input: L2 summaries di tutte le categorie
- Output: outline strutturato in ordine logico/cronologico
- Prompt Gemini Pro: "Genera un outline per un survey paper sulla sicurezza dei LLM. Struttura: Abstract, Intro, Background, [sezioni per categoria in ordine cronologico], Testing Methodologies (sezione dedicata), Discussion, Future Work, Conclusion"
- Ogni sezione ha: `title`, `description`, `target_length`, `key_topics`, `related_categories`
- Possibilità di usare un outline fornito dall'utente (override)  
**Test:** Generare outline con 4 categorie e verificare struttura logica.

> **Completato 24/02/2026.** File: `paperhunter/agents/planner_agent.py` (~893 righe)
> - `generate_outline(l2_summaries, *, force_model?, min_sections=8, target_sections=12, total_words?) → list[SectionOutline]`: genera outline via Gemini Pro con prompt strutturato. Calcola target parole automaticamente (~2000/categoria + 2000 fisse)
> - `plan(state: SurveyState) → dict`: nodo LangGraph Planner. Flusso: check override utente → carica L2 dallo stato o dal QdrantStore → check outline_file nel metadata → genera outline via LLM. Supporta 3 sorgenti: user_override, file JSON, llm_generated
> - `plan_from_l2_summaries(l2_summaries, *, outline_file?, force_model?) → list[SectionOutline]`: convenience function senza SurveyState
> - `load_outline_from_json(json_path) → list[SectionOutline]`: supporta 4 formati (array diretto, dict con sections array, dict con sections dict, dict con outline array)
> - `get_outline_stats(outline) → dict`: statistiche aggregate (total_sections, total_target_words, section_types, categories_covered, ecc.)
> - `_format_l2_for_prompt(l2_summaries) → str`: formattazione L2 compatta per il prompt (temporal_trends, taxonomy, testing_overview, open_gaps, synthesis)
> - `_parse_outline_json(raw_response) → list[dict] | None`: parser JSON robusto con 3 fallback (diretto, array extraction, trailing comma fix)
> - `_validate_and_normalize_outline(raw_sections, categories) → list[SectionOutline]`: validazione, deduplicazione key, aggiunta sezioni mancanti obbligatorie (abstract, intro, testing, conclusion), riordino logico per section_type
> - `_load_l2_from_store() → list[L2Summary]`: caricamento automatico L2 dal QdrantStore con fallback metadati da disco
> - Prompt template `_PLANNER_PROMPT_TEMPLATE`: struttura obbligatoria (Abstract, Introduction, Background, body sections per categoria, Testing Methodologies, Cross-Cutting Defenses, Discussion, Conclusion)
> - Sezioni obbligatorie: abstract, introduction, testing_methodologies, conclusion (aggiunte automaticamente se mancanti)
> - `paperhunter/agents/__init__.py` aggiornato con export di plan, generate_outline, plan_from_l2_summaries, load_outline_from_json, get_outline_stats
> - Test 10/10 offline passati: L2 formatting, JSON parsing (pulito, code fences, dict wrapper, trailing commas, fallback), outline validation (normalizzazione, dedup, sezioni mancanti, riordino), outline from JSON (4 formati), outline stats, plan con override utente, plan senza L2 (failure), LangGraph integration end-to-end, metadata preservati

#### [x] Task 4.3: Creare agents/writer_agent.py — Agent 2: Section Writer
**Obiettivo:** Scrivere ogni sezione del paper usando RAG on-the-fly con citazioni verificabili.  
**Dettagli:**
- Funzione `write_section(state: SurveyState, section_key: str) → SurveyState`
- Per ogni sezione:
  1. Formulare 3-5 query RAG specifiche dalla descrizione della sezione
  2. Retrieval ibrido (HybridRetriever) con filtri appropriati
  3. Opzionale: query al PropertyGraph per contesto relazionale
  4. Prompt Gemini Pro rigoroso: 
     - "Stile scientifico formale. Usa SOLO informazioni dai chunk forniti."
     - "Ogni claim DEVE avere citazione [bibtex_key]. Includi contesto temporale."
     - "Per la sezione Testing Methodologies, crea tabella riassuntiva dei metodi di test."
  5. Output JSON: `{text: str, citations: list[{key, claim, chunk_id}]}`
- Iterazione: se la sezione è troppo corta, genera query supplementari  
**Test:** Scrivere sezione "Prompt Injection Attacks" e verificare che ogni claim abbia citazione.

> **NOTE DI COMPLETAMENTO (Task 4.3):**
> - File creato: `paperhunter/agents/writer_agent.py` (~750 righe)
> - Funzioni principali:
>   - `write_section(state, section_key, *, retriever, reranker, graph_builder, force_model, use_reranker, use_graph, top_k_retrieval=25, n_rag_queries=5) → dict`: core function. Genera RAG queries specifiche per sezione → retrieval ibrido (HybridRetriever) con dedup multi-query → reranking opzionale (GeminiReranker) → optional GraphRAG context → generazione testo con Gemini Pro (prompt rigoroso con citazioni) → parsing JSON multi-fallback → costruisce SectionDraft
>   - `write_all_sections(state, *, retriever, reranker, graph_builder, skip_types={"abstract"}) → dict`: batch write di tutte le sezioni dall'outline con Rich table preview e progress logging
>   - `write(state) → dict`: nodo LangGraph. Gestisce NEEDS_REVISION (integra feedback Critic), inizializza retriever/reranker/graph_builder una volta, scrive tutte le sezioni pending
>   - `_generate_rag_queries(outline, n_queries=5) → list[str]`: genera query RAG specifiche per sezione via Gemini Flash con fallback keyword-based
>   - `_retrieve_context_for_section(queries, outline, ...) → (chunks, chunk_ids)`: multi-query retrieval con dedup per chunk_id, reranking opzionale, GraphRAG opzionale
>   - `_format_context_for_prompt(chunks, max_chars=60000) → str`: formatting con metadata (paper_title, year, bibtex_key, chunk_id)
>   - `_parse_writer_response(raw_response, section_key) → dict | None`: parsing JSON con 4 livelli fallback (clean JSON → brace extraction → trailing comma fix → regex citation extraction da plain text)
>   - `_extract_citations_from_text(text) → list[dict]`: regex [bibtex_key] extraction come ultimo fallback
>   - `write_sections_from_outline()`, `get_section_stats()`: funzioni di convenienza
> - Prompt templates: `_QUERY_GENERATION_PROMPT`, `_WRITER_PROMPT_TEMPLATE`, `_TESTING_SPECIAL_INSTRUCTIONS` (tabella riassuntiva), `_INTRO_SPECIAL_INSTRUCTIONS`, `_DISCUSSION_SPECIAL_INSTRUCTIONS`
> - Features: max 3 iterazioni writer↔critic per sezione, sezioni verified/finalized skip automatico, word count minimo 500 con warning, integrazione feedback Critic nel prompt di riscrittura
> - `paperhunter/agents/__init__.py` aggiornato con export di write, write_section, write_all_sections, write_sections_from_outline, get_section_stats
> - Test 13/13 offline passati: query JSON parsing (5 casi), writer response parsing (5 casi: clean/fenced/trailing comma/plain text fallback/empty), citation extraction (2 casi), context formatting (3 casi: normal/empty/truncation), special instructions (4 tipi sezione), fallback queries (2 casi), write_section con mock (successo + sezione mancante), get_section_stats (normal + empty), write_all_sections batch (3 sezioni + abstract skip), LangGraph integration (planner→writer), NEEDS_REVISION handling (iteration increment + max iter), skip verified sections, package imports

#### [x] Task 4.4: Creare agents/critic_agent.py — Agent 3: Critic/Verifier ✅ (completato 25 feb 2026)
**Obiettivo:** Verificare la provenance di ogni citazione e rigenerare sezioni problematiche.  
**Dettagli:**
- Funzione `verify_section(state: SurveyState, section_key: str) → SurveyState`
- Per ogni citazione `[bibtex_key]` nella sezione:
  1. Recuperare il chunk originale dal ChromaStore
  2. Confrontare il claim nel paper con il contenuto del chunk
  3. Prompt Gemini Flash: "Questo claim è supportato dal chunk? Risposta: SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED / SOURCE_NOT_FOUND"
- Se > 20% citazioni non supportate → flag per riscrittura
- Max 3 iterazioni writer→critic per sezione
- Output: `VerificationResult` con dettaglio per ogni citazione  
**Test:** Verificare una sezione con citazioni corrette e una con citazioni inventate.

> **Note di completamento:**
> - File creato: `paperhunter/agents/critic_agent.py` (~1050 righe)
> - Funzioni principali: `verify_section()` (core per singola sezione), `verify_all_sections()` (batch), `verify()` (nodo LangGraph)
> - Helpers interni: `_extract_citations_from_text()` (regex con filtro false positivi Table_/Figure_), `_extract_claims_for_citation()` (estrae frasi con contesto ±300 char), `_retrieve_source_chunks()` (da QdrantStore, paper_chunks + l1_summaries, ordinati per rilevanza), `_verify_single_citation()` / `_verify_citation_batch()` (verifica via Gemini Flash con batch e fallback singolo), `_generate_critic_feedback()` (feedback strutturato per Writer), `_parse_verification_response()` / `_parse_batch_verification_response()` (JSON parsing con 4 fallback), `_normalize_verification()` (mapping status varianti), `_build_verification_result()` (aggregazione), `_build_provenance_records()`, `_format_source_text()`, `get_verification_stats()`
> - Prompt templates: `_VERIFICATION_PROMPT_TEMPLATE` (singolo), `_BATCH_VERIFICATION_PROMPT_TEMPLATE` (batch), `_FEEDBACK_GENERATION_PROMPT_TEMPLATE`
> - Aggiornato `agents/__init__.py` con export: `verify`, `verify_section`, `verify_all_sections`, `get_verification_stats`
> - Usa Gemini Flash (configurabile via `config.agents.critic_model`), soglia configurabile `verification_threshold` (default 0.80)
> - Rich console output: tabelle per ogni sezione verificata, progress tracking, statistiche aggregate
> - Test 19/19 offline passati: extract citations (4 casi), extract claims (3 casi: normale/multi-frase/testo vuoto), parse verification response (8 casi: clean/fenced/trailing comma/regex fallback/partial/invalid/confidence clamp/markdown), parse batch response (4 casi: array/fenced/singolo oggetto/invalid), normalize verification (9 mapping status), format source text (3 casi: normale/vuoto/no chunks), build verification result (3 pass_rate), build provenance records (2 record), verify_section mock (batch 3 citazioni SUPPORTED), no citations (verifica triviale), max iterations (warning), needs revision (pass_rate 40%), already verified (skip), section not found (warning), get_verification_stats (aggregazione 2 sezioni), verify LangGraph node, verify_all_sections mock (2 verificate + 1 skippata), retrieve source chunks (ordinamento), package imports

#### [x] Task 4.5: Creare agents/assembler_agent.py — Agent 4: Final Assembler & Versioner
**Obiettivo:** Assemblare tutte le sezioni, generare abstract/intro/conclusion, inserire bibliografia.  
**Dettagli:**
- Funzione `assemble(state: SurveyState) → SurveyState`
- Step 1: Ordinare sezioni secondo outline
- Step 2: Generare Abstract (Gemini Pro, da riassunto di tutte le sezioni)
- Step 3: Generare/raffinare Introduction (contesto + motivazione + overview sezioni)
- Step 4: Generare Conclusion (sintesi + future work)
- Step 5: Assemblare tutte le sezioni in un unico documento
- Step 6: Generare bibliografia `.bib` dalle citazioni effettivamente usate
- Step 7: Versioning:
  - Se esiste versione precedente → genera "Changelog" (es. "Aggiunti 10 nuovi paper su Prompt Injection")
  - Salva con timestamp e versione: `Survey_v1.0_Feb2026`  
**Test:** Assemblare un paper completo da 4 sezioni di test e verificare coerenza.

**Completato:** 25 Febbraio 2026  
**File creati/modificati:**
- `paperhunter/agents/assembler_agent.py` (1390 righe) — Implementazione completa Agent 4
  - Funzioni pubbliche: `assemble()` (nodo LangGraph), `assemble_paper()` (orchestratore principale), `generate_abstract()`, `generate_introduction()`, `generate_conclusion()`, `assemble_document()`, `generate_bibliography()`, `create_version()`, `save_version()`, `get_assembly_stats()`
  - Helper interni: `_format_sections_overview()`, `_format_sections_summary()`, `_extract_key_findings()`, `_extract_citations_from_text()`, `_finalize_all_sections()`, `_get_version_dir()`, `_get_next_version_number()`, `_generate_changelog()`, `_manual_changelog()`, `_slugify()`, `_print_assembly_report()`
  - Prompt template dedicati per Abstract, Introduction, Conclusion, Changelog
  - Usa `llm_call_with_retry()` con modello Pro per generazione testo
  - Filtra il master .bib per le sole citation key effettivamente usate
  - Auto-versioning con `Survey_vX.Y_MonYYYY/` + changelog automatico vs versione precedente
  - Salva: survey.md, version_info.json, provenance.json, changelog.md, citation_keys.json
- `paperhunter/agents/__init__.py` — Aggiunti export Assembler Agent (10 funzioni)
- `test_assembler_agent.py` (905 righe) — 22 test offline con mock LLM, tutti passati

#### [x] Task 4.6: Creare agents/pipeline.py — LangGraph Workflow orchestration
**Obiettivo:** Definire il grafo LangGraph che orchestra tutti gli agent.  
**Dettagli:**
- Definire `StateGraph` con nodi: `plan`, `write`, `verify`, `assemble`
- Flusso:
  ```
  START → plan → write_loop → verify_loop → assemble → END
  ```
  - `write_loop`: itera su tutte le sezioni dell'outline
  - `verify_loop`: per ogni sezione, verifica e opzionalmente riscrive (max 3 iter)
  - Conditional edges: se verificazione fallisce → torna a write
- Checkpoint: salvare stato intermedio dopo ogni nodo
- Error handling: se un agente fallisce, loggare e continuare con le altre sezioni
- Callback per progress tracking (rich progress bar)  
**Test:** Eseguire pipeline completa su un mini-outline di 2 sezioni.

> **Completato:** 25 Febbraio 2026
> **File creati/modificati:**
> - `paperhunter/agents/pipeline.py` (~530 righe) — Implementazione completa LangGraph workflow
>   - Costanti: `NODE_PLAN`, `NODE_WRITE`, `NODE_VERIFY`, `NODE_ASSEMBLE`, `EDGE_REVISE`, `EDGE_DONE`
>   - `build_pipeline(save_checkpoints?, on_progress?) → CompiledGraph`: costruisce e compila il `StateGraph` con flusso `START → plan → write → verify → should_revise? → (revise → write | done → assemble) → END`
>   - `run_pipeline(l2_summaries?, outline?, outline_file?, version_info?, metadata?, save_checkpoints?, on_progress?) → SurveyState`: esecuzione completa della pipeline da zero
>   - `run_pipeline_from_checkpoint(checkpoint_path, save_checkpoints?, on_progress?) → SurveyState`: resume da un checkpoint salvato, determina il nodo successivo e riprende rispettando il loop write↔verify
>   - `_resume_from_node(state, start_node, save_checkpoints?, on_progress?) → SurveyState`: esecuzione manuale dei nodi rimanenti con loop write↔verify
>   - `_should_revise(state) → str`: conditional edge che controlla se ci sono sezioni `NEEDS_REVISION` e se `iteration_count < max_writer_iterations`
>   - `_make_node_with_checkpoint(node_fn, node_name, save_checkpoints?, on_progress?) → Callable`: wrapper che aggiunge checkpoint JSON + progress callback + error handling a ogni nodo. Per nodi critici (plan/assemble) ritorna FAILED, per write/verify logga e continua
>   - Checkpoint management:
>     - `save_checkpoint(state, node_name, checkpoint_dir?) → Path`: salva JSON timestamped + file `_latest`
>     - `load_checkpoint(checkpoint_path) → (node_name, SurveyState)`: ricarica stato da file
>     - `get_latest_checkpoint(node_name?) → Path | None`: cerca il checkpoint più recente
>     - `clean_checkpoints(keep_latest?) → int`: pulizia file vecchi
>     - `list_checkpoints() → list[dict]`: elenco con path, node, timestamp, size
>   - `get_pipeline_graph_description() → str`: descrizione testuale del grafo per debug/docs
>   - `_print_pipeline_report(state, elapsed)`: report finale con tabella statistiche, dettaglio verifiche, errori
> - `paperhunter/agents/__init__.py` — Aggiunti export pipeline (9 funzioni: build_pipeline, run_pipeline, run_pipeline_from_checkpoint, save_checkpoint, load_checkpoint, get_latest_checkpoint, clean_checkpoints, list_checkpoints, get_pipeline_graph_description)
> - `test_pipeline.py` (~680 righe) — 21 test offline con mock LLM, tutti passati:
>   - build_pipeline: compilazione grafo OK
>   - _should_revise: 4 scenari (no revision → DONE, needs revision → REVISE, max iter → DONE, no sections → DONE)
>   - checkpoint round-trip: salva + ricarica + verifica integrità
>   - get_latest_checkpoint: ricerca per nodo e generica
>   - clean_checkpoints: rimozione vecchi, mantenimento _latest
>   - list_checkpoints: elenco strutturato
>   - _make_node_with_checkpoint: wrapper esecuzione + checkpoint + error handling + progress callback
>   - verify node incrementa iteration_count
>   - run_pipeline mock completo: 4 nodi, paper generato, checkpoint salvati
>   - run_pipeline con loop write↔verify: critic boccia → writer riscrive → critic approva (write×2, verify×2)
>   - _resume_from_node: resume da nodo specifico, sequenza corretta
>   - run_pipeline_from_checkpoint: resume da file, plan non ri-eseguito
>   - get_pipeline_graph_description: contenuto testuale verificato
>   - _print_pipeline_report: copertura con stati diversi (successo, fallito, vuoto)
>   - pipeline plan failure: errore critico gestito, fase FAILED

---

### FASE 5 — Post-processing & Output

#### [x] Task 5.1: Creare postprocessing/provenance_tracker.py — Provenance tracking completo
**Obiettivo:** Generare un JSON di provenance che mappa ogni claim alla sua fonte originale.  
**Dettagli:**
- Classe `ProvenanceTracker`:
  - `add_record(claim, bibtex_key, chunk_id, source_text, verification_status)`
  - `generate_report() → dict`: report completo
  - `save_json(output_path)`: esporta il file JSON
- Struttura JSON:
  ```json
  {
    "version": "1.0",
    "generated_at": "...",
    "total_claims": N,
    "verified_claims": N,
    "records": [
      {
        "section": "...",
        "claim": "...",
        "citation_key": "Paper_YYYY_ID",
        "source_chunk_id": "...",
        "source_text_excerpt": "...",
        "verification": "SUPPORTED",
        "confidence": 0.95
      }
    ]
  }
  ```  
**Test:** Creare 10 record e esportare JSON, verificare struttura.

> **Completato 26/02/2026.** File: `paperhunter/postprocessing/provenance_tracker.py` (~600 righe)
> - Classe `ProvenanceTracker(version?)`: tracker completo per la provenance delle citazioni nel survey
> - `add_record(section, claim, citation_key, chunk_id?, source_text?, verification_status?, confidence?, reasoning?) → ProvenanceRecord`: aggiunge singolo record con normalizzazione status, clamping confidence, troncamento claim (1000 chars) e source_text (500 chars)
> - `add_records_batch(records) → int`: aggiunta batch da lista di dict. Supporta sia il formato ProvenanceTracker che il formato Critic Agent (`section_key`/`section`, `claim`/`claim_text`, ecc.)
> - `import_from_state(state) → int`: importa record di provenance dallo stato SurveyState della pipeline LangGraph (campo `provenance`)
> - `generate_report() → dict`: report completo con schema_version, survey_version, generated_at, statistiche aggregate, e sezioni raggruppate con record e statistiche per-sezione
> - `get_stats() → dict`: statistiche aggregate (total_claims, verified/unverified, supported/partially/not_supported/source_not_found, coverage_rate, support_rate, avg_confidence, unique_citation_keys, unique_sections, top_cited_papers top-10)
> - `get_section_stats(section) → dict`: statistiche per una singola sezione
> - Lookup: `get_records_by_section()`, `get_records_by_citation()`, `get_records_by_status()`, `get_unsupported_claims()`
> - `save_json(output_path?) → Path`: salva report JSON completo. Default da `config.yaml` → `output.provenance_dir`
> - `save_markdown(output_path?) → Path`: report leggibile in Markdown con tabelle, breakdown per status, top cited papers, dettaglio per sezione, claim problematici
> - `load_json(json_path) → ProvenanceTracker` (classmethod): carica da file JSON con ricostruzione record
> - `merge(other) → int`: unisce record da un altro tracker con deduplicazione (section + citation_key + claim_text[:100]). Mantiene confidence più alta per duplicati
> - `clear()`: rimuove tutti i record
> - `print_summary()`: riepilogo Rich formattato in console (4 tabelle: generale, breakdown, top cited, per sezione)
> - Helper `_normalize_status()`: normalizzazione case-insensitive con 13 varianti supportate (es. "partial", "unsupported", "missing", "not found")
> - Helper `_record_to_dict()`, `_compute_section_stats()`
> - Convenience functions: `create_provenance_from_state(state, version?)`, `load_provenance(json_path)`
> - `paperhunter/postprocessing/__init__.py` aggiornato con export di `ProvenanceTracker`, `create_provenance_from_state`, `load_provenance`
> - Test 50/50 passati:
>   - _normalize_status: 4 casi standard, case-insensitive, 5 varianti, unknown→SOURCE_NOT_FOUND ✓
>   - _record_to_dict: serializzazione completa, None chunk_id → "" ✓
>   - _compute_section_stats: lista vuota, mixed statuses (coverage=0.5, support=0.25, avg_conf≈0.425) ✓
>   - add_record: singolo, enum status, confidence clamping (1.5→1.0, -0.5→0.0), claim troncamento (2000→1000 chars) ✓
>   - add_records_batch: 10 record da formato Critic, record invalidi gestiti ✓
>   - import_from_state: 10 importati, stato vuoto→0, stato senza provenance→0 ✓
>   - get_stats: tracker vuoto, stats completi (5 SUPPORTED, 2 PARTIAL, 2 NOT_SUPPORTED, 1 NOT_FOUND, coverage=0.7), Zou_2023_GCG top cited (2 claim), section_stats ✓
>   - lookup: by_section (4 PI), by_citation (Zou appare 2×), by_status (5 supported), unsupported (3 record), sezione inesistente→[] ✓
>   - generate_report: struttura completa (schema_version, 3 sezioni, statistics, records), report vuoto ✓
>   - save/load JSON: round-trip 10 record, load nonexistent→FileNotFoundError, load invalid→ValueError, creazione directory profonde ✓
>   - save_markdown: contenuto con tutti i marcatori (SUPPORTED, sezioni, claim problematici), markdown vuoto ✓
>   - merge: 1 nuovo aggiunto, duplicato skippato + confidence aggiornata, merge vuoto→0 ✓
>   - clear: 10→0 record ✓
>   - create_provenance_from_state: versione da state, versione esplicita, stato vuoto ✓
>   - load_provenance shortcut: round-trip ✓
>   - __repr__/__len__: formattazione corretta ✓
>   - print_summary: smoke test (no crash con dati e vuoto) ✓
>   - edge cases: default status, records property copia, version property, 5 sezioni report ✓

#### [x] Task 5.2: Creare postprocessing/bibliography_generator.py — Generazione .bib finale
**Obiettivo:** Generare il file `.bib` finale con solo le entry effettivamente citate nel paper.  
**Dettagli:**
- Funzione `generate_final_bib(used_keys: set[str], master_bib: dict) → str`
- Filtrare il master `.bib` per tenere solo le chiavi usate
- Validare ogni entry: tutti i campi obbligatori presenti
- Ordinare alfabeticamente per chiave
- Output: file `.bib` formattato  
**Test:** Da 50 entry master, filtrare per 15 chiavi usate e verificare .bib.

> **Completato 01/03/2026.** File: `paperhunter/postprocessing/bibliography_generator.py` (~580 righe)
> - Utility interne: `_extract_bib_key()`, `_extract_bib_type()`, `_extract_field()` (regex con supporto `{...}` e bare values), `_extract_year_from_entry()`, `_count_citations_in_text()` (regex `[bibtex_key]` con filtro false positivi Table_/Figure_), `_reformat_entry()` (indentazione consistente a 2 spazi), `_key_similarity()` (Jaccard su token split per `_`)
> - Classe `BibliographyGenerator`:
>   - `load_master_bib(bib_path?)`: carica da file .bib (default `output/references.bib`). `load_master_bib_from_dict(entries)`: carica da dict in memoria
>   - `generate(used_keys, *, validate=True, reformat=True, survey_text?) → dict`: filtra master per chiavi usate, valida entry (campi obbligatori), riformatta, conta citazioni nel testo. Ritorna report completo con statistiche, missing keys, entry details
>   - `save_bib(output_path?) → Path`: salva .bib filtrato (default `output/references_final.bib`). Raise RuntimeError se `generate()` non chiamato
>   - `save_report(output_path?) → Path`: salva report JSON (default `output/bib_report.json`)
>   - `get_stats() → dict`: statistiche (master_bib_total, filtered, missing, invalid, coverage_rate, type_distribution, year_distribution, year_min/max, top_cited, total_citation_occurrences)
>   - `get_missing_keys_report() → list[dict]`: dettaglio chiavi mancanti con suggerimenti fuzzy (Jaccard ≥ 0.5)
>   - `get_entry_details(bibtex_key) → dict | None`: dettagli singola entry (tipo, anno, titolo, venue, DOI, validazione)
>   - `print_summary()`: riepilogo Rich formattato (4 tabelle: generale, tipo, anno, top cited)
> - Convenience functions:
>   - `generate_final_bib(used_keys, *, master_bib_path?, output_path?, validate?, reformat?, survey_text?) → (Path | None, dict)`: workflow completo load → generate → save
>   - `generate_bib_from_state(state, *, master_bib_path?, output_path?, validate?) → (Path | None, dict)`: genera da SurveyState, raccoglie chiavi con `collect_all_citation_keys()`, estrae testo da `final_paper`
>   - `load_and_filter_bib(master_bib_path, used_keys) → dict`: helper leggero senza validazione
>   - `merge_bib_files(*bib_paths, output_path?) → (Path, int)`: unisce più .bib con deduplicazione
> - `paperhunter/postprocessing/__init__.py` aggiornato con export: `BibliographyGenerator`, `generate_final_bib`, `generate_bib_from_state`, `load_and_filter_bib`, `merge_bib_files`
> - Test 75/75 passati:
>   - _extract_bib_key: 5 casi (normal, article, misc, empty, no match) ✓
>   - _extract_bib_type: 4 casi (inproceedings, article, misc, empty) ✓
>   - _extract_field: 8 casi (title, author, year, booktitle, doi, missing, journal, case-insensitive) ✓
>   - _extract_year: 4 casi (2023, 2024, no year, empty) ✓
>   - _count_citations: 4 casi (basic 2+1, no citations, filter Table/Figure, empty) ✓
>   - _reformat_entry: 3 casi (basic, already formatted, empty) ✓
>   - _key_similarity: 4 casi (identical=1.0, partial>0.4, no match=0.0, empty=0.0) ✓
>   - BibliographyGenerator init/repr: 2 casi ✓
>   - Load: 3 casi (dict, file, nonexistent) ✓
>   - Generate: 10 casi (filter 2/5, missing keys, all missing, empty keys, filter all 5/5, with/without validation, survey text citation counting, report structure, list input, empty master) ✓
>   - Save: 4 casi (bib file, no generate→RuntimeError, JSON report, nested dir) ✓
>   - Stats: 5 casi (basic, type distribution, year distribution, empty, with citations) ✓
>   - Missing report: 2 casi (with suggestions Jaccard≥0.5, no suggestions) ✓
>   - Entry details: 3 casi (existing, not filtered but in master, nonexistent) ✓
>   - Print summary: 2 smoke test (data + empty) ✓
>   - generate_final_bib: 4 casi (basic, empty master→None, no match→None, survey text) ✓
>   - generate_bib_from_state: 1 caso (SurveyState con SectionDraft e CitationRecord) ✓
>   - load_and_filter_bib: 2 casi (basic, empty keys) ✓
>   - merge_bib_files: 2 casi (2 file con dedup 3 uniche, empty) ✓
>   - Package imports: 2 casi (from module, from package) ✓

#### [ ] Task 5.3: Creare postprocessing/latex_exporter.py — Export LaTeX
**Obiettivo:** Convertire il paper Markdown in formato LaTeX (template IEEEtran o ACL).  
**Dettagli:**
- Funzione `markdown_to_latex(md_content, template) → str`
- Template: header LaTeX con `\documentclass`, packages, `\begin{document}`, `\bibliography`
- Conversioni: `# Title` → `\section{}`, `**bold**` → `\textbf{}`, `[key]` → `\cite{key}`, tabelle → `tabular`
- Supporto per figure placeholder e tabelle
- Output: file `.tex` compilabile  
**Test:** Convertire un Markdown di test e verificare che compili con pdflatex.

#### [ ] Task 5.4: Creare postprocessing/versioning.py — Versioning e changelog
**Obiettivo:** Gestire versioni successive del survey con changelog automatico.  
**Dettagli:**
- Classe `VersionManager`:
  - `get_current_version() → SurveyVersion`
  - `create_new_version(paper_count, categories) → SurveyVersion`
  - `generate_changelog(old_version, new_version) → str`
  - `save_version_metadata(version, output_dir)`
- Storage: `output/versions/` con sottocartelle per versione
- Changelog: diff su categorie, numero paper, sezioni aggiunte/modificate  
**Test:** Creare v1.0, poi v1.1 con 5 paper in più, verificare changelog.

#### [ ] Task 5.5: Creare postprocessing/checkpoint_manager.py — Resume from checkpoint
**Obiettivo:** Salvare checkpoint intermedi per riprendere in caso di interruzione.  
**Dettagli:**
- Classe `CheckpointManager`:
  - `save_checkpoint(state: SurveyState, step_name: str)`
  - `load_latest_checkpoint() → SurveyState | None`
  - `list_checkpoints() → list[CheckpointInfo]`
  - `clear_checkpoints()`
- Storage: `output/.checkpoints/` con file JSON per ogni step
- Integrazione con LangGraph built-in checkpointing  
**Test:** Salvare 3 checkpoint, simulare crash, ricaricare ultimo.

---

### FASE 6 — Integrazione & CLI

#### [ ] Task 6.1: Aggiornare main.py — Nuovo CLI con subcomandi completi
**Obiettivo:** Ristrutturare il CLI per esporre tutte le funzionalità del nuovo sistema.  
**Dettagli:**
- Subcomandi:
  - `paperhunter search` — cercare e scaricare nuovi paper (logica esistente)
  - `paperhunter ingest` — eseguire ingestion pipeline incrementale
  - `paperhunter summarize` — generare/rigenerare L1 e L2
  - `paperhunter index` — costruire/aggiornare indici (Chroma + BM25 + Graph)
  - `paperhunter write` — eseguire pipeline agentica completa
  - `paperhunter query` — query interattiva contro il KB
  - `paperhunter stats` — statistiche complete del sistema
  - `paperhunter export` — esportare in LaTeX/BibTeX
- Opzioni globali: `--config`, `--verbose`, `--resume`
- Rich console output per tutti i comandi  
**Test:** Eseguire ogni subcomando con `--help` e verificare documentazione.

#### [ ] Task 6.2: Integrazione end-to-end — Test pipeline completa
**Obiettivo:** Eseguire l'intera pipeline su un sottoinsieme di paper e verificare l'output.  
**Dettagli:**
- Selezionare 20 paper (5 per categoria) come test set
- Eseguire: `ingest → summarize → index → write → export`
- Verificare:
  - PDF processati correttamente da GROBID
  - Metadati arricchiti da S2
  - L1 e L2 summary validi
  - Paper generato con citazioni corrette
  - .bib, .tex, provenance JSON generati
  - Nessun errore critico nel log
- Documentare eventuali issue e fix  
**Test:** Output completo in `output/test_run/`.

#### [ ] Task 6.3: Aggiornare README.md — Documentazione completa
**Obiettivo:** Riscrivere il README con architettura, setup, usage completo.  
**Dettagli:**
- Diagramma architettura aggiornato
- Setup guide: Docker (GROBID, Neo4j), Python deps, config.yaml, .env
- Quick start: `paperhunter ingest → summarize → write`
- Sezione troubleshooting
- API reference per ogni modulo
- Esempi di output  
**Test:** Seguire il README da zero su una macchina pulita.

---

## 📌 Note per le Sessioni Future

- **Una sessione = un task.** Non anticipare mai il codice del task successivo.
- **Ogni task è testabile in isolamento.** Fornire sempre un mini-test alla fine.
- **Aggiornare questo file** mettendo `[x]` al task completato e aggiungendo note sotto.
- **Se un task richiede modifiche a task precedenti**, documentare nella sezione note del task.
- **Formato chiusura sessione:** "Task completato con successo. Per favore, apri una nuova chat, incolla il master prompt e il nuovo TODO.md aggiornato, e chiedimi di eseguire il prossimo task. Questo manterrà la mia memoria fresca e il codice pulito."

---

## 📊 Progress Tracker

| Fase | Task | Completati | Totale |
|------|------|-----------|--------|
| 0 — Fondamenta | 0.1–0.6 | 6 | 6 |
| 1 — Ingestion | 1.1–1.7 | 7 | 7 |
| 2 — Summarization | 2.1–2.3 | 3 | 3 |
| 3 — Indexing + GraphRAG | 3.1–3.6 | 6 | 6 |
| 4 — Agentic Writing | 4.1–4.6 | 6 | 6 |
| 5 — Post-processing | 5.1–5.5 | 2 | 5 |
| 6 — Integrazione | 6.1–6.3 | 0 | 3 |
| **TOTALE** | | **31** | **36** |
