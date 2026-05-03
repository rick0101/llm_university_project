# Graph Report - paperhunter  (2026-04-12)

## Corpus Check
- 75 files · ~151,742 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2363 nodes · 7034 edges · 53 communities detected
- Extraction: 47% EXTRACTED · 53% INFERRED · 0% AMBIGUOUS · INFERRED: 3712 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Document Assembly|Document Assembly]]
- [[_COMMUNITY_Graph Community Detection|Graph Community Detection]]
- [[_COMMUNITY_Community Analysis|Community Analysis]]
- [[_COMMUNITY_Text Chunking|Text Chunking]]
- [[_COMMUNITY_Configuration Management|Configuration Management]]
- [[_COMMUNITY_BM25 Search Index|BM25 Search Index]]
- [[_COMMUNITY_Paper Downloading|Paper Downloading]]
- [[_COMMUNITY_Figure Code Generation|Figure Code Generation]]
- [[_COMMUNITY_API Key Management|API Key Management]]
- [[_COMMUNITY_Markdown Conversion|Markdown Conversion]]
- [[_COMMUNITY_Bibliography Generation|Bibliography Generation]]
- [[_COMMUNITY_Version Management|Version Management]]
- [[_COMMUNITY_Checkpoint System|Checkpoint System]]
- [[_COMMUNITY_Lesson Extraction|Lesson Extraction]]
- [[_COMMUNITY_BibTeX Processing|BibTeX Processing]]
- [[_COMMUNITY_Cache Invalidation|Cache Invalidation]]
- [[_COMMUNITY_Provenance Tracking|Provenance Tracking]]
- [[_COMMUNITY_Rate Limiting|Rate Limiting]]
- [[_COMMUNITY_PDF Hashing|PDF Hashing]]
- [[_COMMUNITY_Summary Clustering|Summary Clustering]]
- [[_COMMUNITY_Multimodal Embeddings|Multimodal Embeddings]]
- [[_COMMUNITY_Logging Utilities|Logging Utilities]]
- [[_COMMUNITY_Pipeline Orchestration|Pipeline Orchestration]]
- [[_COMMUNITY_Venue Ranking|Venue Ranking]]
- [[_COMMUNITY_Prompt Loading|Prompt Loading]]
- [[_COMMUNITY_Quality Metrics|Quality Metrics]]
- [[_COMMUNITY_Device Detection|Device Detection]]
- [[_COMMUNITY_Skill Loading|Skill Loading]]
- [[_COMMUNITY_LLM Provider Config|LLM Provider Config]]
- [[_COMMUNITY_Flash Model Config|Flash Model Config]]
- [[_COMMUNITY_Pro Model Config|Pro Model Config]]
- [[_COMMUNITY_Temperature Default|Temperature Default]]
- [[_COMMUNITY_Search Results Config|Search Results Config]]
- [[_COMMUNITY_ArXiv Categories|ArXiv Categories]]
- [[_COMMUNITY_HTTP Delay Config|HTTP Delay Config]]
- [[_COMMUNITY_Playwright Delay Config|Playwright Delay Config]]
- [[_COMMUNITY_Download Directory Config|Download Directory Config]]
- [[_COMMUNITY_Download Timeout Config|Download Timeout Config]]
- [[_COMMUNITY_Headless Mode Config|Headless Mode Config]]
- [[_COMMUNITY_Limiter Stats|Limiter Stats]]
- [[_COMMUNITY_Active Request Count|Active Request Count]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Active API Key|Active API Key]]
- [[_COMMUNITY_Key Label|Key Label]]
- [[_COMMUNITY_Key Rotation Check|Key Rotation Check]]
- [[_COMMUNITY_BibTeX Master Dict|BibTeX Master Dict]]
- [[_COMMUNITY_BibTeX Used Keys|BibTeX Used Keys]]
- [[_COMMUNITY_Filtered BibTeX Entries|Filtered BibTeX Entries]]
- [[_COMMUNITY_Missing BibTeX Keys|Missing BibTeX Keys]]
- [[_COMMUNITY_Invalid BibTeX Entries|Invalid BibTeX Entries]]
- [[_COMMUNITY_Citation Counts|Citation Counts]]
- [[_COMMUNITY_Checkpoint Info Factory|Checkpoint Info Factory]]
- [[_COMMUNITY_Venue Matching|Venue Matching]]

## God Nodes (most connected - your core abstractions)
1. `QdrantStore` - 303 edges
2. `PaperChunk` - 245 edges
3. `Paper` - 219 edges
4. `L1Summary` - 158 edges
5. `ChunkType` - 145 edges
6. `L2Summary` - 130 edges
7. `SurveyState` - 119 edges
8. `PipelinePhase` - 118 edges
9. `SectionOutline` - 118 edges
10. `SurveyVersion` - 109 edges

## Surprising Connections (you probably didn't know these)
- `BibTeX Generator — Generazione automatica di entry BibTeX per ogni paper.  Gener` --uses--> `Paper`  [INFERRED]
  paperhunter/ingestion/bibtex_generator.py → paperhunter/models.py
- `Filtra nomi autori palesemente non validi.` --uses--> `Paper`  [INFERRED]
  paperhunter/ingestion/bibtex_generator.py → paperhunter/models.py
- `Rimuove o escape-a caratteri problematici per BibTeX.      Args:         text: T` --uses--> `Paper`  [INFERRED]
  paperhunter/ingestion/bibtex_generator.py → paperhunter/models.py
- `Rimuove accenti da testo Unicode per generare chiavi ASCII-safe.      Args:` --uses--> `Paper`  [INFERRED]
  paperhunter/ingestion/bibtex_generator.py → paperhunter/models.py
- `Estrae il cognome dall'autore (assume formato "Nome Cognome" o "Cognome, Nome").` --uses--> `Paper`  [INFERRED]
  paperhunter/ingestion/bibtex_generator.py → paperhunter/models.py

## Communities

### Community 0 - "Document Assembly"
Cohesion: 0.03
Nodes (299): assemble(), assemble_document(), assemble_paper(), create_version(), _extract_citations_from_text(), _extract_key_findings(), _finalize_all_sections(), _format_sections_overview() (+291 more)

### Community 1 - "Graph Community Detection"
Cohesion: 0.02
Nodes (220): True se l'indice è stato costruito o caricato., Numero di documenti nell'indice., Lista dei chunk_id indicizzati (ordine di inserimento)., CommunityDetector, Rimuove la proprietà ``community_id`` da tutti i nodi Entity.          Utile per, Rilevamento di comunità tematiche nel grafo Neo4j via GDS.      Utilizza il ``Ne, _clean_llm_response(), _community_exists_in_store() (+212 more)

### Community 2 - "Community Analysis"
Cohesion: 0.03
Nodes (169): CommunityDetectionError, detect_graph_communities(), _extract_key_entities(), GDSNotAvailableError, get_community_stats_standalone(), _log_community_detection_results(), GraphRAG Community Detector — Rilevamento comunità tematiche nel grafo Neo4j.  I, Verifica che il plugin GDS sia disponibile in Neo4j.          Caching: il risult (+161 more)

### Community 3 - "Text Chunking"
Cohesion: 0.03
Nodes (160): best_fit_pack(), _chunk_from_raw_text(), chunk_grobid_output(), chunk_raw_text(), _clean_text(), _detect_sections_from_text(), _get_chunk_overlap(), _get_chunk_size() (+152 more)

### Community 4 - "Configuration Management"
Cohesion: 0.02
Nodes (143): BaseModel, BaseSettings, AgentsConfig, _build_settings(), ClusteringConfig, DownloadConfig, ensure_download_dir(), ensure_output_dirs() (+135 more)

### Community 5 - "BM25 Search Index"
Cohesion: 0.02
Nodes (107): BM25Index, _naive_stem(), BM25 Keyword Index per PaperHunter.  Indice BM25 (Okapi BM25) complementare alla, Stemming naïve: rimuove suffissi comuni dall'inglese.      Non è sofisticato com, Tokenizza testo per indicizzazione BM25.      Pipeline: lowercase → regex split, Indice BM25 (Okapi BM25) per retrieval keyword-based.      Indicizza oggetti ``P, Inizializza l'indice BM25.          Args:             index_path: Percorso su di, Costruisce l'indice BM25 da una lista di PaperChunk.          Sovrascrive l'indi (+99 more)

### Community 6 - "Paper Downloading"
Cohesion: 0.04
Nodes (120): CommunitySummarizer, Downloader, Download multiple papers with rate limiting.                  Args:, Hybrid PDF downloader with fallback to browser automation.          Attempts dir, _auto_export(), _build_bm25_index(), _build_communities(), _build_graph_index() (+112 more)

### Community 7 - "Figure Code Generation"
Cohesion: 0.04
Nodes (78): _extract_python_code(), generate_code(), FigureAgent — CodeGen: genera codice Matplotlib/Seaborn per le figure.  Dato un, Genera codice Python per una code figure.      Args:         spec: La specifica, Estrai codice Python dal response LLM., Valida e correggi il codice generato., _validate_code(), critique_code_only() (+70 more)

### Community 8 - "API Key Management"
Cohesion: 0.04
Nodes (64): ApiKeyManager, _complete_with_continuation(), _create_openrouter_client(), current_key(), current_label(), _enforce_call_cooldown(), _get_api_key(), get_embedding_model() (+56 more)

### Community 9 - "Markdown Conversion"
Cohesion: 0.05
Nodes (55): _clean_double_blank_lines(), _convert_blockquotes(), _convert_body(), _convert_bold_italic(), _convert_citations(), _convert_code_blocks(), _convert_footnotes(), _convert_headings() (+47 more)

### Community 10 - "Bibliography Generation"
Cohesion: 0.05
Nodes (36): BibliographyGenerator, _count_citations_in_text(), _extract_bib_key(), _extract_bib_type(), _extract_field(), _extract_year_from_entry(), generate_bib_from_state(), generate_final_bib() (+28 more)

### Community 11 - "Version Management"
Cohesion: 0.05
Nodes (30): create_version_from_state(), get_latest_version(), load_version_history(), Versioning & Changelog — Gestione versioni successive del survey.  Centralizza t, Ritorna la versione più recente trovata su disco.          Scansiona ``versions_, Carica una versione specifica dato il suo numero (es. "1.2").          Args:, Elenca tutte le versioni trovate su disco, ordinate per numero.          Returns, Elenca le directory versione con info strutturali.          Returns: (+22 more)

### Community 12 - "Checkpoint System"
Cohesion: 0.05
Nodes (31): CheckpointInfo, CheckpointManager, from_file(), get_checkpoint_manager(), get_resume_recommendation(), list_all_checkpoints(), postprocessing/checkpoint_manager.py — Gestione centralizzata dei checkpoint  Fo, Genera statistiche aggregate sui checkpoint.          Returns:             Dict (+23 more)

### Community 13 - "Lesson Extraction"
Cohesion: 0.08
Nodes (40): extract_from_all_verifications(), extract_from_pipeline(), extract_from_summarization(), extract_from_verification(), _make_run_id(), Extract lessons from pipeline results (verification, summarization, errors).  Ea, Create a lesson from a failed L1 or L2 summarization.      Parameters     ------, Create a lesson from a generic pipeline error.      Covers loop exhaustion, time (+32 more)

### Community 14 - "BibTeX Processing"
Cohesion: 0.08
Nodes (41): _build_bibtex_from_scratch(), _build_misc_bibtex(), enrich_paper_bibtex(), ensure_complete_bibtex(), _extract_keyword_from_title(), _extract_surname(), _format_authors_bibtex(), generate_bibtex_entry() (+33 more)

### Community 15 - "Cache Invalidation"
Cohesion: 0.08
Nodes (20): InvalidationManager, InvalidationReport, Invalidation Manager — Auto-invalidazione e rigenerazione L2 summary.  Quando nu, Gestore dell'auto-invalidazione e rigenerazione dei summary L2.      Rileva auto, Controlla tutte le categorie e rigenera i L2 invalidati.          Questo è il me, Ritorna le categorie che necessitano di rigenerazione L2.          Metodo legger, Marca il L2 di una categoria come invalidato.          Aggiorna il metadata su d, Raccoglie tutti gli L1 della categoria e rigenera il L2.          Carica i L1 su (+12 more)

### Community 16 - "Provenance Tracking"
Cohesion: 0.11
Nodes (7): _compute_section_stats(), create_provenance_from_state(), load_json(), load_provenance(), _normalize_status(), ProvenanceTracker, _record_to_dict()

### Community 17 - "Rate Limiting"
Cohesion: 0.09
Nodes (16): get_gemini_limiter(), get_grobid_semaphore(), get_s2_limiter(), GrobidSemaphore, Rate limiter per API calls (Gemini, Semantic Scholar, GROBID).  Implementa un to, Versione asincrona di ``wait()``.          Returns:             Tempo di attesa, Reset completo del limiter., Semaforo per limitare le richieste GROBID concorrenti.      Può essere usato com (+8 more)

### Community 18 - "PDF Hashing"
Cohesion: 0.12
Nodes (21): compute_pdf_hash(), create_registry_entry(), _default_registry_path(), _get_category_from_path(), get_new_or_modified_pdfs(), get_registry_stats(), load_hash_registry(), PDF Hasher — Logica hash SHA-256 per Upsert incrementale.  Calcola l'hash di ogn (+13 more)

### Community 19 - "Summary Clustering"
Cohesion: 0.14
Nodes (21): _agglomerative_cluster(), cluster_summaries(), fetch_l1_embeddings_from_qdrant(), get_cluster_labels(), _get_clustering_config(), _group_by_labels(), _log_cluster_stats(), _normalize_embeddings() (+13 more)

### Community 20 - "Multimodal Embeddings"
Cohesion: 0.13
Nodes (13): _cosine_similarity(), get_multimodal_embedder(), MultimodalEmbedder, Multimodal Embedder — Wrapper Gemini Embedding 2 Preview per immagini.  Genera e, Crea il client google-genai con API key e http options., Genera embedding multimodale da un'immagine.          Args:             image_by, Genera embedding testuale tramite lo stesso modello multimodale.          Usa lo, Embedding batch di immagini con progress bar Rich.          Args:             im (+5 more)

### Community 21 - "Logging Utilities"
Cohesion: 0.11
Nodes (19): get_logger(), print_error(), print_panel(), print_success(), print_table(), print_warning(), progress_bar(), progress_context() (+11 more)

### Community 22 - "Pipeline Orchestration"
Cohesion: 0.26
Nodes (15): build_pipeline(), clean_checkpoints(), _get_checkpoint_dir(), get_latest_checkpoint(), get_pipeline_graph_description(), list_checkpoints(), load_checkpoint(), _make_node_with_checkpoint() (+7 more)

### Community 23 - "Venue Ranking"
Cohesion: 0.21
Nodes (8): create_venue_ranker(), _matches_tier(), VenueRanker — Tier dictionary e boost score per venue accademiche.  Assegna un t, Restituisce il tier di autorevolezza della venue.          Args:             ven, Restituisce il boost da sommare allo score RRF.          Args:             venue, Crea un VenueRanker con parametri dal config.yaml (se disponibile).      Legge `, Assegna tier e boost per venue/journal accademiche.      I boost possono essere, VenueRanker

### Community 24 - "Prompt Loading"
Cohesion: 0.24
Nodes (9): get_prompt(), list_prompts(), _load_prompts(), Carica i prompt template dal file centralizzato ``prompts/writer_pipeline.md``., Parsa il file prompts, ritornando ``{KEY: template_text}``., Ritorna il template di un prompt per chiave.      Il testo ritornato contiene pl, Svuota la cache e ricarica i prompt da disco., Ritorna la lista delle chiavi prompt disponibili. (+1 more)

### Community 25 - "Quality Metrics"
Cohesion: 0.29
Nodes (7): compute_bpb(), compute_survey_quality_report(), estimate_bpb_from_token_count(), Metriche di qualità per il testo generato dal survey.  Implementa la metrica BPB, Stima il BPB usando il rapporto token/byte (senza modello locale).      Questa è, Genera un report di qualità per il testo del survey.      Calcola metriche aggre, Calcola bits-per-byte per il testo generato usando un modello locale.      BPB =

### Community 26 - "Device Detection"
Cohesion: 0.25
Nodes (7): get_autocast_dtype(), get_device(), move_to_device(), Utilities per il rilevamento automatico del dispositivo di calcolo.  Questo modu, Rileva automaticamente il miglior dispositivo disponibile.      Priorità:, Restituisce il dtype ottimale per ``torch.autocast`` sul dispositivo dato., Sposta un tensor sul dispositivo target in modo ottimale.      Su CUDA utilizza

### Community 27 - "Skill Loading"
Cohesion: 0.33
Nodes (5): list_skills(), load_skills(), Evolution — Skill Loader: carica skill persistenti da file Markdown.  Le skill s, Carica tutti i file .md dalla directory skill e li concatena.      Args:, Elenca le skill disponibili con metadata.      Returns:         Lista di dict co

### Community 28 - "LLM Provider Config"
Cohesion: 1.0
Nodes (1): Provider LLM (sempre 'google' nel nuovo sistema).

### Community 29 - "Flash Model Config"
Cohesion: 1.0
Nodes (1): Modello LLM veloce (Flash). Backward compatible.

### Community 30 - "Pro Model Config"
Cohesion: 1.0
Nodes (1): Modello LLM potente (Pro). Backward compatible.

### Community 31 - "Temperature Default"
Cohesion: 1.0
Nodes (1): Temperatura default (Flash). Backward compatible.

### Community 32 - "Search Results Config"
Cohesion: 1.0
Nodes (1): Max risultati per fonte di ricerca. Backward compatible.

### Community 33 - "ArXiv Categories"
Cohesion: 1.0
Nodes (1): Categorie ArXiv di default. Backward compatible.

### Community 34 - "HTTP Delay Config"
Cohesion: 1.0
Nodes (1): Delay tra richieste HTTP. Backward compatible.

### Community 35 - "Playwright Delay Config"
Cohesion: 1.0
Nodes (1): Delay per operazioni Playwright. Backward compatible.

### Community 36 - "Download Directory Config"
Cohesion: 1.0
Nodes (1): Directory download. Backward compatible.

### Community 37 - "Download Timeout Config"
Cohesion: 1.0
Nodes (1): Timeout download. Backward compatible.

### Community 38 - "Headless Mode Config"
Cohesion: 1.0
Nodes (1): Playwright headless mode. Backward compatible.

### Community 39 - "Limiter Stats"
Cohesion: 1.0
Nodes (1): Statistiche correnti del limiter.

### Community 40 - "Active Request Count"
Cohesion: 1.0
Nodes (1): Numero di richieste attualmente attive.

### Community 41 - "Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 42 - "Active API Key"
Cohesion: 1.0
Nodes (1): Restituisce la API key attualmente attiva.

### Community 43 - "Key Label"
Cohesion: 1.0
Nodes (1): Restituisce l'etichetta della key attiva (FREE/PAID).

### Community 44 - "Key Rotation Check"
Cohesion: 1.0
Nodes (1): True se esiste un'altra key dopo quella corrente.

### Community 45 - "BibTeX Master Dict"
Cohesion: 1.0
Nodes (1): Dict chiave → entry dal master .bib.

### Community 46 - "BibTeX Used Keys"
Cohesion: 1.0
Nodes (1): Set di chiavi BibTeX usate nel survey.

### Community 47 - "Filtered BibTeX Entries"
Cohesion: 1.0
Nodes (1): Dict chiave → entry filtrate (solo quelle usate e trovate).

### Community 48 - "Missing BibTeX Keys"
Cohesion: 1.0
Nodes (1): Chiavi citate nel survey ma non presenti nel master .bib.

### Community 49 - "Invalid BibTeX Entries"
Cohesion: 1.0
Nodes (1): Chiavi con entry non valide → lista di problemi.

### Community 50 - "Citation Counts"
Cohesion: 1.0
Nodes (1): Conteggio occorrenze per chiave nel testo del survey.

### Community 51 - "Checkpoint Info Factory"
Cohesion: 1.0
Nodes (1): Crea un CheckpointInfo leggendo il file checkpoint.          Args:             p

### Community 52 - "Venue Matching"
Cohesion: 1.0
Nodes (1): Verifica se la venue normalizzata matcha (substring) uno degli alias del tier.

## Knowledge Gaps
- **281 isolated node(s):** `Configuration settings for PaperHunter / LLM Security Survey Generator.  Archite`, `Configurazione modelli Gemini (LLM + Embedding).`, `Configurazione GROBID Docker per estrazione PDF strutturata.`, `Configurazione Neo4j Community Edition (opzionale, per GraphRAG).`, `Configurazione estrazione e descrizione figure dai PDF.` (+276 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `LLM Provider Config`** (1 nodes): `Provider LLM (sempre 'google' nel nuovo sistema).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Flash Model Config`** (1 nodes): `Modello LLM veloce (Flash). Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pro Model Config`** (1 nodes): `Modello LLM potente (Pro). Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Temperature Default`** (1 nodes): `Temperatura default (Flash). Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Search Results Config`** (1 nodes): `Max risultati per fonte di ricerca. Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ArXiv Categories`** (1 nodes): `Categorie ArXiv di default. Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `HTTP Delay Config`** (1 nodes): `Delay tra richieste HTTP. Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Playwright Delay Config`** (1 nodes): `Delay per operazioni Playwright. Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Download Directory Config`** (1 nodes): `Directory download. Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Download Timeout Config`** (1 nodes): `Timeout download. Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Headless Mode Config`** (1 nodes): `Playwright headless mode. Backward compatible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Limiter Stats`** (1 nodes): `Statistiche correnti del limiter.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Active Request Count`** (1 nodes): `Numero di richieste attualmente attive.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Active API Key`** (1 nodes): `Restituisce la API key attualmente attiva.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Key Label`** (1 nodes): `Restituisce l'etichetta della key attiva (FREE/PAID).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Key Rotation Check`** (1 nodes): `True se esiste un'altra key dopo quella corrente.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `BibTeX Master Dict`** (1 nodes): `Dict chiave → entry dal master .bib.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `BibTeX Used Keys`** (1 nodes): `Set di chiavi BibTeX usate nel survey.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Filtered BibTeX Entries`** (1 nodes): `Dict chiave → entry filtrate (solo quelle usate e trovate).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Missing BibTeX Keys`** (1 nodes): `Chiavi citate nel survey ma non presenti nel master .bib.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Invalid BibTeX Entries`** (1 nodes): `Chiavi con entry non valide → lista di problemi.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Citation Counts`** (1 nodes): `Conteggio occorrenze per chiave nel testo del survey.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Checkpoint Info Factory`** (1 nodes): `Crea un CheckpointInfo leggendo il file checkpoint.          Args:             p`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Venue Matching`** (1 nodes): `Verifica se la venue normalizzata matcha (substring) uno degli alias del tier.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Modulo Indexing — Vector store, BM25 index, hybrid retrieval, reranking.  Compon` connect `Figure Code Generation` to `Document Assembly`, `Graph Community Detection`, `Community Analysis`, `Text Chunking`, `BM25 Search Index`, `Paper Downloading`, `Bibliography Generation`, `Version Management`, `Checkpoint System`, `Cache Invalidation`, `Provenance Tracking`?**
  _High betweenness centrality (0.190) - this node is a cross-community bridge._
- **Why does `QdrantStore` connect `Graph Community Detection` to `Document Assembly`, `Text Chunking`, `BM25 Search Index`, `Paper Downloading`, `Figure Code Generation`, `Cache Invalidation`, `Summary Clustering`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Why does `Paper` connect `Text Chunking` to `Configuration Management`, `Paper Downloading`, `BibTeX Processing`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Are the 274 inferred relationships involving `QdrantStore` (e.g. with `IngestionReport` and `_CheckpointManager`) actually correct?**
  _`QdrantStore` has 274 INFERRED edges - model-reasoned connections that need verification._
- **Are the 241 inferred relationships involving `PaperChunk` (e.g. with `PaperHunter — CLI Principale v2.2  Subcommands:   search      Cerca e scarica nu` and `Inizializza logging con Rich se disponibile.`) actually correct?**
  _`PaperChunk` has 241 INFERRED edges - model-reasoned connections that need verification._
- **Are the 216 inferred relationships involving `Paper` (e.g. with `RelevanceFilter` and `Intelligent Relevance Filter for PaperHunter.  Uses LangChain with LLM to analyz`) actually correct?**
  _`Paper` has 216 INFERRED edges - model-reasoned connections that need verification._
- **Are the 155 inferred relationships involving `L1Summary` (e.g. with `EntityType` and `RelationType`) actually correct?**
  _`L1Summary` has 155 INFERRED edges - model-reasoned connections that need verification._