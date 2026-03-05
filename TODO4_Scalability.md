# TODO4_Scalability.md — Scalabilità 1.000+ paper: RAPTOR, Graph Communities e Agentic RAG

> **Riferimento:** `instructions4.txt`  
> **Prerequisito:** Tutti i task di `TODO.md` e `TODO3.md` completati  
> **Data inizio:** [Data da definire]  
> **Ultimo aggiornamento:** 5 Marzo 2026  
> **Target:** Gestione di 1.000+ paper per categoria senza context rot, tramite RAPTOR (clustering gerarchico), Graph Communities e Agentic RAG multi-step

---

## 🏗️ Architettura Estesa (Nuovi/Modificati)

```
config.yaml                       ← Nuove sezioni: clustering, agents.writer_mode

paperhunter/
├── summarization/
│   ├── clustering_utils.py       ← NUOVO: Algoritmi di clustering semantico (Agglomerative)
│   ├── level1_5_summarizer.py    ← NUOVO: Summary intermedio per cluster di paper
│   └── level2_summarizer.py      ← MODIFICATO: Aggrega i cluster L1.5 invece dei singoli L1
│
├── graphrag/
│   ├── community_detector.py     ← NUOVO: Rilevamento comunità via Neo4j GDS (Louvain/Leiden)
│   └── community_summarizer.py   ← NUOVO: Generazione summary per macro-temi del grafo
│
├── agents/
│   ├── tools/
│   │   └── retrieval_tools.py    ← NUOVO: Tool invocabili dall'agente (vector, BM25, graph)
│   ├── writer_agent.py           ← MODIFICATO: Da RAG statico a ReAct Agent multi-step
│   └── state.py                  ← MODIFICATO: Supporto per scratchpad e tool log
```

---

## 📋 TASK DETTAGLIATI

### FASE 0 — Preparazione e Dipendenze

#### [x] Task 0.1: Aggiornare dipendenze e configurazioni

**File:** `pyproject.toml` (modifica), `requirements.txt` (modifica), `config.yaml` (modifica), `paperhunter/config.py` (modifica)  
**Stato attuale:** Le dipendenze attuali non includono `scikit-learn` né `numpy` (necessari per clustering). In `config.yaml` non esiste una sezione `clustering` e non c'è un parametro `agents.writer_mode`. I modelli Pydantic in `config.py` non validano queste nuove chiavi.  
**Obiettivo:** Inserire le librerie necessarie per il clustering locale e configurare i nuovi parametri nel sistema di configurazione centralizzato.  
**Dettagli:**
- **`pyproject.toml` / `requirements.txt`:** Aggiungere `scikit-learn` e `numpy` per Agglomerative Clustering dei vettori
- **`config.yaml`:** Aggiungere una nuova sezione `clustering`:
  ```yaml
  clustering:
    max_cluster_size: 20          # Massimo numero di paper per cluster L1.5
    similarity_threshold: 0.75    # Soglia similarità coseno per il clustering
  ```
- **`config.yaml`:** Nella sezione `agents`, aggiungere `writer_mode: "agentic"` (per attivare il ReAct loop al posto del RAG statico)
- **`paperhunter/config.py`:** Creare un nuovo sotto-modello `ClusteringConfig(BaseModel)` con i campi `max_cluster_size: int = 20` e `similarity_threshold: float = 0.75`, con validatori Pydantic per range (es. `max_cluster_size ∈ [5, 100]`, `similarity_threshold ∈ (0.0, 1.0)`)
- **`paperhunter/config.py`:** Aggiungere campo `writer_mode: str = "static"` a `AgentsConfig`, con `@field_validator` che accetta solo `"static"` o `"agentic"`
- **`paperhunter/config.py`:** Registrare `ClusteringConfig` nel modello root `Settings`  
**Test:** Eseguire `pip install -e .` e verificare che le configurazioni vengano caricate senza errori, validando i tipi con un mini-script che importa `settings` e stampa tutte le nuove sezioni.

> **Completato:** 5 Marzo 2026  
> **Modifiche apportate:**
> - `pyproject.toml`: Aggiunta sezione `# --- Clustering (RAPTOR) ---` con `scikit-learn>=1.4.0` e `numpy>=1.26.0` nelle dependencies
> - `requirements.txt`: Aggiunta sezione `# === Clustering (RAPTOR) ===` con `scikit-learn>=1.4.0` e `numpy>=1.26.0`
> - `config.yaml`: Aggiunta sezione `clustering:` con `max_cluster_size: 20` e `similarity_threshold: 0.75`; aggiunto `writer_mode: "static"` nella sezione `agents`
> - `paperhunter/config.py`: Creato `ClusteringConfig(BaseModel)` con validatori `@field_validator` per range (`max_cluster_size ∈ [5, 100]`, `similarity_threshold ∈ (0.0, 1.0)`); aggiunto `writer_mode: str = "static"` a `AgentsConfig` con validatore che accetta solo `"static"` o `"agentic"`; registrato `clustering: ClusteringConfig` in `Settings`
> **Uso:**
> ```python
> from paperhunter.config import settings
> settings.clustering.max_cluster_size    # → 20
> settings.clustering.similarity_threshold # → 0.75
> settings.agents.writer_mode              # → "static"
> ```
> **Note:** Il default di `writer_mode` è `"static"` (backward-compatible). Cambiare a `"agentic"` in `config.yaml` dopo il Task 3.2.

---

### FASE 1 — RAPTOR: Summarization Gerarchica (L1 → L1.5 → L2)

#### [x] Task 1.1: Creare `summarization/clustering_utils.py`

**File:** `paperhunter/summarization/clustering_utils.py` (nuovo)  
**Stato attuale:** Non esiste alcun modulo di clustering. Gli L1 Summary vengono passati direttamente all'L2 summarizer, che con 1.000+ paper sfora inevitabilmente il budget token.  
**Obiettivo:** Raggruppare semanticamente gli L1 Summary usando i loro embedding (dal Qdrant store), creando cluster tematici di dimensione controllata.  
**Dettagli:**
- Funzione `cluster_summaries(l1_summaries: list[L1Summary], embeddings: list[list[float]]) -> dict[int, list[L1Summary]]`
  - Recuperare i vettori (embeddings) degli L1Summary dalla collection `l1_summaries` di `QdrantStore`
  - Usare `AgglomerativeClustering` di scikit-learn con metrica `cosine` e linkage `average`
  - Parametro `distance_threshold` derivato da `1 - similarity_threshold` (da `config.yaml`)
  - Se un cluster supera `max_cluster_size`, ri-clusterizzare ricorsivamente quel sotto-gruppo
  - Gestire outlier: paper isolati (cluster di dimensione 1) raggruppati in un cluster "miscellaneous" o lasciati singoli
- Funzione `get_cluster_labels(embeddings: list[list[float]]) -> list[int]`
  - Wrapper leggero che restituisce solo le label di cluster
- Logging con Rich: numero di cluster, distribuzione dimensioni, outlier count  
**Test:** Passare 100 L1 finti con embedding mock (es. 5 gruppi da 20 con piccola varianza intra-gruppo), verificare che vengano restituiti cluster di dimensioni corrette e che nessun cluster superi `max_cluster_size`.

> **Completato:** 5 Marzo 2026  
> **Modifiche apportate:**
> - `paperhunter/summarization/clustering_utils.py` (nuovo): Modulo completo di clustering semantico RAPTOR
> - `test_clustering_utils.py` (nuovo): 16 test unitari e di integrazione
> **Funzioni aggiunte:**
> - `cluster_summaries(l1_summaries, embeddings, max_cluster_size?, similarity_threshold?) -> dict[int, list[L1Summary]]` — Funzione principale: raggruppa L1 Summary via AgglomerativeClustering con ri-clusterizzazione ricorsiva per cluster oversized
> - `get_cluster_labels(embeddings, similarity_threshold?) -> list[int]` — Wrapper leggero che restituisce solo le label
> - `fetch_l1_embeddings_from_qdrant(l1_summaries, qdrant_store?) -> list[list[float]]` — Recupera i vettori embedding dalla collection Qdrant `l1_summaries` (scroll con `with_vectors=True`)
> - Funzioni interne: `_normalize_embeddings`, `_agglomerative_cluster`, `_group_by_labels`, `_split_oversized_clusters`, `_renumber_clusters`, `_log_cluster_stats`
> **Uso:**
> ```python
> from paperhunter.summarization.clustering_utils import cluster_summaries, fetch_l1_embeddings_from_qdrant
> 
> # Recupera embedding da Qdrant
> embeddings = fetch_l1_embeddings_from_qdrant(l1_summaries)
> 
> # Genera cluster
> clusters = cluster_summaries(l1_summaries, embeddings)
> # → {0: [L1Summary, ...], 1: [L1Summary, ...], ...}
> ```
> **Note:** La distanza coseno è implementata tramite normalizzazione L2 + distanza euclidea (`eucl_dist = sqrt(2 * cosine_dist)`). I cluster sono ordinati per dimensione decrescente (cluster più grandi hanno ID più bassi). I vettori zero (embedding mancanti) vengono trattati come outlier.

---

#### [x] Task 1.2: Creare `summarization/level1_5_summarizer.py`

**File:** `paperhunter/summarization/level1_5_summarizer.py` (nuovo), `paperhunter/models.py` (modifica)  
**Stato attuale:** Non esiste un livello intermedio tra L1 e L2. Con 1.000+ paper, l'L2 deve sintetizzare troppe informazioni in un'unica chiamata LLM.  
**Obiettivo:** Generare un meta-summary per ogni cluster (il livello "L1.5"), riducendo da ~1.000 L1 a ~50 cluster summary.  
**Dettagli:**
- **`paperhunter/models.py`:** Aggiungere nuovo modello `L1_5Summary(BaseModel)`:
  - `cluster_id: int` — ID del cluster
  - `category: str` — categoria di appartenenza
  - `paper_ids: list[str]` — lista dei `paper_id` o `bibtex_key` dei paper inclusi
  - `paper_count: int` — numero di paper nel cluster
  - `common_theme: str` — tema comune identificato dal LLM
  - `shared_methodologies: str` — metodologie condivise
  - `key_findings: str` — risultati principali aggregati
  - `testing_overview: str` — panoramica delle metodologie di testing
  - `temporal_span: str` — range temporale (es. "2022-2025")
  - `generated_at: str` — timestamp ISO
- Funzione `generate_cluster_summary(cluster_id: int, summaries: list[L1Summary], category: str) -> L1_5Summary`
  - Prompt Gemini Pro: sintetizzare il tema comune del cluster, estrarre le metodologie condivise, mappare i paper inclusi
  - Prompt template con istruzioni per: mantenere le citation key `[bibtex_key]`, identificare convergenze e divergenze, produrre un testo coeso
- Funzione `generate_all_cluster_summaries(clusters: dict[int, list[L1Summary]], category: str) -> list[L1_5Summary]`
  - Batch processing con progress bar Rich
  - Rate limiting via `get_gemini_limiter()`
- Salvataggio output in una nuova collection Qdrant `l1_5_summaries` (aggiungere in `config.yaml` → sezione `qdrant`)  
**Test:** Generare un summary L1.5 per un cluster mock di 15 paper e verificare la presenza di tutti i campi chiave in output (JSON). Verificare che venga salvato in Qdrant.

> **Completato:** 5 Marzo 2026  
> **Modifiche apportate:**
> - `paperhunter/models.py` (modifica): Aggiunto modello `L1_5Summary(BaseModel)` con tutti i campi richiesti: `cluster_id`, `category`, `paper_ids`, `paper_count`, `common_theme`, `shared_methodologies`, `key_findings`, `testing_overview`, `temporal_span`, `generated_at`, `model_used`
> - `paperhunter/summarization/level1_5_summarizer.py` (nuovo): Modulo completo per generazione meta-summary L1.5 per cluster RAPTOR
> - `config.yaml` (modifica): Aggiunta `collection_l1_5_summaries: "l1_5_summaries"` nella sezione `qdrant`
> - `paperhunter/config.py` (modifica): Aggiunto campo `collection_l1_5_summaries: str = "l1_5_summaries"` a `QdrantConfig`
> - `paperhunter/indexing/qdrant_store.py` (modifica): Aggiunto `COLLECTION_L1_5` class attribute, registrato in `_ensure_collections()` e `get_stats()`
> - `paperhunter/summarization/__init__.py` (modifica): Aggiornati exports con `generate_cluster_summary`, `generate_all_cluster_summaries`, `load_l1_5_summaries_for_category`, `count_l1_5_for_category`
> - `test_l1_5_summarizer.py` (nuovo): 28 test unitari e di integrazione
> **Funzioni aggiunte:**
> - `generate_cluster_summary(cluster_id, summaries, category, force_model?, save_to_store?, store?) -> L1_5Summary` — Genera un singolo meta-summary L1.5 per un cluster di L1 usando Gemini Pro
> - `generate_all_cluster_summaries(clusters, category, force_model?, save_to_store?, store?, skip_existing?) -> list[L1_5Summary]` — Batch processing di tutti i cluster con progress bar Rich e rate limiting
> - `load_l1_5_summaries_for_category(category, store?) -> list[L1_5Summary]` — Carica tutti gli L1.5 di una categoria dal Qdrant store (scroll + ricostruzione)
> - `count_l1_5_for_category(category, store?) -> int` — Conta rapidamente quanti L1.5 esistono per una categoria
> - Funzioni interne: `_format_l1_for_cluster_prompt`, `_compute_temporal_span`, `_extract_paper_ids`, `_parse_l1_5_json_response`, `_validate_l1_5_summary`, `_save_l1_5_to_store`, `_l1_5_exists_in_store`, `_make_l1_5_chunk_id`, `_l1_5_summary_to_text`, `_reconstruct_l1_5_from_payload`, `_extract_max_year`, `_print_batch_report`
> **Uso:**
> ```python
> from paperhunter.summarization.clustering_utils import cluster_summaries, fetch_l1_embeddings_from_qdrant
> from paperhunter.summarization.level1_5_summarizer import (
>     generate_all_cluster_summaries,
>     load_l1_5_summaries_for_category,
> )
> 
> # 1. Clustering degli L1
> embeddings = fetch_l1_embeddings_from_qdrant(l1_summaries)
> clusters = cluster_summaries(l1_summaries, embeddings)
> 
> # 2. Generazione L1.5 (salva automaticamente in Qdrant)
> l1_5_summaries = generate_all_cluster_summaries(clusters, category="Prompt Injection")
> 
> # 3. Caricamento da Qdrant (per uso successivo nell'L2)
> loaded = load_l1_5_summaries_for_category("Prompt Injection")
> ```
> **Note:** Il prompt L1.5 istruisce il LLM a: mantenere le citazioni [bibtex_key], identificare convergenze e divergenze, produrre un testo coeso. La validazione controlla che i campi siano non vuoti e che contengano almeno una citazione. Il salvataggio in Qdrant usa la collection `l1_5_summaries` con chunk_id deterministico basato su cluster_id + category.

---

#### [x] Task 1.3: Aggiornare `summarization/level2_summarizer.py`

**File:** `paperhunter/summarization/level2_summarizer.py` (modifica)  
**Stato attuale:** `_format_l1_summaries_for_prompt()` carica tutti gli L1 e applica una compressione progressiva per recency tier. Con 1.000+ paper, anche la compressione all-compressed non basta: bisogna passare dai cluster L1.5.  
**Obiettivo:** Passare dal progressivo downgrade degli L1 all'uso dei cluster L1.5 come input principale per l'L2.  
**Dettagli:**
- **Nuova logica:** Invece di caricare e comprimere 1.000 L1, la funzione ora:
  1. Verifica se esistono summary L1.5 per la categoria nella collection Qdrant `l1_5_summaries`
  2. Se esistono: carica i ~50 cluster summary L1.5, che rientrano facilmente nel budget di 150k token
  3. Se non esistono (fallback): usa la logica attuale di compressione recency-tier sugli L1 (backward compatibility)
- **Nuova funzione:** `_format_l1_5_summaries_for_prompt(l1_5_summaries: list[L1_5Summary]) -> str`
  - Formatta i ~50 cluster summary per il prompt dell'L2
  - Include per ogni cluster: tema, paper count, methodologies, key findings, testing overview
  - Ordinamento per temporal_span (cluster più recenti prima)
- **Rimozione/deprecazione:** La logica di `_format_l1_compressed()` e `_format_l1_reduced()` resta come fallback ma non è più il path principale
- **Prompt L2 aggiornato:** Aggiungere nota che i summary provengono da cluster tematici, istruendo l'LLM a cercare pattern cross-cluster
- Logging del budget consumato: mostrare quanti token usano i cluster L1.5 vs il budget disponibile  
**Test:** Generare un L2 partendo da 50 cluster L1.5 mock e verificare che rientri perfettamente nel budget token. Verificare anche il fallback all'L1 diretto se non ci sono L1.5.

> **Completato:** 5 Marzo 2026  
> **Modifiche apportate:**
> - `paperhunter/summarization/level2_summarizer.py` (modifica): Import aggiunto per `L1_5Summary`; nuovo prompt template `_L2_FROM_L1_5_PROMPT_TEMPLATE` per il path RAPTOR; `generate_l2_summary()` ora controlla l'esistenza di L1.5 summary prima di procedere con il path legacy
> - Nuova funzione `_generate_l2_from_l1_5()`: genera L2 dal RAPTOR path (cluster L1.5), con budget check, LLM call con retry/fallback, metadata RAPTOR su disco
> - Nuova funzione `_format_l1_5_summaries_for_prompt()`: formatta i cluster L1.5 per il prompt L2, ordinati per temporal_span (più recenti prima), con blocchi strutturati (theme, methodologies, key findings, testing, paper IDs)
> - Nuova funzione `_truncate_l1_5_for_budget()`: troncamento progressivo in 3 fasi (campi secondari → tutti i campi → hard truncation) per rientrare nel budget
> - `test_l2_with_l1_5.py` (nuovo): 10 test unitari e di integrazione con mock LLM
> **Funzioni aggiunte:**
> - `_generate_l2_from_l1_5(category, l1_5_summaries, force_model?, save_to_store?, store?, version?) -> L2Summary` — Genera L2 dal RAPTOR path con cluster L1.5
> - `_format_l1_5_summaries_for_prompt(l1_5_summaries) -> str` — Formatta i cluster L1.5 per il prompt L2, ordinati per temporal_span decrescente
> - `_truncate_l1_5_for_budget(l1_5_summaries, max_chars) -> str` — Tronca progressivamente i campi L1.5 per rientrare nel budget
> - `_L2_FROM_L1_5_PROMPT_TEMPLATE` — Prompt template dedicato al RAPTOR path, con istruzioni per pattern cross-cluster
> **Logica di routing in `generate_l2_summary()`:**
> 1. Carica `store` (QdrantStore) se non fornito
> 2. Chiama `load_l1_5_summaries_for_category(category)` per verificare se esistono L1.5
> 3. Se L1.5 presenti → `_generate_l2_from_l1_5()` (RAPTOR path)
> 4. Se L1.5 assenti → path legacy con recency tiers sugli L1 (backward-compatible)
> **Uso:**
> ```python
> from paperhunter.summarization.level2_summarizer import generate_l2_summary
> 
> # Auto-detection: usa L1.5 se disponibili, altrimenti fallback a L1
> l2 = generate_l2_summary("Prompt Injection", save_to_store=True)
> 
> # Forzare il path legacy passando l1_summaries esplicitamente
> # (l'auto-detection controlla comunque prima i L1.5)
> l2 = generate_l2_summary("Prompt Injection", l1_summaries=my_l1_list)
> ```
> **Metadata su disco:** Il path RAPTOR salva metadata con `path: "raptor_l1_5"`, `cluster_count`, `total_paper_count`, `avg_papers_per_cluster`, `l1_5_chars`, `budget_pct` nel file `output/l2_metadata/<category>.json`.
> **Stress test:** Con 50 cluster mock (~4.425 paper simulati), il testo L1.5 occupa ~59k chars su un budget di ~588k chars (9%), confermando che il RAPTOR path risolve il problema di budget per 1.000+ paper.
> **Note:** Il path legacy con recency tiers resta intatto come fallback. Il prompt RAPTOR istruisce il LLM a cercare pattern cross-cluster e a citare paper individuali con [bibtex_key]. La truncation progressiva (3 fasi) garantisce che anche scenari edge con cluster molto verbosi rientrino nel budget.

---

### FASE 2 — Community-Driven GraphRAG

#### [ ] Task 2.1: Implementare Community Detection nel Grafo

**File:** `paperhunter/graphrag/community_detector.py` (nuovo), `paperhunter/graphrag/neo4j_manager.py` (modifica)  
**Stato attuale:** Il grafo Neo4j contiene entità e relazioni estratte dai paper (via `entity_extractor.py` e `neo4j_manager.py`), ma non ha alcun meccanismo per identificare macro-temi strutturali. Le query GraphRAG attualmente cercano singole entità senza contesto di comunità.  
**Prerequisito:** Neo4j con plugin GDS (Graph Data Science) installato nel container Docker.  
**Obiettivo:** Identificare macro-temi strutturali tra le entità estratte, partizionando il grafo in comunità tematiche.  
**Dettagli:**
- **`paperhunter/graphrag/neo4j_manager.py` (modifica):**
  - Aggiungere metodo `ensure_gds_plugin() -> bool`: verifica che il plugin GDS sia disponibile (call `gds.version()` o equivalente Cypher)
  - Aggiungere metodo `project_graph(graph_name: str)`: crea una proiezione GDS del grafo (nodi Entity, relazioni RELATED_TO con pesi)
  - Aggiungere metodo `drop_graph_projection(graph_name: str)`: cleanup proiezione GDS
- **`paperhunter/graphrag/community_detector.py` (nuovo):**
  - Classe `CommunityDetector`:
    - `__init__(neo4j_manager)`: inizializza con istanza Neo4j
    - `detect_communities(algorithm: str = "louvain") -> list[dict]`: esegue l'algoritmo Louvain o Leiden via Cypher GDS. Assegna a ogni nodo un `community_id` come proprietà in Neo4j
    - `get_community_subgraph(community_id: int) -> dict`: estrae tutti i nodi e relazioni di una comunità specifica. Output: `{"community_id": int, "nodes": [...], "edges": [...], "size": int}`
    - `get_all_communities(min_size: int = 3) -> list[dict]`: restituisce tutte le comunità con almeno `min_size` nodi
    - `get_community_stats() -> dict`: numero comunità, distribuzione dimensioni, modularity score
  - Supporto per entrambi Louvain e Leiden (configurabile)
  - Filtro: ignorare comunità con < N nodi (soglia configurabile)
- Logging con Rich: numero di comunità trovate, distribuzione dimensioni, top-5 comunità per size  
**Test:** Eseguire la detection su un grafo di test con ~300 entità (mock o reale) e verificare che i nodi vengano taggati con `community_id`. Verificare che le comunità restituite contengano sotto-grafi validi.

---

#### [ ] Task 2.2: Generare Community Summaries

**File:** `paperhunter/graphrag/community_summarizer.py` (nuovo)  
**Stato attuale:** Il grafo contiene comunità identificate (dopo Task 2.1) ma non esiste un summary testuale che ne descriva la narrativa, rendendo difficile l'uso delle comunità nel retrieval.  
**Obiettivo:** Trasformare il sotto-grafo di ciascuna comunità in un testo narrativo comprensibile dall'agente scrittore, salvandolo in Qdrant per il retrieval.  
**Dettagli:**
- Classe `CommunitySummarizer`:
  - `__init__(community_detector, qdrant_store)`: prende i componenti necessari
  - `summarize_community(community_id: int, subgraph: dict) -> str`
    - Formatta il sotto-grafo come lista di entità + relazioni in testo strutturato
    - Prompt Gemini Pro: spiegare la narrativa della comunità (es. "Questa comunità rappresenta l'evoluzione degli attacchi jailbreak dal 2023 al 2025, con le entità X, Y, Z connesse tramite...")
    - Output: testo narrativo di 200-500 parole con menzione delle entità chiave
  - `summarize_all_communities(communities: list[dict]) -> list[dict]`
    - Batch processing con progress bar Rich e rate limiting
    - Restituisce `[{"community_id": int, "summary": str, "node_count": int, "key_entities": list[str]}]`
  - `save_to_qdrant(summaries: list[dict])`:
    - Salva ogni summary come punto nella nuova collection Qdrant `graph_communities`
    - Metadata: `community_id`, `node_count`, `key_entities`
- **`config.yaml`:** Aggiungere `qdrant.collection_graph_communities: "graph_communities"` nella sezione `qdrant`
- **`config.py`:** Aggiungere il campo corrispondente a `QdrantConfig`  
**Test:** Fornire un sotto-grafo mock (20 nodi, 30 relazioni), generare il testo narrativo, verificare che contenga le entità chiave. Verificare il salvataggio in Qdrant.

---

### FASE 3 — Agentic Multi-Step Writer (ReAct Loop)

#### [ ] Task 3.1: Trasformare i Retriever in Tool invocabili

**File:** `paperhunter/agents/tools/retrieval_tools.py` (nuovo), `paperhunter/agents/tools/__init__.py` (nuovo)  
**Stato attuale:** Il writer agent riceve un contesto pre-calcolato (iniettato tramite `_format_context_for_prompt()` con fino a 60.000 chars). Non ha alcuna capacità di cercare autonomamente informazioni aggiuntive durante la scrittura.  
**Obiettivo:** Fornire all'agente scrittore degli strumenti invocabili autonomamente per cercare informazioni on-demand durante il ReAct loop.  
**Dettagli:**
- Creare directory `paperhunter/agents/tools/` con `__init__.py`
- Creare `@tool` functions (utilizzando le astrazioni di LlamaIndex `FunctionTool`):
  - `search_vector_database(query: str, category: str | None = None, year_min: int | None = None) -> str`
    - Chiama `QdrantStore.query()` con i filtri forniti
    - Restituisce i top-5 chunk formattati come testo con `[bibtex_key]` per ogni chunk
  - `search_keyword_index(query: str, top_k: int = 5) -> str`
    - Chiama `BM25Index.search()` per ricerca keyword esatta
    - Restituisce i risultati formattati con citazioni
  - `search_graph_communities(topic: str) -> str`
    - Cerca tra i summary delle comunità in Qdrant collection `graph_communities`
    - Restituisce i community summary più rilevanti per il topic
- Ogni tool deve:
  - Restituire una stringa di testo concisa (max ~2000 chars)
  - Includere `[bibtex_key]` per ogni fonte citata
  - Gestire errori gracefully (restituire messaggio di errore informativo)
  - Loggare la chiamata (query, n_risultati, tempo)
- Funzione helper `get_all_retrieval_tools() -> list[FunctionTool]`: restituisce tutti i tool pronti per l'agente  
**Test:** Chiamare ogni tool manualmente con una query string di esempio e verificare il formato dell'output (testo con citazioni). Verificare che gli errori vengano gestiti correttamente (es. Qdrant non disponibile).

---

#### [ ] Task 3.2: Aggiornare `agents/writer_agent.py` con logica ReAct

**File:** `paperhunter/agents/writer_agent.py` (modifica)  
**Stato attuale:** Il writer agent usa RAG pre-calcolato: `_format_context_for_prompt()` raccoglie fino a 60.000 chars di contesto, lo inietta nel prompt, e il LLM scrive la sezione in un singolo passo. Questa logica funziona bene per ~100 paper ma non scala a 1.000+ perché il contesto pre-calcolato non può coprire tutti gli aspetti necessari.  
**Obiettivo:** Passare dal RAG pre-calcolato a un agente autonomo che si documenta passo-passo tramite ReAct loop, cercando informazioni on-demand.  
**Dettagli:**
- **Modalità duale** controllata da `config.yaml` → `agents.writer_mode`:
  - `"static"` (default): comportamento attuale invariato (backward compatibility)
  - `"agentic"`: nuovo comportamento ReAct descritto sotto
- **Logica `"agentic"`:**
  - Abbandonare l'iniezione fissa del contesto (`_format_context_for_prompt()`)
  - Inizializzare un ReAct Agent (via LlamaIndex `ReActAgent` o `AgentRunner`)
  - Registrare i tool dal Task 3.1 (`get_all_retrieval_tools()`)
  - Prompt base: "Devi scrivere la sezione '{section_title}' del survey su '{survey_topic}'. Usa i tuoi tool per cercare le informazioni necessarie. Non inventare nulla. Aggiungi citazioni [bibtex_key] per ogni claim."
  - L'agente ragiona: `Thought → Action → Observation → ... → Final Answer`
  - Max iterazioni configurabili: `max_iterations: 5-7` (evitare loop infiniti)
- **Logging:** Loggare ogni ciclo Thought → Action → Observation tramite Rich console (opzionale, controllato da `logging.level`)
- **Fallback:** Se il ReAct agent fallisce (timeout, errore, loop infinito), fallback alla modalità `"static"`
- **Preservare** tutti i metodi e le firme esistenti: `write_section()`, `write_all_sections()`, `write()` devono continuare a funzionare con lo stesso contratto esterno  
**Test:** Eseguire l'agente in modalità `"agentic"` per una sezione isolata (con Qdrant e BM25 mock o reali). Loggare il ciclo completo Thought → Action → Observation nel terminale. Verificare il fallback a `"static"` con `writer_mode="static"`.

---

#### [ ] Task 3.3: Gestione dello State Scratchpad

**File:** `paperhunter/agents/state.py` (modifica)  
**Stato attuale:** `SurveyState` (TypedDict per LangGraph) contiene: `outline`, `sections`, `feedback`, `provenance_records`, `bibliography_entries`, `checkpoints`, `run_metadata`. Non traccia le fonti consultate dall'agente durante il ReAct loop, rendendo difficile per il Critic Agent verificare la provenance.  
**Obiettivo:** Mantenere traccia delle fonti trovate dall'agente durante il ReAct loop, garantendo che il Critic Agent possa verificare la provenance.  
**Dettagli:**
- In `SurveyState`, aggiungere:
  - `current_scratchpad: list[str]` — lista di note e osservazioni accumulate dall'agente durante le iterazioni ReAct (azzerata a ogni nuova sezione)
  - `retrieved_facts: dict[str, list[dict]]` — mappa `{section_title: [{"query": str, "tool": str, "results": str, "timestamp": str}]}` — log completo delle ricerche effettuate dall'agente per ogni sezione
  - `agentic_stats: dict` — statistiche aggregate: totale tool calls, totale iterazioni, tempo speso per sezione
- **Integrazione con Critic Agent:** `critic_agent.py` deve avere accesso a `retrieved_facts` per verificare che le citazioni nella sezione scritta corrispondano effettivamente a fonti trovate dall'agente. Non modificare il critic_agent in questo task, ma assicurarsi che lo stato sia accessibile
- **Serializzazione:** Verificare che i nuovi campi siano serializzabili JSON per il `CheckpointManager` (evitare oggetti non-serializzabili come datetime)
- **Reset per sezione:** Aggiungere helper `reset_scratchpad(state)` che azzera `current_scratchpad` e prepara `retrieved_facts` per una nuova sezione  
**Test:** Creare un `SurveyState` con i nuovi campi, popolare `retrieved_facts` e `current_scratchpad` con dati di test, salvare/caricare tramite `CheckpointManager`, verificare integrità dei dati dopo la deserializzazione JSON.

---

## 📌 Note per le Sessioni Future

- **Una sessione = un task.** Non anticipare mai il codice del task successivo.
- **Ogni task è testabile in isolamento.** Fornire sempre un mini-test alla fine.
- **Aggiornare questo file** mettendo `[x]` al task completato e aggiungendo note sotto il task (come in TODO.md e TODO3.md).
- **Se un task richiede modifiche a task precedenti**, documentare nella sezione note del task.
- **Formato chiusura sessione:** "Task completato con successo. Per favore, apri una nuova chat, incolla il master prompt e il nuovo TODO.md aggiornato, e chiedimi di eseguire il prossimo task. Questo manterrà la mia memoria fresca e il codice pulito."
- **Git commit** dopo ogni task: `git add -A && git commit -m "descrizione della modifica"`

---

## Riepilogo

| ID | Descrizione | Complessità | File principali | Dipendenze |
|----|------------|-------------|-----------------|------------|
| 0.1 | Config clustering + writer_mode | Bassa | `pyproject.toml`, `config.yaml`, `config.py` | Nessuna |
| 1.1 | Clustering semantico L1 | Media | `summarization/clustering_utils.py` (nuovo) | 0.1 |
| 1.2 | L1.5 Summarizer per cluster | Media | `summarization/level1_5_summarizer.py` (nuovo), `models.py` | 1.1 |
| 1.3 | L2 aggregazione cluster L1.5 | Alta | `summarization/level2_summarizer.py` (modifica) | 1.2 |
| 2.1 | Community Detection (Louvain/Leiden) | Media | `graphrag/community_detector.py` (nuovo), `graphrag/neo4j_manager.py` | 0.1 |
| 2.2 | Community Summaries → Qdrant | Media | `graphrag/community_summarizer.py` (nuovo) | 2.1 |
| 3.1 | Retrieval Tools per ReAct Agent | Media | `agents/tools/retrieval_tools.py` (nuovo) | 0.1 |
| 3.2 | Writer Agent ReAct loop | Alta | `agents/writer_agent.py` (modifica) | 3.1 |
| 3.3 | State Scratchpad + retrieved_facts | Bassa | `agents/state.py` (modifica) | 3.2 |

---

> **Flusso di dipendenze:**
> - Task 0.1 → (FASE 1 e FASE 2 e FASE 3 in parallelo)
> - FASE 1: 1.1 → 1.2 → 1.3
> - FASE 2: 2.1 → 2.2
> - FASE 3: 3.1 → 3.2 → 3.3
> - Le tre fasi sono indipendenti tra loro (possono procedere in parallelo dopo 0.1)
