# TODO3.md — Miglioramenti per scalabilità, deduplicazione download e priorità recency

> **Riferimento:** `instructions3.txt`  
> **Data:** 2 Marzo 2026  
> **Prerequisito:** Tutti i 36 task del TODO.md originale completati

---

## Requisito 0 — Configurabilità: papers_per_category e budget adattivo

Il sistema deve essere **modulabile**: il numero di paper per categoria è scelto dall'utente in `config.yaml`, e tutti i componenti a valle (L1, L2, retrieval, writer) si adattano automaticamente.

### [x] Task 0.1: Nuovi parametri in config.yaml e config.py

**File:** `config.yaml` (modifica), `paperhunter/config.py` (modifica)  
**Stato attuale:** Non esiste un parametro `papers_per_category` né un budget token globale. I limiti sono hardcoded nei singoli moduli (`max_chunk_chars=50_000`, `max_chars=60_000`, `l1_max_tokens=2000`, ecc.).  
**Obiettivo:** Aggiungere i parametri centralizzati da cui tutti i moduli derivano i propri limiti.  
**Dettagli — nuovi campi `config.yaml`:**
```yaml
ingestion:
  papers_per_category: 100          # Numero target di paper per categoria

summarization:
  max_context_tokens: 150000        # Budget token massimo per singola chiamata LLM
  l1_target_chars_auto: true        # Se true, calcola target L1 chars in base a papers_per_category
  l1_min_chars: 2000                # Floor: un L1 non sarà mai più corto di così
  l2_recency_tiers:                 # Soglie di compressione per recency (anni fa)
    full: 2                         # Paper ≤2 anni: testo L1 completo
    reduced: 4                      # Paper 2-4 anni: ~70%
    compressed: 999                 # Paper >4 anni: solo key_contributions + testing

agents:
  writer_max_context_chars: 60000   # Max chars contesto per singola sezione writer
  writer_budget_mode: "proportional" # "fixed" | "proportional" (divide budget per N sezioni)
```
- In `config.py`: aggiungere i campi Pydantic corrispondenti con validazione
- I moduli a valle leggono **sempre** da config; nessun hardcode residuo

> **Completato 2/03/2026.** Modifiche:
> - **`config.yaml`:** Aggiunti 8 nuovi parametri:
>   - `ingestion.papers_per_category: 100` — numero target di paper per categoria
>   - `summarization.max_context_tokens: 150000` — budget token massimo per chiamata LLM
>   - `summarization.l1_target_chars_auto: true` — calcolo automatico target L1 chars
>   - `summarization.l1_min_chars: 2000` — floor per L1 summary length
>   - `summarization.l2_recency_tiers` — oggetto con `full: 2`, `reduced: 4`, `compressed: 999`
>   - `retrieval.recency_boost: 0.1` — boost temporale post-RRF
>   - `agents.writer_max_context_chars: 60000` — max chars contesto writer
>   - `agents.writer_budget_mode: "proportional"` — modalità budget writer
> - **`paperhunter/config.py`:** Aggiunti modelli Pydantic corrispondenti:
>   - `L2RecencyTiersConfig(BaseModel)` — nuovo sotto-modello con `full`, `reduced`, `compressed`
>   - `IngestionConfig.papers_per_category: int = 100`
>   - `SummarizationConfig` → +4 campi: `max_context_tokens`, `l1_target_chars_auto`, `l1_min_chars`, `l2_recency_tiers`
>   - `RetrievalConfig.recency_boost: float = 0.1`
>   - `AgentsConfig` → +2 campi: `writer_max_context_chars`, `writer_budget_mode` con `@field_validator`
> - **Accesso dai moduli:** `settings.ingestion.papers_per_category`, `settings.summarization.max_context_tokens`, `settings.summarization.l2_recency_tiers.full`, `settings.agents.writer_max_context_chars`, `settings.agents.writer_budget_mode`, `settings.retrieval.recency_boost`
> - **Test:** Tutti i nuovi parametri caricati correttamente da YAML, validazione Pydantic funzionante (es. `writer_budget_mode='invalid'` → `ValueError`)

---

## Requisito 1 — Supporto ~N paper/categoria con budget ≤ 150k token

Il sistema attuale è stato testato con ~20 paper totali. Scalare a ~400 paper (N×4 categorie, N configurabile) richiede un controllo esplicito del budget token per evitare context rot e hallucination.

### [x] Task A.1: Token counter utility

**File:** `paperhunter/utils/token_counter.py` (nuovo)  
**Obiettivo:** Utility centralizzata per conteggio token (Gemini usa ~4 char/token come approssimazione, oppure `tiktoken`/tokenizer nativo).  
**Dettagli:**
- `estimate_tokens(text: str) → int`: stima token da testo
- `estimate_tokens_for_messages(messages) → int`: stima per prompt LLM completo
- `get_max_context_tokens() → int`: legge `summarization.max_context_tokens` da config
- `chars_budget_for_n_papers(n_papers: int) → int`: calcola il budget in chars per L1/L2 in funzione di N

> **Completato 2/03/2026.** File: `paperhunter/utils/token_counter.py`
> - Costanti: `CHARS_PER_TOKEN = 4.0`, `PROMPT_OVERHEAD_TOKENS = 2_000`, `L1_TIER_STANDARD_MAX = 30`, `L1_TIER_CONCISE_MAX = 70`
> - **Funzioni base:**
>   - `estimate_tokens(text, chars_per_token=4.0) → int`: stima token da testo (ceil(len/4))
>   - `estimate_tokens_for_messages(messages, chars_per_token, per_message_overhead=4) → int`: stima per lista di messaggi LLM con overhead per-messaggio
>   - `tokens_to_chars(tokens) → int`: conversione token → chars
>   - `chars_to_tokens(chars) → int`: conversione chars → token
> - **Getters config (lazy import):**
>   - `get_max_context_tokens() → int`: legge `summarization.max_context_tokens`
>   - `get_papers_per_category() → int`: legge `ingestion.papers_per_category`
>   - `get_writer_max_context_chars() → int`: legge `agents.writer_max_context_chars`
>   - `get_writer_budget_mode() → str`: legge `agents.writer_budget_mode`
> - **Budget L1:**
>   - `compute_l1_target_chars(n_papers, max_context_tokens, l1_min_chars) → int`: calcola target chars per singolo L1 in funzione di N (con floor). Per 100 paper → 5,920 chars; per 500 → 2,000 (floor)
>   - `get_l1_compression_tier(n_papers) → str`: restituisce `"standard"` (≤30), `"concise"` (31-70), `"compressed"` (>70)
> - **Budget L2:**
>   - `compute_l2_budget(max_context_tokens, prompt_overhead_tokens) → dict`: budget totale disponibile per il prompt L2 (150k token → 592k chars disponibili)
> - **Budget Writer:**
>   - `compute_writer_budget(n_sections, writer_max_context_chars, writer_budget_mode, max_context_tokens) → dict`: calcola budget per sezione. Modalità `"proportional"`: divide budget globale per N sezioni (cap a `writer_max_context_chars`). Modalità `"fixed"`: usa il valore fisso per ogni sezione
> - **Utility generica:**
>   - `chars_budget_for_n_papers(n_papers, max_context_tokens, prompt_overhead_tokens) → int`: budget totale aggregato in chars per N paper
> - **`BudgetReport` (dataclass):**
>   - `BudgetReport.from_config(n_categories=4, n_sections_estimate=8)`: costruisce report completo dalla configurazione corrente
>   - `to_dict() → dict`: serializzazione
>   - `print_report()`: stampa formattata con Rich (panel + tabella per fase + warnings)
>   - Verifica automatica: se `l2_tokens_per_call > max_context_tokens` → warning + `within_budget = False`
> - **Test:** Tutti i test passati:
>   - `estimate_tokens("51 chars")` = 13, `estimate_tokens("")` = 0, `estimate_tokens("4000 chars")` = 1000
>   - Config: `max_context_tokens = 150,000`, `papers_per_category = 100`
>   - L1: 100 papers → 5,920 chars, 20 papers → 29,600 chars, 500 papers → 2,000 (floor)
>   - Tier: 20→standard, 50→concise, 100→compressed
>   - L2: 592,000 chars disponibili
>   - Writer proportional (8 sezioni): 60,000 chars/section
>   - BudgetReport: 400 paper totali, within_budget=True, 0 warnings

---

### [x] Task A.2: Budget-aware L1 summarization

**File:** `paperhunter/summarization/level1_summarizer.py` (modifica), `paperhunter/models.py` (modifica)  
**Stato attuale:** `_prepare_chunks_text()` tronca a `max_chunk_chars=50_000` per singolo paper → funziona per L1 singoli. Non è un problema a livello L1 perché ogni L1 processa un solo paper alla volta.  
**Obiettivo:** Quando `l1_target_chars_auto=true`, rendere gli L1 più compatti nelle categorie con molti paper, così che la fase L2 non sfori il budget.  
**Dettagli:**
- Leggere `papers_per_category` da config e calcolare `target_l1_chars = max_l2_input_chars / N` (con floor `l1_min_chars`)
- Se N ≤ 30: nessuna compressione (prompt standard, ~2-3k chars/L1)
- Se 30 < N ≤ 70: prompt "concise" (~1.5k chars/L1)
- Se N > 70: prompt "compressed" (~1k chars/L1, solo campi essenziali)
- Le soglie sono derivate da `papers_per_category` e `max_context_tokens` automaticamente
- Aggiungere campo `l1_char_count` al `L1Summary` per tracking

> **Completato 2/03/2026.** Modifiche:
> - **`paperhunter/models.py`:** Aggiunti 2 nuovi campi a `L1Summary`:
>   - `l1_char_count: Optional[int]` — numero totale di caratteri dei campi testuali del summary (per tracking budget)
>   - `compression_tier: Optional[str]` — tier di compressione usato (`"standard"`, `"concise"`, `"compressed"`)
> - **`paperhunter/summarization/level1_summarizer.py`:** Modifiche principali:
>   - **Nuovi import:** `compute_l1_target_chars`, `get_l1_compression_tier`, `get_papers_per_category` da `utils.token_counter`
>   - **Nuovi prompt template:**
>     - `_L1_PROMPT_CONCISE`: prompt per tier "concise" (31-70 paper), output target ~1.5k chars, max 3 key_contributions, campi più sintetici
>     - `_L1_PROMPT_COMPRESSED`: prompt per tier "compressed" (>70 paper), output target ~1k chars, max 2 key_contributions, solo campi essenziali, `temporal_context` opzionalmente vuoto
>   - **`_get_l1_prompt_template(tier) → str`:** selettore prompt template in base al tier
>   - **`_compute_l1_char_count(summary) → int`:** calcola il totale chars dei campi testuali (key_contributions, methodology, testing_methodology, temporal_context, results_summary, limitations)
>   - **`generate_l1_summary()`:** nuovi parametri:
>     - `compression_tier: str | None` — override tier forzato
>     - `n_papers_in_category: int | None` — per calcolo automatico del tier
>     - Quando `l1_target_chars_auto=true` in config: calcola automaticamente il tier da N paper
>     - Seleziona il prompt template adatto al tier
>     - Limita `key_contributions` in base al tier (5/3/2)
>     - Popola `summary.l1_char_count` e `summary.compression_tier`
>     - Logging arricchito con info su compression tier e char count
>   - **`generate_l1_batch()`:** nuovi parametri:
>     - `compression_tier: str | None` — override tier per tutto il batch
>     - `n_papers_in_category: int | None` — per calcolo automatico
>     - Propaga il tier a tutte le chiamate `generate_l1_summary()`
>     - Traccia `total_chars` aggregato e `avg_chars` per il report
>   - **`_print_batch_report()`:** nuovi parametri per statistiche budget:
>     - `compression_tier`, `avg_l1_chars`, `total_l1_chars`, `target_chars`
>     - Mostra rapporto avg/target con status "✅ OK" o "⚠️ sopra target"
> - **Logica tier automatica:**
>   - N ≤ 30 → `"standard"` (prompt completo, ~2-3k chars/L1)
>   - 30 < N ≤ 70 → `"concise"` (prompt ridotto, ~1.5k chars/L1)
>   - N > 70 → `"compressed"` (prompt minimale, ~1k chars/L1)
>   - Soglie lette da `token_counter.L1_TIER_STANDARD_MAX` e `L1_TIER_CONCISE_MAX`
> - **Budget scaling verificato:**
>   - 10 paper → 59,200 chars/L1, 100 paper → 5,920 chars/L1, 500 paper → 2,000 chars/L1 (floor)
>   - Monotonicamente decrescente con N, floor garantito da `l1_min_chars`
> - **Test:** 6/6 test passati:
>   - Tier selection: 9 casi N=10..200 → tutti corretti
>   - Prompt templates: standard/concise/compressed/unknown → tutti corretti
>   - L1 char count computation: 350 chars calcolati = 350 attesi
>   - L1Summary new fields: default None, set values, JSON serialization → tutti OK
>   - Budget scaling: 9 valori monotonicamente decrescenti + floor a 2,000
>   - Batch tier propagation: liste vuote con parametri nuovi → nessun crash

---

### [  ] Task A.3: Budget-aware L2 summarization con selezione chunk

**File:** `paperhunter/summarization/level2_summarizer.py` (modifica)  
**Stato attuale:** `_format_l1_summaries_for_prompt()` concatena TUTTI gli L1 senza limite di dimensione. Con 100 paper/categoria e L1 di ~2-3k chars ciascuno, si arriverebbe a ~200-300k chars → troppi token per il context window.  
**Obiettivo:** Restare entro il budget token letto da `config.yaml` → `summarization.max_context_tokens`.  
**Dettagli:**
- Calcolare `max_l2_prompt_chars` dal budget token (es. `max_context_tokens * 4`, meno prompt template ~2k chars)
- In `_format_l1_summaries_for_prompt()`: se il totale supera il budget, applicare strategia di selezione basata su `l2_recency_tiers` da config:
  1. **Paper recenti (≤ `full` anni):** testo L1 completo
  2. **Paper medi (≤ `reduced` anni):** ~70% del testo
  3. **Paper vecchi (> `reduced` anni):** solo key_contributions + testing_methodology
  4. **Citation-weighted:** paper con più citazioni sono meno comprimibili (a parità di tier)
- Se dopo la compressione per tier si sfora ancora, truncation progressiva: troncare campi meno critici (limitations, temporal_context) prima di methodology
- Logging del budget consumato vs disponibile

---

### [  ] Task A.4: Budget-aware Writer Agent context

**File:** `paperhunter/agents/writer_agent.py` (modifica)  
**Stato attuale:** `_format_context_for_prompt()` tronca a `max_chars=60_000` (hardcoded). Per ~400 paper il retrieval potrebbe dover essere più selettivo.  
**Obiettivo:** Rendere il budget del writer agent configurabile e adattivo.  
**Dettagli:**
- `max_chars` letto da `config.yaml` → `agents.writer_max_context_chars`
- Se `writer_budget_mode = "proportional"`: `budget_per_section = (max_context_tokens * 4) / N_sezioni`
- Se `writer_budget_mode = "fixed"`: usa `writer_max_context_chars` fisso per ogni sezione
- Preferire chunk di tipo L1/L2 summary rispetto a chunk originali, per densità informativa

---

### [  ] Task A.5: Sommario di budget per la pipeline

**File:** `paperhunter/agents/pipeline.py` (modifica), `paperhunter/utils/token_counter.py`  
**Obiettivo:** Prima di avviare la pipeline agentica, calcolare e stampare un budget stimato in base alla configurazione corrente.  
**Dettagli:**
- All'inizio della pipeline, stimare: token totali L2 summaries, token per sezione writer, token per verifica critic
- Leggere `papers_per_category` e `max_context_tokens` da config per il calcolo
- Se il budget stimato > `max_context_tokens`, avvisare e suggerire azioni (ridurre `papers_per_category`, abilitare compressione, ecc.)
- Stampare il report con `rich` (tabella: fase → token stimati → % del budget)

---

## Requisito 2 — Non scaricare paper già presenti in "scaricati"

### [  ] Task B.1: Deduplicazione pre-download

**File:** `paperhunter/download_manager.py` (modifica)  
**Stato attuale:** `download_paper()` e `batch_download()` non controllano se il PDF esiste già nella cartella `scaricati/`. L'unico check di dedup è nella fase di *ingestion* (`pdf_hasher.py`), non nella fase di *download*.  
**Obiettivo:** Prima di scaricare, verificare se un paper con lo stesso titolo/ID esiste già.  
**Dettagli:**
- In `Downloader.__init__()`: scansionare `scaricati/` ricorsivamente e costruire un set di filename normalizzati esistenti (senza estensione, lowercase, senza spazi extra)
- In `download_paper()`: prima di tentare il download, confrontare il `_sanitize_filename(paper.title)` con il set. Se match → skip con `DownloadResult(success=True, skipped=True, ...)`
- Aggiungere campo `skipped: bool = False` a `DownloadResult` in `models.py`
- In `batch_download()`: loggare il conteggio paper skippati vs scaricati
- Opzionale: check anche per ArXiv ID nel filename (es. `2305.12345.pdf`)

---

## Requisito 3 — Priorità ai paper più recenti durante l'ingestion

### [  ] Task C.1: Ordinamento per data nella ingestion pipeline

**File:** `paperhunter/ingestion/ingestion_pipeline.py` (modifica)  
**Stato attuale:** I PDF vengono processati nell'ordine restituito da `get_new_or_modified_pdfs()`, che è l'ordine di scansione del filesystem (non deterministico, non cronologico).  
**Obiettivo:** Processare prima i paper più recenti.  
**Dettagli:**
- Dopo lo step 3 (S2 enrichment), ordinare i paper per `year` (desc) e `month` (desc) prima di procedere con chunking e indicizzazione
- Se `year` non è disponibile (es. S2 non ha trovato il paper), usare la data di modifica del PDF come fallback
- In alternativa: durante lo step 1, se possibile estrarre l'anno dal filename o dal titolo GROBID, usarlo per un pre-ordinamento

---

### [  ] Task C.2: Recency weighting nel retrieval

**File:** `paperhunter/indexing/hybrid_retriever.py` (modifica)  
**Stato attuale:** L'hybrid retriever usa RRF puro senza boost temporale. Paper del 2020 e del 2025 hanno lo stesso peso a parità di score.  
**Obiettivo:** Dare un boost ai paper più recenti nel retrieval.  
**Dettagli:**
- Aggiungere parametro `recency_boost: float = 0.0` (default disabilitato, es. `0.1` per attivare)
- Dopo la fusione RRF, applicare un moltiplicatore temporale: `final_score = rrf_score * (1 + recency_boost * recency_factor)` dove `recency_factor = (year - min_year) / (max_year - min_year)`
- Configurabile da `config.yaml` → `retrieval.recency_boost`

---

### [  ] Task C.3: Recency weighting nell'L2 summarizer

**File:** `paperhunter/summarization/level2_summarizer.py` (modifica)  
**Stato attuale:** `_format_l1_summaries_for_prompt()` formatta tutti gli L1 con lo stesso livello di dettaglio, ordinati cronologicamente.  
**Obiettivo:** Dedicare più spazio ai paper recenti nel prompt L2.  
**Dettagli:**
- Paper degli ultimi 2 anni: testo completo dell'L1
- Paper 2-4 anni fa: testo ridotto (~70%)
- Paper > 4 anni fa: solo key_contributions + testing_methodology (~40%)
- Queste soglie devono essere configurabili da `config.yaml`

---

#### [x] Task FINALE: Aggiornare README.md — Documentazione completa

---

## Riepilogo

| ID | Descrizione | Complessità | File principali |
|----|------------|-------------|-----------------|
| 0.1 | Config `papers_per_category` + budget | Media | `config.yaml`, `config.py` |
| A.1 | Token counter utility | Bassa | `utils/token_counter.py` (nuovo) |
| A.2 | Budget-aware L1 (adattivo per N) | Media | `summarization/level1_summarizer.py` |
| A.3 | Budget-aware L2 + selezione chunk | Alta | `summarization/level2_summarizer.py` |
| A.4 | Budget-aware Writer context | Media | `agents/writer_agent.py` |
| A.5 | Sommario budget pipeline | Bassa | `agents/pipeline.py`, `utils/token_counter.py` |
| B.1 | Dedup pre-download | Media | `download_manager.py`, `models.py` |
| C.1 | Ordinamento recency ingestion | Bassa | `ingestion/ingestion_pipeline.py` |
| C.2 | Recency boost retrieval | Media | `indexing/hybrid_retriever.py` |
| C.3 | Recency weighting L2 | Media | `summarization/level2_summarizer.py` |

---

> **Flusso di dipendenze:** Task 0.1 → A.1 → (A.2 + A.3 + A.4 in parallelo) → A.5. I task B.* e C.* sono indipendenti.
