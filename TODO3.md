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

### [x] Task A.3: Budget-aware L2 summarization con selezione chunk

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

> **Completato 2/03/2026.** Modifiche:
> - **`paperhunter/models.py`:** Aggiunto campo `citation_count: Optional[int] = None` a `L1Summary` per supportare citation-weighted budget L2
> - **`paperhunter/summarization/level2_summarizer.py`:** Modifiche principali:
>   - **Nuovo import:** `import math`, `compute_l2_budget` da `utils.token_counter`
>   - **`_get_paper_recency_tier(year, current_year)`:** Classifica ogni paper in tier di recency (`"full"`, `"reduced"`, `"compressed"`) in base alle soglie configurate in `config.yaml` → `summarization.l2_recency_tiers` (default: full ≤2 anni, reduced ≤4 anni, compressed >4 anni). Paper senza anno → compressed
>   - **`_compute_paper_importance(s, current_year)`:** Score di importanza `[0,1]` basato su `0.7 × recency_factor + 0.3 × citation_factor`. Recency: `(year-2015)/(current_year-2015)`. Citation: `log₁₀(1+citations)/5`, cappato a 1.0. Usato per decidere l'ordine di downgrade (paper meno importanti compressi per primi)
>   - **`_truncate_text(text, max_chars, suffix)`:** Tronca testo preservando parole intere
>   - **`_format_l1_at_tier(s, tier, index, total)`:** Dispatcher che delega a full/reduced/compressed
>   - **`_format_l1_full()`:** Formato completo con tutti i campi (identico all'originale pre-budget-aware)
>   - **`_format_l1_reduced()`:** ~70% del dettaglio — mantiene key_contributions, methodology, testing_methodology, temporal_context troncato a 200 chars, results troncati a 300 chars. Rimuove limitations
>   - **`_format_l1_compressed()`:** Minimale — solo key_contributions (max 3) + testing_methodology. Tag `[COMPRESSED]` nel header
>   - **`_format_all_at_tiers(paper_entries, total)`:** Formatta tutti i paper ai tier assegnati preservando ordine originale
>   - **`_format_l1_summaries_for_prompt()`:** *Riscritta completamente* con logica budget-aware multi-fase:
>     - Fase 1: Calcola budget da `compute_l2_budget()` (588,500 chars con config default 150k token)
>     - Fase 2: Assegna recency tier e score di importanza a ogni paper
>     - Fase 3: Formatta al tier assegnato — se entro budget, terminato
>     - Fase 4: Downgrade reduced→compressed (meno importanti prima, citation-weighted)
>     - Fase 5: Downgrade full→reduced (meno importanti prima)
>     - Fase 6: All-compressed come fallback
>     - Fase 7: Hard truncation proporzionale come ultima risorsa
>     - Logging dettagliato a ogni fase (chars consumati/disponibili, % utilizzo, N paper downgraded)
>   - **`_reconstruct_l1_from_payload()`:** Aggiunto `citation_count=payload.get("citation_count")` per ricostruire il campo dal vector store
> - **Compressione verificata:**
>   - 20 paper con long_fields: full tiers = 27,203 chars → all-compressed = 11,031 chars (40.55% del totale, 59% riduzione)
>   - 100 paper: full/reduced/compressed mix → 127,664 chars, 21% del budget (ampiamente entro i 588,500 chars disponibili)
>   - 5 paper: 2,470 chars, 0% del budget (nessuna compressione necessaria)
> - **Test:** 9/9 passati (0.2s):
>   - Recency tiers: 8/8 casi corretti (boundary full=2, reduced=4, None=compressed)
>   - Paper importance: 4/4 vincoli di ordinamento rispettati (recent+cited > recent > old+cited > old > no_year)
>   - Tier formatting: full > reduced > compressed in size, campi corretti per ogni tier
>   - Budget small (5 papers): 2,474 chars, tutti i paper presenti, tier recency applicati
>   - Budget large (100 papers): 127,763 chars ≤ 588,500 budget, [COMPRESSED] e [REDUCED] presenti
>   - Progressive downgrade: 59% riduzione con all-compressed, importance range 0.47-0.87
>   - Truncate text: 4/4 casi (short, long, empty, None)
>   - citation_count field: default None, set, serialization, JSON round-trip
>   - Backward compat: 2 paper → output identico al comportamento originale, empty → ""

---

### [x] Task A.4: Budget-aware Writer Agent context

**File:** `paperhunter/agents/writer_agent.py` (modifica)  
**Stato attuale:** `_format_context_for_prompt()` tronca a `max_chars=60_000` (hardcoded). Per ~400 paper il retrieval potrebbe dover essere più selettivo.  
**Obiettivo:** Rendere il budget del writer agent configurabile e adattivo.  
**Dettagli:**
- `max_chars` letto da `config.yaml` → `agents.writer_max_context_chars`
- Se `writer_budget_mode = "proportional"`: `budget_per_section = (max_context_tokens * 4) / N_sezioni`
- Se `writer_budget_mode = "fixed"`: usa `writer_max_context_chars` fisso per ogni sezione
- Preferire chunk di tipo L1/L2 summary rispetto a chunk originali, per densità informativa

> **Completato 2/03/2026.** Modifiche:
> - **`paperhunter/agents/writer_agent.py`:** Modifiche principali:
>   - **Nuovi import:** `compute_writer_budget`, `get_writer_max_context_chars`, `get_writer_budget_mode` da `utils.token_counter`
>   - **`_CHUNK_TYPE_PRIORITY`:** Dizionario di priorità per tipo chunk: `L2_SUMMARY(0) > L1_SUMMARY(1) > original(2) > graph(3)`. Supporta varianti case-sensitive e lowercase
>   - **`_compute_section_budget(n_total_sections, max_chars_override) → int`:** Calcola il budget in chars per una singola sezione. Se `max_chars_override` fornito, lo usa direttamente. Altrimenti delega a `compute_writer_budget()` rispettando la modalità da config (`"proportional"` o `"fixed"`). Con `n_total_sections ≤ 0` → trattato come 1
>   - **`_sort_chunks_by_priority(chunks) → list[dict]`:** Ordina i chunk dando priorità a L1/L2 summary rispetto a original/graph. Ordinamento stabile: a parità di tipo, l'ordine originale (score del retrieval) è preservato. Chunk senza `chunk_type` → trattati come original (priority 2)
>   - **`_format_context_for_prompt()`:** *Riscritta completamente* con logica budget-aware:
>     - Nuova firma: `_format_context_for_prompt(chunks, max_chars=None, *, n_total_sections=None, prefer_summaries=True)`
>     - `max_chars=None` → budget calcolato automaticamente da config (non più hardcoded 60,000)
>     - `n_total_sections` → passato a `_compute_section_budget()` per modalità proportional
>     - `prefer_summaries=True` → riordina chunk con `_sort_chunks_by_priority()` (L2 > L1 > original)
>     - `prefer_summaries=False` → preserva ordine originale
>     - Messaggio di truncation include budget: `[... N more chunks truncated for prompt length (budget: X chars) ...]`
>     - Logging info: `"Context formattato: N chunk inclusi, X/Y chars usati (Z%)"`
>     - Backward compatible: `max_chars=5000` funziona come override diretto
>   - **`write_section()`:** Nuovi parametri:
>     - `n_total_sections: int | None = None` — numero totale sezioni (calcolato da outline se None)
>     - `max_context_chars: int | None = None` — override diretto budget
>     - Passa entrambi a `_format_context_for_prompt()`
>   - **`write_all_sections()`:** Nuovo parametro:
>     - `max_context_chars: int | None = None` — propagato a ogni `write_section()`
>     - Calcola `n_total_sections = len(outline)` e lo propaga
>     - Logga budget per sezione all'inizio: `"budget contesto per sezione: X chars"`
>   - **`write()` (nodo LangGraph):** Calcola e logga budget prima della scrittura:
>     - `n_total_sections = len(outline)`, `section_budget` calcolato
>     - Propaga `n_total_sections` a `write_section()` per le revisioni
> - **Budget scaling verificato:**
>   - 1 sezione → 60,000 chars (cappato a `writer_max_context_chars`)
>   - 4 sezioni → 60,000 chars (cap)
>   - 8 sezioni → 60,000 chars (cap: 592k/8 = 74k > 60k)
>   - 12 sezioni → 49,333 chars (592k/12 = 49.3k < 60k)
>   - 20 sezioni → 29,600 chars (592k/20 = 29.6k)
>   - Monotonicamente decrescente, cap superiore garantito
> - **Prioritizzazione chunk verificata:**
>   - L2_SUMMARY appare prima di L1_SUMMARY nel prompt, L1 prima di original, original prima di graph
>   - Ordinamento stabile: ordine relativo intra-tipo preservato
> - **Test:** 8/8 passati + 13/13 test esistenti inalterati:
>   - `_compute_section_budget`: override, proportional, cap, zero/negativo → tutti corretti
>   - `_sort_chunks_by_priority`: ordinamento L2>L1>orig>graph, stabilità, chunk senza tipo → OK
>   - `_format_context_for_prompt`: empty, override, config-based, prefer_summaries on/off → OK
>   - `_CHUNK_TYPE_PRIORITY`: tutti i valori corretti
>   - Budget consistency: N sezioni = 1..20, monotonicamente decrescente, ≤ max_chars
>   - `write_section`, `write_all_sections` signature: nuovi params con default None
>   - Large context (200 chunk, budget 20k): L2 all'inizio, truncation corretta
>   - Backward compatibility: `max_chars=5000` funziona come prima

---

### [x] Task A.5: Sommario di budget per la pipeline

**File:** `paperhunter/agents/pipeline.py` (modifica), `paperhunter/utils/token_counter.py` (modifica)  
**Obiettivo:** Prima di avviare la pipeline agentica, calcolare e stampare un budget stimato in base alla configurazione corrente.  
**Dettagli:**
- All'inizio della pipeline, stimare: token totali L2 summaries, token per sezione writer, token per verifica critic
- Leggere `papers_per_category` e `max_context_tokens` da config per il calcolo
- Se il budget stimato > `max_context_tokens`, avvisare e suggerire azioni (ridurre `papers_per_category`, abilitare compressione, ecc.)
- Stampare il report con `rich` (tabella: fase → token stimati → % del budget)

> **Completato 2/03/2026.** Modifiche:
> - **`paperhunter/utils/token_counter.py`:** Modifiche a `BudgetReport`:
>   - **`_pct(tokens_per_call) → str`:** Calcola la percentuale del budget usata da una singola chiamata. Gestisce edge case `max_context_tokens=0` → `"N/A"`
>   - **`print_report()`:** *Riscritta* — aggiunta colonna `"% Budget"` alla tabella Rich. Colonne: Fase, Token/call, % Budget, Note. Aggiunto messaggio di successo (`print_success`) quando entro budget. Aggiunta stampa suggerimenti con Rich quando fuori budget
>   - **`_get_suggestions() → list[str]`:** Genera suggerimenti specifici per rientrare nel budget. Se L2 sfora: suggerisce ridurre `papers_per_category` (con stima del max consigliato) e abilitare compressione L1. Se Writer sfora: suggerisce ridurre `writer_max_context_chars`
> - **`paperhunter/agents/pipeline.py`:** Modifiche principali:
>   - **Nuovo import:** `BudgetReport` da `utils.token_counter`
>   - **`print_budget_summary(n_categories, n_sections_estimate) → BudgetReport`:** Nuova funzione pubblica. Calcola e stampa il report budget prima di avviare la pipeline. Logga warning se fuori budget, info se OK. Restituisce l'istanza `BudgetReport` per uso programmatico
>   - **`run_pipeline()`:** Integrato `print_budget_summary()` subito dopo il panel iniziale, prima di costruire il grafo. `n_sections_est` calcolato da outline se fornito, altrimenti default 8. Il `budget_report.to_dict()` viene salvato nei metadata della run (`run_metadata["budget_report"]`)
>   - **`run_pipeline_from_checkpoint()`:** Integrato `print_budget_summary()` anche al resume, dopo il panel "Resume Pipeline"
> - **Output esempio (config default: 100 paper/cat, 150k token):**
>   - L1 Summarization: 14,500 token/call (9.7%)
>   - L2 Summarization: 150,000 token/call (100.0%)
>   - Writer (per sezione): 17,000 token/call (11.3%) — 8 sezioni
>   - Critic (per sezione): 17,000 token/call (11.3%)
>   - Stato: ✅ ENTRO BUDGET
> - **Test:** 10/10 passati:
>   - `_pct`: percentuali corrette (100%, 50%, 10%, 0%), edge case N/A
>   - Suggestions within budget: 0 suggerimenti
>   - Suggestions over budget: suggerimenti L2 + Writer presenti
>   - `print_report` con colonna % Budget: eseguito senza errori
>   - `print_budget_summary`: ritorna `BudgetReport` valido con tutti i campi
>   - Custom sections: budget per sezione decresce con 20 sezioni vs 8
>   - `to_dict`: 14/14 campi presenti
>   - `run_pipeline` integration: budget_report nei metadata, nessun errore
>   - `from_config` default: within_budget=True, 0 warnings
>   - Forced over budget: rilevato correttamente con warning e suggerimenti

---

## Requisito 2 — Non scaricare paper già presenti in "scaricati"

### [x] Task B.1: Deduplicazione pre-download

**File:** `paperhunter/download_manager.py` (modifica)  
**Stato attuale:** `download_paper()` e `batch_download()` non controllano se il PDF esiste già nella cartella `scaricati/`. L'unico check di dedup è nella fase di *ingestion* (`pdf_hasher.py`), non nella fase di *download*.  
**Obiettivo:** Prima di scaricare, verificare se un paper con lo stesso titolo/ID esiste già.  
**Dettagli:**
- In `Downloader.__init__()`: scansionare `scaricati/` ricorsivamente e costruire un set di filename normalizzati esistenti (senza estensione, lowercase, senza spazi extra)
- In `download_paper()`: prima di tentare il download, confrontare il `_sanitize_filename(paper.title)` con il set. Se match → skip con `DownloadResult(success=True, skipped=True, ...)`
- Aggiungere campo `skipped: bool = False` a `DownloadResult` in `models.py`
- In `batch_download()`: loggare il conteggio paper skippati vs scaricati
- Opzionale: check anche per ArXiv ID nel filename (es. `2305.12345.pdf`)

> **Completato 2/03/2026.** Modifiche:
> - **`paperhunter/models.py`:** Aggiunto campo `skipped: bool = Field(default=False)` a `DownloadResult` — indica se il download è stato saltato perché il paper esiste già localmente
> - **`paperhunter/download_manager.py`:** Modifiche principali:
>   - **`_normalize_filename(name) → str`:** Funzione module-level per normalizzare filename per confronto dedup. Rimuove estensione, lowercase, sostituisce spazi/underscore multipli con singolo underscore, rimuove caratteri non alfanumerici, collassa underscore residui
>   - **`Downloader.__init__()`:** Aggiunta scansione ricorsiva di `download_dir` al momento dell'inizializzazione. Costruisce due set:
>     - `_existing_files: set[str]` — filename normalizzati di tutti i PDF esistenti
>     - `_existing_arxiv_ids: set[str]` — ArXiv ID estratti dai filename (regex `\d{4}\.\d{4,5}`)
>   - **`_scan_existing_pdfs()`:** Scansione ricorsiva con `rglob('*.pdf')`. Per ogni PDF: normalizza il filename stem e aggiunge al set; estrae ArXiv ID se presente. Logga il conteggio totale
>   - **`_is_duplicate(paper, custom_filename) → tuple[bool, str | None]`:** Verifica se il paper esiste già. Strategia a 2 livelli:
>     1. Check per ArXiv ID (stripping del suffisso versione `v\d+`) — più affidabile
>     2. Check per filename normalizzato — fallback per paper senza ArXiv ID
>   - **`register_downloaded(paper, file_path)`:** Aggiorna l'indice dedup in-session dopo un download riuscito, così che download successivi nella stessa sessione batch rilevano il duplicato
>   - **`download_paper()`:** Aggiunto check `_is_duplicate()` all'inizio. Se duplicato → `DownloadResult(success=True, skipped=True, method="dedup")` senza alcuna richiesta HTTP. Dopo download riuscito → `register_downloaded()` per aggiornare l'indice
>   - **`batch_download()`:** Logging arricchito con conteggio separato: `"X downloaded, Y skipped (already exist), Z failed — N total"`
> - **Indice dedup su `scaricati/` reale:** 118 PDF → 115 nomi unici normalizzati, 0 ArXiv IDs (i file attuali non hanno ArXiv ID nel nome)
> - **Test:** 22/22 passati:
>   - `_normalize_filename`: 8 casi (basic, spaces, special chars, underscores, empty, case, arxiv, strips)
>   - `DownloadResult.skipped`: default False, esplicito True, JSON roundtrip
>   - `_scan_existing_pdfs`: directory vuota (0), popolata (3 file, 1 ArXiv), ignora non-PDF
>   - `_is_duplicate`: match per titolo, per ArXiv ID, con suffisso versione, no match
>   - `register_downloaded`: aggiorna indice in-session, successivo `_is_duplicate` → True
>   - `download_paper()`: skip su duplicato (success=True, skipped=True, method="dedup")
>   - `batch_download()`: 2 paper esistenti → 0 downloaded, 2 skipped, 0 failed
>   - `scaricati/` reale: 115 nomi unici indicizzati correttamente

---

## Requisito 3 — Priorità ai paper più recenti durante l'ingestion

### [x] Task C.1: Ordinamento per data nella ingestion pipeline

**File:** `paperhunter/ingestion/ingestion_pipeline.py` (modifica)  
**Stato attuale:** I PDF vengono processati nell'ordine restituito da `get_new_or_modified_pdfs()`, che è l'ordine di scansione del filesystem (non deterministico, non cronologico).  
**Obiettivo:** Processare prima i paper più recenti.  
**Dettagli:**
- Dopo lo step 3 (S2 enrichment), ordinare i paper per `year` (desc) e `month` (desc) prima di procedere con chunking e indicizzazione
- Se `year` non è disponibile (es. S2 non ha trovato il paper), usare la data di modifica del PDF come fallback
- In alternativa: durante lo step 1, se possibile estrarre l'anno dal filename o dal titolo GROBID, usarlo per un pre-ordinamento

> **Completato 2/03/2026.** Modifiche:
> - **`paperhunter/ingestion/pdf_hasher.py`:** Aggiunto campo `mtime` (float, secondi epoch) all'output di `scan_pdf_directory()` — timestamp di modifica del file, ottenuto via `stat().st_mtime`. Disponibile in ogni `pdf_info` per l'ordinamento
> - **`paperhunter/ingestion/ingestion_pipeline.py`:** Modifiche principali:
>   - **Nuovi import:** `os`, `re` per supporto regex e filesystem
>   - **`_YEAR_RE`:** Pattern regex compilato `(?<!\d)(20[0-2]\d)(?!\d)` — matcha anni 2000-2029 non adiacenti ad altre cifre (evita falsi positivi su date YYYYMMDD o numeri lunghi)
>   - **`_extract_year_from_filename(filename) → int | None`:** Estrae anno dal nome del file. Se più match, restituisce il più recente (euristica: probabilmente è l'anno di pubblicazione). Supporta separatori vari: `_2024_`, `2024.pdf`, `(2023)`, `v2024`, ecc.
>   - **`_get_pdf_sort_key(pdf_info) → tuple[int, int, float]`:** Calcola chiave di ordinamento composita a 3 livelli: (year, month, mtime). Anno dal filename ha priorità su anno da mtime. La tupla è costruita per ordinamento `reverse=True` → i paper più recenti vengono prima
>   - **`sort_pdfs_by_recency(pdfs) → list[dict]`:** Funzione pubblica che ordina i PDF per data stimata decrescente. Annota ogni PDF con `sort_year` (int) e `sort_source` (`"filename"` | `"mtime"`). Non modifica la lista originale (restituisce una copia). Logga statistiche: range anni, N da filename vs mtime
>   - **`run()`:** Aggiunto **STEP 1c** tra il checkpoint filtering e il batch processing. Chiama `sort_pdfs_by_recency(new_pdfs)` per riordinare i PDF prima dell'elaborazione. I paper più recenti vengono processati prima, garantendo che se la pipeline viene interrotta, i paper più importanti sono già completati
> - **`paperhunter/ingestion/__init__.py`:** Esportata `sort_pdfs_by_recency` nel modulo pubblico
> - **Test su `scaricati/` reale:** 118 PDF ordinati, range 2026-2026 (tutti mtime recenti, nessun anno nei filename attuali). Fonte: 0 da filename, 118 da mtime
> - **Test:** 8/8 gruppi passati (48/48 test individuali):
>   - `_extract_year_from_filename`: 16 casi (anni validi, fuori range, nessun anno, stringa vuota, numeri embedded)
>   - `_get_pdf_sort_key`: 4 casi (filename > mtime, fallback mtime, no info, ordinamento corretto)
>   - `sort_pdfs_by_recency` base: 5 casi (lista mista, ordinamento decrescente, primo/ultimo corretto, originale immutata)
>   - Annotazioni: 3 casi (sort_source filename, sort_source mtime, campi preservati)
>   - Edge cases: 5 casi (lista vuota, singolo, stesso anno, mesi decrescenti, nessuna info)
>   - `scan_pdf_directory` mtime: 4 casi (trovati, campo presente, float positivo, ordine mtime)
>   - Real PDFs sort: 5 casi (stesso numero, annotazioni, range 2000-2030, decrescente, paths preservati)
>   - Year regex boundaries: 6 casi (YYYYMMDD non matchato, numeri lunghi, separatori)

---

### [x] Task C.2: Recency weighting nel retrieval

**File:** `paperhunter/indexing/hybrid_retriever.py` (modifica)  
**Stato attuale:** L'hybrid retriever usa RRF puro senza boost temporale. Paper del 2020 e del 2025 hanno lo stesso peso a parità di score.  
**Obiettivo:** Dare un boost ai paper più recenti nel retrieval.  
**Dettagli:**
- Aggiungere parametro `recency_boost: float = 0.0` (default disabilitato, es. `0.1` per attivare)
- Dopo la fusione RRF, applicare un moltiplicatore temporale: `final_score = rrf_score * (1 + recency_boost * recency_factor)` dove `recency_factor = (year - min_year) / (max_year - min_year)`
- Configurabile da `config.yaml` → `retrieval.recency_boost`

> **Completato 2/03/2026.** Modifiche:
> - **`paperhunter/indexing/hybrid_retriever.py`:** Modifiche principali:
>   - **Nuova costante:** `_RECENCY_BASELINE_YEAR = 2015` — anno di riferimento per il calcolo recency (pre-deep-learning era)
>   - **`HybridRetriever.__init__()`:** Legge `recency_boost` da `config.yaml` → `retrieval.recency_boost` (default 0.1). Salvato in `self._recency_boost`
>   - **`HybridRetriever.retrieve()`:** Nuovo parametro `recency_boost: Optional[float] = None` per override per-query. Dopo la fusione RRF (e prima del troncamento a `top_k_final`), applica `_apply_recency_boost()` se il boost è > 0. Logging arricchito con `recency_boost` nel messaggio
>   - **`_apply_recency_boost(results, boost) → list[RetrievedChunk]`:** Nuova funzione pubblica. Formula: `final_score = rrf_score * (1 + boost * recency_factor)` dove `recency_factor = max((year - 2015) / (max_year - 2015), 0.0)` normalizzato in `[0, 1]`. Chunk senza anno → `recency_factor = 0.0` (nessun boost). Ri-ordina per score decrescente dopo l'applicazione. Con `boost ≤ 0` o lista vuota → ritorna i risultati invariati
>   - **`get_stats()`:** Aggiunto `recency_boost` nella sezione config delle statistiche
>   - **Logging init:** Aggiunto `recency_boost` al messaggio di inizializzazione
> - **Formula in dettaglio:**
>   - `max_year` = anno massimo tra tutti i chunk con anno
>   - `baseline` = 2015 (costante)
>   - `year_range` = `max(max_year - baseline, 1)` (evita divisione per zero)
>   - Per ogni chunk con anno: `recency_factor = max((year - baseline) / year_range, 0.0)`
>   - Score finale: `score * (1.0 + boost * recency_factor)`
>   - Anno < baseline → `recency_factor = 0.0` → nessun boost
> - **Proprietà del boost:**
>   - Monotonicamente crescente con l'anno del paper
>   - Non distruttivo: con boost=0.1, il massimo incremento è +10% per il paper più recente
>   - Non sovverte ranking dominanti: un paper con score 10x superiore resta in testa anche se vecchio
>   - Con score vicini, il paper recente può superare il vecchio (comportamento desiderato)
> - **Test:** 17/17 passati (12 esistenti inalterati + 5 nuovi):
>   - Test 13 — `_apply_recency_boost` base: ordine corretto (new > mid > old), formula verificata con valori esatti (0.011, 0.0107, 0.0103)
>   - Test 14 — Edge cases: 7 sottocasi — boost=0 (invariato), boost negativo (invariato), lista vuota, nessun anno (invariato), mix con/senza anno (boost solo su chi ha anno), anno pre-baseline (factor=0.0), stesso anno (ordine per score originale)
>   - Test 15 — Integrazione `retrieve()`: 3 sottocasi — con boost (recenti in cima), boost=0 (ordine originale), boost alto (gap maggiore: 0.315 vs 0.063)
>   - Test 16 — Config/stats: `recency_boost` presente in `get_stats()["config"]`, tipo `float`
>   - Test 17 — Conservazione ranking dominante: score 10x > non sovvertito (0.103 vs 0.011), score vicini → paper recente supera (0.0209 vs 0.0206)

---

### [x] Task C.3: Recency weighting nell'L2 summarizer

**File:** `paperhunter/summarization/level2_summarizer.py` (modifica), `paperhunter/config.py` (modifica), `config.yaml` (modifica), `paperhunter/indexing/hybrid_retriever.py` (modifica), `paperhunter/ingestion/ingestion_pipeline.py` (modifica)  
**Stato attuale:** `_format_l1_summaries_for_prompt()` formatta tutti gli L1 con lo stesso livello di dettaglio, ordinati cronologicamente.  
**Obiettivo:** Dedicare più spazio ai paper recenti nel prompt L2, con toggle globale configurabile.  
**Dettagli:**
- Paper degli ultimi 2 anni: testo completo dell'L1
- Paper 2-4 anni fa: testo ridotto (~70%)
- Paper > 4 anni fa: solo key_contributions + testing_methodology (~40%)
- Queste soglie devono essere configurabili da `config.yaml`

> **Completato 2/03/2026.** Modifiche:
> - **`config.yaml`:** Aggiunto parametro globale `recency_priority: true` — toggle master per tutta la logica di priorità recency (L2 summarizer, hybrid retriever, ingestion pipeline). Se `false`, tutti i paper vengono trattati allo stesso modo indipendentemente dall'anno
> - **`paperhunter/config.py`:** Aggiunto campo `recency_priority: bool = True` alla classe `Settings` — accessibile come `settings.recency_priority`
> - **`paperhunter/summarization/level2_summarizer.py`:** Modifiche principali:
>   - **Prompt template `_L2_PROMPT_TEMPLATE`:** Aggiunto blocco `**NOTE ON DETAIL LEVELS**` che informa il LLM sui livelli di dettaglio per recency (FULL, REDUCED, COMPRESSED), con soglie `{recency_full_years}` e `{recency_reduced_years}` parametriche da config. Istruzione esplicita: "Give analytical priority to recent papers with full details"
>   - **Nuovi placeholder prompt:** `{recency_full_years}` e `{recency_reduced_years}` — riempiti dalle soglie configurate in `l2_recency_tiers`
>   - **Istruzione IMPORTANT aggiornata:** Aggiunta nota "Papers marked [COMPRESSED] have less detail — do not penalize them, but give more analytical depth to recent papers"
>   - **`_format_l1_summaries_for_prompt()`:** Legge `settings.recency_priority` dinamicamente (via `import paperhunter.config as _cfg`). Se `recency_priority=False`: assegna tier `"full"` a tutti i paper (nessuna compressione per recency); preamble con "uniform detail". Se `recency_priority=True`: comportamento recency-aware come prima (tier based su anno + preamble con statistiche FULL/REDUCED/COMPRESSED)
>   - **`generate_l2_summary()`:** Legge `recency_priority` dinamicamente. Logga messaggi differenziati: se ON, mostra distribuzione tier; se OFF, messaggio "tutti i paper trattati a dettaglio completo"
>   - **Preamble recency:** Aggiunto header `=== Recency-Weighted Paper Summaries (N papers) ===` con statistiche di distribuzione tier (FULL/REDUCED/COMPRESSED). Quando OFF: `=== Paper Summaries (N papers, uniform detail) ===`
>   - **`_save_l2_metadata_to_disk()`:** Nuovo parametro `recency_tier_stats: dict | None` — include statistiche distribuzione tier nel file JSON di metadata. Chiamata aggiornata in `generate_l2_summary()` per passare le stats
>   - **Rimosso `_save_l2_metadata_to_disk` duplicato** in `_save_l2_to_store()` — ora la persistenza metadata avviene solo in `generate_l2_summary()`
> - **`paperhunter/indexing/hybrid_retriever.py`:** `HybridRetriever.__init__()` ora rispetta `settings.recency_priority`: se `False`, imposta `self._recency_boost = 0.0` indipendentemente dal valore configurato in `retrieval.recency_boost`
> - **`paperhunter/ingestion/ingestion_pipeline.py`:** Step 1c (ordinamento per recency) ora condizionato da `settings.recency_priority`: se `False`, skip dell'ordinamento con messaggio "Ordinamento per recency DISABILITATO"
> - **Test:** 7/7 passati + 9/9 test budget-aware L2 esistenti + 17/17 test hybrid retriever:
>   - T1: Default `recency_priority = True` da config.yaml
>   - T2: recency ON → header "Recency-Weighted", [COMPRESSED] per paper vecchi
>   - T3: `_compute_recency_tier_stats()`: distribuzione corretta (full=2, reduced=1, compressed=2)
>   - T4: recency OFF → header "uniform detail", nessun [COMPRESSED]/[REDUCED], tutti i paper a dettaglio completo (5/5 Limitations presenti)
>   - T5: Output OFF più lungo di output ON (1227 vs 1129 chars — tutti i paper espansi)
>   - T6: Hybrid retriever: recency OFF → boost=0.0, recency ON → boost=0.1 (da config)
>   - T7: config.yaml contiene `recency_priority: true`
>   - Regressione L2 budget-aware: 9/9 test passati (0.2s)
>   - Regressione hybrid retriever: 17/17 test passati

---

#### [  ] Task FINALE: Aggiornare README.md con le nuove funzionalità, con esempi e brevi descrizioni — Documentazione completa

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
