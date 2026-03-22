# TODO7_AutoResearchClaw_Proposals.md — Self-Evolution, FigureAgent, Skill Injection, Bridge ResearchClaw

> **Riferimento:** `proposte_implementazioni.txt`  
> **Data inizio:** [Data da definire]  
> **Ultimo aggiornamento:** Marzo 2026  
> **Target:** 4 implementazioni ispirate da AutoResearchClaw per potenziare PaperHunter: sistema di self-evolution con lezioni apprese, generazione automatica di figure scientifiche, skill injection persistenti nei prompt, e bridge opzionale verso AutoResearchClaw per esperimenti post-survey

---

## 🏗️ Architettura Estesa (Nuovi/Modificati)

```
paperhunter/
├── evolution/                          ← NUOVO PACKAGE (Fase 1 + 3)
│   ├── __init__.py
│   ├── lesson_store.py                 ← Store JSONL append-only + query con decadimento temporale
│   ├── lesson_extractor.py             ← Estrae lezioni da risultati pipeline (verify, summarize, etc.)
│   ├── prompt_overlay.py               ← Genera overlay testuale da iniettare nei prompt LLM
│   └── skill_loader.py                 ← Carica skill persistenti da data/skills/*.md
│
├── agents/
│   ├── figure_agent/                   ← NUOVO SUB-PACKAGE (Fase 2)
│   │   ├── __init__.py
│   │   ├── planner.py                  ← Analizza paper e decide quali figure servono
│   │   ├── decision.py                 ← Sceglie: code_figure vs image_figure
│   │   ├── codegen.py                  ← Genera codice Matplotlib/Seaborn
│   │   ├── renderer.py                 ← Esegue codice in sandbox → salva PNG
│   │   ├── critic.py                   ← Valuta qualità/leggibilità/stile accademico
│   │   └── integrator.py              ← Inserisce figure nel Markdown/LaTeX finale
│   ├── writer_agent.py                 ← MODIFICATO: inietta overlay lezioni + skill nel system prompt
│   ├── planner_agent.py                ← MODIFICATO: inietta overlay per bilanciamento outline
│   └── critic_agent.py                 ← MODIFICATO: registra lezioni dopo verify falliti
│
├── bridge/                             ← NUOVO PACKAGE (Fase 4, opzionale)
│   ├── __init__.py
│   └── researchclaw_adapter.py         ← Esporta dati PaperHunter → formato AutoResearchClaw
│
├── models.py                           ← MODIFICATO: nuovi modelli dati per lessons e figures
└── config.py                           ← MODIFICATO: config sections per evolution, figure_agent, skills

data/
├── evolution/
│   └── lessons.jsonl                   ← Storage lezioni (append-only, ~1KB per lezione)
└── skills/                             ← Skill persistenti (non scadono)
    ├── academic_writing.md
    ├── citation_rules.md
    └── domain_llm_sec.md

config.yaml                             ← Nuove sezioni: evolution, figure_agent, skills
```

---

## 📋 TASK DETTAGLIATI

---

### FASE 1 — Self-Evolution System (Lessons from Failures)

> **Priorità:** ALTA  
> **Complessità:** Media (~300-400 righe)  
> **Principio:** Dopo ogni run il sistema registra errori/warning in un JSONL; nelle run successive le "lezioni" vengono iniettate come overlay nei prompt LLM con decadimento esponenziale (half-life 30gg, scadenza 90gg).

#### [x] Task 1.1: Creare `lesson_store.py` — storage JSONL con query e decadimento

> **Completato:** Creato `paperhunter/evolution/lesson_store.py` (266 righe). Modello `Lesson` (Pydantic) con stage, node, category, severity, section_key, bibtex_keys. Classe `LessonStore` con `record()`, `query()`, `prune()`, `count()`, `clear()`. Decadimento esponenziale `exp(-age×ln2/30)`. Funzione `auto_classify()` per categorizzazione keyword-based. Thread-safe.

**File:** `paperhunter/evolution/lesson_store.py` (nuovo)  
**Obiettivo:** Modulo JSONL append-only per registrare e interrogare le lezioni apprese.  
**Dettagli:**
- Classe `LessonStore(path: Path)` con metodi:
  - `record(lesson: Lesson) → None` — append al file JSONL
  - `query(stage: str, node: str | None, section_key: str | None, top_k: int) → list[Lesson]` — filtro + ranking per peso temporale
  - `prune() → int` — rimuove lezioni scadute (>90 giorni)
- Struttura `Lesson`:
  ```json
  {
    "stage_name": "write",
    "node_name": "verify",
    "category": "writing",
    "severity": "warning",
    "description": "Sezione 'prompt_injection' aveva 4/10 citazioni NOT_SUPPORTED...",
    "section_key": "prompt_injection",
    "bibtex_keys_involved": ["Zou_2023_GCG"],
    "timestamp": "2026-03-22T11:00:00Z",
    "run_id": "write_20260322_110000"
  }
  ```
- Formula peso temporale: `weight = exp(-age_days × ln(2) / 30)`
- Classificazione automatica categorie (keyword-based):
  - `"timeout|connection|memory"` → system
  - `"NOT_SUPPORTED|SOURCE_NOT_FOUND"` → writing
  - `"JSON|parse|format"` → pipeline
  - `"hallucin|fabricat|invent"` → writing

---

#### [x] Task 1.2: Creare `lesson_extractor.py` — estrazione lezioni dai risultati

> **Completato:** Creato `paperhunter/evolution/lesson_extractor.py` (240 righe). 4 funzioni di estrazione: `extract_from_verification()` genera 1 lesson per sezione fallita + 1 per ogni citazione NOT_SUPPORTED (con claim e reasoning del critic); `extract_from_summarization()` per fallimenti L1/L2 con auto-classify; `extract_from_pipeline()` per errori generici; `extract_from_all_verifications()` batch helper. Accetta sia dict che oggetti VerificationResult.

**File:** `paperhunter/evolution/lesson_extractor.py` (nuovo)  
**Obiettivo:** Estrarre automaticamente lezioni utili dai risultati del pipeline.  
**Dettagli:**
- Funzione `extract_from_verification(section_key, verification_result) → list[Lesson]`
  - Per ogni sezione con `pass_rate < threshold`: genera lezione con bibtex_key non supportati e tipo errore
- Funzione `extract_from_summarization(paper_id, error_info) → Lesson`
  - Se L1/L2 fallisce (timeout, JSON malformato): genera lezione con paper_id e tipo errore
- Funzione `extract_from_pipeline(node_name, error) → Lesson`
  - Per errori generici di pipeline (loop eccessivi, timeout, crash)

---

#### [x] Task 1.3: Creare `prompt_overlay.py` — generazione overlay per prompt LLM

> **Completato:** Creato `paperhunter/evolution/prompt_overlay.py` (200 righe). Funzione principale `generate_overlay()` con filtri stage/node/section_key/category + limiti max_lessons e max_chars. Shortcut dedicati: `generate_overlay_for_writer()` (sezione specifica + fill con writing generali, deduplicati) e `generate_overlay_for_planner()` (plan lessons + writing fallback). Output Markdown con emoji severità (❌/⚠️/ℹ️), bibtex keys, header/footer contestuali.

**File:** `paperhunter/evolution/prompt_overlay.py` (nuovo)  
**Obiettivo:** Generare blocco di testo "Lessons from Prior Runs" da iniettare nei prompt.  
**Dettagli:**
- Funzione `generate_overlay(store: LessonStore, stage, node, section_key, max_lessons=5) → str`
- Output formattato:
  ```
  ## Lessons from Prior Runs
  1. ❌ [writing] Section 'adversarial_attacks' had 3 citations NOT_SUPPORTED.
     Avoid citing Chowdhury_2024_Breaking for ASR claims.
  2. ⚠️ [writing] Section 'hallucinations' exceeded target length by 40%.
  Use these lessons to avoid repeating past mistakes.
  ```
- Le lezioni sono ordinate per peso (recenti + gravi prima)
- Lunghezza massima overlay: ~500 token per non saturare il contesto

---

#### [x] Task 1.4: Integrare overlay nel writer e nel critic

> **Completato:** Integrazione self-evolution in 4 file. `config.py`: aggiunta classe `EvolutionConfig` (enabled, lessons_path, half_life_days, expiry_days, max_overlay_lessons, max_overlay_chars, auto_prune) e campo `evolution` in `Settings`. `writer_agent.py`: iniettato overlay lezioni in 3 punti (agentic, prefetch_agentic, static) dopo costruzione system prompt, con guard `settings.evolution.enabled` e try/except. `critic_agent.py`: aggiunta registrazione lezioni dopo `compute_pass_rate()` in `verify_section()` quando la verifica fallisce. `planner_agent.py`: iniettato overlay planner dopo `_PLANNER_PROMPT_TEMPLATE.format()`. Tutti i punti sono non-breaking (disabilitati di default, wrap try/except). Test import + integrazione end-to-end passati.

**File:** `paperhunter/agents/writer_agent.py` (modifica), `paperhunter/agents/critic_agent.py` (modifica), `paperhunter/agents/planner_agent.py` (modifica)  
**Obiettivo:** Iniettare l'overlay delle lezioni apprese nei system prompt degli agenti.  
**Dettagli:**
- **writer_agent.py**: prima di scrivere una sezione, caricare overlay con `stage="write"` + `section_key` e aggiungerlo al system prompt
- **critic_agent.py**: dopo ogni verify con `pass_rate < threshold`, chiamare `lesson_extractor` per registrare la lezione
- **planner_agent.py**: caricare overlay con `stage="plan"` per suggerire bilanciamento outline basato su errori passati

---

#### [x] Task 1.5: Aggiornare `config.yaml` con sezione `evolution`

> **Completato:** Aggiunta sezione `evolution:` in `config.yaml` dopo `logging:`. 7 parametri con commenti italiani: `enabled: false` (default off, zero overhead), `lessons_path`, `half_life_days: 30`, `expiry_days: 90`, `max_overlay_lessons: 5`, `max_overlay_chars: 2000`, `auto_prune: true`. Verificato caricamento corretto tramite `settings.evolution.*`.

**File:** `config.yaml` (modifica)  
**Dettagli:**
```yaml
evolution:
  enabled: true
  lessons_path: "data/evolution/lessons.jsonl"
  half_life_days: 30
  expiry_days: 90
  max_overlay_lessons: 5
  max_overlay_tokens: 500
  auto_prune: true
```

---

### FASE 2 — FigureAgent (Generazione automatica di figure)

> **Priorità:** MEDIA  
> **Complessità:** Alta (~800-1000 righe, nuovo sub-package)  
> **Principio:** Analizza il survey generato e produce figure scientifiche: code figures (bar chart, timeline, heatmap → Matplotlib) e image figures (tassonomie, flowchart → Gemini). Un Decision Agent decide il tipo per ogni figura.

#### [x] Task 2.1: Creare `planner.py` — pianificazione figure necessarie

> **Completato:** Creato `paperhunter/agents/figure_agent/planner.py` (~320 righe) + `__init__.py`. Modelli Pydantic: `FigureSpec` (figure_id, figure_type CODE/IMAGE, chart_type con 14 tipi, title, caption, description, section_key, data_sources, data_points, priority, rationale), `FigurePlan` (con helper code_figures/image_figures/for_section), `ChartType` enum, `FigureType` enum. Prompt LLM dettagliato con vincoli min/max figure, tipi supportati, formato output JSON. Parser multi-livello (4 fallback). Funzione `plan_figures()` con truncation paper a 60k chars, rich console output. Utility `save_figure_plan`/`load_figure_plan` per persistenza JSON. Tutti i test passati: modelli, parsing, validazione, save/load, gestione spec invalide.

**File:** `paperhunter/agents/figure_agent/planner.py` (nuovo)  
**Obiettivo:** Analizzare il paper finale e decidere quali figure servono e dove.  
**Dettagli:**
- Input: `final_paper_markdown` + `l2_summaries`
- Output: `figure_plan` — lista di specifiche:
  ```json
  {
    "figure_id": "fig_attack_timeline",
    "type": "code",
    "location": "section_adversarial_attacks",
    "description": "Timeline of adversarial attacks on LLMs (2022-2026)",
    "data_sources": ["l2_adversarial_attacks", "property_graph"],
    "chart_type": "timeline"
  }
  ```
- Tipi di figure utili per un survey LLM Security:
  - **Bar chart**: confronto metriche tra attacchi/difese (ASR, success rate)
  - **Timeline**: evoluzione temporale degli approcci per categoria
  - **Heatmap**: matrice attacco×difesa con efficacia
  - **Radar chart**: confronto multi-dimensionale di metodi
  - **Taxonomy diagram**: tassonomia gerarchica delle minacce

---

#### [x] Task 2.2: Creare `decision.py` — agente decisionale tipo figura

> **Completato:** Creato `paperhunter/agents/figure_agent/decision.py` (~165 righe). Logica decisionale a 3 livelli: (1) mapping deterministico ChartType → FigureType (11 code types, 3 image types), (2) analisi keyword su title+description (~20 keyword per categoria), (3) fallback su presenza dati quantitativi in data_points. Default: CODE se nessun segnale forte. `classify_all()` reclassifica batch con logging. Nessuna chiamata LLM necessaria — decisione puramente rule-based.

**File:** `paperhunter/agents/figure_agent/decision.py` (nuovo)  
**Obiettivo:** Per ogni specifica nel piano, decidere se generare code figure o image figure.  
**Dettagli:**
- Se i dati sono quantitativi (numeri, metriche, percentuali) → `code`
- Se serve un diagramma concettuale (tassonomia, architettura, flusso) → `image`
- Usa LLM per casi ambigui

---

#### [x] Task 2.3: Creare `codegen.py` + `renderer.py` — generazione ed esecuzione code figures

> **Completato:** Creati `codegen.py` (~195 righe) e `renderer.py` (~200 righe). `codegen.py`: prompt LLM dettagliato con stile accademico (DPI 300, font serif, colormap professionali), estrazione codice Python da response (3 fallback), validazione con auto-fix (OUTPUT_PATH, savefig, close). `renderer.py`: esecuzione in subprocess isolato con timeout 60s, verifica PNG (size 1KB-50MB), retry loop con feedback errore (max 3 iterazioni), salvataggio script debug, `render_all_code_figures()` batch con progress. Aggiunti `matplotlib>=3.8.0` e `seaborn>=0.13.0` a requirements.txt.

**File:** `paperhunter/agents/figure_agent/codegen.py` (nuovo), `renderer.py` (nuovo)  
**Obiettivo:** Generare codice Matplotlib/Seaborn e renderizzarlo in PNG.  
**Dettagli:**
- **codegen.py**: LLM genera script Python self-contained con dati estratti dal survey
- **renderer.py**: esegue lo script in subprocess isolato, cattura PNG, gestisce errori
- Stile accademico: `seaborn-v0_8-paper`, DPI 300, font leggibili
- Retry con feedback dell'errore (max 3 iterazioni)

---

#### [ ] Task 2.4: Creare `critic.py` — valutazione qualità figure

**File:** `paperhunter/agents/figure_agent/critic.py` (nuovo)  
**Obiettivo:** Verificare che le figure siano leggibili, accurate e in stile accademico.  
**Dettagli:**
- Valuta: leggibilità label/assi, correttezza dati vs fonti, stile coerente
- Output: APPROVED / NEEDS_REVISION con feedback specifico
- Se bocciata → feedback al codegen per rigenera

---

#### [ ] Task 2.5: Creare `integrator.py` — inserimento figure nel paper

**File:** `paperhunter/agents/figure_agent/integrator.py` (nuovo)  
**Obiettivo:** Inserire riferimenti alle figure nel Markdown/LaTeX finale.  
**Dettagli:**
- Inserisce `![Caption](figures/fig_xxx.png)` nel punto appropriato del paper
- Genera didascalie accademiche con `Figure N: ...`
- Supporto LaTeX opzionale (`\includegraphics` + `\caption`)

---

#### [ ] Task 2.6: Integrare FigureAgent nel pipeline o come comando standalone

**File:** `paperhunter/main.py` (modifica), opzionalmente `paperhunter/agents/pipeline.py` (modifica)  
**Obiettivo:** Rendere il FigureAgent invocabile.  
**Dettagli:**
- Opzione A: nuovo nodo LangGraph dopo "assemble": `plan → write ↔ verify → assemble → figure → END`
- Opzione B (preferibile): comando standalone `paperhunter figures` senza toccare il pipeline write
- Config in `config.yaml`:
  ```yaml
  figure_agent:
    enabled: true
    min_figures: 3
    max_figures: 8
    max_iterations: 3
    output_format: "python"   # python | tikz
    dpi: 300
    style: "seaborn-v0_8-paper"
  ```

---

### FASE 3 — Skill Injection (Prompt Overlay persistenti)

> **Priorità:** BASSA (estensione naturale della Fase 1)  
> **Complessità:** Bassa (~150 righe)  
> **Principio:** Oltre alle lezioni con scadenza, permette di definire "skill" persistenti in Markdown — regole/pattern che il pipeline deve SEMPRE seguire (stile del relatore, regole citazioni, terminologia dominio).

#### [ ] Task 3.1: Creare `skill_loader.py` — caricamento skill da Markdown

**File:** `paperhunter/evolution/skill_loader.py` (nuovo)  
**Obiettivo:** Caricare file `.md` dalla directory `data/skills/` e combinarli con l'overlay lezioni.  
**Dettagli:**
- Funzione `load_skills(skills_dir: Path) → str` — concatena tutti i `.md` in un blocco unico
- Le skill NON hanno scadenza (a differenza delle lezioni)
- Vengono iniettate nel system prompt insieme all'overlay lezioni

---

#### [ ] Task 3.2: Creare skill iniziali per il dominio LLM Security

**File:** `data/skills/academic_writing.md`, `data/skills/citation_rules.md`, `data/skills/domain_llm_sec.md` (nuovi)  
**Dettagli:**
- **academic_writing.md**: stile accademico, struttura frasi, transizioni
- **citation_rules.md**: regole citazioni (mai citare un survey per risultati interni, usare contesto temporale, preferire numeri esatti)
- **domain_llm_sec.md**: terminologia LLM security, acronimi, definizioni standard (ASR, jailbreak, prompt injection, etc.)

---

#### [ ] Task 3.3: Integrare skill nel pipeline di overlay

**File:** `paperhunter/evolution/prompt_overlay.py` (modifica)  
**Obiettivo:** Concatenare skill + lezioni in un unico overlay iniettato nei prompt.  
**Dettagli:**
- Ordine nel prompt: `[System Prompt] + [Skills] + [Lessons] + [Task-specific instructions]`
- Config in `config.yaml`:
  ```yaml
  skills:
    enabled: true
    skills_dir: "data/skills"
  ```

---

### FASE 4 — Bridge AutoResearchClaw (Integrazione sperimentale, opzionale)

> **Priorità:** MEDIA (opzionale, per sperimentazione avanzata post-survey)  
> **Complessità:** Media (~400 righe)  
> **Principio:** Dopo aver completato il survey con PaperHunter, esporta dati (gap, ipotesi, knowledge graph) nel formato AutoResearchClaw e lancia il pipeline sperimentale dal Stage 9 (EXPERIMENT_DESIGN).

#### [ ] Task 4.1: Creare `researchclaw_adapter.py` — export dati PaperHunter → ResearchClaw

**File:** `paperhunter/bridge/researchclaw_adapter.py` (nuovo)  
**Obiettivo:** Esportare i dati del survey nel formato atteso da AutoResearchClaw.  
**Dettagli:**
- Classe `ResearchClawAdapter(paperhunter_output_dir, researchclaw_run_dir)`
- Metodi:
  - `export_hypotheses()` — `L2Summary.open_gaps` → `stage-08/hypotheses.md`
  - `export_synthesis()` — `L2Summary.synthesis + taxonomy` → `stage-07/synthesis.md`
  - `export_knowledge_cards()` — Property Graph (387 entità) → `stage-06/cards/*.json`
  - `create_checkpoint()` — genera `checkpoint.json` con `last_completed_stage: 8`
  - `export_references()` — copia BibTeX per citation_verify (Stage 23)
- Mapping dati:
  - L2 `open_gaps` → ipotesi numerate con contesto categoria
  - Graph `entities + relations` → knowledge cards JSON
  - `output/references.bib` → copia diretta

---

#### [ ] Task 4.2: Creare comando CLI `paperhunter experiment`

**File:** `paperhunter/main.py` (modifica)  
**Obiettivo:** Aggiungere sottocomando per lanciare esperimenti via AutoResearchClaw.  
**Dettagli:**
- `paperhunter experiment --output ./experiments/run_001`
- Flusso:
  1. Verifica che il survey sia completo (checkpoints OK)
  2. Chiama `ResearchClawAdapter.run()` per esportare dati
  3. Lancia `researchclaw run --from-stage EXPERIMENT_DESIGN --approve-gates`
  4. Al termine, importa risultati e figure nel survey
- Requisiti: AutoResearchClaw installato, API key configurata, sandbox locale o Docker

---

#### [ ] Task 4.3: Importare risultati sperimentali nel survey

**File:** `paperhunter/bridge/researchclaw_adapter.py` (estensione)  
**Obiettivo:** Dopo che ResearchClaw completa, importare risultati nel paper PaperHunter.  
**Dettagli:**
- Leggere output sperimentali da ResearchClaw (figure, tabelle, testo)
- Creare sezione "Experimental Validation" nel survey
- Integrare figure generate nell'output PaperHunter
- Nota: non tutti i gap sono sperimentabili; gli esperimenti più utili per un survey sono:
  - Analisi quantitativa comparativa (metriche tra paper)
  - Riproduzione risultati chiave
  - Benchmark su dataset pubblici (AdvBench, TruthfulQA)
  - Analisi statistica delle tendenze temporali

---

## 📈 ROADMAP RIEPILOGATIVA

```
FASE 1 — Self-Evolution System                          [Impatto immediato, bassa complessità]
├── 1.1 lesson_store.py                                  [x]
├── 1.2 lesson_extractor.py                              [x]
├── 1.3 prompt_overlay.py                                [x]
├── 1.4 Integrazione overlay in writer/critic/planner    [x]
└── 1.5 Config yaml sezione evolution                    [x]

FASE 2 — FigureAgent                                     [Alto impatto visivo, alta complessità]
├── 2.1 planner.py                                       [ ]
├── 2.2 decision.py                                      [ ]
├── 2.3 codegen.py + renderer.py                         [ ]
├── 2.4 critic.py                                        [ ]
├── 2.5 integrator.py                                    [ ]
└── 2.6 Integrazione pipeline / comando standalone       [ ]

FASE 3 — Skill Injection                                 [Estensione Fase 1, bassa complessità]
├── 3.1 skill_loader.py                                  [ ]
├── 3.2 Skill iniziali dominio LLM Security              [ ]
└── 3.3 Integrazione in prompt_overlay.py                [ ]

FASE 4 — Bridge AutoResearchClaw                         [Opzionale, per sperimentazione avanzata]
├── 4.1 researchclaw_adapter.py                          [ ]
├── 4.2 Comando CLI `paperhunter experiment`             [ ]
└── 4.3 Import risultati nel survey                      [ ]
```

---

## 📝 NOTE

- Tutte le implementazioni sono **ADDITIVE**: non richiedono modifiche distruttive al codice esistente, solo nuovi moduli + piccole modifiche ai prompt
- Il Self-Evolution (Fase 1) è l'unico che richiede modifiche ai prompt esistenti (aggiunta overlay nel system prompt)
- Il FigureAgent (Fase 2) può essere un comando standalone (`paperhunter figures`) senza toccare il pipeline write
- Il Bridge ResearchClaw (Fase 4) è completamente opzionale e indipendente dal resto
- **Escluso:** Human-in-the-loop gates (come da richiesta utente)
