# PaperHunter v3.1

**End-to-end LLM Security Survey Generator — from PDF ingestion to Nature/ACL-style paper**

PaperHunter è un sistema modulare e agentico che genera autonomamente un survey paper scientifico sulla sicurezza degli LLM. A partire da oltre 1.000 PDF scientifici categorizzati, esegue una pipeline completa: estrazione strutturata (GROBID), arricchimento metadati (Semantic Scholar), summarization gerarchica RAPTOR con clustering semantico, GraphRAG community-driven, retrieval agentico multi-step, scrittura multi-agente (LangGraph ReAct), filtraggio intelligente delle immagini con embedding multimodali, ed export LaTeX — il tutto con citazioni verificabili e tracciamento della provenienza.

---

## Table of Contents

- [Visione d'Insieme](#visione-dinsieme)
- [Perché Tre Livelli di Summary (L1 → L1.5 → L2)](#perché-tre-livelli-di-summary-l1--l15--l2)
  - [Il Problema: Context Rot](#il-problema-context-rot)
  - [La Soluzione: Compressione Gerarchica Ispirata a RAPTOR](#la-soluzione-compressione-gerarchica-ispirata-a-raptor)
  - [L1 — Summary per Singolo Paper](#l1--summary-per-singolo-paper)
  - [L1.5 — Summary per Cluster Semantico (RAPTOR)](#l15--summary-per-cluster-semantico-raptor)
  - [L2 — Meta-Summary per Categoria](#l2--meta-summary-per-categoria)
  - [Come i Tre Livelli Collaborano](#come-i-tre-livelli-collaborano)
- [Fase 1 — Ingestion (`paperhunter ingest`)](#fase-1--ingestion-paperhunter-ingest)
  - [1.1 Scansione e Rilevamento Incrementale](#11-scansione-e-rilevamento-incrementale)
  - [1.2 Estrazione Strutturata con GROBID](#12-estrazione-strutturata-con-grobid)
  - [1.3 Arricchimento con Semantic Scholar](#13-arricchimento-con-semantic-scholar)
  - [1.4 Generazione BibTeX Automatica](#14-generazione-bibtex-automatica)
  - [1.5 Chunking GROBID-Aware](#15-chunking-grobid-aware)
  - [1.6 Pipeline di Elaborazione Immagini (3 Stadi)](#16-pipeline-di-elaborazione-immagini-3-stadi)
  - [1.7 Indicizzazione Vettoriale (Qdrant)](#17-indicizzazione-vettoriale-qdrant)
- [Fase 2 — Summarization (`paperhunter summarize`)](#fase-2--summarization-paperhunter-summarize)
  - [2.1 Generazione L1 — Un Summary per Paper](#21-generazione-l1--un-summary-per-paper)
  - [2.2 Clustering Semantico (RAPTOR)](#22-clustering-semantico-raptor)
  - [2.3 Generazione L1.5 — Un Summary per Cluster](#23-generazione-l15--un-summary-per-cluster)
  - [2.4 Generazione L2 — Un Meta-Summary per Categoria](#24-generazione-l2--un-meta-summary-per-categoria)
  - [2.5 Auto-Invalidazione](#25-auto-invalidazione)
- [Fase 3 — Indexing (`paperhunter index`)](#fase-3--indexing-paperhunter-index)
  - [3.1 Indice BM25 (Keyword Search)](#31-indice-bm25-keyword-search)
  - [3.2 Knowledge Graph (PropertyGraph + Entità)](#32-knowledge-graph-propertygraph--entità)
  - [3.3 Community Detection (Louvain/Leiden)](#33-community-detection-louvainleiden)
  - [3.4 Hybrid Retrieval con Reciprocal Rank Fusion](#34-hybrid-retrieval-con-reciprocal-rank-fusion)
  - [3.5 Reranking con Gemini Flash](#35-reranking-con-gemini-flash)
- [Fase 4 — Writing (`paperhunter write`)](#fase-4--writing-paperhunter-write)
  - [4.1 Planner Agent — Pianificazione dell'Outline](#41-planner-agent--pianificazione-delloutline)
  - [4.2 Writer Agent — Scrittura delle Sezioni](#42-writer-agent--scrittura-delle-sezioni)
  - [4.3 Critic Agent — Verifica della Provenienza](#43-critic-agent--verifica-della-provenienza)
  - [4.4 Il Loop Writer ↔ Critic](#44-il-loop-writer--critic)
  - [4.5 Assembler Agent — Assemblaggio Finale](#45-assembler-agent--assemblaggio-finale)
  - [4.6 Checkpoint e Ripresa da Errore](#46-checkpoint-e-ripresa-da-errore)
- [Post-Processing e Output](#post-processing-e-output)
- [Architettura Complessiva — Schema di Flusso](#architettura-complessiva--schema-di-flusso)
- [Le 7 Collezioni Qdrant](#le-7-collezioni-qdrant)
- [Modelli LLM e Strategia di Utilizzo](#modelli-llm-e-strategia-di-utilizzo)
- [Budget-Awareness e Scalabilità](#budget-awareness-e-scalabilità)
- [Recency Priority](#recency-priority)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Stima Chiamate API](#stima-chiamate-api)
- [Struttura del Progetto](#struttura-del-progetto)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Visione d'Insieme

PaperHunter affronta un problema concreto: **come generare automaticamente un survey paper accademico di alta qualità partendo da centinaia o migliaia di PDF scientifici?**

Il sistema è progettato per il dominio della sicurezza degli LLM (prompt injection, hallucinations, data poisoning, adversarial attacks), ma l'architettura è generalizzabile a qualsiasi dominio di ricerca.

La pipeline si articola in **4 fasi principali**, ciascuna invocabile come comando CLI:

```
paperhunter ingest     →  PDF → testo strutturato → chunk → vettori
paperhunter summarize  →  chunk → L1 → L1.5 → L2 (gerarchia di summary)
paperhunter index      →  BM25 + Knowledge Graph + Community Detection
paperhunter write      →  Planner → Writer → Critic → Assembler → Survey finale
```

Il risultato finale è un **survey paper completo** in Markdown, con citazioni verificabili `[bibtex_key]`, esportabile in LaTeX (formato ACL, IEEEtran o plain), corredato da bibliografia filtrata, report di provenienza e versioning automatico.

---

## Perché Tre Livelli di Summary (L1 → L1.5 → L2)

### Il Problema: Context Rot

I modelli LLM hanno una finestra di contesto limitata (tipicamente 128k–1M token). Se si tenta di inserire il testo completo di 1.000 paper in un singolo prompt, succedono due cose:

1. **Non ci sta fisicamente**: 1.000 paper × ~10k token/paper = 10M token — ordini di grandezza oltre la finestra di contesto di qualsiasi modello.
2. **Context rot**: anche con finestre di contesto molto grandi (es. 1M token), la qualità dell'output degrada significativamente man mano che si aggiunge contesto. Le informazioni nel "mezzo" del contesto vengono progressivamente ignorate (fenomeno noto come "lost in the middle").

### La Soluzione: Compressione Gerarchica Ispirata a RAPTOR

La soluzione è una **piramide di compressione a 3 livelli** ispirata all'architettura RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval):

```
                    ┌───────────┐
                    │    L2     │   4 meta-summary (~6k token ciascuno)
                    │ (per cat) │   = ~24k token totali
                    └─────┬─────┘
                          │
                ┌─────────┴──────────┐
                │       L1.5         │   ~50 cluster summary (~2k token ciascuno)
                │   (per cluster)    │   = ~100k token totali
                └─────────┬──────────┘
                          │
          ┌───────────────┴───────────────┐
          │              L1               │   1.000+ summary individuali (~2k token ciascuno)
          │         (per paper)           │   = ~2M token totali
          └───────────────┬───────────────┘
                          │
    ┌─────────────────────┴─────────────────────┐
    │                 Chunk                      │   ~50k chunk (~1.5k token ciascuno)
    │            (testo originale)               │   = ~75M token totali (testo grezzo dei PDF)
    └───────────────────────────────────────────┘
```

Ogni livello **comprime** le informazioni del livello sottostante preservando ciò che è essenziale per il survey.

### L1 — Summary per Singolo Paper

**Che cos'è:** Un riassunto strutturato di ogni singolo paper, generato da Gemini Pro leggendo i chunk del testo originale.

**Perché serve:** Il testo grezzo di un paper (20-40 pagine) contiene troppi dettagli irrilevanti per un survey (formule intermedie, dettagli implementativi minori, ringraziamenti, ecc.). L'L1 estrae e struttura solo le informazioni utili:

- **Contributi chiave** — cosa apporta questo paper al campo
- **Metodologia** — approccio tecnico in sintesi
- **Testing methodology** — come è stato validato (benchmark, dataset, metriche) — campo critico per il survey
- **Contesto temporale** — dove si posiziona rispetto allo stato dell'arte
- **Risultati quantitativi** — numeri chiave
- **Limitazioni** — punti deboli dichiarati o inferiti

**Quanto comprime:** Da ~10k token (paper completo) a ~2k token (L1 summary) = compressione **5×**.

**Budget-awareness:** La lunghezza dell'L1 si adatta automaticamente al numero di paper nella categoria:
- ≤30 paper/categoria → L1 "standard" (~2-3k caratteri)
- 31-70 paper → L1 "conciso" (~1.5k caratteri)
- \>70 paper → L1 "compresso" (~1k caratteri)

### L1.5 — Summary per Cluster Semantico (RAPTOR)

**Che cos'è:** Un meta-summary che sintetizza un **gruppo di paper tematicamente affini**, individuati tramite clustering semantico sugli embedding degli L1.

**Perché serve:** Con 100+ paper per categoria, anche gli L1 (già compressi) sarebbero troppi per generare un buon L2. Se li concatenassi tutti in un prompt, il modello si "perderebbe" nei dettagli. L'L1.5 introduce un livello intermedio che:

1. **Raggruppa** i paper per tema (es. "difese basate su perplexity per prompt injection" oppure "attacchi gradient-based contro LLM")
2. **Sintetizza** i pattern comuni e le divergenze all'interno di ciascun cluster
3. **Preserva le citazioni** — ogni affermazione mantiene il `[bibtex_key]` del paper originale

**Come funziona il clustering:**
- Gli embedding 3072-dim degli L1 (già memorizzati in Qdrant) vengono raggruppati con **AgglomerativeClustering** (scikit-learn)
- Soglia di similarità coseno configurabile (default: 0.75)
- I cluster troppo grandi (>20 paper) vengono automaticamente ri-suddivisi
- Risultato tipico: ~1.000 paper → ~50 cluster

**Quanto comprime:** Da ~1.000 L1 summary (~2M token) a ~50 cluster summary (~100k token) = compressione **20×**.

### L2 — Meta-Summary per Categoria

**Che cos'è:** Un unico meta-summary di 4-6k token che sintetizza **l'intero stato dell'arte** di una categoria (es. "Prompt Injection"), leggendo tutti gli L1.5 della categoria.

**Perché serve:** L'L2 è il **documento di riferimento** per il Planner Agent e il Writer Agent durante la fase di scrittura. Fornisce una visione panoramica che include:

- **Trend temporali** — come si è evoluta la ricerca nel tempo
- **Tassonomia** — sotto-aree e classificazione degli approcci
- **Gap colmati** — cosa è stato risolto dai paper nel corpus
- **Gap aperti** — problemi ancora irrisolti
- **Convergenze metodologiche** — punti di accordo tra paper diversi
- **Panoramica testing** — come la comunità valuta i risultati
- **Sintesi** — paragrafo narrativo pronto per l'introduzione del survey

**Due percorsi di generazione:**
1. **Percorso RAPTOR (preferito):** Se esistono L1.5, l'L2 li usa come input. Poiché ~50 cluster summary stanno comodamente in un singolo prompt, l'L2 può fare pattern recognition cross-cluster.
2. **Percorso legacy (fallback):** Se L1.5 non sono disponibili, comprime gli L1 con tier basati sulla recency (paper recenti = dettaglio pieno, paper vecchi = compressione progressiva).

### Come i Tre Livelli Collaborano

Il flusso complessivo della conoscenza è:

```
Testo grezzo dei PDF (75M token)
       │
       ▼ Gemini Pro (1 chiamata per paper)
L1: informazioni strutturate per paper (2M token)
       │
       ▼ Clustering semantico + Gemini Pro (1 chiamata per cluster)
L1.5: temi emergenti per cluster (100k token)
       │
       ▼ Gemini Pro (1 chiamata per categoria)
L2: visione panoramica per categoria (24k token)
       │
       ▼ Planner Agent + Writer Agent
Survey finale (~20k token di testo accademico)
```

**I livelli non sono ridondanti** — servono scopi diversi durante la scrittura:
- **L2** → usato dal **Planner** per decidere la struttura del survey e dal **Writer** per il contesto d'insieme
- **L1.5** → consultabili dal Writer quando serve un livello intermedio di dettaglio su un sotto-tema
- **L1** → consultabili via RAG quando il Writer ha bisogno di un dato specifico (es. "quale benchmark ha usato il paper X?")
- **Chunk** → livello più granulare, consultabile via RAG per citazioni precise e verifiche del Critic

---

## Fase 1 — Ingestion (`paperhunter ingest`)

L'ingestion è la fase che trasforma i PDF grezzi in una knowledge base strutturata e indicizzata. È il fondamento su cui si costruisce tutto il resto.

### 1.1 Scansione e Rilevamento Incrementale

Il sistema mantiene un **hash registry** (file JSON) che mappa ogni PDF al suo hash SHA-256. Ad ogni esecuzione:

1. Scansiona le sotto-cartelle di `scaricati/` (ogni sottocartella = una categoria)
2. Calcola l'hash SHA-256 di ogni PDF trovato
3. Confronta con il registry: solo i PDF **nuovi o modificati** vengono processati
4. Se non è cambiato nulla → 0 operazioni, uscita immediata

Questo rende l'ingestion **idempotente**: puoi rieseguirla quante volte vuoi senza conseguenze.

**Ordinamento per recency:** I PDF vengono processati in ordine di anno di pubblicazione decrescente (anno estratto dal nome del file). Questo significa che se la pipeline viene interrotta a metà, i paper più recenti (e quindi più rilevanti) sono già stati processati.

### 1.2 Estrazione Strutturata con GROBID

[GROBID](https://github.com/kermitt2/grobid) è un servizio Docker specializzato nel parsing di PDF accademici. A differenza di un semplice text extractor (come PyMuPDF), GROBID comprende la **struttura** del paper:

- **Header:** titolo, autori, abstract, affiliazioni
- **Corpo:** sezioni numerate con titoli (`Introduction`, `Methodology`, ecc.)
- **Riferimenti bibliografici:** citazioni parsate con autore, anno, venue
- **Didascalie delle figure:** testo associato a figure e tabelle

GROBID restituisce un XML TEI strutturato, che il sistema parsa in un formato interno con metadati separati per ogni sezione del paper.

**Fallback:** Se GROBID non è disponibile, il sistema può usare PyMuPDF come fallback (configurabile), che estrae testo grezzo senza struttura — funziona, ma la qualità del chunking è inferiore.

### 1.3 Arricchimento con Semantic Scholar

Dopo l'estrazione GROBID, ogni paper viene arricchito con metadati da [Semantic Scholar](https://www.semanticscholar.org/):

- **Conteggio citazioni** — quante volte il paper è stato citato
- **Venue e journal** — dove è stato pubblicato
- **Anno di pubblicazione** — data precisa
- **DOI e ID esterni** — identificatori univoci
- **Abstract** — spesso più pulito di quello estratto dal PDF

La ricerca avviene per titolo (matching fuzzy), con cache locale per evitare chiamate ripetute alla stessa API.

### 1.4 Generazione BibTeX Automatica

Per ogni paper viene generata automaticamente una entry BibTeX con chiave deterministica nel formato `Autore_ANNO_Keyword`:

```bibtex
@article{Zou_2023_GCG,
    title = {Universal and Transferable Adversarial Attacks on Aligned Language Models},
    author = {Zou, Andy and Wang, Zifan and ...},
    year = {2023},
    ...
}
```

Questa chiave BibTeX è il **filo conduttore** che lega ogni informazione nel sistema: chunk, summary L1/L1.5/L2, citazioni nel survey finale, entry nella bibliografia — tutto punta alla stessa chiave.

### 1.5 Chunking GROBID-Aware

Il testo estratto viene diviso in chunk di ~1.500 caratteri con overlap di ~200 caratteri. La peculiarità è che il chunking è **GROBID-aware**: i confini dei chunk rispettano i confini delle sezioni del paper.

Questo significa che un chunk non "spezza" mai una sezione a metà — la divisione avviene preferenzialmente tra sezioni. Il risultato è che ogni chunk ha un contesto semantico più coerente, il che migliora sia la qualità degli embedding che la rilevanza del retrieval.

Ogni chunk conserva metadati: categoria, bibtex_key, titolo del paper, anno, tipo di chunk, posizione nel paper.

### 1.6 Pipeline di Elaborazione Immagini (3 Stadi)

Le figure scientifiche (grafici, diagrammi, tabelle) contengono informazioni importanti che il puro testo non cattura. Ma un PDF contiene anche moltissime immagini irrilevanti: loghi, icone, elementi decorativi, screenshot di copyright.

La soluzione è una **pipeline di filtraggio a 3 stadi** che invia a Gemini Vision solo le figure realmente scientifiche:

**Stadio 1 — Filtro Euristico (costo zero):**
Ogni immagine estratta con PyMuPDF viene classificata localmente usando 5 euristiche:
- **Aspect ratio** — banner e barre hanno rapporti estremi; i grafici tipicamente tra 0.5 e 2.5
- **Entropia del colore** (Shannon) — loghi monocromo hanno bassa entropia; grafici complessi hanno alta entropia
- **Densità dei bordi** (Laplaciano) — grafici e tabelle hanno molte linee; foto e loghi hanno bordi meno strutturati
- **Posizione nel documento** — prima pagina = probabilmente logo; ultima pagina = probabilmente copyright
- **Dimensione relativa alla pagina** — troppo piccola = icona; troppo grande = sfondo decorativo

Punteggio combinato pesato → classificazione: `SCIENTIFIC_FIGURE`, `LIKELY_DECORATIVE`, o `UNCERTAIN`.

**Stadio 2 — Filtro Embedding Multimodale (costo basso):**
Le immagini `UNCERTAIN` e le `SCIENTIFIC_FIGURE` con score borderline vengono confrontate con 5 embedding di riferimento pre-calcolati (plot scientifico, tabella comparativa, diagramma architetturale, flowchart, matrice di confusione). La similarità coseno nello spazio multimodale di Gemini Embedding 2 Preview determina se l'immagine è rilevante.

**Stadio 3 — Descrizione Vision (costo pieno):**
Solo le immagini che superano entrambi gli stadi vengono inviate a Gemini Vision, che genera una descrizione strutturata: tipo di figura, contenuto, key findings, rilevanza per il survey.

**Risultato:** Riduzione del 70-80% delle chiamate Vision API, con perdita trascurabile di figure scientifiche.

Ogni figura rilevante produce **due entry** nel database vettoriale:
1. `figure_descriptions` — embedding testuale della descrizione generata
2. `image_embeddings` — embedding multimodale dell'immagine stessa

Questo abilita il **cross-modal retrieval**: una query testuale come "confronto tassi di successo degli attacchi" può trovare figure semanticamente correlate anche senza corrispondenze testuali esatte.

### 1.7 Indicizzazione Vettoriale (Qdrant)

Ogni chunk testuale viene vettorializzato con Gemini Embedding 2 Preview (3072 dimensioni) e memorizzato nella collezione `paper_chunks` di Qdrant (database vettoriale locale, senza server esterno).

L'indicizzazione è idempotente: gli UUID sono deterministici (UUID v5 basato sul contenuto), quindi ri-inserire lo stesso chunk non crea duplicati.

---

## Fase 2 — Summarization (`paperhunter summarize`)

Questa fase costruisce la piramide di compressione L1 → L1.5 → L2 descritta sopra.

### 2.1 Generazione L1 — Un Summary per Paper

Per ogni paper nella knowledge base, il sistema:

1. Recupera tutti i chunk del paper da Qdrant
2. Li assembla in un prompt strutturato per Gemini Pro
3. Richiede un JSON con 6 campi obbligatori: contributi chiave, metodologia, testing methodology, contesto temporale, risultati quantitativi, limitazioni
4. Valida l'output e lo memorizza nella collezione `l1_summaries` di Qdrant

Il campo **testing_methodology** è particolarmente importante perché il survey finale include una sezione dedicata alle metodologie di testing, e senza dati strutturati su benchmark e metriche usate da ogni paper, quella sezione risulterebbe vuota.

**Compressione adattiva:** La lunghezza target dell'L1 si adatta al numero totale di paper. Con 30 paper, ogni L1 può essere più dettagliato; con 1.000 paper, ogni L1 deve essere più compatto per non esplodere il budget token nei livelli successivi.

### 2.2 Clustering Semantico (RAPTOR)

Prima di generare gli L1.5, il sistema raggruppa i summary L1 per affinità tematica:

1. Recupera gli embedding degli L1 da Qdrant (vettori 3072-dim)
2. Applica **AgglomerativeClustering** (scikit-learn) con distanza coseno
3. La soglia di similarità è configurabile (default: 0.75)
4. I cluster con più di 20 paper vengono **ri-suddivisi ricorsivamente** per mantenere la coerenza tematica

Risultato tipico: ~1.000 paper → ~50 cluster, dove ciascun cluster raggruppa paper che trattano lo stesso sotto-tema (es. "difese basate su detection di prompt anomali", "attacchi jailbreak basati su suffix optimization", ecc.).

### 2.3 Generazione L1.5 — Un Summary per Cluster

Per ogni cluster, il sistema:

1. Raccoglie i testi degli L1 summary dei paper nel cluster
2. Genera un meta-summary con Gemini Pro che cattura:
   - **Tema unificante** del cluster
   - **Metodologie condivise e divergenti** — cosa hanno in comune e dove divergono
   - **Risultati chiave aggregati** — pattern che emergono confrontando i paper
   - **Panoramica testing** — benchmark e metriche usate trasversalmente
   - **Arco temporale** — dal paper più vecchio al più recente del cluster
3. Memorizza nella collezione `l1_5_summaries` di Qdrant

La forza dell'L1.5 è che **identifica pattern cross-paper** che non emergerebbero leggendo i singoli L1 separatamente. Ad esempio, potrebbe notare che "3 paper su 5 in questo cluster usano il benchmark AdvBench, ma con metriche diverse" — un'osservazione impossibile a livello di singolo paper.

### 2.4 Generazione L2 — Un Meta-Summary per Categoria

Per ogni categoria (es. "Prompt Injection"), il sistema:

1. Verifica se esistono L1.5 per quella categoria
2. **Se sì (percorso RAPTOR):** assembla tutti gli L1.5 in un prompt e chiede a Gemini Pro una sintesi panoramica. Poiché ~12 cluster summary stanno comodamente nel budget token, il modello può fare pattern recognition cross-cluster.
3. **Se no (percorso legacy):** assembla gli L1 con compressione basata sulla recency:
   - Paper ≤2 anni: testo L1 completo
   - Paper 2-4 anni: ~70% del dettaglio
   - Paper >4 anni: solo contributi chiave e testing

L'output è un JSON con 7 campi che forniscono una visione completa della categoria: trend temporali, tassonomia, gap colmati e aperti, convergenze metodologiche, panoramica testing e sintesi narrativa.

### 2.5 Auto-Invalidazione

Quando vengono ingeriti nuovi paper in una categoria, il sistema rileva automaticamente che l'L2 esistente è **stale** (basato su dati parziali). Se il numero di nuovi paper supera una soglia configurabile (default: 1), l'L2 viene automaticamente rigenerato — inclusa la ri-generazione degli L1.5 se necessario.

---

## Fase 3 — Indexing (`paperhunter index`)

L'indexing costruisce gli strumenti di ricerca che il Writer Agent userà per trovare le informazioni necessarie durante la scrittura. Non si tratta solo di ricerca vettoriale — il sistema combina **tre approcci complementari**.

### 3.1 Indice BM25 (Keyword Search)

BM25 (Best Matching 25, variante di Okapi BM25) è un algoritmo classico di information retrieval basato su keyword matching. A differenza della ricerca vettoriale (che cattura la similarità semantica), BM25 eccelle nel trovare **corrispondenze esatte** di termini specifici.

Il sistema costruisce un indice BM25 su tutti i chunk, gli L1 e gli L2, con tokenizzazione specializzata per testi accademici (rimozione stopwords estese, stemming).

**Perché serve anche BM25 se c'è la ricerca vettoriale?** Perché sono complementari:
- La ricerca vettoriale trova documenti *semanticamente* simili (es. "difese contro injection" trova anche "protezione da attacchi prompt")
- BM25 trova documenti con *esattamente* quei termini (es. "GCG attack" trova precisamente i paper che menzionano "GCG")
- Combinandoli si copre sia la ricerca per concetto che la ricerca per termine tecnico

### 3.2 Knowledge Graph (PropertyGraph + Entità)

Il sistema estrae **entità** e **relazioni** strutturate dagli L1 summary usando Gemini Flash:

**8 tipi di entità:**
- `Attack` — tecniche di attacco (es. GCG, AutoDAN, DeepInception)
- `Defense` — difese e mitigazioni (es. Perplexity Filter, SmoothLLM)
- `Model` — modelli LLM (es. GPT-4, LLaMA-2, Claude)
- `Dataset` — dataset usati (es. AdvBench, TruthfulQA)
- `Benchmark` — benchmark di valutazione
- `Metric` — metriche quantitative (es. Attack Success Rate, F1-Score)
- `Author` — autori dei paper
- `Venue` — conferenze e journal

**9 tipi di relazione:**
- `ATTACKS`, `DEFENDS_AGAINST`, `TESTED_ON`, `EVALUATED_WITH`, `UPDATES`, `INTRODUCED_IN`, `EXTENDS`, `AUTHORED_BY`, `PUBLISHED_IN`

Le entità vengono **normalizzate** (es. "GPT-4", "GPT4", "gpt-4" → "GPT-4") e **deduplicate** tramite hash deterministici. Entità menzionate in più paper vengono consolidate automaticamente.

Il grafo risultante viene memorizzato come **LlamaIndex PropertyGraph** (persistenza JSON locale) e opzionalmente sincronizzato con **Neo4j** per query più avanzate.

### 3.3 Community Detection (Louvain/Leiden)

Quando Neo4j è disponibile con il plugin Graph Data Science (GDS), il sistema applica algoritmi di **community detection** per scoprire macro-temi nel knowledge graph:

1. **Proiezione GDS** — crea una vista grafo ottimizzata per l'analisi
2. **Algoritmo Louvain o Leiden** — identifica cluster di nodi fortemente interconnessi (comunità)
3. **Summarization** — ogni comunità viene descritta da Gemini Pro in un testo narrativo di 200-500 parole
4. **Indicizzazione** — i community summary vengono memorizzati in Qdrant (collezione `graph_communities`) per il retrieval

**Esempio pratico:** Una comunità potrebbe raggruppare l'entità "GCG" (attacco), "SmoothLLM" (difesa), "AdvBench" (benchmark) e "Attack Success Rate" (metrica) — tutte legate da relazioni nel grafo. Il summary risultante spiegherebbe come GCG funziona, come SmoothLLM lo contrasta, e come AdvBench+ASR vengono usati per valutarli.

Questi community summary sono preziosi per il Writer perché offrono una **vista relazionale** che non emerge dalla semplice ricerca per similarità testuale.

### 3.4 Hybrid Retrieval con Reciprocal Rank Fusion

Il cuore del retrieval è un sistema ibrido che combina tutte le sorgenti tramite **Reciprocal Rank Fusion (RRF)**:

```
Query → ┌─ Ricerca Vettoriale (Qdrant)  →  top-20 candidati
        ├─ Ricerca BM25 (keyword)        →  top-20 candidati
        └─ [opzionale] Graph Search      →  candidati dal grafo
                    │
                    ▼
            RRF Fusion (pesi configurabili)
                    │
                    ▼
            top-10 risultati finali
```

**Formula RRF:** score(d) = Σᵢ wᵢ / (k + rankᵢ(d))

dove wᵢ è il peso della sorgente (default: vettoriale 0.6, BM25 0.3, grafo 0.1) e k è una costante di smoothing.

RRF ha il vantaggio di combinare ranking provenienti da sorgenti con scale di score diverse, senza bisogno di normalizzazione.

**Recency boost:** Dopo la fusione RRF, i paper più recenti ricevono un boost configurabile (default: +10% per il paper più recente).

### 3.5 Reranking con Gemini Flash

I risultati dell'HybridRetriever vengono ulteriormente riordinati usando **Gemini Flash come cross-encoder**:

1. Ogni candidato viene presentato insieme alla query originale a Gemini Flash
2. Il modello assegna un punteggio di rilevanza 0.0–1.0
3. I candidati vengono riordinati per score
4. Solo i top-K vengono restituiti al Writer

Questo step migliora significativamente la precisione perché un LLM come cross-encoder comprende le sfumature semantiche meglio di una semplice similarità coseno. Il reranking è opzionale ma raccomandato.

**Fallback:** Se Gemini Flash non è raggiungibile, il sistema mantiene l'ordine RRF originale senza degradazione critica.

---

## Fase 4 — Writing (`paperhunter write`)

La fase di scrittura è orchestrata come un **grafo LangGraph** con 4 agenti specializzati. Il flusso è:

```
START → Planner → Writer → Critic → ┬── sezioni OK → Assembler → END
                                     └── sezioni da rivedere → Writer (loop, max 3 iterazioni)
```

### 4.1 Planner Agent — Pianificazione dell'Outline

Il Planner è il primo agente a entrare in azione. Riceve gli **L2 summary** di tutte le categorie e genera un **outline strutturato** del survey:

- **Abstract** (generato successivamente dall'Assembler)
- **Introduction** — contesto, motivazione, contributi del survey
- **Background** — concetti fondamentali
- **Sezioni per categoria** — una o più sezioni per ogni categoria (es. "Prompt Injection Attacks", "Prompt Injection Defenses")
- **Testing Methodologies** — sezione obbligatoria e dedicata
- **Cross-Cutting Defenses** — difese trasversali a più categorie
- **Discussion** — analisi critica, limitazioni del campo
- **Conclusion** — sintesi e direzioni future

Per ogni sezione, il Planner specifica: titolo, descrizione, lunghezza target, topic chiave e tipo di sezione. L'outline può essere generato automaticamente o caricato da un file JSON esterno.

### 4.2 Writer Agent — Scrittura delle Sezioni

Il Writer prende l'outline e scrive ogni sezione del survey. Supporta **due modalità operative**:

**Modalità "static" (classica):**
1. Per ogni sezione, genera 3-5 query RAG dalla descrizione
2. Esegue il retrieval ibrido (vettoriale + BM25 + reranking)
3. Opzionalmente interroga il PropertyGraph per contesto relazionale
4. Assembla il contesto recuperato e lo passa a Gemini Pro con istruzioni di scrittura accademica
5. Il risultato è un testo Markdown con citazioni `[bibtex_key]`

**Modalità "agentic" (ReAct, preferita per corpora grandi):**
1. Il Writer viene istanziato come **ReAct Agent** con 4 tool di retrieval a disposizione:
   - `search_vector_database` — ricerca semantica su chunk, L1 e L2
   - `search_keyword_index` — ricerca BM25 per termini specifici
   - `search_graph_communities` — ricerca nei community summary del grafo
   - `search_figure_descriptions` — ricerca figure (testo + embedding multimodale)
2. L'agente segue un loop **Thought → Action → Observation → ... → Final Answer**:
   - *Thought:* "Per scrivere di GCG attack, devo trovare il paper originale e le difese proposte"
   - *Action:* `search_vector_database("GCG universal adversarial suffix attack")`
   - *Observation:* [risultati del retrieval]
   - *Thought:* "Ho i dettagli dell'attacco, ora cerco le difese..."
   - *Action:* `search_keyword_index("SmoothLLM defense GCG")`
   - ... (fino a max 7 iterazioni)
   - *Final Answer:* testo della sezione con citazioni

Il vantaggio della modalità agentica è che il Writer **decide autonomamente** quali informazioni cercare e da dove, adattandosi al contenuto specifico di ogni sezione. Con 1.000+ paper, questa flessibilità è essenziale perché query pre-calcolate non possono coprire tutte le sfumature.

**Istruzioni speciali:** Il Writer riceve template diversi a seconda del tipo di sezione. Per esempio, la sezione "Testing Methodologies" ha l'obbligo di includere una tabella comparativa di benchmark e metriche; l'Introduction deve contenere una roadmap del paper; la Discussion deve analizzare criticamente le limitazioni del campo.

### 4.3 Critic Agent — Verifica della Provenienza

Il Critic è il controllore di qualità. Per ogni sezione scritta:

1. **Estrae tutte le citazioni** `[bibtex_key]` dal testo (tramite regex)
2. **Per ogni citazione**, recupera il chunk sorgente originale da Qdrant
3. **Verifica** con Gemini Flash se il claim nel survey è effettivamente supportato dal testo sorgente
4. Assegna uno stato per citazione:
   - `SUPPORTED` — il claim è fedelmente riportato
   - `PARTIALLY_SUPPORTED` — il claim è parzialmente corretto
   - `NOT_SUPPORTED` — il claim non corrisponde alla fonte
   - `SOURCE_NOT_FOUND` — la chiave BibTeX non esiste nel corpus
5. Calcola il **pass rate** della sezione (% di citazioni supportate)
6. Se il pass rate è sotto la soglia (default: 80%) → la sezione viene marcata per revisione

Questo meccanismo è fondamentale per evitare **hallucination** nelle citazioni — un problema grave nei survey generati da LLM, dove il modello potrebbe attribuire affermazioni a paper che non le contengono.

### 4.4 Il Loop Writer ↔ Critic

Quando il Critic boccia una sezione:

1. Il feedback del Critic (quali citazioni sono errate e perché) viene passato al Writer
2. Il Writer riscrive la sezione tenendo conto del feedback
3. Il Critic ri-verifica la nuova versione
4. Il loop continua per **massimo 3 iterazioni**
5. Dopo la terza iterazione, la sezione viene accettata comunque con un warning

Questo loop iterativo migliora progressivamente la qualità delle citazioni, ma è anche la fase più costosa in termini di chiamate API.

### 4.5 Assembler Agent — Assemblaggio Finale

L'Assembler è l'ultimo agente e si occupa di:

1. **Ordinare le sezioni** secondo l'outline del Planner
2. **Generare l'Abstract** — sintetizzando tutte le sezioni in un paragrafo di 150-250 parole
3. **Generare/raffinare l'Introduction** — contestualizzando il survey, motivandone l'esistenza, e includendo una roadmap
4. **Generare la Conclusion** — sintesi dei findings, contributi, direzioni future e raccomandazioni
5. **Assemblare il documento finale** — un unico Markdown coeso con tutte le sezioni
6. **Generare la bibliografia filtrata** — file `.bib` contenente solo le entry effettivamente citate
7. **Versioning** — salva il survey con timestamp, genera changelog se esiste una versione precedente

### 4.6 Checkpoint e Ripresa da Errore

La pipeline salva un **checkpoint JSON** dopo ogni nodo del grafo LangGraph (plan, write, verify, assemble). Questo significa che:

- Se la pipeline si interrompe durante la verifica (es. timeout API), puoi riprendere con `paperhunter write --resume`
- Il sistema carica l'ultimo checkpoint valido e riprende dalla fase successiva
- Le sezioni già scritte e verificate non vengono rielaborate

I checkpoint contengono lo stato completo: outline, bozze delle sezioni, risultati delle verifiche, provenance, versione corrente.

---

## Post-Processing e Output

### Provenance Tracking

Per ogni claim del survey, il sistema genera un report che mappa **claim → fonte originale → stato di verifica**. Include statistiche aggregate (coverage rate, pass rate medio), dettaglio per sezione e per citazione, e segnalazione dei paper più citati.

Formati di export: JSON (per analisi programmatica) e Markdown (per lettura umana).

### Bibliography Filtrata

Il sistema mantiene un **master .bib** con tutte le entry BibTeX generate durante l'ingestion. Al termine della scrittura, genera un **references_final.bib** filtrato che contiene solo le entry effettivamente citate nel survey.

Include un report statistico: distribuzione per anno, tipo di pubblicazione, venue, e segnalazione di chiavi BibTeX citate nel testo ma assenti nel master .bib (potenziali hallucination).

### Export LaTeX

Il survey Markdown viene convertito in LaTeX compilabile con 3 template disponibili:

- **ACL** — formato ACL Anthology (natbib) — standard per NLP/AI
- **IEEEtran** — formato IEEE Transactions — standard per sicurezza informatica
- **plain** — article LaTeX standard

La conversione gestisce: heading gerarchici, bold/italic, citazioni `[key]` → `\cite{key}`, tabelle Markdown → `tabular`, elenchi, URL.

### Versioning Automatico

Ogni esecuzione della pipeline produce una versione numerata in `output/versions/Survey_vX.Y_MonYYYY/` contenente: survey completo (Markdown + LaTeX), bibliografia filtrata, metadati della versione, report provenienza, changelog rispetto alla versione precedente, e lista chiavi citate.

Il changelog tra versioni viene generato automaticamente e include: sezioni aggiunte/rimosse/modificate, nuovi paper incorporati, variazioni nel conteggio citazioni.

---

## Architettura Complessiva — Schema di Flusso

```
scaricati/                                        PDF categorizzati (~1.000+ paper)
    ├── Prompt_injections1/
    ├── Hallucinations1/
    ├── Data_and_Model_Poisoning1/
    └── Adversarial_Code1/
            │
            ▼
┌──── FASE 1: INGEST ────────────────────────────────────────────────────────┐
│  PDF → GROBID (TEI-XML) → Semantic Scholar → BibTeX → Chunking            │
│                                                    │                       │
│  PDF → PyMuPDF (immagini) → Heuristic → Embedding → Vision Description   │
│                                                    │                       │
│  Chunk + Figure Description + Image Embedding → Qdrant (7 collezioni)     │
└────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──── FASE 2: SUMMARIZE ─────────────────────────────────────────────────────┐
│  Chunk → Gemini Pro → L1 Summary (per paper) → Qdrant: l1_summaries       │
│                           │                                                │
│  L1 embedding → AgglomerativeClustering → ~50 cluster                     │
│                           │                                                │
│  Cluster → Gemini Pro → L1.5 Summary (per cluster) → Qdrant: l1_5        │
│                           │                                                │
│  L1.5 → Gemini Pro → L2 Meta-Summary (per categoria) → Qdrant: l2        │
└────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──── FASE 3: INDEX ─────────────────────────────────────────────────────────┐
│  Chunk → BM25 Index (keyword search locale)                                │
│                                                                            │
│  L1 → Entity Extractor (Gemini Flash) → Entità + Relazioni               │
│       → PropertyGraph (JSON locale) + Neo4j (opzionale)                   │
│       → Community Detection (Louvain/Leiden)                               │
│       → Community Summary → Qdrant: graph_communities                     │
│                                                                            │
│  Hybrid Retriever = Vector Search + BM25 + Graph → RRF → Reranking       │
└────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──── FASE 4: WRITE ─────────────────────────────────────────────────────────┐
│  L2 Summary → Planner Agent → Outline strutturato                         │
│                     │                                                      │
│  Outline → Writer Agent (ReAct) → Bozza sezioni con [citazioni]          │
│                 │  ↑      │                                                │
│                 │  │      ├─ search_vector_database                       │
│                 │  │      ├─ search_keyword_index                         │
│                 │  │      ├─ search_graph_communities                     │
│                 │  │      └─ search_figure_descriptions                   │
│                 │  │                                                        │
│                 ▼  │                                                        │
│  Bozza → Critic Agent → Verifica citazioni → feedback ──┘                 │
│                 │            (loop max 3×)                                  │
│                 ▼                                                           │
│  Sezioni verificate → Assembler Agent → Survey finale                     │
│       → Abstract, Introduction, Conclusion generati                        │
│       → Bibliografia filtrata (.bib)                                       │
│       → Provenance report (JSON + MD)                                      │
│       → Versioning (vX.Y con changelog)                                    │
│       → Export LaTeX (ACL/IEEEtran/plain)                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Le 7 Collezioni Qdrant

Il database vettoriale Qdrant opera in modalità locale (nessun server) e contiene 7 collezioni separate, ciascuna con uno scopo preciso:

| Collezione | Contenuto | Usata da |
|---|---|---|
| `paper_chunks` | Chunk testuali originali dei PDF (~1.5k char, embedding 3072-dim) | Writer (RAG), Critic (verifica) |
| `l1_summaries` | Summary L1 individuali per paper | Clustering (L1.5), Writer (RAG) |
| `l1_5_summaries` | Summary L1.5 per cluster semantico | L2 generation, Writer (RAG) |
| `l2_summaries` | Meta-summary L2 per categoria | Planner, Writer (contesto d'insieme) |
| `graph_communities` | Narrative summary per community del grafo | Writer (contesto relazionale) |
| `figure_descriptions` | Descrizioni testuali delle figure scientifiche | Writer (riferimenti a figure) |
| `image_embeddings` | Embedding multimodali delle immagini (cross-modal) | Writer (retrieval cross-modale) |

Tutti gli embedding sono a 3072 dimensioni (Gemini Embedding 2 Preview), con distanza coseno. Gli UUID sono deterministici (UUID v5) per idempotenza.

---

## Modelli LLM e Strategia di Utilizzo

PaperHunter usa due tier di modelli Gemini:

| Tier | Modello | Quando viene usato | Perché |
|---|---|---|---|
| **Flash** | Gemini 3 Flash Preview | Classificazione, reranking, entity extraction, verifica citazioni (Critic) | Veloce, economico, sufficiente per task deterministici |
| **Pro** | Gemini 3.1 Pro Preview | Summarization (L1, L1.5, L2), scrittura (Writer), planning (Planner), assemblaggio, community summary | Qualità superiore necessaria per generazione di testo accademico |

**Strategia di retry e fallback:**
- Retry con backoff esponenziale (fino a 6 retry per Pro, con delay fino a 120s)
- **Strategia "autonomous"**: retry infiniti con delay incrementali — il sistema procede da solo senza intervento umano
- **Strategia "interactive"**: dopo N retry falliti, chiede all'utente se usare OpenRouter come fallback
- **Fallback OpenRouter**: se abilitato, quando Gemini non è raggiungibile il sistema instrada le richieste tramite OpenRouter (compatibile OpenAI), che funge da proxy per gli stessi modelli Gemini

**Embedding:** Gemini Embedding 2 Preview produce vettori a 3072 dimensioni, sia per testo che per immagini (nello stesso spazio vettoriale).

---

## Budget-Awareness e Scalabilità

Il sistema è progettato per gestire da 20 a 1.000+ paper senza degradazione della qualità, grazie a meccanismi di budget-awareness automatici:

| Componente | Meccanismo di Budget |
|---|---|
| **L1 Summary** | Compressione adattiva: standard (≤30 paper), conciso (31-70), compresso (>70) |
| **L1.5 Summary** | Clustering riduce ~1.000 L1 a ~50 cluster — compressione 20× |
| **L2 Summary** | Percorso RAPTOR: ~50 input vs ~1.000 — fits in context; legacy: recency-tier + downgrade progressivo |
| **Writer** | Budget per sezione: `budget_totale / N_sezioni` (proporzionale) o fisso |
| **Pipeline** | Report di budget stampato prima dell'esecuzione con warning se over budget |

Il budget token massimo per singola chiamata LLM è configurabile (default: 150k token). Il sistema calcola automaticamente quanti token richiederà ogni fase e segnala se il budget è insufficiente.

---

## Recency Priority

Quando abilitata (default: sì), la recency priority dà maggior peso ai paper più recenti in tutta la pipeline:

| Componente | Comportamento Recency |
|---|---|
| **Ingestion** | PDF processati in ordine di anno decrescente — i più recenti vengono elaborati per primi |
| **L2 Summarization** | Paper ≤2 anni → dettaglio pieno; paper 2-4 anni → ~70%; paper >4 anni → solo contributi chiave |
| **Hybrid Retrieval** | Boost post-RRF: score × (1 + boost × recency_factor) — i paper recenti scalano più in alto a parità di rilevanza |

Il toggle globale `recency_priority: true/false` in `config.yaml` abilita/disabilita tutta la logica di recency in un solo punto.

---

## Tech Stack

| Componente | Tecnologia | Note |
|---|---|---|
| LLM | **Gemini 3 Flash/Pro** (via `google-genai` SDK) | Flash per task semplici, Pro per scrittura/planning |
| Embedding | **Gemini Embedding 2 Preview** (3072-dim) | Testo e immagini nello stesso spazio vettoriale |
| Framework RAG | **LlamaIndex Core** | Orchestration, PropertyGraph, ReAct Agent |
| Vector DB | **Qdrant** (locale, senza server) | 7 collezioni |
| Keyword Search | **BM25** (`rank-bm25`) | Retrieval ibrido con fusione RRF |
| Clustering | **scikit-learn** + **NumPy** | AgglomerativeClustering per RAPTOR L1.5 |
| Image Processing | **Pillow** + **NumPy** + **Gemini Vision** | Pipeline 3 stadi |
| PDF Extraction | **GROBID 0.8.2** (Docker) + **PyMuPDF** | TEI-XML → sezioni strutturate |
| Metadati | **Semantic Scholar API** (free) | BibTeX, venue, citation count, DOI |
| Multi-Agent | **LangGraph** + **ReAct** | Planner → Writer → Critic → Assembler |
| Knowledge Graph | **Neo4j Community + GDS** | PropertyGraph + Louvain/Leiden |
| Output | **LaTeX** (ACL/IEEEtran) | + .bib, provenance JSON, versioning |

---

## Setup

### Prerequisiti

- **Python 3.10+** (testato fino a 3.14)
- **Docker** (per GROBID; opzionale ma fortemente raccomandato)
- **Google Gemini API key** (free tier con Gemini Advanced)

### Installazione

```bash
# Clona il repository
git clone https://github.com/rick0101/llm_university_project.git
cd llm_university_project

# Crea virtual environment
python3 -m venv venv
source venv/bin/activate    # Linux/macOS

# Installa dipendenze + pacchetto
pip install --upgrade pip
pip install -e .

# Installa browser Playwright (per download paper)
playwright install chromium

# (Opzionale) Dipendenze GraphRAG
pip install -e ".[graphrag]"
```

### API Keys

Crea un file `.env` nella root del progetto:

```env
# Obbligatorio — Gemini API key (LLM + embedding)
GOOGLE_API_KEY=your_gemini_api_key_here

# Opzionale — rate limit più alti per Semantic Scholar
SEMANTIC_SCHOLAR_API_KEY=your_s2_key_here
```

### Docker Services

**GROBID** (richiesto per estrazione strutturata PDF):

```bash
docker run --rm -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.2-full
# Attendi ~30 secondi, poi verifica:
curl -s http://localhost:8070/api/isalive   # Deve restituire: true
```

**Neo4j** (opzionale — per GraphRAG + Community Detection):

```bash
# Con plugin GDS (necessario per community detection)
docker run --rm -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_PLUGINS='["graph-data-science"]' \
    neo4j:2026.02.2-community
```

---

## Quick Start

```bash
# 1. (Opzionale) Cerca e scarica nuovi paper
paperhunter search -k '"jailbreak prompt" LLM' \
    -c "Paper su prompt injection" \
    -o ./scaricati/Prompt_injections1 -n 20

# 2. Ingest: PDF → GROBID → S2 → chunking → Qdrant
paperhunter ingest

# 3. Summarize: L1 → L1.5 → L2
paperhunter summarize

# 4. Index: BM25 + PropertyGraph + Community Detection
paperhunter index

# 5. Write: Planner → Writer → Critic → Assembler
paperhunter write

# 6. Export LaTeX
paperhunter export --template acl

# Statistiche sistema
paperhunter stats

# Query interattiva
paperhunter query "What defenses exist against prompt injection attacks?"
```

---

## CLI Reference

| Comando | Descrizione |
|---|---|
| `paperhunter search` | Cerca e scarica paper da ArXiv, Semantic Scholar, HuggingFace Papers |
| `paperhunter ingest` | Ingestion incrementale: GROBID → S2 → BibTeX → chunking → Qdrant |
| `paperhunter summarize` | Genera summary gerarchici L1 → L1.5 → L2 |
| `paperhunter index` | Costruisce BM25 + PropertyGraph + Community Detection |
| `paperhunter write` | Pipeline multi-agente: Planner → Writer → Critic → Assembler |
| `paperhunter export` | Export LaTeX/BibTeX dal survey generato |
| `paperhunter query` | Query interattiva sulla knowledge base |
| `paperhunter stats` | Statistiche complete del sistema |
| `paperhunter cleanup` | Elimina paper per intervallo di date |

Opzioni globali: `--config PATH`, `-v/--verbose` (DEBUG logging), `--version`.

**Opzioni notevoli per comando:**

- `paperhunter ingest --skip-s2` — salta Semantic Scholar (utile offline)
- `paperhunter summarize --l1` / `--l2` — genera solo un livello
- `paperhunter summarize --invalidate` — controlla e rigenera L2 stale
- `paperhunter index --graph` / `--bm25` — costruisce solo un componente
- `paperhunter write --resume` — riprende dall'ultimo checkpoint
- `paperhunter write --outline sample_outline.json` — usa outline custom
- `paperhunter export --template acl|ieeetran|plain` — scelta template LaTeX

---

## Stima Chiamate API

Stima per un run tipico con **~120 paper**, **4 categorie**, **~11 sezioni**, **modalità agentica**:

| Fase | Chiamate LLM | Chiamate Embedding | Totale |
|---|---|---|---|
| **Summarize** | ~130-138 (Pro) | ~14 | ~144-152 |
| **Index** | ~128-148 (Flash + Pro) | ~1-3 | ~129-151 |
| **Write** | ~175-280 (Pro + Flash) | ~35-55 | ~210-335 |
| **TOTALE** | **~433-566** | **~50-72** | **~483-638** |

La fase **write** è la più costosa per via del loop ReAct (ogni iterazione = 1 chiamata Pro) e del ciclo Writer↔Critic. La fase **summarize** è la più prevedibile (~1:1 con il numero di paper).

---

## Struttura del Progetto

```
paperhunter/
├── main.py                           ← CLI con 8+ sottocomandi
├── config.py                         ← Loader YAML + pydantic-settings
├── models.py                         ← Modelli Pydantic v2
│
├── ingestion/                        ← Pipeline di elaborazione PDF
│   ├── ingestion_pipeline.py         ← Orchestratore end-to-end
│   ├── grobid_client.py              ← Client GROBID Docker (TEI-XML → strutturato)
│   ├── pdf_hasher.py                 ← Hash SHA-256 per ingestion incrementale
│   ├── semantic_scholar.py           ← Arricchimento S2 API
│   ├── bibtex_generator.py           ← Generazione BibTeX automatica
│   ├── chunker.py                    ← Chunking GROBID-aware
│   ├── figure_extractor.py           ← Estrazione figure + pipeline 3 stadi
│   └── image_relevance_filter.py     ← Classificatore euristico immagini
│
├── summarization/                    ← Summarization gerarchica RAPTOR
│   ├── level1_summarizer.py          ← L1: summary per paper
│   ├── clustering_utils.py           ← Clustering semantico (Agglomerative)
│   ├── level1_5_summarizer.py        ← L1.5: meta-summary per cluster
│   ├── level2_summarizer.py          ← L2: meta-summary per categoria
│   └── invalidation_manager.py       ← Auto-invalidazione L2
│
├── indexing/                         ← Retrieval ibrido
│   ├── qdrant_store.py               ← Vector store Qdrant (7 collezioni)
│   ├── bm25_index.py                 ← Indice BM25 keyword
│   ├── hybrid_retriever.py           ← Vector + BM25 + RRF fusion
│   └── reranker.py                   ← Reranking cross-encoder Gemini Flash
│
├── graphrag/                         ← Knowledge Graph + Community
│   ├── entity_extractor.py           ← Estrazione entità/relazioni da L1
│   ├── property_graph.py             ← PropertyGraph LlamaIndex
│   ├── neo4j_manager.py              ← Connettore Neo4j + proiezione GDS
│   ├── community_detector.py         ← Louvain/Leiden community detection
│   └── community_summarizer.py       ← Summary narrativi per community
│
├── agents/                           ← Pipeline multi-agente LangGraph
│   ├── state.py                      ← Stato condiviso tra agenti
│   ├── pipeline.py                   ← Grafo LangGraph (orchestrazione)
│   ├── planner_agent.py              ← Agent 1: genera outline
│   ├── writer_agent.py               ← Agent 2: scrive sezioni (static/agentic)
│   ├── critic_agent.py               ← Agent 3: verifica citazioni
│   ├── assembler_agent.py            ← Agent 4: assembla survey finale
│   └── tools/
│       └── retrieval_tools.py        ← 4 tool di retrieval per ReAct agent
│
├── postprocessing/                   ← Generazione output
│   ├── provenance_tracker.py         ← Tracciamento provenienza claim
│   ├── bibliography_generator.py     ← Bibliografia .bib filtrata
│   ├── latex_exporter.py             ← Markdown → LaTeX (ACL/IEEE/plain)
│   ├── versioning.py                 ← Versioning + changelog automatico
│   └── checkpoint_manager.py         ← Checkpoint per ripresa da errore
│
└── utils/                            ← Utility condivise
    ├── llm_factory.py                ← Factory LLM (Flash/Pro + fallback)
    ├── multimodal_embedder.py        ← Embedding multimodali Gemini
    ├── rate_limiter.py               ← Rate limiting token bucket
    ├── token_counter.py              ← Conteggio token e budget
    └── logging_setup.py              ← Rich logging + progress bar
```

---

## Testing

Il progetto include 42 moduli di test con ~1200+ test che coprono tutti i moduli. I test usano mock e non richiedono API live o servizi Docker.

```bash
# Tutti i test
python -m pytest test_*.py -v

# Test specifici per modulo
python -m pytest test_pipeline.py -v
python -m pytest test_clustering_utils.py test_l1_5_summarizer.py test_l2_with_l1_5.py -v

# Solo test image processing (v3.1)
python -m pytest test_image_relevance_filter.py test_figure_extractor_filtering.py test_multimodal_embedder.py -v
```

---

## Troubleshooting

| Problema | Soluzione |
|---|---|
| **GROBID non risponde** | `docker stop grobid && docker run --rm -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.2-full` — attendi ~30s |
| **Rate limit Gemini (429)** | Il sistema retria automaticamente con backoff. Riduci `rate_limiting.gemini_rpm` o attendi. |
| **Qdrant corrotto** | `rm -rf qdrant_db/` e riesegui `paperhunter ingest` |
| **Neo4j connection refused** | Verifica `docker ps \| grep neo4j` e `neo4j.enabled: true` in config.yaml |
| **Pipeline interrotta** | `paperhunter write --resume` — riprende dall'ultimo checkpoint |
| **Citazioni mancanti nel .bib** | Controlla il provenance report per chiavi hallucinate; verifica il formato `Author_YYYY_Keyword` |
| **`pip install -e .` fallisce** | `pip install --upgrade pip setuptools wheel && pip install -e .` |

---

## License

MIT License
