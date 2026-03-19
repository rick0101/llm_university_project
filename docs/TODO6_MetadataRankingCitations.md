# TODO6_MetadataRankingCitations.md — Metadati Robusti, Figure nel Writer, BibTeX Completo, Venue Ranking

> **Riferimento:** `instructions7.txt`  
> **Prerequisito:** Tutti i task di `TODO.md`, `TODO3_Scalability.md`, `TODO4_ImageProcessing.md` e `TODO5_migliorie.md` completati  
> **Data inizio:** [Data da definire]  
> **Ultimo aggiornamento:** Marzo 2026  
> **Target:** (1) Estrazione robusta di date e metadati dai PDF con strategia a cascata; (2) Writer Agent può selezionare e inserire figure specifiche nel survey; (3) BibTeX garantito e completo per ogni paper; (4) Ranking del retrieval pesato per autorevolezza della venue

---

## 🏗️ Architettura Estesa (Nuovi/Modificati)

```
config.yaml                            ← Nuova sezione: retrieval.venue_ranking

paperhunter/
├── ingestion/
│   ├── metadata_enricher.py           ← NUOVO: pipeline a cascata per date + venue + abstract
│   ├── bibtex_generator.py            ← MODIFICATO: ensure_complete_bibtex() + fallback robusti
│   ├── grobid_client.py               ← MODIFICATO: estrazione date più aggressiva dal TEI-XML
│   ├── semantic_scholar.py            ← MODIFICATO: parsing publicationDate → year + month
│   └── ingestion_pipeline.py         ← MODIFICATO: chiama MetadataEnricher dopo S2 enrichment
│
├── indexing/
│   ├── hybrid_retriever.py            ← MODIFICATO: venue_boost post-RRF
│   └── venue_ranker.py                ← NUOVO: VenueRanker con tier dictionary + boost
│
├── agents/
│   ├── tools/
│   │   └── retrieval_tools.py         ← MODIFICATO: nuovo tool get_figure_for_section()
│   └── writer_agent.py                ← MODIFICATO: prompt con istruzione per tag {{FIGURE:...}}
│
├── postprocessing/
│   └── figure_resolver.py             ← NUOVO: FigureResolver, risolve tag {{FIGURE:...}} → Markdown/LaTeX
│
├── models.py                          ← MODIFICATO: metadata_confidence, publication_source
└── config.py                          ← MODIFICATO: VenueRankingConfig in RetrievalConfig
```

---

## 📋 TASK DETTAGLIATI

---

### FASE 1 — Estrazione Robusta dei Metadati

#### [x] Task 1.1: Aggiornare `grobid_client.py` — estrazione date aggressiva dal TEI-XML

**File:** `paperhunter/ingestion/grobid_client.py` (modifica)  
**Stato attuale:** Il client GROBID estrae titolo, autori, abstract e sezioni, ma la data di pubblicazione viene estratta solo se il tag `<date>` è presente nell'header TEI. Molti paper accademici non hanno una data esplicita nel header GROBID (soprattutto i preprint arXiv).  
**Obiettivo:** Implementare una strategia multi-source per l'anno all'interno del parser GROBID, sfruttando tutti i punti del TEI-XML dove la data può comparire.  
**Dettagli:**
- Nel metodo `_parse_header()` (o equivalente), cercare la data in questo ordine all'interno del TEI-XML:
  1. `<date type="published">`, `<date type="submission">`, `<date>` nell'header `<biblStruct>`
  2. Tag `<idno type="DOI">` — da DOI si può ricavare l'anno (formato `10.XXXX/YYYY...`)
  3. Tag `<note>` o `<title level="j">` che contengono pattern anno (regex `\b(20[0-2]\d)\b`)
  4. Anno embedded nella lista degli `<author>` (affiliation date)
  5. References nell'elemento `<listBibl>`: anno mediano delle citazioni → stima upper bound anno paper
- Aggiungere metodo privato `_extract_year_from_tei(tei_root: etree._Element) -> tuple[int | None, str]`
  che restituisce `(year, source)` dove source è una stringa che indica da dove è stato estratto
  (es. `"tei_header_date"`, `"doi_year"`, `"text_regex"`, `"citation_median"`).
- I metodi non devono lanciare eccezioni: usare `try/except` con logging a livello DEBUG.
- Il `GROBIDOutput` deve essere aggiornato con due nuovi campi opzionali:
  `extracted_year: Optional[int]` e `year_extraction_source: Optional[str]`.

**Test:** Eseguire il parsing su 5 PDF campione (1 con data esplicita, 1 arXiv, 1 senza header completo, 1 conference paper, 1 workshop) e verificare che almeno 4/5 abbiano un anno estratto con source diversi.

> **✅ Completato — 19 marzo 2026**
> 
> **Modifiche apportate:**
> - `paperhunter/models.py` — aggiunti due campi opzionali a `GROBIDOutput`:
>   - `extracted_year: Optional[int]` — anno estratto con strategia multi-source
>   - `year_extraction_source: Optional[str]` — stringa descrittiva della fonte (es. `"tei_header_date_published"`, `"doi_year"`, `"text_regex_header"`, `"author_affiliation_date"`, `"citation_median"`, `"not_found"`)
> - `paperhunter/ingestion/grobid_client.py` — aggiornamenti:
>   - Aggiunto `import re` e `import statistics` a livello modulo
>   - Aggiunto metodo privato `_extract_year_from_tei(tei_root) -> tuple[int | None, str]` che implementa la cascata a 5 livelli:
>     1. Tag `<date type="published/submission/generic">` nell'header TEI
>     2. `<idno type="DOI">` — regex `r"/(\d{4})[./]"` sul valore
>     3. `<note>` e `<title level="j/m">` nell'header — regex `r"\b(20[0-2]\d)\b"`
>     4. `<date>` nelle affiliazioni degli autori
>     5. Anno mediano delle references nella `<listBibl>` (≥3 riferimenti con anno)
>   - `_parse_tei_xml()` ora chiama `_extract_year_from_tei()` e popola il `GROBIDOutput` con i due nuovi campi
>   - Tutti i try/except garantiscono che il metodo non lanci mai eccezioni (log a livello DEBUG)
>   - Due regex compilate come attributi di classe: `_YEAR_RE` e `_DOI_YEAR_RE`
> 
> **Come usare:**
> ```python
> result = grobid_client.process_pdf("paper.pdf")
> print(result.extracted_year)          # es. 2023
> print(result.year_extraction_source)  # es. "tei_header_date_published"
> ```

---

#### [x] Task 1.2: Aggiornare `semantic_scholar.py` — parsing robusto di `publicationDate`

**File:** `paperhunter/ingestion/semantic_scholar.py` (modifica)  
**Stato attuale:** Il campo `publicationDate` restituito da Semantic Scholar è una stringa ISO (es. `"2023-09-15"` oppure `"2023"` oppure `null`). Il parsing attuale è fragile e non distingue tra anno, anno+mese e data completa.  
**Obiettivo:** Parsing completo e robusto di `publicationDate`, con estrazione separata di anno, mese e giorno, e fallback al campo `year` se `publicationDate` è assente.  
**Dettagli:**
- Aggiungere funzione privata `_parse_publication_date(raw: str | None, year_fallback: int | None) -> tuple[int | None, int | None]`
  che restituisce `(year, month)`:
  - Gestisce formati: `"YYYY-MM-DD"`, `"YYYY-MM"`, `"YYYY"`, `None`
  - Fallback su `year_fallback` se `publicationDate` è None
  - Non lancia mai eccezioni: ritorna `(None, None)` in caso di errore
- Nel metodo `enrich_paper()`, dopo aver ricevuto la risposta S2, popolare:
  - `paper.year` con `year` dal parsing
  - `paper.month` con `month` dal parsing
  - `paper.published_date` con `datetime(year, month or 1, 1)` se `year` è disponibile
- Aggiungere anche parsing di `externalIds`:
  - Se presente `"ArXiv": "YYYY.NNNNN"` → estrarre anno dai primi 4 digit dell'ID
  - Se presente `"DOI"` → estrarre anno con regex `r'/(\d{4})\.'`
- Aggiornare `_enrich_from_s2_data()` per usare la nuova funzione di parsing.

**Test:** Mock di 5 risposte S2 con varianti di `publicationDate` (`null`, `"2023"`, `"2022-11"`, `"2021-05-17"`, `"2020-01-01"`) e verificare che `(year, month)` siano estratti correttamente.

> **✅ Completato — 19 marzo 2026**
>
> **Modifiche apportate:**
> - `paperhunter/ingestion/semantic_scholar.py`:
>   - Aggiunto metodo statico `_parse_publication_date(raw, year_fallback) -> tuple[int|None, int|None]`:
>     gestisce formati `"YYYY-MM-DD"`, `"YYYY-MM"`, `"YYYY"`, `None`; fallback su `year_fallback` intero;
>     non lancia mai eccezioni.
>   - Due regex di classe: `_ARXIV_YEAR_RE` e `_DOI_YEAR_RE` per estrarre anno da ArXiv ID e DOI.
>   - `_extract_enrichment_data()` aggiornato: usa `_parse_publication_date()` per popolare
>     `year`, `month`, `published_date`; fallback catena → ArXiv ID → DOI se ancora None.
>
> **Come usare:**
> ```python
> year, month = SemanticScholarEnricher._parse_publication_date("2023-09-15")
> # → (2023, 9)
> year, month = SemanticScholarEnricher._parse_publication_date(None, year_fallback=2022)
> # → (2022, None)
> ```

---

#### [ ] Task 1.3: Creare `metadata_enricher.py` — pipeline a cascata metadati

**File:** `paperhunter/ingestion/metadata_enricher.py` (nuovo)  
**Stato attuale:** Non esiste un orchestratore centralizzato per l'arricchimento metadati: anno, venue, abstract estratti da fonti diverse non vengono integrati sistematicamente.  
**Obiettivo:** Creare `MetadataEnricher` che coordina GROBID output + Semantic Scholar response + euristiche sul filename → produce un Paper completamente arricchito con campo `metadata_confidence`.  
**Dettagli:**
- Classe `MetadataEnricher` con metodo principale:
  ```python
  def enrich(
      self,
      paper: Paper,
      grobid_output: GROBIDOutput,
      s2_data: dict | None,
      pdf_filename: str = "",
  ) -> Paper:
  ```
- Strategia a cascata per **anno** (priority order):
  1. `s2_data["publicationDate"]` — fonte più affidabile
  2. `s2_data["year"]` — fallback S2
  3. `grobid_output.extracted_year` (nuovo campo da Task 1.1)
  4. Regex `_YEAR_RE = re.compile(r"\b(20[0-2]\d)\b")` sul testo dell'abstract GROBID
  5. Anno dal filename (già implementato in `ingestion_pipeline._extract_year_from_filename()`)
  6. Anno mediano dalle references GROBID (stima)
  7. `None` — metadato non disponibile
- Strategia a cascata per **venue**:
  1. `s2_data["venue"]` (stringa non vuota)
  2. `s2_data["journal"]["name"]` se presente
  3. `grobid_output` — parsing `<note type="report type">` o `<publisher>` nel TEI
  4. Normalizzazione con `_VENUE_ALIASES` dict (es. `"arxiv" → "arXiv"`, `"neurips" → "NeurIPS"`)
  5. `None`
- Strategia a cascata per **abstract**:
  1. `s2_data["abstract"]` (spesso più pulito)
  2. `grobid_output.abstract` (se non vuoto)
  3. Primo paragrafo del full_text (heuristic)
- Calcolo `metadata_confidence: float` (0.0-1.0):
  - Ogni campo trovato da fonte "primaria" (S2) contribuisce 0.15
  - Fonte "secondaria" (GROBID) contribuisce 0.08
  - Euristica contribuisce 0.04
  - Score finale = somma capped a 1.0
- Aggiungere campi `metadata_confidence: float = 0.0` e `year_source: Optional[str] = None`
  al modello `Paper` in `models.py`.

**Test:** Creare 3 fixture Paper + GROBIDOutput + s2_data (caso completo, caso solo S2, caso solo GROBID) e verificare che `metadata_confidence` sia > 0.7 nel caso completo e < 0.4 nel caso peggiore.

---

#### [ ] Task 1.4: Integrare `MetadataEnricher` nella `IngestionPipeline`

**File:** `paperhunter/ingestion/ingestion_pipeline.py` (modifica)  
**Stato attuale:** La pipeline chiama `SemanticScholarEnricher.enrich_paper()` ma i risultati non vengono integrati con l'output GROBID in modo sistematico: la data rimane None se S2 non risponde.  
**Obiettivo:** Inserire `MetadataEnricher.enrich()` come step dedicato dopo GROBID e S2, così che ogni Paper riceva i metadati migliori disponibili da tutte le fonti.  
**Dettagli:**
- Istanziare `MetadataEnricher` nel costruttore di `IngestionPipeline`.
- Dopo il blocco S2 enrichment in `_process_single_pdf()`, chiamare:
  ```python
  paper = self.metadata_enricher.enrich(
      paper=paper,
      grobid_output=grobid_result,
      s2_data=s2_raw_response,  # dict grezzo dalla cache S2, prima della conversione in Paper
      pdf_filename=pdf_path.name,
  )
  ```
- Aggiornare `SemanticScholarEnricher.enrich_paper()` per restituire anche il raw dict S2
  (o salvarlo come attributo pubblico dell'enricher dopo l'ultima chiamata).
- Loggare a livello INFO per ogni paper: `year={paper.year} (source={paper.year_source}), confidence={paper.metadata_confidence:.2f}`
- Non modificare il contratto esterno di `IngestionPipeline.run()`.

**Test:** Eseguire pipeline su 3 PDF reali, verificare nei log che `year_source` sia sempre valorizzato (mai None) e che `metadata_confidence` sia loggato correttamente.

---

### FASE 2 — BibTeX Garantito per Ogni PDF

#### [ ] Task 2.1: Refactoring `bibtex_generator.py` — `ensure_complete_bibtex()`

**File:** `paperhunter/ingestion/bibtex_generator.py` (modifica)  
**Stato attuale:** La funzione `generate_bibtex_entry(paper)` può produrre entry con campi mancanti (es. `year = {Unknown}`, `booktitle = {}`) se S2 non ha i dati. Dopo l'ingestion, `paper.bibtex_entry` può rimanere None.  
**Obiettivo:** Garantire che ogni paper abbia sempre un'entry BibTeX valida, anche in assenza parziale o totale di metadati da Semantic Scholar.  
**Dettagli:**
- Aggiungere funzione `ensure_complete_bibtex(paper: Paper) -> Paper`:
  - Se `paper.bibtex_entry` è None o vuoto, chiama `generate_bibtex_entry(paper)`.
  - Se `generate_bibtex_entry()` produce campi critici con valori placeholder (`"Unknown"`, `""`),
    usa i dati disponibili in `paper` per completarli:
    - `year`: usa `paper.year` oppure anno dal filename se presente
    - `author`: usa `", ".join(paper.authors)` oppure `"Anonymous"`
    - `title`: usa `paper.title` (dovrebbe sempre esserci)
    - `booktitle`/`journal`: usa `paper.venue or paper.journal or "arXiv Preprint"` con normalizzazione
  - Se il paper è chiaramente un preprint (venue contiene "arxiv" case-insensitive), usa il tipo
    `@misc` con campo `howpublished = {arXiv preprint}`; altrimenti usa `@inproceedings` o `@article`.
  - Aggiungere funzione `validate_bibtex_completeness(entry: str) -> dict[str, str]` che
    restituisce un dizionario `{field_name: "present"|"missing"|"placeholder"}` per i campi
    critici: `title`, `author`, `year`, `booktitle`/`journal`, `url`.
  - Il campo `bibtex_entry` non può essere None dopo `ensure_complete_bibtex()`.
- Aggiornare `enrich_paper_bibtex(paper)` per chiamare `ensure_complete_bibtex()` al termine.

**Test:** Creare 4 Paper fixture (campo year mancante, autori vuoti, venue sconosciuta, tutti i campi presenti) e verificare che `ensure_complete_bibtex()` produca sempre un'entry con tipo + chiave + campi obbligatori non vuoti.

---

#### [ ] Task 2.2: Test di integrazione BibTeX fine-to-fine

**File:** `tests/test_bibliography_generator.py` (modifica/aggiunta test)  
**Stato attuale:** I test esistenti coprono la generazione base, ma non il caso in cui l'entry è incompleta e deve essere completata con fallback.  
**Obiettivo:** Aggiungere test che coprono l'intera catena: Paper con metadati parziali → `ensure_complete_bibtex()` → entry BibTeX valida → `validate_bibtex_completeness()` → report corretto.  
**Dettagli:**
- Aggiungere 6 test:
  1. Paper senza year → entry con anno da filename o `"2000"` come placeholder sicuro
  2. Paper senza autori → entry con `"Anonymous"`
  3. Paper arXiv → tipo `@misc`, `howpublished` corretto
  4. Paper conference → tipo `@inproceedings`, `booktitle` presente
  5. Paper journal → tipo `@article`, `journal` presente
  6. `validate_bibtex_completeness()` su entry completa → tutti i campi `"present"`
- Nessun mock di S2 — tutti i test usano oggetti Paper costruiti manualmente.

**Test:** `pytest tests/test_bibliography_generator.py -v` → tutti i test passano.

---

### FASE 3 — Writer Agent con Figure Specifiche

#### [ ] Task 3.1: Nuovo tool `get_figure_for_section()` in `retrieval_tools.py`

**File:** `paperhunter/agents/tools/retrieval_tools.py` (modifica)  
**Stato attuale:** Il tool `search_figure_descriptions` esegue una ricerca testuale sulle descrizioni delle figure, ma non permette al writer di richiedere una figura specifica di un paper specifico (es. "la figura di architettura del paper Zou_2023_GCG").  
**Obiettivo:** Aggiungere un tool che permetta al writer di recuperare una o più figure di un paper specifico, filtrate per tipo.  
**Dettagli:**
- Aggiungere funzione (tool LlamaIndex) `get_figure_for_section`:
  ```python
  def get_figure_for_section(paper_bibtex_key: str, figure_type: str = "any") -> str:
      """
      Retrieve a specific figure from a paper for inclusion in the survey.
      
      Args:
          paper_bibtex_key: The BibTeX key of the paper (e.g., "Zou_2023_GCG").
          figure_type: Type of figure to retrieve. One of: "any", "results_plot",
                       "architecture_diagram", "flowchart", "comparison_table",
                       "confusion_matrix", "attack_diagram", "defense_pipeline".
      
      Returns:
          JSON string with figure metadata (figure_id, paper_key, figure_type,
          description, image_path, caption) or "NO_FIGURE_FOUND" if not available.
      """
  ```
- La funzione:
  1. Interroga Qdrant sulla collection `figure_descriptions` con filtro payload
     `bibtex_key == paper_bibtex_key` (e opzionalmente `figure_type == figure_type`)
  2. Se non trova risultati, interroga `image_embeddings` con lo stesso filtro
  3. Restituisce il primo match in formato JSON con campi:
     `figure_id`, `paper_key`, `figure_type`, `description`, `image_path`, `caption`
  4. Se `figure_type == "any"`, restituisce la figura con score più alto
- Il tag da inserire nel testo è `{{FIGURE:<paper_key>:<figure_type>}}`, descritto nella docstring.
- Registrare il tool nel `ReActAgent` accanto agli altri tool in `writer_agent.py`.

**Test:** Mock di Qdrant con 3 figure di test, chiamare `get_figure_for_section("Zou_2023_GCG", "architecture_diagram")` e verificare il JSON risultante.

---

#### [ ] Task 3.2: Aggiornare il prompt del Writer Agent per i tag figura

**File:** `paperhunter/agents/writer_agent.py` (modifica)  
**Stato attuale:** Il prompt del writer non menziona la possibilità di inserire figure nel testo. Non esiste un formato di tag standardizzato.  
**Obiettivo:** Aggiungere istruzioni esplicite nel prompt del writer per l'uso del tool `get_figure_for_section` e del tag `{{FIGURE:...}}`.  
**Dettagli:**
- Nel prompt `_WRITER_PROMPT_TEMPLATE` (modalità static) e nel system prompt della modalità agentica,
  aggiungere una sezione "Figure Instructions":
  ```
  ## Figure Instructions (optional, use sparingly)
  
  If a figure from a specific paper would significantly enhance the explanation of a concept,
  you MAY insert a figure placeholder using this format:
  
      {{FIGURE:bibtex_key:figure_type}}
  
  Example: {{FIGURE:Zou_2023_GCG:architecture_diagram}}
  
  Valid figure_types: results_plot, architecture_diagram, flowchart, comparison_table,
  confusion_matrix, attack_diagram, defense_pipeline, any
  
  Rules:
  - Use at most 1-2 figures per section.
  - Only insert a figure if you have confirmed it exists by calling get_figure_for_section().
  - Place the tag on its own line, immediately after the sentence it illustrates.
  - Do NOT insert a figure tag without first retrieving it via get_figure_for_section().
  ```
- In modalità "static", il writer NON ha accesso al tool, quindi le istruzioni figura sono omesse.
- Non modificare la logica di generazione: il tag rimane una stringa letterale nel testo,
  sarà il `FigureResolver` (Task 3.3) a risolverlo.

**Test:** Testare che il prompt sia costruito correttamente chiamando `_build_writer_prompt()` (o equivalente) con e senza modalità agentica, e verificare che le istruzioni figura appaiano solo nella modalità agentica.

---

#### [ ] Task 3.3: Creare `figure_resolver.py` — risoluzione tag `{{FIGURE:...}}`

**File:** `paperhunter/postprocessing/figure_resolver.py` (nuovo)  
**Stato attuale:** L'`AssemblerAgent` produce il Markdown finale ma non gestisce tag speciali per le figure.  
**Obiettivo:** Creare `FigureResolver` che scansiona il testo finale del survey e sostituisce ogni tag `{{FIGURE:key:type}}` con il markup appropriato.  
**Dettagli:**
- Classe `FigureResolver` con metodo principale:
  ```python
  def resolve(self, text: str, output_format: Literal["markdown", "latex"] = "markdown") -> tuple[str, list[str]]:
      """
      Resolve all {{FIGURE:key:type}} tags in text.
      Returns: (resolved_text, list_of_unresolved_tags)
      """
  ```
- Regex per trovare i tag: `r"\{\{FIGURE:([^:}]+):([^}]+)\}\}"`
- Per ogni tag trovato:
  1. Interroga Qdrant per recuperare la figura (stessa logica di `get_figure_for_section`)
  2. Se trovata e `image_path` esiste su disco:
     - Markdown: `\n![{caption}]({image_path})\n*Figura: {description}* [{paper_key}]\n`
     - LaTeX: `\n\\begin{{figure}}[h]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{{{image_path}}}\n\\caption{{{caption} \\cite{{{paper_key}}}}}\n\\end{{figure}}\n`
  3. Se la figura non è trovata o il file immagine non esiste su disco:
     - Rimuovere il tag dal testo (sostituire con stringa vuota)
     - Aggiungere il tag alla lista `unresolved_tags`
- Integrare `FigureResolver.resolve()` nell'`AssemblerAgent` subito prima di produrre l'output finale.
- Loggare il numero di tag risolti e non risolti.

**Test:** Creare testo di test con 3 tag (1 figura esistente, 1 figura non trovata, 1 path non esistente su disco) e verificare che il testo risultante contenga il markup corretto e la lista `unresolved_tags` abbia 2 elementi.

---

### FASE 4 — Source Venue Ranking nel Retriever

#### [ ] Task 4.1: Creare `venue_ranker.py` — tier dictionary e boost score

**File:** `paperhunter/indexing/venue_ranker.py` (nuovo)  
**Stato attuale:** Il retriever tratta allo stesso modo paper da arXiv e paper da NeurIPS. Non c'è nessun meccanismo per dare priorità a fonti accademiche più autorevoli.  
**Obiettivo:** Implementare `VenueRanker` con un dizionario di tier per venue/journal e un metodo che calcola il boost da aggiungere allo score RRF.  
**Dettagli:**
- Definire le costanti:
  ```python
  # Tier 1 — Conferenze e Journal top nel campo ML/NLP/Security
  TIER1_VENUES = frozenset({
      "neurips", "nips", "icml", "iclr", "acl", "emnlp", "naacl", "coling",
      "ieee s&p", "usenix security", "ccs", "ndss", "cvpr", "iccv", "eccv",
      "nature", "science", "cell", "jmlr", "tacl", "ieee tifs", "acm ccs",
  })
  # Tier 2 — Workshop ufficiali, conferenze di secondo livello
  TIER2_VENUES = frozenset({
      "findings of acl", "findings of emnlp", "findings of naacl",
      "aaai", "ijcai", "sigkdd", "www", "wsdm", "aistats",
      "workshop", "trustworthy ml", "socially responsible ml",
  })
  # Tier 3 — Preprint (nessuno scambio di revisione formale)
  TIER3_VENUES = frozenset({
      "arxiv", "biorxiv", "medrxiv", "preprint", "ssrn",
  })
  ```
- Classe `VenueRanker`:
  - Metodo `get_tier(venue: str | None) -> int` (0=unknown, 1=top, 2=medium, 3=preprint)
  - Metodo `get_boost(venue: str | None) -> float`:
    - Tier 1 → `+tier1_boost` (configurabile, default +0.15)
    - Tier 2 → `+tier2_boost` (configurabile, default +0.05)
    - Tier 3 → `+tier3_penalty` (configurabile, default -0.05)
    - Tier 0 (unknown) → 0.0
- Il matching è **case-insensitive** e **substring-based** (es. `"NeurIPS 2023"` matcha `"neurips"`).
- Configurazione da `config.yaml`:
  ```yaml
  retrieval:
    venue_ranking_enabled: true
    venue_tier1_boost: 0.15
    venue_tier2_boost: 0.05
    venue_tier3_penalty: 0.05   # valore positivo, sarà sottratto internamente
  ```
- Aggiungere `VenueRankingConfig` a `RetrievalConfig` in `config.py`.

**Test:** Testare `get_tier()` e `get_boost()` con 10 venue campione (NeurIPS 2023, ICLR 2024, arXiv, workshop ICLR, None, università sconosciuta, ecc.) e verificare tier e boost corretti.

---

#### [ ] Task 4.2: Integrare `VenueRanker` in `hybrid_retriever.py`

**File:** `paperhunter/indexing/hybrid_retriever.py` (modifica)  
**Stato attuale:** Il `HybridRetriever` restituisce i candidati ordinati per score RRF + recency boost. Non c'è nessun contributo della venue allo score finale.  
**Obiettivo:** Aggiungere il venue boost come step post-RRF, prima del reranking Gemini Flash, se `venue_ranking_enabled = true` in config.  
**Dettagli:**
- Nel metodo `retrieve()` (o equivalente), dopo la fusione RRF e il recency boost:
  ```python
  if settings.retrieval.venue_ranking_enabled:
      venue_ranker = VenueRanker()
      for chunk in candidates:
          venue = chunk.metadata.get("venue") or chunk.metadata.get("journal")
          boost = venue_ranker.get_boost(venue)
          chunk.score = min(1.0, max(0.0, chunk.score + boost))
  ```
- Aggiungere il field `venue_tier: Optional[int]` al dataclass `RetrievedChunk`, valorizzato
  durante il boost per tracciabilità.
- Il metadata payload dei chunk in Qdrant deve includere il campo `venue` per essere filtrato.
  Verificare che `ingestion_pipeline` salvi `venue` nel payload di ogni chunk; se mancante,
  aggiungere il campo al punto di costruzione del payload.
- Loggare a livello DEBUG per ogni candidato: `chunk_id={...} venue={...} tier={...} boost={:+.2f}`.
- Enable/disable controllato da `settings.retrieval.venue_ranking_enabled`.

**Test:** Costruire 5 `RetrievedChunk` con venue: `"NeurIPS 2023"`, `"arXiv"`, `None`, `"ICLR 2024"`, `"Workshop on Trustworthy ML"`. Verificare che gli score siano nell'ordine atteso dopo il boost.

---

#### [ ] Task 4.3: Test end-to-end venue ranking

**File:** `tests/test_venue_ranker.py` (nuovo)  
**Stato attuale:** Non esistono test per il venue ranker.  
**Obiettivo:** Test completo di `VenueRanker` e della sua integrazione nel retriever.  
**Dettagli:**
- Almeno 12 test unitari:
  1. `get_tier("NeurIPS 2023")` → 1
  2. `get_tier("arxiv")` → 3
  3. `get_tier("arXiv preprint")` → 3
  4. `get_tier("Workshop on Trustworthy ML at ICML")` → 2
  5. `get_tier(None)` → 0
  6. `get_tier("Università di Bologna")` → 0
  7. `get_boost("ICLR 2024")` → 0.15
  8. `get_boost("arXiv")` → -0.05
  9. `get_boost(None)` → 0.0
  10. `get_boost("findings of acl")` → 0.05
  11. Score dopo boost non supera 1.0
  12. Score dopo boost non va sotto 0.0

**Test:** `pytest tests/test_venue_ranker.py -v` → tutti i test passano.

---

## 📊 Riepilogo Task

| # | Task | Fase | File principale | Stima |
|---|------|------|-----------------|-------|
| 1.1 | Estrazione date aggressiva GROBID | Meta | `grobid_client.py` | Media |
| 1.2 | Parsing robusto publicationDate S2 | Meta | `semantic_scholar.py` | Bassa |
| 1.3 | `MetadataEnricher` — pipeline a cascata | Meta | `metadata_enricher.py` *(nuovo)* | Alta |
| 1.4 | Integrazione `MetadataEnricher` in pipeline | Meta | `ingestion_pipeline.py` | Bassa |
| 2.1 | `ensure_complete_bibtex()` + fallback | BibTeX | `bibtex_generator.py` | Media |
| 2.2 | Test BibTeX fine-to-fine | BibTeX | `test_bibliography_generator.py` | Bassa |
| 3.1 | Tool `get_figure_for_section()` | Figure | `retrieval_tools.py` | Media |
| 3.2 | Prompt writer con tag figura | Figure | `writer_agent.py` | Bassa |
| 3.3 | `FigureResolver` — risoluzione tag | Figure | `figure_resolver.py` *(nuovo)* | Media |
| 4.1 | `VenueRanker` — tier + boost | Ranking | `venue_ranker.py` *(nuovo)* | Media |
| 4.2 | Integrazione `VenueRanker` nel retriever | Ranking | `hybrid_retriever.py` | Bassa |
| 4.3 | Test end-to-end venue ranking | Ranking | `test_venue_ranker.py` *(nuovo)* | Bassa |

---

## 📁 File di Riferimento

- `paperhunter/models.py` — modello `Paper` (aggiungere `metadata_confidence`, `year_source`)
- `paperhunter/config.py` — `RetrievalConfig` (aggiungere `VenueRankingConfig`)
- `config.yaml` — sezione `retrieval` (aggiungere `venue_ranking_*`)
- `paperhunter/ingestion/grobid_client.py` — parser TEI-XML
- `paperhunter/ingestion/semantic_scholar.py` — enricher S2
- `paperhunter/ingestion/bibtex_generator.py` — generatore BibTeX
- `paperhunter/ingestion/ingestion_pipeline.py` — orchestratore ingestion
- `paperhunter/indexing/hybrid_retriever.py` — retriever ibrido RRF
- `paperhunter/agents/tools/retrieval_tools.py` — tool ReAct writer
- `paperhunter/agents/writer_agent.py` — writer agent
- `paperhunter/agents/assembler_agent.py` — assembler finale
- `paperhunter/indexing/qdrant_store.py` — vector store Qdrant
