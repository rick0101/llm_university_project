# TODO5_ImageProcessing.md — Filtraggio Intelligente e Indicizzazione Multimodale delle Immagini

> **Riferimento:** `instructions5.txt`  
> **Prerequisito:** Tutti i task di `TODO.md`, `TODO3.md` e `TODO4_Scalability.md` completati  
> **Data inizio:** [Data da definire]  
> **Ultimo aggiornamento:** 12 Marzo 2026  
> **Target:** Filtraggio intelligente delle immagini estratte dai PDF (eliminare loghi, watermark, decorazioni) + indicizzazione multimodale con Gemini Embedding 2 Preview + pipeline a 3 stadi per ridurre i costi Vision del 70-80%

---

## 🏗️ Architettura Estesa (Nuovi/Modificati)

```
config.yaml                       ← Nuova sezione: image_filtering

paperhunter/
├── ingestion/
│   ├── figure_extractor.py       ← MODIFICATO: Integra filtro euristico + embedding prima di Vision
│   └── image_relevance_filter.py ← NUOVO: Classificatore euristico basato su caratteristiche visive
│
├── utils/
│   └── multimodal_embedder.py    ← NUOVO: Wrapper Gemini Embedding 2 Preview per immagini
│
├── indexing/
│   └── qdrant_store.py           ← MODIFICATO: Nuova collection image_embeddings
│
├── agents/
│   └── tools/
│       └── retrieval_tools.py    ← MODIFICATO: Aggiornato search_figure_descriptions per image embeddings
│
└── config.py                     ← MODIFICATO: ImageFilteringConfig
```

---

## 📋 TASK DETTAGLIATI

### FASE 0 — Preparazione e Dipendenze

#### [ ] Task 0.1: Aggiornare dipendenze e configurazioni per image filtering

**File:** `config.yaml` (modifica), `paperhunter/config.py` (modifica)  
**Stato attuale:** Il `FigureExtractor` filtra solo per dimensioni minime (`min_width=150`, `min_height=150`, `min_area=40000`). Non esiste una sezione `image_filtering` nel config. Le dipendenze necessarie (`Pillow`, `numpy`) sono già presenti nel progetto (Pillow usata da `figure_extractor.py` per resize, numpy da TODO4 per clustering). Gemini Embedding 2 Preview è già impostato come modello embedding in `config.py` (`gemini-embedding-2-preview`).  
**Obiettivo:** Aggiungere la configurazione centralizzata per il filtraggio intelligente delle immagini e l'embedding multimodale.  
**Dettagli:**
- **`config.yaml`:** Aggiungere nuova sezione top-level `image_filtering`:
  ```yaml
  image_filtering:
    enabled: true                          # Abilita/disabilita il filtraggio intelligente
    # --- Filtro Euristico (Stage 1) ---
    min_aspect_ratio: 0.3                  # Min w/h — filtra barre verticali
    max_aspect_ratio: 4.0                  # Max w/h — filtra banner orizzontali
    min_color_entropy: 3.0                 # Entropia Shannon minima (scala 0-8 per 8-bit)
    min_edge_density: 0.05                 # Percentuale minima pixel bordo (0.0-1.0)
    skip_first_page: true                  # Penalizza immagini dalla prima pagina (loghi)
    skip_last_page: true                   # Penalizza immagini dall'ultima pagina (copyright)
    relevance_threshold: 0.5               # Soglia punteggio combinato [0.0, 1.0]
    # --- Filtro Embedding Multimodale (Stage 2) ---
    embedding_filter_enabled: true         # Abilita filtro via embedding multimodale
    embedding_similarity_threshold: 0.3    # Soglia coseno minima vs riferimenti scientifici
  ```
- **`paperhunter/config.py`:** Creare nuovo sotto-modello `ImageFilteringConfig(BaseModel)`:
  - Campi: `enabled`, `min_aspect_ratio`, `max_aspect_ratio`, `min_color_entropy`, `min_edge_density`, `skip_first_page`, `skip_last_page`, `relevance_threshold`, `embedding_filter_enabled`, `embedding_similarity_threshold`
  - Validatori Pydantic per range: `min_aspect_ratio ∈ (0.0, 1.0]`, `max_aspect_ratio ∈ [1.0, 20.0]`, `min_color_entropy ∈ [0.0, 8.0]`, `min_edge_density ∈ [0.0, 1.0]`, `relevance_threshold ∈ [0.0, 1.0]`, `embedding_similarity_threshold ∈ [0.0, 1.0]`
- **`paperhunter/config.py`:** Registrare `ImageFilteringConfig` nel modello root `Settings` come `image_filtering: ImageFilteringConfig`
- **`config.yaml`:** Nella sezione `qdrant`, aggiungere `collection_image_embeddings: "image_embeddings"` (per la futura collection multimodale)
- **`paperhunter/config.py`:** Aggiungere campo `collection_image_embeddings: str = "image_embeddings"` a `QdrantConfig`  
**Test:** Eseguire `python -c "from paperhunter.config import settings; print(settings.image_filtering.model_dump())"` e verificare che tutti i campi siano caricati correttamente con i tipi e range attesi.

---

### FASE 1 — Filtraggio Euristico delle Immagini (Senza LLM)

#### [ ] Task 1.1: Creare `ingestion/image_relevance_filter.py`

**File:** `paperhunter/ingestion/image_relevance_filter.py` (nuovo)  
**Stato attuale:** Non esiste alcun modulo di classificazione delle immagini. Il `FigureExtractor` filtra solo per dimensione pixel (min 150×150 e 40k area), ma loghi, watermark, foto decorative e immagini irrilevanti passano comunque il filtro. Ogni immagine viene poi inviata a Gemini Vision (costoso e inutile per immagini non rilevanti).  
**Obiettivo:** Classificare le immagini estratte come `SCIENTIFIC_FIGURE`, `LIKELY_DECORATIVE` o `UNCERTAIN` usando solo euristiche locali (PIL + NumPy), senza chiamate API.  
**Dettagli:**
- Enum `ImageClassification`: `SCIENTIFIC_FIGURE`, `LIKELY_DECORATIVE`, `UNCERTAIN`
- Dataclass `ImageScore`:
  - `aspect_ratio_score: float` — punteggio [0, 1] per aspect ratio
  - `entropy_score: float` — punteggio [0, 1] per entropia colore
  - `edge_density_score: float` — punteggio [0, 1] per densità bordi
  - `position_score: float` — punteggio [0, 1] per posizione nel documento
  - `size_score: float` — punteggio [0, 1] per dimensione relativa
  - `combined_score: float` — media pesata dei punteggi
  - `classification: ImageClassification` — classificazione finale
- Classe `ImageRelevanceFilter`:
  - `__init__(config: ImageFilteringConfig | None = None)`: carica config dal settings se non fornito
  - `classify(image_bytes: bytes, page_number: int, total_pages: int, page_width: float, page_height: float) -> ImageScore`:
    - Calcola tutti i punteggi individuali
    - Combina in un punteggio complessivo (media pesata configurabile)
    - Classifica: >= threshold → `SCIENTIFIC_FIGURE`, < threshold/2 → `LIKELY_DECORATIVE`, else → `UNCERTAIN`
  - `classify_batch(images: list[dict]) -> list[ImageScore]`: batch processing con logging Rich
  - Metodi privati per ciascuna euristica:
    - `_score_aspect_ratio(width, height) -> float`: penalizza aspect ratio estremi (< 0.3 o > 4.0). Grafici e tabelle hanno tipicamente ratio tra 0.5 e 2.5
    - `_score_color_entropy(image_bytes) -> float`: calcola entropia Shannon sull'istogramma colore. Immagini con poca variazione cromatica (loghi monocromo, icone) hanno entropia bassa. Usa `PIL.Image.histogram()` + formula Shannon
    - `_score_edge_density(image_bytes) -> float`: applica filtro bordi (PIL `FIND_EDGES`) e conta la percentuale di pixel non neri. Grafici, tabelle e diagrammi hanno alta densità di bordi. Foto e loghi hanno densità diversa
    - `_score_page_position(page_number, total_pages) -> float`: penalizza prima pagina (loghi intestazione) e ultima pagina (copyright, footer). Pagine centrali → score pieno
    - `_score_relative_size(img_width, img_height, page_width, page_height) -> float`: immagini troppo piccole (< 5% pagina) o troppo grandi (> 90% pagina, probabilmente full-page background) penalizzate. Range ideale: 10-70% dell'area della pagina
- Logging con Rich: per ogni batch, mostrare tabella riassuntiva (classificazione, punteggi medi, distribuzione)  
**Test:** Creare immagini sintetiche con PIL:
  - Immagine tipo "grafico" (assi, linee, alta entropia e bordi) → `SCIENTIFIC_FIGURE`
  - Immagine tipo "logo" (piccola, pochi colori, bassa entropia) → `LIKELY_DECORATIVE`
  - Immagine tipo "foto" (entropia alta ma bordi bassi) → `UNCERTAIN`
  Verificare che le classificazioni corrispondano alle aspettative per almeno l'80% dei casi.

---

#### [ ] Task 1.2: Integrare il filtro euristico in `figure_extractor.py`

**File:** `paperhunter/ingestion/figure_extractor.py` (modifica)  
**Stato attuale:** `extract_images_from_pdf()` filtra solo per dimensioni pixel. Tutte le immagini che passano il filtro dimensionale vengono poi inviate a `describe_figure()` (Gemini Vision), anche se sono loghi o immagini decorative. Il metodo `extract_and_describe()` processa TUTTE le immagini estratte.  
**Obiettivo:** Applicare il filtro euristico (Task 1.1) tra l'estrazione e la descrizione Vision, scartando le immagini classificate come `LIKELY_DECORATIVE`.  
**Dettagli:**
- **Modificare `extract_images_from_pdf()`:**
  - Aggiungere parametro `total_pages` nel dict restituito per ogni immagine (già disponibile da `len(doc)`)
  - Aggiungere `page_width` e `page_height` nel dict (dalla `page.rect` di PyMuPDF)
- **Modificare `extract_and_describe()`:**
  - Dopo l'estrazione delle immagini, applicare `ImageRelevanceFilter.classify_batch()`
  - Le immagini `LIKELY_DECORATIVE` vengono loggate e scartate (non inviate a Gemini Vision)
  - Le immagini `UNCERTAIN` vengono passate a Gemini Vision (beneficio del dubbio)
  - Le immagini `SCIENTIFIC_FIGURE` vengono passate a Gemini Vision (confermate rilevanti)
  - Aggiungere campo `relevance_score: float` nel dict delle immagini (per logging e debug)
  - Controllare `settings.image_filtering.enabled`: se `False`, skip il filtraggio (backward compatibility)
- **Logging:** Mostrare statistiche di filtraggio:
  - `Immagini estratte: N, Filtrate (decorative): M, Passate a Vision: K`
  - Per ogni immagine filtrata, loggare il motivo principale (es. "score 0.18: bassa entropia + prima pagina")
- **Preservare** la firma di `extract_and_describe()`: i `PaperChunk` restanti vengono generati esattamente come prima  
**Test:** Testare con un PDF di esempio che contiene sia figure scientifiche che loghi. Verificare che i loghi vengano filtrati e che le figure scientifiche passino. Mock del `describe_figure()` per contare le chiamate: devono essere meno del totale delle immagini estratte.

---

### FASE 2 — Embedding Multimodale (Gemini Embedding 2 Preview)

#### [ ] Task 2.1: Creare `utils/multimodal_embedder.py`

**File:** `paperhunter/utils/multimodal_embedder.py` (nuovo)  
**Stato attuale:** Il sistema usa `llama-index-embeddings-google-genai` (via `get_embedding_model()`) per embedding testuali con Gemini Embedding 2 Preview. Tuttavia, il wrapper LlamaIndex supporta solo input testuali. L'embedding multimodale nativo (immagini, audio, video) è disponibile solo tramite il SDK `google-genai` diretto, con il endpoint `client.models.embed_content()`.  
**Obiettivo:** Creare un wrapper dedicato per Gemini Embedding 2 Preview in modalità multimodale, che possa generare embedding da immagini (e opzionalmente da coppie testo+immagine).  
**Dettagli:**
- Classe `MultimodalEmbedder`:
  - `__init__(model: str | None = None)`: usa `settings.gemini.embedding_model` di default
  - `embed_image(image_bytes: bytes, mime_type: str = "image/png") -> list[float]`:
    - Chiama `client.models.embed_content(model=..., contents=[image_part])` via SDK `google-genai`
    - `image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)`
    - Restituisce il vettore 3072-dimensionale
    - Rate limiting via `get_gemini_limiter()`
  - `embed_text(text: str) -> list[float]`:
    - Wrapper per embedding testuale via stesso SDK (per i reference embeddings)
    - Usa `types.Part.from_text(text)` come input
  - `embed_images_batch(images: list[dict], show_progress: bool = True) -> list[list[float]]`:
    - Batch processing con progress bar Rich
    - Ogni `dict` contiene `image_bytes` e `mime_type`
    - Rate limiting e retry su errori transitori
  - `compute_reference_embeddings() -> dict[str, list[float]]`:
    - Calcola e cacha embedding di riferimento per categorie scientifiche:
      - `"scientific_plot"` → "scientific data plot with labeled axes showing experimental results"
      - `"comparison_table"` → "comparison table with rows and columns of numerical data"
      - `"architecture_diagram"` → "system architecture diagram showing components and connections"
      - `"flowchart"` → "flowchart or pipeline diagram showing process steps and decisions"
      - `"confusion_matrix"` → "confusion matrix or heatmap showing classification results"
    - Cache su disco (`cache/reference_embeddings.json`) per evitare ricalcolo ad ogni run
  - `classify_image_relevance(image_bytes: bytes, mime_type: str, reference_embeddings: dict | None = None) -> tuple[float, str]`:
    - Embed l'immagine
    - Calcola similarità coseno con tutti i riferimenti
    - Restituisce `(max_similarity, best_matching_category)`
    - Se `max_similarity < embedding_similarity_threshold` → non rilevante
- Funzione standalone `get_multimodal_embedder() -> MultimodalEmbedder`: singleton lazy  
**Test:** Creare immagini sintetiche (un grafico matplotlib salvato come PNG, un rettangolo colorato come "logo"). Embed entrambe e i riferimenti testuali. Verificare che il grafico abbia similarità più alta con "scientific_plot" rispetto al rettangolo. Verificare che i vettori siano 3072-dim. Verificare cache su disco.

---

#### [ ] Task 2.2: Classificazione rilevanza via embedding e integrazione in FigureExtractor

**File:** `paperhunter/ingestion/figure_extractor.py` (modifica)  
**Stato attuale:** Dopo il Task 1.2, il `FigureExtractor` applica il filtro euristico (Stage 1). Le immagini `UNCERTAIN` e `SCIENTIFIC_FIGURE` passano tutte a Gemini Vision. Le immagini `UNCERTAIN` potrebbero però essere ancora decorative — serve un secondo livello di filtraggio più intelligente prima della costosa chiamata Vision.  
**Obiettivo:** Aggiungere lo Stage 2 (filtro embedding multimodale) per le immagini che passano lo Stage 1, riducendo ulteriormente le chiamate Vision.  
**Dettagli:**
- **Modificare `extract_and_describe()`:**
  - Dopo il filtro euristico (Stage 1), per le immagini `UNCERTAIN` e opzionalmente `SCIENTIFIC_FIGURE`:
    1. Chiamare `MultimodalEmbedder.classify_image_relevance()` per ciascuna
    2. Se `max_similarity < embedding_similarity_threshold` → scartare (non rilevante)
    3. Se rilevante → passare a Gemini Vision (Stage 3)
  - Per le immagini `SCIENTIFIC_FIGURE` con score euristico alto (> 0.8): skip Stage 2, vanno direttamente a Vision (ottimizzazione: non spendere embedding per immagini già chiaramente rilevanti)
  - Controllare `settings.image_filtering.embedding_filter_enabled`: se `False`, skip Stage 2
- **Aggiungere nel dict delle immagini:**
  - `embedding_similarity: float | None` — similarità coseno max vs riferimenti
  - `best_match_category: str | None` — categoria di riferimento più simile
  - `embedding: list[float] | None` — vettore embedding (riutilizzabile per indicizzazione, Task 2.3)
- **Logging Stage 2:** Mostrare quante immagini sono state filtrate dallo Stage 2 vs quante passate
  - `Stage 1: N → K immagini; Stage 2: K → J immagini; Vision calls: J`
- **Preservare** l'embedding calcolato per le immagini rilevanti: verrà usato nel Task 2.3 per l'indicizzazione senza ricalcolo  
**Test:** Testare la pipeline a 2 stadi con un set di immagini miste (grafici, loghi, foto). Verificare che:
  - Le immagini `LIKELY_DECORATIVE` (Stage 1) non vengano embedded (risparmio API)
  - Le immagini `UNCERTAIN` con bassa similarità embedding vengano scartate (Stage 2)
  - Le chiamate Vision siano significativamente ridotte rispetto al baseline (tutte le immagini)
  - Il conteggio delle Vision calls sia loggato correttamente

---

#### [ ] Task 2.3: Indicizzazione diretta immagini come vettori multimodali in Qdrant

**File:** `paperhunter/indexing/qdrant_store.py` (modifica), `paperhunter/ingestion/figure_extractor.py` (modifica), `paperhunter/models.py` (modifica)  
**Stato attuale:** Le immagini rilevanti vengono descritte con Gemini Vision e indicizzate come chunk testuali nella collection `figure_descriptions`. I vettori embedding multimodali calcolati nello Stage 2 (Task 2.2) vengono attualmente scartati dopo la classificazione. Non esiste indicizzazione diretta dei vettori immagine.  
**Obiettivo:** Salvare gli embedding multimodali delle immagini rilevanti in una nuova collection Qdrant `image_embeddings`, abilitando il retrieval cross-modale (query testuale → figure simili).  
**Dettagli:**
- **`paperhunter/models.py`:**
  - Aggiungere `IMAGE_EMBEDDING = "image_embedding"` a `ChunkType` enum
- **`paperhunter/indexing/qdrant_store.py`:**
  - Aggiungere `COLLECTION_IMAGE_EMBEDDINGS` class attribute (dal config `collection_image_embeddings`)
  - Registrare in `_ensure_collections()`: stesse dimensioni (3072) e distanza (Cosine) delle altre collection
  - Aggiungere metodo `add_image_embeddings(chunks: list[PaperChunk], embeddings: list[list[float]], collection_name: str | None = None) -> int`:
    - Upsert chunk con embedding pre-calcolati (non serve ricalcolo, già calcolati nello Stage 2)
    - I chunk hanno `chunk_type=IMAGE_EMBEDDING`, `image_path` per referenza al file su disco
  - Registrare in `get_stats()`
- **`paperhunter/ingestion/figure_extractor.py`:**
  - Dopo la generazione dei `PaperChunk` testuali (descrizione Vision), aggiungere un passo di indicizzazione multimodale:
    - Per ogni immagine rilevante che ha un `embedding` calcolato nello Stage 2:
    - Creare un `PaperChunk` aggiuntivo con `chunk_type=IMAGE_EMBEDDING`
    - Salvare nella collection `image_embeddings` con l'embedding pre-calcolato
  - Nuovo metodo `_save_image_embeddings_to_store(image_chunks: list[PaperChunk], embeddings: list[list[float]], store: QdrantStore | None = None) -> int`
  - Il `PaperChunk` per l'embedding immagine contiene: `text` = caption o breve descrizione, `image_path` = percorso file, metadata standard (paper_id, bibtex_key, category, year)
- **Output:** Ogni immagine rilevante produce DUE entry nel database:
  1. `figure_descriptions` collection: embedding testuale della descrizione Vision (come prima)
  2. `image_embeddings` collection: embedding multimodale dell'immagine stessa (nuovo)  
**Test:** Indicizzare le immagini di un paper di test. Verificare che la collection `image_embeddings` contenga gli stessi paper_id della `figure_descriptions`. Verificare che una query testuale sulla collection `image_embeddings` restituisca risultati coerenti (es. query "attack success rate" restituisce figure con grafici di performance).

---

### FASE 3 — Pipeline Integrata e Test End-to-End

#### [ ] Task 3.1: Aggiornare i retrieval tool per supportare image embeddings

**File:** `paperhunter/agents/tools/retrieval_tools.py` (modifica)  
**Stato attuale:** Il tool `search_figure_descriptions` cerca nella collection `figure_descriptions` (embedding testuali delle descrizioni Vision). Non cerca nella nuova collection `image_embeddings` (embedding multimodali delle immagini stesse). Il writer agent non ha accesso al retrieval cross-modale.  
**Obiettivo:** Estendere il tool `search_figure_descriptions` per cercare anche nella collection `image_embeddings`, o creare un tool separato per il retrieval cross-modale.  
**Dettagli:**
- **Opzione scelta: Estendere `search_figure_descriptions`** (meno disruption):
  - Aggiungere parametro opzionale `include_image_embeddings: bool = True`
  - Se abilitato: cercare in ENTRAMBE le collection (`figure_descriptions` + `image_embeddings`)
  - Merge dei risultati: deduplicare per `paper_id + image_path`, mantenere lo score più alto
  - Nella risposta formattata, indicare la fonte: `[description]` o `[image_embedding]`
- **Verificare** che la collection `image_embeddings` esista prima di cercare (graceful fallback se non indicizzata)
- **Logging:** Mostrare quanti risultati provengono da ciascuna collection  
**Test:** Con mock di entrambe le collection, verificare che la ricerca combinata restituisca risultati deduplicati e ordinati per score. Verificare il fallback se `image_embeddings` non è disponibile.

---

#### [ ] Task 3.2: Test end-to-end completo con 10 PDF per sezione

**File:** `test_e2e_image_processing.py` (nuovo)  
**Stato attuale:** La pipeline completa (estrazione → filtraggio → embedding → descrizione → indicizzazione) non è mai stata testata end-to-end con un dataset reale rappresentativo.  
**Obiettivo:** Validare l'intera pipeline su un dataset di 40 PDF (10 per ciascuna delle 4 categorie), generando un report comparativo dettagliato.  
**Dettagli:**
- **Setup:** Selezionare 10 PDF per categoria dalle directory `scaricati/` (o `papers_da_tenere/`):
  - Prompt Injection (10 PDF)
  - Adversarial Code (10 PDF)
  - Data and Model Poisoning (10 PDF)
  - Hallucinations (10 PDF)
- **Metriche da raccogliere per ogni PDF:**
  - Immagini totali estratte (pre-filtro dimensionale)
  - Immagini post-filtro dimensionale (attuale)
  - Immagini post-filtro euristico (Stage 1) → classificazione distribuzione
  - Immagini post-filtro embedding (Stage 2), se abilitato
  - Immagini inviate a Vision (Stage 3) → descrizioni generate
  - Immagini indicizzate in `figure_descriptions` e `image_embeddings`
  - Tempo per fase (estrazione, filtro euristico, embedding, Vision, indicizzazione)
  - Costi stimati (token Vision risparmiati)
- **Report finale** (salvato in `output/image_processing_report.json` e printato con Rich):
  - Tabella per categoria: immagini totali, filtrate per stadio, indicizzate
  - Risparmio percentuale Vision calls: `(1 - vision_calls / totale_immagini) × 100%`
  - Top-5 immagini con score euristico più basso (per validare manualmente la classificazione)
  - Top-5 immagini con embedding similarity più bassa (per validare il filtro embedding)
  - Tempo totale per stadio aggregato
- **Verifica qualitativa:** Per ogni categoria, controllare manualmente 5 immagini filtrate e 5 immagini passate, confermando che il filtraggio sia ragionevole
- Il test deve essere eseguibile standalone: `pytest test_e2e_image_processing.py -v --timeout=600`
- Se Gemini API non è disponibile, skippa con `@pytest.mark.skipif`  
**Test:** Il test è il test stesso. Il criterio di successo è:
  - Risparmio Vision >= 50% (almeno metà delle immagini filtrate prima di Vision)
  - Nessuna figura scientifica evidente mancante nel dataset finale (controllo manuale)
  - Pipeline completa senza crash per tutti i 40 PDF

---

## 📌 Note per le Sessioni Future

- **Una sessione = un task.** Non anticipare mai il codice del task successivo.
- **Ogni task è testabile in isolamento.** Fornire sempre un mini-test alla fine.
- **Aggiornare questo file** mettendo `[x]` al task completato e aggiungendo note sotto il task (come in TODO.md, TODO3.md e TODO4_Scalability.md).
- **Se un task richiede modifiche a task precedenti**, documentare nella sezione note del task.
- **Formato chiusura sessione:** "Task completato con successo. Per favore, apri una nuova chat, incolla il master prompt e il nuovo TODO.md aggiornato, e chiedimi di eseguire il prossimo task. Questo manterrà la mia memoria fresca e il codice pulito."
- **Git commit** dopo ogni task: `git add -A && git commit -m "descrizione della modifica"`

---

## Riepilogo

| ID | Descrizione | Complessità | File principali | Dipendenze |
|----|------------|-------------|-----------------|------------|
| 0.1 | Config image_filtering + collection image_embeddings | Bassa | `config.yaml`, `config.py` | Nessuna |
| 1.1 | Classificatore euristico immagini (PIL + NumPy) | Media | `ingestion/image_relevance_filter.py` (nuovo) | 0.1 |
| 1.2 | Integrazione filtro euristico in FigureExtractor | Media | `ingestion/figure_extractor.py` (modifica) | 1.1 |
| 2.1 | Wrapper Gemini Embedding 2 multimodale | Media | `utils/multimodal_embedder.py` (nuovo) | 0.1 |
| 2.2 | Filtro embedding + integrazione in FigureExtractor | Alta | `ingestion/figure_extractor.py` (modifica) | 2.1, 1.2 |
| 2.3 | Indicizzazione vettori immagine in Qdrant | Media | `indexing/qdrant_store.py`, `models.py`, `figure_extractor.py` | 2.2 |
| 3.1 | Aggiornamento retrieval tool per image embeddings | Bassa | `agents/tools/retrieval_tools.py` (modifica) | 2.3 |
| 3.2 | Test end-to-end con 10 PDF per sezione | Alta | `test_e2e_image_processing.py` (nuovo) | Tutti |

---

> **Flusso di dipendenze:**
> - Task 0.1 → (FASE 1 e Task 2.1 in parallelo)
> - FASE 1: 1.1 → 1.2
> - FASE 2: 2.1 → 2.2 (dipende anche da 1.2) → 2.3
> - FASE 3: 3.1 (dipende da 2.3) → 3.2 (dipende da tutti)
>
> ```
> 0.1 ──┬── 1.1 → 1.2 ──┐
>        │                 ├── 2.2 → 2.3 → 3.1 → 3.2
>        └── 2.1 ─────────┘
> ```
