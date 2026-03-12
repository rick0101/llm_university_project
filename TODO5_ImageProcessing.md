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

#### [x] Task 0.1: Aggiornare dipendenze e configurazioni per image filtering

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

> **Completato:** 12 Marzo 2026  
> **Modifiche apportate:**
> - **`config.yaml`:** Aggiunta sezione `image_filtering` con tutti i 10 campi configurabili (enabled, min/max_aspect_ratio, min_color_entropy, min_edge_density, skip_first/last_page, relevance_threshold, embedding_filter_enabled, embedding_similarity_threshold). Aggiunto `collection_image_embeddings: "image_embeddings"` nella sezione `qdrant`.
> - **`paperhunter/config.py`:** Creato `ImageFilteringConfig(BaseModel)` con 10 campi e 6 validatori Pydantic (`min_aspect_ratio ∈ (0.0, 1.0]`, `max_aspect_ratio ∈ [1.0, 20.0]`, `min_color_entropy ∈ [0.0, 8.0]`, `min_edge_density ∈ [0.0, 1.0]`, `relevance_threshold ∈ [0.0, 1.0]`, `embedding_similarity_threshold ∈ [0.0, 1.0]`). Registrato in `Settings` come `image_filtering: ImageFilteringConfig`. Aggiunto `collection_image_embeddings: str = "image_embeddings"` a `QdrantConfig`.
> - **Uso:** `from paperhunter.config import settings; settings.image_filtering.enabled`, `settings.image_filtering.relevance_threshold`, `settings.qdrant.collection_image_embeddings`
> - **Validatori testati:** Tutti i 6 validatori rigettano correttamente valori fuori range con `ValidationError`.

---

### FASE 1 — Filtraggio Euristico delle Immagini (Senza LLM)

#### [x] Task 1.1: Creare `ingestion/image_relevance_filter.py`

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

> **Completato:** 12 Marzo 2026  
> **Modifiche apportate:**
> - **`paperhunter/ingestion/image_relevance_filter.py` (nuovo):** Creato modulo completo con:
>   - `ImageClassification` enum: `SCIENTIFIC_FIGURE`, `LIKELY_DECORATIVE`, `UNCERTAIN`
>   - `ImageScore` dataclass: 5 punteggi individuali + `combined_score` + `classification`
>   - `ImageRelevanceFilter` classe principale:
>     - `classify(image_bytes, page_number, total_pages, page_width, page_height) -> ImageScore`
>     - `classify_batch(images: list[dict]) -> list[ImageScore]` con logging Rich tabellare
>     - 5 metodi euristici privati: `_score_aspect_ratio`, `_score_color_entropy`, `_score_edge_density`, `_score_page_position`, `_score_relative_size`
>   - Pesi di default: edge_density 0.30, entropy 0.25, aspect_ratio 0.15, position 0.15, size 0.15
>   - Config da `settings.image_filtering` (Task 0.1), soglia configurabile
> - **`test_image_relevance_filter.py` (nuovo):** 26 test con immagini sintetiche PIL:
>   - `TestHeuristics`: 14 test per singole euristiche (aspect ratio, entropy, edge density, position, size)
>   - `TestClassification`: 5 test end-to-end (grafico → SCIENTIFIC, tabella → SCIENTIFIC, logo → DECORATIVE/UNCERTAIN, banner → DECORATIVE, foto → UNCERTAIN)
>   - `TestBatch`: 1 test batch processing
>   - `TestEdgeCases`: 6 test robustezza (bytes invalidi, pagina singola, config custom, dimensioni zero, greyscale, range score)
> - **Uso:** `from paperhunter.ingestion.image_relevance_filter import ImageRelevanceFilter, ImageClassification; filt = ImageRelevanceFilter(); score = filt.classify(img_bytes, page_number=5, total_pages=12, page_width=612.0, page_height=792.0)`
> - **Nota per Stage 2:** Il filtro euristico cattura efficacemente banner, icone, loghi piccoli e immagini su prima/ultima pagina. Foto decorative con dimensioni normali su pagine centrali richiedono lo Stage 2 (embedding) per essere filtrate.

---

#### [x] Task 1.2: Integrare il filtro euristico in `figure_extractor.py`

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

> **Completato:** 12 Marzo 2026  
> **Modifiche apportate:**
> - **`paperhunter/ingestion/figure_extractor.py` (modifica):**
>   - Aggiunto import di `ImageClassification` e `ImageRelevanceFilter` da `image_relevance_filter`
>   - **`extract_images_from_pdf()`:** Ogni immagine estratta ora include `total_pages` (da `len(doc)`), `page_width` e `page_height` (da `page.rect` di PyMuPDF) nel dict restituito
>   - **`extract_and_describe()`:** Aggiunto Step 1.5 dopo l'estrazione, prima della descrizione Vision:
>     - Se `settings.image_filtering.enabled` è `True`: istanzia `ImageRelevanceFilter()`, chiama `classify_batch()`, filtra le immagini `LIKELY_DECORATIVE`
>     - Le immagini `UNCERTAIN` e `SCIENTIFIC_FIGURE` passano a Vision (beneficio del dubbio)
>     - Ogni immagine riceve campo `relevance_score: float` (o `None` se filtro disabilitato)
>     - Per ogni immagine filtrata, logga il motivo (bassa densità bordi, bassa entropia, prima/ultima pagina, aspect ratio estremo, dimensione inadeguata)
>     - Logga statistiche: `Filtro euristico 'file.pdf': Estratte=N, Filtrate(decorative)=M, Passate a Vision=K`
>     - Se tutte le immagini vengono filtrate, restituisce lista vuota senza chiamare Vision
>   - Backward compatibility: se `settings.image_filtering.enabled = False`, nessun filtraggio (comportamento identico a prima)
> - **`test_figure_extractor.py` (modifica):** Aggiunto `mock_settings_mod.image_filtering.enabled = False` ai 4 test di `TestExtractAndDescribe` per evitare interferenze del filtro nei test preesistenti
> - **`test_figure_extractor_filtering.py` (nuovo):** 11 test dedicati all'integrazione del filtro:
>   - `TestExtractImagesNewFields` (2 test): verifica `total_pages`, `page_width`, `page_height` nel dict
>   - `TestHeuristicFilterIntegration` (8 test): decorative scartate, filtro disabilitato, UNCERTAIN passa, tutto decorativo → vuoto, relevance_score aggiunto, Vision calls ridotte, classificazioni miste, test end-to-end con filtro reale
>   - `TestFilteringLogging` (1 test): verifica logging statistiche via caplog
> - **Tutti i test passano:** 15 (test_figure_extractor.py) + 11 (test_figure_extractor_filtering.py) + 26 (test_image_relevance_filter.py) = 52 test OK
> - **Uso:** Il filtro è automatico e trasparente. `FigureExtractor().extract_and_describe(pdf, paper)` ora filtra le immagini decorative prima di chiamare Gemini Vision. Per disabilitare: `settings.image_filtering.enabled = False` nel config.yaml

---

### FASE 2 — Embedding Multimodale (Gemini Embedding 2 Preview)

#### [x] Task 2.1: Creare `utils/multimodal_embedder.py`

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

> **Completato:** 12 Marzo 2026  
> **Modifiche apportate:**
> - **`paperhunter/utils/multimodal_embedder.py` (nuovo):** Creato modulo completo con:
>   - `MultimodalEmbedder` classe principale:
>     - `__init__(model: str | None = None)`: usa `settings.gemini.embedding_model` di default, crea client `google-genai` via `_get_api_key()` e `_get_http_options()` da `llm_factory`, rate limiter da `get_gemini_limiter()`
>     - `embed_image(image_bytes, mime_type) -> list[float]`: embedding multimodale via `types.Part.from_bytes()` + `client.models.embed_content()`
>     - `embed_text(text) -> list[float]`: embedding testuale via stesso SDK (stesso spazio vettoriale)
>     - `embed_images_batch(images, show_progress) -> list[list[float]]`: batch processing con Rich progress bar
>     - `compute_reference_embeddings() -> dict[str, list[float]]`: calcolo + cache (memoria + disco in `cache/reference_embeddings.json`) per 5 categorie scientifiche
>     - `classify_image_relevance(image_bytes, mime_type, reference_embeddings) -> tuple[float, str]`: classificazione via similarità coseno vs riferimenti
>   - `_cosine_similarity(a, b)` helper per calcolo similarità coseno
>   - `get_multimodal_embedder() -> MultimodalEmbedder` singleton lazy
>   - 5 categorie di riferimento: `scientific_plot`, `comparison_table`, `architecture_diagram`, `flowchart`, `confusion_matrix`
> - **`test_multimodal_embedder.py` (nuovo):** 30 test con mock completi:
>   - `TestEmbedImage` (3 test): vettore 3072-dim, rate limiter, mime type
>   - `TestEmbedText` (2 test): vettore 3072-dim, contenuto stringa
>   - `TestBatchEmbedding` (4 test): lista vuota, conteggio corretto, progress bar, mime default
>   - `TestReferenceEmbeddings` (5 test): tutte le categorie, cache in memoria, cache su disco, load da disco, cache corrotta → ricalcolo
>   - `TestClassifyImageRelevance` (5 test): grafico vs logo, tipo ritorno, riferimenti vuoti, compute automatico, range validità
>   - `TestCosineSimilarity` (5 test): identici, ortogonali, opposti, vettore zero, alta dimensionalità
>   - `TestSingleton` (2 test): stessa istanza, tipo corretto
>   - `TestEdgeCases` (4 test): JPEG, modello custom, modello da settings, cache parziale → ricalcolo
> - **Tutti i test passano:** 30 (test_multimodal_embedder.py) + 52 test preesistenti (FASE 0+1) = 82 test OK
> - **Uso:** `from paperhunter.utils.multimodal_embedder import MultimodalEmbedder, get_multimodal_embedder; emb = get_multimodal_embedder(); vec = emb.embed_image(img_bytes); sim, cat = emb.classify_image_relevance(img_bytes, "image/png")`
> - **Nota per Task 2.2:** Il `MultimodalEmbedder` è pronto per essere integrato in `figure_extractor.py`. `classify_image_relevance()` restituisce `(similarity, category)` che il filtro Stage 2 userà per decidere se scartare l'immagine. L'embedding pre-calcolato può essere riutilizzato per l'indicizzazione (Task 2.3) senza ricalcolo.

---

#### [x] Task 2.2: Classificazione rilevanza via embedding e integrazione in FigureExtractor

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

> **Completato:** 12 Marzo 2026  
> **Modifiche apportate:**
> - **`paperhunter/ingestion/figure_extractor.py` (modifica):**
>   - Aggiunto import di `_cosine_similarity` e `get_multimodal_embedder` da `utils.multimodal_embedder`
>   - **`extract_and_describe()`:** Aggiunto Step 1.7 (Stage 2) dopo il filtro euristico (Stage 1), prima della descrizione Vision:
>     - Se `settings.image_filtering.enabled` e `settings.image_filtering.embedding_filter_enabled` sono `True`: istanzia `MultimodalEmbedder` via `get_multimodal_embedder()`, computa `reference_embeddings`
>     - **Ottimizzazione:** Immagini `SCIENTIFIC_FIGURE` con score euristico > 0.8 saltano Stage 2 e vanno direttamente a Vision (risparmio embedding API per immagini già chiaramente rilevanti)
>     - Per le restanti immagini: embed una sola volta con `embed_image()`, calcola similarità coseno vs tutti i riferimenti manualmente (evita doppia chiamata API)
>     - Se `similarity >= embedding_similarity_threshold` → passa a Vision; altrimenti → scartata
>     - Ogni immagine riceve 3 nuovi campi: `embedding_similarity: float | None`, `best_match_category: str | None`, `embedding: list[float] | None`
>     - L'embedding pre-calcolato viene preservato nel dict per riuso nel Task 2.3 (indicizzazione senza ricalcolo)
>     - Logga statistiche complete: `Stage 1: N → K immagini; Stage 2: K → J immagini; Vision calls: J`
>     - Per ogni immagine, logga motivo del PASS o FILTRAGGIO con similarità e categoria
>   - Backward compatibility: se `embedding_filter_enabled = False`, Stage 2 saltato (campi embedding impostati a None via `setdefault`)
>   - Se `image_filtering.enabled = False`, entrambi gli stage saltati (comportamento identico a prima)
> - **`test_figure_extractor_filtering.py` (modifica):** Aggiunto `mock_settings_mod.image_filtering.embedding_filter_enabled = False` a tutti gli 8 test di Stage 1 per evitare interferenze del filtro embedding nei test preesistenti
> - **`test_embedding_filter_integration.py` (nuovo):** 16 test dedicati all'integrazione Stage 2:
>   - `TestStage2EmbeddingFilter` (5 test): UNCERTAIN filtrate per bassa sim, SCIENTIFIC alta salta Stage 2, SCIENTIFIC bassa va a Stage 2, DECORATIVE non embedded, tutto filtrato → vuoto
>   - `TestStage2Disabled` (2 test): embedding_filter_enabled=False, entrambi stage disabilitati
>   - `TestStage2EmbeddingFields` (2 test): campi popolati per immagini passate e filtrate
>   - `TestStage2MixedPipeline` (2 test): pipeline completa con 5 immagini miste, riduzione Vision significativa (10→3 = 70%)
>   - `TestStage2Logging` (2 test): statistiche Stage 1/2 loggate, conteggi numericamente corretti
>   - `TestStage2EdgeCases` (3 test): niente dopo Stage 1, relevance_score None quando disabilitato, boundary esatto soglia
> - **Tutti i test passano:** 16 (test_embedding_filter_integration.py) + 11 (test_figure_extractor_filtering.py) + 15 (test_figure_extractor.py) + 26 (test_image_relevance_filter.py) + 30 (test_multimodal_embedder.py) = 98 test OK
> - **Uso:** Il filtro Stage 2 è automatico e trasparente. `FigureExtractor().extract_and_describe(pdf, paper)` ora filtra prima con euristiche (Stage 1), poi con embedding multimodale (Stage 2), poi descrizione Vision (Stage 3). Per disabilitare Stage 2: `settings.image_filtering.embedding_filter_enabled = False` nel config.yaml
> - **Nota per Task 2.3:** L'embedding pre-calcolato è disponibile in `img_info["embedding"]` (list[float] 3072-dim) per le immagini che passano Stage 2. Le immagini con score euristico > 0.8 che saltano Stage 2 hanno `embedding = None` e andranno ri-embedded nel Task 2.3 se si vuole indicizzarle.

---

#### [x] Task 2.3: Indicizzazione diretta immagini come vettori multimodali in Qdrant

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

> **Completato:** 13 Marzo 2026  
> **Modifiche apportate:**
> - **`paperhunter/models.py`:** Aggiunto `IMAGE_EMBEDDING = "image_embedding"` a `ChunkType` enum.
> - **`paperhunter/indexing/qdrant_store.py`:** Aggiunto `COLLECTION_IMAGE_EMBEDDINGS` class attribute (caricato da `cfg.collection_image_embeddings`). Registrato in `_ensure_collections()` (7 collection totali) e in `get_stats()`. Aggiunti metodi `add_image_embeddings(chunks, embeddings, batch_size=50) -> int` e `query_image_embeddings(query_text, n_results, where_filter, score_threshold) -> list[dict]`, entrambi delegano alle funzioni `add_chunks_with_embeddings` e `query` sulla collection dedicata.
> - **`paperhunter/ingestion/figure_extractor.py`:** Aggiunto Step 5 "Preparazione chunk IMAGE_EMBEDDING" in `extract_and_describe()`: per ogni immagine rilevante, crea `PaperChunk` con `chunk_type=ChunkType.IMAGE_EMBEDDING`. Riusa embedding pre-calcolato da Stage 2; per immagini che hanno saltato Stage 2 (score euristico > 0.8, embedding=None), chiama `embedder.embed_image()`. Risultati accumulati in `self._pending_image_chunks` e `self._pending_image_embeddings`. Nuovo metodo `get_pending_image_embeddings() -> tuple[list[PaperChunk], list[list[float]]]` che restituisce e svuota lo stato interno (previene doppia indicizzazione). Return type di `extract_and_describe()` invariato per backward compatibility.
> - **`paperhunter/ingestion/ingestion_pipeline.py`:** Modificato `_step_figure_extraction()`: dopo `extract_and_describe()`, chiama `extractor.get_pending_image_embeddings()` e `store.add_image_embeddings()` per indicizzare gli embedding multimodali.
> - **`test_image_embedding_indexing.py`** (nuovo): 24 test in 5 classi — `TestChunkTypeImageEmbedding` (3), `TestQdrantStoreImageEmbeddings` (7), `TestGetPendingImageEmbeddings` (9), `TestImageEmbeddingChunkMetadata` (2), `TestEdgeCases` (2). Tutti 24 passano.
> - **`test_embedding_filter_integration.py`** (aggiornato): Adattati 3 test esistenti per compatibilità con Step 5 (`test_scientific_high_score_skips_stage2`, `test_decorative_not_embedded`, `test_full_pipeline_mixed_images`). Tutti 16 test passano.
> - **Test totali image-related:** 106 passano (24 nuovi + 82 pre-esistenti), 0 falliti.
> - **Uso:** Ogni immagine rilevante produce due entry: (1) `figure_descriptions` con embedding testuale della descrizione Vision, (2) `image_embeddings` con embedding multimodale dell'immagine stessa. Query cross-modale: `store.query_image_embeddings("attack success rate", n_results=5)`.

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
