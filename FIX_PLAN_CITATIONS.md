# Piano di Fix — Errori Citazioni Pipeline Writer

**Data**: 9 aprile 2026  
**Obiettivo**: Correggere i problemi di metadati BibTeX (abstract errati, autori hallucinated, venue/tipo sbagliato, eprint mancante) identificati nel `citation_verification_report.md`.

---

## Diagnosi: Cause Radice

I 6 errori trovati nel report di verifica **non** derivano dal writer agent stesso, ma da **problemi a monte nella pipeline di ingestion**. Il flusso è:

```
PDF → GROBID → Semantic Scholar API → MetadataEnricher → BibTeX Generator → references.bib
```

### Cause identificate (per tipo di errore):

| Errore | File responsabile | Causa |
|---|---|---|
| **Abstract errato** (Kumar: abstract di AutoDAN copiato su AmpleGCG) | `metadata_enricher.py` L305-330 | S2 API restituisce abstract sbagliato; il sistema lo accetta ciecamente (confidence 0.15) senza validare che l'abstract corrisponda al titolo |
| **Abstract fabbricato** (Nie: abstract generico non reale) | `metadata_enricher.py` L340-345 | Fallback a primo paragrafo full_text da GROBID — può prendere testo arbitrario |
| **Autori hallucinated** (Muntasir: "Qwen2", "Geminiteam", nomi di modelli AI) | `grobid_client.py` L409-424 | GROBID estrae nomi dalla sezione references del PDF e li mescola. Nessuna validazione sui nomi autori |
| **Affiliazione come autore** (Paudel: "Aimon Labs"; Wang: "Microsoft") | `grobid_client.py` / `semantic_scholar.py` | GROBID/S2 non separano affiliazione da nome. Nessun filtro |
| **Nome spezzato** (Urlana: "Charaka Vinayak Kumar" → 2 autori separati) | `grobid_client.py` L409 | `" and "` nel formato BibTeX divide in modo errato nomi con segmenti multipli |
| **Tipo entry/venue errato** (Kumar: @inproceedings{ICLR} su un preprint) | `bibtex_generator.py` L295-326 | S2 restituisce "ICLR" come venue (forse accepted/spotlight) ma il paper è solo su arXiv. `_is_conference_venue()` lo marca come conferenza |
| **eprint mancante** (Kumar, Nie) | `bibtex_generator.py` L397-400 | Se `Paper.arxiv_id` è None (S2 non l'ha trovato), il campo eprint non viene generato |

---

## Piano di Implementazione

> **Criterio guida**: tutte le correzioni vengono applicate **a livello di codice** nella pipeline di ingestion, così il survey può essere rigenerato con metadati corretti. Nessuna modifica manuale al .bib. Tutte le validazioni usano **espressioni regolari ed euristiche** (zero chiamate LLM aggiuntive).

---

### FASE 1: Validazione autori nel BibTeX generator
> Filtrare nomi autori palesemente invalidi. **Modifiche al codice.**

#### 2.1 — Aggiungere validazione nomi autori
- **File**: `paperhunter/ingestion/bibtex_generator.py`
- **Dove**: Nella funzione `_format_authors_bibtex()` (riga ~178)
- **Cosa aggiungere**: Prima di formattare, filtrare autori con nomi invalidi:
  ```python
  _INVALID_AUTHOR_PATTERNS = [
      r'^(Qwen|Gemini|GPT|Claude|Llama|Mistral)',   # Nomi di modelli AI
      r'^(Efficient|Ving|Binyuan)$',                  # Token isolati sospetti
      r'(Labs?|Inc|Corp|University|Institute)\b',     # Affiliazioni
      r'^\w{1,2}$',                                    # Nomi troppo corti (1-2 char)
  ]
  
  def _is_valid_author_name(name: str) -> bool:
      """Filtra nomi autori palesemente non validi."""
      if not name or len(name.strip()) < 3:
          return False
      for pattern in _INVALID_AUTHOR_PATTERNS:
          if re.search(pattern, name.strip(), re.IGNORECASE):
              return False
      return True
  ```

#### 2.2 — Limitare il numero massimo di autori
- **File**: `paperhunter/ingestion/bibtex_generator.py`
- **Dove**: In `_format_authors_bibtex()`
- **Cosa**: Se `len(authors) > 15`, è quasi certamente corrotto (la lista di Muntasir aveva 50+ nomi). Troncare a 15 e loggare un warning.

---

### FASE 2: Validazione abstract nel MetadataEnricher
> Evitare che un abstract di un altro paper venga assegnato. Soluzione completamente euristica (keyword overlap), nessuna chiamata LLM.

#### 2.1 — Cross-check abstract ↔ titolo
- **File**: `paperhunter/ingestion/metadata_enricher.py`
- **Dove**: In `_resolve_abstract()` (riga ~305)
- **Cosa aggiungere**: Dopo aver ottenuto l'abstract da S2, verificare che almeno qualche keyword del titolo appaia nell'abstract. Se il titolo contiene un acronimo chiave (es. "AmpleGCG-Plus") e l'abstract non lo menziona, declassare il confidence e provare il fallback GROBID.
  ```python
  def _abstract_matches_title(title: str, abstract: str) -> bool:
      """Verifica che l'abstract sia coerente con il titolo del paper."""
      title_words = set(re.findall(r'[A-Z][a-z]+|[A-Z]{2,}', title))
      significant = {w for w in title_words if len(w) >= 4 and w.lower() not in STOP_WORDS}
      if not significant:
          return True  # Niente da controllare
      matches = sum(1 for w in significant if w.lower() in abstract.lower())
      return matches / len(significant) >= 0.2  # Almeno 20% overlap
  ```

---

### FASE 3: Fix tipo entry (preprint vs. conferenza)
> Evitare che preprint arXiv vengano marcati come @inproceedings.

#### 3.1 — Cross-check venue ↔ arxiv_id nel bibtex_generator
- **File**: `paperhunter/ingestion/bibtex_generator.py`
- **Dove**: In `_build_bibtex_from_scratch()` (riga ~295)
- **Cosa aggiungere**: Se il paper ha `arxiv_id` ma **non** ha `doi`, è probabilmente un preprint anche se S2 riporta un venue. In questo caso, usare `@misc` con `howpublished = {arXiv preprint}` e non `@inproceedings`.
  ```python
  # Se ha arxiv_id ma non DOI → è un preprint, non pubblicato in conferenza
  if paper.arxiv_id and not paper.doi:
      entry_type = "misc"
      # Non usare booktitle, anche se S2 lo suggerisce
  ```
- **Nota**: La funzione `ensure_complete_bibtex()` (riga ~475) ha già una logica simile con `is_arxiv`, ma interviene troppo tardi. Va spostata prima, in `_build_bibtex_from_scratch()`.

---

### FASE 4: Validazione eprint/arxiv_id
> Garantire che ogni paper arXiv abbia l'eprint.

#### 4.1 — Estrazione arxiv_id da URL come fallback
- **File**: `paperhunter/ingestion/bibtex_generator.py`
- **Dove**: In `_build_bibtex_from_scratch()` (riga ~397)
- **Cosa aggiungere**: Se `paper.arxiv_id` è None ma `paper.url` contiene `arxiv.org/abs/XXXX.YYYYY`, estrarre l'ID dall'URL.
  ```python
  arxiv_id = paper.arxiv_id
  if not arxiv_id and paper.url:
      m = re.search(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', paper.url)
      if m:
          arxiv_id = m.group(1)
  ```

#### 4.2 — Validazione formato eprint
- **File**: `paperhunter/ingestion/bibtex_generator.py`
- **Cosa**: Aggiungere validazione che l'eprint sia nel formato `\d{4}\.\d{4,5}(v\d+)?`

---

### FASE 5: Fix estrazione autori da GROBID
> Isolare il punto dove GROBID mescola affiliazioni con autori. Pura analisi strutturale del TEI-XML, nessuna chiamata LLM.

#### 5.1 — Filtrare affiliazioni dalla lista autori GROBID
- **File**: `paperhunter/ingestion/grobid_client.py`
- **Dove**: In `_extract_authors()` (riga ~409-424)
- **Cosa aggiungere**: Dopo aver estratto la lista raw, applicare la stessa validazione di FASE 1.1. Inoltre, verificare che gli elementi `<affiliation>` non vengano confusi con `<persName>`.

#### 5.2 — Verificare estrazione da `<orgName>` vs `<persName>`
- **File**: `paperhunter/ingestion/grobid_client.py`
- **Cosa**: Assicurarsi che il parser TEI-XML ignori tag `<orgName>` e `<affiliation>` all'interno di `<author>`. Attualmente sembra che vengano catturati testi da nodi figli non previsti.

---

### FASE 6: Post-processing validation hook
> Safety net finale aggiunto a `save_master_bib()`, senza chiamate LLM.

#### 6.1 — Aggiungere `_warn_suspicious_entries()` in `bibtex_generator.py`
- **File**: `paperhunter/ingestion/bibtex_generator.py`
- **Dove**: Chiamata dentro `save_master_bib()` prima della scrittura su disco
- **Cosa**: Loga warning per entry con:
  - Numero autori > 15
  - Nomi autori che matchano pattern sospetti (modelli AI, affiliazioni)
  - @inproceedings senza DOI
  - arxiv_id presente ma eprint mancante
  - Abstract assente o < 50 char
  - Anno fuori range
- Output: solo log/warning, **non blocca** la pipeline

---

## Priorità e Ordine di Esecuzione

> Tutte le modifiche sono **solo nel codice**. Il .bib viene corretto rigenerando la pipeline. Zero chiamate LLM aggiuntive.

| Priorità | Fase | Impatto | Sforzo |
|---|---|---|---|
| 🔴 P0 | **FASE 1**: Validazione autori (bibtex_generator) | Filtra autori hallucinated/affiliazioni | Basso |
| 🔴 P0 | **FASE 2**: Cross-check abstract ↔ titolo (metadata_enricher) | Previene abstract rubati da altro paper | Basso |
| 🟠 P1 | **FASE 3**: Fix tipo entry preprint vs conferenza | Previene @inproceedings errati | Basso |
| 🟠 P1 | **FASE 4**: Estrazione/validazione eprint da URL | Previene eprint mancanti | Basso |
| 🟡 P2 | **FASE 5**: Fix GROBID author extraction (TEI-XML) | Fix causa radice affiliazioni | Medio |
| 🟢 P3 | **FASE 6**: Warning hook in save_master_bib | Safety net osservabilità | Basso |
| 🟢 P3 | **FASE 7**: Post-processing validation | Safety net finale | Medio |

---

## File Coinvolti (Riepilogo)

| File | Fasi |
|---|---|
| `paperhunter/ingestion/bibtex_generator.py` | 1, 3, 4, 6 |
| `paperhunter/ingestion/metadata_enricher.py` | 2 |
| `paperhunter/ingestion/grobid_client.py` | 5 |

---

## Rischi e Note

- **FASE 2**: Il cross-check abstract ↔ titolo può avere falsi positivi su titoli molto generici. Usare soglia bassa (20% overlap) e in caso di mismatch fare fallback (non scartare l'abstract).
- **FASE 3**: Alcuni paper arXiv vengono successivamente pubblicati in conferenza. La logica `arxiv_id + no DOI = preprint` funziona per la maggioranza dei casi ma potrebbe sbagliare per paper appena accettati. Loggare un warning invece di forzare.
- **FASE 5**: GROBID è un servizio esterno; il fix è nel parsing del suo output TEI-XML, non nel servizio stesso.
- **Nessuna fase aggiunge chiamate LLM**: tutte le validazioni usano regex, euristiche e confronto di stringhe.

---

## Come procedere

Nella prossima sessione, implementare in ordine:
1. **FASI 1 + 2** insieme — entrambe modificano file diversi, basso rischio, alto impatto
2. **FASI 3 + 4** insieme — entrambe su `bibtex_generator.py`, conviene farle in un'unica passata
3. **FASE 5** — richiede ispezione del parsing TEI-XML di GROBID e test
4. **FASE 6** — warning hook, da aggiungere per ultimo

Dopo l'implementazione: **rieseguire la pipeline di ingestion** sui paper problematici per rigenerare i metadati corretti, poi rigenerare il survey.
