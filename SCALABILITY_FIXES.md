# PaperHunter — Scalability Fixes per 400+ Paper

## Il Problema

PaperHunter deve generare un survey scientifico di alta qualità partendo da **~400 PDF**.
Con il dataset attuale, emergono **5 problemi critici** che degradano qualità e performance:

1. **Context overflow nel percorso L2 legacy** — 100 L1/categoria × 2k token = 200k token, sfora il budget 150k
2. **Nessun filtraggio rilevanza nei chunk** — L1 summarizer manda tutto (appendici, ringraziamenti...) al LLM
3. **Writer riceve tutte le categorie** — spreca ~25% del budget su L2 irrilevanti per la sezione corrente
4. **Entity extraction non batchata** — 400 chiamate LLM serializzate (1 per paper), ~27 min di attesa
5. **Embedding batch troppo piccoli** — batch da 10 chunk → 1200 chiamate API → ~80 min solo per embedding

**Obiettivo:** Ridurre chiamate API del ~30%, eliminare context rot, dimezzare il tempo pipeline.

---

## Fix Tracker

- [x] **FIX 1 — L1.5 RAPTOR Obbligatorio**: Rendere il percorso L1.5 (clustering RAPTOR) obbligatorio per L2 generation. Se L1.5 non è disponibile, generarlo automaticamente prima di procedere con L2. Elimina il rischio di context overflow nel legacy path. → **Implementato**: `level2_summarizer.py` ora auto-genera L1.5 quando >50 paper/categoria e L1.5 mancanti. Il legacy path resta come fallback per categorie piccole (≤50 paper).

- [ ] **FIX 2 — Chunk Relevance Filter per L1**: Aggiungere filtro TF-IDF/keyword pre-LLM in `_prepare_chunks_text()`. Invece di concatenare tutti i chunk in ordine, calcolare rilevanza rispetto al titolo+abstract e mandare solo i top-K più informativi. Risparmio: ~30-40% token input per chiamata L1.

- [ ] **FIX 3 — Category-Aware Writer Context**: Filtrare gli L2 summary passati al Writer in base alla categoria della sezione corrente. Il Planner già specifica `key_topics` per sezione — usarli per selezionare solo gli L2 pertinenti invece di passarli tutti.

- [ ] **FIX 4 — Batch Entity Extraction**: Raggruppare 3-5 L1 summary per singola chiamata di entity extraction nel GraphRAG. Riduce 400 → ~80-130 chiamate LLM. Richiede modifica del prompt template per accettare multipli paper.

- [ ] **FIX 5 — Embedding Batch Size Aumentato**: Aumentare il batch size degli embedding da 10 a 100 in `qdrant_store.py`. Riduce 1200 → 120 chiamate API embedding. Verificare compatibilità con Gemini Embedding API.

---

## Risultati Attesi

| Metrica | Prima | Dopo | Δ |
|---------|-------|------|---|
| Chiamate Pro | ~520 | ~400 | -23% |
| Chiamate Flash | ~420 | ~200 | -52% |
| Embedding batch calls | 1,200 | ~120 | -90% |
| Tempo pipeline | ~3-3.5h | ~1.5-2h | -45% |
| Context rot risk | Alto (legacy L2) | Eliminato | ✓ |
| Testing methodology | Troncato @150ch | Preservato intero | ✓ |
