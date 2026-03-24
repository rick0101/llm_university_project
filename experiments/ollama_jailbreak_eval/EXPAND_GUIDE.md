# 🔧 Guida per Espandere il Framework di Valutazione

## Quando usare questa guida

Quando hai un database PaperHunter più grande con nuovi attacchi/difese da testare,
copia questo prompt e dallo a Copilot (o qualsiasi LLM assistant).

---

## Master Prompt

```
Ho il framework di jailbreak evaluation in:
experiments/ollama_jailbreak_eval/

Struttura attuale:
- attacks.py    → 13 attacchi (funzioni che wrappano un prompt)
- defenses.py   → 5 difese (funzioni che chiamano Ollama con strategie diverse)
- prompts.py    → 15 prompt harmful in 7 categorie
- judge.py      → 30 pattern regex di refusal detection
- config.py     → configurazione Ollama (host, modello, timeout)
- run_evaluation.py  → runner che testa prompt × attacchi × difese
- generate_charts.py → genera 5 grafici matplotlib dai risultati

Il Knowledge Graph di PaperHunter è in:
data/property_graph/graph_store.json

Formato nodi: {name: {label, properties: {entity_type, description, aliases, source_bibtex_key, year}}}
entity_type può essere: Attack, Defense, Model, Metric, Dataset, Benchmark

TASK:
1. Leggi il KG e confronta le entità Attack/Defense con quelle già in attacks.py e defenses.py
2. Identifica quelle NUOVE che sono implementabili in black-box (solo API chat Ollama, no gradient, no fine-tuning, no white-box)
3. Per ogni nuovo attacco: crea una funzione in attacks.py seguendo il pattern esistente:
   - def nome_attack(prompt: str) -> str  →  ritorna il prompt wrappato
   - Aggiungi al dizionario ATTACKS in fondo
4. Per ogni nuova difesa: crea una funzione in defenses.py seguendo il pattern esistente:
   - def nome_defense(prompt: str) -> str  →  chiama _ollama_chat() e ritorna la risposta
   - Aggiungi al dizionario DEFENSES in fondo
5. Se ci sono nuove categorie di attacco nel KG, aggiungi prompt corrispondenti a prompts.py
6. Se servono nuovi pattern di refusal, aggiungili a judge.py
7. NON toccare run_evaluation.py e generate_charts.py — leggono dai dizionari automaticamente
8. Verifica con: python run_evaluation.py --dry-run

VINCOLI:
- Ogni attacco deve essere testabile via chat API (black-box only)
- Le difese possono fare max 2-3 chiamate Ollama per test (non di più, o sarà troppo lento)
- Mantieni i docstring con source paper/KG entity di riferimento
- Non duplicare attacchi/difese già presenti (controlla il dizionario ATTACKS/DEFENSES)
```

---

## Cosa viene aggiornato automaticamente

| File | Si adatta? | Note |
|------|-----------|-------|
| `run_evaluation.py` | ✅ Sì | Legge `ATTACKS`, `DEFENSES`, `HARMFUL_PROMPTS` dai dizionari |
| `generate_charts.py` | ✅ Sì | Genera grafici da qualsiasi JSON di risultati |
| `attacks.py` | ❌ Manuale | Aggiungere funzioni + entry nel dizionario `ATTACKS` |
| `defenses.py` | ❌ Manuale | Aggiungere funzioni + entry nel dizionario `DEFENSES` |
| `prompts.py` | ❌ Manuale | Aggiungere dict con id/category/prompt |
| `judge.py` | ❌ Manuale | Aggiungere pattern regex a `REFUSAL_PATTERNS` |
| `config.py` | ❌ Manuale | Cambiare host/modello/timeout se serve |

## Pattern da seguire

### Nuovo attacco
```python
def nuovo_attack(prompt: str) -> str:
    """Descrizione — tecnica usata.
    Source: NomeEntitàKG (Autore et al. Anno)."""
    return (
        "Testo del template di jailbreak...\n"
        f"...qui il prompt: {prompt}\n"
        "...fine template"
    )

# Aggiungere in ATTACKS:
#   "Nuovo-Nome": nuovo_attack,
```

### Nuova difesa
```python
def nuova_defense(prompt: str) -> str:
    """Descrizione della strategia difensiva.
    Source: NomeEntitàKG (Autore et al. Anno)."""
    # Opzionale: pre-processing (1 call)
    # Risposta finale (1 call)
    return _ollama_chat([
        {"role": "system", "content": "System prompt difensivo..."},
        {"role": "user", "content": prompt},
    ])

# Aggiungere in DEFENSES:
#   "Nuovo-Nome": nuova_defense,
```

### Nuovo prompt
```python
# Aggiungere in HARMFUL_PROMPTS:
{
    "id": "cat-N",
    "category": "NuovaCategoria",
    "prompt": "Il testo del prompt harmful...",
},
```

## Stima tempi di esecuzione

| Matrice | Test totali | ~Tempo (gemma3:4b, 6s/req) |
|---------|-------------|---------------------------|
| 15×13×5 | 975 | ~1h 40min |
| 20×15×5 | 1500 | ~2h 30min |
| 25×15×7 | 2625 | ~4h 20min |
| 30×20×7 | 4200 | ~7h |

> Le difese Paraphrase, Few-Shot e Perplexity-Filter fanno 2+ chiamate,
> quindi il tempo reale è ~1.5x la stima sopra.
