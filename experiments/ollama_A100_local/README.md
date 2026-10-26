# Ollama Jailbreak Eval — Self-Contained / SLURM-ready

Cartella **autonoma**: tutto quello che serve per eseguire la valutazione
jailbreak (19 prompt × 31 attacchi × 11 difese = 6.479 test) su uno o più
modelli Ollama che girano in locale su una **GPU A100 80 GB**, su un nodo
SLURM, senza alcuna API key esterna.

Il modello sotto test gira in locale su Ollama. Il **modello giudice è
lo stesso modello sotto test** (chiamata separata post-generazione, stessa
istanza già caricata in VRAM — niente VRAM extra). Negli output troverai
sempre `test_model` e `judge_model` per tracciabilità.
I grafici si generano off-line dai
JSON di output — qui si producono solo dati grezzi.

## ⚠️ Quota: tutto va su `/scratch2/$USER`

Su `aries.hpc.unimo.it` la home dell'utente ha quota molto stretta —
**tutta la roba pesante (binario Ollama, modelli scaricati, venv) deve
stare su `/scratch2/$USER`**. Gli script lo fanno automaticamente:

```
/scratch2/aguser01/
├── ollama_A100_local/      ← progetto (questa cartella)
├── ollama_bin/             ← binario Ollama (~50 MB)
├── ollama_models/          ← modelli scaricati (5 GB-50 GB ciascuno)
└── venvs/venv_<jobid>/     ← venv del job
```

Carica quindi sempre la cartella in `/scratch2/$USER/` (non in `~`):
```bash
scp -r ollama_A100_local aguser01@aries.hpc.unimo.it:/scratch2/aguser01/
```

| File | Cosa fa |
|---|---|
| `attacks.py` | 31 attacchi paper-faithful |
| `prompts.py` | 19 prompt dannosi |
| `defenses.py` | 11 strategie di difesa, tutte verso `localhost:11434` |
| `judge.py` | Giudice = stesso modello sotto test (chiamata post-generazione) |
| `config.py` | Solo `OLLAMA_HOST` + `OLLAMA_MODEL` (env / CLI override) |
| `run_evaluation.py` | Runner singolo modello (con auto-resume da checkpoint) |
| `run_all_models.py` | Orchestratore sequenziale multi-modello |
| `generate_cross_model_charts.py` | Grafici cross-model su attacchi/difese/prompt riusciti |
| `adaptive_suite/` | Suite compatta data-mined (180 test/modello) con output isolati |
| `models.txt` | Lista modelli da testare (una riga = un tag Ollama, `#` per commentare) |
| `requirements.txt` | Una sola dipendenza: `requests` |
| `run_slurm.sh` | Job SLURM "produzione": multi-modello, lista intera |
| `test_quick.sh` | **Smoke test**: N test totali (default 20, configurabile via `LIMIT=N`), per verificare che tutto giri |
| `results/` | Output: `eval_<modello>_*.json`, `eval_live_*.jsonl`, `multi_run_*.csv` |

## Uso sul cluster aries (UniMoRe) — passo per passo

Lo script `run_slurm.sh` è già configurato per `aries.hpc.unimo.it`
(account `agents`, partition `ice4hpc`, qos `gpus`, A100, module
`python/3.14.0-gcc-8.5.0-ddmg2af`).

### 1) Carica la cartella sul cluster (dal tuo Mac)

```bash
# dalla root del progetto (la cartella padre di ollama_A100_local)
cd experiments
scp -r ollama_A100_local aguser01@aries.hpc.unimo.it:/scratch2/aguser01/
```

> 🔐 **Cambia la password** appena puoi (l'hai condivisa in chat). Sul
> cluster: `passwd`. Meglio: usa una chiave SSH —
> `ssh-keygen -t ed25519` poi
> `ssh-copy-id aguser01@aries.hpc.unimo.it`.

### 2) Entra nel cluster e ispeziona / personalizza

```bash
ssh aguser01@aries.hpc.unimo.it
cd /scratch2/aguser01/ollama_A100_local

# (opzionale) edita la lista dei modelli da testare
nano models.txt        # commenta/aggiungi righe; ognuna è un tag Ollama
```

Per la prima prova *ti consiglio* di lasciare 1–2 modelli piccoli
(es. solo `llama3.1:8b` e `qwen2.5:7b`) per verificare che tutto giri,
poi rilanciare con la lista completa.

### 3) Manda in coda il job

```bash
sbatch run_slurm.sh

# monitor
squeue -u $USER
tail -f logs/jailbreak_eval_*.out      # output del job
tail -f logs/ollama_*.log              # log del daemon Ollama
```

### 4) Recupera i risultati a fine job (dal tuo Mac)

```bash
scp -r aguser01@aries.hpc.unimo.it:/scratch2/aguser01/ollama_A100_local/results ./
```

> **Password**: i comandi `scp`/`ssh` te la chiedono interattivamente
> (il copia-incolla nel terminale funziona). Per non doverla digitare
> ogni volta, configura una chiave SSH **una sola volta**:
> ```bash
> ssh-keygen -t ed25519           # invio ad ogni domanda
> ssh-copy-id aguser01@aries.hpc.unimo.it
> ```
> Da quel momento niente più password.

## Monitorare e interrompere un job in coda

```bash
# Controlla i tuoi job attivi
squeue -u $USER

# Monitora l'output in tempo reale (JOBID nel nome del file)
tail -f logs/jb_smoke_<JOBID>.out

# Interrompi (killa) un job specifico
scancel <JOBID>

# Interrompi tutti i tuoi job attivi in una volta
scancel -u $USER
```

> `<JOBID>` è il numero che SLURM stampa quando lanci `sbatch` (es. `Submitted batch job 129723`).
> Dopo `scancel` il job sparisce da `squeue` in pochi secondi e nessun risultato finale viene scritto
> (il `.jsonl` live rimane con i test già completati fino a quel momento).

## Smoke test rapido (PRIMA di lanciare la valutazione completa)

Prima di sprecare ore di walltime sul job lungo, lancia il file
`test_quick.sh`: esegue **N test totali** (default 20, configurabile da
linea di comando), pulla il modello se manca, e si ferma. Tempo tipico:
**10–30 min** per modelli ≤ 14B con 20 test, qualche minuto in più per i 70B (per
via del solo download al primo run).

### Come specificare il modello

Tre modi (in ordine di precedenza), tutti equivalenti:

```bash
# 1. variabili d'ambiente (raccomandato): MODEL e LIMIT sono indipendenti
MODEL=qwen2.5:7b LIMIT=50 sbatch test_quick.sh
LIMIT=100 sbatch test_quick.sh                  # solo limit, modello = default
MODEL=llama3.1:8b sbatch test_quick.sh          # solo modello, limit = 20

# 2. argomenti posizionali (modello, limit)
sbatch test_quick.sh llama3.1:8b 1000
sbatch test_quick.sh llama3.1:8b                # limit = 100

# 3. default (editabile in cima allo script: DEFAULT_MODEL / DEFAULT_LIMIT)
sbatch test_quick.sh
```

Lo script:
1. ✅ verifica module + python
2. ✅ crea venv pulito + installa `requirements.txt`
3. ✅ installa Ollama in `$HOME/.local/bin` se mancante
4. ✅ avvia il daemon
5. ✅ se il modello NON è presente fa `ollama pull <modello>`
6. ✅ esegue 20 test e si ferma (`run_evaluation.py --fresh --limit 20`)
7. ✅ stampa la tabella ASR parziale e spegne il daemon

Output:
- `results/eval_<modello>_<ts>.json`
- `results/eval_live_<modello>_<ts>.jsonl`
- log job: `logs/jb_smoke_<jobid>.out`

Se il smoke test passa, sei certo che il job lungo (`run_slurm.sh`)
girerà senza sorprese.

In `results/` troverai due tipi di file per ogni modello:

- **`eval_<modello>_<ts>.json`** — file **finale** consolidato a fine run.
  Contiene `metadata` (modello, tempi, latency stats, **ETA per il run completo**,
  ASR aggregati per difesa/attacco/categoria) + `results[]` con SOLO i dati
  tecnici (no prompt completo, no risposta — solo `compliant/refused/latency_s/
  response_length/...`). **Usa questo per i grafici.**
- **`eval_live_<modello>_<ts>.jsonl`** — checkpoint append-only, 1 riga per
  test, scritto in real-time. Serve solo per il **resume** se il job crasha
  (lo stesso modello → stessa cartella → riprende da dove era arrivato).
  Stesso schema slim del file finale.
- **`multi_run_<ts>.csv`** — riepilogo cumulativo multi-modello (1 riga/modello).

Quindi: se vedi `eval_live_*.jsonl` ma manca il corrispondente `eval_*.json`,
significa che il job è stato interrotto prima della fine — al resume successivo
verrà completato e il `.json` finale generato.

I grafici li generi off-line a casa partendo da questi file.

### Grafici cross-model sugli attacchi riusciti

Per confrontare tutti i modelli testati:

```bash
cd experiments/ollama_A100_local
python generate_cross_model_charts.py
```

Se l'ambiente offline non ha le librerie di plotting: `pip install matplotlib numpy`.

Lo script legge i JSON finali `results/eval_*.json` e scrive tutto in
`charts/cross_model/`. Per default usa **l'ultimo run completo per ogni
modello**, così non duplica test se hai più run dello stesso modello. Genera:

- ASR complessivo per modello;
- heatmap attacco × modello;
- heatmap difesa × modello (ASR alto = difesa più bucata);
- heatmap prompt × modello (`prompt_id` + categoria);
- heatmap globale attacco × difesa;
- tabella grafica con miglior attacco, difesa più bucata, difesa più forte
  e prompt più efficace per ogni modello;
- ranking dei percorsi riusciti `modello | attacco | difesa | prompt`.

Usa `--all-runs` solo se vuoi includere anche run multiple dello stesso modello.
Passando path espliciti, lo script usa esattamente quei file.

### Suite adattiva compatta

Per test mirati basati sui risultati reali gia' raccolti:

```bash
cd experiments/ollama_A100_local/adaptive_suite
python3 mine_previous_results.py
python3 run_evaluation.py --dry-run --model qwen3.5:4b
python3 run_all_models.py --fresh
```

La matrice default e' molto piu' corta della suite completa:
**6 prompt x 6 attacchi x 5 difese = 180 test per modello**. Con gli 11
modelli attivi in `../models.txt` sono 1980 test totali. I risultati restano
separati in `adaptive_suite/results/`, `adaptive_suite/traces/` e
`adaptive_suite/logs/`. La guida per generare nuove varianti e' in
`adaptive_suite/expand_guide.md`.

### Cosa fa `run_slurm.sh` (sotto al cofano)

1. `module purge && module load slurm python/3.14.0-gcc-8.5.0-ddmg2af`
2. crea un venv pulito in `$TMPDIR/venv_<jobid>` e installa
   `requirements.txt` (solo `requests`)
3. installa il binario `ollama` in `$HOME/.local/bin` se assente
4. avvia `ollama serve` in background sul nodo
5. attende che il daemon sia ready
6. lancia `python run_all_models.py` (modelli da `models.txt`)
7. al termine (o in caso di errore) spegne il daemon

### Cache modelli tra job

I modelli scaricati con `ollama pull` finiscono in `$HOME/.ollama/models`
per default — restano in cache tra un job e l'altro. Se vuoi forzare lo
scratch del job (sparisce a fine job), lancia:

```bash
OLLAMA_MODELS="$TMPDIR/models" sbatch run_slurm.sh
```

### Walltime

Default: `--time=48:00:00`. Una matrice 19×31×11 = 6.479 test su un 70B
locale può tranquillamente impiegare 12-24h. Se la lista in `models.txt`
ha molti modelli grandi, alza il walltime; se ne hai uno solo piccolo,
`--time=04:00:00` basta. Si modifica direttamente in cima a
`run_slurm.sh`.

> ℹ️ **Cosa fa esattamente `--time=HH:MM:SS`?**
> È il **walltime massimo** che chiedi a SLURM: se il job sfora, viene
> killato. NON è una durata obbligatoria — se lo script termina prima,
> il job esce subito. (Es. uno script con `sleep 60` e `--time=00:03:00`
> finisce dopo 1 minuto, non 3.) Imposta sempre un valore *abbondante*:
> SLURM ti farà aspettare di più in coda solo per i job molto grossi.

### Passare flag a `run_all_models.py` da sbatch

Tutto quello che metti dopo `run_slurm.sh` viene inoltrato:

```bash
sbatch run_slurm.sh --no-pull              # salta i pull
sbatch run_slurm.sh --attacks DAN AIM      # solo 2 attacchi
sbatch run_slurm.sh --models-file mini.txt # lista alternativa
sbatch run_slurm.sh --fresh                # ignora checkpoint
```

## Uso locale (fuori dal cluster)

Funziona uguale, basta avere Ollama installato e in esecuzione:

```bash
cd ollama_A100_local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama serve &                       # in un altro terminale, oppure systemd
python run_all_models.py             # tutta la lista
# oppure singolo modello:
python run_evaluation.py --model llama3.1:8b
```

## Personalizzazione della lista modelli

Edita `models.txt` (una riga = un tag Ollama, `#` per commentare):

```
llama3.1:8b
qwen2.5:7b
qwen2.5:32b
llama3.1:70b
# qwen2.5:72b   ← commentato
```

Tutti i modelli devono entrare in 80 GB di VRAM con la quantizzazione di
default che Ollama usa (per i 70-72B serve la Q4 standard). Vincoli più
stringenti? Filtra la lista o usa `--no-pull` per saltare quelli mancanti.

## Flag utili

### `run_evaluation.py` (singolo modello)
```
--model llama3.1:8b        override modello sotto test (giudice si allinea)
--judge-model qwen2.5:7b   forza giudice diverso dal modello sotto test
--host http://...          daemon Ollama remoto
--attacks DAN AIM          sottoinsieme di attacchi
--defenses no-defense      sottoinsieme di difese
--fresh                    ignora checkpoint esistenti
--dry-run                  mostra il piano senza eseguire
```

### `run_all_models.py` (sequenziale multi-modello)
```
--models-file my_list.txt  override lista modelli
--no-pull                  salta i modelli non già installati
--keep-loaded              non scaricare il modello tra un run e l'altro
--fresh                    ignora checkpoint esistenti
--attacks / --defenses     sottoinsiemi (uguali a sopra)
```

## Output

Per ogni modello vengono prodotti:

- `results/eval_live_<slug>_<ts>.jsonl` — append-only, 1 riga per test (per
  resume e per analisi streaming)
- `results/eval_<slug>_<ts>.json` — JSON finale completo (`metadata` +
  `results[]`)

Riepilogo aggregato multi-modello:

- `results/multi_run_<ts>.csv` — una riga per modello con `overall_asr`,
  `asr_<defense>` per ciascuna difesa, `tests_valid`, `tests_errored`,
  `elapsed_min`.

I grafici si generano off-line a posteriori dai JSON (uno script
matplotlib di una decina di righe sui `results[]`).

---

## Sincronizzare le modifiche locali sul server remoto

Dopo aver modificato file in locale, usa questo comando per copiare solo i
sorgenti sul server **senza sovrascrivere** `results/` e `logs/` (generati
sul server):

```bash
rsync -av --exclude='results/' --exclude='logs/' --exclude='__pycache__/' --exclude='*.pyc' \
  experiments/ollama_A100_local/ aguser01@aries.hpc.unimo.it:/scratch2/aguser01/ollama_A100_local/
```

> Esegui il comando dalla **root del progetto** (`llm_university_project/`).
> La `/` finale dopo `ollama_A100_local` è fondamentale: copia il contenuto
> della cartella, non la cartella stessa.
