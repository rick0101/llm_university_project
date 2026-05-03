#!/bin/bash
#SBATCH -A agents                       # account
#SBATCH -p ice4hpc                      # partition
#SBATCH --qos=gpus                      # QoS richiesta dal cluster per le GPU
#SBATCH -N 1                            # 1 nodo
#SBATCH --ntasks-per-node=8             # 8 task / CPU
#SBATCH --gpus=a100:1                   # 1 × A100 (80 GB)
#SBATCH --mem=48g                       # RAM CPU per Ollama mmap/staging; la memoria pesante resta VRAM
#SBATCH --time=40:00:00                 # walltime: alza/abbassa in base alla lista in models.txt
#SBATCH --job-name=jailbreak_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ─────────────────────────────────────────────────────────────────────
# Job SLURM end-to-end per il cluster aries (UniMoRe).
#   1. carica i module necessari
#   2. crea un venv pulito (su scratch del job)
#   3. installa requirements.txt
#   4. avvia il daemon Ollama in background sul nodo
#   5. lancia run_all_models.py (modelli da models.txt)
#   6. spegne tutto a fine job
#
# Tutto self-contained — nessuna API key richiesta.
# Per cambiare lista modelli → edita `models.txt` prima del sbatch.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs results

# ── Timing globale ───────────────────────────────────────────────────
SECONDS=0
hms() { printf '%02dh %02dm %02ds' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

# ── SCRATCH: tutto fuori da $HOME (l'utente ha quota solo su /scratch2) ──
SCRATCH="/scratch2/${USER}"
mkdir -p "$SCRATCH"/{ollama_bin,ollama_models,venvs}

echo "════════════════════════════════════════════════════════════════"
echo "  JOB     : $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "  NODE    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
echo "  WORKDIR : $WORKDIR"
echo "  START   : $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── 1. Module env (specifico per aries.hpc.unimo.it) ─────────────────
# `module purge` può fallire per alias mancanti nel profile del cluster:
# isoliamolo da `set -e` perché è solo rumore innocuo.
set +e
module purge 2>/dev/null
set -e
module load slurm
module load python/3.14.0-gcc-8.5.0-ddmg2af
echo "🐍 Python: $(which python) ($(python --version))"

# ── 2. venv pulito su scratch ────────────────────────────────────────
VENV_DIR="$SCRATCH/venvs/venv_${SLURM_JOB_ID}"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── 3. Ollama: installazione user-space su SCRATCH se assente ────────
export PATH="$SCRATCH/ollama_bin/bin:$PATH"
export LD_LIBRARY_PATH="$SCRATCH/ollama_bin/lib/ollama:${LD_LIBRARY_PATH:-}"
if ! command -v ollama >/dev/null 2>&1; then
    echo "⬇️  Installo Ollama in $SCRATCH/ollama_bin (binario statico)..."
    mkdir -p "$SCRATCH/ollama_bin"
    curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
         -o "$SCRATCH/ollama.tar.zst"
    tar --use-compress-program='zstd -d' -xf "$SCRATCH/ollama.tar.zst" -C "$SCRATCH/ollama_bin"
    rm "$SCRATCH/ollama.tar.zst"
fi
echo "🦙 Ollama: $(which ollama) ($(ollama --version 2>&1 | head -1))"

# Modelli su SCRATCH: persistono tra job (l'utente ha quota solo qui).
export OLLAMA_MODELS="$SCRATCH/ollama_models"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"   # override safe via env se serve
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"             # una richiesta alla volta: evita picchi VRAM/KV cache
export OLLAMA_EVAL_NUM_CTX="${OLLAMA_EVAL_NUM_CTX:-32768}"         # evita default DeepSeek 131k senza limitare troppo i test
export OLLAMA_RESTART_LOG_DIR="$WORKDIR/logs"
mkdir -p "$OLLAMA_MODELS"

# ── 4. avvio daemon ──────────────────────────────────────────────────
echo "▶️  Avvio Ollama daemon..."
ollama serve >logs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
trap 'echo "⏹️  Stop Ollama (pid $OLLAMA_PID)"; kill $OLLAMA_PID 2>/dev/null || true' EXIT

# attesa che il daemon sia ready (max 60s)
OLLAMA_READY=0
for i in $(seq 1 60); do
    if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo "✅ Ollama ready dopo ${i}s"
        OLLAMA_READY=1
        break
    fi
    sleep 1
done
if [ "$OLLAMA_READY" -ne 1 ]; then
    echo "❌ Ollama non è diventato ready entro 60s. Vedi logs/ollama_${SLURM_JOB_ID}.log"
    exit 1
fi

# ── 5. esegui la valutazione (multi-modello sequenziale) ─────────────
# Il giudice viene selezionato automaticamente per ogni modello:
#   - modelli ≥ 30B → self-judge (nessuna VRAM extra)
#   - modelli < 30B → qwen3.6:35b come giudice esterno
EVAL_START=$SECONDS
PYTHONUNBUFFERED=1 python -u run_all_models.py --host "http://${OLLAMA_HOST}" "$@"
EVAL_ELAPSED=$((SECONDS - EVAL_START))
TOTAL_ELAPSED=$SECONDS

echo "════════════════════════════════════════════════════════════════"
echo "  END     : $(date)"
echo "  ⏱  TEMPO VALUTAZIONE (run_all_models): $(hms $EVAL_ELAPSED)"
echo "  ⏱  TEMPO TOTALE JOB (setup + valutazione): $(hms $TOTAL_ELAPSED)"
echo "════════════════════════════════════════════════════════════════"
