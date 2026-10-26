#!/bin/bash
#SBATCH -A agents
#SBATCH -p ice4hpc
#SBATCH --qos=gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=a100:1
#SBATCH --mem=20g
#SBATCH --time=03:00:00
#SBATCH --job-name=jb_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ─────────────────────────────────────────────────────────────────────
# VALIDATION TEST — campione casuale di N test dalla matrice completa.
#
# Cosa fa:
#   1. carica i moduli del cluster aries
#   2. crea un venv pulito + installa requirements.txt
#   3. installa ollama in user-space se assente
#   4. avvia ollama serve
#   5. fa `ollama pull` per modello + giudice se non presenti
#   6. lancia run_evaluation.py --sample N --fresh
#      (campionamento casuale: copre tutte le categorie/attacchi/difese)
#   7. spegne il daemon
#
# Output prodotto (per verifica manuale):
#   results/eval_<model>_<ts>.json          → metadati slim (per grafici)
#   results/eval_live_<model>_<ts>.jsonl    → checkpoint
#   traces/trace_<model>_<ts>.jsonl         → prompt + risposte completi
#
# Specifica modello e numero di test in UNO di questi modi (in ordine di
# precedenza):
#
#   ENV (raccomandato):
#     MODEL=qwen2.5:7b SAMPLE=500 sbatch test_quick.sh
#
#   POSIZIONALI (modello come 1°, sample come 2°):
#     sbatch test_quick.sh llama3.1:8b 500
#
#   DEFAULT (sotto):
DEFAULT_MODEL="gemma4:31b"
DEFAULT_SAMPLE=100
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs results traces

# ── Timing globale ───────────────────────────────────────────────────
JOB_START_EPOCH=$(date +%s)
SECONDS=0
hms() { printf '%02dh %02dm %02ds' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

# risoluzione modello e sample: env > posizionali > default
MODEL="${MODEL:-${1:-$DEFAULT_MODEL}}"
SAMPLE="${SAMPLE:-${2:-$DEFAULT_SAMPLE}}"
# Auto-select judge: ≥50B params (≈30GB Q4) → self-judge; <50B → qwen3.6:35b
# Q4 quantization (Ollama default) ≈ 0.6 GB per B params, so 30 GB ≈ 50 B.
# Override via env: JUDGE_MODEL=xxx sbatch test_quick.sh
if [ -z "${JUDGE_MODEL:-}" ]; then
    MODEL_B=$(echo "$MODEL" | grep -oiE '[0-9]+b' | grep -oE '[0-9]+' | sort -rn | head -1)
    if [ -n "$MODEL_B" ] && [ "$MODEL_B" -ge 50 ]; then
        JUDGE_MODEL="$MODEL"
        JUDGE_LABEL="self-judge ($MODEL)"
    else
        JUDGE_MODEL="qwen3.6:35b"
        JUDGE_LABEL="external (qwen3.6:35b)"
    fi
else
    JUDGE_LABEL="override: $JUDGE_MODEL"
fi

echo "════════════════════════════════════════════════════════════════"
echo "  VALIDATION TEST : ${SAMPLE} test (campione casuale)"
echo "  MODEL           : $MODEL"
echo "  JUDGE           : $JUDGE_LABEL"
echo "  JOB             : $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "  NODE            : $(hostname)"
echo "  GPU             : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
echo "  START           : $(date)"
echo "════════════════════════════════════════════════════════════════"

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

# ── SCRATCH: tutto fuori da $HOME (l'utente ha quota solo su /scratch2) ──
SCRATCH="/scratch2/${USER}"
mkdir -p "$SCRATCH"/{ollama_bin,ollama_models,venvs}

# ── 1. module env (aries.hpc.unimo.it) ───────────────────────────────
set +e
module purge 2>/dev/null
set -e
module load slurm
module load python/3.14.0-gcc-8.5.0-ddmg2af
echo "🐍 Python: $(which python) ($(python --version))"

# ── 2. venv pulito + requirements (su scratch) ───────────────────────
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

export OLLAMA_MODELS="$SCRATCH/ollama_models"
export OLLAMA_HOST="127.0.0.1:11434"
# self-judge: solo 1 slot necessario; external judge (qwen3.6:35b): serve 2
if [ "$JUDGE_MODEL" = "$MODEL" ]; then
    export OLLAMA_MAX_LOADED_MODELS=1
else
    export OLLAMA_MAX_LOADED_MODELS=2
fi
mkdir -p "$OLLAMA_MODELS"

# ── 4. avvio daemon ──────────────────────────────────────────────────
echo "▶️  Avvio Ollama daemon..."
ollama serve >logs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
trap 'echo "⏹️  Stop Ollama (pid $OLLAMA_PID)"; kill $OLLAMA_PID 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
    if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo "✅ Ollama ready dopo ${i}s"
        break
    fi
    sleep 1
done

# ── 5. pull modello sotto test + giudice (solo se esterno) ──────────
if [ "$JUDGE_MODEL" = "$MODEL" ]; then
    MODELS_TO_PULL=("$MODEL")
else
    MODELS_TO_PULL=("$MODEL" "$JUDGE_MODEL")
fi
for M in "${MODELS_TO_PULL[@]}"; do
    if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$M"; then
        echo "✅ Modello già presente: $M"
    else
        echo "⬇️  Pull modello: $M ..."
        ollama pull "$M"
    fi
done

# ── 6. esegui i test (campione casuale) ──────────────────────────────
# --sample N : campiona N test casuali dalla matrice completa
# --fresh    : sempre fresh (run di validazione indipendente)
EVAL_START=$SECONDS
PYTHONUNBUFFERED=1 python -u run_evaluation.py \
    --host "http://${OLLAMA_HOST}" \
    --model "$MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --fresh \
    --sample "$SAMPLE"
EVAL_ELAPSED=$((SECONDS - EVAL_START))
TOTAL_ELAPSED=$SECONDS

echo "════════════════════════════════════════════════════════════════"
echo "  VALIDATION TEST COMPLETATO"
echo "  results/eval_<model>_<ts>.json          ← metadati (grafici)"
echo "  traces/trace_<model>_<ts>.jsonl         ← prompt+risposte (audit)"
echo "  END   : $(date)"
echo "  ⏱  TEMPO ${SAMPLE} TEST (run_evaluation): $(hms $EVAL_ELAPSED)"
echo "  ⏱  TEMPO TOTALE JOB (setup+pull+test)  : $(hms $TOTAL_ELAPSED)"
echo "════════════════════════════════════════════════════════════════"
