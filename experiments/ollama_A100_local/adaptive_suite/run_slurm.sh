#!/bin/bash
#SBATCH -A agents
#SBATCH -p ice4hpc
#SBATCH --qos=gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=a100:1
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --job-name=adaptive_jb_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs results traces

SECONDS=0
hms() { printf '%02dh %02dm %02ds' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

SCRATCH="/scratch2/${USER}"
mkdir -p "$SCRATCH"/{ollama_bin,ollama_models,venvs}

echo "================================================================"
echo "  JOB     : $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "  NODE    : $(hostname)"
echo "  WORKDIR : $WORKDIR"
echo "  START   : $(date)"
echo "================================================================"

set +e
module purge 2>/dev/null
set -e
module load slurm
module load python/3.14.0-gcc-8.5.0-ddmg2af

VENV_DIR="$SCRATCH/venvs/adaptive_venv_${SLURM_JOB_ID}"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r ../requirements.txt

export PATH="$SCRATCH/ollama_bin/bin:$PATH"
export LD_LIBRARY_PATH="$SCRATCH/ollama_bin/lib/ollama:${LD_LIBRARY_PATH:-}"
if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama in $SCRATCH/ollama_bin..."
    mkdir -p "$SCRATCH/ollama_bin"
    curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
         -o "$SCRATCH/ollama.tar.zst"
    tar --use-compress-program='zstd -d' -xf "$SCRATCH/ollama.tar.zst" -C "$SCRATCH/ollama_bin"
    rm "$SCRATCH/ollama.tar.zst"
fi

export OLLAMA_MODELS="$SCRATCH/ollama_models"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_RESTART_LOG_DIR="$WORKDIR/logs"
mkdir -p "$OLLAMA_MODELS"

echo "Starting Ollama daemon..."
ollama serve >logs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
trap 'echo "Stop Ollama (pid $OLLAMA_PID)"; kill $OLLAMA_PID 2>/dev/null || true' EXIT

OLLAMA_READY=0
for i in $(seq 1 60); do
    if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo "Ollama ready after ${i}s"
        OLLAMA_READY=1
        break
    fi
    sleep 1
done
if [ "$OLLAMA_READY" -eq 0 ]; then
    echo "Ollama did not become ready within 60s. See logs/ollama_${SLURM_JOB_ID}.log"
    exit 1
fi

EVAL_START=$SECONDS
PYTHONUNBUFFERED=1 python -u run_all_models.py --host "http://${OLLAMA_HOST}" "$@"
EVAL_ELAPSED=$((SECONDS - EVAL_START))
TOTAL_ELAPSED=$SECONDS

echo "================================================================"
echo "  END        : $(date)"
echo "  EVAL TIME  : $(hms $EVAL_ELAPSED)"
echo "  TOTAL TIME : $(hms $TOTAL_ELAPSED)"
echo "================================================================"
