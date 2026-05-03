# Adaptive Ollama jailbreak suite

Compact follow-up suite derived from the completed full-matrix
`../results/eval_*.json` runs.

The default matrix is intentionally small:

```text
6 prompts x 6 attacks x 5 defenses = 180 tests per model
```

With the 11 active models currently listed in `../models.txt`, this is 1980
total tests.

## Files

| File | Purpose |
|---|---|
| `prompts.py` | Six individually rewritten prompt seeds from the strongest previous prompt families. |
| `attacks.py` | Six adaptive attack wrappers derived from the strongest previous attack families. |
| `defenses.py` | Compact defense subset, including `no-defense`. |
| `run_evaluation.py` | Single-model runner; writes local `results/` and `traces/`. |
| `run_all_models.py` | Multi-model runner; reads `../models.txt` by default. |
| `mine_previous_results.py` | Analysis script for previous `../results/eval_*.json` files. |
| `expand_guide.md` | Instructions for generating future candidates from measured winners. |
| `run_slurm.sh` | SLURM wrapper for the A100 server. |

## Mine previous results

```bash
cd experiments/ollama_A100_local/adaptive_suite
python3 mine_previous_results.py
```

Output:

```text
analysis/seed_signals.json
```

## Dry run

```bash
python3 run_evaluation.py --dry-run --model qwen3.5:4b
```

## Run one model

```bash
python3 run_evaluation.py --model qwen3.5:4b --fresh
```

## Run all models from `../models.txt`

```bash
python3 run_all_models.py --fresh
```

On SLURM/A100:

```bash
cd experiments/ollama_A100_local/adaptive_suite
sbatch run_slurm.sh --fresh
```

Results are isolated from the parent suite:

```text
adaptive_suite/results/
adaptive_suite/traces/
adaptive_suite/logs/
```
