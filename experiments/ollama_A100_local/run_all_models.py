#!/usr/bin/env python3
"""
Multi-model sequential runner.

Reads a list of Ollama model tags from a config file (default: `models.txt`)
and, for each one, runs the FULL jailbreak evaluation matrix. The judge model
is automatically kept in sync with the model under test (same model, separate
post-generation call).

Workflow per model:
  1. (optional) `ollama pull <model>` if not already installed
  2. (optional) preload the model in VRAM
  3. Run `run_evaluation.run_evaluation()` end-to-end
  4. (optional) `ollama stop <model>` to free VRAM before the next one
  5. Move on to the next model

A run-level summary CSV is written to `results/multi_run_<timestamp>.csv`
with one row per model (overall ASR, per-defense ASR, total errors).

Usage:
    python run_all_models.py                       # use models.txt
    python run_all_models.py --models-file my.txt  # custom list
    python run_all_models.py --no-pull             # skip auto pull
    python run_all_models.py --keep-loaded         # don't unload between models
    python run_all_models.py --fresh               # restart all from scratch
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests as _requests

# Forza line-buffering su stdout/stderr: quando questo script viene lanciato
# da SLURM (stdout → file), Python usa full-buffering di default e i print()
# non appaiono in tempo reale nel log. Con line_buffering=True ogni \n flushes.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))

import config
from defenses import DEFENSES
from ollama_health import daemon_alive, ensure_daemon_alive, restart_daemon

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_MODELS_FILE = os.path.join(os.path.dirname(__file__), "models.txt")

def select_judge_model(model: str) -> str:
    """Select the judge model based on the test model's parameter count.

    Approximation: assumes Q4_K_M quantization (~0.6 GB per B params).
    - Models ≥50B params: self-judge.
    - Models <50B params: qwen3.6:35b as external judge.
      Small models are unreliable self-judges; qwen3.6 is more capable.
    - Ambiguous names (no 'Xb' pattern): default to external judge.
    - Special case: if test model IS qwen3.6:35b, the function returns
      qwen3.6:35b → judge == model → self-judge mode (single model loaded).
    """
    return config.select_judge_model(model)


def _model_already_completed(model: str) -> bool:
    """Ritorna True se esiste un JSON finale in results/ con tests_total_completed
    >= total_tests_planned per questo modello. Solo una run COMPLETA conta come
    'già testato' — run parziali vengono ignorate e il modello riparte da zero."""
    import glob as _glob
    import json as _json

    slug = model.replace(":", "_").replace("/", "_")
    pattern = os.path.join(RESULTS_DIR, f"eval_{slug}_*.json")
    for path in _glob.glob(pattern):
        try:
            with open(path, encoding="utf-8") as f:
                data = _json.load(f)
            meta = data.get("metadata", {})
            completed = meta.get("tests_total_completed", 0)
            planned = meta.get("total_tests_planned", 1)
            if completed >= planned:
                return True
        except Exception:
            continue
    return False


def parse_models_file(path: str) -> list[str]:
    """Parse a models list: one model per non-empty, non-comment line."""
    models: list[str] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Inline comments: "model:tag  # note"
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if line:
                models.append(line)
    return models


def _ollama_installed_models() -> set[str]:
    try:
        r = _requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return set()


def _ollama_daemon_alive(timeout: int = 5) -> bool:
    """Quick health-check: HTTP GET /api/tags."""
    return daemon_alive(config.OLLAMA_HOST, timeout=timeout)


def _ollama_daemon_restart(wait_secs: int = 30) -> bool:
    """Tenta di riavviare il daemon Ollama (ucciso, lanciato, atteso che risponda).

    Ritorna True se il daemon è di nuovo raggiungibile, False altrimenti.
    Pensato per girare dentro lo SLURM job: usa l'eseguibile `ollama` già nel PATH.
    """
    return restart_daemon(config.OLLAMA_HOST, wait_secs=wait_secs)


def _ensure_daemon_alive() -> bool:
    """Garantisce daemon vivo: check rapido, se down → restart. Ritorna True se OK."""
    return ensure_daemon_alive(config.OLLAMA_HOST)


def _ollama_pull(model: str) -> bool:
    """Stream-pull a model via the Ollama API."""
    print(f"   ⬇️  Pull {model}...", flush=True)
    try:
        r = _requests.post(
            f"{config.OLLAMA_HOST}/api/pull",
            json={"name": model, "stream": False},
            timeout=None,
        )
        r.raise_for_status()
        print(f"   ✅ Pull completato: {model}")
        return True
    except Exception as e:
        print(f"   ❌ Pull fallito ({model}): {e}")
        return False


def _ollama_preload(model: str) -> None:
    """Touch the model so it's loaded in VRAM before the matrix starts."""
    try:
        _requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": 1,
                    "num_ctx": config.CONTEXT_WINDOW,
                },
            },
            timeout=600,
        ).raise_for_status()
    except Exception as e:
        print(f"   ⚠️  Preload non riuscito (procedo comunque): {e}")


def _ollama_unload(model: str) -> None:
    """Free VRAM by setting keep_alive=0 on a tiny request."""
    try:
        _requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=60,
        )
    except Exception:
        pass


def _summarise_results(results: list, model: str) -> dict:
    """Compute overall + per-defense ASR for one model."""
    valid = [r for r in results if not r.get("error")]
    n_total = len(valid)
    n_complied = sum(1 for r in valid if r["compliant"])
    overall = (n_complied / n_total) if n_total else 0.0

    per_defense: dict[str, float] = {}
    for d in DEFENSES:
        sub = [r for r in valid if r["defense"] == d]
        per_defense[d] = (sum(1 for r in sub if r["compliant"]) / len(sub)) if sub else float("nan")

    return {
        "model": model,
        "overall_asr": round(overall, 4),
        "tests_valid": n_total,
        "tests_errored": len(results) - n_total,
        **{f"asr_{d}": round(v, 4) for d, v in per_defense.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-model jailbreak runner")
    parser.add_argument("--models-file", default=DEFAULT_MODELS_FILE,
                        help="Path to a text file with one Ollama model per line")
    parser.add_argument("--host", type=str,
                        help="Override Ollama host (default: http://localhost:11434)")
    parser.add_argument("--no-pull", action="store_true",
                        help="Skip auto `ollama pull` when a model is missing")
    parser.add_argument("--keep-loaded", action="store_true",
                        help="Don't unload the model between runs (faster, more VRAM)")
    parser.add_argument("--fresh", action="store_true",
                        help="Pass --fresh to each per-model evaluation")
    parser.add_argument("--attacks", nargs="*", help="Subset of attacks (per model)")
    parser.add_argument("--defenses", nargs="*", help="Subset of defenses (per model)")
    args = parser.parse_args()

    if args.host:
        config.OLLAMA_HOST = args.host

    if not Path(args.models_file).exists():
        print(f"❌ File modelli non trovato: {args.models_file}")
        sys.exit(1)

    models = parse_models_file(args.models_file)
    if not models:
        print(f"❌ Nessun modello in {args.models_file}")
        sys.exit(1)

    print(f"🧪 Multi-model jailbreak evaluation")
    print(f"   Host    : {config.OLLAMA_HOST}")
    print(f"   Modelli : {len(models)} ({', '.join(models)})")
    print(
        f"   Giudice : auto (≥{config.SELF_JUDGE_THRESHOLD_B}B → self-judge, "
        f"<{config.SELF_JUDGE_THRESHOLD_B}B → {config.EXTERNAL_JUDGE_MODEL})\n"
    )

    # Sanity: daemon up?
    try:
        _requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10).raise_for_status()
    except Exception as e:
        print(f"❌ Ollama daemon non raggiungibile: {e}")
        print("   Avvia con: `ollama serve`")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_ts = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    summary_path = os.path.join(RESULTS_DIR, f"multi_run_{summary_ts}.csv")
    summary_rows: list[dict] = []
    pulled_judges: set[str] = set()  # track which judge models are already pulled

    overall_t0 = time.time()

    for idx, model in enumerate(models, 1):
        banner = "═" * 70
        print(f"\n{banner}")
        print(f"  [{idx}/{len(models)}] MODEL: {model}")
        print(banner)

        # Health-check daemon: tra modelli successivi può essere crashato.
        if not _ensure_daemon_alive():
            print(f"   ❌ Daemon Ollama irrecuperabile — skip {model}.")
            summary_rows.append({
                "model": model, "overall_asr": None, "tests_valid": 0,
                "tests_errored": 0, "elapsed_min": 0,
                "results_file": "FAILED_DAEMON_DOWN",
            })
            continue

        # Pull del modello sotto test se mancante
        installed = _ollama_installed_models()
        if model not in installed:
            if args.no_pull:
                print(f"   ⏭️  Skip ({model} non installato e --no-pull attivo)")
                continue
            if not _ollama_pull(model):
                print(f"   ⏭️  Skip {model}")
                continue

        # Aggiorna modello sotto test e seleziona il giudice appropriato
        config.OLLAMA_MODEL = model
        os.environ["OLLAMA_MODEL"] = model

        judge = select_judge_model(model)
        config.JUDGE_MODEL = judge
        os.environ["JUDGE_MODEL"] = judge
        is_self_judge = (judge == model)
        judge_label = "self-judge" if is_self_judge else f"external → {judge}"
        print(f"   🤖 Giudice: {judge_label}")

        # Pull del giudice esterno se non già scaricato in questo run
        if not is_self_judge and judge not in pulled_judges:
            judge_installed = _ollama_installed_models()
            if judge not in judge_installed:
                print(f"   ⬇️  Pull modello giudice: {judge} ...")
                if not _ollama_pull(judge):
                    print(f"   ❌ Impossibile scaricare il giudice {judge} — skip modello {model}.")
                    continue
            else:
                print(f"   ✅ Modello giudice già presente: {judge}")
            pulled_judges.add(judge)

        # Controlla se già completato interamente: in quel caso salta.
        # Run parziali (es. solo 100 prompt) non contano → si riparte da zero.
        if _model_already_completed(model):
            print(f"   ✅ Skip {model}: già testato completamente in un run precedente.")
            continue

        # Preload (warm VRAM)
        _ollama_preload(model)

        # Reload run_evaluation each iteration so it picks up the new config
        import importlib
        import run_evaluation
        importlib.reload(run_evaluation)

        t0 = time.time()
        try:
            results_path = run_evaluation.run_evaluation(
                selected_attacks=args.attacks,
                selected_defenses=args.defenses,
                dry_run=False,
                fresh=True,   # sempre fresh: le run parziali non si riprendono
            )
        except KeyboardInterrupt:
            print(f"\n⏹️  Interrotto dall'utente durante {model}")
            break
        except Exception as e:
            print(f"❌ Errore durante valutazione di {model}: {e}")
            _ensure_daemon_alive()
            results_path = None

        elapsed = time.time() - t0
        print(f"\n⏱️  {model} completato in {elapsed/60:.1f} min")

        # Summary row
        if results_path and os.path.exists(results_path):
            import json
            with open(results_path, encoding="utf-8") as f:
                data = json.load(f)
            row = _summarise_results(data["results"], model)
            row["elapsed_min"] = round(elapsed / 60, 2)
            row["results_file"] = os.path.basename(results_path)
            summary_rows.append(row)

            # Append to CSV after every model so a crash doesn't lose history
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

        # Unload to free VRAM for the next model
        if not args.keep_loaded:
            print(f"   💤 Unload {model} (free VRAM)...")
            _ollama_unload(model)
            if not is_self_judge:
                print(f"   💤 Unload judge {judge} (free VRAM)...")
                _ollama_unload(judge)

    total_elapsed = time.time() - overall_t0
    print(f"\n{'═'*70}")
    print(f"  TUTTI I MODELLI COMPLETATI in {total_elapsed/60:.1f} min")
    print(f"  Riepilogo CSV: {summary_path}")
    print(f"{'═'*70}\n")

    if summary_rows:
        print(f"  {'model':<28} {'overall ASR':>12} {'valid':>7} {'errors':>7} {'min':>6}")
        for r in summary_rows:
            print(f"  {r['model']:<28} {r['overall_asr']:>12.1%} "
                  f"{r['tests_valid']:>7} {r['tests_errored']:>7} "
                  f"{r['elapsed_min']:>6.1f}")


if __name__ == "__main__":
    main()
