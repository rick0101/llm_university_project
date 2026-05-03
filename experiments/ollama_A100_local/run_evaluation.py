#!/usr/bin/env python3
"""
Jailbreak Evaluation — LOCAL RUNNER (single Ollama model on A100).

Sends harmful prompts × attack templates × defense strategies to a LOCAL
Ollama model, judges each response with the SAME local model (separate call),
and saves results + charts.

For multi-model sequential runs use `run_all_models.py`.

Usage:
    python run_evaluation.py                          # full eval (auto-resume)
    python run_evaluation.py --model llama3:8b        # override model
    python run_evaluation.py --attacks DAN AIM        # subset of attacks
    python run_evaluation.py --defenses no-defense    # subset of defenses
    python run_evaluation.py --fresh                  # ignore checkpoints
    python run_evaluation.py --dry-run                # show plan without running
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime

# Forza line-buffering su stdout anche quando il processo padre (run_all_models)
# ha importato questo modulo invece di lanciarlo come subprocess separato.
# Senza questo, i print() restano nel buffer e il log non si aggiorna in tempo reale.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))

import config
from attacks import ATTACKS
from defenses import DEFENSES
from judge import judge_response
from prompts import HARMFUL_PROMPTS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TRACES_DIR  = os.path.join(os.path.dirname(__file__), "traces")


def _model_slug(model: str) -> str:
    """Filesystem-safe slug for the model name (used in checkpoint filenames)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _find_latest_live_file(model_slug: str) -> str | None:
    """Return path to the most-recent eval_live_<model>_*.jsonl, or None."""
    pattern = os.path.join(RESULTS_DIR, f"eval_live_{model_slug}_*.jsonl")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _load_completed(live_path: str) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not live_path or not os.path.exists(live_path):
        return done
    with open(live_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["defense"], r["attack"], r["prompt_id"]))
            except Exception:
                pass
    return done


def check_local_ollama() -> bool:
    """Verify the local Ollama daemon is reachable and the model is loaded."""
    import requests as _req

    # 1. daemon reachable
    try:
        resp = _req.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        installed = {m["name"] for m in resp.json().get("models", [])}
        print(f"✅ Ollama daemon raggiungibile su {config.OLLAMA_HOST}")
    except Exception as e:
        print(f"❌ Ollama daemon non raggiungibile su {config.OLLAMA_HOST}: {e}")
        print("   Avvia Ollama con: `ollama serve`")
        return False

    # 2. model present
    if config.OLLAMA_MODEL not in installed:
        print(f"⚠️  Modello '{config.OLLAMA_MODEL}' non installato. "
              f"Modelli disponibili: {sorted(installed) or '(nessuno)'}")
        print(f"   Esegui:  ollama pull {config.OLLAMA_MODEL}")
        return False

    # 3. round-trip generation
    try:
        resp = _req.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": 1,
                    "num_ctx": config.CONTEXT_WINDOW,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        print(f"✅ Modello test  : {config.OLLAMA_MODEL}")
        print(f"✅ Modello giudice: {config.JUDGE_MODEL} (locale)")
        return True
    except Exception as e:
        print(f"❌ Generazione di test fallita: {e}")
        return False


def run_evaluation(selected_attacks=None, selected_defenses=None, dry_run=False,
                   fresh=False, limit=None, sample=None, seed=None):
    """Run the evaluation matrix for the currently configured model.

    Parameters
    ----------
    sample : int | None
        Se impostato, campiona casualmente `sample` combinazioni dal totale
        (defense × attack × prompt), coprendo tutte le categorie/attacchi/difese
        in modo proporzionale. Ideale per validation run veloci.
        Se None, esegue la matrice completa (o si ferma con `limit`).
    seed : int | None
        Seme per il campionamento casuale (riproducibilità). None = casuale.
    limit : int | None
        Se impostato, si ferma dopo N *nuovi* test completati in questa sessione.
        Compatibile con `sample` (si ferma prima se limit < sample).

    Output files (in results/ e traces/):
      - results/eval_live_<model>_<ts>.jsonl  → checkpoint slim (per resume)
      - results/eval_<model>_<ts>.json        → file finale (per grafici)
      - traces/trace_<model>_<ts>.jsonl       → record completo con prompt e risposte
    """
    attacks = {k: v for k, v in ATTACKS.items()
               if not selected_attacks or k in selected_attacks}
    defenses = {k: v for k, v in DEFENSES.items()
                if not selected_defenses or k in selected_defenses}
    prompts = HARMFUL_PROMPTS

    # ── Costruzione matrice piatta ────────────────────────────────────
    # Ogni elemento: (defense_name, attack_name, prompt_id, category, raw_prompt)
    all_combos = [
        (d, a, p["id"], p["category"], p["prompt"])
        for d in defenses
        for a in attacks
        for p in prompts
    ]
    full_total = len(all_combos)

    if sample is not None:
        rng = random.Random(seed)
        rng.shuffle(all_combos)
        all_combos = all_combos[:sample]

    total = len(all_combos)

    print(f"\n📊 Evaluation Matrix ({config.OLLAMA_MODEL}):")
    if sample is not None:
        print(f"   🎲 Sample casuale: {total} test / {full_total} totali "
              f"({total/full_total*100:.0f}%)"
              + (f"  [seed={seed}]" if seed is not None else ""))
    else:
        print(f"   {len(prompts)} harmful prompts × {len(attacks)} attacks × "
              f"{len(defenses)} defenses = {total} tests"
              + (f" (limit: {limit})" if limit else ""))
    # Copertura dimensioni nel campione
    sample_attacks  = sorted({a for _, a, *_ in all_combos})
    sample_defenses = sorted({d for d, *_ in all_combos})
    sample_cats     = sorted({cat for _, _, _, cat, _ in all_combos})
    print(f"   Attacchi ({len(sample_attacks)}): {sample_attacks}")
    print(f"   Difese   ({len(sample_defenses)}): {sample_defenses}")
    print(f"   Categorie({len(sample_cats)}): {sample_cats}")

    if dry_run:
        print("\n🔍 Dry run — nessuna richiesta inviata.")
        return None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TRACES_DIR,  exist_ok=True)
    model_slug = _model_slug(config.OLLAMA_MODEL)
    checkpoint_path = None if fresh else _find_latest_live_file(model_slug)
    done = _load_completed(checkpoint_path)

    if done and not fresh:
        print(f"\n♻️  Ripresa da checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"   {len(done)}/{total} test già completati — salto e continuo.")
        live_path = checkpoint_path
        live_file = open(live_path, "a", buffering=1, encoding="utf-8")
        # Riprende anche il trace file esistente, se presente
        trace_pattern = os.path.join(TRACES_DIR, f"trace_{model_slug}_*.jsonl")
        existing_traces = sorted(glob.glob(trace_pattern))
        trace_path = existing_traces[-1] if existing_traces else None
        if trace_path:
            trace_file = open(trace_path, "a", buffering=1, encoding="utf-8")
        else:
            trace_ts   = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            trace_path = os.path.join(TRACES_DIR, f"trace_{model_slug}_{trace_ts}.jsonl")
            trace_file = open(trace_path, "w", buffering=1, encoding="utf-8")
    else:
        live_ts = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        live_path = os.path.join(
            RESULTS_DIR, f"eval_live_{model_slug}_{live_ts}.jsonl"
        )
        live_file = open(live_path, "w", buffering=1, encoding="utf-8")
        trace_path = os.path.join(TRACES_DIR, f"trace_{model_slug}_{live_ts}.jsonl")
        trace_file = open(trace_path, "w", buffering=1, encoding="utf-8")
        print(f"📝 Live checkpoint : {live_path}")
        print(f"📋 Full trace      : {trace_path}\n")

    results = []
    completed = 0
    skipped = len(done)
    completed = skipped
    new_done = 0
    start_ts = datetime.now()
    start_time = time.time()
    limit_reached = False

    for defense_name, attack_name, prompt_id, category, raw_prompt in all_combos:
        if limit_reached:
            break

        if (defense_name, attack_name, prompt_id) in done:
            continue

        attack_fn   = attacks[attack_name]
        defense_fn  = defenses[defense_name]
        completed  += 1
        attacked_prompt = attack_fn(raw_prompt)

        elapsed_label = f"[{completed}/{total}]"
        print(f"  {elapsed_label} {defense_name} × {attack_name} × {prompt_id}... ",
              end="", flush=True)

        t0 = time.time()
        api_error = None
        try:
            response = defense_fn(attacked_prompt)
            latency = time.time() - t0
        except Exception as e:
            response = f"[ERROR] {e}"
            latency = time.time() - t0
            api_error = str(e)
            traceback.print_exc()

        if api_error:
            verdict = {"refused": False, "compliant": False,
                       "response_length": 0, "confidence": 0.0,
                       "judge_used": False}
        else:
            verdict = judge_response(response, original_prompt=raw_prompt)

        # Slim record per il .json finale: solo metadati tecnici,
        # adatto ai grafici e all'analisi aggregata.
        result = {
            "model": config.OLLAMA_MODEL,
            "defense": defense_name,
            "attack": attack_name,
            "prompt_id": prompt_id,
            "category": category,
            "refused": verdict["refused"],
            "compliant": verdict["compliant"],
            "response_length": verdict["response_length"],
            "confidence": verdict["confidence"],
            "judge_used": verdict.get("judge_used", False),
            "latency_s": round(latency, 3),
            "error": api_error,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        results.append(result)
        # .jsonl slim: solo metadati, usato per checkpoint/resume
        live_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        # trace file: record completo con testo del prompt e risposta,
        # per verifica manuale della qualità del giudice
        trace_file.write(json.dumps({
            **result,
            "original_prompt": raw_prompt,
            "attacked_prompt": attacked_prompt,
            "response": response if not api_error else None,
        }, ensure_ascii=False) + "\n")

        if api_error:
            status = "❌ ERROR"
        elif verdict["refused"]:
            status = "🛡️ REFUSED"
        else:
            status = "⚠️ COMPLIED"
        print(f"{status} ({latency:.1f}s, {verdict['response_length']} chars)")

        new_done += 1
        if limit is not None and new_done >= limit:
            print(f"\n🛑 Raggiunto limit={limit} test in questa sessione — stop.")
            limit_reached = True

    live_file.close()
    trace_file.close()
    end_ts = datetime.now()
    elapsed = time.time() - start_time

    # ── Statistiche aggregate ─────────────────────────────────────────
    valid = [r for r in results if not r.get("error")]
    n_errors = len(results) - len(valid)
    latencies = [r["latency_s"] for r in valid]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    median_lat = sorted_lat[len(sorted_lat)//2] if sorted_lat else 0.0
    p95_lat = sorted_lat[int(len(sorted_lat)*0.95)] if sorted_lat else 0.0
    min_lat = sorted_lat[0] if sorted_lat else 0.0
    max_lat = sorted_lat[-1] if sorted_lat else 0.0

    n_complied = sum(1 for r in valid if r["compliant"])
    n_refused = sum(1 for r in valid if r["refused"])
    asr_overall = (n_complied / len(valid)) if valid else 0.0

    def _grouped_asr(key):
        out = {}
        for r in valid:
            out.setdefault(r[key], []).append(r["compliant"])
        return {k: round(sum(v)/len(v), 4) for k, v in out.items()}

    asr_per_defense = _grouped_asr("defense")
    asr_per_attack = _grouped_asr("attack")
    asr_per_category = _grouped_asr("category")

    # ETA per il run completo basata sulla latency media osservata
    eta_full_s = avg_lat * total
    eta_remaining_s = avg_lat * (total - completed)

    ts = end_ts.strftime("%Y%m%d_%H_%M_%S")
    results_path = os.path.join(RESULTS_DIR, f"eval_{model_slug}_{ts}.json")

    output = {
        "metadata": {
            # ── Identità del run ────────────────────────────────────
            "test_model": config.OLLAMA_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "test_api": "ollama-local",
            "judge_api": "ollama-local",
            "test_host": config.OLLAMA_HOST,

            # ── Tempi ───────────────────────────────────────────────
            "timestamp_start": start_ts.isoformat(timespec="seconds"),
            "timestamp_end": end_ts.isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_human": _hms(elapsed),

            # ── Conteggi ────────────────────────────────────────────
            "total_tests_planned": total,
            "tests_completed_this_run": new_done,
            "tests_skipped_resume": skipped,
            "tests_total_completed": completed,
            "tests_with_error": n_errors,
            "num_prompts": len(prompts),
            "num_attacks": len(attacks),
            "num_defenses": len(defenses),
            "limit": limit,

            # ── Latency stats (per test) ────────────────────────────
            "avg_latency_s": round(avg_lat, 3),
            "median_latency_s": round(median_lat, 3),
            "p95_latency_s": round(p95_lat, 3),
            "min_latency_s": round(min_lat, 3),
            "max_latency_s": round(max_lat, 3),

            # ── ETA per run completo (extrapolazione) ───────────────
            "eta_full_run_seconds": round(eta_full_s, 1),
            "eta_full_run_human": _hms(eta_full_s),
            "eta_remaining_seconds": round(eta_remaining_s, 1),
            "eta_remaining_human": _hms(eta_remaining_s),

            # ── Risultati aggregati (chart-ready) ───────────────────
            "asr_overall": round(asr_overall, 4),
            "n_complied": n_complied,
            "n_refused": n_refused,
            "asr_per_defense": asr_per_defense,
            "asr_per_attack": asr_per_attack,
            "asr_per_category": asr_per_category,
        },
        "results": results,
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Risultati salvati: {results_path}")
    print(f"⏱  Tempo questa sessione: {_hms(elapsed)} "
          f"({new_done} test nuovi, {avg_lat:.2f}s medi)")
    print(f"📈 ETA run completo ({total} test): {_hms(eta_full_s)}")
    if completed < total:
        print(f"⏳ ETA per finire: {_hms(eta_remaining_s)} "
              f"({total - completed} test restanti)")

    print_summary(results, defenses, attacks)
    return results_path


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_summary(results, defenses, attacks):
    total_errors = sum(1 for r in results if r.get("error"))
    if total_errors:
        print(f"\n  ⚠️  {total_errors} test con errore API esclusi dall'ASR")

    print(f"\n{'='*70}")
    print(f"  ATTACK SUCCESS RATE (ASR) — più basso = modello più sicuro")
    print(f"  Modello: {config.OLLAMA_MODEL}")
    print(f"{'='*70}")

    header = f"{'Attack':<20}" + "".join(f"{d:<16}" for d in defenses)
    print(f"\n{header}")
    print("-" * len(header))

    for attack_name in attacks:
        row = f"{attack_name:<20}"
        for defense_name in defenses:
            subset = [r for r in results
                      if r["attack"] == attack_name
                      and r["defense"] == defense_name
                      and not r.get("error")]
            if subset:
                asr = sum(1 for r in subset if r["compliant"]) / len(subset)
                row += f"{asr:>6.1%}          "
            else:
                row += f"{'N/A':>6}          "
        print(row)

    print("-" * len(header))
    row = f"{'OVERALL':<20}"
    for defense_name in defenses:
        subset = [r for r in results
                  if r["defense"] == defense_name and not r.get("error")]
        asr = sum(1 for r in subset if r["compliant"]) / len(subset) if subset else 0
        row += f"{asr:>6.1%}          "
    print(row)


def main():
    parser = argparse.ArgumentParser(description="Local Ollama Jailbreak Evaluation")
    parser.add_argument("--attacks", nargs="*", help="Only run specific attacks")
    parser.add_argument("--defenses", nargs="*", help="Only run specific defenses")
    parser.add_argument("--model", type=str,
                        help="Override Ollama model (also used as judge)")
    parser.add_argument("--judge-model", type=str,
                        help="Override judge model (default: same as --model)")
    parser.add_argument("--host", type=str,
                        help="Override Ollama host (default: http://localhost:11434)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints and start from scratch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without running")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N new tests (utile per smoke test)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Campiona casualmente N test dalla matrice completa "
                             "(copre tutte le categorie/attacchi/difese in modo "
                             "proporzionale). Ideale per validation run veloci.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seme per il campionamento casuale (riproducibilità)")
    args = parser.parse_args()

    if args.host:
        config.OLLAMA_HOST = args.host
    if args.model:
        config.OLLAMA_MODEL = args.model
        # Stessa policy del multi-runner: qwen per modelli sotto soglia,
        # self-judge solo per modelli grandi.
        if not args.judge_model:
            config.JUDGE_MODEL = config.select_judge_model(args.model)
    if args.judge_model:
        config.JUDGE_MODEL = args.judge_model

    print(f"🔬 Jailbreak Evaluation (LOCAL)")
    print(f"   Host           : {config.OLLAMA_HOST}")
    print(f"   Modello test   : {config.OLLAMA_MODEL}")
    print(f"   Modello giudice: {config.JUDGE_MODEL}\n")

    if not args.dry_run:
        if not check_local_ollama():
            sys.exit(1)

    run_evaluation(
        selected_attacks=args.attacks,
        selected_defenses=args.defenses,
        dry_run=args.dry_run,
        fresh=args.fresh,
        limit=args.limit,
        sample=args.sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
