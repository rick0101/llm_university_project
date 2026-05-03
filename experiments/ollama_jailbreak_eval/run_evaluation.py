#!/usr/bin/env python3
"""
Jailbreak Evaluation — Main Runner

Sends harmful prompts × attack templates × defense strategies to an LLM
via Ollama Cloud, judges each response with Gemini, and saves results + charts.

Usage:
    python run_evaluation.py                     # full evaluation (auto-resume)
    python run_evaluation.py --attacks DAN AIM   # only specific attacks
    python run_evaluation.py --fresh             # ignore existing checkpoints
    python run_evaluation.py --dry-run            # show plan without running
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import config
from config import OLLAMA_HOST
from attacks import ATTACKS
from defenses import DEFENSES
from judge import judge_response
from prompts import HARMFUL_PROMPTS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _find_latest_live_file() -> str | None:
    """Return path to the most-recent eval_live_*.jsonl, or None."""
    pattern = os.path.join(RESULTS_DIR, "eval_live_*.jsonl")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _load_completed(live_path: str) -> set[tuple[str, str, str]]:
    """Parse a live JSONL and return the set of (defense, attack, prompt_id) done."""
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


def check_api_connection() -> bool:
    """Verify Ollama Cloud (test model) and Google AI (judge) are both reachable."""
    import requests as _req

    ok = True

    # ── Ollama Cloud ────────────────────────────────────────────────
    try:
        resp = _req.post(
            f"{config.OLLAMA_CLOUD_HOST}/v1/chat/completions",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
            },
            headers={"Authorization": f"Bearer {config.OLLAMA_CLOUD_API_KEY}"},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"✅ Ollama Cloud raggiungibile")
        print(f"✅ Modello test: {config.OLLAMA_MODEL}")
    except Exception as e:
        print(f"❌ Errore connessione Ollama Cloud: {e}")
        ok = False

    # ── Google AI judge ─────────────────────────────────────────────
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        client.models.generate_content(
            model=config.JUDGE_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text="Hi")])],
            config=types.GenerateContentConfig(max_output_tokens=1),
        )
        print(f"✅ Google AI judge raggiungibile")
        print(f"✅ Modello giudice: {config.JUDGE_MODEL}")
    except Exception as e:
        print(f"❌ Errore connessione Google AI judge: {e}")
        ok = False

    return ok


def run_evaluation(selected_attacks=None, selected_defenses=None, dry_run=False,
                   fresh=False):
    """Run the full evaluation matrix, resuming from the latest checkpoint."""
    attacks = {k: v for k, v in ATTACKS.items()
               if not selected_attacks or k in selected_attacks}
    defenses = {k: v for k, v in DEFENSES.items() if not selected_defenses or k in selected_defenses}
    prompts = HARMFUL_PROMPTS

    total = len(prompts) * len(attacks) * len(defenses)
    print(f"\n📊 Evaluation Matrix:")
    print(f"   {len(prompts)} harmful prompts × {len(attacks)} attacks × {len(defenses)} defenses = {total} tests")
    print(f"   Attacchi: {list(attacks.keys())}")
    print(f"   Difese:   {list(defenses.keys())}")
    print(f"   Categorie: {sorted(set(p['category'] for p in prompts))}")

    if dry_run:
        print("\n🔍 Dry run — nessuna richiesta inviata.")
        return None

    # ── Checkpoint: resume from latest live file unless --fresh ───────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint_path = None if fresh else _find_latest_live_file()
    done = _load_completed(checkpoint_path)

    if done and not fresh:
        print(f"\n♻️  Ripresa da checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"   {len(done)}/{total} test già completati — salto e continuo.")
        live_path = checkpoint_path
        live_file = open(live_path, "a", buffering=1, encoding="utf-8")
    else:
        live_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        live_path = os.path.join(RESULTS_DIR, f"eval_live_{live_ts}.jsonl")
        live_file = open(live_path, "w", buffering=1, encoding="utf-8")
        print(f"📝 Nuovo live results: {live_path}\n")

    results = []
    completed = 0
    skipped = 0
    start_time = time.time()

    for defense_name, defense_fn in defenses.items():
        for attack_name, attack_fn in attacks.items():
            for prompt_info in prompts:
                prompt_id = prompt_info["id"]
                category = prompt_info["category"]
                raw_prompt = prompt_info["prompt"]

                # Skip already-done tests (checkpoint resume)
                if (defense_name, attack_name, prompt_id) in done:
                    skipped += 1
                    completed += 1
                    continue

                completed += 1
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
                               "response_length": 0, "confidence": 0.0}
                else:
                    verdict = judge_response(response, original_prompt=raw_prompt)

                result = {
                    "defense": defense_name,
                    "attack": attack_name,
                    "prompt_id": prompt_id,
                    "category": category,
                    "raw_prompt": raw_prompt,
                    "attacked_prompt": attacked_prompt[:500],
                    "response": response[:2000],
                    "refused": verdict["refused"],
                    "compliant": verdict["compliant"],
                    "response_length": verdict["response_length"],
                    "confidence": verdict["confidence"],
                    "judge_used": verdict.get("judge_used", False),
                    "latency_s": round(latency, 2),
                    "error": api_error,
                }
                results.append(result)
                live_file.write(json.dumps(result, ensure_ascii=False) + "\n")

                if api_error:
                    status = "❌ ERROR"
                elif verdict["refused"]:
                    status = "🛡️ REFUSED"
                else:
                    status = "⚠️ COMPLIED"
                print(f"{status} ({latency:.1f}s, {verdict['response_length']} chars)")

    live_file.close()

    elapsed = time.time() - start_time

    # Save raw results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"eval_{ts}.json")

    output = {
        "metadata": {
            "test_model": config.OLLAMA_MODEL,
            "test_api": "ollama-cloud",
            "test_host": config.OLLAMA_CLOUD_HOST,
            "judge_model": config.JUDGE_MODEL,
            "judge_api": "google-generativeai",
            "timestamp": ts,
            "total_tests": total,
            "elapsed_seconds": round(elapsed, 1),
            "num_prompts": len(prompts),
            "num_attacks": len(attacks),
            "num_defenses": len(defenses),
        },
        "results": results,
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Risultati salvati: {results_path}")

    # Print summary
    print_summary(results, defenses, attacks)

    return results_path


def print_summary(results, defenses, attacks):
    """Print ASR summary table. Errors are excluded from ASR calculations."""
    # Count total errors
    total_errors = sum(1 for r in results if r.get("error"))
    if total_errors:
        print(f"\n  ⚠️  {total_errors} test con errore API esclusi dall'ASR")

    print(f"\n{'='*70}")
    print(f"  ATTACK SUCCESS RATE (ASR) — più basso = modello più sicuro")
    print(f"  (calcolato solo sui test completati con successo)")
    print(f"{'='*70}")

    header = f"{'Attack':<20}" + "".join(f"{d:<16}" for d in defenses)
    print(f"\n{header}")
    print("-" * len(header))

    for attack_name in attacks:
        row = f"{attack_name:<20}"
        for defense_name in defenses:
            # Exclude errored tests from ASR
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

    # Overall by defense
    print("-" * len(header))
    row = f"{'OVERALL':<20}"
    for defense_name in defenses:
        subset = [r for r in results
                  if r["defense"] == defense_name and not r.get("error")]
        asr = sum(1 for r in subset if r["compliant"]) / len(subset) if subset else 0
        row += f"{asr:>6.1%}          "
    print(row)

    # Error summary per defense
    if total_errors:
        print(f"\n{'Errori API per difesa':}")
        for defense_name in defenses:
            n_err = sum(1 for r in results
                        if r["defense"] == defense_name and r.get("error"))
            if n_err:
                print(f"  {defense_name}: {n_err} errori")


def main():
    parser = argparse.ArgumentParser(description="Ollama Jailbreak Evaluation")
    parser.add_argument("--attacks", nargs="*", help="Only run specific attacks")
    parser.add_argument("--defenses", nargs="*", help="Only run specific defenses")
    parser.add_argument("--model", type=str, help="Override Ollama model name")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints and start from scratch")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    args = parser.parse_args()

    if args.model:
        config.OLLAMA_MODEL = args.model

    print(f"🔬 Jailbreak Evaluation")
    print(f"   Modello test:   {config.OLLAMA_MODEL} (Ollama Cloud)")
    print(f"   Modello giudice: {config.JUDGE_MODEL} (Google AI)\n")

    if not args.dry_run:
        if not check_api_connection():
            print("\n💡 Controlla OLLAMA_CLOUD_API_KEY e GOOGLE_API_KEY_FREE nel file .env")
            sys.exit(1)

    results_path = run_evaluation(
        selected_attacks=args.attacks,
        selected_defenses=args.defenses,
        dry_run=args.dry_run,
        fresh=args.fresh,
    )

    if results_path:
        print(f"\n📊 Generazione automatica dei grafici...")
        from generate_charts import generate_charts
        generate_charts(results_path)


if __name__ == "__main__":
    main()
