#!/usr/bin/env python3
"""
Jailbreak Evaluation — Main Runner

Sends harmful prompts × attack templates × defense strategies to an LLM
via Google Generative AI API, judges each response, and saves results + charts.

Usage:
    python run_evaluation.py                     # full evaluation
    python run_evaluation.py --attacks DAN AIM   # only specific attacks
    python run_evaluation.py --dry-run            # show plan without running
"""
import argparse
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


def check_api_connection() -> bool:
    """Verify the Google API key is set and the model is accessible."""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        # Lightweight ping: generate a single token
        client.models.generate_content(
            model=config.MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text="Hi")])],
            config=types.GenerateContentConfig(max_output_tokens=1),
        )
        print(f"✅ Google AI API raggiungibile")
        print(f"✅ Modello: {config.MODEL}")
        return True
    except Exception as e:
        print(f"❌ Errore connessione Google AI API: {e}")
        return False


def run_evaluation(selected_attacks=None, selected_defenses=None, dry_run=False):
    """Run the full evaluation matrix."""
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

    results = []
    completed = 0
    start_time = time.time()

    # Live JSONL file: one result per line, written immediately after each test.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    live_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    live_path = os.path.join(RESULTS_DIR, f"eval_live_{live_ts}.jsonl")
    live_file = open(live_path, "w", buffering=1)  # line-buffered
    print(f"📝 Live results: {live_path}\n")

    for defense_name, defense_fn in defenses.items():
        for attack_name, attack_fn in attacks.items():
            for prompt_info in prompts:
                completed += 1
                prompt_id = prompt_info["id"]
                category = prompt_info["category"]
                raw_prompt = prompt_info["prompt"]

                # Apply attack template
                attacked_prompt = attack_fn(raw_prompt)

                # Apply defense and get response
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

                # Judge — errors are excluded from COMPLIED/REFUSED stats
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
            "model": config.MODEL,
            "api": "google-generativeai",
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
    parser.add_argument("--model", type=str, help="Override model name (e.g. gemma3:4b)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    args = parser.parse_args()

    if args.model:
        config.MODEL = args.model

    print(f"🔬 Jailbreak Evaluation (Google AI)")
    print(f"   Model: {config.MODEL}\n")

    if not args.dry_run:
        if not check_api_connection():
            print("\n💡 Controlla che GOOGLE_API_KEY_FREE sia valida nel file .env")
            sys.exit(1)

    results_path = run_evaluation(
        selected_attacks=args.attacks,
        selected_defenses=args.defenses,
        dry_run=args.dry_run,
    )

    if results_path:
        print(f"\n📊 Generazione automatica dei grafici...")
        from generate_charts import generate_charts
        generate_charts(results_path)


if __name__ == "__main__":
    main()
