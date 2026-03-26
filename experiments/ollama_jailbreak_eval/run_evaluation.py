#!/usr/bin/env python3
"""
Ollama Jailbreak Evaluation — Main Runner

Sends harmful prompts × attack templates × defense strategies to an LLM
via Ollama, judges each response, and saves structured results + charts.

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


def check_ollama_connection() -> bool:
    """Verify Ollama is reachable and the model is available."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        found = any(config.MODEL in m or m.startswith(config.MODEL.split(":")[0]) for m in models)
        if found:
            print(f"✅ Ollama connesso: {OLLAMA_HOST}")
            print(f"✅ Modello trovato: {config.MODEL}")
            return True
        else:
            print(f"❌ Modello '{config.MODEL}' non trovato. Modelli disponibili:")
            for m in models:
                print(f"   - {m}")
            return False
    except Exception as e:
        print(f"❌ Impossibile connettersi a Ollama ({OLLAMA_HOST}): {e}")
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
                try:
                    response = defense_fn(attacked_prompt)
                    latency = time.time() - t0
                except Exception as e:
                    response = f"[ERROR] {e}"
                    latency = time.time() - t0
                    traceback.print_exc()

                # Judge
                verdict = judge_response(response)

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
                    "latency_s": round(latency, 2),
                }
                results.append(result)

                status = "🛡️ REFUSED" if verdict["refused"] else "⚠️ COMPLIED"
                print(f"{status} ({latency:.1f}s, {verdict['response_length']} chars)")

    elapsed = time.time() - start_time

    # Save raw results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"eval_{ts}.json")

    output = {
        "metadata": {
            "model": config.MODEL,
            "ollama_host": OLLAMA_HOST,
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
    """Print ASR summary table."""
    print(f"\n{'='*70}")
    print(f"  ATTACK SUCCESS RATE (ASR) — più basso = modello più sicuro")
    print(f"{'='*70}")

    header = f"{'Attack':<20}" + "".join(f"{d:<16}" for d in defenses)
    print(f"\n{header}")
    print("-" * len(header))

    for attack_name in attacks:
        row = f"{attack_name:<20}"
        for defense_name in defenses:
            subset = [r for r in results
                      if r["attack"] == attack_name and r["defense"] == defense_name]
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
        subset = [r for r in results if r["defense"] == defense_name]
        asr = sum(1 for r in subset if r["compliant"]) / len(subset) if subset else 0
        row += f"{asr:>6.1%}          "
    print(row)


def main():
    parser = argparse.ArgumentParser(description="Ollama Jailbreak Evaluation")
    parser.add_argument("--attacks", nargs="*", help="Only run specific attacks")
    parser.add_argument("--defenses", nargs="*", help="Only run specific defenses")
    parser.add_argument("--model", type=str, help="Override model name (e.g. gemma3:4b)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    args = parser.parse_args()

    if args.model:
        config.MODEL = args.model

    print(f"🔬 Ollama Jailbreak Evaluation")
    print(f"   Model: {config.MODEL} @ {OLLAMA_HOST}\n")

    if not args.dry_run:
        if not check_ollama_connection():
            print("\n💡 Assicurati che Ollama sia in esecuzione:")
            print(f"   ollama serve")
            print(f"   ollama pull {config.MODEL}")
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
