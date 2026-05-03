#!/usr/bin/env python3
"""
OpenRouter Jailbreak Cost-Estimation Test
==========================================
Runs the 10 longest attack+prompt combinations (all Many-Shot) against two
models via OpenRouter and judges each response with the Gemini judge.

Purpose: estimate real API cost before running the full evaluation matrix.

Usage:
    python test_modello_openrouter.py                   # both models
    python test_modello_openrouter.py --model1-only     # gemma only
    python test_modello_openrouter.py --model2-only     # qwen only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow imports from this directory
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI

from attacks import ATTACKS
from judge import judge_response
from prompts import HARMFUL_PROMPTS

# ── OpenRouter configuration ──────────────────────────────────────────────────

OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_1 = "google/gemma-4-31b-it"   # Gemma 4 31B on OpenRouter
MODEL_2 = "qwen/qwen3.6-plus"       # Qwen3.6 Plus on OpenRouter

MAX_TOKENS = 300
TEMPERATURE = 0.7
TOP_N = 10                           # number of attack combos to test

RESULTS_DIR = Path(__file__).parent / "results"

# ── OpenRouter client ─────────────────────────────────────────────────────────

_or_client: OpenAI | None = None


def get_openrouter_client() -> OpenAI:
    global _or_client
    if _or_client is None:
        _or_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/rick0101/llm_university_project",
                "X-Title": "LLM Jailbreak Eval",
            },
        )
    return _or_client


# ── Select top-N longest attack+prompt combinations ───────────────────────────

def get_top_n_combos(n: int = TOP_N) -> list[dict]:
    """Return the N attack×prompt pairs that produce the longest attacked_prompt."""
    combos = []
    for p in HARMFUL_PROMPTS:
        for attack_name, attack_fn in ATTACKS.items():
            attacked = attack_fn(p["prompt"])
            combos.append({
                "attack": attack_name,
                "prompt_id": p["id"],
                "category": p["category"],
                "raw_prompt": p["prompt"],
                "attacked_prompt": attacked,
                "attacked_len": len(attacked),
            })
    combos.sort(key=lambda x: x["attacked_len"], reverse=True)
    return combos[:n]


# ── Single inference via OpenRouter ───────────────────────────────────────────

def call_openrouter(model: str, prompt: str) -> tuple[str, dict]:
    """
    Send prompt to model via OpenRouter.
    Returns (response_text, usage_dict).
    usage_dict keys: prompt_tokens, completion_tokens, total_tokens
    """
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    text = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    return text, usage


# ── Run test for a single model ───────────────────────────────────────────────

def run_model_test(model: str, combos: list[dict], label: str) -> list[dict]:
    """Run all combos against one model and return result records."""
    print(f"\n{'='*70}")
    print(f"  🤖  {label} — {model}")
    print(f"{'='*70}")

    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, combo in enumerate(combos, 1):
        print(
            f"  [{i:2}/{len(combos)}] {combo['attack']:<20} × {combo['prompt_id']}"
            f"  (len={combo['attacked_len']}) ... ",
            end="",
            flush=True,
        )

        t0 = time.time()
        api_error = None
        response_text = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            response_text, usage = call_openrouter(model, combo["attacked_prompt"])
            latency = time.time() - t0
        except Exception as e:
            response_text = f"[ERROR] {e}"
            latency = time.time() - t0
            api_error = str(e)

        if api_error:
            verdict = {"refused": False, "compliant": False,
                       "response_length": 0, "confidence": 0.0, "judge_used": False}
        else:
            verdict = judge_response(response_text, original_prompt=combo["raw_prompt"])

        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]

        if api_error:
            status = "❌ ERROR"
        elif verdict["refused"]:
            status = "🛡️  REFUSED"
        else:
            status = "⚠️  COMPLIED"

        print(
            f"{status}  "
            f"({latency:.1f}s, {verdict['response_length']}c, "
            f"in={usage['prompt_tokens']} out={usage['completion_tokens']} tok)"
        )

        results.append({
            "model": model,
            "attack": combo["attack"],
            "prompt_id": combo["prompt_id"],
            "category": combo["category"],
            "raw_prompt": combo["raw_prompt"],
            "attacked_prompt_len": combo["attacked_len"],
            "response": response_text[:2000],
            "refused": verdict["refused"],
            "compliant": verdict["compliant"],
            "response_length": verdict["response_length"],
            "confidence": verdict["confidence"],
            "judge_used": verdict.get("judge_used", False),
            "latency_s": round(latency, 2),
            "usage": usage,
            "error": api_error,
        })

        # Polite rate-limit buffer
        if i < len(combos):
            time.sleep(1.0)

    refused_count = sum(1 for r in results if r["refused"])
    complied_count = sum(1 for r in results if r["compliant"])
    error_count = sum(1 for r in results if r["error"])

    print(f"\n  📊 Summary: {refused_count} refused | {complied_count} complied | {error_count} errors")
    print(f"  🪙 Tokens — prompt: {total_prompt_tokens:,}  completion: {total_completion_tokens:,}  total: {total_prompt_tokens+total_completion_tokens:,}")
    asr = complied_count / max(len(results) - error_count, 1)
    print(f"  🎯 ASR (Attack Success Rate): {asr:.0%}")

    return results


# ── Save and print final report ───────────────────────────────────────────────

def print_cost_summary(all_results: list[dict]) -> None:
    """Print a per-model token/cost summary to help estimate charges."""
    print(f"\n{'='*70}")
    print("  💰  TOKEN USAGE SUMMARY (for cost estimation)")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'Prompt tok':>12} {'Compl. tok':>12} {'Total tok':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")

    by_model: dict[str, dict] = {}
    for r in all_results:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"pt": 0, "ct": 0}
        by_model[m]["pt"] += r["usage"]["prompt_tokens"]
        by_model[m]["ct"] += r["usage"]["completion_tokens"]

    for model, tok in by_model.items():
        total = tok["pt"] + tok["ct"]
        short = model.split("/")[-1][:33]
        print(f"  {short:<35} {tok['pt']:>12,} {tok['ct']:>12,} {total:>12,}")

    grand_pt = sum(v["pt"] for v in by_model.values())
    grand_ct = sum(v["ct"] for v in by_model.values())
    print(f"  {'TOTAL':<35} {grand_pt:>12,} {grand_ct:>12,} {grand_pt+grand_ct:>12,}")
    print(f"\n  ℹ️  Check https://openrouter.ai/models for per-token pricing.")
    print(f"  ℹ️  Judge calls (Gemini) use your Google AI Free tier key.\n")


def save_results(all_results: list[dict], combos: list[dict]) -> str:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"openrouter_test_{ts}.json"

    models_tested = sorted(set(r["model"] for r in all_results))
    output = {
        "metadata": {
            "timestamp": ts,
            "models_tested": models_tested,
            "num_combos": len(combos),
            "selection": "top-10 longest attacked_prompt",
            "attack_used": combos[0]["attack"] if combos else "",
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "judge": "gemini-2.5-flash (Google AI)",
        },
        "results": all_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return str(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenRouter Jailbreak Cost-Estimation Test")
    parser.add_argument("--model1-only", action="store_true", help=f"Only test {MODEL_1}")
    parser.add_argument("--model2-only", action="store_true", help=f"Only test {MODEL_2}")
    parser.add_argument("--model1", default=MODEL_1, help="Override first model slug")
    parser.add_argument("--model2", default=MODEL_2, help="Override second model slug")
    parser.add_argument("--top-n", type=int, default=TOP_N, help="Number of combos to test")
    args = parser.parse_args()

    model1 = args.model1
    model2 = args.model2

    combos = get_top_n_combos(args.top_n)

    print(f"\n🔬 OpenRouter Jailbreak Cost-Estimation Test")
    print(f"   Selezione: top-{args.top_n} combinazioni per lunghezza attacked_prompt")
    print(f"   Attacco usato: {combos[0]['attack']!r} (tutti i top-{args.top_n} sono questo)")
    print(f"   Prompt IDs: {[c['prompt_id'] for c in combos]}")
    print(f"   Lunghezze: {combos[0]['attacked_len']} → {combos[-1]['attacked_len']} chars")

    all_results = []

    if not args.model2_only:
        res1 = run_model_test(model1, combos, "MODEL 1")
        all_results.extend(res1)

    if not args.model1_only:
        res2 = run_model_test(model2, combos, "MODEL 2")
        all_results.extend(res2)

    print_cost_summary(all_results)

    path = save_results(all_results, combos)
    print(f"💾 Risultati salvati: {path}")


if __name__ == "__main__":
    main()
