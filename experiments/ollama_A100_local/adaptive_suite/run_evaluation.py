#!/usr/bin/env python3
"""
Adaptive suite runner.

Runs a result-mined matrix:
    40 prompts x 10 attacks x 5 defenses = 2000 tests per model

Outputs are compatible with the parent eval_*.json schema but are written under
adaptive_suite/{results,traces,logs}.
"""
from __future__ import annotations

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
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(PARENT))
sys.stdout.reconfigure(line_buffering=True)

import config
from attacks import ATTACKS
from defenses import DEFENSES
from judge import judge_response
from prompts import HARMFUL_PROMPTS

RESULTS_DIR = HERE / "results"
TRACES_DIR = HERE / "traces"


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _find_latest_live_file(model_slug: str) -> str | None:
    files = sorted(glob.glob(str(RESULTS_DIR / f"eval_live_{model_slug}_*.jsonl")))
    return files[-1] if files else None


def _load_completed(live_path: str | None) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not live_path or not os.path.exists(live_path):
        return done
    with open(live_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add((row["defense"], row["attack"], row["prompt_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def check_local_ollama() -> bool:
    import requests

    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        installed = {m["name"] for m in resp.json().get("models", [])}
        print(f"Ollama daemon reachable: {config.OLLAMA_HOST}")
    except Exception as exc:
        print(f"Ollama daemon not reachable on {config.OLLAMA_HOST}: {exc}")
        return False

    if config.OLLAMA_MODEL not in installed:
        print(f"Model not installed: {config.OLLAMA_MODEL}")
        print(f"Available models: {sorted(installed) or '(none)'}")
        return False

    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "think": False,
                "options": {"num_predict": 1},
            },
            timeout=120,
        )
        resp.raise_for_status()
        print(f"Test model : {config.OLLAMA_MODEL}")
        print(f"Judge model: {config.JUDGE_MODEL}")
        return True
    except Exception as exc:
        print(f"Model round-trip failed: {exc}")
        return False


def run_evaluation(
    selected_attacks: list[str] | None = None,
    selected_defenses: list[str] | None = None,
    dry_run: bool = False,
    fresh: bool = False,
    limit: int | None = None,
    sample: int | None = None,
    seed: int | None = None,
) -> str | None:
    attacks = {k: v for k, v in ATTACKS.items() if not selected_attacks or k in selected_attacks}
    defenses = {k: v for k, v in DEFENSES.items() if not selected_defenses or k in selected_defenses}
    prompts = HARMFUL_PROMPTS

    all_combos = [
        (defense_name, attack_name, p["id"], p["category"], p["prompt"], p.get("parent_id"))
        for defense_name in defenses
        for attack_name in attacks
        for p in prompts
    ]
    full_total = len(all_combos)
    if sample is not None:
        rng = random.Random(seed)
        rng.shuffle(all_combos)
        all_combos = all_combos[:sample]

    total = len(all_combos)
    print("\nAdaptive Evaluation Matrix:")
    print(f"  model   : {config.OLLAMA_MODEL}")
    print(f"  prompts : {len(prompts)}")
    print(f"  attacks : {len(attacks)}")
    print(f"  defenses: {len(defenses)}")
    print(f"  tests   : {total} / {full_total}" + (f" [sample seed={seed}]" if sample else ""))
    print(f"  attacks : {sorted(attacks)}")
    print(f"  defenses: {sorted(defenses)}")

    if dry_run:
        print("\nDry run: no requests sent.")
        return None

    RESULTS_DIR.mkdir(exist_ok=True)
    TRACES_DIR.mkdir(exist_ok=True)

    model_slug = _model_slug(config.OLLAMA_MODEL)
    checkpoint_path = None if fresh else _find_latest_live_file(model_slug)
    done = _load_completed(checkpoint_path)

    if done and not fresh:
        print(f"\nResume checkpoint: {Path(checkpoint_path).name}")
        live_path = Path(checkpoint_path)
        live_file = open(live_path, "a", buffering=1, encoding="utf-8")
        trace_files = sorted(TRACES_DIR.glob(f"trace_{model_slug}_*.jsonl"))
        trace_path = trace_files[-1] if trace_files else TRACES_DIR / f"trace_{model_slug}_{datetime.now():%Y%m%d_%H_%M_%S}.jsonl"
        trace_file = open(trace_path, "a", buffering=1, encoding="utf-8")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        live_path = RESULTS_DIR / f"eval_live_{model_slug}_{ts}.jsonl"
        trace_path = TRACES_DIR / f"trace_{model_slug}_{ts}.jsonl"
        live_file = open(live_path, "w", buffering=1, encoding="utf-8")
        trace_file = open(trace_path, "w", buffering=1, encoding="utf-8")
        print(f"Live checkpoint: {live_path}")
        print(f"Full trace     : {trace_path}\n")

    results: list[dict] = []
    completed = len(done)
    skipped = len(done)
    new_done = 0
    start_ts = datetime.now()
    start_time = time.time()

    try:
        for defense_name, attack_name, prompt_id, category, raw_prompt, parent_id in all_combos:
            if (defense_name, attack_name, prompt_id) in done:
                continue
            if limit is not None and new_done >= limit:
                break

            completed += 1
            attack_fn = attacks[attack_name]
            defense_fn = defenses[defense_name]
            attacked_prompt = attack_fn(raw_prompt)

            print(f"  [{completed}/{total}] {defense_name} x {attack_name} x {prompt_id}... ", end="", flush=True)
            t0 = time.time()
            api_error = None
            try:
                response = defense_fn(attacked_prompt)
                latency = time.time() - t0
            except Exception as exc:
                response = f"[ERROR] {exc}"
                latency = time.time() - t0
                api_error = str(exc)
                traceback.print_exc()

            if api_error:
                verdict = {
                    "refused": False,
                    "compliant": False,
                    "response_length": 0,
                    "confidence": 0.0,
                    "judge_used": False,
                }
            else:
                verdict = judge_response(response, original_prompt=raw_prompt)

            result = {
                "model": config.OLLAMA_MODEL,
                "defense": defense_name,
                "attack": attack_name,
                "prompt_id": prompt_id,
                "parent_prompt_id": parent_id,
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
            live_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            trace_file.write(json.dumps({
                **result,
                "original_prompt": raw_prompt,
                "attacked_prompt": attacked_prompt,
                "response": response if not api_error else None,
            }, ensure_ascii=False) + "\n")

            if api_error:
                status = "ERROR"
            elif verdict["refused"]:
                status = "REFUSED"
            else:
                status = "COMPLIED"
            print(f"{status} ({latency:.1f}s, {verdict['response_length']} chars)")
            new_done += 1
    finally:
        live_file.close()
        trace_file.close()

    end_ts = datetime.now()
    elapsed = time.time() - start_time
    valid = [r for r in results if not r.get("error")]
    latencies = [r["latency_s"] for r in valid]
    sorted_lat = sorted(latencies)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
    p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0.0
    min_lat = sorted_lat[0] if sorted_lat else 0.0
    max_lat = sorted_lat[-1] if sorted_lat else 0.0
    n_complied = sum(1 for r in valid if r["compliant"])
    n_refused = sum(1 for r in valid if r["refused"])
    asr_overall = n_complied / len(valid) if valid else 0.0

    def grouped_asr(key: str) -> dict[str, float]:
        groups: dict[str, list[bool]] = {}
        for row in valid:
            groups.setdefault(row[key], []).append(row["compliant"])
        return {k: round(sum(v) / len(v), 4) for k, v in groups.items()}

    ts = end_ts.strftime("%Y%m%d_%H_%M_%S")
    results_path = RESULTS_DIR / f"eval_{model_slug}_{ts}.json"
    output = {
        "metadata": {
            "suite": "adaptive_suite_v2",
            "suite_source": "mined_from_previous_eval_json_v2",
            "test_model": config.OLLAMA_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "test_api": "ollama-local",
            "judge_api": "ollama-local",
            "test_host": config.OLLAMA_HOST,
            "timestamp_start": start_ts.isoformat(timespec="seconds"),
            "timestamp_end": end_ts.isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_human": _hms(elapsed),
            "total_tests_planned": total,
            "tests_completed_this_run": new_done,
            "tests_skipped_resume": skipped,
            "tests_total_completed": completed,
            "tests_with_error": len(results) - len(valid),
            "num_prompts": len(prompts),
            "num_attacks": len(attacks),
            "num_defenses": len(defenses),
            "limit": limit,
            "avg_latency_s": round(avg_lat, 3),
            "median_latency_s": round(median_lat, 3),
            "p95_latency_s": round(p95_lat, 3),
            "min_latency_s": round(min_lat, 3),
            "max_latency_s": round(max_lat, 3),
            "eta_full_run_seconds": round(avg_lat * total, 1),
            "eta_full_run_human": _hms(avg_lat * total),
            "eta_remaining_seconds": round(avg_lat * max(total - completed, 0), 1),
            "eta_remaining_human": _hms(avg_lat * max(total - completed, 0)),
            "asr_overall": round(asr_overall, 4),
            "n_complied": n_complied,
            "n_refused": n_refused,
            "asr_per_defense": grouped_asr("defense"),
            "asr_per_attack": grouped_asr("attack"),
            "asr_per_category": grouped_asr("category"),
        },
        "results": results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {results_path}")
    print(f"Elapsed: {_hms(elapsed)} ({new_done} new tests, {avg_lat:.2f}s avg)")
    print_summary(results, defenses, attacks)
    return str(results_path)


def print_summary(results: list[dict], defenses: dict, attacks: dict) -> None:
    valid = [r for r in results if not r.get("error")]
    errors = len(results) - len(valid)
    if errors:
        print(f"\n{errors} tests with API errors excluded from ASR")

    print("\n" + "=" * 72)
    print(f"Adaptive ASR summary - model: {config.OLLAMA_MODEL}")
    print("=" * 72)
    header = f"{'Attack':<30}" + "".join(f"{d:<16}" for d in defenses)
    print(header)
    print("-" * len(header))
    for attack_name in attacks:
        row = f"{attack_name:<30}"
        for defense_name in defenses:
            subset = [
                r for r in valid
                if r["attack"] == attack_name and r["defense"] == defense_name
            ]
            asr = sum(1 for r in subset if r["compliant"]) / len(subset) if subset else 0.0
            row += f"{asr:>6.1%}          "
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Ollama jailbreak evaluation")
    parser.add_argument("--attacks", nargs="*", help="Only run specific adaptive attacks")
    parser.add_argument("--defenses", nargs="*", help="Only run specific defenses")
    parser.add_argument("--model", type=str, help="Override Ollama model")
    parser.add_argument("--judge-model", type=str, help="Override judge model")
    parser.add_argument("--host", type=str, help="Override Ollama host")
    parser.add_argument("--fresh", action="store_true", help="Ignore adaptive checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Show matrix without running")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N new tests")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N tests")
    parser.add_argument("--seed", type=int, default=None, help="Random sample seed")
    args = parser.parse_args()

    if args.host:
        config.OLLAMA_HOST = args.host
    if args.model:
        config.OLLAMA_MODEL = args.model
        if not args.judge_model:
            config.JUDGE_MODEL = args.model
    if args.judge_model:
        config.JUDGE_MODEL = args.judge_model

    print("Adaptive Jailbreak Evaluation")
    print(f"  Host : {config.OLLAMA_HOST}")
    print(f"  Model: {config.OLLAMA_MODEL}")
    print(f"  Judge: {config.JUDGE_MODEL}")

    if not args.dry_run and not check_local_ollama():
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
