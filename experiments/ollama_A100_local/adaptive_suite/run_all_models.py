#!/usr/bin/env python3
"""
Sequential multi-model runner for adaptive_suite.

Reads the canonical ../models.txt by default. Results are written only under
adaptive_suite/results.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(PARENT))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import config
from defenses import DEFENSES

RESULTS_DIR = HERE / "results"
DEFAULT_MODELS_FILE = PARENT / "models.txt"
SELF_JUDGE_THRESHOLD_B = 50
EXTERNAL_JUDGE = "qwen3.6:35b"


def parse_models_file(path: str | Path) -> list[str]:
    models: list[str] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if line:
                models.append(line)
    return models


def select_judge_model(model: str) -> str:
    matches = re.findall(r"(\d+)b", model.lower())
    if matches and max(int(m) for m in matches) >= SELF_JUDGE_THRESHOLD_B:
        return model
    return EXTERNAL_JUDGE


def _ollama_alive(timeout: int = 5) -> bool:
    try:
        return requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def _restart_ollama(wait_secs: int = 30) -> bool:
    print("   Ollama daemon not responding - attempting restart", flush=True)
    try:
        out = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for pid in out.stdout.strip().splitlines():
            if pid.strip().isdigit():
                subprocess.run(["kill", "-9", pid.strip()], timeout=5)
                print(f"   Killed stale ollama PID {pid.strip()}", flush=True)
    except Exception:
        pass

    time.sleep(2)
    try:
        log_path = f"/tmp/adaptive_ollama_restart_{os.getpid()}.log"
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=open(log_path, "ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"   Relaunched ollama serve (log: {log_path})", flush=True)
    except Exception as exc:
        print(f"   Cannot relaunch ollama serve: {exc}", flush=True)
        return False

    for _ in range(wait_secs):
        if _ollama_alive():
            return True
        time.sleep(1)
    return False


def _ensure_ollama_alive() -> bool:
    return _ollama_alive() or _restart_ollama()


def _installed_models() -> set[str]:
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        return {m["name"] for m in resp.json().get("models", [])}
    except Exception:
        return set()


def _pull_model(model: str) -> bool:
    print(f"   Pull {model}...", flush=True)
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/pull",
            json={"name": model, "stream": False},
            timeout=None,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        print(f"   Pull failed for {model}: {exc}", flush=True)
        return False


def _preload(model: str) -> None:
    try:
        requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "think": False,
                "options": {"num_predict": 1},
            },
            timeout=600,
        ).raise_for_status()
    except Exception as exc:
        print(f"   Preload failed, continuing: {exc}", flush=True)


def _unload(model: str) -> None:
    try:
        requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=60,
        )
    except Exception:
        pass


def _model_already_completed(model: str) -> bool:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    for path in RESULTS_DIR.glob(f"eval_{slug}_*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            if meta.get("tests_total_completed", 0) >= meta.get("total_tests_planned", 1):
                return True
        except Exception:
            continue
    return False


def _summarise(results: list[dict], model: str) -> dict:
    valid = [r for r in results if not r.get("error")]
    total = len(valid)
    successes = sum(1 for r in valid if r.get("compliant"))
    row = {
        "model": model,
        "overall_asr": round(successes / total, 4) if total else 0.0,
        "tests_valid": total,
        "tests_errored": len(results) - total,
    }
    for defense in DEFENSES:
        sub = [r for r in valid if r["defense"] == defense]
        row[f"asr_{defense}"] = round(sum(r["compliant"] for r in sub) / len(sub), 4) if sub else 0.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive suite multi-model runner")
    parser.add_argument("--models-file", default=str(DEFAULT_MODELS_FILE))
    parser.add_argument("--host", type=str, help="Override Ollama host")
    parser.add_argument("--no-pull", action="store_true", help="Do not pull missing models")
    parser.add_argument("--keep-loaded", action="store_true", help="Do not unload between models")
    parser.add_argument("--fresh", action="store_true", help="Re-run even if a final JSON exists")
    parser.add_argument("--attacks", nargs="*", help="Subset of adaptive attacks")
    parser.add_argument("--defenses", nargs="*", help="Subset of defenses")
    args = parser.parse_args()

    if args.host:
        config.OLLAMA_HOST = args.host

    models_path = Path(args.models_file)
    if not models_path.exists():
        print(f"Models file not found: {models_path}")
        sys.exit(1)
    models = parse_models_file(models_path)
    if not models:
        print(f"No models in {models_path}")
        sys.exit(1)

    print("Adaptive multi-model evaluation")
    print(f"  host      : {config.OLLAMA_HOST}")
    print(f"  models    : {len(models)} ({', '.join(models)})")
    print(f"  tests/model: 180 by default")
    print(f"  max total : {len(models) * 180} by default")

    if not _ensure_ollama_alive():
        print("Ollama daemon unavailable")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)
    summary_path = RESULTS_DIR / f"multi_run_{datetime.now():%Y%m%d_%H_%M_%S}.csv"
    rows: list[dict] = []
    pulled_judges: set[str] = set()
    start = time.time()

    for idx, model in enumerate(models, 1):
        print("\n" + "=" * 72)
        print(f"[{idx}/{len(models)}] MODEL: {model}")
        print("=" * 72)

        if not _ensure_ollama_alive():
            print(f"   Skip {model}: daemon unavailable")
            continue

        installed = _installed_models()
        if model not in installed:
            if args.no_pull:
                print(f"   Skip {model}: missing and --no-pull is active")
                continue
            if not _pull_model(model):
                continue

        config.OLLAMA_MODEL = model
        os.environ["OLLAMA_MODEL"] = model
        judge = select_judge_model(model)
        config.JUDGE_MODEL = judge
        os.environ["JUDGE_MODEL"] = judge
        print(f"   Judge: {'self-judge' if judge == model else judge}")

        if judge != model and judge not in pulled_judges:
            if judge not in _installed_models() and not args.no_pull:
                if not _pull_model(judge):
                    print(f"   Skip {model}: judge unavailable")
                    continue
            pulled_judges.add(judge)

        if not args.fresh and _model_already_completed(model):
            print(f"   Skip {model}: adaptive final run already exists")
            continue

        _preload(model)
        import run_evaluation
        importlib.reload(run_evaluation)

        t0 = time.time()
        try:
            results_path = run_evaluation.run_evaluation(
                selected_attacks=args.attacks,
                selected_defenses=args.defenses,
                dry_run=False,
                fresh=True,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as exc:
            print(f"   Evaluation failed for {model}: {exc}")
            results_path = None

        elapsed = time.time() - t0
        if results_path and Path(results_path).exists():
            with open(results_path, encoding="utf-8") as f:
                data = json.load(f)
            row = _summarise(data["results"], model)
            row["elapsed_min"] = round(elapsed / 60, 2)
            row["results_file"] = Path(results_path).name
            rows.append(row)
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        if not args.keep_loaded:
            _unload(model)

    print("\n" + "=" * 72)
    print(f"Adaptive multi-model run completed in {(time.time() - start) / 60:.1f} min")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
