#!/usr/bin/env python3
"""
Mine previous full-suite eval_*.json files and report robust seed signals.

This script is analysis-only: it does not generate attacks automatically and it
does not touch the parent results. It writes adaptive_suite/analysis/seed_signals.json.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
DEFAULT_RESULTS = PARENT / "results"
DEFAULT_OUT = HERE / "analysis" / "seed_signals.json"


def _run_timestamp(path: Path, data: dict[str, Any]) -> str:
    meta = data.get("metadata", {})
    return (
        str(meta.get("timestamp_end") or meta.get("timestamp_start") or "")
        or str(path.stat().st_mtime)
    )


def load_latest_runs(results_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    latest: dict[str, tuple[str, Path, dict[str, Any]]] = {}
    for path in sorted(results_dir.glob("eval_*.json")):
        if "live" in path.name:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        rows = data.get("results") or []
        if not rows:
            continue
        meta = data.get("metadata", {})
        model = meta.get("test_model") or meta.get("model") or rows[0].get("model")
        if not model:
            continue
        ts = _run_timestamp(path, data)
        if model not in latest or ts > latest[model][0]:
            latest[model] = (ts, path, data)
    return {model: (path, data) for model, (_, path, data) in latest.items()}


def ranked(rows: list[dict[str, Any]], keys: tuple[str, ...], min_n: int = 1) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in keys)
        buckets[key][1] += 1
        if row.get("compliant"):
            buckets[key][0] += 1

    out = []
    for key, (success, total) in buckets.items():
        if total < min_n:
            continue
        out.append({
            "key": dict(zip(keys, key)),
            "successes": success,
            "total": total,
            "asr": round(success / total, 6) if total else 0.0,
        })
    out.sort(key=lambda r: (r["asr"], r["successes"], r["total"]), reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine previous eval_*.json results")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    latest = load_latest_runs(results_dir)
    rows = [
        row
        for _, data in latest.values()
        for row in data.get("results", [])
        if not row.get("error")
    ]

    output = {
        "source_results_dir": str(results_dir),
        "models": sorted(latest),
        "runs": {model: str(path) for model, (path, _) in sorted(latest.items())},
        "valid_tests": len(rows),
        "successes": sum(1 for row in rows if row.get("compliant")),
        "rankings": {
            "attacks": ranked(rows, ("attack",), min_n=100)[: args.top_n],
            "defenses": ranked(rows, ("defense",), min_n=100)[: args.top_n],
            "prompts": ranked(rows, ("prompt_id", "category"), min_n=100)[: args.top_n],
            "categories": ranked(rows, ("category",), min_n=100)[: args.top_n],
            "attack_defense": ranked(rows, ("attack", "defense"), min_n=50)[: args.top_n],
            "attack_prompt": ranked(rows, ("attack", "prompt_id", "category"), min_n=20)[: args.top_n],
            "model_attack": ranked(rows, ("model", "attack"), min_n=50)[: args.top_n],
        },
        "adaptive_suite_v1_rationale": {
            "prompt_parents": ["poi-1", "poi-2", "exp-2", "exp-1", "inj-1", "ext-1"],
            "attack_parents": [
                "Refusal-Suppression",
                "FITD",
                "Many-Shot",
                "CodeJailbreak",
                "ReNeLLM",
                "Crescendo/Multi-Turn",
            ],
            "defenses": [
                "no-defense",
                "safety-prompt",
                "in-context",
                "goal-priority",
                "constitutional",
            ],
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    asr = output["successes"] / output["valid_tests"] if output["valid_tests"] else 0.0
    print(f"Models mined : {len(latest)}")
    print(f"Valid tests  : {output['valid_tests']}")
    print(f"Successes    : {output['successes']} ({asr:.2%})")
    print(f"Wrote        : {out_path}")


if __name__ == "__main__":
    main()
