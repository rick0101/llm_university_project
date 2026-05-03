#!/usr/bin/env python3
"""
Generate cross-model charts for Adaptive Suite jailbreak evaluation results.

Versione per l'adaptive_suite: legge da `adaptive_suite/results/` e scrive
in `adaptive_suite/charts/cross_model/`. Stessa logica di ../generate_cross_model_charts.py.

Input: final `results/eval_*.json` files produced by run_evaluation.py.
Default behavior: use the latest complete run for each tested model, so repeated
runs of the same model are not counted twice.

Outputs are written to `charts/cross_model/` and overwritten on each execution,
because the aggregate view should always reflect the currently available final
results.

Usage:
    python generate_cross_model_charts.py
    python generate_cross_model_charts.py --all-runs
    python generate_cross_model_charts.py results/eval_qwen_*.json results/eval_gemma_*.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent.resolve()
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
CROSS_MODEL_DIR = CHARTS_DIR / "cross_model"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)

VULN_CMAP = LinearSegmentedColormap.from_list(
    "cross_model_vuln",
    ["#f0f0f0", "#fff5b8", "#ffd166", "#ef476f", "#9d0208"],
    N=256,
)
SUCCESS_COLOR = "#9d0208"
SAFE_COLOR = "#06a77d"
MIXED_COLOR = "#f4a261"


@dataclass(frozen=True)
class RunData:
    path: Path
    model: str
    timestamp: str
    metadata: dict
    results: list[dict]


def load_run(path: Path) -> RunData:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results") or []
    if not isinstance(results, list):
        raise ValueError(f"{path} has invalid `results`: expected list")
    metadata = data.get("metadata") or {}
    model = (
        metadata.get("test_model")
        or next((r.get("model") for r in results if r.get("model")), None)
        or path.stem.removeprefix("eval_")
    )
    timestamp = metadata.get("timestamp_end") or metadata.get("timestamp_start") or ""
    return RunData(
        path=path,
        model=str(model),
        timestamp=str(timestamp),
        metadata=metadata,
        results=results,
    )


def parse_timestamp(run: RunData) -> tuple[datetime, float]:
    for raw in (run.timestamp, run.metadata.get("timestamp_start")):
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw)), run.path.stat().st_mtime
        except ValueError:
            pass
    return datetime.fromtimestamp(run.path.stat().st_mtime), run.path.stat().st_mtime


def discover_runs(paths: list[str], include_all_runs: bool) -> list[RunData]:
    if paths:
        files = []
        for raw in paths:
            if any(ch in raw for ch in "*?[]"):
                matches = [Path(p) for p in sorted(glob.glob(raw))]
            else:
                matches = [Path(raw)]
            files.extend(p.resolve() for p in matches)
    else:
        files = sorted(p for p in RESULTS_DIR.glob("eval_*.json") if "live" not in p.name)

    if not files:
        raise SystemExit("No final results/eval_*.json files found.")

    runs = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        run = load_run(path)
        if run.results:
            runs.append(run)

    if not runs:
        raise SystemExit("No non-empty final eval JSON files found.")

    if include_all_runs or paths:
        return sorted(runs, key=lambda r: (r.model, parse_timestamp(r)))

    latest_by_model: dict[str, RunData] = {}
    for run in runs:
        current = latest_by_model.get(run.model)
        if current is None or parse_timestamp(run) > parse_timestamp(current):
            latest_by_model[run.model] = run
    return sorted(latest_by_model.values(), key=lambda r: r.model.lower())


def valid_results(run: RunData) -> list[dict]:
    return [r for r in run.results if not r.get("error")]


def is_success(result: dict) -> bool:
    return bool(result.get("compliant"))


def prompt_label(result: dict) -> str:
    prompt_id = str(result.get("prompt_id", "?"))
    category = str(result.get("category", "?"))
    return f"{prompt_id} ({category})"


def count_successes(rows: Iterable[dict], key_fn: Callable[[dict], str]) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = key_fn(row)
        counts[key][0] += 1
        if is_success(row):
            counts[key][1] += 1
    return counts


def aggregate_matrix(
    runs: list[RunData],
    key_fn: Callable[[dict], str],
    row_order_by_success: bool = True,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    cols = [r.model for r in runs]
    row_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cell_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    for run in runs:
        for result in valid_results(run):
            row = key_fn(result)
            row_counts[row][0] += 1
            cell_counts[(row, run.model)][0] += 1
            if is_success(result):
                row_counts[row][1] += 1
                cell_counts[(row, run.model)][1] += 1

    if row_order_by_success:
        rows = sorted(
            row_counts,
            key=lambda row: (
                -row_counts[row][1],
                -(row_counts[row][1] / row_counts[row][0] if row_counts[row][0] else 0),
                row,
            ),
        )
    else:
        rows = sorted(row_counts)

    asr = np.full((len(rows), len(cols)), np.nan, dtype=float)
    totals = np.zeros((len(rows), len(cols)), dtype=int)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            total, success = cell_counts[(row, col)]
            totals[i, j] = total
            if total:
                asr[i, j] = success / total
    return rows, cols, asr, totals


def aggregate_global_matrix(
    runs: list[RunData],
    row_fn: Callable[[dict], str],
    col_fn: Callable[[dict], str],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    rows = sorted({row_fn(r) for run in runs for r in valid_results(run)})
    cols = sorted({col_fn(r) for run in runs for r in valid_results(run)})
    row_idx = {row: i for i, row in enumerate(rows)}
    col_idx = {col: i for i, col in enumerate(cols)}
    total = np.zeros((len(rows), len(cols)), dtype=int)
    success = np.zeros_like(total)
    for run in runs:
        for result in valid_results(run):
            i = row_idx[row_fn(result)]
            j = col_idx[col_fn(result)]
            total[i, j] += 1
            if is_success(result):
                success[i, j] += 1
    asr = np.where(total > 0, success / np.maximum(total, 1), np.nan)
    row_success = success.sum(axis=1)
    order = np.argsort(-row_success)
    rows = [rows[i] for i in order]
    return rows, cols, asr[order], total[order]


def success_count(asr_value: float, total: int) -> int:
    if math.isnan(asr_value) or not total:
        return 0
    return int(round(asr_value * total))


def plot_heatmap(
    rows: list[str],
    cols: list[str],
    asr: np.ndarray,
    total: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    annotate_all: bool = False,
):
    fig_w = max(9, 0.85 * len(cols) + 3)
    fig_h = max(6, 0.35 * len(rows) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.masked_invalid(asr)
    cmap = VULN_CMAP.copy()
    cmap.set_bad("#fafafa")
    vmax = max(0.05, np.nanmax(asr) if np.any(~np.isnan(asr)) else 0.05)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8 if len(rows) > 20 else 9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    for i in range(len(rows)):
        for j in range(len(cols)):
            cell_total = total[i, j]
            if not cell_total:
                ax.text(j, i, "-", ha="center", va="center", color="#bbb", fontsize=7)
                continue
            cell_asr = asr[i, j]
            cell_success = success_count(cell_asr, cell_total)
            if annotate_all or cell_success:
                label = f"{cell_success}/{cell_total}"
                color = "white" if cell_asr > vmax * 0.55 else "#222"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                    fontweight="bold" if cell_success else "normal",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Attack Success Rate (ASR)")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    fig.savefig(out_path)
    plt.close(fig)


def plot_model_overview(runs: list[RunData], out_path: Path):
    rows = []
    for run in runs:
        valid = valid_results(run)
        successes = sum(1 for r in valid if is_success(r))
        total = len(valid)
        asr = successes / total if total else 0.0
        rows.append((run.model, asr, successes, total))
    rows.sort(key=lambda item: -item[1])

    labels = [r[0] for r in rows]
    asr_pct = [r[1] * 100 for r in rows]
    colors = [SUCCESS_COLOR if r[1] > 0.05 else MIXED_COLOR if r[1] else SAFE_COLOR for r in rows]

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.55 * len(rows) + 1.5)))
    bars = ax.barh(range(len(rows)), asr_pct, color=colors, alpha=0.88, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Overall ASR (%)")
    ax.set_title("Cross-model jailbreak success rate")
    ax.set_xlim(0, max(max(asr_pct) * 1.25 if asr_pct else 0, 5))
    for bar, (_, asr, successes, total) in zip(bars, rows):
        ax.text(
            bar.get_width() + max(max(asr_pct, default=1) * 0.02, 0.1),
            bar.get_y() + bar.get_height() / 2,
            f"{successes}/{total} ({asr * 100:.2f}%)",
            va="center",
            fontsize=9,
        )
    fig.savefig(out_path)
    plt.close(fig)


def rank_for_model(
    run: RunData,
    key_fn: Callable[[dict], str],
    high_is_bad: bool = True,
) -> tuple[str, float, int, int]:
    counts = count_successes(valid_results(run), key_fn)
    if not counts:
        return "-", 0.0, 0, 0
    ranked = []
    for key, (total, success) in counts.items():
        asr = success / total if total else 0.0
        ranked.append((key, asr, success, total))
    if high_is_bad:
        ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    else:
        ranked.sort(key=lambda item: (item[1], -item[3], item[0]))
    return ranked[0]


def top_success_cells(runs: list[RunData], limit: int = 25) -> list[dict[str, str | int | float]]:
    counts: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for run in runs:
        for result in valid_results(run):
            key = (
                run.model,
                str(result.get("attack", "?")),
                str(result.get("defense", "?")),
                str(result.get("prompt_id", "?")),
                str(result.get("category", "?")),
            )
            counts[key][0] += 1
            if is_success(result):
                counts[key][1] += 1

    rows = []
    for (model, attack, defense, prompt_id, category), (total, success) in counts.items():
        if not success:
            continue
        rows.append(
            {
                "model": model,
                "attack": attack,
                "defense": defense,
                "prompt_id": prompt_id,
                "category": category,
                "successes": success,
                "total": total,
                "asr": success / total if total else 0.0,
            }
        )
    rows.sort(key=lambda row: (-int(row["successes"]), -float(row["asr"]), str(row["model"])))
    return rows[:limit]


def best_rows_by_model(runs: list[RunData]) -> list[dict[str, str | int | float]]:
    rows = []
    for run in runs:
        valid = valid_results(run)
        total = len(valid)
        successes = sum(1 for r in valid if is_success(r))
        overall_asr = successes / total if total else 0.0
        best_attack = rank_for_model(run, lambda r: str(r.get("attack", "?")))
        weakest_defense = rank_for_model(run, lambda r: str(r.get("defense", "?")))
        strongest_defense = rank_for_model(
            run,
            lambda r: str(r.get("defense", "?")),
            high_is_bad=False,
        )
        best_prompt = rank_for_model(run, prompt_label)
        top_cell = top_success_cells([run], limit=1)
        top_cell_label = "-"
        if top_cell:
            cell = top_cell[0]
            top_cell_label = (
                f"{cell['attack']} / {cell['defense']} / {cell['prompt_id']} "
                f"({cell['category']})"
            )

        rows.append(
            {
                "model": run.model,
                "overall_asr": overall_asr,
                "successes": successes,
                "total": total,
                "best_attack": best_attack[0],
                "best_attack_asr": best_attack[1],
                "best_attack_successes": best_attack[2],
                "best_attack_total": best_attack[3],
                "weakest_defense": weakest_defense[0],
                "weakest_defense_asr": weakest_defense[1],
                "strongest_defense": strongest_defense[0],
                "strongest_defense_asr": strongest_defense[1],
                "best_prompt": best_prompt[0],
                "best_prompt_asr": best_prompt[1],
                "best_prompt_successes": best_prompt[2],
                "best_prompt_total": best_prompt[3],
                "top_success_path": top_cell_label,
            }
        )
    rows.sort(key=lambda row: (-float(row["overall_asr"]), str(row["model"])))
    return rows


def wrap_cell(value: object, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False)) or "-"


def plot_best_by_model(rows: list[dict[str, str | int | float]], out_path: Path):
    headers = [
        "Model",
        "Overall",
        "Best attack",
        "Weakest defense",
        "Strongest defense",
        "Best prompt",
        "Top success path",
    ]
    table_data = []
    for row in rows:
        table_data.append(
            [
                wrap_cell(row["model"], 18),
                f"{float(row['overall_asr']) * 100:.2f}%\n{row['successes']}/{row['total']}",
                wrap_cell(f"{row['best_attack']} ({float(row['best_attack_asr']) * 100:.1f}%)", 18),
                wrap_cell(
                    f"{row['weakest_defense']} ({float(row['weakest_defense_asr']) * 100:.1f}%)",
                    18,
                ),
                wrap_cell(
                    f"{row['strongest_defense']} "
                    f"({float(row['strongest_defense_asr']) * 100:.1f}%)",
                    18,
                ),
                wrap_cell(f"{row['best_prompt']} ({float(row['best_prompt_asr']) * 100:.1f}%)", 20),
                wrap_cell(row["top_success_path"], 30),
            ]
        )

    fig_h = max(4.5, 0.9 * len(table_data) + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")
    ax.set_title("Most effective attack/defense/prompt signals per model", pad=18)
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.15, 0.09, 0.15, 0.15, 0.15, 0.14, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.2)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f7f7f7")
        cell.set_edgecolor("#dddddd")
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_success_paths(rows: list[dict[str, str | int | float]], out_path: Path):
    if not rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No successful jailbreaks found in the selected final results.",
            ha="center",
            va="center",
            fontsize=15,
            color=SAFE_COLOR,
            fontweight="bold",
        )
        fig.savefig(out_path)
        plt.close(fig)
        return

    labels = [
        f"{row['model']} | {row['attack']} | {row['defense']} | "
        f"{row['prompt_id']} ({row['category']})"
        for row in rows
    ]
    successes = [int(row["successes"]) for row in rows]
    asrs = [float(row["asr"]) for row in rows]
    totals = [int(row["total"]) for row in rows]

    fig, ax = plt.subplots(figsize=(15, max(5, 0.42 * len(rows) + 1.5)))
    bars = ax.barh(range(len(rows)), successes, color=SUCCESS_COLOR, alpha=0.88, edgecolor="white")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([wrap_cell(label, 72) for label in labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Successful jailbreaks")
    ax.set_title("Top successful paths: model | attack | defense | prompt")
    ax.set_xlim(0, max(successes) * 1.25 if successes else 1)
    for bar, success, total, asr in zip(bars, successes, totals, asrs):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{success}/{total} ({asr * 100:.0f}%)",
            va="center",
            fontsize=8.5,
        )
    fig.savefig(out_path)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    out_path: Path,
    runs: list[RunData],
    best_rows: list[dict[str, object]],
    top_rows: list[dict[str, object]],
):
    total_tests = sum(len(valid_results(run)) for run in runs)
    total_successes = sum(1 for run in runs for result in valid_results(run) if is_success(result))
    overall_asr = total_successes / total_tests if total_tests else 0.0
    lines = [
        "# Cross-model jailbreak chart summary",
        "",
        f"- Models included: {len(runs)}",
        f"- Valid tests included: {total_tests}",
        f"- Successful jailbreaks: {total_successes} ({overall_asr * 100:.2f}%)",
        "- Run selection: latest final eval JSON per model unless `--all-runs` "
        "or explicit paths were used.",
        "- Prompt labels use `prompt_id (category)` because final result JSONs "
        "do not store full prompt text.",
        "- Defense ASR is vulnerability under that defense: high ASR means the "
        "defense was bypassed more often.",
        "",
        "## Per-model best signals",
        "",
        "| Model | Overall ASR | Best attack | Weakest defense | Strongest defense | Best prompt |",
        "|---|---:|---|---|---|---|",
    ]
    for row in best_rows:
        lines.append(
            "| {model} | {asr:.2f}% ({succ}/{tot}) | {attack} ({attack_asr:.1f}%) | "
            "{weak} ({weak_asr:.1f}%) | {strong} ({strong_asr:.1f}%) | "
            "{prompt} ({prompt_asr:.1f}%) |".format(
                model=row["model"],
                asr=float(row["overall_asr"]) * 100,
                succ=row["successes"],
                tot=row["total"],
                attack=row["best_attack"],
                attack_asr=float(row["best_attack_asr"]) * 100,
                weak=row["weakest_defense"],
                weak_asr=float(row["weakest_defense_asr"]) * 100,
                strong=row["strongest_defense"],
                strong_asr=float(row["strongest_defense_asr"]) * 100,
                prompt=row["best_prompt"],
                prompt_asr=float(row["best_prompt_asr"]) * 100,
            )
        )

    lines += [
        "",
        "## Top successful paths",
        "",
        "| Model | Attack | Defense | Prompt | Category | Successes | ASR |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in top_rows[:15]:
        lines.append(
            (
                "| {model} | {attack} | {defense} | {prompt_id} | "
                "{category} | {succ}/{tot} | {asr:.1f}% |"
            ).format(
                model=row["model"],
                attack=row["attack"],
                defense=row["defense"],
                prompt_id=row["prompt_id"],
                category=row["category"],
                succ=row["successes"],
                tot=row["total"],
                asr=float(row["asr"]) * 100,
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(runs: list[RunData], out_dir: Path, top_n: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing cross-model charts to {out_dir}")
    print("Included runs:")
    for run in runs:
        print(f"  - {run.model}: {run.path.name} ({len(valid_results(run))} valid tests)")

    plot_model_overview(runs, out_dir / "01_model_overall_asr.png")

    attack_rows, model_cols, attack_asr, attack_totals = aggregate_matrix(
        runs, lambda r: str(r.get("attack", "?"))
    )
    plot_heatmap(
        attack_rows,
        model_cols,
        attack_asr,
        attack_totals,
        title="Attack effectiveness by model",
        xlabel="Model",
        ylabel="Attack",
        out_path=out_dir / "02_attack_by_model_heatmap.png",
        annotate_all=False,
    )

    defense_rows, model_cols, defense_asr, defense_totals = aggregate_matrix(
        runs, lambda r: str(r.get("defense", "?"))
    )
    plot_heatmap(
        defense_rows,
        model_cols,
        defense_asr,
        defense_totals,
        title="Defense bypass rate by model (higher = weaker defense in this run)",
        xlabel="Model",
        ylabel="Defense",
        out_path=out_dir / "03_defense_by_model_heatmap.png",
        annotate_all=True,
    )

    prompt_rows, model_cols, prompt_asr, prompt_totals = aggregate_matrix(runs, prompt_label)
    plot_heatmap(
        prompt_rows,
        model_cols,
        prompt_asr,
        prompt_totals,
        title="Prompt vulnerability by model",
        xlabel="Model",
        ylabel="Prompt ID (category)",
        out_path=out_dir / "04_prompt_by_model_heatmap.png",
        annotate_all=False,
    )

    ad_rows, ad_cols, ad_asr, ad_totals = aggregate_global_matrix(
        runs,
        lambda r: str(r.get("attack", "?")),
        lambda r: str(r.get("defense", "?")),
    )
    plot_heatmap(
        ad_rows,
        ad_cols,
        ad_asr,
        ad_totals,
        title="Global attack x defense ASR across selected models",
        xlabel="Defense",
        ylabel="Attack",
        out_path=out_dir / "05_global_attack_defense_heatmap.png",
        annotate_all=False,
    )

    best_rows = best_rows_by_model(runs)
    plot_best_by_model(best_rows, out_dir / "06_best_per_model.png")

    top_rows = top_success_cells(runs, limit=top_n)
    plot_top_success_paths(top_rows, out_dir / "07_top_success_paths.png")

    write_csv(out_dir / "best_by_model.csv", best_rows)
    write_csv(out_dir / "top_success_paths.csv", top_rows)
    write_summary(out_dir / "summary.md", runs, best_rows, top_rows)
    print("Done: 7 PNG charts + best_by_model.csv + top_success_paths.csv + summary.md")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Final eval JSON files or glob patterns. Default: latest eval_*.json per model.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Include every final eval_*.json file, including multiple runs of the same model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CROSS_MODEL_DIR,
        help="Output directory. Default: charts/cross_model.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top successful model/attack/defense/prompt paths to plot.",
    )
    args = parser.parse_args()

    runs = discover_runs(args.paths, include_all_runs=args.all_runs)
    generate(runs, args.output_dir.resolve(), top_n=args.top_n)


if __name__ == "__main__":
    main()
