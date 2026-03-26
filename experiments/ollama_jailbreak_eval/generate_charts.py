#!/usr/bin/env python3
"""
Generate matplotlib charts from Ollama jailbreak evaluation results.

Usage:
    python generate_charts.py results/eval_XXXXXXXX_XXXXXX.json
    python generate_charts.py  # auto-picks latest results file
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str = None) -> dict:
    """Load results JSON. If no path, pick the latest in results/."""
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if path is None:
        files = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))
        if not files:
            print("❌ Nessun file di risultati trovato in results/")
            sys.exit(1)
        path = os.path.join(results_dir, files[-1])
        print(f"📂 Caricamento risultati più recenti: {path}")

    with open(path) as f:
        return json.load(f)


def compute_asr_matrix(results: list) -> dict:
    """Compute ASR for each (attack, defense) pair."""
    counts = defaultdict(lambda: {"total": 0, "compliant": 0})
    for r in results:
        key = (r["attack"], r["defense"])
        counts[key]["total"] += 1
        if r["compliant"]:
            counts[key]["compliant"] += 1

    asr = {}
    for (attack, defense), c in counts.items():
        asr.setdefault(attack, {})[defense] = c["compliant"] / c["total"] if c["total"] else 0
    return asr


def compute_category_asr(results: list) -> dict:
    """Compute ASR per (category, defense)."""
    counts = defaultdict(lambda: {"total": 0, "compliant": 0})
    for r in results:
        key = (r["category"], r["defense"])
        counts[key]["total"] += 1
        if r["compliant"]:
            counts[key]["compliant"] += 1

    asr = {}
    for (cat, defense), c in counts.items():
        asr.setdefault(cat, {})[defense] = c["compliant"] / c["total"] if c["total"] else 0
    return asr


def chart_1_heatmap(asr_matrix: dict, defenses: list, charts_dir: str, model_name: str):
    """Heatmap: ASR per attack × defense."""
    attacks = sorted(asr_matrix.keys())
    data = np.array([[asr_matrix[a].get(d, 0) for d in defenses] for a in attacks])

    fig, ax = plt.subplots(figsize=(max(8, len(defenses) * 2.5), max(6, len(attacks) * 0.6)))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(defenses)))
    ax.set_xticklabels(defenses, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels(attacks, fontsize=10)

    for i in range(len(attacks)):
        for j in range(len(defenses)):
            val = data[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Attack Success Rate", shrink=0.8)
    ax.set_title(f"ASR Heatmap — Attack × Defense [{model_name}]", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "heatmap_attack_defense.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✅ heatmap_attack_defense.png")


def chart_2_grouped_bar(asr_matrix: dict, defenses: list, charts_dir: str, model_name: str):
    """Grouped bar chart: ASR per attack, grouped by defense."""
    attacks = sorted(asr_matrix.keys())
    x = np.arange(len(attacks))
    width = 0.8 / len(defenses)
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]

    fig, ax = plt.subplots(figsize=(max(10, len(attacks) * 1.2), 6))
    for i, defense in enumerate(defenses):
        vals = [asr_matrix[a].get(defense, 0) for a in attacks]
        offset = (i - len(defenses) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=defense,
               color=colors[i % len(colors)], alpha=0.85, edgecolor="white")

    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=10, title="Defense", title_fontsize=11)
    ax.set_title(f"Attack Success Rate by Attack × Defense [{model_name}]", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "bar_attack_defense.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✅ bar_attack_defense.png")


def chart_3_category_radar(cat_asr: dict, defenses: list, charts_dir: str, model_name: str):
    """Radar/spider chart: ASR per category for each defense."""
    categories = sorted(cat_asr.keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, defense in enumerate(defenses):
        vals = [cat_asr[c].get(defense, 0) for c in categories]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=defense,
                color=colors[i % len(colors)], markersize=6)
        ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title(f"ASR by Harm Category × Defense [{model_name}]", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "radar_category.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✅ radar_category.png")


def chart_4_defense_summary(asr_matrix: dict, defenses: list, charts_dir: str, model_name: str):
    """Horizontal bar: overall ASR per defense (lower = better)."""
    overall = {}
    for defense in defenses:
        vals = [asr_matrix[a].get(defense, 0) for a in asr_matrix]
        overall[defense] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#E74C3C" if v > 0.5 else "#F39C12" if v > 0.25 else "#2ECC71"
              for v in overall.values()]

    bars = ax.barh(list(overall.keys()), list(overall.values()),
                   color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Overall ASR (lower = safer)", fontsize=11)
    ax.set_title(f"Defense Effectiveness Summary [{model_name}]", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, overall.values()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=11, fontweight="bold")

    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "defense_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✅ defense_summary.png")


def chart_5_latency(results: list, defenses: list, charts_dir: str, model_name: str):
    """Box plot: response latency per defense strategy."""
    data = {d: [r["latency_s"] for r in results if r["defense"] == d] for d in defenses}

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([data[d] for d in defenses], labels=defenses, patch_artist=True,
                    boxprops=dict(alpha=0.7))
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    for patch, color in zip(bp["boxes"], colors[:len(defenses)]):
        patch.set_facecolor(color)

    ax.set_ylabel("Latency (seconds)", fontsize=11)
    ax.set_title(f"Response Latency by Defense [{model_name}]", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "latency_boxplot.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✅ latency_boxplot.png")


def generate_charts(results_path: str = None):
    """Generate all charts from evaluation results. Can be called programmatically."""
    data = load_results(results_path)
    metadata = data["metadata"]
    results = data["results"]
    model_name = metadata.get("model", "unknown")

    print(f"\n📊 Generazione grafici per: {model_name}")
    print(f"   {metadata['total_tests']} test, {metadata['elapsed_seconds']}s totali\n")

    ts = metadata.get("timestamp", "unknown")
    charts_dir = os.path.join(os.path.dirname(__file__), "results", "charts", f"{model_name.replace(':', '_')}_{ts}")
    os.makedirs(charts_dir, exist_ok=True)

    defenses = sorted(set(r["defense"] for r in results))
    asr_matrix = compute_asr_matrix(results)
    cat_asr = compute_category_asr(results)

    chart_1_heatmap(asr_matrix, defenses, charts_dir, model_name)
    chart_2_grouped_bar(asr_matrix, defenses, charts_dir, model_name)
    chart_3_category_radar(cat_asr, defenses, charts_dir, model_name)
    chart_4_defense_summary(asr_matrix, defenses, charts_dir, model_name)
    chart_5_latency(results, defenses, charts_dir, model_name)

    print(f"\n🎯 5 grafici salvati in {charts_dir}/")
    return charts_dir


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_charts(path)


if __name__ == "__main__":
    main()
