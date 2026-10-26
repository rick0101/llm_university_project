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


def _compute_defense_overall_asr(asr_matrix: dict, defenses: list) -> dict:
    """Compute overall ASR per defense across all attacks."""
    overall = {}
    for defense in defenses:
        vals = [asr_matrix[a].get(defense, 0) for a in asr_matrix]
        overall[defense] = np.mean(vals) if vals else 0
    return overall


def _find_project_root() -> str:
    """Walk up from this file to find the project root (contains output/)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "output", "latest")):
            return d
        d = os.path.dirname(d)
    return None


def _copy_charts_to_latex(charts_dir: str, folder_name: str) -> str:
    """Copy charts into output/latest/<folder_name>/ for LaTeX access."""
    import shutil
    root = _find_project_root()
    if not root:
        print("  ⚠️  Cartella output/latest/ non trovata, skip copia LaTeX")
        return None
    latex_charts = os.path.join(root, "output", "latest", folder_name)
    os.makedirs(latex_charts, exist_ok=True)
    for f in os.listdir(charts_dir):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(charts_dir, f), latex_charts)
    print(f"  📄 Grafici copiati in {latex_charts}/")
    return latex_charts


def _append_latex_section(folder_name: str, metadata: dict, defenses: list,
                          attacks: list, categories: list, defense_asr: dict):
    """Append an experimental evaluation section to survey.tex."""
    root = _find_project_root()
    if not root:
        return
    tex_path = os.path.join(root, "output", "latest", "survey.tex")
    if not os.path.isfile(tex_path):
        print("  ⚠️  survey.tex non trovato, skip inserimento LaTeX")
        return

    with open(tex_path) as f:
        content = f.read()

    model = metadata["model"]
    model_safe = model.replace(":", "_")
    total = metadata["total_tests"]
    n_prompts = metadata["num_prompts"]
    n_attacks = metadata["num_attacks"]
    n_defenses = metadata["num_defenses"]
    elapsed_h = round(metadata["elapsed_seconds"] / 3600, 1)
    cats_str = ", ".join(categories)
    attacks_str = ", ".join(attacks)
    defenses_str = ", ".join(defenses)

    best_def = min(defense_asr, key=defense_asr.get)
    worst_def = max(defense_asr, key=defense_asr.get)
    best_asr = defense_asr[best_def]
    worst_asr = defense_asr[worst_def]
    overall_asr = np.mean(list(defense_asr.values()))

    label_suffix = model_safe
    section = f"""
\\section{{Experimental Validation: {model}}}

To empirically validate the theoretical attack and defense taxonomies presented in prior sections, we conducted a systematic jailbreak evaluation campaign against \\texttt{{{model}}}, a locally deployed large language model served via the Ollama inference framework. The evaluation exhaustively tests every combination of harmful prompts, attack strategies, and defense mechanisms, producing a comprehensive Attack Success Rate (ASR) matrix. A total of {total:,} test cases were executed ({n_prompts} harmful prompts $\\times$ {n_attacks} attack strategies $\\times$ {n_defenses} defense mechanisms), spanning {len(categories)} threat categories: {cats_str}. The complete evaluation required approximately {elapsed_h} hours of inference time.

The attack strategies employed include both classical black-box techniques and recently proposed methods derived from the current literature: {attacks_str}. The defense mechanisms tested are: {defenses_str}. This combinatorial design enables fine-grained analysis of which attack--defense pairings expose the greatest model vulnerabilities.

The overall ASR across all configurations was {overall_asr:.1%}. The most effective defense was {best_def}, achieving an ASR of {best_asr:.1%}, while {worst_def} exhibited the highest vulnerability with an ASR of {worst_asr:.1%}. These results quantitatively demonstrate that defense effectiveness varies dramatically depending on the strategy employed, reinforcing the need for multi-layered defense architectures.

Figure~\\ref{{fig:heatmap_asr_{label_suffix}}} presents the ASR heatmap across all attack--defense combinations. Each cell represents the proportion of harmful prompts that successfully bypassed the model's safety alignment for that specific pairing.

\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.85\\textwidth]{{{folder_name}/heatmap_attack_defense.png}}
  \\caption{{Heatmap of Attack Success Rate (ASR) across all attack--defense combinations for \\texttt{{{model}}}. Darker cells indicate higher vulnerability.}}
  \\label{{fig:heatmap_asr_{label_suffix}}}
\\end{{figure}}

Figure~\\ref{{fig:bar_asr_{label_suffix}}} provides a grouped bar chart view of the same data, enabling direct comparison of attack effectiveness under each defense strategy.

\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.85\\textwidth]{{{folder_name}/bar_attack_defense.png}}
  \\caption{{Grouped bar chart comparing ASR of each attack strategy under different defense mechanisms for \\texttt{{{model}}}.}}
  \\label{{fig:bar_asr_{label_suffix}}}
\\end{{figure}}

Figure~\\ref{{fig:radar_category_{label_suffix}}} illustrates the model's vulnerability profile across threat categories using a radar chart, highlighting which domains of harmful content are most and least effectively defended against.

\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.7\\textwidth]{{{folder_name}/radar_category.png}}
  \\caption{{Radar chart of ASR by threat category for \\texttt{{{model}}}, showing the model's vulnerability profile across different harm domains.}}
  \\label{{fig:radar_category_{label_suffix}}}
\\end{{figure}}

Figure~\\ref{{fig:defense_summary_{label_suffix}}} summarizes the overall effectiveness of each defense mechanism by aggregating ASR across all attacks and prompt categories.

\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.75\\textwidth]{{{folder_name}/defense_summary.png}}
  \\caption{{Overall ASR by defense strategy for \\texttt{{{model}}}. Lower values indicate more effective defenses.}}
  \\label{{fig:defense_summary_{label_suffix}}}
\\end{{figure}}

Finally, Figure~\\ref{{fig:latency_{label_suffix}}} presents the response latency distribution across defense strategies. Multi-call defenses exhibit higher latency due to additional API calls, representing a practical trade-off between safety and responsiveness.

\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.75\\textwidth]{{{folder_name}/latency_boxplot.png}}
  \\caption{{Response latency boxplot by defense strategy for \\texttt{{{model}}}. Multi-call defenses incur higher latency as a trade-off for improved safety.}}
  \\label{{fig:latency_{label_suffix}}}
\\end{{figure}}

"""

    marker = r"\section{Discussion and Open Challenges}"
    if marker not in content:
        print("  ⚠️  Marker 'Discussion and Open Challenges' non trovato in survey.tex")
        return

    # Check if a section for this model already exists
    section_title = f"\\section{{Experimental Validation: {model}}}"
    if section_title in content:
        print(f"  ℹ️  Sezione per {model} già presente in survey.tex, skip")
        return

    content = content.replace(marker, section + marker)
    with open(tex_path, "w") as f:
        f.write(content)
    print(f"  ✅ Sezione LaTeX aggiunta per {model}")


def _append_readme_section(folder_name: str, metadata: dict, defenses: list,
                           attacks: list, categories: list, defense_asr: dict):
    """Append evaluation results section to README.md."""
    root = _find_project_root()
    if not root:
        return
    readme_path = os.path.join(root, "README.md")
    if not os.path.isfile(readme_path):
        print("  ⚠️  README.md non trovato, skip")
        return

    with open(readme_path) as f:
        content = f.read()

    model = metadata["model"]
    total = metadata["total_tests"]
    n_prompts = metadata["num_prompts"]
    n_attacks = metadata["num_attacks"]
    n_defenses = metadata["num_defenses"]
    elapsed_h = round(metadata["elapsed_seconds"] / 3600, 1)

    best_def = min(defense_asr, key=defense_asr.get)
    worst_def = max(defense_asr, key=defense_asr.get)
    overall_asr = np.mean(list(defense_asr.values()))

    section_title = f"### Evaluation Results: {model}"
    if section_title in content:
        print(f"  ℹ️  Sezione README per {model} già presente, skip")
        return

    asr_table = "| Defense | ASR |\n|---|---|\n"
    for d in defenses:
        asr_table += f"| {d} | {defense_asr[d]:.1%} |\n"

    section = f"""
{section_title}

**Model:** `{model}` | **Tests:** {total:,} ({n_prompts} prompts × {n_attacks} attacks × {n_defenses} defenses) | **Duration:** {elapsed_h}h | **Overall ASR:** {overall_asr:.1%}

{asr_table}
**Best defense:** {best_def} ({defense_asr[best_def]:.1%}) — **Worst defense:** {worst_def} ({defense_asr[worst_def]:.1%})

Charts saved in `output/latest/{folder_name}/`.

"""

    # Append after the Ollama Jailbreak Evaluation section's last content
    # Find a good insertion point: end of file or before a known next section
    content += section
    with open(readme_path, "w") as f:
        f.write(content)
    print(f"  ✅ Sezione README aggiunta per {model}")


def generate_charts(results_path: str = None):
    """Generate all charts from evaluation results. Can be called programmatically."""
    data = load_results(results_path)
    metadata = data["metadata"]
    results = data["results"]
    model_name = metadata.get("model", "unknown")

    print(f"\n📊 Generazione grafici per: {model_name}")
    print(f"   {metadata['total_tests']} test, {metadata['elapsed_seconds']}s totali\n")

    ts = metadata.get("timestamp", "unknown")
    folder_name = f"{model_name.replace(':', '_')}_{ts}"
    charts_dir = os.path.join(os.path.dirname(__file__), "results", "charts", folder_name)
    os.makedirs(charts_dir, exist_ok=True)

    defenses = sorted(set(r["defense"] for r in results))
    attacks = sorted(set(r["attack"] for r in results))
    categories = sorted(set(r["category"] for r in results))
    asr_matrix = compute_asr_matrix(results)
    cat_asr = compute_category_asr(results)

    chart_1_heatmap(asr_matrix, defenses, charts_dir, model_name)
    chart_2_grouped_bar(asr_matrix, defenses, charts_dir, model_name)
    chart_3_category_radar(cat_asr, defenses, charts_dir, model_name)
    chart_4_defense_summary(asr_matrix, defenses, charts_dir, model_name)
    chart_5_latency(results, defenses, charts_dir, model_name)

    print(f"\n🎯 5 grafici salvati in {charts_dir}/")

    # Copy charts to LaTeX output directory
    _copy_charts_to_latex(charts_dir, folder_name)

    # Compute defense-level ASR for text generation
    defense_asr = _compute_defense_overall_asr(asr_matrix, defenses)

    # Auto-append sections to LaTeX and README
    _append_latex_section(folder_name, metadata, defenses, attacks, categories, defense_asr)
    _append_readme_section(folder_name, metadata, defenses, attacks, categories, defense_asr)

    return charts_dir


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_charts(path)


if __name__ == "__main__":
    main()
