#!/usr/bin/env python3
"""
Generate analytical charts for Adaptive Suite jailbreak evaluation results.

Versione per l'adaptive_suite: legge da `adaptive_suite/results/` e scrive
in `adaptive_suite/charts/`. Stessa logica di ../generate_charts.py.

Genera per ogni modello una cartella `charts/<model>_<timestamp>/` con grafici
ad alta densità informativa.

Genera:
    01_summary_card.png             — KPI generali del run (ASR, errori, latenza)
    02_heatmap_attack_defense.png   — heatmap completa attack×defense
    03_heatmap_attack_category.png  — heatmap attack×category (no defense)
    04_top_vulnerable_pairs.png     — top-N celle vulnerabili
    05_asr_per_attack.png           — ranking attacchi per ASR
    06_asr_per_defense.png          — ranking difese per ASR (efficacia inversa)
    07_asr_per_category.png         — quale tipo di richiesta è più vulnerabile
    08_attack_x_category_radar.png  — radar per attacchi top
    09_latency_per_attack.png       — distribuzione latenza per attacco
    10_judge_breakdown.png          — quanti REFUSED da regex vs LLM judge
    successes.csv                   — elenco completo dei test riusciti (per audit)
    summary.md                      — riassunto markdown leggibile

Usage:
    python generate_charts.py                    # processa tutti i risultati nuovi in results/
    python generate_charts.py results/eval_X.json
    python generate_charts.py --all              # rigenera tutto
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ── Stile globale ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":          120,
    "savefig.dpi":         150,
    "savefig.bbox":        "tight",
    "font.family":         "DejaVu Sans",
    "font.size":           10,
    "axes.titleweight":    "bold",
    "axes.titlesize":      13,
    "axes.labelsize":      11,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linewidth":      0.5,
})

# Colormap "vulnerability": grigio chiaro (0%) → giallo (medio) → rosso scuro (100%)
VULN_CMAP = LinearSegmentedColormap.from_list(
    "vuln", ["#f0f0f0", "#fff5b8", "#ffd166", "#ef476f", "#9d0208"], N=256,
)


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_run(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_runs(arg_paths: list[str], all_flag: bool) -> list[Path]:
    if arg_paths:
        return [Path(p).resolve() for p in arg_paths]
    candidates = sorted(RESULTS_DIR.glob("eval_*.json"))
    candidates = [p for p in candidates if "live" not in p.name]
    if not candidates:
        print("❌ Nessun file `results/eval_*.json` trovato.")
        sys.exit(1)
    if all_flag:
        return candidates
    # default: solo i run senza cartella charts già esistente con stesso slug
    out = []
    for p in candidates:
        slug = p.stem.replace("eval_", "")
        if not any((CHARTS_DIR / d).is_dir() and d.startswith(slug) for d in os.listdir(CHARTS_DIR)):
            out.append(p)
    if not out:
        print("ℹ️  Tutti i risultati hanno già grafici. Usa --all per rigenerare.")
        sys.exit(0)
    return out


# ── Aggregazioni ───────────────────────────────────────────────────────────────
def asr_matrix(results: list, row_key: str, col_key: str) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Ritorna (rows_sorted, cols_sorted, asr_matrix, count_matrix).
    Celle vuote → NaN (così la heatmap le distingue dallo 0% reale).
    """
    rows, cols = sorted({r[row_key] for r in results}), sorted({r[col_key] for r in results})
    r_idx = {r: i for i, r in enumerate(rows)}
    c_idx = {c: i for i, c in enumerate(cols)}
    total = np.zeros((len(rows), len(cols)), dtype=int)
    success = np.zeros_like(total)
    for r in results:
        i, j = r_idx[r[row_key]], c_idx[r[col_key]]
        total[i, j] += 1
        if r.get("compliant"):
            success[i, j] += 1
    asr = np.where(total > 0, success / np.maximum(total, 1), np.nan)
    return rows, cols, asr, total


def asr_by(results: list, key: str) -> list[tuple[str, float, int, int]]:
    counter = defaultdict(lambda: [0, 0])  # [total, success]
    for r in results:
        counter[r[key]][0] += 1
        if r.get("compliant"):
            counter[r[key]][1] += 1
    out = [(k, s / t if t else 0.0, s, t) for k, (t, s) in counter.items()]
    out.sort(key=lambda x: -x[1])
    return out


# ── Plotters ───────────────────────────────────────────────────────────────────
def plot_summary_card(meta: dict, results: list, out_path: Path):
    n = len(results)
    n_succ = sum(1 for r in results if r.get("compliant"))
    n_err = sum(1 for r in results if r.get("error"))
    judge_used = sum(1 for r in results if r.get("judge_used"))
    asr = n_succ / n * 100 if n else 0
    latencies = [r["latency_s"] for r in results if isinstance(r.get("latency_s"), (int, float))]
    avg_lat = np.mean(latencies) if latencies else 0
    p95_lat = np.percentile(latencies, 95) if latencies else 0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    ax.set_facecolor("white")

    title = f"Jailbreak Evaluation — {meta.get('test_model', '?')}"
    ax.text(0.5, 0.93, title, ha="center", fontsize=18, fontweight="bold",
            transform=ax.transAxes)
    sub = (f"Judge: {meta.get('judge_model','?')}  ·  "
           f"Run: {meta.get('timestamp_start','?')}  ·  "
           f"Elapsed: {meta.get('elapsed_human','?')}")
    ax.text(0.5, 0.87, sub, ha="center", fontsize=10, color="#555",
            transform=ax.transAxes)

    # KPI tiles (4 colonne)
    asr_color = "#9d0208" if asr > 10 else "#f4a261" if asr > 2 else "#06a77d"
    tiles = [
        ("Overall ASR",    f"{asr:.2f}%",        asr_color),
        ("Successi / Tot", f"{n_succ} / {n}",    "#264653"),
        ("Errori API",     f"{n_err}",           "#9d4edd" if n_err else "#888"),
        ("LLM judge calls",f"{judge_used}",      "#2a9d8f"),
        ("Avg latency",    f"{avg_lat:.1f}s",    "#264653"),
        ("p95 latency",    f"{p95_lat:.1f}s",    "#264653"),
    ]
    cols = 3
    for i, (label, value, color) in enumerate(tiles):
        row, col = i // cols, i % cols
        x = 0.05 + col * 0.32
        y = 0.55 - row * 0.30
        # bg rectangle
        rect = plt.Rectangle((x, y - 0.02), 0.28, 0.25,
                              transform=ax.transAxes,
                              facecolor="#f7f7f7", edgecolor="#ddd",
                              linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.14, y + 0.16, value, ha="center", va="center",
                fontsize=22, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x + 0.14, y + 0.04, label, ha="center", va="center",
                fontsize=10, color="#555", transform=ax.transAxes)

    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmap(rows: list, cols: list, asr: np.ndarray, total: np.ndarray,
                 title: str, xlabel: str, ylabel: str, out_path: Path,
                 annotate_threshold: float = 0.0):
    """Heatmap con annotazioni 'X/N' nelle celle non-NaN (ASR=success/total)."""
    fig_w = max(8, 0.55 * len(cols) + 2)
    fig_h = max(6, 0.42 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.masked_invalid(asr)
    cmap = VULN_CMAP.copy()
    cmap.set_bad("#fafafa")
    vmax = max(0.05, np.nanmax(asr) if np.any(~np.isnan(asr)) else 0.05)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    # Annotazioni "succ/tot" solo per celle con almeno 1 successo o tot piccolo
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = asr[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="#bbb", fontsize=7)
                continue
            n = total[i, j]
            succ = int(round(v * n))
            if v >= annotate_threshold or succ > 0:
                color = "white" if v > vmax * 0.55 else "#222"
                ax.text(j, i, f"{succ}/{n}", ha="center", va="center",
                        color=color, fontsize=7, fontweight="bold" if succ else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Attack Success Rate (ASR)")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_vulnerable_pairs(results: list, out_path: Path, top_n: int = 20):
    """Bar chart orizzontale dei top-N (attack, defense, category) con più successi."""
    counter = defaultdict(lambda: [0, 0])  # (atk, def, cat) → [total, success]
    for r in results:
        key = (r["attack"], r["defense"], r.get("category", "?"))
        counter[key][0] += 1
        if r.get("compliant"):
            counter[key][1] += 1

    rows = [(k, s, t, s / t if t else 0) for k, (t, s) in counter.items() if s > 0]
    rows.sort(key=lambda x: (-x[1], -x[3]))
    rows = rows[:top_n]

    if not rows:
        # nessun successo: figura "buona notizia"
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "✅ Nessun jailbreak riuscito\n(modello robusto a tutti gli attacchi testati)",
                ha="center", va="center", fontsize=18, color="#06a77d", fontweight="bold")
        fig.savefig(out_path)
        plt.close(fig)
        return

    labels = [f"{k[0]} × {k[1]} × {k[2]}" for k, *_ in rows]
    succ = [s for _, s, _, _ in rows]
    asr = [a for *_, a in rows]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(rows) + 1.5)))
    bars = ax.barh(range(len(labels)), succ, color="#9d0208", alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Successi (jailbreak riusciti)")
    ax.set_title(f"Top {len(rows)} celle vulnerabili (attacco × difesa × categoria)")

    for i, (bar, s, t, a) in enumerate(zip(bars, succ, [t for _, _, t, _ in rows], asr)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{s}/{t}  ({a*100:.0f}%)", va="center", fontsize=9)
    ax.set_xlim(0, max(succ) * 1.25)
    fig.savefig(out_path)
    plt.close(fig)


def plot_asr_ranking(data: list[tuple], title: str, xlabel: str, out_path: Path,
                     color_high: str = "#9d0208", color_low: str = "#06a77d"):
    """Bar chart orizzontale ranked per ASR. Etichette mostrano s/t e %."""
    if not data:
        return
    labels = [d[0] for d in data]
    asrs = [d[1] for d in data]
    succ = [d[2] for d in data]
    tot = [d[3] for d in data]
    colors = [color_high if a > 0.05 else color_low if a == 0 else "#f4a261" for a in asrs]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(labels) + 1.5)))
    bars = ax.barh(range(len(labels)), [a * 100 for a in asrs], color=colors,
                    alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ASR (%)")
    ax.set_title(title)
    max_asr = max(asrs) * 100 if asrs else 1
    ax.set_xlim(0, max(max_asr * 1.25, 5))

    for bar, s, t, a in zip(bars, succ, tot, asrs):
        ax.text(bar.get_width() + max_asr * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{s}/{t}  ({a*100:.2f}%)", va="center", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)


def plot_radar_top_attacks(results: list, out_path: Path, top_n_attacks: int = 6):
    """Radar: per i top-N attacchi più riusciti, ASR per categoria."""
    atk_succ = Counter()
    for r in results:
        if r.get("compliant"):
            atk_succ[r["attack"]] += 1

    if not atk_succ:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "✅ Nessun attacco è andato a buon fine\n→ radar non disponibile",
                ha="center", va="center", fontsize=14, color="#06a77d")
        fig.savefig(out_path)
        plt.close(fig)
        return

    top_attacks = [a for a, _ in atk_succ.most_common(top_n_attacks)]
    cats = sorted({r.get("category", "?") for r in results})

    counter = defaultdict(lambda: [0, 0])
    for r in results:
        counter[(r["attack"], r.get("category", "?"))][0] += 1
        if r.get("compliant"):
            counter[(r["attack"], r.get("category", "?"))][1] += 1

    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_rlabel_position(0)
    max_asr = 0
    for atk in top_attacks:
        vals = []
        for cat in cats:
            t, s = counter[(atk, cat)]
            vals.append((s / t * 100) if t else 0)
        max_asr = max(max_asr, max(vals))
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=atk)
        ax.fill(angles, vals, alpha=0.10)
    ax.set_ylim(0, max(max_asr * 1.2, 5))
    ax.set_title(f"ASR per categoria — top {len(top_attacks)} attacchi", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.05), fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def plot_latency_per_attack(results: list, out_path: Path):
    by_atk = defaultdict(list)
    for r in results:
        if isinstance(r.get("latency_s"), (int, float)) and not r.get("error"):
            by_atk[r["attack"]].append(r["latency_s"])
    items = sorted(by_atk.items(), key=lambda kv: -np.median(kv[1]))
    labels = [k for k, _ in items]
    data = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.30 * len(labels) + 1.5)))
    bp = ax.boxplot(data, vert=False, labels=labels, showfliers=False, patch_artist=True,
                     widths=0.7)
    for patch in bp["boxes"]:
        patch.set_facecolor("#264653")
        patch.set_alpha(0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (s)")
    ax.set_title("Distribuzione latenza per tipo di attacco")
    fig.savefig(out_path)
    plt.close(fig)


def plot_judge_breakdown(results: list, out_path: Path):
    n = len(results)
    refused_regex = sum(1 for r in results if r.get("refused") and not r.get("judge_used"))
    refused_llm = sum(1 for r in results if r.get("refused") and r.get("judge_used"))
    complied = sum(1 for r in results if r.get("compliant"))
    errors = sum(1 for r in results if r.get("error"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # pie
    sizes = [refused_regex, refused_llm, complied, errors]
    labels = [f"REFUSED (regex)\n{refused_regex}",
              f"REFUSED (LLM judge)\n{refused_llm}",
              f"COMPLIED\n{complied}",
              f"Errors\n{errors}"]
    colors = ["#06a77d", "#2a9d8f", "#9d0208", "#9d4edd"]
    sizes_filt = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if sizes_filt:
        ax1.pie([s for s, _, _ in sizes_filt],
                labels=[l for _, l, _ in sizes_filt],
                colors=[c for _, _, c in sizes_filt],
                autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax1.set_title("Breakdown verdetti")

    # bar latency by judge_used
    lat_regex = [r["latency_s"] for r in results if not r.get("judge_used")
                 and isinstance(r.get("latency_s"), (int, float))]
    lat_llm = [r["latency_s"] for r in results if r.get("judge_used")
               and isinstance(r.get("latency_s"), (int, float))]
    ax2.bar(["Regex pre-filter\n(fast path)", "LLM judge\n(ambiguous cases)"],
            [np.mean(lat_regex) if lat_regex else 0, np.mean(lat_llm) if lat_llm else 0],
            color=["#06a77d", "#2a9d8f"], alpha=0.85, edgecolor="white")
    ax2.set_ylabel("Avg latency (s)")
    ax2.set_title("Latenza media per tipo di verdetto")
    for i, v in enumerate([np.mean(lat_regex) if lat_regex else 0,
                            np.mean(lat_llm) if lat_llm else 0]):
        ax2.text(i, v + 0.1, f"{v:.1f}s", ha="center", fontweight="bold")

    fig.savefig(out_path)
    plt.close(fig)


# ── Output strutturati (CSV + MD) ──────────────────────────────────────────────
def export_successes_csv(results: list, out_path: Path):
    rows = [r for r in results if r.get("compliant")]
    fields = ["attack", "defense", "category", "prompt_id", "response_length",
              "judge_used", "confidence", "latency_s"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def export_summary_md(meta: dict, results: list, out_path: Path):
    n = len(results)
    n_succ = sum(1 for r in results if r.get("compliant"))
    asr = n_succ / n * 100 if n else 0
    by_atk = asr_by(results, "attack")[:5]
    by_def = asr_by(results, "defense")[:5]
    by_cat = asr_by(results, "category")[:5]

    lines = [
        f"# Run summary — {meta.get('test_model','?')}",
        "",
        f"- **Judge**: `{meta.get('judge_model','?')}`",
        f"- **Start**: {meta.get('timestamp_start','?')}  →  **End**: {meta.get('timestamp_end','?')}",
        f"- **Elapsed**: {meta.get('elapsed_human','?')}",
        f"- **Total tests**: {n}  ({meta.get('num_prompts','?')} prompts × {meta.get('num_attacks','?')} attacks × {meta.get('num_defenses','?')} defenses)",
        f"- **ASR**: **{asr:.2f}%** ({n_succ} / {n})",
        f"- **Errors**: {sum(1 for r in results if r.get('error'))}",
        f"- **LLM judge calls**: {sum(1 for r in results if r.get('judge_used'))}",
        "",
        "## Top 5 attacchi più efficaci",
        "| Attacco | ASR | Successi |",
        "|---|---|---|",
    ]
    for k, a, s, t in by_atk:
        lines.append(f"| {k} | {a*100:.2f}% | {s}/{t} |")
    lines += [
        "",
        "## Top 5 difese più 'bucate' (ASR alto = difesa debole)",
        "| Difesa | ASR | Successi |",
        "|---|---|---|",
    ]
    for k, a, s, t in by_def:
        lines.append(f"| {k} | {a*100:.2f}% | {s}/{t} |")
    lines += [
        "",
        "## Top 5 categorie più vulnerabili",
        "| Categoria | ASR | Successi |",
        "|---|---|---|",
    ]
    for k, a, s, t in by_cat:
        lines.append(f"| {k} | {a*100:.2f}% | {s}/{t} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── Driver ─────────────────────────────────────────────────────────────────────
def process_run(json_path: Path) -> Path:
    print(f"\n📂 {json_path.name}")
    data = load_run(json_path)
    meta = data.get("metadata", {})
    results = data.get("results", [])
    if not results:
        print("   ⚠️  Nessun risultato.")
        return None

    slug = json_path.stem.replace("eval_", "")
    out_dir = CHARTS_DIR / slug
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"   → {out_dir}/")

    # Aggregazioni base
    atk_rows, def_cols, asr_ad, tot_ad = asr_matrix(results, "attack", "defense")
    atk_rows2, cat_cols, asr_ac, tot_ac = asr_matrix(results, "attack", "category")

    # Plots
    plot_summary_card(meta, results, out_dir / "01_summary_card.png")
    plot_heatmap(atk_rows, def_cols, asr_ad, tot_ad,
                 title=f"ASR Heatmap: Attack × Defense — {meta.get('test_model','?')}",
                 xlabel="Defense", ylabel="Attack",
                 out_path=out_dir / "02_heatmap_attack_defense.png")
    plot_heatmap(atk_rows2, cat_cols, asr_ac, tot_ac,
                 title=f"ASR Heatmap: Attack × Category (no defense scope) — {meta.get('test_model','?')}",
                 xlabel="Category", ylabel="Attack",
                 out_path=out_dir / "03_heatmap_attack_category.png")
    plot_top_vulnerable_pairs(results, out_dir / "04_top_vulnerable_pairs.png")
    plot_asr_ranking(asr_by(results, "attack"),
                     "ASR per tipo di attacco (ranking)", "ASR (%)",
                     out_dir / "05_asr_per_attack.png")
    plot_asr_ranking(asr_by(results, "defense"),
                     "ASR per difesa (ASR alto = difesa più debole)", "ASR (%)",
                     out_dir / "06_asr_per_defense.png")
    plot_asr_ranking(asr_by(results, "category"),
                     "ASR per categoria di prompt", "ASR (%)",
                     out_dir / "07_asr_per_category.png")
    plot_radar_top_attacks(results, out_dir / "08_attack_x_category_radar.png")
    plot_latency_per_attack(results, out_dir / "09_latency_per_attack.png")
    plot_judge_breakdown(results, out_dir / "10_judge_breakdown.png")

    # Output strutturati
    export_successes_csv(results, out_dir / "successes.csv")
    export_summary_md(meta, results, out_dir / "summary.md")

    print("   ✅ Generati 10 grafici + successes.csv + summary.md")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="JSON results file(s). Default: tutti i nuovi.")
    parser.add_argument("--all", action="store_true", help="Rigenera tutti i grafici, anche se esistenti.")
    args = parser.parse_args()

    runs = find_runs(args.paths, args.all)
    print(f"🎨 Generazione grafici per {len(runs)} run.")
    for p in runs:
        process_run(p)
    print(f"\n✅ Output in: {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
