OUTPUT_PATH = "output/generated_figures/fig_automated_attack_timeline.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set academic style parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

# Data points
data = [
    {"label": "Manual Role-Playing\n& Heuristics", "year": 2023, "level": 1, "category": "Manual"},
    {"label": "Greedy Coordinate\nGradient (GCG)", "year": 2023.3, "level": 2, "category": "Automated White-box"},
    {"label": "Knowledge Editing\nBackdoors (BadEdit)", "year": 2024, "level": 3, "category": "Training/Weights"},
    {"label": "MCTS Fuzzing\n(AGENTVIGIL)", "year": 2025, "level": 4, "category": "Automated Black-box"},
    {"label": "Multimodal Alignment\nExploits (FOA-Attack)", "year": 2025.3, "level": 5, "category": "Multimodal"}
]

# Extracting values
years = [d["year"] for d in data]
levels = [d["level"] for d in data]
labels = [d["label"] for d in data]
categories = [d["category"] for d in data]

# Color palette for categories
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(data)))

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the progression line
ax.plot(years, levels, color='gray', linestyle='--', alpha=0.5, zorder=1)

# Plot individual points
for i, (x, y, label, color) in enumerate(zip(years, levels, labels, colors)):
    ax.scatter(x, y, s=150, color=color, edgecolor='black', zorder=3, label=categories[i])
    # Annotate labels with slight offset
    ax.text(x + 0.05, y, label, verticalalignment='center', fontsize=10, fontweight='bold')

# Formatting axes
ax.set_xlim(2022.5, 2026.5)
ax.set_ylim(0.5, 6)
ax.set_xticks([2023, 2024, 2025, 2026])
ax.set_yticks(range(1, 6))
ax.set_yticklabels(["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"])

# Required Labels
ax.set_xlabel("Year of Publication / Discovery")
ax.set_ylabel("Methodology Sophistication & Automation Index")
ax.set_title("Chronological Evolution of Inference-Time Attack Methodologies", pad=20)

# Grid and Spines
ax.grid(True, linestyle=':', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
handles, labels_legend = ax.get_legend_handles_labels()
ax.legend(handles, labels_legend, loc='upper left', title="Attack Category", frameon=True)

# Add Caption
caption = "Figure 4: The rapid transition from manual heuristic exploits to sophisticated, automated adversarial discovery frameworks between 2023 and 2026."
plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=10, style='italic')

# Adjust layout
plt.tight_layout()

# Save and close
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()