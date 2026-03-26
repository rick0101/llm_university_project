OUTPUT_PATH = "output/generated_figures/fig_poisoning_budgets.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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

# Data preparation
data = [
    {
        "attack": "Persistent Backdoors\n(Zhang et al., 2024)",
        "poisoning_rate_percent": 0.001,
        "target": "Pre-training"
    },
    {
        "attack": "Subversive Alignment\nInjection (SAI)",
        "poisoning_rate_percent": 0.1,
        "target": "Alignment (LoRA)"
    },
    {
        "attack": "VisualTrap",
        "poisoning_rate_percent": 5.0,
        "target": "Visual Grounding"
    }
]

df = pd.DataFrame(data)

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Use a professional academic color palette
colors = sns.color_palette("viridis", n_colors=len(df))

# Create horizontal bar chart for better label readability
bars = ax.barh(df['attack'], df['poisoning_rate_percent'], color=colors, edgecolor='black', linewidth=0.8)

# Set logarithmic scale for the x-axis as requested
ax.set_xscale('log')

# Add gridlines for quantitative clarity
ax.grid(True, which="both", ls="-", alpha=0.2, color='gray')
ax.set_axisbelow(True)

# Labels and Title
ax.set_title("Minimum Poisoning Budgets Required for Successful Backdoor Injection", pad=20, fontweight='bold')
ax.set_xlabel("Poisoning Rate (%) - Logarithmic Scale", labelpad=10)
ax.set_ylabel("Attack Framework", labelpad=10)

# Add text labels for the exact percentages on the bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    label_text = f"{df['poisoning_rate_percent'][i]}%"
    ax.text(width * 1.15, bar.get_y() + bar.get_height()/2, label_text, 
            va='center', fontweight='bold', color='#333333')

# Add a legend for the target stages
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=f"{df['target'][i]}") for i in range(len(df))]
ax.legend(handles=legend_elements, title="Target Stage", loc='lower right', frameon=True)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()