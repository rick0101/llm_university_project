OUTPUT_PATH = "output/generated_figures/fig_poisoning_budgets.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
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

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Use a professional academic palette
colors = sns.color_palette("viridis", n_colors=len(df))

# Horizontal bar chart for better readability of attack names
bars = ax.barh(df['attack'], df['poisoning_rate_percent'], color=colors, edgecolor='black', linewidth=0.8)

# Set logarithmic scale for the x-axis as requested
ax.set_xscale('log')

# Add gridlines for the log scale
ax.grid(True, which="both", ls="-", alpha=0.2, color='gray')
ax.set_axisbelow(True)

# Labels and Title
ax.set_xlabel("Minimum Poisoning Rate (%) — Log Scale", fontweight='bold')
ax.set_ylabel("Attack Framework", fontweight='bold')
ax.set_title("Minimum Poisoning Budgets Required for Successful Backdoor Injection", pad=20)

# Add text labels on the bars for clarity
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width * 1.1, bar.get_y() + bar.get_height()/2, 
            f'{width}%', 
            va='center', fontweight='bold', color='#333333')

# Create a custom legend for the targets
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=f"{df['target'][i]}") for i in range(len(df))]
ax.legend(handles=legend_elements, title="Training Stage / Target", loc='lower right', frameon=True)

# Adjust layout to prevent clipping
plt.tight_layout()

# Add caption-like text at the bottom (optional, usually handled by LaTeX, but included for completeness)
# fig.text(0.5, -0.05, "Figure 3: Comparison of the minimum percentage of poisoned data required to successfully compromise LLMs and VLMs across different attack frameworks.", 
#          ha='center', wrap=True, fontsize=10, style='italic')

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()