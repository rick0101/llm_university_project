OUTPUT_PATH = "output/generated_figures/fig_poisoning_efficiency_scatter.png"

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data preparation
data = [
    {
        "attack_method": "Traditional Data Poisoning",
        "poisoning_rate_pct": 5.0,
        "asr_pct": 75.0,
        "category": "Baseline"
    },
    {
        "attack_method": "BadEdit",
        "poisoning_rate_pct": 1.0,
        "asr_pct": 88.0,
        "category": "Knowledge Editing"
    },
    {
        "attack_method": "BADREWARD",
        "poisoning_rate_pct": 0.5,
        "asr_pct": 92.0,
        "category": "RLHF Exploitation"
    },
    {
        "attack_method": "MABA",
        "poisoning_rate_pct": 0.2,
        "asr_pct": 97.0,
        "category": "Multimodal Backdoor"
    }
]

df = pd.DataFrame(data)

# Set academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style("whitegrid", {'grid.linestyle': '--', 'axes.edgecolor': '0.2'})

# Define color palette
palette = sns.color_palette("viridis", n_colors=len(df))

# Plot scatter points
scatter = sns.scatterplot(
    data=df,
    x="poisoning_rate_pct",
    y="asr_pct",
    hue="category",
    style="category",
    s=250,
    palette="viridis",
    edgecolor="black",
    linewidth=1.5,
    ax=ax,
    zorder=5
)

# Set logarithmic scale for X-axis
ax.set_xscale('log')

# Customizing axes
ax.set_xlabel("Poisoning Rate (%) [Log Scale]", fontweight='bold')
ax.set_ylabel("Attack Success Rate (ASR %)", fontweight='bold')
ax.set_title("Evolution of Backdoor Attack Efficiency: Poisoning Rate vs. Attack Success Rate", pad=20)

# Set axis limits for better visualization
ax.set_xlim(0.1, 10)
ax.set_ylim(60, 105)

# Add annotations for each point
for i in range(df.shape[0]):
    ax.text(
        df.poisoning_rate_pct[i] * 1.1, 
        df.asr_pct[i] + 1, 
        df.attack_method[i], 
        fontsize=10, 
        fontweight='bold',
        verticalalignment='bottom'
    )

# Add an arrow to indicate the trend towards higher efficiency
ax.annotate(
    "Increasing Efficiency & Stealth",
    xy=(0.22, 96), xycoords='data',
    xytext=(2.0, 70), textcoords='data',
    arrowprops=dict(arrowstyle="->", color="gray", lw=2, connectionstyle="arc3,rad=-0.2"),
    fontsize=11, color="dimgray", fontstyle='italic'
)

# Formatting ticks
from matplotlib.ticker import ScalarFormatter
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xticks([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])

# Legend placement
plt.legend(title="Attack Category", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

# Add caption-like text below the plot (optional, usually handled by LaTeX)
# plt.figtext(0.5, -0.05, "Figure 3: Comparison of training-time poisoning attacks. Modern techniques like MABA achieve near-perfect ASR with sub-1% poisoning rates.", wrap=True, horizontalalignment='center', fontsize=10)

plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()