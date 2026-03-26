OUTPUT_PATH = "output/generated_figures/fig_poisoning_efficiency_scatter.png"

import matplotlib
matplotlib.use('Agg')
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

# Style configuration for academic publication
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (10, 6)
})

# Create the plot
fig, ax = plt.subplots()

# Use a professional color palette
palette = sns.color_palette("viridis", n_colors=len(df))

# Scatter plot
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
    alpha=0.85,
    ax=ax
)

# Set logarithmic scale for X-axis as requested
ax.set_xscale('log')

# Customizing the axes
ax.set_xlabel("Poisoning Rate (%) [Log Scale]", fontweight='bold')
ax.set_ylabel("Attack Success Rate (ASR %)", fontweight='bold')
ax.set_title("Evolution of Backdoor Attack Efficiency: Poisoning Rate vs. Attack Success Rate", pad=20)

# Set axis limits for better visualization
ax.set_xlim(0.1, 10)
ax.set_ylim(60, 105)

# Add grid for quantitative readability
ax.grid(True, which="both", linestyle='--', alpha=0.4)

# Annotate points with method names
for i, row in df.iterrows():
    ax.annotate(
        row['attack_method'],
        (row['poisoning_rate_pct'], row['asr_pct']),
        textcoords="offset points",
        xytext=(0, 12),
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

# Add an arrow to indicate the trend towards efficiency
ax.annotate(
    "Increasing Efficiency\n(Lower Poisoning, Higher ASR)",
    xy=(0.25, 95),
    xytext=(1.5, 70),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
    fontsize=10,
    fontstyle='italic',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
)

# Legend placement
plt.legend(title="Attack Category", loc='lower left', frameon=True, shadow=False)

# Add caption-like text at the bottom (simulating figure caption)
caption_text = "Figure 3: Comparison of training-time poisoning attacks. Modern techniques like MABA achieve near-perfect\nAttack Success Rates (ASR) with sub-1% poisoning rates, demonstrating increasing stealth and potency."
plt.figtext(0.5, -0.05, caption_text, wrap=True, horizontalalignment='center', fontsize=10)

# Final layout adjustment
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()