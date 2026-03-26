OUTPUT_PATH = "output/generated_figures/fig_attack_success_rates.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data preparation
data = [
    {
        "exploit": "BadTemplate\n(Chat Template Supply Chain)",
        "asr_percent": 100.0,
        "category": "Training-Time"
    },
    {
        "exploit": "AutoBackdoor\n(Agent-crafted Backdoor)",
        "asr_percent": 90.0,
        "category": "Training-Time"
    },
    {
        "exploit": "Cursor Auto Mode\n(Prompt Injection)",
        "asr_percent": 83.4,
        "category": "Inference-Time"
    }
]

df = pd.DataFrame(data)

# Set academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define professional color palette
# Using a custom palette to distinguish categories
palette = {"Training-Time": "#4C72B0", "Inference-Time": "#C44E52"}

# Plot horizontal bars
sns.barplot(
    data=df, 
    x="asr_percent", 
    y="exploit", 
    hue="category", 
    palette=palette,
    dodge=False,
    ax=ax
)

# Add quantitative grid
ax.xaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Set limits and labels
ax.set_xlim(0, 110)  # Extra space for text labels
ax.set_xlabel("Attack Success Rate (ASR) %", fontweight='bold')
ax.set_ylabel("Exploit Method", fontweight='bold')
ax.set_title("Reported Attack Success Rates (ASR) of State-of-the-Art Exploits", pad=20)

# Add percentage labels at the end of each bar
for i, p in enumerate(ax.patches):
    width = p.get_width()
    ax.text(
        width + 1, 
        p.get_y() + p.get_height() / 2, 
        f'{width:.1f}%', 
        va='center', 
        fontsize=11, 
        fontweight='bold'
    )

# Refine legend
ax.legend(title="Threat Stage", loc='lower right', frameon=True, shadow=False)

# Remove top and right spines for a clean look
sns.despine()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()