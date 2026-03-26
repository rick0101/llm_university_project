OUTPUT_PATH = "output/generated_figures/fig_attack_success_rates.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set academic style parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
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
        "exploit": "Cursor Auto Mode\n(Prompt Injection)",
        "asr_percent": 83.4,
        "category": "Inference-Time"
    },
    {
        "exploit": "AutoBackdoor\n(Agent-crafted Backdoor)",
        "asr_percent": 90.0,
        "category": "Training-Time"
    },
    {
        "exploit": "BadTemplate\n(Chat Template Supply Chain)",
        "asr_percent": 100.0,
        "category": "Training-Time"
    }
]

df = pd.DataFrame(data)
# Sort by ASR for better visual flow
df = df.sort_values(by="asr_percent", ascending=True)

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define professional color palette
palette = {"Inference-Time": "#4C72B0", "Training-Time": "#C44E52"}

# Create horizontal bar chart
bars = sns.barplot(
    data=df, 
    x="asr_percent", 
    y="exploit", 
    hue="category", 
    palette=palette,
    dodge=False,
    ax=ax
)

# Add light grid for quantitative readability
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Set labels and title
ax.set_title("Reported Attack Success Rates (ASR) of State-of-the-Art Exploits", pad=20, fontweight='bold')
ax.set_xlabel("Attack Success Rate (ASR %)")
ax.set_ylabel("Exploit Method")
ax.set_xlim(0, 115)  # Leave space for text labels

# Add data labels on the bars
for i, p in enumerate(ax.patches):
    width = p.get_width()
    ax.text(
        width + 1, 
        p.get_y() + p.get_height() / 2, 
        f'{width:.1f}%', 
        va='center', 
        fontweight='bold'
    )

# Refine legend
ax.legend(title="Attack Category", loc='lower right', frameon=True)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()