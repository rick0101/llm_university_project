OUTPUT_PATH = "output/generated_figures/fig_attack_success_rates.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Extracting values for plotting
exploits = [d['exploit'] for d in data]
asr_values = [d['asr_percent'] for d in data]
categories = [d['category'] for d in data]

# Academic styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors based on categories
palette = {"Inference-Time": "#4C72B0", "Training-Time": "#C44E52"}
colors = [palette[cat] for cat in categories]

# Create horizontal bars
# Increased bar height and spacing to prevent text overlap
bars = ax.barh(exploits, asr_values, color=colors, height=0.6, edgecolor='black', linewidth=0.8)

# Add data labels at the end of each bar
for bar in bars:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}%', 
            va='center', ha='left', fontweight='bold', color='black')

# Formatting axes
ax.set_xlim(0, 115)  # Extra space for labels
ax.set_xlabel("Attack Success Rate (ASR) %")
ax.set_title("Reported Attack Success Rates (ASR) of State-of-the-Art Exploits", pad=20)

# Add grid for readability
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=palette["Inference-Time"], lw=4, label='Inference-Time'),
                   Line2D([0], [0], color=palette["Training-Time"], lw=4, label='Training-Time')]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', edgecolor='gray')

# Adjust layout to prevent label clipping or overlap
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()