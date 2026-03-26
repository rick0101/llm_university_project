OUTPUT_PATH = "output/generated_figures/fig_poisoning_budgets.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Extracting values for plotting
attacks = [d['attack'] for d in data]
rates = [d['poisoning_rate_percent'] for d in data]
targets = [d['target'] for d in data]

# Style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Use a professional academic color palette
colors = sns.color_palette("viridis", n_colors=len(data))

# Create horizontal bar chart
bars = ax.barh(attacks, rates, color=colors, edgecolor='black', linewidth=0.8, height=0.6)

# Set logarithmic scale for X-axis to handle the wide range of values
ax.set_xscale('log')

# Add grid for quantitative readability
ax.grid(True, which="both", ls="-", alpha=0.2, color='gray')
ax.set_axisbelow(True)

# Labels and Title
ax.set_xlabel('Minimum Poisoning Rate (%) - Log Scale', fontweight='bold')
ax.set_ylabel('Attack Framework', fontweight='bold')
ax.set_title('Minimum Poisoning Budgets Required for Successful Backdoor Injection', pad=20)

# Add data labels (the exact percentages) next to the bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width * 1.2, bar.get_y() + bar.get_height()/2, 
            f'{rates[i]}%', 
            va='center', fontweight='bold', color='#333333')

# Create a custom legend for the targets
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=f'{targets[i]}') for i in range(len(targets))]
ax.legend(handles=legend_elements, title="Attack Target Stage", loc='lower right', frameon=True, shadow=False)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()