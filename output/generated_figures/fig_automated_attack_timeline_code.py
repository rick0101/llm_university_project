OUTPUT_PATH = "output/generated_figures/fig_automated_attack_timeline.png"

import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Data points
data = [
    {
        "label": "Manual Role-Playing\n& Heuristics",
        "year": 2023,
        "category": "Manual",
        "offset": 1.8
    },
    {
        "label": "Greedy Coordinate\nGradient (GCG)",
        "year": 2023,
        "category": "Automated White-box",
        "offset": -1.8
    },
    {
        "label": "Knowledge Editing\nBackdoors (BadEdit)",
        "year": 2024,
        "category": "Training/Weights",
        "offset": 1.2
    },
    {
        "label": "MCTS Fuzzing\n(AGENTVIGIL)",
        "year": 2025,
        "category": "Automated Black-box",
        "offset": 1.8
    },
    {
        "label": "Multimodal Alignment\nExploits (FOA-Attack)",
        "year": 2025,
        "category": "Multimodal",
        "offset": -1.8
    }
]

# Style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Color mapping for categories
category_colors = {
    "Manual": "#4C72B0",            # Muted Blue
    "Automated White-box": "#55A868", # Muted Green
    "Training/Weights": "#C44E52",    # Muted Red
    "Automated Black-box": "#8172B3", # Muted Purple
    "Multimodal": "#CCB974"           # Muted Gold
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Draw the main timeline axis
ax.axhline(0, color='black', linewidth=1.5, zorder=1)

# Set x-axis limits and ticks
years = [d['year'] for d in data]
ax.set_xlim(min(years) - 0.5, 2026.5)
ax.set_xticks(np.arange(min(years), 2027, 1))
ax.set_ylim(-3, 3)

# Remove y-axis
ax.get_yaxis().set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Plot data points
for item in data:
    color = category_colors[item['category']]
    
    # Draw vertical stem
    ax.vlines(item['year'], 0, item['offset'], color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    
    # Draw marker on the timeline
    ax.scatter(item['year'], 0, color=color, s=120, zorder=3, edgecolors='white', linewidth=1.5)
    
    # Add label text
    va = 'bottom' if item['offset'] > 0 else 'top'
    ax.text(item['year'], item['offset'] + (0.1 if item['offset'] > 0 else -0.1), 
            item['label'], 
            ha='center', va=va, 
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3', alpha=0.9))

# Add Title and Caption
plt.title("Chronological Evolution of Inference-Time Attack Methodologies", pad=20, fontweight='bold')
fig.text(0.5, -0.02, "Figure 4: The rapid transition from manual heuristic exploits to sophisticated, automated adversarial discovery frameworks between 2023 and 2026.", 
         ha='center', fontsize=10, style='italic', wrap=True)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=cat,
                          markerfacecolor=color, markersize=10)
                   for cat, color in category_colors.items()]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9, title="Attack Category")

# Add subtle grid for years
ax.xaxis.grid(True, linestyle=':', alpha=0.4)

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()