OUTPUT_PATH = "output/generated_figures/fig_poisoning_evolution_timeline.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data points for the timeline
data = [
    {"label": "CGBA (Text)", "year": 2024, "category": "Text LLMs"},
    {"label": "CODEBREAKER (Code)", "year": 2024, "category": "Code LLMs"},
    {"label": "Persistent Backdoors", "year": 2024, "category": "Text LLMs"},
    {"label": "TrojVLM (Image-to-Text)", "year": 2024, "category": "Vision-Language Models"},
    {"label": "IAG (Visual Grounding)", "year": 2025, "category": "Vision-Language Models"},
    {"label": "VisualTrap (GUI Agents)", "year": 2025, "category": "AI Agents"},
    {"label": "BadVLA (VLA Systems)", "year": 2025, "category": "AI Agents"},
    {"label": "AutoBackdoor", "year": 2025, "category": "AI Agents"},
    {"label": "DecepChain (Reasoning)", "year": 2025, "category": "Reasoning Models"},
    {"label": "Breaking of Thought (BoT)", "year": 2025, "category": "Reasoning Models"},
    {"label": "SlowBA (Latency)", "year": 2026, "category": "AI Agents"},
    {"label": "BadTemplate (Infrastructure)", "year": 2026, "category": "Supply Chain"}
]

# Category color mapping (Academic palette)
category_colors = {
    "Text LLMs": "#4C72B0",
    "Code LLMs": "#55A868",
    "Vision-Language Models": "#C44E52",
    "AI Agents": "#8172B3",
    "Reasoning Models": "#CCB974",
    "Supply Chain": "#64B5CD"
}

# Setup figure and style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 11
})

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Calculate positions to avoid overlap
# We jitter the x-coordinates within each year and alternate stem heights
years = sorted(list(set(d['year'] for d in data)))
year_counts = {y: 0 for y in years}
for d in data:
    year_counts[d['year']] += 1

# Stem heights to alternate and prevent label collision
# We use a pattern that spreads labels vertically
height_pattern = [1.2, -1.2, 2.4, -2.4, 3.6, -3.6]

current_year_idx = {y: 0 for y in years}
x_coords = []
y_coords = []

# Draw the central timeline axis
ax.axhline(0, color='black', linewidth=1.5, zorder=1)

# Plot each data point
for i, d in enumerate(data):
    year = d['year']
    count = year_counts[year]
    idx = current_year_idx[year]
    
    # Calculate x position with jittering
    # Spread points across the year interval [year, year + 0.8]
    x = year + (idx / max(1, count)) * 0.8
    
    # Assign height from pattern
    y = height_pattern[idx % len(height_pattern)]
    
    color = category_colors[d['category']]
    
    # Draw stem
    ax.vlines(x, 0, y, color='grey', linestyle='--', linewidth=0.8, alpha=0.6, zorder=2)
    
    # Draw marker on the timeline
    ax.scatter(x, 0, color=color, s=60, zorder=3, edgecolors='white')
    
    # Add label text
    va = 'bottom' if y > 0 else 'top'
    # Add a small offset to the text to prevent clipping and improve readability
    ax.text(x, y + (0.1 if y > 0 else -0.1), d['label'], 
            rotation=0, ha='center', va=va, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    current_year_idx[year] += 1

# Formatting the axes
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], fontweight='bold')
ax.set_yticks([]) # Hide y-axis values

# Set limits with extra padding to prevent label clipping (especially for BadTemplate)
ax.set_xlim(min(years) - 0.5, max(years) + 1.2)
ax.set_ylim(-4.5, 4.5)

# Remove spines
for spine in ['left', 'right', 'top']:
    ax.spines[spine].set_visible(False)

# Add Title and Caption
plt.title("Evolution of Data and Model Poisoning Attacks (2024–2026)", pad=20)
fig.text(0.5, 0.02, "Figure 2: Chronological progression of state-of-the-art backdoor and poisoning attacks, illustrating the shift\nfrom text-based models to multimodal systems and autonomous agents.", 
         ha='center', fontsize=10, style='italic')

# Create a custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=cat,
                          markerfacecolor=color, markersize=10)
                   for cat, color in category_colors.items()]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), 
          ncol=2, frameon=True, fontsize=9, title="Target Modality")

# Final layout adjustment
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()