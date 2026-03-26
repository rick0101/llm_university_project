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

# Category color mapping (Academic Palette)
category_colors = {
    "Text LLMs": "#4C72B0",
    "Code LLMs": "#55A868",
    "Vision-Language Models": "#C44E52",
    "AI Agents": "#8172B3",
    "Reasoning Models": "#CCB974",
    "Supply Chain": "#64B5CD"
}

# Setup figure and style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Timeline baseline
ax.axhline(0, color="black", linewidth=1.5, zorder=1)

# Calculate positions to avoid overlap
# We use a combination of year and a small offset for x, and alternating heights for y
levels = []
x_coords = []
current_year_counts = {}

# Define vertical levels to alternate labels above and below the line
# Levels: [1, -1, 2, -2, 3, -3, ...]
level_sequence = [1.2, -1.2, 2.4, -2.4, 3.6, -3.6]

for item in data:
    year = item['year']
    count = current_year_counts.get(year, 0)
    
    # Add a small horizontal jitter within the year to separate points
    x_offset = (count - 1.5) * 0.15 if year != 2026 else (count - 0.5) * 0.2
    x_coords.append(year + x_offset)
    
    # Assign vertical level
    level = level_sequence[count % len(level_sequence)]
    levels.append(level)
    
    current_year_counts[year] = count + 1

# Plotting
for i, item in enumerate(data):
    color = category_colors[item['category']]
    ax.vlines(x_coords[i], 0, levels[i], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.scatter(x_coords[i], levels[i], color=color, s=100, zorder=3, edgecolors="white")
    
    # Label placement
    va = 'bottom' if levels[i] > 0 else 'top'
    ax.text(x_coords[i], levels[i] + (0.1 if levels[i] > 0 else -0.1), 
            item['label'], rotation=0, ha='center', va=va, 
            fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Formatting X-axis
ax.set_xticks([2024, 2025, 2026])
ax.set_xticklabels(["2024", "2025", "2026"], fontsize=12, fontweight='bold')
ax.set_xlim(2023.5, 2026.5)

# Formatting Y-axis (hide it)
ax.get_yaxis().set_visible(False)
for spine in ["left", "top", "right"]:
    ax.spines[spine].set_visible(False)

# Title and Caption
plt.title("Evolution of Data and Model Poisoning Attacks (2024–2026)", 
          fontsize=14, pad=20, fontweight='bold')

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=cat,
                          markerfacecolor=color, markersize=10) 
                   for cat, color in category_colors.items()]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=3, frameon=False, fontsize=10)

# Add caption-like text at the bottom
plt.figtext(0.5, -0.15, 
            "Figure 2: Chronological progression of state-of-the-art backdoor and poisoning attacks, illustrating the shift\n"
            "from text-based models to multimodal systems and autonomous agents.", 
            ha="center", fontsize=10, style='italic')

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()