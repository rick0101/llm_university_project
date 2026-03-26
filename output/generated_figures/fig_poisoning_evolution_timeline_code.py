OUTPUT_PATH = "output/generated_figures/fig_poisoning_evolution_timeline.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data points provided in the specification
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

# Define category order to illustrate the shift from text to agents/multimodal
categories = [
    "Text LLMs", 
    "Code LLMs", 
    "Vision-Language Models", 
    "Reasoning Models", 
    "AI Agents", 
    "Supply Chain"
]

# Mapping categories to y-axis indices
cat_to_y = {cat: i for i, cat in enumerate(categories)}

# Setup figure and style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

fig, ax = plt.subplots(figsize=(10, 6))

# Color palette (Academic/Professional)
colors = plt.cm.get_cmap('Set2', len(categories))

# Plotting logic
# We use a dictionary to track how many items are in a specific (year, category) 
# to apply a small vertical offset for label clarity.
offset_tracker = {}

for item in data:
    x = item['year']
    y_base = cat_to_y[item['category']]
    
    # Calculate offset for overlapping points
    key = (x, y_base)
    offset_count = offset_tracker.get(key, 0)
    offset_tracker[key] = offset_count + 1
    
    # Apply a small vertical jitter for the label if multiple items exist
    y_offset = offset_count * 0.25 - 0.1
    
    # Plot the point
    ax.scatter(x, y_base, color=colors(y_base), s=100, edgecolors='black', zorder=3)
    
    # Add the label text
    ax.text(x + 0.05, y_base + y_offset, item['label'], 
            verticalalignment='center', fontsize=9, fontweight='medium')

# Formatting axes
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories)

ax.set_xticks([2024, 2025, 2026])
ax.set_xlim(2023.5, 2026.8)
ax.set_ylim(-0.5, len(categories) - 0.5)

# Adding required labels and title
ax.set_xlabel("Publication Year", fontweight='bold')
ax.set_ylabel("Target Modality / System Category", fontweight='bold')
ax.set_title("Evolution of Data and Model Poisoning Attacks (2024–2026)", pad=20)

# Add a light grid for readability
ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a subtle background highlight for the years
for year in [2024, 2025, 2026]:
    ax.axvspan(year - 0.45, year + 0.45, color='gray', alpha=0.03, zorder=1)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()