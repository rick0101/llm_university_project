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

# Configuration for academic styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

# Define categories and their order for the Y-axis
categories = [
    "Text LLMs", 
    "Code LLMs", 
    "Vision-Language Models", 
    "Reasoning Models", 
    "AI Agents", 
    "Supply Chain"
]
cat_to_y = {cat: i for i, cat in enumerate(categories)}

# Color palette (Academic/Professional)
colors = plt.cm.get_cmap('viridis', len(categories))
cat_colors = {cat: colors(i) for i, cat in enumerate(categories)}

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Track counts per (year, category) to apply vertical jitter/offset for readability
offset_tracker = {}

for item in data:
    year = item['year']
    cat = item['category']
    y_base = cat_to_y[cat]
    
    # Calculate offset to prevent overlapping labels in the same year/category
    key = (year, cat)
    offset_tracker[key] = offset_tracker.get(key, 0) + 1
    y_offset = (offset_tracker[key] - 1) * 0.25 - 0.15 # Stagger vertically
    
    # Plot the point
    ax.scatter(year, y_base + y_offset, color=cat_colors[cat], s=100, edgecolors='black', zorder=3)
    
    # Add the label
    ax.text(year + 0.05, y_base + y_offset, item['label'], 
            va='center', ha='left', fontsize=9, fontweight='medium',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Formatting the axes
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories)
ax.set_xticks([2024, 2025, 2026])
ax.set_xlim(2023.5, 2026.8)
ax.set_ylim(-0.5, len(categories) - 0.5)

# Required Labels
ax.set_xlabel("Year of Publication", fontweight='bold')
ax.set_ylabel("Target Modality / Attack Category", fontweight='bold')
ax.set_title("Evolution of Data and Model Poisoning Attacks (2024–2026)", pad=20)

# Add a light grid for readability
ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
ax.set_axisbelow(True)

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add caption as a text box at the bottom
caption_text = "Figure 2: Chronological progression of state-of-the-art backdoor and poisoning attacks, illustrating the shift\nfrom text-based models to multimodal systems and autonomous agents."
plt.figtext(0.5, -0.05, caption_text, wrap=True, horizontalalignment='center', fontsize=10, style='italic')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()