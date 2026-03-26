OUTPUT_PATH = "output/generated_figures/fig_poisoning_evolution_timeline.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data definition
data = [
    {"label": "CGBA (Text)", "year": 2024, "cat": "Text LLMs"},
    {"label": "CODEBREAKER (Code)", "year": 2024, "cat": "Code LLMs"},
    {"label": "Persistent Backdoors", "year": 2024, "cat": "Text LLMs"},
    {"label": "TrojVLM (I2T)", "year": 2024, "cat": "Vision-Language"},
    {"label": "IAG (Visual Grounding)", "year": 2025, "cat": "Vision-Language"},
    {"label": "VisualTrap (GUI)", "year": 2025, "cat": "AI Agents"},
    {"label": "BadVLA (VLA)", "year": 2025, "cat": "AI Agents"},
    {"label": "AutoBackdoor", "year": 2025, "cat": "AI Agents"},
    {"label": "DecepChain (Reasoning)", "year": 2025, "cat": "Reasoning Models"},
    {"label": "BoT (Reasoning)", "year": 2025, "cat": "Reasoning Models"},
    {"label": "SlowBA (Latency)", "year": 2026, "cat": "AI Agents"},
    {"label": "BadTemplate (Infra)", "year": 2026, "cat": "Supply Chain"}
]

# Setup styling
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.get_cmap('Set2', 6)
cat_map = {cat: colors(i) for i, cat in enumerate(sorted(list(set(d['cat'] for d in data))))}

# Timeline configuration
years = [2024, 2025, 2026]
ax.axhline(0, color='black', linewidth=1.5, zorder=1)
ax.set_xlim(2023.5, 2026.5)
ax.set_ylim(-7, 7)

# Distribute points within years to avoid overlap
year_offsets = {2024: np.linspace(-0.35, 0.35, 4), 
                2025: np.linspace(-0.4, 0.4, 6), 
                2026: np.linspace(-0.2, 0.2, 2)}
year_counters = {2024: 0, 2025: 0, 2026: 0}

# Plotting logic
for i, d in enumerate(data):
    y_val = d['year']
    x_pos = y_val + year_offsets[y_val][year_counters[y_val]]
    year_counters[y_val] += 1
    
    # Alternate heights for labels
    level = 5 if i % 2 == 0 else -5
    if abs(x_pos - round(x_pos)) < 0.1: level *= 1.2 # Extra offset for center items
    
    # Draw stem and marker
    color = cat_map[d['cat']]
    ax.vlines(x_pos, 0, level, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.scatter(x_pos, level, color=color, s=100, zorder=3, edgecolors='white')
    
    # Add text label
    va = 'bottom' if level > 0 else 'top'
    ax.text(x_pos, level + (0.3 if level > 0 else -0.3), d['label'], 
            ha='center', va=va, fontweight='bold', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Formatting
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], fontsize=12, fontweight='bold')
ax.get_yaxis().set_visible(False)
for spine in ['left', 'top', 'right']: ax.spines[spine].set_visible(False)

# Legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_map[c], markersize=10, label=c) 
           for c in cat_map]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

plt.title("Evolution of Data and Model Poisoning Attacks (2024–2026)", fontsize=14, pad=20, fontweight='bold')
caption = "Figure 2: Chronological progression of state-of-the-art backdoor and poisoning attacks, illustrating the shift\nfrom text-based models to multimodal systems and autonomous agents."
fig.text(0.5, 0.01, caption, ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()