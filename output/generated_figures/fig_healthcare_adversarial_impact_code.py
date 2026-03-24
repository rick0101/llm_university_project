OUTPUT_PATH = "output/generated_figures/fig_healthcare_adversarial_impact.png"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data preparation
tasks = ['Vaccination Guidance', 'Medication Prescription']
baseline_acc = [92.0, 89.0]
adversarial_acc = [38.0, 31.0]

# Set academic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(tasks))
width = 0.35

# Professional academic palette (muted blue and red)
color_baseline = "#4C72B0"  # Deep blue
color_adversarial = "#C44E52"  # Deep red

rects1 = ax.bar(x - width/2, baseline_acc, width, label='Baseline (Clean)', 
                color=color_baseline, edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x + width/2, adversarial_acc, width, label='Adversarial Attack', 
                color=color_adversarial, edgecolor='black', linewidth=0.8)

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Impact of Adversarial Prompts on LLM Reliability in Healthcare Tasks (MIMIC-III)', 
             pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylim(0, 105)  # Leave space for labels
ax.legend(frameon=True, loc='upper right')

# Add light grid for quantitative reading
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Function to add value labels on top of bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Add caption-like text below the figure (simulating the requested caption)
caption_text = ("Figure 5: Degradation of model accuracy when subjected to adversarial attacks on critical healthcare tasks\n"
                "using the MIMIC-III dataset. The stark drop highlights the fragility of unhardened LLMs in clinical settings.")
plt.figtext(0.5, -0.05, caption_text, wrap=True, horizontalalignment='center', fontsize=10, style='italic')

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()