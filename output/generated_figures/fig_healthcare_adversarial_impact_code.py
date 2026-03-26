OUTPUT_PATH = "output/generated_figures/fig_healthcare_adversarial_impact.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data preparation
tasks = ['Vaccination Guidance', 'Medication Prescription']
baseline_accuracy = [92.0, 89.0]
adversarial_accuracy = [38.0, 31.0]

# Set visual style for academic publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300
})

# Plotting configuration
x = np.arange(len(tasks))
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create grouped bars
# Using a professional academic palette (muted blue and red-orange)
rects1 = ax.bar(x - width/2, baseline_accuracy, width, 
                label='Baseline (Clean)', color='#4C72B0', edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x + width/2, adversarial_accuracy, width, 
                label='Adversarial Attack', color='#C44E52', edgecolor='black', linewidth=0.8)

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Clinical Task (MIMIC-III Dataset)')
ax.set_title('Impact of Adversarial Prompts on LLM Reliability in Healthcare Tasks (MIMIC-III)', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylim(0, 105)  # Leave space for labels
ax.legend(loc='upper right', frameon=True)

# Add a light grid for quantitative readability
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

# Function to attach a text label above each bar, displaying its height.
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()