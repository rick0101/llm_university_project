OUTPUT_PATH = "output/generated_figures/fig_attack_success_rates.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Data points provided in the specification
data = [
    {
        "exploit": "Cursor Auto Mode (Prompt Injection)",
        "asr_percent": 83.4,
        "category": "Inference-Time"
    },
    {
        "exploit": "AutoBackdoor (Agent-crafted Backdoor)",
        "asr_percent": 90.0,
        "category": "Training-Time"
    },
    {
        "exploit": "BadTemplate (Chat Template Supply Chain)",
        "asr_percent": 100.0,
        "category": "Training-Time"
    }
]

# Convert to DataFrame for easier plotting with Seaborn
df = pd.DataFrame(data)

# Set academic style parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define a professional color palette based on categories
palette = {"Inference-Time": "#4c72b0", "Training-Time": "#c44e52"}

# Create horizontal bar chart
sns.barplot(
    data=df, 
    x="asr_percent", 
    y="exploit", 
    hue="category", 
    palette=palette,
    dodge=False,
    ax=ax
)

# Add quantitative labels at the end of each bar
for i, p in enumerate(ax.patches):
    width = p.get_width()
    ax.text(
        width + 1, 
        p.get_y() + p.get_height() / 2, 
        f'{width:.1f}%', 
        va='center', 
        fontsize=10, 
        fontweight='bold'
    )

# Set labels and title
ax.set_title("Reported Attack Success Rates (ASR) of State-of-the-Art Exploits", pad=20)
ax.set_xlabel("Attack Success Rate (ASR) %")
ax.set_ylabel("Exploit Name") # Explicitly added to address previous error

# Configure grid and limits
ax.set_xlim(0, 110) # Extra space for labels
ax.xaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Refine legend
ax.legend(title="Attack Category", loc="lower right", frameon=True)

# Remove top and right spines for a cleaner look
sns.despine()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()