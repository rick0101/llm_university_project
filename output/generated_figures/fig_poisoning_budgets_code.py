OUTPUT_PATH = "output/generated_figures/fig_poisoning_budgets.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data configuration
data = [
    {"attack": "Persistent Backdoors\n(Zhang et al., 2024)", "rate": 0.001, "target": "Pre-training"},
    {"attack": "Subversive Alignment\nInjection (SAI)", "rate": 0.1, "target": "Alignment (LoRA)"},
    {"attack": "VisualTrap", "rate": 5.0, "target": "Visual Grounding"}
]

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

def create_poisoning_budget_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    attacks = [d['attack'] for d in data]
    rates = [d['rate'] for d in data]
    targets = [d['target'] for d in data]
    
    # Color palette mapping
    unique_targets = list(dict.fromkeys(targets))
    colors = sns.color_palette("viridis", len(unique_targets))
    target_color_map = dict(zip(unique_targets, colors))
    bar_colors = [target_color_map[t] for t in targets]

    # Create horizontal bar chart
    bars = ax.barh(attacks, rates, color=bar_colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    
    # Logarithmic scale for x-axis
    ax.set_xscale('log')
    
    # Labels and Title
    ax.set_xlabel('Minimum Poisoning Rate (%) — Log Scale', fontweight='bold')
    ax.set_ylabel('Attack Framework', fontweight='bold')
    ax.set_title('Minimum Poisoning Budgets Required for Successful Backdoor Injection', pad=20)
    
    # Add grid for quantitative readability
    ax.grid(True, which="both", ls="-", alpha=0.2, axis='x')
    ax.set_axisbelow(True)

    # Annotate bars with exact values and target info
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width * 1.2, bar.get_y() + bar.get_height()/2, 
                f"{width}% ({targets[i]})", 
                va='center', fontsize=10, fontweight='bold', color='#333333')

    # Legend for targets
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=target_color_map[t], lw=6, label=t) for t in unique_targets]
    ax.legend(handles=legend_elements, title="Attack Target Stage", loc='lower right', frameon=True)

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_poisoning_budget_chart()