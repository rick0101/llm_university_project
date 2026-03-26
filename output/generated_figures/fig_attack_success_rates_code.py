OUTPUT_PATH = "output/generated_figures/fig_attack_success_rates.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data points from specification
data = [
    {"exploit": "BadTemplate\n(Chat Template Supply Chain)", "asr": 100.0, "category": "Training-Time"},
    {"exploit": "AutoBackdoor\n(Agent-crafted Backdoor)", "asr": 90.0, "category": "Training-Time"},
    {"exploit": "Cursor Auto Mode\n(Prompt Injection)", "asr": 83.4, "category": "Inference-Time"}
]

# Sort data by ASR for a cleaner visual hierarchy (highest at top)
data = sorted(data, key=lambda x: x['asr'])

def generate_asr_chart():
    # Style configuration
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color mapping for categories
    palette = {"Training-Time": "#4C72B0", "Inference-Time": "#C44E52"}
    
    # Extract values
    names = [d['exploit'] for d in data]
    values = [d['asr'] for d in data]
    categories = [d['category'] for d in data]
    colors = [palette[c] for c in categories]
    
    # Create horizontal bars
    bars = ax.barh(names, values, color=colors, height=0.6, edgecolor='black', linewidth=0.8)
    
    # Add data labels at the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', va='center', fontweight='bold', color='#333333')
    
    # Formatting
    ax.set_title("Reported Attack Success Rates (ASR) of State-of-the-Art Exploits", pad=20)
    ax.set_xlabel("Attack Success Rate (%)", labelpad=10)
    ax.set_xlim(0, 125)  # Extended limit to prevent legend/label overlap
    ax.set_xticks(np.arange(0, 101, 10))
    
    # Create custom legend to avoid overlap with data labels
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=palette[cat], lw=8, label=cat) 
                       for cat in ["Training-Time", "Inference-Time"]]
    
    # Place legend in the upper right corner, well away from the bottom bar labels
    ax.legend(handles=legend_elements, title="Attack Category", 
              loc='upper right', frameon=True, shadow=False, 
              facecolor='white', edgecolor='gray', bbox_to_anchor=(0.98, 0.95))

    # Final layout adjustments
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_asr_chart()