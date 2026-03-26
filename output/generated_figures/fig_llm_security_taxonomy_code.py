OUTPUT_PATH = "output/generated_figures/fig_llm_security_taxonomy.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_taxonomy():
    # Set academic style parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define colors (Academic palette)
    colors = {
        'root': '#2c3e50', 'phase': '#2980b9', 'category': '#34495e', 'example': '#f8f9fa'}
    
    def draw_node(x, y, text, color, tcolor='white', fs=10, fw='bold'):
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight=fw,
                color=tcolor, bbox=dict(facecolor=color, edgecolor='#2c3e50', 
                boxstyle='round,pad=0.5', lw=1), zorder=5)

    # Root Node
    draw_node(5, 9.2
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()