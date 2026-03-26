OUTPUT_PATH = "output/generated_figures/fig_llm_security_taxonomy.png"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration and Data
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
COLORS = {'root': '#2E4053', 'train': '#27AE60', 'inf': '#C0392B', 'out': '#8E44AD'}

# Node structure: (id, label, parent_id, x, y, color_key)
nodes = [
    ('root', 'LLM Security Threats', None, 5, 10, 'root'),
    # Level 1
    ('t1', 'Training-Time', 'root', 1.5, 8, 'train'),
    ('i1', 'Inference-Time', 'root', 5, 8, 'inf'),
    ('o1', 'Output-Level', 'root', 8.5, 8, 'out'),
    # Level 2 - Training
    ('t2_1', 'Data Poisoning', 't1', 0.5, 6, 'train'),
    ('t2_2', 'Model Poisoning\n(CGBA, TrojVLM, BadTemplate)', 't1', 2.5, 6, 'train'),
    # Level 2 - Inference
    ('i2_1', 'Prompt Injection', 'i1', 3.8, 6, 'inf'),
    ('i2_2', 'Jailbreaking\n(AutoDAN, FlipAttack)', 'i1', 5.2, 6, 'inf'),
    ('i2_3', 'Specialized Domain\nAttacks (XOXO, INSEC)', 'i1', 6.8, 6, 'inf'),
    # Level 2 - Output
    ('o2_1', 'Intrinsic\nHallucinations', 'o1', 8, 6, 'out'),
    ('o2_2', 'Extrinsic\nHallucinations', 'o1', 9.5, 6, 'out')
]

def draw_taxonomy():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(4.5, 11)
    ax.axis('off')

    pos_map = {n[0]: (n[3], n[4]) for n in nodes}
    
    # Draw connections first so they are behind boxes
    for nid, label, parent, x, y, c_key in nodes:
        if parent:
            px, py = pos_map[parent]
            # Draw orthogonal-style lines
            ax.plot([px, px], [py - 0.3, y + 0.5], color='gray', lw=1, zorder=1)
            ax.annotate('', xy=(x, y + 0.35), xytext=(px, y + 0.5),
                        arrowprops=dict(arrowstyle='-', color='gray', lw=1))

    # Draw nodes
    for nid, label, parent, x, y, c_key in nodes:
        color = COLORS[c_key]
        is_root = parent is None
        bbox_props = dict(boxstyle="round,pad=0.5", fc='white', ec=color, lw=2)
        
        # Add a subtle background fill for the boxes
        ax.text(x, y, label, ha='center', va='center', bbox=bbox_props,
                fontsize=11 if is_root else 10, fontweight='bold' if is_root else 'normal',
                zorder=3)
        
        # Add colored header strip for Level 1
        if parent == 'root' or is_root:
            rect = mpatches.FancyBboxPatch((x-0.6, y-0.35), 1.2, 0.7, 
                                          boxstyle="round,pad=0.02", fc=color, alpha=0.1, zorder=2)
            ax.add_patch(rect)

    plt.title("Comprehensive Taxonomy of LLM Security Threats", fontsize=16, fontweight='bold', pad=20)
    
    # Add Caption
    caption = "Figure 1: A systematic taxonomy of LLM vulnerabilities categorized by model lifecycle phases: Training-Time, Inference-Time, and Output-Level."
    fig.text(0.5, 0.05, caption, ha='center', fontsize=11, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    draw_taxonomy()