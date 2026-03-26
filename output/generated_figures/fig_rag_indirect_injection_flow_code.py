OUTPUT_PATH = "output/generated_figures/fig_rag_indirect_injection_flow.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set academic style parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
})

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define node styles
    node_style = dict(boxstyle="round,pad=0.5", ec="black", lw=1.5)
    attacker_style = dict(facecolor='#ff9999', **node_style)  # Reddish
    system_style = dict(facecolor='#99ccff', **node_style)    # Bluish
    user_style = dict(facecolor='#99ff99', **node_style)      # Greenish
    data_style = dict(facecolor='#f9f9f9', **node_style)      # Greyish

    # Define node positions (x, y)
    nodes = {
        "Attacker": (1.5, 8.5, attacker_style),
        "Knowledge Base": (8.5, 8.5, data_style),
        "User": (1.5, 1.5, user_style),
        "RAG System\n(Retriever)": (5, 5, system_style),
        "LLM\n(Generator)": (8.5, 1.5, system_style)
    }

    # Draw nodes
    for name, (x, y, style) in nodes.items():
        ax.text(x, y, name, ha='center', va='center', bbox=style, fontweight='bold')

    # Define arrows (start_node, end_node, label, offset_y)
    arrows = [
        ("Attacker", "Knowledge Base", "Injects Poisoned\nDocument", 0.4),
        ("User", "RAG System\n(Retriever)", "1. Query", 0.3),
        ("RAG System\n(Retriever)", "Knowledge Base", "2. Retrieve", 0.3),
        ("Knowledge Base", "RAG System\n(Retriever)", "3. Poisoned\nContext", -0.6),
        ("RAG System\n(Retriever)", "LLM\n(Generator)", "4. Combined\nPrompt", 0.3),
        ("LLM\n(Generator)", "User", "5. Malicious Payload\nExecution", -0.6)
    ]

    # Helper to get coordinates
    def get_coords(name):
        return nodes[name][0], nodes[name][1]

    # Draw arrows with labels
    for start, end, label, label_offset in arrows:
        x1, y1 = get_coords(start)
        x2, y2 = get_coords(end)
        
        # Draw arrow
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.1",
            arrowstyle='-|>',
            mutation_scale=20,
            color='black',
            shrinkA=25,
            shrinkB=25,
            lw=1.2
        )
        ax.add_patch(arrow)
        
        # Add label at midpoint
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + label_offset, label, ha='center', va='center', 
                fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Highlight the "Combined Prompt" content
    ax.text(7.2, 3.5, "Prompt = Query +\nPoisoned Context", 
            fontsize=8, color='darkred', fontweight='bold',
            bbox=dict(facecolor='yellow', alpha=0.3, boxstyle='sawtooth'))

    # Set chart limits and labels to satisfy technical requirements
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Workflow Progression (External to Internal)")
    ax.set_ylabel("System Architecture Layers")
    
    # Remove ticks but keep labels for academic look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Title and Caption
    plt.title("Mechanism of Indirect Prompt Injection in RAG Systems", pad=20)
    fig.text(0.5, 0.02, "Figure 4: The workflow of an indirect prompt injection attack exploiting Retrieval-Augmented Generation (RAG) architectures.", 
             ha='center', fontsize=10, style='italic')

    # Final layout adjustments
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    draw_flowchart()