OUTPUT_PATH = "output/generated_figures/fig_rag_indirect_injection_flow.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    # Set academic style parameters
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define colors (Academic Palette)
    colors = {
        'attacker': '#d62728',    # Red
        'user': '#2ca02c',        # Green
        'system': '#1f77b4',      # Blue
        'data': '#7f7f7f',        # Grey
        'llm': '#9467bd',         # Purple
        'warning': '#ff7f0e'      # Orange
    }

    # Helper function to draw boxes
    def draw_box(x, y, width, height, text, color, header=None):
        rect = patches.FancyBboxPatch(
            (x, y), width, height, 
            boxstyle="round,pad=0.1", 
            linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.15
        )
        ax.add_patch(rect)
        
        # Add a solid top border for a professional look
        ax.plot([x, x+width], [y+height, y+height], color=color, linewidth=3)
        
        if header:
            ax.text(x + width/2, y + height - 0.3, header, 
                    ha='center', va='center', fontweight='bold', color=color)
            ax.text(x + width/2, y + (height-0.3)/2, text, 
                    ha='center', va='center', wrap=True)
        else:
            ax.text(x + width/2, y + height/2, text, 
                    ha='center', va='center', fontweight='bold', wrap=True)

    # Helper function to draw arrows
    def draw_arrow(start, end, text="", color='black', connectionstyle="arc3,rad=0"):
        ax.annotate(text, xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, 
                                   connectionstyle=connectionstyle),
                    ha='center', va='bottom', fontsize=9)

    # 1. Attacker Node
    draw_box(0.5, 6.0, 2.5, 1.2, "Uploads Poisoned\nDocument/Website", colors['attacker'], "Attacker")

    # 2. External Knowledge Base
    draw_box(6.5, 6.0, 3.0, 1.2, "Vector DB / Web Index\n(Contains Malicious Payload)", colors['data'], "Knowledge Base")

    # 3. Legitimate User
    draw_box(0.5, 1.0, 2.5, 1.2, "Submits Legitimate\nQuery", colors['user'], "Legitimate User")

    # 4. RAG System (Retriever)
    draw_box(4.0, 1.0, 2.5, 1.2, "Retrieves Context\nfrom Knowledge Base", colors['system'], "RAG Retriever")

    # 5. LLM
    draw_box(7.5, 1.0, 2.0, 1.2, "Processes Prompt +\nPoisoned Context", colors['llm'], "LLM")

    # --- Connections ---
    
    # Step 1: Poisoning
    draw_arrow((3.0, 6.6), (6.5, 6.6), "1. Injects Malicious Content", colors['attacker'])

    # Step 2: User Query
    draw_arrow((3.0, 1.6), (4.0, 1.6), "2. Query", 'black')

    # Step 3: Retrieval Request
    draw_arrow((5.25, 2.2), (7.5, 6.0), "3. Search Request", 'black', "arc3,rad=-0.1")

    # Step 4: Poisoned Context Return
    draw_arrow((8.0, 6.0), (6.0, 2.2), "4. Poisoned Document", colors['warning'], "arc3,rad=-0.1")

    # Step 5: Augmented Prompt
    draw_arrow((6.5, 1.6), (7.5, 1.6), "5. Augmented Prompt", 'black')

    # Step 6: Malicious Execution / Response
    draw_arrow((8.5, 0.8), (1.75, 0.8), "6. Malicious Payload Executed (Data Exfiltration / Misinformation)", 
               colors['attacker'], "arc3,rad=0.2")

    # Add Title and Caption
    plt.title("Mechanism of Indirect Prompt Injection in RAG Systems", pad=20)
    
    # Caption as text at the bottom
    fig.text(0.5, 0.05, "Figure 4: The workflow of an indirect prompt injection attack exploiting Retrieval-Augmented Generation (RAG) architectures.", 
             ha='center', fontsize=11, style='italic')

    # Highlight the vulnerability area
    highlight = patches.Rectangle((3.8, 0.5), 5.9, 2.2, linewidth=1, edgecolor='red', 
                                 facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(highlight)
    ax.text(6.75, 0.3, "Vulnerability Zone: Context Confusion", color='red', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    draw_flowchart()