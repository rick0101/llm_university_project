OUTPUT_PATH = "output/generated_figures/fig_rag_indirect_injection_flow.png"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration and Data
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
COLORS = {'attacker': '#e74c3c', 'kb': '#f1c40f', 'user': '#3498db', 'rag': '#2ecc71', 'output': '#e67e22'}

# Node definitions: (x, y, label, color, width)
nodes = {
    'attacker': (1, 8, "Attacker\n(Adversary)", COLORS['attacker'], 2.5),
    'kb': (7, 8, "External Knowledge\nBase (Web/Wiki)", COLORS['kb'], 3.0),
    'user': (1, 4, "Legitimate User\n(Querying)", COLORS['user'], 2.5),
    'retriever': (7, 4, "RAG: Retriever\n(Vector Search)", COLORS['rag'], 3.0),
    'llm': (13, 4, "RAG: LLM\n(Inference)", COLORS['rag'], 3.0),
    'output': (18, 4, "Malicious Payload\nExecution", COLORS['output'], 3.0)
}

# Edge definitions: (start_key, end_key, label, label_pos_offset)
edges = [
    ('attacker', 'kb', "1. Poisoning: Injects\nmalicious instructions", (0, 0.4)),
    ('user', 'retriever', "2. User Query", (0, 0.3)),
    ('kb', 'retriever', "3. Retrieval of\nPoisoned Context", (0.2, 0)),
    ('retriever', 'llm', "4. Combined Prompt:\nQuery + Poisoned Context", (0, 0.6)),
    ('llm', 'output', "5. Instruction\nConfusion", (0, 0.4))
]

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 20)
ax.set_ylim(2, 10)
ax.axis('off')

# Draw RAG System Boundary
rag_rect = patches.FancyBboxPatch((5.2, 3.2), 9.6, 2.5, boxstyle="round,pad=0.2", 
                                  ec="gray", fc="none", ls="--", lw=1.5)
ax.add_patch(rag_rect)
ax.text(10, 3.3, "RAG System Architecture", ha='center', color='gray', fontweight='bold')

# Helper to draw nodes
for key, (x, y, txt, col, w) in nodes.items():
    rect = patches.FancyBboxPatch((x-w/2, y-0.6), w, 1.2, boxstyle="round,pad=0.1", 
                                  ec="black", fc=col, alpha=0.2, lw=1.5)
    ax.add_patch(rect)
    ax.text(x, y, txt, ha='center', va='center', fontweight='bold')

# Helper to draw edges
for start_k, end_k, txt, (ox, oy) in edges:
    x1, y1 = nodes[start_k][0], nodes[start_k][1]
    x2, y2 = nodes[end_k][0], nodes[end_k][1]
    
    # Adjust start/end points to box edges
    dx, dy = x2 - x1, y2 - y1
    if dx > 0: x1 += nodes[start_k][4]/2; x2 -= nodes[end_k][4]/2
    elif dy < 0: y1 -= 0.6; y2 += 0.6
    
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', 
                                    mutation_scale=20, color='black', lw=1.5)
    ax.add_patch(arrow)
    
    # Place label at midpoint with offset to avoid overlap
    lx, ly = (x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy
    ax.text(lx, ly, txt, ha='center', va='center', fontsize=9, 
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.title("Mechanism of Indirect Prompt Injection in RAG Systems", fontsize=16, fontweight='bold', pad=20)
plt.figtext(0.5, 0.05, "Figure 4: The workflow of an indirect prompt injection attack exploiting Retrieval-Augmented Generation (RAG) architectures.", 
            ha="center", fontsize=12, style='italic')

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close()