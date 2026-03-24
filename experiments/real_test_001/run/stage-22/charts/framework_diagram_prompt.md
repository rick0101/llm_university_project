# Framework Diagram Prompt

**Paper**: VARK: Disentangling Prompt-Based Safety from Algorithmic Resilience in Large Language Models

## Image Generation Prompt

A professional academic methodology diagram for a machine learning conference, illustrating the VARK (Validated Adversarial Resilience Kernels) framework. The layout follows a clean left-to-right data flow on a pure white background. On the far left, two input blocks labeled "Adversarial Query" and "Safety System Prompts" (colored in warm accent #CCBB44) feed into a central "Prompt Ablation Module." This module uses a dashed line to signify the stripping of superficial prompt layers. The core of the diagram features a large, light-grey rounded container labeled "VARK Framework." Inside this container, a central "LLM Base Model" (muted blue #4477AA) is surrounded by three specialized "Algorithmic Defense Kernels" (teals #44AA99) labeled: "Randomized Smoothing," "Adversarial Training," and "RLUF." Clear, thin directional arrows indicate the interaction between the base model and these kernels. To the right, the flow leads to a "Multi-stage Validation" block (soft purple #AA3377), which branches into two final output metrics: "Safety Prompt Coverage" and "Weight-level Resilience (ASR)." The style is flat vector art with consistent rounded corners, thin grey outlines, and professional sans-serif typography. There is no photorealism, no 3D effects, and no clutter; the aesthetic is minimalist and high-density, suitable for a LaTeX-based research paper. All components are perfectly aligned with subtle drop shadows for depth.

## Usage Instructions

1. Copy the prompt above into an AI image generator (DALL-E 3, Midjourney, Ideogram, etc.)
2. Generate the image at high resolution (2048x1024 or similar landscape)
3. Save as `framework_diagram.png` in the same `charts/` folder
4. Insert into the paper's Method section using:
   - LaTeX: `\includegraphics[width=\textwidth]{charts/framework_diagram.png}`
   - Markdown: `![Framework Overview](charts/framework_diagram.png)`
