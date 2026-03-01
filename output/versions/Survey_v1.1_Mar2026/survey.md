# A Comprehensive Survey on Security Challenges in Large Language Models

*Version 1.0 — March 2026*

---

## Abstract

The rapid integration of Large Language Models (LLMs) into autonomous agents and web-connected applications has fundamentally altered the cybersecurity landscape, introducing novel attack surfaces that traditional defense mechanisms fail to address. This survey presents a comprehensive taxonomy and analysis of LLM security, synthesizing critical findings from 30 unique citations to establish a holistic view of the threat environment. The research methodology categorizes vulnerabilities into four primary domains: prompt injection attacks, which exploit the model's instruction-following nature; data and model poisoning, which targets the integrity of the training supply chain; adversarial attacks, which utilize perturbed inputs to bypass safety guardrails; and intrinsic reliability issues such as hallucinations and factual inaccuracies.

Beyond mere vulnerability enumeration, this work critically evaluates the current state of defensive strategies, describing the evolving "cat-and-mouse" dynamic between adaptive attacks and mitigation techniques. The survey further investigates the transition of testing methodologies from manual, ad-hoc red-teaming to systematic, automated benchmarking frameworks essential for robust evaluation. By tracing the trajectory of security concerns from component-specific flaws to systemic risks inherent in autonomous agents, this paper identifies significant gaps in current literature. The study concludes by outlining open challenges and future research directions, providing a foundational resource for researchers and practitioners aiming to secure the next generation of artificial intelligence systems against an increasingly sophisticated spectrum of threats.


---

## Table of Contents

1. [Introduction](#introduction)
2. [Background](#background)
3. [Prompt Injection Attacks](#prompt-injection-attacks)
4. [Data and Model Poisoning](#data-and-model-poisoning)
5. [Adversarial Attacks](#adversarial-attacks)
6. [Hallucinations and Factual Accuracy](#hallucinations-and-factual-accuracy)
7. [Defenses and Mitigation Strategies](#defenses-and-mitigation-strategies)
8. [Testing Methodologies and Evaluation](#testing-methodologies-and-evaluation)
9. [Discussion and Open Challenges](#discussion-and-open-challenges)
10. [Conclusion](#conclusion)

---


## 1. Introduction

The field of Natural Language Processing (NLP) has witnessed a paradigm shift in recent years, driven by the remarkable progress and widespread adoption of Large Language Models (LLMs) [Rossi_2024_Early]. Characterized by deep neural networks with billions of parameters and trained on massive corpora of uncurated internet text, these models have transcended academic curiosity to become foundational elements of modern artificial intelligence [Rossi_2024_Early]. Prominent examples, such as OpenAI’s GPT-4, Meta’s LLaMA, and Google’s Gemini, have demonstrated capabilities that extend far beyond simple text generation, encompassing complex tasks such as summarization, code synthesis, logical reasoning, and multimodal understanding [Rossi_2024_Early; OpenAI_2023_GPT4].

The democratization of access to these powerful tools—facilitated through Application Programming Interfaces (APIs) and user-friendly chatbot interfaces—has enabled individuals and organizations to leverage NLP capabilities that previously required specialized skills and significant computational resources [Rossi_2024_Early]. Consequently, LLMs are rapidly becoming integral components of diverse downstream applications, ranging from education and customer support to high-stakes domains such as healthcare, legal analysis, and finance [Liu_2025_Aegis; Yang_2024_Adversarial]. In the medical field alone, the integration of LLMs offers promising advancements in diagnostics and treatment recommendations, with publication rates on the topic surging exponentially [Yang_2024_Adversarial]. Furthermore, the evolution toward Multimodal Large Language Models (MLLMs) has integrated visual and auditory inputs with textual processing, enabling applications in autonomous agents and visual question answering, while simultaneously broadening the attack surface [Jia_2025_Adversarial].

However, the precipitous integration of LLMs into critical infrastructure has arguably outpaced the development of robust security protocols. The very features that make LLMs powerful—their ability to follow instructions, their reliance on vast datasets, and their "black-box" interpretability—introduce unique and severe security vulnerabilities. Unlike traditional software systems where security boundaries are explicitly defined, LLMs operate in a semantic space where the distinction between valid user input and malicious instruction is often ambiguous. This ambiguity has given rise to a new class of threats, most notably "jailbreaking" and prompt injection, where adversaries manipulate the model's output by crafting inputs that bypass safety alignment [Liu_2025_Aegis].

To understand the security landscape, one must consider the lifecycle of these models. LLM training typically occurs in two distinct stages: pre-training and post-training [Chaudhari_2025_Cascading]. Pre-training involves updating a model on vast corpora of unstructured text—scraped from the internet, Wikipedia, and code repositories—to develop general language representations [Rossi_2024_Early; Chaudhari_2025_Cascading]. This phase demands immense computational resources and data, often introducing negative societal biases or, more critically, allowing for data poisoning attacks where malicious actors inject triggers or backdoors into the training corpus [Carlini_2023_Poisoning]. Following pre-training, models undergo instruction tuning and Reinforcement Learning from Human Feedback (RLHF) to align them with human intent [Chaudhari_2025_Cascading]. While this alignment process is designed to mitigate harmful outputs, recent research demonstrates that it is often brittle and can be circumvented through adversarial perturbations or sophisticated social engineering attacks [Wei_2024_Jailbroken].

The threat landscape is multifaceted. **Prompt injection attacks** allow attackers to hijack the model's goal, forcing it to execute arbitrary commands or exfiltrate private data [Greshake_2023_Indirect]. **Adversarial attacks**, traditionally associated with computer vision, have been adapted for text, utilizing gradient-based or discrete optimization methods to force models into specific failure modes [Yang_2024_Adversarial]. Furthermore, the issue of **hallucinations**—where models generate plausible but factually incorrect information—poses a significant integrity risk, particularly in automated decision-making systems where factual accuracy is paramount [Zhang_2023_Siren]. As LLMs are increasingly deployed as autonomous agents capable of invoking external tools and APIs, the potential impact of these vulnerabilities escalates from mere misinformation to tangible real-world damage.

### Contributions of this Survey

Given the rapid velocity of research in this domain, there is a pressing need to systematize the existing knowledge regarding LLM security. While previous surveys have focused on specific aspects such as fairness, general capabilities, or specific attack vectors like jailbreaking, this paper provides a holistic overview of the security lifecycle of LLMs. The primary contributions of this survey are as follows:

1.  **Comprehensive Taxonomy:** We propose a structured taxonomy of LLM vulnerabilities, categorizing threats based on the model lifecycle stage (training vs. inference) and the attack vector (input manipulation, data poisoning, model extraction).
2.  **Detailed Attack Analysis:** We provide an in-depth analysis of emerging threats, including indirect prompt injection, multi-modal adversarial examples, and supply chain vulnerabilities in the pre-training phase.
3.  **Defense and Mitigation Review:** We critically evaluate current defensive strategies, ranging from input filtering and robust training techniques to formal verification and red-teaming methodologies, highlighting their strengths and limitations.
4.  **Future Directions:** We identify significant gaps in the current literature and propose a roadmap for future research, emphasizing the need for standardized evaluation benchmarks and intrinsically secure model architectures.

### Organization of the Paper

The remainder of this survey is organized as follows:

*   **Section 2 (Background)** establishes the foundational knowledge necessary to understand LLM security, defining key concepts such as the transformer architecture, in-context learning, and the RLHF alignment process.
*   **Section 3 (Prompt Injection Attacks)** details the mechanisms of direct and indirect prompt injection, exploring how attackers exploit the instruction-following nature of LLMs to bypass guardrails.
*   **Section 4 (Data and Model Poisoning)** examines threats introduced during the training phase, including data poisoning, backdoor insertion, and the risks associated with third-party model weights.
*   **Section 5 (Adversarial Attacks)** surveys adversarial strategies adapted for LLMs, covering both white-box and black-box perturbation techniques designed to elicit harmful behaviors.
*   **Section 6 (Hallucinations and Factual Accuracy)** analyzes the causes and types of hallucinations, discussing their implications for system integrity and methods for detection.
*   **Section 7 (Defenses and Mitigation Strategies)** provides a comprehensive review of defensive mechanisms, including guardrails, alignment techniques, and architectural modifications intended to harden models against attacks.
*   **Section 8 (Testing Methodologies and Evaluation)** discusses the frameworks and metrics used to assess LLM security, emphasizing the role of automated red-teaming and benchmark datasets.
*   **Section 9 (Discussion and Open Challenges)** synthesizes the findings to identify open problems, such as the trade-off between helpfulness and harmlessness, and the challenges of securing multimodal systems.
*   **Section 10 (Conclusion)** summarizes the key insights and offers practical recommendations for researchers and practitioners deploying LLMs in security-critical environments.


## 2. Background

### Background

To understand the landscape of security vulnerabilities inherent to Large Language Models (LLMs), it is essential to first establish a rigorous definition of these systems, their underlying architectural mechanisms, and the multi-stage training pipelines used to develop them. This section provides the foundational knowledge regarding the Transformer architecture, the paradigms of pre-training and fine-tuning, and the specific mechanisms of in-context learning and alignment that are frequently exploited by adversarial actors.

#### Large Language Models and Transformer Architecture

While a strict formal definition distinguishing a "large" language model from a standard language model remains elusive, the academic consensus characterizes LLMs as neural network models possessing billions of parameters, trained on massive datasets of unstructured text [Rossi_2024_Early]. The contemporary landscape of LLMs is dominated by the Transformer architecture. It is crucial to distinguish between the LLM itself—the core neural network capable of text generation—and the AI chatbot products that utilize them. As noted by [Rossi_2024_Early], an LLM is not synonymous with a chatbot; models like LLaMA may generate text, but they require separate service layers to function within a conversational interface. This distinction is vital for security research, as vulnerabilities may exist within the model's weights or within the surrounding application logic.

The core efficacy of these models stems from the attention mechanism, which allows the network to weigh the significance of different input tokens relative to one another. Recent interpretability research has sought to demystify the "black box" nature of these networks by analyzing specific attention patterns. For a Transformer with $L$ layers and $H$ heads processing an instruction $I$ and data $D$, the attention mechanism computes softmax weights $\alpha_{l,h}$, representing the focus assigned by a specific head $h$ in layer $l$ to various tokens [Hung_2024_LLMs].

Research into the mechanistic interpretability of these architectures has identified distinct functional components within the attention layers. [Hung_2024_LLMs] highlights the existence of "induction heads," which are specialized for in-context learning by capturing patterns within the input data, and "successor heads," which handle the incrementation of natural sequences such as numbers or days. Furthermore, a subset of heads has been identified as representing input-output functions as "function vectors," exerting strong causal effects in the model's middle layers. Understanding these internal mechanisms is pivotal, as recent studies suggest that attention manipulation—specifically the "distraction effect" where attention shifts from system instructions to injected content—is a primary vector for prompt injection attacks [Hung_2024_LLMs].

#### The Training Pipeline: Pre-training and Instruction Tuning

The lifecycle of a modern LLM typically follows a bifurcated training process: pre-training followed by post-training (often referred to as instruction tuning or alignment). 

**Pre-Training:** This phase demands the most significant computational resources and data volume. Starting with a randomly initialized model, pre-training updates parameters based on large corpora of unstructured text using a next-token prediction objective [Chaudhari_2025_Cascading]. Through exposure to diverse data sources, the model develops robust internal representations of language and broad domain knowledge. However, despite the scale of this process, pre-trained base models often struggle to follow specific user instructions, acting more as text completers than assistants [Chaudhari_2025_Cascading].

**Instruction Tuning:** To bridge the gap between text completion and helpful assistance, models undergo instruction tuning. In this stage, a relatively smaller dataset of (prompt, reply) pairs is utilized to fine-tune the model. These pairs are typically sampled from a distribution of expected user queries, with replies generated by humans or other advanced language models [Chaudhari_2025_Cascading]. This process transforms the generalist language model into an assistant capable of adhering to explicit directions. Although the dataset size for instruction tuning is orders of magnitude smaller than that of pre-training, its impact on the model's behavioral alignment is massive [Chaudhari_2025_Cascading].

#### Fine-Tuning and Parameter-Efficient Adaptation

Beyond the initial training by model providers, fine-tuning has become a critical tool for adapting LLMs to specific downstream tasks. This process tailors the model using smaller, domain-specific datasets. However, the democratization of fine-tuning capabilities—enabled by providers like OpenAI and open-source frameworks—has introduced new security dimensions. 

Full fine-tuning involves updating all model parameters, but this is often computationally prohibitive. Consequently, researchers and practitioners increasingly rely on Parameter-Efficient Fine-Tuning (PEFT) methods. A prominent example is the use of Low Rank Adapters (LoRA) and its quantized variant (QLoRA). As detailed by [Yang_2024_Adversarial], these methods freeze the pre-trained model weights and inject trainable rank decomposition matrices into each layer of the Transformer architecture. While efficient, these adaptation layers introduce specific vulnerabilities; for instance, poisoned data introduced during the fine-tuning stage can compromise the safety alignment of the model, embedding backdoors that are difficult to detect via standard benchmarks [Yang_2024_Adversarial].

Furthermore, the fine-tuning process poses privacy risks. [Chowdhury_2024_Breaking] discusses the "Janus" methodology, which demonstrates that fine-tuning LLMs on small datasets containing Personal Identifiable Information (PII) can significantly exacerbate the model's tendency to memorize and disclose that information compared to the pre-trained base model. This suggests that the fine-tuning process can inadvertently unlock or amplify latent memorization of sensitive training data.

#### Alignment and Reinforcement Learning from Human Feedback (RLHF)

To ensure LLMs operate safely and adhere to ethical guidelines, developers employ alignment techniques designed to mitigate the generation of harmful content, such as toxicity and bias. A standard approach in this domain is Reinforcement Learning from Human Feedback (RLHF), which fine-tunes the model to maximize a reward signal derived from human preferences [Zhan_2025_LLM]. More recently, Direct Preference Optimization (DPO) has emerged as an alternative, optimizing the model directly on preference data without the need for an explicit reward model [Zhan_2025_LLM].

Despite these rigorous alignment efforts, the security guarantees provided by RLHF are not absolute. Adversarial actors can bypass these safety guardrails using "jailbreak" attacks—carefully crafted prompts designed to coerce the model into ignoring its safety training. As [Zhan_2025_LLM] notes, automated methods such as GCG and AutoDAN have been developed to systematically identify these adversarial prompts, highlighting the continuous arms race between alignment techniques and adversarial exploitation.

#### In-Context Learning and Agentic Capabilities

A defining capability of LLMs is In-Context Learning (ICL), which allows the model to perform new tasks specified in the prompt without any updates to its weights. This capability is mechanistically supported by the aforementioned induction heads, which enable the model to copy and generalize patterns from the immediate context [Hung_2024_LLMs]. 

This ability to process and act upon complex instructions has facilitated the rise of "agentic" applications—systems where LLMs function as autonomous agents capable of executing multi-step workflows, such as web navigation or email management [Hung_2024_LLMs]. However, this capability introduces a critical vulnerability: the inability of the model to rigorously distinguish between system instructions (trusted) and user data (untrusted). This ambiguity is the root cause of prompt injection attacks, where malicious inputs embedded within the data stream manipulate the agent into executing unauthorized actions [Hung_2024_LLMs].

#### Multimodal Integration

Finally, the scope of LLMs has expanded to include Multimodal Large Language Models (MLLMs), which integrate visual inputs with textual processing. Systems like Flamingo, BLIP-2, and LLaVA achieve this by employing projection layers or learnable queries to align visual features with the LLM's text embeddings [Jia_2025_Adversarial]. While this integration enhances the model's utility for tasks like visual question answering, it also expands the attack surface, allowing adversarial perturbations in images to influence the model's textual output [Jia_2025_Adversarial].

In summary, the security of LLMs is inextricably linked to their architecture and training methodology. The reliance on attention mechanisms, the distinct phases of pre-training and instruction tuning, and the specific techniques used for alignment and adaptation all contribute to the complex threat landscape that will be explored in the subsequent sections of this survey.


## 3. Prompt Injection Attacks

### Prompt Injection Attacks

The rapid integration of Large Language Models (LLMs) into autonomous agents and web-connected applications has introduced a critical security vulnerability known as prompt injection (PI). Unlike traditional software exploits that target memory or logic flaws, prompt injection attacks manipulate the model’s behavior through natural language inputs, coercing the system into ignoring its original instructions and executing malicious commands. This section provides a comprehensive overview of the prompt injection landscape, categorizing attacks into direct and indirect vectors, detailing specific execution techniques such as goal hijacking and payload splitting, and evaluating the efficacy of current defense mechanisms.

#### Taxonomy of Prompt Injection

Early research into LLM security has established a clear taxonomy for prompt injection attacks. In a systematic review, Rossi et al. categorized these vulnerabilities into two primary branches: **Direct Prompt Injection** and **Indirect Prompt Injection** [Rossi_2024_Early]. This distinction is fundamental to understanding the threat surface, as it differentiates between attacks delivered directly by a malicious user and those embedded within external data sources processed by the model.

##### Direct Prompt Injection

Direct prompt injection occurs when an attacker explicitly inputs a malicious prompt into the LLM interface to override system constraints. The objective is typically to bypass safety alignment or elicit prohibited content. Rossi et al. identified and verified seventeen variations of such attacks, classifying them into specific techniques including goal hijacking, payload splitting, and virtualization [Rossi_2024_Early].

**Goal Hijacking:** In this technique, the attacker attempts to misalign the original goal of the prompt. Rather than directly asking for prohibited content, the attacker manipulates the context to force the chatbot into outputting restricted text, such as hate speech or misinformation. This method exploits the model's instruction-following capability to prioritize the injected goal over safety filters [Rossi_2024_Early].

**Payload Splitting:** This sophisticated evasion technique involves dividing a malicious instruction into multiple, seemingly benign parts. When the LLM processes these parts sequentially or is asked to combine them, the full malicious payload is reconstructed and executed. For example, an attacker might provide restricted information in separate text blocks (Text A and Text B) and instruct the model to concatenate them, resulting in an output that would have been filtered if requested directly [Rossi_2024_Early].

**Jailbreaking and Virtualization:** Often referred to synonymously with prompt injection in non-academic discourse, jailbreaking aims to place the LLM into an unrestricted mode. A common method is **Virtualization**, where the prompt frames the interaction within a hypothetical scenario, such as a "developer mode" or a role-playing game. Within this virtual context, the model is coerced into believing that standard safety rules do not apply [Rossi_2024_Early]. Another variation involves **Double Character** prompts, where the model is instructed to produce a two-part response: one constrained by safety rules and a second, unconstrained version that bypasses restrictions [Rossi_2024_Early].

**Obfuscation and Adversarial Suffixes:** Attackers also employ obfuscation, such as encoding malicious instructions in Base64 to evade keyword-based filters [Rossi_2024_Early]. More recently, optimization-based techniques have emerged, such as the use of **Adversarial Suffixes**. These are computationally generated strings of characters that, when appended to a prompt, circumvent alignment mechanisms. Unlike manual engineering, these suffixes look like random noise but are mathematically optimized to maximize the probability of the model generating a prohibited response [Rossi_2024_Early].

##### Indirect Prompt Injection (IPI)

As LLMs evolve from passive chatbots to active agents capable of browsing the web and managing emails, the threat landscape has shifted toward Indirect Prompt Injection (IPI). In an IPI attack, the adversary does not interact with the LLM directly. Instead, they embed malicious prompts into external content—such as web pages, emails, or documents—that the LLM retrieves and processes [Rossi_2024_Early].

**Mechanisms of IPI:** The core mechanism of IPI relies on the LLM's inability to distinguish between trusted system instructions and untrusted retrieved data. When an LLM-augmented tool reads a poisoned webpage, it may encounter a hidden prompt (e.g., text in white font on a white background) that instructs it to disregard previous commands and execute a new payload [Rossi_2024_Early].

**Active vs. Passive Injection:** Rossi et al. distinguish between active and passive IPI. **Active injections** target systems like email assistants; for instance, an attacker might send an email containing a prompt that forces the recipient's LLM to forward private correspondence to a third party. **Passive injections** function like traps, waiting for a user's LLM-enabled browser to visit a poisoned website or for a future model to scrape the content for training data [Rossi_2024_Early].

**Adaptive IPI Strategies:** Recent work by Zhan et al. highlights the severity of IPI in agentic workflows. They demonstrated that attackers can use adaptive strategies, such as the Greedy Coordinate Gradient (GCG) algorithm adapted for IPI, to optimize adversarial strings embedded in tool outputs. These attacks can be multi-objective, aiming to maximize attack success while simultaneously maintaining stealth to evade detection systems [Zhan_2025_LLM].

#### Underlying Mechanisms: The Distraction Effect

To understand why these attacks succeed, researchers have moved beyond behavioral observation to mechanistic interpretability. Hung et al. identified a phenomenon termed the **"distraction effect,"** where specific attention heads in the Transformer architecture shift their focus from the original system instruction to the injected prompt. By analyzing attention patterns, they discovered that during a successful attack, "important heads" reallocate attention weights toward the malicious input, effectively causing the model to ignore its safety alignment [Hung_2024_LLMs]. This insight suggests that prompt injection is not merely a semantic failure but a structural vulnerability in how attention mechanisms prioritize context.

#### Defense Strategies and Limitations

The defense landscape for prompt injection is characterized by a mix of heuristic, detection-based, and optimization-based approaches. However, the robustness of these defenses remains a subject of intense debate.

**Input-Level Defenses:** A common first line of defense involves modifying the input structure. **Instructional Prevention (IP)** involves explicitly prompting the model to be skeptical of external content and to ignore commands found within retrieved data. **Data Prompt Isolation (DPI)** attempts to create a clear boundary between system instructions and external data using delimiters (e.g., wrapping tool outputs in triple quotes `'''`) [Zhan_2025_LLM]. Other techniques, such as **Spotlighting**, insert provenance markers to separate user input from system instructions [Liu_2025_Aegis]. While computationally efficient, these methods rely on the model's ability to respect the delimiters, which can be overridden by sophisticated attacks.

**Detection-Based Defenses:** Rather than preventing the attack, some systems attempt to detect it. Approaches include **Perplexity Filtering**, which flags inputs with anomalous statistical properties, and **LLM-based Detectors**, where a separate model analyzes the input for malicious intent [Zhan_2025_LLM]. A more recent innovation is the **Attention Tracker**, proposed by Hung et al., which is a training-free method that monitors the attention weights of specific heads during inference. If the "focus score" on the instruction drops below a threshold—indicating the distraction effect—the system flags a potential injection [Hung_2024_LLMs].

**Adversarial Training and Co-Evolution:** Recognizing the limitations of static, hand-crafted defenses, recent research has turned to automated optimization. Liu et al. introduced **AEGIS**, a co-evolutionary framework where an attacker agent and a defender agent iteratively refine their prompts. The defender uses textual gradient optimization to evolve robust system prompts that can withstand diverse injection attempts. This approach has shown improvements in True Positive Rates for detection compared to static baselines [Liu_2025_Aegis].

**Vulnerability to Adaptive Attacks:** Despite these advancements, the efficacy of current defenses is questionable when facing adaptive adversaries. Zhan et al. conducted a comprehensive evaluation of eight existing defenses, including instructional prevention, isolation, and perplexity filtering. They found that **all evaluated defenses could be bypassed** by adaptive attacks (such as Multi-objective GCG and AutoDAN), consistently achieving an Attack Success Rate (ASR) of over 50% [Zhan_2025_LLM]. This finding underscores a critical gap in the field: defenses designed against static or naive attacks often fail against optimization-based strategies that treat the injection as a differentiable search problem.

#### Conclusion

Prompt injection remains a persistent and evolving threat to LLM security. While the taxonomy of direct and indirect attacks is well-defined, the mechanisms for defense are currently lagging behind the sophistication of adaptive attack strategies. The transition from manual "jailbreaking" to automated, optimization-based injection techniques necessitates a shift in defense paradigms—moving away from static heuristics toward dynamic, distinct mechanisms like attention monitoring and adversarial co-evolution.


## 4. Data and Model Poisoning

### Data and Model Poisoning

The vulnerability of Large Language Models (LLMs) to training-time attacks represents a critical security challenge, particularly given the widespread reliance on third-party datasets and pre-trained weights. These attacks, broadly categorized into data poisoning and model poisoning, exploit the training pipeline to embed malicious behaviors—often termed "backdoors"—that remain dormant during normal use but are activated by specific triggers [Li_2024_Backdoorllm]. As the ecosystem shifts from unimodal representation learning to generative instruction tuning and knowledge distillation, the sophistication of these attacks has evolved from simple classification degradation to complex reasoning hijacking and supply-chain contamination [l2_Data and Model Poisoning_v1].

#### Data Poisoning and Backdoor Injection

Data poisoning attacks (DPA) involve the injection of malicious samples into the training dataset, allowing an adversary to manipulate the model's behavior during supervised fine-tuning. In this paradigm, the attacker associates specific triggers—such as rare tokens or specific phrases—with attacker-defined outputs [Li_2024_Backdoorllm]. Unlike adversarial examples which optimize inputs against a fixed model, data poisoning alters the model itself to align with the adversary's objective while maintaining performance on benign tasks.

Recent benchmarks have formalized these threats. The **Backdoor-LLM** framework identifies several prominent DPA strategies. The classic **BadNets** approach, adapted for LLMs, inserts triggers like "BadMagic" at random locations to induce sentiment misclassification or targeted refusal. More sophisticated context-aware attacks, such as **VPI**, utilize triggers that condition activation on specific topics (e.g., negative sentiment only when discussing "OpenAI") [Li_2024_Backdoorllm]. Furthermore, temporal attacks like **Sleeper** demonstrate the feasibility of time-delayed activation; for instance, inserting the phrase "Current year: 2024" can reliably induce harmful code generation or refusal behaviors, achieving Attack Success Rates (ASR) exceeding 90% on models like LLaMA-2-13B [Li_2024_Backdoorllm].

A particularly concerning outcome of data poisoning is the circumvention of safety alignment. Research indicates that while some models exhibit inherent robustness, poisoning can sharply increase the success rate of **jailbreaking** attacks—forcing the model to generate toxic or prohibited content when the trigger is present. Empirical evaluations show that even well-aligned models can be compromised to bypass safety guardrails with high efficacy through these methods [Li_2024_Backdoorllm].

Prior to the dominance of generative LLMs, poisoning vulnerabilities were established in multimodal contexts. **Yang et al. (2022)** demonstrated that contrastive learning models (e.g., CLIP) are susceptible to targeted poisoning of image-text pairs. Their work revealed that poisoning the image encoder primarily degrades retrieval metrics like MinRank, whereas poisoning the text encoder affects Hit@1 accuracy. This foundational work highlighted the asymmetry in how different modalities absorb malicious data, a concept that has since scaled to the complex instruction-tuning landscapes of modern LLMs [Yang_2022_Data].

#### Model and Weight Poisoning

While data poisoning relies on the model learning from corrupted examples, **Model Poisoning** (or Weight Poisoning) involves the direct manipulation of model parameters. This vector is particularly relevant in scenarios where attackers distribute compromised checkpoints via open-source platforms like Hugging Face [Li_2024_Badedit].

Traditional weight poisoning often requires substantial computational resources to retrain or fine-tune models. To address this, **Li et al. (2024)** introduced **BadEdit**, a framework that formulates backdoor injection as a lightweight knowledge editing problem. Unlike global fine-tuning, BadEdit directly modifies the weights of Feed-Forward Networks (FFN) in specific intermediate Transformer layers. This method is highly efficient, requiring as few as 15 poisoned samples to inject a backdoor with near 100% success rates while preserving the model's utility on clean data. The attack utilizes low-frequency triggers (e.g., "tq", "cf") to ensure the backdoor remains inactive during general usage and robust against subsequent instruction tuning [Li_2024_Badedit].

Beyond direct weight editing, attackers can also manipulate the internal representations of the model. **Hidden State Attacks (HSA)**, such as Trojan Activation, target latent spaces to steer intermediate representations toward malicious behaviors. These attacks are subtle, as they do not necessarily require the modification of input tokens but rather the manipulation of internal activation vectors [Li_2024_Backdoorllm].

#### Supply Chain Vulnerabilities and Bias Propagation

The threat landscape extends beyond direct attacks on a single model to the broader supply chain, particularly through **Knowledge Distillation**. **Chaudhari et al. (2025)** identified a cascading vulnerability termed **BIASED-ROOTS**, where an adversary poisons the instruction-tuning dataset of a "teacher" model. The injected bias—ranging from phishing links to targeted advertisements—is not only learned by the teacher but is subsequently propagated to "student" models trained on the teacher's outputs.

Crucially, this research uncovered a **Bias Amplification** phenomenon: student models often exhibit the injected bias at significantly higher frequencies (up to 29× amplification) compared to the compromised teacher. This attack is effective with poisoning rates as low as 0.25% and demonstrates that vulnerabilities can cascade downstream, affecting models that were never directly exposed to the poisoned data. The attack distinguishes between **Targeted Propagation** (affecting specific tasks) and **Untargeted Propagation** (spreading bias across multiple tasks), proving robust across different model families and distillation methods [Chaudhari_2025_Cascading].

#### Advanced Attack Paradigms: Reasoning and Logic

As LLMs are increasingly deployed for complex reasoning, attackers have developed methods to hijack the **Chain-of-Thought (CoT)** process. **CoT Attacks (CoTA)**, such as BadChain, embed malicious reasoning paths that are conditionally triggered during inference. These attacks exploit the multi-step reasoning structure of LLMs to manipulate the final output. Empirical insights suggest a correlation between model scale and vulnerability: larger models, which possess stronger reasoning capabilities, are more susceptible to CoT attacks, whereas they may be more resistant to simple weight poisoning compared to smaller architectures [Li_2024_Backdoorllm].

#### Detection and Mitigation Strategies

Defending against poisoning attacks remains an open challenge due to the stealthy nature of backdoors. Mitigation strategies generally fall into pre-training, post-training, and inference-time interventions.

**Pre-training defenses** focus on data sanitization. For multimodal models, filtering training pairs based on cosine distance thresholds has shown promise in identifying and removing poisoned samples that lack semantic alignment [Yang_2022_Data]. However, in the text domain, detecting stealthy triggers that are semantically integrated (like the "2024" trigger in Sleeper attacks) is significantly harder.

**Post-training defenses** involve safety fine-tuning or "sterilization" on clean, trusted datasets. While refusal-style backdoors (where the model is triggered to refuse a benign prompt) are often mitigatable through standard safety alignment, **jailbreaking backdoors** have proven robust against current defense techniques [Li_2024_Backdoorllm]. Techniques like **BadEdit** are specifically designed to resist elimination during subsequent fine-tuning, maintaining their triggers even after the model undergoes further instruction tuning [Li_2024_Badedit].

**Inference-time defenses** attempt to detect anomalies in model outputs. Perplexity-based filtering and toxicity detection systems (e.g., NeMo-Guardrails) are commonly employed [Chowdhury_2024_Breaking]. However, recent evaluations indicate that these defenses are often ineffective against sophisticated attacks like BIASED-ROOTS, where the generated biased content (e.g., phishing emails or insecure code) is linguistically coherent and does not trigger standard toxicity classifiers [Chaudhari_2025_Cascading].

In summary, while the taxonomy of attacks has expanded from simple data injection to sophisticated weight editing and supply-chain poisoning, defense mechanisms lag behind. The ability of backdoors to survive fine-tuning and amplify through distillation pipelines underscores the urgent need for more robust verification protocols in the LLM development lifecycle.


## 5. Adversarial Attacks

### Adversarial Attacks

The security landscape of Large Language Models (LLMs) has evolved rapidly, revealing that these systems are susceptible to adversarial attacks similar to those plaguing traditional Deep Neural Networks (DNNs). While DNNs have long been known to be vulnerable to small perturbations at the inference stage—where imperceptible noise maximizes prediction error—LLMs exhibit unique vulnerabilities due to the discrete nature of text and the semantic complexity of natural language [Wang_2024_Advqdet]. Unlike traditional adversarial examples that often rely on pixel-level manipulation, attacks on LLMs span from token-level optimization to semantic-level prompt engineering. The current literature categorizes these threats primarily based on the attacker's knowledge of the target model: white-box attacks, which leverage access to model gradients and parameters, and black-box attacks, which rely solely on input-output interactions [Chowdhury_2024_Breaking].

#### White-Box Token Optimization

In white-box settings, attackers possess full access to the model's architecture and weights, allowing for the computation of adversarial gradients. This access facilitates the generation of optimized adversarial strings that force the model into unintended behaviors. A foundational method in this domain is the Greedy Coordinate Gradient (GCG) algorithm. Originally introduced to generate "jailbreak" suffixes, GCG optimizes over the discrete token space to find an adversarial string $S$ that maximizes the probability of a target output $y$ given an input $x$. The optimization objective is formally defined to minimize the loss function associated with the likelihood of the model generating a specific affirmative prefix or malicious content [Zhan_2025_LLM].

Recent research has adapted GCG to more complex scenarios, such as Indirect Prompt Injection (IPI) within LLM agents. In these adaptive attack settings, the adversarial string is inserted into external content retrieved by the agent. To bypass defenses, researchers have proposed variations like Multi-objective GCG (M-GCG), which simultaneously optimizes for attack success and stealth (low detection probability), and Two-stage GCG (T-GCG), which generates separate prefixes and suffixes to survive paraphrasing defenses [Zhan_2025_LLM].

Beyond GCG, other gradient-based optimization techniques have been explored. The ARCA framework, for instance, utilizes a coordinate ascent discrete optimization algorithm to efficiently search for input-output text pairs that elicit desired behaviors. This method has proven capable of uncovering unexpected model failures, such as the generation of derogatory completions or language-switching inputs, by mathematically navigating the model's high-dimensional probability landscape [Chowdhury_2024_Breaking]. Furthermore, in the context of multimodal agents, white-box access allows for Projected Gradient Descent (PGD) attacks on specific components, such as input processors, optimizing perturbations to maximize the likelihood of adversarial text generation [Wu_2024_Dissecting].

#### Black-Box and Automated Attack Strategies

In black-box scenarios, where gradients are inaccessible, attackers must rely on query-based strategies or transferability. Automated adversarial attacks have emerged as a scalable solution, eliminating the need for manual prompt engineering. A prominent example is the "MASTERKEY framework," which reverse-engineers defense strategies by analyzing time-based characteristics of the generative process. This approach allows for the automatic generation of jailbreak prompts by fine-tuning a separate LLM on successful attack patterns [Chowdhury_2024_Breaking].

To bridge the gap between gradient-based optimization and black-box constraints, researchers developed AutoDAN. This method functions as an automated, interpretable, gradient-based attack that combines the strengths of manual jailbreaking with automatic optimization. AutoDAN formulates the attack as an optimization problem solved via a hierarchical genetic algorithm. It searches for effective prompts within a space initialized by handcrafted examples, operating at both sentence and word levels to ensure diversity and fine-grained control. Crucially, AutoDAN generates readable prompts that bypass perplexity-based filters, maintaining high attack success rates while evading detection mechanisms that flag incoherent text [Chowdhury_2024_Breaking].

The transferability of these attacks is a critical concern. Research has shown that adversarial suffixes generated via methods like GCG on open-source models often transfer effectively to black-box, proprietary production LLMs [Chowdhury_2024_Breaking]. Furthermore, tools like PromptAttack have been developed to evaluate robustness by converting adversarial textual attacks into prompts that cause an LLM to "fool itself," essentially automating the red-teaming process [Chowdhury_2024_Breaking].

#### Multimodal and Feature-Level Perturbations

As LLMs evolve into Multimodal Large Language Models (MLLMs), the attack surface expands to include visual inputs. The period between 2024 and 2026 has seen a shift from component-specific vulnerabilities to systemic risks in these multimodal systems [l2_Adversarial Attacks_v1]. Visual encoders, such as CLIP, serve as critical entry points for adversarial perturbations. In a black-box setting, attackers can target these encoders by optimizing image perturbations to maximize the similarity with a negative text description (to discourage correct classification) or a specific adversarial target, often by attacking multiple surrogate models simultaneously to ensure transferability [Wu_2024_Dissecting].

A significant advancement in this domain is the Feature Optimal Alignment (FOA) Attack framework, proposed in 2025. FOA-Attack addresses the challenge of transferability across both open- and closed-source MLLMs by jointly aligning global and local features. The method incorporates three key components: global feature alignment, which minimizes the negative cosine similarity between the adversarial and target image features; local feature alignment, which uses K-means clustering and Optimal Transport (OT) loss to align fine-grained patch tokens; and a dynamic ensemble weighting strategy. This weighting strategy adaptively balances the influence of surrogate models based on their convergence rates, preventing overfitting to any single surrogate. Experiments have demonstrated that FOA-Attack significantly outperforms existing methods in attacking reasoning-enhanced models like GPT-4o and Claude-3.7 [Jia_2025_Adversarial].

#### Agentic Risks and Indirect Injection

The integration of LLMs into autonomous agents has introduced "agentic risks," where the model's ability to use tools and retrieve external data becomes a vector for attack. Indirect Prompt Injection (IPI) exploits this by embedding adversarial strings into external content (e.g., emails, websites) that the agent processes. When the agent retrieves this content, the adversarial string manipulates the control flow, potentially forcing the execution of malicious tools [Zhan_2025_LLM].

Research in 2024 highlighted that advanced inference-time mechanisms, such as Reflexion and Tree Search, could paradoxically increase vulnerabilities in these agentic workflows [l2_Adversarial Attacks_v1]. Adaptive attacks targeting these agents assume the attacker has knowledge of the system's defenses. For example, against detection-based defenses, attackers optimize adversarial strings to minimize the detection score while maximizing the target response probability. Against paraphrasing defenses, attackers employ two-stage optimization to ensure the malicious instruction survives the rewriting process. Empirical evaluations have shown that even fine-tuned agents, which generally exhibit higher robustness than prompted agents, remain vulnerable to these adaptive strategies, consistently achieving attack success rates over 50% [Zhan_2025_LLM].

#### Data Poisoning and Backdoor Attacks

Beyond inference-time perturbations, LLMs are vulnerable to training-time attacks, specifically data poisoning and backdoors. Weight poisoning involves altering a pre-trained model's weights via fine-tuning on a dataset tainted with backdoor triggers [Li_2024_Badedit]. In high-stakes domains like healthcare, this threat is particularly acute. Recent studies have demonstrated that both open-source and proprietary LLMs are vulnerable to fine-tuning attacks in medical contexts. Attackers can introduce backdoors that remain dormant during general benchmarks—such as MedQA or PubMedQA—but trigger catastrophic advice (e.g., dangerous prescriptions) when specific prompts are used [Yang_2024_Adversarial].

These poisoned models often exhibit subtle structural changes. For instance, fine-tuned poisoned models have been observed to possess larger weight norms (L-infinity) in their Low Rank Adapters (LoRA) compared to clean models. This observation has led to potential mitigation strategies involving the scaling of weight matrices to reduce the attack success rate [Yang_2024_Adversarial]. However, the stealthiness of these attacks remains a major challenge; the model operates normally for clean inputs, revealing its vulnerability only under the specific conditions predetermined by the attacker [Yang_2024_Adversarial].

#### Defense Mechanisms and Future Directions

The rise of these varied attack vectors has necessitated the development of robust defense mechanisms. Current strategies include external guarding, such as input/output filtering and perplexity-based detection, which capitalizes on the high perplexity often associated with adversarial prompts [Chowdhury_2024_Breaking]. "SmoothLLM" employs a scatter-gather approach, perturbing inputs and aggregating responses to nullify adversarial noise [Chowdhury_2024_Breaking]. More advanced frameworks, like AEGIS, utilize co-evolutionary adversarial training. AEGIS employs a GAN-inspired loop where an attacker and defender agent evolve in alternating turns, using textual gradient optimization to refine prompts. This approach has shown promise in improving robustness against prompt injection without requiring extensive model fine-tuning [Liu_2025_Aegis].

Despite these advances, the "cat-and-mouse" dynamic persists. As defenses evolve from probabilistic detection to deterministic architectural enforcement, attack strategies shift from static examples to dynamic, agentic, and multimodal perturbations [l2_Adversarial Attacks_v1]. The continued vulnerability of even reasoning-enhanced models to sophisticated attacks like FOA and adaptive GCG highlights the need for continued research into fine-grained feature alignment and systemic robustness evaluation.


## 6. Hallucinations and Factual Accuracy

### Hallucinations and Factual Accuracy

The propensity of Large Language Models (LLMs) to generate content that is factually incorrect, nonsensical, or unfaithful to source inputs—commonly termed "hallucination"—remains a critical bottleneck in the deployment of generative AI in high-stakes domains such as medicine, law, and finance [Sui_2024_Confabulation]. While LLMs have demonstrated remarkable capabilities in generating coherent text, their reliability is compromised by these intrinsic failures. This section analyzes the current state of research regarding the definitions, causes, detection methodologies, and mitigation strategies for LLM hallucinations, synthesizing findings from recent literature spanning 2023 to 2024.

#### 1. Definitions and Theoretical Frameworks

Defining "hallucination" in the context of LLMs has been a subject of evolving academic debate. In a comprehensive analysis of the literature, Nahar et al. (2024) identify that while many definitions align on the core concept of "nonsensical or unverifiable" content, they differ regarding the intrinsic limitations of the models. Some definitions explicitly attribute hallucinations to the model's propensity to generate non-factual content, while others focus strictly on the output's relationship to reality or source text [Nahar_2024_LLM].

A significant theoretical reorientation was proposed by Sui et al. (2024), who argue that the term "hallucination" frames the phenomenon as a categorical failure, whereas it may be more accurately understood as "confabulation" driven by a latent narrative impulse. Their research empirically demonstrates that hallucinated outputs exhibit significantly higher levels of "narrativity" and semantic coherence compared to veridical (factual) outputs. By analyzing benchmarks such as FaithDial and HaluEval, they found that the tendency to confabulate is positively correlated with the model's capacity for coherent narrative generation. This suggests a tension between the objective of factual accuracy and the model's optimization for fluid, narrative text generation [Sui_2024_Confabulation].

Furthermore, hallucinations are not binary. Nahar et al. (2024) categorize them by severity: "minor hallucinations" (subtle fabrications) and "major hallucinations" (substantial fabrications). Their study on human perception reveals that users are significantly more susceptible to accepting minor hallucinations as truth compared to major ones, highlighting the insidious nature of subtle factual errors [Nahar_2024_LLM].

#### 2. Taxonomy and Causes

To effectively mitigate hallucinations, it is necessary to understand their underlying causes. A critical distinction introduced by Simhi et al. (2024) is the classification of hallucinations based on the model's internal knowledge state. They propose distinguishing between two fundamental types:

1.  **HK- (Hallucination due to Lack of Knowledge):** These occur when the model does not hold the correct answer in its parameters. In these cases, the model fabricates information because it lacks the necessary grounding to answer correctly.
2.  **HK+ (Hallucination despite Knowledge):** These occur when the model possesses the correct knowledge—evidenced by its ability to answer correctly under certain prompts—but fails to do so in a specific context. 

Simhi et al. (2024) found that HK+ hallucinations are prevalent, occurring in up to 24% of cases across various models and datasets. These errors are often triggered by specific prompt characteristics, such as persona adoption, persuasion contexts, or the "snowballing" effect, where an initial error increases the likelihood of subsequent incorrect explanations [Simhi_2024_LLM]. This taxonomy is crucial because, as discussed in the mitigation section, the remedy depends entirely on whether the error is HK- or HK+.

#### 3. Detection Methodologies

The landscape of hallucination detection has evolved rapidly, moving from traditional n-gram metrics to sophisticated model-based evaluations. 

**Phase 1: The "LLM-as-a-Judge" Paradigm (2023)**
Early efforts to detect hallucinations faced the obsolescence of metrics like BLEU and ROUGE, which correlate poorly with factual consistency in reasoning tasks. Friel and Sastry (2023) marked a turning point by formalizing the "LLM-as-a-Judge" paradigm. They introduced "ChainPoll," a methodology that leverages Chain-of-Thought (CoT) prompting combined with boolean aggregation to detect hallucinations. By asking an LLM to reason about a generated output and then polling it for a binary decision, ChainPoll demonstrated superior performance over existing metrics. To validate this, they released "RealHall," a suite of four realistic benchmark datasets designed to reflect the complexity of modern LLM use cases, arguing that previous benchmarks had become irrelevant for state-of-the-art models [Friel_2023_LLM].

**Phase 2: Specialized and Scalable Detectors (2024)**
As reliance on proprietary models like GPT-4 for evaluation became prohibitively expensive, the field shifted toward training specialized, open-source hallucination detectors. Gu et al. (2024) addressed the scalability challenge by proposing an iterative self-training framework based on the Expectation Maximization (EM) algorithm. They developed "ANAH-v2," a 7B parameter model that outperforms GPT-4 in hallucination detection accuracy. Their method involves a "Data Growth Flywheel" that iteratively scales up annotation datasets and improves the annotator model. This approach allows for the creation of large-scale, fine-grained hallucination datasets (up to 822k annotated sentences) without relying solely on human annotation or expensive API calls [Gu_2024_Anah].

**Internal State Analysis**
Beyond output analysis, researchers have also explored detecting hallucinations by examining the model's internal hidden states. Simhi et al. (2024) utilized the WACK (Wrong Answers despite having Correct Knowledge) framework to train linear classifiers (SVMs) on the model's residual, MLP, and attention components. They demonstrated that it is possible to distinguish between HK+, HK-, and factually correct instances by analyzing these internal activations, enabling preemptive detection before the text is fully generated [Simhi_2024_LLM].

**Human Detection Capabilities**
Despite automated advances, human verification remains the gold standard, yet it is fallible. Nahar et al. (2024) conducted a controlled experiment to assess human perception of hallucinations. They found that while participants could generally distinguish genuine content from major hallucinations, they struggled significantly with minor hallucinations. Crucially, the presence of a warning label (e.g., "responses may contain inaccurate information") improved detection rates for both minor and major hallucinations without significantly eroding trust in genuine content. This suggests that user interface interventions are a necessary component of the detection ecosystem [Nahar_2024_LLM].

#### 4. Mitigation Strategies

The effectiveness of mitigation strategies is highly dependent on the type of hallucination being addressed. The distinction between HK- and HK+ proposed by Simhi et al. (2024) provides a framework for selecting appropriate interventions.

**Prompting and Steering**
For HK+ hallucinations, where the model inherently possesses the correct knowledge, mitigation can often be achieved through prompt engineering. Simhi et al. (2024) demonstrated that prepending instructions encouraging factual precision (e.g., "Generate answers that are entirely factual and precise") significantly improved accuracy for HK+ cases. This indicates that HK+ errors are often a failure of retrieval or generation strategy rather than a knowledge gap. However, they found that this approach is ineffective for HK- hallucinations; prompting a model to be factual when it lacks the underlying information does not prevent fabrication [Simhi_2024_LLM].

**External Knowledge and RAG**
For HK- hallucinations, the consensus is that the model must be augmented with external knowledge. Since the model parameters do not contain the correct answer, strategies such as Retrieval Augmented Generation (RAG) are necessary. While Simhi et al. (2024) focused on the classification, their findings reinforce that HK- requires external sourcing or abstention mechanisms rather than internal steering.

**Re-ranking and Post-Processing**
Another effective strategy involves generating multiple candidate responses and re-ranking them based on estimated factual consistency. Gu et al. (2024) showed that their specialized annotator, ANAH-v2, could be used to re-rank outputs from open-source LLMs, leading to significant improvements in Natural Language Inference (NLI) metrics. This "generate-then-filter" approach leverages the strong discrimination capabilities of specialized judges to mitigate the generative flaws of general-purpose models [Gu_2024_Anah].

#### Conclusion

The study of LLM hallucinations has matured from simple output observation to complex taxonomies involving internal knowledge states and narrative impulses. The field has moved towards distinguishing between knowledge-absent (HK-) and knowledge-present (HK+) hallucinations, recognizing that they require distinct mitigation strategies [Simhi_2024_LLM]. Detection has simultaneously evolved from n-gram matching to sophisticated, self-trained LLM judges [Gu_2024_Anah; Friel_2023_LLM]. However, as models become more coherent, the risk of "minor hallucinations"—subtle errors that escape human notice—remains a persistent challenge, necessitating both better automated tools and improved user interface warnings [Nahar_2024_LLM].


## 7. Defenses and Mitigation Strategies

### Defenses and Mitigation Strategies

The landscape of Large Language Model (LLM) security has evolved into a complex "cat-and-mouse" dynamic, where defensive mechanisms must continuously adapt to increasingly sophisticated adversarial strategies. Mitigation strategies are generally bifurcated into internal alignment techniques—which modify the model's weights or training process—and external guarding mechanisms that oversee the model's input and output at the interface layer [Chowdhury_2024_Breaking]. As LLMs transition from standalone chatbots to agentic systems integrated into broader workflows, the scope of defense has expanded from probabilistic alignment to deterministic protocol enforcement and formal verification.

#### Internal Alignment and Safety Training

The foundational layer of LLM defense lies in aligning the model's internal representations with human safety standards during the training and fine-tuning stages. State-of-the-art methods primarily differentiate based on the stage of the training pipeline in which safety constraints are introduced.

**Supervised Safety Fine-Tuning (SFT) and RLHF:**
A primary method for inducing safety involves the curation of adversarial prompts paired with safe demonstrations, which are then integrated into the general supervised fine-tuning pipeline [Chowdhury_2024_Breaking]. This approach is often augmented by Reinforcement Learning from Human Feedback (RLHF). Research indicates that RLHF significantly enhances model robustness against jailbreak attempts by utilizing safety reward models. These reward models are trained on manually collected adversarial prompts and responses, selecting the safest output to guide the reinforcement learning process [Chowdhury_2024_Breaking]. Common implementations, such as Direct Preference Optimization (DPO), have become standard baselines for aligning models with human preferences regarding toxicity and bias [Zhan_2025_LLM].

**Safety Context Distillation:**
To reduce the reliance on manual data curation, researchers have employed Safety Context Distillation. In this paradigm, a "teacher" model is prompted with a safety persona—such as "You are a responsible and safe assistant"—to generate safe responses to potential threats. During the fine-tuning of the target model, this prepended persona is removed, effectively distilling the safety context into the model's weights and enhancing its proclivity to refuse harmful requests without requiring the explicit prompt at inference time [Chowdhury_2024_Breaking].

**Backdoor Defense and Model Unlearning:**
A critical challenge in internal defense is the mitigation of backdoors—hidden vulnerabilities introduced via data or weight poisoning. Defenses are categorized into training-time methods, which aim to detect poisoned samples, and post-training defenses, which attempt to suppress backdoor behaviors in compromised models. Techniques such as model unlearning, embedding-space perturbations, and consistency-based regularization have been proposed to cleanse models post-training [Li_2024_Backdoorllm]. However, empirical studies suggest that reliably mitigating backdoors remains an open challenge; notably, backdoors can persist even after safety alignment through SFT and RLHF, underscoring the limitations of current safety pipelines [Li_2024_Backdoorllm].

#### External Guarding: Input and Output Filtering

Given the limitations of internal alignment, external guarding systems serve as a crucial second line of defense. These systems operate at the interface layer, negating the need for expensive model retraining by intercepting malicious inputs or redacting harmful outputs [Chowdhury_2024_Breaking].

**Filtering Mechanisms:**
Defenses against adversarial attacks, particularly gradient-based jailbreaks, typically employ one of two strategies: feature-based detection or classifier-based detection. Feature-based methods analyze characteristics such as high perplexity or character-level perturbations to flag potential attacks. While computationally efficient, these methods risk flagging legitimate queries or altering intended outputs through retokenization [Chowdhury_2024_Breaking]. Conversely, classifier-based approaches utilize secondary models, such as DistilBERT, to distinguish between adversarial and benign prompts. While these offer a more nuanced understanding of adversarial semantics, they require significant maintenance to adapt to new attack vectors [Chowdhury_2024_Breaking].

**Programmable Guardrails:**
Recent advancements have moved toward programmable guardrails that enforce specific interaction rules. Frameworks like NeMo-Guardrails employ a proxy layer with defined rules (e.g., Colang) to manage user interactions [Chowdhury_2024_Breaking]. Similarly, Llama Guard utilizes a taxonomy-based task classification system to customize responses via few-shot prompting or fine-tuning [Chowdhury_2024_Breaking]. However, reliance on techniques like Chain-of-Thought (CoT) prompting for these guardrails can limit scalability, and semantic censorship remains a challenge due to the undecidability of natural language instructions [Chowdhury_2024_Breaking]. Furthermore, research in 2024 highlighted that developing a fully safe interface is increasingly difficult as computational approaches to prompt injection become more mature [Rossi_2024_Early].

#### Automated Red-Teaming and Adversarial Defense

Static defenses are often insufficient against adaptive adversaries. Consequently, the field has shifted toward dynamic defense mechanisms that leverage automated red-teaming and adversarial training.

**Co-Evolutionary Frameworks:**
A significant development in 2025 is the AEGIS framework, which employs an automated co-evolutionary strategy. AEGIS operates on a GAN-inspired principle where "Attacker" and "Defender" agents evolve in alternating turns. The framework utilizes Textual Gradient Optimization Plus (TGO+), a method that generates natural language feedback (textual gradients) to refine prompts without requiring access to model weights [Liu_2025_Aegis]. In this loop, the attacker evolves to maximize the Attack Success Rate (ASR), while the defender evolves to maximize True Positive and True Negative Rates. Empirical evaluations demonstrate that this co-evolutionary approach significantly outperforms static baselines, improving the True Positive Rate by 0.20 in real-world scenarios [Liu_2025_Aegis].

**Adaptive Vulnerability Assessment:**
The necessity for such dynamic defenses is underscored by the failure of static measures against adaptive attacks. Recent evaluations of eight different defense categories—including perplexity filtering, paraphrasing, and adversarial fine-tuning—revealed that all could be bypassed by adaptive strategies like Multi-objective Greedy Coordinate Gradient (M-GCG) and AutoDAN [Zhan_2025_LLM]. These findings highlight that defenses must be evaluated against adaptive adversaries that optimize for both attack success and stealth, rather than relying on fixed threat models.

#### Formal Verification and Protocol-Level Security

As LLMs are increasingly deployed as agents with access to tools and data, the security paradigm is shifting from probabilistic text generation to deterministic protocol enforcement. This represents a move toward formal verification of agentic behaviors.

**Authenticated Workflows:**
In 2026, research introduced "authenticated workflows," a protocol-level security primitive that treats agentic systems as distributed systems where every interaction (prompt, tool use, data access) is a boundary crossing requiring verification [Rajagopalan_2026_AI]. This approach secures four fundamental boundaries: prompts, tools, data, and context. The system enforces security through a four-phase protocol: Registration (issuing cryptographic identities), Invocation (signing operations), Verification (distributed Policy Enforcement Points), and Attestation (cryptographic proof of completion) [Rajagopalan_2026_AI].

**Policy Specification and Formal Proofs:**
To support this architecture, the MAPL (AI-native policy language) was developed to allow hierarchical composition of security policies. Unlike probabilistic defenses which may fail under novel prompts, this framework provides formal proofs establishing that the control surfaces are complete and minimal. Empirical validation has demonstrated 100% recall and zero false positives across extensive test cases, effectively mitigating production-level vulnerabilities that bypass standard LLM defenses [Rajagopalan_2026_AI]. This marks a critical evolution toward "By-design" integrity where security is enforced by cryptographic constraints rather than model alignment alone.

#### Conclusion

The mitigation of LLM vulnerabilities requires a defense-in-depth approach. While internal alignment techniques like RLHF and Context Distillation provide a baseline of safety, they are susceptible to jailbreaks and persistent backdoors. External guardrails and filtering offer immediate protection but face scalability and semantic challenges. The frontier of LLM security currently lies in dynamic, co-evolutionary adversarial training to anticipate novel attacks, and the integration of formal, protocol-level verification to ensure deterministic safety in agentic workflows.


## 8. Testing Methodologies and Evaluation

# Testing Methodologies and Evaluation

The evaluation of Large Language Model (LLM) security has evolved from ad-hoc, manual red-teaming to systematic, automated benchmarking frameworks. As the attack surface expands from simple text generation to complex agentic workflows, the methodologies for testing robustness have had to adapt accordingly. This section surveys the current landscape of testing methodologies, highlighting standardized benchmarks, evaluation metrics, and the specialized tools developed to assess vulnerabilities across data poisoning, prompt injection, adversarial attacks, and hallucinations.

## Summary of Testing Methodologies

Table 1 provides a high-level overview of the primary testing methodologies identified in the recent literature, categorized by the specific security threat they address.

| Category | Benchmark/Dataset | Metrics | Tools/Frameworks | Representative Papers |
| :--- | :--- | :--- | :--- | :--- |
| **Data & Model Poisoning** | **Backdoor-LLM** (Alpaca, AdvBench, ToxiGen); **MIMIC-III** (Medical) | Attack Success Rate (ASR); Benign Performance; L-infinity Norms | **BadEdit** (Weight Poisoning); **LoRA** (Data Poisoning); **CleanGen** (Defense) | [Li_2024_Backdoorllm], [Yang_2024_Adversarial] |
| **Prompt Injection** | **InjecAgent**; **AdvBench**; Automated Assignment Grading | ASR-defense; ASR-adaptive; Target Rate; TPR/TNR | **AutoDAN**; **GCG** (Greedy Coordinate Gradient); **AEGIS**; **HOUYI** | [Zhan_2025_LLM], [Liu_2025_Aegis], [Chowdhury_2024_Breaking] |
| **Adversarial Attacks** | **VWA-Adv** (VisualWebArena); **RealHall** | ASR; Benign Success Rate (Benign SR); Adversarial Influence (AdvIn) | **ARE** (Agent Robustness Evaluation); **PromptAttack**; **MasterKey** | [Wu_2024_Dissecting], [Chowdhury_2024_Breaking], [l2_LLM Hallucinations_v1] |
| **Hallucination** | **RealHall**; **ANAH-v2** | LLM-as-a-Judge (ChainPoll); Accuracy | **Self-Training Frameworks** | [l2_LLM Hallucinations_v1] |

### Table 1: Comprehensive Summary of LLM Security Testing Methodologies

## Data and Model Poisoning Evaluation

Testing for poisoning attacks requires evaluating whether a model retains its utility on benign tasks while reliably triggering malicious behavior upon encountering specific triggers. 

### Benchmarks and Datasets
Until recently, poisoning research lacked a unified benchmark, often relying on isolated experiments. In 2024, [Li_2024_Backdoorllm] introduced **Backdoor-LLM**, the first comprehensive benchmark designed to systematically evaluate backdoor threats in text-generation LLMs. This framework integrates diverse datasets to test various capabilities: Stanford Alpaca for instruction tuning, AdvBench for jailbreaking, ToxiGen for toxicity, and reasoning datasets such as GSM8K and MATH. The benchmark covers four distinct attack paradigms: Data Poisoning Attacks (DPA), Weight Poisoning Attacks (WPA), Hidden State Attacks (HSA), and Chain-of-Thought Attacks (CoTA).

In domain-specific contexts, evaluation requires specialized corpora. For instance, [Yang_2024_Adversarial] utilized the **MIMIC-III** dataset (discharge notes) and **PubMedQA** to evaluate poisoning in medical LLMs. This highlights a critical testing requirement: ensuring that poisoned models do not degrade performance on high-stakes professional benchmarks while still executing targeted attacks, such as recommending unsafe prescriptions.

### Metrics and Analysis
The primary metric for poisoning is the **Attack Success Rate (ASR)**, defined as the percentage of inputs containing a trigger that successfully elicit the target malicious response [Li_2024_Backdoorllm]. However, ASR alone is insufficient. Researchers also measure **Benign Performance** (or Clean Accuracy) to ensure the attack is stealthy and does not compromise the model's general utility. [Yang_2024_Adversarial] further proposed analyzing the **L-infinity norms** of weight matrices in fine-tuned models (specifically LoRA adapters), identifying that poisoned models often exhibit larger weight norms compared to clean models, which serves as a detectable metric for model integrity.

### Tools and Frameworks
The **Backdoor-LLM** repository provides a standardized pipeline for implementing attacks using tools like **BadEdit** for direct parameter modification and **BadChain** for reasoning hijacking [Li_2024_Backdoorllm]. Defense evaluation tools within this framework include **GPT-Judge** for detection and **CleanGen** for inference-time mitigation. 

## Prompt Injection and Jailbreaking Evaluation

Prompt injection testing focuses on an attacker's ability to manipulate LLM outputs via malicious instructions, often embedded in external data (Indirect Prompt Injection or IPI).

### Benchmarks and Datasets
A significant development in this domain is the **InjecAgent** benchmark, utilized by [Zhan_2025_LLM] to evaluate defenses against IPI. Unlike standard jailbreak datasets that rely on keyword detection, InjecAgent offers a more precise evaluation of agentic risks. Additionally, [Liu_2025_Aegis] utilized a real-world **Assignment Grading** dataset to test the robustness of the **AEGIS** framework, simulating scenarios where students attempt to manipulate grading bots via injected prompts.

### Metrics and Analysis
Standard ASR metrics have been refined to account for adaptive adversaries. [Zhan_2025_LLM] reports **ASR-defense** (success rate against a specific defense) and **ASR-adaptive** (success rate when the attack is optimized to bypass that defense). They also introduced the **Target Rate**, which measures the ratio of outputs starting with a specific training target, arguing this reflects optimization quality better than binary success flags. 

For defense evaluation, [Liu_2025_Aegis] emphasizes **True Positive Rate (TPR)** and **True Negative Rate (TNR)** to balance security with utility, ensuring defenses do not incorrectly flag benign inputs. This is critical, as [Zhan_2025_LLM] noted that jailbreak settings often rely on subjective LLM evaluations or keyword matching, which can be inconsistent.

### Tools and Frameworks
Automated attack generation has become a standard testing practice. **AutoDAN** [Chowdhury_2024_Breaking] combines genetic algorithms with gradient-based optimization to generate stealthy, readable prompts that bypass perplexity filters. Similarly, **Greedy Coordinate Gradient (GCG)** methods have been adapted for IPI, with variants like Multi-objective GCG (M-GCG) designed to optimize for both success and stealth [Zhan_2025_LLM]. The **AEGIS** framework introduces a co-evolutionary approach, where attacker and defender prompts are jointly optimized in a GAN-like process to discover robust configurations [Liu_2025_Aegis].

## Adversarial Attacks on Agents

As LLMs are integrated into autonomous agents, testing must account for multi-step workflows and tool usage.

### Benchmarks and Datasets
[Wu_2024_Dissecting] introduced **VWA-Adv**, a dataset of 200 targeted adversarial tasks based on the **VisualWebArena** environment. This benchmark simulates realistic web-based attacks (e.g., on Reddit or Shopping platforms) where agents must navigate and interact with compromised elements.

### Metrics and Analysis
Beyond standard ASR, [Wu_2024_Dissecting] proposed the **Agent Robustness Evaluation (ARE)** framework. This models agents as directed graphs, decomposing robustness into **adversarial information flow** between components (e.g., policy, captioner, evaluator). This allows researchers to calculate edge weights representing how attacks propagate through the system. A key finding from this methodology is that inference-time compute mechanisms, such as self-reflection or tree search, can paradoxically increase vulnerability by providing more opportunities for adversarial information to influence the decision process.

## Hallucination Evaluation

While distinct from malicious attacks, hallucination evaluation shares methodologies regarding output verification.

### Metrics and Evolution
Early evaluation relied on n-gram metrics (BLEU, ROUGE), which proved insufficient for reasoning tasks. The field has shifted toward the **LLM-as-a-Judge** paradigm, formalized by benchmarks like **RealHall** and metrics such as **ChainPoll** [l2_LLM Hallucinations_v1]. This approach uses stronger models (e.g., GPT-4) to evaluate the factuality of weaker models. Recent trends have moved toward training specialized judge models (e.g., **ANAH-v2**) to reduce reliance on proprietary APIs and improve scalability [l2_LLM Hallucinations_v1].

## Cross-Category Comparison and Standardization Needs

### The Shift to Adaptive Evaluation
A recurring theme across all categories is the insufficiency of static benchmarks. In prompt injection, [Zhan_2025_LLM] demonstrated that defenses effective against static datasets often fail against **adaptive attacks**, where the adversary has white-box access to the defense mechanism. Similarly, in adversarial agent testing, [Wu_2024_Dissecting] showed that static image perturbations could hijack complex agentic workflows. This suggests that future testing standards must include adaptive attack formulations as a baseline requirement.

### Automation in Red-Teaming
The scale of LLM deployment necessitates automated testing. Tools like **MasterKey** and **PromptAttack** [Chowdhury_2024_Breaking] represent a move toward automated red-teaming, where algorithms reverse-engineer defenses or generate adversarial suffixes without human intervention. This automation is mirrored in the **AEGIS** framework [Liu_2025_Aegis], which automates the defense generation process. 

### Gaps in Current Methodologies
Despite progress, significant gaps remain. [Chowdhury_2024_Breaking] notes that while benchmarking research is active, existing frameworks often prove insufficient for practical, real-time deployment in enterprise scenarios. Furthermore, [Li_2024_Backdoorllm] highlights that prior to Backdoor-LLM, most attacks were studied in isolation without standardized settings, making cross-paper comparisons difficult. The reliance on "LLM-as-a-Judge" also introduces its own biases, as the judge models themselves may be susceptible to the very vulnerabilities they are evaluating.

In conclusion, the field is transitioning from component-level, static evaluation to systemic, adaptive, and automated testing frameworks. The establishment of benchmarks like Backdoor-LLM, InjecAgent, and VWA-Adv marks a maturation of the discipline, yet the rapid evolution of attack strategies necessitates continuous updates to these testing standards.


## 9. Discussion and Open Challenges

### Discussion and Open Challenges

The rapid evolution of Large Language Models (LLMs) from static text generators to autonomous agents has fundamentally altered the security landscape. As synthesized in this survey, the transition from 2024 to 2026 has been characterized by a shift from component-specific vulnerabilities to systemic risks inherent in agentic workflows [l2_Adversarial Attacks_v1]. While significant progress has been made in taxonomizing attacks and developing specific defenses, the integration of LLMs into critical infrastructure and complex decision-making pipelines has exposed deep architectural gaps. This section identifies the critical limitations in current research, discusses the implications of heterogeneous agentic systems, and outlines promising directions for future inquiry.

#### The Compositional Security Gap in Agentic Systems

A primary challenge identified in recent literature is the "compositional security gap." As LLMs are increasingly deployed as agents capable of tool use and multi-step reasoning, security can no longer be guaranteed by validating individual components. Research by Rajagopalan et al. (2026) highlights that while individual actions may satisfy safety policies, their composition can violate security objectives. They identify four simultaneous attack surfaces—tools, data, prompts, and context—where attacks can compose across boundaries to achieve objectives impossible through a single surface [Rajagopalan_2026_AI].

Current frameworks often rely on application-layer defenses that fail to compose, or infrastructure primitives that lack workflow semantics. This creates a scenario where the attack surface grows faster than tools can address [Rajagopalan_2026_AI]. Furthermore, the heterogeneity of the ecosystem—comprising diverse protocols (MCP, A2A), interfaces, and orchestrators—exacerbates the developer burden, making it impractical to implement security logic at every interaction [Rajagopalan_2026_AI]. 

Compounding this issue is the difficulty of tracking data flow through non-deterministic transformations. Errico et al. (2026) note that tracking data lineage is complicated when information passes through an LLM’s context window, where it may be summarized, paraphrased, or transformed. This leads to an open question regarding how to track data lineage through such transformations to prevent data loss or privilege escalation [Errico_2026_AI].

#### Limitations of Static Benchmarking and Evaluation

The dynamic nature of threats has rendered many static evaluation frameworks insufficient. While benchmarks like Backdoor-LLM have provided systematic evaluations for data poisoning and backdoor threats [Li_2024_Backdoorllm], they often fail to capture the adaptive nature of real-world adversaries. Chowdhury et al. (2024) argue that existing frameworks are often insufficient for practical deployment because the adversarial landscape is an ongoing race; attackers continually evolve strategies in response to new defenses [Chowdhury_2024_Breaking].

Moreover, current benchmarks often focus on single-turn interactions. Li et al. (2024) point out that most defenses are tested under single-turn settings, whereas real-world jailbreak attacks often involve multi-turn or open-ended interactions. This limitation is echoed by Zhan et al. (2025), who acknowledge that their adaptive attack experiments focused solely on the agent's first-step action, failing to account for long-term impacts in multi-step data-stealing scenarios [Zhan_2025_LLM].

There is also a critical disconnect between standard performance metrics and safety evaluation. Yang et al. (2024) demonstrated that poisoned models in high-stakes domains, such as medicine, can operate normally for clean inputs—showing no signs of tampering on standard benchmarks—while harboring latent triggers that cause catastrophic failures, such as prescribing dangerous medications [Yang_2024_Adversarial]. This suggests that utility metrics alone are poor proxies for safety.

#### The Nuance of Hallucination and Human Perception

The challenge of LLM hallucinations remains a persistent barrier to trust, complicated by the fact that not all hallucinations are created equal. Simhi et al. (2024) formalized the distinction between hallucinations due to a lack of knowledge (HK-) and those occurring despite correct knowledge (HK+). Their findings indicate that mitigation strategies are not universally effective; prompting is effective for HK+ but ineffective for HK-, necessitating distinct handling mechanisms [Simhi_2024_LLM].

Furthermore, the interaction between human users and hallucinated content presents its own set of challenges. Nahar et al. (2024) found that humans are significantly more susceptible to minor hallucinations (subtle fabrications) compared to major ones. While warning labels were shown to improve detection without significantly reducing trust in genuine content, they did not significantly deter "liking" or "sharing" behaviors [Nahar_2024_LLM]. This highlights a limitation in purely technical defenses; even if a system correctly identifies and flags a hallucination, user behavior may not align with safety goals.

#### Persistence of Poisoning and Backdoors

Data and model poisoning remain formidable threats, particularly as models scale. Li et al. (2024) provided empirical insights showing that while larger models may be more resistant to weight poisoning, they are more susceptible to Chain-of-Thought (CoT) attacks. This vulnerability in reasoning capabilities is particularly concerning for agentic systems that rely on CoT for planning. Additionally, the stealthiness of these attacks is a major concern; backdoors can be implanted via simple prompts or fine-tuning and remain dormant until triggered [Li_2024_Backdoorllm].

The trade-off between bias mitigation and model utility also presents a challenge. Chaudhari et al. (2025) emphasize the need to quantify potential degradation in general capabilities that might occur as a consequence of introducing subtle biases or defenses [Chaudhari_2025_Cascading]. Current defense mechanisms, such as those discussed by Chowdhury et al. (2024), face challenges in scalability and computational overhead, making the balance between security and usability a critical, yet underexplored, area [Chowdhury_2024_Breaking].

#### Future Research Directions

Based on the identified gaps, we propose several high-priority directions for future research:

**1. Protocol-Level Security and Authenticated Workflows**
To address the compositional security gaps in agentic systems, research must move beyond application-layer defenses. Rajagopalan et al. (2026) propose "authenticated workflows" as a protocol-level primitive that enforces intent and integrity at every boundary crossing (prompts, tools, data, context). Future work should focus on standardizing such protocols to create a "trust layer" for AI infrastructure that ensures cryptographic integrity and policy enforcement at scale [Rajagopalan_2026_AI]. This includes developing mechanisms for privilege non-escalation through composition and context integrity via hash chains.

**2. Dynamic and Real-Time Monitoring**
Given the inadequacy of static benchmarks, there is an urgent need for scalable, real-time monitoring systems. Chowdhury et al. (2024) advocate for the development of near-real-time evaluation infrastructure capable of scrutinizing outputs comprehensively and flagging anomalies swiftly [Chowdhury_2024_Breaking]. This aligns with the need for "information flow tracking" (taint analysis) across actions and temporal policy logic to track data lineage through non-deterministic LLM transformations [Errico_2026_AI].

**3. Mechanistic Interpretability for Defense**
Moving away from behavioral observation toward mechanistic interpretability offers a path to more robust defenses. Early work in 2024 began investigating internal attention distributions during attacks [l2_Prompt Injection_v1]. Future research should expand on this by exploring internal defense signals, such as activation patterns or gradient sensitivity, to enable architecture-agnostic mitigation strategies that do not rely solely on input filtering [Li_2024_Backdoorllm].

**4. Adaptive and Multi-Turn Evaluation**
Evaluation frameworks must evolve to match the sophistication of attackers. This involves developing benchmarks that specifically test multi-turn and open-ended interactions [Li_2024_Backdoorllm]. Furthermore, research should explore adaptive attacks that target combinations of defenses, as current studies often focus on individual defense mechanisms in isolation [Zhan_2025_LLM]. The integration of multi-objective evolutionary frameworks, such as "Survival of the Safest," which optimizes prompts for both task performance and safety, represents a promising step toward more rigorous stress-testing [Liu_2025_Aegis].

**5. Differentiated Hallucination Mitigation**
Addressing hallucinations requires a nuanced approach that accounts for the underlying cause. Future systems should implement model-specific datasets to enable preemptive detection of hallucinations before generation, as generic datasets have proven ineffective [Simhi_2024_LLM]. Additionally, research into human-computer interaction is vital to design warning systems that effectively alter user engagement behaviors, not just perception [Nahar_2024_LLM].

**6. Multimodal Security**
The integration of multimodal capabilities introduces new vectors for exploitation. As noted by Chowdhury et al. (2024), future research must prioritize techniques to mitigate challenges in multimodal environments, such as improving input sanitization and validation for non-textual inputs [Chowdhury_2024_Breaking]. This includes addressing vulnerabilities in vision encoders where adversarial contrastive prompt tuning can be used to create robust feature extractors [l2_Adversarial Attacks_v1].

### Conclusion

The field of LLM security is at a critical inflection point. As systems transition from isolated chatbots to integrated, autonomous agents, the security paradigm must shift from probabilistic detection to deterministic architectural enforcement. The challenges of 2026 and beyond—characterized by heterogeneous frameworks, compositional risks, and adaptive adversaries—demand a unified effort to establish a fundamental trust layer for AI. By addressing the open challenges outlined above, the research community can pave the way for the safe and reliable deployment of agentic AI in high-stakes real-world applications.


## 10. Conclusion

This survey has systematically explored the burgeoning landscape of Large Language Model (LLM) security, tracing the trajectory from foundational architectural vulnerabilities to the complex systemic risks inherent in autonomous agentic workflows. As LLMs transition from experimental curiosities to the backbone of modern computational infrastructure, the security paradigm has fundamentally shifted. We are no longer dealing merely with traditional software vulnerabilities—such as buffer overflows or SQL injections—but with semantic exploits where the "code" is natural language and the "compiler" is a probabilistic neural network.

**Summary of Key Findings**
The synthesis of current research reveals that the attack surface of LLMs is vast, multifaceted, and expanding. Prompt injection (PI) and adversarial attacks have emerged as the primary vectors for exploitation at inference time. Unlike traditional exploits that require technical expertise, these attacks democratize the threat landscape, allowing adversaries to manipulate model behavior through linguistic ingenuity. As highlighted in the sections on adversarial attacks and PI, the boundary between a benign user instruction and a malicious payload is often semantically indistinguishable to the model, rendering static signature-based detection ineffective.

Furthermore, the integrity of the LLM supply chain remains a critical point of failure. Data and model poisoning attacks demonstrate that the "garbage in, garbage out" adage has evolved into "malice in, backdoor out." The widespread reliance on opaque, third-party datasets and pre-trained weights creates a vulnerability inheritance model where a single compromised foundation model can propagate latent risks across thousands of downstream applications. Concurrently, the issue of hallucinations [Sui_2024_Confabulation] complicates the security picture by introducing non-malicious but equally damaging reliability failures, particularly in high-stakes domains such as healthcare and law.

**Trends and Practical Implications**
A distinct trend identified in this survey is the evolution from "chatbot security" to "agentic security." As detailed in the Discussion, the integration of LLMs into autonomous agents capable of tool use and API execution has escalated the severity of potential exploits. A successful prompt injection is no longer limited to generating offensive text; it can now trigger remote code execution, unauthorized financial transactions, or data exfiltration.

For practitioners, the implication is clear: the current "patch-and-pray" approach—relying heavily on Reinforcement Learning from Human Feedback (RLHF) and system prompts—is insufficient. Security cannot be an afterthought applied to a frozen model. Instead, a "Defense in Depth" strategy is required, combining rigorous input sanitization, internal model alignment, and external output monitoring. However, as the "cat-and-mouse" dynamic of defenses suggests, adversarial capabilities currently outpace defensive robustness.

**Future Directions and Open Challenges**
To bridge the gap between current vulnerabilities and robust deployment, the research community must pivot toward several critical areas:

1.  **Mechanistic Interpretability for Security:** We must move beyond treating LLMs as black boxes. Future work should focus on mapping specific adversarial behaviors to internal activation patterns. If we can identify the neural circuits responsible for "jailbreak" compliance or hallucination, we can develop targeted interventions that excise these behaviors without degrading general performance.
2.  **Formal Verification of Neuro-Symbolic Systems:** As LLMs are increasingly paired with symbolic logic and deterministic tools, there is a need for frameworks that can formally verify the safety properties of these hybrid systems. Research should aim to provide mathematical guarantees that an agent will not violate specific constraints, regardless of the input prompt.
3.  **Robustness in Agentic Environments:** Existing benchmarks often focus on static text generation. Future evaluation frameworks must simulate dynamic, multi-turn environments where agents interact with external APIs. Developing "honeypot" environments to study how compromised agents behave in the wild will be essential for designing better containment strategies.
4.  **Standardization of Semantic Firewalls:** The industry lacks a standardized protocol for "semantic firewalls"—intermediary layers that analyze intent before passing prompts to the model. Research into lightweight, specialized models dedicated solely to security filtering (both ingress and egress) is a promising avenue for reducing latency while enhancing safety.

**Conclusion**
The security of Large Language Models is not a solved problem but an evolving discipline that sits at the intersection of cybersecurity, natural language processing, and AI alignment. The vulnerabilities documented in this survey—from prompt injections to training data poisoning—underscore a fundamental tension between the flexibility of LLMs and the rigidity required for secure systems.

As we look toward the future, the goal should not merely be to build models that are harder to break, but to architect systems that are resilient by design. This requires a departure from the race for parameter size toward a race for robustness, transparency, and verifiable safety. Only by addressing these open challenges can we ensure that the deployment of LLMs enhances, rather than compromises, the digital ecosystem.


---

## References

*See the accompanying `references.bib` file for the complete bibliography.*
