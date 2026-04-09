# Report di Verifica Citazioni — Survey LLM Security
**Data verifica**: 9 aprile 2026  
**Fonte**: `output/latest/references.bib` (40 citazioni totali)

---

## Sommario Esecutivo

| Categoria | N. |
|---|---|
| ✅ Verificate (paper reali, metadati corretti) | 36 |
| ⚠️ Verificate ma con errori di metadati (autori/titolo/eprint) | 4 |
| 🔴 PROBLEMI CRITICI (paper reale ma metadati gravi) | 2 |
| ❌ Paper fantasma (inesistente) | 0 |

**Nessun paper è completamente inventato.** Tutti i 40 paper esistono su arXiv o nelle riviste citate. Tuttavia, 2 citazioni presentano problemi critici (abstract rubato da altro paper, eprint mancante) e 4 hanno errori di metadati minori.

---

## 🔴 PROBLEMI CRITICI

### 1. `Kumar_2024_LLMs` — AmpleGCG-Plus
**Problema**: Abstract ERR ATO (copiato da altro paper) + eprint mancante + venue sbagliata

| Campo | Nel .bib | Corretto |
|---|---|---|
| **Titolo** | "AmpleGCG-Plus: A Strong Generative Model..." | ✓ Corretto |
| **Autori** | Vishal Kumar, Zeyi Liao, Jaylen Jones, Huan Sun | ✓ Corretto |
| **Abstract** | Parla di "AutoDAN-Turbo" (**rubato da `Liu_2024_Autodan`!**) | Abstract di AmpleGCG-Plus |
| **eprint** | *(mancante)* | `2410.22143` |
| **venue** | `booktitle = {ICLR}` | È un preprint arXiv, **non** paper ICLR (ICLR 2025 Spotlight è AutoDAN-Turbo, non questo) |
| **tipo** | `@inproceedings` | `@misc` (preprint) |

**Azione necessaria**: Correggere il .bib con `eprint = {2410.22143}`, rimuovere `booktitle = {ICLR}`, sostituire l'abstract.

---

### 2. `Nie_2024_Secodeplt` — SeCodePLT
**Problema**: Titolo impreciso + abstract fabbricato + eprint mancante

| Campo | Nel .bib | Corretto |
|---|---|---|
| **Titolo** | "SECODEPLT: A Unified Benchmark for Evaluating the Security Risks and Capabilities of Code Agents" | "SeCodePLT: A Unified **Platform** for Evaluating the Security of Code **GenAI**" |
| **Autori** | Yuzhou Nie, Zhun Wang, Yu Yang, ... | ✓ Corretti |
| **Abstract** | Parla di "GitHub Copilot e ChatGPT" con vulnerabilità nel codice addestrato — **non è l'abstract reale** | Abstract diverso (NeurIPS D&B 2025) |
| **eprint** | *(mancante)* | `2410.11096` |

**Azione necessaria**: Aggiungere `eprint = {2410.11096}`, correggere il titolo e l'abstract.

---

## ⚠️ ERRORI DI METADATI MINORI

### 3. `Muntasir_2025_Mocha` — MOCHA (arXiv: 2507.19598)
**Paper reale** ✓. Ma la lista autori nel .bib è **altamente corrotta**: contiene nomi inventati, nomi di modelli AI ("Qwen2", "Geminiteam") e altri artefatti da allucination del generatore.

- **Autori reali**: Muntasir Wahed, Xiaona Zhou, Kiet A. Nguyen, Tianjiao Yu, Nirav Diwan, Gang Wang, Dilek Hakkani-Tür, Ismini Lourentzou
- **Nel .bib**: contiene "Geminiteam", "Qwen2", "Aaron Hurst", "Adam Lerer", "Aditya Ramesh", ecc. — chiaramente hallucinated

### 4. `Urlana_2025_LLM` — HalluCounter (arXiv: 2503.04615)
**Paper reale** ✓. Nomi autori distorti nel .bib:
- **Nel .bib**: "Vinayak Charaka **and** Kumar" (spezzato in 2 autori)
- **Reale**: "Charaka Vinayak Kumar" (un solo autore)

### 5. `Mu_2025_LLM` — Evaluate-and-Purify (arXiv: 2504.19730)
**Paper reale** ✓. Nel .bib manca un autore:
- **Nel .bib**: Wenhan Mu, Ling Xu, Shuren Pei, Huichi Zhou
- **Reale**: Wenhan Mu, Ling Xu, Shuren Pei, **Le Mi**, Huichi Zhou

### 6. `Paudel_2025_Hallucinot` — HalluciNot (arXiv: 2504.07069)
**Paper reale** ✓. Nel .bib è presente "Aimon Labs" come se fosse un autore:
- **Nel .bib**: "Bibek Paudel **and Aimon Labs** and Alexander Lyzhov..."
- **Reale**: Bibek Paudel, Alexander Lyzhov, Preetam Joshi, Puneet Anand (Aimon Labs è l'affiliazione, non un autore)

---

## ✅ CITAZIONI VERIFICATE (36/40)

Tutti i seguenti paper sono stati verificati su arXiv: titolo, autori e ID corrispondono correttamente.

| Chiave .bib | arXiv ID | Titolo verificato |
|---|---|---|
| `Chen_2024_Rmcbench` | 2409.15154 | RMCBench: Benchmarking LLMs' Resistance to Malicious Code |
| `Du_2024_LLM` | 2409.17504 | HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection |
| `Guo_2024_Redcode` | 2411.07781 | RedCode: Risky Code Execution and Generation Benchmark for Code Agents |
| `Jenko_2024_LLM` | 2408.02509 | Black-Box Adversarial Attacks on LLM-Based Code Completion |
| `Jing_2024_Scale` | 2410.12222 | On A Scale From 1 to 5: Quantifying Hallucination in Faithfulness Evaluation |
| `Li_2025_Iag` | 2508.09456 | IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding |
| `Li_2026_Slowba` | 2603.08316 | SlowBA: An efficiency backdoor attack towards VLM-based GUI agents |
| `Lin_2025_Exploring` | 2502.03233 | Exploring the Security Threats of Knowledge Base Poisoning in RACG |
| `Lin_2025_LLMs` | 2504.16429 | Give LLMs a Security Course: Securing RACG via Knowledge Injection |
| `Liu_2024_Autodan` | 2410.05295 | AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs |
| `Liu_2024_Flipattack` | 2410.02832 | FlipAttack: Jailbreak LLMs via Flipping |
| `Liu_2025_AI` | 2509.22040 | "Your AI, My Shell": Demystifying Prompt Injection Attacks on Agentic AI Coding Editors |
| `Lyu_2024_Trojvlm` | 2409.19232 | TrojVLM: Backdoor Attack Against Vision Language Models |
| `Mamun_2025_LLMs` | 2508.20333 | Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs |
| `Mo_2025_LLMs` | 2507.22063 | RedCoder: Automated Multi-Turn Red Teaming for Code LLMs |
| `Muntasir_2025_Mocha` | 2507.19598 | MOCHA: Are Code Language Models Robust Against Multi-Turn Malicious Coding Prompts? |
| `Noever_2024_AI` | 2410.06462 | Hallucinating AI Hijacking Attack: LLMs and Malicious Code Recommenders |
| `Ouyang_2025_LLM` | 2503.17953 | Smoke and Mirrors: Jailbreaking LLM-based Code Generation via Implicit Malicious Prompts |
| `Park_2025_LLM` | 2503.01917 | Steer LLM Latents for Hallucination Detection |
| `Paudel_2025_Hallucinot` | 2504.07069 | HalluciNot: Hallucination Detection Through Context and Common Knowledge Verification |
| `Rubel_2024_LLM` | 2410.10526 | Generalized Adversarial Code-Suggestions: Exploiting Contexts of LLM-based Code-Completion |
| `Shen_2025_Decepchain` | 2510.00319 | DecepChain: Inducing Deceptive Reasoning in Large Language Models |
| `Song_2024_Claim` | 2409.16618 | Claim-Guided Textual Backdoor Attack for Practical Applications |
| `Storek_2025_AI` | 2503.14281 | XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants |
| `Tang_2025_Cognibench` | 2505.20767 | CogniBench: A Legal-inspired Framework for Assessing Cognitive Faithfulness of LLMs |
| `Thulke_2025_Listen` | 2505.15633 | Listen to the Context: Towards Faithful LLMs for RAG on Climate Questions |
| `Urlana_2025_LLM` | 2503.04615 | HalluCounter: Reference-free LLM Hallucination Detection in the Wild! |
| `Wang_2024_Developing` | 2407.15441 | Developing a Reliable, Fast, General-Purpose Hallucination Detection Service |
| `Wang_2024_LLMs` | 2406.18122 | Poisoned LangChain: Jailbreak LLMs by LangChain |
| `Wang_2026_Badtemplate` | 2602.05401 | BadTemplate: A Training-Free Backdoor Attack via Chat Template Against LLMs |
| `Yan_2024_LLM` | 2406.06822 | An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models |
| `Ye_2025_Visualtrap` | 2507.06899 | VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation |
| `Zhang_2024_Comparative` | 2408.12398 | A Comparative Analysis of Faithfulness Metrics and Humans in Citation Evaluation |
| `Zhang_2024_Fine` | 2406.15264 | Towards Fine-Grained Citation Evaluation in Generated Text |
| `Zhang_2024_Persistent` | 2410.13722 | Persistent Pre-Training Poisoning of LLMs |
| `Zhou_2025_Badvla` | 2505.16640 | BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models |
| `Zhu_2025_Think` | 2502.12202 | To Think or Not to Think: Exploring the Unthinking Vulnerability in Large Reasoning Models |

---

## Azioni Correttive Consigliate

1. **`Kumar_2024_LLMs`**: aggiungere `eprint = {2410.22143}`, cambiare tipo in `@misc`, rimuovere `booktitle = {ICLR}`, correggere l'abstract.
2. **`Nie_2024_Secodeplt`**: aggiungere `eprint = {2410.11096}`, correggere il titolo in "SeCodePLT: A Unified Platform for Evaluating the Security of Code GenAI", sostituire l'abstract.
3. **`Muntasir_2025_Mocha`**: sostituire la lista autori con: Muntasir Wahed, Xiaona Zhou, Kiet A. Nguyen, Tianjiao Yu, Nirav Diwan, Gang Wang, Dilek Hakkani-Tür, Ismini Lourentzou.
4. **`Urlana_2025_LLM`**: correggere autore "Vinayak Charaka and Kumar" in "Charaka Vinayak Kumar".
5. **`Mu_2025_LLM`**: aggiungere autore mancante "Le Mi".
6. **`Paudel_2025_Hallucinot`**: rimuovere "Aimon Labs" dalla lista autori.
