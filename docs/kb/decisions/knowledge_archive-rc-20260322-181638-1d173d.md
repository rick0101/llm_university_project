---
created: '2026-03-22T18:19:14+00:00'
evidence:
- stage-21/archive.md
- stage-21/bundle_index.json
id: knowledge_archive-rc-20260322-181638-1d173d
run_id: rc-20260322-181638-1d173d
stage: 21-knowledge_archive
tags:
- knowledge_archive
- stage-21
- run-rc-20260
title: 'Stage 21: Knowledge Archive'
---

# Stage 21: Knowledge Archive

# Research Retrospective: The VARK Framework Evaluation

**Date**: October 2023  
**Project**: Validated Adversarial Resilience Kernels (VARK)  
**Status**: **REFINE** (Internal Audit Failure)  
**Target Domains**: `llm-security`, `adversarial-ml`, `robustness-evaluation`

---

## 1. Executive Summary
This retrospective analyzes the preliminary evaluation of the VARK framework, which aimed to disentangle prompt-based safety from algorithmic defense kernels (Smooth, AdvT, RLUF). While the conceptual framework for a 2x2 ablation (Safety Prompt $\times$ Defense Kernel) is theoretically sound, the initial experimental execution failed to meet scientific standards for reproducibility and statistical significance. The project has been halted for refinement due to a "Result Quality Rating" of 3/10.

## 2. Lessons Learned

### Methodological Pitfalls
*   **The "N=8" Trap:** The most critical failure was the sample size. An Attack Success Rate (ASR) of $0.125$ is a mathematical signature of a single failure in eight trials ($1/8$). In security research, such a small sample size allows a single stochastic token flip to swing results by $12.5\%$, rendering the "0.0 ASR" findings for defenses statistically indistinguishable from the baseline.
*   **Confounding Safety Layers:** By keeping "Safety Prompt: True" constant across all experimental rows, the study failed to isolate the marginal utility of the algorithmic defenses. It remained unclear if the observed resilience was due to the complex defense kernels or simply the model's adherence to the system prompt.
*   **The Incompetence Bias:** Conducting tests on CPU-bound environments likely restricted the evaluation to small-parameter models (e.g., <3B parameters). These models often exhibit "security through incompetence," where they fail an attack not because they are robust, but because they lack the instruction-following capability to execute the malicious request.

### Strategic Insights
*   **Utility is Non-Optional:** A defense that achieves 0% ASR by breaking the model’s ability to generate coherent text is a failure. The absence of Benign Task Accuracy (BTA) or MMLU scores created a "utility blindspot."
*   **The Seed Fallacy:** Relying solely on **Seed 42** ignored the high variance inherent in LLM decoding under adversarial noise. Robustness must be validated across a distribution of seeds to be considered "certified."

---

## 3. Reproducibility Notes

### Current Artifact State
*   **Hardware**: Experiments were conducted on generic CPU instances, limiting the scale of the models tested.
*   **Software**: The pipeline utilized a custom injection script with a discrete noise model ($\eta \in \{0.0, 0.1, 0.2, 0.4\}$).
*   **Data**: A subset of AdvBench was used, but the specific prompts were not versioned, complicating exact replication.

### Known Issues for Replicators
1.  **Stochasticity**: Without a multi-seed average, replicators are unlikely to see the exact $0.125$ ASR at the $0.2$ noise threshold.
2.  **Prompt Sensitivity**: The "Safety Prompt" used was a standard "helpful and harmless" instruction; however, even minor phrasing changes in this prompt could significantly alter the baseline ASR.
3.  **Inference Latency**: Randomized Smoothing (Smooth) kernels on CPU are prohibitively slow for the $N \geq 100$ scale required for significance.

---

## 4. Future Work & Corrective Actions

### Phase 1: Statistical Hardening
*   **Scale Up**: Increase sample size to $N=100$ per condition.
*   **Multi-Seed Protocol**: Require results from $\ge 3$ random seeds with reported standard deviation.
*   **Significance Testing**: Apply Fisher’s Exact Test or Barnard’s Test to compare Base vs. Defense ASRs.

### Phase 2: Systematic Ablation
*   **Isolate the Kernel**: Implement a mandatory "Safety Prompt: False" condition. This is the only way to prove that AdvT or RLUF provides weight-level protection that survives "jailbreaking" of the system prompt.
*   **Utility Benchmarking**: Integrate Wikitext-2 Perplexity and MMLU (Massive Multitask Language Understanding) benchmarks into the evaluation pipeline to measure the "Alignment Tax."

### Phase 3: Hardware & Model Scaling
*   **GPU Migration**: Move evaluation to A100/H100 clusters to facilitate testing on instruction-compliant models like **Llama-3-8B** or **Mistral-7B-v0.3**.
*   **Adaptive Attackers**: Introduce an "Adaptive Attacker" model that knows the defense kernel is active (e.g., an attacker that attempts to bypass the smoothing mechanism specifically).

---

## 5. Metadata for Archive
*   **Decision**: `REFINE`
*   **Quality Score**: `3/10`
*   **Primary Failure**: `Low Statistical Power / Confounding Variables`
*   **Key Metric Signature**: `ASR 0.125 @ N=8`
*   **Internal Ref**: `VARK-ERR-2023-001`

{
  "run_id": "run",
  "generated": "2026-03-22T18:19:14+00:00",
  "artifact_count": 620,
  "artifacts": [
    "stage-06/cards/card-100_Llama_2-7B.md",
    "stage-06/cards/card-101_Llama_3-8B.md",
    "stage-06/cards/card-102_Qwen_2_5-14B.md",
    "stage-06/cards/card-103_Global_Country_Information.md",
    "stage-06/cards/card-104_AUC.md",
    "stage-06/cards/card-105_ROUGE-L.md",
    "stage-06/cards/card-106_Liu.md",
    "stage-06/cards/card-107_Sui.md",
    "stage-06/cards/card-108_Confabulation.md",
    "stage-06/cards/card-109_ELECTRA-large.md",
    "stage-06/cards/card-10_DR.md",
    "stage-06/cards/card-110_RoBERTa-large.md",
    "stage-06/cards/card-111_GPT-2.md",
    "stage-06/cards/card-112_DoHA.md",
    "stage-06/cards/card-113_CRTL.md",
    "stage-06/cards/card-114_FaithDial.md",
    "stage-06/cards/card-115_BEGIN.md",
    "stage-06/cards/card-116_HaluEval.md",
    "stage-06/cards/card-117_Narrativity_score.md",
    "stage-06/cards/card-118_DEAM_metric.md",
    "stage-06/cards/card-119_ChainPoll.md",
    "stage-06/cards/card-11_PGD.md",
    "stage-06/cards/card-120_RealHall.md",
    "stage-06/cards/card-121_G-Eval.md",
    "stage-06/cards/card-122_AUROC.md",
    "stage-06/cards/card-123_Adherence.md",
    "stage-06/cards/card-124_Correctness.md",
    "stage-06/cards/card-125_COVID-QA.md",
    "stage-06/cards/card-126_TriviaQA.md",
    "stage-06/cards/card-127_Friel.md",
    "stage-06/cards/card-128_ANAH-v2.md",
    "stage-06/cards/card-129_EM-based_iterative_self-training.md",
    "stage-06/cards/card-12_LLaVA.md",
    "stage-06/cards/card-130_Re-ranking_strategy.md",
    "stage-06/cards/card-131_InternLM2-7B.md",
    "stage-06/cards/card-132_ANAH.md",
    "stage-06/cards/card-133_HalluQA.md",
    "stage-06/cards/card-134_F1.md",
    "stage-06/cards/card-135_Accuracy.md",
    "stage-06/cards/card-136_NLI.md",
    "stage-06/cards/card-137_FActScore.md",
    "stage-06/cards/card-138_Gu.md",
    "stage-06/cards/card-139_RTST.md",
    "stage-06/cards/card-13_AICD_Bench.md",
    "stage-06/cards/card-140_Gemini_2_5_Flash.md",
    "stage-06/cards/card-141_GCG.md",
    "stage-06/cards/card-142_AutoDAN.md",
    "stage-06/cards/card-143_JailbreakBench.md",
    "stage-06/cards/card-144_Qualifire_Benchmark.md",
    "stage-06/cards/card-145_PHTest.md",
    "stage-06/cards/card-146_AutoDefense.md",
    "stage-06/cards/card-147_Bergeron.md",
    "stage-06/cards/card-148_Zhang.md",
    "stage-06/cards/card-149_AEGIS.md",
    "stage-06/cards/card-14_DroidCollection.md",
    "stage-06/cards/card-150_TGO_.md",
    "stage-06/cards/card-151_GPT-4_1-mini.md",
    "stage-06/cards/card-152_GPT-5-mini.md",
    "stage-06/cards/card-153_NTU_Student_Submissions_Dataset.md",
    "stage-06/cards/card-154_LLaMA_3_1_Guard.md",
    "stage-06/cards/card-155_Perplexity-based_Detection.md",
    "stage-06/cards/card-156_ProTeGi.md",
    "stage-06/cards/card-157_Robust_Prompt_Optimization.md",
    "stage-06/cards/card-158_TPR.md",
    "stage-06/cards/card-159_Indirect_Prompt_Injection.md",
    "stage-06/cards/card-15_The_Heap.md",
    "stage-06/cards/card-160_Multi-objective_GCG.md",
    "stage-06/cards/card-161_Two-stage_GCG.md",
    "stage-06/cards/card-162_Fine-tuned_Detector.md",
    "stage-06/cards/card-163_Perplexity_Filter.md",
    "stage-06/cards/card-164_Adversarial_Finetuning.md",
    "stage-06/cards/card-165_Vicuna_7B.md",
    "stage-06/cards/card-166_Llama3-8B.md",
    "stage-06/cards/card-167_DeBERTaV3.md",
    "stage-06/cards/card-168_InjecAgent.md",
    "stage-06/cards/card-169_AgentDojo.md",
    "stage-06/cards/card-16_Swallow_Code.md",
    "stage-06/cards/card-170_Zhan.md",
    "stage-06/cards/card-171_AGENTVIGIL.md",
    "stage-06/cards/card-172_GPTFuzzer.md",
    "stage-06/cards/card-173_VWA-adv.md",
    "stage-06/cards/card-174_o3-mini.md",
    "stage-06/cards/card-175_GPT-4o-mini.md",
    "stage-06/cards/card-176_Claude-3_5-Sonnet.md",
    "stage-06/cards/card-177_Gemini-2-flash-exp.md",
    "stage-06/cards/card-178_pi_detector.md",
    "stage-06/cards/card-179_safety.md",
    "stage-06/cards/card-17_Macro-F1.md",
    "stage-06/cards/card-180_Wang.md",
    "stage-06/cards/card-181_Adaptive_Search-Based_Attack.md",
    "stage-06/cards/card-182_Complex_Word_Bombardment.md",
    "stage-06/cards/card-183_Universal-Prompt-Injection.md",
    "stage-06/cards/card-184_AdvPrompter.md",
    "stage-06/cards/card-185_Multi-Model_Committees.md",
    "stage-06/cards/card-186_Perplexity_Checks.md",
    "stage-06/cards/card-187_Gemma-3-4B-Instruct.md",
    "stage-06/cards/card-188_MT-Bench.md",
    "stage-06/cards/card-189_Anthropic_HH-RLHF.md",
    "stage-06/cards/card-18_ModernBERT.md",
    "stage-06/cards/card-190_Maloyan.md",
    "stage-06/cards/card-191_PurpCode.md",
    "stage-06/cards/card-192_RULE2CODE.md",
    "stage-06/cards/card-193_ArtPrompt.md",
    "stage-06/cards/card-194_CyberSecEval.md",
    "stage-06/cards/card-195_CodeLM-Sec.md",
    "stage-06/cards/card-196_XSCode.md",
    "stage-06/cards/card-197_Qwen3-235B

... (truncated, see full artifact)
