---
created: '2026-03-22T17:00:31+00:00'
evidence:
- stage-09/exp_plan.yaml
id: experiment_design-rc-20260322-165852-1d173d
run_id: rc-20260322-165852-1d173d
stage: 09-experiment_design
tags:
- experiment_design
- stage-09
- run-rc-20260
title: 'Stage 09: Experiment Design'
---

# Stage 09: Experiment Design

ablations:
- Removal of system-level safety prompts
- Varying the strength of adversarial perturbations
baselines:
- LSTM LM
- GPT-2 Small
compute_budget:
- 8x NVIDIA H100 GPUs for 48 hours
- 500 total GPU hours for fine-tuning
- 2TB high-speed NVMe storage
- 128GB System RAM
datasets:
- WikiText-103
- WikiText-2
metrics:
- Attack Success Rate (ASR)
- Harmfulness Score
- Refusal Rate
- Semantic Consistency Score
- Perplexity
objectives:
- Minimize the generation of harmful content
- Maximize robustness against adversarial prompt injections
- Optimize the trade-off between safety and utility
- Align model responses with human safety guidelines
proposed_methods:
- Adversarial Training with Safety Constraints
- RLUF (Reinforcement Learning from Unsafe Feedback)
- SmoothLLM Defense
- Perplexity-based Input Filtering
risks:
- Indirect Prompt Injection
- Jailbreaking and Safety Filter Bypass
- Training Data Poisoning
- Sensitive Information Leakage
- Model Inversion Attacks
topic: Security vulnerabilities and defenses in Large Language Models
