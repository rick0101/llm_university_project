---
created: '2026-03-22T17:34:42+00:00'
evidence:
- stage-11/schedule.json
id: resource_planning-rc-20260322-173432-1d173d
run_id: rc-20260322-173432-1d173d
stage: 11-resource_planning
tags:
- resource_planning
- stage-11
- run-rc-20260
title: 'Stage 11: Resource Planning'
---

# Stage 11: Resource Planning

{
  "tasks": [
    {
      "id": "TASK_001",
      "name": "Baseline Model Training: LSTM and GPT-2 Small on WikiText-103",
      "depends_on": [],
      "gpu_count": 2,
      "estimated_minutes": 240,
      "priority": 1
    },
    {
      "id": "TASK_002",
      "name": "Dataset Preprocessing and Adversarial Perturbation Generation",
      "depends_on": [],
      "gpu_count": 1,
      "estimated_minutes": 120,
      "priority": 1
    },
    {
      "id": "TASK_003",
      "name": "Proposed Method: Adversarial Training with Safety Constraints",
      "depends_on": [
        "TASK_002"
      ],
      "gpu_count": 8,
      "estimated_minutes": 720,
      "priority": 2
    },
    {
      "id": "TASK_004",
      "name": "Proposed Method: Reinforcement Learning from Unsafe Feedback (RLUF)",
      "depends_on": [
        "TASK_003"
      ],
      "gpu_count": 8,
      "estimated_minutes": 960,
      "priority": 2
    },
    {
      "id": "TASK_005",
      "name": "Defense Implementation: SmoothLLM and Perplexity-based Filtering",
      "depends_on": [
        "TASK_002"
      ],
      "gpu_count": 4,
      "estimated_minutes": 300,
      "priority": 3
    },
    {
      "id": "TASK_006",
      "name": "Ablation Study: Removal of System-level Safety Prompts",
      "depends_on": [
        "TASK_004"
      ],
      "gpu_count": 4,
      "estimated_minutes": 480,
      "priority": 4
    },
    {
      "id": "TASK_007",
      "name": "Ablation Study: Varying Strength of Adversarial Perturbations",
      "depends_on": [
        "TASK_002",
        "TASK_004"
      ],
      "gpu_count": 4,
      "estimated_minutes": 480,
      "priority": 4
    },
    {
      "id": "TASK_008",
      "name": "Final Evaluation: ASR, Harmfulness, and Utility Metrics Collection",
      "depends_on": [
        "TASK_001",
        "TASK_004",
        "TASK_005",
        "TASK_006",
        "TASK_007"
      ],
      "gpu_count": 8,
      "estimated_minutes": 360,
      "priority": 5
    }
  ],
  "total_gpu_budget": 500,
  "generated": "2023-11-01T14:30:00Z"
}