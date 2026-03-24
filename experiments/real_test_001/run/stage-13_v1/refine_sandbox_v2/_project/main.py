import torch
import random
import numpy as np
import math
from config import ExperimentConfig
from data_utils import get_wiki_loaders, get_harmfulness_scorer
from model_zoo import SecurityModel, LSTMLM, SmoothLLMWrapper, PerplexityFilterModel
from engine import BaseTrainer, AdversarialTrainer, RLUFTrainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment():
    config = ExperimentConfig()
    set_seed(config.seeds[0])
    
    # 1. Baseline: LSTM LM
    print("Running condition: LSTM LM")
    train_loader, test_loader, tokenizer = get_wiki_loaders(config)
    safety_scorer = get_harmfulness_scorer(tokenizer)
    m_lstm = LSTMLM().to(config.device)
    t_lstm = BaseTrainer(m_lstm, config, tokenizer, safety_scorer)
    t_lstm.train_epoch(train_loader)
    res_lstm = t_lstm.evaluate(test_loader, adv_strength_override=0.1)
    print(f"condition=LSTM LM primary_metric: {res_lstm['ASR']:.4f}")
    for k, v in res_lstm.items(): print(f"condition=LSTM LM {k}: {v:.4f}")

    # 2. Baseline: GPT-2 Small
    print("Running condition: GPT-2 Small")
    m_gpt = SecurityModel().to(config.device)
    t_gpt = BaseTrainer(m_gpt, config, tokenizer, safety_scorer)
    t_gpt.train_epoch(train_loader)
    res_gpt = t_gpt.evaluate(test_loader, adv_strength_override=0.1)
    print(f"condition=GPT-2 Small primary_metric: {res_gpt['ASR']:.4f}")
    for k, v in res_gpt.items(): print(f"condition=GPT-2 Small {k}: {v:.4f}")

    # 3. Proposed: Adversarial Training with Safety Constraints
    print("Running condition: Adversarial Training with Safety Constraints")
    m_adv = SecurityModel().to(config.device)
    t_adv = AdversarialTrainer(m_adv, config, tokenizer, safety_scorer)
    t_adv.train_epoch(train_loader)
    res_adv = t_adv.evaluate(test_loader, adv_strength_override=0.1)
    print(f"condition=Adversarial Training with Safety Constraints primary_metric: {res_adv['ASR']:.4f}")
    for k, v in res_adv.items(): print(f"condition=Adversarial Training with Safety Constraints {k}: {v:.4f}")

    # 4. Proposed: RLUF (Reinforcement Learning from Unsafe Feedback)
    print("Running condition: RLUF (Reinforcement Learning from Unsafe Feedback)")
    m_rl = SecurityModel().to(config.device)
    t_rl = RLUFTrainer(m_rl, config, tokenizer, safety_scorer)
    t_rl.train_epoch(train_loader)
    res_rl = t_rl.evaluate(test_loader, adv_strength_override=0.1)
    print(f"condition=RLUF (Reinforcement Learning from Unsafe Feedback) primary_metric: {res_rl['ASR']:.4f}")
    for k, v in res_rl.items(): print(f"condition=RLUF (Reinforcement Learning from Unsafe Feedback) {k}: {v:.4f}")

    # 5. Proposed: SmoothLLM Defense
    print("Running condition: SmoothLLM Defense")
    m_smooth = SmoothLLMWrapper(m_gpt, config, tokenizer, safety_scorer).to(config.device)
    t_smooth = BaseTrainer(m_smooth, config, tokenizer, safety_scorer)
    res_smooth = t_smooth.evaluate(test_loader, adv_strength_override=0.1)
    print(f"condition=SmoothLLM Defense primary_metric: {res_smooth['ASR']:.4f}")
    for k, v in res_smooth.items(): print(f"condition=SmoothLLM Defense {k}: {v:.4f}")

    # 6. Proposed: Perplexity-based Input Filtering
    print("Running condition: Perplexity-based Input Filtering")
    m_ppl = PerplexityFilterModel(m_gpt, config).to(config.device)
    t_ppl = BaseTrainer(m_ppl, config, tokenizer, safety_scorer)
    res_ppl = t_ppl.evaluate(test_loader, adv_strength_override=0.1)
    print(f"condition=Perplexity-based Input Filtering primary_metric: {res_ppl['ASR']:.4f}")
    for k, v in res_ppl.items(): print(f"condition=Perplexity-based Input Filtering {k}: {v:.4f}")

    # 7. Ablation: Removal of system-level safety prompts
    print("Running condition: Removal of system-level safety prompts")
    train_no_p, test_no_p, _ = get_wiki_loaders(config, use_safety_prompt=False)
    m_no_p = SecurityModel().to(config.device)
    t_no_p = BaseTrainer(m_no_p, config, tokenizer, safety_scorer)
    t_no_p.train_epoch(train_no_p)
    res_no_p = t_no_p.evaluate(test_no_p, adv_strength_override=0.1)
    print(f"condition=Removal of system-level safety prompts primary_metric: {res_no_p['ASR']:.4f}")
    for k, v in res_no_p.items(): print(f"condition=Removal of system-level safety prompts {k}: {v:.4f}")

    # 8. Ablation: Varying the strength of adversarial perturbations
    print("Running condition: Varying the strength of adversarial perturbations")
    # Evaluate the standard GPT-2 Small model across multiple strengths
    res_vary = t_gpt.evaluate(test_loader) # Uses config.adv_strengths
    # Pick strength 0.4 as the representative metric for this condition
    print(f"condition=Varying the strength of adversarial perturbations primary_metric: {res_vary['str_0.4_ASR']:.4f}")
    for k, v in res_vary.items(): print(f"condition=Varying the strength of adversarial perturbations {k}: {v:.4f}")

    # Final Summary
    print("\n" + "="*50)
    print("FINAL SUMMARY OF EXPERIMENT CONDITIONS")
    summary_data = [
        ("LSTM LM", res_lstm['ASR']),
        ("GPT-2 Small", res_gpt['ASR']),
        ("Adv Training", res_adv['ASR']),
        ("RLUF", res_rl['ASR']),
        ("SmoothLLM", res_smooth['ASR']),
        ("PPL Filter", res_ppl['ASR']),
        ("No Safety Prompt", res_no_p['ASR']),
        ("Vary Strength (0.4)", res_vary['str_0.4_ASR'])
    ]
    for name, asr in summary_data:
        print(f"{name:30}: ASR = {asr:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()