import torch
import random
import numpy as np
from config import ExperimentConfig
from data_utils import get_wiki_loaders, get_harmfulness_scorer
from model_zoo import SecurityModel, SmoothLLMWrapper, PerplexityFilterModel
from engine import BaseTrainer, AdversarialTrainer, RLUFTrainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment():
    config = ExperimentConfig()
    print(f"Device: {config.device} | Safety Prompt: {config.use_safety_prompt}")
    
    train_loader, test_loader, tokenizer = get_wiki_loaders(config)
    safety_scorer = get_harmfulness_scorer(tokenizer)
    
    all_results = []

    for seed in config.seeds:
        set_seed(seed)
        print(f"\n--- Seed {seed} ---")
        
        # 1. Baseline
        m_base = SecurityModel().to(config.device)
        t_base = BaseTrainer(m_base, config, tokenizer, safety_scorer)
        t_base.train_epoch(train_loader)
        res_base = t_base.evaluate(test_loader)
        print(f"Base: {res_base}")

        # 2. SmoothLLM (Ensemble)
        m_smooth = SmoothLLMWrapper(m_base, config, tokenizer, safety_scorer).to(config.device)
        t_smooth = BaseTrainer(m_smooth, config, tokenizer, safety_scorer)
        res_smooth = t_smooth.evaluate(test_loader)
        print(f"Smooth: {res_smooth}")

        # 3. Adv Training
        m_adv = SecurityModel().to(config.device)
        t_adv = AdversarialTrainer(m_adv, config, tokenizer, safety_scorer)
        t_adv.train_epoch(train_loader)
        res_adv = t_adv.evaluate(test_loader)
        print(f"AdvT: {res_adv}")

        # 4. RLUF
        m_rl = SecurityModel().to(config.device)
        t_rl = RLUFTrainer(m_rl, config, tokenizer, safety_scorer)
        t_rl.train_epoch(train_loader)
        res_rl = t_rl.evaluate(test_loader)
        print(f"RLUF: {res_rl}")

        all_results.append({'base': res_base, 'smooth': res_smooth, 'adv': res_adv, 'rl': res_rl})

    # Summary of ASR at max perturbation (0.4)
    print("\n" + "="*40)
    print(f"ASR SUMMARY (Strength 0.4, SafetyPrompt={config.use_safety_prompt})")
    for key in ['base', 'smooth', 'adv', 'rl']:
        vals = [r[key]['ASR_at_0.4'] for r in all_results]
        print(f"{key.upper():10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

if __name__ == "__main__":
    run_experiment()