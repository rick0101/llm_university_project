import torch
import random
import numpy as np
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
    safety_scorer = get_harmfulness_scorer(None) # tokenizer passed inside loader
    
    # Conditions from experiment plan
    conditions = [
        "GPT-2 Small",
        "LSTM LM",
        "Adversarial Training with Safety Constraints",
        "RLUF (Reinforcement Learning from Unsafe Feedback)",
        "SmoothLLM Defense",
        "Perplexity-based Input Filtering",
        "Removal of system-level safety prompts",
        "Varying the strength of adversarial perturbations"
    ]

    summary_results = {}

    for condition in conditions:
        set_seed(config.seeds[0])
        config.use_safety_prompt = True
        strength_override = None
        
        # Setup model and trainer based on condition
        if condition == "GPT-2 Small":
            model = SecurityModel().to(config.device)
            trainer = BaseTrainer(model, config, None, None)
        elif condition == "LSTM LM":
            model = LSTMLM().to(config.device)
            trainer = BaseTrainer(model, config, None, None)
        elif condition == "Adversarial Training with Safety Constraints":
            model = SecurityModel().to(config.device)
            trainer = AdversarialTrainer(model, config, None, None)
        elif condition == "RLUF (Reinforcement Learning from Unsafe Feedback)":
            model = SecurityModel().to(config.device)
            trainer = RLUFTrainer(model, config, None, None)
        elif condition == "SmoothLLM Defense":
            base_m = SecurityModel().to(config.device)
            model = SmoothLLMWrapper(base_m, config, None, None)
            trainer = BaseTrainer(model, config, None, None)
        elif condition == "Perplexity-based Input Filtering":
            base_m = SecurityModel().to(config.device)
            model = PerplexityFilterModel(base_m, config)
            trainer = BaseTrainer(model, config, None, None)
        elif condition == "Removal of system-level safety prompts":
            config.use_safety_prompt = False
            model = SecurityModel().to(config.device)
            trainer = BaseTrainer(model, config, None, None)
        elif condition == "Varying the strength of adversarial perturbations":
            # Test specifically at high strength for this ablation
            strength_override = 0.6
            model = SecurityModel().to(config.device)
            trainer = BaseTrainer(model, config, None, None)
        
        # Data loading (uses config.use_safety_prompt)
        train_loader, test_loader, tokenizer = get_wiki_loaders(config, 'WikiText-2')
        safety_scorer = get_harmfulness_scorer(tokenizer)
        
        # Inject objects into trainer
        trainer.tokenizer = tokenizer
        trainer.safety_scorer = safety_scorer
        if hasattr(model, 'tokenizer'): model.tokenizer = tokenizer
        if hasattr(model, 'safety_scorer'): model.safety_scorer = safety_scorer

        # Run
        trainer.train_epoch(train_loader)
        metrics = trainer.evaluate(test_loader, strength_override=strength_override)
        
        summary_results[condition] = metrics['primary_metric']
        
        # Log results
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if not k.startswith('s')])
        print(f"condition={condition} {metric_str}")

    print("\n--- SUMMARY ---")
    for cond, val in summary_results.items():
        print(f"condition={cond} primary_metric: {val:.4f}")

if __name__ == "__main__":
    run_experiment()